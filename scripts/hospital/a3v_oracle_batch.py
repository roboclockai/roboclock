"""A3 bag-valve ventilation: real-physics oracle runs over levels x seeds in one Isaac session.

    python scripts/hospital/a3v_oracle_batch.py

Headless (sim.no_render), zero inference latency and control delay: the privileged oracle (joint
actions from Lula IK, starting from the tucked rest pose) is stepped directly. Writes one
A3_<level>_<seed>.json per episode and a summary (default results/hospital/A3/
oracle_summary.json) in which physical success (all breaths delivered by measured bag
compression) and timing success (every breath inside its window, nothing extra) are counted
separately, with each breath's error from its beat and the squeeze duration.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"min": float(min(values)), "mean": float(np.mean(values)), "max": float(max(values))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--output", type=Path, default=Path("results/hospital/A3/a3v_oracle"))
    p.add_argument("--summary", type=Path, default=Path("results/hospital/A3/oracle_summary.json"))
    p.add_argument("--start-pose", choices=["task", "zero"], default="task")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    import yaml

    from roboclock.envs.hospital.config import load_config

    config = load_config()
    config["sim"]["no_render"] = True
    if a.start_pose == "zero":
        config["robot"]["start_pose"] = "zero"
    config_path = a.output / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from roboclock.envs.hospital.bedside.window_bvm import PRESS_LEAD
    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    episodes = []
    try:
        env = HospitalEnv(
            task_id="A3", difficulty=a.levels[0], oracle=True, runtime=runtime, config_path=config_path
        )
        for level in a.levels:
            for seed in a.seeds:
                env.difficulty = level
                started = time.perf_counter()
                obs, _ = env.reset(seed=seed)
                policy = OraclePickPlace(env)
                policy.reset(obs)
                task = env.task
                peak = 0.0
                while not env.done:
                    obs = env.step(policy.act(obs))[0]
                    bag = (task.previous.controls if task.previous else {}).get("bag", {})
                    peak = max(peak, float(bag.get("compression_m") or 0.0))
                m = task.metrics()
                events = [e for e in task.event_log if e.get("type") != "planner_trace"]
                starts = [e["time"] for e in events if e["type"] == "oracle_press_start"]
                record = {
                    "level": level,
                    "seed": seed,
                    "physical_success": bool(task.physical_success),
                    "within_window_success": bool(task.temporal_success),
                    "joint_success": bool(task.joint_success),
                    "window_width_s": m["window_width"],
                    "beats_s": m["beats"],
                    "breath_times_s": m["breath_times"],
                    "beat_errors_s": m["beat_errors"],
                    "breath_intervals_s": m["breath_intervals"],
                    "squeeze_durations_s": [b - s for s, b in zip(starts, m["breath_times"])],
                    "arrival_margin_s": None if not starts else starts[0],
                    "peak_compression_m": peak,
                    "device_moved_time_s": task.device_moved_time,
                    "rhythm_outcome": m["rhythm_outcome"],
                    "violations": m.get("temporal_violations"),
                    "physical_failures": m.get("physical_failures"),
                    "collisions": task.collision_count,
                    "episode_seconds": task.time,
                    "wall_seconds": time.perf_counter() - started,
                    "events": events,
                }
                (a.output / f"A3_{level}_{seed}.json").write_text(json.dumps(record, indent=2, default=str))
                episodes.append(record)
                errors = [round(e, 3) for e in record["beat_errors_s"]]
                print(
                    f"A3 {level} {seed}: physical={record['physical_success']} "
                    f"timing={record['within_window_success']} beat_errors={errors} "
                    f"{record['rhythm_outcome']} {record['violations']}",
                    flush=True,
                )
        per_level = {}
        for level in a.levels:
            rows = [e for e in episodes if e["level"] == level]
            errors = [abs(x) for e in rows for x in e["beat_errors_s"]]
            per_level[level] = {
                "episodes": len(rows),
                "physical_success": sum(e["physical_success"] for e in rows),
                "within_window_success": sum(e["within_window_success"] for e in rows),
                "joint_success": sum(e["joint_success"] for e in rows),
                "window_width_s": rows[0]["window_width_s"] if rows else None,
                "abs_beat_error_s": stats(errors),
                "margin_to_window_edge_s": None
                if not errors or not rows
                else rows[0]["window_width_s"] / 2 - max(errors),
            }
        summary = {
            "task": "A3 bvm_ventilation",
            "simulator": "Isaac Sim 6.0.1 (actual physics, privileged Lula-IK joint oracle)",
            "start_pose": a.start_pose,
            "oracle": "from the tucked rest pose to a hover over the bag; each squeeze starts "
            "PRESS_LEAD s before its beat; the bag is released after every breath",
            "press_lead_s": PRESS_LEAD,
            "squeeze_duration_s": stats([x for e in episodes for x in e["squeeze_durations_s"]]),
            "abs_beat_error_s": stats([abs(x) for e in episodes for x in e["beat_errors_s"]]),
            "totals": {
                "episodes": len(episodes),
                "physical_success": sum(e["physical_success"] for e in episodes),
                "within_window_success": sum(e["within_window_success"] for e in episodes),
                "joint_success": sum(e["joint_success"] for e in episodes),
            },
            "per_level": per_level,
            "episodes": [{k: v for k, v in e.items() if k != "events"} for e in episodes],
        }
        # Written before the runtime closes: Kit's fast shutdown exits the process.
        a.summary.parent.mkdir(parents=True, exist_ok=True)
        a.summary.write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary["totals"]), summary["abs_beat_error_s"])
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
