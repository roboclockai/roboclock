"""A3 syringe pump cutoff: actual-physics oracle check over levels x seeds.

Runs the privileged oracle (joint actions from Lula IK) in Isaac Sim and records, per episode,
physical success (the measured STOP press), within-window success, the action time relative to the
window (seconds after it opens and as a fraction of its width), the measured press duration (start of
the descent to the end of the 40 ms hold) and the frozen volume. Writes one JSON per episode and the
summary (default results/hospital/A3/oracle_summary.json).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=Path, default=Path("results/hospital/A3/a3w_oracle"))
    p.add_argument("--summary", type=Path, default=Path("results/hospital/A3/oracle_summary.json"))
    p.add_argument(
        "--from-results", action="store_true", help="only rebuild the summary from --output"
    )
    a = p.parse_args()
    if a.from_results:
        episodes = [
            json.loads((a.output / f"A3_{level}_{seed}.json").read_text())
            for level in a.levels
            for seed in a.seeds
        ]
        from roboclock.envs.hospital.bedside.window_syringe import PRESS_LEAD

        summarize(episodes, a, PRESS_LEAD)
        return
    a.output.mkdir(parents=True, exist_ok=True)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from roboclock.envs.hospital.bedside.window_syringe import PRESS_LEAD
    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    episodes = []
    try:
        env = HospitalEnv(task_id="A3", difficulty=a.levels[0], oracle=True, runtime=runtime)
        for level in a.levels:
            for seed in a.seeds:
                env.difficulty = level
                started = time.perf_counter()
                obs, _ = env.reset(seed=seed)
                policy = OraclePickPlace(env)
                policy.reset(obs)
                task = env.task
                while not env.done:
                    # Interface order (rig-left arm first) in and out of env.step.
                    obs = env.step(policy.act(obs))[0]
                m = task.metrics()
                events = [e for e in task.event_log if e.get("type") != "planner_trace"]
                start = next((e["time"] for e in events if e["type"] == "oracle_press_start"), None)
                t_o, t_c = task.window_open, task.window_close
                act = task.action_time
                record = {
                    "level": level,
                    "seed": seed,
                    "physical_success": bool(task.physical_success),
                    "within_window_success": bool(task.temporal_success),
                    "joint_success": bool(task.joint_success),
                    "window_open_s": t_o,
                    "window_close_s": t_c,
                    "window_width_s": t_c - t_o,
                    "action_time_s": act,
                    "action_after_open_s": None if act is None else act - t_o,
                    "action_before_close_s": None if act is None else t_c - act,
                    "action_fraction_of_window": None if act is None else (act - t_o) / (t_c - t_o),
                    "press_start_s": start,
                    "press_duration_s": None if act is None or start is None else act - start,
                    "device_moved_time_s": task.device_moved_time,
                    "initial_volume_ml": m["initial_quantity"],
                    "final_volume_ml": m["final_quantity"],
                    "quantity_outcome": m["quantity_outcome"],
                    "violations": m.get("temporal_violations"),
                    "collisions": task.collision_count,
                    "episode_seconds": task.time,
                    "wall_seconds": time.perf_counter() - started,
                    "events": events,
                }
                (a.output / f"A3_{level}_{seed}.json").write_text(
                    json.dumps(record, indent=2, default=str)
                )
                episodes.append(record)
                print(
                    f"A3 {level} {seed}: physical={record['physical_success']} window={record['within_window_success']} "
                    f"act={act} window=({t_o:.3f},{t_c:.3f}) frac={record['action_fraction_of_window']} "
                    f"press={record['press_duration_s']} vol={m['final_quantity']:.3f} {m['quantity_outcome']}",
                    flush=True,
                )
        # Written before the runtime closes: Kit's fast shutdown exits the process.
        summarize(episodes, a, PRESS_LEAD)
    finally:
        env.close()
        runtime.close()


def summarize(episodes, a, press_lead):
    def stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}

    per_level = {}
    for level in a.levels:
        rows = [e for e in episodes if e["level"] == level]
        per_level[level] = {
            "episodes": len(rows),
            "physical_success": sum(e["physical_success"] for e in rows),
            "within_window_success": sum(e["within_window_success"] for e in rows),
            "joint_success": sum(e["joint_success"] for e in rows),
            "window_width_s": rows[0]["window_width_s"] if rows else None,
            "action_after_open_s": stats([e["action_after_open_s"] for e in rows]),
            "action_fraction_of_window": stats([e["action_fraction_of_window"] for e in rows]),
            "final_volume_ml": stats([e["final_volume_ml"] for e in rows]),
        }
    summary = {
        "task": "A3 syringe_pump_cutoff",
        "simulator": "Isaac Sim 6.0.1 (actual physics, privileged Lula-IK joint oracle)",
        "band_ml": [4.5, 5.5],
        "oracle": "hover above STOP; start the press PRESS_LEAD s before the window middle; lift after the stop",
        "press_lead_s": press_lead,
        "press_duration_s": stats([e["press_duration_s"] for e in episodes]),
        "totals": {
            "episodes": len(episodes),
            "physical_success": sum(e["physical_success"] for e in episodes),
            "within_window_success": sum(e["within_window_success"] for e in episodes),
            "joint_success": sum(e["joint_success"] for e in episodes),
        },
        "per_level": per_level,
        "episodes": [{k: v for k, v in e.items() if k != "events"} for e in episodes],
    }
    a.summary.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["totals"]), summary["press_duration_s"])


if __name__ == "__main__":
    main()
