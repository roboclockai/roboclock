"""A4 hang and start infusion: real-physics oracle runs over levels x seeds in one Isaac session.

python scripts/hospital/a4_oracle_batch.py --output results/hospital/A4/oracle \
    --summary results/hospital/B3/oracle_summary.json

Writes one <level>_<seed>.json per episode (metrics, events, planner phases, episode metadata),
frames from the task camera (the first and last, plus any --frames-at simulation times) and a
summary with physical and within-deadline pass counts reported separately. Frame captures never
advance the physics clock.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def level_counts(rows):
    """Physical success and within-deadline success are reported separately."""
    latencies = [r["completion_latency"] for r in rows if r["completion_latency"] is not None]
    hangs = [r["hang_latency"] for r in rows if r["hang_latency"] is not None]
    limit = rows[0]["response_limit"] if rows else None
    return {
        "episodes": len(rows),
        "physical": sum(bool(r["physical"]) for r in rows),
        "within_deadline": sum(bool(r["temporal"]) for r in rows),
        "joint": sum(bool(r["joint"]) for r in rows),
        "response_limit_s": limit,
        "latency_range_s": [min(latencies), max(latencies)] if latencies else None,
        "hang_latency_range_s": [min(hangs), max(hangs)] if hangs else None,
        "max_fraction_of_limit": round(max(latencies) / limit, 3) if latencies else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--seconds", type=float, default=20.0, help="per-episode simulation cap")
    p.add_argument("--output", type=Path, default=Path("results/hospital/A4/oracle"))
    p.add_argument("--summary", type=Path, default=None, help="also write the summary here")
    p.add_argument("--no-frames", action="store_true")
    p.add_argument("--frames-at", nargs="*", type=float, default=[], help="extra frame times (s)")
    p.add_argument("--trace", action="store_true", help="save per-control-step hand/bag/key samples")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.bedside.iv_bag import BAG, INSTRUCTION, START_KEY, start_key_state
    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    rows = []
    try:
        env = HospitalEnv(task_id="A4", difficulty=a.levels[0], oracle=True, runtime=runtime)
        for level in a.levels:
            for seed in a.seeds:
                env.difficulty = level
                started = time.perf_counter()
                obs, _ = env.reset(seed=seed)
                tag = f"{level}_{seed}"
                if not a.no_frames:
                    Image.fromarray(obs["rgb"]).save(a.output / f"{tag}_initial.png")
                policy = OraclePickPlace(env)
                policy.reset(obs)
                pending = sorted(a.frames_at)
                trace = []
                while env.task.time < a.seconds and not env.done:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    env.task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                    if a.trace:
                        state = env.backend.state()
                        bag = state.objects[BAG]
                        trace.append(
                            {
                                "t": round(env.task.time, 4),
                                "phase": getattr(policy.planner, "iv_bag_phase", None),
                                "ee": state.ee_position,
                                "goal": policy.planner.last_goal,
                                "bag": bag.position,
                                "grasped": bag.grasped,
                                "key": start_key_state(env.task.layout, state.objects[START_KEY]),
                                "hang": state.controls.get("iv_bag"),
                                "retained": getattr(env.task, "_retained", None),
                                "bag_quat": bag.orientation,
                                "bag_contact": bag.robot_contact,
                            }
                        )
                    while pending and env.task.time >= pending[0] and not a.no_frames:
                        env.backend.capture_camera(subframes=4)
                        Image.fromarray(env.render()).save(a.output / f"{tag}_t{pending.pop(0):05.2f}.png")
                if not a.no_frames:
                    env.backend.capture_camera(subframes=4)
                    Image.fromarray(env.render()).save(a.output / f"{tag}_final.png")
                m = env.task.metrics()
                phases = [e for e in env.task.event_log if e["type"] == "iv_bag_planner_phase"]
                (a.output / f"{tag}.json").write_text(
                    json.dumps(
                        {
                            "metrics": m,
                            "events": env.task.event_log,
                            "phases": phases,
                            "episode": env.episode_metadata(),
                            "wall_seconds": time.perf_counter() - started,
                        },
                        default=plain,
                        indent=1,
                    )
                )
                if a.trace:
                    (a.output / f"{tag}_trace.json").write_text(json.dumps(trace, default=plain))

                def rounded(value):
                    return None if value is None else round(value, 3)

                row = {
                    "level": level,
                    "seed": seed,
                    "event_time": rounded(env.task.event_time),
                    "response_limit": env.task.spec.response_limit,
                    "hang_latency": rounded(m["hang_latency"]),
                    "completion_latency": rounded(m["completion_latency"]),
                    "physical": m["task_success"],
                    "temporal": m["temporal_success"],
                    "joint": m["joint_success"],
                    "violations": m.get("violations"),
                    "physical_failures": m.get("physical_failures"),
                    "collisions": m.get("collision_count"),
                    "phases": {e["phase"]: round(e["time"], 3) for e in phases},
                }
                rows.append(row)
                print("A4", json.dumps(row, default=plain), flush=True)
        env.close()
    finally:
        counts = {}
        for level in a.levels:
            level_rows = [r for r in rows if r["level"] == level]
            counts[level] = level_counts(level_rows)
        summary = {
            "task": "A4 hang_and_start_infusion",
            "environment": "hospital",
            "policy": "privileged diagnostic oracle (bedside/iv_bag.py command), Isaac Sim physics",
            "simulator": simulator_version(),
            "instruction": INSTRUCTION,
            "setup": (
                "scene-02 left bay, robot on the bed's open side at the patient's shoulder, arm "
                "mounts raised to 1.50 m, tucked start at mount + (0.20, -0.04, 0.02) m; floor IV "
                "pole with a clamp-on hook at 1.55 m (crown 1.70-1.79 m out of top-down reach), "
                "spike clamp and the infusion pump on a side clamp (screen toward the foot of the "
                "bed, START on its top adapter); narrow prep cart (top 0.95 m) under the pump with "
                "the bag cradle, the dial and the order card"
            ),
            "distances_from_left_mount_m": {
                "cradle": 0.24, "hanging_bag": 0.37, "threading_entry": 0.37, "start_key": 0.26,
            },
            "start_key_bearing_off_bag_deg": 38,
            "limits_s": {"easy": 13.2, "medium": 11.0, "hard": 8.8},
            "seeds": a.seeds,
            "pass_counts": counts,
            "episodes": rows,
        }
        text = json.dumps(summary, indent=1, default=plain)
        (a.output / "summary.json").write_text(text)
        if a.summary is not None:
            a.summary.parent.mkdir(parents=True, exist_ok=True)
            a.summary.write_text(text)
        runtime.close()


if __name__ == "__main__":
    main()
