"""B3 hazardous drug spill: actual-physics oracle check over levels x seeds, one simulator session.

Rendering happens only at reset (and at chosen times with --frames), so the physics clock is never
advanced by rendering. Per episode: the break (bottle tip to first counter contact, where it
landed), containment and clean-up latencies against their limits, and any motion before the
break. Writes one JSON per episode and a summary that reports physical success and within-deadline
success separately.

LD_LIBRARY_PATH=/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/hospital/b3_spill_oracle.py \
    --output results/hospital/B3/oracle --seeds 0 1 2 3 4 5 6 7 8 9
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


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "min": round(min(values), 3),
        "mean": round(float(np.mean(values)), 3),
        "max": round(max(values), 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=Path, default=Path("results/hospital/B3/oracle"))
    p.add_argument("--frames", nargs="*", type=float, help="Save frames this long after the break (s)")
    p.add_argument("--no-render", action="store_true", help="Build no cameras (physics only)")
    p.add_argument("--stop-after", type=float, help="End each episode this long after the break")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    rows = []
    try:
        env = HospitalEnv(task_id="B3", difficulty=a.levels[0], oracle=True, runtime=runtime)
        if a.no_render:
            env.config["sim"]["no_render"] = True
        for level in a.levels:
            for seed in a.seeds:
                env.difficulty = level
                started = time.perf_counter()
                obs, _ = env.reset(seed=seed)
                tag = f"B3_{level}_{seed}"
                frames = sorted(a.frames or [])
                if a.frames is not None:
                    Image.fromarray(obs["rgb"]).save(a.output / f"{tag}_reset.png")
                    for key in ("right_wrist_rgb", "left_wrist_rgb"):
                        if key in obs:
                            Image.fromarray(obs[key]).save(a.output / f"{tag}_reset_{key}.png")
                policy = OraclePickPlace(env)
                policy.reset(obs)
                task = env.task
                start_ee = env.backend.state().ee_position.copy()
                pre_motion = 0.0
                while task.time < task.spec.max_time - 1e-9 and not env.done:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                    if task.event_time is None:
                        ee = env.backend.state().ee_position
                        pre_motion = max(pre_motion, float(np.linalg.norm(ee - start_ee)))
                        continue
                    since = task.time - task.event_time
                    while frames and since >= frames[0] - 1e-9:
                        env.backend.capture_camera(subframes=8)
                        Image.fromarray(env.render()).save(
                            a.output / f"{tag}_t{frames.pop(0):05.2f}.png"
                        )
                    if a.stop_after is not None and since >= a.stop_after:
                        break
                if a.frames is not None:
                    env.backend.capture_camera(subframes=8)
                    Image.fromarray(env.render()).save(a.output / f"{tag}_final.png")
                m = task.metrics()
                objects = env.backend.state().objects
                row = {
                    "level": level,
                    "seed": seed,
                    "physical": bool(m["task_success"]),
                    "temporal": bool(m["temporal_success"]),
                    "joint": bool(m["joint_success"]),
                    "response_limit": task.spec.response_limit,
                    "containment_limit": m["containment_limit"],
                    "fall_duration": None
                    if m["break_time"] is None
                    else m["break_time"] - m["fall_time"],
                    "landing_offset": None
                    if task.spill_center is None
                    else (task.spill_center - task.landing).round(4).tolist(),
                    "containment_latency": m["containment_latency"],
                    "shard_latencies": {
                        k: v - m["break_time"] for k, v in m["shard_times"].items()
                    },
                    "response_time": m["response_time"],
                    "puddle_radius": m["puddle_radius"],
                    "puddle_full_radius": m["puddle_full_radius"],
                    "violations": m["temporal_violations"],
                    "physical_failures": m["physical_failures"],
                    "drops": m["drops"],
                    "pre_break_motion_m": pre_motion,
                    "final_positions": {
                        k: np.round(v.position, 3).tolist() for k, v in objects.items()
                    },
                    "planner": [
                        e for e in task.event_log if e["type"].startswith("planner_") and "phase" not in e["type"]
                    ],
                    "sim_time": task.time,
                    "wall_s": round(time.perf_counter() - started, 1),
                }
                rows.append(row)
                (a.output / f"{tag}.json").write_text(
                    json.dumps({"row": row, "metrics": m, "events": task.event_log}, indent=1, default=plain)
                )
                print(
                    f"{tag}: physical={row['physical']} temporal={row['temporal']} "
                    f"contain={row['containment_latency']} / {row['containment_limit']:.2f} "
                    f"done={row['response_time']} / {row['response_limit']:.2f} "
                    f"land={row['landing_offset']} fall={row['fall_duration']} "
                    f"viol={row['violations']} fail={row['physical_failures']} drops={row['drops']}",
                    flush=True,
                )
        summary = {
            "episodes": len(rows),
            "by_level": {
                level: {
                    "episodes": sum(r["level"] == level for r in rows),
                    "physical": sum(r["physical"] for r in rows if r["level"] == level),
                    "within_deadline": sum(r["joint"] for r in rows if r["level"] == level),
                }
                for level in a.levels
            },
            "containment_latency": stats([r["containment_latency"] for r in rows]),
            "response_time": stats([r["response_time"] for r in rows]),
            "fall_duration": stats([r["fall_duration"] for r in rows]),
            "pre_break_motion_m": stats([r["pre_break_motion_m"] for r in rows]),
            "rows": rows,
        }
        (a.output / "oracle_summary.json").write_text(json.dumps(summary, indent=1, default=plain))
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
