"""C1 pharmacy order bins: oracle check over difficulties x seeds in one Isaac Sim session.

Writes results/hospital/C1/oracle/<level>_<seed>.json (metrics + events) and a summary.json;
--still saves the initial frame of each episode as a PNG.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--seconds", type=float, default=None, help="stop early (diagnosis only)")
    p.add_argument("--still", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/hospital/C1/oracle"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    summary = []
    try:
        for level in a.levels:
            env = HospitalEnv(task_id="C1", difficulty=level, oracle=True, runtime=runtime)
            try:
                for seed in a.seeds:
                    started = time.perf_counter()
                    obs, _ = env.reset(seed=seed)
                    if a.still:
                        Image.fromarray(obs["rgb"]).save(a.output / f"{level}_{seed}_initial.png")
                    policy = OraclePickPlace(env)
                    policy.reset(obs)
                    limit = a.seconds or env.task.spec.max_time
                    while env.task.time < limit - 1e-9 and not env.done:
                        action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                        env.backend.apply_action(action)
                        env.task.action_count += 1
                        env._ticks(env.config["sim"]["control_decimation"], render=False)
                    task = env.task
                    if a.still:
                        Image.fromarray(env.render()).save(a.output / f"{level}_{seed}_final.png")
                    m = task.metrics()
                    times = sorted(
                        e["time"] for e in task.event_log if e.get("type") == "task_action"
                    )
                    placed = [
                        e["time"]
                        for e in task.event_log
                        if e.get("type") == "planner_phase" and e.get("phase") == "release"
                    ]
                    arms = {}
                    for e in task.event_log:
                        if e.get("type") == "planner_phase" and e.get("phase") == "release":
                            arms[e.get("arm", 0)] = arms.get(e.get("arm", 0), 0) + 1
                    row = {
                        "releases_per_arm": arms,
                        "level": level,
                        "seed": seed,
                        "horizon": task.spec.max_time,
                        "jobs": f"{m['jobs_done']}/{m['jobs_required']}",
                        "placements": task.placements,
                        "accuracy": task.accuracy,
                        "t_q": m["t_q_s"],
                        "end": m["termination_reason"],
                        "physical": m["task_success"],
                        "temporal_by_tier": m["temporal_success_by_tier"],
                        "joint_by_tier": m["joint_success_by_tier"],
                        "wall_s": round(time.perf_counter() - started, 1),
                        "retreats": placed,
                        "first_action": times[:1],
                    }
                    summary.append(row)
                    print("C1", json.dumps(row, default=plain), flush=True)
                    (a.output / f"{level}_{seed}.json").write_text(
                        json.dumps(
                            {"metrics": m, "events": task.event_log}, default=plain, indent=1
                        )
                    )
            finally:
                env.close()
        (a.output / "summary.json").write_text(json.dumps(summary, default=plain, indent=1))
    except BaseException:
        import traceback

        traceback.print_exc()
        runtime.exit_code = 1
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
