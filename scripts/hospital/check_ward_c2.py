"""C2 instrument tray sort: actual-physics oracle check over levels x seeds, plus stills.

Runs every episode to its end (all six sorted and settled, the stall stop or the horizon H) and
writes one result.json per episode and a summary.json. Renders are taken only at the start and the end, so
the physics clock is never advanced by rendering.
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
    p.add_argument("--output", type=Path, default=Path("results/hospital/C2/check"))
    p.add_argument("--seconds", type=float, default=None, help="Stop early (diagnosis only)")
    p.add_argument("--still-only", action="store_true", help="Only render the initial frame")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    summary = []
    try:
        env = HospitalEnv(task_id="C2", difficulty=a.levels[0], oracle=True, runtime=runtime)
        for level in a.levels:
            for seed in a.seeds:
                env.difficulty = level
                started = time.perf_counter()
                obs, _ = env.reset(seed=seed)
                tag = f"C2_{level}_{seed}"
                Image.fromarray(obs["rgb"]).save(a.output / f"{tag}_initial.png")
                if a.still_only:
                    continue
                policy = OraclePickPlace(env)
                policy.reset(obs)
                limit = env.task.spec.max_time if a.seconds is None else a.seconds
                while env.task.time < limit - 1e-9 and not env.done:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    env.task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                for _ in range(24):  # flush stale RTX frames after rendering-free physics
                    env.backend.world.render()
                Image.fromarray(env.render()).save(a.output / f"{tag}_final.png")
                m = env.task.metrics()
                result = {
                    "metrics": m,
                    "events": env.task.event_log,
                    "episode": env.episode_metadata(),
                    "wall_seconds": time.perf_counter() - started,
                }
                (a.output / f"{tag}.json").write_text(json.dumps(result, default=plain, indent=1))
                row = {
                    "level": level,
                    "seed": seed,
                    "horizon": env.task.spec.max_time,
                    "correct": m["correct_operations"],
                    "placements": m["attempted_operations"],
                    "sort_times": [round(t, 2) for t in m["sort_times"]],
                    "t_q": m["t_q_s"],
                    "end": m["termination_reason"],
                    "physical": m["task_success"],
                    "temporal_by_tier": m["temporal_success_by_tier"],
                    "joint_by_tier": m["joint_success_by_tier"],
                }
                summary.append(row)
                print("C2", json.dumps(row), flush=True)
                (a.output / "summary.json").write_text(json.dumps(summary, indent=1))
        env.close()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
