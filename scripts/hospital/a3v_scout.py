"""A3 bag-valve ventilation: one oracle episode in Isaac Sim with frames from the policy cameras.

Saves head / wrist frames every --every seconds (and the third-person view) plus a compression
trace, to check the scene, visibility and the press timing before recording.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="medium")
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--every", type=float, default=1.0)
    p.add_argument("--until", type=float, default=None)
    p.add_argument("--output", type=Path, default=Path("results/hospital/A3/a3v_scout"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    try:
        env = HospitalEnv(task_id="A3", difficulty=a.level, oracle=True, runtime=runtime)
        obs, _ = env.reset(seed=a.seed)
        policy = OraclePickPlace(env)
        policy.reset(obs)
        task = env.task
        print("instruction:", task.gen_instruction())
        print("windows:", task.windows)
        trace, shot = [], -1e9

        def save(tag):
            for key, value in obs.items():
                if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] in (3, 4):
                    Image.fromarray(value[..., :3].astype(np.uint8)).save(a.output / f"{tag}_{key}.jpg")

        save("t00.00")
        print("obs keys:", {k: getattr(v, "shape", None) for k, v in obs.items()})
        while not env.done and (a.until is None or task.time < a.until):
            obs = env.step(policy.act(obs))[0]
            bag = (task.previous.controls if task.previous else {}).get("bag", {})
            trace.append([task.time, bag.get("compression_m"), bag.get("contact"), bag.get("lung_fraction")])
            if task.time - shot >= a.every - 1e-9:
                shot = task.time
                save(f"t{task.time:05.2f}")
        m = task.metrics()
        events = [e for e in task.event_log if e.get("type") != "planner_trace"]
        print(json.dumps({k: m[k] for k in ("task_success", "temporal_success", "breath_times", "beats", "beat_errors", "temporal_violations", "rhythm_outcome")}, indent=1))
        for e in events:
            print(e)
        (a.output / "trace.json").write_text(json.dumps(trace, default=float))
        (a.output / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
