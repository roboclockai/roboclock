"""Physics + oracle only lab runs: `sim.no_render: true` (no cameras, no render calls).

Runs the same oracle and the same env.step loop as `roboclock eval --policy oracle`
(control rate, zero injected latency). With `sim.no_render` the backend creates no
camera, never renders and observations carry zero images of the spec shapes. The
oracle uses privileged state, so its actions do not depend on pixels. The measured rig
check (bases, link6 vs X5A.urdf) still runs at every reset. Use the rendered
`roboclock eval` only for final acceptance.

    python scripts/lab/headless_oracle.py --tasks A2 B1 --difficulties easy medium hard \
        --seeds 0 --output /tmp/run.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime(headless=True) as runtime:
        from roboclock.envs.lab.env import LabEnv
        from roboclock.envs.lab.layout import apparatus_frame
        from roboclock.envs.lab.policies.oracle import OraclePickPlace

        env = LabEnv(
            task_id=args.tasks[0],
            difficulty=args.difficulties[0],
            runtime=runtime,
            config_path=args.config,
            oracle=True,
        )
        env.config["sim"]["no_render"] = True  # shared with the backend
        try:
            with args.output.open("a", encoding="utf-8") as stream:
                for task_id in args.tasks:
                    for difficulty in args.difficulties:
                        for seed in args.seeds:
                            started = time.perf_counter()
                            obs, _ = env.reset(
                                seed=seed, options={"task_id": task_id, "difficulty": difficulty}
                            )
                            policy = OraclePickPlace(env)
                            policy.reset(obs)
                            while not env.done:
                                obs = env.step(np.asarray(policy.act(obs), dtype=np.float32))[0]
                            row = env.task.metrics()
                            row["wall_seconds"] = time.perf_counter() - started
                            row["ik_failures"] = list(policy.ik_failures)
                            row["mode"] = "sim.no_render"
                            row["apparatus_placement"] = apparatus_frame(env.task.layout).info()
                            row["sample_0_initial"] = env.task.layout.objects[
                                "sample_0"
                            ].position.tolist()
                            meta = env.robodojo_metadata()
                            row["rig_check"] = meta["rig_check"]
                            row["rates"] = meta["rates"]
                            row["observation_shapes"] = {
                                key: list(np.shape(value))
                                for key, value in obs.items()
                                if key.endswith("rgb")
                            }
                            row["events"] = env.task.event_log[-40:]
                            stream.write(json.dumps(row, default=str) + "\n")
                            stream.flush()
                            print(
                                f"RESULT {task_id} {difficulty} seed={seed} "
                                f"physical={row['task_success']} "
                                f"temporal={row['temporal_success']} "
                                f"joint={row['joint_success']} "
                                f"correct={row.get('correct_operations')} "
                                f"attempted={row.get('attempted_operations')} "
                                f"collisions={row.get('collision_count')} "
                                f"t={row.get('completion_time')} "
                                f"wall={row['wall_seconds']:.0f}s",
                                flush=True,
                            )
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
