"""Check imported C1 holes with a gravity seating probe, separate from benchmark scores."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/lab/c1-rack-probe.json"))
    args = parser.parse_args()
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        from roboclock.envs.lab.env import LabEnv
        from roboclock.envs.lab.mdp.geometry import seated
        from roboclock.envs.lab.validation import _scene_checks

        env = LabEnv(runtime=runtime, oracle=True)
        try:
            env.reset(seed=0, options={"task_id": "C1"})
            errors = _scene_checks(env)
            task = env.task
            for name, body in env.backend.objects.items():
                target = task.layout.targets[task.layout.assignments[name]]
                start = target.position.copy()
                start[2] = target.rim_z + 0.07 + 0.018
                body.set_world_pose(start, np.array([1.0, 0.0, 0.0, 0.0]))
                body.set_linear_velocity(np.zeros(3))
                body.set_angular_velocity(np.zeros(3))
            for _ in range(240):
                env.backend.tick(render=False)
            objects = {}
            for name, obj in env.backend.state().objects.items():
                assignment = task.layout.assignments[name]
                target = task.layout.targets[assignment]
                objects[name] = {
                    "seated": bool(seated(task, name, obj, target)),
                    "floor_contact": assignment in obj.fixture_contacts,
                    "position": obj.position.tolist(),
                    "target": target.position.tolist(),
                    "speed": float(np.linalg.norm(obj.velocity)),
                }
            result = {
                "purpose": "gravity fixture probe; no benchmark scores",
                "initial_scene_errors": errors,
                "samples": objects,
                "pass": not errors
                and all(
                    v["seated"] and v["floor_contact"] and v["speed"] < 0.08
                    for v in objects.values()
                ),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
            if not result["pass"]:
                raise RuntimeError("C1 rack gravity probe failed; inspect output")
        finally:
            env.close()


if __name__ == "__main__":
    main()
