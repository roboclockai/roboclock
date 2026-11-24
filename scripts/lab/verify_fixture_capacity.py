"""Separate physical fixture experiments; no benchmark scores are collected.

Run with Isaac Sim's Python environment from this repository:
    python scripts/verify_fixture_capacity.py --output results/lab/fixtures
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from roboclock.config import PACKAGE_ROOT
from roboclock.envs.lab.env import LabEnv
from roboclock.envs.lab.mdp.events import SceneCommand
from roboclock.envs.lab.mdp.geometry import seated
from roboclock.envs.lab.policies.trajectories import sorting_landing
from roboclock.simulators.isaac.runtime import IsaacRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fixture_capacity"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = Path(__file__).with_name("high_capacity_config.yaml")
    rows = []
    manifest = {
        "purpose": "50-sample bin gravity seating and actual moving-fixture tracking",
        "benchmark_scores_used": False,
        "code_sha256": {
            str(p.relative_to(PACKAGE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PACKAGE_ROOT.rglob("*.py"))
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with IsaacRuntime() as runtime:
        env = LabEnv(runtime=runtime, config_path=config_path)
        try:
            for seed in range(3):
                env.reset(seed=seed, options={"task_id": "C1", "difficulty": "easy"})
                task = env.task
                for name, body in env.backend.objects.items():
                    if name in env.backend.supports:
                        env.backend.command(SceneCommand("activate", name, None))
                    target = task.layout.targets[task.layout.assignments[name]]
                    start = sorting_landing(task, name, target)
                    start[2] = target.rim_z + task.layout.dimensions[name][1] / 2 + 0.018
                    body.set_world_pose(start, np.array([1.0, 0.0, 0.0, 0.0]))
                    body.set_linear_velocity(np.zeros(3))
                    body.set_angular_velocity(np.zeros(3))
                for _ in range(120):
                    env.backend.tick(render=False)
                samples = {}
                for name, obj in env.backend.state().objects.items():
                    assignment = task.layout.assignments[name]
                    samples[name] = {
                        "seated": bool(seated(task, name, obj, task.layout.targets[assignment])),
                        "floor_contact": assignment in obj.fixture_contacts,
                        "speed": float(np.linalg.norm(obj.velocity)),
                        "position": obj.position.tolist(),
                    }
                row = {
                    "experiment": "50_sample_sorting_bin_gravity_seating",
                    "seed": seed,
                    "simulated_seconds": 1.0,
                    "samples": samples,
                    "pass": all(
                        s["seated"] and s["floor_contact"] and s["speed"] < 0.08
                        for s in samples.values()
                    ),
                }
                rows.append(row)
                print(f"C1 bin capacity seed={seed}: {row['pass']}", flush=True)
            for difficulty in ("easy", "medium", "hard"):
                for seed in range(3):
                    env.reset(seed=seed, options={"task_id": "A4", "difficulty": difficulty})
                    task = env.task
                    base = env.backend.target_parts["destination"][0][1]
                    dt = env.config["sim"]["physics_dt"]
                    errors = []
                    for _ in range(120):
                        for command in task.advance(dt):
                            env.backend.command(command)
                        env.backend.tick(render=False)
                        actual = env.backend.kinematic_views["destination"].get_transforms()[0, :3]
                        expected = base + np.array([task.fixture_offset, 0.0, 0.0])
                        errors.append(float(np.linalg.norm(actual - expected)))
                    row = {
                        "experiment": "actual_kinematic_fixture_tracking",
                        "difficulty": difficulty,
                        "seed": seed,
                        "simulated_seconds": 1.0,
                        "max_position_error_m": max(errors),
                        "pass": max(errors) < 0.0001,
                    }
                    rows.append(row)
                    print(f"A4 {difficulty} seed={seed}: {row}", flush=True)
            (args.output / "physics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
            summary = {
                "experiments": len(rows),
                "passed": sum(r["pass"] for r in rows),
                "benchmark_scores_used": False,
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(summary, flush=True)
            runtime.exit_code = 0 if all(r["pass"] for r in rows) else 1
        finally:
            env.close()


if __name__ == "__main__":
    main()
