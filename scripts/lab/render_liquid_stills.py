"""Render actual high-sample simulator stills without advancing the task clock."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from roboclock.envs.lab.config import load_config
from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/lab/previews"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1440])
    parser.add_argument(
        "--tasks", nargs="+", choices=["A1", "A2", "A3", "A4"], default=["A1", "A2", "A3", "A4"]
    )
    parser.add_argument(
        "--a1-final",
        action="store_true",
        help="Run the actual A1 controller before an additional final closeup",
    )
    parser.add_argument(
        "--a3-final",
        action="store_true",
        help="Run the actual A3 controller before a final fraction-collector closeup",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    config["sim"].update(overview_camera=True, overview_resolution=args.resolution)
    config["visuals"].update(
        renderer="PathTracing", path_tracing_spp=32, path_tracing_samples=args.samples
    )
    path = args.output / "stills_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    records = []
    with IsaacRuntime(renderer="PathTracing") as runtime:
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from pxr import Gf

        from roboclock.envs.lab.env import LabEnv
        from roboclock.simulators.isaac.rendering import capture_camera

        env = LabEnv(runtime=runtime, task_id="A1", config_path=path, oracle=True)
        for task in args.tasks:
            env.task_id = task
            env.reset(seed=0)
            views = [("overview", None, None)]
            if task == "A1":
                views += [
                    ("office_detail", [-0.15, -0.55, 1.60], [-1.25, -1.9, 0.91]),
                    ("bench_detail", [2.7, 0.0, 1.6], [1.25, -2.0, 0.70]),
                ]
            if task == "A3":
                views += [
                    ("column_detail", [0.86, 1.08, 1.51], [0.48, 0.21, 1.255]),
                    ("collector_detail", [0.94, 0.90, 1.13], [0.55, 0.21, 0.95]),
                ]
            if task == "A1" and args.a1_final:
                views.append(("final_closeup", None, None))
            if task == "A3" and args.a3_final:
                views.append(("final_closeup", [0.94, 0.90, 1.13], [0.40, 0.21, 0.95]))
            for label, position, target in views:
                if label == "final_closeup":
                    from roboclock.envs.lab.policies.oracle import OraclePickPlace
                    from roboclock.envs.lab.wet_bench import closeup_camera

                    policy = OraclePickPlace(env)
                    while not env.done:
                        action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                        env.backend.apply_action(action)
                        env.task.action_count += 1
                        env._ticks(env.config["sim"]["control_decimation"], render=False)
                    print(task + "_FINAL_METRICS", json.dumps(env.task.metrics()), flush=True)
                    if task == "A1":
                        closeup_camera(env.backend)
                if position is not None:
                    view = Gf.Matrix4d().SetLookAt(
                        Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)
                    )
                    orientation = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
                    env.backend.camera.set_world_pose(
                        np.array(position), orientation, camera_axes="usd"
                    )
                    env.backend.camera.set_focal_length(0.035)
                before = env.backend.world.current_time
                capture_camera(max(4, (args.samples + 31) // 32))
                assert env.backend.world.current_time == before, "Still capture advanced physics"
                output = args.output / f"{task}_{label}.png"
                Image.fromarray(env.backend.camera.get_rgba()[..., :3]).save(output)
                records.append(
                    dict(
                        file=output.name,
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        task=task,
                        physics_time=before,
                        task_time=env.task.time,
                        final_metrics=env.task.metrics() if label == "final_closeup" else None,
                        capture_physics_delta=0.0,
                    )
                )
                print("RENDERED", output, flush=True)
        env.backend.stage.GetRootLayer().Export(str(args.output / "laboratory_scene.usda"))
        (args.output / "stills_manifest.json").write_text(
            json.dumps(
                dict(
                    simulator="isaacsim==" + simulator_version(),
                    renderer="PathTracing",
                    samples_per_pixel=args.samples,
                    resolution=args.resolution,
                    stills=records,
                    purpose="Actual simulator stills; optional final views follow actual physical diagnostic episodes",
                    render_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                ),
                indent=2,
            )
        )
        env.close()


if __name__ == "__main__":
    main()
