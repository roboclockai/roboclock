"""Render controlled A3/A4 optical states in Isaac Sim, without policy claims."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from roboclock.envs.lab.config import load_config
from roboclock.simulators.isaac.runtime import IsaacRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/lab/previews/liquid-optics"))
    parser.add_argument("--tasks", nargs="+", default=["A3", "A4"], choices=["A3", "A4"])
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 960])
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    config = load_config()
    config["sim"].update(overview_camera=True, overview_resolution=args.resolution)
    config["visuals"].update(
        renderer="PathTracing", path_tracing_spp=32, path_tracing_samples=args.samples
    )
    path = args.output / "optics_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    records = []
    with IsaacRuntime(renderer="PathTracing") as runtime:
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from pxr import Gf

        from roboclock.envs.lab.env import LabEnv
        from roboclock.simulators.isaac.rendering import capture_camera

        env = LabEnv(runtime=runtime, task_id=args.tasks[0], config_path=path, oracle=True)

        def capture(label, position=None, target=None):
            if position is not None:
                view = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)
                )
                env.backend.camera.set_world_pose(
                    np.array(position),
                    gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat()),
                    camera_axes="usd",
                )
                env.backend.camera.set_focal_length(0.035)
            env.backend.liquid_scene.update(env.task.visual_state())
            before = env.backend.world.current_time
            capture_camera(max(4, (args.samples + 31) // 32))
            assert env.backend.world.current_time == before
            output = args.output / (label + ".png")
            Image.fromarray(env.backend.camera.get_rgba()[..., :3]).save(output)
            records.append(
                dict(
                    file=output.name,
                    task=env.task.task_id,
                    process_state=env.task.visual_state(),
                    capture_physics_delta=env.backend.world.current_time - before,
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                )
            )
            print("RENDERED", output, flush=True)

        for task in args.tasks:
            env.task_id = task
            env.reset(seed=0)
            capture(task + "_overview")
            if task == "A3":
                capture("A3_four_bands", [0.86, 1.08, 1.51], [0.48, 0.21, 1.255])
                env.task.time = 10.0
                capture("A3_bands_descending")
                env.task.time = env.task.intervals[-1][1] + 0.2
                for i in range(len(env.task.colors)):
                    env.task.collection[i + 1, i] = 13.5
                    env.task.tube_volumes[i + 1] = 13.5
                env.task.tube_volumes[0] = 60
                capture("A3_four_fractions", [0.99, 0.90, 1.13], [0.62, 0.21, 0.95])
            else:
                capture("A4_apparatus", [0.91, 1.60, 1.55], [0.50, 0.15, 1.15])
                for label, volume in [
                    ("before_endpoint", 16.0),
                    ("pale_pink_endpoint", 18.2),
                    ("excess_titrant", 20.0),
                ]:
                    env.task.volume = volume
                    env.task.flow = env.task.nominal_flow
                    env.task.time = volume / env.task.nominal_flow + 0.03
                    capture("A4_" + label, [0.68, 0.64, 1.04], [0.50, 0.15, 0.90])
        manifest = dict(
            purpose="Controlled optical validation states; process values are set explicitly. Not robot-policy episodes.",
            simulator="isaacsim==5.1.0",
            renderer="PathTracing",
            resolution=args.resolution,
            samples_per_pixel=args.samples,
            images=records,
            source_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path("src/roboclock/envs/lab/liquid_scene.py"),
                    Path("src/roboclock/envs/lab/chromatography_visuals.py"),
                    Path("src/roboclock/envs/lab/titration_visuals.py"),
                ]
            },
        )
        (args.output / "optics_manifest.json").write_text(json.dumps(manifest, indent=2))
        env.close()


if __name__ == "__main__":
    main()
