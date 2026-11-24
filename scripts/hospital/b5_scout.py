"""B5 placement scouting: one reset, several camera views (no physics stepping).

python scripts/hospital/b5_scout.py --output results/hospital/B5/scout \
    --view side@24 0.9 -1.2 1.5 0.4 0.0 0.9 --view top 0.3 0.0 2.6 0.45 0.0 0.8 \
    [--translate 2.29 -0.95 0 --yaw 180] [--iso 250] [--focal 21]
Without --translate/--yaw the module's own ward pose is used.
"""

import argparse
from importlib import import_module
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="B5")
    p.add_argument("--level", default="hard")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--translate", nargs=3, type=float, default=None)
    p.add_argument("--yaw", type=float, default=None)
    p.add_argument("--iso", type=float, default=None)
    p.add_argument("--focal", type=float, default=None)
    p.add_argument("--task-camera", action="store_true", help="also save the task's own camera")
    p.add_argument(
        "--view",
        nargs=7,
        action="append",
        default=[],
        metavar=("NAME", "X", "Y", "Z", "LX", "LY", "LZ"),
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from PIL import Image
    from pxr import Gf

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.tasks.registry import make_task

    module = import_module(f"roboclock.envs.hospital.bedside.{make_task(a.task).bedside_module}")
    original = module.build_layout

    def build_layout(b):
        original(b)
        if a.translate is not None or a.yaw is not None:
            pose = dict(
                b.info.get("ward_pose") or {"usd": "assets/hospital/ward_02_preview_left.usdc"}
            )
            if a.translate is not None:
                pose["translate"] = a.translate
            if a.yaw is not None:
                pose["yaw"] = a.yaw
            b.info["ward_pose"] = pose
            b.info["hide_bench"] = True

    module.build_layout = build_layout
    env = HospitalEnv(task_id=a.task, difficulty=a.level, runtime=runtime)
    if a.iso is not None:
        env.config["tasks"].setdefault(a.task, {})["film_iso"] = a.iso
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        for seed in a.seeds:
            obs, _ = env.reset(seed=seed)
            if a.task_camera:
                Image.fromarray(obs["rgb"]).save(a.output / f"task_s{seed}.png")
            camera = env.backend.camera
            if a.focal is not None:
                camera.set_focal_length(a.focal / 1000)
            for name, *values in a.view:
                x, y, z, lx, ly, lz = map(float, values)
                if "@" in name:  # NAME@FOCAL sets this view's focal length in mm
                    name, focal = name.split("@")
                    camera.set_focal_length(float(focal) / 1000)
                matrix = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(x, y, z), Gf.Vec3d(lx, ly, lz), Gf.Vec3d(0, 0, 1)
                )
                orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
                camera.set_world_pose(np.array([x, y, z]), orientation, camera_axes="usd")
                for _ in range(4):
                    env.backend.world.render()
                rgb = env.backend.observation()["rgb"]
                Image.fromarray(rgb).save(a.output / f"{name}_s{seed}.png")
                print("saved", a.output / f"{name}_s{seed}.png", flush=True)
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
