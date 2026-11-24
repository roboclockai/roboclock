"""B6 placement scouting: one reset, several camera views (frame 0), optional trial overrides.

python scripts/hospital/b6_scout.py --output results/hospital/B6/scout \
    --tuck 0.0 -0.16 0.10 --view wide 0.05 -1.15 1.75 0.42 0.0 1.0 --focal 24
Without overrides the task's own placement, rest pose and camera are kept; without --view only the
task camera is saved (task.png).
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--translate", nargs=3, type=float, help="trial ward translate")
    p.add_argument("--yaw", type=float, help="trial ward yaw (deg)")
    p.add_argument("--hide", nargs="*", default=None, help="extra ward source objects to hide")
    p.add_argument(
        "--tuck", nargs=3, type=float, help="trial rest pose of the pressing hand from its mount"
    )
    p.add_argument("--mount", nargs=3, type=float, help="trial arm-0 mount (robot frame); moves the whole rig")
    p.add_argument("--iso", type=float)
    p.add_argument("--light", type=float)
    p.add_argument("--subframes", type=int, default=8)
    p.add_argument("--focal", type=float, help="focal length (mm) for the extra views")
    p.add_argument("--resolution", nargs=2, type=int, help="camera width height")
    p.add_argument(
        "--view",
        nargs=7,
        action="append",
        default=[],
        metavar=("NAME", "PX", "PY", "PZ", "LX", "LY", "LZ"),
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.bedside import pump_panel
    from roboclock.envs.hospital.env import HospitalEnv

    if a.tuck:
        pump_panel.TUCK = tuple(a.tuck)
    original = pump_panel.build_layout

    def build_layout(b):
        original(b)
        pose = dict(b.info.get("ward_pose") or {})
        if a.translate:
            pose["translate"] = a.translate
        if a.yaw is not None:
            pose["yaw"] = a.yaw
        if a.hide is not None:
            pose["hide"] = list(pose.get("hide", [])) + a.hide
        if pose:
            b.info["ward_pose"] = pose
        return b

    pump_panel.build_layout = build_layout
    env = HospitalEnv(task_id="B6", difficulty=a.level, runtime=runtime)
    if a.mount:  # the common rig: arm 0 sits half a separation to the rig's right (-y) of its origin
        rig = env.config["robot"]["rig"]
        x, y, z = a.mount
        rig["tasks"].setdefault("B6", {})["origin"] = [x, y + rig["separation"] / 2, z]
    if a.iso is not None:
        env.config["tasks"].setdefault("B6", {})["film_iso"] = a.iso
    if a.light is not None:
        env.config["scene"]["bedside_ward_light"] = a.light
    if a.resolution:
        env.config["sim"]["camera_resolution"] = list(a.resolution)
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        obs, _ = env.reset(seed=a.seed)
        Image.fromarray(obs["rgb"]).save(a.output / "task.png")
        print("saved", a.output / "task.png", flush=True)
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from pxr import Gf

        backend = env.backend
        for name, *values in a.view:
            values = [float(v) for v in values]
            position, look_at = np.array(values[:3]), values[3:]
            matrix = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*position.tolist()), Gf.Vec3d(*look_at), Gf.Vec3d(0, 0, 1)
            )
            orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
            backend.camera.set_world_pose(position, orientation, camera_axes="usd")
            if a.focal:
                backend.camera.set_focal_length(a.focal / 1000)
            backend.capture_camera(subframes=a.subframes)
            rgba = backend.camera.get_rgba()
            Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8)).save(
                a.output / f"{name}.png"
            )
            print("saved", a.output / f"{name}.png", flush=True)
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
