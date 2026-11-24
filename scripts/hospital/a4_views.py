"""Render A4's first frame from its task camera and from extra trial views in one Isaac session.

python scripts/hospital/a4_views.py --output results/hospital/A4/views \
    --view 0.98 -0.95 1.62 0.36 0.02 1.12 21 --view 0.2 -1.2 1.7 0.42 0.1 1.1 24

Each --view is camera xyz, look-at xyz and focal length (mm). Placement scouting only: the
episode clock never advances.
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iso", type=float, default=None, help="override visuals.film_iso")
    p.add_argument("--view", nargs=7, type=float, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    import numpy as np
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv

    env = HospitalEnv(task_id="A4", difficulty=a.level, runtime=runtime)
    if a.iso is not None:
        env.config["visuals"]["film_iso"] = a.iso
        env.config["tasks"]["A4"]["film_iso"] = a.iso
    try:
        obs, _ = env.reset(seed=a.seed)
        a.output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(obs["rgb"]).save(a.output / "task_camera.png")
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from pxr import Gf

        camera = env.backend.camera
        for k, (*xyz, lx, ly, lz, focal) in enumerate(a.view):
            matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*xyz[:3]), Gf.Vec3d(lx, ly, lz), Gf.Vec3d(0, 0, 1))
            orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
            camera.set_world_pose(np.array(xyz[:3]), orientation, camera_axes="usd")
            camera.set_focal_length(focal / 1000)
            for _ in range(8):
                env.backend.world.render()
            Image.fromarray(env.render()).save(a.output / f"view_{k}.png")
        print("saved", a.output)
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
