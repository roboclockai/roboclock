"""Render a ward task's first frame with a trial ward pose, bench and lighting (placement scouting).

python scripts/hospital/ward_pose_preview.py --task C1 --usd assets/hospital/ward_02_preview_left.usdc \
    --translate -2.63 2.0 -0.07 --surface 0.53 0 0.70 0.56 1.4 0.1 --light 3 \
    --camera -0.6 1.1 1.65 --look-at 0.5 0 0.85 --output results/hospital/ward_pose/C1.png
"""

import argparse
from importlib import import_module
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--usd", default=None)
    p.add_argument("--translate", nargs=3, type=float, default=None)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--surface", nargs=6, type=float, default=None, help="center xyz, size xyz")
    p.add_argument("--keep-bench", action="store_true")
    p.add_argument("--light", type=float, default=None, help="default: scene.bedside_ward_light")
    p.add_argument("--iso", type=float, default=None, help="visuals.film_iso exposure")
    p.add_argument("--camera", nargs=3, type=float, default=None)
    p.add_argument("--look-at", nargs=3, type=float, default=None)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.tasks.registry import make_task

    module = import_module(f"roboclock.envs.hospital.bedside.{make_task(a.task).bedside_module}")
    original = module.build_layout

    def build_layout(b):
        original(b)
        if a.usd or a.translate or a.yaw:  # otherwise keep the task's own placement
            pose = dict(b.info.get("ward_pose") or {})
            pose["yaw"] = a.yaw
            if a.usd:
                pose["usd"] = a.usd
            if a.translate:
                pose["translate"] = a.translate
            b.info["ward_pose"] = pose
            b.info["hide_bench"] = not a.keep_bench
        if a.surface:
            b.info["work_surface"] = {"center": a.surface[:3], "size": a.surface[3:]}

    module.build_layout = build_layout
    env = HospitalEnv(task_id=a.task, difficulty=a.level, runtime=runtime)
    scene = env.config["scene"]
    if a.iso is not None:
        env.config["visuals"]["film_iso"] = a.iso
    if a.light is not None:
        scene["bedside_ward_light"] = a.light
    if a.camera:
        scene["bedside_camera_position"] = a.camera
    if a.look_at:
        scene["bedside_camera_look_at"] = a.look_at
    try:
        obs, _ = env.reset(seed=a.seed)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(obs["rgb"]).save(a.output)
        print("saved", a.output)
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
