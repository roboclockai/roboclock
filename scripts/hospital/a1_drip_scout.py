"""A1 drip chamber: render frame 0 for trial ward poses / cameras in one simulator session.

Each --view is "name:cx,cy,cz:lx,ly,lz:focal_mm" (camera position, look-at, focal length in the
robot frame before the seed shift). --translate/--yaw/--hide override the module's WARD_POSE.

python scripts/hospital/a1_drip_scout.py --translate 0.5 -0.2 0 --yaw 90 \
    --view top:0.6,0,4.5:0.6,0.01,0:18 --output results/hospital/A1/drip/scout
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--translate", nargs=3, type=float, default=None)
    p.add_argument("--yaw", type=float, default=None)
    p.add_argument("--hide", nargs="*", default=None)
    p.add_argument("--iso", type=float, default=None)
    p.add_argument("--time", type=float, default=0.0, help="also step the oracle to this time")
    p.add_argument("--view", action="append", default=[])
    p.add_argument(
        "--set", action="append", default=[], help="module constant override NAME=python"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    try:
        from PIL import Image

        from roboclock.envs.hospital.bedside import window_drip as module
        from roboclock.envs.hospital.env import HospitalEnv

        for item in a.set:
            name, _, value = item.partition("=")
            setattr(module, name, eval(value))  # noqa: S307 (scouting tool)
        if a.translate is not None:
            module.WARD_POSE["translate"] = list(a.translate)
        if a.yaw is not None:
            module.WARD_POSE["yaw"] = a.yaw
        if a.hide is not None:
            module.WARD_POSE["hide"] = list(a.hide)
        env = HospitalEnv(task_id="A1", difficulty=a.level, runtime=runtime, oracle=True)
        if a.iso is not None:
            env.config["tasks"]["A1"]["film_iso"] = a.iso
        try:
            obs, _ = env.reset(seed=a.seed)
            Image.fromarray(obs["rgb"]).save(a.output / "frame0.png")
            if a.time > 0:
                from roboclock.envs.hospital.policies.oracle import OraclePickPlace

                policy = OraclePickPlace(env)
                policy.reset(obs)
                task = env.task
                while task.time < a.time and not env.done:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
            import numpy as np
            from isaacsim.core.utils.rotations import gf_quat_to_np_array
            from pxr import Gf

            from roboclock.simulators.isaac.rendering import capture_camera

            shift = np.array([*env.task.layout.info["drip"]["shift"], 0.0])
            for view in a.view:
                name, position, target, focal = view.split(":")
                position = np.array([float(v) for v in position.split(",")]) + shift
                target = np.array([float(v) for v in target.split(",")]) + shift
                matrix = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*position.tolist()),
                    Gf.Vec3d(*target.tolist()),
                    Gf.Vec3d(0, 1, 0)
                    if abs(position[0] - target[0]) + abs(position[1] - target[1]) < 1e-3
                    else Gf.Vec3d(0, 0, 1),
                )
                orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
                env.backend.camera.set_world_pose(position, orientation, camera_axes="usd")
                env.backend.camera.set_focal_length(float(focal) / 1000)
                capture_camera(4)
                rgba = env.backend.camera.get_rgba()
                Image.fromarray(np.asarray(rgba[..., :3], dtype=np.uint8)).save(
                    a.output / f"{name}.png"
                )
                print("saved", a.output / f"{name}.png", flush=True)
            if a.time > 0:
                print(
                    "controls", {k: v for k, v in env.backend.state().controls.items()}, flush=True
                )
        finally:
            env.close()
    except BaseException:
        runtime.exit_code = 1
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
