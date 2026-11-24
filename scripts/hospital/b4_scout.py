"""B4 placement scouting: render camera views at reset and at chosen times after the order.

Each --view is "x y z  lx ly lz  focal_mm" (camera position, look-at point, focal length). The task
camera (layout.info["stopcock_scene"]["camera"]) is always view 0. With --at T [T ...] the oracle runs
and every view is rendered again T seconds after the order (and at the end with --run). Rendering
never advances the physics clock.

python scripts/hospital/b4_scout.py --output results/hospital/B4/scout/v1 \
    --view 0.9 -0.3 1.3 0.41 0.06 0.93 24 --at 0.8 1.6 --run
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", nargs=7, type=float, action="append", default=[])
    p.add_argument("--camera", nargs=7, type=float, default=None, help="Override the task camera")
    p.add_argument("--iso", type=float, default=None)
    p.add_argument("--at", nargs="+", type=float, default=[])
    p.add_argument("--run", action="store_true")
    p.add_argument("--resolution", nargs=2, type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.camera:
        from roboclock.envs.hospital.bedside import stopcock as module

        module.CAMERA = {
            "position": list(a.camera[:3]), "look_at": list(a.camera[3:6]), "focal_mm": a.camera[6]
        }
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from PIL import Image
    from pxr import Gf

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace
    from roboclock.simulators.isaac.rendering import capture_camera

    env = HospitalEnv(task_id="B4", difficulty=a.level, oracle=True, runtime=runtime)
    if a.iso is not None:
        env.config["tasks"]["B4"]["film_iso"] = a.iso
    if a.resolution:
        env.config["sim"]["camera_resolution"] = a.resolution

    def shoot(tag):
        backend = env.backend
        task_view = backend.layout.info["stopcock_scene"]["camera"]
        views = [[*task_view["position"], *task_view["look_at"], task_view.get("focal_mm", 21.0)]]
        for k, v in enumerate(views + a.view):
            position = np.asarray(v[:3]) + (backend.layout.camera_jitter if k == 0 else 0.0)
            matrix = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*position.tolist()), Gf.Vec3d(*v[3:6]), Gf.Vec3d(0, 0, 1)
            )
            q = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
            backend.camera.set_world_pose(position, q, camera_axes="usd")
            backend.camera.set_focal_length(v[6] / 1000)
            capture_camera(8)
            rgb = np.asarray(backend.camera.get_rgba()[..., :3], dtype=np.uint8)
            Image.fromarray(rgb).save(a.output / f"{tag}_view{k}.png")
        # Restore the task camera for later observations.
        view = views[0]
        position = np.asarray(view[:3]) + backend.layout.camera_jitter
        matrix = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view[3:6]), Gf.Vec3d(0, 0, 1)
        )
        q = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
        backend.camera.set_world_pose(position, q, camera_axes="usd")
        backend.camera.set_focal_length(view[6] / 1000)
        print("saved", tag, flush=True)

    try:
        obs, _ = env.reset(seed=a.seed)
        Image.fromarray(obs["rgb"]).save(a.output / "obs0.png")
        shoot("initial")
        if a.run or a.at:
            task = env.task
            policy = OraclePickPlace(env)
            policy.reset(obs)
            limit = task.spec.event_time + (
                task.spec.response_limit + 1.0 if a.run else max(a.at) + 0.01
            )
            pending = sorted(a.at)
            while task.time < limit and not env.done:
                action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                env.backend.apply_action(action)
                task.action_count += 1
                env._ticks(env.config["sim"]["control_decimation"], render=False)
                if pending and task.event_time is not None:
                    if task.time - task.event_time >= pending[0]:
                        shoot(f"t{pending.pop(0):.2f}_{policy.planner.phase}")
            print("joint_success", task.joint_success, "physical", task.physical_success, flush=True)
            if a.run:
                shoot("final")
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
