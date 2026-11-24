"""A3 syringe pump cutoff: scouting renders (task camera and trial views) and an optional oracle run.

python scripts/hospital/a3w_scout.py --output results/hospital/A3/a3w_scout1 \
    --view 0.36 0.80 1.30 0.50 -0.03 0.90 26 --run --frames 0 3 6.5 7.4 9
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--view",
        nargs=7,
        type=float,
        action="append",
        default=[],
        help="extra camera: position xyz, look-at xyz, focal mm (robot frame, before shift)",
    )
    p.add_argument("--iso", type=float, default=None)
    p.add_argument("--run", action="store_true", help="run the oracle and save frames at --frames")
    p.add_argument("--frames", nargs="*", type=float, default=[])
    p.add_argument(
        "--no-action", action="store_true", help="hold the hover pose (late / no action)"
    )
    p.add_argument("--resolution", nargs=2, type=int, default=None)
    p.add_argument(
        "--cadence",
        type=int,
        default=0,
        help="also render every N control steps, like the video recorder (0: only at --frames)",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    env = HospitalEnv(task_id="A3", difficulty=a.level, oracle=True, runtime=runtime)
    if a.iso is not None:
        env.config["tasks"]["A3"]["film_iso"] = a.iso
    if a.resolution:
        env.config["sim"]["camera_resolution"] = a.resolution
    tag = f"A3_{a.level}_{a.seed}"
    try:
        obs, _ = env.reset(seed=a.seed)
        Image.fromarray(obs["rgb"]).save(a.output / f"{tag}_t0.png")
        backend = env.backend
        if a.view:
            from isaacsim.core.utils.rotations import gf_quat_to_np_array
            from pxr import Gf

            shift = np.asarray(
                env.task.layout.info["window_syringe"]["camera"]["position"][:2]
            ) - np.asarray(
                __import__(
                    "roboclock.envs.hospital.bedside.window_syringe", fromlist=["CAMERA"]
                ).CAMERA["position"][:2]
            )
            for i, view in enumerate(a.view):
                position = np.array([view[0] + shift[0], view[1] + shift[1], view[2]])
                look_at = [view[3] + shift[0], view[4] + shift[1], view[5]]
                matrix = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*position.tolist()), Gf.Vec3d(*look_at), Gf.Vec3d(0, 0, 1)
                )
                orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
                backend.camera.set_world_pose(position, orientation, camera_axes="usd")
                backend.camera.set_focal_length(view[6] / 1000)
                for _ in range(6):
                    backend.world.render()
                Image.fromarray(np.asarray(backend.camera.get_rgba()[..., :3])).save(
                    a.output / f"{tag}_view{i}.png"
                )
            from roboclock.envs.hospital.bedside.window_syringe import _camera

            _camera(backend)
        if a.run:
            policy = OraclePickPlace(env)
            policy.reset(obs)
            task = env.task
            pending = sorted(a.frames)
            hover = None
            while not env.done:
                if a.no_action:
                    if hover is None:
                        hover = env.backend.joint_positions().copy()
                    action = hover
                else:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                env.backend.apply_action(action)
                task.action_count += 1
                env._ticks(env.config["sim"]["control_decimation"], render=False)
                steps = getattr(env, "_a3w_steps", 0) + 1
                env._a3w_steps = steps
                if (
                    a.cadence
                    and steps % a.cadence == 0
                    and not (pending and task.time >= pending[0] - 1e-9)
                ):
                    from roboclock.simulators.isaac.rendering import capture_camera

                    capture_camera(4)
                if pending and task.time >= pending[0] - 1e-9:
                    from roboclock.simulators.isaac.rendering import capture_camera

                    stamp = pending.pop(0)
                    for suffix in (
                        "",
                        "_next",
                    ):  # the next render shows visibility switches made before it
                        capture_camera(4)
                        rgb = np.asarray(env.backend.camera.get_rgba()[..., :3])
                        Image.fromarray(rgb).save(a.output / f"{tag}_t{stamp:05.2f}{suffix}.png")
            from roboclock.simulators.isaac.rendering import capture_camera

            capture_camera(4)
            Image.fromarray(np.asarray(env.backend.camera.get_rgba()[..., :3])).save(
                a.output / f"{tag}_end.png"
            )
            m = task.metrics()
            keep = [
                "temporal_success",
                "task_success",
                "window_open",
                "window_close",
                "final_quantity",
                "quantity_outcome",
                "device_moved_time",
                "temporal_violations",
            ]
            result = {k: m.get(k) for k in keep}
            result["action_time"] = task.action_time
            result["time"] = task.time
            result["events"] = [e for e in task.event_log if e.get("type") != "planner_trace"]
            (a.output / f"{tag}.json").write_text(json.dumps(result, indent=2, default=str))
            print(json.dumps(result, indent=1, default=str))
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
