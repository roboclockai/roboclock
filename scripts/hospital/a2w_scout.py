"""A2 (window_bags) scouting renders: frame 0 from the task camera plus optional extra views.

python scripts/hospital/a2w_scout.py --output results/hospital/A2/w_scout1 \
    --view over 1.6 -1.2 2.6 0.4 0.1 0.8 --times 3 7 9

--set NAME=VALUE overrides a module constant (python literal) before the layout is built.
--times runs the privileged oracle (or --idle: no action) and saves task-camera frames at those
simulation times.
"""

import argparse
import ast
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--view", nargs=7, action="append", default=[], metavar=("NAME", "X", "Y", "Z", "LX", "LY", "LZ"))
    p.add_argument("--focal", type=float, default=None, help="focal length (mm) for extra views")
    p.add_argument("--times", nargs="*", type=float, default=[])
    p.add_argument("--idle", action="store_true", help="no robot action (late case)")
    p.add_argument("--iso", type=float, default=None)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    import numpy as np
    from PIL import Image

    from roboclock.envs.hospital.bedside import window_bags as module
    from roboclock.envs.hospital.env import HospitalEnv

    for item in a.set:
        key, value = item.split("=", 1)
        setattr(module, key, ast.literal_eval(value))
    env = HospitalEnv(task_id="A2", difficulty=a.level, runtime=runtime, oracle=True)
    if a.iso is not None:
        env.config["visuals"]["film_iso"] = a.iso
        env.config["tasks"]["A2"]["film_iso"] = a.iso
    try:
        obs, _ = env.reset(seed=a.seed)
        Image.fromarray(obs["rgb"]).save(a.output / "frame0.png")
        task = env.task
        print("window", task.window_open, task.window_close, "max", task.spec.max_time, flush=True)
        backend = env.backend
        if a.view:
            from isaacsim.core.utils.rotations import gf_quat_to_np_array
            from pxr import Gf

            pose = backend.camera.get_world_pose(camera_axes="usd")
            for name, *values in a.view:
                pos, look = [float(v) for v in values[:3]], [float(v) for v in values[3:]]
                m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1))
                q = gf_quat_to_np_array(m.GetInverse().ExtractRotationQuat())
                backend.camera.set_world_pose(np.array(pos), q, camera_axes="usd")
                if a.focal:
                    backend.camera.set_focal_length(a.focal / 1000)
                backend.capture_camera(subframes=4)
                Image.fromarray(env.render()).save(a.output / f"view_{name}.png")
            backend.camera.set_world_pose(*pose, camera_axes="usd")
            backend.camera.set_focal_length(module.CAMERA["focal_mm"] / 1000)
        if a.times:
            from roboclock.envs.hospital.policies.oracle import OraclePickPlace

            policy = OraclePickPlace(env)
            policy.reset(obs)
            hold = env.backend.joint_positions().copy()
            pending = sorted(a.times)
            while pending and not env.done:
                action = hold if a.idle else policy.act_internal({"joint_positions": env.backend.joint_positions()})
                env.backend.apply_action(action)
                env.task.action_count += 1
                env._ticks(env.config["sim"]["control_decimation"], render=False)
                if task.time + 1e-6 >= pending[0]:
                    backend.capture_camera(subframes=4)
                    Image.fromarray(env.render()).save(a.output / f"t{pending.pop(0):05.2f}.png")
            m = task.metrics()
            print(
                {k: m.get(k) for k in ("task_success", "temporal_success", "action_time",
                                         "window_open", "window_close", "final_quantity",
                                         "quantity_outcome", "device_moved_time")},
                flush=True,
            )
            print([e for e in task.event_log if e.get("type") != "planner_trace"], flush=True)
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
