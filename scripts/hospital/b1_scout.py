"""B1 oxygen desaturation scouting: save the robot's camera images and extra views, optionally
while the oracle runs.

Each --view is "x y z  lx ly lz  focal_mm" (camera position, look-at point, focal length), rendered
with the recording camera. With --at T [T ...] the oracle runs and the images are saved again T
seconds after the alarm (and at the end with --run). Rendering never advances the physics clock.

python scripts/hospital/b1_scout.py --output results/hospital/B1/scout/v1 \
    --view -0.25 -0.75 1.75 0.60 0.25 1.35 22 --at 1.5 3.0 --run
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", nargs=7, type=float, action="append", default=[])
    p.add_argument("--iso", type=float, default=None)
    p.add_argument("--at", nargs="+", type=float, default=[])
    p.add_argument("--before", nargs="+", type=float, default=[], help="Seconds before the alarm")
    p.add_argument("--run", action="store_true")
    p.add_argument("--trace", action="store_true", help="Print the oracle's state every 0.2 s")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from PIL import Image
    from pxr import Gf

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace
    from roboclock.simulators.isaac.rendering import capture_camera

    env = HospitalEnv(task_id="B1", difficulty=a.level, oracle=True, runtime=runtime)
    env.config["sim"]["task_camera"] = bool(a.view)
    if a.iso is not None:
        env.config["tasks"]["B1"]["film_iso"] = a.iso

    def shoot(tag):
        obs = env.get_obs() if hasattr(env, "get_obs") else None
        if obs is not None:
            for key, value in obs.items():
                if key.endswith("rgb") and getattr(value, "ndim", 0) == 3:
                    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(a.output / f"{tag}_{key}.png")
        camera = getattr(env.backend, "task_camera", None)
        if camera is not None:
            for k, v in enumerate(a.view):
                matrix = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*v[:3]), Gf.Vec3d(*v[3:6]), Gf.Vec3d(0, 0, 1)
                )
                q = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
                camera.set_world_pose(np.asarray(v[:3]), q, camera_axes="usd")
                camera.set_focal_length(v[6] / 1000)
                capture_camera(8)
                rgb = np.asarray(camera.get_rgba()[..., :3], dtype=np.uint8)
                Image.fromarray(rgb).save(a.output / f"{tag}_view{k}.png")
        print("saved", tag, flush=True)

    try:
        obs, _ = env.reset(seed=a.seed)
        task = env.task
        print("instruction:", task.gen_instruction(), flush=True)
        print("event", task.spec.event_time, "limit", task.spec.response_limit, flush=True)
        head = np.asarray(obs["rgb"], dtype=float)
        luma = head @ [0.299, 0.587, 0.114]
        print("head mean luma", round(float(luma.mean()), 1), "clipped %",
              round(100 * float((head.min(axis=-1) >= 254).mean()), 2), flush=True)
        shoot("reset")
        if a.run or a.at or a.before:
            policy = OraclePickPlace(env)
            policy.reset(obs)
            pending = sorted(a.at)
            before = sorted(a.before, reverse=True)
            last_trace = -1.0
            while not env.done and task.time < task.spec.max_time:
                action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                env.backend.apply_action(action)
                task.action_count += 1
                env._ticks(env.config["sim"]["control_decimation"], render=False)
                if before and task.event_time is None and task.spec.event_time - task.time <= before[0]:
                    shoot(f"pre{before.pop(0):.2f}")
                if a.trace and task.time - last_trace >= 0.2:
                    last_trace = task.time
                    c = task.previous.controls if task.previous is not None else {}
                    print(
                        f"t={task.time:6.2f} phase={policy.planner.phase:10s} spo2={task.spo2:5.1f} "
                        f"angle={np.degrees(c.get('knob_angle_rad', 0.0)):7.1f} flow={task.float_flow:5.2f} "
                        f"contact={c.get('knob_contact')} grasp={c.get('knob_grasped')} "
                        f"ee={np.round(task.previous.ee_position, 3).tolist() if task.previous else None} "
                        f"j={np.round(env.backend.joint_positions()[:6], 2).tolist()}",
                        flush=True,
                    )
                if pending and task.event_time is not None and task.time - task.event_time >= pending[0]:
                    shoot(f"t{pending.pop(0):.2f}_{policy.planner.phase}")
                if not a.run and not pending and not before and task.event_time is not None:
                    break
            m = task.metrics()
            print({k: m[k] for k in (
                "task_success", "temporal_success", "joint_success", "reaction_latency",
                "knob_moved_time", "band_entry_time", "settled_time", "float_flow_l_min",
                "knob_angle_rad", "temporal_violations", "physical_failures", "final_spo2", "min_spo2",
            )}, flush=True)
            if a.run:
                shoot("final")
    except Exception:  # runtime.close() exits the process before a traceback would print
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
