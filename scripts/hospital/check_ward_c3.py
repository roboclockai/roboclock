"""C3 crash-cart vial restock: stills, oracle checks over levels x seeds, and one oracle video.

Run with the Isaac Sim Python from the repository root, for example
    OMNI_KIT_ACCEPT_EULA=YES python scripts/hospital/check_ward_c3.py --levels easy medium hard --seeds 0 1 2
    OMNI_KIT_ACCEPT_EULA=YES python scripts/hospital/check_ward_c3.py --still-only --seeds 0
    OMNI_KIT_ACCEPT_EULA=YES python scripts/hospital/check_ward_c3.py --video --levels hard --seeds 0
"""

import argparse
import json
from pathlib import Path

import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def closeup(env, path, eye=(0.95, 0.95, 1.55), target=(0.38, 0.25, 0.95)):
    """A second view of the supply tray and drawer, then the benchmark camera is restored."""
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from PIL import Image
    from pxr import Gf

    camera = env.backend.camera
    position, orientation = camera.get_world_pose(camera_axes="usd")
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    camera.set_world_pose(
        np.array(eye), gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat()), camera_axes="usd"
    )
    for _ in range(12):
        env.backend.world.render()
    Image.fromarray(np.asarray(camera.get_rgba()[..., :3], dtype=np.uint8)).save(path)
    camera.set_world_pose(position, orientation, camera_axes="usd")
    for _ in range(4):
        env.backend.world.render()


def run_episode(env, seed, output, still_only=False):
    from PIL import Image

    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    obs, _ = env.reset(seed=seed)
    Image.fromarray(obs["rgb"]).save(output.with_name(output.name + "_initial.png"))
    if still_only:
        closeup(env, output.with_name(output.name + "_closeup.png"))
        closeup(env, output.with_name(output.name + "_drawer.png"), (0.95, 0.95, 1.40), (0.32, 0.78, 0.90))
        return None
    policy = OraclePickPlace(env)
    policy.reset(obs)
    task = env.task
    while not env.done:
        action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
        env.backend.apply_action(action)
        task.action_count += 1
        env._ticks(env.config["sim"]["control_decimation"], render=False)
    Image.fromarray(env.render()).save(output.with_name(output.name + "_final.png"))
    metrics = task.metrics()
    output.with_suffix(".json").write_text(
        json.dumps({"metrics": metrics, "events": task.event_log}, default=plain, indent=1)
    )
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--output", type=Path, default=Path("results/hospital/C3"))
    p.add_argument("--still-only", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--fps", type=int, default=25)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from roboclock.envs.hospital.env import HospitalEnv

    summary = []
    try:
        for level in a.levels:
            env = HospitalEnv(task_id="C3", difficulty=level, oracle=True, runtime=runtime)
            try:
                for seed in a.seeds:
                    if a.video:
                        from roboclock.envs.hospital.videos import record_episode

                        meta = record_episode(
                            env,
                            a.output / f"C3_crash_cart_vials_{level}_seed{seed}.mp4",
                            seed=seed,
                            fps=a.fps,
                            seconds=1e9,
                            subframes=4,
                        )
                        metrics = meta["metrics"]
                    else:
                        metrics = run_episode(
                            env, seed, a.output / f"C3_{level}_{seed}", still_only=a.still_only
                        )
                    if metrics is None:
                        continue
                    row = {
                        "level": level,
                        "seed": seed,
                        "H": metrics["horizon_s"],
                        "correct": metrics["correct_operations"],
                        "attempted": metrics["attempted_operations"],
                        "t_q": metrics["t_q_s"],
                        "end": metrics["termination_reason"],
                        "physical": metrics["task_success"],
                        "temporal_by_tier": metrics["temporal_success_by_tier"],
                        "joint_by_tier": metrics["joint_success_by_tier"],
                        "drops": metrics["drops"],
                    }
                    summary.append(row)
                    print("C3", json.dumps(row, default=plain), flush=True)
            finally:
                env.close()
    finally:
        if summary:
            name = "video_summary.json" if a.video else f"summary_{'_'.join(a.levels)}.json"
            (a.output / name).write_text(json.dumps(summary, default=plain, indent=1))
        runtime.close()


if __name__ == "__main__":
    main()
