"""Superseded on the common dual-arm rig (2026-11-03) by scripts/hospital/record_rig.py: the
observation `rgb` is now the rig's head camera (the task view needs sim.task_camera), and Isaac Sim
6.0.1 hangs on the second reset this script makes. Kept for reference.

Record the A3 syringe-pump-cutoff deliverables: still.png (frame 0, task camera) and one oracle video.

Uses roboclock.envs.hospital.videos.record_episode (actual physics; the JSON sidecar keeps the real
outcome). Needs ffmpeg with libx264 on PATH.

python scripts/hospital/a3w_record.py --output results/hospital/A3
"""

import argparse
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difficulty", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--resolution", nargs=2, type=int, default=[1280, 960])
    p.add_argument("--subframes", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("results/hospital/A3"))
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from roboclock.envs.hospital.config import load_config

    config = load_config()
    config["sim"]["camera_resolution"] = a.resolution
    work = Path("results/hospital/A3/a3w_record").resolve()
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "render_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        from PIL import Image

        from roboclock.envs.hospital.env import HospitalEnv
        from roboclock.envs.hospital.videos import record_episode

        env = HospitalEnv(
            runtime=runtime,
            task_id="A3",
            difficulty=a.difficulty,
            config_path=config_path,
            oracle=True,
        )
        try:
            obs, _ = env.reset(seed=a.seed)
            Image.fromarray(obs["rgb"]).save(output / "still.png")
            print("saved", output / "still.png", flush=True)
            if not a.no_video:
                record_episode(
                    env,
                    output / f"A3_syringe_pump_cutoff_{a.difficulty}_{a.seed}.mp4",
                    seed=a.seed,
                    fps=a.fps,
                    seconds=1e9,
                    subframes=a.subframes,
                )
        except BaseException:
            import traceback

            traceback.print_exc()
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
