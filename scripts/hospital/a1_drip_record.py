"""Superseded on the common dual-arm rig (2026-11-03) by scripts/hospital/record_rig.py: the
observation `rgb` is now the rig's head camera (the task view needs sim.task_camera), and Isaac Sim
6.0.1 hangs on the second reset this script makes. Kept for reference.

Record the A1 drip-chamber deliverables: still.png and one oracle video with its JSON sidecar.

The still is frame 0 of the episode from the task camera (the reset observation); the video uses
roboclock.envs.hospital.videos.record_episode, so its physics and render clocks stay independent
and the actual outcome is reported in the sidecar. --frames also saves a few video frames as PNG.

PATH=/tmp/ffmpeg/bin:$PATH python scripts/hospital/a1_drip_record.py --output results/hospital/A1
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
    p.add_argument("--output", type=Path, default=Path("results/hospital/A1/drip/video"))
    p.add_argument("--frames", type=Path, default=None, help="directory for sampled video frames")
    a = p.parse_args()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from roboclock.envs.hospital.config import load_config

    config = load_config()
    config["sim"]["camera_resolution"] = a.resolution
    config_path = output / "render_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        from PIL import Image

        from roboclock.envs.hospital.env import HospitalEnv
        from roboclock.envs.hospital.videos import record_episode

        env = HospitalEnv(
            runtime=runtime,
            task_id="A1",
            difficulty=a.difficulty,
            config_path=config_path,
            oracle=True,
        )
        try:
            observation, _ = env.reset(seed=a.seed)
            Image.fromarray(observation["rgb"]).save(output / "still.png")
            if a.frames is not None:
                from roboclock.envs.lab import videos as lab_videos

                a.frames.mkdir(parents=True, exist_ok=True)
                original = lab_videos.VideoEncoder.write
                counter = {"n": 0}

                def write(self, frame):
                    if counter["n"] % 5 == 0:
                        Image.fromarray(frame).save(a.frames / f"frame_{counter['n']:04d}.png")
                    counter["n"] += 1
                    return original(self, frame)

                lab_videos.VideoEncoder.write = write
            record_episode(
                env,
                output / f"A1_drip_chamber_level_{a.difficulty}_{a.seed}.mp4",
                seed=a.seed,
                fps=a.fps,
                seconds=1e9,
                subframes=a.subframes,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
