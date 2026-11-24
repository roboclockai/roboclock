"""Record one hospital task on the RoboDojo rig into results/hospital/<data ID>/.

robodojo_rig format: head.mp4, left_wrist.mp4, right_wrist.mp4 (640x480, the policy cameras) and the
third-person overview <data ID>_<task name>_<difficulty>_<seed>.mp4 (1280x960), all 25 fps and
frame-synchronized, plus still.png / still_head.png (first frames) and the JSON sidecar (with
robodojo_rig.describe(), the measured rig check and the renderer read-back). The task camera
(sim.task_camera) is only created for recordings. --supersede first moves the folder's current
recordings into SUPERSEDED/ (never deleted). One episode per process (Isaac Sim 6.0.1 hangs on a
second reset).

    python scripts/hospital/record_rig.py --task B4 --data-id B1 --supersede
"""

import argparse
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SUPERSEDED = "superseded_2026-11-03_rig_head50"  # common rig, 50-degree 960x720 head camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="environment task ID (A1-A4, B1-B6, C1-C4)")
    p.add_argument("--data-id", required=True, help="results/hospital folder")
    p.add_argument("--difficulty", default="hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--resolution", nargs=2, type=int, default=[1280, 960], help="task-camera size")
    p.add_argument("--subframes", type=int, default=4)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--work", type=Path, default=REPO / "results/hospital/rig_record")
    p.add_argument("--supersede", action="store_true", help=f"move current recordings to {SUPERSEDED}/")
    a = p.parse_args()
    if shutil.which("ffmpeg") is None:
        p.error("FFmpeg with libx264 is required")
    output = (a.output or REPO / "results/hospital" / a.data_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if a.supersede:
        old = output / SUPERSEDED
        current = [f for f in output.iterdir() if f.is_file() and f.suffix in (".mp4", ".json", ".png")]
        if current:
            old.mkdir(exist_ok=False)
            for f in current:
                f.rename(old / f.name)
    from roboclock.envs.hospital.config import load_config
    from roboclock.envs.hospital.tasks.registry import TASKS

    config = load_config()
    config["sim"]["no_render"] = False
    config["sim"]["task_camera"] = True
    config["sim"]["task_camera_resolution"] = list(a.resolution)
    work = (a.work / a.task).resolve()
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "render_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    name = f"{a.data_id}_{TASKS[a.task].name}_{a.difficulty}_{a.seed}"
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime(renderer=config["visuals"].get("renderer", "RaytracedLighting")) as runtime:
        from roboclock.envs.hospital.env import HospitalEnv
        from roboclock.envs.hospital.videos import record_episode

        env = HospitalEnv(
            runtime=runtime, task_id=a.task, difficulty=a.difficulty, config_path=config_path,
            oracle=True,
        )
        try:
            metadata = record_episode(
                env, output, name, seed=a.seed, fps=a.fps, seconds=1e9, subframes=a.subframes,
            )
        finally:
            env.close()
    print("RECORDED", output / f"{name}.mp4", metadata["metrics"].get("task_success"), flush=True)


if __name__ == "__main__":
    main()
