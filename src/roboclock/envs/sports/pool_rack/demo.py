"""Run the fixed dual-X5 eight-ball racking task; actual PhysX and optional native video."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ, VIDEO_FPS
from .config import config


def source_hashes():
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in ("src/roboclock/envs", "src/roboclock/simulators/isaac")
        for p in sorted(Path(root).rglob("*.py"))
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float)
    p.add_argument("--overrides", type=Path)
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--presentation-only", action="store_true")
    p.add_argument("--embodiment", choices=["dual_arx_x5"], default="dual_arx_x5")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c = config(args.difficulty, args.seed)
    if args.overrides:
        for key, values in json.loads(args.overrides.read_text()).items():
            c[key].update(values)
    from roboclock.envs.sports.recording import configure as configure_recording

    configure_recording(c)
    hashes = source_hashes()
    (args.output / "config.json").write_text(json.dumps(c, indent=2))
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        from PIL import Image

        from roboclock.envs.sports import recording
        from roboclock.envs.sports.arcade.demo import serial
        from roboclock.envs.sports._shared.workflow.scenes import update_cues
        from roboclock.simulators.isaac.embodiments.backend import ReplacementBackend
        from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_camera
        from roboclock.simulators.isaac.rendering import renderer_state
        from roboclock.simulators.isaac.version import simulator_label
        from roboclock.simulators.isaac.video import VideoEncoder

        from .oracle import Oracle
        from .task import Task

        b = ReplacementBackend(runtime, c)
        b.reset()
        apply_presentation_camera(b)
        # Finger speed as in roboclock eval; head camera before the first render.
        rig_setup = recording.prepare(b, video=args.video, task_id="C1")
        (args.output / "initial_state.json").write_text(json.dumps(serial(b.last_state), indent=2))
        t = Task(b)
        o = Oracle(b, t)
        update_cues(b, t)
        encoder = None
        frames = 0
        frame_times = []
        for _ in range(30):
            b.world.render()
        rgb = b.frame()
        Image.fromarray(rgb).save(args.output / "initial.png")
        b.stage.GetRootLayer().Export(str((args.output / "scene.usda").resolve()))
        if args.video:
            encoder = VideoEncoder(args.output / "overview.mp4", c["sim"]["camera_resolution"], VIDEO_FPS)
            encoder = recording.attach(encoder, b, args.output)
            encoder.write(rgb)
            frames = 1
            frame_times = [0.0]
        limit = (
            0
            if args.preview
            else args.duration
            if args.duration is not None
            else c["rack"]["horizon"] + 1
        )
        last_print = -5
        last_frame = 0
        with gzip.open(args.output / "trace.jsonl.gz", "wt") as f:
            while not t.done and b.time < limit:
                if args.stop_on_failure and t.failures:
                    break
                o.act()
                for _ in range(c["sim"]["control_decimation"]):
                    s = b.tick()
                    t.update(s)
                    f.write(
                        json.dumps(
                            serial(
                                dict(time=b.time, phase=o.phase, state=s, command=b.command.copy())
                            )
                        )
                        + "\n"
                    )
                    if t.done:
                        break
                if b.time - last_print >= 3:
                    print(
                        "PROGRESS",
                        round(b.time, 3),
                        o.phase,
                        "accepted",
                        len(t.accepted),
                        "current",
                        len(t.current),
                        "events",
                        t.events[-2:],
                        "failures",
                        t.failures,
                        flush=True,
                    )
                    last_print = b.time
                if encoder and int(b.time * VIDEO_FPS) > last_frame:
                    update_cues(b, t)
                    encoder.write(b.frame())
                    frames += 1
                    last_frame = int(b.time * VIDEO_FPS)
                    frame_times.append(b.time)
            evaluation_end = b.time
            if encoder and t.done:
                while b.time < evaluation_end + 2.0:
                    o.act()
                    for _ in range(c["sim"]["control_decimation"]):
                        s = b.tick()
                        f.write(
                            json.dumps(serial(dict(time=b.time, phase="post_roll", state=s))) + "\n"
                        )
                    if int(b.time * VIDEO_FPS) > last_frame:
                        update_cues(b, t)
                        encoder.write(b.frame())
                        frames += 1
                        last_frame = int(b.time * VIDEO_FPS)
                        frame_times.append(b.time)
            update_cues(b, t)
            Image.fromarray(b.frame()).save(args.output / "final.png")
        if encoder:
            encoder.close()
        (args.output / "final_state.json").write_text(json.dumps(serial(b.last_state), indent=2))
        result = t.result()
        result.update(
            **rig_setup,
            full_episode=t.done,
            seed=args.seed,
            difficulty=args.difficulty,
            video_frames=frames,
            frame_times=frame_times,
            recorded_sim_time=b.time,
            evaluation_end=evaluation_end,
            simulator=simulator_label(),
            physics_hz=PHYSICS_HZ,
            control_hz=CONTROL_HZ,
            controller_transitions=o.transitions,
            actual_drive_limits=b.actual_drive_limits,
            actual_drive_gains=b.actual_drive_gains,
            asset_hashes={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.joinpath("assets").glob("*.png")
            },
            source_hashes=hashes,
            embodiment=c["embodiment"],
            renderer_settings=renderer_state(),
            initial_ik=b.rig.initial_ik,
            validation_scope="Preview only"
            if args.preview
            else "Full episode"
            if t.done
            else "Truncated diagnostic",
        )
        (args.output / "result.json").write_text(json.dumps(serial(result), indent=2) + "\n")
        print(
            "RESULT",
            json.dumps(
                serial(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("source_hashes", "frame_times", "occupancy")
                    }
                )
            ),
            flush=True,
        )
        b.close()


if __name__ == "__main__":
    main()
