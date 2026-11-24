"""Run and record the workflow sports tasks in real Isaac Sim physics."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ, VIDEO_FPS
from roboclock.envs.sports.arcade.demo import serial
from roboclock.envs.sports._shared.precision.config import config
from roboclock.envs.sports._shared.precision.provenance import source_hashes
from roboclock.simulators.isaac.version import simulator_label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["B3", "C1"], required=True)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float)
    p.add_argument("--video", action="store_true")
    p.add_argument("--presentation-only", action="store_true", help="Record only the presentation camera")
    p.add_argument(
        "--embodiment", choices=["dual_arx_x5"], default="dual_arx_x5"
    )
    p.add_argument("--stop-when-stalled", action="store_true", help="End an abbreviated C1 diagnostic after a settled planner stall")
    p.add_argument("--pool-overrides", type=Path, help="Experimental C1 pool configuration JSON")
    p.add_argument("--post-roll", type=float, default=2.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    c = config(args.task, args.difficulty, args.seed)
    from roboclock.simulators.isaac.embodiments.config import configure

    configure(c, args.task, args.embodiment)
    if args.pool_overrides:
        if args.task != "C1":
            raise ValueError("Pool overrides require C1")
        overrides = json.loads(args.pool_overrides.read_text())
        c["pool"].update(overrides.get("pool", {}))
        c["workflow"].update(overrides.get("workflow", {}))
    from roboclock.envs.sports.recording import configure as configure_recording

    configure_recording(c)
    (args.output / "config.json").write_text(json.dumps(c, indent=2))
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        from roboclock.simulators.isaac.embodiments.backend import ReplacementBackend as Backend
        from roboclock.envs.sports._shared.precision.oracle import Oracle
        from roboclock.envs.sports._shared.precision.tasks import Task
        from roboclock.simulators.isaac.video import VideoEncoder

        b = Backend(runtime, c)
        b.reset()
        from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_camera

        apply_presentation_camera(b)
        from roboclock.envs.sports import recording

        # Finger speed as in roboclock eval; head camera before the first render.
        rig_setup = recording.prepare(b, video=args.video, task_id=args.task)
        t = Task(b)
        oracle = Oracle(b, t)
        from roboclock.envs.sports._shared.workflow.scenes import update_cues

        update_cues(b, t)
        for _ in range(30):
            b.world.render()
        rgb = b.frame()
        Image.fromarray(rgb).save(args.output / "initial.png")
        b.stage.GetRootLayer().Export(str((args.output / "scene.usda").resolve()))
        encoder = (
            VideoEncoder(args.output / f"{args.task}.mp4", c["sim"]["camera_resolution"], VIDEO_FPS)
            if args.video
            else None
        )
        encoder = recording.attach(encoder, b, args.output)
        if encoder:
            encoder.write(rgb)
        last_frame = 0
        last_print = -5
        frames = 1
        duration = args.duration if args.duration is not None else 200
        with gzip.open(args.output / "trace.jsonl.gz", "wt") as f:
            while not t.done and b.time < duration:
                if (args.stop_when_stalled and args.task == "C1" and oracle.phase == "finished"
                        and t.ready and len(t.pots) < len(b.pool_targets)):
                    break
                oracle.act()
                for _ in range(c["sim"]["control_decimation"]):
                    s = b.tick()
                    t.update(s)
                    f.write(
                        json.dumps(
                            serial(
                                dict(
                                    time=b.time,
                                    phase=oracle.phase,
                                    state=s,
                                    command=b.command.copy(),
                                    controller_goal=getattr(oracle, "goal", None),
                                    controller_rotation=getattr(oracle, "rotation", None),
                                )
                            )
                        )
                        + "\n"
                    )
                    if t.done:
                        break
                if b.time - last_print >= 3:
                    print(
                        "PROGRESS",
                        round(b.time, 2),
                        oracle.phase,
                        "objects",
                        {n: np.round(o["position"], 3).tolist() for n, o in s["objects"].items()},
                        "events",
                        t.events[-2:],
                        flush=True,
                    )
                    last_print = b.time
                from roboclock.envs.sports._shared.workflow.scenes import update_cues

                update_cues(b, t)
                if encoder and int(b.time * VIDEO_FPS) > last_frame:
                    rgb = b.frame()
                    encoder.write(rgb)
                    last_frame = int(b.time * VIDEO_FPS)
                    frames += 1
            evaluation_end = b.time
            if encoder and t.done:
                while b.time < evaluation_end + args.post_roll:
                    oracle.act()
                    for _ in range(c["sim"]["control_decimation"]):
                        s = b.tick()
                        f.write(
                            json.dumps(serial(dict(time=b.time, phase="post_roll", state=s))) + "\n"
                        )
                    if int(b.time * VIDEO_FPS) > last_frame:
                        update_cues(b, t)
                        encoder.write(b.frame())
                        last_frame = int(b.time * VIDEO_FPS)
                        frames += 1
            Image.fromarray(b.frame()).save(args.output / "final.png")
        if encoder:
            encoder.close()
        result = t.result()
        if hasattr(oracle, "path_checks"):
            result["trajectory_checks"] = oracle.path_checks
            result["controller_transitions"] = oracle.diagnostics
        result["embodiment"] = c["embodiment"]
        result["native_joint_names"] = b.rig.names
        if hasattr(b, "actual_drive_limits"):
            result["actual_drive_limits"] = b.actual_drive_limits
            result["actual_drive_gains"] = b.actual_drive_gains
            result["reset_drive_settings"] = b.reset_drive_settings
        result["actuated_joint_names"] = b.rig.actuated_names
        from roboclock.simulators.isaac.rendering import renderer_state

        result.update(
            **rig_setup,
            full_episode=t.done,
            stop_when_stalled=args.stop_when_stalled,
            difficulty=args.difficulty,
            seed=args.seed,
            video_frames=frames if encoder else 0,
            simulator=simulator_label(),
            physics_hz=PHYSICS_HZ,
            control_hz=CONTROL_HZ,
            trace_hz=240,
            recorded_sim_time=b.time,
            evaluation_end=evaluation_end,
            source_hashes=hashes,
            collision_repairs=getattr(b, "collision_repairs", []),
            renderer_settings=renderer_state(),
            presentation_camera="wide ring / action view / wide ring"
            if args.task == "B2" and encoder
            else "fixed",
        )
        (args.output / "result.json").write_text(json.dumps(serial(result), indent=2) + "\n")
        print("RESULT", json.dumps(serial(result)), flush=True)
        b.close()


if __name__ == "__main__":
    main()
