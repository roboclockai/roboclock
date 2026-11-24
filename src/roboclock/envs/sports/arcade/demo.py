"""Run and record complete physical sports-v2 diagnostic episodes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from roboclock.envs.rates import VIDEO_FPS
from roboclock.simulators.isaac.version import simulator_label


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serial(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["A1", "A2", "B1"], required=True)
    p.add_argument("--variant", choices=["basic", "roll", "capture", "castle"], default="basic")
    p.add_argument("--room-style", choices=["v5", "v6"], default="v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--duration", type=float)
    p.add_argument("--video", action="store_true")
    p.add_argument("--presentation-only", action="store_true", help="Record only the presentation camera")
    p.add_argument(
        "--embodiment", choices=["dual_arx_x5"], default="dual_arx_x5"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    allowed = {"A1": {"basic", "capture", "castle"}, "A2": {"basic", "roll"}, "B1": {"basic"}}
    if args.variant not in allowed[args.task]:
        p.error("Unsupported task/variant combination")
    if args.duration is not None and args.duration < 0:
        p.error("Duration must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    # The code this recording runs: the sports package, the shared simulator layer and
    # the shared rig/rate modules. Other domains' sources (edited concurrently in the
    # same tree) are not part of a sports recording.
    source_hashes = {
        str(f.relative_to(Path("src"))): hashlib.sha256(f.read_bytes()).hexdigest()
        for root in ("src/roboclock/envs/sports", "src/roboclock/simulators")
        for f in Path(root).rglob("*.py")
    }
    for f in (Path("src/roboclock/envs/robodojo_rig.py"), Path("src/roboclock/envs/rates.py")):
        source_hashes[str(f.relative_to(Path("src")))] = hashlib.sha256(f.read_bytes()).hexdigest()
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        from roboclock.envs.sports.adapters.base import ensure_extensions

        # Isaac Sim 6.0.x does not enable the manipulator examples the backends import.
        ensure_extensions()
        from roboclock.envs.sports.arcade.tasks import MalletMoles, TaikoTask, configuration
        from roboclock.simulators.isaac.rendering import capture_camera
        from roboclock.simulators.isaac.video import VideoEncoder

        c = configuration(args.task, args.variant, args.room_style)
        from roboclock.simulators.isaac.embodiments.config import configure

        configure(c, args.task, args.embodiment)
        from roboclock.envs.sports.recording import configure as configure_recording

        configure_recording(c)
        if args.task == "A1":
            from roboclock.envs.chess.config import load_config
            from roboclock.envs.sports.arcade.chess_backend import MeshChessTask, RoomChessBackend
            from roboclock.envs.sports.arcade.chess_oracle import RefinedChessOracle

            cc = load_config()
            cc["sim"] = c["sim"]
            cc["visuals"] = c["visuals"]
            cc["scene"]["camera_position"] = c["scene"]["camera_position"]
            cc["scene"]["room_style"] = args.room_style
            cc["scene"]["camera_look_at"] = (
                [0.45, 0.05, 0.87] if args.room_style == "v5" else [0.55, 0.15, 0.88]
            )
            cc["scene"]["piece_radius"] = 0.012
            cc["scene"]["piece_height"] = 0.050
            if args.variant == "capture":
                cc["episode"].update(history=["e2e4", "d7d5"], move="e4d5")
            elif args.variant == "castle":
                cc["episode"].update(
                    history=["g1f3", "g8f6", "e2e4", "e7e5", "f1e2", "f8e7"], move="e1g1"
                )
            c = cc
            configure(c, args.task, args.embodiment)
            task = MeshChessTask(args.difficulty, args.seed, c)
            b = RoomChessBackend(runtime, c)
            policy_cls = RefinedChessOracle
        else:
            from roboclock.envs.sports.arcade.backend import ToolBackend
            from roboclock.envs.sports.arcade.oracle import ToolOracle

            task = (TaikoTask if args.task == "A2" else MalletMoles)(args.difficulty, args.seed, c)
            b = ToolBackend(runtime, c)
            policy_cls = ToolOracle
            if args.task == "A2" and c["v2"].get("dual_sticks"):
                from roboclock.envs.sports.arcade.dual_taiko import DualTaikoBackend, DualToolOracle

                b = DualTaikoBackend(runtime, c)
                policy_cls = DualToolOracle
            if args.task == "B1" and c["v2"].get("dual_mallets"):
                from roboclock.envs.sports.arcade.dual_moles import DualMoleOracle
                from roboclock.envs.sports.arcade.dual_taiko import DualTaikoBackend

                b = DualTaikoBackend(runtime, c)
                policy_cls = DualMoleOracle
        b.reset(task)
        from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_camera

        apply_presentation_camera(b)
        from roboclock.envs.sports import recording

        # Finger speed as in roboclock eval; head camera before the first render.
        rig_setup = recording.prepare(b, video=args.video, task_id=args.task)
        from pxr import UsdGeom

        title = b.stage.GetPrimAtPath("/World/Title")
        if title:
            UsdGeom.Imageable(title).MakeInvisible()
        if args.task == "A1":
            task.piece_envelopes = b.piece_envelopes
        env = SimpleNamespace(task=task, backend=b, config=c)
        policy = policy_cls(env)

        def obs():
            return {"joint_positions": b.robot.get_joint_positions().copy()}

        if hasattr(policy, "reset"):
            policy.reset(obs())

        # Warm rendering only; no extra physics ticks or task-clock advancement.
        display_checks = []
        if args.task == "A2":
            import omni.replicator.core as rep

            b.display_camera_params = rep.AnnotatorRegistry.get_annotator("camera_params")
            b.display_camera_params.attach(b.camera.get_render_product_path())

        def frame(check_display=False):
            before = float(b.world.current_time)
            for _ in range(3):
                b.world.render()
            capture_camera()
            rgb = np.asarray(b.camera.get_rgba()[:, :, :3], dtype=np.uint8).copy()
            assert abs(float(b.world.current_time) - before) < 1e-9
            if check_display and args.task == "A2":
                from roboclock.envs.sports.arcade.visibility import display_visibility

                display_checks.append({"time": float(task.time), **display_visibility(b)})
            return rgb

        for _ in range(24):
            b.world.render()
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from pxr import Gf

        cp, cq = b.camera.get_world_pose(camera_axes="usd")
        room_p = np.array([3.4, -4.2, 2.45])
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*room_p), Gf.Vec3d(0.35, 0.45, 1.05), Gf.Vec3d(0, 0, 1)
        )
        room_q = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
        b.camera.set_world_pose(room_p, room_q, camera_axes="usd")
        Image.fromarray(frame()).save(args.output / "room.png")
        b.camera.set_world_pose(cp, cq, camera_axes="usd")
        first = frame(True)
        Image.fromarray(first).save(args.output / "initial.png")
        b.stage.GetRootLayer().Export(str((args.output / "scene.usda").resolve()))
        encoder = (
            VideoEncoder(
                args.output / f"{args.task}_{args.variant}.mp4",
                c["sim"]["camera_resolution"],
                VIDEO_FPS,
            )
            if args.video
            else None
        )
        stamps = []
        encoder = recording.attach(encoder, b, args.output)
        if encoder:
            encoder.write(first)
            stamps.append(0.0)
        budget = (
            min(task.spec.max_time, args.duration)
            if args.duration is not None
            else task.spec.max_time
        )
        last_frame = 0
        last_report = -5
        with gzip.open(args.output / "trace.jsonl.gz", "wt") as trace:
            while task.time < budget - 1e-9 and not task.terminated:
                action = policy.act(obs())
                b.apply_action(action)
                task.action_count += 1
                for _ in range(c["sim"]["control_decimation"]):
                    b.tick(render=False)
                    task.observe(b.state(), c["sim"]["physics_dt"])
                    task.collision_count = b.collision_count
                    row = {
                        "time": task.time,
                        "phase": policy.phase,
                        "state": b.state(),
                        "joints": b.robot.get_joint_positions(),
                        "controller": getattr(policy, "diagnostics", {}),
                    }
                    trace.write(json.dumps(serial(row)) + "\n")
                    if task.terminated or task.time >= budget - 1e-9:
                        break
                if hasattr(b, "update_cues"):
                    b.update_cues()
                if encoder and (task.time - last_frame >= 1 / VIDEO_FPS - 1e-9 or task.terminated):
                    encoder.write(frame(True))
                    stamps.append(task.time)
                    last_frame = task.time
                if task.time - last_report >= 5:
                    s = b.state()
                    print(
                        "PROGRESS",
                        round(task.time, 2),
                        policy.phase,
                        serial(s.get("objects", {}).get("mallet", {})),
                        flush=True,
                    )
                    last_report = task.time
        evaluation_end = task.time
        # Keep the outcome visible while real physics continues under the last
        # bounded joint command. Evaluation and task clocks have already ended.
        post_roll = 0.0
        if encoder and task.terminated and args.task in ("A2", "B1"):
            with gzip.open(args.output / "trace.jsonl.gz", "at") as trace:
                for tick in range(round(2.0 / c["sim"]["physics_dt"])):
                    b.tick(render=False)
                    post_roll = (tick + 1) * c["sim"]["physics_dt"]
                    now = evaluation_end + post_roll
                    trace.write(json.dumps(serial({
                        "time": now, "phase": "post_roll", "state": b.state(),
                        "joints": b.robot.get_joint_positions(),
                    })) + "\n")
                    if now - last_frame >= 1 / VIDEO_FPS - 1e-9:
                        encoder.write(frame(True))
                        stamps.append(now)
                        last_frame = now
        final = frame(True)
        Image.fromarray(final).save(args.output / "final.png")
        if encoder:
            encoder.close()
        from roboclock.simulators.isaac.rendering import renderer_state

        result = {
            "presentation_revision": "measured-score-feedback-v1",
            "room_style": args.room_style,
            "renderer_settings": renderer_state(),
            "display_visibility": display_checks,
            "task": args.task,
            "variant": args.variant,
            "seed": args.seed,
            "difficulty": args.difficulty,
            "metrics": task.metrics(),
            "events": task.event_log,
            "configuration": c,
            "controller": "privileged bounded native robot IK; not a learned visual policy",
            "simulator": simulator_label(),
            "full_episode": args.duration is None,
            "frame_times": stamps,
            "evaluation_end": evaluation_end,
            "recorded_sim_time": evaluation_end + post_roll,
            "post_roll_seconds": post_roll,
            "source_sha256": source_hashes,
            "asset_sources": [
                json.loads(f.read_text())
                for f in (Path(__file__).parent / "assets").rglob("source.json")
            ],
        }
        result.update(rig_setup)
        if encoder:
            from roboclock.envs.sports.arcade.audio import add_event_audio

            result["audio"] = add_event_audio(
                encoder.path, task.event_log, encoder.frames / VIDEO_FPS, args.task
            )
            result["video_sha256"] = hashlib.sha256(encoder.path.read_bytes()).hexdigest()
        assert all(
            hashlib.sha256((Path("src") / name).read_bytes()).hexdigest() == digest
            for name, digest in source_hashes.items()
        ), "Source changed during recording"
        (args.output / "result.json").write_text(json.dumps(serial(result), indent=2) + "\n")
        print("RESULT", json.dumps(serial(task.metrics())), flush=True)
        b.close()


if __name__ == "__main__":
    main()
