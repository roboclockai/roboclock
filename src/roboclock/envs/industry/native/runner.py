"""Standalone native Industry episode: one task, its privileged controller, the
recorded runner loop and the recorded result/trace files.

This merges the four recorded runner scripts (plating 41, rotary 42, cell 43/44,
ARX manipulation 45-50). Per-task loop differences come from ``catalog.TASKS``.
The benchmark environment (``envs/industry/env.py``) steps the same scenes through
joint-position actions; this runner is the reference/recording path used by
``python -m roboclock.envs.industry.videos``. Diagnostic camera cutaways of the old
plating/rotary runners (``--diagnostic-cutaway``, ``--inspect-*``) are not carried
over; they never ran in a recording and remain in git history.

Example::

    python -m roboclock.envs.industry.native.runner --task 45 \
        --output results/industry/native/task45
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np

from roboclock.envs.rates import CONTROL_HZ, VIDEO_FPS

from .catalog import TASKS, verify
from .paths import local_output_root, portable_json_loads, robot_manifest, scene_assets

PACKAGE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CAPTURE_PHASES = ("close", "lift", "hold", "lower", "release", "withdraw", "turn", "open")


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported report type: {type(value)}")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--task", type=int, choices=sorted(TASKS), required=True)
    p.add_argument("--manifest", type=Path, default=None, help="ARX robot manifest")
    p.add_argument("--assets", type=Path, default=None, help="Tasks 41/42 environment bundle")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=90)
    p.add_argument(
        "--mode",
        default="nominal",
        choices=["nominal", "passive", "early", "late", "absent", "grasp", "seat", "delivery"],
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Episode seed; default: the reviewed episode (configs/task<NN>/reviewed.json)",
    )
    for name in ["layout", "timing", "physics"]:
        p.add_argument("--randomize-" + name, action="store_true")
    p.add_argument("--replay", type=Path)
    p.add_argument("--record", action="store_true")
    p.add_argument("--capture-start", type=float, default=0.0, help="First recorded time (s)")
    p.add_argument("--capture-end", type=float, default=120.0, help="Last recorded time (s)")
    p.add_argument(
        "--capture-subframes",
        type=int,
        choices=[1, 2, 4, 8, 16],
        default=2,
        help="RTX subframes per recorded frame (one render per 25 fps frame)",
    )
    p.add_argument("--visual-context", choices=["minimal", "debug"], default="minimal")
    p.add_argument(
        "--no-render",
        action="store_true",
        help="Physics and controller only: no cameras are created and nothing is rendered "
        "(alias of the environment's sim.no_render)",
    )
    args = p.parse_args(argv)

    if not args.output.resolve().is_relative_to(Path(local_output_root()).resolve()):
        p.error(f"Write native outputs below {local_output_root()} (ROBOCLOCK_INDUSTRY_LOCAL_ROOT)")
    if not 0 < args.seconds <= 180:
        p.error("--seconds must be greater than 0 and at most 180")
    if args.record and args.mode != "nominal":
        p.error("Record only the full nominal workflow")
    if args.record and args.no_render:
        p.error("--record needs rendering")
    return args


def snapshot_sources(directory):
    """Copy the merged native sources used for this run, with their hashes."""
    directory.mkdir()
    hashes = {}
    for source in sorted(PACKAGE.rglob("*.py")):
        relative = source.relative_to(PACKAGE).as_posix()
        target = directory / relative.replace("/", "__")
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def debug_overlay(frame, task, scene, control, o, resolved, font):
    """Optional diagnostic HUD (``--visual-context debug``); never part of policy RGB."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(frame)
    t = o["time_s"]
    if task.number == 41:
        elapsed = o["elapsed_s"] or 0
        state = (
            "PROCESSING"
            if elapsed < resolved["process_min_s"]
            else "TRANSFER WINDOW"
            if elapsed <= resolved["process_max_s"]
            else "WINDOW CLOSED"
        )
        if scene.final_exit is not None:
            state = "COMPLETE" if control.complete else "RINSE TRANSFER"
        lines = [
            f"TASK 41 | {state} | episode {t:05.1f}s",
            f"Demo recipe: {resolved['process_min_s']:.0f}-{resolved['process_max_s']:.0f}s"
            f" | {control.phase.replace('_', ' ')}",
        ]
    elif task.number == 42:
        state = "LOADING WINDOW" if o["window_open"] else "CARRIER OUTSIDE LOADING ZONE"
        if scene.success:
            state = "SUPPORTED RELEASE / COMPLETE"
        lines = [
            f"TASK42 | ROTARY PUMP LOADING | {o['elapsed_s']:05.1f} s | {state}",
            f"Carrier: {o['carrier_angle_deg']:.1f} deg | zone -120 to -60 deg | 5 deg/s / 12 s",
        ]
    else:
        title = getattr(scene, "title", None) or {
            43: "ENCLOSURE COVER BONDING",
            44: "COOLING-WATER LEAK ISOLATION",
        }.get(task.number, "")
        detail = getattr(scene, "detail", "") or ""
        events = scene.events
        if task.number == 43 and "glue_completed_s" in events:
            detail += (
                f" | Seated after glue: {events['seated_release_s'] - events['glue_completed_s']:.2f}s"
                if "seated_release_s" in events
                else f" | Since glue: {t - events['glue_completed_s']:.1f}s"
            )
        if task.number == 44 and "leak_onset_s" in events:
            detail += (
                f" | Closed in {events['response_s']:.2f}s"
                if "response_s" in events
                else f" | Since leak: {t - events['leak_onset_s']:.1f}s"
            )
        lines = [f"TASK{task.number} | {title} | {t:05.1f}s | {scene.cue_state or 'PROCESSING'}", detail]
        if task.number == 44:
            lines.append(
                f"Estimated spill: {o['estimated_spill_l']:.2f} L / MAX {o['estimated_spill_limit_l']:.2f} L"
            )
    draw.rectangle((12, 12, 1190, 17 + 25 * len(lines)), fill=(18, 32, 38))
    for i, line in enumerate(lines):
        draw.text((23, 18 + 24 * i), line, font=font, fill="white" if i == 0 else (200, 225, 225))


def result_for(task, args, scene, control, o, extras):
    nominal = args.mode == "nominal"
    if task.family == "plating":
        report = control.result()
        if args.mode == "passive":
            tail = extras["trace"][-30:]
            report["passive_support_component_passed"] = bool(
                tail
                and all(
                    r["forces"].get("load_support", 0) > 0.4
                    and np.linalg.norm(r["linear_velocity"]) < 0.015
                    for r in tail
                )
                and not scene.unexpected
            )
        report.update(
            simulation_time_s=float(scene.world.current_time),
            asset_record=scene.asset_record,
            collisions=scene.collisions,
            unexpected_contacts=scene.unexpected,
            first_contacts=scene.first_contacts,
            native_contact_geometry=scene.contact_geometry,
        )
        return report
    if task.family == "rotary":
        return dict(
            task_id=42,
            mode=args.mode,
            phase=control.phase,
            failure=control.failure,
            complete=control.complete,
            task_success=bool(
                control.complete
                and scene.success
                and o["supported"]
                and o["released"]
                and not control.failure
                and nominal
            ),
            physical_placement_completed=bool(
                "supported_release_s" in scene.events and o["supported"] and o["released"]
            ),
            final=o,
            events=scene.events,
            contact_pairs=scene.contacts,
            minimum_separations_m=scene.geometry,
            initial_part_position=scene.initial_part.tolist(),
            collider_paths=scene.collider_paths,
            product_collision="Merged unaltered original triangles, PhysX SDF256; declared box-envelope mass/inertia",
            source_product_manifest=portable_json_loads(
                (scene_assets(42) / "pump/manifest.json").read_text()
            ),
        )
    if task.family == "cell":
        final_state = (
            o.get("physical_seating", False)
            if task.number == 43
            else o.get("closed", False) and not o["leaking"]
        )
    else:
        final_state = scene.final_valid(o)
    return dict(
        task_id=task.number,
        physics_hz=task.physics_hz,
        control_hz=CONTROL_HZ,
        native_readback_checks=getattr(scene, "readback_checks", None),
        mode=args.mode,
        phase=control.phase,
        failure=control.failure,
        complete=bool(control.complete),
        task_success=bool(
            control.complete
            and scene.success
            and not control.failure
            and nominal
            and final_state
            and not scene.contacts.unexpected
        ),
        physical_completed=bool(scene.physical_completed),
        manipulation_success=bool(getattr(scene, "manipulation_success", False)),
        final=o,
        events=scene.events,
        contact_pairs=scene.contacts.pairs,
        minimum_contact_separation_m=scene.contacts.minimum,
        minimum_contact_events=scene.contacts.minimum_events,
        unexpected_contacts=scene.contacts.unexpected,
        product_attachments=False,
        scope="Native task episode; no calibrated material/chemical/fluid model or full randomized-suite claim",
    )


def main(argv=None):
    args = parse(argv)
    task = TASKS[args.task]
    args.output.mkdir(parents=True, exist_ok=False)
    if args.replay:
        resolved = verify(portable_json_loads(args.replay.read_text()))
    else:
        randomized = args.randomize_layout or args.randomize_timing or args.randomize_physics
        resolved = task.resolved(
            0 if args.seed is None and randomized else args.seed,
            layout=args.randomize_layout,
            timing=args.randomize_timing,
            physical=args.randomize_physics,
        )
    (args.output / "resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
    source_hashes = snapshot_sources(args.output / "sources")
    # videos.py sets these too; the default profile would be wrong for Task 41.
    os.environ.setdefault("ROBOCLOCK_V2_RUNTIME_PROFILE", task.runtime_profile)
    os.environ.setdefault("ROBOCLOCK_V2_RENDERER", task.renderer)
    from . import rig
    from .v2.runtime import IndustryV2Runtime

    # robodojo_rig spec: no cameras at all in no-render runs; the overview only when recording.
    rig.configure(no_render=args.no_render, overview=args.record)

    trace = []
    with IndustryV2Runtime(single_tick_rendering=task.single_tick_rendering), (
        args.output / "trace.jsonl"
    ).open("w") as trace_log:
        from PIL import Image, ImageFont

        from roboclock.simulators.isaac.rendering import capture_camera

        renderer = None if args.no_render else rig.apply_renderer()
        manifest = args.manifest or robot_manifest()
        if task.family in ("plating", "rotary"):
            scene_class, _, _ = task.load()
            scene = scene_class(
                args.assets or scene_assets(task.number),
                manifest,
                args.output,
                resolved,
                **dict(task.scene_kwargs),
            )
        else:
            scene = task.make_scene(args.output, resolved, args.mode, manifest=manifest)
        control = task.load()[1](scene, args.mode)
        if args.visual_context == "minimal":
            from .minimal_context import MinimalContext

            MinimalContext(scene, task.number, args.output)
        contacts = getattr(scene, "contacts", None)
        if task.family == "arx" and task.number != 45:
            contacts.minimum_events.clear()
        if task.family in ("cell", "arx"):
            provenance = scene.valve_metadata if task.number == 44 else scene.asset_manifest
            if task.number >= 46:
                provenance["robot_linear_velocity_overrides"] = scene.dual.linear_velocity_overrides
            (args.output / "asset-metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if task.readback_cache:
            scene.dual.native_readback_cache_enabled = True
        last_cue = [getattr(scene, "cue_state", None)]
        camera_refreshes = []
        interarm_log = None
        if task.number == 49:
            interarm_log = (args.output / "interarm-native.jsonl").open("w")
            (args.output / "interarm-body-names.json").write_text(
                json.dumps(
                    dict(
                        body_names={
                            side: list(robot._articulation_view.body_names)
                            for side, robot in scene.dual.robots.items()
                        },
                        pose_format="XYZ + quaternion XYZW",
                        physics_hz=round(1 / scene.world.get_physics_dt()),
                        scope="Native PhysX rigid-link world transforms at every physics step; "
                        "no inferred pose writes",
                    ),
                    indent=2,
                )
                + "\n"
            )

        def refresh_cue_capture():
            # Kit 6 RT2 can show stale signal materials; a fresh render product at the
            # same camera pose fixes that without advancing physics.
            if args.no_render or not hasattr(scene, "cue_state") or scene.cue_state == last_cue[0]:
                return
            before = float(scene.world.current_time)
            scene.camera.destroy()
            scene.camera.initialize()
            rig.reapply_intrinsics(scene)
            for _ in range(6):
                scene.world.render()
                capture_camera(1)
            assert scene.world.current_time == before
            camera_refreshes.append(dict(time_s=before, cue=scene.cue_state))
            last_cue[0] = scene.cue_state

        def capture(name, camera=None, subframes=3, shape=None):
            if args.no_render:
                return
            if camera is None:
                refresh_cue_capture()
            before = float(scene.world.current_time)
            for _ in range(6):
                scene.world.render()
                capture_camera(subframes)
            assert scene.world.current_time == before
            views = [(name, camera)] if camera is not None else [
                (name if key == "rgb" else f"{name}_{key.replace('_rgb', '')}", cam) for key, cam in rig.policy_cameras(scene)
            ] + ([(f"{name}_overview", scene.overview_camera)] if scene.overview_camera is not None else [])
            for label, cam in views:
                rgb = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8)
                width, height = cam.get_resolution()
                assert rgb.shape == (shape or (height, width, 3)) and rgb.std() > 5, (label, rgb.shape)
                Image.fromarray(rgb).save(args.output / (label + ".png"))

        inspection_captured = set()

        def capture_inspection_events():
            if task.number != 49 or args.no_render:
                return
            for key, event in scene.events.items():
                if key.endswith("_verified") and key not in inspection_captured:
                    name = "inspection_" + key
                    capture(name, scene.inspection_camera, shape=(480, 640, 3))
                    event["native_rgb_evidence"] = name + ".png"
                    inspection_captured.add(key)

        capture("initial")
        rig_check = rig.measure(scene)
        scene.rig["measured"] = rig_check
        encoders, frames, completed_at, captured = {}, 0, None, set()
        if args.record:
            # robodojo_rig recording format: head / left_wrist / right_wrist at 640x480 and a
            # third-person overview at 1280x960, all 25 fps from the same renders.
            streams = [(key.replace("_rgb", "").replace("rgb", "head"), cam) for key, cam in rig.policy_cameras(scene)]
            streams.append(("overview", scene.overview_camera))
            for stream, cam in streams:
                width, height = cam.get_resolution()
                encoders[stream] = subprocess.Popen(
                    ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                     "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(VIDEO_FPS),
                     "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast",
                     "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     str(args.output / f"{stream}.mp4")],
                    stdin=subprocess.PIPE,
                    stderr=(args.output / f"encoder-{stream}.log").open("wb"),
                )
            stream_cameras = dict(streams)
        font = ImageFont.truetype(FONT, 20 if task.number == 41 else 18)
        physics_hz = round(1 / scene.world.get_physics_dt())
        if physics_hz != task.physics_hz:
            raise RuntimeError(f"Task {task.number} physics rate {physics_hz} != {task.physics_hz}")
        stride, video_stride = task.control_stride, physics_hz // VIDEO_FPS
        # The rotary runner only waited after completion when recording.
        post_complete = task.post_complete_s
        if task.family == "rotary" and not args.record:
            post_complete = 0.0
        o = None
        for step in range(round(args.seconds * physics_hz)):
            task.interval(scene).clear()
            scene.dual.apply()
            scene.world.step(render=False)
            if interarm_log:
                interarm_log.write(
                    json.dumps(
                        dict(
                            time_s=float(scene.world.current_time),
                            poses={
                                side: np.asarray(
                                    robot._articulation_view._physics_view.get_link_transforms()
                                )[0].tolist()
                                for side, robot in scene.dual.robots.items()
                            },
                            joint_state={
                                side: dict(
                                    names=scene.dual.names[side],
                                    q=robot.get_joint_positions().tolist(),
                                    qd=robot.get_joint_velocities().tolist(),
                                )
                                for side, robot in scene.dual.robots.items()
                            },
                        )
                    )
                    + "\n"
                )
                if scene.contacts.interarm_events:
                    control.failure = "Inter-arm native contact, including zero-impulse touch"
            if task.observe_every_step or step % stride == 0:
                o = scene.observe()
            if step % stride == 0:
                # Scene processes (lamps, dispenser, outfeed/gate) the recorded
                # controllers ran first in their own step; see catalog/env.py.
                process = getattr(scene, "process_step", None)
                if process is not None:
                    process(o)
                control.step(o)
                capture_inspection_events()
                row = dict(**o, phase=control.phase)
                if task.family == "plating":
                    row["right_command"] = scene.dual.commands["right"].tolist()
                elif task.family == "rotary":
                    row["command"] = scene.dual.commands["left"].tolist()
                else:
                    row["commands"] = {k: v.tolist() for k, v in scene.dual.commands.items()}
                # Measured joints for every task (the pre-merge 41/42 traces lacked them).
                row["robot_joints"] = {
                    side: dict(
                        names=scene.dual.names[side],
                        q=robot.get_joint_positions().tolist(),
                        qd=robot.get_joint_velocities().tolist(),
                    )
                    for side, robot in scene.dual.robots.items()
                }
                trace.append(row)
                trace_log.write(json.dumps(row, default=scalar) + "\n")
            if step % physics_hz == 0:
                trace_log.flush()
                print(
                    json.dumps(
                        dict(time_s=o["time_s"], phase=control.phase, failure=control.failure),
                        default=scalar,
                    ),
                    flush=True,
                )
            if not args.record and control.phase not in captured and control.phase in CAPTURE_PHASES:
                capture(control.phase)
                captured.add(control.phase)
            if (
                args.record
                and step % video_stride == 0
                and args.capture_start <= o["time_s"] <= args.capture_end
            ):
                refresh_cue_capture()
                before = scene.world.current_time
                scene.world.render()
                capture_camera(args.capture_subframes)
                assert scene.world.current_time == before
                for stream, cam in stream_cameras.items():
                    frame = Image.fromarray(np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8))
                    if stream == "head" and args.visual_context == "debug":
                        debug_overlay(frame, task, scene, control, o, resolved, font)
                    encoders[stream].stdin.write(frame.tobytes())
                frames += 1
            if control.complete and completed_at is None:
                completed_at = o["time_s"]
            if control.failure:
                break
            if (
                post_complete is not None
                and control.complete
                and (post_complete == 0 or o["time_s"] - completed_at > post_complete)
            ):
                break
        if task.final_observe:
            o = scene.observe()
        for stream, encoder in encoders.items():
            encoder.stdin.close()
            if encoder.wait(timeout=120) != 0:
                raise RuntimeError(f"Video encoding failed: {stream}")
        capture("final")
        trace_log.flush()
        if interarm_log:
            interarm_log.close()
        result = result_for(task, args, scene, control, o, dict(trace=trace))
        result.update(
            isaacsim_version=version("isaacsim"),
            simulation_time_s=float(scene.world.current_time),
            controller_uses_privileged_state=True,
            capture_advanced_physics=False,
            camera_render_product_refreshes=camera_refreshes,
            video_frames=frames,
            video_fps=VIDEO_FPS if args.record else None,
            resolved=resolved,
            source_hashes=source_hashes,
            # Common Dual ARX rig and head-camera pose (rig.py); None off the rig.
            rig=getattr(scene, "rig", None),
        )
        result["renderer"] = renderer if renderer is None else {**renderer, "readback_at_end": rig.renderer_readback()}
        result["video_streams"] = sorted(encoders)
        if task.number == 49:
            result["interarm_native_contact_events"] = scene.contacts.interarm_events
        (args.output / "result.json").write_text(json.dumps(result, indent=2, default=scalar) + "\n")
        (args.output / "trace.json").write_text(json.dumps(trace, default=scalar) + "\n")
        print(
            "RESULT",
            json.dumps(
                {k: result.get(k) for k in ("task_success", "complete", "failure", "events")},
                default=scalar,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
