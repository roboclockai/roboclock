"""Offline recording of real Isaac Sim episodes, with independent physics/render clocks.

Each task folder holds the RoboDojo policy cameras head.mp4, left_wrist.mp4 and
right_wrist.mp4 (640x480) and the third-person overview <task>.mp4 (1280x960), all at
25 fps from one render per frame, plus result.json and preview PNGs.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from roboclock.envs import robodojo_rig
from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.robot import arms_to_external
from roboclock.envs.lab.tasks.registry import TASKS
from roboclock.envs.rates import VIDEO_FPS
from roboclock.resources import PACKAGE_ROOT, resolve_asset
from roboclock.simulators.isaac.runtime import simulator_version


class VideoEncoder:
    """Stream RGB directly to H.264; publish only after FFmpeg flushes successfully."""

    def __init__(self, path, resolution, fps):
        self.path = Path(path)
        self.temporary = self.path.with_suffix(".partial.mp4")
        self.log = self.path.with_suffix(".ffmpeg.log")
        self.frames = 0
        width, height = resolution
        self.shape = (height, width, 3)
        self.stderr = self.log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.temporary),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.stderr,
        )

    def write(self, rgb):
        if rgb.shape != self.shape or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB {self.shape}, got {rgb.shape}/{rgb.dtype}")
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.frames += 1

    def close(self, *, publish=True):
        try:
            self.process.stdin.close()
        finally:
            code = self.process.wait()
            self.stderr.close()
        if code:
            raise RuntimeError(f"FFmpeg failed ({code}): {self.log.read_text()}")
        if publish:
            self.temporary.replace(self.path)
            self.log.unlink()
        else:
            self.temporary.unlink(missing_ok=True)


VIEW_FILES = {
    "head": "head.mp4",
    "left_wrist": "left_wrist.mp4",
    "right_wrist": "right_wrist.mp4",
}


def task_folder(task_id):
    return f"{task_id}_{TASKS[task_id].name}"


def view_files(task_id):
    """Recording file per view: the three RoboDojo policy cameras plus the overview."""
    return {**VIEW_FILES, "overview": f"{task_id}.mp4"}


def read_views(cameras):
    """Copy every camera's current render product (no render call)."""
    views = {}
    for name, camera in cameras.items():
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            raise RuntimeError(f"Isaac Sim camera {name} produced an empty frame")
        views[name] = np.asarray(rgba[..., :3], dtype=np.uint8).copy()
    return views


def save_previews(task_dir, stage, views):
    from PIL import Image

    names = {}
    for name in ("overview", "head"):
        preview = Path(task_dir) / f"{stage}_{name}.png"
        Image.fromarray(views[name]).save(preview)
        names[name] = preview.name
    return names


def record_episode(
    env,
    path,
    *,
    seed,
    fps,
    seconds,
    subframes,
    hold_seconds=1.0,
    idle=False,
    hold_after_loading=False,
):
    """Record one oracle episode into the task folder ``path``.

    head.mp4, left_wrist.mp4 and right_wrist.mp4 are the RoboDojo policy cameras
    (640x480); <task>.mp4 is the third-person overview (sim.overview_resolution). All
    four come from one render per frame, so they are frame-synchronized.
    """
    from roboclock.envs.lab.policies.oracle import OraclePickPlace
    from roboclock.envs.lab.validation import _scene_checks
    from roboclock.simulators.isaac.rendering import capture_camera

    started = time.perf_counter()
    observation, _ = env.reset(seed=seed)
    initial_geometry_errors = _scene_checks(env)
    if initial_geometry_errors:
        raise RuntimeError(f"Initial physical scene is invalid: {initial_geometry_errors}")
    policy = OraclePickPlace(env)
    policy.reset(observation)
    idle_joints = env.backend.joint_positions().copy()
    if getattr(env.task, "safety_task", False):
        idle_joints = env.backend.initial_hold_joints.copy()
    task = env.task
    loaded_hold_joints = None
    if task.task_id == "C1":
        from roboclock.envs.lab.rack_organization import configure_recording_logging

        configure_recording_logging()
    observer = None
    if env.config.get("active_observation", False):
        from roboclock.envs.lab.policies.active_observation import ActiveObservation

        observer = ActiveObservation(env, policy)
    if env.backend.camera is None:
        raise RuntimeError("Recordings need the overview camera (sim.overview_camera: true)")
    if task.task_id == "A4":
        from roboclock.envs.lab.titration_visuals import review_camera

        review_camera(env.backend)
    rig = env.backend.rig_cameras.cameras
    cameras = {
        "head": rig["cam_head"],
        "left_wrist": rig["cam_left_wrist"],
        "right_wrist": rig["cam_right_wrist"],
        "overview": env.backend.camera,
    }
    physics_dt = env.config["sim"]["physics_dt"]
    decimation = env.config["sim"]["control_decimation"]
    control_dt = physics_dt * decimation
    controls_per_frame = round(1 / (fps * control_dt))
    if controls_per_frame < 1 or not math.isclose(
        controls_per_frame * control_dt, 1 / fps, abs_tol=1e-9
    ):
        raise ValueError("Recording FPS must divide the configured control frequency")
    # A1 includes opening, the entire fill from an empty beaker, and closure.
    # Its fill duration follows the faucet flow rate, not a preview cap.
    limit = task.spec.max_time if task.task_id == "A1" else min(seconds, task.spec.max_time)
    task_dir = Path(path)
    task_dir.mkdir(parents=True, exist_ok=True)
    files = view_files(task.task_id)
    resolutions = {name: [int(v) for v in cameras[name].get_resolution()] for name in files}
    encoders = {
        name: VideoEncoder(task_dir / file, resolutions[name], fps) for name, file in files.items()
    }
    encoder = encoders["head"]
    timestamps = []
    preview_paths = {}
    process_trace = []
    wrist_trace = []
    capture_deltas = []
    pixel_changes = {name: [] for name in files}
    previous = {}
    camera_cuts = []
    throughput_review = task.task_id in {"C1", "C2", "C3"}
    last_report = time.perf_counter()

    def write(views):
        for name, frame in views.items():
            sample = frame[::8, ::8].astype(np.float32)
            if name in previous:
                pixel_changes[name].append(float(np.mean(np.abs(sample - previous[name]))))
            previous[name] = sample
            encoders[name].write(frame)

    def capture():
        # Overview cuts move only the third-person camera; the policy cameras are rigid.
        if (
            task.task_id == "A4"
            and task.time >= max(0.0, task.spec.valid_start - 6.0)
            and not camera_cuts
        ):
            from roboclock.envs.lab.titration_visuals import review_camera

            review_camera(env.backend, closeup=True)
            camera_cuts.append({"time": task.time, "view": "indicator_endpoint_closeup"})
        if task.task_id == "A1" and task.opening >= 0.96 and task.time >= 8.0 and not camera_cuts:
            from roboclock.envs.lab.wet_bench import closeup_camera

            closeup_camera(env.backend)
            camera_cuts.append({"time": task.time, "view": "500_mL_graduation_closeup"})
        if task.task_id == "B4" and task.event_time is not None and not camera_cuts:
            from roboclock.envs.lab.tube_liquid_visuals import closeup_camera

            closeup_camera(env.backend)
            camera_cuts.append({"time": task.time, "view": "tube_liquid_rescue_closeup"})
        before = env.backend.world.current_time
        high_quality = getattr(task, "liquid_process", False) and (
            not timestamps
            or (task.time >= task.spec.valid_start and "near_target" not in preview_paths)
        )
        if task.task_id == "A1":
            high_quality = not timestamps or (
                task.volume >= task.minimum_volume and "near_target" not in preview_paths
            )
        if task.task_id == "B4":
            high_quality = not timestamps or (
                task.first_grasp_time is not None and "near_target" not in preview_paths
            )
        capture_camera(max(subframes, 4) if high_quality else subframes)
        # Kit can allocate a correctly sized, all-black buffer before the first
        # RTX frame is ready. Warm up at zero simulation time, never record it.
        if not timestamps:
            for _ in range(20):
                frames = [camera.get_rgba() for camera in cameras.values()]
                if all(rgba is not None and rgba.size and np.any(rgba[..., :3]) for rgba in frames):
                    break
                capture_camera(4)
            else:
                raise RuntimeError("Camera did not produce a nonblank initial frame")
        after = env.backend.world.current_time
        if not math.isclose(before, after, abs_tol=1e-12):
            raise RuntimeError("Video capture advanced the physics clock")
        views = read_views(cameras)
        wrist_trace.append({"time": task.time, **env.backend.rig_cameras.wrist_poses()})
        timestamps.append(task.time)
        capture_deltas.append(after - before)
        write(views)
        stage = "initial" if len(timestamps) == 1 else None
        if (
            getattr(task, "liquid_process", False)
            or getattr(task, "laboratory_v2", False)
            or throughput_review
        ):
            visual_state = (
                task.visual_state()
                if hasattr(task, "visual_state")
                else {
                    "correct_operations": len(task.correct),
                    "attempted_operations": len(task.attempted),
                }
            )
            process_trace.append(
                {
                    "time": task.time,
                    **visual_state,
                    "controls": env.backend.state().controls,
                    "robot_ee_position": env.backend.state().ee_position.tolist(),
                    "secondary_ee_position": env.backend._tool_pose(env.backend.secondary_robot)[
                        0
                    ].tolist(),
                    # Interface order (rig-left arm first, episode joint_order).
                    "robot_joint_positions": env.external_observation(
                        {"joint_positions": env.backend.joint_positions()}
                    )["joint_positions"].tolist(),
                    "planner": policy.diagnostics(),
                }
            )
            near_target_ready = (
                task.first_grasp_time is not None
                if task.task_id == "B4"
                else task.time
                >= (
                    task.event_time
                    if getattr(task, "laboratory_v2", False) and task.event_time is not None
                    else task.spec.valid_start
                )
            )
            if task.task_id == "A1":
                near_target_ready = task.volume >= task.minimum_volume
            if getattr(task, "bath_task", False):
                near_target_ready = task.deadline is not None and task.time >= task.deadline
            stage = (
                "initial"
                if len(timestamps) == 1
                else "faucet_open"
                if task.task_id == "A1"
                and task.fully_open_at is not None
                and "faucet_open" not in preview_paths
                else "developer_entry"
                if task.task_id == "A5"
                and task.source_entry_time is not None
                and "developer_entry" not in preview_paths
                else "source_exit"
                if task.task_id == "A5"
                and task.exit_time is not None
                and "source_exit" not in preview_paths
                else "near_target"
                if near_target_ready and "near_target" not in preview_paths
                else None
            )
            if throughput_review:
                stage = (
                    "initial"
                    if len(timestamps) == 1
                    else (
                        "midpoint"
                        if task.time >= limit / 2 and "midpoint" not in preview_paths
                        else None
                    )
                )
        if stage:
            preview_paths[stage] = save_previews(task_dir, stage, views)

    try:
        capture()
        control_index = 0
        while not env.done and task.time < limit - 1e-9:
            # This diagnostic policy uses measured joints plus privileged task/contact state.
            # Physics follows LabEnv's standard actuator, substep and termination path;
            # RGB capture is only requested at the recording frequency.
            if (
                hold_after_loading
                and loaded_hold_joints is None
                and task.controls.get("source_supported")
                and task.source_entry_time is not None
            ):
                loaded_hold_joints = env.backend.joint_positions().copy()
            action = (
                loaded_hold_joints.copy()
                if loaded_hold_joints is not None
                else idle_joints.copy()
                if idle
                else policy.act_internal({"joint_positions": env.backend.joint_positions()})
            )
            if observer is not None and not idle and loaded_hold_joints is None:
                action = observer.apply(action)
            if not env.action_space.contains(np.asarray(action, dtype=np.float32)):
                raise RuntimeError("Controller produced an invalid joint target")
            env.backend.apply_action(action)
            task.action_count += 1
            remaining = math.ceil((limit - task.time - 1e-9) / physics_dt)
            env._ticks(min(decimation, remaining), render=False)
            control_index += 1
            if control_index % controls_per_frame == 0 or env.done or task.time >= limit - 1e-9:
                capture()
            now = time.perf_counter()
            if now - last_report >= 15:
                print(
                    f"{task.task_id}: sim={task.time:.2f}/{limit:.0f}s "
                    f"frames={encoder.frames} wall={now - started:.1f}s "
                    f"correct={len(task.correct)}",
                    flush=True,
                )
                last_report = now
        if task.task_id == "A1":
            from roboclock.envs.lab.wet_bench import closeup_camera

            closeup_camera(env.backend)
        simulation_frames = encoder.frames
        hold_frames = round(hold_seconds * fps)
        views = None
        for index in range(max(1, hold_frames)):
            # Re-render the frozen final state instead of repeating one image: RTX
            # shows some USD changes several frames late, and a repeated frame
            # froze B5's readout in its alarm colour after the door was latched.
            before = env.backend.world.current_time
            capture_camera(max(subframes, 4) if index == 0 else subframes)
            if not math.isclose(before, env.backend.world.current_time, abs_tol=1e-12):
                raise RuntimeError("End-hold capture advanced physics time")
            views = read_views(cameras)
            if index < hold_frames:
                write(views)
        preview_paths["final"] = save_previews(task_dir, "final", views)
    except BaseException:
        for item in encoders.values():
            item.close(publish=False)
        raise
    else:
        # Flush before IsaacRuntime closes: Kit's fast shutdown exits the process.
        for item in encoders.values():
            item.close()
    counts = {name: item.frames for name, item in encoders.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Recordings are not frame-synchronized: {counts}")
    overview_path = task_dir / files["overview"]

    audio = None
    if getattr(task, "hotplate_task", False):
        from roboclock.envs.lab.hotplate_audio import add_warning_track

        audio = add_warning_track(
            overview_path, encoder.frames / fps, task.event_time, task.action_time
        )

    if getattr(task, "safety_task", False):
        from roboclock.envs.lab.safety_audio import add_track

        audio = add_track(overview_path, encoder.frames / fps, task)

    episode = env.episode_metadata()
    metadata = {
        "task_id": task.task_id,
        "task_name": TASKS[task.task_id].name,
        "files": files,
        "environment": "lab",
        "simulator": "isaacsim==" + simulator_version(),
        "policy": "loaded_hold_failure_diagnostic"
        if hold_after_loading
        else "idle_failure_diagnostic"
        if idle
        else "privileged_lula_ik_trajectory_diagnostic",
        # Interface order (rig-left arm first); the loop above drives the backend internally.
        "loaded_hold_joints": None
        if loaded_hold_joints is None
        else arms_to_external(loaded_hold_joints, env.external_arm_order).tolist(),
        "purpose": "representative actual-physics video, not a validated expert demonstration",
        "instruction": task.gen_instruction(),
        "metrics": task.metrics(),
        "events": task.event_log,
        "controller_ik_failures": policy.external_ik_failures(),  # rig left, rig right
        "cooperative_sample_planning": policy.bimanual is not None,
        "audio": audio,
        "previews": preview_paths,
        "overview_camera_cuts": camera_cuts,
        "robodojo_rig": episode.get("robodojo_rig"),
        "rig_check": episode.get("rig_check"),
        "renderer_readback": episode.get("renderer"),
        "camera_intrinsics": episode.get("camera_intrinsics"),
        "rates": episode.get("rates"),
        "wrist_camera_trace": wrist_trace,
        "active_observation": observer.metadata() if observer else None,
        "process_trace": process_trace,
        "initial_state_and_targets": episode,
        "initial_geometry_errors": initial_geometry_errors,
        "clutter_final_state": {
            name: {
                "position": body.get_world_pose()[0].tolist(),
                "orientation": body.get_world_pose()[1].tolist(),
                "velocity": body.get_linear_velocity().tolist(),
            }
            for name, body in env.backend.clutter.items()
        },
        "resolutions": resolutions,
        "fps": fps,
        "codec": "H.264/yuv420p",
        "renderer": env.backend.appearance.config["visuals"]["renderer"],
        "effective_visuals": env.backend.appearance.config["visuals"],
        "render_subframes": subframes,
        "physics_dt": physics_dt,
        "control_dt": control_dt,
        "requested_simulation_seconds": seconds,
        "simulation_seconds": task.time,
        "full_budget_episode": task.time >= task.spec.max_time - 1e-9,
        "episode_terminated": task.terminated,
        # C tasks name their own reason (all_done, stall, horizon; docs/c-class-timing.md).
        "termination_reason": getattr(task, "termination_reason", None)
        or (
            "physical_goal_or_drop"
            if task.terminated
            else "time_budget"
            if env.done
            else "recording_duration_limit"
        ),
        "simulation_frames": simulation_frames,
        "end_hold_frames": hold_frames,
        "frame_counts": counts,
        "total_frames": encoder.frames,
        "video_seconds": encoder.frames / fps,
        "frame_simulation_timestamps": timestamps,
        "maximum_capture_physics_delta": max(map(abs, capture_deltas)),
        "mean_adjacent_frame_pixel_change": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in pixel_changes.items()
        },
        "wall_seconds": time.perf_counter() - started,
        "sha256": {
            name: hashlib.sha256((task_dir / file).read_bytes()).hexdigest()
            for name, file in files.items()
        },
    }
    if task.task_id == "C1" and not idle:
        metadata["policy"] = "ground_truth_closed_loop_color_rack_expert"
        metadata["purpose"] = (
            "successful scripted GT expert demonstration for this recorded episode"
            if task.metrics()["joint_success"]
            else "scripted GT expert candidate; see measured outcomes"
        )
        metadata["expert_control"] = {
            "inputs": [
                "tube color",
                "measured object and end-effector poses",
                "grasp contacts",
                "slot occupancy",
                "arm mount positions (reach-aware slot and queue selection)",
            ],
            "actions": "articulation motor targets; no object teleportation",
            "grasp_stabilization": (
                "demo-only physical fixed joint after measured bilateral grasp; removed on release"
                if env.config["control"].get("c1_expert_grasp_assist", False)
                else "disabled; friction-only grasp"
            ),
            "active_grasp_constraint_at_end": any(
                env.backend.stage.GetPrimAtPath(path).IsValid()
                for path in ("/World/C1ExpertGrasp", "/World/C1ExpertGraspRight")
            ),
            "validation_scope": "this seed and configuration, not a multi-seed learned-policy evaluation",
        }
        metadata["object_final_state"] = {
            name: {
                "position": obj.position.tolist(),
                "orientation": obj.orientation.tolist(),
                "grasped": obj.grasped,
                "fixture_contacts": sorted(obj.fixture_contacts),
                "assigned_target": task.layout.assignments[name],
            }
            for name, obj in env.backend.state().objects.items()
        }
    if throughput_review:
        source = (
            Path(__file__).resolve().parents[4]
            / "docs/validation/throughput_source"
            / f"{task.task_id}_{TASKS[task.task_id].name}.json"
        )
        metadata["appearance_revision"] = {
            "scope": (
                "new rack organization task in the shared daylit laboratory"
                if task.task_id == "C1"
                else "shared background, environment lighting and materials; original task and physics configuration"
            ),
            "source_sidecar": str(source) if source.exists() else None,
            "source_sidecar_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
            if source.exists()
            else None,
            "background_tables": 2,
            "mountain_windows": True,
        }
    (task_dir / "result.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"SAVED {task.task_id}: {task_dir} ({metadata['video_seconds']:.2f}s; "
        f"physical={task.physical_success}, temporal={task.temporal_success})",
        flush=True,
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/lab/videos-v2"))
    parser.add_argument("--config", type=Path, help="Use an existing scene/render configuration")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idle", action="store_true", help="Record an idle failure diagnostic")
    parser.add_argument(
        "--hold-after-loading",
        action="store_true",
        help="A5: load the stop bath normally, then hold in place past the countdown",
    )
    parser.add_argument("--task-seed", action="append", default=[], metavar="A2=1")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument(
        "--active-observation",
        action="store_true",
        help="Move the otherwise idle arm to inspect task details with its wrist camera",
    )
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--subframes", type=int, default=4)
    args = parser.parse_args(argv)
    if args.active_observation and (args.idle or args.hold_after_loading):
        parser.error("--active-observation requires a task policy")
    if args.hold_after_loading and (args.tasks != ["A5"] or args.idle):
        parser.error("--hold-after-loading requires --tasks A5 and cannot be combined with --idle")
    if shutil.which("ffmpeg") is None:
        parser.error("FFmpeg is required (with the libx264 encoder)")
    if (
        args.seconds <= 0
        or not math.isfinite(args.seconds)
        or args.fps != robodojo_rig.VIDEO_FPS
        or args.subframes < 1
    ):
        parser.error(
            f"Require positive duration/subframes and {robodojo_rig.VIDEO_FPS} fps (RoboDojo)"
        )
    task_seeds = {task: args.seed for task in args.tasks}
    for item in args.task_seed:
        task, separator, seed = item.partition("=")
        if not separator or task not in TASKS:
            parser.error(f"Invalid --task-seed {item!r}")
        task_seeds[task] = int(seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if (output / task_folder(task)).exists():
            parser.error(
                f"Recording already exists for {task} in {output}; use another output directory"
            )
    config = load_config(args.config)
    # RoboDojo cameras (head + wrists, 640x480) plus the 1280x960 third-person overview.
    config["sim"]["overview_camera"] = True
    config["sim"]["no_render"] = False
    config["active_observation"] = args.active_observation
    config["scene"]["camera_position"] = [1.6, 1.35, 2.1]
    config["scene"]["camera_look_at"] = [0.42, 0.0, 1.07]
    config["scene"]["camera_focal_length"] = 22.0
    config["scene"].update(
        liquid_camera_position=[1.45, 2.3, 2.05],
        liquid_camera_look_at=[0.25, -0.45, 1.15],
        liquid_camera_focal_length=29.0,
    )
    config["visuals"]["renderer"] = robodojo_rig.RENDER_MODE
    config.setdefault("liquid_visuals", {})["renderer"] = robodojo_rig.RENDER_MODE
    # Additional tasks can join an existing collection without replacing its
    # configuration or manifest (another render may still be writing them).
    run_suffix = "_" + "_".join(args.tasks) if (output / "manifest.json").exists() else ""
    config_path = output / f"render_config{run_suffix}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "simulator": "isaacsim==" + simulator_version(),
        "purpose": "actual Isaac Sim laboratory task videos",
        "difficulty": args.difficulty,
        "seeds": task_seeds,
        "config": config,
        "texture_provenance": "lab://textures/sources.json",
        "code_sha256": {
            str(path.relative_to(PACKAGE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        },
        "asset_sha256": {
            str(path.relative_to(PACKAGE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for root in [
                resolve_asset("lab://textures"),
                resolve_asset("lab://hotplate"),
                resolve_asset("lab://knife_switch"),
                resolve_asset("lab://safety_equipment"),
                resolve_asset("lab://darkroom"),
                resolve_asset("lab://arx_x5"),
                resolve_asset("shared://hdr"),
            ]
            for path in sorted(root.rglob("*"))
            if path.is_file() and "converted" not in path.relative_to(root).parts
        },
        "imported_backgrounds": {
            p.name: json.loads(p.read_text())
            for p in (Path(__file__).resolve().parents[4] / "assets/lab").glob("*.json")
        },
        "imported_usd_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve().parents[4] / "assets/lab").glob("*.usdc")
        },
        "scene_asset_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[4])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (Path(__file__).resolve().parents[4] / "assets/lab").rglob("*")
            if p.is_file()
        },
        "videos": [],
    }
    manifest_path = output / f"manifest{run_suffix}.json"
    from roboclock.envs.lab.env import LabEnv
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime(renderer=robodojo_rig.RENDER_MODE) as runtime:
        env = LabEnv(
            runtime=runtime, difficulty=args.difficulty, config_path=config_path, oracle=True
        )
        try:
            for index, task_id in enumerate(args.tasks):
                env.task_id = task_id
                print(f"RECORDING [{index + 1}/{len(args.tasks)}] {task_id}", flush=True)
                metadata = record_episode(
                    env,
                    output / task_folder(task_id),
                    seed=task_seeds[task_id],
                    fps=args.fps,
                    seconds=args.seconds,
                    subframes=args.subframes,
                    hold_seconds=3.0
                    if task_id in {"A1", "A4"}
                    else 2.0
                    if getattr(TASKS[task_id], "safety_task", False)
                    or getattr(TASKS[task_id], "bath_task", False)
                    else 1.0,
                    idle=args.idle,
                    hold_after_loading=args.hold_after_loading,
                )
                manifest["videos"].append(
                    {
                        "folder": task_folder(task_id),
                        **{
                            key: metadata[key]
                            for key in ["files", "metrics", "total_frames", "video_seconds", "sha256"]
                        },
                    }
                )
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        finally:
            env.close()


if __name__ == "__main__":
    main()
