"""Offline recording of real Isaac Sim hospital episodes in the robodojo_rig format.

One episode per process: scripts/hospital/record_rig.py. Uses the privileged hospital
trajectory diagnostic. Videos show what the simulator did; failed
outcomes are kept and reported in each video's JSON sidecar.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from roboclock.envs.lab.videos import VideoEncoder
from roboclock.simulators.isaac.runtime import simulator_version

INTRINSICS_NOTE = (
    "camera_intrinsics is what Isaac's get_intrinsics_matrix() reports (fy from the vertical "
    "aperture). RTX ignores the vertical aperture and renders square pixels, so the image "
    "follows camera_intrinsics_rendered (robodojo_rig.intrinsics: fy = fx, principal point at "
    "the centre). Use the rendered matrix for projection and depth unprojection."
)


def _rgb(camera):
    rgba = camera.get_rgba()
    if rgba is None or rgba.size == 0 or not np.any(rgba[..., :3]):
        raise RuntimeError("Isaac Sim camera produced an empty frame")
    return np.asarray(rgba[..., :3], dtype=np.uint8).copy()


STREAMS = {"head": "rgb", "left_wrist": "left_wrist_rgb", "right_wrist": "right_wrist_rgb"}


def frame_size(config):
    """Overview (task camera) size; without a task camera the head camera's."""
    if not config["sim"].get("task_camera"):
        return list(config["sim"]["camera_resolution"])
    return list(config["sim"].get("task_camera_resolution", config["sim"]["camera_resolution"]))


def head_camera_metadata(env):
    """The rig placement this recording used (hospital/rig.py, robodojo_rig)."""
    from roboclock.envs.hospital.rig import head_camera_pose, rig_mounts, rig_pose

    config, task_id = env.config, env.task.task_id
    mounts, yaw = rig_mounts(config, task_id)
    origin, _ = rig_pose(config, task_id)
    position, target, pitch = head_camera_pose(config, task_id)
    return {
        "rig_origin": [float(v) for v in origin],
        "rig_yaw_deg": math.degrees(yaw),
        "arm_mounts": {"arm0_right": [float(v) for v in mounts[0]], "arm1_left": [float(v) for v in mounts[1]]},
        "position": [float(v) for v in position],
        "look_at": [float(v) for v in target],
        "pitch_deg": math.degrees(pitch),
    }


def record_episode(env, directory, name, *, seed, fps, seconds, subframes, hold_seconds=1.0):
    """Reset once and record the oracle episode in the robodojo_rig format: `head.mp4`,
    `left_wrist.mp4`, `right_wrist.mp4` (640x480, the policy cameras) and the third-person
    overview `<name>.mp4` (task camera, sim.task_camera_resolution), all `fps`, frame-synchronized
    (one render feeds every stream), plus `still.png` (first overview frame) and `<name>.json`.
    Isaac Sim 6.0.1 hangs on a second reset in one process: one episode per process."""
    from roboclock.envs import robodojo_rig
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    directory = Path(directory)
    started = time.perf_counter()
    observation, _ = env.reset(seed=seed)
    policy = OraclePickPlace(env)
    policy.reset(observation)
    task, backend = env.task, env.backend
    if backend.task_camera is None or backend.camera is None:
        raise RuntimeError("Recording needs sim.task_camera and rendering (not sim.no_render)")
    physics_dt = env.config["sim"]["physics_dt"]
    decimation = env.config["sim"]["control_decimation"]
    control_dt = physics_dt * decimation
    controls_per_frame = round(1 / (fps * control_dt))
    if controls_per_frame < 1 or not math.isclose(
        controls_per_frame * control_dt, 1 / fps, abs_tol=1e-9
    ):
        raise ValueError("Recording FPS must divide the configured control frequency")
    limit = min(seconds, task.spec.max_time)
    paths = {stream: directory / f"{stream}.mp4" for stream in STREAMS}
    paths["overview"] = directory / f"{name}.mp4"
    sizes = {stream: list(robodojo_rig.HEAD_CAMERA["resolution"] if stream == "head" else robodojo_rig.WRIST_CAMERA["resolution"]) for stream in STREAMS}
    sizes["overview"] = frame_size(env.config)
    encoders = {stream: VideoEncoder(path, sizes[stream], fps) for stream, path in paths.items()}
    timestamps, capture_deltas, pixel_changes = [], [], []
    previous = None
    last_report = time.perf_counter()

    def capture():
        nonlocal previous
        before = backend.world.current_time
        frames = backend.render_frame(subframes)
        after = backend.world.current_time
        if not math.isclose(before, after, abs_tol=1e-12):
            raise RuntimeError("Video capture advanced the physics clock")
        images = {stream: frames[key] for stream, key in STREAMS.items()}
        images["overview"] = _rgb(backend.task_camera)
        sample = images["overview"][::8, ::8].astype(np.float32)
        if previous is not None:
            pixel_changes.append(float(np.mean(np.abs(sample - previous))))
        previous = sample
        timestamps.append(task.time)
        capture_deltas.append(after - before)
        for stream, image in images.items():
            if not np.any(image):
                raise RuntimeError(f"{stream}: Isaac Sim camera produced an empty frame")
            encoders[stream].write(image)
        return images

    try:
        last = capture()
        from PIL import Image

        Image.fromarray(last["overview"]).save(directory / "still.png")
        Image.fromarray(last["head"]).save(directory / "still_head.png")
        control_index = 0
        while not env.done and task.time < limit - 1e-9:
            # Backend-level loop: internal arm order (arm 0 = rig right), see LabEnv.
            action = policy.act_internal({"joint_positions": backend.joint_positions()})
            if not env.action_space.contains(np.asarray(action, dtype=np.float32)):
                raise RuntimeError("Controller produced an invalid joint target")
            backend.apply_action(action)
            task.action_count += 1
            remaining = math.ceil((limit - task.time - 1e-9) / physics_dt)
            env._ticks(min(decimation, remaining), render=False)
            control_index += 1
            if control_index % controls_per_frame == 0 or env.done or task.time >= limit - 1e-9:
                last = capture()
            now = time.perf_counter()
            if now - last_report >= 15:
                print(
                    f"{task.task_id}: sim={task.time:.2f}/{limit:.0f}s "
                    f"frames={encoders['head'].frames} wall={now - started:.1f}s "
                    f"correct={len(task.correct)}",
                    flush=True,
                )
                last_report = now
        simulation_frames = encoders["head"].frames
        hold_frames = round(hold_seconds * fps)
        for _ in range(hold_frames):
            for stream, image in last.items():
                encoders[stream].write(image)
    except BaseException:
        for encoder in encoders.values():
            encoder.close(publish=False)
        raise
    else:
        # Flush before IsaacRuntime closes: Kit's fast shutdown exits the process.
        for encoder in encoders.values():
            encoder.close()
    counts = {stream: encoder.frames for stream, encoder in encoders.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Streams are not frame-synchronized: {counts}")

    metadata = {
        "files": {stream: path.name for stream, path in paths.items()},
        "environment": "hospital",
        "simulator": "isaacsim==" + simulator_version(),
        "policy": "privileged_lula_ik_trajectory_diagnostic",
        "purpose": "representative actual-physics video, not a validated expert demonstration",
        "instruction": task.gen_instruction(),
        "metrics": task.metrics(),
        "events": task.event_log,
        "initial_state_and_targets": env.episode_metadata(),
        "robodojo_rig": robodojo_rig.describe(),
        "rig_check": backend.rig_check,
        "renderer_readback": backend.render_state,
        "camera_intrinsics_reported": backend.camera_intrinsics,
        "camera_intrinsics_rendered": backend.camera_intrinsics_rendered,
        "camera_intrinsics_note": INTRINSICS_NOTE.replace(
            "camera_intrinsics is", "camera_intrinsics_reported is"
        ),
        "rig_placement": head_camera_metadata(env),
        "resolutions": sizes,
        "fps": fps,
        "codec": "H.264/yuv420p",
        "render_subframes": subframes,
        "physics_dt": physics_dt,
        "control_dt": control_dt,
        "requested_simulation_seconds": seconds,
        "simulation_seconds": task.time,
        "full_budget_episode": task.time >= task.spec.max_time - 1e-9,
        "episode_terminated": task.terminated,
        "simulation_frames": simulation_frames,
        "end_hold_frames": hold_frames,
        "frame_counts": counts,
        "total_frames": counts["head"],
        "video_seconds": counts["head"] / fps,
        "frame_simulation_timestamps": timestamps,
        "maximum_capture_physics_delta": max(map(abs, capture_deltas)),
        "mean_adjacent_frame_pixel_change": float(np.mean(pixel_changes)) if pixel_changes else 0,
        "wall_seconds": time.perf_counter() - started,
        "sha256": {stream: hashlib.sha256(path.read_bytes()).hexdigest() for stream, path in paths.items()},
    }
    (directory / f"{name}.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"SAVED {task.task_id}: {paths['overview']} ({metadata['video_seconds']:.2f}s; "
        f"physical={task.physical_success}, temporal={task.temporal_success})",
        flush=True,
    )
    return metadata
