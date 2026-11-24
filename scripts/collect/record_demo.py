"""Record one oracle demonstration as a RoboDojo v1.0 HDF5 episode.

  PYTHONPATH=src python scripts/collect/record_demo.py \\
      --root $DEMOS_DIR/roboclock_oracle_v1 --env lab --task A1 --difficulty easy --seed 0

The diagnostic oracle drives the episode at the 50 Hz control rate, exactly as `roboclock eval
--policy oracle --oracle` does (zero inference latency, no control delay). Every second control
step (25 Hz, RoboDojo's collect_freq) the observation is recorded: the three RoboDojo cameras
rendered at that instant, the joint state and the end-effector poses. The file layout and keys
follow RoboDojo's arx_x5 demonstrations (see FORMAT below), so XPolicyLab's Pi_05
process_data.py and OpenWAM's robodojo dataloader read them unchanged:

  <root>/<env>_<task>/arx_x5/data/episode_<seed:07d>.hdf5   (successful demonstrations only)
  <root>/_meta/<env>_<task>/seed_<seed>.json                (every finished episode: metrics,
                                                             episode metadata, status)

Only a demonstration whose episode ends in joint (physical + temporal) success is written; a
failed rollout gets status "fail" and its seed is replaced by a spare (campaign.py).
"""

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

H5PY_PREBUNDLE = (
    Path(sys.prefix)
    / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/isaacsim/exts"
    / "isaacsim.replicator.episode_recorder/pip_prebundle"
)

CAMERAS = (("cam_head", "rgb"), ("cam_left_wrist", "left_wrist_rgb"), ("cam_right_wrist", "right_wrist_rgb"))
SIDES = ("left", "right")
GRIPPER_SCALE = (-0.01, 0.044)  # RoboDojo X5 gripper_scale; XPolicyLab client uses the same
RGB_MARKER = b"XPL-RGB1"
JPEG_QUALITY = 95
# roboclock ee_pose is the X5 grasp_center: link6 * tool_frame (arx_x5.urdf: xyz 0.145 0 0,
# rpy 0 pi/2 0). RoboDojo's ee_poses are link6.
TOOL_OFFSET_M = 0.145


def encode_jpeg(image):
    """RGB JPEG tagged the way XPolicyLab's encoder tags it (COM segment after JFIF APP0)."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(buffer, "JPEG", quality=JPEG_QUALITY)
    data = buffer.getvalue()
    offset = 4 + int.from_bytes(data[4:6], "big") if data[2:4] == b"\xff\xe0" else 2
    segment = b"\xff\xfe" + (len(RGB_MARKER) + 2).to_bytes(2, "big") + RGB_MARKER
    return data[:offset] + segment + data[offset:]


def normalise_gripper(finger):
    return (np.asarray(finger, dtype=np.float64) - GRIPPER_SCALE[0]) / (GRIPPER_SCALE[1] - GRIPPER_SCALE[0])


def matrix(position, quat_wxyz):
    from scipy.spatial.transform import Rotation

    w, x, y, z = quat_wxyz
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = position
    return pose


def tool_frame():
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", math.pi / 2).as_matrix()
    pose[:3, 3] = (TOOL_OFFSET_M, 0.0, 0.0)
    return pose


def ee_matrices(ee_pose):
    """(2, 4, 4) world poses of the 14-value roboclock ee_pose (per arm xyz + wxyz)."""
    ee = np.asarray(ee_pose, dtype=np.float64)
    return np.stack([matrix(ee[7 * a : 7 * a + 3], ee[7 * a + 3 : 7 * a + 7]) for a in range(2)])


class Arms:
    """Both arms' link6 poses by forward kinematics, in RoboDojo's world frame.

    The bases are calibrated once, from the first observation (`X5Kinematics.calibrated`, as
    the vision-LLM agent does): base = ee_pose * FK_grasp(q)^-1. After that every link6 pose
    is base * FK_link6(q), for measured joints (state) and joint targets (action) alike, so the
    two are the same function of the joints. A domain whose ee_pose tool offset differs from
    the URDF's 0.145 m (sports: 0.14 m) shifts the calibrated base by at most that difference.

    roboclock places the rig per task; RoboDojo always puts it at the base midpoint
    (0, -0.45, 0.765) facing world +y. Poses and camera extrinsics are re-expressed in that
    canonical placement, so OpenWAM's sim variant (world -> arm base) and any reader that
    assumes RoboDojo's base poses see the same geometry as in RoboDojo's own data.
    """

    ROBODOJO_ORIGIN = (0.0, -0.45, 0.765)
    ROBODOJO_YAW_DEG = 90.0

    def __init__(self, joints, ee_pose):
        from roboclock.envs import robodojo_rig
        from roboclock.policies.x5_kinematics import X5Kinematics

        self.kin = X5Kinematics()
        self.inv_tool = np.linalg.inv(tool_frame())
        ee = ee_matrices(ee_pose)
        self.scene_bases = [
            ee[a] @ np.linalg.inv(self.kin.tool_pose(joints[8 * a : 8 * a + 6], base=False)) for a in range(2)
        ]
        origin = (self.scene_bases[0][:3, 3] + self.scene_bases[1][:3, 3]) / 2.0
        forward = self.scene_bases[0][:3, 0] + self.scene_bases[1][:3, 0]
        self.scene_yaw_deg = math.degrees(math.atan2(forward[1], forward[0]))
        self.scene_origin = origin
        scene_rig = robodojo_rig.rig_to_world(origin, self.scene_yaw_deg)
        robodojo = robodojo_rig.rig_to_world(self.ROBODOJO_ORIGIN, self.ROBODOJO_YAW_DEG)
        self.to_robodojo = robodojo @ np.linalg.inv(scene_rig)
        self.bases = [self.to_robodojo @ b for b in self.scene_bases]
        self.base_separation_m = float(np.linalg.norm(self.scene_bases[0][:3, 3] - self.scene_bases[1][:3, 3]))
        # Calibration residual check: the rig is level and the bases face the same way.
        self.base_tilt_deg = [
            float(math.degrees(math.acos(np.clip(b[2, 2], -1, 1)))) for b in self.scene_bases
        ]
        self.head = self.to_robodojo @ robodojo_rig.head_camera_world(origin, self.scene_yaw_deg)
        self.wrist_in_link6 = robodojo_rig.wrist_camera_in_link6()

    def link6(self, joints16):
        q = np.asarray(joints16, dtype=np.float64)
        return np.stack(
            [self.bases[a] @ self.kin.tool_pose(q[8 * a : 8 * a + 6], base=False) @ self.inv_tool for a in range(2)]
        )

    def ee_error_m(self, joints16, ee_pose):
        """Distance between the FK grasp point and the observed ee_pose (checks the model)."""
        q = np.asarray(joints16, dtype=np.float64)
        ee = ee_matrices(ee_pose)
        errors = []
        for a in range(2):
            fk = self.scene_bases[a] @ self.kin.tool_pose(q[8 * a : 8 * a + 6], base=False)
            errors.append(float(np.linalg.norm(fk[:3, 3] - ee[a][:3, 3])))
        return errors


def pose7(pose):
    """FORMAT.ee: xyz + quaternion, in RoboDojo's order."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(pose[:3, :3]).as_quat()
    quat = np.array((w, x, y, z))
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:  # RoboDojo's quaternions have w >= 0
        quat = -quat
    return np.r_[pose[:3, 3], quat]


# RoboDojo v1.0 conventions (RoboDojo env/robot_manager, env/camera_manager,
# XPolicyLab/utils/process_data.py; checked against RoboDojo-Benchmark/RoboDojo episodes):
# - action[t] = state[t+1] for joints and gripper; action ee_poses = FK of action joints.
# - ee_poses: link6, [x, y, z, qw, qx, qy, qz], position relative to the env origin, world
#   orientation, bases at (-0.3 | 0.3, -0.45, 0.765) facing +y; delta_ee_poses = ee_poses.
# - ee_joint_states: last commanded opening normalised over gripper_scale [-0.01, 0.044].
# - extrinsic_matrix: camera-to-world, USD camera axes (looks along -Z), env frame.
# - colors: RGB JPEG q95 tagged XPL-RGB1, fixed-length bytes per camera.
FORMAT = {
    "frequency": 25,
    "data_format_version": b"v1.0",
}


class Recorder:
    def __init__(self):
        from roboclock.envs import robodojo_rig

        self.frames = []
        self.arms = None
        self.ee_errors = []
        self.gripper_clipped = 0
        self.intrinsics = {
            "cam_head": robodojo_rig.intrinsics(robodojo_rig.HEAD_CAMERA),
            "cam_left_wrist": robodojo_rig.intrinsics(robodojo_rig.WRIST_CAMERA),
            "cam_right_wrist": robodojo_rig.intrinsics(robodojo_rig.WRIST_CAMERA),
        }

    def capture(self, obs, gripper_command, images=True):
        joints = np.asarray(obs["joint_positions"], dtype=np.float64)
        if self.arms is None:
            self.arms = Arms(joints, obs["ee_pose"])
        links = self.arms.link6(joints)
        self.ee_errors.append(max(self.arms.ee_error_m(joints, obs["ee_pose"])))
        command = np.asarray(gripper_command, dtype=np.float64).copy()
        self.gripper_clipped += int(np.any((command < 0) | (command > 1)))
        if not images:
            self.frames.append({"t": float(np.asarray(obs["sim_time"]).reshape(-1)[0]),
                                "joints": joints, "link6": links, "gripper": command})
            return
        start = time.perf_counter()
        images = {camera: np.asarray(obs[key]) for camera, key in CAMERAS}
        render = time.perf_counter() - start
        frame = {
            "t": float(np.asarray(obs["sim_time"]).reshape(-1)[0]),
            "joints": joints,
            "link6": links,
            "gripper": command,
            "jpeg": {camera: encode_jpeg(image) for camera, image in images.items()},
            "shape": {camera: image.shape for camera, image in images.items()},
            "extrinsic": {
                "cam_head": self.arms.head,
                "cam_left_wrist": links[0] @ self.arms.wrist_in_link6,
                "cam_right_wrist": links[1] @ self.arms.wrist_in_link6,
            },
            "render_s": render,
        }
        self.frames.append(frame)
        return frame


VULKAN_PROBE_TIMEOUT_S = 180


def patient_vulkan_probe():
    """roboclock's start-up check gives `vulkaninfo --summary` 15 s. On a loaded pod it takes
    longer and succeeds (37 s, exit 0, measured 2026-11-06 with ~20 Isaac processes running), so
    every start fails with "Vulkan cannot enumerate a GPU". Give the probe in this process more
    time; a real enumeration failure still fails it."""
    import subprocess
    import types

    from roboclock.simulators.isaac import diagnostics

    def run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]).endswith("vulkaninfo"):
            kw["timeout"] = max(kw.get("timeout") or 0, VULKAN_PROBE_TIMEOUT_S)
        return subprocess.run(cmd, *a, **kw)

    diagnostics.subprocess = types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)

    # vulkaninfo also fails outright (vkCreateDevice ERROR_INITIALIZATION_FAILED) when any one GPU of the pod
    # has no free memory, since it creates a device on every GPU; Kit, rendering on this process's own GPU,
    # still starts (measured 2026-11-06 with GPU 4 at 9 MiB free). Report such a probe as unchecked; a real
    # rendering failure still stops the episode, and verify_demos.py checks every recorded image.
    status = diagnostics.graphics_status

    def tolerant():
        result = status()
        if result.get("vulkan_gpu_enumeration") is False:
            print(f"[record] vulkaninfo failed ({str(result.get('detail'))[-160:]!r}); continuing on this GPU", flush=True)
            result = {**result, "vulkan_gpu_enumeration": None}
        return result

    diagnostics.graphics_status = tolerant


def run(args, finish):
    """Roll out the oracle and call finish(result) inside the runtime: Kit's shutdown exits the
    process when the runtime closes, so nothing after the `with` block would run."""
    from roboclock.config import load_config
    from roboclock.envs.rates import CONTROL_HZ
    from roboclock.envs.registry import create_env, get_environment, import_symbol
    from roboclock.policies.base import load_policy

    sys.path.insert(0, str(H5PY_PREBUNDLE))
    import h5py  # noqa: F401  (fail before the simulator starts)

    patient_vulkan_probe()

    decimation = round(CONTROL_HZ / FORMAT["frequency"])
    spec = get_environment(args.env)
    renderer = load_config(None, environment=args.env)["visuals"].get("renderer", "RaytracedLighting")
    wall = time.time()
    with import_symbol(spec.runtime_factory)(headless=True, renderer=renderer) as runtime:
        env = create_env(args.env, args.task, args.difficulty, runtime=runtime, oracle=True)
        try:
            startup = time.time() - wall
            episode = dict(environment=args.env, task_id=args.task, difficulty=args.difficulty,
                           seed=args.seed, control_delay=0.0)
            obs, _ = env.reset(seed=args.seed, options=episode)
            reset_s = time.time() - wall - startup
            print(f"[record] reset done: startup {startup:.0f}s reset {reset_s:.0f}s", flush=True)
            instruction = str(obs["instruction"])
            policy = load_policy("oracle", env)
            policy.reset(obs)
            low, high = env.action_space.low, env.action_space.high
            fingers = [6, 14]
            # RoboDojo's state gripper is the last commanded opening; before any command, the measured one.
            command = normalise_gripper(np.asarray(obs["joint_positions"])[fingers])
            recorder = Recorder()
            step = 0
            rollout = time.time()
            while not env.done:
                if step % decimation == 0:
                    recorder.capture(obs, command)
                action = np.clip(np.asarray(policy.act(obs), dtype=np.float32), low, high)
                command = normalise_gripper(action[fingers])
                obs = env.step(action)[0]
                step += 1
                if step % (decimation * 25) == 0:  # every second of simulated time (campaign stall check)
                    rendered = sum(fr.get("render_s", 0.0) for fr in recorder.frames)
                    print(f"[record] t={env.task.time:.1f}s frames={len(recorder.frames)} "
                          f"wall={time.time() - rollout:.0f}s render={rendered:.0f}s", flush=True)
            # RoboDojo's action[t] is state[t+1] (shift_drop_first_drop_last), so the state after
            # the last recorded frame's period is kept too, without images.
            recorder.capture(obs, command, images=False)
            rollout = time.time() - rollout
            row = env.task.metrics()
            row.update(env.episode_metadata())
            row["events"] = env.task.event_log
            row["terminated"] = bool(env.task.terminated)
        finally:
            env.close()
        finish(dict(row=row, recorder=recorder, instruction=instruction, startup_s=startup,
                    reset_s=reset_s, rollout_s=rollout, control_steps=step))


def write_hdf5(path, result):
    import h5py

    every = result["recorder"].frames
    # Frames 0..N-1 carry images; frame N is the state after the last period (state only).
    frames, following = every[:-1], every[1:]
    n = len(frames)
    if n < 2:
        raise RuntimeError(f"Episode too short to write ({n} frames)")
    tmp = path.with_suffix(".hdf5.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp, "w") as f:
        for a, side in enumerate(SIDES):
            arm = slice(8 * a, 8 * a + 6)
            for group, rows in (("state", frames), ("action", following)):
                f[f"{group}/{side}_arm_joint_states"] = np.stack([fr["joints"][arm] for fr in rows])
                f[f"{group}/{side}_ee_joint_states"] = np.stack([gripper(fr["gripper"][a : a + 1]) for fr in rows])
                f[f"{group}/{side}_ee_poses"] = np.stack([pose7(fr["link6"][a]) for fr in rows])
            # RoboDojo's x5 delta_matrix is the identity: delta_ee_poses equals ee_poses.
            f[f"state/{side}_delta_ee_poses"] = f[f"state/{side}_ee_poses"][()]
        for camera, _ in CAMERAS:
            jpegs = [fr["jpeg"][camera] for fr in frames]
            width = max(len(j) for j in jpegs)
            f.create_dataset(f"vision/{camera}/colors", data=np.array(jpegs, dtype=f"S{width}"))
            f[f"vision/{camera}/extrinsic_matrix"] = np.stack([fr["extrinsic"][camera] for fr in frames])
            f[f"vision/{camera}/intrinsic_matrix"] = np.asarray(result["recorder"].intrinsics[camera], dtype=np.float64)
            f[f"vision/{camera}/shape"] = np.asarray(frames[0]["shape"][camera], dtype=np.int64)
        f["additional_info/frequency"] = np.int64(FORMAT["frequency"])
        f.create_dataset("data_format_version", data=FORMAT["data_format_version"],
                         dtype=h5py.string_dtype("ascii"))
        f.create_dataset("instruction", data=result["instruction"], dtype=h5py.string_dtype("utf-8"))
        f.attrs["fix_obs_shift_rgb"] = "shift_drop_first_drop_last+rgb_jpeg"
    os.replace(tmp, path)
    return n


def gripper(value):
    """RoboDojo's [0, 1] opening (0 closed). A command beyond the closed stop (the household
    oracle squeezes to -0.03 m) is clipped: OpenWAM rejects values outside [0, 1], and the
    XPolicyLab client clips a policy's command the same way."""
    return np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--difficulty", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--keep-failed", action="store_true", help="Also write failed rollouts (debug)")
    args = parser.parse_args()

    task_dir = f"{args.env}_{args.task}"
    meta = args.root / "_meta" / task_dir
    meta.mkdir(parents=True, exist_ok=True)
    run(args, lambda result: save(args, meta, task_dir, result))


def save(args, meta, task_dir, result):
    row = result["row"]
    success = bool(row.get("joint_success"))
    frames = result["recorder"].frames
    status = "success" if success else "fail"
    out = args.root / task_dir / "arx_x5" / "data" / f"episode_{args.seed:07d}.hdf5"
    written = None
    if success or args.keep_failed:
        target = out if success else meta / f"failed_seed_{args.seed}.hdf5"
        written = write_hdf5(target, result)
    arms = result["recorder"].arms
    sidecar = {
        "status": status,
        "environment": args.env,
        "task_id": args.task,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "split": "train",
        "file": str(out.relative_to(args.root)) if success else None,
        "frames": written if written is not None else len(frames) - 1,
        "frequency": FORMAT["frequency"],
        "instruction": result["instruction"],
        "sim_time": frames[-1]["t"],
        "control_steps": result["control_steps"],
        "startup_s": result["startup_s"],
        "reset_s": result["reset_s"],
        "rollout_s": result["rollout_s"],
        "render_s": sum(fr.get("render_s", 0.0) for fr in frames),
        "gripper_clipped_frames": result["recorder"].gripper_clipped,
        "base_separation_m": arms.base_separation_m,
        "base_tilt_deg": arms.base_tilt_deg,
        "scene_rig_origin": arms.scene_origin,
        "scene_rig_yaw_deg": arms.scene_yaw_deg,
        "scene_to_robodojo_world": arms.to_robodojo,
        "fk_ee_error_max_m": max(result["recorder"].ee_errors),
        "joint_success": row.get("joint_success"),
        "task_success": row.get("task_success"),
        "temporal_success": row.get("temporal_success"),
        "termination_reason": row.get("termination_reason")
        or ("task_or_drop" if row.get("terminated") else "time_budget"),
        "metrics": row,
        "code": {"worktree": str(Path(__file__).resolve().parents[2]), "commit": git_head(),
                 "record_demo_sha256": file_sha256(Path(__file__)),
                 "x5_kinematics_sha256": file_sha256(
                     Path(__file__).resolve().parents[2] / "src/roboclock/policies/x5_kinematics.py")},
    }
    tmp = meta / f"seed_{args.seed}.json.tmp"
    tmp.write_text(json.dumps(jsonable(sidecar), indent=1, allow_nan=False))
    os.replace(tmp, meta / f"seed_{args.seed}.json")
    print(json.dumps({k: sidecar[k] for k in ("status", "frames", "sim_time", "startup_s", "reset_s", "rollout_s", "render_s")}))


def file_sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
