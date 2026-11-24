"""Validated configuration and reproducible episode sampling (no simulator imports)."""

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from roboclock.envs import robodojo_rig
from roboclock.envs.lab.robot import (
    HEAD_CAMERA_OFFSET,
    RIG_LEFT_ARM,
    RIG_RIGHT_ARM,
    RIG_SEPARATION,
    head_camera_position,
    mount_positions,
    mount_yaws,
    rig_axes,
)
from roboclock.envs.rates import check_rates
from roboclock.evaluation import throughput

LAB_ROOT = Path(__file__).parent


def load_config(path: str | Path | None = None) -> dict:
    with open(path or LAB_ROOT / "configs/benchmark.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("sim", "scene", "difficulty", "episode", "visuals", "physics", "control"):
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
    robot = config.setdefault("robot", {"model": "dual_arx_x5"})
    if robot.get("model", "dual_arx_x5") != "dual_arx_x5":
        raise ValueError("Laboratory benchmark requires robot.model=dual_arx_x5")
    # Laboratory configs carry the common rig; configs of domains that reuse this
    # loader without a robot section (household) keep the legacy defaults.
    if "rig" in robot:
        check_rig(config)
    elif any(key in robot for key in LEGACY_MOUNT_KEYS) and "liquid_tasks" in config:
        raise ValueError("Laboratory configs must use robot.rig, not per-arm mount positions")
    if config["sim"]["physics_dt"] <= 0 or config["sim"]["control_decimation"] < 1:
        raise ValueError("physics_dt and control_decimation must be positive")
    if not isinstance(config["sim"]["control_decimation"], int):
        raise ValueError("control_decimation must be an integer")
    check_rates(config["sim"]["physics_dt"], config["sim"]["control_decimation"])
    if config["episode"]["max_time"] <= 0:
        raise ValueError("max_time must be positive")
    ticks = config["episode"]["max_time"] / config["sim"]["physics_dt"]
    if not math.isfinite(ticks) or not math.isclose(ticks, round(ticks), abs_tol=1e-6):
        raise ValueError("max_time must be a finite whole number of physics ticks")
    if not 0 < config["episode"]["required_accuracy"] <= 1:
        raise ValueError("required_accuracy must be in (0, 1]")
    if (
        not 1
        <= config["episode"]["num_samples_range"][0]
        <= config["episode"]["num_samples_range"][1]
        <= 50
    ):
        raise ValueError("num_samples_range must be an inclusive range within [1, 50]")
    for level in ("easy", "medium", "hard"):
        spec = config["difficulty"][level]
        for key in ("window_width", "reaction_latency", "transfer_deadline", "required_rate"):
            if key == "required_rate" and key not in spec:
                continue  # lab C tasks have no rate (score_version 2); household still sets one
            lo, hi = spec[key]
            if not 0 < lo <= hi or not math.isfinite(hi):
                raise ValueError(f"Invalid {level}.{key} range")
    throughput_tasks = config.get("throughput_tasks")
    if throughput_tasks is not None and throughput_tasks.get("episode_level") not in (
        config["difficulty"]
    ):
        raise ValueError("throughput_tasks.episode_level must name a difficulty level")
    scene, physics, control = config["scene"], config["physics"], config["control"]
    if not math.isclose(
        scene["table_center"][2] + scene["table_size"][2] / 2, scene["table_height"], abs_tol=1e-6
    ):
        raise ValueError("table_height must equal the physical tabletop surface")
    if not 0 <= physics["rest_offset"] < physics["contact_offset"]:
        raise ValueError("Require 0 <= rest_offset < contact_offset")
    for key in (
        "minimum_grasp_time",
        "contact_force_threshold",
        "relative_hold_speed",
        "recovery_dwell",
        "catch_dwell",
    ):
        if not math.isfinite(physics[key]) or physics[key] <= 0:
            raise ValueError(f"physics.{key} must be finite and positive")
    for key in (
        "max_cartesian_speed",
        "insertion_speed",
        "reactive_speed",
        "joint_target_speed",
        "phase_timeout",
    ):
        if not math.isfinite(control[key]) or control[key] <= 0:
            raise ValueError(f"control.{key} must be finite and positive")
    if len(config["sim"]["camera_resolution"]) != 2 or any(
        not isinstance(v, int) or v < 64 for v in config["sim"]["camera_resolution"]
    ):
        raise ValueError("camera_resolution must contain two integers >= 64")
    camera = config["sim"].get("observation_camera")
    if camera not in {None, "head", "scene"}:
        raise ValueError("sim.observation_camera must be 'head' or 'scene'")
    if camera == "head" and "rig" not in robot:
        raise ValueError("sim.observation_camera=head requires robot.rig")
    return config


LEGACY_MOUNT_KEYS = ("mount_positions", "task_mount_positions", "task_mount_yaws")


# Rig values owned by roboclock.envs.robodojo_rig; a lab config may not carry copies.
SHARED_RIG_KEYS = ("separation", "head_camera_offset", "head_camera_pitch_deg")


def check_robodojo_sim(config):
    """The RoboDojo time base, policy cameras and renderer for every lab task.

    Physics runs at exactly robodojo_rig.PHYSICS_HZ (not a multiple), the policy
    cameras are 640x480 and the policy `rgb` is the rig's head camera.
    """
    sim = config["sim"]
    if not math.isclose(1 / sim["physics_dt"], robodojo_rig.PHYSICS_HZ, abs_tol=1e-9):
        raise ValueError(f"sim.physics_dt must be exactly 1/{robodojo_rig.PHYSICS_HZ} s")
    if list(sim["camera_resolution"]) != list(robodojo_rig.HEAD_CAMERA["resolution"]):
        raise ValueError(
            f"sim.camera_resolution must be {list(robodojo_rig.HEAD_CAMERA['resolution'])} "
            "(policy head/wrist cameras)"
        )
    if sim.get("robodojo_cameras") is not True:
        raise ValueError("Laboratory configs must set sim.robodojo_cameras: true")
    if sim.get("observation_camera", "head") != "head":
        raise ValueError("Laboratory policy rgb is the rig head camera (sim.observation_camera: head)")
    overview = sim.get("overview_resolution", [1280, 960])
    if len(overview) != 2 or any(not isinstance(v, int) or v < 64 or v % 2 for v in overview):
        raise ValueError("sim.overview_resolution must contain two even integers >= 64")
    if not isinstance(sim.get("no_render", False), bool):
        raise ValueError("sim.no_render must be true or false")


def check_rig(config):
    """Validate the common dual-X5 rig and the mounts/camera derived from it.

    Every lab task uses the same RoboDojo rig (roboclock.envs.robodojo_rig): both bases at
    one height and one forward coordinate, both facing the rig's forward direction,
    exactly ARM_SEPARATION_M apart, and the head camera at HEAD_IN_RIG_M, pitched
    HEAD_PITCH_DOWN_DEG, without yaw relative to the rig. Only the rig origin (including
    mount height) and yaw may differ by task.
    """
    from roboclock.envs.lab.tasks.registry import TASKS

    robot = config["robot"]
    legacy = [key for key in LEGACY_MOUNT_KEYS if key in robot]
    if legacy:
        raise ValueError(f"robot.{legacy[0]} is replaced by robot.rig (one common dual-X5 rig)")
    rig = robot.get("rig")
    if not isinstance(rig, dict) or "default" not in rig:
        raise ValueError("robot.rig with a default origin/yaw is required")
    copies = [key for key in SHARED_RIG_KEYS if key in rig]
    if copies:
        raise ValueError(f"robot.rig.{copies[0]} is fixed by roboclock.envs.robodojo_rig")
    check_robodojo_sim(config)
    unknown = set(rig.get("tasks", {})) - set(TASKS)
    if unknown:
        raise ValueError(f"robot.rig.tasks has unknown task ids: {sorted(unknown)}")
    if config["episode"]["num_samples_range"][1] > 32:
        raise ValueError("The rig's C2/C3 grids hold at most 32 samples (num_samples_range)")
    table = config["scene"]["table_height"]
    for task_id in TASKS:
        pose = {**rig["default"], **rig.get("tasks", {}).get(task_id, {})}
        if set(pose) - {"origin", "yaw"}:
            raise ValueError(
                f"robot.rig {task_id}: only origin and yaw are allowed (the head camera "
                "looks along the rig's forward direction)"
            )
        origin = np.asarray(pose.get("origin"), dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError(f"robot.rig {task_id}: origin must be a finite xyz point")
        if origin[2] <= table:
            raise ValueError(f"robot.rig {task_id}: mount plane must be above the table")
        if not math.isfinite(float(pose.get("yaw", 0.0))):
            raise ValueError(f"robot.rig {task_id}: yaw must be finite")
        # Check the geometry actually derived for the simulator, not just the inputs.
        mounts = np.asarray(mount_positions(config, task_id), dtype=float)
        yaws = mount_yaws(config, task_id)
        forward, left = rig_axes(yaws[0])
        if not math.isclose(yaws[0], yaws[1], abs_tol=1e-12):
            raise ValueError(f"{task_id}: both arms must face the same direction")
        if not math.isclose(mounts[0][2], mounts[1][2], abs_tol=1e-9):
            raise ValueError(f"{task_id}: both arm bases must be at the same height")
        if not math.isclose(float((mounts[1] - mounts[0]) @ forward), 0.0, abs_tol=1e-9):
            raise ValueError(f"{task_id}: both arm bases must share one forward coordinate")
        if not math.isclose(float((mounts[1] - mounts[0]) @ left), RIG_SEPARATION, abs_tol=1e-9):
            raise ValueError(f"{task_id}: arm bases must be {RIG_SEPARATION} m apart, side by side")
        camera = head_camera_position(config, task_id)
        expected = mounts.mean(axis=0) + forward * HEAD_CAMERA_OFFSET[0] + [0, 0, HEAD_CAMERA_OFFSET[2]]
        if not np.allclose(camera, expected, atol=1e-9):
            raise ValueError(f"{task_id}: head camera must sit at the rig's head mount")
        # The same check every reset runs on measured stage poses.
        left_base, right_base = (
            robodojo_rig.rig_to_world(mounts[side], math.degrees(yaws[side]))
            for side in (RIG_LEFT_ARM, RIG_RIGHT_ARM)
        )
        robodojo_rig.check_rig(
            left_base,
            right_base,
            robodojo_rig.head_camera_world(mounts.mean(axis=0), math.degrees(yaws[0])),
        )


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: str
    difficulty: str
    seed: int
    valid_start: float
    valid_end: float
    event_time: float
    max_latency: float
    transfer_deadline: float
    # Operations / second. None in lab: the C tasks judge t_Q against time tiers instead
    # (score_version 2, docs/c-class-timing.md); household configs still set one.
    required_rate: float | None
    required_accuracy: float
    num_samples: int
    max_time: float
    position_tolerance: float
    orientation_tolerance: float
    phase: float


def sample_episode(task_id: str, difficulty: str, seed: int, config: dict) -> EpisodeSpec:
    if difficulty not in config["difficulty"]:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    level, episode = config["difficulty"][difficulty], config["episode"]
    level = {**level, **config.get("reactive_tasks", {}).get(task_id, {}).get(difficulty, {})}
    for key in ("reaction_latency", "transfer_deadline"):
        lo, hi = level[key]
        if not 0 < lo <= hi or not math.isfinite(hi):
            raise ValueError(f"Invalid {task_id}.{difficulty}.{key} range")
    start = float(rng.uniform(*episode["window_start_range"]))
    width = float(rng.uniform(*level["window_width"]))
    event_time = float(rng.uniform(*episode["event_time_range"]))
    max_latency = float(rng.uniform(*level["reaction_latency"]))
    transfer_deadline = float(rng.uniform(*level["transfer_deadline"]))
    if "required_rate" in level:
        required_rate = float(rng.uniform(*level["required_rate"]))
    else:
        # The draw the rate used to take, so every seed keeps its sample count and phase.
        rng.random()
        required_rate = None
    return EpisodeSpec(
        task_id,
        difficulty,
        seed,
        start,
        start + width,
        event_time,
        max_latency,
        transfer_deadline,
        required_rate,
        episode["required_accuracy"],
        int(rng.integers(episode["num_samples_range"][0], episode["num_samples_range"][1] + 1)),
        episode["max_time"],
        level["position_tolerance"],
        level["orientation_tolerance"],
        float(rng.uniform(-np.pi, np.pi)),
    )


def throughput_episode(task_id: str, difficulty: str, seed: int, config: dict) -> EpisodeSpec:
    """Episode of a lab C task (batch C task, score_version 2; docs/c-class-timing.md).

    Difficulty changes nothing in the episode: every difficulty samples the
    ``throughput_tasks.episode_level`` values (the former medium tolerances), so the spec differs
    only in its ``difficulty`` label, which picks the time tier the row is judged at. max_time is
    the horizon H = 5 t_o(Q) from the frozen oracle time of the task and its sample count Q.
    """
    if difficulty not in config["difficulty"]:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    level = (config.get("throughput_tasks") or {}).get("episode_level", "medium")
    spec = sample_episode(task_id, level, seed, config)
    t_o = throughput.oracle_time("lab", task_id, spec.num_samples)
    return replace(spec, difficulty=difficulty, max_time=throughput.horizon(t_o))
