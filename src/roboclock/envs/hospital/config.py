"""Validated hospital configuration and reproducible episode sampling (no simulator imports)."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from roboclock.envs.rates import check_rates

HOSPITAL_ROOT = Path(__file__).parent
LEVELS = ("easy", "medium", "hard")
STAT_ARRIVAL = 0.3  # A1: the STAT request arrives this fraction of a gap after the previous one
PREP_EXTRA_FIRST = 14.0  # A4 with dose preparation: extra seconds before the first expiry
PREP_EXTRA_GAP = (
    7.0  # and between expiries (one dock, one robot: the doses go through it one at a time)
)
CRITICAL_WIDTH = 0.7  # B1: the critical alarm's limit is this fraction of a normal one
STAT_WIDTH = 0.7  # A1: its window is this fraction of a normal one

# Quantities each task samples per difficulty level; the order fixes the random stream.
TASK_QUANTITIES = {
    "A1": ("window_width", "window_gap"),
    "A2": ("transit_limit",),
    "A3": ("window_width", "window_gap"),
    "A4": ("expiry_time", "expiry_gap"),  # legacy profile; the v2 A4 is a bedside task
    "B1": ("response_limit", "alarm_gap", "alarm_count"),
    "B2": ("response_limit",),
    "B3": ("response_limit",),
    # Ward bedside tasks B4-B6 and A4 (bedside_profile): one response limit from the instruction.
    "B4": ("response_limit",),
    "B5": ("response_limit",),
    "B6": ("response_limit",),
    "C1": ("cycle_time", "required_rate_per_min"),
    "C2": ("cycle_time", "required_rate_per_min"),
    "C3": ("cycle_time", "required_rate_per_min"),
    "C4": ("cycle_time", "required_rate_per_min"),
}


def profile_quantities(task_id, task):
    """Per-level quantities a task block samples; ward profiles replace the historical ones."""
    if task.get("bedside_profile") == "bedside":
        return ("response_limit",)
    if task.get("window_profile") in ("process", "rhythm"):
        return ("window_width",)
    if task.get("ward_profile") == "ward":
        return ()  # ward C1-C4: nothing changes with difficulty (docs/c-class-timing.md)
    return TASK_QUANTITIES[task_id]


def ward_horizon(task_id: str) -> float:
    """Ward C1-C4 episode length H = throughput.horizon(t_o), the same at every difficulty."""
    from roboclock.evaluation import throughput

    return round(throughput.horizon(throughput.oracle_time("hospital", task_id)), 6)


def _check_range(value, label):
    lo, hi = value
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
        raise ValueError(f"Invalid range for {label}: {value}")


def load_config(path: str | Path | None = None) -> dict:
    with open(path or HOSPITAL_ROOT / "configs/benchmark.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("sim", "scene", "difficulty", "episode", "visuals", "physics", "control", "tasks"):
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
    sim, episode = config["sim"], config["episode"]
    if sim["physics_dt"] <= 0 or sim["control_decimation"] < 1:
        raise ValueError("physics_dt and control_decimation must be positive")
    if not isinstance(sim["control_decimation"], int):
        raise ValueError("control_decimation must be an integer")
    check_rates(sim["physics_dt"], sim["control_decimation"])
    if episode["max_time"] <= 0:
        raise ValueError("max_time must be positive")
    ticks = episode["max_time"] / sim["physics_dt"]
    if not math.isfinite(ticks) or not math.isclose(ticks, round(ticks), abs_tol=1e-6):
        raise ValueError("max_time must be a finite whole number of physics ticks")
    for level in LEVELS:
        if not 0 < config["difficulty"][level]["orientation_tolerance"] < math.pi / 2:
            raise ValueError(f"Invalid {level}.orientation_tolerance")
    unknown = set(config["tasks"]) - set(TASK_QUANTITIES)
    if unknown:
        raise ValueError(f"Unknown task configuration: {', '.join(sorted(unknown))}")
    for task_id, quantities in TASK_QUANTITIES.items():
        if task_id not in config["tasks"]:
            continue  # the v2 profile carries A1-A4/B1-B6/C1-C4; the archived one A1-A4/B1-B3/C1-C4
        task = config["tasks"][task_id]
        count_lo, count_hi = task["num_items_range"]
        if not 1 <= count_lo <= count_hi <= 30:
            raise ValueError(f"{task_id}.num_items_range must be an inclusive range in [1, 30]")
        accuracy = task.get("required_accuracy")
        if accuracy is not None and not 0 < accuracy <= 1:
            raise ValueError(f"{task_id}.required_accuracy must be in (0, 1] or null")
        if task.get("window_profile") == "process":
            key = "fill_lead_range" if "fill_lead_range" in task else "window_open_range"
            _check_range(task[key], f"{task_id}.{key}")
        if task.get("window_profile") == "rhythm":
            _check_range(task["first_beat_range"], f"{task_id}.first_beat_range")
            if not float(task["beat_period"]) > 0:
                raise ValueError(f"{task_id}.beat_period must be positive")
            for level in LEVELS:  # windows of consecutive beats must not touch
                if task[level]["window_width"][1] >= float(task["beat_period"]):
                    raise ValueError(f"{task_id}.{level}.window_width must be shorter than beat_period")
                if task[level]["window_width"][1] / 2 >= task["first_beat_range"][0]:
                    raise ValueError(f"{task_id}.{level}: the first window opens before the episode")
        if task.get("ward_profile") == "ward":
            # The ward C tasks are judged from frozen oracle times (roboclock.evaluation.throughput)
            # and run to H; no per-level budget or rate may be configured.
            # Q is fixed: the frozen t_o is the time of the Q-th job.
            for level in LEVELS:
                if task.get(level):
                    raise ValueError(f"{task_id}: ward C tasks take no per-level quantities")
            if count_lo != count_hi:
                raise ValueError(f"{task_id}: ward C tasks need a fixed num_items_range (Q)")
            continue
        for level in LEVELS:
            for quantity in profile_quantities(task_id, task):
                _check_range(task[level][quantity], f"{task_id}.{level}.{quantity}")
            if "required_rate_per_min" in quantities:
                # The rate must be reachable with the items present in the time budget.
                budget = task[level].get("cycle_time", [episode["max_time"]] * 2)[1]
                if budget > episode["max_time"]:
                    raise ValueError(f"{task_id}.{level} budget exceeds episode.max_time")
                if task[level]["required_rate_per_min"][1] * budget / 60 > count_lo:
                    raise ValueError(
                        f"{task_id}.{level} throughput target exceeds the available items per "
                        "time budget"
                    )
    if not 0.02 <= config["tasks"]["B3"]["cover_radius"] <= 0.12:
        raise ValueError("B3 cover_radius is outside the supported range")
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
    if len(sim["camera_resolution"]) != 2 or any(
        not isinstance(v, int) or v < 64 for v in sim["camera_resolution"]
    ):
        raise ValueError("camera_resolution must contain two integers >= 64")
    if "rig" in config.get("robot", {}):  # the RoboDojo rig; archived profiles keep mounts
        from roboclock.envs import robodojo_rig
        from roboclock.envs.hospital.rig import check_rig

        check_rig(config, list(config["tasks"]))
        # RoboDojo spec: physics exactly 250 Hz (not a multiple), the head image at the spec size.
        if not math.isclose(sim["physics_dt"] * robodojo_rig.PHYSICS_HZ, 1.0, abs_tol=1e-12):
            raise ValueError(f"sim.physics_dt must be 1/{robodojo_rig.PHYSICS_HZ} s on the rig")
        if list(sim["camera_resolution"]) != list(robodojo_rig.HEAD_CAMERA["resolution"]):
            raise ValueError("sim.camera_resolution must be the robodojo_rig head camera resolution")
        for task_id, task in config["tasks"].items():
            if any(key in task for key in ("physics_dt", "control_decimation")):
                raise ValueError(f"{task_id}: per-task physics rates are not allowed on the rig")
    return config


@dataclass(frozen=True)
class EpisodeSpec:
    """Quantities a task does not use stay None. Times are seconds of simulation time."""

    task_id: str
    difficulty: str
    seed: int
    max_time: float
    num_items: int
    orientation_tolerance: float
    windows: tuple[tuple[float, float], ...] = ()  # A1 per-patient, A3 per-step [t_o, t_c]
    alarms: tuple[float, ...] = ()  # B1: alarm times when there is more than one
    alarm_limits: tuple[float, ...] = ()  # B1: response limit of each alarm; the last is critical
    stat: int | None = None  # A1: index of the urgent (STAT) request, which arrives between others
    transit_limit: float | None = None  # A2: D, t_place - t_remove < D
    expiry_time: float | None = None  # A4: earliest T_exp, t_place < T_exp
    expiries: tuple[float, ...] = ()  # A4: beyond-use time of each syringe, earliest first
    event_time: float | None = None  # B1 alarm, B2 cardiac arrest, B3 fire
    response_limit: float | None = None  # B1-B3: delta t, strict
    cycle_time: float | None = None  # legacy C1-C3: the deadline, also the episode length
    required_rate_per_min: float | None = None  # legacy C1-C3: R, items per minute
    required_accuracy: float | None = None  # C tasks: sets the allowed final misplacements
    oracle_time: float | None = None  # ward C1-C4: frozen t_o (roboclock.evaluation.throughput)


def sample_episode(task_id: str, difficulty: str, seed: int, config: dict) -> EpisodeSpec:
    """Draw from fixed uniform streams so a seed keeps its geometry across difficulty levels."""
    if difficulty not in LEVELS:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if task_id not in TASK_QUANTITIES:
        raise ValueError(f"Unknown task: {task_id}")
    rng = np.random.default_rng(seed)
    draws = rng.uniform(size=4)
    start_u, event_u, count_u = rng.uniform(size=3)
    task, episode = config["tasks"][task_id], config["episode"]
    level = task.get(difficulty) or {}
    values = {
        key: float(lo + (hi - lo) * u)
        for u, key in zip(
            draws,
            profile_quantities(task_id, task),
        )
        for lo, hi in [level[key]]
    }
    lo, hi = task["num_items_range"]
    count = int(lo + min(int(count_u * (hi - lo + 1)), hi - lo))
    start = float(
        episode["window_start_range"][0]
        + (episode["window_start_range"][1] - episode["window_start_range"][0]) * start_u
    )
    event = float(
        episode["event_time_range"][0]
        + (episode["event_time_range"][1] - episode["event_time_range"][0]) * event_u
    )
    fields = {
        "max_time": episode["max_time"],
        "num_items": count,
        "orientation_tolerance": config["difficulty"][difficulty]["orientation_tolerance"],
    }
    if task.get("ward_profile") == "ward":
        # Ward C1-C4 (score_version 2): one episode for all three tiers. It runs to H = 5 t_o,
        # Q = num_items, and tolerances stay at their former medium values whatever the level;
        # the level only picks which time tier fills temporal/joint success.
        from roboclock.evaluation import throughput

        fields["orientation_tolerance"] = config["difficulty"]["medium"]["orientation_tolerance"]
        fields["required_accuracy"] = task.get("required_accuracy")
        fields["oracle_time"] = throughput.oracle_time("hospital", task_id)
        fields["max_time"] = ward_horizon(task_id)
        return EpisodeSpec(task_id, difficulty, seed, **fields)
    if task.get("bedside_profile") == "bedside":
        # A4 and B4-B6 may issue their instruction at a fixed time (instruction_time, e.g. right after the
        # scene settles): the robot starts from rest and the limit runs from the instruction to
        # completion. Otherwise the cue arrives at a sampled event time.
        instruction = task.get("instruction_time")
        fields["event_time"] = event if instruction is None else float(instruction)
        fields["response_limit"] = values["response_limit"]
        fields["max_time"] = min(
            float(episode["max_time"]),
            fields["event_time"] + max(12.0, values["response_limit"] + 4.0),
        )
    elif task.get("window_profile") == "process":
        # Ward A1-A3: the monitored quantity enters its band at a seeded time (the same for every
        # level of a seed) and leaves it window_width later; the episode shows a short tail after.
        # With `fill_lead_range` (A1) the process starts only when the robot opens the device: the
        # window is then relative to that opening (seconds after it), and the episode allows
        # `open_allowance` seconds for the opening on top.
        relative = "fill_lead_range" in task
        lo, hi = task["fill_lead_range" if relative else "window_open_range"]
        opening = float(lo + (hi - lo) * start_u)
        closing = opening + values["window_width"]
        fields["windows"] = ((opening, closing),)
        fields["event_time"] = opening
        allowance = float(task.get("open_allowance", 0.0)) if relative else 0.0
        fields["max_time"] = min(
            float(episode["max_time"]), allowance + closing + float(task.get("tail", 3.0))
        )
    elif task.get("window_profile") == "rhythm":
        # Ward A3 (bag-valve ventilation): one action per beat. The beats are a seeded first beat
        # (the same for every level of a seed) and then one every beat_period seconds, num_items in
        # all; each window is centered on its beat and window_width wide.
        lo, hi = task["first_beat_range"]
        first = float(lo + (hi - lo) * start_u)
        period, half = float(task["beat_period"]), values["window_width"] / 2
        fields["windows"] = tuple(
            (first + k * period - half, first + k * period + half) for k in range(count)
        )
        fields["event_time"] = fields["windows"][0][0]
        fields["max_time"] = min(
            float(episode["max_time"]), fields["windows"][-1][1] + float(task.get("tail", 3.0))
        )
    elif task_id in ("A1", "A3"):
        fields["windows"] = tuple(
            (
                start + i * values["window_gap"],
                start + i * values["window_gap"] + values["window_width"],
            )
            for i in range(count)
        )
        if task_id == "A1" and count >= 3:
            # The last request is STAT: it arrives shortly after the previous request and has a shorter
            # window, so its deadline falls before that of the request that arrived just before it.
            width, gap = values["window_width"], values["window_gap"]
            urgent = start + (count - 2) * gap + STAT_ARRIVAL * gap
            windows = list(fields["windows"])
            windows[-1] = (urgent, urgent + STAT_WIDTH * width)
            fields["windows"] = tuple(windows)
            fields["stat"] = count - 1
    elif task_id == "A2":
        fields["transit_limit"] = values["transit_limit"]
    elif task_id == "A4":
        # With dose preparation every dose also spends time on the filling dock, so its deadline moves out.
        prep = bool(task.get("dose_prep", False))
        first = values["expiry_time"] + (PREP_EXTRA_FIRST if prep else 0.0)
        gap = values["expiry_gap"] + (PREP_EXTRA_GAP if prep else 0.0)
        fields["expiry_time"] = first
        fields["expiries"] = tuple(first + i * gap for i in range(count))
    elif task_id in ("B1", "B2", "B3"):
        fields["event_time"] = event
        fields["response_limit"] = values["response_limit"]
        if task_id == "B1" and round(values["alarm_count"]) > 1:
            # Several beds alarm one after another, faster than a delivery takes. The last alarm is a
            # critical "code" with a shorter limit, so the earliest deadline is not the first alarm.
            alarms = round(values["alarm_count"])
            fields["alarms"] = tuple(event + i * values["alarm_gap"] for i in range(alarms))
            fields["alarm_limits"] = (values["response_limit"],) * (alarms - 1) + (
                CRITICAL_WIDTH * values["response_limit"],
            )
    else:
        fields["required_rate_per_min"] = values["required_rate_per_min"]
        fields["required_accuracy"] = task.get("required_accuracy")
        if task_id in ("C1", "C2", "C3", "C4"):
            fields["cycle_time"] = round(values["cycle_time"] * 2) / 2
            fields["max_time"] = fields["cycle_time"]
    return EpisodeSpec(task_id, difficulty, seed, **fields)


def load_legacy_config(path=None):
    """Historical A1-A4 profile used only for archived demonstrations/regressions."""
    return load_config(path or HOSPITAL_ROOT / "configs/legacy_benchmark.yaml")
