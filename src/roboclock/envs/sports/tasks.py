"""Catalog, episode specs and result rows of the ten current Dual ARX X5 sports tasks.

Each task keeps the scene, physics, privileged controller and scoring of its
sub-package (see README.md). This module is simulator-free: it builds the
sub-package's native configuration exactly as the recording driver
(``python -m roboclock.envs.sports``) does, derives the episode budget, and turns
the native measured result into the shared result row (``results.normalize``).
"""

import copy
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from roboclock.envs.sports import c_class, spatial
from roboclock.envs.sports.config import check_rates, load_config, task_rates
from roboclock.envs.sports.results import normalize

CATEGORY = {"A": "temporal_window", "B": "event_reactive", "C": "throughput"}

# (task ID, name, discipline, sub-package providing scene/scoring/controller)
CATALOG = (
    ("A1", "chess_exchange_and_clock", "chess", "chess"),
    ("A2", "basketball_shot_clock", "basketball", "arena"),
    ("B1", "nine_hole_mallet_moles", "arcade", "arcade"),
    ("B2", "reactive_boxing", "boxing", "arena"),
    ("B3", "bench_press_assistance", "strength", "bench_press"),
    ("B4", "stop_go_puck_push", "table_game", "freeze"),
    ("B5", "two_stick_taiko", "rhythm", "arcade"),
    ("C1", "pool_ball_racking", "billiards", "pool_rack"),
    ("C2", "speed_cups", "cup_stacking", "arena"),
    ("C3", "ball_service", "tennis", "ball_service"),
)
SOURCES = {task_id: source for task_id, _, _, source in CATALOG}
# Native task IDs written by the arcade sub-package's original task classes.
ARCADE_NATIVE_IDS = {"B5": "32", "B1": "34"}
# Benchmark ID -> the sub-packages' own task kind where they differ. On 2026-11-19 taiko
# (kind A2) became B5, an event-reactive task, and basketball (kind A3) became A2; the
# sub-packages (arcade, arena, embodiment config) keep their kinds.
NATIVE_KINDS = {"A2": "A3", "B5": "A2"}


def native_kind(task_id):
    """The sub-package task kind of a benchmark task ID."""
    return NATIVE_KINDS.get(task_id, task_id)


@dataclass(frozen=True)
class SportsEpisode:
    task_id: str
    difficulty: str
    seed: int
    max_time: float
    physics_dt: float
    control_decimation: int
    source: str
    instruction: str
    timing: dict = field(default_factory=dict)
    # Seeded spatial offsets of the station (``spatial.py``); empty for tasks without them.
    layout: dict = field(default_factory=dict)


def native_config(task_id, difficulty="easy", seed=0, config=None):
    """The sub-package configuration the recording driver uses, plus the policy camera.

    Only rendering settings change: the policy camera resolution, a native-scale
    render and no extra robot-view recording cameras. Physics, layout, timing,
    controller and scoring parameters are those of the recording driver.
    """
    from roboclock.envs.start_pose import start_pose
    from roboclock.simulators.isaac.embodiments.config import configure

    config = config or load_config()
    if task_id not in SOURCES:
        raise ValueError(f"Unknown sports task {task_id!r}; choose one of {', '.join(SOURCES)}")
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(f"Unknown sports difficulty {difficulty!r}")
    seed = int(seed)
    kind = native_kind(task_id)
    if task_id == "A1":
        from roboclock.envs.sports.chess.config import config as chess_config

        c = chess_config(difficulty, seed)
        configure(c, "A1", "dual_arx_x5")
    elif kind in ("A2", "B1"):  # arcade: taiko (B5) and moles
        from roboclock.envs.sports.arcade.tasks import configuration

        c = configuration(kind, "basic", "v5")
        configure(c, kind, "dual_arx_x5")
    elif kind in ("A3", "B2", "C2"):  # arena: basketball (A2), boxing and cups
        from roboclock.envs.sports.arena.config import config as arena_config

        c = arena_config(kind, difficulty, seed)
        configure(c, kind, "dual_arx_x5")
    elif task_id == "B3":
        from roboclock.envs.sports._shared.precision.config import config as precision_config

        c = precision_config("B3", difficulty, seed)
        configure(c, "B3", "dual_arx_x5")
    elif task_id == "B4":
        from roboclock.envs.sports.freeze.config import config as freeze_config

        c = freeze_config(difficulty, seed)
    elif task_id == "C1":
        from roboclock.envs.sports.pool_rack.config import config as rack_config

        c = rack_config(difficulty, seed)
    else:
        from roboclock.envs.sports.ball_service.config import config as service_config

        # Optional calibration override of the demand schedule (configs/c3_eager.yaml).
        c = service_config(difficulty, seed, schedule=config.get("c3_schedule"))
    check_rates(config, task_id, c)
    # Seeded station offsets from the task's own layout stream (2026-11-20, ``spatial.py``).
    spatial.apply(task_id, seed, c)
    c["sim"]["camera_resolution"] = list(config["sim"]["camera_resolution"])
    # Shared key robot.start_pose (roboclock.envs.start_pose): "zero" starts the policy's arms
    # at joint 0 instead of the task's IK pose; read where the rig initializes.
    c["embodiment"]["start_pose"] = start_pose(config)
    c["visuals"]["render_scale"] = 1
    c["visuals"]["record_robot_views"] = False
    c.setdefault("seed", seed)
    c.setdefault("difficulty", difficulty)
    return c


def _throughput_timing(task_id, jobs, start):
    """Spec timing of a batch C task: every tier's budget and the horizon (no tier picked)."""
    from roboclock.evaluation import throughput

    t_o = c_class.oracle_time(task_id)
    return dict(
        jobs=jobs,
        time_origin=start,
        oracle_time=t_o,
        budget_by_tier=throughput.budgets(t_o),
        horizon=start + c_class.horizon(task_id),
    )


def _episode(task_id, difficulty, seed, c):
    """Budget = the native evaluator's own end plus one second, or its native cap.

    C-class tasks (score_version 2) run to their horizon instead: batch tasks (C1, C2) to
    H = 5 t_o after the time origin at every difficulty, C3 to 1.25 x its last demand.

    The instruction states every rule the task's evaluator judges that a policy cannot read
    from the scene: limits, tolerances, forbidden contacts and, where only one arm is read,
    which arm. Its numbers come from the evaluators' own constants and configuration. It
    must fit the observation space (512 characters, ``env.py``).
    """
    timing = {}
    kind = native_kind(task_id)
    if task_id == "A1":
        timing = dict(
            clock_deadline=c["difficulties"][difficulty]["deadline"], moves=list(c["duel"]["moves"])
        )
        max_time = float(c["duel"]["max_time"])
        # Lazy: the adapter module is only needed for its handoff tolerance.
        from roboclock.envs.sports.adapters.chess import RETURN_TOLERANCE

        instruction = (
            "You play WHITE with the right X5 arm. After BLACK presses the clock, grasp the "
            "scheduled piece, place it upright and wholly inside its destination square, "
            "release it and press your clock button within "
            f"{timing['clock_deadline']:g} s of BLACK's press. Then return the right arm to "
            f"the joint pose it started in (every joint within {RETURN_TOLERANCE:g} rad): "
            "BLACK gets the turn only then. Do not move other pieces, touch BLACK's button "
            "or let an arm link touch the table, board, clock or pieces."
        )
    elif task_id == "B5":  # taiko (arcade kind A2)
        settings, limits = c["tasks"]["32"], c["difficulties"][difficulty]
        timing = dict(note_window=limits["note_window"], first_note=12.0)
        max_time = float(settings["max_time"])
        instruction = (
            "Pick up both sticks. LEFT is robot-left (+Y), RIGHT is robot-right (-Y). Strike "
            "each cued drum with the head of a held stick within "
            f"{limits['note_window']:g} s before or after its note; paired notes are "
            "simultaneous. Any other drum strike is a fault. Lift the sticks between hits. "
            "Do not drop a stick or let an arm link touch the table, drums or stands."
        )
    elif task_id == "B1":
        settings, limits = c["tasks"]["34"], c["difficulties"][difficulty]
        timing = dict(deadline=limits["deadline"])
        max_time = float(settings["max_time"])
        instruction = (
            "Pick up both mallets. Hit each green mole with the head of a held mallet within "
            f"{limits['deadline']:g} s of its rise, before it retracts. Never touch a red "
            "mole with a mallet or gripper. Do not drop a mallet or let an arm link touch "
            "the table, moles or stands."
        )
    elif kind in ("A3", "B2", "C2"):
        from roboclock.envs.sports.arena import tasks as arena

        w = c["workflow"]
        if task_id == "A2":  # basketball (arena kind A3)
            timing = dict(
                shot_start=w["shot_start"],
                shot_clock=w["shot_clock"],
                flight_grace=w["flight_grace"],
            )
            end = w["shot_start"] + w["shot_clock"] + w["flight_grace"]
            # Only the active arm's jaws are read for the grasp and release (the X5 backend
            # builds A3 single-armed), so the instruction names it.
            instruction = (
                f"Grasp the ball with the {c['embodiment']['active_side']} gripper and throw "
                "it through the hoop from above. Release it after READY and before the "
                f"{w['shot_clock']:g} s shot clock expires, with the ball higher than "
                f"{arena.RELEASE_MIN_HEIGHT:g} m and at least {arena.NO_DUNK_RADIUS:g} m "
                "(horizontally) from the hoop's centre: a dunk does not count. Do not touch "
                "the ball again after release, and do not let an arm link hit the table."
            )
        elif task_id == "B2":
            from roboclock.envs.sports.arena.boxing import schedule

            commands = schedule(w, c["seed"])
            timing = dict(
                cue_response=w["cue_response"],
                guard_deadline=w["guard_deadline"],
                strikes=sum(len(command["hands"]) for command in commands),
            )
            end = commands[-1]["end"]
            instruction = (
                "Hit the lit pad with the indicated hand's glove within "
                f"{w['cue_response']:g} s of each GO cue, once per cue. Do not punch during "
                "WAIT or HOLD; during HOLD keep both gloves within "
                f"{arena.GUARD_HOLD * 100:g} cm of guard. After a hit return that glove to "
                f"guard within {w['guard_deadline']:g} s; the other glove stays within "
                f"{arena.GUARD_OTHER * 100:g} cm of guard. Guard is the glove centre (ee "
                f"position) within {arena.GUARD_RETURN * 100:g} cm of x {arena.GUARD_X:g}, "
                f"y +/-{arena.GUARD_Y:g} (left +), z {arena.GUARD_Z:g} m. Do not let an arm "
                "link hit the pads or table."
            )
        else:
            timing = dict(
                preparation=w["preparation"],
                cups_required=w["cups_required"],
                **_throughput_timing("C2", w["cups_required"], w["preparation"]),
            )
            instruction = (
                "Match the displayed cup pattern: every cup released, upright and at rest "
                f"within {arena.CUP_TOLERANCE * 1000:g} mm of its place, stacked cups resting "
                "on the cups below. Then press the bell once to advance. A bell rung on an "
                "incomplete pattern or before the preparation phase ends fails the run and "
                "cannot be undone. After the preparation phase, complete the "
                f"{w['cups_required']} cards as quickly as possible. Do not knock a cup off "
                "the table."
            )
        max_time = float(timing["horizon"]) if task_id == "C2" else float(end) + 1.0
    elif task_id == "B3":
        w = c["workflow"]
        timing = dict(catch_deadline=w["catch_deadline"], timeout=w["bench_timeout"])
        max_time = float(w["bench_timeout"]) + 1.0
        from roboclock.envs.sports.bench_press import task as bench

        instruction = (
            "Wait in standby with both hands off the bar and both grippers open wider than "
            f"{bench.STANDBY_GAP * 1000:g} mm. When the lifter stalls and the bar starts to "
            f"fall, catch it with both hands within {w['catch_deadline']:g} s: each gripper "
            f"closed on the bar within {bench.GRIP_TOLERANCE * 100:g} cm of the point "
            f"{w['bench_grip_y']:g} m from the bar's centre on its side. Then carry it back "
            "onto both rack hooks and release it, opening both grippers wider than "
            f"{bench.RELEASE_GAP * 1000:g} mm. The bar must never touch the lifter or drop "
            f"below {w['human_danger_height']:g} m, and no arm link may hit the bar or rack."
        )
    elif task_id == "B4":
        f = c["freeze"]
        timing = dict(response_limit=f["response_limit"], rounds=f["rounds"], horizon=f["horizon"])
        max_time = float(f["horizon"]) + 1.0
        from roboclock.envs.sports.freeze import task as freeze

        rest = min(f["hand_linear_limit"], f["body_linear_limit"])
        instruction = (
            "Grasp both pushers. On GREEN, push the puck with the indicated hand's pusher "
            "toward the next goal, keeping it on the table; the indicated hand changes once "
            f"the puck is {freeze.CHECKPOINT_PAST_MIDLINE * 100:g} cm past the midline within "
            f"{freeze.CHECKPOINT_LANE_HALF_WIDTH * 100:g} cm of the lane's centre line, and "
            "the goal counts when that hand has pushed the whole puck into it. On RED, bring "
            f"hands, pushers and puck to rest (under {rest * 100:g} cm/s) within "
            f"{f['response_limit']:g} s and hold them within "
            f"{f['hold_translation'] * 100:g} cm until GREEN. Complete {f['rounds']} goals. "
            "Do not drop a pusher or let an arm link hit the table or the other arm."
        )
    elif task_id == "C1":
        timing = dict(balls=15, **_throughput_timing("C1", 15, 0.0))
        max_time = float(timing["horizon"])
        from roboclock.envs.sports.pool_rack import task as rack

        instruction = (
            "Rack all 15 numbered balls into a legal eight-ball triangle (8 in the center, a "
            "solid and a stripe in the back corners) as quickly as possible, then withdraw "
            "both hands. A ball counts once it has been grasped, lifted at least "
            f"{rack.LIFT_HEIGHT * 100:g} cm, released and rests within "
            f"{c['rack']['placement_tolerance'] * 1000:g} mm of its rack spot, neighbours "
            f"no more than {c['rack']['neighbor_gap_tolerance'] * 1000:g} mm apart; the rack "
            "is finished when both grippers are at least "
            f"{rack.WITHDRAW_DISTANCE * 100:g} cm from every spot. Do not touch the cue ball."
        )
    else:
        s = c["service"]
        timing = dict(
            preparation=s["preparation"],
            firing_interval=s["firing_interval"],
            demand_count=s["demand_count"],
            capacity=s["capacity"],
            deadlines=[
                s["preparation"] + i * s["firing_interval"] for i in range(s["demand_count"])
            ],
            horizon=s["horizon"],
        )
        max_time = float(s["horizon"])
        instruction = (
            "Stock the gated service hopper from the side baskets: grasp each ball and release "
            f"it into a free cell so that each of the {s['demand_count']} timed demands (every "
            f"{s['firing_interval']:g} s from {s['preparation']:g} s) finds a ball; a demand "
            "that finds none is served late by the next ball you stock. Never exceed "
            f"{s['capacity']} balls in the hopper."
        )
    return max_time, instruction, timing


class SportsTask:
    """Scoring view of one episode; the bound runtime adapter supplies measured state."""

    task_id = ""
    name = ""
    discipline = ""
    category = ""
    source = ""

    def __init__(self, spec: SportsEpisode, native: dict):
        self.spec = spec
        self.native_config = native
        self.adapter = None
        self.action_count = 0

    def bind(self, adapter):
        self.adapter = adapter

    @property
    def time(self):
        return 0.0 if self.adapter is None else float(self.adapter.time)

    @property
    def steps(self):
        return 0 if self.adapter is None else self.adapter.ticks

    @property
    def terminated(self):
        return False if self.adapter is None else bool(self.adapter.done)

    @property
    def joint_success(self):
        return False if self.adapter is None else bool(self.adapter.joint_success())

    @property
    def event_log(self):
        return [] if self.adapter is None else jsonable(self.adapter.events())

    def gen_instruction(self):
        return self.spec.instruction

    def metrics(self):
        if self.adapter is None:
            raise RuntimeError("Reset the environment before scoring")
        document = jsonable(self.adapter.result())
        document.update(task=self.task_id, difficulty=self.spec.difficulty, seed=self.spec.seed)
        row = dict(normalize(document))
        native_id = row.get("task_id")
        task = row.get("task_success", False)
        temporal = row.get("temporal_success", False)
        row.update(
            task_id=self.task_id,
            task_name=self.name,
            category=self.category,
            discipline=self.discipline,
            source_package=f"roboclock.envs.sports.{self.source}",
            difficulty=self.spec.difficulty,
            seed=self.spec.seed,
            task_success=bool(task),
            manipulation_success=bool(row.get("manipulation_success", task)),
            temporal_success=bool(temporal),
            joint_success=bool(row.get("joint_success")) and bool(task) and bool(temporal),
            number_of_actions=self.action_count,
            elapsed_time=self.time,
            physics_hz=round(1 / self.spec.physics_dt),
            control_hz=round(1 / (self.spec.physics_dt * self.spec.control_decimation)),
            # C-class verdicts (``throughput``) are already the row's own fields.
            native_result={
                k: v for k, v in document.items() if k not in ("events", "metrics", "throughput")
            },
        )
        if native_id is not None and str(native_id) != self.task_id:
            row["legacy_task_id"] = str(native_id)
        if not isinstance(row.get("temporal_violations"), list):
            row["temporal_violations"] = list(row.get("temporal_violations") or [])
        if row["temporal_success"]:
            row["temporal_violations"] = []
        return jsonable(row)


def _task_class(task_id, name, discipline, source):
    return type(
        f"Sports{task_id}Task",
        (SportsTask,),
        dict(
            task_id=task_id,
            name=name,
            discipline=discipline,
            source=source,
            category=CATEGORY[task_id[0]],
            __module__=__name__,
        ),
    )


TASKS = {task_id: _task_class(task_id, *rest) for task_id, *rest in CATALOG}


def make_task(task_id, difficulty="easy", seed=0, config=None):
    config = config or load_config()
    task_id = str(task_id)
    native = native_config(task_id, difficulty, seed, config)
    max_time, instruction, timing = _episode(task_id, difficulty, int(seed), native)
    dt, decimation = task_rates(config, task_id)
    spec = SportsEpisode(
        task_id,
        difficulty,
        int(seed),
        max_time,
        dt,
        decimation,
        f"roboclock.envs.sports.{SOURCES[task_id]}",
        instruction,
        timing,
        copy.deepcopy(native.get("spatial_randomization") or {}),
    )
    return TASKS[task_id](spec, native)


def spec_dict(spec):
    return asdict(spec)


def jsonable(value):
    """JSON-safe copy: numpy to Python, sets/tuples to lists, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return copy.deepcopy(str(value))
