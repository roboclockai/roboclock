"""Industry task catalog: class IDs (A1–C4) with the original 41–50 numbers as aliases.

Instructions come from ``configs/tasks.json``. The episode specification is the native
resolver's output.

- ``easy`` / ``medium`` / ``hard`` (``configs/difficulty.yaml``): the seed draws the
  layout, timing (event onsets) and physics, identically at every level; then, for an A or
  B task, the level sets the temporal requirement (window or reaction deadline) and the
  episode budget. The resolved spec records ``difficulty`` and ``temporal`` and is
  re-hashed. ``time_requirement`` states the level's numbers.
- C (throughput) tasks follow ``docs/c-class-timing.md``: the level changes nothing in the
  episode. Every C episode runs to the horizon H = 5 t_o of the task's frozen oracle time
  (``roboclock.evaluation.throughput``), unless it ends early (all jobs done and settled, or
  the stall stop), and is judged at all three tiers from its Q-th job time; the difficulty
  only picks the tier that fills ``temporal_success`` and ``joint_success``. Its resolved
  spec, hash included, is the same at every level, and ``time_requirement`` asks for the
  jobs "as quickly as possible" without a number of seconds.
- ``reviewed_nominal``: the recorded episodes. For the reviewed seed it is
  ``configs/task<NN>/reviewed.json`` (hash-checked); other seeds resolve the same nominal
  layout, timing and physics (only seed-derived quantities such as Task 49's QC results
  change). An A/B task keeps the approved time requirement and the recorded 90 s budget; a
  C task runs and is judged as at the levels, with its tiers' verdict reported at
  ``NOMINAL_TIER``.

Importing this module does not start Isaac Sim.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from roboclock.evaluation import throughput

CONFIGS = Path(__file__).parent / "configs"
_DESCRIPTIONS = {
    row["task_id"]: row
    for row in json.loads((CONFIGS / "tasks.json").read_text())["task_descriptions"]
}
CATEGORY = {"A": "temporal_window", "B": "event_reactive", "C": "throughput"}
# Every recorded episode ran the native runner's 90 s budget (videos.py --seconds 90);
# A/B episodes ended early once complete, throughput episodes always used all of it.
# This is the A/B reviewed_nominal budget; the levels set their own
# (configs/difficulty.yaml). C tasks run to their horizon instead (throughput_spec).
NATIVE_BUDGET_S = 90.0
# The tier whose verdict fills temporal_success / joint_success of a reviewed_nominal C
# episode (not a level, so it names no tier). Every tier is in the row either way.
NOMINAL_TIER = "medium"

# (class ID, original number, name, discipline)
CATALOG = (
    ("A1", 41, "plating_to_rinse", "surface_finishing"),
    ("A2", 42, "rotary_pump_loading", "assembly"),
    ("A3", 43, "enclosure_cover_bonding", "assembly"),
    ("B1", 44, "cooling_water_isolation", "utilities"),
    ("B2", 45, "motor_emergency_stop", "machine_safety"),
    ("B3", 46, "bearing_arrest_recovery", "material_handling"),
    ("C1", 47, "maintenance_kitting_operation", "kitting"),
    ("C2", 48, "carton_shelf_operation", "warehousing"),
    ("C3", 49, "weigh_and_sort", "quality_inspection"),
    ("C4", 50, "eyeshadow_assembly_output", "assembly"),
)
LEGACY_IDS = {str(number): task_id for task_id, number, *_ in CATALOG}
CATEGORY_OF = {task_id: task_id[0] for task_id, *_ in CATALOG}  # "A", "B" or "C"
LEVELS = ("easy", "medium", "hard")
LEVEL_TABLE = yaml.safe_load((CONFIGS / "difficulty.yaml").read_text())
DIFFICULTIES = {
    "reviewed_nominal": dict(layout=False, timing=False, physical=False),
    **{level: dict(LEVEL_TABLE["randomization"]) for level in LEVELS},
}


def _window_width(spec, value):
    spec["process_max_s"] = spec["process_min_s"] + value


def _sector_window(spec, value):
    # The table keeps its speed; the sector is narrowed from its entry edge (the carrier
    # turns toward negative angles, entering at -60 deg and leaving at -120 deg).
    close = spec["loading_sector_deg"][0]
    spec["loading_sector_deg"] = [close, close - spec["rotation_deg_s"] * value]
    spec["window_s"] = value


def _window_close(spec, value):
    spec["window_relative_s"] = [spec["window_relative_s"][0], value]


def _deadline(spec, value):
    spec["response_deadline_s"] = value


# Per task: the level parameter, how it sets the resolved spec, and the requirement text.
TEMPORAL = {
    "A1": ("window_width_s", _window_width,
           "For the adjustable demonstration recipe, complete whole-part exit from the zincate "
           "bath {process_min_s:.1f}–{process_max_s:.1f} s after full immersion. This is a "
           "demonstration recipe on the time scale of a zincate pre-treatment dip on aluminium, "
           "not calibrated chemistry."),
    "A2": ("window_s", _sector_window,
           "Complete supported placement while the carrier is inside the {width:g}° loading "
           "sector at {speed:g}°/s, {article} {window_s:g} s window."),
    "A3": ("window_close_s", _window_close,
           "Complete released, stable seating {opens:g}–{closes:g} s after hot-melt adhesive "
           "application finishes."),
    "B1": ("response_deadline_s", _deadline,
           "Fully close the valve within {response_deadline_s:g} s of leak onset. The current "
           "demonstration's tray limit is {estimated_spill_limit_l:.1f} L."),
    "B2": ("response_deadline_s", _deadline,
           "Actuate the stop button within {response_deadline_s:g} s of the warning."),
    "B3": ("response_deadline_s", _deadline,
           "Arrest within {response_deadline_s:g} s of measured safety-zone entry. Recovery "
           "follows the arrest and must finish with released tray support."),
}

# C tasks (docs/c-class-timing.md): the scene holds exactly the Q jobs; no seconds are stated,
# the tiers' budgets come from the frozen oracle time.
THROUGHPUT_REQUIREMENT = {
    "C1": "Complete both fully packed maintenance kits as quickly as possible. No mobile-base "
          "travel or delivery is required.",
    "C2": "Correctly shelve all 3 cartons as quickly as possible at the fixed pallet-and-shelf "
          "workcell. No mobile-base travel or delivery is required.",
    "C3": "Weigh each of the 3 parts and sort every part correctly, as quickly as possible.",
    "C4": "Assemble and deliver both complete four-color palettes as quickly as possible.",
}


# What an arm may touch. The contact callbacks (``native/arx_cell_task.ContactBook``; A1 and A2
# have their own in ``native/plating_task.py`` and ``native/rotary_task.py``) record every arm
# contact except a finger on the product. ``scoring.contact_voids`` decides which of them void
# physical success: an arm link other than a finger pressing with more than
# ``scoring.ARM_COLLISION_N``. The fingers may touch parts and fixtures. A policy cannot read
# this off the images, so its instruction states it.
CONTACT_RULE = "Arm links other than fingers: no contact over {force:g} N."
# A2 and A3 also judge the grasp and the release by numbers a policy cannot see:
# - the grasp counts once both fingers press with more than the monitor's ``grasp_force_n`` for
#   ``GraspEvents.DWELL_S`` (``scoring.RotaryMonitor``, ``scoring.BondingMonitor``); without it
#   the lift off the source and, with that, the placement are never credited;
# - "released" needs each finger slider of the arm that held the product open past
#   ``RELEASED_JAW_M`` of its 44 mm (``native/rotary_task.py``, ``native/bonding_task.py``).
# A light pinch or a gripper opened just enough to let go does not count, so the instruction
# gives the numbers.
GRIP_RULE = (
    "The grasp counts with over {force:g} N on both fingers for {dwell:g} s, the release once "
    "each finger is open past {jaw_mm:g} mm."
)
GRIP_TASKS = ("A2", "A3")
# Rules of single tasks that their text leaves open and the views do not show.
# - B1: which way the quarter-turn handle closes. It cannot be tried before the leak (moving
#   the valve early voids the timing) and a wrong first swing uses up the deadline.
TASK_RULES = {
    "B1": "The handle closes counterclockwise seen from above: a quarter turn, to its stop.",
}
# Every C episode can end before its horizon (``scoring.ThroughputMonitor.credit_complete``,
# ``evaluation.throughput.stalled``): a policy that is not told works on while nothing counts.
STALL_RULE = (
    "From {armed:.0f} s on, the episode ends once {window:.0f} s pass without a newly credited "
    "step."
)


def contact_rule():
    """The contact rule as every instruction states it."""
    from roboclock.envs.industry.scoring import ARM_COLLISION_N

    return CONTACT_RULE.format(force=ARM_COLLISION_N)


def stall_rule(task_id, resolved):
    """A C task's stall stop in seconds, as its instruction states it."""
    from roboclock.envs.industry.scoring import progress_steps

    t_o = oracle_time(task_id)
    return STALL_RULE.format(
        armed=throughput.MULTIPLIER[throughput.STALL_FROM] * t_o,
        window=throughput.stall_window(t_o, progress_steps(task_id, resolved)),
    )


def task_rule(task_id):
    """The task's own unstated rule (``TASK_RULES``), or ''."""
    return TASK_RULES.get(task_id, "")


def judged_rules(task_id, resolved=None):
    """Rules every level of a task is judged by that its approved text does not state: the
    task's own (``TASK_RULES``), a C task's stall stop, what the arms may touch and, for A2
    and A3, how firmly to grasp and how far to open."""
    rules = [task_rule(task_id)]
    if CATEGORY_OF[task_id] == "C" and resolved is not None:
        rules.append(stall_rule(task_id, resolved))
    rules.append(contact_rule())
    if task_id in GRIP_TASKS:
        from roboclock.envs.industry.native.arx_cell_task import RELEASED_JAW_M
        from roboclock.envs.industry.scoring import MONITORS, GraspEvents

        monitor = MONITORS[TASKS[task_id].legacy_task_id]
        rules.append(
            GRIP_RULE.format(
                force=monitor.grasp_force_n, dwell=GraspEvents.DWELL_S, jaw_mm=RELEASED_JAW_M * 1000
            )
        )
    return " ".join(rule for rule in rules if rule)


def level_spec(task_id, difficulty, resolved):
    """An A/B task's resolved spec with the level's temporal requirement applied and
    re-hashed, the level's episode budget, and the requirement text."""
    from roboclock.envs.industry.native.catalog import seal

    key, apply, text = TEMPORAL[task_id]
    level = LEVEL_TABLE["tasks"][task_id][difficulty]
    spec = copy.deepcopy(resolved)
    apply(spec, float(level[key]))
    spec["difficulty"] = difficulty
    spec["temporal"] = {key: float(level[key])}
    seal(spec)
    budget = float(level["max_time"])
    window = spec.get("window_relative_s", [None, None])
    sector = spec.get("loading_sector_deg", [0.0, 0.0])
    # "an 8 s", "an 11 s", "an 18 s"
    article = "an" if f"{spec.get('window_s', 0):g}".startswith(("8", "11", "18")) else "a"
    requirement = text.format(
        **spec, speed=-spec.get("rotation_deg_s", 0.0), width=sector[1] - sector[0],
        opens=window[0], closes=window[1], article=article,
    )
    return spec, budget, requirement


def oracle_time(task_id):
    """A C task's frozen oracle time t_o (roboclock.evaluation.throughput.ORACLE_TIMES)."""
    return throughput.oracle_time("industry", task_id)


def throughput_spec(task_id, resolved):
    """A C task's resolved spec at a level, and its episode length H.

    The same at every level (docs/c-class-timing.md: one run, three tiers): no ``difficulty``
    key, and ``temporal`` records the frozen oracle time, the tiers' budgets and H. The native
    controllers (the oracle) give up after ``budget_s``; it is H, the whole episode.
    """
    from roboclock.envs.industry.native.catalog import seal

    t_o = oracle_time(task_id)
    horizon = throughput.horizon(t_o)
    spec = copy.deepcopy(resolved)
    spec["budget_s"] = horizon
    spec["temporal"] = {
        "oracle_time_s": t_o, "budget_s_by_tier": throughput.budgets(t_o), "horizon_s": horizon,
    }
    seal(spec)
    return spec, horizon


@dataclass(frozen=True)
class IndustryEpisode:
    task_id: str
    difficulty: str
    seed: int
    legacy_task_id: int
    max_time: float
    instruction: str
    time_requirement: str
    resolved: dict = field(default_factory=dict)


class IndustryTask:
    """Scoring state for one episode; the environment's monitor reports measured outcomes."""

    def __init__(self, spec: IndustryEpisode):
        self.spec = spec
        self.time = 0.0
        self.steps = 0
        self.action_count = 0
        self.terminated = False
        self.event_log = []
        self.outcome = {}

    def gen_instruction(self):
        """What the policy is told: the task, its time requirement at this level and the
        judged rules its text does not state.

        The requirement (a window, a reaction deadline, or "as quickly as possible") is part of
        the task: a policy that is not told it cannot meet it. The other domains word it inside
        their instruction; here it is a separate field of the spec, so it is appended. The
        contact rule (``CONTACT_RULE``) is judged in every task, the grip force and jaw opening in
        A2 and A3 (``judged_rules``); they are appended for the same reason.
        """
        return (
            f"{self.spec.instruction} {self.spec.time_requirement} "
            f"{judged_rules(self.task_id, self.spec.resolved)}"
        ).strip()

    def record(self, outcome: dict):
        """Store the measured result (success flags and metrics)."""
        self.outcome = dict(outcome)

    @property
    def joint_success(self):
        return bool(self.metrics()["joint_success"])

    def metrics(self):
        physical = bool(self.outcome.get("task_success", False))
        temporal = bool(self.outcome.get("temporal_success", False))
        row = {
            **{k: v for k, v in self.outcome.items() if k not in ("task_id", "category")},
            "task_id": self.task_id,
            "legacy_task_id": self.spec.legacy_task_id,
            "task_name": self.name,
            "category": self.category,
            "discipline": self.discipline,
            "difficulty": self.spec.difficulty,
            "seed": self.spec.seed,
            "task_success": physical,
            "manipulation_success": bool(self.outcome.get("manipulation_success", physical)),
            "temporal_success": temporal,
            "joint_success": bool(self.outcome.get("joint_success", physical and temporal))
            and physical
            and temporal,
            "temporal_violations": list(self.outcome.get("temporal_violations", [])),
            # Why task_success is false (scoring.physical_failures); empty before any outcome.
            "physical_failures": list(self.outcome.get("physical_failures", [])),
            "elapsed_time": self.time,
            "number_of_actions": self.action_count,
        }
        if self.category == "throughput":
            row.update(self.throughput_fields())
        return row

    @property
    def tier(self):
        """The time tier that fills temporal_success / joint_success (C tasks)."""
        return self.spec.difficulty if self.spec.difficulty in throughput.TIERS else NOMINAL_TIER

    def throughput_fields(self):
        """docs/c-class-timing.md: physical from the monitor (all Q jobs done and valid at the
        end, no arm collision; no time term), temporal per tier from t_Q, where t_Q is the
        monitor's completion time (the tick of the Q-th job credit)."""
        from roboclock.envs.industry.scoring import MONITORS

        q = MONITORS[self.legacy_task_id].required_operations
        t_o = oracle_time(self.task_id)
        jobs = self.outcome.get("job_times_s") or [None] * q
        fields = throughput.judge(
            jobs, q, bool(self.outcome.get("task_success", False)), t_o, self.tier,
            t_q=self.outcome.get("completion_time"),
        )
        violations = list(self.outcome.get("temporal_violations", []))
        if fields["t_q_s"] is not None and not fields["temporal_success"]:
            violations.append(
                f"all {q} jobs done at {fields['t_q_s']:.2f} s, after the {self.tier} budget "
                f"{fields['budget_s_by_tier'][self.tier]:.2f} s"
            )
        return {
            **fields,
            "judged_tier": self.tier,
            "temporal_violations": violations,
            "termination_reason": self.outcome.get("termination_reason"),
        }


def _task_class(task_id, number, name, discipline):
    category = CATEGORY[_DESCRIPTIONS[number]["temporal_category"]]
    return type(
        f"Industry{task_id}",
        (IndustryTask,),
        {
            "task_id": task_id,
            "legacy_task_id": number,
            "name": name,
            "category": category,
            "discipline": discipline,
            # Episode length: A/B the recorded budget (reviewed_nominal), C the horizon H.
            "max_time": (
                throughput.horizon(oracle_time(task_id)) if category == "throughput" else NATIVE_BUDGET_S
            ),
        },
    )


TASKS = {row[0]: _task_class(*row) for row in CATALOG}


def make_task(task_id, difficulty="reviewed_nominal", seed=0, config=None):
    """One episode. ``config["tasks"][task]["max_time"]``, when set, overrides the episode
    length (short debugging runs). It does not touch the judging: a C task's tiers' budgets
    and stall stop still come from its frozen oracle time, and a cut episode ends with
    termination_reason "horizon"."""
    from roboclock.envs.industry.native.catalog import TASKS as NATIVE

    task_id = LEGACY_IDS.get(str(task_id), str(task_id))
    if task_id not in TASKS:
        raise ValueError(f"Unknown industry task {task_id!r}; choose one of {', '.join(TASKS)}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown industry difficulty {difficulty!r}; choose {', '.join(DIFFICULTIES)}")
    cls = TASKS[task_id]
    description = _DESCRIPTIONS[cls.legacy_task_id]
    flags = dict(DIFFICULTIES[difficulty])
    if difficulty in LEVELS:
        flags.update((LEVEL_TABLE["tasks"].get(task_id) or {}).get("randomization", {}))
    resolved = NATIVE[cls.legacy_task_id].resolved(seed, **flags)
    if cls.category == "throughput":
        # The reviewed replay keeps reviewed.json (hash-checked) and its 90 s controller limit,
        # which every recorded oracle finishes inside; its episode is the levels' one.
        if difficulty in LEVELS:
            resolved, _ = throughput_spec(task_id, resolved)
        budget, requirement = cls.max_time, THROUGHPUT_REQUIREMENT[task_id]
    elif difficulty == "reviewed_nominal":
        budget, requirement = cls.max_time, description["time_requirement"]
    else:
        resolved, budget, requirement = level_spec(task_id, difficulty, resolved)
    budget = (config or {}).get("tasks", {}).get(task_id, {}).get("max_time", budget)
    return cls(
        IndustryEpisode(
            task_id=task_id,
            difficulty=difficulty,
            seed=int(seed),
            legacy_task_id=cls.legacy_task_id,
            max_time=float(budget),
            instruction=description["instruction"],
            time_requirement=requirement,
            resolved=resolved,
        )
    )
