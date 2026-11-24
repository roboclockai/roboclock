"""Dual-X5 household task lifecycle, measured-signal input and shared scoring schema.

A physics backend calls, every control step:

    task.update(signals)   # measured physical facts, keys declared in task.SIGNALS
    task.advance(dt)       # process evolution over the elapsed simulation time

``cues()`` returns the process values that drive visible appearance (tea colour,
scale reading, flame size, ...). Hidden parameters live only in ``spec`` and
``metrics()``; they must not be exposed to an evaluated agent.

All times are simulation seconds from the first agent observation (t = 0). The world
keeps evolving while the agent deliberates; nothing here pauses the process.
"""

from dataclasses import asdict, dataclass, fields

import numpy as np

DIFFICULTY_TIME_SCALE = {"easy": 0.75, "medium": 1.0, "hard": 1.5}
"""Process-rate multiplier per difficulty. ``medium`` is the nominal (alpha = 1) contract."""


LAYOUT_STREAM = 5407
"""Second seed word of the spatial-layout rng: a stream of its own, so the hidden-parameter draws stay as they were."""


def sample_layout(task_id, seed, jitter):
    """Per-seed rigid placement offsets: ((name, dx m, dy m, dyaw deg), ...), uniform per component in jitter[name]
    = (x, y, yaw), each a half-range h (-h .. h) or a (low, high) pair (a reach-limited, one-sided range).

    Drawn from a dedicated stream ([seed, LAYOUT_STREAM, *ord(task_id)]: not sum(ord), which A4, B3 and C2 share),
    independent of the difficulty, so a seed shows the same scene at every tier. Seed 0 is the nominal layout (all
    zeros), as in C1/C3."""
    if not seed:
        return tuple((name, 0.0, 0.0, 0.0) for name in jitter)
    rng = np.random.default_rng([int(seed), LAYOUT_STREAM, *map(ord, task_id)])
    out = []
    for name, ranges in jitter.items():
        u = rng.uniform(0.0, 1.0, 3)
        lo_hi = [(-r, r) if np.isscalar(r) else r for r in ranges]
        out.append((name, *(round(float(lo+(hi-lo)*k), 6) for (lo, hi), k in zip(lo_hi, u))))
    return tuple(out)


def place(points, offset, pivot):
    """Apply a layout offset (dx, dy, dyaw deg) to world points (..., 2 or 3): turn dyaw about the vertical through
    pivot (x, y), then translate by (dx, dy). z is unchanged."""
    p = np.array(points, float)
    dx, dy, yaw = offset
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    x, y = p[..., 0]-pivot[0], p[..., 1]-pivot[1]
    p[..., 0], p[..., 1] = pivot[0]+c*x-s*y+dx, pivot[1]+s*x+c*y+dy
    return p


@dataclass(frozen=True)
class X5Spec:
    task_id: str = ""
    difficulty: str = "medium"
    seed: int = 0
    max_time: float = 60.0
    time_scale: float = 1.0
    # Spatial randomization: ((name, dx m, dy m, dyaw deg), ...) from sample_layout() with the class's LAYOUT_JITTER.
    # Applied by the task's sim Oracle.setup() before the world resets (every policy, not only the oracle).
    layout: tuple = ()

    LAYOUT_JITTER = {}
    """{name: (x m, y m, yaw deg)} half-ranges of the per-seed placement offsets; {} = no spatial randomization here."""

    @classmethod
    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        """Nominal values for seed 0 / hidden values drawn from ``hidden_ranges`` otherwise."""
        if difficulty not in DIFFICULTY_TIME_SCALE:
            raise ValueError(f"Unknown difficulty {difficulty!r}")
        values = dict(task_id=task_id, difficulty=difficulty, seed=int(seed),
                      time_scale=DIFFICULTY_TIME_SCALE[difficulty],
                      layout=sample_layout(task_id, seed, cls.LAYOUT_JITTER))
        if seed:
            rng = np.random.default_rng([int(seed), sum(map(ord, task_id))])
            for name, (lo, hi) in cls.hidden_ranges().items():
                values[name] = float(rng.uniform(lo, hi))
        values.update(overrides)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    @classmethod
    def hidden_ranges(cls) -> dict:
        return {}

    def offset(self, name):
        """(dx m, dy m, dyaw deg) of a layout entry; zeros when the layout does not name it."""
        for entry in self.layout:
            if entry[0] == name:
                return tuple(float(v) for v in entry[1:4])
        return (0.0, 0.0, 0.0)


def band_quality(value, low, high, zero_low, zero_high):
    """1 inside [low, high], linear to 0 at zero_low / zero_high."""
    if value is None:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return float(np.clip((value - zero_low) / (low - zero_low), 0.0, 1.0))
    return float(np.clip((zero_high - value) / (zero_high - high), 0.0, 1.0))


def gripped(jaw_forces, threshold):
    """True when both jaws of one gripper press on an object harder than `threshold` (N).

    ``jaw_forces`` are the contact-force norms on the object, one (jawP, jawN) pair per arm. A task
    is judged the same whichever arm does the work, so a judged "held" signal reads both arms'
    pairs through this. One jaw of each arm touching is not a grip.
    """
    pairs = np.asarray(jaw_forces, dtype=float).reshape(-1, 2)
    return bool((pairs.min(axis=1) > threshold).any())


class Hold:
    """Seconds a boolean condition has held continuously."""

    def __init__(self):
        self.seconds = 0.0

    def step(self, condition, dt):
        self.seconds = self.seconds + dt if condition else 0.0
        return self.seconds


class X5Task:
    """Base class: subclasses define SIGNALS, Spec, process() and scoring properties."""

    task_id = name = category = instruction = ""
    discipline = "household"
    Spec = X5Spec
    SIGNALS: dict = {}

    def __init__(self, spec):
        self.spec = spec
        self.time = 0.0
        self.steps = 0
        self.signals = {k: v[1] for k, v in self.SIGNALS.items()}
        self.event_log = []
        self.violations = []
        self.physical_failures = []
        self.events = {}
        self.done = False
        self.end_reason = None  # C-class tasks: why the episode ended early (see end())

    # ----------------------------------------------------------------- inputs
    def update(self, signals: dict):
        unknown = set(signals) - set(self.SIGNALS)
        if unknown:
            raise KeyError(f"{self.task_id}: undeclared signals {sorted(unknown)}")
        self.signals.update(signals)

    def advance(self, dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        # Plain summation, as in the recorder (scripts/household/run_x5_episode.py): event times
        # such as C1's 300.004 s pickup depend on it. It can end a hair short of a whole-second
        # horizon (75000 x 1/250 s = 299.99999999998 s), so the environment runs until
        # time >= max_time.
        self.time += dt
        self.steps += 1
        self.process(dt)

    def process(self, dt):
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    def mark(self, kind, **data):
        """Record the first occurrence of a named event."""
        if kind not in self.events:
            self.events[kind] = self.time
            self.event_log.append(dict(time=self.time, type=kind, **data))
        return self.events[kind]

    def violation(self, kind):
        if kind not in self.violations:
            self.violations.append(kind)
            self.event_log.append(dict(time=self.time, type=kind))

    def failure(self, kind):
        if kind not in self.physical_failures:
            self.physical_failures.append(kind)
            self.event_log.append(dict(time=self.time, type=kind))

    def end(self, reason):
        """End the episode early (C-class tasks: all_done, stall or failure); the first reason stands."""
        if not self.done:
            self.done, self.end_reason = True, reason
            self.mark("episode_end", reason=reason)

    @property
    def termination_reason(self):
        """C-class row field: all_done, stall, failure or horizon (None while the episode runs)."""
        if self.end_reason:
            return self.end_reason
        return "horizon" if self.time >= self.spec.max_time - 1e-6 else None  # summed steps can end a hair short

    # ----------------------------------------------------------------- scoring
    def gen_instruction(self):
        return self.instruction

    def cues(self) -> dict:
        """Process values that drive rendered appearance (not privileged scores)."""
        return {}

    @property
    def physical_success(self) -> bool:
        raise NotImplementedError

    @property
    def temporal_success(self) -> bool:
        raise NotImplementedError

    @property
    def joint_success(self) -> bool:
        return self.physical_success and self.temporal_success

    @property
    def terminated(self) -> bool:
        return self.done

    def task_metrics(self) -> dict:
        return {}

    def metrics(self) -> dict:
        return _plain({
            "task_id": self.task_id,
            "task_name": self.name,
            "domain": "household",
            "embodiment": "dual_arx_x5",
            "discipline": self.discipline,
            "category": self.category,
            "difficulty": self.spec.difficulty,
            "seed": self.spec.seed,
            "task_success": self.physical_success,
            "manipulation_success": self.physical_success,
            "temporal_success": self.temporal_success,
            "joint_success": self.joint_success,
            "elapsed_time": self.time,
            "temporal_violations": list(self.violations),
            "physical_failures": list(self.physical_failures),
            # Same list the shared evaluator writes as row["events"] (task.event_log);
            # first-occurrence times by name are in event_times.
            "events": [dict(e) for e in self.event_log],
            "event_times": dict(self.events),
            "physics_steps": self.steps,
            **self.task_metrics(),
            "episode_spec": asdict(self.spec),
        })


def _plain(value):
    """JSON-ready copy: numpy scalars/arrays from measured signals become Python values."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
