"""Timing-only interventions for the RoboClock ablations (docs/ablations.md).

Experiment 1 crosses a task's *temporal* difficulty with a fixed inference budget L. Both axes
have to move without anything physical moving with them, so this module owns three things:

- ``EXPERIMENT1``: for each task, the three temporal tiers and how each one is produced. A tier
  is either a whitelisted patch of the domain's configuration (the level stays fixed, so the
  position and orientation tolerances, the window width and every seeded draw stay fixed) or the
  domain's own level, where the domain guarantees that only the temporal requirement changes.
- ``write_config``: the domain's default configuration with a tier's patch applied. ``--config``
  reads a whole domain configuration, so a tier is a generated copy of the default, never an
  edit of the benchmark's own file. Every patched path is checked against ``WHITELIST``.
- ``instance_hash``: the episode's non-temporal fields, hashed. Two tiers of one task at one seed
  must agree on it; the same integer seed is not on its own evidence of the same instance.

Nothing here touches a scene or needs a GPU: the tiers and their hashes are checked on CPU
before any Isaac time is spent.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import yaml

from roboclock.envs.registry import get_environment

# The control interface's own delay, charged on every model query on top of the inference
# budget L (docs/ablations.md, "Latency and input permissions"). L = 0 still costs delta.
INTERFACE_DELAY_S = 0.02
# The inference budgets L swept in experiment 1, in seconds.
BUDGETS_S = (0.0, 0.1, 0.3, 1.0)

# Configuration paths an ablation may patch, per environment. Anything else is refused: a
# timing-only tier must not reach a physical parameter, a layout range or a randomization flag.
WHITELIST = {
    "lab": (
        "bath_tasks.A5.tolerance_s",
        "bath_tasks.A6.tolerance_s",
        "reactive_tasks.B2.easy.reaction_latency",
        "reactive_tasks.B2.medium.reaction_latency",
        "reactive_tasks.B2.hard.reaction_latency",
    ),
    "industry": (
        "tasks.B1.max_time",
        "tasks.B2.max_time",
        "tasks.B3.max_time",
    ),
    "sports": (),
}

# Fields of a resolved episode that a temporal tier is allowed to change. They are left out of
# the instance hash; everything else must match across the tiers of one task and seed.
TEMPORAL_FIELDS = frozenset(
    {
        # lab EpisodeSpec
        "difficulty",
        "max_latency",
        "valid_start",
        "valid_end",
        "transfer_deadline",
        # industry resolved spec
        "temporal",
        "response_deadline_s",
        "window_width_s",
        "window_s",
        "window_close_s",
        "window_relative_s",
        "spec_hash",
        "sha256",
        # the episode length, unified across tiers by a whitelisted patch where it differs
        "max_time",
        "budget_s",
        # sports C3: the tier's demand schedule is the intervention itself
        "preparation",
        "firing_interval",
        "horizon",
        "deadlines",
        # What the model is told about the requirement. It has to track the tier, so it is not
        # part of the instance; ``statement_differs`` checks that it does track it.
        "instruction",
        "time_requirement",
    }
)

# The resolved fields that state the temporal requirement to the model.
STATEMENT_FIELDS = ("instruction", "time_requirement", "requirement")


class Tier:
    """One temporal tier of one task: the level to run at, and the configuration patch."""

    __slots__ = ("label", "difficulty", "patch", "note")

    def __init__(self, label, difficulty, patch=None, note=""):
        self.label = label
        self.difficulty = difficulty
        self.patch = dict(patch or {})
        self.note = note

    def __repr__(self):
        return f"Tier({self.label!r}, {self.difficulty!r}, {self.patch!r})"


class Condition:
    """One task of the matrix: its environment, its tiers and how it is scored."""

    __slots__ = ("environment", "task_id", "tiers", "patch", "scoring", "note")

    def __init__(self, environment, task_id, tiers, patch=None, scoring="episode", note=""):
        self.environment = environment
        self.task_id = task_id
        self.tiers = tiers
        # Applied to every tier of this task (e.g. the unified episode length).
        self.patch = dict(patch or {})
        # "episode": one run per tier. "by_tier": one run, judged at the three tiers from the
        # same trajectory (a C-class threshold sensitivity, docs/c-class-timing.md).
        self.scoring = scoring
        self.note = note

    @property
    def key(self):
        return f"{self.environment}/{self.task_id}"

    def tier_patch(self, tier):
        merged = dict(self.patch)
        merged.update(tier.patch)
        return merged


# A5/A6: the level's tolerance values, applied as an explicit override so the level - and with
# it every physical draw of the seed - stays at medium.
_BATH_TOLERANCES = (("t1", 4.0), ("t2", 2.0), ("t3", 1.0))  # the 2026-11-17 easy/medium/hard values
# lab B2: the level's reaction latencies, applied the same way. sample_episode draws
# uniform(lo, hi) whatever the bounds are, so a degenerate range keeps the RNG stream.
_LAB_B2_LATENCIES = (("t1", 4.0), ("t2", 2.74), ("t3", 1.88))  # the 2026-11-16 easy/medium/hard values

EXPERIMENT1 = {
    condition.key: condition
    for condition in (
        Condition(
            "lab",
            "A5",
            [
                Tier(label, "medium", {"bath_tasks.A5.tolerance_s": value})
                for label, value in _BATH_TOLERANCES
            ],
            note="12 s soak, late-only tolerance; the level stays medium",
        ),
        Condition(
            "lab",
            "A6",
            [
                Tier(label, "medium", {"bath_tasks.A6.tolerance_s": value})
                for label, value in _BATH_TOLERANCES
            ],
            note="30 s soak, early-or-late tolerance; the level stays medium",
        ),
        Condition(
            "lab",
            "B2",
            [
                Tier(label, "medium", {"reactive_tasks.B2.medium.reaction_latency": [v, v]})
                for label, v in _LAB_B2_LATENCIES
            ],
            note="reaction deadline; the level stays medium, so the window width and the "
            "position/orientation tolerances do not move",
        ),
        Condition(
            "industry",
            "B2",
            [Tier("t1", "easy"), Tier("t2", "medium"), Tier("t3", "hard")],
            # The levels differ in response_deadline_s (3/2.75/2.53 s since 2026-11-16); the
            # episode budget (30 s at every level) is pinned here, so a tighter tier does not
            # also end earlier (docs/ablations.md).
            patch={"tasks.B2.max_time": 30.0},
            note="industry levels change only the temporal requirement and the budget; the "
            "budget is unified here",
        ),
        Condition(
            "sports",
            "C3",
            [Tier("t1", "easy"), Tier("t2", "medium"), Tier("t3", "hard")],
            note="stream schedule intervention: the tier changes the demand schedule and the "
            "horizon, so each tier is its own closed-loop run",
        ),
        Condition(
            "lab",
            "C2",
            [Tier("t1", "easy"), Tier("t2", "medium"), Tier("t3", "hard")],
            scoring="by_tier",
            note="batch throughput: one run per budget L, judged at all three tiers from "
            "joint_success_by_tier",
        ),
    )
}


def conditions(keys=None):
    """The matrix's conditions, in order; ``keys`` selects a subset by "env/TASK"."""
    if keys is None:
        return list(EXPERIMENT1.values())
    missing = [key for key in keys if key not in EXPERIMENT1]
    if missing:
        raise ValueError(f"Unknown condition(s): {', '.join(missing)}")
    return [EXPERIMENT1[key] for key in keys]


def runs(keys=None, budgets=BUDGETS_S):
    """Every (condition, tier, L) that has to be run closed-loop.

    A "by_tier" condition runs once per L and is judged at all of its tiers, so it yields one
    row per L with ``tier`` None.
    """
    out = []
    for condition in conditions(keys):
        for budget in budgets:
            if condition.scoring == "by_tier":
                out.append((condition, None, float(budget)))
            else:
                out.extend((condition, tier, float(budget)) for tier in condition.tiers)
    return out


def default_config(environment):
    """The domain's default configuration, as written (not through its loader)."""
    path = Path(get_environment(environment).default_config)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def apply_patch(config, patch, environment):
    """``config`` with each whitelisted dotted path set. Returns a new dict."""
    allowed = WHITELIST.get(environment)
    if allowed is None:
        raise ValueError(f"No ablation whitelist for environment {environment!r}")
    result = copy.deepcopy(config)
    for path, value in patch.items():
        if path not in allowed:
            raise ValueError(
                f"{environment}: {path!r} is not an ablation parameter "
                f"(allowed: {', '.join(allowed) or 'none'})"
            )
        node = result
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"{environment}: {path!r} passes through a non-mapping")
        # A whitelisted leaf may be absent from the default configuration: an override the
        # benchmark never sets (bath_tasks.*.tolerance_s) exists only in a generated tier.
        # WHITELIST, not the default file, is what catches a mistyped path.
        node[leaf] = value
    return result


def write_config(environment, patch, path):
    """Write the domain's default configuration with ``patch`` applied, and return ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = apply_patch(default_config(environment), patch, environment)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def resolved_spec(environment, task_id, difficulty, seed, config=None):
    """The episode's resolved spec, built without a scene (CPU only).

    ``config`` is a loaded domain configuration, or None for the domain's default.
    """
    from roboclock.tasks.registry import make_task

    task = make_task(task_id, difficulty, seed, config, environment=environment)
    spec = task.spec
    if is_dataclass(spec):
        return asdict(spec)
    if isinstance(spec, dict):
        return dict(spec)
    # industry: the episode dataclass holds the resolved spec and the budget.
    return {
        **{k: getattr(spec, k) for k in getattr(spec, "__dataclass_fields__", ())},
        **(getattr(spec, "resolved", None) or {}),
    }


def _plain(value):
    """``value`` as something json.dumps can write, for hashing."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if is_dataclass(value):
        return _plain(asdict(value))
    return repr(value)


def strip_temporal(spec):
    """``spec`` without the fields a temporal tier may change, at every depth."""
    if isinstance(spec, dict):
        return {
            key: strip_temporal(value)
            for key, value in spec.items()
            if key not in TEMPORAL_FIELDS
        }
    if isinstance(spec, (list, tuple)):
        return [strip_temporal(v) for v in spec]
    return spec


def instance_hash(environment, task_id, difficulty, seed, config=None):
    """SHA-256 of the episode's non-temporal resolved fields.

    Two tiers of one task at one seed must give the same digest; that is the paired-instance
    check, and it is what ``instance_id`` stands for in the results.
    """
    spec = strip_temporal(_plain(resolved_spec(environment, task_id, difficulty, seed, config)))
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def statement(environment, task_id, difficulty, seed, config=None):
    """What the resolved episode states about its temporal requirement, as one string.

    The instruction has to move with the tier: a tier that changed a deadline the model is
    never told about would measure a different thing than the paper claims. The tasks whose
    requirement is only in the generated instruction (lab) answer with the task's instruction.
    """
    from roboclock.tasks.registry import make_task

    spec = resolved_spec(environment, task_id, difficulty, seed, config)
    parts = [str(spec[key]) for key in STATEMENT_FIELDS if spec.get(key)]
    if not parts:
        task = make_task(task_id, difficulty, seed, config, environment=environment)
        parts = [task.gen_instruction()]
    return "\n".join(parts)


def instance_id(environment, task_id, seed):
    """The identifier a task's paired instances share across tiers and budgets."""
    return f"{environment}/{task_id}/s{seed}"
