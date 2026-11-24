"""C-class (throughput) jobs, time budgets and judging, score_version 2.

``docs/c-class-timing.md`` states the rule. This module holds its constants and arithmetic so
that the five domains judge their C tasks the same way:

- A C task is Q jobs. Each done job has a completion time; t_Q is the time of the Q-th.
- physical: all Q jobs done correctly by the end of the episode, still valid at the end, no
  physical failure, final misplacements within the accuracy allowance. The domain decides
  this (it owns the scene); time plays no part.
- temporal, per tier: batch tasks need t_Q <= T_tier = MULTIPLIER[tier] * t_o; stream tasks
  need every job by its own deadline. No physical terms.
- joint = physical and temporal.
- t_o is the slowest t_Q of the scripted reference oracle over seeds 0-9, frozen in ORACLE_TIMES
  with its source. It is a reference level, not an optimal time: hard (1.25 t_o) leaves it 20 %
  headroom, easy is 4 t_o and medium their geometric mean (sqrt(5) t_o).
- Every episode runs to H = HORIZON_FACTOR * T_easy, unless all Q jobs are done and settled,
  or the stall stop fires (from T_medium on, no progress for ``stall_window``).

Batch tasks run once per seed and are scored at all three tiers from t_Q (``*_by_tier`` row
fields); the requested difficulty only picks which tier fills ``temporal_success`` and
``joint_success``. Stream tasks keep one run per difficulty because their schedule changes:
their deadlines at a tier are the tier's factor x the task's frozen reference schedule (household C1:
``c1_x5/task.py`` HARD_PICKUPS_S; sports C3: ``ball_service/config.py`` HARD_PREPARATION and
HARD_INTERVAL), judged per job by ``judge_stream``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SCORE_VERSION = 2
TIERS = ("easy", "medium", "hard")
# T_tier = MULTIPLIER[tier] * t_o. Hard leaves the reference oracle 20 % headroom (t_o / 0.8);
# easy keeps its 4 x t_o; medium is their geometric mean, sqrt(4 * 1.25) = sqrt(5).
MULTIPLIER = {"easy": 4.0, "medium": math.sqrt(5.0), "hard": 1.25}
HORIZON_FACTOR = 1.25  # H = 1.25 * T_easy = 5 * t_o
STALL_FROM = "medium"  # the stall stop is armed from T_medium on ...
STALL_WINDOW = 0.5  # ... and fires after this many t_o without progress, or after
STALL_PACE = 1.5  # this many times the easy tier's time per progress step, if longer
SETTLE_S = 2.0  # all Q jobs done and valid for this long ends the episode
TIME_TOLERANCE_S = 0.02  # one control step at 50 Hz; job times are sampled on control steps


@dataclass(frozen=True)
class OracleTime:
    """Slowest recorded reference-oracle time to the Q-th job (per job when ``per_item``)."""

    seconds: float
    per_item: bool = False
    source: str = ""


_E1 = (
    "oracle matrix E1 (2026-11-16, seeds 0-9 easy, zero inference latency and control delay),"
    " slowest successful seed, rounded up (0.01 s; lab 0.001 s per item)"
)

# Frozen. Never derive these at run time: retuning an oracle must not move the benchmark.
# t_o is the reference oracle's slowest time, not an optimal time. Re-measure deliberately
# (scripts/throughput/oracle_times.py) and update this table and docs/c-class-timing.md together.
ORACLE_TIMES: dict[tuple[str, str], OracleTime] = {
    ("lab", "C1"): OracleTime(3.977, True, f"{_E1}: max t_Q/Q, seed 5"),
    ("lab", "C2"): OracleTime(2.807, True, f"{_E1}: max t_Q/Q, seed 1"),
    ("lab", "C3"): OracleTime(3.025, True, f"{_E1}: max t_Q/Q, seed 1"),
    ("hospital", "C1"): OracleTime(16.09, source=f"{_E1}: seed 7, all 8 orders"),
    ("hospital", "C2"): OracleTime(18.27, source=f"{_E1}: seed 2, all 6 items"),
    ("hospital", "C3"): OracleTime(28.44, source=f"{_E1}: seed 1, 12 vials"),
    ("hospital", "C4"): OracleTime(26.34, source=f"{_E1}: seed 8, 12 tubes"),
    ("household", "C2"): OracleTime(43.21, source=f"{_E1}: seed 9, finish_time"),
    ("household", "C3"): OracleTime(25.97, source=f"{_E1}: seed 7, 6 utensils"),
    ("sports", "C1"): OracleTime(127.49, source=f"{_E1}: seed 4, 15th ball"),
    # Measured after the 2.0 s preparation.
    ("sports", "C2"): OracleTime(31.41, source=f"{_E1}: seed 9, 2nd card minus 2.0 s preparation"),
    ("industry", "C1"): OracleTime(64.50, source=f"{_E1}: seed 4, 2nd kit"),
    ("industry", "C2"): OracleTime(45.32, source=f"{_E1}: seed 4, 3rd carton"),
    # Re-measured 2026-11-19 when C3 became a weighing task (was 83.20, E1 seed 4).
    ("industry", "C3"): OracleTime(69.36, source=(
        "C3 weighing re-measure (2026-11-19, seeds 0-9 easy, zero inference latency and control"
        " delay), slowest successful seed, rounded up (0.01 s): seed 4, 3rd part")),
    ("industry", "C4"): OracleTime(81.46, source=f"{_E1}: seed 4, 2nd palette"),
}

# Jobs arrive or are demanded on a schedule; one run per difficulty, judged per job deadline.
STREAM_TASKS = frozenset({("household", "C1"), ("sports", "C3")})


def is_batch(environment: str, task_id: str) -> bool:
    return (environment, task_id) in ORACLE_TIMES


def is_stream(environment: str, task_id: str) -> bool:
    return (environment, task_id) in STREAM_TASKS


def oracle_time(environment: str, task_id: str, q: int | None = None) -> float:
    """t_o of a batch task in seconds; per-item entries need the episode's Q."""
    entry = ORACLE_TIMES[(environment, task_id)]
    if entry.per_item:
        if not q or q < 1:
            raise ValueError(f"{environment}/{task_id}: t_o is per item, so Q is needed")
        return round(entry.seconds * q, 6)
    return entry.seconds


def budgets(t_o: float) -> dict[str, float]:
    """T per tier for an oracle time t_o."""
    return {tier: round(MULTIPLIER[tier] * t_o, 6) for tier in TIERS}


def horizon(t_o: float) -> float:
    """Episode horizon H = HORIZON_FACTOR * T_easy."""
    return round(HORIZON_FACTOR * MULTIPLIER["easy"] * t_o, 6)


def stall_window(t_o: float, steps: int | None = None) -> float:
    """Seconds without progress that end an armed episode.

    STALL_WINDOW * t_o, or STALL_PACE x the easy tier's time per progress step
    (MULTIPLIER["easy"] * t_o / steps) when that is longer: with few steps (sports C2's two
    cards, industry's 2-3 jobs) half of t_o would stop a policy working at easy-tier pace.
    """
    window = STALL_WINDOW * t_o
    if steps:
        window = max(window, STALL_PACE * MULTIPLIER["easy"] * t_o / steps)
    return window


def stalled(
    now: float,
    last_job_time: float | None,
    t_o: float,
    start: float = 0.0,
    steps: int | None = None,
) -> bool:
    """True when the stall stop should end the episode at time `now`.

    `start` is when the clock for the budget starts (a preparation phase, if any). The stop is
    armed from start + T_medium and fires after ``stall_window(t_o, steps)`` without progress.
    `last_job_time` is the latest progress: a done job, or a credited sub-step where a job is
    made of several (an industry kit's components); `steps` counts those progress steps in
    the whole task (Q, or the number of sub-steps).
    """
    elapsed = now - start
    if elapsed < MULTIPLIER[STALL_FROM] * t_o:
        return False
    last = start if last_job_time is None else max(start, last_job_time)
    return now - last >= stall_window(t_o, steps)


def q_th_time(job_times, q: int) -> float | None:
    """Time of the Q-th done job, or None when fewer than Q are done."""
    done = sorted(t for t in job_times if t is not None)
    return done[q - 1] if q >= 1 and len(done) >= q else None


_FROM_JOBS = object()


def judge(
    job_times,
    q: int,
    physical: bool,
    t_o: float,
    difficulty: str,
    *,
    start: float = 0.0,
    t_q=_FROM_JOBS,
    misplaced: int | None = None,
) -> dict:
    """Row fields of a batch C task.

    `job_times` lists each job's completion time (None if not done). `physical` is the domain's
    physical verdict (all Q done and valid at the end, no failure, misplacements allowed).
    `t_q` overrides the Q-th time when the domain tracks it better (for example the first time
    Q jobs were done at once); an explicit None means "not reached". Left out, it is the Q-th
    smallest job time. Times are absolute episode times; budgets count from `start`.
    """
    if difficulty not in TIERS:
        raise ValueError(f"difficulty must be one of {TIERS}, got {difficulty!r}")
    job_times = [None if t is None else float(t) for t in job_times]
    if t_q is _FROM_JOBS:
        t_q = q_th_time(job_times, q)
    limits = budgets(t_o)
    temporal = {
        tier: t_q is not None and t_q - start <= limits[tier] + TIME_TOLERANCE_S for tier in TIERS
    }
    physical = bool(physical)
    joint = {tier: physical and temporal[tier] for tier in TIERS}
    return {
        "score_version": SCORE_VERSION,
        "job_times_s": job_times,
        "jobs_done": sum(t is not None for t in job_times),
        "jobs_required": q,
        "t_q_s": t_q,
        "completion_time": t_q,
        "oracle_time_s": t_o,
        "time_origin_s": start,
        "budget_s_by_tier": limits,
        "horizon_s": start + horizon(t_o),
        "temporal_success_by_tier": temporal,
        "joint_success_by_tier": joint,
        "task_success": physical,
        "manipulation_success": physical,
        "temporal_success": temporal[difficulty],
        "joint_success": joint[difficulty],
        **({"misplaced": misplaced} if misplaced is not None else {}),
    }


def judge_stream(job_times, deadlines, physical: bool) -> dict:
    """Row fields of a stream C task: every job by its own deadline (one run per difficulty)."""
    job_times = [None if t is None else float(t) for t in job_times]
    deadlines = [float(d) for d in deadlines]
    if len(job_times) != len(deadlines):
        raise ValueError("one deadline per job")
    late = [
        i
        for i, (t, d) in enumerate(zip(job_times, deadlines))
        if t is None or t > d + TIME_TOLERANCE_S
    ]
    physical = bool(physical)
    temporal = not late
    return {
        "score_version": SCORE_VERSION,
        "job_times_s": job_times,
        "job_deadlines_s": deadlines,
        "late_jobs": late,
        "jobs_done": sum(t is not None for t in job_times),
        "jobs_required": len(job_times),
        "t_q_s": q_th_time(job_times, len(job_times)),
        "task_success": physical,
        "manipulation_success": physical,
        "temporal_success": temporal,
        "joint_success": physical and temporal,
    }


def allowed_misplacements(required_accuracy: float | None, q: int) -> int:
    """Final misplacements physical success tolerates: floor((1 - accuracy) * Q)."""
    if not required_accuracy:
        return 0
    return math.floor((1.0 - float(required_accuracy)) * q + 1e-9)
