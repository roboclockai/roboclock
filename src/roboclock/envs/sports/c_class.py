"""Sports C-class jobs, horizons and judging (score_version 2, docs/c-class-timing.md).

C1 pool racking (15 balls) and C2 speed cups (2 cards) are batch tasks: one run per seed to
the horizon H = 5 t_o, scored at every time tier from the time of the Q-th job by
``roboclock.evaluation.throughput.judge``; the difficulty only picks which tier fills
``temporal_success``. C3 ball service is a stream task: one run per difficulty, every demand
judged against its own scheduled firing time by ``throughput.judge_stream``.

Each native task returns the judged fields under ``throughput`` in its result; the result
adapter (``results.normalize``) copies them into the row.
"""

from roboclock.evaluation import throughput

ENVIRONMENT = "sports"
TERMINATIONS = ("all_done", "stall", "failure", "horizon")
SETTLE_S = throughput.SETTLE_S


def oracle_time(task_id):
    """Frozen t_o of a sports batch task (seconds after the task's time origin)."""
    return throughput.oracle_time(ENVIRONMENT, task_id)


def horizon(task_id):
    """Episode length H = 1.25 T_easy = 5 t_o after the task's time origin."""
    return throughput.horizon(oracle_time(task_id))


# Arm collision, the same physical failure in C1-C3: in some control step an arm link (not the
# gripper) presses on the table, a scene object or the other arm with more than ARM_COLLISION_N.
# A real arm grazing the table with a few newtons is harmless and goes unnoticed; contact forces of
# tens of newtons are impacts, the scale at which a collaborative arm's collision detection stops
# it. Grazes are only reported (arm_contact_peak); objects they move or knock over are judged by
# each task's own checks. Until 2026-11-09 the three tasks used 3 N for 40 ms, 2 N and 8 N; policy
# runs showed grazes at <= 4 N and collisions at >= 96 N, the oracle at 0 N.
ARM_COLLISION_N = 20.0


def arm_collision(state) -> bool:
    """Whether this control step's arm contact is a collision (see ARM_COLLISION_N)."""
    return state.get("arm_contact_peak", 0.0) > ARM_COLLISION_N


# Progress steps per task for the stall window: C1 the 15 accepted balls, C2 the 2 cards.
PROGRESS_STEPS = {"C1": 15, "C2": 2}


def stalled(task_id, now, last_job_time, start=0.0):
    return throughput.stalled(
        now, last_job_time, oracle_time(task_id), start, steps=PROGRESS_STEPS[task_id]
    )


def judge(task_id, job_times, q, physical, difficulty, *, start, t_q, termination_reason):
    """Row fields of a batch task, from the task's own Q-th time `t_q` (None: not reached).

    Sports measures t_q as the frozen t_o was measured (C1: the rack handed over), so a None
    here means the Q-th time was never reached and no tier is met (``throughput.judge`` treats
    an explicit None the same way; the reset below is kept as a guard).
    """
    fields = throughput.judge(
        job_times, q, physical, oracle_time(task_id), difficulty, start=start, t_q=t_q
    )
    if t_q is None:
        never = {tier: False for tier in throughput.TIERS}
        fields.update(
            t_q_s=None,
            completion_time=None,
            temporal_success_by_tier=never,
            joint_success_by_tier=dict(never),
            temporal_success=False,
            joint_success=False,
        )
    fields["termination_reason"] = termination_reason
    return fields


def judge_stream(job_times, deadlines, physical, *, termination_reason):
    """Row fields of a stream task: every job by its own deadline."""
    fields = throughput.judge_stream(job_times, deadlines, physical)
    fields["termination_reason"] = termination_reason
    return fields


def temporal_violations(fields):
    """Why a judged row is not temporally successful (empty when it is)."""
    if fields.get("temporal_success"):
        return []
    if "late_jobs" in fields:
        return [f"late_job_{i}" for i in fields["late_jobs"]] or ["temporal_requirement_not_met"]
    if fields.get("t_q_s") is None:
        return ["jobs_incomplete"]
    return ["t_q_over_budget"]
