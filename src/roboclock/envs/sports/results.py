"""Convert the sports demos' native ``result.json`` into the shared result row.

Taiko and moles (arcade) already store a row under ``metrics``; the other eight
demos write ``task``, ``success``, ``physical_success``, ``temporal_success`` and
a ``failures`` list, plus task-specific measurements mapped below. The unified
runtime (``roboclock eval --env sports``) scores its episodes with the same function.

C-class results since score_version 2 (C1-C3) also carry their judged fields under
``throughput`` (``roboclock.envs.sports.c_class``); those fields are the row's verdicts
(task, manipulation, temporal and joint success, completion time, job times, per-tier
verdicts and the termination reason) and are applied last.
"""

import statistics

from roboclock.envs.sports.c_class import temporal_violations

# Failure codes that describe the manipulation outcome rather than timing: the
# codes each task's evaluator excludes from (or reports for) physical success.
PHYSICAL_FAILURES = {
    "lost_robot_grasp",
    "direct_target_contact",
    "episode_timeout",
    "arm_collision",
    # A1 chess
    "invalid_or_dropped_piece",
    "unrelated_piece_moved",
    "placement_lost_after_clock",
    # A2 basketball (A3 until 2026-11-19)
    "no_grasp",
    "no_release",
    "missed_basket",
    "ball_recontact_after_release",
    "release_inside_no_dunk_region",
    # B2 boxing, C2 cups
    "incomplete_combinations",
    "moved_during_hold",
    "early_or_incorrect_bell",
    "dropped_cup",
    # B3 bench press
    "human_safety_contact",
    # B4 stop/go
    "puck_lifted",
    "puck_out_of_bounds",
    "incomplete_route",
    # C1 racking, C3 service
    "cue_ball_touched",
    "incomplete_rack_at_deadline",
    "occupied_cell",
    "overflow",
    "incomplete_cards",
}
# Per-object codes (dropped_tool_left, ball_dropped_ball07, controller_timeout_...).
PHYSICAL_PREFIXES = ("dropped_", "ball_dropped_", "controller_timeout_")
# Current task IDs; the arcade rows keep their original numeric task_id (32, 34).
CURRENT_TASKS = {"A1", "A2", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3"}
ARCADE_TASKS = {"32": "B5", "34": "B1"}  # taiko, moles
# Renumbered 2026-11-19: taiko A2 -> B5 (event-reactive), basketball A3 -> A2. Rows recorded
# before carry the old ID with their task name; native results carry the sub-package kind.
RENAMED = {
    ("A2", "two_stick_taiko"): "B5",
    ("A3", "basketball_shot_clock"): "A2",
}
NATIVE_KINDS = {"A3": "A2"}  # arena kind -> benchmark ID (arcade rows go by ARCADE_TASKS)


def _category(task_id):
    return {"A": "temporal_window", "B": "event_reactive", "C": "throughput"}[task_id[0]]


def _renamed(row, task_id):
    if task_id == row.get("task_id"):
        return row
    return {**row, "task_id": task_id, "category": _category(task_id),
            "legacy_task_id": str(row.get("task_id"))}


def _physical(failure):
    # Chess failures carry their turn index ("2:deadline_missed").
    code = failure.split(":", 1)[-1]
    return code in PHYSICAL_FAILURES or code.startswith(PHYSICAL_PREFIXES)


def _rate(document):
    for key in ("balls_per_budget_minute", "cards_per_minute", "served_per_minute"):
        if document.get(key) is not None:
            return float(document[key])
    return None


def _reaction_latency(document):
    if document.get("reaction_latency") is not None:
        return float(document["reaction_latency"])
    latencies = [
        r["latency"] for r in document.get("reactions") or [] if r.get("latency") is not None
    ]
    return statistics.mean(latencies) if latencies else None


def _collisions(document):
    count = document.get("arm_collision_count")
    if isinstance(count, dict):
        return sum(count.values())
    return count


def _episode(document, key):
    # Early arena results keep the episode identity only in their configuration.
    configuration = document.get("configuration")
    configuration = configuration if isinstance(configuration, dict) else {}
    value = document.get(key, configuration.get(key))
    if value is None:
        raise ValueError(f"Sports result for task {document.get('task')!r} has no {key}")
    return value


def normalize(document: dict) -> dict | None:
    metrics = document.get("metrics")
    if isinstance(metrics, dict) and "task_success" in metrics:
        task = ARCADE_TASKS.get(str(metrics.get("task_id")), str(document.get("task")))
        if task in CURRENT_TASKS and str(metrics.get("task_id")) != task:
            return _renamed(metrics, task)
        return metrics
    if "task_success" in document:
        key = (str(document.get("task_id")), document.get("task_name"))
        return _renamed(document, RENAMED[key]) if key in RENAMED else document
    if "task" not in document or "physical_success" not in document:
        return None
    if document["physical_success"] is None:
        # Run-status records (supplemental matrix *.status.json) point to the
        # result file that carries the outcome; they are not results themselves.
        return None
    task_id = NATIVE_KINDS.get(str(document["task"]), str(document["task"]))
    physical = bool(document.get("physical_success"))
    temporal = bool(document.get("temporal_success"))
    success = document.get("success")
    failures = [str(f) for f in document.get("failures") or []]
    temporal_failures = [f for f in failures if not _physical(f)]
    row = {
        "task_id": task_id,
        "difficulty": _episode(document, "difficulty"),
        "seed": _episode(document, "seed"),
        "category": document.get("primary_temporal_category"),
        "variant": document.get("variant"),
        "task_success": physical,
        "manipulation_success": physical,
        "temporal_success": temporal,
        "joint_success": bool(success) if success is not None else physical and temporal,
        "temporal_violations": []
        if temporal
        else temporal_failures or ["temporal_requirement_not_met"],
        "physical_failures": [f for f in failures if _physical(f)],
        "elapsed_time": document.get("evaluation_end", document.get("sim_time")),
        "completion_time": document.get("completed_at"),
        "reaction_latency": _reaction_latency(document),
        "operations_per_minute": _rate(document),
        "collision_count": _collisions(document),
        "controller": document.get("controller"),
        "result_adapter": "sports.native_result",
    }
    judged = document.get("throughput")
    if isinstance(judged, dict) and judged.get("score_version"):
        # C-class verdicts (score_version 2) replace the native ones; nothing below overrides them.
        row.update(judged)
        row["temporal_violations"] = temporal_violations(judged)
    return row
