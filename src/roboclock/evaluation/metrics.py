"""Aggregate episode artifacts by task and temporal category."""

import math
import statistics
from collections import defaultdict

from roboclock.evaluation.schema import NUMERIC_METRICS, SCORING_FIELDS, require_valid


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No episodes to summarize")
    require_valid(rows, fields=SCORING_FIELDS, source="Episodes (try summarize --normalize)")

    def summarize(group):
        result = {"episodes": len(group)}
        for key in ("task_success", "manipulation_success", "temporal_success", "joint_success"):
            result[key + "_rate"] = sum(bool(row[key]) for row in group) / len(group)
        for key in NUMERIC_METRICS:
            values = [
                row[key] for row in group if row.get(key) is not None and math.isfinite(row[key])
            ]
            if values:
                result["mean_" + key.removeprefix("mean_")] = statistics.mean(values)
        result["temporal_violation_rate"] = sum(
            bool(r["temporal_violations"]) for r in group
        ) / len(group)
        reactive = [r for r in group if r["category"] == "event_reactive"]
        if reactive:
            result["deadline_miss_rate"] = sum(
                bool(r.get("deadline_missed")) for r in reactive
            ) / len(reactive)
        return result

    groups = defaultdict(list)
    for row in rows:
        prefix = f"{row['environment']}/" if "environment" in row else ""
        # Rows recorded before the block-and-replay evaluator call the fixed delay inference_latency.
        delay = row.get("control_delay", row.get("inference_latency", 0))
        groups[f"{prefix}{row['task_id']}/{row['difficulty']}/{delay}"].append(row)
    categories = defaultdict(list)
    environments = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
        environments[row.get("environment", "unspecified")].append(row)
    return {
        "overall": summarize(rows),
        "by_task_difficulty_latency": {k: summarize(v) for k, v in sorted(groups.items())},
        "by_category": {k: summarize(v) for k, v in sorted(categories.items())},
        "by_environment": {k: summarize(v) for k, v in sorted(environments.items())},
    }
