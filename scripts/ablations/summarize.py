#!/usr/bin/env python3
"""Summarize experiment 1: LTSR and physical success per model, tier and inference budget.

The paper's score is a task macro average: the seeds of a task are averaged first, then the
tasks are weighted equally. ``roboclock.evaluation.metrics.aggregate`` averages over episodes
instead, which lets a task with more episodes pull the total, so this is its own summary and
not a call into that one.

Uncertainty is a paired bootstrap clustered on the task: a resample draws tasks with
replacement and keeps, inside each drawn task, the paired seeds of every condition, so that a
difference between two budgets is taken on the same instances.

    scripts/ablations/summarize.py $WORK_DIR/roboclock_ablation1 --out summary.json
"""

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Judged per tier from one rollout (lab C2): the row carries all three verdicts.
BY_TIER_FIELD = "joint_success_by_tier"
TIER_OF_LEVEL = {"easy": "t1", "medium": "t2", "hard": "t3"}


def read_rows(root):
    """Every episode under ``root``, tagged with the cell its directory names."""
    rows = []
    for model_dir in sorted(p for p in root.iterdir() if (p / "out").is_dir()):
        for out in sorted((model_dir / "out").iterdir()):
            episodes = out / "episodes.jsonl"
            if not episodes.is_file():
                continue
            # <env>_<task>_<tier>_L<budget>_s<seed>
            env, task, tier, budget, seed = out.name.rsplit("_", 4)
            for line in episodes.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_model"] = model_dir.name
                row["_env"], row["_task"], row["_tier"] = env, task, tier
                row["_budget"] = float(budget[1:])
                row["_seed"] = int(seed[1:])
                rows.append(row)
    return rows


def expand(rows):
    """One record per (model, task, tier, budget, seed), with the judged outcomes.

    A by_tier cell ("all") becomes one record per tier, all from the same rollout. That is a
    threshold sensitivity on one trajectory, so the records carry ``shared_rollout``.
    """
    out = []
    for row in rows:
        base = {
            "model": row["_model"],
            "task": f"{row['_env']}/{row['_task']}",
            "budget": row["_budget"],
            "seed": row["_seed"],
            "physical": bool(row.get("task_success")),
            "control_delay": row.get("control_delay"),
            "inference_budget_s": row.get("inference_budget_s"),
            "interface_delay_s": row.get("interface_delay_s"),
            "queries": row.get("model_queries"),
            "termination_reason": row.get("termination_reason"),
        }
        if row["_tier"] == "all":
            by_tier = row.get(BY_TIER_FIELD) or {}
            for level, tier in TIER_OF_LEVEL.items():
                out.append(
                    {**base, "tier": tier, "joint": bool(by_tier.get(level)),
                     "temporal": bool(by_tier.get(level)), "shared_rollout": True}
                )
        else:
            out.append(
                {**base, "tier": row["_tier"], "joint": bool(row.get("joint_success")),
                 "temporal": bool(row.get("temporal_success")), "shared_rollout": False}
            )
    return out


def macro(records, field):
    """Task macro average of ``field``: mean over a task's seeds, then over the tasks."""
    per_task = defaultdict(list)
    for record in records:
        per_task[record["task"]].append(float(record[field]))
    if not per_task:
        return None, 0, 0
    means = [statistics.mean(v) for v in per_task.values()]
    return statistics.mean(means), len(per_task), sum(len(v) for v in per_task.values())


def paired_bootstrap(records_a, records_b, field, draws=10000, seed=0):
    """95 % CI of the task macro difference a - b, resampling tasks (the cluster).

    Inside a drawn task the paired seeds of both conditions are kept, so the difference is
    taken on the same instances.
    """
    by_task = defaultdict(lambda: ([], []))
    for record in records_a:
        by_task[record["task"]][0].append(float(record[field]))
    for record in records_b:
        by_task[record["task"]][1].append(float(record[field]))
    tasks = [t for t, (a, b) in by_task.items() if a and b]
    if len(tasks) < 2:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        drawn = [by_task[rng.choice(tasks)] for _ in tasks]
        diffs.append(
            statistics.mean(statistics.mean(a) for a, _ in drawn)
            - statistics.mean(statistics.mean(b) for _, b in drawn)
        )
    diffs.sort()
    point = statistics.mean(statistics.mean(by_task[t][0]) for t in tasks) - statistics.mean(
        statistics.mean(by_task[t][1]) for t in tasks
    )
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return {"difference": point, "ci95": [lo, hi], "tasks": len(tasks), "draws": draws}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args(argv)

    records = expand(read_rows(args.root))
    if not records:
        print(f"no episodes under {args.root}", file=sys.stderr)
        return 1

    cells = defaultdict(list)
    for record in records:
        cells[(record["model"], record["tier"], record["budget"])].append(record)

    table = []
    for (model, tier, budget), group in sorted(cells.items()):
        ltsr, tasks, episodes = macro(group, "joint")
        physical, _, _ = macro(group, "physical")
        table.append(
            {
                "model": model, "tier": tier, "inference_budget_s": budget,
                "ltsr_task_macro": ltsr, "physical_task_macro": physical,
                "tasks": tasks, "episodes": episodes,
                "shared_rollout_tasks": sorted(
                    {r["task"] for r in group if r["shared_rollout"]}
                ),
            }
        )

    # Every budget against L = 0 at the same tier and model: the latency effect, paired.
    contrasts = []
    budgets = sorted({r["budget"] for r in records})
    for (model, tier, budget), group in sorted(cells.items()):
        if budget == budgets[0]:
            continue
        reference = cells.get((model, tier, budgets[0]))
        if not reference:
            continue
        result = paired_bootstrap(group, reference, "joint", args.draws)
        if result:
            contrasts.append(
                {"model": model, "tier": tier, "inference_budget_s": budget,
                 "versus_budget_s": budgets[0], "metric": "ltsr_task_macro", **result}
            )

    summary = {
        "root": str(args.root),
        "records": len(records),
        "budgets_s": budgets,
        "note": "LTSR and physical success are task macro averages (seeds within a task, then "
                "tasks equally weighted). CIs are a paired bootstrap clustered on the task.",
        "cells": table,
        "contrasts_vs_lowest_budget": contrasts,
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(f"{'model':34}{'tier':6}{'L':>6} {'LTSR':>7} {'PSR':>7} {'tasks':>6} {'eps':>5}")
    for row in table:
        print(f"{row['model']:34}{row['tier']:6}{row['inference_budget_s']:6g} "
              f"{row['ltsr_task_macro']:7.3f} {row['physical_task_macro']:7.3f} "
              f"{row['tasks']:6} {row['episodes']:5}")
    if args.out:
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
