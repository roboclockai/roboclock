"""Paper-table metrics of a train45 evaluation, split into the 45 seen and the 15 held-out tasks.

PS = stage score, PSR = task_success, TSR = joint_success (in %). Per task the mean over its seeds,
then the macro-average over the tasks of each category (A temporal_window, B event_reactive,
C throughput) and over all of them: the formula of the earlier RoboDojo table (docs/robodojo-sweep.md).
As there, household C1 with no qualified meal never counts as a success.

  PYTHONPATH=src python scripts/train45/eval_summary.py $TRAIN45_DIR/eval/D_zero/internw0ft/out [--tasks]
"""

import collections
import glob
import json
import os
import sys

from roboclock.envs.hospital.results import RENAMED as HOSPITAL_RENAMED
from roboclock.envs.sports.results import RENAMED as SPORTS_RENAMED
from roboclock.evaluation.score import score_row, task_stages

# docs/five-domain-training.md, 3.1: one A, one B and one C per domain.
HELD_OUT = {
    ("hospital", "A3"), ("hospital", "B6"), ("hospital", "C4"),
    ("household", "A3"), ("household", "B1"), ("household", "C2"),
    ("industry", "A3"), ("industry", "B2"), ("industry", "C4"),
    ("lab", "A6"), ("lab", "B5"), ("lab", "C3"),
    ("sports", "A2"), ("sports", "B2"), ("sports", "C3"),
}
CATS = {"A": "temporal_window", "B": "event_reactive", "C": "throughput"}


def load(out):
    per_task, category = collections.defaultdict(list), {}
    for path in sorted(glob.glob(f"{out}/*/episodes.jsonl")):
        if os.path.getsize(path) == 0:
            continue
        row = json.loads(open(path).readline())
        env, task = row["environment"], row["task_id"]
        # Recorded before the 2026-11-19 renumbering: hospital B6 -> A4, B7 -> B6;
        # sports A2 (taiko) -> B5, A3 (basketball) -> A2.
        renamed = {"hospital": HOSPITAL_RENAMED, "sports": SPORTS_RENAMED}.get(env, {})
        if (task, row.get("task_name")) in renamed:
            task = renamed[task, row.get("task_name")]
            row = dict(row, task_id=task, category=CATS[task[0]])
        if env == "household" and task == "C1" and not row.get("qualified_meals"):
            row = dict(row, task_success=False, joint_success=False, temporal_success=False)
        ps = score_row(row, task_stages(env, task, row))["score"]
        per_task[(env, task)].append(
            (row["seed"], ps, 100.0 * bool(row["task_success"]), 100.0 * bool(row["joint_success"]))
        )
        category[(env, task)] = row["category"]
    return per_task, category


def macro(per_task, tasks):
    means = [[sum(x[i] for x in per_task[t]) / len(per_task[t]) for i in (1, 2, 3)] for t in tasks]
    return [sum(m[i] for m in means) / len(means) for i in range(3)] if means else [float("nan")] * 3


def main():
    out = sys.argv[1]
    per_task, category = load(out)
    episodes = sum(len(v) for v in per_task.values())
    print(f"{out}: {episodes} episodes, {len(per_task)} tasks")
    print(f"{'split':8s} {'cat':4s} {'tasks':>5s} {'PS':>6s} {'PSR':>6s} {'TSR':>6s}")
    for split, keep in (("seen", lambda t: t not in HELD_OUT), ("unseen", lambda t: t in HELD_OUT), ("all", lambda t: True)):
        for key, name in list(CATS.items()) + [("all", None)]:
            tasks = [t for t in per_task if keep(t) and (name is None or category[t] == name)]
            ps, psr, tsr = macro(per_task, tasks)
            print(f"{split:8s} {key:4s} {len(tasks):5d} {ps:6.1f} {psr:6.1f} {tsr:6.1f}")
    if "--tasks" in sys.argv:
        print("\nper task (seed: TSR/PSR/PS)")
        for t in sorted(per_task):
            cells = "  ".join(f"{s}: {int(j > 0)}/{int(p > 0)}/{ps:.0f}" for s, ps, p, j in sorted(per_task[t]))
            print(f"  {'unseen' if t in HELD_OUT else 'seen  '} {t[0]:12s} {t[1]:3s} {cells}")


if __name__ == "__main__":
    main()
