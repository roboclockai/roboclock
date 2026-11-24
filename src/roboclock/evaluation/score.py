"""Stage scores: partial credit for how far an episode got, in the manner of RoboDojo.

RoboDojo gives each task a process score: ordered stages, each worth a fixed number of
points, where only physical task-state changes count (an object lifted, a key pressed,
a part placed) and the last tier needs success. roboclock follows that rule:

- A domain lists each task's intermediate stages (``EnvironmentSpec.stages``), ordered,
  with scores strictly increasing inside (0, 80). A stage is reached when its check holds
  on the episode's result row; it counts only if every earlier stage was reached too
  (RoboDojo's ``transition`` mode).
- Physical completion (``task_success``) scores ``PHYSICAL`` = 80 on its own, since it
  implies every physical stage. Joint success (physical and temporal) scores 100.
  So an episode that does the task but misses its timing caps at 80.
- A task without listed stages scores 0, 80 or 100 from the success flags alone.

The episode row gains ``score`` (0-100) and ``score_stages`` (names reached, in order).
``python -m roboclock.evaluation.score DIR...`` scores existing ``episodes.jsonl`` files.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PHYSICAL = 80.0
JOINT = 100.0
PHYSICAL_STAGE = "physically_complete"
JOINT_STAGE = "completed_on_time"


@dataclass(frozen=True)
class Stage:
    name: str
    score: float
    reached: Callable[[dict], bool]


def events(row):
    """Names of the events and milestones an episode row records, whatever its domain."""
    names = set()
    for event in row.get("events") or []:
        if isinstance(event, dict):
            for key in ("type", "kind", "event"):
                if event.get(key):
                    names.add(str(event[key]))
    for key in ("event_times", "scene_events", "measured_events"):
        names.update(str(name) for name in (row.get(key) or {}))
    return names


def given(*fields):
    """Check: every named row field is set: not None and not False (0.0 counts as set)."""
    return lambda row: all(
        row.get(field) is not None and row.get(field) is not False for field in fields
    )


def happened(*names):
    """Check: every named event or milestone appears in the row."""
    return lambda row: set(names) <= events(row)


def check_stages(stages, where=""):
    """Raise unless the stages are named, ordered and scored strictly inside (0, PHYSICAL)."""
    previous = 0.0
    names = set()
    for stage in stages:
        if not stage.name or stage.name in names or stage.name in (PHYSICAL_STAGE, JOINT_STAGE):
            raise ValueError(f"{where}: stage name {stage.name!r} is empty, repeated or reserved")
        if not previous < stage.score < PHYSICAL:
            raise ValueError(
                f"{where}: stage {stage.name!r} scores {stage.score}; scores must increase "
                f"strictly inside (0, {PHYSICAL:g})"
            )
        names.add(stage.name)
        previous = stage.score
    return stages


def score_row(row, stages=()):
    """``{"score", "score_stages"}`` for one episode row under the given intermediate stages."""
    value, reached = 0.0, []
    for stage in stages:
        if not stage.reached(row):
            break
        value = stage.score
        reached.append(stage.name)
    if row.get("task_success"):
        value = PHYSICAL
        reached.append(PHYSICAL_STAGE)
    if row.get("joint_success"):
        value = JOINT
        reached.append(JOINT_STAGE)
    return {"score": value, "score_stages": reached}


def task_stages(environment, task_id, row=None):
    """The intermediate stages of one task; a domain may build them from the row
    (count-based tasks). Empty when the domain lists none for the task."""
    from roboclock.envs.registry import get_environment, import_symbol

    reference = get_environment(environment).stages
    if reference is None:
        return ()
    table = import_symbol(reference)
    entry = table.get(task_id, ())
    stages = entry(row or {}) if callable(entry) else entry
    return check_stages(tuple(stages), f"{environment}/{task_id}")


def score_fields(environment, row):
    """Score fields for an episode row of `environment`.

    A batch C-class row (score_version 2) is judged at every time tier from one run; it also
    gains ``score_by_tier``, each tier scored with that tier's joint success.
    """
    stages = task_stages(environment, row["task_id"], row)
    fields = score_row(row, stages)
    tiers = row.get("joint_success_by_tier")
    if isinstance(tiers, dict):
        fields["score_by_tier"] = {
            tier: score_row({**row, "joint_success": bool(joint)}, stages)["score"]
            for tier, joint in tiers.items()
        }
    return fields


def _rows(paths):
    for path in paths:
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.rglob("episodes.jsonl"))
        for file in files:
            for line in file.read_text().splitlines():
                if line.strip():
                    yield file, json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m roboclock.evaluation.score",
        description="Score existing episodes.jsonl files with the current stage definitions.",
    )
    parser.add_argument("paths", nargs="+", help="episodes.jsonl files or directories holding them")
    parser.add_argument("--episodes", action="store_true", help="also print one line per episode")
    args = parser.parse_args(argv)
    groups = defaultdict(list)
    for file, row in _rows(args.paths):
        environment = row.get("environment", "lab")
        scored = score_fields(environment, row)
        groups[(environment, row["task_id"])].append((row, scored))
        if args.episodes:
            print(
                f"{environment}/{row['task_id']} seed={row.get('seed')} score={scored['score']:g} "
                f"stages={','.join(scored['score_stages']) or '-'}  ({file.parent.name})"
            )
    if not groups:
        print("No episodes found", file=sys.stderr)
        return 1
    print(f"{'task':<16} {'episodes':>8} {'joint':>7} {'score':>7}  stages reached (episodes)")
    for (environment, task), items in sorted(groups.items()):
        joint = sum(bool(row["joint_success"]) for row, _ in items) / len(items)
        mean = statistics.mean(scored["score"] for _, scored in items)
        counts = defaultdict(int)
        for _, scored in items:
            for name in scored["score_stages"]:
                counts[name] += 1
        listed = [stage.name for stage in task_stages(environment, task, items[0][0])]
        reached = ", ".join(
            f"{name} {counts[name]}"
            for name in [*listed, PHYSICAL_STAGE, JOINT_STAGE]
            if counts[name]
        )
        print(
            f"{environment + '/' + task:<16} {len(items):>8} {joint:>7.0%} {mean:>7.1f}  {reached or '-'}"
        )
        tiered = [
            row["joint_success_by_tier"] for row, _ in items if row.get("joint_success_by_tier")
        ]
        if tiered:
            rates = ", ".join(
                f"{tier} {sum(bool(t.get(tier)) for t in tiered) / len(tiered):.0%}"
                for tier in ("easy", "medium", "hard")
            )
            print(f"{'':<16} {'':>8} joint by tier (one run each): {rates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
