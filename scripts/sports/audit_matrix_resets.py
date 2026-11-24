"""Compare measured first-step object poses for the same seed across levels.

This is an initial-state reproducibility check, not a repeated-policy success test.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def poses(row):
    s = row.get("state")
    if s is None:
        # Chess exports both actor-facing observations of the shared board.
        s = row["states"]["white"]
    return {name: np.array(o["position"]) for name, o in s["objects"].items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    groups = {}
    for status in args.root.glob("*.status.json"):
        d = json.loads(status.read_text())
        if not d.get("runtime_completed"):
            continue
        run = args.root / Path(d["result"]).parent
        with gzip.open(run / "trace.jsonl.gz", "rt") as f:
            row = json.loads(next(f))
        groups.setdefault((d["task"], d["seed"]), {})[d["difficulty"]] = poses(row)
    records = []
    for (task, seed), levels in sorted(groups.items()):
        if "easy" not in levels:
            continue
        for level, objects in levels.items():
            if level == "easy":
                continue
            a = levels["easy"]
            same_ids = a.keys() == objects.keys()
            errors = {
                name: float(np.linalg.norm(a[name] - objects[name]))
                for name in a.keys() & objects.keys()
            }
            records.append(
                dict(
                    task=task,
                    seed=seed,
                    comparison="easy vs " + level,
                    same_object_ids=same_ids,
                    maximum_position_difference_m=max(errors.values(), default=0),
                    within_10_micrometers=same_ids and max(errors.values(), default=0) <= 1e-5,
                )
            )
    report = dict(
        scope="Measured first physics step, same seed across difficulty; 10 micrometer reporting tolerance, no scoring change",
        comparisons=records,
        all_within_tolerance=all(r["within_10_micrometers"] for r in records),
    )
    (args.root / "initial-state-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        len(records),
        "comparisons;",
        sum(not r["within_10_micrometers"] for r in records),
        "outside tolerance",
    )


if __name__ == "__main__":
    main()
