"""Re-evaluate recorded measured states without executing the controller.

This tests logging/evaluator agreement, not an independent contact simulator.
"""

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace


def replay(path):
    recorded = json.loads((path / "result.json").read_text())
    task_id = recorded["task"]
    config = recorded.get("configuration") or json.loads((path / "config.json").read_text())
    backend = SimpleNamespace(config=config, time=0.0)
    if task_id in ("A2", "B1"):
        from roboclock.envs.sports.arcade.tasks import MalletMoles, TaikoTask

        cls = TaikoTask if task_id == "A2" else MalletMoles
        task = cls(recorded["difficulty"], recorded["seed"], config)
        expected = recorded["metrics"]
        keys = [
            "joint_success",
            "manipulation_success",
            "temporal_success",
            "physical_failures",
            "temporal_violations",
            "correct_operations",
            "attempted_operations",
        ]
    elif task_id == "B3":
        from roboclock.envs.sports.bench_press.task import BenchPress

        backend.human_effort = SimpleNamespace(released=False)
        task = BenchPress(backend)
        expected = recorded
        keys = [
            "success",
            "physical_success",
            "temporal_success",
            "standby_success",
            "events",
            "failures",
            "catch_time",
            "exhaustion_time",
            "rack_time",
        ]
    elif task_id == "C1":
        from roboclock.envs.sports.pool_rack.task import Task

        task = Task(backend)
        expected = recorded
        keys = [
            "success",
            "physical_success",
            "temporal_success",
            "events",
            "failures",
            "accepted",
            "final_occupancy",
            "completed_at",
        ]
    elif task_id == "C3":
        from roboclock.envs.sports.ball_service.task import Task

        backend.gate_drives = {
            i: SimpleNamespace(
                GetTargetPositionAttr=lambda: SimpleNamespace(Set=lambda target: None)
            )
            for i in range(4)
        }
        task = Task(backend)
        expected = recorded
        keys = [
            "success",
            "physical_success",
            "temporal_success",
            "events",
            "failures",
            "accepted",
            "served",
            "demand",
            "max_occupancy",
        ]
    else:
        raise ValueError("Use audit_freeze.py for B4; this replay supports A2/B1/B3/C1/C3")
    samples = 0
    previous = 0.0
    maximum_gap = 0.0
    with gzip.open(path / "trace.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            if row.get("phase") == "post_roll":
                break
            now = row["time"]
            assert now > previous
            maximum_gap = max(maximum_gap, now - previous)
            backend.time = now
            backend.last_state = row["state"]
            if task_id in ("A2", "B1"):
                task.observe(row["state"], config["sim"]["physics_dt"])
            else:
                task.update(row["state"])
            previous = now
            samples += 1
    if task_id in ("A2", "B1"):
        # Collision event count is exported separately from the sampled state.
        task.collision_count = expected["collision_count"]
        actual = task.metrics()
    else:
        actual = task.result()
    matches = {key: actual[key] == expected[key] for key in keys}
    result = dict(
        scope="Re-evaluation of measured full-rate trace; no controller and no rerun of physics",
        task=task_id,
        samples=samples,
        maximum_gap=maximum_gap,
        checks=matches,
        passed=all(matches.values()),
    )
    (path / "trace_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(path.name, result["passed"], samples, flush=True)
    if not result["passed"]:
        print(
            {
                k: dict(recorded=expected[k], replayed=actual[k])
                for k, v in matches.items()
                if not v
            },
            flush=True,
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()
    results = [replay(path) for path in args.runs]
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
