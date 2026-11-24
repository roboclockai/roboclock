"""Actual PhysX negative controls: alter controller decisions, never task predicates.

Use an independent Isaac process. Expected task failures are successful fault
checks only if the requested event actually occurred and was rejected.
"""

import argparse
import contextlib
import gzip
import hashlib
import importlib
import json
import shutil
import sys
import traceback
from pathlib import Path


class IgnoreRedView:
    """Controller-only view; the real evaluator still reads real red windows."""

    def __init__(self, task):
        self.task = task

    def red(self):
        return None

    def __getattr__(self, name):
        return getattr(self.task, name)


def install(task):
    if task == "A3":
        cls = importlib.import_module("roboclock.envs.sports.arena.oracle").Oracle
        original = cls.act

        def action(self):
            end = self.task.c["shot_start"] + self.task.c["shot_clock"]
            if self.phase == "launch" and self.b.time < end + 1 / 30:
                self.start = self.b.time
                return
            original(self)

        method = "act"
        description = "Hold the held ball at aim until after the shot buzzer, then launch normally."
    elif task == "B3":
        cls = importlib.import_module("roboclock.envs.sports.bench_press.oracle").BenchOracle
        original = cls.act

        def action(self):
            if self.task.event_time is not None:
                return  # Continue bounded standby command; do not assist after exhaustion.
            original(self)

        method = "act"
        description = "Do not assist after measured exhaustion; retain the last standby command."
    elif task == "B4":
        cls = importlib.import_module("roboclock.envs.sports.freeze.oracle").Oracle
        original = cls.__init__

        def action(self, backend, task):
            original(self, backend, IgnoreRedView(task))

        method = "__init__"
        description = (
            "Controller ignores red cues; actual evaluator and rendered cues are unchanged."
        )
    else:
        cls = importlib.import_module("roboclock.envs.sports.ball_service.oracle").Oracle
        original = cls.act

        def action(self):
            pass  # Initial bounded drives remain active; no ball is loaded.

        method = "act"
        description = (
            "Keep initial joint targets instead of loading balls; demand proceeds normally."
        )
    setattr(cls, method, action)
    return description, lambda: setattr(cls, method, original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--tasks", nargs="+", choices=["A3", "B3", "B4", "C3"], default=["A3", "B3", "B4", "C3"]
    )
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__)
    shutil.copy2(script, args.output / "fault_runner.py")
    (args.output / "runner-sha256.txt").write_text(
        hashlib.sha256(script.read_bytes()).hexdigest() + "\n"
    )
    from roboclock.simulators.isaac import runtime as rt

    runtime = rt.IsaacRuntime()
    rt.IsaacRuntime = lambda: contextlib.nullcontext(runtime)
    from roboclock.envs.sports.demo import main as demo

    rows = []
    try:
        for task in args.tasks:
            description, restore = install(task)
            row = dict(task=task, fault=description, difficulty="easy", seed=0)
            output = args.output / task
            try:
                sys.argv = [
                    "fault-check",
                    "--task",
                    task,
                    "--difficulty",
                    "easy",
                    "--seed",
                    "0",
                    "--embodiment",
                    "dual_arx_x5",
                    "--presentation-only",
                    "--output",
                    str(output),
                ]
                demo()
                result = json.loads((output / "result.json").read_text())
                failures = result.get("failures", [])
                required = {
                    "A3": {"release_outside_round"},
                    "B3": {"human_safety_contact"},
                    "B4": {"red_motion", "late_stop", "red_drift"},
                    "C3": {"starvation"},
                }[task]
                triggered = bool(set(failures) & required)
                row.update(
                    runtime_completed=True,
                    full_episode=result.get("full_episode", False),
                    task_success=result["success"],
                    failures=failures,
                    requested_fault_observed=triggered,
                    negative_control_passed=not result["success"] and triggered,
                    result=str((output / "result.json").relative_to(args.output)),
                )
            except Exception:
                row.update(
                    runtime_completed=False,
                    traceback=traceback.format_exc(),
                    negative_control_passed=False,
                )
                traceback.print_exc()
            finally:
                restore()
            stage = output / "scene.usda"
            if stage.exists():
                with (
                    stage.open("rb") as src,
                    gzip.open(str(stage) + ".gz", "wb", compresslevel=3) as dst,
                ):
                    shutil.copyfileobj(src, dst)
                stage.unlink()
            rows.append(row)
            (args.output / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
            print("NEGATIVE_RESULT", json.dumps(row), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
