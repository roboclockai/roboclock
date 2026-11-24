"""Run `roboclock eval` on household C1 with the pickup schedule overridden (re-measuring the hard schedule).

    C1_PICKUPS=0.5,1,1.5,2 C1_MAX_TIME=150 C1_EVENTS=events.jsonl PYTHONPATH=src \
        python scripts/household/c1_oracle_free_run.py eval --env household --task C1 --difficulty medium \
        --seeds 1 --seed-start 2 --policy oracle --oracle --output OUT

C1_PICKUPS (wall seconds, one per order) replaces the schedule and C1_MAX_TIME the horizon; C1_EVENTS appends the
probe events (oracle phases such as the right arm's ``wait_board`` beside the board, waiter pickups) as JSONL.
With every pickup before the first set-down, the waiter takes each meal 1 s after it was set down (the task's late
rule), so the board never holds the oracle back and ``served_s`` are its free-running set-down times. The task
model and the oracle are the code's own; only the spec is replaced. c1_x5/task.py (HARD_PICKUPS_S) records how the
frozen schedule was derived from such runs. One episode per process.
"""
import json
import os
from dataclasses import replace

from roboclock.envs.household.c1_x5 import task as T
from roboclock.envs.household.x5 import episode as E

if os.environ.get("C1_PICKUPS"):
    WALL = tuple(float(x) for x in os.environ["C1_PICKUPS"].split(","))
    T.C1Spec.pickup_times = property(lambda self: WALL)
    _sample = T.C1Spec.sample.__func__

    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        spec = _sample(cls, task_id, difficulty, seed, **overrides)
        if len(spec.order) != len(WALL):
            raise ValueError(f"C1_PICKUPS: {len(WALL)} pickups for {len(spec.order)} orders")
        max_time = float(os.environ.get("C1_MAX_TIME", 1.25*WALL[-1]))
        return replace(spec, pickups_s=WALL, max_time=max_time)

    T.C1Spec.sample = classmethod(sample)

if os.environ.get("C1_EVENTS"):
    _event = E.X5Episode._event

    def _logged(self, name, t, **data):
        _event(self, name, t, **data)
        with open(os.environ["C1_EVENTS"], "a") as f:
            f.write(json.dumps(dict(event=name, t_s=float(t), **data), default=str)+"\n")

    E.X5Episode._event = _logged

from roboclock.cli import main  # noqa: E402

raise SystemExit(main())
