"""Current Dual-X5 household tasks (tasks.json), without importing the simulator.

``make_task`` returns a task object with the shared evaluator contract: a dataclass
``spec`` (with ``max_time``), ``time``, ``terminated``, ``event_log``,
``gen_instruction()`` and ``metrics()``. A physics backend feeds it measured signals
through ``update()`` and advances it with ``advance(dt)``; see ``x5/task_base.py``.
"""

import importlib
import json
from pathlib import Path

CATALOG = json.loads((Path(__file__).parents[1] / "tasks.json").read_text())
TASKS = {row["id"]: importlib.import_module(row["module"]).TASK for row in CATALOG["tasks"]}


def make_task(task_id: str, difficulty="medium", seed=0, config=None, **overrides):
    matches = [cls for key, cls in TASKS.items() if task_id in (key, cls.name)]
    if not matches:
        raise ValueError(f"Unknown task {task_id!r}; choose one of {', '.join(TASKS)}")
    cls = matches[0]
    return cls(cls.Spec.sample(cls.task_id, difficulty, seed, **overrides))
