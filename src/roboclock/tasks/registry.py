"""Task lookup scoped by environment, so domains may reuse the same task IDs."""

from roboclock.envs.registry import get_environment, import_symbol


def task_registry(environment="lab"):
    return import_symbol(get_environment(environment).task_registry)


def make_task(task_id, difficulty="easy", seed=0, config=None, *, environment="lab"):
    return import_symbol(get_environment(environment).task_factory)(
        task_id, difficulty, seed, config
    )
