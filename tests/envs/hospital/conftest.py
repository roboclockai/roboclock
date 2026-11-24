from dataclasses import replace

import pytest

from roboclock.envs.hospital.config import load_legacy_config as load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task


@pytest.fixture
def task_factory():
    def factory(task_id, **overrides):
        config = load_config()
        task = make_task(task_id, seed=7, config=config)
        task.spec = replace(task.spec, **overrides)
        task.bind(make_layout(task, config))
        return task

    return factory
