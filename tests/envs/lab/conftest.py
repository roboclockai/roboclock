from dataclasses import replace

import pytest

from roboclock.config import load_config
from roboclock.envs.lab.layout import make_layout
from roboclock.tasks.registry import make_task


@pytest.fixture
def task_factory():
    def factory(task_id, **overrides):
        config = load_config()
        task = make_task(task_id, seed=7, config=config)
        task.spec = replace(task.spec, **overrides)
        task.bind(make_layout(task, config))
        return task

    return factory


@pytest.fixture
def legacy_task_factory():
    """Retain geometry/predicate regressions for the superseded V1 tasks."""
    from roboclock.envs.lab.config import sample_episode
    from roboclock.envs.lab.tasks.temporal_window.catalyst_addition import CatalystAddition
    from roboclock.envs.lab.tasks.temporal_window.incubation_transfer import IncubationTransfer
    from roboclock.envs.lab.tasks.temporal_window.probe_alignment import ProbeAlignment
    from roboclock.envs.lab.tasks.temporal_window.temperature_removal import TemperatureRemoval

    from roboclock.envs.lab.tasks.reactive.slip_recovery import SlipRecovery
    from roboclock.envs.lab.tasks.reactive.falling_sample import FallingSample

    from roboclock.envs.lab.tasks.throughput.sample_sorting import SampleSorting

    legacy = {
        c.task_id: c
        for c in (
            TemperatureRemoval,
            IncubationTransfer,
            CatalystAddition,
            ProbeAlignment,
            SlipRecovery,
            FallingSample,
            SampleSorting,
        )
    }

    def factory(task_id, **overrides):
        config = load_config()
        task = (
            legacy[task_id](sample_episode(task_id, "easy", 7, config))
            if task_id in legacy
            else make_task(task_id, seed=7, config=config)
        )
        task.spec = replace(task.spec, **overrides)
        task.bind(make_layout(task, config))
        return task

    return factory
