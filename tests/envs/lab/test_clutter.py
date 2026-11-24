import numpy as np
import pytest

from roboclock.envs.lab.clutter import reserved_regions, sample_clutter
from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.layout import make_layout
from roboclock.envs.lab.tasks.registry import TASKS, make_task


@pytest.mark.parametrize("task_id", list(TASKS))
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_clutter_is_supported_reproducible_and_keeps_task_clearance(task_id, seed):
    config = load_config()
    task = make_task(task_id, seed=seed, config=config)
    task.bind(make_layout(task, config))
    objects = sample_clutter(task, task.layout, config)
    replay = sample_clutter(task, task.layout, config)
    assert len(objects) == config["scene"]["clutter"]["count"]
    low, high = task.layout.workspace_bounds
    for index, item in enumerate(objects):
        assert np.array_equal(item.position, replay[index].position)
        assert item.position[2] - item.size[2] / 2 == pytest.approx(
            task.layout.table_height + 0.001
        )
        assert np.all(item.position[:2] - item.size[:2] / 2 >= low[:2])
        assert np.all(item.position[:2] + item.size[:2] / 2 <= high[:2])
        for center, extent in reserved_regions(task, task.layout, config):
            assert np.any(np.abs(item.position[:2] - center) >= item.size[:2] / 2 + extent)
        for other in objects[index + 1 :]:
            assert np.any(
                np.abs(item.position[:2] - other.position[:2])
                >= (item.size[:2] + other.size[:2]) / 2
                + config["scene"]["clutter"]["pair_clearance"]
            )


def test_clutter_can_be_disabled_without_changing_task_sampling():
    config = load_config()
    task = make_task("A2", seed=1, config=config)
    task.bind(make_layout(task, config))
    initial = task.layout.objects["sample_0"].position.copy()
    sample_clutter(task, task.layout, config)
    config["scene"]["clutter"]["enabled"] = False
    assert sample_clutter(task, task.layout, config) == []
    assert np.array_equal(make_layout(task, config).objects["sample_0"].position, initial)


def test_clutter_density_stays_independent_of_timing_difficulty():
    config = load_config()
    poses = []
    for difficulty in ["easy", "medium", "hard"]:
        task = make_task("A4", difficulty=difficulty, seed=1, config=config)
        task.bind(make_layout(task, config))
        poses.append(
            np.array([item.position for item in sample_clutter(task, task.layout, config)])
        )
    assert np.array_equal(poses[0], poses[1])
    assert np.array_equal(poses[0], poses[2])
