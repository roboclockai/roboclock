"""Pool layout revisions must not move unrelated tabletop robots."""

import pytest

from roboclock.envs.sports.arcade.tasks import configuration
from roboclock.simulators.isaac.embodiments.config import configure


@pytest.mark.parametrize("task", ["A2", "B1"])
def test_tool_task_mount_matches_recorded_reachable_workspace(task):
    c = configuration(task, "basic", "v5")
    configure(c, task, "dual_arx_x5")
    assert c["embodiment"]["position"] == [0.15, 0, 1.0]


def test_pool_rack_keeps_its_own_mount():
    from roboclock.envs.sports.pool_rack.config import config

    assert config()["embodiment"]["position"] == [0.14, 0, 1.02]
