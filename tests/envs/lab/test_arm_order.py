"""Lab interface arm order: the rig's LEFT arm (+y in the rig frame) comes first.

Internally side 0 (/World/ARXLeft) stands on the rig's right; LabEnv converts actions and
per-arm observations once, so the first 8 action values must reach the +y arm.
"""

import numpy as np
import pytest
from test_environment import InstrumentedBackend

from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.env import LabEnv
from roboclock.envs.lab.rig_cameras import VIEWS
from roboclock.envs.lab.robot import (
    EXTERNAL_ARMS,
    EXTERNAL_JOINT_ORDER,
    HOME,
    arms_to_external,
    arms_to_internal,
    mount_positions,
    rig_axes,
    rig_pose,
)
from roboclock.envs.lab.tasks.registry import TASKS


def rig_lateral(config, task_id):
    """Each internal side's base offset along the rig's left (+y) axis."""
    origin, yaw = rig_pose(config, task_id)
    _, left = rig_axes(yaw)
    return [float(np.dot(np.asarray(m) - origin, left)) for m in mount_positions(config, task_id)]


class DistinctArmsBackend(InstrumentedBackend):
    """Reports a different joint vector, pose and width per internal side."""

    def observation(self, depth=False, images=True):
        result = super().observation(depth, images)
        result["joint_positions"] = np.r_[np.full(8, 0.0), np.full(8, 1.0)].astype(np.float32)
        result["ee_pose"] = np.r_[np.full(7, 0.0), np.full(7, 1.0)].astype(np.float32)
        result["gripper_width"] = np.array([0.01, 0.02], dtype=np.float32)
        return result


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr("roboclock.envs.lab.backend.IsaacBackend", DistinctArmsBackend)
    instance = LabEnv("A2", runtime=object())
    instance.reset(seed=0)
    yield instance
    instance.close()


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_first_interface_arm_is_mounted_on_the_rig_left(task_id):
    config = load_config()
    lateral = rig_lateral(config, task_id)
    first, second = EXTERNAL_ARMS
    assert lateral[first] > 0 > lateral[second], (task_id, lateral)
    # RoboDojo's left wrist camera rides the same arm.
    assert VIEWS["cam_left_wrist"][2] == first
    assert VIEWS["cam_right_wrist"][2] == second


def test_first_eight_action_values_drive_the_rig_left_arm(env):
    action = np.r_[HOME + [0.1, 0, 0, 0, 0, 0, 0, 0], HOME].astype(np.float32)
    env.step(action)
    sent = env.backend.actions[-1]
    left = EXTERNAL_ARMS[0]
    np.testing.assert_array_equal(sent[8 * left : 8 * left + 8], action[:8])
    np.testing.assert_array_equal(sent[8 * (1 - left) : 8 * (1 - left) + 8], action[8:])
    assert rig_lateral(env.config, env.task.task_id)[left] > 0


def test_per_arm_observations_list_the_rig_left_arm_first(env):
    obs = env.get_obs()
    left = EXTERNAL_ARMS[0]
    np.testing.assert_array_equal(obs["joint_positions"][:8], np.full(8, float(left)))
    np.testing.assert_array_equal(obs["ee_pose"][:7], np.full(7, float(left)))
    assert obs["gripper_width"][0] == pytest.approx([0.01, 0.02][left])
    assert env.observation_space.contains(obs)


def test_joint_order_labels_follow_the_physical_side(env):
    order = env.episode_metadata()["joint_order"]
    assert order == list(EXTERNAL_JOINT_ORDER)
    assert order[:8] == [f"left/joint{i}" for i in range(1, 9)]
    assert order[8:] == [f"right/joint{i}" for i in range(1, 9)]


def test_arm_conversions_are_inverse():
    values = np.arange(32, dtype=np.float32).reshape(2, 16)
    np.testing.assert_array_equal(arms_to_internal(arms_to_external(values)), values)
    np.testing.assert_array_equal(arms_to_external(values)[..., :8], values[..., 8:])
