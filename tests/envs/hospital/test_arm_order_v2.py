"""hospital inherits the lab interface arm order: the rig's LEFT arm (+y) comes first."""

import math

import numpy as np
import pytest
from test_hospital_environment import FakeBackend

from roboclock.envs import create_env, robodojo_rig
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.rig import ARM_SIDES, WRIST_KEYS, rig_mounts, rig_pose
from roboclock.envs.hospital.tasks.registry import TASKS
from roboclock.envs.lab.robot import EXTERNAL_ARMS, EXTERNAL_JOINT_ORDER, HOME, mount_positions


def rig_lateral(config, task_id):
    origin, yaw = rig_pose(config, task_id)
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return [float(np.dot(np.asarray(m) - origin, left)) for m in mount_positions(config, task_id)]


class DistinctArmsBackend(FakeBackend):
    def observation(self, depth=False, images=True):
        result = super().observation(depth, images)
        result["joint_positions"] = np.r_[np.full(8, 0.0), np.full(8, 1.0)].astype(np.float32)
        result["ee_pose"] = np.r_[np.full(7, 0.0), np.full(7, 1.0)].astype(np.float32)
        result["gripper_width"] = np.array([0.01, 0.02], dtype=np.float32)
        return result


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr("roboclock.envs.hospital.backend.HospitalBackend", DistinctArmsBackend)
    instance = create_env("hospital", "B1", "easy", runtime=object())
    instance.reset(seed=0)
    yield instance
    instance.close()


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_first_interface_arm_is_mounted_on_the_rig_left(task_id):
    config = load_config()
    lateral = rig_lateral(config, task_id)
    first, second = EXTERNAL_ARMS
    assert lateral[first] > 0 > lateral[second], (task_id, lateral)
    # The scene places the bases where the spec says: rig_mounts agrees with the lab mounts.
    mounts, _ = rig_mounts(config, task_id)
    np.testing.assert_allclose(mounts, mount_positions(config, task_id), atol=1e-9)
    origin, yaw = rig_pose(config, task_id)
    spec_left, _ = robodojo_rig.base_positions(origin, math.degrees(yaw))
    np.testing.assert_allclose(mounts[first], spec_left, atol=1e-9)
    # `left_wrist_rgb` rides the first interface arm.
    assert ARM_SIDES[first] == "left" and WRIST_KEYS[ARM_SIDES[first]] == "left_wrist_rgb"


def test_first_eight_action_values_drive_the_rig_left_arm(env):
    action = np.r_[HOME + [0.1, 0, 0, 0, 0, 0, 0, 0], HOME].astype(np.float32)
    env.step(action)
    sent = env.backend.actions[-1]
    left = EXTERNAL_ARMS[0]
    np.testing.assert_array_equal(sent[8 * left : 8 * left + 8], action[:8])
    np.testing.assert_array_equal(sent[8 * (1 - left) : 8 * (1 - left) + 8], action[8:])


def test_per_arm_observations_list_the_rig_left_arm_first(env):
    obs = env.get_obs()
    left = EXTERNAL_ARMS[0]
    np.testing.assert_array_equal(obs["joint_positions"][:8], np.full(8, float(left)))
    np.testing.assert_array_equal(obs["ee_pose"][:7], np.full(7, float(left)))
    assert obs["gripper_width"][0] == pytest.approx([0.01, 0.02][left])


def test_joint_order_labels_follow_the_physical_side(env):
    assert env.joint_order() == list(EXTERNAL_JOINT_ORDER)
    assert env.joint_order()[0] == "left/joint1" and env.joint_order()[8] == "right/joint1"
