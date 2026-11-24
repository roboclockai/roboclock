"""robot.start_pose: the optional RoboDojo start pose (every arm joint at 0) in each domain."""

import numpy as np
import pytest

from roboclock.envs.start_pose import START_POSES, set_start_pose, start_pose, zero_start


def test_start_pose_defaults_to_the_task_posture_and_validates():
    assert START_POSES == ("task", "zero")
    assert start_pose({}) == "task" and start_pose(None) == "task"
    assert start_pose({"robot": None}) == "task"
    assert zero_start({"robot": {"start_pose": "zero"}})
    with pytest.raises(ValueError, match="robot.start_pose"):
        start_pose({"robot": {"start_pose": "home"}})


def test_set_start_pose_records_the_value_in_the_robot_section():
    config = {"robot": {"model": "x"}}
    assert set_start_pose(config, None) == {"robot": {"model": "x"}}
    set_start_pose(config, "zero")
    assert config == {"robot": {"model": "x", "start_pose": "zero"}}
    assert set_start_pose({}, "task") == {"robot": {"start_pose": "task"}}
    with pytest.raises(ValueError, match="start_pose"):
        set_start_pose({}, "folded")


def test_lab_zero_start_overrides_home_and_per_task_postures():
    from roboclock.envs.lab.config import load_config
    from roboclock.envs.lab.robot import ARM_DOF, HOME, home_joints
    from roboclock.envs.lab.tasks.registry import TASKS

    config = load_config()
    custom = config.get("robot", {}).get("task_home_joints", {})
    set_start_pose(config, "zero")
    for task_id in TASKS:
        joints = home_joints(config, task_id)
        assert joints.shape == (2, 8)
        np.testing.assert_array_equal(joints[:, :ARM_DOF], 0.0)
        np.testing.assert_array_equal(joints[:, ARM_DOF:], np.tile(HOME[ARM_DOF:], (2, 1)))
    # Tasks with their own posture lose it too.
    assert all(np.all(home_joints(config, task_id)[:, :ARM_DOF] == 0) for task_id in custom)


def test_hospital_zero_start_uses_the_lab_home_joints():
    from roboclock.envs.hospital.config import load_config
    from roboclock.envs.lab.robot import home_joints

    config = set_start_pose(load_config(), "zero")
    for task_id in ("A1", "A3", "B4", "C1"):
        assert np.all(home_joints(config, task_id)[:, :6] == 0)


def test_household_zero_start_reaches_the_env_config_and_reset_targets():
    from roboclock.envs.household.x5 import interface as X
    from roboclock.envs.household.x5.config import env_config, load_config

    config = env_config(load_config())
    assert config["robot"] == {"start_pose": "task"}
    assert X.start_joints(config) == X.READY_Q
    set_start_pose(config, "zero")
    assert X.start_joints(config) == (0.0,) * 6

    ready, zero = X.ready_action(), X.ready_action(X.start_joints(config))
    for arm in (slice(0, 8), slice(8, 16)):
        np.testing.assert_array_equal(zero[arm][:6], 0.0)
        np.testing.assert_array_equal(zero[arm][6:], ready[arm][6:])  # jaws open either way
    described = X.robot_configuration(config)
    assert described["start_pose"] == "zero" and described["start_joints_rad"] == [0.0] * 6
    assert described["ready_joints_rad"] == list(X.READY_Q)
    assert X.robot_configuration()["start_joints_rad"] == list(X.READY_Q)


def test_household_config_file_key_reaches_the_env_config(tmp_path):
    import yaml

    from roboclock.envs.household.x5.config import DEFAULT_PATH, env_config, load_config

    raw = yaml.safe_load(DEFAULT_PATH.read_text())
    raw.setdefault("robot", {})["start_pose"] = "zero"
    path = tmp_path / "household.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert env_config(load_config(path))["robot"]["start_pose"] == "zero"


def test_sports_native_config_carries_the_start_pose_to_every_rig():
    from roboclock.envs.sports.config import load_config
    from roboclock.envs.sports.tasks import SOURCES, native_config

    config = load_config()
    for task_id in SOURCES:
        assert native_config(task_id, config=config)["embodiment"]["start_pose"] == "task"
    set_start_pose(config, "zero")
    for task_id in SOURCES:
        assert native_config(task_id, config=config)["embodiment"]["start_pose"] == "zero"


class _FakeController:
    def set_gains(self, **_):
        pass

    def set_max_efforts(self, _):
        pass


class _FakeRobot:
    dof_names = [f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 9)]

    def __init__(self):
        self.q = np.full(len(self.dof_names), 0.5)

    def get_joint_positions(self):
        return self.q.copy()

    def get_articulation_controller(self):
        return _FakeController()

    def set_joints_default_state(self, positions):
        self.default = np.asarray(positions).copy()

    def set_joint_positions(self, q):
        self.q = np.asarray(q).copy()


class _FakeKinematics:
    def solve(self, position, rotation, multistart=False):
        return np.full(6, 1.0), 0.001


def _fake_rig():
    from roboclock.simulators.isaac.embodiments.rig import RobotRig

    rig = object.__new__(RobotRig)
    rig.model, rig.robot, rig.sides = "dual_arx_x5", _FakeRobot(), ("left", "right")
    rig.arm_names = {s: [f"{s}_joint{i}" for i in range(1, 7)] for s in rig.sides}
    rig.grip_names = {s: [f"{s}_joint7", f"{s}_joint8"] for s in rig.sides}
    rig.kin = {s: _FakeKinematics() for s in rig.sides}
    rig.targets = {}
    rig.apply = lambda: None
    return rig


@pytest.mark.parametrize("zero_arms", [False, True])
def test_rig_initialize_zero_arms_skips_ik_and_keeps_the_gripper(zero_arms):
    rig = _fake_rig()
    rotation = np.eye(3)
    rig.initialize(
        {"left": (np.array([0.3, 0.3, 1.0]), rotation, 0.04), "right": (np.zeros(3), rotation, 0.0)},
        zero_arms=zero_arms,
    )
    for side, grip in (("left", 0.04), ("right", 0.0)):
        arm = rig.robot.q[rig.arm_ids[side]]
        np.testing.assert_array_equal(arm, 0.0 if zero_arms else 1.0)
        np.testing.assert_array_equal(rig.robot.q[rig.grip_ids[side]], grip)
        np.testing.assert_array_equal(rig.robot.default, rig.robot.q)
        assert (side in rig.targets) is not zero_arms
        assert rig.initial_ik[side] == (None if zero_arms else 0.001)
