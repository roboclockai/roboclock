"""Embodiment, grasp-frame, and full-cycle recording regressions."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.robot import ASSET_DIR, HOME, JOINT_HIGH, JOINT_LOW, tool_pose
from roboclock.envs.lab.tasks.registry import TASKS, make_task


def test_description_has_six_arm_and_two_physical_finger_joints():
    root = ET.parse(ASSET_DIR / "arx_x5.urdf").getroot()
    joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
    assert [j.get("type") for j in joints] == ["revolute"] * 6 + ["prismatic"] * 2
    for i, joint in enumerate(joints):
        assert float(joint.find("limit").get("lower")) == pytest.approx(JOINT_LOW[i])
        assert float(joint.find("limit").get("upper")) == pytest.approx(JOINT_HIGH[i])
    assert np.all(HOME >= JOINT_LOW) and np.all(HOME <= JOINT_HIGH)


def test_tool_frame_matches_urdf_with_rotated_mount():
    from roboclock.physics.geometry import rotation_matrix

    quat = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    robot = SimpleNamespace(
        end_effector=SimpleNamespace(
            get_world_pose=lambda: (np.array([1, 2, 3]), quat),
        )
    )
    position, orientation = tool_pose(robot)
    np.testing.assert_allclose(position, [1, 2.145, 3], atol=1e-8)
    np.testing.assert_allclose(rotation_matrix(orientation)[:, 2], [0, 1, 0], atol=1e-8)


@pytest.mark.parametrize("task_id", TASKS)
def test_every_task_has_the_same_dual_arm_interface(task_id, monkeypatch):
    from test_environment import InstrumentedBackend

    from roboclock.envs.lab.env import LabEnv

    monkeypatch.setattr("roboclock.envs.lab.backend.IsaacBackend", InstrumentedBackend)
    env = LabEnv(task_id, runtime=object())
    observation, _ = env.reset(seed=0)
    assert env.action_space.shape == (16,)
    assert env.observation_space.contains(observation)
    assert env.episode_metadata()["robot"] == "dual_arx_x5"
    env.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_a1_empty_to_full_duration_is_budgeted(difficulty):
    from roboclock.envs.lab.layout import make_layout

    task = make_task("A1", difficulty=difficulty, config=load_config())
    task.bind(make_layout(task, load_config()))
    assert task.volume == 0
    assert task.opening == 0
    assert task.spec.max_time > 500 / task.nominal_flow
    assert task.metrics()["initial_volume_ml"] == 0


@pytest.mark.parametrize("task_id", ["C1", "C2", "C3"])
def test_bimanual_queues_partition_samples_without_changing_task(task_id, task_factory):
    from roboclock.envs.lab.policies.bimanual import ArmTask, BimanualThroughput

    task = task_factory(task_id)
    active = task.active.copy()
    policy = BimanualThroughput(SimpleNamespace(task=task, config=load_config()), 1 / 30)
    left, right = policy.names
    assert not left.intersection(right)
    assert left.union(right) == set(task.layout.objects)
    assert task.active == active
    view = ArmTask(task, left, set())
    assert view.active == active.intersection(left)


def test_placement_reservations_do_not_modify_scoring_targets(task_factory):
    from helpers import snapshot

    from roboclock.envs.lab.policies.bimanual import ArmTask

    task = task_factory("C1")
    state = snapshot(task)
    targets = task.free_targets("sample_0", state)
    reserved = targets[0]
    view = ArmTask(task, {"sample_0"}, {reserved})
    assert reserved not in view.free_targets("sample_0", state)
    assert reserved in task.free_targets("sample_0", state)


def test_parallel_selection_keeps_neighboring_columns_exclusive(task_factory):
    from helpers import snapshot

    from roboclock.envs.lab.policies.bimanual import BimanualThroughput
    from roboclock.envs.lab.policies.trajectories import CartesianCommand

    task = task_factory("C2")
    state = snapshot(task)

    class Planner:
        def __init__(self, sample=None):
            self.sample = sample
            self.phase = "select" if sample is None else "approach"

        def command(self, view, observation):
            if self.phase == "select" and view.active:
                self.sample = min(view.active)
                self.phase = "approach"
            return CartesianCommand(observation.ee_position, False)

    robots = [SimpleNamespace(get_joint_positions=lambda: HOME.copy()) for _ in range(2)]
    backend = SimpleNamespace(
        robot=robots[0],
        secondary_robot=robots[1],
        state=lambda: state,
        grasp_owners={},
        # Each tool in front of its own base on the side-by-side rig.
        _tool_pose=lambda robot: (
            np.array([0.3, -0.3 if robot is robots[0] else 0.3, 1.0]),
            state.ee_orientation,
        ),
    )
    policy = BimanualThroughput(
        SimpleNamespace(task=task, config=load_config(), backend=backend), 1 / 30
    )
    policy.planners = [Planner("sample_3"), Planner()]
    policy.commands()
    selected = policy.planners[1].sample
    if selected is not None:
        # Columns run across the rig (mount-to-mount axis, world y for yaw 0).
        assert (
            abs(
                task.layout.objects[selected].position[1]
                - task.layout.objects["sample_3"].position[1]
            )
            >= 0.18
        )
    assert policy.planners[0].sample == "sample_3"
    task.correct.add("sample_3")
    policy.commands()
    assert policy.planners[1].sample is not None


def test_collision_onsets_count_both_arms_once_and_include_secondary_clutter():
    from roboclock.envs.lab.backend import IsaacBackend

    left = np.zeros((6, 1, 3))
    right = np.zeros((6, 1, 3))
    between = np.zeros((8, 8, 3))
    backend = object.__new__(IsaacBackend)
    backend.config = load_config()
    backend.task = SimpleNamespace(liquid_process=False)
    backend.world = SimpleNamespace(step=lambda **kwargs: None)
    backend.arm_contacts = SimpleNamespace(get_contact_force_matrix=lambda **kwargs: left)
    backend.secondary_contacts = SimpleNamespace(get_contact_force_matrix=lambda **kwargs: right)
    backend.inter_arm_contacts = SimpleNamespace(get_contact_force_matrix=lambda **kwargs: between)
    backend.secondary_robot = object()
    backend.clutter = {"box": object()}
    backend.secondary_clutter_filter_indices = [0]
    backend.last_collision = backend._last_clutter_contact = False
    backend.collision_count = backend.clutter_robot_contact_count = 0
    backend._measure_state = lambda elapsed: None

    left[0, 0, 0] = right[0, 0, 0] = between[0, 0, 0] = 1.0
    backend.tick()
    backend.tick()
    assert backend.collision_count == backend.clutter_robot_contact_count == 1
    left[:] = right[:] = between[:] = 0
    backend.tick()
    right[0, 0, 0] = 1.0
    backend.tick()
    assert backend.collision_count == backend.clutter_robot_contact_count == 2
