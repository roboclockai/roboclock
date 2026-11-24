"""Inspection aiming, task priority and inspection motion limits of the active observer."""

from types import SimpleNamespace

import numpy as np

from roboclock.envs.lab.policies.active_observation import (
    ActiveObservation,
    inspection_orientation,
)
from roboclock.envs.lab.robot import HOME
from roboclock.physics.geometry import rotation_matrix


def test_inspection_aims_tool_toward_detail_for_all_roll_candidates():
    eye = np.array([0.45, 0.32, 1.2])
    detail = np.array([0.5, 0.1, 0.9])
    direction = (detail - eye) / np.linalg.norm(detail - eye)
    for roll in (0.0, np.pi / 2, -np.pi / 2, np.pi):
        orientation = inspection_orientation(eye, detail, roll)
        np.testing.assert_allclose(rotation_matrix(orientation)[:, 2], direction, atol=1e-12)


def observer_fixture():
    observer = object.__new__(ActiveObservation)
    observer.enabled = True
    observer.goal = HOME.copy()
    observer.commanded = HOME.copy()
    observer.goal[:6] += 0.1
    observer.env = SimpleNamespace(
        task=SimpleNamespace(task_id="B4", time=2.0, event_time=None),
        backend=SimpleNamespace(
            secondary_robot=SimpleNamespace(get_joint_positions=lambda: HOME.copy())
        ),
    )
    observer.policy = SimpleNamespace(dt=1 / 30)
    observer.speed = 1.2
    observer.started_at = observer.arrived_at = None
    return observer


def test_inspector_preserves_operating_arm_and_waits_for_reactive_event():
    observer = observer_fixture()
    action = np.tile(HOME, 2)
    np.testing.assert_array_equal(observer.apply(action), action)
    assert observer.started_at is None
    observer.env.task.event_time = 2.0
    observed = observer.apply(action)
    np.testing.assert_array_equal(observed[:8], action[:8])
    assert 0 < np.max(np.abs(observed[8:] - HOME)) <= 1.2 / 30 + 1e-12
    np.testing.assert_array_equal(action, np.tile(HOME, 2))


def test_cooperative_arm_commands_are_never_overridden():
    observer = observer_fixture()
    observer.enabled = False
    observer.env.task.event_time = 1.0
    action = np.linspace(0.0, 0.04, 16)
    np.testing.assert_array_equal(observer.apply(action), action)
    assert observer.started_at is None
