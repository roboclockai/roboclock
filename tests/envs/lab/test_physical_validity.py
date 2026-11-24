from dataclasses import replace

import numpy as np
import pytest
from helpers import approach_targets, at_target, observe_for, snapshot

from roboclock.envs.lab.mdp.geometry import fits_xy, safe_workspace
from roboclock.envs.lab.mdp.state import ObjectState
from roboclock.envs.lab.policies.trajectories import LabTrajectory, limit_step
from roboclock.physics.contacts import GraspTracker


def test_hover_above_rack_does_not_count(legacy_task_factory):
    task = legacy_task_factory("A2", valid_start=1.0, valid_end=5.0)
    task.time = 2.0
    obj = at_target(task)
    obj = replace(obj, position=obj.position + np.array([0.0, 0.0, 0.03]))
    observe_for(task, snapshot(task, sample_0=obj), duration=0.5)
    assert not task.physical_success and task.action_time is None


def test_slot_requires_downward_entry_not_side_loading(legacy_task_factory):
    task = legacy_task_factory("A2")
    obj = at_target(task)
    task.previous = snapshot(
        task, sample_0=replace(obj, position=obj.position + np.array([0.15, 0.0, 0.0]))
    )
    observe_for(task, snapshot(task, sample_0=obj), duration=0.5)
    assert not task.physical_success and task.action_time is None


def test_center_in_slot_cannot_hide_body_wall_overlap(legacy_task_factory):
    task = legacy_task_factory("A2")
    obj = at_target(task)
    obj = replace(obj, position=obj.position + np.array([0.014, 0.0, 0.0]))
    assert task.layout.targets["destination"].contains(obj.position)
    assert not fits_xy(task, "sample_0", obj, task.layout.targets["destination"])
    state = snapshot(task, sample_0=obj)
    approach_targets(task, state)
    observe_for(task, state, duration=0.5)
    assert not task.physical_success


def test_seated_pose_requires_measured_fixture_contact(legacy_task_factory):
    task = legacy_task_factory("C3")
    state = snapshot(
        task, **{n: at_target(task, n, fixture_contacts=frozenset()) for n in task.active}
    )
    observe_for(task, state, duration=0.5)
    assert not task.correct


def test_grasp_requires_opposing_forces_and_continuous_retention():
    tracker = GraspTracker(minimum_time=0.04)
    origin = np.zeros(3)

    def measure(forces, dt=0.01, velocity=None):
        return tracker.update(
            "sample",
            np.array(forces),
            0.04,
            origin,
            origin,
            origin if velocity is None else velocity,
            origin,
            0.02,
            0.075,
            dt,
        )[0]

    assert not measure([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 0.1)
    assert not measure([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], 0.02)
    assert not measure([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 0.01)
    assert not measure([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], 0.02)
    assert measure([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], 0.02)
    assert not measure([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], velocity=np.array([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("moving_first,table_contact", [(False, True), (True, False)])
def test_slip_incidental_contact_or_airborne_touch_is_not_recovery(
    legacy_task_factory, moving_first, table_contact
):
    task = legacy_task_factory("B2")
    task.time, task.event_time = 2.0, 1.0
    source = task.layout.objects["sample_0"]
    if moving_first:
        task.observe(
            snapshot(
                task,
                sample_0=ObjectState(
                    source.position, source.orientation, velocity=np.array([0.0, -0.5, 0.0])
                ),
            ),
            0.01,
        )
    observe_for(
        task,
        snapshot(
            task,
            sample_0=ObjectState(
                source.position, source.orientation, robot_contact=True, table_contact=table_contact
            ),
        ),
        duration=0.5,
    )
    assert not task.physical_success


def test_recovery_requires_whole_sample_inside_bench(legacy_task_factory):
    task = legacy_task_factory("B2")
    task.time, task.event_time = 2.0, 1.0
    obj = at_target(task, grasped=True)
    position = obj.position.copy()
    position[0] = task.layout.workspace_bounds[1][0] - 0.005
    obj = replace(obj, position=position)
    assert not safe_workspace(task, "sample_0", obj)
    observe_for(task, snapshot(task, sample_0=obj), duration=0.5)
    assert task.terminated and not task.physical_success


def test_brief_catch_must_be_retained(legacy_task_factory):
    task = legacy_task_factory("B3")
    task.time, task.event_time = 2.0, 1.0
    source = task.layout.objects["sample_0"]
    obj = ObjectState(source.position, source.orientation, grasped=True, relative_tool_speed=0.0)
    observe_for(task, snapshot(task, sample_0=obj), duration=0.1)
    assert not task.physical_success
    observe_for(task, snapshot(task, sample_0=replace(obj, grasped=False)), duration=0.1)
    observe_for(task, snapshot(task, sample_0=obj), duration=0.2)
    assert not task.physical_success
    observe_for(task, snapshot(task, sample_0=obj), duration=0.1)
    assert task.physical_success


def test_fixture_impact_invalidates_catch_even_while_grasped(legacy_task_factory):
    task = legacy_task_factory("B3")
    task.time, task.event_time = 2.0, 1.0
    source = task.layout.objects["sample_0"]
    obj = ObjectState(
        source.position,
        source.orientation,
        grasped=True,
        relative_tool_speed=0.0,
        fixture_contacts=frozenset(["holder"]),
    )
    observe_for(task, snapshot(task, sample_0=obj), duration=0.4)
    assert task.hit_table and not task.physical_success


def test_cartesian_trajectory_caps_speed():
    result = limit_step(np.zeros(3), np.array([2.0, 1.0, 1.0]), 0.3, 1 / 30)
    assert np.linalg.norm(result) == pytest.approx(0.01)
    np.testing.assert_allclose(limit_step(result, result, 0.3, 1 / 30), result)


def test_rack_release_pose_keeps_fingers_above_lip(legacy_task_factory):
    task = legacy_task_factory("A2")
    planner = LabTrajectory(
        {
            "insertion_speed": 0.07,
            "max_cartesian_speed": 0.35,
            "reactive_speed": 0.65,
            "phase_timeout": 10.0,
        },
        1 / 30,
    )
    planner.sample, planner.phase = "sample_0", "insert"
    state = snapshot(task, sample_0=at_target(task, grasped=True))
    command = planner.command(task, state)
    assert command.position[2] >= task.layout.targets["destination"].rim_z
    assert command.close


def test_hold_uses_same_motion_interval_for_sample_and_tool(legacy_task_factory):
    from types import SimpleNamespace

    from roboclock.config import load_config
    from roboclock.envs.lab.backend import IsaacBackend

    task = legacy_task_factory("A2")
    backend = IsaacBackend(None, load_config())
    backend.layout = task.layout
    backend._previous_tool = None
    backend._previous_object_positions = {}
    backend.grasps = GraspTracker(minimum_time=0.04)
    pose = np.array([0.4, 0.0, 1.0])
    # Constraint-solver velocities can fluctuate although observed relative pose is constant.
    body = SimpleNamespace(
        get_world_pose=lambda: (pose.copy(), np.array([1.0, 0.0, 0.0, 0.0])),
        get_linear_velocity=lambda: np.array([0.0, 0.0, 1.0]),
    )
    backend.objects = {"sample_0": body}
    backend.contacts = SimpleNamespace(
        prim_paths=["/World/Objects/sample_0"],
        get_contact_force_matrix=lambda **kwargs: np.array(
            [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
    )
    backend.surface_paths = {}
    backend.robot = SimpleNamespace(
        get_joint_positions=lambda: np.array([0.0] * 6 + [0.02, 0.02]),
        get_joint_velocities=lambda: np.zeros(8),
    )
    backend._tool_pose = lambda robot=None: (pose.copy(), np.array([0.0, 0.0, 1.0, 0.0]))
    backend.secondary_robot = backend.robot
    backend.secondary_sample_contacts = SimpleNamespace(
        prim_paths=["/World/Objects/sample_0"],
        get_contact_force_matrix=lambda **kwargs: np.zeros((1, 3, 3)),
    )
    for _ in range(5):
        pose[0] += 0.001
        backend._cached_state = backend._measure_state(0.01)
    assert backend.state().objects["sample_0"].grasped
    assert backend.state().objects["sample_0"].relative_tool_speed == pytest.approx(0.0)
    duration = backend.grasps.durations["sample_0"]
    for _ in range(20):
        backend.state()
    assert backend.grasps.durations["sample_0"] == duration


def test_block_can_stop_applied_slip_within_one_physics_tick(legacy_task_factory):
    task = legacy_task_factory("B2")
    task.event_time, task.time = 1.0, 1.01
    source = task.layout.objects["sample_0"]
    obj = ObjectState(
        source.position,
        source.orientation,
        robot_contact=True,
        table_contact=True,
        release_speed=0.6,
    )
    task.observe(snapshot(task, sample_0=obj), 0.01)
    observe_for(task, snapshot(task, sample_0=replace(obj, release_speed=0.0)), duration=0.31)
    assert task.physical_success and task.temporal_success


def test_large_source_grid_clears_rack_fronts(legacy_task_factory):
    task = legacy_task_factory("C1", num_samples=50)  # superseded V1 bin sorting
    for obj in task.layout.objects.values():
        radius = task.layout.dimensions[obj.name][0]
        source_max_y = obj.position[1] + radius * 1.2
        front = min(t.position[1] - t.half_extents[1] - 0.006 for t in task.layout.targets.values())
        assert source_max_y < front - 0.002


@pytest.mark.parametrize("task_id", ["C2", "C3"])
def test_full_rig_grid_keeps_sources_in_front_of_targets(task_id, task_factory):
    # Sources sit between the rig and the targets (rig forward = +x).
    task = task_factory(task_id, num_samples=32)
    for obj in task.layout.objects.values():
        radius = task.layout.dimensions[obj.name][0]
        front = min(t.position[0] - t.half_extents[0] - 0.006 for t in task.layout.targets.values())
        assert obj.position[0] + radius * 1.2 < front - 0.002


def test_fifty_sample_sorting_has_distinct_supported_landing_positions(legacy_task_factory):
    from roboclock.envs.lab.policies.trajectories import sorting_landing

    task = legacy_task_factory("C1", num_samples=50)
    groups = {}
    for name in task.layout.objects:
        target_name = task.layout.assignments[name]
        target = task.layout.targets[target_name]
        position = sorting_landing(task, name, target)
        radius = task.layout.dimensions[name][0]
        assert np.all(
            np.abs(position[:2] - target.position[:2]) + radius < target.half_extents[:2] - 0.0005
        )
        for previous in groups.setdefault(target_name, []):
            assert np.linalg.norm(position[:2] - previous[:2]) >= 2 * radius + 0.001
        groups[target_name].append(position)


def test_full_rack_clears_indicator_housing(legacy_task_factory):
    from roboclock.envs.lab.layout import indicator_position

    task = legacy_task_factory("C2", num_samples=32)
    housing = indicator_position(task)
    for target in task.layout.targets.values():
        # Include the physical walls, not just the interior opening.
        limits = target.half_extents[:2] + np.array([0.004, 0.004]) + np.array([0.0325, 0.020])
        assert np.any(np.abs(target.position[:2] - housing[:2]) > limits + 0.001)
