from dataclasses import replace

import numpy as np
import pytest
from helpers import approach_targets, at_target, observe_for, snapshot

from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.layout import make_layout
from roboclock.envs.lab.mdp.geometry import safe_workspace
from roboclock.envs.lab.mdp.state import ObjectState
from roboclock.envs.lab.tasks.registry import make_task


def test_c1_uses_tubes_and_individual_slots(task_factory):
    task = task_factory("C1")
    assert task.name == "rack_organization"
    assert task.asset == "test_tube"
    assert task.active == set(task.layout.objects)
    assert len(set(task.layout.assignments.values())) == task.spec.num_samples
    assert all(t.kind == "slot" for t in task.layout.targets.values())
    assert task.advance(2.0) == []


@pytest.mark.parametrize("count", [20, 25, 36])
def test_rack_slots_fit_table_and_do_not_overlap_sources(count):
    config = load_config()
    task = make_task("C1")
    task.spec = replace(task.spec, num_samples=count)
    task.bind(make_layout(task, config))
    positions = []
    for name, target in task.layout.targets.items():
        sample = "sample_0"
        obj = ObjectState(target.position, target.orientation)
        assert safe_workspace(task, sample, obj)
        assert all(np.linalg.norm(target.position[:2] - p[:2]) > 0.04 for p in positions)
        positions.append(target.position)
    for obj in task.layout.objects.values():  # source racks are separate from color racks
        assert min(np.linalg.norm(obj.position[:2] - p[:2]) for p in positions) > 0.045
        assert abs(obj.position[1]) < 0.55 - 0.03


def test_c1_requires_downward_insertion_contact_and_release(task_factory):
    task = task_factory("C1")
    placed = at_target(task)
    observe_for(task, snapshot(task, sample_0=placed))
    assert not task.correct  # Teleporting into a slot is not a scored insertion.
    state = snapshot(task, sample_0=replace(placed, grasped=True))
    approach_targets(task, state)
    observe_for(task, state)
    assert not task.correct
    observe_for(task, snapshot(task, sample_0=replace(placed, fixture_contacts=frozenset())))
    assert not task.correct
    observe_for(task, snapshot(task, sample_0=placed))
    assert task.correct == {"sample_0"}
    task.time = 120
    assert task.metrics()["operations_per_minute"] == pytest.approx(0.5)
    assert not task.terminated


def test_c1_rejects_wrong_slot_and_tipped_tube(task_factory):
    task = task_factory("C1")
    wrong = at_target(task, "sample_1")
    state = snapshot(task, sample_0=wrong)
    approach_targets(task, state)
    observe_for(task, state)
    assert not task.correct
    tipped = replace(at_target(task), orientation=np.array([0.70710678, 0.70710678, 0, 0]))
    state = snapshot(task, sample_0=tipped)
    approach_targets(task, state)
    observe_for(task, state)
    assert not task.correct


def test_c1_diagnostic_stops_retrying_failed_sample(task_factory):
    from roboclock.envs.lab.policies.rack_organization import RackOrganizationTrajectory

    task = task_factory("C1")
    planner = RackOrganizationTrajectory(load_config()["control"], 1 / 30)
    planner.sample = "sample_0"
    planner.phase = "approach"
    planner.retries["sample_0"] = 3
    planner.command(task, snapshot(task))
    assert planner.sample != "sample_0"
    assert planner.phase == "approach"


def test_color_racks_reject_more_than_36_tubes():
    config = load_config()
    task = make_task("C1", seed=43)
    task.spec = replace(task.spec, num_samples=50)
    with pytest.raises(ValueError, match="at most 36"):
        make_layout(task, config)


def test_c1_has_three_color_racks_and_mixed_sources(task_factory):
    from roboclock.envs.lab.layout import COLORS
    from roboclock.envs.lab.rack_organization import rack_center

    task = task_factory("C1")
    assert len(task.layout.targets) == 36
    source_colors = [set(), set(), set()]
    for name, obj in task.layout.objects.items():
        color = int(name.split("_")[-1]) % 3
        np.testing.assert_array_equal(obj.color, COLORS[color])
        target = int(task.layout.assignments[name].split("_")[-1])
        assert target // 12 == color
        rack = min(
            range(3),
            key=lambda i: np.linalg.norm(obj.position[:2] - rack_center(i, 0, source=True)[:2]),
        )
        source_colors[rack].add(color)
    # Red tubes start on the left arm's side, green on the right, blue on either;
    # every source rack still holds mixed colors.
    assert 2 not in source_colors[0] | source_colors[1]
    assert 0 not in source_colors[2]
    assert all(len(colors) >= 2 for colors in source_colors)


def test_c1_every_tube_has_an_arm_for_source_and_rack():
    from roboclock.envs.lab.policies.rack_organization import RackOrganizationTrajectory
    from roboclock.envs.lab.robot import mount_positions

    config = load_config()
    mounts = [np.asarray(m) for m in mount_positions(config, "C1")]
    for seed in range(10):
        task = make_task("C1", seed=seed, config=config)
        task.bind(make_layout(task, config))
        planners = [
            RackOrganizationTrajectory(
                config["control"], 0.02, base=m, keep_out=(mounts[1 - k][:2], 0.25)
            )
            for k, m in enumerate(mounts)
        ]
        for name in task.layout.objects:
            assert any(p.can_handle(task, name) for p in planners), (seed, name)


def test_c1_accepts_spare_same_color_slot(task_factory):
    task = task_factory("C1")
    key = "target_11"  # Empty red slot, not sample_0's suggested destination.
    target = task.layout.targets[key]
    obj = ObjectState(
        target.position.copy(), target.orientation.copy(), fixture_contacts=frozenset({key})
    )
    above = replace(obj, position=obj.position + [0, 0, 0.12], fixture_contacts=frozenset())
    task.previous = snapshot(task, sample_0=above)
    observe_for(task, snapshot(task, sample_0=obj))
    assert task.correct == {"sample_0"}
    assert task.layout.assignments["sample_0"] == key
    assert task.metrics()["correct_by_color"] == {"red": 1, "blue": 0, "green": 0}


def test_c1_does_not_count_two_tubes_in_one_slot(task_factory):
    task = task_factory("C1")
    obj = at_target(task)
    above = replace(obj, position=obj.position + [0, 0, 0.12], fixture_contacts=frozenset())
    task.previous = snapshot(task, sample_0=above, sample_3=above)
    observe_for(task, snapshot(task, sample_0=obj, sample_3=obj))
    assert len(task.correct) == 1


def test_c1_planner_avoids_occupied_color_slot(task_factory):
    task = task_factory("C1")
    obj = at_target(task)
    assert "target_0" not in task.free_targets("sample_3", snapshot(task, sample_0=obj))
    assert "target_1" in task.free_targets("sample_3", snapshot(task, sample_0=obj))


def test_expert_waits_for_actual_grasp_before_lifting(task_factory):
    from roboclock.envs.lab.policies.rack_organization import RackOrganizationTrajectory

    task = task_factory("C1")
    planner = RackOrganizationTrajectory(load_config()["control"], 1 / 30)
    planner.sample, planner.phase = "sample_0", "grasp"
    state = snapshot(task)
    assert planner.command(task, state).close
    assert planner.phase == "grasp"
    state.objects["sample_0"] = replace(state.objects["sample_0"], grasped=True)
    planner.command(task, state)
    assert planner.phase == "lift"


@pytest.mark.parametrize(
    "offset,release", [([0, 0, 0.02], False), ([0.01, 0, 0], False), ([0, 0, 0], True)]
)
def test_expert_releases_only_after_measured_alignment(task_factory, offset, release):
    from roboclock.envs.lab.policies.rack_organization import RackOrganizationTrajectory

    task = task_factory("C1")
    planner = RackOrganizationTrajectory(load_config()["control"], 1 / 30)
    planner.sample, planner.phase = "sample_0", "insert"
    task.insertion_times["sample_0"] = 0.1
    obj = at_target(task, grasped=True)
    obj = replace(obj, position=obj.position + offset)
    command = planner.command(task, snapshot(task, sample_0=obj))
    assert (planner.phase == "release") == release
    assert command.close != release


def test_expert_reinserts_if_downward_crossing_was_not_observed(task_factory):
    from roboclock.envs.lab.policies.rack_organization import RackOrganizationTrajectory

    task = task_factory("C1")
    planner = RackOrganizationTrajectory(load_config()["control"], 1 / 30)
    planner.sample, planner.phase = "sample_0", "insert"
    command = planner.command(task, snapshot(task, sample_0=at_target(task, grasped=True)))
    assert command.close
    assert planner.phase == "lift"


def test_expert_can_finish_seating_after_valid_crossing(task_factory):
    from roboclock.envs.lab.policies.rack_organization import RackOrganizationTrajectory

    task = task_factory("C1")
    planner = RackOrganizationTrajectory(load_config()["control"], 1 / 30)
    planner.sample, planner.phase = "sample_0", "insert"
    task.insertion_times["sample_0"] = 0.1
    obj = at_target(task, grasped=True)
    obj = replace(obj, position=obj.position + [0.003, 0, 0.02])
    state = snapshot(task, sample_0=obj)
    command = planner.command(task, state)
    assert command.close
    assert command.position[2] < state.ee_position[2]


def test_demo_grasp_assist_requires_contact_and_releases_on_open(task_factory):
    from types import SimpleNamespace

    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from roboclock.envs.lab.rack_organization import update_expert_grasp

    task = task_factory("C1")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/ARXLeft/link6")
    UsdGeom.Xform.Define(stage, "/World/Objects/sample_0")
    backend = SimpleNamespace(
        stage=stage,
        task=task,
        robot=SimpleNamespace(
            end_effector=SimpleNamespace(
                get_world_pose=lambda: (np.zeros(3), np.array([1.0, 0, 0, 0]))
            )
        ),
    )
    state = snapshot(task)
    update_expert_grasp(backend, "sample_0", True, state)
    assert not stage.GetPrimAtPath("/World/C1ExpertGrasp").IsValid()
    state.objects["sample_0"] = replace(state.objects["sample_0"], grasped=True)
    update_expert_grasp(backend, "sample_0", True, state)
    assert stage.GetPrimAtPath("/World/C1ExpertGrasp").IsValid()
    update_expert_grasp(backend, "sample_0", False, state)
    assert not stage.GetPrimAtPath("/World/C1ExpertGrasp").IsValid()
    assert [e["type"] for e in task.event_log[-2:]] == [
        "expert_grasp_assist_enabled",
        "expert_grasp_assist_released",
    ]
