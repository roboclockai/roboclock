from dataclasses import replace

import numpy as np
import pytest
from helpers import approach_targets, at_target, observe_for, snapshot

from roboclock.config import load_config
from roboclock.envs.lab.layout import make_layout
from roboclock.envs.lab.mdp.events import window_error
from roboclock.envs.lab.mdp.state import ObjectState, orientation_error
from roboclock.envs.lab.tasks.registry import TASKS, make_task
from roboclock.evaluation.evaluator import episode_plan


def test_throughput_episode_level_must_be_a_difficulty(tmp_path):
    import yaml

    config = load_config()
    assert all("required_rate" not in level for level in config["difficulty"].values())
    config["throughput_tasks"]["episode_level"] = "extreme"
    path = tmp_path / "bad_level.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="episode_level"):
        load_config(path)


def test_upright_rack_insertion_ignores_cylinder_yaw(legacy_task_factory):
    task = legacy_task_factory("A2", valid_start=1.0, valid_end=3.0)
    task.time = 2.0
    obj = at_target(task)
    obj = replace(obj, orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    state = snapshot(task, sample_0=obj)
    approach_targets(task, state)
    observe_for(task, state)
    assert task.joint_success


def test_suite_size_and_seed_reproducibility():
    assert len(episode_plan()) == len(TASKS) * 3 * 10
    config = load_config()
    for task_id in TASKS:
        a, b = make_task(task_id, seed=42), make_task(task_id, seed=42)
        assert a.spec == b.spec
        x, y = make_layout(a, config), make_layout(b, config)
        for name in x.objects:
            np.testing.assert_array_equal(x.objects[name].position, y.objects[name].position)
        c = make_task(task_id, seed=43)
        assert a.spec != c.spec


@pytest.mark.parametrize("task_id", TASKS)
def test_physical_setup_constant_across_difficulties(task_id):
    config = load_config()
    easy, hard = make_task(task_id, "easy", 11), make_task(task_id, "hard", 11)
    x, y = make_layout(easy, config), make_layout(hard, config)
    for name in x.objects:
        np.testing.assert_array_equal(x.objects[name].position, y.objects[name].position)
    if task_id.startswith("C"):
        # Batch C tasks: one episode for every difficulty (docs/c-class-timing.md).
        assert replace(easy.spec, difficulty="hard") == hard.spec
    else:
        assert easy.spec.max_latency > hard.spec.max_latency


@pytest.mark.parametrize("time,error", [(2.0, 1.0), (3.0, 0.0), (5.0, 0.0), (6.0, 1.0)])
def test_window_includes_boundaries(time, error):
    assert window_error(time, 3.0, 5.0) == error


@pytest.mark.parametrize("time,valid", [(2.0, False), (3.0, True), (5.0, True), (6.0, False)])
def test_incubation_scores_insertion_time_not_settling_time(legacy_task_factory, time, valid):
    task = legacy_task_factory("A2", valid_start=3.0, valid_end=5.0)
    task.time = time
    state = snapshot(task, sample_0=at_target(task))
    approach_targets(task, state)
    task.observe(state, 0.01)
    approach_targets(task, state)
    observe_for(task, state)
    assert task.physical_success
    assert task.temporal_success == valid
    assert task.action_time == time


def test_early_insertion_cannot_wait_in_slot_to_erase_violation(legacy_task_factory):
    task = legacy_task_factory("A2", valid_start=3.0, valid_end=5.0)
    task.time = 2.0
    state = snapshot(task, sample_0=at_target(task, grasped=True))
    approach_targets(task, state)
    task.observe(state, 0.01)
    task.time = 4.0
    state = snapshot(task, sample_0=at_target(task))
    approach_targets(task, state)
    observe_for(task, state)
    assert task.physical_success
    assert not task.temporal_success
    assert "early_action" in task.violations


def test_temperature_checks_grasp_then_destination(legacy_task_factory):
    task = legacy_task_factory("A1")
    task.temperature = 60.0
    source = task.layout.objects["sample_0"]
    task.observe(
        snapshot(task, sample_0=ObjectState(source.position, source.orientation, grasped=True)),
        0.01,
    )
    assert task.temporal_success
    assert not task.physical_success
    state = snapshot(task, sample_0=at_target(task))
    approach_targets(task, state)
    observe_for(task, state)
    assert task.joint_success


def test_temperature_early_grasp_stays_invalid(legacy_task_factory):
    task = legacy_task_factory("A1")
    task.temperature = 40.0
    source = task.layout.objects["sample_0"]
    task.observe(
        snapshot(task, sample_0=ObjectState(source.position, source.orientation, grasped=True)),
        0.01,
    )
    task.temperature = 60.0
    state = snapshot(task, sample_0=at_target(task))
    approach_targets(task, state)
    observe_for(task, state)
    assert task.physical_success and not task.temporal_success


def test_exactly_one_catalyst(legacy_task_factory):
    task = legacy_task_factory("A3", valid_start=1.0, valid_end=3.0)
    task.time = 2.0
    state = snapshot(task, sample_0=at_target(task), sample_1=at_target(task, "sample_1"))
    approach_targets(task, state)
    observe_for(task, state)
    assert not task.physical_success
    assert "multiple_pellets" in task.violations


def test_catalyst_success(legacy_task_factory):
    task = legacy_task_factory("A3", valid_start=1.0, valid_end=3.0)
    task.time = 2.0
    state = snapshot(task, sample_0=at_target(task))
    approach_targets(task, state)
    observe_for(task, state)
    assert task.joint_success


@pytest.mark.parametrize("latency,valid", [(0.49, True), (0.5, False), (0.51, False)])
def test_alarm_strict_reaction_deadline(legacy_task_factory, latency, valid):
    task = legacy_task_factory("B1", max_latency=0.5, transfer_deadline=3.0)
    task.event_time = 1.0
    task.time = 1.0 + latency - 0.05
    source = task.layout.objects["sample_0"]
    observe_for(
        task,
        snapshot(
            task,
            sample_0=ObjectState(source.position + [0, 0, 0.04], source.orientation, grasped=True),
        ),
        duration=0.05,
    )
    state = snapshot(task, sample_0=at_target(task))
    approach_targets(task, state)
    observe_for(task, state)
    assert task.physical_success
    assert task.temporal_success == valid


def test_alarm_rejects_pre_event_grasp(legacy_task_factory):
    task = legacy_task_factory("B1")
    source = task.layout.objects["sample_0"]
    task.observe(
        snapshot(task, sample_0=ObjectState(source.position, source.orientation, grasped=True)),
        0.01,
    )
    assert "pre_event_action" in task.violations
    assert not task.temporal_success


def test_alarm_transfer_has_separate_completion_deadline(legacy_task_factory):
    task = legacy_task_factory("B1", max_latency=1.0, transfer_deadline=2.0)
    task.event_time, task.time = 1.0, 1.1
    source = task.layout.objects["sample_0"]
    observe_for(
        task,
        snapshot(
            task,
            sample_0=ObjectState(source.position + [0, 0, 0.04], source.orientation, grasped=True),
        ),
        duration=0.05,
    )
    task.time = 4.0
    state = snapshot(task, sample_0=at_target(task))
    approach_targets(task, state)
    observe_for(task, state)
    assert task.physical_success and not task.temporal_success
    assert "transfer_deadline" in task.violations


@pytest.mark.parametrize("task_id", ["B2", "B3"])
def test_physical_release_emitted_once(legacy_task_factory, task_id):
    task = legacy_task_factory(task_id, event_time=0.02)
    assert not any(c.kind == "release" for c in task.advance(0.01))
    assert sum(c.kind == "release" for c in task.advance(0.01)) == 1
    assert not any(c.kind == "release" for c in task.advance(0.01))
    assert task.event_time == 0.02


def test_slip_can_be_blocked_with_contact(legacy_task_factory):
    task = legacy_task_factory("B2", max_latency=2.0)
    task.event_time = 1.0
    task.time = 1.1
    source = task.layout.objects["sample_0"]
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
                source.position, source.orientation, robot_contact=True, table_contact=True
            ),
        ),
        duration=0.35,
    )
    assert task.joint_success


def test_falling_sample_cannot_be_picked_up_after_impact(legacy_task_factory):
    task = legacy_task_factory("B3")
    task.event_time = 1.0
    task.time = 1.1
    landed = task.layout.objects["sample_0"].position.copy()
    landed[2] = task.layout.table_height + 0.025
    task.observe(
        snapshot(
            task, sample_0=ObjectState(landed, np.array([1.0, 0.0, 0.0, 0.0]), table_contact=True)
        ),
        0.01,
    )
    observe_for(task, snapshot(task, sample_0=at_target(task, grasped=True)))
    assert task.hit_table and task.terminated
    assert not task.physical_success


def test_probe_scores_periodic_alignment_not_one_off_window(legacy_task_factory):
    task = legacy_task_factory("A4")
    task.fixture_offset = 0.0
    task.time = task.spec.valid_end + 30.0
    obj = at_target(task, grasped=True)
    state = snapshot(task, sample_0=obj)
    approach_targets(task, state)
    observe_for(task, state)
    assert task.joint_success


def test_probe_wrong_alignment_stays_temporally_invalid(legacy_task_factory):
    task = legacy_task_factory("A4")
    task.fixture_offset = 0.04
    task.omega = 1.0
    obj = at_target(task, grasped=True)
    shifted = replace(obj, position=obj.position + np.array([0.04, 0.0, 0.0]))
    state = snapshot(task, sample_0=shifted)
    approach_targets(task, state)
    observe_for(task, state)
    assert task.physical_success and not task.temporal_success


def test_throughput_counts_each_sample_once(legacy_task_factory):
    task = legacy_task_factory("C2")
    state = snapshot(task, **{name: at_target(task, name) for name in task.layout.objects})
    approach_targets(task, state)
    observe_for(task, state, 1.0)
    count = len(task.correct)
    assert count == len(task.layout.objects)
    times = dict(task.job_times)
    assert set(times) == set(task.layout.objects)
    approach_targets(task, state)
    observe_for(task, state, 0.5)
    assert len(task.correct) == count and task.job_times == times
    assert task.metrics()["operations_per_minute"] == pytest.approx(60 * count / task.time)


def test_final_throughput_goal_requires_samples_still_in_targets(legacy_task_factory):
    task = legacy_task_factory("C2")
    state = snapshot(task, **{name: at_target(task, name) for name in task.layout.objects})
    approach_targets(task, state)
    observe_for(task, state)
    assert task.physical_success
    task.observe(snapshot(task), 0.01)
    assert not task.physical_success


def test_continuous_arrivals_and_drop_accounting(legacy_task_factory):
    task = legacy_task_factory("C1")
    assert task.active == {"sample_0"}
    commands = task.advance(2.0)
    assert any(c.kind == "activate" and c.name == "sample_1" for c in commands)
    fallen = ObjectState(np.array([0.4, 0.0, 0.1]), np.array([1.0, 0.0, 0.0, 0.0]))
    task.observe(snapshot(task, sample_1=fallen), 0.01)
    assert task.dropped == {"sample_1"}
    assert task.metrics()["attempted_operations"] == 1


def test_quaternion_sign_and_angle():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert orientation_error(q, -q) == 0.0
    assert orientation_error(q, np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(np.pi)
