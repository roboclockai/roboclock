from dataclasses import replace

import numpy as np
import pytest
from helpers import snapshot

from roboclock.envs.lab.safety_audio import samples
from roboclock.envs.lab.tasks.registry import make_task

IDS = ["B5", "B6", "B7"]


def measured(task, *, secured=False, contact=False, grasp=False, opened=True):
    state = snapshot(task)
    obj = replace(state.objects["sample_0"], robot_contact=contact, grasped=grasp)
    return replace(
        state,
        objects={"sample_0": obj},
        controls={"secured": secured, "opening_reached": opened},
    )


def start_alarm(task):
    task.advance(0.1)
    if task.task_id == "B5":
        task.observe(measured(task), 0.001)


@pytest.mark.parametrize("task_id,window", zip(IDS, [7.28, 1.3, 1.3]))
def test_default_response_windows(task_id, window):
    assert make_task(task_id).spec.max_latency == window


# Oracle-calibrated tiers (configs/benchmark.yaml reactive_tasks, 2026-11-16), with every
# limit at least 1.0 s since 2026-11-17.
@pytest.mark.parametrize(
    "task_id,windows",
    zip(IDS, [(7.28, 4.07, 2.28), (1.3, 1.14, 1.0), (1.3, 1.14, 1.0)]),
)
def test_tiered_response_windows(task_id, windows):
    for difficulty, window in zip(("easy", "medium", "hard"), windows):
        assert make_task(task_id, difficulty).spec.max_latency == window


@pytest.mark.parametrize("task_id", IDS)
def test_touch_alone_cannot_complete(task_factory, task_id):
    task = task_factory(task_id, event_time=0.1)
    start_alarm(task)
    task.observe(measured(task, contact=True, grasp=True), 0.1)
    assert task.action_time is None
    assert not task.physical_success


@pytest.mark.parametrize("task_id", IDS)
def test_completed_contact_stroke_and_retention(task_factory, task_id):
    task = task_factory(task_id, event_time=0.1)
    start_alarm(task)
    task.observe(measured(task, contact=True, grasp=True), 0.01)
    task.advance(0.05)
    task.observe(measured(task, secured=True, contact=True, grasp=True), 0.05)
    assert task.action_time == pytest.approx(0.15)
    assert not task.goal_achieved
    for _ in range(20):
        task.advance(0.01)
        task.observe(measured(task, secured=True), 0.01)
    assert task.joint_success


@pytest.mark.parametrize("task_id", IDS)
def test_exact_deadline_is_terminal_failure_even_with_completed_stroke(task_factory, task_id):
    task = task_factory(task_id, event_time=0.1, max_latency=1.0)
    start_alarm(task)
    task.advance(1.0)
    task.observe(measured(task, secured=True, contact=True, grasp=True), 0.01)
    assert task.failed and task.terminated and not task.joint_success
    assert task.failure_kind in task.physical_failures
    assert task.action_time is None


@pytest.mark.parametrize("task_id", IDS)
def test_early_contact_cannot_be_erased(task_factory, task_id):
    task = task_factory(task_id, event_time=0.5)
    task.observe(measured(task, contact=True), 0.01)
    task.advance(0.5)
    for _ in range(25):
        task.advance(0.01)
        task.observe(measured(task, secured=True, contact=True, grasp=True), 0.01)
    assert "pre_event_action" in task.violations
    assert not task.joint_success


def test_freezer_requires_bilateral_grasp(task_factory):
    task = task_factory("B5", event_time=0.1)
    start_alarm(task)
    task.observe(measured(task, secured=True, contact=True), 0.3)
    assert task.action_time is None


def test_reopening_during_retention_fails(task_factory):
    task = task_factory("B6", event_time=0.1)
    start_alarm(task)
    task.observe(measured(task, secured=True, contact=True), 0.01)
    task.advance(0.01)
    task.observe(measured(task), 0.01)
    assert task.failed and "intervention_reversed" in task.physical_failures


def test_waste_flow_stops_only_when_clamped(task_factory):
    task = task_factory("B7", event_time=0.1, max_latency=2.0)
    start_alarm(task)
    task.observe(measured(task), 0.1)
    assert task.volume_ml == pytest.approx(9990.5)
    task.observe(measured(task, secured=True, contact=True), 0.1)
    assert task.volume_ml == pytest.approx(9990.5)
    assert task.visual_state()["flow_ml_s"] == 0.0


@pytest.mark.parametrize("task_id", IDS)
def test_audio_alignment_and_failure_tone(task_id):
    t = np.arange(0, 4, 1 / 24000)
    success = samples(t, task_id, 1.0, action_time=2.0)
    failure = samples(t, task_id, 1.0, failure_time=2.0)
    assert np.max(np.abs(success[t < 1])) == 0
    assert np.max(np.abs(success[(t > 1) & (t < 1.1)])) > 0.1
    assert np.max(np.abs(success[t > 2.5])) < 1e-6
    assert np.max(np.abs(failure[t > 3])) > 0.2


def test_alarm_requires_measured_opening_after_internal_catch_release(task_factory):
    task = task_factory("B5", event_time=0.1)
    initial = measured(task, opened=False)
    task.observe(initial, 0.01)
    commands = task.advance(0.1)
    assert any(c.kind == "safety_fault_release" for c in commands)
    task.observe(initial, 0.01)
    assert task.event_time is None
    task.advance(0.05)
    task.observe(measured(task, opened=False), 0.01)
    assert task.event_time is None
    task.advance(0.05)
    task.observe(measured(task, opened=True), 0.01)
    assert task.event_time == pytest.approx(0.2)
    assert task.action_time is None


@pytest.mark.parametrize("task_id", IDS)
def test_early_reaching_without_contact_is_allowed(task_factory, task_id):
    task = task_factory(task_id, event_time=2.0)
    initial = measured(task, opened=False)
    task.observe(initial, 0.01)
    task.observe(replace(initial, ee_position=initial.ee_position + [0.10, 0, 0]), 0.01)
    assert not task.violations
    assert task.pre_event_max_ee_drift == pytest.approx(0.10)
    assert not task.contact_seen


@pytest.mark.parametrize("task_id", IDS)
def test_oracle_holds_original_joint_command_before_event(task_id):
    from types import SimpleNamespace

    from roboclock.envs.lab.policies.oracle import OraclePickPlace

    policy = OraclePickPlace.__new__(OraclePickPlace)
    policy.env = SimpleNamespace(
        task=SimpleNamespace(task_id=task_id, safety_task=True, event_time=None)
    )
    policy.planner = SimpleNamespace()
    from roboclock.envs.lab.robot import HOME

    initial = np.tile(HOME, 2)
    policy.reset({"joint_positions": initial.copy()})
    for drift in (0, 0.001, -0.002):
        assert np.array_equal(policy.act({"joint_positions": initial + drift}), initial)
