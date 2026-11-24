from dataclasses import replace

import numpy as np
import pytest
from helpers import snapshot

from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.tasks.registry import make_task


# Oracle-calibrated tiers (configs/benchmark.yaml reactive_tasks; easy = 4 t since 2026-11-23).
@pytest.mark.parametrize(
    "task_id,windows",
    [("B2", (6.02, 3.36, 1.88)), ("B3", (6.37, 3.56, 1.99)), ("B4", (6.76, 3.79, 2.12))],
)
def test_v2_reaction_windows(task_id, windows):
    for difficulty, seconds in zip(("easy", "medium", "hard"), windows):
        task = make_task(task_id, difficulty, config=load_config())
        assert task.laboratory_v2
        assert task.spec.max_latency == seconds


@pytest.mark.parametrize("task_id", ["B2", "B3"])
def test_contact_alone_does_not_cut_power(task_factory, task_id):
    task = task_factory(task_id, event_time=1, max_latency=2)
    task.time = task.event_time = 1
    state = snapshot(task)
    state.controls.update({"power_on": True, "switch_robot_contact": True, "switch_angle_rad": 0.1})
    task.observe(state, 0.1)
    assert task.action_time is None
    assert not task.physical_success


@pytest.mark.parametrize("latency,late", [(1.99, False), (2.0, True), (2.01, True)])
def test_disconnect_deadline_is_strict_and_latched(task_factory, latency, late):
    task = task_factory("B2", event_time=1, max_latency=2)
    task.event_time = 1
    task.time = 1 + latency
    state = snapshot(task)
    state.controls.update({"power_on": False, "switch_angle_rad": 0.7})
    task.observe(state, 0.6)
    assert task.physical_success
    assert task.temporal_success is not late
    assert ("late_reaction" in task.violations) is late


def test_premature_power_cut_is_invalid(task_factory):
    task = task_factory("B2")
    state = snapshot(task)
    state.controls.update({"power_on": False})
    task.observe(state, 1)
    assert "pre_event_action" in task.violations
    assert task.action_time is None


def test_b2_warning_precedes_latched_short_circuit(task_factory):
    task = task_factory("B2", event_time=1, max_latency=2)
    assert task.visual_state()["status_indicator"] == "green"
    task.advance(1)
    assert task.visual_state()["status_indicator"] == "red"
    assert task.visual_state()["warning_red"]
    assert not task.visual_state()["short_indicator_red"]
    task.advance(1.99)
    assert not task.short_circuit
    task.advance(0.01)
    assert task.short_circuit
    assert task.visual_state()["status_indicator"] == "flashing_red"
    assert task.visual_state()["short_indicator_red"]
    assert not task.visual_state()["warning_red"]
    assert "short_circuit" in task.physical_failures
    assert task.terminated
    # Cutting power after damage does not erase the failed episode or red indication.
    state = snapshot(task)
    state.controls.update(power_on=False, switch_angle_rad=0.8)
    task.observe(state, 0.6)
    task.advance(1)
    assert task.visual_state()["short_indicator_red"]
    assert not task.joint_success


def test_b2_timely_isolation_prevents_insulation_failure(task_factory):
    task = task_factory("B2", event_time=1, max_latency=2)
    task.advance(1)
    task.advance(1.5)
    state = snapshot(task)
    state.controls.update(power_on=False, switch_angle_rad=0.8)
    task.observe(state, 0.6)
    damage = task.insulation_damage
    task.advance(3)
    assert task.insulation_damage == damage < 1
    assert task.insulation_temperature == 25
    assert task.visual_state()["phase"] == "isolated"
    assert task.visual_state()["status_indicator"] == "off"
    assert not task.visual_state()["short_indicator_red"]
    assert not task.visual_state()["warning_red"]
    assert task.joint_success


def test_motor_event_requires_prior_rotation_and_measured_stop(task_factory):
    task = task_factory("B3", event_time=1)
    state = snapshot(task)
    task.advance(1)
    state.controls.update({"power_on": True, "wheel_speed_rad_s": 0.0})
    task.observe(state, 0.01)
    assert task.event_time is None
    state.controls["wheel_speed_rad_s"] = 4.0
    task.observe(state, 0.01)
    assert task.event_time is None
    state.controls["wheel_speed_rad_s"] = 0.1
    task.observe(state, 0.01)
    assert task.event_time == task.time


def test_motor_policy_waits_in_place_until_measured_stop(task_factory):
    from roboclock.envs.lab.policies.power_isolation import PowerIsolationTrajectory

    task = task_factory("B3", event_time=1)
    state = snapshot(task)
    state = replace(state, ee_position=state.objects["sample_0"].position + [0.15, 0, 0.25])
    policy = PowerIsolationTrajectory(1 / 30)
    task.advance(1)
    state.controls.update(power_on=True, wheel_speed_rad_s=4.0)
    task.observe(state, 1 / 120)
    for _ in range(10):
        command = policy.command(task, state)
        np.testing.assert_array_equal(command.position, state.ee_position)
        assert not command.close
        assert policy.phase == "approach"
    state.controls["wheel_speed_rad_s"] = 0.0
    task.observe(state, 1 / 120)
    assert task.event_time is not None
    command = policy.command(task, state)
    assert np.linalg.norm(command.position - state.ee_position) > 0
    assert not command.close


@pytest.mark.parametrize("task_id", ["B1", "B2", "B3", "B4"])
def test_reactive_oracle_holds_all_joints_before_event(task_factory, task_id):
    from types import SimpleNamespace

    from roboclock.envs.lab.policies.oracle import OraclePickPlace

    oracle = object.__new__(OraclePickPlace)
    oracle.env = SimpleNamespace(task=task_factory(task_id))
    oracle.planner = SimpleNamespace()
    initial = np.arange(16, dtype=float) / 10
    np.testing.assert_array_equal(oracle.act({"joint_positions": initial}), initial)
    # Measured drift must not change the held target or start an IK correction.
    np.testing.assert_array_equal(oracle.act({"joint_positions": initial + 0.001}), initial)


def test_reactive_reset_preserves_initial_pose_before_first_action(task_factory):
    from types import SimpleNamespace

    from roboclock.envs.lab.policies.oracle import OraclePickPlace

    oracle = object.__new__(OraclePickPlace)
    oracle.env = SimpleNamespace(task=task_factory("B2"))
    oracle.planner = SimpleNamespace()
    for initial in (np.zeros(16), np.ones(16)):
        oracle.reset({"joint_positions": initial})
        np.testing.assert_array_equal(oracle.act({"joint_positions": initial + 0.001}), initial)


def rescued_state(task, *, grasped=True, clearance=0.025, tilt=0, contacts=frozenset()):
    state = snapshot(task)
    obj = state.objects["sample_0"]
    position = obj.position.copy()
    position[2] = task.layout.table_height + task.rack_rim + 0.07 + clearance
    state.objects["sample_0"] = replace(
        obj,
        position=position,
        grasped=grasped,
        orientation=np.array([np.cos(tilt / 2), 0, np.sin(tilt / 2), 0]),
        fixture_contacts=contacts,
    )
    return state


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grasped": False},
        {"clearance": 0.0},
        {"tilt": 0.4},
        {"contacts": frozenset(["rack_rail1"])},
    ],
)
def test_tube_rescue_requires_retained_clear_upright_lift(task_factory, kwargs):
    task = task_factory("B4", event_time=1)
    task.event_time = task.time = 1
    task.observe(rescued_state(task, **kwargs), 0.2)
    assert task.action_time is None


def test_tube_retention_requires_continuous_dwell(task_factory):
    task = task_factory("B4", event_time=1)
    task.event_time = task.time = 1
    task.observe(rescued_state(task), 0.1)
    task.observe(rescued_state(task, grasped=False), 0.01)
    task.observe(rescued_state(task), 0.1)
    assert task.action_time is None
    task.observe(rescued_state(task), 0.05)
    assert task.joint_success


def test_spill_failure_cannot_be_erased_by_late_rescue(task_factory):
    task = task_factory("B4", event_time=1)
    task.event_time = task.time = 1
    task.observe(rescued_state(task, tilt=np.deg2rad(86)), 0.3)
    task.observe(rescued_state(task), 0.2)
    assert "sample_spilled" in task.physical_failures
    assert not task.physical_success


def test_disconnect_requires_separation_contact_and_continuous_dwell():
    from types import SimpleNamespace

    from roboclock.envs.lab.mdp.state import ObjectState
    from roboclock.envs.lab.reactive_scene import ReactiveEquipmentScene

    rig = ReactiveEquipmentScene.__new__(ReactiveEquipmentScene)
    rig.task = SimpleNamespace(power_task=True, task_id="B2")
    rig.power_on, rig.press_dwell, rig.wiring = True, 0.0, []

    def sample(angle, contact):
        return ObjectState(
            np.zeros(3),
            np.array([np.cos(angle / 2), 0, -np.sin(angle / 2), 0]),
            robot_contact=contact,
        )

    assert rig.measure(sample(0.8, False), 0.1)["power_on"]
    assert rig.measure(sample(0.1, True), 0.1)["power_on"]
    assert rig.measure(sample(0.8, True), 0.02)["power_on"]
    assert rig.measure(sample(0.8, False), 0.01)["power_on"]
    assert rig.measure(sample(0.8, True), 0.02)["power_on"]
    assert not rig.measure(sample(0.8, True), 0.01)["power_on"]
    assert rig.measure(sample(0.05, False), 0.1)["power_on"]


def test_reclosing_disconnect_latches_failure(task_factory):
    task = task_factory("B2")
    task.event_time = task.time = 1
    state = snapshot(task)
    state.controls.update(power_on=False, switch_angle_rad=0.8)
    task.observe(state, 0.1)
    state.controls.update(power_on=True, switch_angle_rad=0.0)
    task.observe(state, 0.1)
    assert "power_reconnected" in task.violations
    assert task.isolated_dwell == 0


def test_downloaded_disconnect_is_portable_and_attributed():
    import hashlib
    import json
    from pathlib import Path

    pxr = pytest.importorskip("pxr.Usd")
    root = Path(__file__).parents[3] / "src/roboclock/envs/lab/assets/knife_switch"
    source = json.loads((root / "provenance.json").read_text())
    assert source["author"] == "Sam Feng"
    assert source["license"] == "CC-BY-4.0"
    assert (
        hashlib.sha256((root / "source.glb").read_bytes()).hexdigest()
        == source["source_glb_sha256"]
    )
    for part in ("handle", "panel"):
        stage = pxr.Stage.Open(str(root / (part + ".usdc")))
        assert stage.GetPrimAtPath("/Switch/Mesh")


def test_tube_starts_upright_half_full_and_waits_for_measured_near_fall(task_factory):
    task = task_factory("B4", event_time=1)
    assert np.allclose(task.layout.objects["sample_0"].orientation, [1, 0, 0, 0])
    assert task.liquid.volume == pytest.approx(task.liquid.capacity / 2)
    task.advance(1)
    assert task.disturbance_time == 1
    assert task.event_time is None
    task.observe(rescued_state(task, grasped=False, tilt=np.deg2rad(25)), 0.01)
    assert task.event_time is None
    task.time = 2
    task.observe(rescued_state(task, grasped=False, tilt=np.deg2rad(33)), 0.01)
    assert task.event_time == 2
    assert not task.violations


def test_tube_premature_reach_without_contact_is_allowed(task_factory):
    task = task_factory("B4", event_time=1)
    state = snapshot(task)
    task.observe(state, 0.01)
    state = replace(state, ee_position=state.ee_position + [0.05, 0, 0])
    task.observe(state, 0.01)
    assert "pre_event_action" not in task.violations
    assert task.maximum_pre_event_ee_displacement == pytest.approx(0.05)


def test_tube_controller_keeps_fingers_open_between_hover_and_grasp(task_factory):
    from roboclock.envs.lab.policies.tube_rescue import TubeRescueTrajectory
    from roboclock.physics.geometry import rotation_matrix

    task = task_factory("B4")
    task.event_time = task.time = 1
    state = rescued_state(task, grasped=False, tilt=np.deg2rad(45))
    obj = state.objects["sample_0"]
    pick = obj.position + rotation_matrix(obj.orientation)[:, 2] * 0.0525
    upright = np.array([0.0, 0.0, 1.0, 0.0])
    policy = TubeRescueTrajectory(1 / 30)
    policy.phase = "hover"
    state = replace(state, ee_position=pick + [0, 0, 0.08], ee_orientation=upright)
    assert not policy.command(task, state).close
    assert policy.phase == "reach"
    assert not policy.command(task, state).close
    state = replace(state, ee_position=pick)
    assert policy.command(task, state).close


@pytest.mark.parametrize("task_id", ["B1", "B2", "B3", "B4", "B5", "B6", "B7"])
def test_pre_event_tool_drift_measures_motion_before_the_cue_only(task_factory, task_id):
    task = task_factory(task_id, event_time=5.0)
    still = replace(snapshot(task), secondary_ee_position=np.array([0.4, 0.5, 1.0]))
    for _ in range(3):
        task.advance(1 / 250)
        task.observe(still, 1 / 250)
    assert task.event_time is None
    assert task.metrics()["pre_event_tool_drift_m"] == 0.0
    # Either arm leaving its start pose before the cue counts.
    moved = replace(still, secondary_ee_position=np.array([0.4, 0.5, 1.03]))
    task.advance(1 / 250)
    task.observe(moved, 1 / 250)
    assert task.metrics()["pre_event_tool_drift_m"] == pytest.approx(0.03)
    # Motion after the cue is the reaction, not drift.
    task.event_time = task.time
    task.advance(1 / 250)
    task.observe(replace(still, ee_position=np.array([0.6, 0.0, 1.0])), 1 / 250)
    assert task.metrics()["pre_event_tool_drift_m"] == pytest.approx(0.03)
