"""B1 oxygen desaturation: registry, process model, scoring, layout and oracle logic (no simulator)."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import oxygen_flow as o2
from roboclock.envs.hospital.config import load_config, load_legacy_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory
from roboclock.envs.hospital.rig import head_pixels
from roboclock.envs.hospital.tasks.reactive import oxygen_desaturation as task_module
from roboclock.envs.hospital.tasks.registry import TASKS, make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState
from roboclock.envs.lab.robot import mount_positions

LEVELS = ("easy", "medium", "hard")
DT = 0.004


@pytest.fixture(scope="module")
def config():
    return load_config()


def bound(config, level="hard", seed=0):
    task = make_task("B1", level, seed, config)
    task.bind(make_layout(task, config))
    return task


def controls(angle, contact=True):
    return SimpleNamespace(controls={"knob_angle_rad": angle, "knob_contact": contact})


def run(task, until, angle=lambda t: 0.0, contact=True):
    """Advance the task clock to `until`, feeding the knob angle as a function of task time."""
    while task.time < until - 1e-9 and not task.terminated:
        task.advance(DT)
        task.evaluate(controls(angle(task.time), contact), DT)


def turn_to(task, flow, start, rate=1.1):
    """A knob turn that starts `start` seconds after the alarm and stops at the angle for `flow`."""
    target = task_module.angle_for_flow(flow)
    t0 = task.spec.event_time + start
    return lambda t: min(max(t - t0, 0.0) * rate, target)


# --- registry and episode sampling ---------------------------------------------------------------


def test_b1_is_the_oxygen_task_and_the_legacy_profile_keeps_the_pump_alarms(config):
    task = make_task("B1", "medium", 0, config)
    assert TASKS["B1"] is task_module.OxygenDesaturation and type(task) is TASKS["B1"]
    assert (task.name, task.category, task.bedside_module) == (
        "oxygen_desaturation", "event_reactive", "oxygen_flow"
    )
    legacy = make_task("B1", "medium", 0, load_legacy_config())
    assert legacy.name == "pump_alarm"


@pytest.mark.parametrize("seed", range(6))
def test_alarm_time_is_seeded_and_limits_tighten_with_the_level(config, seed):
    specs = [make_task("B1", level, seed, config).spec for level in LEVELS]
    lo, hi = config["episode"]["event_time_range"]
    assert len({s.event_time for s in specs}) == 1 and lo <= specs[0].event_time <= hi
    limits = [s.response_limit for s in specs]
    assert limits[0] > limits[1] > limits[2] >= 1.0
    assert limits[1] == pytest.approx(math.sqrt(limits[0] * limits[2]), abs=0.01)
    for spec in specs:  # the episode outlasts the deadline
        assert spec.max_time >= spec.event_time + spec.response_limit + 4.0 - 1e-9


def test_order_and_instruction_follow_the_seed(config):
    for seed in range(6):
        task = make_task("B1", "medium", seed, config)
        assert task.ordered_flow == task_module.ORDERED_FLOWS[seed % 3]
        assert task.band == (task.ordered_flow - 1.0, task.ordered_flow + 1.0)
        text = task.gen_instruction()
        assert f"{task.band[0]:g}-{task.band[1]:g} L/min" in text
        assert f"within {task.spec.response_limit:g} s of the alarm" in text
        assert len(text) <= 512  # the observation space's instruction length


# --- process model -------------------------------------------------------------------------------


def test_valve_flow_is_linear_in_the_knob_angle_and_clamped_to_the_scale():
    assert task_module.flow_from_angle(0.0) == task_module.BASELINE_FLOW
    assert task_module.flow_from_angle(math.radians(72.0)) == pytest.approx(6.0)
    assert task_module.flow_from_angle(math.radians(-90.0)) == 0.0
    assert task_module.flow_from_angle(math.radians(400.0)) == task_module.MAX_FLOW
    assert task_module.flow_from_angle(float("nan")) == task_module.BASELINE_FLOW
    assert task_module.angle_for_flow(6.0) == pytest.approx(math.radians(72.0))


def test_float_lags_the_valve_and_converges():
    shown = 2.0
    for _ in range(round(task_module.FLOAT_LAG / DT)):
        shown = task_module.float_step(shown, 6.0, DT)
    assert shown == pytest.approx(2.0 + 4.0 * (1 - math.exp(-1)), abs=0.02)
    for _ in range(500):
        shown = task_module.float_step(shown, 6.0, DT)
    assert shown == pytest.approx(6.0, abs=1e-6)


def test_spo2_falls_to_cross_90_at_the_event_time_and_the_alarm_starts_there(config):
    task = bound(config)
    event = task.spec.event_time
    # Real-rate lead-in (2026-11-23): the episode starts mid-fall, just above the alarm line.
    run(task, event - 3.0)
    assert task.spo2 == pytest.approx(90.0 + 3.0 * task_module.FALL_RATE, abs=0.02) and not task.alarm_active
    assert task.spo2 < task.baseline_spo2
    run(task, event - 1.0)
    assert task.spo2 == pytest.approx(90.0 + task_module.FALL_RATE, abs=0.02) and task.event_time is None
    run(task, event + 0.1)
    assert task.event_time == pytest.approx(event, abs=DT)
    assert task.alarm_active and task.displayed_spo2() == 89
    assert any(e["type"] == "spo2_alarm" for e in task.event_log)
    # Untreated, the saturation keeps falling, more slowly, to its floor.
    run(task, event + 3.0)
    assert task.spo2 == pytest.approx(90.0 - task_module.FALL_RATE_ALARM * 3.0, abs=0.05)
    assert task_module.spo2_step(80.0, 1.0, None) == task_module.SPO2_FLOOR


def test_spo2_recovers_after_the_float_reaches_the_band(config):
    task = bound(config, "easy")
    run(task, task.spec.event_time + 6.0, turn_to(task, task.ordered_flow, 1.0))
    assert task.goal_achieved and task.spo2 > task_module.ALARM_THRESHOLD
    assert not task.alarm_active and task.min_spo2 < task_module.ALARM_THRESHOLD
    assert o2.monitor_state(task) == (task.displayed_spo2(), False)


# --- scoring -------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_a_timely_turn_into_the_band_succeeds_after_the_dwell(config, level):
    task = bound(config, level)
    event, limit = task.spec.event_time, task.spec.response_limit
    run(task, event + limit, turn_to(task, task.ordered_flow, 1.0))
    assert task.goal_achieved and task.physical_success and task.temporal_success
    assert task.knob_moved_time > event
    # The response ends when the float has stayed in the band for the dwell.
    assert task.settled_time - task.band_entry_time == pytest.approx(task_module.DWELL, abs=2 * DT)
    assert task.action_time == task.settled_time < event + limit
    m = task.metrics()
    assert m["band_l_min"][0] <= m["float_flow_l_min"] <= m["band_l_min"][1]
    assert m["reaction_latency"] == pytest.approx(task.settled_time - event, abs=DT)


def test_the_episode_shows_the_recovery_before_it_ends(config):
    task = bound(config, "easy")
    run(task, task.spec.max_time, turn_to(task, task.ordered_flow, 1.0))
    assert task.terminated
    assert task.time == pytest.approx(task.settled_time + task_module.HOLD_AFTER, abs=2 * DT)


def test_turning_before_the_alarm_is_an_early_action(config):
    task = bound(config)
    run(task, task.spec.event_time - 1.0, lambda t: math.radians(10.0))
    assert "early_action" in task.violations and task.event_time is None
    run(task, task.spec.event_time + 4.0, lambda t: task_module.angle_for_flow(task.ordered_flow))
    assert task.physical_success and not task.temporal_success and not task.joint_success


def test_a_late_settle_is_physical_success_but_misses_the_deadline(config):
    task = bound(config, "hard")
    limit = task.spec.response_limit
    run(task, task.spec.max_time, turn_to(task, task.ordered_flow, limit - 0.2))
    assert task.physical_success and "late_response" in task.violations
    assert task.metrics()["deadline_missed"] and not task.joint_success


def test_no_response_runs_out_the_limit(config):
    task = bound(config)
    run(task, task.spec.event_time + task.spec.response_limit + 0.1)
    assert not task.goal_achieved and task.violations == ["late_response"]
    assert task.knob_moved_time is None and task.spo2 < task_module.ALARM_THRESHOLD


def test_passing_through_the_band_without_staying_does_not_count(config):
    task = bound(config, "easy")
    event = task.spec.event_time
    top = task_module.angle_for_flow(task.band[1] + 2.0)
    run(task, event + 3.0, lambda t: min(max(t - event - 0.5, 0.0) * 4.0, top))
    assert task.band_entry_time is not None and not task.goal_achieved
    assert not task.physical_failures  # 2 L/min past the band is not yet flooding


def test_flooding_the_patient_or_turning_the_flow_down_is_a_physical_failure(config):
    flood = bound(config, "easy")
    wide = task_module.angle_for_flow(flood.band[1] + task_module.OVERSHOOT + 1.0)
    run(flood, flood.spec.event_time + 3.0, lambda t: wide if t > flood.spec.event_time + 0.5 else 0.0)
    assert flood.physical_failures == ["flow_overshoot"] and flood.terminated
    assert not flood.physical_success
    down = bound(config, "easy")
    run(down, down.spec.event_time + 3.0, lambda t: math.radians(-30.0) if t > down.spec.event_time + 0.5 else 0.0)
    assert down.physical_failures == ["flow_reduced"]


def test_a_knob_that_moved_without_robot_contact_is_not_credited(config):
    task = bound(config, "easy")
    run(task, task.spec.event_time + 5.0, turn_to(task, task.ordered_flow, 1.0), contact=False)
    assert task.knob_moved_time is None and not task.goal_achieved


def test_countdown_dial_runs_from_the_alarm_and_turns_green_on_success(config):
    task = bound(config, "easy")
    commands = task.advance(DT)
    assert [c.kind for c in commands] == ["indicator"]  # no countdown before the alarm
    seen = []
    while not task.terminated:
        seen += task.advance(DT)
        task.evaluate(controls(turn_to(task, task.ordered_flow, 1.0)(task.time)), DT)
    timers = [c.value for c in seen if c.kind == "timer"]
    assert timers[0][0] == pytest.approx(1.0, abs=0.02) and timers[-1][1] == "green"
    assert all(a[0] >= b[0] for a, b in zip(timers, timers[1:]))


# --- measured knob angle ---------------------------------------------------------------------------


def quat_x(angle):
    """Rotation by `angle` about world +x (w, x, y, z)."""
    return np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])


def test_knob_angle_is_counterclockwise_from_the_robot_and_unwraps():
    # The robot looks along +x: counterclockwise for it is a negative rotation about +x.
    angle, axis = o2.knob_angle(quat_x(-math.radians(72.0)))
    assert angle == pytest.approx(math.radians(72.0)) and axis == pytest.approx(0.0, abs=1e-6)
    previous = 0.0
    for degrees in range(0, 231, 10):  # past 180 degrees without a jump
        previous, _ = o2.knob_angle(quat_x(-math.radians(degrees)), previous)
        assert previous == pytest.approx(math.radians(degrees), abs=1e-6)
    tilted = np.array([math.cos(0.1), 0.0, math.sin(0.1), 0.0])
    assert o2.knob_angle(tilted)[1] == pytest.approx(0.2, abs=1e-6)
    assert math.isnan(o2.knob_angle([0.0, 0.0, 0.0, 0.0])[0])


def test_joint_travel_spans_closed_to_the_top_of_the_scale():
    assert task_module.flow_from_angle(-math.radians(o2.KNOB_CLOSE_DEG)) == pytest.approx(0.0)
    assert task_module.flow_from_angle(math.radians(o2.KNOB_OPEN_DEG)) == pytest.approx(
        task_module.MAX_FLOW
    )
    assert o2.ball_height(0.0) == 0.0 and o2.ball_height(99.0) == o2.BALL_TRAVEL


# --- layout ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_layout_puts_the_knob_where_the_forward_hand_reaches_it(config, seed):
    task = bound(config, seed=seed)
    info = task.layout.info
    assert info["bedside_task"] == "oxygen_flow" and info["hide_bench"]
    scene = info["oxygen_flow"]
    knob = np.asarray(scene["knob"])
    np.testing.assert_allclose(task.layout.objects[o2.NAME].position, knob, atol=1e-9)
    mount = np.asarray(mount_positions(config, "B1")[0])
    offset = knob - mount
    assert 0.38 <= offset[0] <= 0.50 and abs(offset[1]) <= 0.12 and 0.20 <= offset[2] <= 0.25
    # At rest the hand is tucked at its own column, away from the flowmeter.
    np.testing.assert_allclose(info["initial_ee_position"], mount + o2.TUCK)
    assert np.linalg.norm(np.asarray(info["initial_ee_position"]) - knob) > 0.35
    # The tube stands above the valve, the float rests at its foot and the band lies on the tube.
    assert scene["tube_bottom"][2] < scene["ball_rest"][2] < scene["tube_top"][2]
    top = scene["ball_rest"][2] + o2.ball_height(scene["band_l_min"][1])
    assert top < scene["tube_top"][2]
    assert scene["ordered_flow_l_min"] == task.ordered_flow
    # The ledge under the panel is the support collider; the panel has its own.
    surface, ledge = info["work_surface"], scene["ledge"]
    assert surface["center"] == ledge["center"] and surface["size"] == ledge["size"]
    assert {"gas_panel", "oxygen_valve", "oxygen_tube", "monitor_head"} <= {
        f.name for f in task.layout.fixtures
    }
    # The ward moves with the seeded shift, like every apparatus point.
    shift = knob[:2] - (mount[:2] + o2.KNOB_OFFSET[:2])
    pose = info["ward_pose"]
    np.testing.assert_allclose(
        pose["translate"][:2], np.array([o2.bay.ROBOT_Y, -o2.bay.ROBOT_X]) + shift, atol=1e-9
    )
    assert o2.WARD_PILLOW in pose["hide"]


@pytest.mark.parametrize("seed", range(10))
def test_cues_are_inside_the_head_camera_image(config, seed):
    scene = bound(config, seed=seed).layout.info["oxygen_flow"]
    screen = np.asarray(scene["screen"])
    half_w, half_h = o2.SCREEN_SIZE[0] / 2, o2.SCREEN_SIZE[1] / 2
    points = {
        "knob": scene["knob"],
        "tube_top": scene["tube_top"],
        "tube_bottom": scene["tube_bottom"],
        "screen": screen,
        "screen_left": screen + [0.0, half_w, 0.0],
        "screen_right": screen + [0.0, -half_w, 0.0],
        "screen_top": screen + [0.0, 0.0, half_h],
        "screen_bottom": screen + [0.0, 0.0, -half_h],
        "dial": np.asarray(scene["dial"]) + [0.0, 0.0, 0.13],
    }
    pixels, depth = head_pixels(config, "B1", list(points.values()))
    for name, (u, v), d in zip(points, pixels, depth):
        assert d > 0 and 8 <= u <= 632 and 8 <= v <= 472, (name, u, v)


def test_model_constants_match_the_imported_flowmeter():
    credits = json.loads((o2.MODELS / "credits.json").read_text())
    meter = credits["oxygen_flowmeter"]
    assert meter["license"] == "CC-BY-4.0" and meter["author"] == "Dwayne.Barnett"
    assert meter["tube_axis_xy_m"][0] == pytest.approx(o2.TUBE_AXIS_X, abs=1e-3)
    assert meter["tube_z_m"] == pytest.approx(list(o2.TUBE_Z), abs=1e-3)
    assert meter["ball_rest_m"][2] == pytest.approx(o2.BALL_REST_Z, abs=1e-3)
    assert meter["valve_front_x_m"] == pytest.approx(o2.VALVE_FRONT_X, abs=1e-3)
    assert meter["knob_radius_m"] == pytest.approx(o2.KNOB_RADIUS, abs=5e-4)
    assert meter["barb_tip_m"] == pytest.approx(list(o2.BARB_TIP), abs=1e-3)
    # The float's full travel stays inside the glass tube.
    assert o2.BALL_REST_Z + o2.BALL_TRAVEL < o2.TUBE_Z[1]
    for name in ("oxygen_flowmeter", "patient_supine", "oxygen_mask", "patient_monitor_head"):
        assert credits[name]["license"] == "CC-BY-4.0" and credits[name]["source"].startswith("https://")
        assert (o2.MODELS / credits[name]["usd"]).is_file()
    for part in meter["parts"].values():
        assert (o2.MODELS / part).is_file()


# --- oracle logic ----------------------------------------------------------------------------------


def oracle_state(task, ee, angle=0.0, grasped=False):
    knob = task.layout.objects[o2.NAME]
    obj = ObjectState(knob.position.copy(), knob.orientation.copy(), grasped=grasped)
    return PhysicalState(
        {o2.NAME: obj}, np.asarray(ee, dtype=float), controls={"knob_angle_rad": angle}
    )


def test_oracle_holds_the_tuck_until_the_alarm_then_reaches_pinches_and_turns(config):
    task = bound(config, "hard", seed=1)
    info = task.layout.info
    planner = HospitalTrajectory(config["control"], 0.02)
    planner.dt = 0.02
    tuck = np.asarray(info["initial_ee_position"])
    knob = np.asarray(info["oxygen_flow"]["knob"])
    planner.last_goal = tuck.copy()
    command = o2.command(planner, task, oracle_state(task, tuck))
    assert planner.phase == "wait_alarm" and planner.tool_tilt is None and not command.close
    np.testing.assert_allclose(command.position, tuck)
    # The alarm: one joint-space swing to the forward pose in front of the knob.
    task.event_time = task.spec.event_time
    task.time = task.event_time + 0.02
    command = o2.command(planner, task, oracle_state(task, tuck))
    pre = knob - [o2.GRIP_BACK + o2.STANDOFF, 0.0, 0.0]
    assert planner.phase == "swing" and planner.sync_joints
    np.testing.assert_allclose(planner.tool_tilt, o2.FORWARD)
    np.testing.assert_allclose(command.position, pre)
    assert command.yaw == o2.HAND_YAW and not command.close
    # Arrived: advance onto the knob, then pinch.
    grip = knob - [o2.GRIP_BACK, 0.0, 0.0]
    command = o2.command(planner, task, oracle_state(task, pre))
    assert planner.phase == "advance" and command.position[0] > pre[0]
    planner.last_goal = grip.copy()
    command = o2.command(planner, task, oracle_state(task, grip))
    assert planner.phase == "pinch" and command.close
    # Grasped: the commanded roll leads the measured knob by at most TURN_LEAD.
    angle = 0.0
    for _ in range(200):
        command = o2.command(planner, task, oracle_state(task, grip, angle, grasped=True))
        assert planner.phase == "turn" and command.close
        assert command.yaw - o2.HAND_YAW <= angle + o2.TURN_LEAD + 1e-9
        angle = command.yaw - o2.HAND_YAW  # the knob follows the wrist
    target = task_module.angle_for_flow(task.ordered_flow)
    assert angle == pytest.approx(target, abs=math.radians(18.0) * o2.FLOW_TOLERANCE + 1e-6)
    # The turn stays inside the wrist's roll range (+-2 rad).
    assert abs(o2.HAND_YAW + task_module.angle_for_flow(max(task_module.ORDERED_FLOWS))) < 1.9
    # Settled: let go and back off.
    task.goal_achieved = True
    command = o2.command(planner, task, oracle_state(task, grip, angle, grasped=True))
    assert planner.phase == "release" and not command.close and command.position[0] < grip[0]


# --- results and stages ----------------------------------------------------------------------------


def test_old_pump_alarm_rows_are_retired_and_new_rows_score_their_stages():
    from roboclock.envs.hospital.results import normalize
    from roboclock.envs.hospital.stages import STAGES

    old = normalize({"task_id": "B1", "task_name": "pump_alarm", "task_success": True})
    assert old["retired"] and old["task_id"] == "B1-pump_alarm" and old["benchmark_id"] is None
    new = normalize({"task_id": "B1", "task_name": "oxygen_desaturation", "task_success": False})
    assert new["benchmark_id"] == "B1" and "retired" not in new
    assert STAGES["B1"]({"task_name": "pump_alarm"}) == ()
    stages = STAGES["B1"]({"task_name": "oxygen_desaturation"})
    assert [s.name for s in stages] == ["knob_turned", "float_in_band"]
