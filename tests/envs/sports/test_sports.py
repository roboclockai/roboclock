"""Temporal predicates and contact requirements, independent of the Isaac runtime."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import yaml

from roboclock.envs.sports.common.config import load_config
from roboclock.envs.sports.common.tasks import TASKS, make_task


def state(t):
    return {
        "objects": {
            n: {
                "position": np.array(b.position),
                "orientation": np.array([1.0, 0, 0, 0]),
                "velocity": np.zeros(3),
                "angular_velocity": np.zeros(3),
                "supported": True,
                "robot_contact": False,
                "grasped": False,
            }
            for n, b in t.layout.bodies.items()
        },
        "controls": {
            n: {"activated": False, "contact": False, "exposed": True} for n in t.layout.controls
        },
        "tool_velocity": np.zeros(3),
    }


def advance(t, s, seconds):
    for _ in range(round(seconds / 0.01)):
        t.observe(s, 0.01)


def press(t, s, name, duration=0.04):
    s["controls"][name].update(activated=True, contact=True)
    advance(t, s, duration)
    s["controls"][name].update(activated=False, contact=False)
    advance(t, s, 0.12)


@pytest.mark.parametrize("task_id", TASKS)
def test_serializable_reproducible_plans_and_difficulty_independent_geometry(task_id):
    a, b = make_task(task_id, "easy", 7), make_task(task_id, "hard", 7)
    json.dumps(asdict(a.spec))
    if task_id != "31":
        assert asdict(a.layout) == asdict(b.layout)
        assert asdict(a.spec) == asdict(make_task(task_id, "easy", 7).spec)
        assert a.limits["deadline"] > b.limits["deadline"]
    else:
        assert a.move in a.board.legal_moves


def test_buzzer_early_lockout_is_irrecoverable():
    t = make_task("35")
    s = state(t)
    press(t, s, "buzzer")
    assert t.terminated and "early_press_lockout" in t.violations
    assert not t.joint_success


@pytest.mark.parametrize("delay,success", [(0.1, True), (3.1, False)])
def test_buzzer_late_action_separates_physical_and_timing(delay, success):
    t = make_task("35")
    s = state(t)
    t.time = t.events[0]["time"] + delay
    press(t, s, "buzzer")
    assert t.physical_success
    assert t.joint_success is success


def test_rhythm_requires_continuous_hold_and_correct_pad():
    t = make_task("32")
    s = state(t)
    e = t.events[1]
    t.time = e["time"] + 0.1
    press(t, s, e["control"], 0.1)
    assert 1 not in t.correct and "note_1_hold_released_early" in t.violations
    press(t, s, e["control"], e["hold"] + 0.05)
    assert 1 in t.correct and not t.temporal_success


def test_rhythm_complete_schedule():
    t = make_task("32")
    s = state(t)
    for i, e in enumerate(t.events):
        t.time = e["time"] + 0.1
        press(t, s, e["control"], e["hold"] + 0.05)
    assert t.joint_success


def test_decoy_contact_without_depression_fails():
    t = make_task("34")
    s = state(t)
    e = next(e for e in t.events if not e["valid"])
    t.time = e["time"] + 0.1
    s["controls"][e["control"]]["contact"] = True
    advance(t, s, 0.01)
    assert "decoy_hit" in t.violations


def test_command_deadlines_shrink_and_wrong_action_latches():
    t = make_task("36")
    s = state(t)
    assert all(a["deadline"] > b["deadline"] for a, b in zip(t.events, t.events[1:]))
    t.time = 2.1
    wrong = next(c for c in s["controls"] if c != t.events[0]["control"])
    press(t, s, wrong)
    assert "wrong_command_action" in t.violations


def test_dumbbell_must_hold_then_lower_on_signal_and_release():
    t = make_task("33")
    s = state(t)
    o = s["objects"]["dumbbell"]
    initial = o["position"].copy()
    o["position"][2] += 0.13
    o.update(grasped=True, robot_contact=True, supported=False)
    advance(t, s, 1.3)
    assert t.down_time is not None
    o["position"] = initial
    o.update(grasped=False, robot_contact=False, supported=True)
    advance(t, s, 0.3)
    assert t.joint_success and t.lower_time >= t.down_time


def test_dumbbell_early_lowering_latches():
    t = make_task("33")
    s = state(t)
    o = s["objects"]["dumbbell"]
    o["position"][2] += 0.13
    o["grasped"] = True
    advance(t, s, 0.1)
    o["position"][2] -= 0.10
    advance(t, s, 0.1)
    assert "lowered_before_down" in t.violations


def test_red_stop_then_resume_before_green_fails():
    t = make_task("37")
    s = state(t)
    t.time = 2.01
    advance(t, s, 0.1)
    s["tool_velocity"][0] = 0.1
    advance(t, s, 0.01)
    assert "moved_during_red" in t.violations


def test_puck_pose_without_contact_is_not_completion():
    t = make_task("37")
    s = state(t)
    s["objects"]["puck"]["position"] = np.array(t.layout.targets["puck"])
    t.red_seen = True
    advance(t, s, 0.4)
    assert not t.physical_success


@pytest.mark.parametrize("bad", ["unsupported", "held", "tilted", "never_grasped", "hovering"])
def test_pin_pose_alone_does_not_pass(bad):
    t = make_task("38")
    s = state(t)
    t.grasped.update(s["objects"])
    for n, o in s["objects"].items():
        o["position"] = np.array(t.layout.targets[n])
    o = s["objects"]["pin_0"]
    if bad == "unsupported":
        o["supported"] = False
    if bad == "held":
        o["robot_contact"] = True
    if bad == "tilted":
        o["orientation"] = np.array([0.707, 0.707, 0, 0])
    if bad == "never_grasped":
        t.grasped.remove("pin_0")
    if bad == "hovering":
        o["position"][2] += 0.04
    advance(t, s, 0.3)
    assert not t.physical_success


def test_pin_count_unique_full_budget_rate_and_ball_deadline():
    t = make_task("38")
    s = state(t)
    t.grasped.update(s["objects"])
    for n, o in s["objects"].items():
        o["position"] = np.array(t.layout.targets[n])
    advance(t, s, 1)
    assert t.physical_success and not t.terminated
    assert t.metrics()["operations_per_minute"] == 5
    t.time = t.limits["ball_deadline"]
    s["objects"]["pin_0"]["position"][0] += 0.1
    advance(t, s, 0.01)
    assert "pins_missing_at_ball" in t.violations


def test_cups_flip_only_on_correct_released_arrangement_and_bell():
    t = make_task("39")
    s = state(t)
    press(t, s, "bell")
    assert t.card_index == 0 and "bell_with_wrong_arrangement" in t.violations
    t.grasped.update(s["objects"])
    for n, p in t.targets().items():
        s["objects"][n]["position"] = np.array(p)
    advance(t, s, 0.3)
    press(t, s, "bell")
    assert t.card_index == 1 and len(t.correct) == 1
    assert not t.joint_success


def test_hopper_starvation_is_recorded_at_demand_even_without_actual_shot():
    t = make_task("40")
    s = state(t)
    t.time = t.settings["warmup"]
    advance(t, s, 0.01)
    assert t.shots_due == 1 and "hopper_starved_0" in t.violations
    assert "fed_less_than_demand" in t.violations


def test_hopper_release_unique_admission_capacity_and_physical_exit():
    t = make_task("40")
    s = state(t)
    t.grasped.update(s["objects"])
    for i in range(4):
        o = s["objects"][f"ball_{i}"]
        o["position"] = np.array(t.layout.targets["hopper"]) - [0, 0, 0.06]
        o["position"][2] = 0.80
    advance(t, s, 0.3)
    assert len(t.admitted) == 4 and "hopper_over_capacity" in t.violations
    t.time = t.settings["warmup"]
    advance(t, s, 0.01)
    assert len(t.launched) == 1 and not t.exited
    name = next(iter(t.launched))
    s["objects"][name]["position"][2] = 1.05
    advance(t, s, 0.01)
    assert name in t.exited


@pytest.mark.parametrize(
    "path,value",
    [
        (("sim", "physics_dt"), 0),
        (("sim", "settle_steps"), -1),
        (("scene", "spatial_jitter"), float("nan")),
        (("tasks", "38", "pins"), 9),
        (("tasks", "40", "capacity"), 0),
        (("tasks", "40", "balls"), 4),
        (("difficulties", "easy", "pins_per_min"), 6),
    ],
)
def test_impossible_or_invalid_config_is_rejected(tmp_path, path, value):
    config = load_config()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(file)


def test_dumbbell_handle_axis_symmetry_does_not_require_an_arbitrary_roll():
    task = make_task("33")
    obj = state(task)["objects"]["dumbbell"]
    obj["orientation"] = np.array([2**-0.5, 0, 2**-0.5, 0])
    assert task.stable(obj, "dumbbell")
    obj["orientation"] = np.array([2**-0.5, 2**-0.5, 0, 0])
    assert not task.stable(obj, "dumbbell")


def test_command_press_begun_before_cue_is_not_accepted_as_timely():
    task = make_task("36")
    s = state(task)
    event = task.events[0]
    task.time = event["time"] - 0.015
    press(task, s, event["control"])
    assert "early_command_0" in task.violations


def test_short_contact_bounce_does_not_create_another_press_edge():
    task = make_task("32")
    s = state(task)
    event = task.events[0]
    control = s["controls"][event["control"]]
    task.time = event["time"] + 0.1
    control.update(activated=True, contact=True)
    advance(task, s, 0.15)
    control.update(activated=False, contact=False)
    advance(task, s, 0.04)
    control.update(activated=True, contact=True)
    advance(task, s, 0.10)
    assert len(task.attempted) == 1
    control.update(activated=False, contact=False)
    advance(task, s, 0.12)
    control.update(activated=True, contact=True)
    advance(task, s, 0.04)
    assert "wrong_or_extra_note" in task.violations


def test_moles_have_nine_clear_retractable_holes_and_randomized_decoys():
    a, b = make_task("34", seed=0), make_task("34", seed=1)
    assert len(a.layout.controls) == 9
    assert len({e["control"] for e in a.events[:9]}) == 9
    assert a.events != b.events
    assert any(not e["valid"] for e in a.events)
    assert any(e["valid"] for e in a.events)
    h = a.config["scene"]["table_height"]
    for m in a.layout.controls.values():
        assert m.position[2] - m.travel - m.height / 2 > h + 0.008
        assert m.position[2] - m.travel + m.height / 2 < h + 0.085


def test_mole_complete_random_schedule():
    t = make_task("34")
    s = state(t)
    for e in t.events:
        t.time = e["time"] + 0.1
        advance(t, s, 0.01)
        if e["valid"]:
            press(t, s, e["control"])
    t.time = t.mole_up_times[len(t.events) - 1] + t.limits["deadline"]
    advance(t, s, 0.01)
    assert t.joint_success


def test_mole_deadline_starts_at_measured_emergence():
    t = make_task("34", "hard")
    s = state(t)
    for c in s["controls"].values():
        c["exposed"] = False
    t.time = t.events[0]["time"]
    advance(t, s, 0.4)
    assert 0 not in t.mole_up_times
    s["controls"][t.events[0]["control"]]["exposed"] = True
    advance(t, s, 0.01)
    assert t.mole_up_times[0] == pytest.approx(2.41)
    assert t.mole_active(0, 3.0)
    assert not t.mole_active(0, 3.12)


def test_mole_failed_actuator_is_a_physical_failure():
    t = make_task("34")
    s = state(t)
    for c in s["controls"].values():
        c["exposed"] = False
    t.time = t.events[0]["time"]
    advance(t, s, 0.8)
    assert "mole_failed_to_rise_0" in t.physical_failures


@pytest.mark.parametrize("offset", [[0, 0, 0.12], [0.083, 0, 0.06], [0, 0, 0.02]])
def test_partially_contained_ball_is_not_admitted(offset):
    t = make_task("40")
    s = state(t)
    t.grasped.add("ball_0")
    p = np.array(t.layout.targets["hopper"])
    p[2] = t.config["scene"]["table_height"]
    s["objects"]["ball_0"]["position"] = p + offset
    advance(t, s, 0.2)
    assert "ball_0" not in t.admitted
    assert t.occupancy == 1


def test_held_partially_inserted_balls_count_toward_capacity():
    t = make_task("40")
    s = state(t)
    for i in range(4):
        o = s["objects"][f"ball_{i}"]
        o["position"] = np.array(t.layout.targets["hopper"])
        o.update(robot_contact=True, grasped=True)
    advance(t, s, 0.01)
    assert "hopper_over_capacity" in t.violations
    assert not t.admitted


def test_spinning_dumbbell_does_not_complete_still_hold():
    t = make_task("33")
    s = state(t)
    o = s["objects"]["dumbbell"]
    o["position"][2] += 0.13
    o.update(grasped=True, robot_contact=True, supported=False)
    o["angular_velocity"] = np.array([0, 0, 1.0])
    advance(t, s, 2.0)
    assert t.lift_time is not None and t.down_time is None


@pytest.mark.parametrize("offset,success", [(0.0, True), (0.035, False)])
def test_puck_goal_requires_whole_disk_inside(offset, success):
    t = make_task("37")
    s = state(t)
    t.touched.add("puck")
    t.red_seen = True
    s["objects"]["puck"]["position"] = np.array(t.layout.targets["puck"]) + [offset, 0, 0]
    advance(t, s, 0.3)
    assert t.physical_success is success


def test_pose_angular_velocity_handles_quaternion_sign_and_world_axis():
    from roboclock.physics.geometry import angular_velocity_from_poses

    q = np.array([1.0, 0.0, 0.0, 0.0])
    quarter_turn = np.array([2**-0.5, 0.0, 0.0, 2**-0.5])
    np.testing.assert_allclose(angular_velocity_from_poses(q, -q, 0.01), [0, 0, 0])
    np.testing.assert_allclose(angular_velocity_from_poses(q, quarter_turn, 0.5), [0, 0, np.pi])
    np.testing.assert_allclose(angular_velocity_from_poses(quarter_turn, q, 0.5), [0, 0, -np.pi])
