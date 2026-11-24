"""A2 sequential bag switch: layout, process visuals, selector latch and oracle timing (no simulator)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import window_bags as bags
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.lab.robot import mount_positions

LEVELS = ("easy", "medium", "hard")


@pytest.fixture(scope="module")
def config():
    return load_config()


def layout_for(config, level="hard", seed=0):
    task = make_task("A2", level, seed, config)
    layout = make_layout(task, config)
    task.bind(layout)
    return task, layout


def test_task_uses_this_module_and_the_sheet_window(config):
    for level, width in zip(LEVELS, (8.0, 5.5, 3.0)):
        task = make_task("A2", level, 0, config)
        assert task.bedside_module == "window_bags" and task.band == (95.0, 100.0)
        assert task.window_close - task.window_open == pytest.approx(width)
        assert 3.0 <= width <= 8.0  # the sheet's 3-8 s window
        assert f"{width:g} seconds" in task.gen_instruction()


@pytest.mark.parametrize("seed", range(5))
def test_layout_places_the_selector_within_reach_on_its_tray(config, seed):
    task, layout = layout_for(config, seed=seed)
    info = layout.info
    assert info["bedside_task"] == "window_bags" and info["hide_bench"]
    pivot = np.asarray(info["a2_bags"]["pivot"])
    np.testing.assert_allclose(layout.objects[bags.NAME].position, pivot)
    left = np.asarray(mount_positions(config, "A2")[0])
    assert 0.25 <= np.hypot(*(pivot[:2] - left[:2])) <= 0.40
    # The ready pose straddles the hub at grasp height, the support collider is the tray.
    np.testing.assert_allclose(info["initial_ee_position"], pivot + [0, 0, bags.GRIP_HEIGHT])
    surface = info["work_surface"]
    shelf = info["a2_bags"]["shelf"]
    assert surface["center"][2] + surface["size"][2] / 2 == pytest.approx(shelf["top"])
    assert shelf["x"][0] < pivot[0] < shelf["x"][1] and shelf["y"][0] < pivot[1] < shelf["y"][1]
    # The ward turns 180 degrees and moves with the seeded shift, like every apparatus point.
    pose = info["ward_pose"]
    assert pose["yaw"] == 180.0 and pose["usd"].endswith("ward_02_preview_left.usdc")
    shift = pivot[:2] - np.asarray(bags.PIVOT_XY)
    np.testing.assert_allclose(pose["translate"][:2], np.asarray(bags.ROBOT_WARD) + shift)


def test_hooks_hang_bag_1_nearer_the_selector_and_both_clear_the_line_of_sight(config):
    task, layout = layout_for(config)
    info = layout.info["a2_bags"]
    hook1, hook2 = (np.asarray(h) for h in info["hooks"])
    pivot = np.asarray(info["pivot"])
    assert hook1[1] < hook2[1] and hook1[2] == pytest.approx(hook2[2])
    assert hook1[2] > pivot[2] + 0.4  # bags hang well above the selector
    # Neither bag hangs over the hand: both are offset sideways from the selector.
    for hook in (hook1, hook2):
        assert np.hypot(*(hook[:2] - pivot[:2])) > 0.18


def test_quantity_band_maps_to_the_visible_endpoint_and_empty_chamber(config):
    task = make_task("A2", "hard", 0, config)
    rate = task.rate
    start = bags.fluid_state(task.initial_quantity(), rate=rate, switched=False)
    assert start["bag1_level"] > 0.05 and start["chamber1"] == 1.0 and start["drip1"]
    at_open = bags.fluid_state(task.quantity(task.window_open), rate=rate, switched=False)
    assert at_open["bag1_level"] == pytest.approx(0.0, abs=1e-9) and at_open["chamber1"] == 1.0
    middle = 0.5 * (task.window_open + task.window_close)
    mid = bags.fluid_state(task.quantity(middle), rate=rate, switched=False)
    assert mid["chamber1"] == pytest.approx(0.5) and mid["air1"] == 0.0 and not mid["drip1"]
    at_close = bags.fluid_state(task.quantity(task.window_close), rate=rate, switched=False)
    assert at_close["chamber1"] == pytest.approx(0.0, abs=1e-9) and at_close["air1"] == 0.0
    late = bags.fluid_state(task.quantity(task.spec.max_time), rate=rate, switched=False)
    assert late["air1"] == pytest.approx(
        (task.spec.max_time - task.window_close) * bags.AIR_SPEED
    )


def test_bag_1_level_falls_monotonically_and_is_capped_at_full():
    levels = [bags.fluid_state(q, rate=1.0, switched=False)["bag1_level"] for q in range(0, 101, 5)]
    assert all(a >= b for a, b in zip(levels, levels[1:]))
    full = bags.BAG_SIZE[2] - bags.BAG_HEADER - bags.END_HEIGHT - 0.012
    assert levels[0] == pytest.approx(full)
    assert levels[-1] == 0.0


def test_switch_stops_bag_1_and_starts_bag_2():
    before = bags.fluid_state(90.0, rate=1.0, switched=False)
    assert before["drip1"] and not before["drip2"]
    after = bags.fluid_state(90.0, rate=1.0, switched=True, since_switch=2.0)
    assert not after["drip1"] and after["drip2"]
    # An early switch freezes fluid above END (bag 1 interrupted); bag 2 starts to drain.
    assert after["bag1_level"] == before["bag1_level"] > 0
    assert after["bag2_level"] == pytest.approx(before["bag2_level"] - 2.0 * bags.BAG2_DRAIN)
    late = bags.fluid_state(101.0, rate=1.0, switched=True, since_switch=0.0)
    assert late["air1"] > 0 and late["chamber1"] == 0.0  # a late switch keeps the air it let in


def test_selector_latch_needs_a_contact_driven_turn():
    target = bags.TARGET_ANGLE
    state = bags.selector_state(None, 0.0, False, False)
    assert state == {"moved": False, "contact_turn": False, "actuated": False}
    # A turn without robot contact (e.g. the bearing settling) never actuates.
    ghost = bags.selector_state(state, target, False, False)
    assert ghost["moved"] and not ghost["actuated"]
    # A contact-driven turn into the detent actuates and stays latched.
    state = bags.selector_state(state, 0.4, True, True)
    assert state["moved"] and state["contact_turn"] and not state["actuated"]
    state = bags.selector_state(state, target - bags.DETENT_TOLERANCE / 2, True, True)
    assert state["actuated"]
    state = bags.selector_state(state, 0.3, False, False)
    assert state["actuated"] and state["moved"]
    # Missing or misaligned measurements never look like the detent.
    fresh = bags.selector_state(None, 0.4, True, True)
    assert not bags.selector_state(fresh, float("nan"), True, True)["actuated"]
    assert not bags.selector_state(fresh, target, True, True, axis_error=0.2)["actuated"]
    assert not bags.selector_state(fresh, target, True, True, pivot_error=0.01)["actuated"]
    assert not bags.selector_state(fresh, target - 2 * bags.DETENT_TOLERANCE, True, True)["actuated"]


@pytest.mark.parametrize("level", LEVELS)
def test_oracle_plans_its_detent_for_the_window_middle(config, level):
    for seed in range(10):
        task = make_task("A2", level, seed, config)
        start = bags.switch_start_time(task)
        landing = start + bags.ACTUATION_DURATION
        middle = 0.5 * (task.window_open + task.window_close)
        assert start > 1.0 and landing == pytest.approx(middle)
        # Half the hard window (1.5 s) of margin on either side of the planned detent.
        assert landing - task.window_open >= 1.5 - 1e-9
        assert task.window_close - landing >= 1.5 - 1e-9


def _state(ee, obj_position, angle, grasped=False):
    obj = SimpleNamespace(position=np.asarray(obj_position, dtype=float), grasped=grasped)
    return SimpleNamespace(
        ee_position=np.asarray(ee, dtype=float), objects={bags.NAME: obj},
        controls={"selector_angle_rad": angle},
    )


def test_oracle_waits_open_then_pinches_turns_and_lets_go(config):
    from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory

    task, layout = layout_for(config)
    planner = HospitalTrajectory(config["control"], 1 / 30)
    pivot = np.asarray(layout.info["a2_bags"]["pivot"])
    grip = pivot + [0, 0, bags.GRIP_HEIGHT]
    task.time = bags.switch_start_time(task) - 0.5
    command = bags.command(planner, task, _state(grip, pivot, 0.0))
    assert planner.phase == "wait_window" and not command.close
    task.time += 0.6
    command = bags.command(planner, task, _state(grip, pivot, 0.0))
    assert planner.phase == "pinch_selector" and command.close
    assert any(e["type"] == "oracle_switch_start" for e in task.event_log)
    yaws = []
    for k in range(80):
        angle = min(bags.TARGET_ANGLE, 0.05 * k)
        command = bags.command(planner, task, _state(grip, pivot, angle, grasped=True))
        yaws.append(command.yaw)
    assert planner.phase == "turn_selector" and command.close
    assert all(b >= a for a, b in zip(yaws, yaws[1:]))
    assert max(yaws) <= bags.TARGET_ANGLE + 1e-9  # never drives past the hard stop
    task.goal_achieved = True
    command = bags.command(planner, task, _state(grip, pivot, bags.TARGET_ANGLE, grasped=True))
    assert planner.phase == "release" and not command.close


@pytest.mark.parametrize("seed", range(4))
def test_idle_arm_stands_clear_of_the_stand_and_the_trolley(config, seed):
    """Common rig: the idle arm 1 stands 0.60 m to +y, past the IV stand; the ward (bed side and
    its two-tier supply trolley) moved 0.45 m along the bed so the trolley clears it."""
    import json
    from pathlib import Path

    from roboclock.envs.hospital.bedside.ward import furniture, part_bounds, ward_path

    left, right = (np.asarray(m) for m in mount_positions(config, "A2"))
    pole = np.asarray(bags.POLE_XY)
    assert left[1] < pole[1] < right[1] and math.isclose(left[2], right[2])
    assert np.hypot(*(right[:2] - pole)) > 0.40  # the stand's hooks stay out of the parked arm
    _, layout = layout_for(config, seed=seed)
    pose = layout.info["ward_pose"]
    parts = json.loads(ward_path(config, pose["usd"]).with_suffix(".json").read_text())["parts"]
    lo, _ = part_bounds(pose, parts, set(furniture("supply_trolley")))
    assert lo[1] > right[1] + 0.13 + 0.02  # past the wheeled base's shell (0.13 m) with a margin
