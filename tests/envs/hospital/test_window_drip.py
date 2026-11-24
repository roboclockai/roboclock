"""A1 drip chamber module logic that needs no simulator: level mapping, marks, placement, controls."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import window_drip as drip
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import REACH, make_layout
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState
from roboclock.envs.lab.robot import mount_positions

LEVELS = ("easy", "medium", "hard")
SHOULDER_ABOVE_MOUNT = 0.1005
ROLLER_HALF_X = drip.ROLLER_HALF_X  # catalog iv_roller box, along the slide


def episode(level="hard", seed=0):
    config = load_config()
    task = make_task("A1", level, seed, config)
    task.bind(make_layout(task, config))
    return task, config


# --- level mapping and printed marks -------------------------------------------------------------


def test_level_maps_percent_to_liquid_height_and_clips():
    assert drip.liquid_height(0.0) == 0.0
    assert drip.liquid_height(100.0) == pytest.approx(drip.CHAMBER_HEIGHT)
    assert drip.liquid_height(50.0) == pytest.approx(drip.CHAMBER_HEIGHT / 2)
    assert drip.liquid_height(-5.0) == 0.0 and drip.liquid_height(130.0) == pytest.approx(
        drip.CHAMBER_HEIGHT
    )
    heights = [drip.liquid_height(q) for q in np.linspace(0, 100, 21)]
    assert all(b > a for a, b in zip(heights, heights[1:]))


def test_band_marks_sit_inside_the_chamber_at_the_task_band():
    task, _ = episode()
    marks = drip.band_marks(task.band)
    assert task.band == (55.0, 85.0)
    assert 0.0 < marks["band_low"] < marks["band_high"] < drip.CHAMBER_HEIGHT
    assert marks["band_low"] == pytest.approx(0.55 * drip.CHAMBER_HEIGHT)
    assert marks["band_high"] == pytest.approx(0.85 * drip.CHAMBER_HEIGHT)
    # The band is at least 4 cm tall on the trainer chamber, so it reads in the 60-degree head view.
    assert marks["band_high"] - marks["band_low"] >= 0.04


def test_scale_card_rows_order_over_band_low():
    rows = 600
    top, bottom, zero, full = drip.scale_rows((55.0, 85.0), rows)
    assert 0 < full < top < bottom < zero < rows  # image rows grow downward
    # The band's rows match the chamber marks (the card spans the chamber plus both caps).
    per_meter = rows / (drip.CHAMBER_HEIGHT + 2 * drip.CAP_HEIGHT)
    assert bottom - top == pytest.approx(0.30 * drip.CHAMBER_HEIGHT * per_meter, abs=1.0)


def test_drop_hangs_at_the_orifice_then_falls_to_the_surface():
    orifice, surface = 1.30, 1.22
    samples = [
        drip.drop_state(t, surface, orifice)
        for t in np.linspace(0, drip.DROP_PERIOD, 200, endpoint=False)
    ]
    zs = [z for z, _ in samples]
    assert max(zs) <= orifice and min(zs) >= surface - 1e-9
    hanging = [z for z, scale in samples if z >= orifice - drip.DROP_RADIUS - 1e-9]
    assert len(hanging) > 0.6 * len(samples)


# --- placement --------------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_hand_goals_are_within_reach_and_straddle_the_roller(level):
    for seed in range(10):
        task, config = episode(level, seed)
        layout = task.layout
        info = layout.info
        roller = layout.objects[drip.ROLLER_NAME].position
        origin = np.asarray(info["drip"]["origin"])
        mount = np.asarray(mount_positions(config, "A1")[0], dtype=float)
        shoulder = mount + [0.0, 0.0, SHOULDER_ABOVE_MOUNT]
        start = np.asarray(info["initial_ee_position"])
        assert REACH[0] <= np.hypot(*(start[:2] - mount[:2])) <= REACH[1]
        goals = drip.hand_goals(origin)
        # Measured X5 reach boundary is ~0.62 m from the shoulder; keep a margin for the slides.
        for goal in [start, *goals.values()]:
            assert np.linalg.norm(goal - shoulder) < 0.55
        # The start pose hovers over the CLOSED roller; the fingers open along the slide.
        np.testing.assert_allclose(start[:2], roller[:2])
        assert info["initial_ee_yaw"] == pytest.approx(np.pi / 2)
        assert info["initial_gripper_width"] == pytest.approx(drip.GRIP_WIDTH)
        # Straddling, each finger's inner face clears the roller by the slack, with no contact.
        assert drip.SLACK == pytest.approx(drip.FINGER_HALF_GAP - ROLLER_HALF_X)
        assert 0.004 < drip.SLACK < 0.012
        np.testing.assert_allclose(goals["closed"][:2], roller[:2])
        # The pull and the push drive the roller past its stops by the overdrive only.
        front_face = goals["pulled"][0] + drip.FINGER_HALF_GAP
        assert front_face == pytest.approx(origin[0] + ROLLER_HALF_X - drip.OVERDRIVE)
        rear_face = goals["pushed"][0] - drip.FINGER_HALF_GAP
        assert rear_face == pytest.approx(origin[0] + drip.TRAVEL - ROLLER_HALF_X + drip.OVERDRIVE)
        # The fingertips (12.57 mm below the tool point) stay above the rails and the end stops.
        rail_top = roller[2] - 0.029 + 0.019 / 2
        assert goals["closed"][2] - 0.01257 > rail_top + 0.003


def test_roller_starts_closed_and_rides_in_its_organizer_on_the_support_collider():
    task, _ = episode()
    layout = task.layout
    roller = layout.objects[drip.ROLLER_NAME].position
    origin = np.asarray(layout.info["drip"]["origin"])
    np.testing.assert_allclose(roller, origin + [drip.TRAVEL, 0.0, 0.0])  # at the CLOSED end
    low, high = layout.workspace_bounds
    for x in (origin[0], origin[0] + drip.TRAVEL):  # both ends of travel lie over the tray
        assert low[0] < x - ROLLER_HALF_X and x + ROLLER_HALF_X < high[0]
    assert low[1] < roller[1] < high[1]
    assert high[2] <= roller[2] - 0.011 + 1e-6  # tray top under the roller's bottom face
    stops = {f.name: f for f in layout.fixtures if f.name.startswith("roller_stop")}
    assert (
        stops["roller_stop_open"].position[0]
        < origin[0]
        < origin[0] + drip.TRAVEL
        < stops["roller_stop_closed"].position[0]
    )


def test_bag_chamber_and_organizer_on_the_table_in_the_head_view():
    from roboclock.envs.hospital.rig import head_pixels

    task, config = episode()
    info = task.layout.info["drip"]
    frame = drip.chamber_frame(info)
    hook, origin, table = np.asarray(info["hook"]), np.asarray(info["origin"]), info["table"]
    # Bag above the chamber above the organizer; the tray rests on the table top.
    assert hook[2] > frame["bag_bottom"] > frame["top"] > frame["floor"] > origin[2] + 0.06
    assert origin[2] - 0.047 == pytest.approx(table["top"])
    for xy in (hook[:2], origin[:2], np.asarray(info["pole"])):
        assert table["x"][0] < xy[0] < table["x"][1] and table["y"][0] < xy[1] < table["y"][1]
    # The whole bag, the chamber and the roller are inside the 640x480 head image, and the chamber
    # stands left of the roller (the arm comes in from the right and never covers it).
    bag_top = hook[2] + drip.BAG_HANG
    points = [
        [hook[0], hook[1] + drip.BAG_SIZE[0] / 2, bag_top],
        [hook[0], hook[1] - drip.BAG_SIZE[0] / 2, bag_top],
        [*frame["xy"], frame["floor"]],
        origin + [drip.TRAVEL, 0.0, 0.0],
    ]
    u, v = head_pixels(config, "A1", points)[0].T
    assert np.all((u > 20) & (u < 620) & (v > 10) & (v < 470))
    assert u[3] - u[2] > 50


def test_ward_moves_with_the_seed_shift_and_is_reproducible():
    a, _ = episode("easy", 3)
    b, _ = episode("hard", 3)
    pose_a, pose_b = a.layout.info["ward_pose"], b.layout.info["ward_pose"]
    assert pose_a == pose_b and pose_a["usd"] == drip.WARD_USD
    shift = np.asarray(a.layout.info["drip"]["shift"])
    np.testing.assert_allclose(
        pose_a["translate"][:2], np.asarray(drip.WARD_POSE["translate"][:2]) + shift
    )
    assert a.layout.info["hide_bench"] is True and a.layout.info["bedside_task"] == "window_drip"


# --- measured controls -----------------------------------------------------------------------------


def roller_state(task, position, contact=False):
    origin = np.asarray(task.layout.info["drip"]["origin"])
    return ObjectState(
        origin + [position, 0.0, 0.0], np.array([1.0, 0.0, 0.0, 0.0]), robot_contact=contact
    )


def measure_at(backend, task, position, contact=False):
    state = PhysicalState({drip.ROLLER_NAME: roller_state(task, position, contact)}, np.zeros(3))
    return drip.measure(backend, state)


def test_controls_open_then_actuate_only_back_at_the_closed_end():
    task, _ = episode()
    backend = SimpleNamespace(layout=task.layout, task=task)
    T = drip.TRAVEL
    controls = measure_at(backend, task, T)
    assert not controls["device_moved"] and not controls["device_opened"]
    assert not controls["device_actuated"] and controls["roller_at_closed_end"]
    assert measure_at(backend, task, T - 0.001)["device_moved"] is False  # below 1.5 mm
    # Reaching the OPEN end without any robot contact (a spurious motion) never opens the clamp.
    assert measure_at(backend, task, 0.0)["device_opened"] is False
    assert measure_at(backend, task, T - 0.010, contact=True)["device_moved"] is True
    assert measure_at(backend, task, 2 * drip.END_TOLERANCE)["device_opened"] is False
    opened = measure_at(backend, task, 0.5 * drip.END_TOLERANCE)
    assert opened["device_opened"] and opened["roller_at_open_end"]
    assert not opened["device_actuated"]
    almost = measure_at(backend, task, T - 2 * drip.END_TOLERANCE)
    assert almost["device_opened"] and not almost["device_actuated"]
    done = measure_at(backend, task, T - 0.5 * drip.END_TOLERANCE)
    assert done["device_actuated"] and done["roller_at_closed_end"]
    # Latched: a roller that settles back a little stays actuated, and opened stays opened.
    later = measure_at(backend, task, T - 0.004)
    assert later["device_actuated"] and later["device_opened"] and later["device_moved"]
    assert later["chamber_level_pct"] == pytest.approx(task.quantity())


def test_latches_reset_with_a_new_episode():
    task, _ = episode()
    backend = SimpleNamespace(layout=task.layout, task=task)
    measure_at(backend, task, 0.0, contact=True)
    assert measure_at(backend, task, drip.TRAVEL)["device_actuated"]
    fresh, _ = episode()
    backend.task, backend.layout = fresh, fresh.layout
    controls = measure_at(backend, fresh, drip.TRAVEL)
    assert not controls["device_opened"] and not controls["device_actuated"]


# --- the process, relative to the opening -------------------------------------------------------


def test_level_is_still_until_the_opening_then_crosses_the_band_in_the_window():
    task, _ = episode("medium", 2)
    lead, width = task.fill_lead, task.window_width
    assert 3.0 <= lead <= 5.0 and width == pytest.approx(4.5)
    assert task.window_open == np.inf and task.quantity(10.0) == task.initial_quantity()
    assert task.initial_quantity() == pytest.approx(task.band[0] - lead * task.rate)
    task.time, task.open_time = 2.0, 2.0
    assert task.window_open == pytest.approx(2.0 + lead)
    assert task.window_close == pytest.approx(2.0 + lead + width)
    assert task.quantity(2.0 + lead) == pytest.approx(task.band[0])
    assert task.quantity(2.0 + lead + width) == pytest.approx(task.band[1])


def test_fill_lead_is_the_same_at_every_level_of_a_seed():
    for seed in range(5):
        leads = {episode(level, seed)[0].fill_lead for level in LEVELS}
        assert len(leads) == 1


def drive(task, open_at, close_at, dt=1 / 120, duration=0.24):
    """Measured slides: open over `duration` ending at `open_at`, close ending at `close_at`."""
    backend = SimpleNamespace(layout=task.layout, task=task)
    T = drip.TRAVEL
    while not task.terminated:
        task.advance(dt)
        t = task.time
        if t < open_at - duration:
            position, contact = T, False
        elif t <= open_at:
            position, contact = T * (open_at - t) / duration, True
        elif close_at is None or t < close_at - duration:
            position, contact = 0.0, False
        elif t <= close_at:
            position, contact = T * (1 - (close_at - t) / duration), True
        else:
            position, contact = T, False
        state = PhysicalState(
            {drip.ROLLER_NAME: roller_state(task, position, contact)}, np.zeros(3)
        )
        state.controls.update(drip.measure(backend, state))
        task.evaluate(state, dt)
    return task


def test_measured_open_and_close_score_through_the_window_task():
    task, _ = episode("hard", 1)
    open_at = 1.6
    middle = open_at + task.fill_lead + task.window_width / 2
    drive(task, open_at, middle)
    assert task.open_time == pytest.approx(open_at, abs=0.02)
    assert task.joint_success and task.metrics()["quantity_outcome"] == "in_band"
    assert task.action_time == pytest.approx(middle, abs=0.02)
    assert task.metrics()["open_time"] == pytest.approx(open_at, abs=0.02)


def test_early_and_late_closings_fail_on_timing_only():
    task, _ = episode("hard", 1)
    early = drive(task, 1.6, 1.6 + task.fill_lead - 0.5)
    assert early.outcome() == "short" and not early.temporal_success
    assert early.quantity() < early.band[0]
    task, _ = episode("hard", 1)
    late = drive(task, 1.6, 1.6 + task.fill_lead + task.window_width + 0.5)
    assert late.outcome() == "overshoot" and not late.temporal_success
    assert late.quantity() > late.band[1]


def test_never_opening_is_no_action_and_the_level_stays_low():
    task, _ = episode("medium", 0)
    backend = SimpleNamespace(layout=task.layout, task=task)
    while not task.terminated:
        task.advance(1 / 60)
        state = PhysicalState(
            {drip.ROLLER_NAME: roller_state(task, drip.TRAVEL, False)}, np.zeros(3)
        )
        state.controls.update(drip.measure(backend, state))
        task.evaluate(state, 1 / 60)
    assert task.outcome() == "no_action" and not task.joint_success
    assert task.quantity() == task.initial_quantity() < task.band[0]
    assert task.metrics()["window_open"] is None


@pytest.mark.parametrize("level", LEVELS)
def test_episode_leaves_room_for_the_opening_and_oracle_timing_lands_in_every_window(level):
    for seed in range(20):
        task, _ = episode(level, seed)
        # Opening by OPEN_AT + 1 s still leaves the whole window and the tail inside max_time.
        open_at = drip.OPEN_AT + 1.0
        assert open_at + task.fill_lead + task.window_width + 1.5 < task.spec.max_time
        start = drip.slide_start(open_at + task.fill_lead, open_at + task.fill_lead + task.window_width)
        closes = start + drip.PUSH_LEAD
        assert closes == pytest.approx(open_at + task.fill_lead + task.window_width / 2)
        # Even a slide 0.3 s off its calibrated lead stays inside the narrowest (3 s) window.
        assert open_at + task.fill_lead + 0.3 < closes < open_at + task.fill_lead + task.window_width - 0.3
        # A missed window visibly overfills (up to the brim: the level is capped at 100 %).
        task.open_time = open_at
        overfill = task.quantity(task.spec.max_time)
        assert overfill >= task.band[1] + 10.0 and overfill <= 100.0
        assert 5.0 - 1e-9 <= task.initial_quantity() < task.band[0] - 10.0


def test_module_copies_mechanics_instead_of_importing_other_bedside_modules():
    source = Path(drip.__file__).read_text()
    for other in ("roller_clamp", "stopcock", "pump_panel", "iv_bag", "ward_c"):
        assert f"bedside.{other}" not in source and f"bedside import {other}" not in source


def test_band_and_chamber_read_in_the_head_image():
    """The band spans >= 10 px vertically and the chamber >= 35 px in the 640x480 head image."""
    from roboclock.envs.hospital.rig import head_pixels

    task, config = episode()
    info = task.layout.info["drip"]
    frame, marks = drip.chamber_frame(info), drip.band_marks(task.band)
    (cx, cy) = frame["xy"]
    points = [
        [cx, cy, frame["floor"] + marks["band_low"]],
        [cx, cy, frame["floor"] + marks["band_high"]],
        [cx, cy, frame["floor"]],
        [cx, cy, frame["floor"] + drip.CHAMBER_HEIGHT],
    ]
    uv, _ = head_pixels(config, "A1", points)
    assert uv[0, 1] - uv[1, 1] >= 10.0
    assert uv[2, 1] - uv[3, 1] >= 35.0
