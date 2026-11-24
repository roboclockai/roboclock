"""B5 anaphylaxis auto-injector: geometry, scoring and registration (not PhysX episodes)."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import autoinjector
from roboclock.envs.hospital.bedside.autoinjector import PEN_HALF, TIP_LENGTH, TIP_REST
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.bedside import autoinjector as rules
from roboclock.envs.hospital.tasks.bedside.autoinjector import (
    ACTIVATE_TRAVEL,
    HOLD,
    HOLD_AFTER,
    PEN,
    TIP,
    TIP_TRAVEL,
    AnaphylaxisAutoInjector,
)
from roboclock.envs.hospital.tasks.registry import RETIRED_BEDSIDE_TASKS, TASKS, make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState

DT = 0.004
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
INVERTED = np.array([0.0, 1.0, 0.0, 0.0])  # half a turn about x: the tip points up


@pytest.fixture
def task():
    task = make_task("B5", "hard", 3)
    task.bind(make_layout(task, load_config()))
    return task


def info(task):
    return task.layout.info["autoinjector"]


def injector(task, where="holder", travel=0.0, height=0.0, dxy=(0.0, 0.0), held=False, orientation=UPRIGHT):
    """The two bodies with the tip's end `height` above the pad (or resting in the holder) and the
    tip pushed `travel` into the barrel."""
    sign = -1.0 if orientation is INVERTED else 1.0
    if where == "holder":
        pen = task.layout.objects[PEN].position + [*dxy, height]
    else:
        pad = np.array(info(task)["pad"]["center"])
        end = pad + [*dxy, height]
        pen = end + [0.0, 0.0, sign * (TIP_TRAVEL + PEN_HALF - travel)]
        if orientation is INVERTED:  # the blunt end is on the pad instead
            pen = pad + [*dxy, height + PEN_HALF]
    tip = pen + [0.0, 0.0, sign * (TIP_REST + travel)]
    contacts = frozenset(["fixture:thigh_pad"]) if where == "pad" and height < 0.002 else frozenset()
    return (
        ObjectState(pen, orientation.copy(), grasped=held, robot_contact=held),
        ObjectState(tip, orientation.copy(), fixture_contacts=contacts),
    )


def step(task, bodies, duration=DT):
    """Advance the task clock and score one measured state per physics tick."""
    pen, tip = bodies
    for _ in range(max(1, round(duration / DT))):
        task.advance(DT)
        measured = autoinjector.geometry(task.layout, pen, tip)
        state = PhysicalState({PEN: pen, TIP: tip}, np.array([0.4, 0.0, 1.0]), controls={"autoinjector": measured})
        task.observe(state, DT)


def alarm(task):
    step(task, injector(task), task.spec.event_time + DT)
    assert task.event_time == pytest.approx(task.spec.event_time, abs=2 * DT)


def bring(task):
    """Grasp the injector, carry it over the pad and set its tip down."""
    step(task, injector(task, held=True), 0.2)
    step(task, injector(task, height=0.07, held=True), 0.2)
    step(task, injector(task, "pad", height=0.02, held=True), 0.3)
    step(task, injector(task, "pad", held=True), 0.1)


def press(task, duration, travel=0.009):
    step(task, injector(task, "pad", travel=travel, held=True), duration)


# --- registration and episode sampling -----------------------------------------------------------


def test_b5_is_the_anaphylaxis_autoinjector():
    assert TASKS["B5"] is AnaphylaxisAutoInjector
    assert AnaphylaxisAutoInjector.name == "anaphylaxis_autoinjector"
    assert AnaphylaxisAutoInjector.category == "event_reactive"
    assert "hang_bag_open_clamp" not in {cls.name for cls in TASKS.values()}
    assert "hang_bag_open_clamp" in RETIRED_BEDSIDE_TASKS


def test_alarm_time_is_seeded_and_the_same_at_every_level():
    lo, hi = load_config()["episode"]["event_time_range"]
    times = set()
    for seed in range(10):
        events = {make_task("B5", level, seed).spec.event_time for level in ("easy", "medium", "hard")}
        assert len(events) == 1
        times |= events
    assert len(times) == 10 and all(lo <= t <= hi for t in times)


def test_limits_shrink_with_difficulty_and_leave_room_for_the_hold():
    limits = [make_task("B5", level, 0).spec.response_limit for level in ("easy", "medium", "hard")]
    assert limits[0] > limits[1] > limits[2] > HOLD + 1.0
    assert limits[1] == pytest.approx(np.sqrt(limits[0] * limits[2]), abs=0.01)


def test_instruction_names_the_hold_and_the_limit_and_fits_the_prompt_budget(task):
    text = task.gen_instruction()
    assert f"{task.spec.response_limit:.1f} s" in text and f"{HOLD:g} s" in text
    assert len(text) <= 512


def test_old_roller_clamp_rows_are_retired_not_mistaken_for_the_new_b5():
    from roboclock.envs.hospital.results import normalize
    from roboclock.envs.hospital.stages import STAGES

    old = normalize({"task_id": "B5", "task_name": "hang_bag_open_clamp", "task_success": True})
    assert old["retired"] and old["task_id"] == "B5-hang_bag_open_clamp" and old["benchmark_id"] is None
    new = normalize({"task_id": "B5", "task_name": "anaphylaxis_autoinjector", "task_success": True})
    assert new["benchmark_id"] == "B5" and not new.get("retired")
    assert [s.name for s in STAGES["B5"]({"task_name": "hang_bag_open_clamp"})] == ["bag_hung"]
    assert [s.name for s in STAGES["B5"]({"task_name": "anaphylaxis_autoinjector"})] == [
        "injector_grasped",
        "tip_on_injection_pad",
        "injector_fired",
    ]


# --- layout ----------------------------------------------------------------------------------


def test_injector_stands_on_its_tip_in_the_holder_with_the_tip_extended(task):
    layout = task.layout
    pen, tip = layout.objects[PEN], layout.objects[TIP]
    floor = info(task)["holder"][2]
    assert tip.position[2] - TIP_LENGTH / 2 == pytest.approx(floor + 0.0005, abs=1e-6)
    assert pen.position[2] - PEN_HALF == pytest.approx(floor + 0.0005 + TIP_TRAVEL, abs=1e-6)
    measured = autoinjector.geometry(layout, *injector(task))
    assert measured["travel"] == pytest.approx(0.0, abs=1e-9)
    assert not measured["on_site"] and not measured["pressed"]
    assert (PEN, TIP) in [tuple(pair) for pair in layout.info["attached"]]


def test_holder_walls_surround_the_barrel_with_clearance(task):
    walls = {f.name: f for f in task.layout.fixtures if f.name.startswith("well_")}
    assert len(walls) == 4
    pen = task.layout.objects[PEN].position
    size = task.layout.shapes[PEN]
    del size
    gap_x = walls["well_xp"].position[0] - walls["well_xp"].size[0] / 2 - pen[0] - 0.0264 / 2
    gap_y = walls["well_yp"].position[1] - walls["well_yp"].size[1] / 2 - pen[1] - 0.0164 / 2
    assert 0.001 < gap_x < 0.004 and 0.001 < gap_y < 0.004


def test_pad_is_on_the_patient_and_below_the_holder(task):
    pad = np.array(info(task)["pad"]["center"])
    assert pad[2] > info(task)["mattress_top"] + 0.2  # on the thigh, not on the mattress
    assert pad[2] < info(task)["holder"][2]  # the carry goes down to it
    assert np.linalg.norm(pad[:2] - info(task)["holder"][:2]) > 0.2


def test_everything_the_policy_needs_is_in_the_head_image(task):
    from roboclock.envs.hospital.rig import head_pixels

    points = [
        info(task)["pad"]["center"],
        task.layout.objects[PEN].position,
        info(task)["monitor"],
        task.layout.indicators[0].position,
    ]
    pixels, depth = head_pixels(load_config(), "B5", points, conservative=True)
    assert np.all(depth > 0)
    assert np.all((pixels[:, 0] > 10) & (pixels[:, 0] < 630))
    assert np.all((pixels[:, 1] > 10) & (pixels[:, 1] < 470))


def test_start_and_work_poses_are_within_the_arm_reach(task):
    for point in (info(task)["pad"]["center"], task.layout.objects[PEN].position):
        assert 0.30 < np.linalg.norm(np.array(point)[:2]) < 0.50  # arm 0's mount is at the origin
    assert np.linalg.norm(np.array(task.layout.info["initial_ee_position"])[:2]) < 0.25


# --- measurement -------------------------------------------------------------------------------


def test_travel_is_measured_along_the_barrel(task):
    for travel in (0.0, 0.004, 0.009):
        measured = autoinjector.geometry(task.layout, *injector(task, "pad", travel=travel))
        assert measured["travel"] == pytest.approx(travel, abs=1e-9)
        assert measured["on_site"] and measured["aligned"]
    assert autoinjector.geometry(task.layout, *injector(task, "pad", travel=0.009))["pressed"]
    assert not autoinjector.geometry(task.layout, *injector(task, "pad", travel=0.007))["pressed"]


def test_on_site_needs_the_tip_end_on_the_pad(task):
    size = info(task)["pad"]["size"]
    assert not autoinjector.geometry(task.layout, *injector(task, "pad", height=0.02))["on_site"]
    off = (size[0] / 2 + 0.005, 0.0)
    assert not autoinjector.geometry(task.layout, *injector(task, "pad", dxy=off))["on_site"]
    edge = (size[0] / 2 - 0.012, 0.0)
    assert autoinjector.geometry(task.layout, *injector(task, "pad", dxy=edge))["on_site"]


def test_inverted_injector_is_the_blunt_end_on_the_pad(task):
    measured = autoinjector.geometry(task.layout, *injector(task, "pad", orientation=INVERTED))
    assert measured["blunt_on_site"] and not measured["on_site"] and not measured["aligned"]


def test_blood_pressure_model_crosses_the_alarm_at_the_event_and_recovers():
    # Real-rate lead-in (2026-11-23): already falling at the start, never above the baseline.
    assert rules.systolic_before_alarm(0.0, 8.0, 118.0) == pytest.approx(rules.ALARM_SYSTOLIC + 8.0 * rules.FALL_RATE)
    assert rules.systolic_before_alarm(0.0, 8.0, 90.2) == 90.2
    assert rules.systolic_before_alarm(8.0, 8.0, 118.0) == pytest.approx(rules.ALARM_SYSTOLIC)
    falling = rules.systolic_step(85.0, 1.0, None)
    assert falling == pytest.approx(85.0 - rules.FALL_RATE_ALARM)
    assert rules.systolic_step(85.0, 1.0, rules.RECOVERY_DELAY / 2) == falling
    assert rules.systolic_step(85.0, 1.0, rules.RECOVERY_DELAY) == pytest.approx(85.0 + rules.RECOVERY_RATE)
    assert rules.systolic_step(109.9, 1.0, 5.0) == rules.RECOVERY_TARGET
    assert rules.heart_rate(90.0, 118.0) > rules.heart_rate(118.0, 118.0) == 84


def test_monitor_shows_a_card_for_every_reading(task):
    lo, hi = int(rules.SYSTOLIC_FLOOR), int(rules.SYSTOLIC_RANGE[1])
    cards = set(range(lo, hi + 1, autoinjector.SYSTOLIC_STEP))
    for systolic in np.linspace(lo, hi, 50):
        task.systolic = float(systolic)
        assert autoinjector.monitor_state(task)[0] in cards


# --- scoring -----------------------------------------------------------------------------------


def test_injection_held_three_seconds_in_time_is_a_joint_success(task):
    alarm(task)
    bring(task)
    press(task, HOLD + 0.2)
    assert task.fire_time is not None and task.delivered_time == pytest.approx(task.fire_time + HOLD, abs=2 * DT)
    assert task.physical_success and task.temporal_success and task.joint_success
    m = task.metrics()
    assert m["response_latency"] == pytest.approx(task.delivered_time - task.event_time)
    assert m["grasp_time"] < m["site_time"] <= m["fire_time"] < m["delivered_time"]
    assert task.correct == {PEN, "site", "hold"}


def test_hold_is_timed_from_the_first_pressed_instant(task):
    alarm(task)
    bring(task)
    before = task.time
    press(task, 1.0)
    assert task.fire_time == pytest.approx(before + DT, abs=2 * DT)
    assert task.delivered_time is None and not task.goal_achieved


def test_lifting_before_the_hold_is_over_loses_the_dose(task):
    alarm(task)
    bring(task)
    press(task, HOLD - 0.5)
    step(task, injector(task, "pad", height=0.03, held=True), 0.1)
    assert "released_early" in task.physical_failures
    press(task, HOLD + 0.2)  # pressing again does not bring the dose back
    assert task.delivered_time is None and not task.physical_success and task.terminated


def test_easing_off_within_the_hold_travel_keeps_the_hold(task):
    alarm(task)
    bring(task)
    press(task, 1.0)
    press(task, HOLD, travel=rules.HOLD_TRAVEL + 0.0005)
    assert task.physical_success and not task.physical_failures


def test_tip_pushed_in_off_the_pad_is_a_wrong_site(task):
    alarm(task)
    step(task, injector(task, held=True), 0.2)
    size = info(task)["pad"]["size"]
    off = injector(task, "pad", travel=0.009, dxy=(0.0, -size[1]), held=True)
    step(task, off, 0.1)
    assert "wrong_site" in task.physical_failures and task.terminated and not task.physical_success


def test_pushing_the_tip_in_inside_the_holder_is_a_wrong_site(task):
    alarm(task)
    pen, tip = injector(task, held=True)
    pen.position[2] -= 0.009
    step(task, (pen, tip), 0.1)
    assert "wrong_site" in task.physical_failures


def test_a_tilted_injector_does_not_fire(task):
    alarm(task)
    bring(task)
    pen, tip = injector(task, "pad", travel=0.009, held=True)
    angle = 0.6
    tilted = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    axis = np.array([0.0, -np.sin(angle), np.cos(angle)])
    end = np.array(info(task)["pad"]["center"])
    tip_center = end + axis * TIP_LENGTH / 2
    pen_center = tip_center - axis * (TIP_REST + 0.009)
    bodies = (
        ObjectState(pen_center, tilted, grasped=True, robot_contact=True),
        ObjectState(tip_center, tilted.copy(), fixture_contacts=tip.fixture_contacts),
    )
    measured = autoinjector.geometry(task.layout, *bodies)
    assert measured["pressed"] and measured["on_site"] and not measured["aligned"]
    step(task, bodies, 0.5)
    assert task.fire_time is None and not task.physical_failures
    del pen


def test_blunt_end_on_the_pad_is_the_wrong_end(task):
    alarm(task)
    step(task, injector(task, held=True), 0.2)
    step(task, injector(task, "pad", orientation=INVERTED, held=True), 0.3)
    assert "wrong_end" in task.physical_failures


def test_late_injection_is_physical_but_not_temporal(task):
    alarm(task)
    step(task, injector(task), task.spec.response_limit - HOLD + 0.5)
    bring(task)
    press(task, HOLD + 0.2)
    assert task.physical_success and not task.temporal_success
    assert "late_response" in task.violations and task.metrics()["deadline_missed"]


def test_never_injected_is_late_and_not_physical(task):
    alarm(task)
    step(task, injector(task), task.spec.response_limit + 0.2)
    assert "late_response" in task.violations and not task.physical_success and not task.terminated


def test_touching_the_injector_before_the_alarm_is_an_early_action(task):
    step(task, injector(task), 1.0)
    step(task, injector(task, held=True), 0.1)
    assert "early_action" in task.violations
    alarm(task)
    bring(task)
    press(task, HOLD + 0.2)
    assert task.physical_success and not task.temporal_success


def test_moving_the_injector_before_the_alarm_is_an_early_action(task):
    step(task, injector(task, height=0.02), 0.1)
    assert "early_action" in task.violations


def test_monitor_alarm_follows_the_blood_pressure(task):
    step(task, injector(task), task.spec.event_time - 0.5)
    assert not task.alarm_active and task.systolic > rules.ALARM_SYSTOLIC
    alarm(task)
    step(task, injector(task), 0.2)
    assert task.alarm_active and task.displayed()[0] < rules.ALARM_SYSTOLIC
    bring(task)
    press(task, HOLD + rules.RECOVERY_DELAY + 2.5)
    assert not task.alarm_active and task.systolic > rules.ALARM_SYSTOLIC


def test_episode_ends_after_the_recovery_is_shown(task):
    alarm(task)
    bring(task)
    press(task, HOLD + 0.1)
    assert not task.terminated
    press(task, HOLD_AFTER)
    assert task.terminated


def test_dropped_injector_ends_the_episode_as_a_failure(task):
    alarm(task)
    pen, tip = injector(task)
    pen.position[2] = tip.position[2] = info(task)["mattress_top"] - 0.5
    step(task, (pen, tip), 0.1)
    assert task.dropped and task.terminated and not task.physical_success


# --- authoring ---------------------------------------------------------------------------------


def test_author_joins_the_tip_to_the_barrel_with_a_sprung_prismatic_joint(task):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for name in (PEN, TIP):
        UsdGeom.Xform.Define(stage, f"/World/Objects/{name}")
    backend = SimpleNamespace(stage=stage, appearance=None, layout=task.layout)
    autoinjector.author(backend)
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/epi_tip"))
    assert joint.GetLowerLimitAttr().Get() == 0.0
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(TIP_TRAVEL)
    assert joint.GetLocalPos0Attr().Get()[2] == pytest.approx(TIP_REST)
    drive = UsdPhysics.DriveAPI(joint.GetPrim(), "linear")
    assert drive.GetStiffnessAttr().Get() > 0 and drive.GetTargetPositionAttr().Get() < 0
    # At the firing travel the spring pushes back with more than the injector's weight.
    force = drive.GetStiffnessAttr().Get() * (ACTIVATE_TRAVEL - drive.GetTargetPositionAttr().Get())
    assert 2.0 < force < drive.GetMaxForceAttr().Get()
    del pxr


# --- seeded placement ----------------------------------------------------------------------------


def _layout(seed, level="hard", config=None):
    config = config or load_config()
    task = make_task("B5", level, seed, config)
    return make_layout(task, config)


def test_placement_is_deterministic_and_the_same_at_every_level():
    for seed in range(6):
        reference = _layout(seed, "easy")
        for level in ("easy", "medium", "hard"):
            layout = _layout(seed, level)
            assert layout.info["autoinjector"]["spatial"] == reference.info["autoinjector"]["spatial"]
            for name in (PEN, TIP):
                np.testing.assert_array_equal(layout.objects[name].position, reference.objects[name].position)
            np.testing.assert_array_equal(
                layout.info["autoinjector"]["pad"]["center"], reference.info["autoinjector"]["pad"]["center"]
            )


def test_placement_offsets_stay_inside_the_configured_ranges_and_vary():
    ranges = load_config()["tasks"]["B5"]["spatial"]
    assert set(ranges) == set(autoinjector.SPATIAL_KEYS)
    drawn = {key: [] for key in ranges}
    for seed in range(60):
        spatial = _layout(seed).info["autoinjector"]["spatial"]
        for key, (lo, hi) in ranges.items():
            assert lo <= spatial[key] <= hi
            drawn[key].append(spatial[key])
    for key, (lo, hi) in ranges.items():  # the seeds cover most of every range
        assert max(drawn[key]) - min(drawn[key]) > 0.8 * (hi - lo)


def test_tray_and_pad_move_relative_to_each_other_and_to_the_scene():
    gaps, pads = [], []
    for seed in range(40):
        layout = _layout(seed)
        own = layout.info["autoinjector"]
        pad, pen = np.array(own["pad"]["center"]), layout.objects[PEN].position
        gaps.append(pad[:2] - pen[:2])
        pads.append(pad[:2] - np.array(own["table"]["center"]))  # the table carries only the scene shift
        # The holder, the safety release and the tray move together.
        np.testing.assert_allclose(np.array(own["holder"][:2]), pen[:2], atol=1e-9)
        offset = np.array(own["tray"]["center"][:2]) - pen[:2]
        np.testing.assert_allclose(offset, np.subtract(autoinjector.TRAY_XY, autoinjector.HOLDER_XY), atol=1e-9)
    assert np.all(np.ptp(gaps, axis=0) > (0.06, 0.12)) and np.all(np.ptp(pads, axis=0) > (0.04, 0.08))


def test_pad_stays_on_the_outer_middle_of_the_thigh():
    for seed in range(60):
        x, y = _layout(seed).info["autoinjector"]["pad_on_patient"]
        assert 0.12 <= x <= 0.16  # outer half of the near thigh (its axis is at 0.12 m)
        assert 0.155 <= y - 0.07 and y + 0.07 <= 0.365  # the pad inside the thigh's middle half


def test_every_placement_is_reachable_and_in_the_head_image():
    from roboclock.envs.hospital.rig import head_pixels

    config = load_config()
    corners = [dict.fromkeys(autoinjector.SPATIAL_KEYS, u) for u in (0.0, 1.0)]
    corners += [
        {"tray_dx": 1.0, "tray_dy": 0.0, "pad_across": 0.0, "pad_along": 0.0, "patient_dx": 1.0, "patient_dy": 1.0},
        {"tray_dx": 0.0, "tray_dy": 1.0, "pad_across": 1.0, "pad_along": 1.0, "patient_dx": 0.0, "patient_dy": 0.0},
    ]
    for forced in [None, *corners]:
        config["tasks"]["B5"]["spatial_force"] = forced
        for seed in range(12):
            layout = _layout(seed, config=config)
            own = layout.info["autoinjector"]
            if forced:
                lo, hi = config["tasks"]["B5"]["spatial"]["tray_dx"]
                assert own["spatial"]["tray_dx"] == pytest.approx(lo + (hi - lo) * forced["tray_dx"])
            pad, pen = np.array(own["pad"]["center"]), layout.objects[PEN].position
            for point in (pad, pen):
                assert 0.30 < np.linalg.norm(point[:2]) < 0.52  # arm 0's mount is at the origin
            # The tray stays on the table and clear of the dial; the pad clears the table's edge.
            table, tray, size = np.array(own["table"]["center"]), np.array(own["tray"]["center"]), own["tray"]["size"]
            assert abs(tray[1] - table[1]) + size[1] / 2 < 0.394 / 2
            assert layout.indicators[0].position[0] - 0.045 > tray[0] + size[0] / 2
            assert pad[1] - 0.07 > table[1] + 0.394 / 2 - 0.03
            pixels, depth = head_pixels(config, "B5", [pad, pen, own["monitor"], layout.indicators[0].position], conservative=True)
            assert np.all(depth > 0)
            assert np.all((pixels[:, 0] > 10) & (pixels[:, 0] < 630) & (pixels[:, 1] > 10) & (pixels[:, 1] < 470))


def test_forcing_an_offset_leaves_the_rest_of_the_layout_unchanged():
    config = load_config()
    free = _layout(4, config=config)
    config["tasks"]["B5"]["spatial_force"] = {"tray_dx": 1.0}
    pinned = _layout(4, config=config)
    assert pinned.objects[PEN].mass == free.objects[PEN].mass
    for key in ("tray_dy", "pad_across", "pad_along", "patient_dx", "patient_dy"):
        assert pinned.info["autoinjector"]["spatial"][key] == free.info["autoinjector"]["spatial"][key]
    assert pinned.objects[PEN].position[0] > free.objects[PEN].position[0] - 1e-9


def test_metrics_report_the_sampled_placement(task):
    assert task.metrics()["spatial"] == task.layout.info["autoinjector"]["spatial"]
