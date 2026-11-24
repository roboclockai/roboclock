"""B6 haemorrhage tourniquet: geometry, process model, scoring and registration (not PhysX episodes)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import tourniquet
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.bedside import tourniquet as rules
from roboclock.envs.hospital.tasks.bedside.tourniquet import (
    BLEED_RATE,
    CLIP_TOLERANCE,
    HOLD_AFTER,
    OCCLUDE_ANGLE,
    OVERTURN,
    ROD,
    SECURE_DWELL,
    TURN_ANGLE,
    HaemorrhageTourniquet,
    bleed_rate,
    clip_engaged,
    pool_radius,
)
from roboclock.envs.hospital.tasks.registry import TASKS, make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState

DT = 0.004


@pytest.fixture
def task():
    task = make_task("B6", "hard", 3)
    task.bind(make_layout(task, load_config()))
    return task


def info(task):
    return task.layout.info["tourniquet"]


def yaw_quaternion(angle, start=tourniquet.START_YAW):
    """The rod's orientation after turning `angle` counterclockwise from its start yaw."""
    yaw = start + angle
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def step(task, angle=0.0, duration=DT, contact=False, speed=0.0):
    """Advance the task clock and score one measured windlass state per physics tick."""
    obj = ObjectState(
        np.array(info(task)["pivot"]), yaw_quaternion(angle, info(task)["start_yaw_rad"]),
        grasped=contact, robot_contact=contact,
    )
    measured = {
        "angle_rad": float(angle), "speed_rad_s": float(speed), "contact": bool(contact),
        "grasped": bool(contact), "clip_engaged": clip_engaged(angle, task.turn_required),
    }
    for _ in range(max(1, round(duration / DT))):
        task.advance(DT)
        state = PhysicalState({ROD: obj}, np.array([0.4, 0.0, 1.0]), controls={"tourniquet": measured})
        task.observe(state, DT)


def bleed(task, wait=0.0):
    """Run to the seeded event (and `wait` seconds on) with the rod untouched."""
    step(task, 0.0, task.spec.event_time + DT + wait)
    assert task.event_time == pytest.approx(task.spec.event_time, abs=2 * DT)


def turn(task, to=None, start=0.0, rate=1.1):
    """Turn the pinched rod from `start` to `to` (default: the episode's clip angle)."""
    to = task.turn_required if to is None else to
    angle = start
    while angle < to - 1e-9:
        angle = min(to, angle + rate * DT)
        step(task, angle, DT, contact=True, speed=rate)


# --- registration and episode sampling -----------------------------------------------------------


def test_b6_is_the_haemorrhage_tourniquet():
    assert TASKS["B6"] is HaemorrhageTourniquet
    assert HaemorrhageTourniquet.name == "haemorrhage_tourniquet"
    assert HaemorrhageTourniquet.category == "event_reactive"
    assert "pump_panel_response" not in {cls.name for cls in TASKS.values()}


def test_bleeding_start_is_seeded_and_the_same_at_every_level():
    lo, hi = load_config()["episode"]["event_time_range"]
    times = set()
    for seed in range(10):
        events = {make_task("B6", level, seed).spec.event_time for level in ("easy", "medium", "hard")}
        assert len(events) == 1
        times |= events
    assert len(times) == 10 and all(lo <= t <= hi for t in times)


def test_limits_shrink_with_difficulty_and_never_fall_below_one_second():
    limits = [make_task("B6", level, 0).spec.response_limit for level in ("easy", "medium", "hard")]
    assert limits[0] > limits[1] > limits[2] >= 1.0
    assert limits[1] == pytest.approx(np.sqrt(limits[0] * limits[2]), abs=0.01)


def test_easy_limit_is_the_documented_application_time():
    # Experienced providers: CAT on a thigh bleeding trainer in 24 +- 4 s (Lechner et al. 2024).
    assert make_task("B6", "easy", 0).spec.response_limit == pytest.approx(24.0)


def test_instruction_names_the_limit_and_fits_the_prompt_budget(task):
    text = task.gen_instruction()
    assert f"{task.spec.response_limit:.1f} s" in text
    assert len(text) <= 512


def test_old_pump_panel_rows_are_retired_not_mistaken_for_the_new_b6():
    from roboclock.envs.hospital.results import normalize
    from roboclock.envs.hospital.stages import STAGES

    for recorded in ("B6", "B7"):  # B7 until 2026-11-19, B6 from then until 2026-11-20
        old = normalize({"task_id": recorded, "task_name": "pump_panel_response", "task_success": True})
        assert old["retired"] and old["benchmark_id"] is None
        assert old["task_id"] == "B6-pump_panel_response" and old["recorded_task_id"] == recorded
    start = normalize({"task_id": "B6", "task_name": "hang_and_start_infusion", "task_success": True})
    assert start["task_id"] == "A4"
    new = normalize({"task_id": "B6", "task_name": "haemorrhage_tourniquet", "task_success": True})
    assert new["benchmark_id"] == "B6" and not new.get("retired")
    assert [s.name for s in STAGES["B6"]({"task_name": "pump_panel_response"})] == ["requested_key_moved"]
    assert [s.name for s in STAGES["B6"]({"task_name": "haemorrhage_tourniquet"})] == [
        "windlass_turned",
        "bleeding_stopped",
    ]


# --- layout ----------------------------------------------------------------------------------


def test_layout_puts_the_rod_on_its_pivot_along_the_limb(task):
    obj = task.layout.objects[ROD]
    assert np.allclose(obj.position, info(task)["pivot"], atol=0.003)
    angle, tilt = tourniquet.rod_angle(obj.orientation, None, info(task)["start_yaw_rad"])
    assert angle == pytest.approx(0.0, abs=1e-6) and tilt == pytest.approx(0.0, abs=1e-6)


def test_rod_clears_the_plate_and_sits_on_top_of_the_thigh(task):
    g = info(task)
    assert g["pivot"][2] - tourniquet.ROD_HEIGHT / 2 - g["plate_top"] >= 0.006  # room for fingertips
    shift = g["placement"]["band_shift_m"]
    assert g["plate_top"] - g["mattress_top"] == pytest.approx(
        tourniquet.THIGH[2] + tourniquet.THIGH_TOP_SLOPE * shift + 0.010, abs=1e-6
    )
    wound, pool = np.array(g["wound"]), np.array(g["pool"])
    assert wound[1] < g["pivot"][1] - 0.08  # the wound is distal to the band
    assert pool[2] < wound[2] and pool[2] == pytest.approx(g["mattress_top"] + 0.0095, abs=1e-6)


def test_everything_the_policy_needs_is_in_the_head_image(task):
    from roboclock.envs.hospital.rig import head_pixels

    g = info(task)
    points = [g["pivot"], g["wound"], g["pool"], g["display"], task.layout.indicators[0].position]
    pixels, depth = head_pixels(load_config(), "B6", points, conservative=True)
    assert np.all(depth > 0)
    assert np.all((pixels[:, 0] > 10) & (pixels[:, 0] < 630))
    assert np.all((pixels[:, 1] > 10) & (pixels[:, 1] < 470))


def test_start_and_work_poses_are_within_the_arm_reach(task):
    reach = np.linalg.norm(np.array(info(task)["pivot"])[:2])  # arm 0's mount is at the origin
    assert 0.33 < reach < 0.50
    assert np.linalg.norm(np.array(task.layout.info["initial_ee_position"])[:2]) < 0.25


def test_turn_stays_inside_the_joint_limits_at_every_seed():
    # PhysX clamps revolute limits to +-180 degrees: the joint angle is the rod's world yaw.
    config = load_config()
    for seed in range(40):
        t = make_task("B6", "hard", seed, config)
        t.bind(make_layout(t, config))
        g = info(t)
        lower, upper = tourniquet.joint_limits(g)
        start, clip = math.degrees(g["start_yaw_rad"]), math.degrees(g["clip_yaw_rad"])
        assert -180.0 < lower < start < clip + math.degrees(OVERTURN) < upper < 180.0
        assert tourniquet.drive_values(g, "strap")["target"] == pytest.approx(start)
        assert tourniquet.drive_values(g, "clip")["target"] == pytest.approx(clip)
        # The X5 wrist (joint 6, +-2 rad) follows the hand yaw from the start to the clip.
        assert start > -106.0 and clip + 2.0 < 104.0


# --- seeded placement ----------------------------------------------------------------------------


def placement_ranges():
    return load_config()["tasks"]["B6"]["placement"]


def test_placement_ranges_are_configured_not_hard_coded():
    ranges = placement_ranges()
    assert set(ranges) == set(tourniquet.PLACEMENT_KEYS)
    assert ranges["band_shift_m"][1] - ranges["band_shift_m"][0] >= 0.04  # several cm along the thigh
    assert all(lo < hi for lo, hi in ranges.values())


def test_placement_is_deterministic_and_the_same_at_every_level():
    config = load_config()
    for seed in range(10):
        drawn = []
        for level in ("easy", "medium", "hard", "hard"):
            t = make_task("B6", level, seed, config)
            t.bind(make_layout(t, config))
            drawn.append((info(t)["placement"], info(t)["pivot"], info(t)["start_yaw_rad"]))
        assert all(d == drawn[0] for d in drawn)


def test_placement_varies_with_the_seed_inside_its_ranges():
    config, ranges = load_config(), placement_ranges()
    seen = {key: [] for key in ranges}
    for seed in range(60):
        t = make_task("B6", "medium", seed, config)
        t.bind(make_layout(t, config))
        for key, (lo, hi) in ranges.items():
            value = info(t)["placement"][key]
            assert lo <= value <= hi
            seen[key].append(value)
    for key, (lo, hi) in ranges.items():  # the draws cover most of each range
        assert max(seen[key]) - min(seen[key]) > 0.8 * (hi - lo)


def test_turn_to_the_clip_follows_the_rod_start_and_clip_angles():
    config = load_config()
    turns = []
    for seed in range(30):
        t = make_task("B6", "easy", seed, config)
        t.bind(make_layout(t, config))
        p, g = info(t)["placement"], info(t)
        assert math.degrees(g["turn_required_rad"]) == pytest.approx(180.0 + p["clip_deg"] - p["rod_start_deg"])
        assert math.degrees(g["start_yaw_rad"]) == pytest.approx(-90.0 + p["leg_yaw_deg"] + p["rod_start_deg"])
        assert t.turn_required == pytest.approx(g["turn_required_rad"])
        assert t.occlude_angle == pytest.approx(t.turn_required - rules.OCCLUDE_BEFORE_CLIP)
        assert t.metrics()["placement"] == p
        turns.append(math.degrees(t.turn_required))
    assert 164.0 <= min(turns) < 175.0 and 185.0 < max(turns) <= 196.0


def test_band_and_wound_move_together_along_the_thigh():
    config = load_config()
    base = tourniquet.geometry(config)
    for shift in (-0.03, 0.02):
        g = tourniquet.geometry(config, placement={"band_shift_m": shift})
        assert g["pivot"][1] - base["pivot"][1] == pytest.approx(shift)
        assert g["wound"][1] - base["wound"][1] == pytest.approx(shift)
        assert np.allclose(g["origin"], base["origin"])  # the patient stays where it lies


def test_leg_yaw_turns_the_patient_about_the_windlass():
    config = load_config()
    base = tourniquet.geometry(config)
    g = tourniquet.geometry(config, placement={"leg_yaw_deg": 4.0})
    assert np.allclose(g["pivot"], base["pivot"])
    d0, d1 = base["wound"][:2] - base["pivot"][:2], g["wound"][:2] - g["pivot"][:2]
    assert np.linalg.norm(d1) == pytest.approx(np.linalg.norm(d0))
    assert math.degrees(math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])) == pytest.approx(4.0)


def test_cues_stay_in_view_and_in_reach_at_the_range_extremes():
    from itertools import product

    from roboclock.envs.hospital.rig import head_pixels

    config, ranges = load_config(), placement_ranges()
    jitter = config["scene"].get("bedside_spatial_jitter", config["scene"]["spatial_jitter"])
    for corner in product(*[ranges[key] for key in tourniquet.PLACEMENT_KEYS]):
        placement = dict(zip(tourniquet.PLACEMENT_KEYS, corner))
        for shift in product((-jitter, jitter), repeat=2):
            g = tourniquet.geometry(config, shift, placement=placement)
            pixels, depth = head_pixels(config, "B6", [g["pivot"], g["wound"], g["pool"]], conservative=True)
            assert np.all(depth > 0)
            assert np.all((pixels[:, 0] > 10) & (pixels[:, 0] < 630))
            assert np.all((pixels[:, 1] > 10) & (pixels[:, 1] < 470))
            assert 0.33 < np.linalg.norm(g["pivot"][:2]) < 0.50  # arm 0's mount is at the origin


# --- measurement and process model -------------------------------------------------------------


def test_rod_angle_is_measured_from_the_start_and_unwrapped():
    assert tourniquet.rod_angle(yaw_quaternion(0.0))[0] == pytest.approx(0.0, abs=1e-9)
    assert tourniquet.rod_angle(yaw_quaternion(1.0), 0.9)[0] == pytest.approx(1.0)
    # Past the half-turn the yaw wraps; the angle follows the previous measurement.
    assert tourniquet.rod_angle(yaw_quaternion(math.radians(195.0)), 3.2)[0] == pytest.approx(math.radians(195.0))
    assert tourniquet.rod_angle(yaw_quaternion(-0.03), 0.0)[0] == pytest.approx(-0.03)
    tilted = np.array([math.cos(0.1), math.sin(0.1), 0.0, 0.0])
    assert tourniquet.rod_angle(tilted)[1] == pytest.approx(0.2, abs=1e-6)
    assert math.isnan(tourniquet.rod_angle([0.0, 0.0, 0.0, 0.0])[0])


def test_bleeding_falls_with_the_turn_and_stops_before_the_clip():
    assert bleed_rate(0.0) == pytest.approx(BLEED_RATE)
    assert bleed_rate(OCCLUDE_ANGLE / 2) == pytest.approx(BLEED_RATE / 2)
    assert bleed_rate(OCCLUDE_ANGLE) == 0.0 and bleed_rate(TURN_ANGLE) == 0.0
    assert bleed_rate(math.radians(160.0), math.radians(166.0)) > 0.0  # a longer turn stops later
    assert bleed_rate(-0.05) == pytest.approx(BLEED_RATE)
    assert bleed_rate(float("nan")) == pytest.approx(BLEED_RATE)
    assert OCCLUDE_ANGLE < TURN_ANGLE - CLIP_TOLERANCE  # secured beyond where the bleeding stops


def test_bleed_rate_matches_the_cited_trainer_figure():
    # Lechner et al. 2024: 99 +- 17 mL lost during a 24 +- 4 s application.
    assert BLEED_RATE == pytest.approx(99.0 / 24.0, abs=0.2)


def test_clip_captures_the_rod_only_near_the_half_turn():
    assert not clip_engaged(TURN_ANGLE - CLIP_TOLERANCE - 0.01)
    assert clip_engaged(TURN_ANGLE - CLIP_TOLERANCE + 0.001) and clip_engaged(TURN_ANGLE + 0.1)
    assert not clip_engaged(float("nan"))


def test_pool_grows_with_the_lost_volume():
    assert pool_radius(0.0) == 0.0
    assert pool_radius(40.0) == pytest.approx(2 * pool_radius(10.0))
    assert 0.04 < pool_radius(10.0) < 0.08 and pool_radius(100.0) < 0.20


def test_display_follows_the_task(task):
    assert tourniquet.display_state(task) == "ok"
    bleed(task, 0.2)
    assert tourniquet.display_state(task) == "alarm"
    turn(task, task.occlude_angle + 0.02)
    assert tourniquet.display_state(task) == "stopped"
    turn(task, task.turn_required, start=task.occlude_angle + 0.02)
    step(task, task.turn_required, SECURE_DWELL + 0.05)
    assert tourniquet.display_state(task) == "secured"


# --- scoring -----------------------------------------------------------------------------------


def test_securing_the_rod_in_time_is_a_joint_success(task):
    bleed(task, 1.0)
    turn(task)
    step(task, task.turn_required, SECURE_DWELL + 0.05)
    assert task.goal_achieved and task.physical_success and task.temporal_success and task.joint_success
    assert task.metrics()["response_latency"] < task.spec.response_limit
    assert task.correct == {"windlass", "clip"}


def test_securing_is_timed_at_the_first_released_instant_not_after_the_dwell(task):
    bleed(task, 1.0)
    turn(task)
    released = task.time
    step(task, task.turn_required, SECURE_DWELL + 0.2)
    assert task.secure_time == pytest.approx(released + DT, abs=2 * DT)
    assert task.action_time == task.secure_time


def test_holding_the_rod_at_the_clip_is_not_secured(task):
    bleed(task, 0.5)
    turn(task)
    step(task, task.turn_required, 1.0, contact=True)
    assert task.secure_time is None and not task.goal_achieved
    assert task.occlusion_time is not None  # the bleeding has stopped, though


def test_blood_loss_integrates_the_bleeding_until_it_stops(task):
    bleed(task, 2.0)
    assert task.blood_lost == pytest.approx(2.0 * BLEED_RATE, rel=0.02)
    turn(task)
    lost = task.blood_lost
    step(task, task.turn_required, 1.0)
    assert task.blood_lost == pytest.approx(lost)  # nothing more once occluded
    assert task.metrics()["blood_lost_at_secure_ml"] == pytest.approx(lost)


def test_late_securing_is_physical_but_not_temporal(task):
    bleed(task, task.spec.response_limit + 0.5)
    turn(task)
    step(task, task.turn_required, SECURE_DWELL + 0.05)
    assert task.physical_success and not task.temporal_success
    assert "late_response" in task.violations


def test_never_secured_is_late_and_not_physical(task):
    bleed(task, task.spec.response_limit + 1.0)
    assert not task.physical_success and "late_response" in task.violations


def test_touching_the_rod_before_the_bleeding_is_an_early_action(task):
    step(task, 0.0, 1.0, contact=True)
    assert "early_action" in task.violations
    bleed(task)
    turn(task)
    step(task, task.turn_required, SECURE_DWELL + 0.05)
    assert task.physical_success and not task.temporal_success


def test_turning_the_rod_before_the_bleeding_is_an_early_action(task):
    step(task, 0.3, 0.5)
    assert "early_action" in task.violations


def test_letting_go_before_the_clip_fails_physically(task):
    bleed(task, 0.5)
    turn(task, math.radians(100.0))
    step(task, math.radians(95.0), 0.2, speed=-2.0)  # released: the strap unwinds the rod
    assert "released_unsecured" in task.physical_failures
    assert not task.physical_success and task.terminated


def test_a_brief_loss_of_contact_while_turning_is_tolerated(task):
    bleed(task, 0.5)
    turn(task, math.radians(100.0))
    step(task, math.radians(100.0), 0.05)  # shorter than the release grace
    turn(task, task.turn_required, start=math.radians(100.0))
    step(task, task.turn_required, SECURE_DWELL + 0.05)
    assert task.physical_success and not task.physical_failures


def test_forcing_the_rod_past_the_clip_fails_physically(task):
    bleed(task, 0.5)
    turn(task, task.turn_required + OVERTURN + 0.02)
    assert "over_tightened" in task.physical_failures and not task.physical_success


def test_rod_pulled_out_of_the_clip_after_securing_fails_physically(task):
    bleed(task, 0.5)
    turn(task)
    step(task, task.turn_required, SECURE_DWELL + 0.05)
    assert task.goal_achieved
    step(task, math.radians(120.0), 0.1, contact=True)
    assert "rod_left_clip" in task.physical_failures and not task.goal_achieved
    assert task.rate > 0.0  # and the bleeding resumes


def test_episode_ends_after_the_hold_once_secured(task):
    bleed(task, 0.5)
    turn(task)
    step(task, task.turn_required, SECURE_DWELL + 0.05)
    assert not task.terminated
    step(task, task.turn_required, HOLD_AFTER)
    assert task.terminated


def test_stage_fields_are_reported(task):
    bleed(task, 0.5)
    turn(task)
    step(task, task.turn_required, SECURE_DWELL + 0.05)
    m = task.metrics()
    assert m["turn_time"] < m["occlusion_time"] < m["secure_time"]
    assert m["turn_latency"] == pytest.approx(0.5, abs=0.2)
    assert m["task_name"] == "haemorrhage_tourniquet" and m["category"] == "event_reactive"
    assert rules.INSTRUCTION == m["instruction"]


# --- authoring ---------------------------------------------------------------------------------


def test_author_gives_the_rod_a_sprung_revolute_joint(task):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, tourniquet.ROOT)
    backend = SimpleNamespace(stage=stage, appearance=None, layout=task.layout)
    tourniquet.author(backend)
    colliders = [
        p for p in stage.GetPrimAtPath(tourniquet.ROOT).GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert len(colliders) == 1
    joint = UsdPhysics.RevoluteJoint.Get(stage, tourniquet.JOINT_PATH)
    assert joint.GetAxisAttr().Get() == "Z"
    assert np.allclose(joint.GetLocalPos0Attr().Get(), info(task)["pivot"], atol=1e-5)
    drive = UsdPhysics.DriveAPI(joint.GetPrim(), "angular")
    assert drive.GetStiffnessAttr().Get() == pytest.approx(tourniquet.STRAP["stiffness"])
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(info(task)["start_yaw_rad"]))
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(tourniquet.joint_limits(info(task))[1])
    tourniquet._set_drive(backend, "clip")
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(info(task)["clip_yaw_rad"]))
    assert drive.GetMaxForceAttr().Get() == pytest.approx(tourniquet.CLIP["max_force"])
    del pxr
