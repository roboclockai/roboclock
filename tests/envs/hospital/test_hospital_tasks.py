from dataclasses import replace

import numpy as np
import pytest
from hospital_helpers import observe_for, place, snapshot

from roboclock.envs.hospital.config import load_legacy_config as load_config
from roboclock.envs.hospital.mdp.events import RED
from roboclock.envs.hospital.mdp.geometry import object_extents
from roboclock.envs.hospital.tasks.reactive.aed_response import ANALYSIS as AED_ANALYSIS
from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task
from roboclock.envs.lab.mdp.state import ObjectState

# --- A1: pill cups in per-patient windows ----------------------------------------------------


STEPS = ((3.0, 5.0), (20.0, 25.0), (32.0, 36.0))


@pytest.mark.parametrize("time,valid", [(2.99, False), (3.0, True), (5.0, True), (5.01, False)])
def test_instrument_step_window_boundaries_are_inclusive(task_factory, time, valid):
    task = task_factory("A3", windows=STEPS)
    place(task, "instr_0", "tray", at=time)
    assert task.action_times["instr_0"] == pytest.approx(time)
    assert ("early_action" in task.violations or "late_action" in task.violations) != valid


def test_each_step_needs_its_own_instrument_and_window(task_factory):
    task = task_factory("A3", windows=STEPS)
    for k, (start, end) in enumerate(STEPS):
        place(task, f"instr_{k}", "tray", at=(start + end) / 2)
    assert task.physical_success and task.temporal_success and task.violations == []


def test_an_instrument_no_step_asks_for_is_a_failure(task_factory):
    task = task_factory("A3", windows=STEPS)
    place(task, "extra_0", "tray", at=4.0)
    assert "wrong_item" in task.violations and "wrong_item" in task.physical_failures


def test_a3_steps_name_three_distinct_instruments():
    from roboclock.envs.hospital.config import load_legacy_config as load_config
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    for seed in range(15):
        layout = make_layout(make_task("A3", "easy", seed, config), config)
        assert len(set(layout.steps)) == 3
        assert [layout.objects[f"instr_{k}"].asset for k in range(3)] == layout.steps
        assert len(layout.tags) == 5


def test_each_cup_uses_its_own_window(task_factory):
    task = task_factory("A1")
    windows = task.spec.windows
    place(task, "cup_0", "tray_0", at=(windows[0][0] + windows[0][1]) / 2)
    assert task.temporal_success is False  # the other cups are still outstanding
    for i in range(1, len(windows)):
        place(task, f"cup_{i}", f"tray_{i}", at=(windows[i][0] + windows[i][1]) / 2)
    assert task.physical_success and task.temporal_success and task.joint_success
    assert task.violations == []


def test_cup_placed_in_another_patients_window_is_early(task_factory):
    task = task_factory("A1")
    windows = task.spec.windows
    # Window 0 is open, but cup_1 belongs to the later window.
    place(task, "cup_1", "tray_1", at=windows[0][0] + 0.5)
    assert "early_action" in task.violations
    assert task.physical_success is False


def test_cup_in_wrong_tray_is_a_failure(task_factory):
    task = task_factory("A1")
    start = task.spec.windows[0][0] + 0.5
    place(task, "cup_0", "tray_1", at=start)
    assert "wrong_container" in task.violations
    assert "wrong_container" in task.physical_failures
    assert not task.joint_success


def test_missed_window_is_latched_without_any_placement(task_factory):
    task = task_factory("A3", windows=STEPS)
    for _ in range(600):
        task.advance(0.01)
    assert "missed_window" in task.violations
    assert task.temporal_success is False


def test_tilted_cup_is_not_accepted(task_factory):
    task = task_factory("A1")
    start = task.spec.windows[0][0] + 0.5
    tilted = np.array([np.cos(0.4), np.sin(0.4), 0.0, 0.0])
    place(task, "cup_0", "tray_0", at=start, orientation=tilted)
    assert "cup_0" not in task.correct


# --- A2: blood bag, lifted from the fridge drawer and hung on the hook ----------------------------

HOOK = "fixture:hook_peg"


def leave_fridge(task, at, grasped=True):
    task.time = at
    obj = task.layout.objects["bag_0"]
    lifted = ObjectState(obj.position + [0.0, 0.0, 0.08], obj.orientation.copy(), grasped=grasped)
    task.observe(snapshot(task, bag_0=lifted), 0.01)


def hang(task, at, dwell=0.7, speed=0.0):
    """The eyelet rests on the peg, released and (nearly) still."""
    task.time = at
    info = task.layout.info["hook"]
    obj = task.layout.objects["bag_0"]
    position = np.array([info["x"], info["tip_y"] + 0.07, info["peg_z"] - 0.108])
    state = snapshot(
        task,
        bag_0=ObjectState(
            position,
            obj.orientation.copy(),
            velocity=np.array([0.0, speed, 0.0]),
            fixture_contacts=frozenset([HOOK]),
        ),
    )
    task.observe(state, 0.01)
    observe_for(task, state, dwell)


@pytest.mark.parametrize("transit,valid", [(9.99, True), (10.0, False), (10.5, False)])
def test_blood_bag_transit_limit_is_strict(task_factory, transit, valid):
    task = task_factory("A2", transit_limit=10.0)
    leave_fridge(task, 5.0)
    assert task.removal_time == 5.0
    hang(task, 5.0 + transit)
    assert task.physical_success
    assert task.temporal_success == valid
    assert task.metrics()["transit_time"] == pytest.approx(transit, abs=0.02)


def test_resting_in_the_cradle_does_not_start_the_clock_but_lifting_does(task_factory):
    task = task_factory("A2", transit_limit=10.0)
    obj = task.layout.objects["bag_0"]
    task.time = 1.0
    wobble = ObjectState(obj.position + [0.0, 0.03, 0.002], obj.orientation.copy())
    task.observe(snapshot(task, bag_0=wobble), 0.01)  # the drawer carries it; not lifted
    assert task.removal_time is None
    leave_fridge(task, 2.0)
    assert task.removal_time == 2.0


def test_expired_transit_is_latched_before_the_bag_is_hung(task_factory):
    task = task_factory("A2", transit_limit=1.0)
    leave_fridge(task, 2.0)
    task.time = 3.0
    task.advance(0.01)
    assert "late_transit" in task.violations


def test_bag_knocked_out_of_the_drawer_without_a_grasp_is_a_failure(task_factory):
    task = task_factory("A2")
    leave_fridge(task, 2.0, grasped=False)
    for _ in range(40):  # it stays loose in the air for 0.4 s
        task.time += 0.01
        task.observe(snapshot(task, bag_0=task.previous.objects["bag_0"]), 0.01)
    assert "removed_without_grasp" in task.physical_failures


def test_a_brief_gap_in_the_grasp_reading_during_the_lift_is_tolerated(task_factory):
    task = task_factory("A2")
    leave_fridge(task, 2.0, grasped=False)  # one tick without a grasp reading
    task.time += 0.01
    obj = task.previous.objects["bag_0"]
    task.observe(
        snapshot(task, bag_0=ObjectState(obj.position, obj.orientation, grasped=True)), 0.01
    )
    assert "removed_without_grasp" not in task.physical_failures


def test_swinging_bag_does_not_count_until_it_settles(task_factory):
    task = task_factory("A2", transit_limit=30.0)
    leave_fridge(task, 2.0)
    hang(task, 5.0, dwell=0.7, speed=0.6)
    assert not task.physical_success
    hang(task, 6.0, dwell=0.7, speed=0.0)
    assert task.physical_success


def test_timer_light_goes_green_amber_red(task_factory):
    task = task_factory("A2", transit_limit=10.0)
    leave_fridge(task, 1.0)
    colors = []
    for seconds in (2.0, 8.0, 12.0):
        task.time = seconds - 0.01
        colors.append(task.advance(0.01)[0].value)
    assert colors[0] != colors[1] != colors[2] and colors[0] != colors[2]


# --- A4: syringe doses with their own beyond-use times ----------------------------------------

EXPIRIES = (20.0, 30.0, 40.0)


def place_all(task, times):
    for k, at in enumerate(times):
        place(task, f"syringe_{k}", task.layout.assignments[f"syringe_{k}"], at=at)


def test_each_dose_must_be_placed_strictly_before_its_own_expiry(task_factory):
    task = task_factory("A4", expiries=EXPIRIES, expiry_time=20.0)
    place_all(task, [e - 1.0 for e in EXPIRIES])
    assert task.physical_success and task.temporal_success


@pytest.mark.parametrize("late", [0, 1, 2])
def test_one_late_dose_fails_the_timing_but_not_the_manipulation(task_factory, late):
    task = task_factory("A4", expiries=EXPIRIES, expiry_time=20.0)
    times = [e - 1.0 for e in EXPIRIES]
    times[late] = EXPIRIES[late]  # exactly at the beyond-use time is already too late
    place_all(task, times)
    assert task.physical_success
    assert not task.temporal_success and "expired_dose" in task.violations


def test_delivering_in_the_wrong_order_misses_the_earliest_expiry(task_factory):
    task = task_factory("A4", expiries=(15.0, 20.0, 40.0), expiry_time=15.0)
    # Each delivery takes about 9 s: the latest-expiring dose first leaves the early one too late.
    place_all(task, [27.0, 18.0, 9.0])
    assert task.physical_success and not task.temporal_success


def test_a_dose_in_another_beds_tray_is_a_physical_failure(task_factory):
    task = task_factory("A4", expiries=EXPIRIES, expiry_time=20.0)
    own = task.layout.assignments["syringe_0"]
    other = next(t for t in task.layout.targets if t != own)
    place(task, "syringe_0", other, at=5.0)
    assert "wrong_container" in task.physical_failures and not task.physical_success


def test_expiry_is_latched_per_dose_when_nothing_is_placed(task_factory):
    task = task_factory("A4", expiries=(1.0, 2.0, 3.0), expiry_time=1.0)
    for _ in range(150):
        task.advance(0.01)
    assert "expired_dose" in task.violations


def test_every_dose_has_a_label_with_its_bed_and_time(task_factory):
    task = task_factory("A4", expiries=EXPIRIES, expiry_time=20.0)
    texts = {t.text for t in task.layout.tags}
    for k, expiry in enumerate(EXPIRIES):
        bed = task.layout.assignments[f"syringe_{k}"].split("_")[-1]
        assert f"BED {int(bed) + 1}  0:{int(expiry):02d}" in texts


def test_sampled_doses_expire_in_increasing_order():
    config = load_config()
    for level in ("easy", "medium", "hard"):
        spec = make_task("A4", level, 3, config).spec
        assert len(spec.expiries) == 3 == spec.num_items
        assert list(spec.expiries) == sorted(spec.expiries) and spec.expiries[0] == spec.expiry_time


# --- B1: supply alarm ------------------------------------------------------------------------


def b1(task_factory, limit=8.0, event=5.0):
    return task_factory("B1", response_limit=limit, event_time=event)


def trigger(task):
    while task.event_time is None:
        task.advance(0.01)


def press_pump(task, j, seconds=0.4):
    """Hold the tool tip on pump j's SILENCE button."""
    button = task.buttons[j]
    ee = np.array([*button[:2], button[2] + 0.004])
    state = replace(task.previous or snapshot(task), ee_position=ee)
    for _ in range(round(seconds / 0.01)):
        task.time += 0.01
        task.observe(state, 0.01)


@pytest.mark.parametrize("delay,valid", [(7.0, True), (8.0, False), (9.0, False)])
def test_silencing_must_be_strictly_inside_the_response_limit(task_factory, delay, valid):
    task = b1(task_factory)
    trigger(task)
    j = task.layout.alarm_station
    task.time = task.event_time + delay - 0.2  # the press settles 0.2 s later
    press_pump(task, j)
    assert task.physical_success
    assert task.temporal_success == valid
    assert task.metrics()["reaction_latency"] == pytest.approx(delay, abs=0.05)


def test_pressing_a_pump_that_is_not_alarming_is_a_physical_failure(task_factory):
    task = b1(task_factory)
    trigger(task)
    j = task.layout.alarm_station
    press_pump(task, (j + 1) % 3)
    assert "wrong_pump" in task.physical_failures


def test_silencing_before_the_alarm_is_a_violation(task_factory):
    task = b1(task_factory)
    j = task.layout.alarm_station
    task.time = 1.0
    press_pump(task, j)
    assert "pre_event_action" in task.violations and not task.press_done
    assert not task.joint_success


def test_unanswered_alarm_is_late_at_the_limit(task_factory):
    task = b1(task_factory, limit=2.0, event=1.0)
    for _ in range(320):
        task.advance(0.01)
    assert "late_delivery" in task.violations


def test_the_oracle_target_is_the_earliest_deadline_among_the_pumps_that_alarm(task_factory):
    task = task_factory(
        "B1",
        alarms=(5.0, 8.0, 11.0),
        alarm_limits=(14.0, 14.0, 9.8),
        event_time=5.0,
        response_limit=14.0,
    )
    stations = [s for s, _, _ in task.alarms()]
    task.time = 4.0
    # No pump alarms yet: there is no target and no button to go to (the cue is not observable).
    assert task.target() is None and task.button is None and not task.press_ready
    assert not task.cue_observed
    task.time = 5.0
    assert task.cue_observed and task.target()[0] == stations[0]
    task.time = 8.5  # the third pump (11.0) has not alarmed: it is not a target, deadline or not
    assert {task.target()[0]} <= {stations[0], stations[1]}
    task.time = 12.0  # all three have alarmed: deadlines 19, 22 and 20.8
    assert task.target()[0] == stations[0] and task.press_ready
    task.delivery_times[stations[0]] = 12.0
    assert task.target()[0] == stations[2]  # 20.8 comes before 22


# --- B2: cardiac arrest and the AED -----------------------------------------------------------


def b2(task_factory, limit=40.0, event=2.0):
    task = task_factory("B2", response_limit=limit, event_time=event)
    trigger(task)
    return task


def settle(task, **changes):
    """Hold the given object states for a while, as the robot would after letting go."""
    state = snapshot(task, **changes)
    task.observe(state, 0.01)  # the first frame after release: the objects count as moved
    task.observe(state, 0.01)
    observe_for(task, state, 0.4)
    return state


def pull_tab(task):
    tab = task.layout.objects["tab_0"]
    away = ObjectState(
        tab.position + [*task.layout.info["aed"]["tab_pull"], 0.0], tab.orientation.copy()
    )
    return settle(task, tab_0=away)


def pad_on(task, k, mark=None, **others):
    zone = task.layout.zones[f"mark_{mark if mark is not None else k}"]
    pad = task.layout.objects[f"pad_{k}"]
    z = zone.support_z + task.layout.shapes[f"pad_{k}"].half[2]
    return {f"pad_{k}": ObjectState(np.array([*zone.center, z]), pad.orientation.copy()), **others}


def press(task, seconds=0.4):
    ee = np.array([*task.button[:2], task.button[2] + 0.004])
    state = snapshot(task)
    state = replace(state, ee_position=ee)
    for _ in range(round(seconds / 0.01)):
        task.time += 0.01
        task.observe(state, 0.01)


def do_everything(task, ready_wait=AED_ANALYSIS + 0.5):
    pull_tab(task)
    for k in range(2):
        settle(task, **pad_on(task, k))
    task.time = max(task.pad_times.values()) + ready_wait
    task.advance(0.01)
    press(task)


def test_the_pads_are_locked_until_the_tab_is_pulled(task_factory):
    task = task_factory("B2")
    assert set(task.layout.held) == {"pad_0", "pad_1"} and "tab_0" not in task.layout.held
    commands = []
    pull_tab_task = b2(task_factory)
    commands += [c for c in pull_tab_task.advance(0.01) if c.kind == "release"]
    assert commands == []  # nothing is released before the tab moves
    pull_tab(pull_tab_task)
    released = [c.name for c in pull_tab_task.advance(0.01) if c.kind == "release"]
    assert sorted(released) == ["pad_0", "pad_1"]


def test_full_sequence_in_time_is_a_joint_success(task_factory):
    task = b2(task_factory)
    do_everything(task)
    assert task.physical_success and task.temporal_success, (
        task.violations,
        task.physical_failures,
    )
    metrics = task.metrics()
    assert metrics["shock_time"] >= metrics["shock_ready_time"]
    assert metrics["shock_ready_time"] == pytest.approx(
        max(metrics["pad_times"].values()) + AED_ANALYSIS, abs=0.05
    )


def test_unit_needs_eight_seconds_to_analyze_after_both_pads_are_on(task_factory):
    task = b2(task_factory)
    pull_tab(task)
    for k in range(2):
        settle(task, **pad_on(task, k))
    assert AED_ANALYSIS == 8.0  # manufacturer figure (Philips FRx: "typically in 8 seconds")
    assert task.shock_ready_time == pytest.approx(max(task.pad_times.values()) + AED_ANALYSIS)
    task.time = max(task.pad_times.values()) + AED_ANALYSIS - 1.0
    assert not task.shock_ready
    press(task)  # too early: ignored
    assert task.shock_time is None and task.premature_press and not task.physical_failures


def test_shock_without_both_pads_is_a_physical_failure(task_factory):
    task = b2(task_factory)
    pull_tab(task)
    settle(task, **pad_on(task, 0))
    press(task)
    assert "shock_without_pads" in task.physical_failures and not task.physical_success


def test_pads_swapped_onto_the_wrong_marks_are_a_failure(task_factory):
    task = b2(task_factory)
    pull_tab(task)
    settle(task, **pad_on(task, 0, mark=1))
    assert "wrong_pad_placement" in task.physical_failures


def test_a_late_shock_is_physically_fine_but_misses_the_limit(task_factory):
    task = b2(task_factory, limit=10.0, event=2.0)
    pull_tab(task)
    for k in range(2):
        settle(task, **pad_on(task, k))
    task.time = task.event_time + 12.0
    press(task)
    assert task.physical_success and not task.temporal_success
    assert "late_shock" in task.violations


def test_the_response_limit_is_strict(task_factory):
    task = b2(task_factory, limit=10.0, event=2.0)
    pull_tab(task)
    for k in range(2):
        settle(task, **pad_on(task, k))
    task.time = task.event_time + 10.0 - 0.2  # the press registers at the limit
    press(task, seconds=0.25)
    assert "late_shock" in task.violations


def test_touching_the_kit_before_the_arrest_is_a_violation(task_factory):
    task = task_factory("B2", response_limit=40.0, event_time=5.0)
    tab = task.layout.objects["tab_0"]
    state = snapshot(
        task, tab_0=ObjectState(tab.position.copy(), tab.orientation.copy(), grasped=True)
    )
    task.observe(state, 0.01)
    assert "pre_event_action" in task.violations


# --- B3: fire and the blanket -----------------------------------------------------------------


def b3(task_factory, limit=40.0, event=2.0):
    task = task_factory("B3", response_limit=limit, event_time=event)
    trigger(task)
    return task


def pull_tabs(task, which=("tab_0", "tab_1")):
    changes = {}
    for name in which:
        obj = task.layout.objects[name]
        changes[name] = ObjectState(obj.position + [0.0, 0.08, 0.0], obj.orientation.copy())
    return settle(task, **changes)


def press_alarm(task, seconds=0.4):
    """Hold the tool tip on the red FIRE ALARM button."""
    ee = np.array([*task.button[:2], task.button[2] + 0.004])
    state = replace(snapshot(task), ee_position=ee)
    for _ in range(round(seconds / 0.01)):
        task.time += 0.01
        task.observe(state, 0.01)


def cover(task, offset=(0.0, 0.0), lift=0.0):
    if task.alarm_time is None and task.event_time is not None:
        press_alarm(task)
    blanket = task.layout.objects["blanket"]
    z = task.dish_top + task.layout.shapes["blanket"].half[2] + lift
    position = np.array([*(task.fire_center + np.array(offset)), z])
    state = settle(task, blanket=ObjectState(position, blanket.orientation.copy()))
    observe_for(task, state, 1.7)  # the scene keeps running a moment after the fire is covered
    return state


def test_the_blanket_is_locked_until_both_tabs_are_pulled(task_factory):
    task = b3(task_factory)
    assert set(task.layout.held) == {"blanket"}
    assert [c for c in task.advance(0.01) if c.kind == "release"] == []
    pull_tabs(task, ("tab_0",))
    assert [c for c in task.advance(0.01) if c.kind == "release"] == []  # one tab is not enough
    pull_tabs(task)
    released = [c.name for c in task.advance(0.01) if c.kind == "release"]
    assert released == ["blanket"]


def test_flames_show_at_ignition_and_hide_when_covered(task_factory):
    task = task_factory("B3", event_time=1.0, response_limit=40.0)
    shown = []
    for _ in range(150):
        shown += [(c.name, c.value) for c in task.advance(0.01) if c.kind == "show"]
    assert shown == [("flames", True)]
    pull_tabs(task)
    cover(task)
    shown = [(c.name, c.value) for c in task.advance(0.01) if c.kind == "show"]
    assert ("flames", False) in shown and ("blanket_cover", True) in shown


def test_fire_full_sequence_in_time_is_a_joint_success(task_factory):
    task = b3(task_factory)
    pull_tabs(task)
    cover(task)
    assert task.physical_success and task.temporal_success, (
        task.violations,
        task.physical_failures,
    )
    assert task.metrics()["response_time"] is not None
    assert task.action_time == pytest.approx(task.cover_time)


def test_a_blanket_beside_the_dish_does_not_smother_the_fire(task_factory):
    task = b3(task_factory)
    pull_tabs(task)
    cover(task, offset=(0.12, 0.0))
    assert task.cover_time is None and not task.goal_achieved


def test_a_blanket_held_above_the_dish_does_not_count(task_factory):
    task = b3(task_factory)
    pull_tabs(task)
    cover(task, lift=0.05)
    assert task.cover_time is None


def test_covering_at_exactly_the_limit_is_late(task_factory):
    task = b3(task_factory, limit=10.0, event=2.0)
    pull_tabs(task)
    task.time = task.event_time + 10.0 - 0.1
    cover(task)
    assert "late_cover" in task.violations and not task.temporal_success


def test_an_unanswered_fire_is_late_at_the_limit(task_factory):
    task = b3(task_factory, limit=3.0, event=1.0)
    for _ in range(450):
        task.advance(0.01)
    assert "late_cover" in task.violations


def test_touching_the_pouch_before_the_fire_is_a_violation(task_factory):
    task = task_factory("B3", response_limit=40.0, event_time=5.0)
    tab = task.layout.objects["tab_0"]
    task.observe(
        snapshot(
            task, tab_0=ObjectState(tab.position.copy(), tab.orientation.copy(), grasped=True)
        ),
        0.01,
    )
    assert "pre_event_action" in task.violations


# --- C1-C3: throughput -----------------------------------------------------------------------


def test_c1_counts_only_items_the_order_still_needs(task_factory):
    task = task_factory("C1", num_items=18)
    layout = task.layout
    # Find a bin that needs an item type and an item of that type.
    container, order = next((c, o) for c, o in layout.orders.items() if o)
    kind = next(iter(order))
    matching = [n for n, k in layout.item_types.items() if k == kind]
    for name in matching[: order[kind]]:
        place(task, name, container)
    assert len(task.correct) == order[kind]
    # One more of that type exceeds the order: an attempt, but not a correct operation.
    extra = matching[order[kind]] if len(matching) > order[kind] else None
    if extra:
        place(task, extra, container)
        assert extra not in task.correct and extra in task.attempted
        assert task.metrics()["accuracy"] < 1.0


def test_c1_accuracy_threshold_gates_temporal_success(task_factory):
    task = task_factory("C1", required_rate_per_min=0.1)
    layout = task.layout
    stock = {}
    for name, kind in layout.item_types.items():
        stock.setdefault(kind, []).append(name)
    # An item type that another bin also needs lets this bin be over-filled by one.
    container, kind = next(
        (c, k)
        for c, order in layout.orders.items()
        for k, count in order.items()
        if 0 < count < len(stock[k])
    )
    quota = layout.orders[container][kind]
    for name in stock[kind][: quota + 1]:
        place(task, name, container)
    task.time = 60.0
    assert len(task.correct) == quota
    assert task.metrics()["accuracy"] == pytest.approx(quota / (quota + 1))
    assert task.temporal_success is False  # below the 95% order-accuracy requirement


def test_c2_requires_the_matching_category_bin(task_factory):
    task = task_factory("C2")
    layout = task.layout
    name, kind = next(iter(layout.item_types.items()))
    right = next(c for c, k in layout.bin_types.items() if k == kind)
    wrong = next(c for c, k in layout.bin_types.items() if k != kind)
    place(task, name, wrong)
    assert name not in task.correct and name in task.attempted
    other = next(n for n, k in layout.item_types.items() if k == kind and n != name)
    place(task, other, right)
    assert other in task.correct


def test_throughput_item_is_credited_once(task_factory):
    task = task_factory("C2")
    layout = task.layout
    name, kind = next(iter(layout.item_types.items()))
    right = next(c for c, k in layout.bin_types.items() if k == kind)
    place(task, name, right)
    place(task, name, right, at=task.time + 1.0)
    assert len(task.correct) == 1


def test_throughput_never_terminates_early(task_factory):
    task = task_factory("C2", num_items=2)
    for name in list(task.layout.objects)[:1]:
        kind = task.layout.item_types[name]
        container = next(c for c, k in task.layout.bin_types.items() if k == kind)
        place(task, name, container)
    assert task.terminated is False


def press_start(task, seconds=0.4):
    """Hold the tool tip on the green START button."""
    ee = np.array([*task.button[:2], task.button[2] + 0.004])
    state = replace(task.previous or snapshot(task), ee_position=ee)
    for _ in range(round(seconds / 0.01)):
        task.time += 0.01
        task.observe(state, 0.01)


def load_all(task, at=5.0):
    for k, name in enumerate(list(task.layout.objects)):
        place(task, name, "basket", at=at + k)


def test_c3_needs_every_instrument_in_and_start_pressed_before_the_cutoff(task_factory):
    task = task_factory("C3", cycle_time=60.0, max_time=60.0)
    load_all(task)
    assert task.press_ready and not task.press_done and not task.goal_achieved
    press_start(task)
    assert task.press_done and task.goal_achieved
    assert task.temporal_success and not task.terminated  # the cycle still runs on the dial
    task.time += 5.0
    assert task.terminated


def test_c3_start_pressed_before_the_instruments_are_in_does_not_count(task_factory):
    task = task_factory("C3", cycle_time=60.0, max_time=60.0)
    press_start(task)
    assert not task.press_done
    place(task, list(task.layout.objects)[0], "basket")
    press_start(task)
    assert not task.press_done


def test_c3_missing_the_cutoff_fails_the_timing(task_factory):
    task = task_factory("C3", cycle_time=20.0, max_time=20.0)
    load_all(task, at=5.0)
    task.time = 20.5
    task.observe(snapshot(task), 0.01)
    assert not task.temporal_success and "late_start" in task.violations and task.terminated


def test_c3_indicator_turns_red_when_the_cycle_starts(task_factory):
    task = task_factory("C3", cycle_time=1.0, max_time=1.0)
    colors = [task.advance(0.01)[0].value for _ in range(120)]
    assert colors[0] != colors[-1]


def test_basket_accepts_stacked_instruments(task_factory):
    task = task_factory("C3")
    names = list(task.layout.objects)[:2]
    target = task.layout.targets["basket"]
    for k, name in enumerate(names):
        obj = task.layout.objects[name]
        extent = object_extents(task.layout.shapes[name], obj.orientation)
        lift = 0.025 * k
        above = ObjectState(
            np.array([*target.position[:2], target.rim_z + extent[2] + 0.02]),
            obj.orientation.copy(),
            grasped=True,
        )
        task.time += 1.0
        task.observe(snapshot(task, **{name: above}), 0.01)
        rest = ObjectState(
            np.array([*target.position[:2], target.support_z + extent[2] + 0.001 + lift]),
            obj.orientation.copy(),
        )
        state = snapshot(task, **{name: rest})
        task.observe(state, 0.01)
        observe_for(task, state, 0.3)
    assert len(task.correct) == 2


# --- shared behavior -------------------------------------------------------------------------


def test_dropping_an_item_ends_the_episode(task_factory):
    task = task_factory("A4")
    obj = task.layout.objects["syringe_0"]
    below = ObjectState(np.array([0.3, 0.0, 0.2]), obj.orientation.copy())
    task.observe(snapshot(task, syringe_0=below), 0.01)
    assert task.terminated and task.dropped == {"syringe_0"}
    assert not task.physical_success


def test_metrics_follow_the_shared_schema(task_factory):
    for task_id in ("A1", "B1", "C1"):
        metrics = task_factory(task_id).metrics()
        for key in (
            "task_id",
            "difficulty",
            "category",
            "task_success",
            "manipulation_success",
            "temporal_success",
            "joint_success",
            "temporal_violations",
            "completion_time",
        ):
            assert key in metrics
        assert metrics["domain"] == "hospital"


def test_throughput_items_may_lie_on_their_side(task_factory):
    task = task_factory("C3")
    on_side = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    place(task, "item_0", "basket", at=2.0, orientation=on_side)
    assert "item_0" in task.correct


def test_a1_has_decoy_vials_that_belong_to_no_tray(task_factory):
    task = task_factory("A1")
    decoys = [n for n in task.layout.objects if n.startswith("decoy_")]
    assert len(decoys) == 2
    assert all(n not in task.layout.assignments for n in decoys)


def test_decoy_vial_in_a_tray_is_a_failure(task_factory):
    task = task_factory("A1")
    start = task.spec.windows[0][0] + 0.5
    place(task, "decoy_0", "tray_0", at=start)
    assert "wrong_item" in task.violations
    assert "wrong_item" in task.physical_failures
    assert not task.joint_success


def test_a1_vials_are_shuffled_over_the_pad_slots():
    from roboclock.envs.hospital.config import load_legacy_config as load_config
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_legacy_task as make_task

    config = load_config()
    orders = set()
    for seed in range(12):
        task = make_task("A1", "easy", seed, config)
        layout = make_layout(task, config)
        orders.add(
            tuple(layout.objects[f"cup_{i}"].position[0] > 0.3 for i in range(task.spec.num_items))
        )
    assert len(orders) > 1


def test_an_instrument_may_rest_on_either_face_but_not_on_end(task_factory):
    """Rolling a square-section tool over its long axis is harmless; standing it on end is not."""
    c = np.cos(np.pi / 4)
    rolled, on_end = np.array([c, c, 0.0, 0.0]), np.array([c, 0.0, c, 0.0])
    task = task_factory("A3", windows=STEPS)
    place(task, "instr_0", "tray", at=STEPS[0][0] + 0.5, orientation=rolled)
    assert "instr_0" in task.correct
    task = task_factory("A3", windows=STEPS)
    place(task, "instr_0", "tray", at=STEPS[0][0] + 0.5, orientation=on_end)
    assert "instr_0" not in task.correct


def test_surgeon_calls_for_each_instrument_once_at_its_window_start(task_factory):
    task = task_factory("A3", windows=STEPS)
    shown = {}
    for _ in range(4000):  # 40 s
        for command in task.advance(0.01):
            if command.kind == "show":
                shown.setdefault(command.name, []).append((round(task.time, 2), command.value))
    for k, (start, _) in enumerate(STEPS):
        assert shown[f"call_{k}"] == [(pytest.approx(start, abs=0.02), True)]
        assert shown[f"wait_{k}"] == [(pytest.approx(start, abs=0.02), False)]
    calls = [e for e in task.event_log if e["type"] == "call"]
    assert [e["step"] for e in calls] == [0, 1, 2]
    assert [e["tool"] for e in calls] == task.layout.steps


def test_response_time_is_measured_from_the_call(task_factory):
    task = task_factory("A3", windows=STEPS)
    for _ in range(int((STEPS[0][0] + 0.5) * 100)):
        task.advance(0.01)
    place(task, "instr_0", "tray", at=STEPS[0][0] + 1.5)
    metrics = task.metrics()
    assert metrics["call_times"][0] == pytest.approx(STEPS[0][0], abs=0.02)
    assert metrics["response_times"]["instr_0"] == pytest.approx(1.5, abs=0.05)


def multi_alarm(task_factory, times=(5.0, 8.0, 11.0), limits=(14.0, 14.0, 9.8)):
    return task_factory(
        "B1", alarms=times, alarm_limits=limits, event_time=times[0], response_limit=14.0
    )


def test_each_alarm_has_its_own_station_and_deadline(task_factory):
    task = multi_alarm(task_factory)
    alarms = task.alarms()
    assert len({station for station, _, _ in alarms}) == 3  # three different beds
    assert [t for _, t, _ in alarms] == [5.0, 8.0, 11.0]
    assert alarms[-1][2] < alarms[0][2]  # the last one is critical


def test_all_alarms_must_be_answered_inside_their_own_limits(task_factory):
    task = multi_alarm(task_factory)
    for station, time, limit in task.alarms():
        task.time = time + limit - 1.5
        press_pump(task, station)
    assert task.physical_success and task.temporal_success
    assert set(task.metrics()["response_times"]) == {s for s, _, _ in task.alarms()}


def test_the_critical_alarm_has_a_shorter_limit(task_factory):
    task = multi_alarm(task_factory)
    for station, time, limit in task.alarms()[:-1]:
        task.time = time + 3.0
        press_pump(task, station)
    station, time, limit = task.alarms()[-1]
    task.time = time + limit + 0.5  # fine for the others
    press_pump(task, station)
    assert task.physical_success and not task.temporal_success
    assert "late_delivery" in task.violations


def test_silencing_a_pump_before_its_alarm_is_a_violation(task_factory):
    task = multi_alarm(task_factory)
    station = task.alarms()[2][0]  # the third alarm has not gone off at t = 6
    task.time = 6.0
    press_pump(task, station)
    assert "pre_event_action" in task.violations


def test_sampled_b1_levels_have_more_alarms_and_a_critical_last_one():
    config = load_config()
    for level, count in (("easy", 1), ("medium", 2), ("hard", 3)):
        spec = make_task("B1", level, 4, config).spec
        if count == 1:
            assert spec.alarms == () and spec.alarm_limits == ()
            continue
        assert len(spec.alarms) == count == len(spec.alarm_limits)
        assert spec.alarms[0] == spec.event_time
        assert spec.alarm_limits[-1] == pytest.approx(0.7 * spec.response_limit)
        assert all(b > a for a, b in zip(spec.alarms, spec.alarms[1:]))


def timer_commands(task, seconds, dt=0.05):
    out = []
    for _ in range(int(seconds / dt)):
        out += [c for c in task.advance(dt) if c.kind == "timer"]
    return out


def lights(task, seconds, dt=0.05):
    out = []
    for _ in range(int(seconds / dt)):
        out += [
            c.value for c in task.advance(dt) if c.kind == "indicator" and c.name == "indicator"
        ]
    return out


def check_drains(commands):
    fractions = [c.value[0] for c in commands]
    assert fractions == sorted(fractions, reverse=True)
    assert fractions[0] > 0.9 and fractions[-1] < 0.1
    colors = [c.value[1] for c in commands]
    assert colors[0] == "green" and "amber" in colors and colors[-1] == "red"
    assert len(commands) < 150  # only visible changes are sent, not one per tick


def test_fire_timer_counts_down_the_response_limit_after_ignition(task_factory):
    task = task_factory("B3", response_limit=10.0, event_time=2.0)
    assert timer_commands(task, 1.9) == []  # nothing before the ignition
    check_drains(timer_commands(task, 9.9))


def test_arrest_timer_runs_until_the_shock_is_delivered(task_factory):
    task = task_factory("B2", response_limit=20.0, event_time=2.0)
    assert timer_commands(task, 1.9) == []
    commands = timer_commands(task, 6.0)  # the arrest happens, nobody responds yet
    assert commands and commands[-1].value[0] < 0.8
    do_everything(task)
    assert timer_commands(task, 3.0) == []  # the bar stops once the shock is delivered


@pytest.mark.parametrize("task_id", ["C1", "C2"])
def test_throughput_tasks_show_the_time_budget_running_out(task_factory, task_id):
    task = task_factory(task_id)
    check_drains(timer_commands(task, task.spec.max_time - 1.0, dt=0.1))
    assert lights(task, 5.0, dt=0.1)[-1] == RED


def test_sterilizer_timer_counts_down_to_the_cycle_start(task_factory):
    task = task_factory("C3")
    check_drains(timer_commands(task, task.spec.cycle_time - 1.0, dt=0.1))


def test_every_timed_task_has_a_dial_that_runs_while_its_clock_runs(task_factory):
    """Each deadline has a physical countdown: window tasks while a window is open, B1 per alarm, A4 per dose."""
    for task_id in ("A1", "A3", "A4", "B1"):
        task = task_factory(task_id)
        commands = timer_commands(task, 40.0, dt=0.1)
        assert commands, task_id
        assert all(c.kind == "timer" for c in commands)
        assert commands[0].value[0] > 0.8, task_id  # starts full
        assert any(c.value[0] < 0.2 for c in commands), task_id  # and runs down


def test_blood_bag_countdown_runs_from_leaving_the_fridge_and_stops_when_it_hangs(task_factory):
    task = task_factory("A2", transit_limit=20.0)
    assert timer_commands(task, 5.0) == []  # nothing before the bag is out of the fridge
    task.removal_time = task.time
    check_drains(timer_commands(task, 19.0, dt=0.1))
    task.placement_time = task.time
    assert timer_commands(task, 2.0) == []  # stopped once it hangs


def test_pharmacy_requests_appear_on_the_wall_screen_when_they_arrive(task_factory):
    task = task_factory("A1")
    shown = {}
    for _ in range(int(60 / 0.05)):
        for c in task.advance(0.05):
            if c.kind == "show":
                shown.setdefault(c.name, task.time)
    for k, (start, _) in enumerate(task.spec.windows):
        assert shown[f"request_{k}"] == pytest.approx(start, abs=0.1)


def test_a_spent_tab_falling_off_the_bench_does_not_fail_the_task(task_factory):
    for task_id, tab in (("B2", "tab_0"), ("B3", "tab_0")):
        task = task_factory(task_id, response_limit=40.0, event_time=2.0)
        trigger(task)
        obj = task.layout.objects[tab]
        floor = ObjectState(
            np.array([obj.position[0], obj.position[1], 0.2]), obj.orientation.copy()
        )
        task.observe(snapshot(task, **{tab: floor}), 0.01)
        assert not task.dropped, task_id
        assert tab in task.dropped_consumables


def test_a_dropped_patient_item_still_fails_the_task(task_factory):
    task = task_factory("B2", response_limit=40.0, event_time=2.0)
    trigger(task)
    pad = task.layout.objects["pad_0"]
    floor = ObjectState(np.array([pad.position[0], pad.position[1], 0.2]), pad.orientation.copy())
    task.observe(snapshot(task, pad_0=floor), 0.01)
    assert "pad_0" in task.dropped and not task.physical_success


def test_the_oracle_touches_no_pad_until_the_tab_has_registered_as_pulled(task_factory):
    from roboclock.envs.hospital.policies.jobs import HospitalJobs

    task = b2(task_factory)
    jobs = HospitalJobs()
    assert jobs.choose(task, snapshot(task)) == "tab_0"
    # Even while the tab is "recently released", the welded pads must not be chosen.
    assert jobs.choose(task, snapshot(task), exclude=("tab_0",)) is None
    assert not jobs.describe(task, snapshot(task), "pad_0").grasp_ok
    pull_tab(task)
    assert jobs.choose(task, snapshot(task), exclude=("tab_0",)) == "pad_0"
    assert jobs.describe(task, snapshot(task), "pad_0").grasp_ok


def test_a_tab_set_down_a_few_centimeters_short_of_its_goal_still_counts_as_pulled(task_factory):
    task = b2(task_factory)
    tab = task.layout.objects["tab_0"]
    pull = np.array(task.layout.info["aed"]["tab_pull"])
    short = ObjectState(
        tab.position + [*(0.6 * pull), 0.0], tab.orientation.copy()
    )  # 60 % of the way
    settle(task, tab_0=short)
    assert task.tab_time is not None


def test_the_fire_alarm_is_raised_by_pressing_its_button_and_logged(task_factory):
    task = b3(task_factory)
    assert task.alarm_time is None
    press_alarm(task)
    assert task.alarm_time is not None
    assert any(e["type"] == "alarm_raised" for e in task.event_log)


def test_a_covered_fire_without_the_alarm_does_not_finish_the_task(task_factory):
    task = b3(task_factory)
    pull_tabs(task)
    blanket = task.layout.objects["blanket"]
    z = task.dish_top + task.layout.shapes["blanket"].half[2]
    settle(task, blanket=ObjectState(np.array([*task.fire_center, z]), blanket.orientation.copy()))
    observe_for(task, snapshot(task), 2.0)
    assert task.cover_time is None and not task.goal_achieved  # RACE: the alarm comes first


def test_a_late_alarm_is_a_violation(task_factory):
    task = b3(task_factory, limit=20.0, event=2.0)
    task.time = task.event_time + 0.4 * 20.0 + 0.5
    task.observe(snapshot(task), 0.01)
    task.advance(0.01)
    assert "late_alarm" in task.violations


def test_pressing_the_alarm_before_the_fire_is_a_violation(task_factory):
    task = task_factory("B3", response_limit=40.0, event_time=5.0)
    press_alarm(task)
    assert "pre_event_action" in task.violations


def test_the_oracle_raises_the_alarm_before_it_touches_the_tabs(task_factory):
    from roboclock.envs.hospital.policies.jobs import HospitalJobs

    task = b3(task_factory)
    jobs = HospitalJobs()
    assert (
        jobs.choose(task, snapshot(task)) is None
    )  # nothing to carry yet: it goes to the alarm button
    press_alarm(task)
    assert jobs.choose(task, snapshot(task)) == "tab_0"


def test_fridge_temperature_readout_warms_while_the_bag_is_out_and_freezes_when_it_hangs(
    task_factory,
):
    task = task_factory("A2", transit_limit=21.0)
    shown = []
    for _ in range(40):
        shown += [c for c in task.advance(0.1) if c.kind == "show"]
    assert shown == []  # 4.0 C while the bag is in the fridge
    task.removal_time = task.time
    for _ in range(int(21.0 / 0.1)):
        shown += [(c.name, c.value) for c in task.advance(0.1) if c.kind == "show"]
    levels = [n for n, v in shown if v]
    assert levels == [f"temp_{k}" for k in range(1, 7)]  # one degree at a time up to 10.0 C
    assert ("temp_0", False) in shown


def test_emergency_beacons_flash_only_while_the_emergency_is_open(task_factory):
    def flashes(task, seconds, dt=0.05):
        colors = []
        for _ in range(int(seconds / dt)):
            colors += [
                c.value for c in task.advance(dt) if c.kind == "indicator" and c.name == "alert"
            ]
        return colors

    # A3: dark until a call, flashing during it
    task = task_factory("A3")
    assert len(set(flashes(task, 3.0))) <= 1
    start = task.spec.windows[0][0]
    task.time = start
    open_colors = flashes(task, 3.0)
    assert len(set(open_colors)) == 2  # bright and dim alternate

    # B2: the code-blue lamp flashes from the arrest until the shock
    task = task_factory("B2", event_time=1.0, response_limit=40.0)
    assert len(set(flashes(task, 4.0))) == 2

    # B3: the fire-alarm strobe starts only once the alarm has been pressed
    task = task_factory("B3", event_time=1.0, response_limit=40.0)
    assert len(set(flashes(task, 4.0))) <= 1
    task.alarm_time = task.time
    assert len(set(flashes(task, 3.0))) == 2
    task.cover_time = task.time
    flashes(task, 3.0)
    assert task._flash["alert"] == (0.22, 0.03, 0.03)  # dark again once the fire is covered


def test_the_alert_beacons_exist_in_the_layouts_of_the_emergency_tasks():
    from roboclock.envs.hospital.config import load_legacy_config as _load
    from roboclock.envs.hospital.layout import make_layout

    config = _load()
    for task_id in ("A3", "B2", "B3"):
        layout = make_layout(make_task(task_id, "easy", 0, config), config)
        beacons = [i for i in layout.indicators if i.name == "alert"]
        assert len(beacons) == 1 and beacons[0].style == "beacon", task_id


def prep_task(**spec):
    from dataclasses import replace as _replace

    from roboclock.envs.hospital.config import load_legacy_config as _load
    from roboclock.envs.hospital.layout import make_layout

    config = _load()
    config["tasks"]["A4"]["dose_prep"] = True
    task = make_task("A4", "easy", 0, config)
    if spec:
        task.spec = _replace(task.spec, **spec)
    task.bind(make_layout(task, config))
    return task


def dock_state(task, name, grasped=False):
    prep = task.layout.info["dose_prep"]
    obj = task.layout.objects[name]
    z = prep["support_z"] + task.layout.shapes[name].half[2]
    return ObjectState(np.array([*prep["dock"], z]), obj.orientation.copy(), grasped=grasped)


def test_a_syringe_on_the_dock_fills_after_the_fill_time_and_shows_its_liquid():
    from roboclock.envs.hospital.tasks.temporal_window.syringe_dose import FILL_TIME

    task = prep_task()
    shown, timers = [], []
    state = snapshot(task, syringe_0=dock_state(task, "syringe_0"))
    for _ in range(int((FILL_TIME + 1.0) / 0.05)):
        for c in task.advance(0.05):
            (shown if c.kind == "show" else timers).append(c)
        task.observe(state, 0.05)
    assert "syringe_0" in task.drawn
    assert task.drawn["syringe_0"] == pytest.approx(FILL_TIME, abs=0.2)
    assert [c.name for c in shown] == ["liquid_0"]
    assert any(
        c.name == "dock" and c.value[1] == "amber" for c in timers
    )  # the dock dial counted the fill


def test_a_syringe_carried_through_the_dock_does_not_fill():
    task = prep_task()
    state = snapshot(task, syringe_0=dock_state(task, "syringe_0", grasped=True))
    for _ in range(100):
        task.time += 0.05
        task.observe(state, 0.05)
    assert task.drawn == {}


def test_an_empty_syringe_in_a_patients_tray_is_a_physical_failure():
    task = prep_task()
    tray = task.layout.assignments["syringe_0"]
    place(task, "syringe_0", tray, at=5.0)
    assert "undrawn_dose" in task.physical_failures and not task.physical_success


def test_the_oracle_leaves_a_filling_syringe_alone_and_never_starts_a_second_dose():
    from roboclock.envs.hospital.policies.jobs import HospitalJobs

    task = prep_task()
    jobs = HospitalJobs()
    state = snapshot(task, syringe_0=dock_state(task, "syringe_0"))
    assert jobs.choose(task, state) == "syringe_0"
    assert (
        jobs.choose(task, state, exclude=("syringe_0",)) is None
    )  # waits instead of starting syringe_1
    assert not jobs.describe(task, state, "syringe_0").grasp_ok  # hovers until it has filled
    task.drawn["syringe_0"] = task.time
    assert (
        jobs.describe(task, state, "syringe_0").destination[0]
        != task.layout.info["dose_prep"]["dock"][0]
    )


def test_doses_expire_later_when_they_have_to_be_prepared_first():
    from roboclock.envs.hospital.config import load_legacy_config as _load
    from roboclock.envs.hospital.config import sample_episode

    config = _load()
    plain = sample_episode("A4", "easy", 0, config)
    config["tasks"]["A4"]["dose_prep"] = True
    prepared = sample_episode("A4", "easy", 0, config)
    assert prepared.expiries[0] > plain.expiries[0] + 10
    assert prepared.expiries[1] - prepared.expiries[0] > plain.expiries[1] - plain.expiries[0] + 5
