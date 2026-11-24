"""B3 hazardous drug spill (tasks/reactive/drug_spill.py): task logic on scripted physical states."""

from dataclasses import replace

import numpy as np
import pytest
from hospital_helpers import above_container, in_container, observe_for, snapshot

from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout, validate_layout
from roboclock.envs.hospital.tasks.reactive.drug_spill import (
    CONTAIN_SHARE,
    PUDDLE_FULL,
    PUDDLE_START,
    HazardousDrugSpill,
)
from roboclock.envs.hospital.tasks.registry import TASKS, make_legacy_task, make_task
from roboclock.envs.lab.mdp.state import ObjectState

DT = 0.01


def spill(limit=20.0, event=2.0, seed=7):
    config = load_config()
    task = make_task("B3", "medium", seed, config)
    task.spec = replace(task.spec, response_limit=limit, event_time=event)
    task.bind(make_layout(task, config))
    return task


def landed(task, offset=(0.0, 0.0)):
    """The bottle lying on the counter where it fell."""
    bottle = task.layout.objects["bottle"]
    radius = task.layout.shapes["bottle"].radius
    lying = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # axis horizontal
    xy = task.landing + np.array(offset)
    return ObjectState(np.array([*xy, task.layout.table_height + radius + 0.001]), lying)


def tip(task):
    """Run the scene clock until the bottle has been released from the shelf."""
    commands = []
    while task.fall_time is None:
        commands += task.advance(DT)
        task.observe(snapshot(task), DT)
    return commands


def break_bottle(task, offset=(0.0, 0.0)):
    commands = tip(task)
    state = snapshot(task, bottle=landed(task, offset))
    task.time += DT
    task.observe(state, DT)
    assert task.event_time is not None
    commands += task.advance(DT)
    return commands


def world(task, **changes):
    """After the break: the bottle is gone (parked on the floor), the rest as given."""
    bottle = task.layout.objects["bottle"]
    gone = ObjectState(np.array([*task.spill_center, 0.05]), bottle.orientation.copy())
    return snapshot(task, bottle=gone, **changes)


def hold(task, state, seconds=0.45):
    task.observe(state, DT)
    task.observe(state, DT)
    observe_for(task, state, seconds)
    return state


def pad_on_spill(task, offset=(0.0, 0.0), lift=0.0, **kwargs):
    pad = task.layout.objects["pad"]
    z = task.layout.table_height + task.layout.shapes["pad"].half[2] + lift
    position = np.array([*(task.spill_center + np.array(offset)), z])
    return {"pad": ObjectState(position, pad.orientation.copy(), **kwargs)}


def bin_shard(task, name, placed):
    """Lower one glass piece over the bin's lip and let it settle, keeping earlier placements."""
    approach = world(task, **placed, **{name: above_container(task, name, "hd_bin")})
    task.time += DT
    task.observe(approach, DT)
    placed[name] = in_container(task, name, "hd_bin")
    hold(task, world(task, **placed))


def clean_up(task):
    placed = pad_on_spill(task)
    hold(task, world(task, **placed))
    for name in task.shards:
        bin_shard(task, name, placed)
    return placed


# --- registration and layout ---------------------------------------------------------------------


def test_b3_is_the_spill_in_v2_and_the_fire_blanket_in_the_legacy_profile():
    assert TASKS["B3"] is HazardousDrugSpill
    task = make_task("B3")
    assert (task.name, task.category) == ("hazardous_drug_spill", "event_reactive")
    assert make_task("hazardous_drug_spill").task_id == "B3"
    assert make_legacy_task("B3").name == "fire_blanket"


@pytest.mark.parametrize("seed", range(10))
def test_layout_is_valid_and_keeps_the_bottle_and_the_glass_held(seed):
    config = load_config()
    task = make_task("B3", "hard", seed, config)
    layout = make_layout(task, config)
    validate_layout(task, layout, config)
    assert set(layout.objects) == {"bottle", "pad", "shard_0", "shard_1"}
    assert layout.held == {"bottle", "shard_0", "shard_1"}
    assert set(layout.targets) == {"hd_bin"}
    spill_info = layout.info["spill"]
    center, half = np.array(spill_info["center"]), PUDDLE_FULL
    assert "zone_half" not in spill_info  # no marked zone on the counter (review 2026-11-19)
    # The glass comes to rest clear of the spread towel; the kit and the bin stand clear of the puddle.
    for name, (dx, dy) in spill_info["shard_offsets"].items():
        extent = max(layout.shapes[name].half[:2])
        assert max(abs(dx), abs(dy)) - extent > 0.10, name
    for xy in (spill_info["kit"], spill_info["bin"]):
        assert np.max(np.abs(np.array(xy) - center)) > half + 0.08
    # The bottle stands on the lower shelf, above the counter, behind where it lands.
    bottle = layout.objects["bottle"]
    assert bottle.position[2] > layout.table_height + 0.15
    assert bottle.position[0] > center[0] + half


def test_instruction_names_both_limits_and_stays_short():
    task = spill(limit=23.47)
    text = task.gen_instruction()
    assert "9.3 seconds" in text and "23.4 seconds" in text  # rounded down, never overstated
    assert len(text) <= 512


# --- the fall and the break --------------------------------------------------------------------


def test_bottle_is_released_at_the_event_time_and_breaks_on_the_counter():
    task = spill(event=2.0)
    commands = tip(task)
    assert task.fall_time == pytest.approx(2.0, abs=DT)
    released = [c for c in commands if c.kind == "release"]
    assert [c.name for c in released] == ["bottle"] and released[0].value[0] < 0  # toward the robot
    assert task.event_time is None  # still falling: the clock has not started
    # In the air above the counter nothing happens.
    bottle = task.layout.objects["bottle"]
    falling = ObjectState(bottle.position - [0.05, 0.0, 0.06], bottle.orientation.copy())
    task.time += DT
    task.observe(snapshot(task, bottle=falling), DT)
    assert task.event_time is None
    task.observe(snapshot(task, bottle=landed(task, (0.02, -0.01))), DT)
    assert task.event_time == pytest.approx(task.time)
    assert np.allclose(task.spill_center, task.landing + [0.02, -0.01])
    commands = task.advance(DT)
    spill_commands = {c.name: c.value for c in commands if c.kind == "spill"}
    assert np.allclose(spill_commands["break"], task.spill_center)
    assert spill_commands["puddle"] == pytest.approx(PUDDLE_START, abs=0.002)
    assert [c for c in task.advance(DT) if c.kind == "spill" and c.name == "break"] == []


def test_a_bottle_that_never_reaches_the_counter_still_breaks_at_the_landing_point():
    task = spill()
    tip(task)
    stuck = snapshot(task)  # still on the shelf
    observe_for(task, stuck, 1.6)
    assert task.event_time is not None
    assert np.allclose(task.spill_center, task.landing)


def test_touching_anything_before_the_break_is_a_violation():
    task = spill(event=5.0)
    pad = task.layout.objects["pad"]
    state = snapshot(task, pad=ObjectState(pad.position.copy(), pad.orientation.copy(), grasped=True))
    task.time += DT
    task.observe(state, DT)
    assert "pre_event_action" in task.violations


# --- the puddle -----------------------------------------------------------------------------------


def test_puddle_reaches_its_full_extent_exactly_at_the_containment_limit_and_stops():
    task = spill(limit=20.0)
    break_bottle(task)
    assert task.containment_limit == pytest.approx(CONTAIN_SHARE * 20.0)
    assert task.puddle_radius(task.event_time) == pytest.approx(PUDDLE_START)
    at_limit = task.puddle_radius(task.event_time + task.containment_limit)
    assert at_limit == pytest.approx(PUDDLE_FULL)
    assert task.puddle_radius(task.event_time + 0.5 * task.containment_limit) < at_limit
    assert task.puddle_radius(task.event_time + 3 * task.containment_limit) == pytest.approx(PUDDLE_FULL)


def test_an_off_centre_break_spreads_the_same_from_where_it_broke():
    task = spill(limit=20.0)
    break_bottle(task, offset=(0.03, 0.0))
    assert np.allclose(task.spill_center, task.landing + [0.03, 0.0])
    at_limit = task.puddle_radius(task.event_time + task.containment_limit)
    assert at_limit == pytest.approx(PUDDLE_FULL)


def test_instruction_and_layout_name_no_zone_line():
    task = spill()
    text = task.gen_instruction().lower()
    assert "line" not in text and "zone" not in text and "contain dial" in text
    assert "towel" in text


def test_the_pad_is_a_thick_towel_the_gripper_still_spans():
    task = spill()
    half = task.layout.shapes["pad"].half
    assert 2 * half[2] >= 0.05 and 2 * half[1] <= 0.05  # 5 cm thick, 5 cm across the fingers
    assert task.layout.shapes["pad"].grasp_radius == pytest.approx(half[1])


def test_the_pad_freezes_the_puddle_and_the_scene_shows_the_spread_sheet():
    task = spill(limit=20.0)
    break_bottle(task)
    task.time = task.event_time + 3.0
    hold(task, world(task, **pad_on_spill(task)))
    assert task.contained and task.pad in task.correct
    frozen = task.puddle_radius()
    commands = task.advance(DT)
    assert any(c.kind == "spill" and c.name == "contain" for c in commands)
    assert task.puddle_radius(task.time + 30.0) == pytest.approx(frozen)
    assert frozen < PUDDLE_FULL


# --- scoring ----------------------------------------------------------------------------------------


def test_full_sequence_in_time_is_a_joint_success():
    task = spill(limit=30.0)
    break_bottle(task)
    clean_up(task)
    assert task.cleanup_time is not None and not task.goal_achieved  # the scene runs on briefly
    observe_for(task, world(task, **pad_on_spill(task), **{n: in_container(task, n, "hd_bin") for n in task.shards}), 1.6)
    assert task.physical_success and task.temporal_success and task.joint_success
    metrics = task.metrics()
    assert metrics["task_name"] == "hazardous_drug_spill"
    assert metrics["containment_latency"] < metrics["containment_limit"]
    assert set(metrics["shard_times"]) == set(task.shards)
    assert metrics["response_time"] < 30.0 and metrics["drops"] == 0


@pytest.mark.parametrize(
    "change", [{"offset": (0.08, 0.0)}, {"lift": 0.05}, {"grasped": True}]
)
def test_a_pad_beside_the_liquid_held_or_in_the_air_does_not_contain_it(change):
    task = spill()
    break_bottle(task)
    hold(task, world(task, **pad_on_spill(task, **change)))
    assert not task.contained


def test_late_containment_is_physically_fine_but_misses_its_limit():
    task = spill(limit=20.0)
    break_bottle(task)
    task.time = task.event_time + task.containment_limit + 1.0
    clean_up(task)
    assert "late_containment" in task.violations and "late_cleanup" not in task.violations
    observe_for(task, task.previous, 1.6)
    assert task.physical_success and not task.temporal_success
    assert task.puddle_radius() == pytest.approx(PUDDLE_FULL)  # the liquid spread to its full extent


def test_the_containment_limit_is_strict():
    task = spill(limit=20.0)
    break_bottle(task)
    task.time = task.event_time + task.containment_limit - 0.1  # the dwell ends past the limit
    hold(task, world(task, **pad_on_spill(task)), 0.25)
    assert "late_containment" in task.violations


def test_late_clean_up_misses_the_response_limit():
    task = spill(limit=20.0)
    break_bottle(task)
    placed = pad_on_spill(task)
    hold(task, world(task, **placed))
    bin_shard(task, "shard_0", placed)
    task.time = task.event_time + 21.0
    task.advance(DT)
    assert "late_cleanup" in task.violations
    bin_shard(task, "shard_1", placed)
    observe_for(task, task.previous, 1.6)
    assert task.physical_success and not task.temporal_success


def test_an_unanswered_spill_is_late_at_both_limits():
    task = spill(limit=10.0)
    break_bottle(task)
    while task.time < task.event_time + 10.5:
        task.advance(DT)
    assert {"late_containment", "late_cleanup"} <= set(task.violations)
    assert not task.goal_achieved


def test_gripping_glass_before_the_liquid_is_contained_is_a_wrong_order_failure():
    task = spill()
    break_bottle(task)
    shard = task.layout.objects["shard_0"]
    state = world(task, shard_0=ObjectState(shard.position.copy(), shard.orientation.copy(), grasped=True))
    task.time += DT
    task.observe(state, DT)
    assert "glass_before_containment" in task.physical_failures
    clean_up(task)
    observe_for(task, task.previous, 1.6)
    assert not task.physical_success


def test_glass_beside_the_bin_does_not_count_and_a_dropped_piece_fails():
    task = spill()
    break_bottle(task)
    placed = pad_on_spill(task)
    hold(task, world(task, **placed))
    target = task.layout.targets["hd_bin"]
    shard = task.layout.objects["shard_0"]
    beside = ObjectState(
        np.array([target.position[0], target.position[1] + 0.20, shard.position[2]]), shard.orientation.copy()
    )
    hold(task, world(task, **placed, shard_0=beside))
    assert "shard_0" not in task.shard_times
    floor = ObjectState(np.array([0.5, 0.7, 0.03]), shard.orientation.copy())
    hold(task, world(task, **placed, shard_0=floor))
    assert "shard_0" in task.dropped and task.terminated and not task.physical_success


def test_the_parked_bottle_is_not_a_drop():
    task = spill()
    break_bottle(task)
    hold(task, world(task))
    assert not task.dropped and task.dropped_consumables == {"bottle"}


# --- dials, results and stages -----------------------------------------------------------------------


def test_dials_wait_blue_then_count_down_both_limits():
    task = spill(limit=10.0, event=2.0)
    colors = {c.name: c.value for c in task.advance(DT) if c.kind == "indicator"}
    assert set(colors) == {"indicator", "contain"}
    break_bottle(task)
    task.time = task.event_time + 2.0
    timers = {c.name: c.value[0] for c in task.advance(DT) if c.kind == "timer"}
    assert timers["indicator"] == pytest.approx(0.8, abs=0.01)
    assert timers["contain"] == pytest.approx(0.5, abs=0.01)


def test_old_fire_rows_are_retired_and_spill_rows_score_their_stages():
    from roboclock.envs.hospital.results import normalize
    from roboclock.envs.hospital.stages import STAGES

    old = normalize({"task_id": "B3", "task_name": "fire_blanket", "task_success": True})
    assert old["retired"] and old["task_id"] == "B3-fire_blanket" and old["benchmark_id"] is None
    new = normalize({"task_id": "B3", "task_name": "hazardous_drug_spill", "task_success": True})
    assert new["benchmark_id"] == "B3" and not new.get("retired")
    names = [s.name for s in STAGES["B3"]({"task_name": "hazardous_drug_spill"})]
    assert names == ["spill_contained", "first_glass_in_bin"]
    assert [s.name for s in STAGES["B3"]({"task_name": "fire_blanket"})][0] == "first_tab_pulled"


def test_glass_leaning_on_the_bin_wall_is_in_the_bin():
    """Seed 6 of the first oracle batch: the neck piece bounced against the wall and never counted."""
    task = spill()
    break_bottle(task)
    placed = pad_on_spill(task)
    hold(task, world(task, **placed))
    target = task.layout.targets["hd_bin"]
    resting = in_container(task, "shard_0", "hd_bin")
    half = task.layout.shapes["shard_0"].half
    against = resting.position + [0.0, target.half_extents[1] - half[1] + 0.005, 0.0]  # 5 mm into the wall's lean
    hold(task, world(task, **placed, shard_0=replace(resting, position=against)))
    assert "shard_0" in task.shard_times
