"""Dual-X5 household task/process/scoring models driven by synthetic measured signals.

These tests exercise the evaluator contract and the timing causality of each task
without Isaac Sim: the same scripted physical sequence succeeds on time and fails
when it happens too early or too late.
"""

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

from roboclock.envs.household.c1_x5.task import HARD_PICKUPS_S
from roboclock.envs.household.tasks.registry import CATALOG, TASKS, make_task
from roboclock.evaluation import throughput as TP
from roboclock.tasks.registry import make_task as shared_make_task

DT = 0.05


def run(task, script, until=None):
    """script(t) -> signals dict for the current step; runs until terminated or `until`."""
    end = task.spec.max_time if until is None else until
    while task.time < end - 1e-9 and not task.terminated:
        task.update(script(task.time))
        task.advance(DT)
    return task.metrics()


def test_catalog_contract():
    assert [row["id"] for row in CATALOG["tasks"]] == list(TASKS)
    for tid, cls in TASKS.items():
        task = make_task(tid, "medium", 0)
        assert task.spec.max_time > 0 and task.gen_instruction()
        json.dumps(dataclasses.asdict(task.spec))
        for key, (desc, _default) in cls.SIGNALS.items():
            assert desc
        metrics = task.metrics()
        for field in ("task_id", "difficulty", "category", "task_success", "manipulation_success",
                      "temporal_success", "joint_success", "temporal_violations"):
            assert field in metrics
    assert shared_make_task("A2", "hard", 4, environment="household").spec.time_scale == 1.5


def test_hidden_parameters_are_seeded_and_bounded():
    a, b = make_task("A2", "medium", 7), make_task("A2", "medium", 7)
    assert a.spec == b.spec
    assert 0.6 <= a.spec.spin_up_s <= 1.3 and 1.4 <= a.spec.interval_s <= 1.7
    assert make_task("A2", "medium", 0).spec.interval_s == 1.5
    with pytest.raises(KeyError):
        a.update({"not_a_signal": 1})


@pytest.mark.parametrize("task_id", ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C2"])
def test_spatial_layout_is_seeded_bounded_and_independent_of_difficulty(task_id):
    jitter = make_task(task_id, "medium", 0).spec.LAYOUT_JITTER
    assert jitter
    # seed 0 is the nominal layout
    assert all(entry[1:] == (0.0, 0.0, 0.0) for entry in make_task(task_id, "medium", 0).spec.layout)
    layouts = set()
    for seed in range(1, 20):
        specs = [make_task(task_id, d, seed).spec for d in ("easy", "medium", "hard")]
        assert len({s.layout for s in specs}) == 1  # the same scene at every tier
        layout = specs[0].layout
        assert [e[0] for e in layout] == list(jitter)
        for name, *values in layout:
            for v, r in zip(values, jitter[name]):
                lo, hi = (-r, r) if isinstance(r, float) else r
                assert lo - 1e-9 <= v <= hi + 1e-9
            assert specs[0].offset(name) == tuple(values)
        layouts.add(layout)
        json.dumps(dataclasses.asdict(specs[0]))
    assert len(layouts) == 19
    # a dedicated stream: the hidden-parameter draws are those of the same seed without a layout
    spec = make_task(task_id, "medium", 5).spec
    plain = type(spec).sample.__func__(type(spec), task_id, "medium", 5, layout=())
    assert dataclasses.replace(spec, layout=()) == plain


def a1_script(pour_s, over_glass=True, start=2.0):
    """Tilt the jar to 90 degrees (full flow) for pour_s seconds from `start`, stand it upright and let go 0.5 s
    later, stir two turns from 3 s after that and put the spoon back."""
    down = start + pour_s + 0.5

    def script(t):
        pouring = start <= t < start + pour_s
        return dict(jar_tilt_deg=90.0 if pouring else 0.0, mouth_over_glass=over_glass,
                    jar_upright_released=not (start - 1 <= t < down), spoon_in_cup=down + 3 <= t < down + 9,
                    stir_turns=0.0 if t < down + 3 else min(2.1, (t - down - 3) / 2.5),
                    spoon_in_basket_released=t >= down + 10)
    return script


@pytest.mark.parametrize("pour_s,ok", [(1.5, False), (4.5, True), (9.0, False)])
def test_a1_window(pour_s, ok):
    m = run(make_task("A1", "medium", 0), a1_script(pour_s))
    assert m["manipulation_success"] and m["temporal_success"] is ok, m


def test_a1_band_is_the_tier():
    # The user's bands (2026-11-19): easy 10-30 g, medium 10-25 g, hard 10-15 g; the pour is the same at every tier.
    bands = {d: make_task("A1", d, 0).spec.band for d in ("easy", "medium", "hard")}
    assert bands == {"easy": (10.0, 30.0), "medium": (10.0, 25.0), "hard": (10.0, 15.0)}
    assert len({make_task("A1", d, 0).spec.max_flow_g_s for d in bands}) == 1
    about_22 = {d: run(make_task("A1", d, 0), a1_script(6.0)) for d in bands}  # ~22 g
    assert 20 < about_22["easy"]["sugar_g"] < 24
    assert [about_22[d]["temporal_success"] for d in ("easy", "medium", "hard")] == [True, True, False]
    about_12 = run(make_task("A1", "hard", 0), a1_script(3.0))
    assert 10 < about_12["sugar_g"] < 15 and about_12["temporal_success"]


def test_a1_hard_band_lasts_at_least_one_second():
    # Time-limit rule (2026-11-17): no window below 1 s; at the fastest hidden flow, full tilt, the 5 g band lasts
    # 5 / (4 x 1.15) s.
    from roboclock.envs.household.a1_x5.task import A1Spec

    lo, hi = A1Spec.hidden_ranges()["flow_coefficient"]
    spec = make_task("A1", "hard", 0).spec
    assert (spec.band[1] - spec.band[0]) / (spec.max_flow_g_s * hi) >= 1.0


def test_a1_spill_and_early_stir_fail():
    spilled = run(make_task("A1", "medium", 0), a1_script(4.5, over_glass=False))
    assert not spilled["manipulation_success"] and "sugar_spilled" in spilled["physical_failures"]

    def early(t):
        return dict(a1_script(4.5)(t), spoon_in_cup=3.0 <= t < 4.0 or a1_script(4.5)(t)["spoon_in_cup"])
    m = run(make_task("A1", "medium", 0), early)
    assert not m["manipulation_success"]


def a2_script(task, press_at, release_after, after_cube=5, per_cube=0.10):
    """Press the paddle at press_at; release `release_after` s after cube `after_cube` is delivered; every delivered
    cube lands in the glass and raises the ice by `per_cube` of the glass; the glass stands released on the counter
    2 s after the release."""
    state = {}

    def script(t):
        if "fifth" not in state and task.dispensed >= after_cube:
            state["fifth"] = t
        released = "fifth" in state and t >= state["fifth"] + release_after
        return dict(paddle_deg=8.0 if t >= press_at and not released else 0.0, cubes_in_glass=task.dispensed,
                    ice_level=per_cube * task.dispensed,
                    glass_on_counter_released=t < press_at - 3 or (released and t >= state["fifth"] + release_after + 2))
    return script


@pytest.mark.parametrize("after_cube,release_after,level,ok",
                         [(3, 0.2, 0.3, False), (4, 0.2, 0.4, True), (5, 0.2, 0.5, True), (5, 1.0, 0.5, True),
                          (6, 1.4, 0.7, False)])
def test_a2_release_window(after_cube, release_after, level, ok):
    # medium seed 0: one cube every 1.5 s after a 0.9 s spin-up; the auger coasts 0.12 s after the release (the
    # scripted release lags the delivery by up to two 0.05 s steps). Band 0.40-0.60 of the glass.
    task = make_task("A2", "medium", 0)
    m = run(task, a2_script(task, 5.0, release_after, after_cube))
    assert m["manipulation_success"] and m["final_ice_level"] == pytest.approx(level) and m["temporal_success"] is ok


def test_a2_hard_runs_faster():
    task = make_task("A2", "hard", 3)
    m = run(task, a2_script(task, 5.0, 0.1))
    d = m["delivery_times_if_held_s"]
    pressed = m["event_times"]["paddle_pressed"]
    assert m["temporal_success"] and d[0] - pressed == pytest.approx(task.spec.spin_up_s / 1.5, abs=1e-6)
    assert (d[5] - d[0]) / 5 == pytest.approx(task.spec.interval_s / 1.5, rel=0.06)
    assert (d[5] - d[0]) / 5 < (make_task("A2", "medium", 3).due[5] - make_task("A2", "medium", 3).due[0]) / 5


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_a2_release_window_at_least_one_second(difficulty):
    # The band can hold a single cube count (measured): then the release must come after that cube and at least the
    # coast before the next one is due. That worst case, one wall interval minus the coast, is >= 1.0 s everywhere.
    worst = []
    for seed in range(50):
        task = make_task("A2", difficulty, seed)
        gaps = [b - a for a, b in zip(task.due, task.due[1:])]
        worst.append((min(gaps) - task.spec.coast_s) / task.spec.time_scale)
    assert min(worst) >= 1.0


def test_a2_cube_outside_the_glass_fails():
    task = make_task("A2", "medium", 0)

    def script(t):
        return dict(a2_script(task, 5.0, 0.2)(t), cubes_outside=1 if t > 9 else 0)
    m = run(task, script)
    assert not m["manipulation_success"] and "cube_missed_glass" in m["physical_failures"]


@pytest.mark.parametrize("flip_at,ok", [(15.0, False), (50.0, True), (95.0, False)])
def test_a3_window(flip_at, ok):
    def script(t):
        return dict(pancake_on_pan=not (flip_at - 1 <= t < flip_at + 1), pancake_rotation_deg=180.0 if t >= flip_at else 0.0,
                    pancake_in_pan_released=t >= flip_at + 2, turner_clear=t >= flip_at + 2)
    m = run(make_task("A3", "medium", 0), script)
    assert m["manipulation_success"] and m["temporal_success"] is ok


@pytest.mark.parametrize("stop_at,ok", [(9.5, False), (17.85, True), (22.0, False)])
def test_a4_stop_interval(stop_at, ok):
    # Tilt fixed at 75 deg from t = 5 s: ~19 ml/s with a full 1.0 L kettle.
    def script(t):
        pouring = 5 <= t < stop_at
        return dict(kettle_tilt_deg=75.0 if pouring else 0.0, spout_over_jug=True,
                    kettle_upright_released=not (5 <= t < stop_at + 1))
    m = run(make_task("A4", "medium", 0), script)
    assert m["manipulation_success"] and m["temporal_success"] is ok, m["final_dose_g"]


def test_a4_a_tilt_that_pours_nothing_is_not_a_pour():
    # The kettle is bumped past 5 deg (no water leaves it below ~43 deg) and stands again. That
    # used to end the episode 3 s later as a physical success with a 0 g dose.
    def bump(t):
        return dict(kettle_tilt_deg=12.0 if 2 <= t < 3 else 0.0, spout_over_jug=False,
                    kettle_upright_released=not (2 <= t < 3))
    task = make_task("A4", "medium", 0)
    m = run(task, bump)
    assert "dose_final" not in task.events and task.time >= task.spec.max_time - 1e-6
    assert not m["manipulation_success"] and m["final_dose_g"] is None


def test_a4_water_that_all_misses_the_jug_is_a_failure():
    def beside(t):
        pouring = 5 <= t < 5.2
        return dict(kettle_tilt_deg=60.0 if pouring else 0.0, spout_over_jug=False,
                    kettle_upright_released=not (5 <= t < 6))
    task = make_task("A4", "medium", 0)
    m = run(task, beside)
    assert 0 < task.spill_g < task.spec.spill_limit_g and task.jug_g == 0
    assert not m["manipulation_success"] and task.physical_failures == ["no_water_in_jug"]


def test_a4_instruction_states_the_scored_budget():
    for difficulty, budget in (("easy", 115), ("medium", 64), ("hard", 36)):
        task = make_task("A4", difficulty, 0)
        assert task.spec.budget_s == budget
        assert f"finish within {budget} seconds" in task.gen_instruction()
    assert make_task("A4", "medium", 0).gen_instruction() == CATALOG["tasks"][3]["goal"]


def test_a4_scale_display_lags():
    task = make_task("A4", "medium", 0)
    run(task, lambda t: dict(kettle_tilt_deg=90.0, spout_over_jug=True, kettle_upright_released=False), until=3.0)
    assert task.cues()["scale_reading_g"] < task.jug_g


@pytest.mark.parametrize("close_at,overflow_l,ok", [(4.0, 0.0, True), (5.2, 0.29, True), (6.0, 0.5, False)])
def test_b1_overflow_loss(close_at, overflow_l, ok):
    # Gentle flow for 2 s, then 0.45 L/s: nominal overflow starts near 4.56 s.
    def script(t):
        return dict(lever_opening=0.0 if t >= close_at else 1.0, hand_on_lever=close_at - 1 <= t < close_at + 0.5)
    m = run(make_task("B1", "medium", 0), script)
    assert m["overflow_l"] == pytest.approx(overflow_l, abs=0.06)
    assert m["temporal_success"] is ok and (m["manipulation_success"] or not ok)


def test_b1_surge_requires_reaction_after_event():
    task = make_task("B1", "medium", 0)
    assert task.inflow() == pytest.approx(0.025)
    run(task, lambda t: dict(lever_opening=1.0, hand_on_lever=False), until=2.1)
    assert task.inflow() == pytest.approx(0.45)
    assert task.events["event"] == pytest.approx(2.0, abs=0.02)
    assert task.cues()["surging"]

    early = make_task("B1", "medium", 0)
    m = run(early, lambda t: dict(lever_opening=0.0, hand_on_lever=t < 0.5))
    assert not m["temporal_success"]
    assert "pre_event_action" in early.physical_failures


def b2_script(fall=False, peak=5.0, others=0.0):
    def script(t):
        moving = 2.0 <= t < 3.2
        edge = 0.5 - max(0.0, min(t, 3.2) - 2.0) * 0.35
        if fall and t >= 3.2:
            edge = 0.5 - (t - 2.0) * 0.5
        return dict(released=t >= 2.0, object_speed=0.5 if moving or fall else 0.0, edge_distance=edge,
                    object_fell=fall and edge < -0.05, object_held=not fall and t >= 3.8,
                    peak_contact_n=peak, others_moved_m=others)
    return script


def test_b2_catch_succeeds_and_times_are_reported():
    m = run(make_task("B2", "medium", 0), b2_script())
    assert m["joint_success"] and m["object"] == "egg" and m["board_deg"] == 6.0
    assert m["stop_latency"] == pytest.approx(1.4, abs=0.1) and m["edge_margin_m"] > 0


def test_b2_fall_fails():
    m = run(make_task("B2", "medium", 0), b2_script(fall=True))
    assert not m["joint_success"] and "object_fell" in m["physical_failures"]


@pytest.mark.parametrize("kw,failure", [(dict(peak=40.0), "object_crushed"), (dict(others=0.05), "wrong_object_disturbed")])
def test_b2_physical_failures(kw, failure):
    m = run(make_task("B2", "medium", 0), b2_script(**kw))
    assert not m["manipulation_success"] and failure in m["physical_failures"]


def test_b2_difficulty_steepens_the_board_and_seeds_pick_the_object():
    assert [make_task("B2", d, 0).spec.board_deg for d in ("easy", "medium", "hard")] == [4.5, 6.0, 9.0]
    specs = [make_task("B2", "medium", s).spec for s in range(1, 30)]
    assert {s.target_name for s in specs} == {"cup", "egg", "pill_bottle", "apple"}
    assert all(1.0 <= s.release_time <= 3.0 for s in specs)


@pytest.mark.parametrize("off_at,ok", [(6.0, True), (55.0, False)])
def test_b3_burner_off(off_at, ok):
    # nominal flame 0.15 growing at 0.065/s reaches the spread threshold 0.8 at about 48 s
    def script(t):
        return dict(knob_angle_deg=0.0 if t >= off_at else 90.0, hand_on_knob=off_at - 1 <= t < off_at + 0.5)
    m = run(make_task("B3", "medium", 0), script)
    assert m["temporal_success"] is ok
    if ok:
        assert m["flame_out_time"] < off_at + 5 and m["flame_exposure"] < 3


def test_b3_water_is_failure():
    m = run(make_task("B3", "medium", 0), lambda t: dict(water_contact=t > 5))
    assert "water_on_oil_fire" in m["physical_failures"]


def test_b3_ignition_is_seeded_bounded_and_independent_of_difficulty():
    times = [make_task("B3", "medium", seed).spec.ignition_time for seed in range(1, 30)]
    assert all(2.0 <= t <= 8.0 for t in times)
    assert len(set(times)) == len(times)
    for seed in (0, 7, 50):
        specs = [make_task("B3", difficulty, seed).spec for difficulty in ("easy", "medium", "hard")]
        assert len({s.ignition_time for s in specs}) == 1
        assert specs[1] == make_task("B3", "medium", seed).spec
    assert make_task("B3", "medium", 0).spec.ignition_time == 5.0


def test_b3_has_no_flame_exposure_or_completion_before_ignition():
    task = make_task("B3", "medium", 50)
    assert task.cues()["flame_intensity"] == 0.0 and not task.events
    task.advance(task.spec.ignition_time - 0.01)
    assert task.flame == task.exposure == 0.0
    assert not task.events and not task.terminated
    task.advance(0.01)
    assert task.events["event"] == pytest.approx(task.spec.ignition_time)
    assert task.flame == pytest.approx(task.spec.f0)
    assert task.exposure == pytest.approx(0.0)


def test_b3_turning_off_before_ignition_does_not_complete_the_task():
    task = make_task("B3", "medium", 50)
    m = run(task, lambda t: dict(knob_angle_deg=0.0), until=12.0)
    assert not m["joint_success"] and not m["event_times"]
    assert task.flame == task.exposure == 0.0


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_b3_response_and_spread_clocks_start_at_ignition(difficulty):
    outcomes = []
    for delay in (2.0, 8.0):
        task = make_task("B3", difficulty, 0, ignition_time=delay)
        m = run(task, lambda t: dict(knob_angle_deg=0.0 if t >= delay + 6.0 - 1e-9 else 90.0))
        assert m["joint_success"]
        outcomes.append((m["flame_exposure"], m["flame_out_time"] - m["ignition_time"]))
    assert outcomes[0] == pytest.approx(outcomes[1], abs=0.06)


def c1_script(spec, lead=6.0, serve=None, late=None):
    """Signals as the relay serves `serve[i]` (default: the wanted meal) `lead` s before pickup i; the
    waiter takes it at the pickup. `late` = {i: s}: the meal of order i is set down s after its pickup instead,
    and stays on the board until the task's waiter has taken it (as the simulation's waiter does)."""
    serve = spec.order if serve is None else serve
    late = late or {}
    state = {"task": None}

    def script(t):
        task = state["task"]
        loc, board = ["counter"] * 4, -1
        for i, b in enumerate(serve):
            down = spec.pickup_times[i] + late[i] if i in late else spec.pickup_times[i] - lead
            if task is not None and task.taken[b] is not None:
                loc[b] = "taken"  # the waiter took it (the simulation moves it off the counter on its next tick)
            elif t >= down:
                loc[b], board = "board", b
        return dict(bowl_location=tuple(loc), board_bowl=board)
    script.state = state
    return script


def run_c1(task, script):
    script.state["task"] = task
    return run(task, script)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_c1_serves_the_wanted_meal_at_each_pickup(difficulty):
    task = make_task("C1", difficulty, 3)
    m = run_c1(task, c1_script(task.spec))
    assert m["qualified_meals"] == m["orders"] == 4  # the same four orders at every tier
    assert m["joint_success"] and m["wrong_meals"] == 0 and m["late_jobs"] == []
    assert all(s == pytest.approx(6.0, abs=0.1) for s in m["slack_s"])
    assert sum(e["type"].endswith("_served") for e in m["events"]) == m["orders"]  # stage scores count these
    # Every ordered meal served and the last pickup passed: the episode ends SETTLE_S later.
    assert m["termination_reason"] == "all_done"
    assert m["elapsed_time"] == pytest.approx(task.spec.pickup_times[-1] + 2.0, abs=2 * DT)
    assert m["job_deadlines_s"] == pytest.approx(list(task.spec.pickup_times))


def test_c1_wrong_meal_late_meal_and_drop():
    spec = make_task("C1", "medium", 3).spec
    swapped = (spec.order[1], spec.order[0], *spec.order[2:])
    task = make_task("C1", "medium", 3)
    m = run_c1(task, c1_script(spec, serve=swapped))
    # Pickup 1 takes the second order's meal (wrong); the first order's meal, set down later, is overdue and
    # taken as soon as it rests on the board, so pickup 2 finds the board empty. The second order's meal went to
    # the first guest, so the second order is never done: not physical, and late.
    assert m["wrong_meals"] == 1 and "wrong_meal_served" in m["temporal_violations"] and m["late_pickups"] == [1]
    assert m["qualified_meals"] == 2 and not m["task_success"] and not m["temporal_success"]
    assert m["late_jobs"] == [0, 1] and m["job_times_s"][1] is None
    assert m["termination_reason"] == "horizon"  # an undone order never ends the episode early
    nothing = run(make_task("C1", "medium", 3), lambda t: dict(bowl_location=("counter",) * 4, board_bowl=-1))
    assert nothing["qualified_meals"] == 0 and not nothing["temporal_success"]
    assert not nothing["task_success"]  # a robot that serves nothing is not physically complete
    assert nothing["termination_reason"] == "horizon"
    assert nothing["elapsed_time"] == pytest.approx(1.25 * TP.MULTIPLIER['medium'] * HARD_PICKUPS_S[-1], abs=DT)  # medium H
    dropped = run(make_task("C1", "medium", 3), lambda t: dict(bowl_location=("dropped",) + ("counter",) * 3, board_bowl=-1))
    assert "bowl1_dropped" in dropped["physical_failures"] and not dropped["task_success"]


def test_c1_all_wrong_deliveries_are_not_physical():
    # Each meal set down for the next order: every pickup takes a wrong meal, and the last meal is taken late
    # for the first order. Only the first order is ever done.
    spec = make_task("C1", "medium", 3).spec
    rotated = (*spec.order[1:], spec.order[0])
    m = run_c1(make_task("C1", "medium", 3), c1_script(spec, serve=rotated))
    assert m["wrong_meals"] == 3 and m["qualified_meals"] == 0
    assert not m["task_success"] and not m["joint_success"]
    assert [t is None for t in m["job_times_s"]] == [False, True, True, True]


def test_c1_bowl_tipped_on_the_board_loses_its_served_time():
    task = make_task("C1", "medium", 3)
    spec = task.spec
    base = c1_script(spec)
    first = spec.order[0]

    def script(t):  # the first meal tips past the upright limit on the board 1 s after it was set down
        sig = base(t)
        if sig["board_bowl"] == first and t >= spec.pickup_times[0] - 5.0:
            return dict(sig, board_bowl=-1)  # still located on the board, but not upright
        return sig
    script.state = base.state
    m = run_c1(task, script)
    assert m["job_times_s"][0] is None and not m["task_success"] and 0 in m["late_jobs"]


def test_c1_late_meal_counts_for_physical_and_frees_the_board():
    task = make_task("C1", "medium", 3)
    spec = task.spec
    m = run_c1(task, c1_script(spec, late={1: 3.0}))  # order 2 set down 3 s after its pickup
    assert m["task_success"] and not m["temporal_success"] and not m["joint_success"]
    assert m["late_jobs"] == [1] and m["late_pickups"] == [2]
    assert m["job_times_s"][1] == pytest.approx(spec.pickup_times[1] + 3.0, abs=2 * DT)
    late = m["event_times"]["late_pickup2"]
    assert late == pytest.approx(spec.pickup_times[1] + 4.0, abs=2 * DT)  # rested LATE_TAKE_S = 1 s on the board
    assert m["qualified_meals"] == 3 and m["wrong_meals"] == 0  # the next orders were served on time
    assert m["termination_reason"] == "all_done"


def test_c1_served_meal_lifted_off_the_board_does_not_count():
    task = make_task("C1", "medium", 3)
    spec = task.spec
    first = spec.order[0]

    def script(t):
        loc, board = ["counter"] * 4, -1
        if 10.0 <= t < 12.0:
            loc[first], board = "board", first
        elif t >= 12.0:
            loc[first] = "held"  # picked up again before the waiter came, never set down again
        return dict(bowl_location=tuple(loc), board_bowl=board)
    m = run(task, script)
    assert f"bowl{first + 1}_served" in m["event_times"]  # the stage still records the first set-down
    assert m["job_times_s"][0] is None and m["served_s"][0] is None
    assert not m["task_success"] and not m["temporal_success"] and m["qualified_meals"] == 0


def test_c1_difficulty_schedule():
    # C-class stream rule (docs/c-class-timing.md): every tier serves the same four orders; its pickups are
    # MULTIPLIER = 4 / sqrt(5) / 1.25 x the reference schedule, the first pickup and the intervals alike, and nothing else
    # scales (time_scale 1). The horizon is 1.25 x the tier's last pickup.
    # Frozen in task.py from oracle runs: first set-down 10.524 s on every seed, slowest relay turnaround 12.96 s.
    assert HARD_PICKUPS_S == (10.53, 23.49, 36.45, 49.41)
    for d, m in TP.MULTIPLIER.items():
        for seed in (0, 5):
            s = make_task("C1", d, seed).spec
            assert len(s.order) == 4 and sorted(s.order) == [0, 1, 2, 3] and s.time_scale == 1.0
            assert s.pickup_times == s.pickups_s == pytest.approx([m * p for p in HARD_PICKUPS_S])
            assert s.max_time == pytest.approx(1.25 * s.pickup_times[-1])
    assert make_task("C1", "easy", 0).spec.pickup_times == pytest.approx([42.12, 93.96, 145.8, 197.64])  # easy unchanged
    horizons = [make_task("C1", d, 0).spec.max_time for d in ("easy", "medium", "hard")]
    assert horizons == pytest.approx([1.25 * m * 49.41 for m in (4.0, 5**0.5, 1.25)])
    with pytest.raises(ValueError):
        make_task("C1", "extreme", 0)


def test_c1_layout_is_seeded():
    from roboclock.envs.household.c1_x5.task import SLOTS, layout

    order0, slots0 = layout(0)
    assert order0 == (0, 1, 2, 3) and slots0[0] is None and slots0[1:] == SLOTS
    seen = set()
    for seed in range(1, 40):
        order, slots = layout(seed)
        assert layout(seed) == (order, slots) and sorted(order) == [0, 1, 2, 3]
        assert slots[order[0]] is None and all(slots[b] is not None for b in order[1:])  # first wanted meal on the mat
        for b in order[1:]:  # each waiting meal in its own slot, within the jitter
            k = min(range(3), key=lambda k: abs(slots[b][0] - SLOTS[k][0]) + abs(slots[b][1] - SLOTS[k][1]))
            assert abs(slots[b][0] - SLOTS[k][0]) <= .012 and abs(slots[b][1] - SLOTS[k][1]) <= .012
            assert abs(slots[b][2] - SLOTS[k][2]) <= 8.0
        seen.add(order)
    assert len(seen) > 10  # the wanted order varies
    task = make_task("C1", "medium", 5)
    assert all(m in task.gen_instruction() for m in ("curry", "rice", "soup", "pasta"))
    assert "23.55, 52.53, 81.5 and 110.48 seconds" in task.gen_instruction()  # medium: sqrt(5) x the reference pickups


def c2_script(task, rate=20.0, final=92.0, spill_rate=0.0):
    """Measured volumes as each arm fills its three glasses in turn at `rate` ml/s to `final` ml, 1 s apart."""
    def script(t):
        ml, air, bottle = [0.0] * 6, 0.0, [330.0, 330.0]
        for b in range(2):
            start = 2.0
            for j in (3 * b, 3 * b + 1, 3 * b + 2):
                dur = final / rate
                fill = min(final, max(0.0, (t - start) * rate))
                ml[j] = fill
                bottle[b] -= fill
                if start <= t < start + dur:
                    air += 2.0
                start += dur + 1.0
        done = t >= 2.0 + 3 * (final / rate + 1.0) + 2.0
        return dict(glass_ml=tuple(ml), air_ml=air, bottle_ml=tuple(bottle), spilled_ml=spill_rate * t,
                    bottle_released_upright=(done, done))
    return script


def test_c2_two_arms_serve_six_glasses():
    task = make_task("C2", "medium", 0)
    m = run(task, c2_script(task))
    assert m["served_glasses"] == 6 and m["joint_success"], m
    # t_Q is all_served (six served, both bottles back for 1 s), as in the frozen t_o; the episode ends
    # SETTLE_S after all is done.
    assert m["t_q_s"] == m["finish_time"] == m["completion_time"] == pytest.approx(21.8, abs=2 * DT)
    assert m["temporal_success_by_tier"] == {"easy": True, "medium": True, "hard": True}
    assert m["termination_reason"] == "all_done" and m["elapsed_time"] == pytest.approx(22.8, abs=2 * DT)
    assert m["jobs_done"] == 6 and all(t is not None for t in m["job_times_s"])


def test_c2_overfill_and_spill():
    task = make_task("C2", "medium", 0)
    m = run(task, c2_script(task, final=120.0))
    # An overfilled glass never comes back into the band, so physical success fails on the final band check.
    assert "glass1_overfilled" in m["event_times"] and m["overfilled_glasses"] == [1, 2, 3, 4, 5, 6]
    assert m["served_glasses"] == 0 and not m["task_success"] and not m["temporal_success"]
    assert m["temporal_violations"] == []
    wet = make_task("C2", "medium", 0)
    m = run(wet, c2_script(wet, spill_rate=1.0))
    assert "spill" in m["physical_failures"] and m["termination_reason"] == "failure"
    assert m["elapsed_time"] == pytest.approx(15.0, abs=2 * DT)


def test_c2_one_run_scored_at_every_tier():
    rows = {d: run(make_task("C2", d, 0), c2_script(make_task("C2", d, 0), rate=5.0)) for d in ("easy", "medium", "hard")}
    for d, m in rows.items():  # the same episode at every difficulty ...
        assert m["task_success"] and m["t_q_s"] == pytest.approx(rows["easy"]["t_q_s"])
        assert m["temporal_success_by_tier"] == {"easy": True, "medium": True, "hard": False}
        assert m["temporal_success"] is m["temporal_success_by_tier"][d]  # ... the difficulty picks the tier
    assert 1.25 * 43.21 < rows["easy"]["t_q_s"] < TP.MULTIPLIER['medium'] * 43.21


def test_c2_slow_pairs_are_not_stalled():
    # 3 ml/s: the glasses come in pairs every ~32 s, slower than the oracle but within the stall window
    # (1.5 x the easy tier's time per glass = 43.3 s > 0.5 t_o), so the episode runs on.
    task = make_task("C2", "medium", 0)
    m = run(task, c2_script(task, rate=3.0))
    gaps = sorted(t for t in m["job_times_s"] if t is not None)
    assert max(b - a for a, b in zip(gaps, gaps[1:])) < TP.stall_window(43.21, 6)
    assert m["termination_reason"] != "stall"


def test_c2_stall_stop():
    # Two glasses served, then nothing: from T_medium the stall stop ends the episode once no glass was
    # served for the stall window.
    task = make_task("C2", "medium", 0)
    base = c2_script(task)
    first = run(make_task("C2", "medium", 0), base)["job_times_s"]
    cut = sorted(t for t in first if t is not None)[1] + 0.5  # just after the 2nd glass

    def script(t):
        return base(min(t, cut))
    m = run(task, script)
    last = max(t for t in m["job_times_s"] if t is not None)
    assert m["termination_reason"] == "stall" and not m["task_success"]
    assert m["elapsed_time"] == pytest.approx(max(TP.MULTIPLIER['medium'] * 43.21, last + TP.stall_window(43.21, 6)), abs=2 * DT)
    assert not m["temporal_success"] and m["t_q_s"] is None


def test_c2_physical_needs_all_six_valid_at_the_end():
    task = make_task("C2", "medium", 0)
    base = c2_script(task)

    def script(t):  # all served at 21.8 s (t_Q), then glass 6 is topped up past the band before the early end
        sig = base(t)
        if t >= 22.2:
            ml = list(sig["glass_ml"])
            ml[5] = 110.0
            sig = dict(sig, glass_ml=tuple(ml))
        return sig
    m = run(task, script)
    assert m["t_q_s"] == pytest.approx(21.8, abs=2 * DT) and m["temporal_success"]  # on time ...
    assert m["served_glasses"] == 5 and not m["task_success"] and not m["joint_success"]  # ... but not complete
    assert m["termination_reason"] == "stall"


def c3_script(times, names=("Spoon1", "Spoon2", "Fork1", "Fork2", "Knife1", "Knife2")):
    """One more utensil sorted at each of `times` (s)."""
    def script(t):
        k = sum(t >= x for x in times)
        return dict(correct=k, placed=tuple(names[:k]))
    return script


def test_c3_cutlery_all_six_judged_by_tier():
    task = make_task("C3", "medium", 0)
    assert "second" not in task.gen_instruction() and "as quickly as possible" in task.gen_instruction()
    m = run(task, c3_script([6, 12, 18, 24, 30, 36]))  # one more utensil every 6 s
    assert m["correct"] == 6 and m["task_success"] and m["joint_success"] and task.terminated
    assert m["t_q_s"] == pytest.approx(36.0, abs=2 * DT) and m["first_sorted_times"][0] == pytest.approx(6.0, abs=2 * DT)
    assert m["temporal_success_by_tier"] == {"easy": True, "medium": True, "hard": False}  # T_hard = 1.25 t_o = 32.46 s
    assert m["termination_reason"] == "all_done" and m["elapsed_time"] == pytest.approx(38.0, abs=2 * DT)
    assert m["jobs_required"] == 6 and m["jobs_done"] == 6 and m["misplaced"] == 0
    hard = run(make_task("C3", "hard", 0), c3_script([6, 12, 18, 24, 30, 36]))
    assert hard["task_success"] and not hard["temporal_success"] and hard["t_q_s"] == m["t_q_s"]
    # Sorted at a slower pace: still physically complete, on time only at easy.
    slow = run(make_task("C3", "medium", 0), c3_script([11, 22, 33, 44, 55, 66]))
    assert slow["task_success"] and slow["temporal_success_by_tier"] == {"easy": True, "medium": False, "hard": False}


def test_c3_physical_needs_all_six_valid_at_the_end():
    def leaves(t):  # all six sorted by 30 s, then one is knocked out of its compartment for good
        sig = c3_script([5, 10, 15, 20, 25, 30])(t)
        return dict(correct=5, placed=sig["placed"][:5]) if t >= 31.0 else sig
    m = run(make_task("C3", "medium", 0), leaves)
    assert m["temporal_success"] and not m["task_success"] and not m["joint_success"]  # t_Q 30 s, but 5 at the end
    assert m["termination_reason"] == "stall"
    wrong = run(make_task("C3", "medium", 0), lambda t: dict(correct=5, wrong=1))
    assert not wrong["task_success"] and wrong["misplaced"] == 1
    dropped = run(make_task("C3", "medium", 0), lambda t: dict(correct=5, dropped=int(t > 10)))
    assert "utensil_dropped" in dropped["physical_failures"] and not dropped["joint_success"]


def test_c3_stall_stop():
    t_o = 25.97
    m = run(make_task("C3", "medium", 0), c3_script([5, 10, 15]))  # three sorted, then nothing
    assert m["termination_reason"] == "stall" and m["elapsed_time"] == pytest.approx(TP.MULTIPLIER["medium"] * t_o, abs=2 * DT)
    late = run(make_task("C3", "medium", 0), c3_script([5, 10, 15, 50]))  # the 4th at 50 s keeps it going
    fourth = late["job_times_s"][3]
    assert fourth == pytest.approx(50.0, abs=2 * DT) and not late["task_success"]
    assert late["termination_reason"] == "stall"
    assert late["elapsed_time"] == pytest.approx(fourth + TP.stall_window(t_o, 6), abs=DT)


def test_c_batch_tasks_identical_across_difficulties():
    from roboclock.evaluation import throughput as TP

    for tid in ("C2", "C3"):
        specs = {d: make_task(tid, d, 4).spec for d in ("easy", "medium", "hard")}
        assert {dataclasses.replace(s, difficulty="medium") for s in specs.values()} == {specs["medium"]}
        assert specs["hard"].max_time == pytest.approx(TP.horizon(TP.oracle_time("household", tid)))
        assert specs["hard"].time_scale == 1.0
        texts = {make_task(tid, d, 4).gen_instruction() for d in ("easy", "medium", "hard")}
        assert len(texts) == 1
        text = texts.pop()
        # No time is stated (one run is scored at three budgets). Other judged limits (ml, cm) may be.
        assert "as quickly as possible" in text and "second" not in text
        assert not re.search(r"\d\s*(s|sec|min|minutes?)\b", text)


def test_c_rows_validate_and_score():
    from roboclock.evaluation.schema import validate_row
    from roboclock.evaluation.score import score_fields

    task = make_task("C3", "hard", 0)  # all six by 36 s: physical, on time at easy and medium only
    row = dict(run(task, c3_script([6, 12, 18, 24, 30, 36])), environment="household")
    assert validate_row(row) == [] and row["score_version"] == 2
    scored = score_fields("household", row)
    assert scored["score"] == 80.0 and scored["score_by_tier"] == {"easy": 100.0, "medium": 100.0, "hard": 80.0}
    partial = dict(run(make_task("C3", "medium", 0), c3_script([5, 10, 15])), environment="household")
    assert score_fields("household", partial)["score"] == 36.0  # three of six utensils sorted
    task = make_task("C2", "medium", 0)
    row = dict(run(task, c2_script(task)), environment="household")
    assert validate_row(row) == [] and score_fields("household", row)["score_by_tier"]["hard"] == 100.0
    task = make_task("C1", "medium", 3)
    row = dict(run_c1(task, c1_script(task.spec, late={1: 3.0})), environment="household")
    assert validate_row(row) == [] and score_fields("household", row)["score"] == 80.0  # served, one late
    json.dumps(row, allow_nan=False)


# ------------------------------------------------- judging independent of the arm, instructions
def sim_source(task_id):
    """<task>_x5/sim.py as text: it imports Isaac modules, so tests read it instead of importing it."""
    root = Path(__file__).resolve().parents[3] / "src/roboclock/envs/household"
    return (root / f"{task_id.lower()}_x5/sim.py").read_text()


def sim_constant(task_id, name):
    for node in ast.parse(sim_source(task_id)).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(f"{task_id}: no {name}")


def test_gripped_reads_either_arm():
    from roboclock.envs.household.x5.task_base import gripped

    assert gripped([5.0, 5.0, 0.0, 0.0], 1.0) and gripped([0.0, 0.0, 5.0, 5.0], 1.0)  # either arm's pair
    assert not gripped([5.0, 0.0, 0.0, 5.0], 1.0)  # one jaw of each arm is not a grip
    assert not gripped([0.5, 5.0, 0.0, 0.0], 1.0) and not gripped([0.0] * 4, 1.0)
    assert gripped([2.0, 3.0], 1.0) and not gripped([1.0, 3.0], 1.0)  # a single arm's pair; the threshold is strict


def test_judged_contacts_read_both_arms():
    # The 2026-11-10 sweep caught B2's object with the right arm and was never "held": the judged contact
    # read the left jaws only. Every judged jaw contact must list both arms; the oracle's own control may
    # keep reading its own hand.
    both = ("/World/Arms/Left/jawP", "/World/Arms/Left/jawN", "/World/Arms/Right/jawP", "/World/Arms/Right/jawN")
    for task_id, name in (("A1", "FILTERS"), ("A2", "JAWS"), ("B1", "FILTERS"), ("B2", "JAWS"), ("C1", "JAWS"),
                          ("C3", "JAWS")):
        assert set(both) <= set(sim_constant(task_id, name)), task_id
    assert {"/World/Arms/Left/link6", "/World/Arms/Right/link6"} <= set(sim_constant("B1", "FILTERS"))
    for task_id in ("A4", "B3", "C2"):  # a dict per arm: the rigid body's contact filter must join both
        assert sorted(sim_constant(task_id, "JAWS")) == ["Left", "Right"]
        assert re.search(r"JAWS\['(Left|Right)'\]\+JAWS\['(Left|Right)'\]", sim_source(task_id)), task_id
    # The oracle's own hand comes first where its control unpacks a pair.
    assert sim_constant("B2", "JAWS")[:2] == list(both[:2]) and sim_constant("A2", "JAWS")[:2] == list(both[2:])
    # The judged signals go through both pairs.
    assert "gripped(f, 1.)" in sim_source("B2") and "float(f.max())" in sim_source("B2")
    assert "jaws.max() < .3" in sim_source("A2")
    assert "jar_held = self.grasped['Left'] or other['SugarJar']" in sim_source("A1") and "in_cup and spoon_held" in sim_source("A1")
    # A3 is the exception: its jaw contacts serve the oracle's own grasp of the turner only; the judged signals
    # are the pancake's pose, its contact with the turner and the blade tip.


def test_a1_pinch_is_the_same_test_for_either_hand():
    from roboclock.envs.household.a1_x5.task import PINCH_DWELL, PINCH_N, PINCH_SLIP, Pinch

    dt = 0.004
    pinch = Pinch()
    held = [pinch.step((0.5, 0.5), (0.0, 0.0, 0.1), dt) for _ in range(30)]
    assert held.index(True) == int(PINCH_DWELL / dt) and all(held[held.index(True):])  # held after the dwell
    assert not pinch.step((0.5, PINCH_N / 2), (0.0, 0.0, 0.1), dt)  # one jaw lets go
    assert not any(Pinch().step((0.5, PINCH_N / 2), (0.0, 0.0, 0.1), dt) for _ in range(30))
    slipping = Pinch()  # both jaws touch, but the tool slides through the hand
    assert not any(slipping.step((0.5, 0.5), (0.0, 0.0, 2 * PINCH_SLIP * dt * k), dt) for k in range(30))


def instruction(task_id, difficulty="medium", seed=0):
    return make_task(task_id, difficulty, seed).gen_instruction()


def test_instructions_fit_the_observation_and_match_the_catalog():
    import string

    charset = set(string.printable + "–°")  # x5/env.py: gym.spaces.Text(512, charset=...)
    goals = {row["id"]: row["goal"] for row in CATALOG["tasks"]}
    for tid, cls in TASKS.items():
        assert cls.instruction == goals[tid], tid  # the catalog's goal is the task's instruction
        for difficulty in ("easy", "medium", "hard"):
            for seed in range(8):
                text = instruction(tid, difficulty, seed)
                assert len(text) <= 512 and set(text) <= charset, (tid, difficulty, seed, len(text))


def test_instructions_state_the_judged_rules():
    # An instruction states every judged rule the policy cannot see. The numbers are read from the specs
    # where they live there (or matched against the sim.py line that applies them), so the text and the
    # judging cannot drift apart silently.
    from roboclock.envs.household.a1_x5.task import mound_radius

    for difficulty in ("easy", "medium", "hard"):
        a1 = make_task("A1", difficulty, 0)
        lo, hi = a1.spec.band
        text = a1.gen_instruction()
        assert f"Add {lo:g} to {hi:g} g of sugar" in text  # the tier's band
        for g in (lo, hi):  # the mound's size at the band's edges, as sim.py draws it
            assert f"{g:g} g is {200 * mound_radius(g):.1f} cm" in text or f"{g:g} g {200 * mound_radius(g):.1f} cm" in text
        assert f"max {a1.spec.max_flow_g_s:g} g/s" in text and f"{a1.spec.spill_limit_g:g} g spilled" in text
        assert "upright and let go" in text and "clockwise turns (seen from above)" in text
        assert "whole head under the surface" in text and "released" in text and "until the jar is down" in text
    assert "tilt < 10." in sim_source("A1")  # 'upright' of jar_upright_released

    a2 = make_task("A2", "medium", 0)
    lo, hi = a2.spec.level_band
    text = a2.gen_instruction()
    assert f"{lo * 100:g} to {hi * 100:g} %" in text and "top of the highest cube" in text
    assert "first press only" in text and "let go" in text and "Every cube must end up in the glass" in text
    assert f"tilt more than {a2.spec.max_tilt_deg:g} degrees" in text

    text = instruction("A3")
    assert "underside" in text and "cannot be seen from above" in text  # the only cue of the browning
    assert "turned over" in text and "wholly inside the pan" in text
    assert "3 cm outside the rim" in text and "12 cm above the pan's floor" in text  # sim.py turner_clear
    assert "> RIM_R+.03 or blade_tip[2] > fz+.12" in sim_source("A3")

    a4 = make_task("A4", "medium", 0)
    text = a4.gen_instruction()
    assert f"{a4.spec.target_g:g} g (+/- {a4.spec.tolerance_g:g} g)" in text and "let go" in text
    assert f"{a4.spec.spill_limit_g:g} g spilled fails" in text and "not above the jug's mouth" in text
    assert f"the dose is read {a4.spec.settle_s:g} s later" in text and "standing released" in text
    lo, hi = type(a4.spec).hidden_ranges()["kettle_ml"]
    assert f"weighs {0.8 + lo / 1000:g} to {0.8 + hi / 1000:g} kg" in text  # sim.py SHELL_KG + the water
    for ml in (lo, hi):  # the pouring angle the text gives covers every seed's kettle while it is full
        full = dataclasses.replace(a4.spec, kettle_ml=ml)
        assert "40 to 60 degrees" in text and 38 <= type(a4)(full).theta0() <= 60

    b1 = make_task("B1", "medium", 0)
    text = b1.gen_instruction()
    assert b1.spec.fail_overflow_l == 0.5 and "half a litre" in text
    assert "cannot be pressed down" in text and "about 2 cm sideways toward the neck" in text
    assert "to the left, seen from the robot" in text and "stands upright" in text
    assert f"untouched for {b1.spec.closure_dwell:g} second" in text and "let go" in text

    b2 = make_task("B2", "medium", 0)
    text = b2.gen_instruction()
    assert "before its centre passes the counter's edge" in text and "both jaws of one gripper" in text
    assert "at least 1.5 cm off the counter" in text and f"for {b2.spec.hold_s:g} second" in text
    assert "clear = me['p'][2] > H+r+.015" in sim_source("B2")
    assert f"more than {b2.spec.crush_limit_n:g} N" in text and f"within {b2.spec.others_limit_m * 100:g} cm" in text

    b3 = make_task("B3", "medium", 0)
    text = b3.gen_instruction()
    assert "middle one of the five" in text and "quarter turn clockwise (seen from above)" in text
    assert f"within {b3.spec.off_angle_deg:g} degrees of that stop" in text and "let go" in text
    assert "front handle" in text

    from roboclock.envs.household.c1_x5.task import LATE_TAKE_S, RULES

    for difficulty in ("easy", "medium", "hard"):
        text = instruction("C1", difficulty, 5)
        assert text.endswith(RULES) and "board." + RULES in text
    assert RULES.startswith(" A meal") and f"stood there {LATE_TAKE_S:g} second" in RULES
    assert "upright and released on the board" in RULES and "whatever bowl is there" in RULES
    assert "tilted more than 40 degrees, even in a gripper" in RULES  # sim.py: tested before "held"
    c1 = sim_source("C1")
    assert c1.index("elif c[2] < .84 or tilt > 40:") < c1.index("elif held:")

    c2 = make_task("C2", "medium", 0)
    lo, hi = c2.spec.band_ml
    text = c2.gen_instruction()
    assert f"{lo:g} to {hi:g} ml" in text and f"the line marks {c2.spec.target_ml:g} ml" in text
    assert f"Spilling {c2.spec.spill_limit_ml:g} ml" in text and "more than 3 cm or tipping it" in text
    assert "dropping a bottle" in text and "released" in text
    assert "gtilt[j] < 20. and np.linalg.norm(self.gp[j][:2]-G['glass0'][j][:2]) < .03" in sim_source("C2")

    text = instruction("C3")
    assert "at rest, released" in text and "below the tops of its walls" in text
    assert "across a wall does not count" in text and "wrong compartment" in text and "off the counter" in text
    assert "already in the drawer must stay" in text


def test_c3_layout_is_seeded():
    from roboclock.envs.household.c3_x5.task import LOOSE, REFERENCES

    pytest.importorskip("pxr")
    from roboclock.envs.household.c3_x5.scene import layout, slots

    assert LOOSE == ("Spoon1", "Spoon2", "Fork1", "Fork2", "Knife1", "Knife2")
    assert REFERENCES == ("Spoon3", "Fork3", "Knife3")
    order0, poses0 = layout(0)
    assert order0 == ["spoon", "fork", "knife"] and layout(0)[1].keys() == poses0.keys()
    assert len(slots()) == 6 and len(poses0) == 9
    # seed 0: each arm has one utensil of each kind on its side of the drawer (x = 0.30)
    for left in (True, False):
        side = [n for n in LOOSE if (poses0[n][0][0] < .30) == left]
        assert sorted(n[:-1] for n in side) == ["Fork", "Knife", "Spoon"]
    orders = {tuple(layout(seed)[0]) for seed in range(1, 12)}
    assert len(orders) > 2
    order1, poses1 = layout(1)
    assert sorted(order1) == sorted(order0) and len(poses1) == 9
    assert any(abs(poses1[n][0][0]-poses0[n][0][0]) > .02 for n in LOOSE)
    for seed in range(1, 12):  # every loose utensil in its own table slot, within the jitter
        poses = layout(seed)[1]
        used = sorted(min(range(6), key=lambda k: abs(poses[n][0][0] - slots()[k][0])) for n in LOOSE)
        assert used == list(range(6))


def test_designs_scene_builders_and_demos_exist():
    # Initial scenes are built in code at reset (data/household/x5_initial was removed in
    # b35aedd); test_x5_env.py rebuilds them and compares with initial_scene_sha256.
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    household = root / "src/roboclock/envs/household"
    stale = []  # demos recorded on an earlier scene: C1 (R3, recording pending), C3 (six utensils since 2026-11-09)
    for row in CATALOG["tasks"]:
        design = json.loads((household / row["design"]).read_text())
        assert design["id"] == row["id"] and design["instruction"] == row["goal"]
        assert importlib.util.find_spec(row["scene"]) is not None
        assert importlib.util.find_spec(row["module"].rsplit(".", 1)[0] + ".sim") is not None
        demo = root / row["demo"]["path"]
        result = json.loads((demo / "result.json").read_text())
        assert result["task"] == row["id"] and result["success"]
        assert (result["difficulty"], result["seed"]) == (row["demo"]["difficulty"], row["demo"]["seed"])
        if result["initial_scene_sha256"] != row["initial_scene_sha256"]:
            stale.append(row["id"])
        assert (demo / "initial.png").exists()
        if (demo / f"{row['id']}.mp4").exists():  # RoboDojo rig format: overview + three policy videos
            for name in ("head.mp4", "left_wrist.mp4", "right_wrist.mp4", "frames.json"):
                assert (demo / name).exists(), (row["id"], name)
            assert result["robodojo_rig_check"]["ok"] and result["robodojo_rig"]["head_pitch_down_deg"] == 60.0
            assert result["renderer"]["rendermode"] == "RaytracedLighting"
            validation = json.loads((demo / "validation.json").read_text())
            assert validation["joint_success"] and validation["media"]["checks"]["frame_synchronized"]
        else:  # legacy single video (A2 until it moves to the RoboDojo rig)
            assert (demo / "demo.mp4").exists()
    assert not stale, f"demo recorded on another initial scene than tasks.json's: {stale}; re-record them"
