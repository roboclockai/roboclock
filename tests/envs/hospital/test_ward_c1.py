"""C1 pharmacy order bins: scoring logic only (no simulator)."""

from dataclasses import replace

import numpy as np
import pytest
from hospital_helpers import above_container, in_container, observe_for, snapshot

from roboclock.envs.hospital.bedside import ward_c1
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.hospital.tasks.ward.pharmacy_order_bins import COUNT, ERRORS, PharmacyOrderBins
from roboclock.envs.lab.mdp.state import ObjectState
from roboclock.evaluation import throughput


def c1(seed=0, difficulty="hard", **spec):
    config = load_config()
    task = make_task("C1", difficulty, seed, config)
    task.spec = replace(task.spec, **spec)
    task.bind(make_layout(task, config))
    return task


def plan(task):
    """(vial, bin) pairs that exactly fill every order, plus one vial no order can use any more."""
    needs = {(b, k): n for b, order in task.layout.orders.items() for k, n in order.items()}
    pairs, used = [], set()
    for (bin_name, kind), count in needs.items():
        for _ in range(count):
            vial = next(
                v for v, t in task.layout.item_types.items() if t == kind and v not in used
            )
            used.add(vial)
            pairs.append((vial, bin_name))
    spare = next(v for v in task.layout.objects if v not in used)
    return pairs, spare


def place(task, vial, bin_name, at):
    """Lower one vial into a bin at time `at`; vials placed earlier stay where they are."""
    task.time = at
    kept = {v: task.previous.objects[v] for v in task.attempted} if task.previous else {}
    task.observe(snapshot(task, **kept, **{vial: above_container(task, vial, bin_name)}), 0.01)
    final = snapshot(task, **kept, **{vial: in_container(task, vial, bin_name)})
    task.observe(final, 0.01)
    observe_for(task, final, 0.25)


def settle(task, seconds=throughput.SETTLE_S + 0.05):
    """Hold the current scene still for `seconds` (vials placed so far stay where they are)."""
    observe_for(task, snapshot(task, **{v: task.previous.objects[v] for v in task.attempted}), seconds)


def test_layout_orders_and_registry():
    task = c1()
    assert isinstance(task, PharmacyOrderBins) and task.task_id == "C1"
    assert task.bedside_module == "ward_c1"
    assert task.layout.info["bedside_task"] == "ward_c1"
    assert sorted(task.layout.targets) == ["bin_0", "bin_1", "bin_2", "bin_3"]
    ordered = sum(n for order in task.layout.orders.values() for n in order.values())
    assert ordered == task.spec.num_items == 8
    assert len(task.layout.objects) == task.spec.num_items + 2  # one spare vial per stock riser
    # Q = every ordered vial; the episode runs to H = 5 t_o whatever the level.
    assert task.jobs_required == 8
    assert task.spec.max_time == pytest.approx(5 * throughput.oracle_time("hospital", "C1"))


def test_seed_keeps_geometry_across_levels_and_differs_across_seeds():
    easy, hard = c1(3, "easy"), c1(3, "hard")
    assert easy.layout.orders == hard.layout.orders
    for name, obj in easy.layout.objects.items():
        np.testing.assert_allclose(obj.position, hard.layout.objects[name].position)
    assert easy.spec.max_time == hard.spec.max_time
    assert any(c1(s).layout.orders != easy.layout.orders for s in range(4))


def test_all_eight_orders_filled_is_physical_and_t_q_is_the_eighth_credit():
    task = c1()
    pairs, _ = plan(task)
    for k, (vial, bin_name) in enumerate(pairs[:7]):
        place(task, vial, bin_name, at=1.0 + 2 * k)
    assert len(task.correct) == 7 and not task.physical_success  # Q is all 8 orders
    place(task, *pairs[7], at=20.0)
    assert task.goal_achieved and task.physical_success and task.accuracy == 1.0
    settle(task)
    assert task.termination_reason == "all_done"
    metrics = task.metrics()
    t_o = throughput.oracle_time("hospital", "C1")
    assert metrics["t_q_s"] == pytest.approx(20.2, abs=0.02)  # credited after the 0.2 s dwell
    assert t_o < metrics["t_q_s"] < 2 * t_o
    assert metrics["temporal_success_by_tier"] == {"easy": True, "medium": True, "hard": False}
    assert metrics["task_success"] and not metrics["joint_success"]  # hard
    assert metrics["orders_fulfilled"] == metrics["orders_total"] == 4
    assert sorted(t for t in metrics["job_times_s"]) == sorted(task.job_times.values())


def test_a_spare_vial_left_in_a_bin_fails_physical():
    task = c1(difficulty="easy")
    pairs, spare = plan(task)
    for k, (vial, bin_name) in enumerate(pairs):
        place(task, vial, bin_name, at=1.0 + 2 * k)
    assert task.goal_achieved
    # Every order is already full: one more vial is an over-fill, i.e. a misplacement.
    place(task, spare, "bin_0", at=17.0)
    assert spare not in task.correct and task.placements == len(pairs) + 1
    assert task.misplaced() == 1 and task.allowed_misplacements == 0
    assert not task.physical_success
    metrics = task.metrics()
    assert metrics["accuracy"] == pytest.approx(len(pairs) / (len(pairs) + 1))
    assert metrics["temporal_success"] and not metrics["task_success"]  # time is not the problem
    # Taken out again and left on the counter: nothing is misplaced at the end.
    start = task.layout.objects[spare]
    aside = ObjectState(start.position + np.array([0.0, 0.10, 0.0]), start.orientation.copy())
    kept = {v: task.previous.objects[v] for v in task.correct}
    observe_for(task, snapshot(task, **kept, **{spare: aside}), throughput.SETTLE_S + 0.1)
    assert task.misplaced() == 0 and task.physical_success
    assert task.termination_reason == "all_done"


def test_vial_of_a_drug_the_bin_does_not_need_is_wrong():
    task = c1()
    bin_name, order = next(iter(task.layout.orders.items()))
    vial = next(v for v, kind in task.layout.item_types.items() if kind not in order)
    place(task, vial, bin_name, at=3.0)
    assert vial in task.attempted and vial not in task.correct
    assert task.metrics()["wrong_placements"] == 1


def test_vials_count_until_the_end_of_the_episode():
    """No shift cut-off: credit continues past the old easy shift (33 s) while work goes on."""
    task = c1()
    pairs, _ = plan(task)
    for k, pair in enumerate(pairs[:6]):
        place(task, *pair, at=5.0 + 7.0 * k)  # 7 s apart: within the 0.5 t_o stall window
    assert len(task.correct) == 6 and max(task.job_times.values()) > 40.0
    assert not task.terminated


def test_released_vial_left_off_target_counts_as_an_attempt():
    task = c1()
    vial = next(iter(task.layout.objects))
    start = task.layout.objects[vial]
    moved = ObjectState(start.position + np.array([0.0, 0.12, 0.0]), start.orientation.copy())
    observe_for(task, snapshot(task, **{vial: moved}), duration=0.3)
    assert task.attempted == {vial} and not task.correct
    assert task.accuracy == 0.0


def test_board_readouts_follow_the_counts():
    task = c1()
    commands = {(c.kind, c.name): c.value for c in task.advance(0.01)}
    assert commands[("timer", COUNT)][0] == 0 and commands[("timer", ERRORS)][0] == 0
    fraction, _ = commands[("timer", "c1_shift_bar")]
    assert fraction == pytest.approx(1 - 0.01 / task.spec.max_time, abs=0.01)
    pairs, _ = plan(task)
    place(task, *pairs[0], at=2.0)
    commands = {(c.kind, c.name): c.value for c in task.advance(0.01)}
    assert commands[("timer", COUNT)][0] == 1
    assert ("timer", ERRORS) not in commands  # unchanged readouts are not resent


def test_config_has_no_per_level_budget():
    block = load_config()["tasks"]["C1"]
    assert not any(level in block for level in ("easy", "medium", "hard"))
    assert block["num_items_range"] == [8, 8] and block["required_accuracy"] == 0.95


def test_each_arm_fills_its_own_bins_from_its_own_stock():
    for seed in range(6):
        task = c1(seed)
        layout, sides = task.layout, task.layout.info["c1"]["sides"]
        assert set(sides["0"]["items"]).isdisjoint(sides["1"]["items"])
        assert set(sides["0"]["bins"]).isdisjoint(sides["1"]["bins"])
        for side, own in sides.items():
            # The halves are split at robot y = 0: side 0 at -y, side 1 at +y (the rig).
            for name in own["items"]:
                assert ward_c1.side_of(layout.objects[name].position[1]) == int(side)
            for name in own["bins"]:
                assert ward_c1.side_of(layout.targets[name].position[1]) == int(side)
            stock = {}
            for name in own["items"]:
                stock[layout.item_types[name]] = stock.get(layout.item_types[name], 0) + 1
            for bin_name in own["bins"]:
                for kind, count in layout.orders[bin_name].items():
                    stock[kind] = stock.get(kind, 0) - count
            assert min(stock.values()) >= 0  # this arm's stock covers this arm's orders


def test_oracle_choice_never_crosses_into_the_other_arms_half():
    task = c1(2)
    state = snapshot(task)
    sides = task.layout.info["c1"]["sides"]
    for side in (0, 1):
        memory = {"pending": {}, "skip": set()}
        pick = ward_c1.choose(task, state, memory, side, np.array([0.3, 0.0, 1.0]))
        assert pick is not None
        vial, bin_name = pick
        assert vial in sides[str(side)]["items"] and bin_name in sides[str(side)]["bins"]
        assert task.layout.item_types[vial] in task.layout.orders[bin_name]


def test_two_arm_mounts_and_counter_pose():
    from roboclock.envs.lab.robot import mount_positions

    task = c1()
    left, right = mount_positions(load_config(), "C1")
    assert left[1] < 0 < right[1]  # side 0 at -y, side 1 at +y: the common rig in front of the counter
    assert right[1] - left[1] == pytest.approx(0.60) and left[0] == right[0] and left[2] == right[2]
    info = task.layout.info
    assert info["hide_bench"] and info["ward_pose"]["usd"].endswith("ward_02_preview_left.usdc")
    assert "iMac_5k_Monitor054" in info["ward_pose"]["hide"]  # the ultrasound cart is hidden
    surface = info["work_surface"]
    top = surface["center"][2] + surface["size"][2] / 2
    assert top == pytest.approx(task.layout.table_height)


def test_stage_milestones_lift_then_container_entry():
    """The stage score's measured steps: a vial lifted off its start, then lowered into a bin."""
    task = c1()
    (vial, bin_name), *_ = plan(task)[0]
    start = task.layout.objects[vial]
    task.time = 1.0
    task.observe(snapshot(task), 0.01)
    nudged = ObjectState(start.position + np.array([0.06, 0.0, 0.02]), start.orientation.copy())
    task.observe(snapshot(task, **{vial: nudged}), 0.01)  # pushed, not lifted
    assert task.metrics()["first_lift_time"] is None
    task.time = 2.0
    lifted = ObjectState(start.position + np.array([0.0, 0.0, 0.05]), start.orientation.copy())
    task.observe(snapshot(task, **{vial: lifted}), 0.01)
    assert task.metrics()["first_lift_time"] == pytest.approx(2.0)
    assert task.metrics()["first_entry_time"] is None
    place(task, vial, bin_name, at=3.0)
    assert task.metrics()["first_entry_time"] == pytest.approx(3.0)
