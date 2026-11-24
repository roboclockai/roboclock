from dataclasses import replace

import numpy as np
import pytest
import yaml

from roboclock.config import load_config
from roboclock.envs.hospital.config import load_legacy_config
from roboclock.envs.hospital.layout import check_config, make_layout
from roboclock.envs.hospital.tasks.registry import TASKS, make_legacy_task, make_task
from roboclock.evaluation.evaluator import episode_plan


def test_suite_size_is_fourteen_tasks_three_levels_ten_seeds():
    plan = episode_plan(environment="hospital")
    assert len(plan) == 420
    assert {item["task_id"] for item in plan} == set(TASKS) and len(TASKS) == 14
    assert {item["environment"] for item in plan} == {"hospital"}


def test_every_layout_is_valid_across_levels_and_seeds():
    config = load_config(environment="hospital")
    for task_id in TASKS:
        for difficulty in ("easy", "medium", "hard"):
            for seed in range(10):
                task = make_task(task_id, difficulty, seed, config)
                task.bind(make_layout(task, config))
    check_config(config)


@pytest.mark.parametrize("task_id", TASKS)
def test_seed_is_reproducible_and_geometry_is_constant_across_difficulty(task_id):
    config = load_config(environment="hospital")
    a, b = make_task(task_id, "easy", 42, config), make_task(task_id, "easy", 42, config)
    assert a.spec == b.spec
    easy, hard = make_task(task_id, "easy", 11, config), make_task(task_id, "hard", 11, config)
    x, y = make_layout(easy, config), make_layout(hard, config)
    assert x.objects.keys() == y.objects.keys()
    for name in x.objects:
        np.testing.assert_array_equal(x.objects[name].position, y.objects[name].position)
        assert x.objects[name].mass == y.objects[name].mass
    assert easy.spec.num_items == hard.spec.num_items
    assert make_task(task_id, "easy", 43, config).spec != a.spec


def test_temporal_pressure_tightens_with_difficulty():
    config = load_config(environment="hospital")

    def spec(task_id, level):
        return make_task(task_id, level, 5, config).spec

    for level_a, level_b in (("easy", "medium"), ("medium", "hard")):
        for task_id in ("A4", "B4", "B5", "B6"):
            assert spec(task_id, level_a).response_limit > spec(task_id, level_b).response_limit
            assert spec(task_id, level_a).event_time == spec(task_id, level_b).event_time
        for task_id in ("B1", "B2", "B3"):
            assert spec(task_id, level_a).response_limit > spec(task_id, level_b).response_limit
        # Ward C1-C4: one episode scored at every time tier, so the level changes nothing in it.
        for task_id in ("C1", "C2", "C3", "C4"):
            a, b = spec(task_id, level_a), spec(task_id, level_b)
            assert replace(a, difficulty=b.difficulty) == b


def test_ward_c_horizon_is_five_oracle_times():
    from roboclock.evaluation import throughput

    for task_id in ("C1", "C2", "C3", "C4"):
        spec = make_task(task_id, "hard", 3).spec
        assert spec.cycle_time is None
        assert spec.max_time == pytest.approx(5 * throughput.oracle_time("hospital", task_id))


def test_sheet_quantities_are_present_for_each_task():
    config = load_config(environment="hospital")
    for task_id in ("A4", "B4", "B5", "B6"):
        task = make_task(task_id, seed=1, config=config)
        assert task.spec.num_items == 1
        if task_id in ("B4", "B5", "B6"):  # the circuit comes off / the patient reacts / the wound bleeds at a seeded time
            assert task.spec.event_time > 0
        else:  # the instruction is in the first observation, so it is issued at frame 0
            assert task.spec.event_time == 0
        assert task.spec.response_limit > 0
        assert task.spec.windows == () and task.spec.expiries == ()
        assert task.spec.transit_limit is None and task.spec.expiry_time is None
        assert task.category == ("temporal_window" if task_id == "A4" else "event_reactive")
    for task_id in ("B1", "B2", "B3"):
        spec = make_task(task_id, seed=1, config=config).spec
        assert spec.event_time > 0 and spec.response_limit > 0
    assert make_task("C1", seed=1, config=config).spec.required_accuracy == 0.95  # ward C1
    assert make_task("C2", seed=1, config=config).spec.required_accuracy == 0.95  # ward C2


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("C1", [40, 44], "inclusive range"),
    ],
)
def test_impossible_throughput_configuration_is_rejected(tmp_path, key, value, match):
    config = load_config(environment="hospital")
    config["tasks"][key]["num_items_range"] = value
    path = tmp_path / "impossible.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        load_config(path, environment="hospital")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"easy": {"cycle_time": [30.0, 33.0]}}, "no per-level quantities"),
        ({"num_items_range": [10, 12]}, "fixed num_items_range"),
    ],
)
def test_ward_c_budgets_are_not_configurable(tmp_path, change, match):
    config = load_config(environment="hospital")
    config["tasks"]["C3"].update(change)
    path = tmp_path / "ward_c.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=match):
        load_config(path, environment="hospital")


def test_legacy_c_budget_beyond_episode_is_still_rejected(tmp_path):
    config = load_legacy_config()
    config["tasks"]["C3"]["easy"]["cycle_time"] = [200.0, 230.0]
    path = tmp_path / "long.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="budget exceeds"):
        load_legacy_config(path)


def test_fire_scene_keeps_the_blanket_locked_and_everything_within_reach():
    config = load_legacy_config()  # the fire blanket is the legacy profile's B3 since 2026-11-19
    for seed in range(10):
        layout = make_layout(make_legacy_task("B3", "easy", seed, config), config)
        assert layout.held == {"blanket"}
        assert set(layout.objects) == {"tab_0", "tab_1", "blanket"}
        fire = layout.info["fire"]
        assert 0.25 < float(np.linalg.norm(fire["center"])) < 0.74
        for name in layout.objects:
            assert float(np.linalg.norm(layout.objects[name].position[:2])) < 0.74, name
        # Pulled tabs must stay on the bench, with a margin from its edge.
        center = np.array(config["scene"]["table_center"][:2])
        half = np.array(config["scene"]["table_size"][:2]) / 2
        for tab in ("tab_0", "tab_1"):
            end = layout.objects[tab].position[:2] + np.array(fire["tab_pull"])
            assert np.all(np.abs(end - center) < half - 0.05), (seed, tab, end)
        # The tabs and the blanket do not overlap each other or the dish.
        names = list(layout.objects)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                gap = np.linalg.norm(
                    layout.objects[names[a]].position[:2] - layout.objects[names[b]].position[:2]
                )
                assert gap > 0.06, (names[a], names[b], gap)


def test_pharmacy_orders_match_shelf_stock_exactly():
    config = load_config(environment="hospital")
    for seed in range(10):
        task = make_task("C1", "easy", seed, config)
        layout = make_layout(task, config)
        stock, demand = {}, {}
        for kind in layout.item_types.values():
            stock[kind] = stock.get(kind, 0) + 1
        for order in layout.orders.values():
            for kind, count in order.items():
                demand[kind] = demand.get(kind, 0) + count
        # Ward C1 keeps one spare vial per arm's stock riser: stock covers every order, plus two.
        assert all(stock.get(kind, 0) >= count for kind, count in demand.items())
        assert sum(stock.values()) == sum(demand.values()) + 2


def test_aed_pads_are_held_until_release_and_start_on_the_unit():
    config = load_config(environment="hospital")
    task = make_task("B2", "easy", 0, config)
    layout = make_layout(task, config)
    assert layout.held == {"pad_0", "pad_1"}
    aed = next(f for f in layout.fixtures if f.name == "aed")
    for name in ("pad_0", "pad_1", "tab_0"):
        position = layout.objects[name].position
        assert abs(position[0] - aed.position[0]) < aed.size[0] / 2
        assert abs(position[1] - aed.position[1]) < aed.size[1] / 2
        assert position[2] > aed.position[2] + aed.size[2] / 2  # resting on top of it


def test_aed_marks_and_button_are_within_the_robots_reach():
    from roboclock.envs.lab.robot import mount_positions

    config = load_config(environment="hospital")
    base = np.array(mount_positions(config, "B2")[0][:2])
    for seed in range(5):
        layout = make_layout(make_task("B2", "easy", seed, config), config)
        for zone in (layout.zones["mark_0"], layout.zones["mark_1"]):
            # Chest height makes the reach shorter than the flat distance suggests: the second pad stalled at 0.78 m.
            assert 0.2 < float(np.linalg.norm(zone.center[:2] - base)) < 0.72
        assert 0.2 < float(np.linalg.norm(layout.info["aed"]["button"][:2] - base)) < 0.72


def test_a2_drawer_and_bag_never_overlap_the_fridge_through_the_whole_pull():
    """Sweep the drawer (with the bag riding in it) from closed to fully open."""
    import numpy as np

    from roboclock.envs.hospital.layout import load_catalog, make_layout

    def box(center, size):
        center, size = np.asarray(center, float), np.asarray(size, float)
        return center - size / 2, center + size / 2

    def overlap(a, b):
        return bool(np.all(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]) > 1e-6))

    config = load_legacy_config()
    spec = load_catalog()["blood_bag"]
    bx, by, bz = spec["size"]
    eye = spec["eyelet"]
    for seed in range(3):
        layout = make_layout(make_legacy_task("A2", "easy", seed, config), config)
        drawer, bag = layout.drawers[0], layout.objects["bag_0"].position
        fixed = [(f.name, box(f.position, f.size)) for f in layout.fixtures if f.collision]
        fixed = [f for f in fixed if f[0].startswith("fridge")]
        for shift in np.linspace(0.0, drawer.travel, 19):
            move = np.array([0.0, shift, 0.0])
            moving = [
                (p.name, box(drawer.position + move + p.offset, p.size))
                for p in drawer.parts
                if p.collision
            ]
            moving += [
                ("bag_body", box(bag + move, [bx, by, bz])),
                (
                    "eyelet_top",
                    box(
                        bag + move + [0, 0, bz / 2 + eye["height"] - eye["bar"] / 2],
                        [eye["width"], eye["thickness"], eye["bar"]],
                    ),
                ),
                (
                    "eyelet_l",
                    box(
                        bag
                        + move
                        + [-(eye["width"] / 2 - eye["bar"] / 2), 0, bz / 2 + eye["height"] / 2],
                        [eye["bar"], eye["thickness"], eye["height"]],
                    ),
                ),
            ]
            for name, a in moving:
                for fname, b in fixed:
                    assert not overlap(a, b), (seed, round(float(shift), 3), name, fname)
            for i, (n1, a) in enumerate(moving[: len(moving) - 3]):
                for n2, b in moving[len(moving) - 3 :]:
                    assert not overlap(a, b), (seed, "drawer part touches the bag", n1, n2)


def test_a2_loop_grasp_is_within_the_simulators_grasp_distance():
    """The grasp detector needs the object center within grasp_height / 2 + 2.5 cm of the fingers."""
    config = load_legacy_config()
    for seed in range(3):
        layout = make_layout(make_legacy_task("A2", "easy", seed, config), config)
        _, grasp_height = layout.dimensions["bag_0"]
        assert grasp_height / 2 + 0.025 > layout.info["bag_grasp_dz"] + 0.01


def test_a1_last_request_is_stat_with_a_shorter_window_that_arrives_between_others():
    config = load_legacy_config()
    for level in ("easy", "medium", "hard"):
        for seed in range(10):
            spec = make_legacy_task("A1", level, seed, config).spec
            if spec.num_items < 3:
                continue
            assert spec.stat == spec.num_items - 1
            previous, urgent = spec.windows[-2], spec.windows[-1]
            widths = [end - start for start, end in spec.windows]
            assert widths[-1] == pytest.approx(0.7 * widths[0])
            assert previous[0] < urgent[0] < previous[1]  # it arrives while the previous is open
            assert urgent[1] < previous[1]  # and it has the earlier deadline


# Measured on the oracle: a delivery takes ~4.5 s when idle over the vial, ~7.5 s back to back.
IDLE_DELIVERY, BACK_TO_BACK_DELIVERY = 4.5, 7.5


def _serve(windows, policy):
    """Worst slack when requests are served one at a time in the order chosen by `policy`."""
    time, todo, worst = 0.0, set(range(len(windows))), 1e9
    while todo:
        arrived = [i for i in todo if windows[i][0] <= time]
        if arrived:
            key = (lambda k: windows[k][1]) if policy == "deadline" else (lambda k: windows[k][0])
            i = min(arrived, key=key)
            time += BACK_TO_BACK_DELIVERY
        else:
            i = min(todo, key=lambda k: windows[k][0])
            time = windows[i][0] + IDLE_DELIVERY
        worst = min(worst, windows[i][1] - time)
        todo.discard(i)
    return worst


def test_a1_medium_and_hard_reward_earliest_deadline_first():
    """With a queue, serving in arrival order starts missing the STAT request; deadline order does not."""
    config = load_legacy_config()
    for level, minimum_fifo_failures in (("easy", 0), ("medium", 10), ("hard", 20)):
        deadline, arrival = [], []
        for seed in range(60):
            windows = make_legacy_task("A1", level, seed, config).spec.windows
            deadline.append(_serve(windows, "deadline"))
            arrival.append(_serve(windows, "arrival"))
        assert all(slack >= 0 for slack in deadline), level
        assert sum(slack < 0 for slack in arrival) >= minimum_fifo_failures, level


def test_aed_tab_lands_clear_of_the_unit_and_on_the_bench_for_every_seed():
    """The pulled tab must leave the AED (or it cannot be set down) without leaving the bench (or it falls)."""
    config = load_config(environment="hospital")
    center = np.array(config["scene"]["table_center"][:2])
    half = np.array(config["scene"]["table_size"][:2]) / 2
    for seed in range(40):
        layout = make_layout(make_task("B2", "easy", seed, config), config)
        aed = next(f for f in layout.fixtures if f.name == "aed")
        end = layout.objects["tab_0"].position[:2] + np.array(layout.info["aed"]["tab_pull"])
        assert np.all(np.abs(end - center) < half - 0.012), (seed, end)  # on the bench
        outside = np.any(np.abs(end - aed.position[:2]) > aed.size[:2] / 2 + 0.025)
        assert outside, (seed, end)  # fully off the unit, so it rests on the bench


def test_a3_calls_a_situation_that_maps_to_the_right_instrument_and_varies_by_seed():
    from roboclock.envs.hospital.layout import SITUATIONS

    config = load_legacy_config()
    seen = set()
    for seed in range(40):
        layout = make_layout(make_legacy_task("A3", "easy", seed, config), config)
        situations = layout.info["situations"]
        assert len(situations) == len(layout.steps) == 3
        for tool, situation in zip(layout.steps, situations):
            assert situation in SITUATIONS[tool]  # the call really means that instrument
            assert tool.upper() not in situation  # and it never names it
        seen.add(tuple(situations))
    assert len(seen) > 20  # different situations in different episodes


def test_aed_tab_does_not_overlap_the_pads_the_gripper_straddles():
    """The fingers open 4 cm either side of the tab: they must not land on a pad (the pads then block the grasp)."""
    config = load_config(environment="hospital")
    layout = make_layout(make_task("B2", "easy", 0, config), config)
    tab = layout.objects["tab_0"].position
    opening = 0.04 + 0.004  # finger half-opening plus the finger thickness
    for name in ("pad_0", "pad_1"):
        pad = layout.objects[name].position
        half = layout.shapes[name].half
        x_overlap = abs(tab[0] - pad[0]) < layout.shapes["tab_0"].half[0] + half[0] - 0.01
        y_inside = (
            abs(tab[1] - pad[1]) - half[1] < opening
        )  # the finger would come down on this pad
        assert not (x_overlap and y_inside), name


def test_blood_request_and_pharmacy_requests_are_random_per_seed():
    config = load_legacy_config()
    meds, beds = set(), set()
    for seed in range(20):
        a1 = make_layout(make_legacy_task("A1", "easy", seed, config), config)
        a2 = make_layout(make_legacy_task("A2", "easy", seed, config), config)
        assert len({m for m in a1.info["requests"]}) == len(
            a1.info["requests"]
        )  # no drug twice in one episode
        meds.add(tuple(a1.info["requests"]))
        beds.add((a2.info["request"]["bed"], a2.info["request"]["unit"]))
    assert len(meds) > 10 and len(beds) > 5


@pytest.mark.parametrize(
    "task_id,limits",
    [
        ("B4", (20.0, 8.87, 3.93)),  # ventilator reconnection since 2026-11-19
        ("B5", (33.7, 17.05, 8.63)),  # anaphylaxis auto-injector since 2026-11-20
        ("A4", (26.64, 14.9, 8.33)),  # easy = 4 t since 2026-11-23
        ("B6", (24.0, 11.43, 5.44)),  # haemorrhage tourniquet since 2026-11-20
    ],
)
def test_bedside_deadlines_and_single_operation_across_geometry_matrix(task_id, limits):
    """All 120 new bedside layouts preserve one focused response and prepared access."""
    config = load_config(environment="hospital")
    for level, limit in zip(("easy", "medium", "hard"), limits):
        for seed in range(10):
            task = make_task(task_id, level, seed, config)
            layout = make_layout(task, config)
            assert task.spec.response_limit == pytest.approx(limit)
            assert task.spec.num_items == 1 and task.category == ("temporal_window" if task_id == "A4" else "event_reactive")
            assert layout.info["bedside_task"]
            initial = np.asarray(layout.info["initial_ee_position"])
            assert initial.shape == (3,) and np.isfinite(initial).all()
            assert initial[2] > layout.table_height
            assert layout.objects and layout.indicators
            assert all(np.isfinite(obj.position).all() for obj in layout.objects.values())
            assert "fridge" not in layout.info and "dose_prep" not in layout.info


def test_bedside_tasks_are_a4_and_b4_to_b6_and_history_keeps_a1_to_a4():
    from roboclock.envs.hospital.tasks.registry import LEGACY_TASKS, TASKS

    # The bedside tasks (2026-11-19 numbering): the infusion start counts with the A class.
    bedside = {"A4": "temporal_window", "B4": "event_reactive", "B5": "event_reactive",
               "B6": "event_reactive"}
    for task_id, category in bedside.items():
        task = make_task(task_id)
        assert task.task_id == task_id and task.category == category
        assert make_task(task.name).task_id == task_id
    assert list(TASKS) == ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6",
                           "C1", "C2", "C3", "C4"]
    for old_id in ("A1", "A2", "A3", "A4"):
        assert make_legacy_task(LEGACY_TASKS[old_id].name).task_id == old_id
        # v2 reuses A1-A4 for its own A-class tasks, never the historical ones
        assert make_task(old_id).category == "temporal_window"
        assert type(make_task(old_id)) is not LEGACY_TASKS[old_id]


@pytest.mark.parametrize("task_id", ("A1", "A2", "A3", "A4"))
def test_legacy_config_accepts_historical_names_and_ids(task_id):
    from roboclock.envs.hospital.tasks.registry import LEGACY_TASKS

    config = load_legacy_config()
    historical = LEGACY_TASKS[task_id]
    named = make_task(historical.name, "medium", 5, config)
    identified = make_task(task_id, "medium", 5, config)
    assert type(named) is type(identified) is historical
    assert named.spec == identified.spec
