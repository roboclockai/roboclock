"""C2 instrument tray sort: scoring rule and placement accounting, without the simulator."""

import math

import numpy as np
import pytest

from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.mdp.geometry import object_extents
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.hospital.tasks.ward.instrument_tray_sort import InstrumentTraySort
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState
from roboclock.evaluation import throughput

DT = 0.01


@pytest.fixture
def task():
    config = load_config()
    task = make_task("C2", "hard", 0, config)
    assert isinstance(task, InstrumentTraySort)
    task.bind(make_layout(task, config))
    return task


def _state(task, **changes):
    objects = {
        name: ObjectState(obj.position.copy(), obj.orientation.copy())
        for name, obj in task.layout.objects.items()
    }
    objects.update(changes)
    return PhysicalState(objects, np.array([0.4, 0.0, 1.0]), robot_moving=True)


def _in_slot(task, name, slot, grasped=False):
    target = task.layout.targets[slot]
    q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])  # long axis along y
    extent = object_extents(task.layout.shapes[name], q)
    position = np.array([*target.position[:2], target.support_z + extent[2] + 0.0005])
    return ObjectState(position, q, grasped=grasped, fixture_contacts=frozenset([slot]))


def _run(task, seconds, **changes):
    for _ in range(round(seconds / DT)):
        task.advance(DT)
        task.observe(_state(task, **changes), DT)


def _slot_for(task, name, right=True):
    info = task.layout.info["c2"]
    kind = info["item_types"][name]
    return next(s for s, k in info["slot_types"].items() if (k == kind) == right)


def test_layout_two_arms_two_trays(task):
    info = task.layout.info["c2"]
    assert sorted(info["slot_types"].values()) == sorted(["scalpel", "forceps", "hemostat"] * 2)
    assert [t["case"] for t in info["trays"]] == [1, 2]
    assert len(task.layout.objects) == task.spec.num_items == 6
    for name in task.layout.objects:
        assert task.in_basin(task.layout.objects[name])
    assert task.spec.max_time == pytest.approx(5 * throughput.oracle_time("hospital", "C2"))
    assert task.jobs_required == 6
    # Each arm owns one basin row and one tray, with exactly one instrument of each type per slot.
    arms = info["arms"]
    assert sorted(arms["0"]["items"] + arms["1"]["items"]) == sorted(task.layout.objects)
    for side in ("0", "1"):
        kinds = sorted(info["item_types"][n] for n in arms[side]["items"])
        assert kinds == sorted(info["slot_types"][s] for s in arms[side]["slots"])
        for name in arms[side]["items"]:
            assert task.layout.assignments[name] in arms[side]["slots"]
    # Zones do not overlap: the left arm works at -y, the right arm at +y.
    for name in arms["0"]["items"]:
        assert task.layout.objects[name].position[1] < info["basin"]["center"][1]
    for name in arms["1"]["items"]:
        assert task.layout.objects[name].position[1] > info["basin"]["center"][1]
    assert all(task.layout.targets[s].position[1] < 0 for s in arms["0"]["slots"])
    assert all(task.layout.targets[s].position[1] > 0 for s in arms["1"]["slots"])


def test_no_per_level_budget_or_rate():
    config = load_config()
    specs = [make_task("C2", level, 0, config).spec for level in ("easy", "medium", "hard")]
    assert len({(s.max_time, s.num_items, s.orientation_tolerance) for s in specs}) == 1
    assert all(s.cycle_time is None and s.required_rate_per_min is None for s in specs)


def test_correct_and_wrong_placements(task):
    types = task.layout.info["c2"]["item_types"]
    # b is a narrow instrument so it physically fits a (wider) hemostat slot of the wrong type.
    b = next(n for n in task.layout.objects if types[n] != "hemostat")
    a = next(n for n in task.layout.objects if n != b)
    lifted = ObjectState(task.layout.objects[a].position + [0, 0, 0.1], np.array([1.0, 0, 0, 0]), grasped=True)
    _run(task, 0.1, **{a: lifted})
    _run(task, 0.3, **{a: _in_slot(task, a, _slot_for(task, a))})
    assert task.correct == {a}
    _run(task, 0.1, **{a: _in_slot(task, a, _slot_for(task, a)), b: lifted})
    hemostat_slot = next(
        s
        for s, k in task.layout.info["c2"]["slot_types"].items()
        if k == "hemostat" and s != _slot_for(task, a)
    )
    wrong = _in_slot(task, b, hemostat_slot)
    _run(task, 0.3, **{a: _in_slot(task, a, _slot_for(task, a)), b: wrong})
    assert task.correct == {a} and len(task.wrong) == 1
    # Fixing the wrong placement still leaves the wrong placement on the record.
    _run(task, 0.1, **{a: _in_slot(task, a, _slot_for(task, a)), b: lifted})
    _run(task, 0.3, **{a: _in_slot(task, a, _slot_for(task, a)), b: _in_slot(task, b, _slot_for(task, b))})
    assert task.correct == {a, b} and task.placements == 3
    metrics = task.metrics()
    assert math.isclose(metrics["accuracy"], 2 / 3)
    assert not task.temporal_success and not task.physical_success  # 2 of 6
    assert metrics["jobs_done"] == 2 and metrics["t_q_s"] is None


def test_nudged_item_in_basin_is_not_a_placement(task):
    name = next(iter(task.layout.objects))
    obj = task.layout.objects[name]
    nudged = ObjectState(obj.position + [0.0, 0.05, 0.0], obj.orientation.copy())
    _run(task, 0.5, **{name: nudged})
    assert name in task.moved and task.placements == 0


def test_release_on_bench_counts_as_stray(task):
    name = next(iter(task.layout.objects))
    bench = ObjectState(np.array([0.70, 0.40, task.layout.table_height + 0.007]), np.array([1.0, 0, 0, 0]))
    _run(task, 0.5, **{name: bench})
    assert [s[0] for s in task.strays] == [name] and task.placements == 1


def test_all_six_sorted_pass_and_readout(task):
    names = list(task.layout.objects)
    used, placed = set(), {}
    for name in names:
        slot = next(
            s
            for s, k in task.layout.info["c2"]["slot_types"].items()
            if k == task.layout.info["c2"]["item_types"][name] and s not in used
        )
        used.add(slot)
        placed[name] = _in_slot(task, name, slot)
    _run(task, 0.3, **placed)
    assert len(task.correct) == 6 and task.placements == 6
    assert task.temporal_success and task.physical_success  # t_Q = 0.2 s <= T_hard
    shows = {c.name: c.value for c in task.process(DT) if c.kind == "show"}
    assert shows == {} or shows.get("c2_count_6") is True
    assert not any(name.startswith("c2_pace") for name in shows)  # no pace: no rate to keep
    _run(task, throughput.SETTLE_S, **placed)
    assert task.terminated and task.metrics()["termination_reason"] == "all_done"
