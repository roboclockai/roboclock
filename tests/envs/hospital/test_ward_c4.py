"""C4 blood-tube triage scoring, without the simulator."""

import numpy as np
import pytest

from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState
from roboclock.evaluation import throughput


def bound(difficulty="hard", seed=0):
    config = load_config()
    task = make_task("C4", difficulty, seed, config)
    task.bind(make_layout(task, config))
    return task


def snapshot(task, **changes):
    objects = {
        name: ObjectState(obj.position.copy(), obj.orientation.copy())
        for name, obj in task.layout.objects.items()
    }
    objects.update(changes)
    return PhysicalState(objects, np.array([0.4, 0.0, 1.0]), robot_moving=True)


def in_well(task, name, well, tilt=0.0):
    target = task.layout.targets[well]
    h = task.layout.shapes[name].height
    q = np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0.0, 0.0])
    axis = np.array([0.0, -np.sin(tilt), np.cos(tilt)])
    bottom = np.array([*target.position[:2], target.support_z + 0.001])
    return ObjectState(bottom + axis * h / 2, q)


def run(task, state, seconds, dt=0.01):
    for _ in range(round(seconds / dt)):
        task.time += dt
        task.observe(state, dt)


def wells_for(task, name, correct=True):
    key = task.tube_class[name]
    return [w for w, k in task.well_class.items() if (k == key) == correct]


def test_registered_ward_task_and_balanced_colors():
    task = bound()
    assert task.task_id == "C4" and task.category == "throughput"
    assert task.spec.max_time == pytest.approx(5 * throughput.oracle_time("hospital", "C4"))
    assert task.jobs_required == 12  # every tube, at every level
    counts = {k: list(task.tube_class.values()).count(k) for k in ("purple", "blue", "red")}
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(len(wells_for(task, n)) >= max(counts.values()) for n in task.tube_class)
    assert not task.terminated


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_q_is_every_tube_at_every_level(difficulty):
    task = bound(difficulty)
    assert task.jobs_required == len(task.tube_class) == 12
    assert task.spec.cycle_time is None and task.spec.required_rate_per_min is None


def test_all_tubes_racked_is_physical_and_judged_from_t_q():
    task = bound()
    run(task, snapshot(task), 0.1)
    placed = {}
    used = set()
    for name in task.tube_class:
        well = next(w for w in wells_for(task, name) if w not in used)
        used.add(well)
        placed[name] = in_well(task, name, well)
        task.moved.add(name)
        run(task, snapshot(task, **placed), 0.3)
    assert task.correct == set(task.tube_class)
    assert task.accuracy == 1.0 and task.physical_success
    t_q = max(task.job_times.values())
    assert t_q == pytest.approx(0.1 + 12 * 0.3, abs=0.12)  # the 12th tube, after its dwell
    assert task.temporal_success and task.joint_success  # t_Q far inside T_hard
    metrics = task.metrics()
    assert metrics["t_q_s"] == pytest.approx(t_q) == pytest.approx(metrics["last_correct_time"])
    assert all(metrics["temporal_success_by_tier"].values())


def test_wrong_rack_lowers_accuracy_and_is_a_misplacement_until_moved():
    task = bound()
    name = next(iter(task.tube_class))
    wrong = wells_for(task, name, correct=False)[0]
    task.moved.add(name)
    run(task, snapshot(task, **{name: in_well(task, name, wrong)}), 0.3)
    assert name in task.attempted and name not in task.correct
    assert task.misplaced() == 1
    # Moved to the right rack afterwards it earns its job; the first well stays on the record.
    right = wells_for(task, name)[0]
    run(task, snapshot(task, **{name: in_well(task, name, right)}), 0.3)
    assert name in task.correct and task.misplaced() == 0 and task.accuracy == 0.0
    assert task.metrics()["placements"][name] == {
        "well": wrong,
        "time": pytest.approx(0.2, abs=0.02),
        "correct": False,
    }


def test_needs_dwell_release_and_upright():
    task = bound()
    name = next(iter(task.tube_class))
    well = wells_for(task, name)[0]
    task.moved.add(name)
    run(task, snapshot(task, **{name: in_well(task, name, well)}), 0.1)
    assert name not in task.correct  # placement dwell not yet reached
    held = ObjectState(
        in_well(task, name, well).position, in_well(task, name, well).orientation, grasped=True
    )
    other = bound()
    other.moved.add(name)
    run(other, snapshot(other, **{name: held}), 0.5)
    assert name not in other.correct
    tilted = bound()
    tilted.moved.add(name)
    run(tilted, snapshot(tilted, **{name: in_well(tilted, name, well, tilt=0.5)}), 0.5)
    assert name not in tilted.correct


def test_one_tube_is_not_physical_success():
    task = bound()
    name = next(iter(task.tube_class))
    task.moved.add(name)
    run(task, snapshot(task, **{name: in_well(task, name, wells_for(task, name)[0])}), 0.3)
    assert task.accuracy == 1.0 and name in task.correct
    assert not task.physical_success and not task.temporal_success  # 1 of 12


def test_each_arm_owns_a_balanced_tray_and_rack_half():
    task = bound()
    info = task.layout.info["ward_c4"]
    split = info["split_y"]
    for side, sign in (("left", 1.0), ("right", -1.0)):
        tubes = [n for n, s in info["tube_side"].items() if s == side]
        assert len(tubes) == task.spec.num_items // 2
        for key in ("purple", "blue", "red"):
            count = sum(info["tube_class"][t] == key for t in tubes)
            own = [w for w in info["wells"][key] if info["well_side"][w] == side]
            assert count <= len(own)
        assert all(sign * (task.layout.objects[t].position[1] - split) > 0.15 for t in tubes)
        own_wells = [w for w, s in info["well_side"].items() if s == side]
        assert all(sign * (task.layout.targets[w].position[1] - split) > 0.05 for w in own_wells)


def test_board_shows_count_and_no_pace():
    task = bound()
    assert not any(i.style == "beacon" for i in task.layout.indicators)
    task.time = 0.5 * task.spec.max_time  # nothing racked halfway through
    names = {c.name for c in task.process(0.01) if c.kind == "show"}
    assert not any(name.startswith("c4_pace") for name in names)
