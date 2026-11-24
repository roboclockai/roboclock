"""C3 crash-cart vial restock: scoring rules on synthetic states (no simulator)."""

from dataclasses import replace

import numpy as np
import pytest

from roboclock.envs.hospital.config import load_config, sample_episode
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.hospital.tasks.ward.crash_cart_vials import CrashCartVials
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState
from roboclock.evaluation import throughput

DT = 1 / 120


@pytest.fixture
def task():
    config = load_config()
    task = CrashCartVials(sample_episode("C3", "hard", 0, config))
    task.bind(make_layout(task, config))
    return task


def resting(task, overrides=None):
    """Every vial at its supply place, plus any (name -> ObjectState) overrides."""
    objects = {
        name: ObjectState(obj.position.copy(), obj.orientation.copy())
        for name, obj in task.layout.objects.items()
    }
    objects.update(overrides or {})
    return PhysicalState(objects, ee_position=np.array([0.4, 0.3, 1.0]))


def in_pocket(task, name, pocket, *, tilt=0.0):
    target = task.layout.targets[pocket]
    height = task.layout.shapes[name].height
    position = np.array([*target.position[:2], target.support_z + height / 2])
    orientation = np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0.0, 0.0])
    return ObjectState(position, orientation, fixture_contacts=frozenset([pocket]))


def run(task, state, seconds):
    for _ in range(round(seconds / DT)):
        task.advance(DT)
        task.observe(state, DT)


def pocket_for(task, name, *, matching=True):
    info = task.layout.info["c3"]
    drug = info["item_drug"][name]
    return next(p for p, d in info["pocket_drug"].items() if (d == drug) == matching)


def test_registry_builds_ward_task_with_one_horizon():
    config = load_config()
    for level in ("easy", "medium", "hard"):
        task = make_task("C3", level, 1, config)
        assert isinstance(task, CrashCartVials) and task.task_id == "C3"
        assert task.spec.max_time == pytest.approx(5 * throughput.oracle_time("hospital", "C3"))
        assert task.jobs_required == task.spec.num_items == 12
        assert task.spec.orientation_tolerance == config["difficulty"]["medium"]["orientation_tolerance"]
        layout = make_layout(task, config)
        drugs = list(layout.info["c3"]["item_drug"].values())
        pockets = list(layout.info["c3"]["pocket_drug"].values())
        assert sorted(drugs) == sorted(pockets)  # exactly one pocket per vial, per drug


def test_matching_pocket_counts_after_dwell(task):
    name = "vial_0"
    pocket = pocket_for(task, name)
    moved = in_pocket(task, name, pocket)
    run(task, resting(task, {name: moved}), 0.1)
    assert name not in task.correct  # placement dwell not yet reached
    run(task, resting(task, {name: moved}), 0.15)
    assert task.correct == {name} and task.counted[name] == pocket
    assert task.accuracy == 1.0


def test_wrong_pocket_is_an_attempt_not_credit(task):
    name = "vial_1"
    run(task, resting(task, {name: in_pocket(task, name, pocket_for(task, name, matching=False))}), 0.5)
    assert name in task.attempted and name not in task.correct
    assert any(e["type"] == "wrong_pocket" for e in task.event_log)


def test_tipped_vial_does_not_count(task):
    name = "vial_2"
    tilted = in_pocket(task, name, pocket_for(task, name), tilt=0.4)
    run(task, resting(task, {name: tilted}), 0.5)
    assert name not in task.correct


def test_held_vial_does_not_count(task):
    name = "vial_3"
    held = replace(in_pocket(task, name, pocket_for(task, name)), grasped=True)
    run(task, resting(task, {name: held}), 0.5)
    assert name not in task.correct and name not in task.attempted


def test_all_twelve_stocked_is_physical_and_t_q_decides_the_tier(task):
    placed = {}
    used = set()
    for name in task.layout.objects:
        pocket = next(
            p
            for p, d in task.layout.info["c3"]["pocket_drug"].items()
            if d == task.layout.info["c3"]["item_drug"][name] and p not in used
        )
        used.add(pocket)
        placed[name] = in_pocket(task, name, pocket)
    run(task, resting(task, placed), 1.0)
    assert len(task.correct) == task.spec.num_items
    assert task.temporal_success and task.physical_success  # t_Q ~ 0.2 s, far inside T_hard
    run(task, resting(task, placed), throughput.SETTLE_S)
    metrics = task.metrics()
    assert metrics["termination_reason"] == "all_done" and metrics["joint_success"]
    assert metrics["t_q_s"] == pytest.approx(max(task.placement_times.values()))


def test_too_few_vials_fail_physical(task):
    name = "vial_0"
    run(task, resting(task, {name: in_pocket(task, name, pocket_for(task, name))}), 0.5)
    assert name in task.correct
    assert not task.physical_success and not task.temporal_success  # 1 of 12
    assert task.metrics()["t_q_s"] is None


def test_count_readout_commands(task):
    name = "vial_0"
    state = resting(task, {name: in_pocket(task, name, pocket_for(task, name))})
    shown = []
    for _ in range(40):
        for command in task.advance(DT):
            if command.kind == "show":
                shown.append((command.name, command.value))
        task.observe(state, DT)
    assert ("c3_count_0", False) in shown and ("c3_count_1", True) in shown


def test_two_arm_split_is_balanced_and_reachable(task):
    from roboclock.envs.hospital.bedside import ward_c3

    info = task.layout.info["c3"]
    for side in (0, 1):
        vials = [n for n, arm in info["item_arm"].items() if arm == side]
        pockets = [p for p, arm in info["pocket_arm"].items() if arm == side]
        assert len(vials) == len(pockets) == task.spec.num_items // 2
        assert sorted(info["item_drug"][n] for n in vials) == sorted(info["pocket_drug"][p] for p in pockets)
        planner = type("P", (), {"side": side})()
        plan = ward_c3._plan_order(task, side)
        assert sorted(n for n, _ in plan) == sorted(vials)
        assert len({p for _, p in plan}) == len(plan)  # one pocket per vial
        mount = np.array(info["mounts"][side])
        for name, pocket in plan:
            assert info["item_drug"][name] == info["pocket_drug"][pocket]
            assert info["pocket_arm"][pocket] == side
            for point in (task.layout.objects[name].position, task.layout.targets[pocket].position):
                assert 0.2 < np.hypot(*(point[:2] - mount)) < 0.5  # comfortable X5 reach
            # Wrist-friendly yaw: the diagonal on the side of the arm's bearing.
            yaw = ward_c3._yaw_for(planner, task, task.layout.targets[pocket].position)
            assert abs(abs(yaw) - np.pi / 4) < 1e-9
    # Each arm keeps to its own side of the insert's center line: the side its mount stands on
    # (the common rig puts arm 0 at -y and arm 1 at +y).
    y0 = info["insert_center_y"]
    assert info["mounts"][0][1] < info["mounts"][1][1]
    for pocket, arm in info["pocket_arm"].items():
        y = task.layout.targets[pocket].position[1] - y0
        assert (y > 0) == (arm == 1)


def test_crash_cart_parts_get_a_texture_by_their_source_color():
    """The converted crash-cart model has flat colors only; each class maps to a textured surface."""
    from roboclock.envs.hospital.bedside.ward_c3 import cart_surface

    assert cart_surface((0.8, 0.12, 0.07)) == "red"  # shell and drawers
    assert cart_surface((0.765, 0.765, 0.765)) == "pull"  # drawer pulls
    assert cart_surface((0.86, 0.78, 0.50)) == "beige"  # worktop
    assert cart_surface((0.88, 0.69, 0.08)) == "yellow"  # side bins
    assert cart_surface((0.46, 0.46, 0.43)) == "grey"  # rail bins
    assert cart_surface((0.20, 0.20, 0.20)) == "frame"
    assert cart_surface((0.01, 0.01, 0.01)) == "rubber"  # casters and bumpers
