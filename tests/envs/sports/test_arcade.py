"""Boundary and no-double-count checks for the new drum timing contract."""

import pytest

from roboclock.envs.rates import PHYSICS_HZ
from roboclock.envs.sports.arcade.tasks import MalletMoles, TaikoTask, configuration


def task():
    return TaikoTask("easy", 0, configuration("A2"))


def impact(t, name, onset, step):
    t.time = onset
    t.steps = step
    t.rises = {name}
    t.presses[name] = onset
    state = {"controls": {name: {"tool_force": 2.0}}}
    t.evaluate(state, 1 / 120)


def test_early_contact_is_not_a_note():
    t = task()
    e = t.events[0]
    impact(t, e["control"], e["time"] - t.limits["note_window"] - 0.01, 1)
    assert not t.correct
    assert any("offbeat" in x for x in t.violations)


def test_wrong_drum_cannot_score():
    t = task()
    e = t.events[0]
    impact(t, "pad_1" if e["control"] == "pad_0" else "pad_0", e["time"], 1)
    assert not t.correct


def test_contact_cannot_score_a_note_twice():
    t = task()
    e = t.events[0]
    impact(t, e["control"], e["time"], 1)
    impact(t, e["control"], e["time"] + 0.1, 2)
    assert t.correct == {0}
    assert len(t.hits) == 1
    assert t.violations


def test_missed_notes_do_not_silently_disappear():
    t = task()
    t.time = t.events[-1]["time"] + t.limits["note_window"] + 1
    t.rises = set()
    t.evaluate({}, 1 / 120)
    assert t.terminated and not t.joint_success
    assert len(t.violations) == len(t.events)


def test_roll_requires_separate_timed_impacts():
    t = TaikoTask("easy", 0, configuration("A2", "roll"))
    assert [e["kind"] for e in t.events[-2:]] == ["roll", "roll"]
    assert t.events[-2]["time"] < t.events[-1]["time"] < t.spec.max_time
    assert all(e["hold"] == 0 for e in t.events)


def test_mole_cue_waits_for_fixed_preparation_period():
    t = MalletMoles("easy", 0, configuration("B1"))
    assert min(e["time"] for e in t.events) == 12
    assert len(t.layout.controls) == 9
    assert sorted(e["control"] for e in t.events) == [f"mole_{i}" for i in range(9)]
    assert sum(not e["valid"] for e in t.events) == 3
    assert not t.mole_active(0, time=11.9)


def test_shaft_contact_does_not_score_a_head_strike():
    from roboclock.envs.sports.arcade.contacts import head_contacts

    force, points = head_contacts([[0, 0, 0], [0, 0, -0.149]], [10, 2], [0, 0, 0], [1, 0, 0, 0])
    assert force == 2
    assert len(points) == 1


def test_head_contact_uses_measured_rotation():
    from roboclock.envs.sports.arcade.contacts import head_contacts

    force, _ = head_contacts([[0, 0, 0.149]], [2], [0, 0, 0], [0, 1, 0, 0])
    assert force == 2


def test_rectangular_mallet_edge_contact_is_valid():
    from roboclock.envs.sports.arcade.contacts import head_contacts

    force, _ = head_contacts([[0.045, 0, -0.148]], [3], [0, 0, 0], [1, 0, 0, 0], True)
    assert force == 3


def test_horizontal_tool_head_contact_and_shaft_rejection():
    from roboclock.envs.sports.arcade.contacts import head_contacts

    # Local -Z points along world +X. The lower world-facing head surface
    # belongs to local +X, so the old local-bottom-only test would reject it.
    q = [2**-0.5, 0, -(2**-0.5), 0]
    force, points = head_contacts(
        [[0.125, 0, -0.016], [0.02, 0, -0.011]],
        [3, 25],
        [0, 0, 0],
        q,
        radius=0.016,
        lower_only=False,
    )
    assert force == 3 and len(points) == 1


def test_wrist_arc_lowers_the_head_without_changing_lever_length():
    import numpy as np

    from roboclock.envs.sports.arcade.oracle import pitch_matrix

    lever = np.array([0.145, 0, 0])
    ready = pitch_matrix(np.deg2rad(-25)) @ lever
    impact = pitch_matrix(np.deg2rad(22)) @ lever
    assert ready[2] > 0 and impact[2] < 0
    assert ready[2] - impact[2] > 0.10
    assert np.isclose(np.linalg.norm(ready), np.linalg.norm(impact))


def dual_task(seed=0):
    from roboclock.simulators.isaac.embodiments.config import configure

    c = configuration("A2")
    configure(c, "A2")
    return TaikoTask("easy", seed, c)


def test_dual_taiko_labels_follow_robot_forward_x_left_y():
    t = dual_task()
    assert t.layout.controls["pad_0"].position[1] > 0
    assert t.layout.controls["pad_1"].position[1] < 0
    assert t.layout.bodies["mallet_left"].position[1] > 0
    assert t.layout.bodies["mallet"].position[1] < 0
    assert t.events == dual_task().events
    assert t.events != dual_task(12).events


def test_simultaneous_notes_require_two_distinct_contact_edges():
    t = dual_task()
    first = t.events[0]["time"]
    pair = [(i, e) for i, e in enumerate(t.events) if e["time"] == first]
    assert len(pair) == 2 and {e["control"] for _, e in pair} == {"pad_0", "pad_1"}
    impact(t, "pad_0", first, 1)
    assert len(t.correct) == 1
    impact(t, "pad_0", first + 0.01, 2)
    assert len(t.correct) == 1
    impact(t, "pad_1", first + 0.02, 3)
    assert len(t.correct) == 2


def test_a_missed_chord_hand_still_fails_the_note_window():
    t = dual_task()
    first = t.events[0]["time"]
    impact(t, "pad_0", first, 1)
    t.time = first + t.limits["note_window"] + 0.01
    t.rises = set()
    t.evaluate({}, 1 / 120)
    assert len(t.correct) == 1
    assert any("missed_note" in reason for reason in t.violations)


def test_long_stick_contacts_use_actual_head_not_old_shaft_location():
    from roboclock.envs.sports.arcade.contacts import head_contacts

    points = [[0, 0, -0.225], [0, 0, -0.125], [0, 0, 0.01]]
    force, contacts = head_contacts(
        points,
        [2, 10, 20],
        [0, 0, 0],
        [1, 0, 0, 0],
        radius=0.016,
        head_offset=0.225,
        lower_only=False,
    )
    assert force == 2
    assert len(contacts) == 1


# Injected evaluator states below are predicate checks, not physics rollouts.


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("edge", ["early", "late"])
def test_supplemental_note_just_outside_each_window_fails(level, edge):
    t = TaikoTask(level, 2, configuration("A2"))
    event = t.events[0]
    offset = (t.limits["note_window"] + 1 / PHYSICS_HZ) * (-1 if edge == "early" else 1)
    impact(t, event["control"], event["time"] + offset, 1)
    assert not t.correct and t.violations


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_supplemental_touching_decoy_without_depressing_is_rejected(level, seed):
    t = MalletMoles(level, seed, configuration("B1"))
    i, event = next((i, e) for i, e in enumerate(t.events) if not e["valid"])
    t.time = event["time"] + 0.1
    t.mole_up_times[i] = t.time
    t.rises = set()  # A touch must fail even without a scored depression edge.
    controls = {n: dict(exposed=n == event["control"], contact=n == event["control"])
                for n in t.layout.controls}
    t.evaluate(dict(controls=controls), 1 / PHYSICS_HZ)
    assert "decoy_hit" in t.violations and i not in t.correct


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_supplemental_mole_exact_expiry_does_not_count(level):
    t = MalletMoles(level, 0, configuration("B1"))
    i, event = next((i, e) for i, e in enumerate(t.events) if e["valid"])
    t.mole_up_times[i] = event["time"] + 0.1
    t.time = t.mole_up_times[i] + t.limits["deadline"]
    t.rises = {event["control"]}
    t.presses[event["control"]] = t.time
    controls = {n: dict(exposed=False, contact=False) for n in t.layout.controls}
    t.evaluate(dict(controls=controls), 1 / PHYSICS_HZ)
    assert i not in t.correct
    assert "inactive_mole_hit" in t.violations


@pytest.mark.parametrize("task_id", ["A2", "B1"])
def test_supplemental_recorded_arm_collision_must_block_joint_success(task_id):
    """An otherwise completed episode must not hide detected arm collisions."""
    cls = TaikoTask if task_id == "A2" else MalletMoles
    t = cls("easy", 0, configuration(task_id))
    t.goal_achieved = True
    t._terminated = True
    t.collision_count = 1
    metrics = t.metrics()
    assert not metrics["joint_success"]
    assert not metrics["manipulation_success"]
    assert metrics["temporal_success"]
    assert metrics["physical_failures"] == ["arm_collision"]
    if task_id == "B1":
        from roboclock.envs.sports.arcade.feedback import mole_score
        assert mole_score(t) == ("CHECK RESULT", "HITS 00 / ERRORS 01")
