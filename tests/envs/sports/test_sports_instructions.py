"""The instruction states every rule a task's evaluator judges (sweep v2, 2026-11-10).

Each number an instruction gives is read from the evaluator's own constant or configuration,
so the tests compare the two. The A1 turn handoff runs on stand-in rigs (no simulator).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.sports.adapters import chess as chess_adapter
from roboclock.envs.sports.tasks import make_task

LEVELS = ("easy", "medium", "hard")


def text(task_id, level="easy", seed=0):
    return make_task(task_id, level, seed).gen_instruction()


# ---------------------------------------------------------------- A1 turn handoff
def handoff_adapter(*, since_press, joint_error, tool_z=1.06, turn_index=1, actor="white"):
    """A ChessAdapter with stand-in task, rigs and states: `actor` pressed `since_press`
    seconds ago, its right arm is `joint_error` rad from its starting pose and its tool at
    `tool_z` (the waiting pose holds the tool at about 1.06 m)."""
    home = np.linspace(-0.4, 0.6, 8)
    joints = home.copy()
    joints[2] += joint_error
    began, resets = [], []
    a = object.__new__(chess_adapter.ChessAdapter)  # build() needs the simulator
    a.t = SimpleNamespace(turn_index=turn_index, time=20.0, sequence=["m"] * 4, done=False)

    def begin_turn():
        began.append(a.t.turn_index)
        a.t.actor = "black" if actor == "white" else "white"

    a.t.begin_turn = begin_turn
    a.backend = SimpleNamespace(states={actor: {"tool_position": np.array([0.4, -0.3, tool_z])}})
    a.views = {actor: SimpleNamespace(rig=SimpleNamespace(arm_ids={"right": list(range(6))}), active_side="right")}
    a.robots = {actor: SimpleNamespace(get_joint_positions=lambda: joints)}
    a.home = {actor: home}
    a.black = SimpleNamespace(reset=resets.append)
    a.end_time = None
    origin = np.array([0.4, -0.3, 0.8])
    a.retire = dict(turn=turn_index, start=20.0 - since_press, origin=origin, command=origin.copy(), lifted=False)
    return a, began, resets


def test_returning_to_the_waiting_pose_hands_the_turn_over_without_a_lift():
    """Both sweep seeds: WHITE went back to its waiting pose (tool at 1.06 m), the handoff
    also wanted the tool above 1.09 m first, and BLACK never moved."""
    a, began, resets = handoff_adapter(since_press=2.0, joint_error=0.01)
    assert a.backend.states["white"]["tool_position"][2] < chess_adapter.LIFTED_HEIGHT
    a.handoff("white")
    assert a.t.turn_index == 2 and began == [2] and a.retire is None
    assert resets == [{}]  # BLACK's controller starts its turn


@pytest.mark.parametrize(
    ("since_press", "joint_error"),
    [
        (1.0, 0.0),  # back already, but inside the 1.5 s dwell after the press
        (5.0, 0.03),  # outside the stated 0.025 rad
    ],
)
def test_the_turn_stays_until_the_arm_is_back_and_the_dwell_is_over(since_press, joint_error):
    a, began, _ = handoff_adapter(since_press=since_press, joint_error=joint_error, tool_z=1.12)
    a.handoff("white")
    assert a.t.turn_index == 1 and not began and a.retire is not None
    # The scripted retire motion's lift latch is still kept for it.
    assert a.retire["lifted"]


def test_the_last_return_ends_the_episode_after_the_post_roll():
    a, began, _ = handoff_adapter(since_press=3.0, joint_error=0.0, turn_index=3)
    a.handoff("white")
    assert a.t.done and not began and a.end_time == pytest.approx(20.0 + chess_adapter.END_POST_ROLL)


@pytest.mark.parametrize("level", LEVELS)
def test_chess_instruction_states_the_handoff_it_is_judged_by(level):
    task = make_task("A1", level, 0)
    instruction = task.gen_instruction()
    assert f"within {task.spec.timing['clock_deadline']:g} s of BLACK's press" in instruction
    assert (
        "return the right arm to the joint pose it started in (every joint within "
        f"{chess_adapter.RETURN_TOLERANCE:g} rad)" in instruction
    )
    assert "lift" not in instruction  # the lift is the scripted motion's, not a rule
    for rule in ("other pieces", "BLACK's button", "arm link"):
        assert rule in instruction


# ---------------------------------------------------------------- stated tolerances
def test_basketball_states_the_no_dunk_distance_the_height_and_the_arm():
    from roboclock.envs.sports.arena import tasks as arena

    task = make_task("A2", "medium", 1)
    instruction = task.gen_instruction()
    # The X5 backend reads only the active arm's jaws for the grasp and the release.
    assert task.native_config["embodiment"]["active_side"] == "right"
    assert "with the right gripper" in instruction
    assert f"at least {arena.NO_DUNK_RADIUS:g} m (horizontally) from the hoop's centre" in instruction
    assert f"higher than {arena.RELEASE_MIN_HEIGHT:g} m" in instruction
    assert "a dunk does not count" in instruction
    assert f"{task.spec.timing['shot_clock']:g} s shot clock" in instruction


def test_cups_state_the_tolerance_and_that_a_wrong_bell_is_final():
    from roboclock.envs.sports.arena import tasks as arena

    instruction = text("C2")
    assert f"within {arena.CUP_TOLERANCE * 1000:g} mm of its place" in instruction
    assert "press the bell once" in instruction
    assert "incomplete pattern or before the preparation phase ends fails the run" in instruction
    assert "cannot be undone" in instruction


def test_rack_states_the_slot_tolerance_and_the_handover():
    from roboclock.envs.sports.pool_rack import task as rack

    task = make_task("C1", "hard", 2)
    c, instruction = task.native_config["rack"], task.gen_instruction()
    assert c["placement_tolerance"] == 0.004 and c["neighbor_gap_tolerance"] == 0.002
    assert f"within {c['placement_tolerance'] * 1000:g} mm of its rack spot" in instruction
    assert f"no more than {c['neighbor_gap_tolerance'] * 1000:g} mm apart" in instruction
    assert f"lifted at least {rack.LIFT_HEIGHT * 100:g} cm" in instruction
    assert f"at least {rack.WITHDRAW_DISTANCE * 100:g} cm from every spot" in instruction
    assert "Do not touch the cue ball" in instruction


@pytest.mark.parametrize("level", LEVELS)
def test_boxing_states_where_guard_is(level):
    from roboclock.envs.sports.arena import tasks as arena

    task = make_task("B2", level, 1)
    w, instruction = task.native_config["workflow"], task.gen_instruction()
    assert f"within {w['cue_response']:g} s of each GO cue, once per cue" in instruction
    assert f"to guard within {w['guard_deadline']:g} s" in instruction
    assert "x 0.35, y +/-0.23 (left +), z 1.06 m" in instruction
    for radius in (arena.GUARD_RETURN, arena.GUARD_HOLD, arena.GUARD_OTHER):
        assert f"within {radius * 100:g} cm of" in instruction


@pytest.mark.parametrize("level", LEVELS)
def test_bench_press_states_the_jaw_gaps_and_the_grip_points(level):
    from roboclock.envs.sports.bench_press import task as bench

    task = make_task("B3", level, 0)
    w, instruction = task.native_config["workflow"], task.gen_instruction()
    assert f"within {w['catch_deadline']:g} s" in instruction
    assert f"open wider than {bench.STANDBY_GAP * 1000:g} mm" in instruction
    assert f"within {bench.GRIP_TOLERANCE * 100:g} cm of the point {w['bench_grip_y']:g} m" in instruction
    assert f"opening both grippers wider than {bench.RELEASE_GAP * 1000:g} mm" in instruction
    assert f"drop below {w['human_danger_height']:g} m" in instruction
    # The standby pose itself satisfies the standby rule.
    assert 2 * w["standby_grip"] > bench.STANDBY_GAP


@pytest.mark.parametrize("level", LEVELS)
def test_stop_go_states_what_must_stop_and_how_still(level):
    task = make_task("B4", level, 0)
    f, instruction = task.native_config["freeze"], task.gen_instruction()
    from roboclock.envs.sports.freeze import task as freeze

    assert (freeze.CHECKPOINT_PAST_MIDLINE, freeze.CHECKPOINT_LANE_HALF_WIDTH) == (0.060, 0.040)
    assert (
        "the indicated hand changes once the puck is 6 cm past the midline within 4 cm of the "
        "lane's centre line" in instruction
    )
    assert "hands, pushers and puck to rest" in instruction
    assert f"(under {min(f['hand_linear_limit'], f['body_linear_limit']) * 100:g} cm/s)" in instruction
    assert f"within {f['response_limit']:g} s" in instruction
    assert f"hold them within {f['hold_translation'] * 100:g} cm until GREEN" in instruction


@pytest.mark.parametrize("level", LEVELS)
def test_moles_and_taiko_state_their_windows_and_faults(level):
    moles = make_task("B1", level, 0)
    assert f"within {moles.spec.timing['deadline']:g} s of its rise" in moles.gen_instruction()
    assert "Never touch a red mole" in moles.gen_instruction()
    taiko = make_task("B5", level, 0)
    assert f"within {taiko.spec.timing['note_window']:g} s before or after its note" in taiko.gen_instruction()
    assert "Any other drum strike is a fault" in taiko.gen_instruction()


@pytest.mark.parametrize("task_id", ["A1", "A2", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3"])
def test_every_instruction_fits_the_observation_space(task_id):
    import string

    charset = set(string.printable + "–°")  # SportsBenchmarkEnv's `instruction` space
    for level in LEVELS:
        for seed in range(4):
            instruction = text(task_id, level, seed)
            assert len(instruction) <= 512 and set(instruction) <= charset
