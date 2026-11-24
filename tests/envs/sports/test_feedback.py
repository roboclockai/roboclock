"""Display observed outcomes without leaking future decoys or false success."""

from types import SimpleNamespace

from roboclock.envs.sports.arcade.feedback import mole_score, taiko_instruction


def task(**changes):
    values = dict(
        terminated=False,
        time=11.0,
        correct=set(),
        violations=[],
        physical_failures=[],
        limits={"note_window": 0.3},
        events=[dict(time=12.0, control="pad_0"), dict(time=12.0, control="pad_1")],
        metrics=lambda: {"joint_success": True},
    )
    return SimpleNamespace(**(values | changes))


def test_mole_score_does_not_disclose_future_holes_or_decoys():
    a = task(events=[dict(valid=True, control="mole_0")])
    b = task(events=[dict(valid=False, control="mole_8")] * 20)
    assert mole_score(a) == mole_score(b) == ("PLAYING", "HITS 00 / ERRORS 00")


def test_final_score_keeps_failures_visible_even_when_all_targets_were_hit():
    t = task(
        terminated=True,
        correct={0, 1},
        violations=["decoy_hit"],
        metrics=lambda: {"joint_success": False},
    )
    assert mole_score(t) == ("CHECK RESULT", "HITS 02 / ERRORS 01")
    assert taiko_instruction(t) == "CHECK RESULT 2/2"


def test_taiko_cues_use_robot_side_and_wait_for_evaluation():
    t = task()
    assert taiko_instruction(t) == "BOTH"
    t.correct = {0}
    assert taiko_instruction(t) == "ROBOT RIGHT"
    t.correct = {0, 1}
    assert taiko_instruction(t) == "FINISHING"
    t.terminated = True
    assert taiko_instruction(t) == "COMPLETE 2/2"
