"""Sports stage scores (roboclock.evaluation.score): physical milestones before completion.

Physical completion scores 80 and joint success 100 for every task; the stages below are
read from the events and native result the tasks' own evaluators record from measured
scene state. The oracles' bookkeeping (B1 ``mallet_assigned`` events, C1
``controller_transitions``) and the robot's own motion (B3 ``approach_response``, B4
``confirmed_stop``) never score.
Repetition tasks score repetition k of n at 80 k / (n + 1), as the boxing strikes.
"""

from roboclock.evaluation.score import PHYSICAL, Stage, happened


def count(row, name, **fields):
    """How many of the row's events are `name`, with the given field values."""
    return sum(
        1
        for event in row.get("events") or ()
        if isinstance(event, dict)
        and name in (event.get("type"), event.get("kind"), event.get("event"))
        and all(event.get(key) == value for key, value in fields.items())
    )


def repetitions(stage, event, n):
    """One stage per repetition: the k-th of n `event`s scores 80 k / (n + 1)."""
    return tuple(
        Stage(
            f"{stage}_{k}",
            round(PHYSICAL * k / (n + 1), 1),
            lambda row, k=k: count(row, event) >= k,
        )
        for k in range(1, n + 1)
    )


def white_placements(row):
    """Chess (A1): moves WHITE, the policy, has placed (BLACK is the environment)."""
    return count(row, "legal_move_placed", actor="white")


def strikes(row):
    """Boxing (B2): one stage per correct-pad strike, as RoboDojo's by-count scores.

    Strike k of n required scores 80 k / (n + 1), so all n strikes stay below the 80 of
    completion, which also needs every guard return and no movement during HOLD.
    """
    native = row.get("native_result") or {}
    required = sum(
        len(command.get("hands") or ()) for command in native.get("boxing_schedule") or ()
    )
    landed = len(native.get("hits") or ())
    return tuple(
        Stage(f"strike_{k}", round(PHYSICAL * k / (required + 1), 1), lambda _row, k=k: landed >= k)
        for k in range(1, required + 1)
    )


def cups(row, cards=2):
    """Speed cups (C2): cups placed on the required cards (``cup_placed`` once per card and cup)."""
    return sum(count(row, "cup_placed", card=card) for card in range(cards))


TAIKO = repetitions("note_hit", "stick_hit", 11)
BASKETBALL = (
    Stage("ball_held", 25, happened("grasp")),
    Stage("ball_released", 55, happened("release")),
)

STAGES = {
    # Chess and clock: WHITE (the policy) plays plies 2 and 4, BLACK (the environment's
    # opponent) plies 1 and 3, so only WHITE's events count. A move is placed when the
    # grasped piece rests released and settled on its destination; the clock press is the
    # button's travel and dwell after that placement. Completion is WHITE's second press.
    "A1": (
        Stage("white_move_1_placed", 20, lambda row: white_placements(row) >= 1),
        Stage(
            "white_clock_1_pressed",
            40,
            lambda row: count(row, "clock_handoff", actor="white", physical_success=True) >= 1,
        ),
        Stage("white_move_2_placed", 60, lambda row: white_placements(row) >= 2),
    ),
    # Basketball shot clock (A3 until 2026-11-19): the ball held, then released (thrown);
    # completion is the ball through the hoop (the shot clock decides the timing of the release).
    # Rows recorded before the renumbering call taiko A2 (above) and basketball A3; the result
    # adapter (results.py) gives them today's IDs.
    "A2": lambda row: TAIKO if row.get("task_name") == "two_stick_taiko" else BASKETBALL,
    # Nine-hole moles: each of the 6 green moles (3 of the 9 risers are red decoys) struck
    # by the head of a held mallet while it is up; completion is all 6.
    "B1": repetitions("mole_hit", "mole_hit", 6),
    "B2": strikes,
    # Bench-press assistance: the falling bar arrested in both retained grips (the timed
    # response); completion is the bar released and seated on both rack hooks.
    "B3": (Stage("bar_caught", 40, happened("bilateral_catch")),),
    # Stop/go puck: each of the two goals is a pusher-driven puck past the lane checkpoint,
    # then into the goal; completion is the second goal.
    "B4": (
        Stage("puck_past_checkpoint_1", 20, lambda row: count(row, "checkpoint") >= 1),
        Stage("puck_in_goal_1", 40, lambda row: count(row, "goal") >= 1),
        Stage("puck_past_checkpoint_2", 60, lambda row: count(row, "checkpoint") >= 2),
    ),
    # Pool racking: each of the 15 balls lifted, released and settled in a rack slot;
    # completion is the legal triangle with both hands withdrawn.
    # Two-stick taiko (A2 until 2026-11-19): each of the 11 notes (8 beats, every third a
    # two-drum chord) struck on its drum by the head of a held stick, within the note window;
    # completion is all 11.
    "B5": TAIKO,
    "C1": repetitions("ball_racked", "placed", 15),
    # Speed cups: the 2 required cards (Q = 2 jobs) of 6 cups each; a cup counts when it
    # first rests released on its target with the card's support (the arena's
    # ``cup_placed``). Cup k of the 12 scores 80 k / 13; a second card's cups are placed only
    # after the first card's bell. Completion is both cards stable and rung.
    "C2": tuple(
        Stage(f"cup_placed_{k}", round(PHYSICAL * k / 13, 1), lambda row, k=k: cups(row) >= k)
        for k in range(1, 13)
    ),
    # Ball service: a ball for each of the 6 demands grasped, released into a hopper cell and
    # resting on its gate; completion is all 6 served through the outlet.
    "C3": repetitions("ball_stocked", "accepted", 6),
}
