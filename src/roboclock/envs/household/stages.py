"""Household stage scores (roboclock.evaluation.score): physical milestones before completion.

Physical completion scores 80 and joint success 100 for every task; the stages below are
the milestones before it, read from the events the task marks from measured signals (the
oracle's own phase events never reach the row). Every task is listed except B1, whose row
records no physical change before the faucet counts as closed (only first lever contact);
it scores from the success flags alone. The throughput tasks C1-C3 score one stage per
bowl, glass or utensil, as RoboDojo's by-count scores.
"""

from roboclock.evaluation.score import Stage, events, happened


def at_least(n, prefix, suffix=""):
    """Check: at least n distinct events named prefix...suffix (one per bowl, glass or utensil)."""
    return lambda row: sum(e.startswith(prefix) and e.endswith(suffix) for e in events(row)) >= n


STAGES = {
    # Sugar: sugar running from the tilted jar, the jar set down upright and released on the
    # counter, then the tea stirred two clockwise turns with the spoon; completion is the spoon
    # released in its basket (the grams in the glass when the jar is down decide the timing).
    "A1": (
        Stage("sugar_poured", 20, happened("pour_started")),
        Stage("jar_set_down", 40, happened("jar_set_down")),
        Stage("tea_stirred", 60, happened("stirred")),
    ),
    # Ice dispenser: the glass off the counter, the paddle held in until dispensing starts,
    # the paddle released after dispensing; completion is the glass set down with the ice
    # settled (the ice level decides the timing).
    "A2": (
        Stage("glass_picked_up", 20, happened("glass_picked_up")),
        Stage("ice_dispensing", 40, happened("dispense_started")),
        Stage("paddle_released", 60, happened("released")),
    ),
    # Pancake: turned over (rotated past 150 degrees); completion is the pancake settled in
    # the pan, untouched, with the turner withdrawn (the doneness at the flip decides the timing).
    "A3": (Stage("pancake_flipped", 40, happened("flip")),),
    # Pour to weight: the kettle tipped over 5 degrees toward its spout, off its rest (water
    # reaching the jug is not recorded before completion); completion is the kettle back
    # upright and released for 3 s with the stream stopped (dose and budget decide the timing).
    "A4": (Stage("kettle_tipped_to_pour", 40, happened("pour_started")),),
    # Rolling object: stopped on the board before the counter edge; completion is the
    # object held clear of the counter for the hold time.
    "B2": (Stage("object_stopped_before_edge", 40, happened("stopped")),),
    # Pot fire: the burner knob turned to OFF and released; completion is the flame out with
    # the knob left off for 3 s (the fire spreading decides the timing).
    "B3": (Stage("burner_knob_off", 40, happened("heat_off")),),
    # Serving relay (R3): one stage per bowl set down on the serving board. Completion is every
    # ordered meal served by the end, late or not, with no bowl dropped (four orders at every tier);
    # each meal on the board by its pickup decides the timing (C-class stream judging).
    "C1": tuple(Stage(f"bowls_served_{k}", 16 * k, at_least(k, "bowl", "_served")) for k in range(1, 5)),
    # Cola: cola counted in a glass, then one stage per glass filled into the line band and
    # settled; completion is all six in the band at the end and both bottles back upright and
    # released, with nothing dropped, knocked or spilled (t_Q against the tier budget decides
    # the timing).
    "C2": (
        Stage("cola_in_a_glass", 10, happened("pour_started")),
        *(Stage(f"glasses_filled_{k}", 10 + 10 * k, at_least(k, "glass", "_served")) for k in range(1, 7)),
    ),
    # Cutlery: one stage per loose utensil resting, released, in its own compartment (six since
    # 2026-11-09); completion is all six sorted and none in a wrong compartment at the end, with
    # nothing dropped (t_Q against the tier budget decides the timing).
    "C3": tuple(Stage(f"utensils_sorted_{k}", 12 * k, at_least(k, "sorted_")) for k in range(1, 7)),
}
