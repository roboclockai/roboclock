"""Laboratory stage scores (roboclock.evaluation.score): physical milestones before completion.

Physical completion scores 80 and joint success 100 for every task; the stages below are
the milestones before it, read from the task's own metrics and events. Every task is
listed except B4 (test-tube rescue), whose only step before completion is the grasp;
it scores from the success flags alone.
"""

from roboclock.evaluation.score import PHYSICAL, Stage, given, happened


def at_least(k):
    """Check: the task credited at least k operations (fractions collected, samples placed)."""
    return lambda row: (row.get("correct_operations") or 0) >= k


def placements(row):
    """C1-C3: one stage per correctly placed sample, as RoboDojo's by-count scores.

    Placement k of n scores 80 k / (n + 1), so all n stay below the 80 of completion,
    which also needs every sample still seated at the end and none dropped.
    """
    n = (row.get("episode_spec") or {}).get("num_samples") or 0
    return tuple(
        Stage(f"sample_{k}_placed", round(PHYSICAL * k / (n + 1), 1), at_least(k))
        for k in range(1, n + 1)
    )


STAGES = {
    # Water collection: the closed faucet handle turned open, then turned closed again;
    # completion is the closed faucet held with the volume in band.
    "A1": (
        Stage("faucet_opened", 25, given("faucet_opened_at")),
        Stage("faucet_closed", 55, given("faucet_closed_at")),
    ),
    # Phase separation: the open stopcock turned closed (first closure); completion is
    # the closure held with the right volume drained.
    "A2": (Stage("stopcock_closed", 40, given("action_time")),),
    # Chromatography: each colour band collected cleanly in its own tube (rack slid under
    # the outlet, then away); completion is all four with the waste beaker back.
    "A3": (
        Stage("fraction_1_collected", 16, at_least(1)),
        Stage("fraction_2_collected", 32, at_least(2)),
        Stage("fraction_3_collected", 48, at_least(3)),
        Stage("fraction_4_collected", 64, at_least(4)),
    ),
    # Titration: the open burette stopcock turned closed (first closure); completion is
    # the closure held with the delivered volume in band.
    "A4": (Stage("stopcock_closed", 40, given("action_time")),),
    # Film stop bath and fixer: stop-bath immersion (grasped), out of the stop bath, into the
    # fixer; completion is the released reel retained on the fixer support. The stage names keep
    # their earlier spelling for result compatibility: "developer" is the first, timed bath
    # (labelled STOP BATH) and "stop_bath" the second one (labelled FIXER).
    "A5": (
        Stage("film_in_developer", 20, given("source_entry_time")),
        Stage("film_out_of_developer", 40, given("source_exit_time")),
        Stage("film_in_stop_bath", 60, given("destination_immersion_time")),
    ),
    # Heat shock: the tube from the bench stand into the warm bath (grasped), out of it, into
    # the ice bath; completion is the released tube retained on the ice-bath support.
    "A6": (
        Stage("tube_in_warm_bath", 20, given("source_entry_time")),
        Stage("tube_out_of_warm_bath", 40, given("source_exit_time")),
        Stage("tube_in_ice_bath", 60, given("destination_immersion_time")),
    ),
    # Hotplate alarm: the beaker held 2 cm clear of the hotplate (the timed response);
    # completion is the beaker placed in the safety tray.
    "B1": (Stage("beaker_lifted_off_hotplate", 40, given("isolation_time")),),
    # Thermal short / motor stall: the knife switch opened past 35 degrees under contact
    # (power isolated); completion is the switch held open for 0.6 s.
    "B2": (Stage("switch_opened", 40, happened("power_isolated")),),
    "B3": (Stage("switch_opened", 40, happened("power_isolated")),),
    # Freezer door: the door pushed shut onto its latch (grasped); completion is the
    # latch held for 0.2 s.
    "B5": (Stage("door_latched", 40, happened("freezer_door_latched")),),
    # Vacuum vent: the VENT plunger pressed through its 16 mm travel; completion is the
    # stroke held for 0.2 s.
    "B6": (Stage("vent_plunger_pressed", 40, happened("vacuum_breaker_opened")),),
    # Waste carboy: the hose pinch clamp pressed through its 16 mm travel; completion is
    # the clamp held for 0.2 s.
    "B7": (Stage("hose_clamp_closed", 40, happened("waste_hose_clamped")),),
    # Rack organization / loading / calibration: one stage per sample placed and settled
    # in a correct slot or marker; completion is every sample placed and still seated.
    "C1": placements,
    "C2": placements,
    "C3": placements,
}
