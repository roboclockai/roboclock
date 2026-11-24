"""Industry stage scores (roboclock.evaluation.score): physical milestones before completion.

Physical completion scores 80 and joint success 100 for every task; the stages below are
events the environment measures from scene state for every policy: the scene's own
``scene_events`` and the scoring monitors' ``measured_events`` (``scoring.py``: Task 46's
arrest, the throughput tasks' credited parts). Grasps and contacts (``bilateral_grasp_s``,
``first_hand_contact_s``) never score, and nothing the native controllers record
(``phase_*_s``, their own grasp events) is used: other policies do not produce it.
"""

from fnmatch import fnmatchcase

from roboclock.evaluation.score import PHYSICAL, Stage, happened


def credits(name, total, *patterns):
    """Throughput tasks (C): one stage per credited operation, as RoboDojo's by-count scores.

    Stage k needs k ``measured_events`` matching the patterns, in any order, and scores
    80 k / (total + 1), so all of them stay below the 80 of physical success, which also
    needs all Q jobs still valid at the end of the episode (docs/c-class-timing.md). The
    totals count each task's credits: C1 6 components (2 kits), C2 3 cartons, C3 3 parts,
    C4 8 pans and 2 deliveries (2 palettes).
    """

    def reached(k):
        return lambda row: (
            k
            <= sum(
                any(fnmatchcase(event, pattern) for pattern in patterns)
                for event in row.get("measured_events") or {}
            )
        )

    return tuple(
        Stage(f"{name}_{k}", round(PHYSICAL * k / (total + 1), 1), reached(k))
        for k in range(1, total + 1)
    )


STAGES = {
    # Zincate dip to rinse (41): the part, picked from the loading pad, is fully immersed in
    # the dip bath, starts to leave it, then clears it entirely; completion is the part released and retained in the rinse bath. The stage
    # names keep the earlier "plating" spelling for result compatibility.
    "A1": (
        Stage("part_immersed_in_plating_liquid", 20, happened("plating_started_s")),
        Stage("part_leaving_plating_liquid", 35, happened("first_surface_exit_s")),
        Stage("part_cleared_plating_liquid", 55, happened("whole_part_exit_s")),
    ),
    # Rotary pump loading (42): the grasped housing lifted clear of its source fixture;
    # completion is the released housing retained upright in the moving carrier (the
    # loading sector decides the timing).
    "A2": (Stage("housing_lifted_from_source", 40, happened("source_cleared_s")),),
    # Cover bonding (43): the grasped cover lifted clear of its source pallet; completion is
    # the released cover seated aligned on the housing (the adhesive window decides the
    # timing).
    "A3": (Stage("cover_lifted_from_source", 40, happened("source_cleared_s")),),
    # Valve isolation (44): after the leak, the gripped handle turned past half the closing
    # angle; completion is the valve held closed and released.
    "B1": (Stage("valve_half_closed", 40, happened("valve_half_closed_s")),),
    # Motor emergency stop (45): the stop button pressed through its actuation travel with
    # finger contact; completion is the braked motor staying stopped.
    "B2": (Stage("stop_button_pressed", 40, happened("button_confirmed_s")),),
    # Bearing arrest (46): the rolling bearing held still wholly inside the safety zone (the
    # timed response); completion is the released bearing retained on the rework tray.
    "B3": (Stage("bearing_arrested_in_zone", 40, happened("arrest_confirmed_s")),),
    # Maintenance kitting (47): each of the six bearings, brackets and pins released and
    # settled in its kit insert; completion is both kits still packed at the end.
    "C1": credits("component_kitted", 6, "kit?_*_credited_s"),
    # Carton putaway (48): each carton released at rest in its SKU's shelf location.
    "C2": credits("carton_shelved", 3, "*_correct_slot_credit_s"),
    # Weighing (49): each weighed part released at rest in the tray of
    # its weight result.
    "C3": credits("part_sorted", 3, "*_sorted_s"),
    # Eyeshadow palettes (50): each of the eight pans seated in its case pocket and each of
    # the two filled cases delivered to output.
    "C4": credits("placement", 10, "pan?_?_credited_s", "delivery?_s"),
}
