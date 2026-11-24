"""Hospital stage scores (roboclock.evaluation.score): physical milestones before completion.

Physical completion scores 80 and joint success 100 for every task; the stages below are
the milestones before it, read from the task's own metrics. Rows of the ICU pump alarms,
B1 until 2026-11-19, have no stage and score from the success flags alone: their SILENCE
buttons were static fixtures, pressed by fingertip pose.
"""

from roboclock.evaluation.score import PHYSICAL, Stage, given


def at_least(field, count):
    """Check: the row's `field` (milestone times by object) holds at least `count` entries."""
    return lambda row: len(row.get(field) or ()) >= count


HANG_AND_START = (
    Stage("port_on_spike", 20, given("spike_entry_time")),
    Stage("spike_fully_inserted", 40, given("insertion_time")),
    Stage("bag_hung", 60, given("hanging_time")),
)
PUMP_PANEL = (Stage("requested_key_moved", 40, given("key_moved_time")),)

STAGES = {
    # Drip chamber level: the roller leaves its CLOSED rest (1.5 mm of its 36 mm slide), then
    # reaches the OPEN end (the bag drips); completion is the roller back at the CLOSED end stop
    # (the window, relative to the opening, decides the timing).
    "A1": (
        Stage("roller_moved", 20, given("device_moved_time")),
        Stage("clamp_opened", 40, given("open_time")),
    ),
    # Bag switch: the selector turned off its BAG 1 port (3 degrees); completion is the
    # pointer in the BAG 2 detent after a contact-driven turn (the window decides the timing).
    "A2": (Stage("selector_turned", 40, given("device_moved_time")),),
    # Bag-valve ventilation (A3 since 2026-11-19): the bag compressed under contact (3 mm), the
    # first breath delivered (a full squeeze), then half of the breaths; completion is all of
    # them (the breath windows decide the timing). Rows of the syringe pump cutoff, A3 until
    # then, keep their one stage: the STOP key leaving its rest position under contact.
    "A3": lambda row: (
        (Stage("stop_key_moved", 40, given("device_moved_time")),)
        if row.get("task_name") == "syringe_pump_cutoff"
        else (
            Stage("bag_compressed", 15, given("device_moved_time")),
            Stage("first_breath_delivered", 35, given("first_breath_time")),
            Stage(
                "half_of_breaths_delivered",
                55,
                lambda r: 2 * int(r.get("breaths_delivered") or 0) >= int(r.get("breaths_required") or 1),
            ),
        )
    ),
    # Hang and start infusion (B6 until 2026-11-19): the held bag's port enters the spike aligned,
    # is inserted 18 mm, then hangs released on the hook (step 1); completion is the START key held.
    "A4": HANG_AND_START,
    # Oxygen desaturation (B1 since 2026-11-19): the flowmeter knob turned under contact after
    # the alarm (3 degrees), then the ball float inside the ordered band; completion is the
    # float staying there for the dwell. Rows of the ICU pump alarms (B1 until then) have no stage.
    "B1": lambda row: (
        ()
        if row.get("task_name") == "pump_alarm"
        else (
            Stage("knob_turned", 30, given("knob_moved_time")),
            Stage("float_in_band", 60, given("band_entry_time")),
        )
    ),
    # AED: the green tab pulled off the unit and left at rest (frees the pads), then one and
    # both pads resting on their chest marks; completion is the SHOCK press after analysis.
    "B2": (
        Stage("pad_tab_pulled", 20, given("tab_time")),
        Stage("first_pad_on_mark", 40, at_least("pad_times", 1)),
        Stage("both_pads_on_marks", 60, at_least("pad_times", 2)),
    ),
    # Hazardous drug spill (B3 since 2026-11-19): the absorbent pad resting on the liquid, then
    # the first glass piece at rest in the sharps bin; completion is both pieces in the bin.
    # Rows of the fire blanket, B3 until then, keep their stages: one, then both holder tabs
    # pulled away and left at rest (frees the blanket).
    "B3": lambda row: (
        (
            Stage("first_tab_pulled", 25, at_least("tab_times", 1)),
            Stage("blanket_freed", 55, at_least("tab_times", 2)),
        )
        if row.get("task_name") == "fire_blanket"
        else (
            Stage("spill_contained", 30, given("contain_time")),
            Stage("first_glass_in_bin", 55, at_least("shard_times", 1)),
        )
    ),
    # Ventilator reconnection (B4 since 2026-11-19): the circuit's elbow grasped after it came
    # off, then its socket lowered over the 15 mm connector to the seat depth; completion is the
    # released elbow staying on the connector. Rows of the infusion-line switch, B4 until then,
    # keep their stages: the maintenance roller clamp closed and released, then the stopcock
    # turned into its target detent.
    "B4": lambda row: (
        (
            Stage("maintenance_clamp_closed", 25, given("clamp_closed_time")),
            Stage("stopcock_at_target_detent", 55, given("first_target_entry")),
        )
        if row.get("task_name") == "line_switch"
        else (
            Stage("elbow_grasped", 25, given("grasp_time")),
            Stage("connector_seated", 55, given("seat_time")),
        )
    ),
    # Hang bag, open clamp: the bag hanging released by its eyelet on the hook (step 1);
    # completion is the roller retained at the OPEN stop below the hanging bag.
    # Since 2026-11-20 B5 is the anaphylaxis auto-injector: the injector grasped after the alarm,
    # its tip on the thigh's injection pad, then the tip pushed in there (fired); completion is
    # the 3 s hold. Rows of "hang the bag, open the clamp" keep their one stage.
    "B5": lambda row: (
        (Stage("bag_hung", 40, given("hang_time")),)
        if row.get("task_name") == "hang_bag_open_clamp"
        else (
            Stage("injector_grasped", 20, given("grasp_time")),
            Stage("tip_on_injection_pad", 40, given("site_time")),
            Stage("injector_fired", 60, given("fire_time")),
        )
    ),
    # Pump panel (B7 until 2026-11-19): the requested key pressed in (1.5 mm of travel under
    # contact) after the request; completion is the key held at full press for its hold time.
    # Haemorrhage tourniquet (B6 since 2026-11-20): the windlass rod turned under contact after
    # the bleeding started (5 degrees), then turned far enough that the bleeding has stopped;
    # completion is the released rod staying in the retaining clip. Rows of the pump panel, B6
    # until then, keep their one stage.
    "B6": lambda row: (
        HANG_AND_START
        if row.get("task_name") == "hang_and_start_infusion"
        else PUMP_PANEL
        if row.get("task_name") == "pump_panel_response"
        else (
            Stage("windlass_turned", 20, given("turn_time")),
            Stage("bleeding_stopped", 50, given("occlusion_time")),
        )
    ),
    # Rows recorded before 2026-11-19 call hang-and-start B6 (above) and the pump panel B7;
    # the result adapter (results.py) gives them today's IDs.
}


def jobs_done(row):
    """Jobs credited so far: score_version 2 rows carry jobs_done, older rows correct_operations."""
    value = row.get("jobs_done")
    return int(row.get("correct_operations") or 0) if value is None else int(value)


def ward_jobs(row):
    """Ward throughput (C1-C4): counted progress below completion (docs/c-class-timing.md).

    Q + 2 evenly spaced stages below 80: an item lifted 3 cm off its starting pose, an item
    lowered over a container lip into a container, then one stage per job done (k of Q, Q =
    jobs_required or the episode's num_items). Completion (80) needs all Q still valid at the end.
    """
    q = int(row.get("jobs_required") or (row.get("episode_spec") or {}).get("num_items") or 0)
    steps = q + 3

    def score(i):
        return round(PHYSICAL * i / steps, 1)

    def lifted(r):
        return r.get("first_lift_time") is not None or jobs_done(r) >= 1

    def entered(r):
        return r.get("first_entry_time") is not None or jobs_done(r) >= 1

    def done(k):
        return lambda r: jobs_done(r) >= k

    return (
        Stage("item_lifted", score(1), lifted),
        Stage("item_into_container", score(2), entered),
        *(Stage(f"job_{k}_done", score(k + 2), done(k)) for k in range(1, q + 1)),
    )


STAGES.update(dict.fromkeys(("C1", "C2", "C3", "C4"), ward_jobs))
