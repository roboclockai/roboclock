"""Identify recorded hospital sidecars by registered task, without ID collisions.

Sidecars keep the env task ID they were recorded with.
The early single-step bedside recordings were made as env A1-A4, which are now the
process-window tasks, so they become ``<ID>-early`` retired rows. Rows recorded before
2026-11-19 call the infusion start B6 and the pump panel B7; the infusion start gets today's ID
(A4), the pump panel's rows are retired (B6 is the haemorrhage tourniquet since 2026-11-20).
Rows also get the data folder ID as ``benchmark_id``: since the 2026-11-19 demo refresh the
data folders use the env IDs.
"""

from roboclock.envs.hospital.tasks.registry import TASKS

EARLY_BEDSIDE = {
    ("A1", "stopcock_switch"),
    ("A2", "iv_roller_clamp"),
    ("A3", "iv_bag_spiking_hanging"),
    ("A4", "pump_panel_response"),
}
# (recorded ID, task name) -> current ID, for rows recorded before the 2026-11-19 renumbering.
RENAMED = {
    ("B6", "hang_and_start_infusion"): "A4",
}
# (recorded ID, task name) of tasks whose ID now belongs to a different task: their rows are kept
# as retired ``<ID>-<name>`` rows. A3 was the syringe pump cutoff until 2026-11-19 (now the
# bag-valve ventilation rhythm).
# The pump-panel key press was B7 until 2026-11-19 and B6 from then until 2026-11-20, when the
# haemorrhage tourniquet took B6: its rows are retired as ``B6-pump_panel_response`` whichever ID
# they were recorded with.
RETIRED_AS = {("B7", "pump_panel_response"): "B6"}
REPLACED = {
    ("B6", "pump_panel_response"),
    ("B7", "pump_panel_response"),
    ("B1", "pump_alarm"),  # the ICU pump alarms; B1 is the oxygen desaturation response since 2026-11-19
    ("A3", "syringe_pump_cutoff"),
    ("B3", "fire_blanket"),  # B3 is the hazardous drug spill since 2026-11-19
    ("B5", "hang_bag_open_clamp"),  # hang the bag, open the clamp; B5 is the anaphylaxis auto-injector since 2026-11-20
    ("B4", "line_switch"),  # the infusion-line switch; B4 is the ventilator circuit reconnection since 2026-11-19
}


def normalize(document: dict) -> dict | None:
    row = document.get("metrics", document)
    if not isinstance(row, dict) or not {"task_id", "task_success"} <= row.keys():
        return None
    row = dict(row)
    task_id, name = str(row["task_id"]), row.get("task_name")
    if (task_id, name) in EARLY_BEDSIDE:
        row["benchmark_id"] = None  # their data folders (B5-B8) were retired in the refresh
        row["task_id"] = f"{task_id}-early"
        row["retired"] = True
        return row
    if (task_id, name) in REPLACED:
        row["benchmark_id"] = None  # the data folder now shows the task that took over the ID
        row["recorded_task_id"] = task_id
        row["task_id"] = f"{RETIRED_AS.get((task_id, name), task_id)}-{name}"
        row["retired"] = True
        return row
    if (task_id, name) in RENAMED:
        row["recorded_task_id"] = task_id
        task_id = row["task_id"] = RENAMED[task_id, name]
        if row.get("category") == "event_reactive" and task_id.startswith("A"):
            row["category"] = "temporal_window"
    if task_id in TASKS and name in (None, TASKS[task_id].name):
        row["benchmark_id"] = task_id
    else:
        raise ValueError(f"hospital result {task_id}/{name} matches no registered task")
    return row
