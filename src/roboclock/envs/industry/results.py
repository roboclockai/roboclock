"""Recorded industry sidecars keep a result row under ``metrics`` with the original
41–50 task numbers and ``reactive`` as the B category; map them to the shared IDs.
The runners' raw ``native-result.json`` evidence is not a row (its sidecar is)."""

from roboclock.envs.industry.tasks import LEGACY_IDS


def normalize(document: dict) -> dict | None:
    row = document.get("metrics", document)
    if not isinstance(row, dict) or not {"task_id", "task_success", "seed"} <= row.keys():
        return None
    row = dict(row)
    number = str(row.get("legacy_task_id", row["task_id"]))
    if number in LEGACY_IDS:
        row["legacy_task_id"] = int(number)
        row["task_id"] = LEGACY_IDS[number]
    return row
