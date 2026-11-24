"""Summarize measured spotting evidence without replaying or changing a run."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def audit(path):
    result = json.loads((path / "result.json").read_text())
    config = json.loads((path / "config.json").read_text())
    with gzip.open(path / "trace.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    c = config["workflow"]
    event, catch = result["exhaustion_time"], result["catch_time"]
    fatigue_loss = 3 + (c["fatigue_rep"] - 1) * 5 + 2 + 2*c["fatigue_progress"] + c["fatigue_stall"]
    records = []
    for label, when in [("descent", event), ("catch", catch), ("rack", result["rack_time"])]:
        if when is None:
            continue
        row = min(rows, key=lambda r: abs(r["time"]-when))
        s, o = row["state"], row["state"]["objects"]["bar"]
        records.append(dict(event=label, event_time=when, sample_time=row["time"],
                            bar_position=o["position"], bar_velocity=o["velocity"],
                            finger_contacts=o["forces"][:4], finger_gaps=s["finger_gaps"],
                            hook_contacts=o["support_contacts"], human_actuator=s["human_actuator"]))
    waiting = [r for r in rows if .2 < r["time"] < (event or 0.)]
    assisted = [r for r in rows if catch is not None and r["time"] > catch+.02]
    report = dict(
        success=result["success"], fatigue_rep=c["fatigue_rep"],
        human_support_loss_starts=fatigue_loss, detected_descent=event,
        no_pre_fatigue_trigger=event is not None and event >= fatigue_loss,
        normal_repetitions_before_fatigue=c["fatigue_rep"]-1,
        minimum_open_gap_before_event=min((min(r["state"]["finger_gaps"]) for r in waiting), default=None),
        maximum_pre_event_bar_contact=max((max(r["state"]["objects"]["bar"]["forces"][:4]) for r in waiting), default=None),
        standby_success=result.get("standby_success"), standby_peak_error=result.get("standby_peak_error"),
        motion_response_time=result.get("motion_response_time"),
        maximum_human_force_after_catch=max((float(np.linalg.norm(r["state"]["human_actuator"]["force"])) for r in assisted), default=None),
        reaction_latency=result["reaction_latency"], samples=records,
        trace_samples=len(rows), trace_hz=result["trace_hz"],
        note="Event predicates run on every physics tick; nearest exported samples are at the control rate. Human force ceases on the physics tick after catch.",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.run), indent=2)+"\n")
