"""Replay recorded physical observations and quantify a complete C1 rack."""

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roboclock.envs.sports.pool_rack.task import Task


def audit(run):
    config = json.loads((run / "config.json").read_text())
    result = json.loads((run / "result.json").read_text())
    b = SimpleNamespace(config=config, time=0.0)
    task = Task(b)
    initial = json.loads((run / "initial_state.json").read_text())
    white_origin = np.array(initial["objects"]["cue_ball"]["position"])
    white_motion, step_max, arm_force, count = 0.0, 0.0, 0.0, 0
    previous = initial
    evaluated = initial
    with gzip.open(run / "trace.jsonl.gz", "rt") as f:
        for line in f:
            entry = json.loads(line)
            state = entry["state"]
            b.time = state["time"]
            task.update(state)
            count += 1
            white_motion = max(
                white_motion,
                float(
                    np.linalg.norm(
                        np.array(state["objects"]["cue_ball"]["position"]) - white_origin
                    )
                ),
            )
            arm_force = max(arm_force, state.get("arm_contact_peak", 0.0))
            for name, o in state["objects"].items():
                step_max = max(
                    step_max,
                    float(
                        np.linalg.norm(
                            np.array(o["position"]) - previous["objects"][name]["position"]
                        )
                    ),
                )
            if state["time"] <= result["evaluation_end"] + 1e-8:
                evaluated = state
            previous = state
    replay = task.result()
    for name in (
        "success",
        "physical_success",
        "temporal_success",
        "completed_at",
        "accepted",
        "events",
        "final_occupancy",
    ):
        assert replay[name] == result[name], name
    slots = np.array(config["rack"]["slots"])
    occupancy = result["final_occupancy"]
    by_slot = {slot: name for name, slot in occupancy.items()}
    errors, gaps = [], []
    for slot, name in by_slot.items():
        p = np.array(evaluated["objects"][name]["position"])
        errors.append(float(np.linalg.norm(p[:2] - slots[slot, :2])))
        for j in range(slot):
            if (
                j in by_slot
                and np.linalg.norm(slots[slot] - slots[j]) < 2 * config["rack"]["radius"] + 0.001
            ):
                q = np.array(evaluated["objects"][by_slot[j]]["position"])
                gaps.append(float(np.linalg.norm(p - q) - 2 * config["rack"]["radius"]))
    report = dict(
        replay_matches_report=True,
        physics_samples=count,
        white_ball_max_displacement_m=white_motion,
        max_ball_displacement_per_physics_step_m=step_max,
        peak_arm_contact_n=arm_force,
        valid_final_balls=len(occupancy),
        max_final_slot_xy_error_m=max(errors, default=None),
        neighbor_surface_gap_min_m=min(gaps, default=None),
        neighbor_surface_gap_max_m=max(gaps, default=None),
        final_apex_ball=by_slot.get(0),
        final_center_ball=by_slot.get(4),
        final_corner_balls=[by_slot.get(i) for i in (10, 14)],
        completed_at=result["completed_at"],
        budget=result["budget"],
    )
    (run / "trace_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
