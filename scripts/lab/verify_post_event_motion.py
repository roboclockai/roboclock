"""Audit measured arm motion in the B1/B2/B4 videos against observed event times."""

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = {"B1": "hotplate_isolation", "B2": "thermal_short_circuit", "B4": "test_tube_rescue"}


def verify(directory, tasks):
    reports = []
    for task in tasks:
        data = json.loads((directory / f"{task}_{NAMES[task]}.json").read_text())
        trace, metrics = data["process_trace"], data["metrics"]
        event = metrics["event_time"]
        times = np.array([x["time"] for x in trace])
        joints = np.array([x["robot_joint_positions"] for x in trace])
        ee = np.array([x["robot_ee_position"] for x in trace])
        joint_motion = np.max(np.abs(joints - joints[0]), axis=1)
        ee_motion = np.linalg.norm(ee - ee[0], axis=1)
        before = times < event
        # Gravity loads the stationary PD servos in the first few physics ticks
        # (~3 mm / 4 mrad from the initial frame). This is distinct from a reach.
        moved = (joint_motion > 0.005) | (ee_motion > 0.005)
        onset = float(times[moved][0]) if np.any(moved) else None
        report = {
            "task_id": task,
            "event_time": event,
            "first_measured_motion_time": onset,
            "maximum_pre_event_joint_drift": float(np.max(joint_motion[before])),
            "maximum_pre_event_ee_drift_m": float(np.max(ee_motion[before])),
            "reaction_latency": metrics["reaction_latency"],
            "response_window_s": metrics["episode_spec"]["max_latency"],
            "joint_success": metrics["joint_success"],
            "pass": bool(
                onset is not None
                and onset > event
                and not np.any(moved[before])
                and metrics["joint_success"]
                and metrics["collision_count"] == 0
                and 0 <= metrics["reaction_latency"] < metrics["episode_spec"]["max_latency"]
            ),
        }
        reports.append(report)
    result = {
        "scope": "Measured PhysX poses, one seed per video",
        "settling_tolerance": {"joint_rad": 0.005, "end_effector_m": 0.005},
        "tasks": reports,
    }
    (directory / ("verification_post_event_" + "_".join(tasks) + ".json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if not all(x["pass"] for x in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/lab/videos-v2"))
    parser.add_argument("--tasks", nargs="+", choices=list(NAMES), default=list(NAMES))
    args = parser.parse_args()
    verify(args.directory, args.tasks)
