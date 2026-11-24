"""Verify B2–B4 video integrity, measured response evidence and the shared v2 room."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from PIL import Image

NAMES = {"B2": "thermal_short_circuit", "B3": "motor_stall", "B4": "test_tube_rescue"}
SHARED = [
    "renderer",
    "hdr_path",
    "hdr_intensity",
    "hdr_rotation",
    "key_intensity",
    "fill_intensity",
    "path_tracing_bounces",
]


def canonical_hdr_path(value):
    """Resolve the asset directory migration recorded in older video sidecars."""
    root = Path(__file__).resolve().parents[2]
    path = Path(value)
    legacy = root / "data/lab/assets"
    if path.is_relative_to(legacy):
        migrated = root / "assets/lab" / path.relative_to(legacy)
        if not path.exists() and migrated.is_file():
            path = migrated
    return path.resolve()


def verify(directory, task_ids):
    reference = json.loads((directory / "A1_water_collection.json").read_text())[
        "effective_visuals"
    ]
    reports = []
    for task_id in task_ids:
        sidecar = directory / f"{task_id}_{NAMES[task_id]}.json"
        data = json.loads(sidecar.read_text())
        video = sidecar.with_suffix(".mp4")
        errors = []
        movement_evidence = {}

        def check(condition, message):
            if not condition:
                errors.append(message)

        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_streams",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )
        stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        check(stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p", "encoding")
        check([stream["width"], stream["height"]] == data["resolution"], "resolution")
        check(int(stream["nb_read_frames"]) == data["total_frames"], "frame count")
        check(hashlib.sha256(video.read_bytes()).hexdigest() == data["sha256"], "checksum")
        check(data["maximum_capture_physics_delta"] < 1e-12, "render advanced physics clock")
        check(not data["initial_geometry_errors"], "invalid initial physics")
        check(data["mean_adjacent_frame_pixel_change"] > 0, "static video")
        times = data["frame_simulation_timestamps"]
        check(
            len(times) == data["simulation_frames"]
            and all(b > a for a, b in zip(times, times[1:])),
            "simulation timestamps",
        )
        check(abs(times[-1] - data["simulation_seconds"]) < 1e-8, "final timestamp")
        for key in SHARED:
            actual, expected = data["effective_visuals"][key], reference[key]
            if key == "hdr_path":
                actual, expected = canonical_hdr_path(actual), canonical_hdr_path(expected)
            check(actual == expected, "A1 room profile: " + key)
        check(
            data["initial_state_and_targets"]["scene_id"] == "laboratory_equipment_faults_v2",
            "scene ID",
        )
        check(set(data["previews"]) == {"initial", "near_target", "final"}, "previews")
        for name in data["previews"].values():
            with Image.open(directory / name) as im:
                check(list(im.size) == data["resolution"], "preview dimensions")
        metrics = data["metrics"]
        trace = data["process_trace"]
        check(metrics["joint_success"], "diagnostic did not succeed")
        check(
            0 <= metrics["reaction_latency"] < metrics["episode_spec"]["max_latency"],
            "response window",
        )
        check(metrics["collision_count"] == 0, "arm collisions")
        if task_id in ("B2", "B3"):
            check(metrics["power_isolated"], "supply still connected")
            check(metrics["switch_angle_rad"] >= 0.6108, "disconnect did not open 35 degrees")
            check(
                any(
                    x["controls"].get("switch_robot_contact", False)
                    and x["controls"].get("switch_angle_rad", 0) > 0.1
                    for x in trace
                ),
                "missing physical lever contact",
            )
            wiring = trace[-1]["controls"]["wiring"]
            check(
                len(wiring) == 4
                and len({tuple(x["start"]) for x in wiring}) == 4
                and len({tuple(x["end"]) for x in wiring}) == 4,
                "four distinct connected leads",
            )
            if task_id == "B2":
                before_fault = [x for x in trace if not x["fault"]]
                check(
                    bool(before_fault)
                    and all(
                        x["controls"]["power_on"]
                        and abs(x["controls"]["switch_angle_rad"]) < 0.05
                        and not x["controls"]["switch_robot_contact"]
                        for x in before_fault
                    ),
                    "lamp must remain connected with untouched closed switch before fault",
                )
                phases = []
                for x in trace:
                    phase = x["phase"]
                    if not phases or phase != phases[-1]:
                        phases.append(phase)
                check(
                    phases == ["normal", "overheating", "isolated"],
                    "normal → overheating → isolated sequence",
                )
                colors = {"normal": "green", "overheating": "red", "isolated": "off"}
                check(
                    all(x["status_indicator"] == colors[x["phase"]] for x in trace),
                    "same status LED must change green → red → off",
                )
                check(
                    not any(x["short_indicator_red"] for x in trace),
                    "successful isolation must prevent the flashing short-circuit indicator",
                )
                check(any(x["warning_red"] for x in trace), "missing red overheat warning")
                failure_path = directory / "B2_thermal_short_circuit_failure.json"
                check(failure_path.exists(), "missing idle short-circuit diagnostic")
                if failure_path.exists():
                    failure = json.loads(failure_path.read_text())
                    fm = failure["metrics"]
                    check(
                        failure["trace"][-1]["visual"]["status_indicator"] == "flashing_red",
                        "short circuit must flash the same status LED red",
                    )
                    check(
                        failure["mode"] == "idle"
                        and not fm["joint_success"]
                        and fm["short_circuit"]
                        and "short_circuit" in fm["physical_failures"],
                        "idle branch must fail with a short circuit",
                    )
                    check(
                        abs(
                            fm["elapsed_time"]
                            - fm["event_time"]
                            - fm["episode_spec"]["max_latency"]
                        )
                        < 1e-8,
                        "short circuit must occur at the reaction deadline",
                    )
                    check(
                        all(not x["visual"]["short_indicator_red"] for x in failure["trace"][:-1])
                        and failure["trace"][-1]["visual"]["short_indicator_red"]
                        and failure["trace"][-1]["controls"]["power_on"],
                        "flashing red indication must follow timeout while supply remains connected",
                    )
                    with Image.open(directory / failure["previews"]["failure"]) as im:
                        check(list(im.size) == data["resolution"], "failure preview dimensions")
            if task_id == "B3":
                motion_trace = trace
                motion_source = "video_process_trace"
                if "robot_ee_position" not in trace[0]:
                    # Older recording schemas omit robot poses. Keep the independent
                    # PhysX diagnostic explicit; never present replay poses as frame data.
                    motion_path = directory / "B3_motor_stall_motion_diagnostic.json"
                    diagnostic = json.loads(motion_path.read_text())
                    motion_trace = diagnostic["trace"]
                    motion_source = motion_path.name
                    dm = diagnostic["metrics"]
                    check(
                        dm["episode_spec"] == metrics["episode_spec"]
                        and all(
                            abs(dm[k] - metrics[k]) < 1e-8
                            for k in ("event_time", "action_time", "switch_angle_rad")
                        ),
                        "motion diagnostic episode differs from recorded episode",
                    )
                    for frame in trace[1:]:
                        sample = min(motion_trace, key=lambda x: abs(x["time"] - frame["time"]))
                        check(
                            abs(sample["time"] - frame["time"]) < 1e-8
                            and sample["controls"]["power_on"] == frame["controls"]["power_on"]
                            and all(
                                abs(sample["controls"][k] - frame["controls"][k]) < 1e-6
                                for k in ("wheel_speed_rad_s", "switch_angle_rad")
                            ),
                            "motion diagnostic does not match recorded mechanism measurements",
                        )
                before_stop = [x for x in motion_trace if x["time"] < metrics["event_time"]]
                initial_position = motion_trace[0]["robot_ee_position"]
                initial_joints = motion_trace[0]["robot_joint_positions"]
                check(
                    bool(before_stop)
                    and all(
                        math.dist(x["robot_ee_position"], initial_position) < 0.002
                        and max(
                            abs(a - b) for a, b in zip(x["robot_joint_positions"], initial_joints)
                        )
                        < 0.01
                        and not x["controls"]["switch_robot_contact"]
                        for x in before_stop
                    ),
                    "robot moved from initial posture before observed stop",
                )
                first_move = next(
                    (
                        x["time"]
                        for x in motion_trace
                        if math.dist(x["robot_ee_position"], initial_position) >= 0.002
                    ),
                    None,
                )
                movement_evidence = {
                    "motion_evidence_source": motion_source,
                    "pre_stop_max_ee_drift_m": max(
                        (math.dist(x["robot_ee_position"], initial_position) for x in before_stop),
                        default=None,
                    ),
                    "motion_start_time": first_move,
                    "motion_start_latency": None
                    if first_move is None
                    else first_move - metrics["event_time"],
                    "motion_detection_threshold_m": 0.002,
                }
                check(
                    first_move is not None and 0 <= first_move - metrics["event_time"] <= 0.2,
                    "robot must begin moving immediately after observed stop",
                )
                check(metrics["rotor_previously_rotating"], "rotor never rotated")
                check(
                    any(abs(x["controls"].get("wheel_speed_rad_s", 0)) > 2 for x in trace),
                    "missing running rotor measurement",
                )
                check(
                    any(
                        x["controls"].get("power_on", False)
                        and abs(x["controls"].get("wheel_speed_rad_s", 0)) < 0.35
                        and x["time"] >= metrics["event_time"]
                        for x in trace
                    ),
                    "missing measured stop before power isolation",
                )
        else:
            check(metrics["retained_clearance_dwell"] >= 0.15 - 1e-9, "insufficient retained lift")
            check(not metrics["physical_failures"], "test-tube spill/drop")
            check(
                0 <= metrics["liquid_spilled_ml"] < metrics["liquid_spill_limit_ml"],
                "rescue must precede excessive liquid loss",
            )
            check(metrics["liquid_rim_clearance_mm"] > 0, "liquid remains above rim after rescue")
            check(abs(metrics["liquid_initial_ml"] - 31.361807746171916) < 0.01, "half-filled tube")
            check(trace[0]["tilt_degrees"] < 1, "tube must start upright")
            check(metrics["maximum_pre_event_ee_displacement"] < 0.015, "premature robot reach")
            check(
                metrics["event_time"] > metrics["disturbance_time"] + 1,
                "tilt must develop before reaction",
            )
            check(metrics["first_grasp_time"] > metrics["event_time"], "premature grasp")
            check(metrics["grasp_tilt_degrees"] >= 32, "grasp must begin at a substantial tilt")
            check(
                all(
                    abs(x["liquid_initial_ml"] - x["liquid_remaining_ml"] - x["liquid_spilled_ml"])
                    < 1e-7
                    for x in trace
                ),
                "liquid volume conservation",
            )
            check(
                any(c["view"] == "tube_liquid_rescue_closeup" for c in data["camera_cuts"]),
                "missing liquid close-up",
            )
            failure_path = directory / "B4_test_tube_rescue_failure.json"
            check(failure_path.exists(), "missing idle spill diagnostic")
            if failure_path.exists():
                failure = json.loads(failure_path.read_text())
                check(
                    failure["mode"] == "idle"
                    and "sample_spilled" in failure["metrics"]["physical_failures"]
                    and not failure["metrics"]["joint_success"],
                    "idle run must fail from cumulative liquid loss",
                )

        decode = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        check(
            decode.returncode == 0 and not decode.stderr.strip(),
            "full decode: " + decode.stderr.strip(),
        )
        reports.append(
            {
                "task_id": task_id,
                "file": video.name,
                "pass": not errors,
                "errors": errors,
                "reaction_latency": metrics["reaction_latency"],
                "window_s": metrics["episode_spec"]["max_latency"],
                "joint_success": metrics["joint_success"],
                "video_seconds": data["video_seconds"],
                "sha256": data["sha256"],
                **movement_evidence,
            }
        )
    result = {
        "scope": "One seed per task; actual-PhysX privileged diagnostics, not a multi-seed benchmark",
        "shared_A1_visual_profile": {k: reference[k] for k in SHARED},
        "videos": reports,
    }
    (directory / ("verification_" + "_".join(task_ids) + ".json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if any(not r["pass"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/lab/videos-v2"))
    parser.add_argument("--tasks", nargs="+", choices=list(NAMES), default=list(NAMES))
    args = parser.parse_args()
    verify(args.directory, args.tasks)
