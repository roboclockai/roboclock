"""Audit B5–B7 success/idle recordings and build a local review page."""

import argparse
import hashlib
import html
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from roboclock.envs.lab.safety_audio import RATE, samples
from roboclock.resources import PACKAGE_ROOT

NAMES = {
    "B5": "freezer_door_rescue",
    "B6": "vacuum_bumping_relief",
    "B7": "waste_carboy_overflow",
}
FAILURES = {
    "B5": "sample_thawed_scenario",
    "B6": "vacuum_bumping",
    "B7": "hazardous_waste_overflow",
}


def verify(root):
    reports = []
    # Historical flat recordings keep the manifest name they were rendered with.
    group = "B5_B7" if (root / "manifest_B5_B7.json").exists() else "B5_B8"
    flat = (root / f"manifest_{group}.json").exists()
    for mode in ("success", "failure"):
        directory = root if flat else root / mode
        suffix = "_failure" if mode == "failure" else ""
        manifest_paths = (
            [directory / f"manifest_{group}{suffix}.json"]
            if flat
            else sorted(directory.glob("manifest*.json"))
        )
        manifests = [json.loads(path.read_text()) for path in manifest_paths]
        manifest = dict(manifests[0])
        manifest["videos"] = [entry for item in manifests for entry in item["videos"]]
        asset_errors = []
        bundled = {
            name: digest
            for name, digest in manifest["asset_sha256"].items()
            if name.startswith("envs/lab/assets/safety_equipment/")
        }
        if not bundled:
            asset_errors.append("missing bundled equipment provenance")
        for name, digest in bundled.items():
            path = PACKAGE_ROOT / name
            # Preserve the descriptive asset metadata as it existed at recording
            # time; subsequent task revisions do not rewrite historical inputs.
            if name.rsplit("/", 1)[-1] in ("selection.json", "provenance.json"):
                snapshot = manifest.get("asset_metadata_snapshots", {}).get(name)
                if snapshot:
                    path = directory / snapshot
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                asset_errors.append("asset changed since recording: " + name)
        for task_id, name in NAMES.items():
            sidecar = directory / f"{task_id}_{name}{suffix if flat else ''}.json"
            data = json.loads(sidecar.read_text())
            video = sidecar.with_suffix(".mp4")
            errors = asset_errors.copy()

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
            audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
            check(
                stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p", "video encoding"
            )
            check(
                audio["codec_name"] == "aac" and int(audio["sample_rate"]) == RATE, "audio encoding"
            )
            check([stream["width"], stream["height"]] == data["resolution"], "resolution")
            check(int(stream["nb_read_frames"]) == data["total_frames"], "frame count")
            check(abs(float(stream["duration"]) - data["video_seconds"]) < 0.01, "duration")
            checksum = hashlib.sha256(video.read_bytes()).hexdigest()
            check(checksum == data["sha256"], "video checksum")
            entry = next(x for x in manifest["videos"] if x["file"] == video.name)
            check(entry["sha256"] == checksum, "manifest checksum")
            decode = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(video),
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
            )
            check(decode.returncode == 0 and not decode.stderr, "full video/audio decode")
            check(data["maximum_capture_physics_delta"] < 1e-12, "capture advanced physics")
            check(not data["initial_geometry_errors"], "initial physics errors")
            check(data["mean_adjacent_frame_pixel_change"] > 0, "static recording")
            times = data["frame_simulation_timestamps"]
            check(
                len(times) == data["simulation_frames"]
                and all(b > a for a, b in zip(times, times[1:])),
                "frame timestamps",
            )
            check(abs(times[-1] - data["simulation_seconds"]) < 1e-8, "final timestamp")
            check(data["end_hold_frames"] == 2 * data["fps"], "two-second result hold")
            check(set(data["previews"]) == {"initial", "near_target", "final"}, "preview set")
            for preview in data["previews"].values():
                with Image.open(directory / preview) as im:
                    check(list(im.size) == data["resolution"], "preview dimensions")

            metrics, trace = data["metrics"], data["process_trace"]
            event, action = metrics["event_time"], metrics["action_time"]
            window = metrics["episode_spec"]["max_latency"]
            check(metrics["collision_count"] == 0, "arm collision")
            check(data["episode_terminated"], "episode not terminal")
            before = [x for x in trace if x["time"] < event]
            initial_ee = np.array(trace[0]["robot_ee_position"])
            initial_joints = np.array(trace[0]["robot_joint_positions"])
            pre_drift = max(
                float(np.linalg.norm(np.array(x["robot_ee_position"]) - initial_ee)) for x in before
            )
            pre_joint_drift = max(
                float(np.max(np.abs(np.array(x["robot_joint_positions"]) - initial_joints)))
                for x in before
            )
            check(pre_drift < 0.002 and pre_joint_drift < 0.01, "robot anticipated the event")
            check(metrics["pre_event_max_ee_drift_m"] < 0.002, "subframe pre-event reaching")
            first_move = next(
                (
                    x["time"]
                    for x in trace
                    if np.linalg.norm(np.array(x["robot_ee_position"]) - initial_ee) >= 0.002
                ),
                None,
            )
            if mode == "success":
                check(
                    first_move is not None and 0 <= first_move - event <= 0.2,
                    "response must start after observed alarm",
                )
            else:
                check(first_move is None, "idle robot moved")
            if task_id == "B5":
                released = metrics["fault_release_time"]
                check(
                    released is not None and released < event,
                    "alarm must follow internal catch release",
                )
                check(
                    abs(trace[0]["controls"]["angle_rad"]) < np.deg2rad(1),
                    "door must start closed",
                )
                check(
                    all(
                        abs(x["controls"]["angle_rad"]) < np.deg2rad(1)
                        for x in trace
                        if x["time"] < released
                    ),
                    "door opened before catch release",
                )
                check(
                    any(
                        x["controls"]["fault_released"] and x["controls"]["opening_reached"]
                        for x in trace
                    ),
                    "no measured opening after catch release",
                )
            check(
                bool(before)
                and all(
                    not x["controls"]["secured"] and not x["controls"]["robot_contact"]
                    for x in before
                ),
                "pre-alarm interference",
            )
            check(any(x["alarm"] for x in trace), "missing visible warning")
            if mode == "success":
                check(
                    metrics["joint_success"] and metrics["intervention_secured"],
                    "intervention failed",
                )
                check(action is not None and 0 <= action - event < window, "late intervention")
                check(metrics["robot_contact_observed"], "no robot contact")
                check(metrics["secured_dwell"] >= 0.20 - 1e-9, "unstable closure")
                check(
                    not metrics["physical_failures"] and not metrics["temporal_violations"],
                    "failure on success branch",
                )
                check(trace[-1]["completed"] and not trace[-1]["failed"], "final success cue")
                controls = metrics["final_controls"]
                if task_id == "B5":
                    check(abs(controls["angle_rad"]) < np.deg2rad(2), "door not closed")
                else:
                    check(controls["travel_m"] >= 0.016, "plunger stroke incomplete")
                if task_id == "B5":
                    check(metrics["bilateral_grasp_observed"], "freezer was not grasped")
            else:
                check(
                    not metrics["joint_success"] and not metrics["intervention_secured"],
                    "idle falsely succeeded",
                )
                check(action is None, "idle recorded intervention")
                check(
                    FAILURES[task_id] in metrics["physical_failures"], "wrong timeout consequence"
                )
                check(
                    abs(metrics["elapsed_time"] - event - window) < 1e-8, "wrong failure deadline"
                )
                check(
                    trace[-1]["failed"] and not any(x["failed"] for x in trace[:-1]),
                    "failure cue timing",
                )
            if task_id == "B7":
                levels = [x["volume_ml"] for x in trace]
                check(
                    all(b >= a - 1e-8 for a, b in zip(levels, levels[1:])), "waste volume reversed"
                )
                if mode == "success":
                    secured = [x for x in trace if x["secured"]]
                    check(
                        bool(secured) and all(x["flow_ml_s"] == 0 for x in secured),
                        "inflow did not stop",
                    )
                    check(
                        max(x["volume_ml"] for x in secured) - min(x["volume_ml"] for x in secured)
                        < 1e-8,
                        "volume rose after clamp",
                    )
                    check(metrics["volume_ml"] < 10000, "successful response overflowed")
                else:
                    check(metrics["volume_ml"] > 10000, "missing overflow")

            cues = data["audio"]
            check(
                cues["start_sim_time"] == event and cues["stop_sim_time"] == action,
                "audio event alignment",
            )
            check(
                cues["failure_sim_time"]
                == (metrics["elapsed_time"] if mode == "failure" else None),
                "failure audio alignment",
            )
            with wave.open(str(directory / cues["wav"]), "rb") as wav:
                check(
                    wav.getframerate() == RATE
                    and wav.getnchannels() == 1
                    and wav.getsampwidth() == 2,
                    "WAV format",
                )
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            expected = np.round(
                np.clip(
                    samples(
                        np.arange(len(pcm)) / RATE, task_id, event, action, cues["failure_sim_time"]
                    ),
                    -1,
                    1,
                )
                * 32767
            ).astype("<i2")
            check(np.array_equal(pcm, expected), "WAV does not match event schedule")
            check(np.any(pcm != 0), "silent alarm")
            reports.append(
                {
                    "task_id": task_id,
                    "mode": mode,
                    "passed": not errors,
                    "errors": errors,
                    "video": str(video.relative_to(root)),
                    "sha256": checksum,
                    "poster": str(
                        (
                            directory
                            / data["previews"]["initial" if mode == "success" else "final"]
                        ).relative_to(root)
                    ),
                    "instrument_preview": str(
                        (directory / data["previews"]["initial"]).relative_to(root)
                    ),
                    "event_time": event,
                    "reaction_latency": metrics["reaction_latency"],
                    "pre_event_max_ee_drift_m": pre_drift,
                    "pre_event_max_joint_drift_rad": pre_joint_drift,
                    "motion_start_time": first_move,
                    "fault_release_time": metrics["fault_release_time"],
                    "deadline_seconds": window,
                    "video_seconds": data["video_seconds"],
                    "physical_failures": metrics["physical_failures"],
                }
            )
    result = {
        "passed": all(x["passed"] for x in reports),
        "scope": "easy / seed 0, three privileged-controller episodes and three idle timeout episodes; not the full benchmark matrix",
        "checks": "complete media decode, hashes, timestamps, absence of pre-event reaching/rotation, closed-to-open ordering, contacts, mechanical closure/stroke, deadlines, waste flow and event-aligned audio",
        "limitations": "Rigid-body actuation with symbolic thermal/pressure states and volume balance; spill overlays are visual consequences, not fluid simulation.",
        "recordings": reports,
    }
    report_name = f"verification_{group}.json" if flat else "verification.json"
    (root / report_name).write_text(json.dumps(result, indent=2) + "\n")
    cards = []
    for task_id, name in NAMES.items():
        cells = []
        for row in (x for x in reports if x["task_id"] == task_id):
            latency = row["reaction_latency"]
            detail = (
                f"Intervention completed: {latency:.3f} s after alarm"
                if latency is not None
                else f"Timeout: {row['deadline_seconds']:g} s"
            )
            motion = (
                f"First visible robot movement: {row['motion_start_time'] - row['event_time']:.3f} s after alarm."
                if row["motion_start_time"] is not None
                else "Robot remains at its initial pose throughout."
            )
            cells.append(
                f'<article><h3>{row["mode"].title()} · {detail}</h3><video controls preload="none" poster="{row["poster"]}" src="{row["video"]}"></video><p>{motion}</p><p><a href="{row["video"]}">MP4</a> · <a href="{row["video"].replace(".mp4", ".json")}">Physics trace</a> · <a href="{row["instrument_preview"]}">Instrument view</a></p></article>'
            )
        cards.append(
            f'<section><h2>{task_id} · {html.escape(name.replace("_", " ").title())}</h2><div class="pair">{"".join(cells)}</div></section>'
        )
    (root / (f"index_{group}.html" if flat else "index.html")).write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>B5–B7 Laboratory Safety Tasks</title><style>body{font:16px system-ui;margin:32px auto;max-width:1300px;padding:0 24px;background:#f3f5f7;color:#173043}h1{margin-bottom:8px}p{line-height:1.6}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{padding:16px;background:white;border-radius:12px}video{width:100%;border-radius:6px}a{color:#176ca1}section{margin:32px 0}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>B5–B7 Laboratory Safety Tasks</h1><p>Actual Isaac Sim physics · easy / seed 0 · synchronized alarm audio.<br>Privileged controller success and idle timeout diagnostics. These are representative episodes, not a full benchmark evaluation.</p><p><strong>Event-response revision:</strong> the robot holds its original posture until the observed alarm. B5 starts closed; an internal catch fault lets the door spring open, and the observed opening triggers the alarm.</p><p>Instruments: cabinet-mounted temperature readout, circular absolute-pressure gauge and wired level-alarm module. No freestanding instructional signs. Assets: <a href="https://opengameart.org/content/lab-assets">MilkAndBanana Lab Assets (CC0)</a>; authored freezer and actuation fixtures. Temperature/pressure and spill consequences are scenario models. <a href="verification.json">Verification report</a></p>"""
        + "".join(cards)
        + "</html>\n"
    )
    if flat:
        index = root / f"index_{group}.html"
        index.write_text(
            index.read_text().replace('href="verification.json"', f'href="{report_name}"')
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("results/lab/videos-v2"))
    args = parser.parse_args()
    result = verify(args.directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
