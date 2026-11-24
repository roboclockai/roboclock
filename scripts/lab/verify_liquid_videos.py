"""Verify liquid-task video integrity separately from diagnostic policy outcomes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image


def verify(directory, tasks=("A1", "A2", "A3", "A4")):
    rows = []
    for task in tasks:
        candidates = list(directory.glob(f"{task}_*.json"))
        candidates = [p for p in candidates if p.with_suffix(".mp4").exists()]
        if len(candidates) != 1:
            raise ValueError(f"{task}: expected one video sidecar, found {candidates}")
        sidecar = candidates[0]
        data = json.loads(sidecar.read_text())
        video = sidecar.with_suffix(".mp4")
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )["streams"][0]
        errors = []

        def check(condition, message):
            if not condition:
                errors.append(message)

        check(
            probe["codec_name"] == "h264" and probe["pix_fmt"] == "yuv420p",
            "unexpected codec/pixel format",
        )
        check([probe["width"], probe["height"]] == data["resolution"], "resolution mismatch")
        check(int(probe["nb_read_frames"]) == data["total_frames"], "decoded frame count mismatch")
        check(hashlib.sha256(video.read_bytes()).hexdigest() == data["sha256"], "checksum mismatch")
        check(data["maximum_capture_physics_delta"] < 1e-12, "capture advanced physics")
        times = data["frame_simulation_timestamps"]
        check(len(times) == data["simulation_frames"], "timestamp count mismatch")
        check(all(b > a for a, b in zip(times, times[1:])), "non-monotonic simulation timestamps")
        check(
            abs(times[-1] - data["simulation_seconds"]) < 1e-8,
            "last frame does not match final simulation time",
        )
        check(data["mean_adjacent_frame_pixel_change"] > 0, "no changing RGB frames")
        decode = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        check(
            decode.returncode == 0 and not decode.stderr.strip(),
            "full decode errors: " + decode.stderr.strip(),
        )
        for name in data.get("previews", {}).values():
            with Image.open(directory / name) as im:
                check(list(im.size) == data["resolution"], "preview dimensions mismatch")
        metrics = data["metrics"]
        if data.get("initial_state_and_targets", {}).get("robot") == "dual_arx_x5":
            check(
                all(len(row["robot_joint_positions"]) == 16 for row in data["process_trace"]),
                "Dual ARX X5 requires both eight-joint arm traces",
            )
            if task == "A1":
                check(
                    data["process_trace"][0]["volume_ml"] == 0, "A1 must start with an empty beaker"
                )
        if task == "A1" and "faucet_opened_at" in metrics and metrics["joint_success"]:
            opened, full, closed = (
                metrics[key]
                for key in ("faucet_opened_at", "faucet_fully_open_at", "faucet_closed_at")
            )
            check(
                all(value is not None for value in (opened, full, closed))
                and 0 < opened < full < closed,
                "missing ordered faucet cycle",
            )
            trace = data["process_trace"]
            check(
                trace[0]["opening"] < 0.04 and trace[0]["flow_ml_s"] == 0,
                "faucet must start closed without flow",
            )
            check(
                any(row["opening"] >= 0.96 and row["flow_ml_s"] > 0 for row in trace),
                "no fully open flowing interval",
            )
            check(
                trace[-1]["opening"] < 0.04 and trace[-1]["flow_ml_s"] == 0,
                "faucet must finish closed without flow",
            )
            check(trace[-1]["volume_ml"] > trace[0]["volume_ml"], "no water collected")
            check("faucet_open" in data["previews"], "missing open-faucet preview")
        rows.append(
            {
                "task_id": task,
                "file": video.name,
                "integrity_pass": not errors,
                "errors": errors,
                "decoded_frames": int(probe["nb_read_frames"]),
                "resolution": data["resolution"],
                "fps": data["fps"],
                "simulation_seconds": data["simulation_seconds"],
                "video_seconds": data["video_seconds"],
                "physical_success": metrics["manipulation_success"],
                "temporal_success": metrics["temporal_success"],
                "joint_success": metrics["joint_success"],
                "failures": metrics["physical_failures"],
                "temporal_violations": metrics["temporal_violations"],
                "sha256": data["sha256"],
            }
        )
    result = {
        "scope": "Selected recorded actual-PhysX episodes; video integrity and policy outcomes are reported separately.",
        "all_integrity_checks_passed": all(r["integrity_pass"] for r in rows),
        "videos": rows,
    }
    suffix = "" if set(tasks) == {"A1", "A2", "A3", "A4"} else "_" + "_".join(tasks)
    (directory / f"verification{suffix}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["all_integrity_checks_passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--tasks", nargs="+", choices=["A1", "A2", "A3", "A4"], default=["A1", "A2", "A3", "A4"]
    )
    args = parser.parse_args()
    verify(args.directory, args.tasks)
