"""Audit full-clock A5/A6 recordings and generate their local review page."""

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path

import numpy as np

NAMES = {"A5": "photographic_film_development", "A6": "heat_shock_ice_transfer"}


def verify(root):
    rows = []
    for mode in ("success", "failure"):
        directory = root / mode
        manifest = json.loads((directory / "manifest.json").read_text())
        for task_id, name in NAMES.items():
            p = directory / f"{task_id}_{name}.json"
            data = json.loads(p.read_text())
            m = data["metrics"]
            trace = data["process_trace"]
            video = p.with_suffix(".mp4")
            errors = []

            def check(ok, message):
                if not ok:
                    errors.append(message)

            media = json.loads(
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
            stream = next(x for x in media["streams"] if x["codec_type"] == "video")
            check(
                stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p", "video encoding"
            )
            check([stream["width"], stream["height"]] == data["resolution"], "resolution")
            check(int(stream["nb_read_frames"]) == data["total_frames"], "frame count")
            check(abs(float(stream["duration"]) - data["video_seconds"]) < 0.01, "media duration")
            decoded = subprocess.run(
                ["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"],
                capture_output=True,
            )
            check(decoded.returncode == 0 and not decoded.stderr, "complete decode")
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            check(digest == data["sha256"], "sidecar checksum")
            entry = next(x for x in manifest["videos"] if x["file"] == video.name)
            check(entry["sha256"] == digest, "manifest checksum")
            if "source_manifest" in entry:
                run_manifest = json.loads((directory / entry["source_manifest"]).read_text())
                run_entry = next(x for x in run_manifest["videos"] if x["file"] == video.name)
                check(run_entry["sha256"] == digest, "recording provenance checksum")
            check(not data["initial_geometry_errors"], "initial geometry")
            check(data["maximum_capture_physics_delta"] < 1e-12, "capture advanced physics")
            times = data["frame_simulation_timestamps"]
            check(
                len(times) == data["simulation_frames"]
                and all(b > a for a, b in zip(times, times[1:])),
                "simulation timestamps",
            )
            check(
                all(
                    abs((b - a) - 1 / data["fps"]) < data["physics_dt"] + 1e-8
                    for a, b in zip(times[:-2], times[1:-1])
                ),
                "compressed or skipped waiting interval",
            )
            check(data["episode_terminated"], "episode did not terminate")
            check(m["collision_count"] == 0 and not m["drops"], "arm collision or drop")
            if task_id == "A5":
                start = m.get("source_entry_time")
                check(start is not None and start > 0, "missing developer loading event")
                check(
                    not trace[0]["controls"]["source_immersed"], "reel did not start on dry stand"
                )
                if start is not None:
                    check(
                        any(x["controls"].get("source_fully_immersed") for x in trace),
                        "no full developer immersion",
                    )
                    check(
                        all(
                            abs(x["remaining"] - max(0, 12 - max(0, x["time"] - start))) < 1e-7
                            for x in trace
                        ),
                        "countdown is not anchored to developer entry",
                    )
            else:
                start = 0.0
                check(trace[0]["controls"]["source_immersed"], "sample did not start immersed")
            deadline = m.get("timing_deadline", m["prescribed_duration_s"])
            check(
                m["prescribed_duration_s"] == (12 if task_id == "A5" else 30), "prescribed duration"
            )
            if task_id == "A6":
                check(data["simulation_seconds"] >= 30 - 1e-8, "30-second exposure not recorded")
            if mode == "success":
                check(
                    m["joint_success"]
                    and not m["physical_failures"]
                    and not m["temporal_violations"],
                    "unsuccessful transfer",
                )
                check(m["grasp_observed"], "no measured bilateral grasp")
                expected = m["developer_lift_time"] if task_id == "A5" else m["source_exit_time"]
                check(m["action_time"] == expected, "wrong timing event")
                check(
                    -(0 if task_id == "A5" else m["timing_tolerance_s"]) - 1e-8
                    <= expected - deadline
                    <= m["timing_tolerance_s"] + 1e-8,
                    "missed timing window",
                )
                check(
                    m["destination_immersion_time"] >= m["source_exit_time"],
                    "immersion preceded source exit",
                )
                c = m["final_controls"]
                check(
                    c["destination_immersed"]
                    and c["destination_supported"]
                    and c["gripper_width"] > 0.065
                    and not c["sample_grasped"],
                    "sample was not released on destination support",
                )
                check(
                    m["destination_retention_s"] >= 0.25 - 1e-9, "insufficient supported retention"
                )
                check(trace[-1]["phase"] == "complete", "missing final success cue")
            else:
                check(not m["joint_success"] and m["action_time"] is None, "idle falsely succeeded")
                reason = "overdevelopment" if task_id == "A5" else "heat_exposure_exceeded"
                check(reason in m["temporal_violations"], "wrong timeout consequence")
                check(trace[-1]["phase"] == "failed", "missing failure cue")
                held_trace = (
                    [x for x in trace if x["time"] >= start + 0.5]
                    if task_id == "A5" and start is not None
                    else trace
                )
                check(
                    all(x["controls"]["source_immersed"] for x in held_trace),
                    "idle sample left its bath",
                )
                initial = np.asarray(held_trace[0]["robot_joint_positions"])
                check(
                    all(
                        np.max(np.abs(np.asarray(x["robot_joint_positions"]) - initial)) < 0.01
                        for x in held_trace
                    ),
                    "idle arm moved",
                )
                if task_id == "A5":
                    check(
                        data["policy"] == "loaded_hold_failure_diagnostic",
                        "wrong full-cycle failure policy",
                    )
            for preview in data["previews"].values():
                check((directory / preview).exists(), "missing preview")
            rows.append(
                dict(
                    task_id=task_id,
                    mode=mode,
                    passed=not errors,
                    errors=errors,
                    video=str(video.relative_to(root)),
                    poster=str((directory / data["previews"]["final"]).relative_to(root)),
                    sha256=digest,
                    timing_event=m["timing_event"],
                    action_time=m["action_time"],
                    target_time=deadline,
                    source_entry_time=start,
                    tolerance=m["timing_tolerance_s"],
                    simulation_seconds=data["simulation_seconds"],
                )
            )
    result = dict(
        passed=all(r["passed"] for r in rows),
        scope="easy / seed 0; two controller success episodes, A5 hold-after-loading timeout and A6 idle timeout",
        limitations="Rigid-body contact and geometric immersion; symbolic chemistry/biology. A5 uses a configurable 12-second demonstration, not a processing recipe. A6 uses the requested 30-second simulation exposure; the exit tolerance is the level's TOLERANCE_S (tasks/temporal_window/bath_transfer.py).",
        recordings=rows,
    )
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    cards = []
    for task_id, name in NAMES.items():
        videos = []
        for r in rows:
            if r["task_id"] != task_id:
                continue
            timing = (
                f"Measured interval: {r['action_time'] - r['source_entry_time']:.4f} s"
                if r["action_time"] is not None
                else "Loaded into developer, then held past the deadline"
                if task_id == "A5"
                else "No intervention; deadline exceeded"
            )
            if task_id == "A5":
                timing += f" · Developer entry at video time {r['source_entry_time']:.3f} s"
            videos.append(
                f'<article><h3>{r["mode"].title()}</h3><video controls preload="none" poster="{r["poster"]}" src="{r["video"]}"></video><p>{timing}</p><a href="{r["video"]}">MP4</a> · <a href="{r["video"].replace(".mp4", ".json")}">Physics trace</a></article>'
            )
        cards.append(
            f'<section><h2>{task_id} · {html.escape(name.replace("_", " ").title())}</h2><div class="pair">{"".join(videos)}</div></section>'
        )
    (root / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A5–A6 Timed Bath Transfers</title><style>body{font:16px system-ui;background:#f3f5f7;color:#173043;max-width:1300px;margin:32px auto;padding:0 24px}p{line-height:1.6}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;padding:16px;border-radius:12px}video{width:100%}section{margin:32px 0}a{color:#176ca1}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>A5–A6 Timed Bath Transfers</h1><p>Actual Isaac Sim physics · easy / seed 0 · full simulation clock, no time compression. A4 titration is retained.</p><p>A5 starts with a loaded film reel on a dry stand. The robot grips its center spindle and loads the developer; full immersion starts the 12-second demonstration countdown. It keeps the reel in the developer for the full countdown, lifts it out once the countdown has ended, and transfers it into the stop bath. A6 measures warm-bath exit at 30 seconds, followed by ice-bucket placement. Both panels show only countdown digits; the single orange-capped tube is enlarged 1.5 times for visibility, and the warm bath uses a refractive water volume with surface ripples. The robot may grasp in advance while keeping the sample immersed.</p><p>Authored apparatus, loaded film reel with an MIT-licensed center column, and capped tube; imported laboratory background. Contact-backed rigid manipulation with geometric bath immersion, not a chemical or biological simulator. <a href="verification.json">Verification report</a></p>"""
        + "".join(cards)
        + "</html>\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = verify(parser.parse_args().directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
