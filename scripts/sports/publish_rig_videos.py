"""Publish the RoboDojo-rig sports recordings into data/sports/videos.

Each task delivers the third-person overview ``<task>.mp4`` (1280x960, 25 fps) and
the three RoboDojo rig cameras ``evidence/<task>/{head,left_wrist,right_wrist}.mp4``
(640x480, 25 fps, frame-synchronized with the overview: equal frame counts, checked
here by decoding every stream).

Inputs are fresh recording-driver outputs, one directory per task ID::

    PATH=<ffmpeg dir>:$PATH python -m roboclock.envs.sports --task B4 --difficulty easy \
        --seed 0 --video --output <recordings>/B4

and, optionally, ``roboclock eval --env sports --policy oracle --oracle`` outputs
(``<evals>/<mode>/<ID>/episodes.jsonl``) whose outcomes are recorded beside each
video. The previous collection is moved, unchanged, to
``data/sports/videos/superseded/<date>/`` before the new files are copied.

    python scripts/sports/publish_rig_videos.py --recordings R --evals E \
        --revision <commit> [--supersede 2026-10-31]
"""

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

from roboclock.envs import robodojo_rig

VIDEOS = Path("data/sports/videos")
TASKS = {
    "A1": ("chess", "Chess"),
    "A2": ("taiko", "Taiko"),
    "A3": ("basketball", "Basketball"),
    "B1": ("moles", "Moles"),
    "B2": ("boxing", "Boxing"),
    "B3": ("bench_press", "Bench Press"),
    "B4": ("freeze", "Freeze"),
    "C1": ("pool", "Pool"),
    "C2": ("cups", "Cups"),
    "C3": ("ball_service", "Ball Service"),
}
EVIDENCE = (
    "config.json",
    "initial.png",
    "final.png",
    "initial_state.json",
    "final_state.json",
    "rig_views.json",
    *(f"{view}{suffix}" for view in ("head", "left_wrist", "right_wrist")
      for suffix in (".mp4", "_initial.png", "_final.png")),
)
RIG_VIDEOS = ("head", "left_wrist", "right_wrist")
FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "/tmp/ffmpeg/bin/ffmpeg")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream(path):
    """(width, height, fps, decoded frame count), duration: every frame is decoded."""
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    video = re.search(r"Video: .*?, (\d+)x(\d+).*?, ([\d.]+) fps", out)
    frames = int(re.findall(r"frame=\s*(\d+)", out)[-1])
    clock = re.findall(r"time=(\d+):(\d+):([\d.]+)", out)[-1]
    duration = int(clock[0]) * 3600 + int(clock[1]) * 60 + float(clock[2])
    info = dict(width=int(video[1]), height=int(video[2]), fps=float(video[3]), frames=frames)
    return info, duration


def outcome(result):
    metrics = result.get("metrics", result)
    keys = ("success", "physical_success", "temporal_success")
    if all(k in result for k in keys):
        return {k: bool(result[k]) for k in keys}
    return dict(
        success=bool(metrics["joint_success"]),
        physical_success=bool(metrics["task_success"]),
        temporal_success=bool(metrics["temporal_success"]),
    )


def eval_row(root, mode, task):
    path = Path(root) / mode / task / "episodes.jsonl" if root else None
    if path is None or not path.exists():
        return None
    row = json.loads(path.read_text().splitlines()[0])
    native = row.get("native_result", {})
    return dict(
        mode=mode,
        physical=row["task_success"],
        temporal=row["temporal_success"],
        joint=row["joint_success"],
        evaluation_end=native.get("evaluation_end", native.get("sim_time")),
        observation_rendered=row.get("observation_rendered"),
        temporal_violations=row.get("temporal_violations"),
        physical_failures=row.get("physical_failures"),
    )


def supersede(label, tasks=None):
    target = VIDEOS / "superseded" / label
    if target.exists():
        raise SystemExit(f"{target} exists; refusing to overwrite superseded media")
    (target / "evidence").mkdir(parents=True)
    for task, (name, _) in TASKS.items():
        if tasks is not None and task not in tasks:
            continue
        for suffix in (".mp4", ".json"):
            source = VIDEOS / f"{task}_{name}{suffix}"
            if source.exists():
                shutil.move(source, target / source.name)
        source = VIDEOS / "evidence" / f"{task}_{name}"
        if source.exists():
            shutil.move(source, target / "evidence" / source.name)
    for extra in ("manifest.json", "index.html", "README.md"):
        if (VIDEOS / extra).exists():
            shutil.copy2(VIDEOS / extra, target / extra)
    runtime = VIDEOS / "evidence/runtime"
    if runtime.exists():
        shutil.move(runtime, target / "evidence/runtime")
    return target


def runtime_record(path):
    import importlib.metadata as md
    import platform
    import sys

    packages = {d.metadata["Name"]: d.version for d in md.distributions()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(
                python=sys.version,
                executable=sys.executable,
                platform=platform.platform(),
                packages=dict(sorted(packages.items(), key=lambda kv: kv[0].lower())),
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--evals", type=Path)
    parser.add_argument("--revision", required=True, help="Commit the recordings ran from")
    parser.add_argument("--supersede", help="Move the current collection to superseded/<label>")
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    missing = [t for t in tasks if not (args.recordings / t / "result.json").exists()]
    if missing:
        raise SystemExit(f"No recording result for {missing}")
    partial = set(tasks) != set(TASKS)
    previous = []
    if partial and (VIDEOS / "manifest.json").exists():
        old_manifest = json.loads((VIDEOS / "manifest.json").read_text())
        for e in old_manifest.get("videos", []):
            e.setdefault("recording_revision", old_manifest.get("recording_revision"))
            previous.append(e)
    superseded = (
        supersede(args.supersede, tasks if partial else None) if args.supersede else None
    )
    entries, cards = [], []
    for task in tasks:
        name, title = TASKS[task]
        raw = args.recordings / task
        result = json.loads((raw / "result.json").read_text())
        overview = [p for p in raw.glob("*.mp4") if p.stem not in RIG_VIDEOS]
        if len(overview) != 1 or not all((raw / f"{v}.mp4").exists() for v in RIG_VIDEOS):
            raise SystemExit(f"{task}: expected one overview MP4 and {RIG_VIDEOS} in {raw}")
        video, result_file = f"{task}_{name}.mp4", f"{task}_{name}.json"
        shutil.copy2(overview[0], VIDEOS / video)
        shutil.copy2(raw / "result.json", VIDEOS / result_file)
        evidence = VIDEOS / "evidence" / f"{task}_{name}"
        evidence.mkdir(parents=True, exist_ok=True)
        for item in EVIDENCE:
            if (raw / item).exists():
                shutil.copy2(raw / item, evidence / item)
        if not (evidence / "config.json").exists() and "configuration" in result:
            # The arcade driver (A2, B1) stores its configuration inside result.json.
            (evidence / "config.json").write_text(
                json.dumps(result["configuration"], indent=2) + "\n"
            )
        info, duration = stream(VIDEOS / video)
        rig = {view: stream(evidence / f"{view}.mp4") for view in RIG_VIDEOS}
        views = json.loads((evidence / "rig_views.json").read_text())
        if (info["width"], info["height"], info["fps"]) != (1280, 960, robodojo_rig.VIDEO_FPS):
            raise SystemExit(f"{task}: overview stream {info}")
        for view, (head_info, _) in rig.items():
            if (head_info["width"], head_info["height"], head_info["fps"]) != (
                640, 480, robodojo_rig.VIDEO_FPS
            ) or head_info["frames"] != info["frames"] or views["frames"] != info["frames"]:
                raise SystemExit(f"{task}: {view} stream {head_info} vs overview {info}")
        state = outcome(result)
        sim = result.get("configuration", {}).get("sim", {})
        physics_hz = result.get("physics_hz") or (
            round(1 / sim["physics_dt"]) if "physics_dt" in sim else None
        )
        control_hz = result.get("control_hz") or (
            round(1 / (sim["physics_dt"] * sim["control_decimation"])) if physics_hz else None
        )
        entry = dict(
            task=task,
            name=name,
            file=video,
            result=result_file,
            evidence=f"evidence/{task}_{name}",
            configuration=f"evidence/{task}_{name}/config.json",
            sha256=digest(VIDEOS / video),
            bytes=(VIDEOS / video).stat().st_size,
            rig_videos={
                view: dict(
                    file=f"evidence/{task}_{name}/{view}.mp4",
                    sha256=digest(evidence / f"{view}.mp4"),
                    bytes=(evidence / f"{view}.mp4").stat().st_size,
                    stream=rig[view][0],
                    duration_seconds=round(rig[view][1], 6),
                )
                for view in RIG_VIDEOS
            },
            frames_synchronized=True,
            rig_check=result.get("robodojo_rig", {}).get("rig_check"),
            renderer=result.get("robodojo_rig", {}).get("renderer"),
            head_view=result.get("head_view"),
            full_episode=bool(result.get("full_episode", True)),
            **state,
            difficulty=result.get("difficulty", "easy"),
            seed=result.get("seed", 0),
            simulator=result.get("simulator"),
            physics_hz=physics_hz,
            control_hz=control_hz,
            stream=dict(info),
            duration_seconds=round(duration, 6),
            unified_eval=[
                row for row in (
                    eval_row(args.evals, "norender", task), eval_row(args.evals, "render", task)
                ) if row
            ],
        )
        entries.append(entry)
    for e in entries:
        e["recording_revision"] = args.revision
    # A partial publish keeps the other tasks' current entries unchanged.
    entries = sorted(
        [e for e in previous if e["task"] not in tasks] + entries, key=lambda e: e["task"]
    )
    for entry in entries:
        task, name = entry["task"], entry["name"]
        title = TASKS[task][1]
        verdict = "PASS" if entry["success"] else "FAIL"
        cards.append(
            f'<article><h2>{task} · {html.escape(title)}</h2>'
            f'<video controls preload="metadata" poster="evidence/{task}_{name}/initial.png" '
            f'src="{entry["file"]}"></video>'
            + "".join(
                f'<p>{label}</p><video controls preload="metadata" muted '
                f'poster="evidence/{task}_{name}/{view}_initial.png" '
                f'src="evidence/{task}_{name}/{view}.mp4"></video>'
                for view, label in (("head", "Head camera (rgb)"),
                                    ("left_wrist", "Left wrist (left_wrist_rgb)"),
                                    ("right_wrist", "Right wrist (right_wrist_rgb)"))
            )
            + f"<p>{entry['duration_seconds']:.1f} s · easy / seed 0 · recorded {verdict}</p>"
            f'<a href="{entry["result"]}">Recorded metrics</a></article>'
        )
    runtime_record(VIDEOS / "evidence/runtime/runtime-6.0.1.json")
    manifest = dict(
        delivery="Sports / Games: Dual ARX X5 on the RoboDojo rig (head + two wrist cameras)",
        date=date.today().isoformat(),
        recording_revision=args.revision,
        rig=robodojo_rig.describe(),
        videos_format=dict(
            overview="<task>.mp4, 1280x960, 25 fps, task presentation camera",
            rig_cameras="evidence/<task>/{head,left_wrist,right_wrist}.mp4, 640x480, 25 fps, "
            "same frame count and timestamps as the overview",
        ),
        controller="Privileged scripted feedback baseline; not a trained visual policy",
        superseded=str(superseded.relative_to(VIDEOS)) if superseded else None,
        runtime="evidence/runtime/runtime-6.0.1.json",
        videos=entries,
    )
    (VIDEOS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (VIDEOS / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Sports / Games</title><style>'
        "body{max-width:1400px;margin:32px auto;padding:0 24px;background:#f4f2ee;"
        "font:16px system-ui;color:#222}main{display:grid;grid-template-columns:"
        "repeat(auto-fit,minmax(320px,1fr));gap:24px}article{background:white;padding:18px;"
        "border-radius:12px}video{width:100%}a{color:#28604c}</style><h1>Sports / Games</h1>"
        "<p>Dual ARX X5 on the RoboDojo rig (arms 0.60 m apart) · presentation overview, "
        "head and both wrist cameras · Isaac Sim RTX, 250 Hz physics.</p><p><a href=\"README.md\">Recording details</a> · "
        '<a href="../validation/README.md">Validation and known failures</a></p><main>'
        + "".join(cards)
        + "</main></html>\n"
    )
    print(json.dumps([{k: e[k] for k in ("task", "success", "duration_seconds")} for e in entries]))


if __name__ == "__main__":
    main()
