"""Publish one native Industry recording (``python -m roboclock.envs.industry.videos --record``)
into ``data/industry/videos-v2/dual-arx-x5`` next to the recordings it supersedes.

Nothing is deleted: the new movie, sidecar, trace, previews and evidence get the
``--suffix`` (default ``_v5``, the RoboDojo-spec recordings; ``_v3``/``_v4`` were the
first common-rig recordings), the collection manifest points the task at the new
files and lists the earlier ones under ``superseded``, and the task folder gains
links to the new files beside its existing links. CPU only; no Isaac Sim.

Example::

    python scripts/industry/package_recording.py --task 45 --run <videos.py output> \
        --eval <roboclock eval output>/episodes.jsonl
"""

import argparse
import datetime
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "data/industry/videos-v2/dual-arx-x5"
CONFIGS = ROOT / "src/roboclock/envs/industry/configs"
# Review clips of the reactive tasks start shortly before the event (as before).
LEAD_IN_S = {44: 19.5, 45: 13.5}
CATEGORY = {"A": "temporal_window", "B": "reactive", "C": "throughput"}
REQUIRED = {47: 2, 48: 3, 49: 3, 50: 2}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(movie):
    info = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_streams",
             "-of", "json", str(movie)]
        )
    )["streams"][0]
    return float(info["duration"]), [info["width"], info["height"]], int(info["nb_read_frames"])


def correct_operations(task, result):
    final = result.get("final", {})
    if task == 47:
        credited = set(map(tuple, final.get("credited", [])))
        return sum(all((k, p) in credited for p in ("bearing", "bracket", "pin")) for k in (0, 1))
    if task == 48:
        return len(set(final.get("placed_ids", [])))
    if task == 49:
        return len(final.get("sorted", []))
    if task == 50:
        return len(final.get("delivered", []))
    return int(bool(result.get("task_success")))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--task", type=int, required=True, choices=range(41, 51))
    p.add_argument("--run", type=Path, required=True, help="videos.py --record output directory")
    p.add_argument("--suffix", default="_v5")
    p.add_argument("--date", default=datetime.date.today().isoformat(), help="recording date for the task README")
    p.add_argument("--eval", type=Path, help="rendered roboclock eval episodes.jsonl for this task")
    p.add_argument("--note", default="", help="free-text purpose/layout note for the sidecar")
    args = p.parse_args(argv)
    task, run = args.task, args.run
    collection = json.loads((FOLDER / "collection_manifest.json").read_text())
    entry = next(v for v in collection["videos"] if int(v["task"]) == task)
    base = entry.get("superseded_stem") or Path(entry["file"]).stem
    if base.endswith(args.suffix):
        raise SystemExit(f"Task {task} already points at a {args.suffix} recording")
    stem = base + args.suffix
    result = json.loads((run / "result.json").read_text())
    resolved = json.loads((run / "resolved.json").read_text())
    reviewed = json.loads((CONFIGS / f"task{task}/reviewed.json").read_text())
    if resolved != reviewed:
        raise SystemExit("Recording did not replay the current reviewed.json")
    if not result.get("task_success") or result.get("failure"):
        raise SystemExit(f"Native episode did not succeed: {result.get('failure')}")
    descriptions = {
        row["task_id"]: row for row in json.loads((CONFIGS / "tasks.json").read_text())["task_descriptions"]
    }
    instruction = descriptions[task]

    movie = FOLDER / f"{stem}.mp4"
    lead_in = LEAD_IN_S.get(task, 0.0)
    # robodojo_rig format: overview <stem>.mp4 (1280x960) plus head / left_wrist / right_wrist
    # (640x480), all 25 fps from the same renders; older runs wrote one candidate.mp4.
    sources = {"": run / "overview.mp4"} if (run / "overview.mp4").exists() else {"": run / "candidate.mp4"}
    for stream in ("head", "left_wrist", "right_wrist"):
        if (run / f"{stream}.mp4").exists():
            sources[stream] = run / f"{stream}.mp4"
    streams = {}
    for stream, source in sources.items():
        target = movie if not stream else FOLDER / f"{stem}_{stream}.mp4"
        if lead_in:
            # Same trim on every stream keeps them frame-synchronized (re-encoded identically).
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(lead_in), "-i",
                 str(source), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                check=True,
            )
        else:
            shutil.copy2(source, target)
        if stream:
            duration, size, count = probe(target)
            streams[stream] = dict(file=target.name, sha256=sha256(target), resolution=size, frames=count, seconds=duration)
    seconds, resolution, frames = probe(movie)
    if any(item["frames"] != frames for item in streams.values()):
        raise SystemExit(f"Camera streams are not frame-synchronized: {frames} vs {streams}")

    rows = [json.loads(line) for line in (run / "trace.jsonl").open() if line.strip()]
    trace = FOLDER / f"{stem}_trace.jsonl.gz"
    with gzip.open(trace, "wt", compresslevel=9) as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    sampled = rows[:: max(1, len(rows) // 60)]

    previews = {}
    for name in ("initial", "final", "initial_overview", "final_overview", "initial_left_wrist", "initial_right_wrist",
                 "final_left_wrist", "final_right_wrist"):
        if (run / f"{name}.png").exists():
            shutil.copy2(run / f"{name}.png", FOLDER / f"{stem}_{name}.png")
            previews[name] = f"{stem}_{name}.png"

    evidence = FOLDER / f"{stem}_evidence"
    evidence.mkdir(exist_ok=False)
    shutil.copy2(run / "result.json", evidence / "native-result.json")
    # videos.py episode.json is not copied: it would read as a second result row.
    for name in ("asset-metadata.json",):
        if (run / name).exists():
            shutil.copy2(run / name, evidence / name)
    extra = {}
    if task == 49:
        images = []
        for image in sorted(run.glob("inspection_*_verified.png")):
            shutil.copy2(image, evidence / image.name)
            images.append(dict(file=image.name, sha256=sha256(image)))
        (evidence / "inspection-images.json").write_text(json.dumps(dict(images=images), indent=2) + "\n")
        extra["inspection_rgb_evidence"] = dict(
            manifest=f"{stem}_evidence/inspection-images.json", count=len(images)
        )
        if (run / "interarm-native.jsonl").exists():
            with (run / "interarm-native.jsonl").open("rb") as src, gzip.open(
                evidence / "interarm-native.jsonl.gz", "wb", compresslevel=9
            ) as dst:
                shutil.copyfileobj(src, dst)

    events = result.get("events", {})
    elapsed = float(result["simulation_time_s"])
    correct = correct_operations(task, result)
    category = instruction["temporal_category"]
    benchmark = None
    if args.eval:
        lines = [json.loads(line) for line in args.eval.read_text().splitlines() if line.strip()]
        row = lines[-1]
        keep = (
            "task_id", "seed", "difficulty", "policy", "oracle", "task_success", "temporal_success",
            "joint_success", "temporal_violations", "correct_operations", "completion_time",
            "reaction_latency", "collision_count", "unexpected_contacts", "elapsed_time", "rig",
        )
        benchmark = {k: row.get(k) for k in keep}
        benchmark["rendered"] = True
    metrics = dict(
        task_id=task,
        task_name=Path(base).stem.split("_", 1)[1],
        domain="industry",
        category=CATEGORY[category],
        temporal_category=category,
        difficulty="reviewed_nominal",
        seed=int(resolved["seed"]),
        task_success=True,
        # The 43-45 scenes never set manipulation_success; a native success is a
        # physical success (as in the environment row and the earlier sidecars).
        manipulation_success=True,
        correct_operations=correct,
        full_throughput_target_tested=task >= 47,
        mobile_base_required=False,
        physical_failures=[],
        joint_success=True,
        temporal_success=True,
        temporal_violations=[],
        attempted_operations=REQUIRED.get(task, 1),
        accuracy=correct / REQUIRED.get(task, 1),
        elapsed_time=elapsed,
        operations_per_minute=correct * 60.0 / elapsed,
        episode_spec=resolved,
    )
    sidecar = dict(
        file=movie.name,
        environment="industry",
        simulator="isaacsim==6.0.1.0",
        robot="dual_arx_x5",
        policy="privileged_native_contact_ik_diagnostic",
        purpose=(
            "Native-physics diagnostic on the RoboDojo-spec Dual ARX rig (docs/robodojo-rig-spec.md: "
            "bases 0.60 m apart, head camera +0.04 m / +0.543 m pitched 60 deg, D435 wrist cameras, "
            "250 Hz physics, RaytracedLighting + DLAA); the movie is the recording-only overview, "
            "the policy cameras are in camera_streams"
            if result.get("rig") and streams
            else "Native-physics diagnostic on the common Dual ARX rig (0.60 m side by side, "
            "head camera +0.04 m / +0.54 m)"
            if result.get("rig")
            else "Native-physics diagnostic re-recorded on the task's recorded layout (not on the common rig)"
        )
        + "; not a learned-policy validation. " + args.note,
        instruction=instruction["instruction"],
        time_requirement=instruction["time_requirement"],
        metrics=metrics,
        episode_spec=resolved,
        events=events,
        previews=previews,
        process_trace=sampled,
        native_trace=dict(
            file=trace.name, rows=len(rows), sha256=sha256(trace), compression="gzip",
            clock="native simulation time; no rescaling",
        ),
        simulation_seconds=elapsed,
        video_seconds=seconds,
        review_edit=dict(lead_in_removed_s=lead_in, speed=1),
        sha256=sha256(movie),
        generation=dict(
            entrypoint="python -m roboclock.envs.industry.videos --record",
            task=task,
            replay=f"src/roboclock/envs/industry/configs/task{task}/reviewed.json",
            sources="src/roboclock/envs/industry/native (source hashes in the evidence native-result.json)",
        ),
        rig=result.get("rig"),
        renderer=result.get("renderer"),
        camera_streams=streams,
        rendered_benchmark_eval=benchmark,
        supersedes=dict(file=entry["file"], sha256=entry["sha256"]),
        verification=dict(native_success=True, full_video_decode=True, native_rows=len(rows)),
        limitations=[
            "Privileged controller; not evidence of learned-policy performance.",
            "Recorded by the native runner under reviewed_nominal (90 s); the A/B easy/medium/hard levels are "
            "configs/difficulty.yaml, and C episodes are judged per docs/c-class-timing.md.",
        ],
        resolution=resolution,
        total_frames=frames,
        full_budget_episode=bool(task >= 47 and elapsed >= 89.98),
        camera_cuts=[],
        native_joint_readback_in_trace=True,
        maximum_capture_physics_delta=0.0 if result.get("capture_advanced_physics") is False else None,
        **extra,
    )
    (FOLDER / f"{stem}.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    previous = entry["file"]
    superseded = collection.setdefault("superseded", [])
    superseded.append({k: entry[k] for k in ("task", "file", "sha256") if k in entry})
    entry.update(
        file=movie.name,
        sha256=sidecar["sha256"],
        integrity_pass=True,
        joint_success=True,
        correct_operations=correct,
        simulation_seconds=elapsed,
        video_seconds=seconds,
        errors=[],
        superseded_stem=base,
    )
    (FOLDER / "collection_manifest.json").write_text(json.dumps(collection, indent=2) + "\n")
    manifest = json.loads((FOLDER / "manifest.json").read_text())
    for item in manifest["videos"]:
        if int(item["task"]) == task:
            item.update(file=movie.name, metadata=f"{stem}.json", superseded=previous)
    (FOLDER / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    task_folder = ROOT / f"data/industry/task{task}"
    for ext in (".mp4", ".json"):
        link = task_folder / f"{stem}{ext}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f"../videos-v2/dual-arx-x5/{stem}{ext}")
    readme = task_folder / "README.md"
    text = readme.read_text()
    where = (
        "on the RoboDojo-spec Dual ARX rig with head and wrist cameras"
        if result.get("rig") and streams
        else "on the common Dual ARX rig"
        if result.get("rig")
        else "on the task's recorded layout (not on the common rig; see docs/industry/README.md)"
    )
    older = [item["file"] for item in collection["superseded"] if int(item["task"]) == task]
    current = (
        f"Current recording ({args.date}, {where}): [{stem}.mp4](../videos-v2/dual-arx-x5/{stem}.mp4)"
        + (
            " (overview); policy cameras "
            + ", ".join(f"[{item['file']}](../videos-v2/dual-arx-x5/{item['file']})" for item in streams.values())
            if streams
            else ""
        )
        + f". Metadata: [{stem}.json](../videos-v2/dual-arx-x5/{stem}.json).\n\n"
        "Superseded recordings, kept unchanged: "
        + ", ".join(f"[{name}](../videos-v2/dual-arx-x5/{name})" for name in older)
        + ".\n"
    )
    lines = [
        line for line in text.splitlines(keepends=True)
        if not line.startswith(("Superseded recording", "Superseded recordings"))
    ]
    index = next(
        i for i, line in enumerate(lines) if line.startswith(("Latest reviewed recording:", "Current recording"))
    )
    lines[index] = current
    readme.write_text("".join(lines).replace("\n\n\n", "\n\n"))
    print(json.dumps(dict(task=task, movie=str(movie), seconds=seconds, frames=frames, sha256=sidecar["sha256"])))


if __name__ == "__main__":
    main()
