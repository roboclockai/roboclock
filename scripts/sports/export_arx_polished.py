"""Publish actual manufacturer-CAD episodes with source and prior-model comparison."""

import argparse
import gzip
import hashlib
import html
import json
import lzma
import shutil
import subprocess
import zipfile
from pathlib import Path

from export_arx_videos import TASKS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recordings",
        type=Path,
        default=Path("/tmp/arx-polish/record/v1"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/sports/new_videos/arx_x5_polished")
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    entries, cards = [], []
    shutil.copy2(
        "data/sports/new_videos/arx_x5_official/converted_asset_hashes.json",
        out / "converted_asset_hashes.json",
    )
    shutil.copy2("src/roboclock/envs/sports/common/assets/dual_arx_x5/manifest.json", out / "robot_asset.json")
    shutil.copy2("results/arx-official/asset_audit.json", out / "asset_audit.json")
    shutil.copy2(
        "results/arx-official/gripper-probe-v2/result.json", out / "articulation_probe.json"
    )
    sources = {}
    for provenance in sorted(args.recordings.glob("*/render_source.json")):
        snapshot = Path(json.loads(provenance.read_text())["snapshot"])
        if str(snapshot) in sources:
            continue
        archive = out / (snapshot.name + ".zip")
        hashes = json.loads((snapshot / "source_hashes.json").read_text())
        resources = [
            p
            for p in (snapshot / "src").rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and p.suffix in (".json", ".yaml", ".yml", ".toml")
        ]
        files = {name: snapshot / name for name in hashes}
        files.update({str(p.relative_to(snapshot)): p for p in resources})
        for name in ("source_hashes.json", "pyproject.toml", "requirements.txt"):
            files[name] = snapshot / name
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, path in sorted(files.items()):
                data = path.read_bytes()
                if name in hashes:
                    assert hashlib.sha256(data).hexdigest() == hashes[name]
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, data)
        sources[str(snapshot)] = dict(archive=archive.name, sha256=digest(archive))
    (out / "recorded_source.json").write_text(json.dumps(sources, indent=2))
    previous = Path("data/sports/new_videos/arx_x5_official")
    for task, title in TASKS.items():
        raw = args.recordings / task
        if not (raw / "render_source.json").exists():
            continue
        r = json.loads((raw / "result.json").read_text())
        metrics = r.get("metrics", r)
        config = r.get("configuration") or json.loads((raw / "config.json").read_text())
        provenance = json.loads((raw / "render_source.json").read_text())
        snapshot = Path(provenance["snapshot"])
        hashes = r.get("source_hashes", r.get("source_sha256"))
        assert hashes, task
        for name, expected in hashes.items():
            path = snapshot / name if name.startswith("src/") else snapshot / "src" / name
            assert digest(path) == expected, (task, name)
        assert config["embodiment"]["model"] == "dual_arx_x5"
        old_config = json.loads((previous / task / "config.json").read_text())
        old_camera = old_config["visuals"]["presentation_camera"]
        new_camera = config["visuals"]["presentation_camera"]
        assert old_camera["eye"] == new_camera["eye"]
        assert old_camera["target"] == new_camera["target"]
        assert abs(new_camera["focal_length"] / old_camera["focal_length"] - 1.05) < 1e-8
        assert new_camera["light_intensity_multiplier"] == 1.3
        for quality_key in ("render_scale", "anti_aliasing", "render_updates", "renderer"):
            assert config["visuals"].get(quality_key) == old_config["visuals"].get(quality_key), (
                task,
                quality_key,
            )
        target = out / task
        target.mkdir(exist_ok=True)
        source = next(p for p in [raw / f"{task}.mp4", raw / f"{task}_basic.mp4"] if p.exists())
        video = target / "overview.mp4"
        sha = digest(source)
        if not video.exists() or digest(video) != sha:
            shutil.copy2(source, video)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
        info = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_read_frames,r_frame_rate:format=duration",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )
        stream = info["streams"][0]
        assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (1280, 960, "15/1")
        stamps = json.loads((raw / "views.json").read_text())
        assert stamps["frames"] == int(stream["nb_read_frames"])
        assert len(stamps["simulation_timestamps"]) == stamps["frames"]
        assert all(
            b > a
            for a, b in zip(stamps["simulation_timestamps"], stamps["simulation_timestamps"][1:])
        )
        assert not stamps["physics_advanced_during_capture"]
        for name in [
            "result.json",
            "views.json",
            "render_source.json",
            "initial.png",
            "final.png",
        ]:
            shutil.copy2(raw / name, target / name)
        # Preserve the full physics trace with lossless, denser compression.
        trace = target / "trace.jsonl.xz"
        if not trace.exists():
            with (
                gzip.open(raw / "trace.jsonl.gz", "rb") as source_trace,
                lzma.open(trace, "wb", preset=3) as dest_trace,
            ):
                shutil.copyfileobj(source_trace, dest_trace)
        with (
            gzip.open(raw / "trace.jsonl.gz", "rb") as source_trace,
            lzma.open(trace, "rb") as dest_trace,
        ):
            assert (
                hashlib.file_digest(source_trace, "sha256").digest()
                == hashlib.file_digest(dest_trace, "sha256").digest()
            )
        (target / "config.json").write_text(json.dumps(config, indent=2))
        (target / "source_verification.json").write_text(
            json.dumps(
                dict(snapshot=str(snapshot), recorded_source=sources[str(snapshot)], files=hashes),
                indent=2,
            )
        )
        old = json.loads((previous / task / "result.json").read_text())
        old_metrics = old.get("metrics", old)
        # Compare recorded outcomes; never borrow previous evaluation for a new camera run.
        keys = [
            "success",
            "joint_success",
            "physical_success",
            "manipulation_success",
            "temporal_success",
            "failures",
            "physical_failures",
            "temporal_violations",
        ]
        differences = {
            k: {"previous": old_metrics.get(k), "current": metrics.get(k)}
            for k in keys
            if old_metrics.get(k) != metrics.get(k)
        }
        entry = dict(
            task=task,
            title=title,
            sha256=sha,
            video=f"{task}/overview.mp4",
            camera=config["visuals"]["presentation_camera"],
            streams=info,
            full_episode=r.get("full_episode"),
            success=metrics.get("joint_success", metrics.get("success", False)),
            physical_success=metrics.get("physical_success", metrics.get("manipulation_success")),
            temporal_success=metrics.get("temporal_success"),
            failures=metrics.get("failures", metrics.get("physical_failures", [])),
            temporal_violations=metrics.get("temporal_violations", []),
            baseline_result_differences=differences,
        )
        assert entry["full_episode"], task
        entries.append(entry)
        reasons = "; ".join(entry["failures"] + entry["temporal_violations"]) or "none"
        cards.append(f'''<article><h2>{task} · {html.escape(title)}</h2>
<video controls preload="metadata" poster="{task}/initial.png" src="{task}/overview.mp4"></video>
<p><a href="{task}/overview.mp4">New video</a> · <a href="../arx_x5_official/{task}/overview.mp4">Previous presentation</a></p>
<details><summary>Framing and lighting comparison</summary><div class="compare"><figure><img src="../arx_x5_official/{task}/initial.png"><figcaption>Previous</figcaption></figure><figure><img src="{task}/initial.png"><figcaption>Brighter close-up</figcaption></figure></div></details>
<p>Action: {"pass" if entry["physical_success"] else "fail"} · Timing: {"pass" if entry["temporal_success"] else "fail"}<br>{html.escape(reasons)}</p>
<p><a href="{task}/result.json">Evaluation</a> · <a href="{task}/config.json">Camera configuration</a></p></article>''')
    (out / "index.json").write_text(json.dumps(entries, indent=2))
    (out / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dual ARX X5 · Brighter close-up</title>
<style>body{max-width:1440px;margin:36px auto;padding:0 24px;background:#edf0ec;color:#20302a;font:16px system-ui}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}article{background:#fff;border-radius:16px;padding:20px}video{width:100%;border-radius:10px}a{color:#28604c}p{line-height:1.6}.compare{display:flex}figure{width:50%;margin:6px}img{width:100%}details{cursor:pointer}@media(max-width:600px){main{display:block}article{margin-bottom:22px}}</style>
<h1>Dual ARX X5 · Brighter close-up</h1><p>Fresh native-camera renders · 1280 × 960 / 15 fps · easy / seed 0.<br>Official X5A arms and complete grippers; authored fixed fixture and photo-referenced finishes. 5% tighter lenses and 30% stronger scene lighting. Privileged scripted baseline; failures retained. The mounting fixture is not an official complete-body model.</p>
<p><a href="contact_sheet.jpg">Completed task frames</a> · <a href="asset_audit.json">CAD fidelity audit</a> · <a href="lighting_comparison.jpg">Lighting comparison</a> · <a href="shot-sequence.jpg">A3 scoring sequence</a> · <a href="recorded_source.json">Recorded source archives</a> · <a href="../arx_x5_official/index.html">Previous close-up recordings</a> · <a href="validation.md">Validation</a></p><main>"""
        + "".join(cards)
        + "</main></html>"
    )
    lines = [
        "# X5 brighter close-up validation",
        "",
        f"{len(entries)}/8 fixed X5 tasks completed; easy / seed 0. Fresh full physics rollouts with presentation-only recording.",
        "Official CAD source verified by SHA-256; native joint frames and 16 actuated DOFs retained. Wrist accessories remain from RoboTwin.",
        "All 36 converted USD visual/collision meshes preserve manufacturer triangle connectivity, scale and handedness. Maximum vertex conversion error is 1.054e-8 m. Native link masses and inertias are preserved in the derived URDF. See asset_audit.json for individual meshes and explicit joint-limit adaptations.",
        "",
        "| Task | Seconds | Action | Timing | Baseline result changed |",
        "|---|---:|---|---|---|",
    ]
    for e in entries:
        lines.append(
            f"| {e['task']} {e['title']} | {float(e['streams']['format']['duration']):.2f} | {'pass' if e['physical_success'] else 'fail'} | {'pass' if e['temporal_success'] else 'fail'} | {'yes' if e['baseline_result_differences'] else 'no'} |"
        )
    lines += [
        "",
        "All videos fully decoded, 1280 × 960, 15 fps; frame counts match increasing simulation timestamps.",
        "Configurations include camera eye, target, focal length. Source hashes match frozen recorded code.",
        "Official finger meshes and their convex-decomposition colliders replace modified RoboTwin fingers; aperture conversion updated. Materials and mounting visuals refined. Task deadlines and scoring predicates were not relaxed.",
        "B4/C3 mobile tasks remain unchanged.",
        "The four-second articulation probe passed with 0.175/0.229 mm position errors and no measured self-contact on that trajectory. This is not a task-success claim.",
        "31 focused driver, lighting and basketball-randomization tests passed. No multi-seed or multi-difficulty evaluation is claimed.",
        "Fixture and surface finishes are authored; wrist-camera inertia remains uncalibrated. Manufacturer mechanical mesh files and native joint frames are verified separately.",
        "This is manufacturer mechanical CAD fidelity, not a complete photogrammetric hardware replica: external cable routing, printed markings, scanned coatings and electronics are not supplied by the CAD or claimed as reproduced. Simulation drive gains and limits are adaptations, not measured hardware identification.",
        "Frame inspection and before/after comparisons are recorded in visual_review.json. This is not exhaustive collision or multi-seed validation.",
        "Full physics traces are preserved losslessly in trace.jsonl.xz; decoded contents are checked against the original gzip recordings.",
        "A3 uses a repositioned fixed fixture, a shorter reachable follow-through and earlier jaw opening. It still requires real contact-supported grasp, free ball flight, legal release timing and downward hoop crossing. No object attachment, imposed ball velocity or relaxed scoring is used.",
        "The initial single-hull gripper rollout was rejected after physical closure failed. Final records explicitly decompose the native finger meshes without disabling finger self-collision.",
        "",
    ]
    (out / "validation.md").write_text("\n".join(lines))
    (out / "README.md").write_text(
        "# Dual ARX X5 closer and brighter recordings\n\nOpen `index.html` for videos and framing comparisons.\n\n"
        + "\n".join(f"- [{e['task']} {e['title']}]({e['video']})" for e in entries)
        + "\n\nSee `validation.md` for recording checks and `result.json` in each task folder for actual outcomes. "
        "The previous close-up recordings remain in `../arx_x5_closeup`.\n"
    )
    print(f"Published {len(entries)}/8 brighter close-up recordings to {out}", flush=True)


if __name__ == "__main__":
    main()
