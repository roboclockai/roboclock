"""Publish actual manufacturer-CAD episodes with source and prior-model comparison."""

import argparse
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path

from export_arx_videos import TASKS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recordings",
        type=Path,
        default=Path("/tmp/roboclock-arx-official-renders/record/official-v3"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/sports/new_videos/arx_x5_official")
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    entries, cards = [], []
    shutil.copy2("src/roboclock/envs/sports/common/assets/dual_arx_x5/manifest.json", out / "robot_asset.json")
    shutil.copy2("results/arx-official/asset_audit.json", out / "asset_audit.json")
    shutil.copy2(
        "results/arx-official/gripper-probe-v2/result.json", out / "articulation_probe.json"
    )
    recorded_source = json.loads(Path("reproduction/arx_x5_official/manifest.json").read_text())
    recorded_source.update(
        archive="reproduction/arx_x5_official/source.zip", path_base="repository_root"
    )
    (out / "recorded_source.json").write_text(json.dumps(recorded_source, indent=2))
    previous = Path("data/sports/new_videos/arx_x5_closeup")
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
        assert (
            old_config["visuals"]["presentation_camera"] == config["visuals"]["presentation_camera"]
        )
        target = out / task
        target.mkdir(exist_ok=True)
        source = next(p for p in [raw / f"{task}.mp4", raw / f"{task}_basic.mp4"] if p.exists())
        video = target / "overview.mp4"
        sha = digest(source)
        if not video.exists() or digest(video) != sha:
            shutil.copy2(source, video)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True
        )
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
            "trace.jsonl.gz",
        ]:
            shutil.copy2(raw / name, target / name)
        (target / "config.json").write_text(json.dumps(config, indent=2))
        (target / "source_verification.json").write_text(
            json.dumps(dict(snapshot=str(snapshot), files=hashes), indent=2)
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
<p><a href="{task}/overview.mp4">New video</a> · <a href="../arx_x5_closeup/{task}/overview.mp4">Previous model</a></p>
<details><summary>Model comparison</summary><div class="compare"><figure><img src="../arx_x5_closeup/{task}/initial.png"><figcaption>Previous</figcaption></figure><figure><img src="{task}/initial.png"><figcaption>Manufacturer CAD</figcaption></figure></div></details>
<p>Action: {"pass" if entry["physical_success"] else "fail"} · Timing: {"pass" if entry["temporal_success"] else "fail"}<br>{html.escape(reasons)}</p>
<p><a href="{task}/result.json">Evaluation</a> · <a href="{task}/config.json">Camera configuration</a></p></article>''')
    (out / "index.json").write_text(json.dumps(entries, indent=2))
    (out / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dual ARX X5 · Manufacturer CAD</title>
<style>body{max-width:1440px;margin:36px auto;padding:0 24px;background:#edf0ec;color:#20302a;font:16px system-ui}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}article{background:#fff;border-radius:16px;padding:20px}video{width:100%;border-radius:10px}a{color:#28604c}p{line-height:1.6}.compare{display:flex}figure{width:50%;margin:6px}img{width:100%}details{cursor:pointer}@media(max-width:600px){main{display:block}article{margin-bottom:22px}}</style>
<h1>Dual ARX X5 · Manufacturer CAD</h1><p>Fresh native-camera renders · 1280 × 960 / 15 fps · easy / seed 0.<br>Official X5A arms and complete grippers; authored fixed fixture and photo-referenced finishes. Same close-up cameras. Privileged scripted baseline; failures retained. The mounting fixture is not an official complete-body model.</p>
<p><a href="contact_sheet.jpg">All eight task frames</a> · <a href="asset_audit.json">CAD fidelity audit</a> · <a href="../arx_x5_closeup/index.html">Previous close-up recordings</a> · <a href="validation.md">Validation</a></p><main>"""
        + "".join(cards)
        + "</main></html>"
    )
    lines = [
        "# Manufacturer X5A validation",
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
        "11 focused driver/configuration tests passed. No multi-seed or multi-difficulty evaluation is claimed.",
        "Fixture and surface finishes are authored; wrist-camera inertia remains uncalibrated. Manufacturer mechanical mesh files and native joint frames are verified separately.",
        "This is manufacturer mechanical CAD fidelity, not a complete photogrammetric hardware replica: external cable routing, printed markings, scanned coatings and electronics are not supplied by the CAD or claimed as reproduced. Simulation drive gains and limits are adaptations, not measured hardware identification.",
        "Manual inspection of eight action frames is recorded in visual_review.json; contact_sheet.jpg provides an overview. Frame inspection is not exhaustive collision or multi-seed validation.",
        "The initial single-hull gripper rollout was rejected after physical closure failed. Final records explicitly decompose the native finger meshes without disabling finger self-collision.",
        "",
    ]
    (out / "validation.md").write_text("\n".join(lines))
    (out / "README.md").write_text(
        "# Dual ARX X5 manufacturer-CAD recordings\n\nOpen `index.html` for videos and framing comparisons.\n\n"
        + "\n".join(f"- [{e['task']} {e['title']}]({e['video']})" for e in entries)
        + "\n\nSee `validation.md` for recording checks and `result.json` in each task folder for actual outcomes. "
        "The previous close-up recordings remain in `../arx_x5_closeup`.\n"
    )
    print(f"Published {len(entries)}/8 manufacturer-CAD recordings to {out}", flush=True)


if __name__ == "__main__":
    main()
