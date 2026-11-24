"""Publish actual X5 recordings and their honest diagnostic/evaluation status."""

import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path

TASKS = {
    "A1": "Chess",
    "A2": "Taiko",
    "A3": "Basketball",
    "B1": "Moles",
    "B2": "Boxing",
    "B3": "Bench Assist",
    "C1": "Pool",
    "C2": "Cups",
}
ROOT = Path("results/arx-migration/recordings")
OUT = Path("data/sports/new_videos/arx_x5")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    shutil.copy2("results/arx-migration/assets/manifest.json", OUT / "robot_asset.json")
    for task, title in TASKS.items():
        raw = ROOT / task
        if not (raw / "result.json").exists():
            continue
        result = json.loads((raw / "result.json").read_text())
        provenance = json.loads((raw / "render_source.json").read_text())
        snapshot = Path(provenance["snapshot"])
        verified = {}
        if not result.get("source_hashes", result.get("source_sha256")):
            raise ValueError(f"Missing render source hashes: {task}")
        for name, expected in result.get("source_hashes", result.get("source_sha256", {})).items():
            source = snapshot / name if name.startswith("src/") else snapshot / "src" / name
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Render source mismatch: {task} {name}")
            verified[name] = actual
        metric = result.get("metrics", result)
        config = result.get("configuration") or json.loads((raw / "config.json").read_text())
        assert config["embodiment"]["model"] == "dual_arx_x5"
        main = next(
            (
                p
                for p in [
                    raw / f"{task}.mp4",
                    raw / f"{task}_basic.mp4",
                    raw / "chess_duel.mp4",
                    raw / "A1_duel.mp4",
                ]
                if p.exists()
            ),
            None,
        )
        if main is None:
            raise FileNotFoundError(f"No finished main video for {task}")
        target = OUT / task
        target.mkdir(exist_ok=True)
        archive = Path("reproduction/arx_x5") / (snapshot.name + ".zip")
        source_record = dict(snapshot=str(snapshot), files=verified)
        if archive.exists():
            source_record.update(
                code_archive=str(archive),
                code_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
        (target / "source_verification.json").write_text(json.dumps(source_record, indent=2))
        clips = {}
        for view, src in [
            ("overview", main),
            ("ego", raw / "ego.mp4"),
            ("left_wrist", raw / "left_wrist.mp4"),
        ]:
            if not src.exists():
                raise FileNotFoundError(src)
            dst = target / (view + ".mp4")
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
                shutil.copy2(src, dst)
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(dst), "-f", "null", "-"], check=True
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-ss",
                        "0",
                        "-i",
                        str(dst),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        "-update",
                        "1",
                        "-y",
                        str(target / (view + ".jpg")),
                    ],
                    check=True,
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
                        str(dst),
                    ]
                )
            )
            clips[view] = {"file": str(dst.relative_to(OUT)), "sha256": digest, **info}
        counts = [int(v["streams"][0]["nb_read_frames"]) for v in clips.values()]
        assert len(set(counts)) == 1, (task, counts)
        stamps = json.loads((raw / "views.json").read_text())
        assert stamps["frames"] == counts[0]
        times = stamps["simulation_timestamps"]
        assert len(times) == counts[0]
        assert all(b > a for a, b in zip(times, times[1:])), task
        assert stamps["physics_advanced_during_capture"] is False
        assert all(v["streams"][0]["r_frame_rate"] == "15/1" for v in clips.values())
        for name in [
            "result.json",
            "config.json",
            "views.json",
            "render_source.json",
            "initial.png",
            "final.png",
        ]:
            if (raw / name).exists():
                shutil.copy2(raw / name, target / name)
        (target / "config.json").write_text(json.dumps(config, indent=2))
        success = metric.get("joint_success", metric.get("success", False))
        full = result.get("full_episode", False)
        entry = dict(
            task=task,
            title=title,
            success=success,
            full_episode=full,
            physical_success=metric.get("physical_success", metric.get("manipulation_success")),
            temporal_success=metric.get("temporal_success"),
            status="Passed" if success else "Failed baseline" if full else "Diagnostic / capped",
            failures=metric.get("failures", metric.get("physical_failures", [])),
            temporal_violations=metric.get("temporal_violations", []),
            clips=clips,
        )
        entries.append(entry)
    (OUT / "index.json").write_text(json.dumps(entries, indent=2))
    cards = []
    for e in entries:
        task = e["task"]
        reason = "; ".join(e["failures"] + e["temporal_violations"])
        cards.append(f'''<article><h2>{task} · {html.escape(e["title"])}</h2>
<p class="status">{e["status"]}</p>
<p>Action: {"passed" if e["physical_success"] else "failed" if e["physical_success"] is not None else "unreported"} · Timing: {"passed" if e["temporal_success"] else "failed" if e["temporal_success"] is not None else "unreported"}</p><p>{html.escape(reason)}</p>
<video controls preload="metadata" poster="{task}/overview.jpg" src="{task}/overview.mp4"></video>
<div>{"".join(f'<button data-task="{task}" data-view="{v}">{label}</button>' for v, label in [("overview", "Presentation"), ("ego", "Robot view"), ("left_wrist", "Left wrist")])}</div>
<p><a href="{task}/result.json">Evaluation</a> · <a href="{task}/config.json">Configuration</a> · <a href="{task}/views.json">Camera timestamps</a></p></article>''')
    (OUT / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual ARX X5 · Sports</title><style>body{margin:40px auto;max-width:1400px;padding:0 24px;background:#eef0ec;color:#21302e;font:16px system-ui}h1{font-size:38px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:24px}article{background:white;border-radius:18px;padding:20px}video{width:100%;border-radius:10px}button{margin:12px 8px 0 0;padding:10px 14px;border:1px solid #bdc9c4;border-radius:7px;background:#e8eee9;cursor:pointer}a{color:#246654}.status{font-weight:600}p{line-height:1.5}@media(max-width:600px){main{display:block}article{margin-bottom:20px}}</style>
<h1>Dual ARX X5 · Sports</h1><p>Actual Isaac Sim rollouts · fixed dual arms · easy / seed 0 · 1280 × 960, 15 fps.<br>Presentation, robot-view and wrist-view videos share simulation timestamps. A privileged scripted baseline controls the arms. Failures are retained.</p>
<p>"""
        + f"{len(entries)}/8 replacement recordings exported. B4/C3 migration scope is pending."
        + "</p><main>"
        + "".join(cards)
        + """</main>
<script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>{const v=b.closest('article').querySelector('video');const t=v.currentTime;const playing=!v.paused;v.src=b.dataset.task+'/'+b.dataset.view+'.mp4';v.poster=b.dataset.task+'/'+b.dataset.view+'.jpg';v.onloadedmetadata=()=>{v.currentTime=Math.min(t,v.duration);if(playing)v.play()}})</script></html>"""
    )
    (OUT / "README.md").write_text(
        "# Dual ARX X5 sports recordings\n\nOpen `index.html` for synchronized view selection.\n\n"
        "Each task folder contains actual presentation/ego/left-wrist videos, evaluated failures, configuration and camera timestamps.\n"
        "These easy/seed-0 scripted-baseline recordings are not multi-seed or learned-agent validation.\n"
        + "\n| Task | Presentation | Robot view | Left wrist | Baseline |\n|---|---|---|---|---|\n"
        + "".join(
            f"| {e['task']} {e['title']} | [Video]({e['task']}/overview.mp4) | "
            f"[Video]({e['task']}/ego.mp4) | [Video]({e['task']}/left_wrist.mp4) | {e['status']} |\n"
            for e in entries
        )
        + "\nB4 Freeze and C3 Ball Service retain their previous mobile implementations.\n"
    )
    labels = {True: "pass", False: "fail", None: "unreported"}
    report = [
        "# Native X5 recording validation",
        "",
        "Scope: easy / seed 0, privileged scripted baselines. This is not a multi-seed score.",
        "",
        "| Task | Video seconds | Action | Timing | Full episode | Failures |",
        "|---|---:|---|---|---|---|",
    ]
    for e in entries:
        duration = float(e["clips"]["overview"]["format"]["duration"])
        reasons = "; ".join(e["failures"] + e["temporal_violations"]) or "none"
        report.append(
            f"| [{e['task']} {e['title']}]({e['task']}/result.json) | {duration:.2f} | "
            f"{labels[e['physical_success']]} | {labels[e['temporal_success']]} | "
            f"{e['full_episode']} | {reasons} |"
        )
    report += [
        "",
        "Video duration includes any post-roll; task scores use their evaluation horizon.",
        "All views have equal frame counts, increasing simulation timestamps and 15 fps.",
        "Newly copied clips are fully decoded. SHA-256 and stream metadata are in `index.json`.",
        "Each task's `source_verification.json` matches its actual recorded source snapshot.",
        "The shared `robot_asset.json` records robot provenance and asset limitations.",
        "B4 and C3 are outside this fixed-arm replacement batch.",
        "",
    ]
    (OUT / "validation.md").write_text("\n".join(report))
    print(f"Exported and decoded {len(entries)} task recordings to {OUT}")


if __name__ == "__main__":
    main()
