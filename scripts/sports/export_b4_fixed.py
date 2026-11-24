"""Verify and package a complete fixed Dual X5 stop/go recording."""

import argparse
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from export_c3_fixed import compress, sha

from roboclock.envs.rates import VIDEO_FPS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--matrix", type=Path)
    a = p.parse_args()
    r = json.loads((a.recording / "result.json").read_text())
    assert r["full_episode"]
    provenance = json.loads((a.recording / "render_source.json").read_text())
    snapshot = Path(provenance["snapshot"])
    for path, digest in r["source_hashes"].items():
        assert sha(snapshot / path) == digest, path
    a.output.mkdir(parents=True, exist_ok=True)
    for name in (
        "overview.mp4",
        "initial.png",
        "final.png",
        "result.json",
        "config.json",
        "render_source.json",
    ):
        shutil.copy2(a.recording / name, a.output / name)
    video = a.output / "overview.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
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
                "stream=width,height,r_frame_rate,nb_read_frames:format=duration",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    stream = probe["streams"][0]
    assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (1280, 960, "15/1")
    times = r["frame_times"]
    assert int(stream["nb_read_frames"]) == r["video_frames"] == len(times)
    assert all(x < y for x, y in zip(times, times[1:]))
    gap = max(y - x for x, y in zip(times, times[1:]))
    assert gap < 1 / VIDEO_FPS + 1 / r["control_hz"] + 1e-5
    assert abs(float(probe["format"]["duration"]) - r["recorded_sim_time"]) < 0.071
    trace_hash = compress(
        a.recording / "trace.jsonl.gz", a.output / "trace.jsonl.xz", gzip_input=True
    )
    compress(a.recording / "scene.usda", a.output / "scene.usda.xz")
    with zipfile.ZipFile(a.output / "recorded_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root in ("src", "pyproject.toml", "requirements.txt"):
            base = snapshot / root
            for f in sorted(base.rglob("*")) if base.is_dir() else [base]:
                if (
                    f.is_file()
                    and not f.is_symlink()
                    and f.suffix in (".py", ".json", ".yaml", ".yml", ".toml", ".txt")
                ):
                    z.write(f, f.relative_to(snapshot))
    shutil.copy2("docs/sports-b4-fixed-freeze.md", a.output / "implementation.md")
    for name in ("manifest.json", "converted_asset_hashes.json"):
        source = Path("src/roboclock/envs/sports/common/assets/dual_arx_x5") / name
        if source.exists():
            shutil.copy2(source, a.output / ("robot_" + name))
    assets = []
    for reference in sorted(
        set(re.findall(r"@([^@]+)@", (a.recording / "scene.usda").read_text()))
    ):
        path = Path(reference)
        assets.append(
            dict(
                reference=reference,
                sha256=sha(path) if path.is_file() else None,
                available=path.is_file(),
            )
        )
    (a.output / "asset_references.json").write_text(json.dumps(assets, indent=2) + "\n")
    runs = []

    def summary(label, result):
        return dict(
            label=label,
            success=result["success"],
            physical=result["physical_success"],
            temporal=result["temporal_success"],
            rounds=result["rounds_completed"],
            reactions=len(result["reactions"]),
            on_time=sum(x["within_deadline"] for x in result["reactions"]),
            holds=sum(x["hold_ok"] for x in result["reactions"]),
            failures=result["failures"],
        )

    runs.append(summary("Recorded easy / seed 0", r))
    if a.matrix:
        for path in sorted(a.matrix.glob("*/result.json")):
            rr = json.loads(path.read_text())
            dst = a.output / "evaluation" / path.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("result.json", "config.json"):
                shutil.copy2(path.parent / name, dst / name)
            runs.append(summary(path.parent.name, rr))
    candidates = [("grasp", 3.0), ("push", 8.0)]
    red = next((e for e in r["events"] if e["event"] == "confirmed_stop"), None)
    handoff = next((e for e in r["events"] if e["event"] == "checkpoint"), None)
    candidates += [
        ("red_hold", red["time"] + 0.3 if red else 10.0),
        ("handoff", handoff["time"] + 2 if handoff else 16.0),
    ]
    candidates += [
        ("return", max(0, r["evaluation_end"] - 5)),
        ("completion", r["recorded_sim_time"] - 0.3),
    ]
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (1280, 1560), (20, 29, 35))
    draw = ImageDraw.Draw(sheet)
    for i, (name, seconds) in enumerate(candidates):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(seconds),
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(a.output / (name + ".jpg")),
            ],
            check=True,
        )
        x = (i % 2) * 640
        y = (i // 2) * 520
        sheet.paste(Image.open(a.output / (name + ".jpg")).resize((640, 480)), (x, y + 32))
        draw.text((x + 12, y + 10), name.upper(), fill="white")
    sheet.save(a.output / "contact_sheet.jpg", quality=94)
    manifest = dict(
        video="overview.mp4",
        video_sha256=sha(video),
        probe=probe,
        full_episode=True,
        max_frame_gap=gap,
        evaluation_end=r["evaluation_end"],
        source_archive="recorded_source.zip",
        source_archive_sha256=sha(a.output / "recorded_source.zip"),
        source_hashes_verified=True,
        trace_uncompressed_sha256=trace_hash,
        runs=runs,
        validation_scope=f"{len(runs)} diagnostic difficulty/seed combinations; not exhaustive",
        visual_review="See visual_review.json",
        extracted_frames=candidates,
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = "".join(
        f"<tr><td>{x['label']}</td><td>{'PASS' if x['success'] else 'FAIL'}</td><td>{x['rounds']}/2</td><td>{x['on_time']}/{x['reactions']}</td><td>{x['holds']}/{x['reactions']}</td><td>{', '.join(x['failures']) or 'None'}</td></tr>"
        for x in runs
    )
    (a.output / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>B4 · Tabletop Freeze</title><style>body{margin:40px auto;max-width:1100px;padding:0 20px;background:#142a32;color:#f3ecdc;font:17px/1.6 system-ui}h1{font-size:32px}a{color:#9be0cd}video,img{width:100%;border-radius:12px}td,th{text-align:left;padding:10px 20px;border-bottom:1px solid #456}p{max-width:950px}</style><h1>B4 · Tabletop Freeze</h1><p>Fixed Dual ARX X5 · Green-light pushing · Red-light reaction and hold</p><video controls preload="metadata" poster="initial.png" src="overview.mp4"></video><p>Full actual PhysX episode at native 1280 × 960, 15 fps. Real gripper-held pushers move a supported puck across the center checkpoint. Both hands, both pushers and the puck must stop after red and hold until green. The return traversal reverses hand roles.</p><p>This is a privileged scripted baseline, not a trained visual agent. The table uses physical sliding friction, without an air cushion. Reaction deadlines come from independent visible game cues. Results include failures; diagnostic seeds are not exhaustive.</p><table><tr><th>Run</th><th>Result</th><th>Rounds</th><th>On-time stops</th><th>Holds</th><th>Failures</th></tr>"""
        + rows
        + """</table><p><a href="contact_sheet.jpg">Action frames</a> · <a href="result.json">Recorded evaluation</a> · <a href="config.json">Configuration</a> · <a href="manifest.json">Verification</a> · <a href="recorded_source.zip">Recorded source</a> · <a href="implementation.md">Implementation and limitations</a> · <a href="../arx_x5_c3/index.html">C3 Ball Service</a> · <a href="../arx_x5_polished/index.html">Other X5 tasks</a></p><img src="contact_sheet.jpg" alt="Actual simulation frames"></html>"""
    )
    (a.output / "README.md").write_text(
        "# B4 fixed Dual ARX X5\n\nOpen `index.html` for the full clip and evaluation. `recorded_source.zip` is an overlay for the existing repository, without binary assets. `scene.usda.xz` retains original references; it is not a portable asset pack. `trace.jsonl.xz` is the lossless full physics trace. See `implementation.md` and `validation.md`.\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
