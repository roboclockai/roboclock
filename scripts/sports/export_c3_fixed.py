"""Package and verify a completed fixed-X5 C3 simulation recording."""

import argparse
import gzip
import hashlib
import json
import lzma
import shutil
import subprocess
import zipfile
from pathlib import Path

from roboclock.envs.rates import VIDEO_FPS


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compress(source, target, *, gzip_input=False):
    digest = hashlib.sha256()
    opener = gzip.open if gzip_input else open
    with opener(source, "rb") as src, lzma.open(target, "wb", preset=3) as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
            dst.write(block)
    check = hashlib.sha256()
    with lzma.open(target, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            check.update(block)
    assert digest.digest() == check.digest()
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--matrix", type=Path)
    a = p.parse_args()
    r = json.loads((a.recording / "result.json").read_text())
    assert r["full_episode"], "Only a complete episode can be published"
    provenance = json.loads((a.recording / "render_source.json").read_text())
    snapshot = Path(provenance["snapshot"])
    for name, digest in r["source_hashes"].items():
        assert sha(snapshot / name) == digest, name
    a.output.mkdir(parents=True, exist_ok=True)
    for name in [
        "overview.mp4",
        "initial.png",
        "final.png",
        "result.json",
        "config.json",
        "render_source.json",
    ]:
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
    assert int(stream["nb_read_frames"]) == r["video_frames"] == len(r["frame_times"])
    times = r["frame_times"]
    assert all(x < y for x, y in zip(times, times[1:]))
    assert max(y - x for x, y in zip(times, times[1:])) < 1 / VIDEO_FPS + 1 / r["control_hz"] + 1e-5
    assert abs(float(probe["format"]["duration"]) - r["recorded_sim_time"]) < 0.071
    trace_hash = compress(
        a.recording / "trace.jsonl.gz", a.output / "trace.jsonl.xz", gzip_input=True
    )
    compress(a.recording / "scene.usda", a.output / "scene.usda.xz")
    with zipfile.ZipFile(a.output / "recorded_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(snapshot.rglob("*")):
            if (
                f.is_file()
                and not f.is_symlink()
                and f.suffix in (".py", ".json", ".yaml", ".yml", ".toml", ".txt")
            ):
                z.write(f, f.relative_to(snapshot))
    shutil.copy2("docs/sports-c3-fixed-service.md", a.output / "implementation.md")
    for name in ["manifest.json", "converted_asset_hashes.json"]:
        source = Path("src/roboclock/envs/sports/common/assets/dual_arx_x5") / name
        if source.exists():
            shutil.copy2(source, a.output / ("robot_" + name))
    runs = [
        dict(
            label="Recorded easy / seed 0",
            success=r["success"],
            served=r["served"],
            failures=r["failures"],
        )
    ]
    if a.matrix:
        for f in sorted(a.matrix.glob("*/result.json")):
            result = json.loads(f.read_text())
            dst = a.output / "evaluation" / f.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("result.json", "config.json"):
                shutil.copy2(f.parent / name, dst / name)
            runs.append(
                dict(
                    label=f.parent.name,
                    success=result["success"],
                    served=result["served"],
                    failures=result["failures"],
                )
            )
    for name, seconds in [("grasp", 2.6), ("feed", 11.5), ("discharge", 28.8), ("refill", 31.2)]:
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
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (1280, 1040), (20, 29, 35))
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(["grasp", "feed", "discharge", "refill"]):
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
        max_frame_gap=max(y - x for x, y in zip(times, times[1:])),
        evaluation_end=r.get("evaluation_end", r["recorded_sim_time"]),
        source_archive="recorded_source.zip",
        source_archive_sha256=sha(a.output / "recorded_source.zip"),
        source_hashes_verified=True,
        trace_uncompressed_sha256=trace_hash,
        runs=runs,
        validation_scope=f"{len(runs)} diagnostic difficulty/seed combinations; not an exhaustive seed sweep",
        visual_review="See visual_review.json; extracted frames alone are not manual review",
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = "".join(
        f"<tr><td>{x['label']}</td><td>{'PASS' if x['success'] else 'FAIL'}</td><td>{x['served']}/6</td><td>{', '.join(x['failures']) or 'None'}</td></tr>"
        for x in runs
    )
    (a.output / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>C3 · Dual X5 Ball Service</title><style>body{margin:40px auto;max-width:1100px;padding:0 20px;background:#102225;color:#f3ecdc;font:17px/1.6 system-ui}h1{font-size:32px}a{color:#9be0cd}video,img{width:100%;border-radius:12px}td,th{text-align:left;padding:10px 20px;border-bottom:1px solid #456}p{max-width:950px}</style><h1>C3 · Dual X5 Ball Service</h1><p>Fixed dual arms · Tennis training club · Continuous replenishment under scheduled demand</p><video controls preload="metadata" poster="initial.png" src="overview.mp4"></video><p>Full episode at native 1280 × 960, 15 fps. Both hands grasp real balls; accepted balls require release and supported containment. Each served ball physically passes through a driven gate. The later portion drains the buffer at the declared cadence; waiting is intentional.</p><p>The machine models gravity metering into a collector, not powered tennis launches. This is a privileged scripted baseline, not a trained visual agent.</p><table><tr><th>Run</th><th>Result</th><th>Served</th><th>Failures</th></tr>"""
        + rows
        + """</table><p>Limited diagnostic coverage; no claim of exhaustive seed robustness.</p><p><a href="contact_sheet.jpg">Action frames</a> · <a href="result.json">Recorded evaluation</a> · <a href="config.json">Configuration</a> · <a href="manifest.json">Verification</a> · <a href="recorded_source.zip">Recorded source</a> · <a href="implementation.md">Implementation and limitations</a> · <a href="../arx_x5_polished/index.html">Other fixed-X5 videos</a></p><img src="contact_sheet.jpg" alt="Actual simulation frames"></html>"""
    )
    (a.output / "README.md").write_text(
        "# C3 fixed Dual ARX X5\n\nOpen `index.html` for the full clip and evaluation. Source archive is an overlay for the existing repository; binary assets remain in the repository/cache. `scene.usda.xz` retains original absolute asset references and is not a portable asset pack. `trace.jsonl.xz` is the lossless full physics trace. See `implementation.md` for task rules, commands, assets and limitations.\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
