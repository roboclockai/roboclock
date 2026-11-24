"""Decode all delivered clips and extract full-episode samples for human review.

Successful decoding is not a visual or physical success verdict. Findings are
recorded separately after examining actual frames and measured task evidence.
"""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def inspect(entry, root, output):
    video = root / entry["file"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-threads", "1", "-i", str(video), "-f", "null", "-"],
        check=True,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames:format=duration",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    assert digest == entry["sha256"], entry["task"]
    duration = float(probe["format"]["duration"])
    sheet = Image.new("RGB", (1920, 2048), "#eff0ed")
    draw = ImageDraw.Draw(sheet)
    times = [0.2 + (duration - 0.5) * i / 11 for i in range(12)]
    temporary = output / (entry["task"] + "-frame.jpg")
    for i, stamp in enumerate(times):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-y",
                "-ss",
                str(stamp),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(temporary),
            ],
            check=True,
        )
        x, y = i % 3 * 640, i // 3 * 512
        with Image.open(temporary) as frame:
            sheet.paste(frame.resize((640, 480)), (x, y + 32))
        draw.text((x + 10, y + 8), f"{entry['task']} | {stamp:.2f} s", fill="#172b30")
    temporary.unlink()
    sheet.save(output / (entry["task"] + "-sheet.jpg"), quality=94)
    result = json.loads((root / entry["result"]).read_text())
    times_recorded = result.get("frame_times", [])
    metrics = result.get("metrics", result)
    return dict(
        task=entry["task"],
        video=entry["file"],
        sha256=digest,
        decoded=True,
        probe=probe,
        sample_times=times,
        physical=metrics.get("physical_success", metrics.get("manipulation_success")),
        temporal=metrics.get("temporal_success"),
        joint=metrics.get("success", metrics.get("joint_success")),
        recorded_timestamps_monotonic=all(
            a < b for a, b in zip(times_recorded, times_recorded[1:])
        ),
        max_recorded_frame_gap=max(
            (b - a for a, b in zip(times_recorded, times_recorded[1:])), default=None
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, default=Path("data/sports/videos"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = json.loads((args.collection / "manifest.json").read_text())["videos"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda e: inspect(e, args.collection, args.output), entries))
    (args.output / "decode-and-sampling.json").write_text(
        json.dumps(
            dict(
                scope="Full decode and timestamp checks; twelve uniformly sampled frames per clip. "
                "Manual findings and contact-trace checks are separate; not every-frame visual inspection.",
                videos=results,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({r["task"]: r["joint"] for r in results}))


if __name__ == "__main__":
    main()
