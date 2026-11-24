"""Verify lab recordings in the RoboDojo format (one folder per task).

Each folder must hold head.mp4, left_wrist.mp4 and right_wrist.mp4 (640x480) and the
overview <task>.mp4 (1280x960), all H.264 at 25 fps with equal decoded frame counts,
plus result.json whose checksums match and which carries robodojo_rig.describe(), a
passed measured rig check, the renderer read-back and 250 Hz physics.

    python scripts/lab/verify_robodojo_videos.py results/lab/videos-v2 --output verification.json
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from roboclock.envs import robodojo_rig

FFMPEG = shutil.which("ffmpeg") or "/tmp/ffmpeg/bin/ffmpeg"


def probe(path):
    """(width, height, fps, decoded frames) from FFmpeg's own decode."""
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = re.search(r"Video: h264.*?, (\d+)x(\d+).*?, ([\d.]+) fps", result.stderr)
    frames = re.findall(r"frame=\s*(\d+)", result.stderr)
    if not stream or not frames:
        raise ValueError(f"Cannot read {path}")
    return int(stream[1]), int(stream[2]), float(stream[3]), int(frames[-1])


def verify(folder):
    errors = []
    result_path = folder / "result.json"
    if not result_path.exists():
        return {"folder": folder.name, "passed": False, "errors": ["missing result.json"]}
    data = json.loads(result_path.read_text())
    expected = {
        "head": robodojo_rig.HEAD_CAMERA["resolution"],
        "left_wrist": robodojo_rig.WRIST_CAMERA["resolution"],
        "right_wrist": robodojo_rig.WRIST_CAMERA["resolution"],
        "overview": (1280, 960),
    }
    files = data.get("files", {})
    if files.get("overview") != f"{data.get('task_id')}.mp4":
        errors.append("overview must be <task>.mp4")
    videos = {}
    for name, size in expected.items():
        path = folder / files.get(name, f"{name}.mp4")
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        width, height, fps, frames = probe(path)
        videos[name] = {"file": path.name, "size": [width, height], "fps": fps, "frames": frames}
        if (width, height) != tuple(size):
            errors.append(f"{path.name} is {width}x{height}, expected {size[0]}x{size[1]}")
        if abs(fps - robodojo_rig.VIDEO_FPS) > 1e-6:
            errors.append(f"{path.name} runs at {fps} fps")
        if hashlib.sha256(path.read_bytes()).hexdigest() != data.get("sha256", {}).get(name):
            errors.append(f"{path.name} checksum differs from result.json")
    counts = {v["frames"] for v in videos.values()}
    if len(counts) != 1:
        errors.append(f"frame counts differ: { {k: v['frames'] for k, v in videos.items()} }")
    elif counts != {data.get("total_frames")}:
        errors.append("decoded frame count differs from result.json total_frames")
    if data.get("robodojo_rig") != json.loads(json.dumps(robodojo_rig.describe())):
        errors.append("result.json robodojo_rig differs from robodojo_rig.describe()")
    if not (data.get("rig_check") or {}).get("passed"):
        errors.append("measured rig check missing or failed")
    renderer = data.get("renderer_readback") or {}
    if renderer.get("rendermode") != robodojo_rig.RENDER_MODE:
        errors.append(f"renderer {renderer.get('rendermode')}")
    if renderer.get("antialiasing") != robodojo_rig.ANTIALIASING and not renderer.get(
        "antialiasing_fallback"
    ):
        errors.append("antialiasing neither DLAA nor a recorded fallback")
    if (data.get("rates") or {}).get("physics_hz") != robodojo_rig.PHYSICS_HZ:
        errors.append(f"physics at {(data.get('rates') or {}).get('physics_hz')} Hz")
    if data.get("maximum_capture_physics_delta", 1) > 1e-12:
        errors.append("capture advanced physics time")
    return {
        "folder": folder.name,
        "task_id": data.get("task_id"),
        "passed": not errors,
        "errors": errors,
        "videos": videos,
        "joint_success": data.get("metrics", {}).get("joint_success"),
        "video_seconds": data.get("video_seconds"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    folders = sorted(p.parent for p in args.root.glob("*/result.json"))
    rows = [verify(folder) for folder in folders]
    for row in rows:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['folder']} {row['errors']}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0 if rows and all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
