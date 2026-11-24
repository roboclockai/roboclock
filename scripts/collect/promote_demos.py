"""Promote one collected oracle demonstration per task into data/<env>/<task>/.

Reads the collection written by campaign.py (RoboDojo v1.0 arx_x5 episodes, default
$DEMOS_DIR/roboclock_oracle_v3) and renders nothing: every image comes from the
episode's stored JPEGs. Per task it writes exactly two files:

    data/<env>/<task>/demo.json   the collection sidecar (outcome, metrics, instruction,
                                  rig calibration, code) plus the HDF5's path, sha256 and size
    data/<env>/<task>/demo.mp4    head camera above both wrist cameras, 25 fps, every frame

The episode shown is the first successful medium seed (the default difficulty), else the
first easy one. The HDF5 itself stays in the collection ("file" in demo.json).

Run with a Python that has h5py, Pillow and numpy (e.g. the Pi_05 openpi venv):

    python scripts/collect/promote_demos.py --domains lab hospital
"""

import os
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(os.path.expandvars("$DEMOS_DIR/roboclock_oracle_v3"))
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PREFERENCE = ("medium", "easy")


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def successes(source, task_dir):
    """Successful sidecars of one collected task whose HDF5 exists, best first."""
    found = []
    for side in sorted((source / "_meta" / task_dir).glob("seed_*.json")):
        doc = json.loads(side.read_text())
        if doc.get("status") != "success" or doc.get("difficulty") not in PREFERENCE:
            continue
        if not (source / doc["file"]).is_file():
            continue
        found.append(doc)
    found.sort(key=lambda d: (PREFERENCE.index(d["difficulty"]), d["seed"]))
    return found


def label(img, text, size=15):
    draw = ImageDraw.Draw(img)
    f = font(size)
    box = draw.textbbox((0, 0), text, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.rectangle((0, 0, w + 12, h + 10), fill=(0, 0, 0))
    draw.text((6, 4 - box[1]), text, font=f, fill=(255, 255, 255))


def decode(blob):
    return Image.open(io.BytesIO(bytes(blob))).convert("RGB")


def write_preview(h5, out, title, fps):
    """head (640x480) above left and right wrist (320x240 each): 640x720 H.264."""
    colors = [h5[f"vision/{c}/colors"] for c in CAMERAS]
    n = len(colors[0])
    w, hh = 640, 480
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{hh + hh // 2}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "27",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for t in range(n):
            canvas = Image.new("RGB", (w, hh + hh // 2))
            head = decode(colors[0][t]).resize((w, hh))
            label(head, f"{title}   t = {t / fps:5.2f} s")
            canvas.paste(head, (0, 0))
            for k, name in ((1, "left wrist"), (2, "right wrist")):
                tile = decode(colors[k][t]).resize((w // 2, hh // 2))
                label(tile, name, 12)
                canvas.paste(tile, ((k - 1) * w // 2, hh))
            proc.stdin.write(np.asarray(canvas).tobytes())
    finally:
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"ffmpeg failed for {out}")
    return n


def current_ids(source, env, recorded):
    """Current task ID of each recorded folder ID, by the sidecars' task_name.

    Collections keep the IDs they were recorded with; tasks renumbered since (2026-11-19:
    hospital B6 -> A4, B7 -> B6; sports A2 -> B5, A3 -> A2) are promoted under today's ID.
    A folder whose task_name the registry does not know keeps its own ID.
    """
    from roboclock.tasks.registry import task_registry

    by_name = {getattr(cls, "name", None): task_id for task_id, cls in task_registry(env).items()}
    ids = {}
    for task_id in recorded:
        name = None
        for side in sorted((source / "_meta" / f"{env}_{task_id}").glob("seed_*.json")):
            name = (json.loads(side.read_text()).get("metrics") or {}).get("task_name")
            if name:
                break
        ids[task_id] = by_name.get(name, task_id)
    clashes = [i for i in set(ids.values()) if list(ids.values()).count(i) > 1]
    if clashes:
        raise SystemExit(f"{env}: several recorded tasks map to {clashes}: {ids}")
    return ids


def promote(source, data, env, task_id, force=False, recorded_id=None):
    task_dir = f"{env}_{recorded_id or task_id}"
    found = successes(source, task_dir)
    if not found:
        return None
    doc = found[0]
    hdf5 = (source / doc["file"]).resolve()
    out = data / env / task_id
    demo_json = out / "demo.json"
    if demo_json.is_file() and not force:
        old = json.loads(demo_json.read_text())
        if old.get("source", {}).get("file") == doc["file"] and (out / "demo.mp4").is_file():
            return old
    out.mkdir(parents=True, exist_ok=True)
    title = f"{env} {task_id} | {doc['difficulty']} seed {doc['seed']}"
    with h5py.File(hdf5, "r") as h5:
        fps = int(h5["additional_info/frequency"][()])
        frames = write_preview(h5, out / "demo.mp4", title, fps)
        cameras = {
            c: {
                "shape": [int(v) for v in h5[f"vision/{c}/shape"][()]],
                "intrinsic_matrix": h5[f"vision/{c}/intrinsic_matrix"][()].tolist(),
            }
            for c in CAMERAS
        }
    if recorded_id and recorded_id != task_id:
        doc = renumbered(doc, task_id, recorded_id)
    record = {
        **doc,
        "file": str(hdf5),
        "source": {
            "collection": str(source),
            "file": doc["file"],
            "sha256": sha256(hdf5),
            "bytes": hdf5.stat().st_size,
            "successful_episodes_in_collection": len(found),
        },
        "preview": {
            "file": "demo.mp4",
            "frames": frames,
            "fps": fps,
            "layout": "cam_head 640x480 above cam_left_wrist | cam_right_wrist at 320x240",
        },
        "cameras": cameras,
    }
    demo_json.write_text(json.dumps(record, indent=2) + "\n")
    return record


CATEGORIES = {"A": "temporal_window", "B": "event_reactive", "C": "throughput"}


def renumbered(doc, task_id, recorded_id):
    """The sidecar under today's task ID; the ID it was recorded with stays in recorded_task_id
    (scene IDs, native results and source paths keep the recorded ID)."""
    doc = json.loads(json.dumps(doc))
    doc["task_id"], doc["recorded_task_id"] = task_id, recorded_id
    metrics = doc.get("metrics")
    if isinstance(metrics, dict):
        metrics["task_id"] = task_id
        metrics["category"] = CATEGORIES[task_id[0]]
        for key in ("episode_spec", "rig_check"):
            if isinstance(metrics.get(key), dict) and "task_id" in metrics[key]:
                metrics[key]["task_id"] = task_id
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--data", type=Path, default=REPO / "data")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--force", action="store_true", help="rewrite demos that are already promoted")
    a = p.parse_args()
    for env in a.domains:
        tasks = sorted(
            d.name[len(env) + 1 :]
            for d in (a.source / "_meta").glob(f"{env}_*")
            if d.is_dir() and "_" not in d.name[len(env) + 1 :]
        )
        ids = current_ids(a.source, env, tasks)
        for recorded_id in tasks:
            task_id = ids[recorded_id]
            record = promote(a.source, a.data, env, task_id, a.force, recorded_id)
            print(
                f"{env} {task_id}"
                + (f" (recorded as {recorded_id})" if task_id != recorded_id else "")
                + ": "
                + (
                    "no successful episode yet"
                    if record is None
                    else f"{record['difficulty']} seed {record['seed']}, {record['frames']} frames"
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
