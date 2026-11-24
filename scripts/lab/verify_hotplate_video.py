"""Check B1 video/audio integrity, timing evidence and the shared A1 v2 render profile."""

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image


def verify(directory):
    sidecar = directory / "B1_hotplate_isolation.json"
    data = json.loads(sidecar.read_text())
    video = sidecar.with_suffix(".mp4")
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    streams = {s["codec_type"]: s for s in probe["streams"]}
    visual, audio = streams["video"], streams["audio"]
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(visual["codec_name"] == "h264" and visual["pix_fmt"] == "yuv420p", "video encoding")
    check(audio["codec_name"] == "aac", "missing AAC warning track")
    check([visual["width"], visual["height"]] == data["resolution"], "resolution mismatch")
    check(int(visual["nb_read_frames"]) == data["total_frames"], "frame count mismatch")
    check(hashlib.sha256(video.read_bytes()).hexdigest() == data["sha256"], "checksum mismatch")
    times = data["frame_simulation_timestamps"]
    check(len(times) == data["simulation_frames"], "timestamp count mismatch")
    check(all(b > a for a, b in zip(times, times[1:])), "non-monotonic simulation timestamps")
    check(abs(times[-1] - data["simulation_seconds"]) < 1e-8, "final timestamp mismatch")
    check(data["maximum_capture_physics_delta"] < 1e-12, "capture advanced physics")
    check(not data["initial_geometry_errors"], "invalid initial physics")
    check(data["mean_adjacent_frame_pixel_change"] > 0, "no changing frames")
    check(
        data["initial_state_and_targets"]["scene_id"] == "laboratory_hotplate_isolation_v2",
        "wrong scene",
    )
    check(set(data["previews"]) == {"initial", "near_target", "final"}, "missing previews")
    for name in data["previews"].values():
        with Image.open(directory / name) as im:
            check(list(im.size) == data["resolution"], "preview size mismatch")
    reference = directory / "A1_water_collection.json"
    shared_keys = [
        "renderer",
        "hdr_path",
        "hdr_intensity",
        "hdr_rotation",
        "key_intensity",
        "fill_intensity",
        "path_tracing_bounces",
    ]
    if reference.exists():
        a1 = json.loads(reference.read_text())["effective_visuals"]
        for key in shared_keys:
            check(
                data["effective_visuals"][key] == a1[key], "A1 background profile mismatch: " + key
            )
    with wave.open(str(directory / data["audio"]["wav"])) as stream:
        rate = stream.getframerate()
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    sample_times = np.arange(len(pcm)) / rate
    metrics = data["metrics"]
    event, isolation = metrics["event_time"], metrics["isolation_time"]
    check(event is not None and isolation is not None, "missing alarm/isolation evidence")
    if event is not None and isolation is not None:
        check(not pcm[sample_times < event].any(), "audio before alarm")
        check(pcm[(sample_times >= event) & (sample_times < isolation)].any(), "silent warning")
        check(not pcm[sample_times >= isolation].any(), "audio after isolation")
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    check(
        decode.returncode == 0 and not decode.stderr.strip(),
        "full decode errors: " + decode.stderr.strip(),
    )
    result = {
        "task_id": "B1",
        "file": video.name,
        "integrity_pass": not errors,
        "errors": errors,
        "scope": "One actual-PhysX privileged diagnostic; not a multi-seed benchmark",
        "joint_success": metrics["joint_success"],
        "reaction_latency": metrics["reaction_latency"],
        "isolation_window": metrics["isolation_window"],
        "completion_time": metrics["completion_time"],
        "video_seconds": data["video_seconds"],
        "sha256": data["sha256"],
        "shared_A1_visual_profile": {k: data["effective_visuals"][k] for k in shared_keys},
    }
    (directory / "verification_B1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/lab/videos-v2"))
    verify(parser.parse_args().directory)
