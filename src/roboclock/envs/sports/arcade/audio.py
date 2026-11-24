"""Procedural review-video sounds synchronized to measured events, not policy input."""

import subprocess
import wave

import numpy as np


def add_event_audio(video, events, duration, task_id):
    rate = 48000
    signal = np.zeros(int((duration + 0.1) * rate))
    count = 0
    for event in events:
        kind = event.get("type", event.get("event"))
        if kind not in ("stick_hit", "mole_hit", "robot_press"):
            continue
        onset = float(event["time"])
        t = np.arange(int(0.22 * rate)) / rate
        freq = 130 if kind == "stick_hit" else 220 if kind == "mole_hit" else 900
        pulse = 0.35 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 24)
        pulse += 0.10 * np.sin(2 * np.pi * freq * 2.1 * t) * np.exp(-t * 65)
        start = int(onset * rate)
        n = min(len(pulse), len(signal) - start)
        if n > 0:
            signal[start : start + n] += pulse[:n]
            count += 1
    audio = video.with_suffix(".wav")
    with wave.open(str(audio), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    target = video.with_name(video.stem + "_audio.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ],
        check=True,
    )
    target.replace(video)
    return {
        "source": "procedural impact tones triggered by measured simulation events",
        "events": count,
        "policy_observation": False,
        "acoustic_simulation": False,
    }
