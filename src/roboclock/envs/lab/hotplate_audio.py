"""Deterministic two-tone warning for recorded B1 videos (not a microphone model)."""

import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


def warning_samples(times, event_time, isolation_time=None):
    times = np.asarray(times, dtype=float)
    if event_time is None:
        return np.zeros(times.shape, dtype=np.float32)
    relative = times - event_time
    phase = np.mod(relative, 0.5)
    tone_time = np.where(phase < 0.15, phase, phase - 0.15)
    frequency = np.where(phase < 0.15, 880.0, 1320.0)
    envelope = np.minimum(
        np.clip(tone_time / 0.005, 0, 1), np.clip((0.15 - tone_time) / 0.005, 0, 1)
    )
    active = (relative >= 0) & (phase < 0.30)
    if isolation_time is not None:
        active &= times < isolation_time
    return (0.25 * active * envelope * np.sin(2 * np.pi * frequency * tone_time)).astype(np.float32)


def add_warning_track(video, duration, event_time, isolation_time):
    video = Path(video)
    wav = video.with_name(video.stem + "_alarm.wav")
    samples = warning_samples(
        np.arange(round(duration * SAMPLE_RATE)) / SAMPLE_RATE, event_time, isolation_time
    )
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(np.round(samples * 32767).astype("<i2").tobytes())
    output = video.with_suffix(".audio.partial.mp4")
    try:
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
                str(wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                "-t",
                str(duration),
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        output.replace(video)
    finally:
        output.unlink(missing_ok=True)
    return {
        "source": "synthetic two-tone alarm; no acoustic propagation or microphone observation",
        "wav": wav.name,
        "sample_rate": SAMPLE_RATE,
        "frequencies_hz": [880, 1320],
        "start_sim_time": event_time,
        "stop_sim_time": isolation_time,
    }
