"""Deterministic audible cues aligned to episode events, for exported videos only."""

import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 24000


def samples(times, task_id, event_time, action_time=None, failure_time=None):
    t = np.asarray(times)
    output = np.zeros(t.shape)
    if event_time is None:
        return output.astype(np.float32)
    rel = t - event_time
    phase = np.mod(rel, 0.30)
    active = rel >= 0
    for end in (action_time, failure_time):
        if end is not None:
            active &= t < end
    envelope = np.clip(phase / 0.006, 0, 1) * np.clip((0.13 - phase) / 0.006, 0, 1)
    output += 0.25 * active * envelope * np.sin(2 * np.pi * 1800 * phase)
    if action_time is not None:
        q = np.maximum(0, t - action_time)
        output += 0.3 * (t >= action_time) * np.exp(-q * 80) * np.sin(2 * np.pi * 1300 * q)
    if failure_time is not None:
        q = np.maximum(0, t - failure_time)
        output += 0.25 * (t >= failure_time) * np.minimum(q / 0.01, 1) * np.sin(2 * np.pi * 660 * q)
    return output.astype(np.float32)


def add_track(video, duration, task):
    video = Path(video)
    failure_time = task.time if task.failed else None
    signal = samples(
        np.arange(round(duration * RATE)) / RATE,
        task.task_id,
        task.event_time,
        task.action_time,
        failure_time,
    )
    wav = video.with_name(video.stem + "_alarm.wav")
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(np.round(np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    temporary = video.with_suffix(".audio.partial.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
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
                str(temporary),
            ],
            check=True,
            capture_output=True,
        )
        temporary.replace(video)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(
        source="synthetic event-aligned cues; no acoustic sensor simulation",
        wav=wav.name,
        sample_rate=RATE,
        start_sim_time=task.event_time,
        stop_sim_time=task.action_time,
        failure_sim_time=failure_time,
        pattern="pulsed beep and latch click",
        failure_pattern="continuous low tone",
    )
