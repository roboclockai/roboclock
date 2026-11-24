"""Shared streaming H.264 encoder for actual simulator camera frames."""

import subprocess
from pathlib import Path

import numpy as np


class VideoEncoder:
    """Stream RGB directly to H.264; publish only after FFmpeg flushes successfully."""

    def __init__(self, path, resolution, fps):
        self.path = Path(path)
        self.temporary = self.path.with_suffix(".partial.mp4")
        self.log = self.path.with_suffix(".ffmpeg.log")
        self.frames = 0
        width, height = resolution
        self.shape = (height, width, 3)
        self.stderr = self.log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.temporary),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.stderr,
        )

    def write(self, rgb):
        if rgb.shape != self.shape or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB {self.shape}, got {rgb.shape}/{rgb.dtype}")
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.frames += 1

    def close(self, *, publish=True):
        try:
            self.process.stdin.close()
        finally:
            code = self.process.wait()
            self.stderr.close()
        if code:
            raise RuntimeError(f"FFmpeg failed ({code}): {self.log.read_text()}")
        if publish:
            self.temporary.replace(self.path)
            self.log.unlink()
        else:
            self.temporary.unlink(missing_ok=True)
