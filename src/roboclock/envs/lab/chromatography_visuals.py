"""Continuous dye concentration cues for the A3 packed column (no fluid dynamics)."""

import numpy as np

BED_BOTTOM = 0.275
BED_TOP = 0.665
BAND_SPEED = 0.0035
BED_RADIUS = 0.0228
BED_SEGMENTS = 48
CLEAR_SOLVENT = np.array([0.985, 0.995, 1.0])


def band_concentrations(heights, time, intervals):
    """Compact, soft bands whose leading/trailing edges match collection times.

    Height is relative to the worktop. The bed-to-tip transit is abstracted by
    the existing process model, so visible elution and scoring stay synchronized.
    """
    heights = np.asarray(heights)
    result = []
    for start, end in intervals:
        phase = (time + (heights - BED_BOTTOM) / BAND_SPEED - start) / (end - start)
        envelope = np.sin(np.pi * np.clip(phase, 0, 1)) ** 0.65
        result.append(np.where((phase > 0) & (phase < 1), envelope, 0.0))
    return np.stack(result, axis=-1)


def packing_colors(heights, time, intervals, colors):
    concentration = band_concentrations(heights, time, intervals)
    # Wet silica stays pale between bands; dye stains the bed rather than
    # forming separate opaque cylinders that overlap the stationary phase.
    base = np.array([0.995, 0.998, 1.0])
    dye = np.asarray(colors) * 0.90 + 0.025
    weights = concentration.sum(axis=-1, keepdims=True)
    tint = concentration @ dye / np.maximum(weights, 1e-9)
    return base * (1 - np.clip(weights, 0, 1)) + tint * np.clip(weights, 0, 1)


def solution_tint(component_ml, total_ml, colors):
    """Absorption tint accounts for dilution by the collected clear eluent."""
    fractions = np.asarray(component_ml) / max(float(total_ml), 1e-9)
    absorbance = -np.log(0.22 + 0.78 * np.asarray(colors))
    return CLEAR_SOLVENT * np.exp(-(fractions @ absorbance))


def column_texture(intervals, colors):
    """One padded absorption texture, advected by UV offset without disk churn."""
    import hashlib
    import json
    import os
    from pathlib import Path

    from PIL import Image

    key = hashlib.sha256(json.dumps([intervals, colors]).encode()).hexdigest()[:20]
    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "liquids"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / ("column_v1_" + key + ".png")
    if not path.exists():
        span = BED_TOP - BED_BOTTOM
        # PNG top is UV v=1. Padding prevents wrapping a band back into the tube.
        heights = np.linspace(BED_TOP + span, BED_BOTTOM - span, 3072)
        linear = 0.22 + 0.78 * packing_colors(heights, 0, intervals, colors)
        srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
        pixels = np.repeat(np.round(srgb[:, None, :] * 255).astype(np.uint8), 16, axis=1)
        Image.fromarray(pixels).save(path)
    return path
