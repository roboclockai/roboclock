"""Rebuild authored seamless epoxy, brushed-steel and powder-coat texture maps.

These are deterministic procedural materials, not photographs. Poly Haven's
photographic wall/floor maps have separate source attribution.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2] / "src/roboclock/envs/lab/assets/textures"


def save_material(name, color, height, roughness, strength):
    # Periodic derivatives keep normal-map seams continuous.
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * strength
    dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * strength
    normal = np.dstack([-dx, -dy, np.ones_like(height)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    maps = {"diff": color, "rough": roughness, "nor_gl": normal * 0.5 + 0.5}
    for key, data in maps.items():
        Image.fromarray(np.round(np.clip(data, 0, 1) * 255).astype("uint8")).save(
            ROOT / f"{name}_{key}_1k.png"
        )


def main():
    ROOT.mkdir(exist_ok=True)
    rng = np.random.default_rng(7401)
    noise = rng.normal(0, 1, (1024, 1024))
    grain = noise * 0.012
    color = np.stack([0.15 + grain, 0.19 + grain, 0.20 + grain], axis=-1)
    save_material("epoxy", color, noise * 0.009, 0.38 + noise * 0.016, 1.0)
    lines = np.repeat(rng.normal(0, 1, (1024, 1)), 1024, axis=1)
    brushing = lines * 0.011 + noise * 0.002
    color = np.repeat((0.63 + brushing)[..., None], 3, axis=2)
    save_material("brushed_steel", color, lines * 0.016 + noise * 0.001, 0.27 + lines * 0.025, 0.8)
    coat = (
        np.asarray(
            Image.fromarray(((noise - noise.min()) / np.ptp(noise) * 255).astype("uint8")).filter(
                ImageFilter.GaussianBlur(0.5)
            )
        )
        / 255
    )
    save_material(
        "powder_coat",
        np.stack([0.57 + grain, 0.61 + grain, 0.62 + grain], axis=-1),
        coat * 0.018,
        0.46 + noise * 0.012,
        1.0,
    )
    (ROOT / "PROCEDURAL.txt").write_text(
        "Epoxy, brushed_steel, powder_coat: authored by scripts/lab/build_surface_textures.py. Deterministic seed 7401; these maps are procedural, not scans.\n"
    )


if __name__ == "__main__":
    main()
