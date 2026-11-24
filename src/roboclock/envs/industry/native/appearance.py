"""Industry materials: the laboratory appearance plus the recorded label layout.

The recorded Industry sources carried a frozen copy of ``envs/lab/appearance.py``.
The current laboratory module is a superset of it (named glass materials, printed
labels, emissive materials that honour their requested color) except for label
textures: the Industry copy shrank long identifiers until they fit ("fit-v2").
That label renderer is kept here; everything else is the laboratory module.
Textures and HDR resolve to the laboratory/shared assets, which are byte-identical
to the copies bundled under ``assets/industry/runtime/isaac601-test``.
"""

import hashlib
import os
from pathlib import Path

from roboclock.envs.lab.appearance import LabAppearance

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def label_texture(text, color=(0.1, 0.35, 0.5)) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(("fit-v2:" + text + repr(tuple(color))).encode()).hexdigest()[:20]
    path = cache / f"{key}.png"
    if not path.exists():
        image = Image.new("RGB", (512, 256), (238, 243, 245))
        draw = ImageDraw.Draw(image)
        rgb = tuple(round(255 * x) for x in color)
        draw.rectangle((0, 0, 512, 28), fill=rgb)
        exists = Path(FONT).exists()
        size = 72
        font = ImageFont.truetype(FONT, size) if exists else ImageFont.load_default(size=size)
        while draw.textbbox((0, 0), text, font=font)[2] > 464 and size > 12:
            size -= 2
            font = ImageFont.truetype(FONT, size) if exists else ImageFont.load_default(size=size)
        draw.text((256, 116), text, fill=(24, 37, 48), font=font, anchor="mm")
        # A visual serial pattern, not a machine-readable simulator ID channel.
        for i, byte in enumerate(hashlib.sha256(text.encode()).digest()[:24]):
            if byte & 1:
                draw.rectangle((28 + 18 * i, 202, 34 + 18 * i, 239), fill=(39, 50, 56))
        image.save(path)
    return path


class IndustryAppearance(LabAppearance):
    def label_image(self, text, color):
        return label_texture(text, color)
