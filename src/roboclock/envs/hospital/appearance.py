"""Hospital room, props and lighting. Visual geometry only: nothing here collides.

Replaces the laboratory room (plaster wall, outlet strip, lab sign, analyzer) with a clinical bay:
pale walls, a light bench, a wall monitor, medical gas outlets, a hand-sanitizer dispenser, a sharps
bin, an infusion pump, a privacy curtain and an IV pole. Labels carry no hidden event schedule.

NOT YET RENDERED: tune positions, colors and light levels after looking at an actual frame.
"""

from pathlib import Path

from roboclock.envs.hospital.emergency_scenes import EmergencyScenes
from roboclock.envs.lab.appearance import LabAppearance

# name -> (color, roughness, metallic)
HOSPITAL_PALETTE = {
    "bench_top": ([0.58, 0.61, 0.62], 0.40, 0.0),
    "hosp_wall": ([0.80, 0.83, 0.85], 0.85, 0.0),
    "hosp_floor": ([0.66, 0.70, 0.70], 0.22, 0.0),
    "hosp_base": ([0.74, 0.77, 0.79], 0.45, 0.0),
    "cabinet_white": ([0.90, 0.92, 0.92], 0.50, 0.0),
    "rail": ([0.30, 0.62, 0.66], 0.50, 0.0),
    "curtain": ([0.36, 0.62, 0.68], 0.90, 0.0),
    "curtain_dark": ([0.22, 0.45, 0.52], 0.90, 0.0),
    "sterile_blue": ([0.20, 0.46, 0.72], 0.70, 0.0),
    "stainless": ([0.80, 0.82, 0.84], 0.28, 0.85),
    "sharps_red": ([0.78, 0.08, 0.06], 0.45, 0.0),
    "sharps_lid": ([0.95, 0.78, 0.10], 0.45, 0.0),
    "screen_dark": ([0.02, 0.03, 0.04], 0.15, 0.0),
    "iv_bag": ([0.86, 0.93, 0.96], 0.12, 0.0),
    "liquid": ([0.93, 0.96, 0.98], 0.90, 0.0),  # matte, so it cannot mirror dark surroundings
    "pill_orange": ([0.95, 0.55, 0.20], 0.40, 0.0),
    "pill_red": ([0.80, 0.18, 0.15], 0.40, 0.0),
    "clock_face": ([0.96, 0.96, 0.94], 0.50, 0.0),
    "label_paper": ([0.96, 0.96, 0.94], 0.60, 0.0),
    "label_blue": ([0.69, 0.82, 0.90], 0.60, 0.0),
    "tissue_white": ([0.97, 0.97, 0.96], 0.90, 0.0),
    "mug_white": ([0.95, 0.95, 0.93], 0.25, 0.0),
    "pen_blue": ([0.10, 0.22, 0.80], 0.40, 0.0),
    "pen_black": ([0.05, 0.05, 0.06], 0.40, 0.0),
    "pen_red": ([0.80, 0.10, 0.10], 0.40, 0.0),
    "bottle_white": ([0.93, 0.94, 0.95], 0.40, 0.0),
    "or_wall": ([0.80, 0.89, 0.87], 0.80, 0.0),
    "drape_green": ([0.10, 0.42, 0.38], 0.85, 0.0),
    "skin": ([0.88, 0.68, 0.58], 0.60, 0.0),
    "lamp_white": ([0.95, 0.96, 0.96], 0.30, 0.0),
    "screen_green": ([0.15, 0.85, 0.40], 0.30, 0.0),
    "gas_yellow": ([0.95, 0.80, 0.10], 0.35, 0.0),
    "gas_green": ([0.15, 0.70, 0.30], 0.35, 0.0),
    "aed_blue": ([0.10, 0.25, 0.55], 0.40, 0.0),
    "fire_red": ([0.78, 0.06, 0.05], 0.45, 0.0),
    "shock_orange": ([1.00, 0.45, 0.05], 0.35, 0.0),
    "betadine": ([0.72, 0.38, 0.10], 0.45, 0.0),
    "blood_red": ([0.55, 0.02, 0.05], 0.18, 0.0),
    "cap_white": ([0.93, 0.94, 0.95], 0.45, 0.0),
    "cap_blue": ([0.10, 0.35, 0.70], 0.45, 0.0),
    "pill_yellow": ([0.96, 0.88, 0.45], 0.40, 0.0),
    "pill_white": ([0.95, 0.95, 0.93], 0.40, 0.0),
    "glove_blue": ([0.15, 0.42, 0.78], 0.55, 0.0),
    "glove_purple": ([0.48, 0.28, 0.66], 0.55, 0.0),
    "glove_white": ([0.90, 0.92, 0.93], 0.55, 0.0),
    "shelf_white": ([0.88, 0.90, 0.91], 0.40, 0.0),
    "clip_board": ([0.55, 0.38, 0.22], 0.60, 0.0),
}
# Palette names drawn with a texture set instead of a flat color: name -> (set, tint or None).
PALETTE_TEXTURES = {
    "stainless": ("stainless_brushed", None),
    "steel": ("stainless_brushed", None),
    "aluminum": ("aluminum_brushed", None),
    "rubber": ("rubber", None),
    "polymer": ("plastic_matte", (0.74, 0.77, 0.79)),
    "cabinet_white": ("plastic_matte", (0.98, 0.98, 0.98)),
    "shelf_white": ("plastic_matte", (0.96, 0.97, 0.97)),
    "bottle_white": ("plastic_white", (0.97, 0.97, 0.98)),
    "cap_white": ("plastic_white", (0.97, 0.97, 0.98)),
    "cap_blue": ("plastic_white", (0.16, 0.42, 0.80)),
    "sterile_blue": ("fabric_wrap", (0.34, 0.60, 0.92)),
    "drape_green": ("fabric_wrap", (0.18, 0.62, 0.55)),
    "label_paper": ("paper", None),
    "tissue_white": ("paper", None),
}
GAS_OUTLETS = (("O2", [0.15, 0.70, 0.30]), ("AIR", [0.92, 0.92, 0.90]), ("VAC", [0.95, 0.80, 0.15]))
# Fixed bench equipment the clutter sampler must keep clear: (center xy, half extents xy).
DECOR_KEEPOUTS = (((0.86, -0.32), (0.10, 0.13)),)


PATIENTS = ("J. SMITH", "M. GARCIA", "A. PATEL", "L. CHEN")


def rx_label_texture(bed: int, color):
    """Pharmacy label: patient-colored band with Rx number, name, directions and quantity."""
    import hashlib
    import os

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"rx|{bed}|{tuple(color)}".encode()).hexdigest()[:20]
    path = cache / f"{key}.png"
    if path.exists():
        return path

    def font(size, bold=False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        file = Path("/usr/share/fonts/truetype/dejavu") / name
        return (
            ImageFont.truetype(str(file), size)
            if file.exists()
            else ImageFont.load_default(size=size)
        )

    width, height = 512, 320
    image = Image.new("RGB", (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    rgb = tuple(round(255 * x) for x in color)
    draw.rectangle((0, 0, width, 74), fill=rgb)
    draw.text((22, 37), f"Rx 29458{bed}1", fill=(255, 255, 255), font=font(46, True), anchor="lm")
    draw.text((22, 108), f"BED {bed}", fill=(20, 30, 40), font=font(40, True), anchor="lm")
    draw.text(
        (22, 150),
        PATIENTS[(bed - 1) % len(PATIENTS)],
        fill=(70, 80, 90),
        font=font(30),
        anchor="lm",
    )
    draw.text((22, 188), "TAKE TWO TABLETS", fill=(15, 15, 15), font=font(34, True), anchor="lm")
    draw.text((22, 224), "EVERY DAY", fill=(15, 15, 15), font=font(34, True), anchor="lm")
    draw.rectangle((0, 262, width, height), fill=(176, 208, 230))
    draw.text((22, 291), "QTY 50", fill=(25, 45, 70), font=font(36, True), anchor="lm")
    image.save(path)
    return path


def cap_label_texture(color=(0.10, 0.30, 0.70)):
    """Printed top of a child-resistant cap: PUSH DOWN & TURN with OPEN/CLOSE arrows."""
    import os

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "cap_top_push_down_turn.png"
    if path.exists():
        return path

    def font(size):
        file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        return (
            ImageFont.truetype(str(file), size)
            if file.exists()
            else ImageFont.load_default(size=size)
        )

    size = 384
    image = Image.new("RGB", (size, size), (246, 246, 244))
    draw = ImageDraw.Draw(image)
    rgb = tuple(round(255 * x) for x in color)
    draw.text((size / 2, 58), "OPEN", fill=rgb, font=font(34), anchor="mm")
    draw.text((size / 2, size - 52), "CLOSE", fill=rgb, font=font(34), anchor="mm")
    for y, text in ((140, "PUSH"), (192, "DOWN"), (244, "& TURN")):
        draw.text((size / 2, y), text, fill=rgb, font=font(46), anchor="mm")
    for flip in (0, 1):  # curved arrows at the left and right edges
        box = (18, 70, 150, 314) if not flip else (234, 70, 366, 314)
        draw.arc(box, 110 if not flip else 290, 250 if not flip else 430, fill=rgb, width=9)
        tip = (36, 262) if not flip else (348, 122)
        draw.polygon(
            [
                tip,
                (tip[0] + (24 if not flip else -24), tip[1] - 6),
                (tip[0] + (6 if not flip else -6), tip[1] - 26),
            ],
            fill=rgb,
        )
    image.save(path)
    return path


def call_label_texture(step, tool=None):
    """One procedure-board panel: standing by, or the surgeon calling for an instrument."""
    import hashlib
    import os

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{step}|{tool}".encode()).hexdigest()[:16]
    path = cache / f"call2_{key}.png"
    if path.exists():
        return path

    def font(size):
        file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        return (
            ImageFont.truetype(str(file), size)
            if file.exists()
            else ImageFont.load_default(size=size)
        )

    width, height = 400, 760
    waiting = tool is None
    image = Image.new("RGB", (width, height), (232, 236, 238) if waiting else (250, 244, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 110), fill=(95, 105, 112) if waiting else (190, 40, 40))
    draw.text((width // 2, 56), f"STEP {step}", fill=(255, 255, 255), font=font(64), anchor="mm")
    if waiting:
        draw.text((width // 2, 380), "STANDING", fill=(120, 130, 135), font=font(60), anchor="mm")
        draw.text((width // 2, 450), "BY", fill=(120, 130, 135), font=font(60), anchor="mm")
    else:
        draw.text((width // 2, 210), "SURGEON", fill=(60, 60, 60), font=font(46), anchor="mm")
        draw.text((width // 2, 262), "CALLS", fill=(60, 60, 60), font=font(46), anchor="mm")
        words, lines, size = tool.upper().split(), [], 64
        while size > 28:  # the largest type that fits the panel in at most three lines
            lines, line = [], ""
            for word in words:
                trial = f"{line} {word}".strip()
                if draw.textlength(trial, font=font(size)) <= width - 50:
                    line = trial
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
            if len(lines) <= 3 and all(
                draw.textlength(t, font=font(size)) <= width - 50 for t in lines
            ):
                break
            size -= 4
        top = 430 - (len(lines) - 1) * (size + 8) // 2
        for k, text in enumerate(lines):
            draw.text(
                (width // 2, top + k * (size + 8)),
                text,
                fill=(150, 20, 20),
                font=font(size),
                anchor="mm",
            )
        draw.rectangle((40, 590, width - 40, 700), fill=(30, 130, 70))
        draw.text((width // 2, 645), "NOW", fill=(255, 255, 255), font=font(72), anchor="mm")
    image.save(path)
    return path


def request_label_texture(text, color):
    """One line on a dark wall screen: left-aligned, as large as fits the width."""
    import hashlib
    import os

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((text + repr(tuple(color))).encode()).hexdigest()[:16]
    path = cache / f"req_{key}.png"
    if path.exists():
        return path
    file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

    def font(size):
        return (
            ImageFont.truetype(str(file), size)
            if file.exists()
            else ImageFont.load_default(size=size)
        )

    width, height = 1024, 128
    image = Image.new("RGB", (width, height), (10, 18, 28))
    draw = ImageDraw.Draw(image)
    size = 100
    while size > 24 and draw.textlength(text, font=font(size)) > width - 48:
        size -= 4
    rgb = tuple(round(255 * float(x)) for x in color)
    draw.text((24, height // 2), text, fill=rgb, font=font(size), anchor="lm")
    image.save(path)
    return path


def blood_label_texture(group):
    """Blood unit label: red header, big ABO/Rh group, unit number."""
    import hashlib
    import os

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"blood_{hashlib.sha256(group.encode()).hexdigest()[:12]}.png"
    if path.exists():
        return path

    def font(size):
        file = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        return (
            ImageFont.truetype(str(file), size)
            if file.exists()
            else ImageFont.load_default(size=size)
        )

    image = Image.new("RGB", (384, 368), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 384, 86), fill=(176, 28, 36))
    draw.text((192, 44), "BLOOD", fill=(255, 255, 255), font=font(54), anchor="mm")
    draw.text((192, 190), group, fill=(150, 20, 28), font=font(92), anchor="mm")
    draw.text((192, 290), "UNIT 4471  RBC", fill=(40, 40, 40), font=font(34), anchor="mm")
    draw.rectangle((0, 330, 384, 368), fill=(215, 215, 210))
    image.save(path)
    return path


# NVIDIA Isaac hospital props (CC BY 4.0) per task: (asset, floor xy of its center, yaw in radians).
_SIDE = ("SM_SupplyCart_02a", (1.32, 0.50), 1.5708)
_BIN = ("SM_TrashCan", (1.62, 0.05), 0.0)
HOSPITAL_PROPS = {
    "A1": (_SIDE, _BIN),
    "A2": (("SM_GasCart_01b", (1.40, 0.62), 0.0), _BIN),
    "A4": (_SIDE, _BIN),
    "B1": (("SM_GasCart_01c", (1.40, 0.62), 0.0), _SIDE, _BIN),
    "B2": (("SM_GasCart_01b", (1.45, 0.62), 0.0), _BIN),
    "B3": (("SM_FirstAidKit_01a", (1.40, 0.45), 1.5708), _BIN),
    "C1": (_SIDE, _BIN),
    "C2": (_BIN,),
    "C3": (_BIN,),
}

BOARD_Z = 1.98  # the surgery call board hangs above the surgeons' heads so nobody hides it

# Converted Sketchfab models (CC BY; see assets/models/credits.json) on the floor: (model, xy, yaw).
HOSPITAL_MODELS = {
    "A3": (("instrument_table", (1.30, 0.20), 1.5708),),  # CC BY-NC
    "B1": (("hospital_bed", (1.95, 0.80), 1.5708), ("medical_cart", (1.30, -0.28), 3.1416)),
    "B2": (("aed", (1.00, -0.68), 3.1416, 0.7675),),  # a second, wall-side AED on the counter
    "C2": (("crash_cart", (1.30, -0.28), 3.1416),),
    "C3": (("gas_sterilizer", (1.25, 0.80), 1.5708),),  # clear of the trash can at (1.62, 0.05)
    "B3": (("fire_extinguisher", (1.40, 0.62), 0.0),),
}


# B1 (ICU bedside): the drug each pump runs, a critical short half-life vasopressor infusion.
PUMP_DRUGS = ("NORADRENALINE", "ADRENALINE", "DOPAMINE")


class HospitalAppearance(EmergencyScenes, LabAppearance):
    serial_pattern = False  # the barcode strip read as black residue through the glass

    def __init__(self, stage, world, config):
        super().__init__(stage, world, config)
        # Shared palette names that stand for a real surface get its CC0 texture set, so every
        # part built with them (bezels, trays, poles, casters, wraps, cards) is textured.
        for name, (texture_set, tint) in PALETTE_TEXTURES.items():
            self.textured(name, texture_set, color=tint)
        for name, (color, roughness, metallic) in HOSPITAL_PALETTE.items():
            self.material(name, color=color, roughness=roughness, metallic=metallic)
        self.material("monitor_green", color=[0.10, 0.90, 0.40], roughness=0.3, emissive=True)
        self.material("monitor_amber", color=[0.95, 0.70, 0.10], roughness=0.3, emissive=True)

    def label_image(self, text, color):
        if text.startswith("RX|"):
            return rx_label_texture(int(text.split("|")[1]), color)
        if text == "CAP|":
            return cap_label_texture()
        if text.startswith("BLOOD|"):
            return blood_label_texture(text.split("|")[1])
        if text.startswith("REQ|"):
            return request_label_texture(text[4:], color)
        if text.startswith("WAIT|"):
            return call_label_texture(int(text.split("|")[1]))
        if text.startswith("CALL|"):
            _, step, tool = text.split("|")
            return call_label_texture(int(step), tool)
        return super().label_image(text, color)

    def amber_glass(self):
        from isaacsim.core.api.materials.visual_material import VisualMaterial
        from pxr import Gf, Sdf, UsdShade

        if "amber_glass" not in self.materials:
            path = "/World/Materials/AmberGlass"
            material = UsdShade.Material.Define(self.stage, path)
            shader = UsdShade.Shader.Define(self.stage, path + "/Shader")
            shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(1.0, 0.62, 0.14)
            )
            shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.5)
            shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.006)
            # Thin-walled: tinted transparency without refraction, so contents stay visible.
            shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.20)
            out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput("mdl").ConnectToSource(out)
            self.materials["amber_glass"] = VisualMaterial(
                name="amber_glass",
                prim_path=path,
                prim=material.GetPrim(),
                shaders_list=[shader],
                material=material,
            )
        return self.materials["amber_glass"]

    def amber_plastic(self):
        """Plain opaque orange plastic, the way a prescription vial reads from a distance."""
        return self.material("amber_plastic", color=[0.88, 0.40, 0.04], roughness=0.55)

    def pill_vial_visual(self, path, radius, height, bed, color, seed=0):
        """Prescription vial: amber plastic bottle, pile of pills, white cap, Rx label."""
        import numpy as np

        top, bottom = height / 2, -height / 2
        body_height = height - 0.0165
        body = self.amber_plastic()
        self.cylinder(
            f"{path}/AmberBody", [0, 0, bottom + body_height / 2], radius, body_height, body
        )
        self.cylinder(f"{path}/SnapRing", [0, 0, top - 0.0155], radius * 1.04, 0.003, body)
        rng = np.random.default_rng(seed)
        materials = ("pill_white", "pill_yellow", "pill_orange", "pill_white", "pill_red")
        pills = 0
        for layer, count in enumerate((15, 12, 9, 6)):
            for _ in range(count):
                angle = rng.uniform(0, 2 * np.pi)
                spread = rng.uniform(0, radius * (0.72 - 0.10 * layer))
                center = [
                    spread * np.cos(angle),
                    spread * np.sin(angle),
                    bottom + 0.0035 + layer * 0.0042 + rng.uniform(0, 0.0012),
                ]
                if pills % 5 == 4:
                    self.cube(
                        f"{path}/Pill_{pills}",
                        center,
                        [0.011, 0.0045, 0.0045],
                        materials[pills % len(materials)],
                    )
                else:
                    self.cylinder(
                        f"{path}/Pill_{pills}",
                        center,
                        0.0042,
                        0.0021,
                        materials[pills % len(materials)],
                    )
                pills += 1
        cap_top = top + 0.0005
        cap_radius = radius * 1.06
        self.cylinder(f"{path}/Cap", [0, 0, top - 0.0085], cap_radius, 0.018, "cap_white")
        for i in range(26):
            angle = 2 * np.pi * i / 26
            self.cube(
                f"{path}/CapRib_{i}",
                [
                    cap_radius * 1.01 * np.cos(angle),
                    cap_radius * 1.01 * np.sin(angle),
                    top - 0.0085,
                ],
                [0.0013, 0.0013, 0.0165],
                "cap_white",
            )
        # Printed top: blue ring, white face and three bars standing in for PUSH DOWN & TURN.
        self.cylinder(
            f"{path}/CapPrintRing", [0, 0, cap_top + 0.0003], cap_radius * 0.82, 0.0006, "cap_blue"
        )
        self.cylinder(
            f"{path}/CapPrintFace", [0, 0, cap_top + 0.0007], cap_radius * 0.72, 0.0006, "cap_white"
        )
        for k, dy in enumerate((-0.0045, 0.0, 0.0045)):
            self.cube(
                f"{path}/CapPrintBar_{k}",
                [0, dy, cap_top + 0.0012],
                [0.014, 0.0021, 0.0006],
                "cap_blue",
            )
        # Small front sticker (the rest of the bottle stays orange): paper with a colored strip.
        band = self.material(f"band_{bed}", color=color, roughness=0.5)
        front = radius + 0.0012
        self.cube(f"{path}/LabelSticker", [0, front, -0.003], [0.032, 0.0016, 0.024], "label_paper")
        self.cube(f"{path}/LabelStrip", [0, front + 0.0006, 0.0065], [0.032, 0.0016, 0.0045], band)
        self.cube(
            f"{path}/LabelLines_0",
            [0, front + 0.0006, -0.0015],
            [0.022, 0.0016, 0.0012],
            "screen_dark",
        )
        self.cube(
            f"{path}/LabelLines_1",
            [0, front + 0.0006, -0.0055],
            [0.026, 0.0016, 0.0012],
            "screen_dark",
        )
        self.cube(
            f"{path}/LabelLines_2",
            [0, front + 0.0006, -0.0095],
            [0.018, 0.0016, 0.0012],
            "screen_dark",
        )

    # --- per-task background dressing ------------------------------------------------------

    def request_screen(self, header, rows, hidden=False, row_height=0.078):
        """A large wall screen listing the requests: `rows` is [(text, color)], optionally revealed on arrival.

        Each row is its own group (callout `request_<k>`), so the task can reveal it when the request arrives.
        """
        from pxr import UsdGeom

        cx, cz = 0.40, 1.47
        self.cube("/World/Room/RequestBezel", [cx, -0.7545, cz], [0.66, 0.010, 0.46], "rubber")
        self.cube(
            "/World/Room/RequestScreen", [cx, -0.7490, cz], [0.62, 0.002, 0.42], "screen_dark"
        )
        self.label(
            "/World/Room/RequestHeader",
            f"REQ|{header}",
            [cx, -0.7478, cz + 0.165],
            [0.60, 0.075],
            (0.35, 0.80, 0.95),
        )
        callouts = dict(getattr(self, "callouts", {}))
        for k, (text, color) in enumerate(rows):
            root = f"/World/Room/Request_{k}"
            UsdGeom.Xform.Define(self.stage, root)
            self.label(
                f"{root}/Row",
                f"REQ|{text}",
                [cx, -0.7476, cz + 0.075 - row_height * k],
                [0.60, 0.075],
                color,
            )
            callouts[f"request_{k}"] = root
            if hidden:
                UsdGeom.Imageable(self.stage.GetPrimAtPath(root)).MakeInvisible()
        self.callouts = callouts

    def dress_task(self, layout):
        if layout.task_id == "A1":
            self.dress_medication_station()
            rows = []
            for i, (drug, dose) in enumerate(layout.info["requests"]):
                tag = "STAT " if layout.labels.get(f"tray_{i}", "").startswith("STAT") else ""
                color = (
                    (1.0, 0.25, 0.2)
                    if tag
                    else tuple(float(c) for c in layout.objects[f"cup_{i}"].color)
                )
                rows.append((f"{tag}BED {i + 1}  {drug} {dose}", color))
            self.request_screen("PHARMACY REQUESTS", rows, hidden=True)
        elif layout.task_id == "A2":
            self.dress_blood_bank(layout)
            request = layout.info["request"]
            self.request_screen(
                "TRANSFUSION REQUEST",
                [
                    (f"BED {request['bed']}  1 UNIT {request['unit']}", (1.0, 0.35, 0.3)),
                    ("OUT OF THE FRIDGE: CLOCK RUNNING", (1.0, 0.85, 0.3)),
                ],
            )
        elif layout.task_id == "B2":
            self.dress_cardiac_arrest(layout)
        elif layout.task_id == "B3" and "spill" in layout.info:
            from roboclock.envs.hospital import spill_scene

            spill_scene.dress(self, layout)
        elif layout.task_id == "B3":
            self.dress_fire(layout)
        elif layout.task_id == "A4" and "dose_prep" in layout.info:
            self.dress_dose_dock(layout)
        elif layout.task_id == "B1":
            self.dress_pumps(layout)
        elif layout.task_id == "C1":
            rows = []
            for j in range(len(layout.targets)):
                order = layout.orders.get(f"bin_{j}", {})
                items = ", ".join(f"{n}x {key.split(':')[1]}" for key, n in order.items())
                rows.append((f"RX {j + 1}:  {items or 'nothing'}", (0.95, 0.95, 0.9)))
            rows.append(("STAT: COURIER LEAVES WHEN THE DIAL ENDS", (1.0, 0.45, 0.35)))
            self.request_screen("STAT PHARMACY ORDER", rows)
        elif layout.task_id == "C2":
            rows = [
                (f"{layout.bin_types[f'bin_{j}'].split(':')[1]} BIN", (0.95, 0.95, 0.9))
                for j in range(len(layout.targets))
            ]
            rows.append(("RESTOCK BEFORE THE NEXT CODE", (1.0, 0.45, 0.35)))
            self.request_screen("CRASH CART RESTOCK", rows)

    def dress_pumps(self, layout):
        """One infusion pump behind each bed's bin: the alarming pump is the one whose bin light turns red.

        B1 is set at an ICU bedside: each pump carries the label of a critical short half-life
        infusion (a vasopressor), lying on its top and read from the robot's side.
        """
        from roboclock.envs.hospital.props import add_model, model_size

        if model_size("infusion_pump") is None:
            return
        z = float(layout.table_height)
        scale = 0.65
        model_top = z + scale * model_size("infusion_pump")[2]
        index = 0
        for fixture in layout.fixtures:
            if not fixture.name.startswith("pump_") or fixture.name.startswith("pump_button"):
                continue
            x, y = float(fixture.position[0]), float(fixture.position[1])
            add_model(
                self.stage,
                "infusion_pump",
                f"/World/Pumps/{fixture.name}",
                (x, y, z),
                scale=(scale, scale, scale),
                yaw=3.1416,
            )
            if self.config["visuals"]["labels"]:
                drug = PUMP_DRUGS[index % len(PUMP_DRUGS)]
                # On top of whatever is seen from above: the model or the fixture's white housing.
                top = max(model_top, float(fixture.position[2] + fixture.size[2] / 2))
                card = LabAppearance.label(  # bold sign text: the bench is bright from above
                    self,
                    f"/World/Pumps/{fixture.name}_drug",
                    drug,
                    (x, y, top + 0.002),
                    [0.13, 0.036],
                    (0.75, 0.10, 0.10),
                    sign=True,
                )
                if card is not None:  # lay the upright card flat: text toward -y, its top toward +x
                    card.AddRotateZOp().Set(90.0)
                    card.AddRotateXOp().Set(90.0)
            index += 1

    def dress_hospital_props(self, layout):
        """Real NVIDIA hospital models on the floor beside the bench, where the camera sees them.

        Low pieces only: they must not hide the bench from the camera at the front right, and they
        stay clear of the arm's reach. Visual only (no colliders).
        """
        from roboclock.envs.hospital.props import add_prop

        for index, (name, xy, yaw) in enumerate(HOSPITAL_PROPS.get(layout.task_id, ())):
            add_prop(self.stage, name, f"/World/HospitalProps/prop_{index}_{name}", xy, yaw)
        from roboclock.envs.hospital.props import add_model, model_size

        for index, (name, xy, yaw, *height) in enumerate(HOSPITAL_MODELS.get(layout.task_id, ())):
            if model_size(name) is not None:
                add_model(
                    self.stage,
                    name,
                    f"/World/HospitalModels/model_{index}_{name}",
                    (*xy, height[0] if height else 0.0),
                    yaw=yaw,
                )

    def dress_dose_dock(self, layout):
        """The drug vial the doses are drawn from, beside the filling dock (the dock itself is a layout fixture)."""
        from roboclock.envs.hospital.props import add_model, model_size

        dx, dy = layout.info["dose_prep"]["dock"]
        z = layout.table_height
        self.rounded(
            "/World/DoseDock/VialStand",
            [dx, dy + 0.125, z + 0.005],
            [0.10, 0.075, 0.010],
            "stainless",
            0.003,
        )
        if model_size("vial") is not None:
            add_model(self.stage, "vial", "/World/DoseDock/Vial", (dx, dy + 0.125, z + 0.010))
        else:
            self.cylinder(
                "/World/DoseDock/Vial", [dx, dy + 0.125, z + 0.042], 0.016, 0.062, "glass"
            )
        liquid = self.material(
            "dose_liquid", color=(0.95, 0.65, 0.10), roughness=0.2, emissive=True
        )
        self.cylinder(
            "/World/DoseDock/VialLiquid", [dx, dy + 0.125, z + 0.032], 0.0115, 0.040, liquid
        )
        self.label(
            "/World/DoseDock/VialLabel",
            "DRUG|10 mg",
            [dx, dy + 0.125 + 0.0405, z + 0.020],
            [0.050, 0.030],
            (0.1, 0.35, 0.6),
        )

    def dress_blood_bank(self, layout):
        """Blood-bank refrigerator details and a finished IV stand, placed from the layout."""
        handle, hook = layout.info["handle"], layout.info["hook"]
        cx, y_f, z = handle[0], handle[1] - 0.14, handle[2] - 0.17
        from roboclock.envs.hospital.props import add_model, model_size

        if model_size("blood_bank_fridge_open") is not None:
            # The real refrigerator, doors and drawer fronts removed, squeezed over the cabinet the drawer slides out of.
            add_model(
                self.stage,
                "blood_bank_fridge_open",
                "/World/BloodBank/Fridge",
                (cx, y_f - 0.15, z),
                scale=(0.40 / 0.633, 0.30 / 0.305, 0.33 / 0.85),
                yaw=3.1416,
            )
        self.cube(
            "/World/BloodBank/Band", [cx, y_f + 0.017, z + 0.322], [0.36, 0.004, 0.012], "blood_red"
        )
        self.cube(
            "/World/BloodBank/Seam", [cx, y_f + 0.0165, z + 0.2475], [0.34, 0.003, 0.003], "rubber"
        )
        self.cube(
            "/World/BloodBank/Display",
            [cx + 0.12, y_f + 0.017, z + 0.295],
            [0.07, 0.004, 0.040],
            "screen_dark",
        )
        # The fridge's temperature readout: one label per degree, the task reveals the next as the bag warms up.
        from pxr import UsdGeom

        callouts = dict(getattr(self, "callouts", {}))
        for k in range(7):
            degrees = 4 + k
            color = (0.1, 0.8, 0.3) if k < 3 else (1.0, 0.7, 0.1) if k < 5 else (1.0, 0.2, 0.15)
            root = f"/World/BloodBank/Temp_{k}"
            UsdGeom.Xform.Define(self.stage, root)
            self.label(
                f"{root}/Label",
                f"{degrees}.0 C",
                [cx + 0.12, y_f + 0.0195, z + 0.295],
                [0.062, 0.030],
                color,
            )
            callouts[f"temp_{k}"] = root
            if k:
                UsdGeom.Imageable(self.stage.GetPrimAtPath(root)).MakeInvisible()
        self.callouts = callouts
        self.cube(
            "/World/BloodBank/Plinth", [cx, y_f - 0.15, z + 0.004], [0.37, 0.31, 0.008], "rubber"
        )
        x, tip, base, peg_z = hook["x"], hook["tip_y"], hook["base_y"], hook["peg_z"]
        self.cylinder(
            "/World/BloodBank/PoleCollar", [x, base, z + 0.032], 0.020, 0.040, "stainless"
        )
        self.sphere("/World/BloodBank/PoleCap", [x, base, peg_z + 0.028], 0.016, "stainless")
        self.sphere("/World/BloodBank/PegTip", [x, tip - 0.002, peg_z], 0.0085, "rubber")

    def static_vial(self, name, center, bed=1, color=(0.2, 0.45, 0.8), radius=0.0165, height=0.062):
        from pxr import Gf, UsdGeom

        xform = UsdGeom.Xform.Define(self.stage, name)
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        self.pill_vial_visual(name, radius, height, bed, color, seed=bed)

    def dress_medication_station(self):
        """Nurse medication station: glove dispensers, a vial shelf and counter supplies."""
        counter_top = 0.7675
        # Wall-mounted glove dispenser rack with three boxes.
        self.cube(
            "/World/Station/GloveRack", [0.34, -0.800, 1.12], [0.46, 0.012, 0.17], "stainless"
        )
        self.cube(
            "/World/Station/GloveLip", [0.34, -0.752, 1.045], [0.46, 0.100, 0.012], "stainless"
        )
        for i, (material, text) in enumerate(
            (
                ("glove_blue", "NITRILE M"),
                ("glove_purple", "NITRILE L"),
                ("glove_white", "LATEX-FREE"),
            )
        ):
            x = 0.34 + (i - 1) * 0.15
            self.cube(
                f"/World/Station/GloveBox_{i}", [x, -0.762, 1.100], [0.135, 0.070, 0.095], material
            )
            self.cube(
                f"/World/Station/GloveSlot_{i}",
                [x, -0.7255, 1.125],
                [0.075, 0.002, 0.012],
                "rubber",
            )
            self.label(f"/World/Station/GloveLabel_{i}", text, [x, -0.7250, 1.082], [0.10, 0.032])
            self.gloves(f"/World/Station/Gloves_{i}", x, -0.722, 1.120, material)
        # Shelf of prescription vials and boxes above the counter.
        self.cube(
            "/World/Station/VialShelf", [0.74, -0.760, 1.300], [0.34, 0.100, 0.014], "shelf_white"
        )
        for i in range(7):
            self.static_vial(
                f"/World/Station/ShelfVial_{i}",
                [0.60 + 0.040 * i, -0.760, 1.307 + 0.031],
                bed=i % 4 + 1,
                color=((0.9, 0.2, 0.15), (0.15, 0.4, 0.95), (0.15, 0.75, 0.3), (0.95, 0.7, 0.1))[
                    i % 4
                ],
            )
        for i in range(2):
            self.cube(
                f"/World/Station/ShelfBox_{i}",
                [0.88 + 0.045 * i, -0.760, 1.307 + 0.034],
                [0.040, 0.050, 0.068],
                "cabinet_white",
            )
        # Counter supplies.
        self.cylinder(
            "/World/Station/WipesTub",
            [0.30, -0.690, counter_top + 0.085],
            0.040,
            0.17,
            "cabinet_white",
        )
        self.cylinder(
            "/World/Station/WipesLid",
            [0.30, -0.690, counter_top + 0.176],
            0.043,
            0.012,
            "sterile_blue",
        )
        self.cube(
            "/World/Station/TissueBox",
            [0.46, -0.700, counter_top + 0.045],
            [0.13, 0.065, 0.09],
            "sterile_blue",
        )
        self.tissue_plume("/World/Station/Tissue", 0.46, -0.700, counter_top + 0.090)
        self.cube(
            "/World/Station/TissueSlot",
            [0.46, -0.700, counter_top + 0.091],
            [0.08, 0.020, 0.002],
            "cabinet_white",
        )
        self.cube(
            "/World/Station/Clipboard",
            [0.62, -0.775, counter_top + 0.135],
            [0.20, 0.012, 0.27],
            "clip_board",
        )
        self.cube(
            "/World/Station/ClipboardSheet",
            [0.62, -0.7685, counter_top + 0.13],
            [0.17, 0.002, 0.23],
            "paper",
        )
        self.label(
            "/World/Station/ClipboardLabel",
            "MAR  BEDS 1-4",
            [0.62, -0.7665, counter_top + 0.20],
            [0.15, 0.040],
        )
        for i in range(3):
            self.cylinder(
                f"/World/Station/MedCup_{i}",
                [0.76 + 0.045 * i, -0.690, counter_top + 0.02],
                0.020,
                0.040,
                "mug_white",
            )
        self.cube(
            "/World/Station/PillOrganizer",
            [0.92, -0.690, counter_top + 0.014],
            [0.14, 0.055, 0.028],
            "cabinet_white",
        )
        for i in range(7):
            self.cube(
                f"/World/Station/OrganizerLid_{i}",
                [0.866 + 0.018 * i, -0.690, counter_top + 0.0295],
                [0.014, 0.050, 0.003],
                "sterile_blue",
            )

    # --- lighting ------------------------------------------------------------------------

    def lights(self, layout):
        super().lights(layout)
        from pxr import Gf, UsdGeom, UsdLux

        # Even, bright overhead panel like a ward ceiling light.
        panel = UsdLux.RectLight.Define(self.stage, "/World/Lighting/CeilingPanel")
        panel.CreateIntensityAttr(self.config["visuals"].get("ceiling_intensity", 260.0))
        panel.CreateWidthAttr(1.4)
        panel.CreateHeightAttr(0.9)
        panel.CreateEnableColorTemperatureAttr(True)
        panel.CreateColorTemperatureAttr(5600.0)
        UsdGeom.Xformable(panel).AddTranslateOp().Set(Gf.Vec3d(0.45, -0.05, 2.3))

    def glass(self):
        """Clearer glass than the lab's: the thick absorption depth made vials look black."""
        from pxr import Gf, Sdf

        first = "glass" not in self.materials
        material = super().glass()
        if first:
            shader = material.shaders_list[0]
            shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.004)
            shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.97, 0.99, 1.0)
            )
        return material

    def vial(self, path, radius, height, cap_material):
        """Opaque white plastic bottle with a colored cap, in place of the lab's glass vial."""
        body_height = height - 0.018
        self.cylinder(
            f"{path}/PlasticBody",
            [0, 0, -height / 2 + body_height / 2],
            radius,
            body_height,
            "bottle_white",
        )
        self.cylinder(
            f"{path}/PlasticShoulder",
            [0, 0, height / 2 - 0.022],
            radius * 0.82,
            0.008,
            "bottle_white",
        )
        self.cylinder(
            f"{path}/PlasticCap", [0, 0, height / 2 - 0.008], radius * 0.86, 0.016, cap_material
        )

    def label(self, path, text, center, size, color=(0.1, 0.35, 0.5), *, cylindrical_radius=None):
        """No label on the small curved vial: it rendered as a black patch in front of the liquid."""
        if cylindrical_radius is not None and path.endswith("/Label"):
            return None
        return super().label(path, text, center, size, color, cylindrical_radius=cylindrical_radius)

    # --- room ----------------------------------------------------------------------------

    def cylinder(self, path, center, radius, height, material, axis="Z"):
        from pxr import Gf, UsdGeom, UsdShade

        shape = UsdGeom.Cylinder.Define(self.stage, path)
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(height))
        shape.CreateAxisAttr(axis)
        shape.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        visual = self.material(material) if isinstance(material, str) else material
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(visual.material)
        return shape

    def workbench(self, layout):
        from pxr import Usd, UsdGeom, UsdShade

        z = layout.table_height
        if not layout.info.get("hide_bench"):
            for i, (x, y) in enumerate(((-0.005, -0.48), (0.90, -0.48), (-0.005, 0.48), (0.90, 0.48))):
                self.cube(f"/World/Bench/Leg_{i}", [x, y, 0.34], [0.045, 0.045, 0.68], "stainless")
            self.cube("/World/Bench/Apron", [0.45, 0.0, 0.66], [0.94, 1.02, 0.06], "stainless")
            self.cube("/World/Bench/Mount", [0.0, 0.0, z + 0.003], [0.14, 0.16, 0.006], "aluminum")
        if layout.info.get("bedside_task"):
            from roboclock.envs.hospital.bedside.ward import build_ward

            build_ward(self, layout)
        elif layout.task_id == "A3":
            self.operating_room(layout)
        else:
            self.room(z)
            self.wall_fixtures()
            self.wall_extras()
            self.everyday_objects()
            self.equipment(
                z,
                sharps_x=0.76 if layout.task_id == "A2" else 0.10,
                pump=layout.task_id not in ("B1", "B2"),
            )
            self.props(layout)
            self.dress_task(layout)
            self.dress_hospital_props(layout)
        ground = self.stage.GetPrimAtPath("/World/defaultGroundPlane")
        if ground.IsValid():
            for child in Usd.PrimRange(ground):
                if child.IsA(UsdGeom.Gprim):
                    UsdShade.MaterialBindingAPI.Apply(child).Bind(self.floor_material().material)

    # CC0 PBR sets in assets/textures (sources.json): name -> (repeats per meter, metallic,
    # roughness, normal strength). Tintable sets are near-white and take the part's color.
    TEXTURE_SETS = {
        "stainless_brushed": (6.0, 0.9, 0.32, 0.25),
        "aluminum_brushed": (8.0, 0.85, 0.28, 0.2),
        "powder_coat_dark": (5.0, 0.35, 0.55, 0.3),
        "plastic_white": (5.0, 0.0, 0.35, 0.15),
        "plastic_matte": (5.0, 0.0, 0.6, 0.2),
        "silicone": (8.0, 0.0, 0.55, 0.2),
        "foam": (6.0, 0.0, 0.9, 0.5),
        "fabric_wrap": (10.0, 0.0, 0.85, 0.5),
        "paper": (4.0, 0.0, 0.8, 0.1),
        "rubber": (12.0, 0.0, 0.85, 0.35),
    }

    def textured(self, name, texture_set, *, color=None, roughness=None, metallic=None,
                 repeats=None, normal_strength=None):
        """A textured PBR material for any visual part (mesh, cube or cylinder).

        The maps are projected in the part's own space (meters for authored meshes), so the grain
        keeps one density on every part and follows it when it moves. ``color`` tints the map:
        use it with the tintable white sets (plastic_*, silicone, fabric_wrap, paper).
        """
        if name in self.materials:
            return self.materials[name]
        from pxr import Gf, Sdf

        from roboclock.resources import resolve_asset

        per_meter, metal, rough, bump = self.TEXTURE_SETS[texture_set]
        root = resolve_asset("env://textures", environment="hospital")
        files = {s: root / f"{texture_set}_{s}_1k.png" for s in ("diff", "rough", "nor_gl")}
        missing = [str(f) for f in files.values() if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing texture maps: {missing}")
        material = self.material(
            name,
            color=[1.0, 1.0, 1.0],
            roughness=rough if roughness is None else roughness,
            metallic=metal if metallic is None else metallic,
        )
        shaders = getattr(material, "shaders_list", None)
        if not shaders:  # a test stub without a shader network keeps its plain material
            return material
        shader = shaders[0]
        for suffix, parameter, space in (
            ("diff", "diffuse_texture", "sRGB"),
            ("rough", "reflectionroughness_texture", "raw"),
            ("nor_gl", "normalmap_texture", "raw"),
        ):
            texture = shader.CreateInput(parameter, Sdf.ValueTypeNames.Asset)
            texture.Set(Sdf.AssetPath(str(files[suffix])))
            texture.GetAttr().SetMetadata("colorSpace", space)
        inputs = (
            ("diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*(color or (1.0, 1.0, 1.0)))),
            ("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float, 0.6),
            ("bump_factor", Sdf.ValueTypeNames.Float, bump if normal_strength is None else normal_strength),
            ("flip_tangent_v", Sdf.ValueTypeNames.Bool, False),
            ("project_uvw", Sdf.ValueTypeNames.Bool, True),
            ("world_or_object", Sdf.ValueTypeNames.Bool, False),
            ("texture_scale", Sdf.ValueTypeNames.Float2,
             Gf.Vec2f(*(2 * [per_meter if repeats is None else repeats]))),
        )
        for parameter, kind, value in inputs:
            shader.CreateInput(parameter, kind).Set(value)
        return material

    def texture_robot_mounts(self):
        """Light grey matte shells on the arm mounting columns, matching the white X5 arms."""
        from pxr import UsdShade

        coat = self.textured("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90))
        for side in (0, 1):
            prim = self.stage.GetPrimAtPath(f"/World/ARXMount{side}")
            if prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    coat.material, UsdShade.Tokens.strongerThanDescendants
                )

    def floor_material(self):
        """Speckled terrazzo (procedural maps); falls back to plain vinyl if they are missing."""
        if "terrazzo" in self.materials:
            return self.materials["terrazzo"]
        import numpy as np
        from isaacsim.core.api.materials import OmniPBR
        from pxr import Sdf

        from roboclock.resources import resolve_asset

        root = resolve_asset("env://textures", environment="hospital")
        files = {s: root / f"terrazzo_{s}_1k.png" for s in ("diff", "rough", "nor_gl")}
        if not all(f.exists() for f in files.values()):
            return self.material("hosp_floor")
        material = OmniPBR("/World/Materials/TerrazzoFloor", color=np.array([0.80, 0.82, 0.83]))
        shader = material.shaders_list[0]
        for suffix, parameter, space in (
            ("diff", "diffuse_texture", "sRGB"),
            ("rough", "reflectionroughness_texture", "raw"),
            ("nor_gl", "normalmap_texture", "raw"),
        ):
            texture = shader.CreateInput(parameter, Sdf.ValueTypeNames.Asset)
            texture.Set(Sdf.AssetPath(str(files[suffix])))
            texture.GetAttr().SetMetadata("colorSpace", space)
        shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(
            0.7
        )
        shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.2)
        shader.CreateInput("flip_tangent_v", Sdf.ValueTypeNames.Bool).Set(False)
        shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
        material.set_texture_scale(1.2, 1.2)
        material.set_reflection_roughness(0.3)
        self.materials["terrazzo"] = material
        return material

    def room(self, z):
        self.cube("/World/Room/BackWall", [0.4, -0.84, 1.4], [7.0, 0.06, 2.8], "hosp_wall")
        self.cube(
            "/World/Room/SideWall", [-1.45, 0.9, 1.4], [0.06, 3.5, 2.8], "hosp_wall"
        )  # closes the room behind the curtain
        self.cube("/World/Room/Baseboard", [0.4, -0.797, 0.07], [7.0, 0.02, 0.10], "hosp_base")
        # Upper wall cabinets at the two ends; the middle of the wall carries the equipment.
        upper = [-0.30, 0.96]
        for i, x in enumerate(upper):
            self.cube(
                f"/World/Room/Cabinet_{i}", [x, -0.91, 1.62], [0.40, 0.18, 0.46], "cabinet_white"
            )
            self.cube(
                f"/World/Room/CabinetDoor_{i}",
                [x, -0.81, 1.62],
                [0.386, 0.016, 0.44],
                "cabinet_white",
            )
            self.cube(
                f"/World/Room/CabinetHandle_{i}",
                [x + 0.12, -0.792, 1.58],
                [0.012, 0.018, 0.10],
                "steel",
            )
        # Protective wall rail with brackets.
        self.cube("/World/Room/WallRail", [0.4, -0.785, 0.95], [7.0, 0.035, 0.12], "rail")
        self.cube(
            "/World/Room/WallRailTrim", [0.4, -0.7655, 0.95], [7.0, 0.004, 0.125], "cabinet_white"
        )
        for i, x in enumerate((-0.80, 1.60)):  # end caps
            self.cube(f"/World/Room/RailCap_{i}", [x, -0.785, 0.95], [0.02, 0.04, 0.125], "rail")
        for i, x in enumerate([-0.30, 0.40, 1.10]):
            self.cube(
                f"/World/Room/RailBracket_{i}", [x, -0.805, 0.95], [0.03, 0.02, 0.05], "steel"
            )
        # Back counter and lower cabinets: light top, white cabinets.
        self.cube("/World/Room/BackCounter", [0.40, -0.66, 0.74], [1.8, 0.25, 0.055], "bench_top")
        for i, x in enumerate([-0.16, 0.28, 0.72, 1.16]):
            self.cube(
                f"/World/Room/LowerCabinet_{i}",
                [x, -0.72, 0.38],
                [0.425, 0.26, 0.66],
                "cabinet_white",
            )
            for row in range(3):
                self.cube(
                    f"/World/Room/Drawer_{i}_{row}",
                    [x, -0.58, 0.18 + row * 0.20],
                    [0.407, 0.015, 0.185],
                    "paper",
                )
                self.cube(
                    f"/World/Room/DrawerHandle_{i}_{row}",
                    [x, -0.561, 0.19 + row * 0.20],
                    [0.12, 0.020, 0.010],
                    "steel",
                )

    def wall_fixtures(self):
        # Patient monitor on the wall: dark screen with simple green and amber traces.
        self.cube("/World/Room/MonitorBezel", [0.33, -0.775, 1.50], [0.48, 0.03, 0.31], "rubber")
        self.cube(
            "/World/Room/MonitorScreen", [0.33, -0.758, 1.50], [0.45, 0.004, 0.28], "screen_dark"
        )
        for i, (dz, width, material) in enumerate(
            (
                (0.07, 0.36, "monitor_green"),
                (0.0, 0.36, "monitor_amber"),
                (-0.07, 0.30, "monitor_green"),
            )
        ):
            self.cube(
                f"/World/Room/MonitorTrace_{i}",
                [0.30, -0.755, 1.50 + dz],
                [width, 0.002, 0.008],
                material,
            )
        self.label(
            "/World/Room/MonitorReadout", "HR 72  SpO2 98", [0.33, -0.753, 1.405], [0.30, 0.040]
        )
        # Medical gas outlet panel.
        self.cube("/World/Room/GasPanel", [0.82, -0.785, 1.12], [0.34, 0.025, 0.14], "aluminum")
        for i, (text, color) in enumerate(GAS_OUTLETS):
            x = 0.82 + (i - 1) * 0.10
            self.cube(
                f"/World/Room/GasOutlet_{i}",
                [x, -0.770, 1.12],
                [0.060, 0.012, 0.060],
                self.material(f"Gas_{text}", color=color, roughness=0.4),
            )
            self.label(f"/World/Room/GasLabel_{i}", text, [x, -0.763, 1.065], [0.055, 0.022])
        # Hand-sanitizer dispenser.
        self.cube(
            "/World/Room/Sanitizer", [0.05, -0.77, 1.22], [0.075, 0.06, 0.20], "cabinet_white"
        )
        self.cube("/World/Room/SanitizerLever", [0.05, -0.735, 1.17], [0.020, 0.040, 0.012], "rail")
        self.label("/World/Room/SanitizerLabel", "SANITIZER", [0.05, -0.739, 1.27], [0.065, 0.026])

    def wall_extras(self):
        """Clock, soap dispenser, hygiene poster and a records computer, as in a clinic bay."""
        # Wall clock (disc facing into the room).
        self.cylinder(
            "/World/Room/ClockRim", [0.66, -0.797, 1.80], 0.112, 0.016, "rubber", axis="Y"
        )
        self.cylinder(
            "/World/Room/ClockFace", [0.66, -0.788, 1.80], 0.104, 0.014, "clock_face", axis="Y"
        )
        self.cube(
            "/World/Room/ClockMinute", [0.66, -0.7795, 1.832], [0.006, 0.002, 0.070], "rubber"
        )
        self.cube("/World/Room/ClockHour", [0.688, -0.7795, 1.80], [0.052, 0.002, 0.007], "rubber")
        self.cylinder(
            "/World/Room/ClockHub", [0.66, -0.7785, 1.80], 0.007, 0.004, "rubber", axis="Y"
        )
        # Blue soap dispenser beside the sanitizer, and a hand-hygiene poster above both.
        self.cube(
            "/World/Room/SoapDispenser", [-0.07, -0.77, 1.22], [0.07, 0.06, 0.20], "sterile_blue"
        )
        self.cube(
            "/World/Room/SoapLever", [-0.07, -0.735, 1.17], [0.020, 0.040, 0.012], "cabinet_white"
        )
        self.label("/World/Room/SoapLabel", "SOAP", [-0.07, -0.739, 1.27], [0.055, 0.026])
        self.cube("/World/Room/PosterFrame", [-0.01, -0.792, 1.60], [0.20, 0.008, 0.27], "steel")
        self.cube("/World/Room/Poster", [-0.01, -0.786, 1.60], [0.18, 0.004, 0.25], "paper")
        self.label(
            "/World/Room/PosterText",
            "HAND HYGIENE",
            [-0.01, -0.783, 1.60],
            [0.16, 0.05],
            [0.10, 0.45, 0.50],
        )
        # Records computer on the back counter.
        top = 0.7675
        self.cube(
            "/World/Equipment/PcBase", [-0.16, -0.70, top + 0.006], [0.14, 0.10, 0.012], "stainless"
        )
        self.cube(
            "/World/Equipment/PcNeck", [-0.16, -0.72, top + 0.075], [0.035, 0.03, 0.13], "stainless"
        )
        self.cube(
            "/World/Equipment/PcBezel", [-0.16, -0.735, top + 0.20], [0.30, 0.020, 0.19], "rubber"
        )
        self.cube(
            "/World/Equipment/PcScreen",
            [-0.16, -0.724, top + 0.20],
            [0.28, 0.003, 0.17],
            "screen_dark",
        )
        self.label(
            "/World/Equipment/PcText",
            "PATIENT RECORD",
            [-0.16, -0.7215, top + 0.20],
            [0.24, 0.12],
            [0.15, 0.45, 0.60],
        )
        self.cube(
            "/World/Equipment/Keyboard",
            [-0.16, -0.62, top + 0.008],
            [0.26, 0.09, 0.016],
            "cabinet_white",
        )

    def group(self, path, center, rx=0.0, ry=0.0, rz=0.0):
        """A posed parent prim: children use local coordinates and tilt with it."""
        from pxr import Gf, UsdGeom

        xform = UsdGeom.Xform.Define(self.stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        for angle, op in (
            (rx, xform.AddRotateXOp),
            (ry, xform.AddRotateYOp),
            (rz, xform.AddRotateZOp),
        ):
            if angle:
                op().Set(float(angle))
        return path

    def sphere(self, path, center, radius, material):
        from pxr import Gf, UsdGeom, UsdShade

        shape = UsdGeom.Sphere.Define(self.stage, path)
        shape.CreateRadiusAttr(float(radius))
        shape.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        visual = self.material(material) if isinstance(material, str) else material
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(visual.material)
        return shape

    def gloves(self, path, x, y, z, material):
        """Two disposable gloves pulled up out of a dispenser slot, with cuffs and fingers."""
        poses = ((-0.011, 0.000, 0.000, -7, 4), (0.012, -0.004, 0.006, 6, -5))
        for g, (dx, dy, dz, tilt, spin) in enumerate(poses):
            base = self.group(f"{path}/Glove_{g}", [x + dx, y + dy, z + dz], ry=tilt, rz=spin)
            self.cube(f"{base}/Cuff", [0, 0, 0.004], [0.050, 0.006, 0.014], material)
            self.cube(f"{base}/Palm", [0, 0, 0.034], [0.046, 0.004, 0.050], material)
            fingers = (
                (-0.0165, -9, 0.040),
                (-0.0055, -3, 0.047),
                (0.0055, 3, 0.045),
                (0.0165, 9, 0.036),
            )
            for f, (fx, splay, length) in enumerate(fingers):
                finger = self.group(
                    f"{base}/Finger_{f}", [fx, 0, 0.056 + length / 2 - 0.006], ry=splay
                )
                self.cube(f"{finger}/Body", [0, 0, 0], [0.0092, 0.0042, length], material)
            thumb = self.group(f"{base}/Thumb", [0.027, 0, 0.040], ry=-48)
            self.cube(f"{thumb}/Body", [0, 0, 0.012], [0.012, 0.0042, 0.034], material)

    def tissue_plume(self, path, x, y, z):
        """A crumpled tissue pulled up out of the box opening."""
        sheets = (
            (0.000, 0.085, -10, 6, 0.075),
            (-0.012, 0.070, 9, -14, 0.065),
            (0.014, 0.060, -5, 20, 0.060),
            (0.002, 0.045, 14, -4, 0.080),
        )
        for k, (dx, h, rx, rz, w) in enumerate(sheets):
            sheet = self.group(f"{path}/Sheet_{k}", [x + dx, y, z], rx=rx, rz=rz)
            self.cube(f"{sheet}/Body", [0, 0, h / 2], [w, 0.0025, h], "tissue_white")

    def everyday_objects(self):
        """Ordinary things found around a nursing counter, as in a lived-in clinic bay."""
        import numpy as np

        top = 0.7675
        self.cylinder("/World/Everyday/PenCup", [0.21, -0.690, top + 0.043], 0.033, 0.086, "rubber")
        pens = (
            (-0.010, 0.004, -8, 6, "pen_blue", 0.150),
            (0.008, -0.006, 5, -9, "pen_black", 0.140),
            (0.004, 0.010, -12, 3, "pen_red", 0.145),
            (-0.004, -0.010, 7, 11, "pen_blue", 0.135),
        )
        for k, (dx, dy, rx, ry, material, h) in enumerate(pens):
            pen = self.group(
                f"/World/Everyday/Pen_{k}", [0.21 + dx, -0.690 + dy, top + 0.040], rx=rx, ry=ry
            )
            self.cylinder(f"{pen}/Barrel", [0, 0, h / 2], 0.0045, h, material)
            self.cylinder(f"{pen}/Tip", [0, 0, 0.003], 0.0025, 0.006, "steel")
        self.cylinder(
            "/World/Everyday/CreamBottle",
            [0.575, -0.640, top + 0.060],
            0.028,
            0.120,
            "cabinet_white",
        )
        self.cylinder(
            "/World/Everyday/CreamCap", [0.575, -0.640, top + 0.125], 0.012, 0.012, "sterile_blue"
        )
        self.cube(
            "/World/Everyday/CreamPump",
            [0.585, -0.640, top + 0.145],
            [0.030, 0.010, 0.012],
            "sterile_blue",
        )
        self.cylinder(
            "/World/Everyday/Mug", [0.670, -0.655, top + 0.045], 0.036, 0.090, "mug_white"
        )
        self.cube(
            "/World/Everyday/MugHandle",
            [0.714, -0.655, top + 0.048],
            [0.012, 0.010, 0.050],
            "mug_white",
        )
        self.cube(
            "/World/Everyday/MugHandleTop",
            [0.710, -0.655, top + 0.070],
            [0.020, 0.010, 0.010],
            "mug_white",
        )
        self.cylinder(
            "/World/Everyday/MugCoffee", [0.670, -0.655, top + 0.089], 0.032, 0.002, "clip_board"
        )
        cx, cy = 0.96, -0.592
        for i in range(30):
            angle = 2 * np.pi * i / 30
            self.sphere(
                f"/World/Everyday/Stethoscope/Tube_{i}",
                [cx + 0.050 * np.cos(angle), cy + 0.050 * np.sin(angle), top + 0.0048],
                0.0048,
                "rubber",
            )
        self.cylinder(
            "/World/Everyday/Stethoscope/Chest",
            [cx + 0.050, cy, top + 0.006],
            0.022,
            0.012,
            "stainless",
        )
        self.cylinder(
            "/World/Everyday/Stethoscope/ChestFace",
            [cx + 0.050, cy, top + 0.0125],
            0.016,
            0.002,
            "rubber",
        )
        for side in (-1, 1):
            for i in range(7):
                self.sphere(
                    f"/World/Everyday/Stethoscope/Ear{'L' if side < 0 else 'R'}_{i}",
                    [cx - 0.050 + 0.004 * i, cy + side * (0.004 + 0.004 * i), top + 0.0048],
                    0.0042,
                    "rubber",
                )
            self.sphere(
                f"/World/Everyday/Stethoscope/EarTip{'L' if side < 0 else 'R'}",
                [cx - 0.020, cy + side * 0.034, top + 0.0048],
                0.0065,
                "stainless",
            )
        self.cylinder(
            "/World/Everyday/SprayBottle",
            [1.080, -0.690, top + 0.075],
            0.030,
            0.150,
            "cabinet_white",
        )
        self.cylinder(
            "/World/Everyday/SprayNeck", [1.080, -0.690, top + 0.160], 0.013, 0.025, "sterile_blue"
        )
        self.cube(
            "/World/Everyday/SprayHead",
            [1.080, -0.676, top + 0.180],
            [0.030, 0.052, 0.026],
            "sterile_blue",
        )
        self.cube(
            "/World/Everyday/SprayTrigger",
            [1.080, -0.652, top + 0.160],
            [0.016, 0.006, 0.034],
            "sterile_blue",
        )
        self.label(
            "/World/Everyday/SprayLabel",
            "ALCOHOL",
            [1.080, -0.7195, top + 0.085],
            [0.040, 0.050],
            [0.1, 0.4, 0.6],
        )
        self.cube(
            "/World/Everyday/TowelDispenser",
            [1.12, -0.755, 1.14],
            [0.22, 0.085, 0.30],
            "cabinet_white",
        )
        self.cube(
            "/World/Everyday/TowelSlot", [1.12, -0.7125, 1.015], [0.17, 0.004, 0.012], "rubber"
        )
        self.cube(
            "/World/Everyday/TowelSheet",
            [1.12, -0.7135, 0.975],
            [0.15, 0.003, 0.075],
            "tissue_white",
        )
        self.label("/World/Everyday/TowelLabel", "TOWELS", [1.12, -0.7115, 1.16], [0.12, 0.040])

    def ring(self, path, center, width, height, thickness, material, depth=0.006):
        """A flat loop (finger ring or hanging eyelet) in the x-z plane, built from four bars."""
        cx, cy, cz = center
        bar = thickness
        for k, (dx, dz, w, h) in enumerate(
            (
                (0, height / 2 - bar / 2, width, bar),
                (0, -height / 2 + bar / 2, width, bar),
                (-width / 2 + bar / 2, 0, bar, height - 2 * bar),
                (width / 2 - bar / 2, 0, bar, height - 2 * bar),
            )
        ):
            self.cube(
                f"{path}_{k}",
                [cx + dx, cy, cz + dz],
                [w, depth, h],
                material,
            )

    def surgical_tool(self, path, kind, size):
        """Recognizable instrument silhouettes inside the (box) collision envelope; long axis is x."""
        sx, sy, sz = size
        if kind == "scalpel":
            self.cube(
                f"{path}/Handle", [-sx * 0.12, 0, 0], [sx * 0.74, sy * 0.70, sz * 0.72], "stainless"
            )
            for k in range(6):
                self.cube(
                    f"{path}/Groove_{k}",
                    [-sx * 0.42 + 0.008 * k, 0, 0],
                    [0.0016, sy * 0.78, sz * 0.78],
                    "rubber",
                )
            self.cube(
                f"{path}/Blade", [sx * 0.40, 0, 0], [sx * 0.22, 0.0018, sz * 0.56], "stainless"
            )
            self.cube(
                f"{path}/Tip",
                [sx * 0.505, 0, -sz * 0.08],
                [sx * 0.04, 0.0018, sz * 0.30],
                "stainless",
            )
        elif kind == "forceps":
            for side in (-1, 1):
                prong = self.group(
                    f"{path}/Prong_{'L' if side < 0 else 'R'}",
                    [0, side * sy * 0.20, 0],
                    rz=-side * 1.2,
                )
                self.cube(f"{prong}/Arm", [0, 0, 0], [sx * 0.96, sy * 0.30, sz * 0.60], "stainless")
                self.cube(
                    f"{prong}/Tip",
                    [sx * 0.46, -side * sy * 0.03, 0],
                    [sx * 0.05, sy * 0.22, sz * 0.36],
                    "stainless",
                )
            for k in range(5):
                self.cube(
                    f"{path}/Serration_{k}",
                    [-sx * 0.30 + 0.012 * k, 0, 0],
                    [0.0016, sy * 0.95, sz * 0.7],
                    "rubber",
                )
        elif kind in ("scissors", "clamp"):
            jaw = sx * (0.52 if kind == "scissors" else 0.38)
            for side in (-1, 1):
                tag = "L" if side < 0 else "R"
                blade = self.group(
                    f"{path}/Blade_{tag}", [sx * 0.10, side * sy * 0.10, 0], rz=-side * 4
                )
                self.cube(f"{blade}/Body", [jaw * 0.5, 0, 0], [jaw, 0.004, sz * 0.60], "stainless")
                shank = self.group(
                    f"{path}/Shank_{tag}", [-sx * 0.15, side * sy * 0.10, 0], rz=side * 3
                )
                self.cube(f"{shank}/Body", [0, 0, 0], [sx * 0.34, 0.004, sz * 0.55], "stainless")
                self.ring(
                    f"{path}/Ring_{tag}",
                    [-sx * 0.39, side * sy * 0.30, 0],
                    sx * 0.20,
                    sz * 1.5,
                    0.0045,
                    "rubber",
                )
            self.cylinder(f"{path}/Pivot", [sx * 0.10, 0, 0], 0.0045, 0.0045, "steel", axis="Y")
            if kind == "clamp":
                for k in range(4):
                    self.cube(
                        f"{path}/Ratchet_{k}",
                        [-sx * 0.20 + 0.007 * k, 0, 0.005],
                        [0.003, sy * 0.30, 0.004],
                        "stainless",
                    )
        elif kind == "retractor":
            self.cube(
                f"{path}/Handle", [-sx * 0.12, 0, 0], [sx * 0.62, sy * 0.60, sz * 0.74], "stainless"
            )
            self.cube(
                f"{path}/Neck", [sx * 0.27, 0, 0], [sx * 0.16, sy * 0.40, sz * 0.40], "stainless"
            )
            self.cube(
                f"{path}/Blade", [sx * 0.455, 0, 0], [0.004, sy * 1.55, sz * 0.62], "stainless"
            )

    def procedure_board(self, steps):
        """Wall display with a panel per step: standing by until the surgeon calls for something.

        The call panels start hidden; the task swaps them in at each call time.
        """
        from pxr import UsdGeom

        self.cube("/World/Room/BoardFrame", [0.45, -1.540, BOARD_Z], [0.80, 0.020, 0.54], "steel")
        self.callouts = {}
        for k, tool in enumerate(steps):
            x = 0.45 + (k - 1) * 0.25
            for kind, text, color in (
                ("wait", f"WAIT|{k + 1}", (0.4, 0.4, 0.4)),
                ("call", f"CALL|{k + 1}|{tool}", (0.8, 0.15, 0.15)),
            ):
                root = f"/World/Room/{kind.title()}_{k}"
                UsdGeom.Xform.Define(self.stage, root)
                self.label(f"{root}/Label", text, [x, -1.528, BOARD_Z], [0.22, 0.42], color)
                self.callouts[f"{kind}_{k}"] = root
                if kind == "call":
                    UsdGeom.Imageable(self.stage.GetPrimAtPath(root)).MakeInvisible()

    def model_visual(self, path, name, size, mode="length"):
        """Mount a converted model on an object's collision box (centered; box size in meters).

        "length": uniform scale so the model's length matches the box's x extent (tools, syringes).
        "fit": stretch to the box (blood bag, with the hanging loop above it: size is the overall extent).
        Returns False when the model is not installed, so the caller can fall back to the built shapes.
        """
        from roboclock.envs.hospital.props import add_model, model_size

        native = model_size(name)
        if native is None:
            return False
        sx, sy, sz = size
        if mode == "fit":
            scale = (sx / native[0], sy / native[1], sz / native[2])
        else:
            uniform = min(max(sx / native[0], 0.8), 1.6 if name.startswith("tool_") else 1.25)
            scale = (uniform, uniform, uniform)
        add_model(
            self.stage,
            name,
            f"{path}/Model",
            (0.0, 0.0, -size[2] / 2 if mode == "length" else 0.0),
            scale,
        )
        return True

    def model_visual_at(self, path, name, size, z_bottom):
        """Stretch a converted model to `size` with its bottom at z_bottom (relative to the object's center)."""
        from roboclock.envs.hospital.props import add_model, model_size

        native = model_size(name)
        if native is None:
            return False
        scale = tuple(a / b for a, b in zip(size, native))
        add_model(self.stage, name, f"{path}/Model", (0.0, 0.0, z_bottom), scale)
        return True

    def blood_bag_visual(self, path, size, eyelet):
        """A blood unit: red pouch with weld seams, ports, a hanging eyelet and a group label."""
        bx, by, bz = size
        # The real bag (maroon, taped label, hanging hole) over body + loop; the colliders stay as built.
        total = bz + eyelet["height"]
        if self.model_visual_at(path, "blood_bag", (bx, by, total), z_bottom=-bz / 2):
            return
        self.rounded(
            f"{path}/Pouch", [0, 0, 0], [bx - 0.002, by - 0.004, bz - 0.004], "blood_red", 0.016
        )
        self.cube(
            f"{path}/WeldTop", [0, 0, bz / 2 - 0.005], [bx, by * 0.92, 0.010], "cabinet_white"
        )
        self.cube(
            f"{path}/WeldBottom", [0, 0, -bz / 2 + 0.005], [bx, by * 0.92, 0.010], "cabinet_white"
        )
        for k, dx in enumerate((-0.014, 0.014)):
            self.cylinder(
                f"{path}/Port_{k}", [dx, 0, -bz / 2 + 0.006], 0.006, 0.014, "cabinet_white"
            )
        self.ring(
            f"{path}/Eyelet",
            [0, 0, bz / 2 + eyelet["height"] / 2],
            eyelet["width"],
            eyelet["height"],
            eyelet["bar"],
            "cabinet_white",
            depth=eyelet["thickness"],
        )
        if self.config["visuals"]["labels"]:
            self.label(
                f"{path}/BagLabel", "BLOOD|O NEG", [0, by / 2 + 0.0008, 0.004], [0.072, 0.070]
            )

    def operating_room(self, layout):
        """Operating room: a draped back table at the robot, the patient on the table behind it."""
        import numpy as np

        z = layout.table_height
        self.cube("/World/Room/BackWall", [0.40, -1.60, 1.40], [9.0, 0.06, 2.8], "or_wall")
        self.cube("/World/Room/SideWall", [-2.6, 0.2, 1.4], [0.06, 4.5, 2.8], "or_wall")
        self.cube("/World/Room/Baseboard", [0.40, -1.565, 0.07], [9.0, 0.02, 0.10], "hosp_base")
        self.cube("/World/Room/WallRail", [0.40, -1.565, 0.95], [9.0, 0.035, 0.12], "rail")
        self.procedure_board(layout.info.get("situations") or layout.steps)
        self.wall_clock(1.05, -1.572, 1.85)
        # Instrument cabinet against the back wall.
        self.cube("/World/OR/Cabinet", [-1.25, -1.43, 0.95], [0.60, 0.30, 1.90], "cabinet_white")
        self.cube(
            "/World/OR/CabinetGlass", [-1.25, -1.275, 1.30], [0.52, 0.004, 1.00], "screen_dark"
        )
        self.cube("/World/OR/CabinetBase", [-1.25, -1.275, 0.38], [0.52, 0.004, 0.66], "stainless")
        for k in range(3):
            self.cube(
                f"/World/OR/CabinetShelf_{k}",
                [-1.25, -1.285, 0.95 + 0.30 * k],
                [0.50, 0.02, 0.008],
                "shelf_white",
            )
        # The robot's back table: sterile drape over the bench, hanging over its edges.
        self.cube(
            "/World/OR/BenchDrape", [0.54, 0.0, z + 0.0012], [0.84, 1.06, 0.0016], "drape_green"
        )
        self.cube(
            "/World/OR/DrapeFront", [0.962, 0.0, z - 0.09], [0.004, 1.06, 0.19], "drape_green"
        )
        for k, y in enumerate((-0.532, 0.532)):
            self.cube(
                f"/World/OR/DrapeSide_{k}", [0.54, y, z - 0.09], [0.84, 0.004, 0.19], "drape_green"
            )
        for name, fixture in ((f.name, f) for f in layout.fixtures):
            if name == "source_pad":  # instruments lie on a folded sterile wrap
                cx, cy = fixture.position[:2]
                self.cube(
                    "/World/OR/Wrap", [cx, cy, z + 0.0124], [0.21, 0.33, 0.0008], "cabinet_white"
                )
        tray = layout.targets.get("tray")
        if tray is not None:  # blue towel lining the Mayo tray
            self.cube(
                "/World/OR/TrayTowel",
                [*tray.position[:2], tray.support_z + 0.0006],
                [2 * tray.half_extents[0] - 0.012, 2 * tray.half_extents[1] - 0.012, 0.001],
                "sterile_blue",
            )
        # Operating table: the real hospital bed (backrest laid flat, head at +x like the patient), else
        # base, column, frame, padded top and side rails.
        tx, ty = (
            0.42,
            -1.27,
        )  # the bed lies against the far wall; the surgeons stand in the aisle by the bench
        from roboclock.envs.hospital.props import add_model, model_info

        top = 0.88
        info = model_info("hospital_bed_flat")
        if info is not None:
            add_model(
                self.stage,
                "hospital_bed_flat",
                "/World/OR/Bed",
                (tx + 0.05, ty, 0.0),
                scale=(0.91, 0.36, top / (info["top_z"] + 0.02)),
            )
        else:
            self.rounded(
                "/World/OR/TableBase", [tx, ty, 0.035], [0.95, 0.60, 0.07], "stainless", 0.02
            )
            self.cylinder("/World/OR/TableColumn", [tx, ty, 0.42], 0.075, 0.70, "stainless")
            self.rounded(
                "/World/OR/TableFrame", [tx, ty, 0.80], [1.95, 0.52, 0.06], "stainless", 0.015
            )
            self.rounded(
                "/World/OR/Mattress", [tx, ty, 0.865], [1.90, 0.56, 0.07], "sterile_blue", 0.03
            )
        self.surgical_patient(tx, ty, top)
        # Surgical field: a blue fenestrated drape with the prepped skin showing, on the abdomen.
        field_z = top + 0.004 + 0.17
        self.rounded(
            "/World/OR/Field",
            [tx + 0.10, ty, field_z + 0.002],
            [0.30, 0.28, 0.006],
            "sterile_blue",
            0.002,
        )
        self.cube(
            "/World/OR/FieldWindow",
            [tx + 0.10, ty, field_z + 0.0055],
            [0.12, 0.10, 0.003],
            "betadine",
        )
        # Ether screen between the surgical field and the anesthesia side.
        for k, dy in enumerate((-0.30, 0.30)):
            self.cylinder(
                f"/World/OR/ScreenPole_{k}", [tx + 0.62, ty + dy, 1.10], 0.008, 0.46, "stainless"
            )
        self.cube(
            "/World/OR/ScreenSheet", [tx + 0.62, ty, 1.22], [0.006, 0.60, 0.26], "sterile_blue"
        )
        # Surgical light on a ceiling arm, off to the side so it never hides the call board.
        lx = tx + 0.80
        self.cylinder("/World/OR/LampRod", [lx, ty, 2.10], 0.025, 0.60, "stainless")
        self.cylinder("/World/OR/LampHead", [lx, ty, 1.80], 0.26, 0.07, "lamp_white")
        self.cylinder("/World/OR/LampLens", [lx, ty, 1.763], 0.20, 0.006, "clock_face")
        for i in range(8):
            angle = 2 * np.pi * i / 8
            self.sphere(
                f"/World/OR/LampLed_{i}",
                [lx + 0.15 * np.cos(angle), ty + 0.15 * np.sin(angle), 1.758],
                0.018,
                "clock_face",
            )
        # Anesthesia cart at the head of the table, with a monitor and gas bottles.
        ax, ay = tx + 1.35, ty
        self.cube("/World/OR/AnesthesiaCart", [ax, ay, 0.50], [0.52, 0.46, 1.00], "cabinet_white")
        self.cube("/World/OR/AnesthesiaTop", [ax, ay, 1.03], [0.54, 0.48, 0.05], "stainless")
        self.cube(
            "/World/OR/AnesthesiaMonitor", [ax, ay + 0.03, 1.30], [0.34, 0.03, 0.26], "rubber"
        )
        self.cube(
            "/World/OR/AnesthesiaScreen", [ax, ay + 0.048, 1.30], [0.30, 0.004, 0.22], "screen_dark"
        )
        for k, dz in enumerate((0.07, 0.0, -0.07)):
            self.cube(
                f"/World/OR/Trace_{k}",
                [ax - 0.02, ay + 0.052, 1.30 + dz],
                [0.24, 0.002, 0.008],
                "screen_green",
            )
        for k, color in enumerate(("glove_blue", "gas_yellow", "gas_green")):
            self.cylinder(
                f"/World/OR/Gas_{k}", [ax - 0.15 + 0.15 * k, ay - 0.10, 1.13], 0.035, 0.18, color
            )
        # IV pole with a bag beside the table.
        px, py = tx - 1.20, ty - 0.20
        self.cylinder("/World/OR/IVPole", [px, py, 0.95], 0.010, 1.90, "stainless")
        self.cylinder("/World/OR/IVBase", [px, py, 0.015], 0.20, 0.03, "stainless")
        self.cube("/World/OR/IVBag", [px + 0.05, py, 1.65], [0.10, 0.03, 0.17], "iv_bag")

    def wall_clock(self, x, y, z):
        self.cylinder("/World/OR/ClockRim", [x, y, z], 0.13, 0.02, "stainless", axis="Y")
        self.cylinder(
            "/World/OR/ClockFace", [x, y + 0.012, z], 0.115, 0.006, "clock_face", axis="Y"
        )
        self.cube("/World/OR/ClockHour", [x, y + 0.017, z + 0.03], [0.008, 0.003, 0.06], "rubber")
        self.cube(
            "/World/OR/ClockMinute", [x + 0.035, y + 0.017, z], [0.07, 0.003, 0.006], "rubber"
        )

    def equipment(self, z, sharps_x=0.10, pump=True):
        # Sharps container on the back counter.
        self.cube(
            "/World/Equipment/SharpsBin",
            [sharps_x, -0.66, 0.7675 + 0.10],
            [0.12, 0.10, 0.20],
            "sharps_red",
        )
        self.cube(
            "/World/Equipment/SharpsLid",
            [sharps_x, -0.66, 0.7675 + 0.205],
            [0.125, 0.105, 0.020],
            "sharps_lid",
        )
        self.label(
            "/World/Equipment/SharpsLabel",
            "SHARPS",
            [sharps_x, -0.608, 0.7675 + 0.10],
            [0.10, 0.04],
        )
        if not pump:
            return
        # Infusion pump on the bench (decor; clutter keeps clear of DECOR_KEEPOUTS).
        self.cube(
            "/World/Equipment/Pump", [0.86, -0.32, z + 0.085], [0.15, 0.21, 0.17], "cabinet_white"
        )
        self.cube(
            "/World/Equipment/PumpTop",
            [0.86, -0.32, z + 0.177],
            [0.155, 0.21, 0.014],
            "sterile_blue",
        )
        self.cube(
            "/World/Equipment/PumpBezel", [0.86, -0.214, z + 0.115], [0.102, 0.010, 0.053], "rubber"
        )
        self.cube(
            "/World/Equipment/PumpScreen",
            [0.86, -0.208, z + 0.115],
            [0.085, 0.002, 0.036],
            "screen_dark",
        )
        self.label(
            "/World/Equipment/PumpReadout",
            "RATE 125",
            [0.86, -0.2065, z + 0.115],
            [0.076, 0.026],
            [0.05, 0.45, 0.45],
        )
        self.label(
            "/World/Equipment/PumpName", "INFUSION", [0.86, -0.208, z + 0.055], [0.08, 0.028]
        )
        for i, x in enumerate([0.802, 0.918]):
            for j, y in enumerate([-0.39, -0.25]):
                self.cube(
                    f"/World/Equipment/PumpFoot_{i}_{j}",
                    [x, y, z + 0.007],
                    [0.017, 0.025, 0.014],
                    "rubber",
                )

    def props(self, layout=None):
        """Privacy curtain behind the robot and an IV pole, both outside the work area."""
        for i in range(8):
            y = -0.78 + i * 0.14
            self.cube(
                f"/World/Room/Curtain_{i}",
                [-0.55 + (0.035 if i % 2 else -0.035), y, 1.12],
                [0.03, 0.15, 1.76],
                "curtain_dark" if i % 2 else "curtain",
            )
        self.cube("/World/Room/CurtainTrack", [-0.55, -0.22, 2.02], [0.04, 1.16, 0.03], "steel")
        # IV stand: the real model when installed, otherwise a pole with a hanging bag. On the common
        # rig (hospital/rig.py) the idle arm stands at robot y ~0.6, where the stand used to be:
        # the stand then moves along the curtain line to 0.40 m beyond the rig's +y column.
        from roboclock.envs.hospital.props import add_model, model_size

        sy = 0.62
        if layout is not None and "rig" in self.config.get("robot", {}):
            from roboclock.envs.lab.robot import mount_positions

            mounts = mount_positions(self.config, layout.task_id)
            sy = max(sy, max(float(m[1]) for m in mounts) + 0.40)
        if model_size("iv_stand") is not None:
            add_model(self.stage, "iv_stand", "/World/Props/IVStand", (-0.22, sy, 0.0))
            return
        self.cylinder("/World/Props/IVPole", [-0.22, sy, 0.90], 0.010, 1.80, "stainless")
        self.cylinder("/World/Props/IVBase", [-0.22, sy, 0.015], 0.17, 0.03, "stainless")
        self.cube("/World/Props/IVArm", [-0.22, sy, 1.78], [0.20, 0.014, 0.014], "stainless")
        self.cube("/World/Props/IVBag", [-0.30, sy, 1.64], [0.10, 0.03, 0.19], "iv_bag")
        self.label("/World/Props/IVLabel", "NS 0.9%", [-0.30, sy - 0.017, 1.64], [0.085, 0.04])
