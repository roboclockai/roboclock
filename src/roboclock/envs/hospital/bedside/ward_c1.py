"""C1 apparatus on the scene-02 ward counter: two stock risers of color-coded vials, four RX order
bins with order cards and a shift board (countdown bar plus filled/wrong readouts), worked by both
X5 arms of one mobile base, each in its own half of the counter.

Assets: the vial mesh is the local CC-BY-4.0 "Empty Vial" by Feras Sultan
(`assets/models/vial/vial.usd`, see `models/credits.json`), mounted on a rigid cylinder collider. The
model ships flat untextured materials (matte grey body), so its glass body, crimp and flip-off cap
are re-bound here (glossy flat glass, brushed aluminum, drug-colored molded plastic) and a printed
label sleeve in the drug color is added. Risers, bins, card holders, board and the mobile base are
primitives with CC0 PBR texture sets (hospital/assets/textures). Nothing here moves a vial or
scores anything at runtime.
"""

import math

import numpy as np

from roboclock.envs.hospital.tasks.ward.pharmacy_order_bins import (
    BAR,
    COUNT,
    ERRORS,
    MAX_ERRORS_SHOWN,
)

VIAL = "c1_rx_vial"
VIAL_SPEC = {
    "proxy": "cylinder",
    "usd_path": None,
    "scale": [1.0, 1.0, 1.0],
    "radius": 0.018,
    "height": 0.074,
    "grasp_radius": 0.018,
    "grip_force": 8.0,
    "mass_range": [0.025, 0.040],
    "min_friction": 0.8,
}
# Drug name and color; the color is on the cap, the label sleeve, the shelf tag and the RX card.
DRUGS = (
    ("METOPROLOL", (0.90, 0.15, 0.12)),
    ("LISINOPRIL", (0.12, 0.38, 0.95)),
    ("FUROSEMIDE", (0.12, 0.72, 0.28)),
    ("ASPIRIN", (0.98, 0.78, 0.10)),
)
COLOR_NAMES = ("RED", "BLUE", "GREEN", "YELLOW")
RACK_TIER = 0.040  # stock riser height above the counter
# Two-arm layout, mirrored about robot y = 0 (the rig origin). The common rig (hospital/rig.py)
# stands in front of the counter with X5 side 0 at -y and side 1 at +y, 0.60 m apart. Each arm owns
# one half: a five-vial stock riser against the wall and two RX bins in front of it, so the arms
# never share a zone. Offsets are from the arm's own mount, with +dy pointing away from the centre
# line. The top-down X5 hand reaches
# about 0.5 m from its mount and drifts nearer than ~0.27 m, so everything sits in that ring.
STOCK_X = 0.515
STOCK_DY = (-0.10, -0.032, 0.036, 0.104, 0.172)
BIN_X, BIN_DY = 0.40, (-0.0675, 0.0675)
BIN_HALF, BIN_WALL = (0.06, 0.06), 0.055
SLOTS = ((0.0, -0.028), (0.0, 0.028), (0.0, 0.0))  # where vials are set down inside a bin
CARRY = 0.20  # tool height above the table while carrying
READY = (0.24, 0.0, 0.26)  # rest point: x, outward dy from the mount, height above the table
WALL_T = 0.006  # bin wall thickness (layout.WALL)
# Shift board against the wall between the two stock risers, facing the rig's head camera. RoboDojo
# rig (2026-11-03): at (0.40, -0.70), past side 0's riser, it stood beyond the right edge of the
# 60-degree head view; here no hand goes (the hands stay within x 0.56 and |y| >= 0.17).
BOARD_XY = (0.67, 0.0)
# The real ward counter (preview left bay, blue base cabinets): ward x 2.88 front .. 3.44 wall,
# top z 0.82. This pose puts its front edge at robot x 0.20, the wall at 0.76, the top at the
# table height 0.75, and the window-end section (ward y -1.9) at robot y 0; the stethoscope
# (ward y -1.0) lies past the +y bins. The ultrasound cart in front of this section (ward
# y -2.6..-1.7) would stand where the -y arm's base goes, so it is hidden (scenery only).
CART = [f"Obj24of_591669_{k}_{suffix}" for k, suffix in (
    (4, "386"), (5, "498"), (6, "853"), (7, "182"), (8, "786"), (9, "616"), (10, "520"),
    (11, "051"), (12, "779"), (13, "830"), (14, "113"))] + ["iMac_5k_Monitor054"]
WARD_POSE = {"usd": "assets/hospital/ward_02_preview_left.usdc", "translate": [-2.68, 1.9, -0.07],
             "yaw": 0.0, "hide": CART}
COUNTER = {"front_x": 0.20, "wall_x": 0.76, "y": (-0.95, 0.80)}
# A raised 3/4 view from the room at the window (-y) end, like a ward ceiling camera. Checked by
# ray casting: no sight line to the bins, RX cards, stock or board crosses a mount column. Any
# lower room-side view puts a column in front of a bin, since the base stands between the room and
# the counter; only a view along the counter edge avoids that, and it hides the far arm's half.
CAMERA = {"position": [-0.20, -0.65, 2.25], "look_at": [0.44, -0.02, 0.84], "focal_mm": 34.0}


def side_of(y):
    """Which arm owns a counter position: 0 (the rig's -y arm) or 1 (its +y arm)."""
    return 0 if y < 0 else 1


def build_layout(b):
    """Seeded stock and orders: each arm's two bins are filled from its own stock riser."""
    from roboclock.envs.lab.robot import mount_positions

    b.catalog.setdefault(VIAL, dict(VIAL_SPEC))
    rng, z, n = b.rng, b.z, b.task.spec.num_items
    per_side = len(STOCK_DY)
    if n % 2 or not 4 <= n <= 2 * (per_side - 1):
        raise ValueError(f"C1 needs an even 4..{2 * (per_side - 1)} order items, got {n}")
    b.shift = b.shift * 0.4  # the far vials sit near the edge of each hand's reach
    mounts = [m[:2] for m in mount_positions(b.config, b.task.task_id)]
    center_y = (mounts[0][1] + mounts[1][1]) / 2
    kinds = [int(k) for k in rng.permutation(np.arange(2 * per_side) % len(DRUGS))]
    sides = {}
    for side, (_, my) in enumerate(mounts):
        out = 1.0 if my > center_y else -1.0  # away from the centre line
        items = [f"item_{side * per_side + k}" for k in range(per_side)]
        bins = [f"bin_{2 * side + k}" for k in range(2)]
        mid = my + out * (STOCK_DY[0] + STOCK_DY[-1]) / 2
        b.fixture(f"c1_rack_{side}", (STOCK_X, mid), (0.09, 0.34, RACK_TIER), "cabinet_white")
        for name, dy in zip(items, STOCK_DY):
            drug, color = DRUGS[kinds[int(name.split("_")[1])]]
            b.item(name, VIAL, (STOCK_X, my + out * dy), z + RACK_TIER, color=color, label=drug,
                   yaw=0.0)
        # RX 1..4 run from side 0's outer bin (-y) to side 1's outer bin (+y).
        for name, dy in zip(bins, sorted(BIN_DY, reverse=side == 0)):
            b.target(name, "bin", (BIN_X, my + out * dy), BIN_HALF, BIN_WALL)
        # This arm's orders: n/2 of its five vials, dealt round-robin to its two bins.
        chosen = [items[int(i)] for i in rng.permutation(per_side)[: n // 2]]
        for k, name in enumerate(chosen):
            b.order_bins[name] = bins[k % 2]
        sides[str(side)] = {"items": items, "bins": bins}
    b.types = tuple((VIAL, color, name) for name, color in DRUGS)
    b.kinds = kinds
    b.info.update(
        bedside_task="ward_c1",
        c1={
            "drugs": [name for name, _ in DRUGS],
            "carry_z": float(z + CARRY),
            "board": [*map(float, b.xy(*BOARD_XY))],
            "camera": CAMERA,
            "floor_z": 0.0,  # the visible floor is the terrazzo ground plane (ward slab hidden)
            "sides": sides,
        },
        ward_pose={**WARD_POSE, "hide": list(CART)},
        # Both arms start (and rest when done) tucked up just in front of their mount, clear of the
        # bins in the camera view (arm 1's pose is applied by measure() at reset).
        initial_ee_position=[READY[0], float(mounts[0][1] + READY[1]), float(z + READY[2])],
        initial_secondary_ee_position=[READY[0], float(mounts[1][1] - READY[1]), float(z + READY[2])],
        initial_ee_yaw=0.0,
        initial_secondary_ee_yaw=0.0,
        initial_gripper_width=float(VIAL_SPEC["radius"] + 0.010),
        hide_bench=True,
        work_surface={
            "center": [(COUNTER["front_x"] + COUNTER["wall_x"]) / 2, sum(COUNTER["y"]) / 2, z - 0.05],
            "size": [COUNTER["wall_x"] - COUNTER["front_x"], COUNTER["y"][1] - COUNTER["y"][0], 0.10],
        },
    )
    return b


# --- appearance --------------------------------------------------------------------------------


def _skins(look):
    """Textured PBR materials for the C1 props; color-coded parts keep their color as the tint."""
    t = look.textured
    return {
        # Stock risers: pale grey-blue molded matte plastic, set off from the white counter.
        "riser": t("c1_riser", "plastic_matte", color=(0.86, 0.90, 0.95)),
        # RX bins: neutral grey molded polypropylene, so the drug colors inside read clearly.
        "bin": t("c1_bin", "plastic_white", color=(0.72, 0.76, 0.80), roughness=0.42),
        # RX card holders and the shift board housing: dark matte plastic.
        "holder": t("c1_holder", "plastic_matte", color=(0.20, 0.21, 0.23)),
        # One mobile robot look: the ARXMount shells' own light grey matte plastic, a dark
        # powder-coated frame and skirt, rubber casters and brushed aluminum trim.
        "robot_shell": t("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90)),
        "robot_frame": t("robot_frame_dark", "powder_coat_dark"),
    }


def _rebind(stage, path, material):
    """Bind a material to a layout fixture's or target's visual (`<path>_Visual` if it exists)."""
    from pxr import UsdShade

    for candidate in (path + "_Visual", path):
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material)
            return prim
    return None


def _ring(stage, path, radius, z0, z1, material, thickness=0.0006):
    """A smooth thin sleeve (96 segments): a USD Cylinder is tessellated coarsely enough that the
    vial's glass pokes through its flat facets."""
    from pxr import UsdShade

    from roboclock.simulators.isaac.visual_geometry import (
        author_mesh,
        revolved,
        smooth_revolved_normals,
    )

    inner = radius - thickness
    profile = [(inner, z0), (radius, z0), (radius, z1), (inner, z1)]
    mesh = author_mesh(stage, path, revolved(profile))
    smooth_revolved_normals(mesh, profile)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material.material)
    return mesh


def _texture(key, draw_fn, size):
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"c1_{hashlib.sha256(key.encode()).hexdigest()[:16]}.png"
    if not path.exists():
        image = Image.new("RGB", size, (12, 20, 30))
        draw_fn(ImageDraw.Draw(image), image)
        image.save(path)
    return path


def _font(size, bold=True):
    from pathlib import Path

    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    file = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(file), size) if file.exists() else ImageFont.load_default(size=size)


def _rgb(color):
    return tuple(round(255 * float(c)) for c in color)


def rx_card_texture(index, rows):
    """Order screen: 'RX n' header, then one row per drug: count, a big color swatch and the name."""

    def draw(d, image):
        w, h = image.size
        d.rectangle((0, 0, w, 96), fill=(30, 95, 150))
        d.text((w / 2, 48), f"RX {index + 1}", fill=(255, 255, 255), font=_font(76), anchor="mm")
        step = (h - 104) / max(2, len(rows))
        for k, (count, name, color) in enumerate(rows):
            y = 104 + step * (k + 0.5)
            d.text((22, y), f"{count}x", fill=(255, 255, 255), font=_font(84), anchor="lm")
            d.rounded_rectangle((150, y - 62, 290, y + 62), radius=18, fill=_rgb(color))
            f = 70
            while f > 28 and d.textlength(name, font=_font(f)) > w - 330:
                f -= 2
            d.text((312, y), name, fill=_rgb(color), font=_font(f), anchor="lm")

    return _texture(f"rx2|{index}|{rows!r}", draw, (720, 460))


def text_texture(text, color, size=(640, 128), font=72, background=(12, 20, 30)):
    def draw(d, image):
        w, h = image.size
        if background != (12, 20, 30):
            d.rectangle((0, 0, w, h), fill=background)
        f = font
        while f > 20 and d.textlength(text, font=_font(f)) > w - 24:
            f -= 2
        d.text((w / 2, h / 2), text, fill=_rgb(color), font=_font(f), anchor="mm")

    return _texture(f"txt|{text}|{color!r}|{size!r}|{font}|{background!r}", draw, size)


def _card(stage, path, texture, center, size):
    """A textured quad in the parent's local xz plane, readable from local +y."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    material = UsdShade.Material.Define(stage, path + "_material")
    shader = UsdShade.Shader.Define(stage, path + "_material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    tex = UsdShade.Shader.Define(stage, path + "_material/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, path + "_material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    # Self-lit a little, like a screen, so the text stays legible in the dim ward.
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, path)
    w, h = size
    mesh.CreatePointsAttr(
        [Gf.Vec3f(w / 2, 0, -h / 2), Gf.Vec3f(w / 2, 0, h / 2),
         Gf.Vec3f(-w / 2, 0, h / 2), Gf.Vec3f(-w / 2, 0, -h / 2)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([Gf.Vec2f(0, 0), Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0)])
    mesh.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _group(stage, path, center, yaw_deg=0.0, tilt_deg=0.0):
    """Local frame at `center`: rotated about z by yaw, then tilted back about local x."""
    from pxr import Gf, UsdGeom

    group = UsdGeom.Xform.Define(stage, path)
    group.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    group.AddRotateZOp().Set(float(yaw_deg))
    group.AddRotateXOp().Set(float(tilt_deg))
    return group


def _facing(camera, point):
    """Yaw (degrees) that turns local +y toward the camera, in the horizontal plane."""
    dx, dy = camera[0] - point[0], camera[1] - point[1]
    return math.degrees(math.atan2(-dx, dy))


class ShiftBar:
    """A horizontal countdown bar on the shift board (local frame).

    Seen from the board's front (local +y), local +x is on the viewer's left; the bar stays anchored
    there and shrinks toward it as the shift runs out.
    """

    def __init__(self, stage, look, root, center, length, height):
        from pxr import UsdGeom, UsdShade

        from roboclock.envs.hospital.timers import COLORS

        self.center, self.length, self.height = np.asarray(center, float), length, height
        self.materials = {
            key: look.material(f"timer_{key}", color=color, roughness=0.4, emissive=True)
            for key, color in COLORS.items()
        }
        look.rounded(root + "/BarTrack", self.center + [0, -0.001, 0],
                     [length + 0.012, 0.004, height + 0.012], "rubber", 0.002)
        self.bar = UsdGeom.Cube.Define(stage, root + "/Bar")
        self.bar.CreateSizeAttr(1.0)
        self.translate = self.bar.AddTranslateOp()
        self.scale = self.bar.AddScaleOp()
        self.color = None
        self._binding = UsdShade.MaterialBindingAPI.Apply(self.bar.GetPrim())
        self.set(1.0, "green")

    def set(self, fraction, color):
        from pxr import Gf

        fraction = min(max(float(fraction), 0.0), 1.0)
        width = max(fraction * self.length, 1e-4)
        x_left = self.center[0] + self.length / 2  # the viewer's left end
        self.translate.Set(Gf.Vec3d(x_left - width / 2, self.center[1] + 0.002, self.center[2]))
        self.scale.Set(Gf.Vec3f(width, 0.004, self.height))
        if color != self.color:
            self.color = color
            self._binding.Bind(self.materials[color].material)

    def state(self, rgb):
        pass


class Readout:
    """Prebuilt text cards at one spot; `set(k)` slides card k to the front and the rest behind.

    Moving the cards (a transform change) shows reliably in RTX captures, unlike toggling USD
    visibility of these meshes, which the renderer did not pick up during an episode.
    """

    def __init__(self, cards, center, hidden_offset=-0.02):
        self.ops = [card.GetOrderedXformOps()[0] for card in cards]
        self.center = np.asarray(center, float)
        self.hidden = self.center + [0.0, hidden_offset, 0.0]
        self.value = None
        self.set(0)

    def set(self, value, color=None):
        from pxr import Gf

        value = int(round(value))
        if value == self.value:
            return
        self.value = value
        for k, op in enumerate(self.ops):
            op.Set(Gf.Vec3d(*map(float, self.center if k == value else self.hidden)))

    def state(self, rgb):
        pass


def _vial_visual(backend, name, obj):
    from pxr import UsdGeom, UsdShade

    from roboclock.envs.hospital.props import add_model, model_size

    stage, look = backend.stage, backend.appearance
    root = f"/World/Objects/{name}"
    prim = stage.GetPrimAtPath(root)
    for child in list(prim.GetChildren()):
        if child.GetName() != "Body":  # keep the rigid cylinder collider
            stage.RemovePrim(child.GetPath())
    radius, height = VIAL_SPEC["radius"], VIAL_SPEC["height"]
    color = [float(c) for c in obj.color]
    # Flip-off cap: molded plastic in the drug color; label sleeve: printed paper in the same color.
    cap = look.textured(f"c1_cap_{obj.label}", "plastic_white", color=color, roughness=0.35)
    sleeve = look.textured(f"c1_sleeve_{obj.label}", "paper", color=color)
    glass = look.material("c1_vial_glass", color=[0.84, 0.90, 0.93], roughness=0.05, metallic=0.0)
    native = model_size("vial")
    if native is not None:
        s = 2 * radius / native[0]
        add_model(stage, "vial", root + "/Model", (0.0, 0.0, -height / 2), (s, s, height / native[2]))
        # The model's parts carry flat untextured materials: part_002 is the glass body, part_001
        # the crimp seal and part_000 the flip-off cap. The glass stays a flat glossy material: the
        # refractive OmniGlass rendered the empty vials black under the ward's RTX lighting.
        for part, material in (("part_000", cap), ("part_001", look.material("aluminum")),
                               ("part_002", glass)):
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{root}/Model/{part}")).Bind(
                material.material, UsdShade.Tokens.strongerThanDescendants
            )
    else:  # primitive proxy if the model pack is missing
        look.cylinder(root + "/Glass", [0, 0, -0.006], radius, height - 0.012, glass)
        look.cylinder(root + "/Cap", [0, 0, height / 2 - 0.006], radius * 0.95, 0.012, cap)
    # Colored label sleeve with a white band, readable as the drug color from any side.
    _ring(stage, root + "/Sleeve", radius + 0.0007, -0.024, 0.016, sleeve)
    _ring(stage, root + "/SleeveBand", radius + 0.0010, -0.0075, -0.0005, look.material("label_paper"))
    UsdGeom.Imageable(stage.GetPrimAtPath(root + "/Body")).MakeInvisible()


def author(backend):
    """Visuals only: vial meshes, shelf tags, RX cards and the shift board. No joints, no motion."""
    stage, look, layout, task = backend.stage, backend.appearance, backend.layout, backend.task
    z = layout.table_height
    camera = layout.info["c1"]["camera"]["position"]
    skins = _skins(look)
    _pedestals(backend, skins)
    for name, obj in layout.objects.items():
        _vial_visual(backend, name, obj)
    for side in (0, 1):
        _rebind(stage, f"/World/Fixtures/c1_rack_{side}", skins["riser"])
    for bin_name in layout.targets:
        for part in ("floor", "left", "right", "front", "back"):
            _rebind(stage, f"/World/Targets/{bin_name}/{part}_Visual", skins["bin"])
    base = "/World/WardC1"
    # Stock risers: a low backboard against the wall with a header turned toward the camera, and a
    # colored drug tag on the riser's room-side edge in front of each vial.
    for side in (0, 1):
        rack = next(f for f in layout.fixtures if f.name == f"c1_rack_{side}")
        back_x = rack.position[0] + rack.size[0] / 2
        front_x = rack.position[0] - rack.size[0] / 2
        ry, length = rack.position[1], rack.size[1]
        look.rounded(f"{base}/RackBack_{side}", [back_x + 0.006, ry, z + 0.075],
                     [0.012, length, 0.15], skins["riser"], 0.003)
        for k, end in enumerate((-1, 1)):
            look.rounded(f"{base}/RackSide_{side}_{k}",
                         [rack.position[0], ry + end * (length / 2 + 0.006), z + 0.035],
                         [rack.size[0], 0.012, 0.07], skins["riser"], 0.003)
        header = _group(stage, f"{base}/RackHeader_{side}", [back_x - 0.004, ry, z + 0.125],
                        _facing(camera, (back_x, ry)), 0.0)
        _card(stage, header.GetPath().pathString + "/Card",
              text_texture(f"STOCK {'A' if side == 0 else 'B'}", (0.95, 0.95, 0.95),
                           background=(30, 95, 150)), [0, 0, 0], (0.16, 0.04))
        top = rack.position[2] + rack.size[2] / 2
        for name in layout.info["c1"]["sides"][str(side)]["items"]:
            obj = layout.objects[name]
            tag = _group(stage, f"{base}/Tag_{name}",
                         [front_x - 0.0008, obj.position[1], top - 0.018], 90.0)
            _card(stage, tag.GetPath().pathString + "/Card",
                  text_texture(obj.label, (1, 1, 1), size=(512, 128), background=_rgb(obj.color)),
                  [0, 0, 0], (0.064, 0.030))
    # RX order cards: a tilted screen in front of each bin, low enough not to hide its contents.
    for j, (bin_name, target) in enumerate(sorted(layout.targets.items())):
        label = stage.GetPrimAtPath(f"/World/Labels/{bin_name}")
        if label.IsValid():
            stage.RemovePrim(label.GetPath())
        rows = []
        for key, count in sorted(layout.orders.get(bin_name, {}).items()):
            drug = key.split(":")[1]
            index = [d for d, _ in DRUGS].index(drug)
            rows.append((count, drug, DRUGS[index][1]))
        cx, cy = target.position[:2]
        # The order card stands on the bin's back wall like a shelf-edge label holder, facing the
        # room and leaning back a little: behind the bin it is clear of the arm roots' shadow in
        # the camera view and low enough not to hide the stock vials behind it.
        back = cx + target.half_extents[0] + WALL_T + 0.002
        height = 0.072
        card = _group(stage, f"{base}/Rx_{j}", [back, cy, target.rim_z + height / 2 - 0.012],
                      100.0, 10.0)
        path = card.GetPath().pathString
        look.rounded(path + "/Bezel", [0, -0.003, 0], [0.122, 0.005, height + 0.006], skins["holder"],
                     0.002)
        _card(stage, path + "/Screen", rx_card_texture(j, rows), [0, 0.0005, 0], (0.116, height))
    # Shift board: header, countdown bar and the filled / wrong readouts.
    bx, by = layout.info["c1"]["board"]
    spec = task.spec
    from roboclock.envs.hospital.rig import head_camera_pose

    head = head_camera_pose(backend.config, backend.task.task_id)[0]
    # Low and tilted back 55 degrees, square to the head camera: the wall unit over the counter
    # hides anything above ~0.91 m this far back.
    board = _group(stage, base + "/ShiftBoard", [bx, by, z + 0.075], _facing(head, (bx, by)), 55)
    root = board.GetPath().pathString
    look.rounded(f"{base}/BoardPost", [bx, by - 0.012, z + 0.02], [0.018, 0.018, 0.04],
                 "stainless", 0.002)
    look.rounded(f"{base}/BoardFoot", [bx, by - 0.012, z + 0.003], [0.11, 0.08, 0.006],
                 "stainless", 0.002)
    width, height = 0.30, 0.19
    look.rounded(root + "/Bezel", [0, -0.006, 0], [width + 0.012, 0.010, height + 0.012],
                 skins["holder"], 0.005)
    look.rounded(root + "/Face", [0, -0.0005, 0], [width, 0.001, height], "screen_dark", 0.002)
    need = task.jobs_required  # every ordered vial (score_version 2: no time on the board)
    _card(stage, root + "/Header",
          text_texture(f"GOAL {need} VIALS", (0.45, 0.85, 1.0)),
          [0, 0.001, 0.071], (0.285, 0.040))
    backend.timer_bars[BAR] = ShiftBar(stage, look, root, [0, 0.001, 0.030], 0.27, 0.026)
    cards = []
    for k in range(spec.num_items + 1):
        color = (0.3, 1.0, 0.45) if k >= need else (1.0, 0.95, 0.85)
        cards.append(_card(stage, f"{root}/Count_{k}", text_texture(f"FILLED {k} / {need}", color),
                           [0, 0.0015, -0.020], (0.285, 0.052)))
    backend.timer_bars[COUNT] = Readout(cards, [0, 0.0015, -0.020])
    cards = []
    for k in range(MAX_ERRORS_SHOWN + 1):
        text = f"WRONG {k}{'+' if k == MAX_ERRORS_SHOWN else ''}"
        color = (0.55, 0.9, 0.6) if k == 0 else (1.0, 0.3, 0.25)
        cards.append(_card(stage, f"{root}/Errors_{k}", text_texture(text, color),
                           [0, 0.0015, -0.068], (0.285, 0.036)))
    backend.timer_bars[ERRORS] = Readout(cards, [0, 0.0015, -0.068])


def _pedestals(backend, skins=None):
    """The X5 mount columns stop at counter height: extend both to the floor on one mobile base.

    Same robot look as the bedside tasks: light grey matte shells (the ARXMount shells' own
    material, so each pedestal reads as one column up to the arm), a dark powder-coated frame and
    skirt, rubber casters, aluminum trim and an aluminum collar under each arm column.
    """
    from roboclock.envs.lab.robot import mount_positions

    look, layout = backend.appearance, backend.layout
    skins = skins or _skins(look)
    shell, frame = skins["robot_shell"], skins["robot_frame"]
    floor, top = layout.info["c1"]["floor_z"], layout.table_height
    mounts = mount_positions(backend.config, backend.task.task_id)
    for side, (x, y, _) in enumerate(mounts):
        look.rounded(f"/World/WardC1/Pedestal_{side}", [x, y, (floor + top) / 2],
                     [0.12, 0.12, top - floor], shell, 0.004)
        look.rounded(f"/World/WardC1/Collar_{side}", [x, y, top], [0.126, 0.126, 0.014],
                     "aluminum", 0.003)
    xs, ys = [m[0] for m in mounts], [m[1] for m in mounts]
    cx, cy, width = min(xs) - 0.05, (min(ys) + max(ys)) / 2, max(ys) - min(ys) + 0.30
    look.rounded("/World/WardC1/Base", [cx, cy, floor + 0.05], [0.26, width, 0.08], shell, 0.01)
    look.rounded("/World/WardC1/BaseSkirt", [cx, cy, floor + 0.022], [0.272, width + 0.012, 0.026],
                 frame, 0.008)
    look.rounded("/World/WardC1/BaseTrim", [cx, cy, floor + 0.088], [0.264, width + 0.004, 0.008],
                 "aluminum", 0.003)
    look.rounded("/World/WardC1/BaseSpine", [min(xs), cy, top - 0.10],
                 [0.08, max(ys) - min(ys), 0.06], frame, 0.004)
    for k, (dx, dy) in enumerate(((-0.10, -1), (-0.10, 1), (0.06, -1), (0.06, 1))):
        wheel = [cx + dx, cy + dy * (width / 2 - 0.03), floor + 0.025]
        look.cylinder(f"/World/WardC1/Wheel_{k}", wheel, 0.025, 0.02, "rubber", axis="Y")


def _camera(backend):
    """Room-side view of the robot at the counter, set once per reset (the shared camera is fixed)."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["c1"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    if "focal_mm" in view:
        camera.set_focal_length(view["focal_mm"] / 1000)
    for _ in range(3):
        backend.world.render()


def _ready_secondary(backend):
    """Reset-time ready pose for internal arm 1 (secondary, the rig's left X5), mirroring what the
    backend does for arm 0 (the rig's right X5).

    The X5 home posture stands upright in front of arm 0's bins in the camera view, so arm 1
    starts low over its own outer bin instead. This happens once, at reset, before the
    episode clock starts; nothing moves the arm afterwards except its joint commands.
    """
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    from roboclock.envs.lab.robot import DOF, JOINT_HIGH, JOINT_LOW, make_solver
    from roboclock.envs.start_pose import zero_start

    info, robot = backend.layout.info, getattr(backend, "secondary_robot", None)
    if robot is None or "initial_secondary_ee_position" not in info:
        return
    if zero_start(backend.config):
        # robot.start_pose: zero. The arm stays at joint 0 (lab.robot.home_joints); only the
        # task's start gripper width applies.
        q = robot.get_joint_positions().copy()
        q[6:] = info.get("initial_gripper_width", 0.04)
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros(DOF))
        hold = np.concatenate([backend.robot.get_joint_positions(), q])
        backend.initial_hold_joints = hold
        backend.apply_action(hold)
        for _ in range(10):
            backend.world.step(render=False)
        return
    action, solved = make_solver(robot).compute_inverse_kinematics(
        target_position=np.asarray(info["initial_secondary_ee_position"]),
        target_orientation=euler_angles_to_quat(
            np.array([0.0, np.pi, info.get("initial_secondary_ee_yaw", 0.0)])
        ),
    )
    if not solved:
        raise RuntimeError("C1: prescribed right-arm ready pose is unreachable")
    q = robot.get_joint_positions().copy()
    indices = action.joint_indices if action.joint_indices is not None else range(DOF)
    for i, value in zip(indices, action.joint_positions):
        if value is not None:
            q[i] = value
    q[6:] = info.get("initial_gripper_width", 0.04)
    if np.any(q < JOINT_LOW - 0.001) or np.any(q > JOINT_HIGH + 0.001):
        raise RuntimeError("C1: right-arm ready pose exceeds joint limits")
    robot.set_joint_positions(q)
    robot.set_joint_velocities(np.zeros(DOF))
    hold = np.concatenate([backend.robot.get_joint_positions(), q])
    backend.initial_hold_joints = hold
    backend.apply_action(hold)
    # Joint writes reach the rendered transforms only after a physics step; these settle steps
    # are part of the reset, like the backend's own settle_steps, and never advance the task clock.
    for _ in range(10):
        backend.world.step(render=False)


def measure(backend, state):
    """No device state; the first measurement after a reset sets arm 1's (rig left) ready pose and
    aims the C1 camera."""
    # The ready pose is physics, so it must not depend on a camera existing: under sim.no_render
    # (no cameras) arm 1 used to start from HOME and finish 0.04 s earlier than rendered.
    if getattr(backend, "_c1_camera_for", None) is not backend.task:
        backend._c1_camera_for = backend.task
        _ready_secondary(backend)
        _camera(backend)  # aims the task camera; returns at once without one
    return {}


# --- privileged diagnostic oracle (joint actions through the shared IK planner) ---------------


def _memory(planner, task):
    memory = getattr(planner, "c1", None)
    if memory is None or memory.get("task") is not task:
        memory = {"task": task, "phase": "select", "vial": None, "bin": None, "slot": None,
                  "since": task.time, "pending": {}, "fills": {}, "tries": {}, "skip": set()}
        planner.c1 = memory
    return memory


def _needs(task, memory):
    """Remaining need per (bin, drug), minus vials released there that have not settled yet."""
    needs = {(b, k): v for b, order in task.remaining.items() for k, v in order.items() if v > 0}
    for vial, (bin_name, since) in list(memory["pending"].items()):
        if vial in task.correct or vial in task.attempted or task.time - since > 2.5:
            memory["pending"].pop(vial)
            continue
        key = (bin_name, task.layout.item_types[vial])
        if key in needs:
            needs[key] -= 1
    return {key: v for key, v in needs.items() if v > 0}


def choose(task, state, memory, side=0, ee=None):
    """Nearest vial in this arm's stock that one of its unfilled orders still needs, and that bin.

    Each arm only touches its own half of the counter (layout.info["c1"]["sides"]), so the two arms
    never pick the same vial or reach into the same bin.
    """
    own = task.layout.info.get("c1", {}).get("sides", {}).get(str(side))
    ee = state.ee_position if ee is None else ee
    needs = {
        key: v for key, v in _needs(task, memory).items() if own is None or key[0] in own["bins"]
    }
    busy = set(memory["pending"]) | memory["skip"] | task.correct | task.dropped | task.attempted
    best = None
    for vial, obj in state.objects.items():
        if vial in busy or (own is not None and vial not in own["items"]):
            continue
        kind = task.layout.item_types[vial]
        bins = [b for (b, k) in needs if k == kind]
        if not bins:
            continue
        reach = float(np.linalg.norm(obj.position[:2] - ee[:2]))
        for bin_name in bins:
            target = task.layout.targets[bin_name].position[:2]
            cost = reach + float(np.linalg.norm(obj.position[:2] - target))
            if best is None or cost < best[0]:
                best = (cost, vial, bin_name)
    return None if best is None else best[1:]


def command(planner, task, state):
    """X5 side 0 (ARXLeft, the rig's -y arm): the -y half of the counter."""
    return _drive(planner, task, state, 0)


def command_secondary(planner, task, state):
    """X5 side 1 (ARXRight, the rig's +y arm): the +y half of the counter, at the same time."""
    return _drive(planner, task, state, 1)


def _drive(planner, task, state, side):
    memory = _memory(planner, task)
    layout = task.layout
    z_table = layout.table_height
    carry = layout.info["c1"]["carry_z"]
    own_ee = getattr(planner, "ee_position", None)
    ee = state.ee_position if own_ee is None else np.asarray(own_ee, dtype=float)
    held = getattr(planner, "held", None)
    planner.yaw = 0.0
    planner.open_width = VIAL_SPEC["radius"] + 0.010
    planner.close_width = VIAL_SPEC["radius"] - VIAL_SPEC["grip_force"] / 1000
    planner.gentle = True
    phase, vial = memory["phase"], memory["vial"]

    def go(new):
        memory["phase"], memory["since"] = new, task.time
        task.event_log.append({"time": round(task.time, 3), "type": "planner_phase",
                               "object": memory["vial"], "phase": new, "arm": side})

    elapsed = task.time - memory["since"]
    if task.time >= task.spec.max_time - 1e-9:  # the episode horizon: stop where the hand is
        return planner.finish(state, np.array([*ee[:2], carry]), False, 0.2)
    if phase == "select":
        pick = choose(task, state, memory, side, ee)
        if pick is None:
            key = "initial_ee_position" if side == 0 else "initial_secondary_ee_position"
            rest = layout.info.get(key, [*ee[:2], carry])
            if ee[2] < carry - 0.03:  # rise out of the bin before tucking away
                rest = [*ee[:2], carry]
            return planner.finish(state, np.asarray(rest, dtype=float), False, 0.3)
        memory["vial"], memory["bin"] = pick
        fills = memory["fills"].get(pick[1], 0)
        memory["slot"] = SLOTS[fills % len(SLOTS)]
        vial = pick[0]
        go("approach")
        phase = "approach"
    obj = state.objects[vial]
    in_hand = obj.grasped if held is None else vial in held
    grasp = obj.position.copy()
    target = layout.targets[memory["bin"]]
    slot = np.array([*(target.position[:2] + memory["slot"]), 0.0])
    # A vial that leaves the hand while it is carried is abandoned; the next choice re-plans.
    if phase in ("lift", "transfer", "lower") and not in_hand and np.linalg.norm(
        obj.position - ee
    ) > 0.04:
        memory["tries"][vial] = memory["tries"].get(vial, 0) + 1
        go("retreat")
        phase = "retreat"
    if phase == "approach":
        # Travel at carry height (clear of every vial top), then go straight down onto the vial.
        goal = np.array([*grasp[:2], carry])
        error = np.linalg.norm(ee[:2] - grasp[:2])
        # Near the edge of its reach an arm can settle ~1 cm short; accept that after a pause.
        if (error < 0.006 or (elapsed > 1.0 and error < 0.012)) and abs(ee[2] - carry) < 0.03:
            go("descend")
        return planner.finish(state, goal, False, 0.6)
    if phase == "descend":
        if np.linalg.norm(ee[:2] - grasp[:2]) > 0.022:
            go("approach")
        elif np.linalg.norm(ee - grasp) < 0.006 or (elapsed > 0.8 and np.linalg.norm(ee - grasp) < 0.012):
            go("grasp")
        return planner.finish(state, grasp, False, 0.35)
    if phase == "grasp":
        if in_hand and elapsed > 0.12:
            memory["offset"] = ee - obj.position
            memory["lift_xy"] = ee[:2].copy()
            go("lift")
        elif elapsed > 1.2:
            memory["tries"][vial] = memory["tries"].get(vial, 0) + 1
            if memory["tries"][vial] >= 3:
                memory["skip"].add(vial)
            go("retreat")
        return planner.finish(state, grasp, True, 0.2)
    if phase == "lift":
        goal = np.array([*memory["lift_xy"], carry])
        if ee[2] > carry - 0.035:
            go("transfer")
        return planner.finish(state, goal, True, 0.45)
    if phase == "transfer":
        goal = slot + [*memory["offset"][:2], carry]
        if np.linalg.norm(ee[:2] - goal[:2]) < 0.012:
            go("lower")
        return planner.finish(state, goal, True, 0.6)
    if phase == "lower":
        bottom = target.support_z + VIAL_SPEC["height"] / 2 + 0.008
        goal = slot + memory["offset"] + [0.0, 0.0, bottom]
        if np.linalg.norm(ee - goal) < 0.008 or (elapsed > 1.5 and np.linalg.norm(ee - goal) < 0.02):
            go("release")
        return planner.finish(state, goal, True, 0.35)
    if phase == "release":
        if elapsed > 0.18:
            memory["pending"][vial] = (memory["bin"], task.time)
            memory["fills"][memory["bin"]] = memory["fills"].get(memory["bin"], 0) + 1
            go("retreat")
        return planner.finish(state, ee.copy(), False, 0.1)
    if phase == "retreat":
        goal = np.array([*ee[:2], carry])
        if ee[2] > max(target.rim_z, z_table + 0.12) + 0.05:
            go("select")
        return planner.finish(state, goal, False, 0.5)
    raise RuntimeError(f"Unknown C1 planner phase {phase}")
