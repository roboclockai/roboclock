"""C2 instrument tray sort at the foot of the bed in the scene-02 left ward: two-arm oracle.

Instruments (all CC-BY-4.0, Wenschel, see assets/models/credits.json):
  scalpel  -> tool_scalpel model on a thin box collider,
  hemostat -> tool_clamp model on a two-box collider (narrow shaft + ring handles),
  forceps  -> the repository's procedural thumb-forceps visual (appearance.surgical_tool) on a
              thin box collider; the local "tool_forceps" mesh is a wide ring instrument, not
              tissue forceps, so it is not used.
The colliders replace the catalog boxes in `author` (before physics starts), and the same
dimensions are used for the layout shapes, so scoring and contact geometry agree.

The basin, tray and slots are static colliders; the robot moves every instrument with real
finger contact. The diagnostic `command` is privileged (reads measured poses) and only emits
Cartesian goals that the oracle turns into bounded joint targets.
"""

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roboclock.envs.hospital.mdp.geometry import Shape

# type key -> (catalog asset for mass/friction, slot label, tape/silicone color)
TYPES = {
    "scalpel": ("scalpel", "SCALPEL", (0.10, 0.42, 0.95)),
    "forceps": ("forceps", "FORCEPS", (0.96, 0.52, 0.08)),
    "hemostat": ("clamp", "HEMOSTAT", (0.12, 0.68, 0.30)),
}
TYPE_ORDER = ("scalpel", "forceps", "hemostat")
SLOT_ORDER = ("hemostat", "forceps", "scalpel")  # slots on each tray, from the robot outward (+x)
THICK = 0.012  # collider thickness of every instrument (lying flat)
# Collider half extents per type (x is the instrument's long axis).
HALF = {
    "scalpel": (0.065, 0.006, THICK / 2),
    "forceps": (0.065, 0.008, THICK / 2),
    "hemostat": (0.0604, 0.0284, THICK / 2),  # tool_clamp at native scale, rings included
}
GRASP_X = {"scalpel": 0.0, "forceps": 0.0, "hemostat": 0.006}  # along the long axis
SLOT_HALF_X = {"scalpel": 0.016, "forceps": 0.018, "hemostat": 0.036}  # slot inner half widths
SLOT_HALF_Y = 0.088
SLOT_WALL_H = 0.012
WALL = 0.006
BASIN_HALF = (0.140, 0.155)
BASIN_RIM = 0.030
TRAY_TOP = 0.030  # each Mayo tray deck sits on short rails above the stand's arm

# Bedside instrument reprocessing at the foot of the bed in the preview's true left bay
# (assets/hospital/ward_02_preview_left.usdc). The ward is shifted so the open floor at the bed's foot
# lands under the work area and the bed stands behind it in the camera view. An instrument trolley
# carries the basin; a dual-arm mobile base faces it; a Mayo stand on each side holds the tray for
# one of two back-to-back cases. Internal arm 0 (mount 0, -y: the rig's right arm) sorts the basin's -y
# row into the CASE 1 tray, arm 1 (mount 1, +y: the rig's left arm) the +y row into the CASE 2 tray,
# so the arms never cross.
WARD_USD = "assets/hospital/ward_02_preview_left.usdc"
WARD_TRANSLATE = (-0.85, -0.10, 0.0)  # bed foot at y -0.72, bed x -0.05..0.94 (behind the trays)
FLOOR_Z = WARD_TRANSLATE[2]
# The bay is open beyond y ~0.6 and x ~-0.6 (the source scene's neighbouring bay was cut away),
# so the camera stands on the open +y side and looks along -y: the bed, the ultrasound cart, the
# equipment pendant and the supply trolley fill the background instead of the empty floor. The
# privacy curtain bunched at the +x end of its ceiling track (x 1.13-1.63, y 0-0.18), right beyond
# the trolley's handle, would stand as a dark edge in that view and is hidden.
CAMERA = {"position": [0.95, 1.25, 1.80], "look_at": [0.43, -0.38, 0.82], "focal_length": 34.0}
WARD_HIDE = ["对象32326195"]
BASIN_C = (0.44, 0.0)
ROW_Y = 0.078  # basin rows at +/- ROW_Y: +y row -> arm 1 (rig left), -y row -> arm 0 (rig right)
TROLLEY = {"center": (0.54, 0.0), "size": (0.60, 0.36)}  # top at table height, push handle at +x
TRAY_Y = 0.33  # tray centers at +/- TRAY_Y
TRAY_X0 = 0.33  # inner (robot-side) x edge of the first slot's wall
LABEL_STRIP = 0.050  # deck strip beyond each tray's outer slot ends for its CASE label

# Oracle geometry.
FINGERTIP_BELOW = 0.01257  # X5 fingertips project this far below the IK grasp center
GRIP_LIFT = 0.009  # grasp center above the item center: fingertips 2.4 mm above the floor
OPEN = 0.022
RELEASE_OPEN = 0.014
FAST, DESCEND, PLACE = 0.50, 0.20, 0.14


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _extent(kind, yaw):
    hx, hy, _ = HALF[kind]
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return np.array([c * hx + s * hy, s * hx + c * hy])


def build_layout(b):
    z, rng = b.z, b.rng
    b.shift = b.shift * np.array([0.4, 0.6])  # the trays sit near the arms' reach
    n = b.task.spec.num_items
    if n != 6:
        raise ValueError("C2 uses six instruments: one of each type per basin row and per tray")
    from roboclock.envs.lab.robot import mount_positions

    bases = [list(map(float, m[:2])) for m in mount_positions(b.config, "C2")]

    # Sterilization basin: shallow open stainless pan on the trolley, a static collider.
    bc = BASIN_C
    hx, hy = BASIN_HALF
    floor = 0.006
    b.fixture("c2_basin_floor", bc, (2 * hx + 2 * WALL, 2 * hy + 2 * WALL, floor), "polymer")
    for name, xy, size in (
        ("c2_basin_wall_n", (bc[0], bc[1] + hy + WALL / 2), (2 * hx + 2 * WALL, WALL, BASIN_RIM)),
        ("c2_basin_wall_s", (bc[0], bc[1] - hy - WALL / 2), (2 * hx + 2 * WALL, WALL, BASIN_RIM)),
        ("c2_basin_wall_e", (bc[0] + hx + WALL / 2, bc[1]), (WALL, 2 * hy, BASIN_RIM)),
        ("c2_basin_wall_w", (bc[0] - hx - WALL / 2, bc[1]), (WALL, 2 * hy, BASIN_RIM)),
    ):
        b.fixture(name, xy, size, "stainless", z=z + floor)
    support = z + floor

    # Loose instruments: two rows of three (one of each type, shuffled), long axis roughly along y,
    # random flip and skew. Row 0 (+y) is arm 1's (rig left), row 1 (-y) arm 0's (rig right).
    item_types, arm_items, index = {}, {0: [], 1: []}, 0
    for r, side in ((0, 1), (1, 0)):
        row = [TYPE_ORDER[i] for i in rng.permutation(3)]
        yaws = [
            math.pi / 2 + float(rng.uniform(-0.2, 0.2)) + math.pi * int(rng.integers(0, 2))
            for _ in row
        ]
        widths = [2 * _extent(k, y)[0] for k, y in zip(row, yaws)]
        gap = 0.020
        total = sum(widths) + gap * (len(row) - 1)
        slack = max(0.0, 2 * hx - 0.040 - total)
        x = bc[0] - total / 2 + float(rng.uniform(-slack / 2, slack / 2))
        y_row = bc[1] + (ROW_Y if r == 0 else -ROW_Y)
        for kind, yaw, width in zip(row, yaws, widths):
            name = f"item_{index}"
            asset, _, color = TYPES[kind]
            cx = x + width / 2
            cy = y_row + float(rng.uniform(-0.005, 0.005))
            b.item(name, asset, (cx, cy), support, color=color, yaw=yaw)
            half = HALF[kind]
            b.shapes[name] = Shape(
                "box",
                half=half,
                grasp_radius=half[1] if kind != "hemostat" else 0.007,
                grasp_height=2 * half[0],
                grip_force=12.0,
            )
            obj = b.objects[name]
            b.objects[name] = replace(
                obj,
                position=np.array([obj.position[0], obj.position[1], support + half[2] + 0.001]),
                orientation=_yaw_quat(yaw),
                label="",
            )
            item_types[name] = kind
            arm_items[side].append(name)
            x += width + gap
            index += 1

    # Two Mayo trays (CASE 1 at -y for arm 0 / rig right, CASE 2 at +y for arm 1 / rig left), each with one
    # silicone slot per instrument type, long axis along y.
    slot_types, trays, arm_slots = {}, [], {0: [], 1: []}
    for side, sign, case in ((0, -1, 1), (1, 1, 2)):
        ty = sign * TRAY_Y
        x, labels = TRAY_X0, []
        for kind in SLOT_ORDER:
            half_x = SLOT_HALF_X[kind]
            x += half_x + WALL
            name = f"slot_{case}_{kind}"
            b.target(name, "tray", (x, ty), (half_x, SLOT_HALF_Y), SLOT_WALL_H, z0=z + TRAY_TOP)
            slot_types[name] = kind
            arm_slots[side].append(name)
            labels.append({"type": kind, "x": float(b.xy(x, 0)[0]), "width": 2 * half_x})
            x += half_x + WALL + 0.001
        x0, x1 = TRAY_X0 - 0.010, x + 0.009
        inner = SLOT_HALF_Y + WALL + 0.008
        outer = SLOT_HALF_Y + WALL + LABEL_STRIP
        y0, y1 = (ty - outer, ty + inner) if sign < 0 else (ty - inner, ty + outer)
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        size = (x1 - x0, y1 - y0, TRAY_TOP)
        b.fixture(f"c2_mayo_tray_{case}", center, size, "stainless", visible=False)
        trays.append(
            {
                "case": case,
                "side": side,
                "center": [float(v) for v in b.xy(*center)],
                "size": [float(v) for v in size],
                "deck_z": float(z + TRAY_TOP),
                "slot_y": float(b.xy(0, ty)[1]),
                "label_y": float(b.xy(0, ty + sign * (SLOT_HALF_Y + WALL + LABEL_STRIP / 2))[1]),
                "outer_y": float(b.xy(0, y0 if sign < 0 else y1)[1]),
                "labels": labels,
            }
        )

    # Designated slot: each arm puts an instrument in its own tray's slot of that type.
    for side in (0, 1):
        for name in arm_items[side]:
            kind = item_types[name]
            b.assignments[name] = next(s for s in arm_slots[side] if slot_types[s] == kind)

    b.indicator("indicator", (0.80, -0.115), "SHIFT")  # on the trolley's far half
    # For make_layout's shared C1/C2 type bookkeeping.
    b.types = tuple((TYPES[k][0], TYPES[k][2], TYPES[k][1]) for k in TYPE_ORDER)
    b.kinds = [TYPE_ORDER.index(item_types[f"item_{i}"]) for i in range(n)]

    basin_center = b.xy(*bc)
    ready = np.array([basin_center[0], basin_center[1] - ROW_Y, z + 0.17])
    b.info.update(
        bedside_task="ward_c2",
        c2={
            "item_types": item_types,
            "slot_types": slot_types,
            "arms": {
                str(side): {"items": arm_items[side], "slots": arm_slots[side]} for side in (0, 1)
            },
            "basin": {
                "center": [float(v) for v in basin_center],
                "half": list(BASIN_HALF),
                "support_z": float(support),
                "rim_z": float(support + BASIN_RIM),
            },
            "trays": trays,
            "trolley": {"center": list(TROLLEY["center"]), "size": list(TROLLEY["size"])},
            "readouts": {"center": [float(v) for v in b.xy(0.79, 0.095)], "z": float(z - 0.03)},
            "camera": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in CAMERA.items()},
            "grasp_x": GRASP_X,
            "bases": bases,
        },
        ward_pose={
            "usd": WARD_USD, "translate": list(WARD_TRANSLATE), "yaw": 0.0, "hide": list(WARD_HIDE),
        },
        hide_bench=True,
        work_surface={
            "center": [*TROLLEY["center"], float(z - 0.05)],
            "size": [*TROLLEY["size"], 0.10],
        },
        initial_ee_position=[float(v) for v in ready],
        initial_ee_yaw=_reachable_yaw(math.pi / 2, ready, bases[0]),
        initial_gripper_width=OPEN,
    )
    return b


# --- scene authoring (before physics starts) ---------------------------------------------------


def _text_texture(lines, fg, bg, size=(1024, 256)):
    """A display texture: one or more centered lines of text."""
    import hashlib
    import os

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr((lines, fg, bg, size)).encode()).hexdigest()[:20]
    path = cache / f"c2_{key}.png"
    if path.exists():
        return path
    image = Image.new("RGB", size, tuple(round(255 * c) for c in bg))
    draw = ImageDraw.Draw(image)
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    count = len(lines)
    for i, (text, scale) in enumerate(lines):
        font_size = int(size[1] / count * scale)
        try:
            font = ImageFont.truetype(bold, font_size)
        except OSError:
            font = ImageFont.load_default(size=font_size)
        while font.getlength(text) > size[0] * 0.94 and font_size > 10:
            font_size -= 4
            try:
                font = ImageFont.truetype(bold, font_size)
            except OSError:
                font = ImageFont.load_default(size=font_size)
        draw.text(
            (size[0] / 2, size[1] * (i + 0.5) / count),
            text,
            fill=tuple(round(255 * c) for c in fg),
            font=font,
            anchor="mm",
        )
    image.save(path)
    return path


def _panel(stage, path, texture, center, size, *, flat=False, yaw_deg=0.0, emissive=0.0):
    """A textured quad. Upright panels face +y (toward the camera); flat ones face up."""
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
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    if emissive:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, path)
    w, h = size
    mesh.CreatePointsAttr(
        [Gf.Vec3f(-w / 2, 0, -h / 2), Gf.Vec3f(-w / 2, 0, h / 2),
         Gf.Vec3f(w / 2, 0, -h / 2), Gf.Vec3f(w / 2, 0, h / 2)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(1, 0), (1, 1), (0, 0), (0, 1)])
    mesh.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        mesh.AddRotateZOp().Set(float(yaw_deg))
    if flat:
        mesh.AddRotateXOp().Set(90.0)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _collider(backend, path, offset, size, name):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(backend.stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in offset]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    backend._collision_settings(prim)
    material = backend.stage.GetPrimAtPath(f"/World/Materials/{name}")
    if material.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(material), materialPurpose="physics"
        )
    UsdGeom.Imageable(prim).MakeInvisible()


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# Mean linear albedo of the CC0 diffuse maps used with a color tint (assets/textures/*_diff_1k.png):
# the tint multiplies the map, so tint = target / mean reproduces a target sRGB color on average.
MAP_MEAN = {"silicone": (0.032, 0.033, 0.035), "plastic_matte": (0.847, 0.820, 0.812)}


def _tint(texture_set, srgb):
    return tuple(float(v) for v in _srgb_to_linear(srgb) / np.asarray(MAP_MEAN[texture_set]))


def _materials(look):
    """Textured PBR materials (CC0 sets in hospital/assets/textures) for C2's own parts.

    Shared palette names ("stainless", "steel", "rubber", "sterile_blue", ...) are textured
    centrally; these cover the parts whose look is specific to C2.
    """
    mats = {
        # Instruments: fine brushed grain at instrument scale; serrations a darker, rougher steel.
        "instrument": look.textured(
            "c2_instrument_steel", "stainless_brushed", repeats=40.0, roughness=0.38
        ),
        "serration": look.textured(
            "c2_instrument_serration", "stainless_brushed", color=(0.55, 0.55, 0.57),
            repeats=40.0, roughness=0.5,
        ),
        # Robot base: shells match the central mount-column coat (texture_robot_mounts).
        "shell": look.textured("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90)),
        "cover": look.textured("c2_robot_cover", "plastic_matte", color=(0.70, 0.73, 0.76)),
        "skirt": look.textured("c2_robot_skirt", "powder_coat_dark"),
        "trim": look.textured("c2_robot_trim", "aluminum_brushed"),
        "estop": look.textured("c2_estop_red", "plastic_matte", color=_tint("plastic_matte", (0.80, 0.08, 0.06))),
        "estop_collar": look.textured(
            "c2_estop_yellow", "plastic_matte", color=_tint("plastic_matte", (0.98, 0.80, 0.10))
        ),
        # Shift display: powder-coated housing on an aluminum stand.
        "housing": look.textured("c2_display_housing", "powder_coat_dark"),
        "bracket": look.textured("c2_display_bracket", "aluminum_brushed"),
    }
    # Silicone slot liners keep each type's color code: walls in the full color, floors lighter.
    for kind, (_, _, color) in TYPES.items():
        color = np.asarray(color)
        mats[f"slot_wall_{kind}"] = look.textured(
            f"c2_slot_wall_{kind}", "silicone", color=_tint("silicone", 0.80 * color + 0.06)
        )
        mats[f"slot_floor_{kind}"] = look.textured(
            f"c2_slot_floor_{kind}", "silicone", color=_tint("silicone", 0.45 * color + 0.48)
        )
    return mats


def _bind(stage, path, material, stronger=False):
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return False
    strength = (
        UsdShade.Tokens.strongerThanDescendants if stronger else UsdShade.Tokens.weakerThanDescendants
    )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, strength)
    return True


def _instrument(backend, name, kind, color, mats):
    from pxr import Usd, UsdGeom, UsdPhysics

    from roboclock.envs.hospital.props import add_model, model_size

    stage, look = backend.stage, backend.appearance
    root = f"/World/Objects/{name}"
    for child in list(stage.GetPrimAtPath(root).GetChildren()):
        stage.RemovePrim(child.GetPath())
    hx, hy, hz = HALF[kind]
    tape = look.material(f"c2_tape_{kind}", color=list(color), roughness=0.45)
    if kind == "hemostat":
        # Narrow shaft and jaws where the fingers close, plus the ring handles at -x.
        _collider(backend, root + "/Shaft", (0.018, 0, 0), (0.085, 0.014, 2 * hz), name)
        _collider(backend, root + "/Rings", (-0.0429, 0, 0), (0.035, 2 * hy, 2 * hz), name)
        if model_size("tool_clamp") is not None:
            add_model(stage, "tool_clamp", root + "/Model", (0, 0, -hz), (1.0, 1.0, 0.75))
        else:
            look.surgical_tool(root, "clamp", (2 * hx, 2 * hy, 2 * hz))
        look.rounded(root + "/Tape", (-0.012, 0, -hz + 0.004), (0.006, 0.010, 0.006), tape, 0.001)
    else:
        _collider(backend, root + "/Body", (0, 0, 0), (2 * hx, 2 * hy, 2 * hz), name)
        if kind == "scalpel" and model_size("tool_scalpel") is not None:
            native = model_size("tool_scalpel")
            s = 2 * hx / native[0]
            add_model(stage, "tool_scalpel", root + "/Model", (0, 0, -hz), (s, s, 1.6))
            look.rounded(
                root + "/Tape", (0.040, 0, -hz + 0.0045), (0.010, 0.0115, 0.0075), tape, 0.001
            )
        else:
            look.surgical_tool(root, kind, (2 * hx, 2 * hy, 2 * hz))
            if kind == "forceps":
                # The shared silhouette wraps both prongs in black "serration" bands that read as
                # rings: drop them and join the prongs at a welded heel like real thumb forceps.
                for k in range(5):
                    stage.RemovePrim(f"{root}/Serration_{k}")
                look.rounded(
                    root + "/Heel", (-0.86 * hx, 0, 0), (0.12 * hx, 1.5 * hy, 1.25 * hz),
                    mats["instrument"], 0.001,
                )
            look.rounded(root + "/Tape", (-0.045, 0, 0), (0.010, 2 * hy + 0.001, 0.0085), tape, 0.001)
    # Steel everywhere but the colored tape: the imported tool_scalpel / tool_clamp meshes carry a
    # flat grey, non-metallic map (they read as grey plastic), and the procedural forceps get the
    # same fine-grained brushed stainless.
    model = stage.GetPrimAtPath(root + "/Model")
    if model.IsValid():
        # The imported meshes are single-sided with some faces wound inward: in steel they read as
        # see-through outlines. Render both sides.
        for prim in Usd.PrimRange(model):
            if prim.IsA(UsdGeom.Mesh):
                UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
    if not _bind(stage, root + "/Model", mats["instrument"], stronger=True):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if prim.GetName() == "Tape":
                continue
            key = "serration" if prim.GetName().startswith(("Serration", "Groove")) else "instrument"
            _bind(stage, str(prim.GetPath()), mats[key])


def author(backend):
    layout, stage, look = backend.layout, backend.stage, backend.appearance
    info = layout.info["c2"]
    z = layout.table_height
    mats = _materials(look)
    for name, kind in info["item_types"].items():
        _instrument(backend, name, kind, TYPES[kind][2], mats)

    # Silicone slot liners in each type's color.
    for slot, kind in info["slot_types"].items():
        for part in ("floor", "left", "right", "front", "back"):
            key = f"slot_{'floor' if part == 'floor' else 'wall'}_{kind}"
            _bind(stage, f"/World/Targets/{slot}/{part}_Visual", mats[key])

    _basin(backend, info)
    _trolley(look, info, z)
    for tray in info["trays"]:
        _mayo_tray(backend, info, tray, z)
        _mayo_stand(look, tray)
    if backend.config["visuals"]["labels"]:
        basin = info["basin"]
        texture = _text_texture(
            [("STERILIZATION BASIN", 0.62)], (0.1, 0.1, 0.12), (0.86, 0.88, 0.9), (1024, 124)
        )
        bx, by = basin["center"]
        # On the +y wall's face, below the rolled rim.
        _panel(
            stage, "/World/C2/BasinLabel", texture,
            (bx, by + basin["half"][1] + WALL + 0.0005, basin["rim_z"] - 0.0165), (0.20, 0.024),
        )
    _readout_board(backend, info, mats)
    _robot_base(backend, mats)


def _basin(backend, info):
    """A stainless pan (the floor collider's visual) lined with a blue sterile wrap."""
    stage, look = backend.stage, backend.appearance
    basin = info["basin"]
    (bx, by), (hx, hy) = basin["center"], basin["half"]
    _bind(stage, "/World/Fixtures/c2_basin_floor_Visual", look.material("stainless"))
    # The liner lies on the pan floor inside the walls (visual only; 1.5 mm, its top 0.8 mm above
    # the support face the instruments rest on).
    support = basin["support_z"]
    look.rounded(
        "/World/C2/BasinLiner", (bx, by, support - 0.0007), (2 * hx - 0.002, 2 * hy - 0.002, 0.0015),
        "sterile_blue", 0.0007,
    )
    # A rolled stainless rim along the top of the four walls.
    rim = basin["rim_z"]
    for k, (center, size) in enumerate(
        (
            ((bx, by + hy + WALL / 2), (2 * hx + 2 * WALL + 0.004, WALL + 0.004)),
            ((bx, by - hy - WALL / 2), (2 * hx + 2 * WALL + 0.004, WALL + 0.004)),
            ((bx + hx + WALL / 2, by), (WALL + 0.004, 2 * hy)),
            ((bx - hx - WALL / 2, by), (WALL + 0.004, 2 * hy)),
        )
    ):
        look.rounded(f"/World/C2/BasinRim_{k}", (*center, rim - 0.002), (*size, 0.005), "stainless", 0.002)


def _caster(look, path, x, y, radius=0.025, fork="steel"):
    look.cylinder(path + "_Wheel", (x, y, FLOOR_Z + radius), radius, 0.018, "rubber", axis="Y")
    look.cylinder(path + "_Hub", (x, y, FLOOR_Z + radius), radius * 0.45, 0.020, "aluminum", axis="Y")
    look.rounded(path + "_Fork", (x, y, FLOOR_Z + 2 * radius + 0.009), (0.026, 0.030, 0.020), fork, 0.003)


def _trolley(look, info, z):
    """Stainless two-tier instrument trolley under the basin, push handle at its far (+x) end."""
    (cx, cy), (sx, sy) = info["trolley"]["center"], info["trolley"]["size"]
    look.rounded("/World/C2/TrolleyTop", (cx, cy, z - 0.012), (sx, sy, 0.024), "stainless", 0.004)
    for k, (dy, w) in enumerate(((-sy / 2, 0.006), (sy / 2, 0.006))):
        look.rounded(f"/World/C2/TrolleyLip_{k}", (cx, cy + dy, z + 0.006), (sx, w, 0.012), "stainless", 0.002)
    look.rounded("/World/C2/TrolleyShelf", (cx, cy, 0.30), (sx - 0.02, sy - 0.02, 0.018), "stainless", 0.004)
    for i, dx in enumerate((-sx / 2 + 0.02, sx / 2 - 0.02)):
        for j, dy in enumerate((-sy / 2 + 0.02, sy / 2 - 0.02)):
            leg_h = z - 0.024 - FLOOR_Z - 0.06
            look.cylinder(
                f"/World/C2/TrolleyLeg_{i}{j}", (cx + dx, cy + dy, FLOOR_Z + 0.06 + leg_h / 2), 0.012, leg_h, "stainless"
            )
            _caster(look, f"/World/C2/TrolleyCaster_{i}{j}", cx + dx, cy + dy)
    hx = cx + sx / 2 + 0.03
    for k, dy in enumerate((-sy / 2 + 0.03, sy / 2 - 0.03)):
        look.cylinder(f"/World/C2/TrolleyHandlePost_{k}", (hx, cy + dy, z + 0.06), 0.010, 0.14, "stainless")
    look.cylinder("/World/C2/TrolleyHandle", (hx, cy, z + 0.13), 0.013, sy - 0.04, "rubber", axis="Y")


def _mayo_tray(backend, info, tray, z):
    """A stainless deck on two rails (one invisible collider block), with slot and case labels."""
    stage, look = backend.stage, backend.appearance
    (cx, cy), (sx, sy, sz) = tray["center"], tray["size"]
    case = tray["case"]
    look.rounded(f"/World/C2/MayoDeck_{case}", (cx, cy, z + sz - 0.003), (sx, sy, 0.006), "stainless", 0.003)
    look.rounded(
        f"/World/C2/MayoLip_{case}", (cx, tray["outer_y"], z + sz + 0.004), (sx, 0.006, 0.008), "stainless", 0.002
    )
    for k, dx in enumerate((-sx / 2 + 0.03, sx / 2 - 0.03)):
        look.rounded(
            f"/World/C2/MayoRail_{case}_{k}", (cx + dx, cy, z + (sz - 0.006) / 2), (0.016, sy - 0.02, sz - 0.006),
            "steel", 0.002,
        )
    if not backend.config["visuals"]["labels"]:
        return
    texture = _text_texture([(f"CASE {case}", 0.7)], (1, 1, 1), (0.16, 0.20, 0.26), (768, 192))
    _panel(
        stage, f"/World/C2/CaseLabel_{case}", texture, (cx, tray["label_y"], tray["deck_z"] + 0.0012),
        (min(0.16, sx - 0.02), 0.036), flat=True,
    )
    support = info["basin"]["support_z"] - 0.006 + TRAY_TOP + 0.004  # slot floor top
    for label in tray["labels"]:
        color = TYPES[label["type"]][2]
        texture = _text_texture([(TYPES[label["type"]][1], 0.8)], color, (1, 1, 1), (1024, 192))
        # Printed along the empty slot's floor, read from the camera side; the instrument covers it.
        _panel(
            stage, f"/World/C2/SlotLabel_{case}_{label['type']}", texture,
            (label["x"], tray["slot_y"], support + 0.0006), (0.15, min(0.026, label["width"] * 0.8)),
            flat=True, yaw_deg=-90.0,
        )


def _mayo_stand(look, tray):
    """A Mayo stand: the tray rides on a cantilever arm and one post over a wheeled U base."""
    (cx, cy), (sx, sy, _) = tray["center"], tray["size"]
    case = tray["case"]
    sign = 1.0 if tray["outer_y"] > cy else -1.0
    deck = tray["deck_z"] - TRAY_TOP  # bottom of the tray assembly
    post = (cx, tray["outer_y"] + sign * 0.030)
    height = deck - FLOOR_Z - 0.07
    look.cylinder(f"/World/C2/MayoPost_{case}", (*post, FLOOR_Z + 0.07 + height / 2), 0.016, height, "stainless")
    look.rounded(
        f"/World/C2/MayoArm_{case}", (cx, post[1] - sign * 0.07, deck - 0.012), (0.05, 0.16, 0.018), "stainless", 0.004
    )
    look.rounded(f"/World/C2/MayoCollar_{case}", (*post, deck - 0.05), (0.05, 0.05, 0.06), "steel", 0.006)
    # U base on the post's side, its legs reaching in under the tray (clear of the trolley).
    look.rounded(f"/World/C2/MayoBaseBack_{case}", (cx, post[1], FLOOR_Z + 0.078), (0.34, 0.04, 0.026), "stainless", 0.006)
    for k, dx in enumerate((-0.15, 0.15)):
        look.rounded(
            f"/World/C2/MayoBaseLeg_{case}_{k}", (cx + dx, post[1] - sign * 0.11, FLOOR_Z + 0.078), (0.04, 0.24, 0.026),
            "stainless", 0.006,
        )
        _caster(look, f"/World/C2/MayoCaster_{case}_{k}a", cx + dx, post[1])
        _caster(look, f"/World/C2/MayoCaster_{case}_{k}b", cx + dx, post[1] - sign * 0.21)


def _robot_base(backend, mats):
    """A compact wheeled dual-arm base: low chassis, one torso post and a shoulder crossbar that
    carries both arm columns (the columns themselves are built by the robot loader).

    Molded light grey shells like the mount columns over a dark powder-coated skirt with rubber
    bumper strips, an inset service cover, side vent slats, an e-stop, an aluminum shoulder bar
    and push handle, and rubber casters on dark forks.
    """
    from roboclock.envs.lab.robot import mount_positions

    look = backend.appearance
    mounts = np.array(mount_positions(backend.config, "C2"))[:, :2]
    top = backend.layout.table_height
    cx = float(mounts[:, 0].mean())
    cy = float(mounts[:, 1].mean())
    span = float(np.ptp(mounts[:, 1]))
    sx, sy = 0.36, span + 0.16
    ccx = cx - 0.06
    p = "/World/C2/Robot"
    # Chassis: skirt band (0.03-0.09 m) under the molded shell (0.08-0.19 m).
    look.rounded(p + "Skirt", (ccx, cy, FLOOR_Z + 0.06), (sx + 0.012, sy + 0.012, 0.06), mats["skirt"], 0.016)
    look.rounded(p + "Chassis", (ccx, cy, FLOOR_Z + 0.135), (sx, sy, 0.11), mats["shell"], 0.03)
    look.rounded(p + "Cover", (ccx - 0.015, cy, FLOOR_Z + 0.1915), (sx - 0.09, sy - 0.07, 0.005), mats["cover"], 0.012)
    for k, sign in enumerate((-1, 1)):  # rubber bumper strips on the front and back of the skirt
        look.rounded(
            p + f"BumperStrip_{k}", (ccx + sign * (sx / 2 + 0.008), cy, FLOOR_Z + 0.062),
            (0.012, sy - 0.06, 0.026), "rubber", 0.005,
        )
    for k, sign in enumerate((-1, 1)):  # vent slats on both sides of the shell
        for j in range(4):
            look.rounded(
                p + f"Vent_{k}_{j}", (ccx - 0.05, cy + sign * (sy / 2 + 0.0005), FLOOR_Z + 0.115 + 0.014 * j),
                (0.14, 0.003, 0.005), mats["skirt"], 0.0012,
            )
    # E-stop on the rear of the cover: red mushroom on a yellow collar.
    stop = (ccx - sx / 2 + 0.06, cy + sy / 2 - 0.08)
    look.cylinder(p + "EStopCollar", (*stop, FLOOR_Z + 0.197), 0.022, 0.008, mats["estop_collar"])
    look.cylinder(p + "EStopStem", (*stop, FLOOR_Z + 0.205), 0.008, 0.010, mats["skirt"])
    look.cylinder(p + "EStop", (*stop, FLOOR_Z + 0.214), 0.016, 0.010, mats["estop"])
    # Torso: shell with a dark front panel and the status strip, push handle on the back.
    torso_bottom = FLOOR_Z + 0.19
    torso_h = top - 0.05 - torso_bottom
    tx = cx - 0.03
    look.rounded(p + "Torso", (tx, cy, torso_bottom + torso_h / 2), (0.16, 0.20, torso_h), mats["shell"], 0.03)
    look.rounded(
        p + "TorsoPanel", (tx + 0.079, cy, torso_bottom + torso_h * 0.52), (0.004, 0.15, torso_h * 0.72),
        mats["skirt"], 0.004,
    )
    look.rounded(
        p + "StatusStrip", (tx + 0.0815, cy, torso_bottom + torso_h * 0.78), (0.003, 0.11, 0.010),
        look.material("c2_status_led", color=[0.2, 0.75, 1.0], roughness=0.3, emissive=True), 0.002,
    )
    hz = torso_bottom + torso_h * 0.70
    for k, dy in enumerate((-0.07, 0.07)):
        look.cylinder(p + f"HandlePost_{k}", (tx - 0.095, cy + dy, hz), 0.007, 0.03, mats["trim"], axis="X")
    look.cylinder(p + "Handle", (tx - 0.11, cy, hz), 0.011, 0.17, mats["trim"], axis="Y")
    # Shoulder crossbar under both arm columns, with dark end caps.
    look.rounded(p + "Shoulder", (cx, cy, top - 0.025), (0.14, span + 0.11, 0.05), mats["trim"], 0.008)
    for k, sign in enumerate((-1, 1)):
        look.rounded(
            p + f"ShoulderCap_{k}", (cx, cy + sign * (span / 2 + 0.065), top - 0.025), (0.146, 0.022, 0.056),
            mats["skirt"], 0.008,
        )
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        _caster(
            look, f"/World/C2/RobotCaster_{k}", ccx + dx * (sx / 2 - 0.05), cy + dy * (sy / 2 - 0.05), 0.03,
            fork=mats["skirt"],
        )


def _readout_board(backend, info, mats):
    """Tray-side display: the goal and the live sorted count, beside the countdown dial."""
    stage, look = backend.stage, backend.appearance
    from pxr import UsdGeom

    (x, y), z = info["readouts"]["center"], info["readouts"]["z"]
    task = backend.task
    # The panels face the group's +y: turn it toward the rig's head camera (the policy's `rgb`;
    # RoboDojo rig 2026-11-04: toward the third-person C2 camera it stood edge-on to the head).
    from roboclock.envs.hospital.rig import dial_target

    toward = dial_target(backend, info["camera"]["position"])
    yaw = math.degrees(math.atan2(toward[1] - y, toward[0] - x)) - 90.0
    n = len(info["item_types"])
    group = look.group("/World/C2/Display", (x, y, z), rz=yaw)
    look.rounded(group + "/Foot", (0, 0.0, 0.036), (0.16, 0.09, 0.012), mats["bracket"], 0.004)
    look.rounded(group + "/Post", (0, -0.02, 0.09), (0.022, 0.016, 0.10), mats["bracket"], 0.003)
    look.rounded(group + "/Bezel", (0, -0.014, 0.19), (0.25, 0.012, 0.15), mats["housing"], 0.005)
    look.rounded(group + "/Back", (0, -0.024, 0.19), (0.17, 0.010, 0.09), mats["housing"], 0.004)
    dark = (0.02, 0.03, 0.04)
    header = _text_texture(
        [("INSTRUMENT SORT", 0.55), (f"GOAL  ALL {task.jobs_required} SORTED", 0.45)],
        (0.85, 0.92, 1.0), dark, (1024, 256),
    )
    _panel(stage, group + "/Header", header, (0, -0.0075, 0.235), (0.23, 0.0575), emissive=1.0)
    backend.zone_prims = getattr(backend, "zone_prims", {})
    for k in range(n + 1):
        texture = _text_texture([(f"SORTED {k}/{n}", 0.8)], (0.35, 1.0, 0.45), dark, (1024, 256))
        mesh = _panel(stage, group + f"/Count_{k}", texture, (0, -0.0076, 0.180), (0.23, 0.0575), emissive=1.0)
        backend.zone_prims[f"c2_count_{k}"] = mesh.GetPrim()
        if k:
            UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()


def _camera(backend):
    """A closer view of the instruments than the shared bedside camera, set once per reset."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["c2"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    if view.get("focal_length"):
        camera.set_focal_length(float(view["focal_length"]) / 1000)
    for _ in range(3):
        backend.world.render()


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_c2_camera_for", None) is not backend.task:
        backend._c2_camera_for = backend.task
        _camera(backend)
    controls = {}
    robot = getattr(backend, "secondary_robot", None)
    if robot is not None:
        # The right hand's measured yaw, for the privileged two-arm diagnostic oracle.
        _, orientation = backend._tool_pose(robot)
        controls["c2_right_ee_yaw"] = _ee_yaw(np.asarray(orientation, dtype=float))
    return controls


# --- privileged diagnostic oracle ----------------------------------------------------------------


def _wrap_half(angle):
    """Wrap to [-pi/2, pi/2): the gripper and the slots are symmetric under a half turn."""
    return (angle + math.pi / 2) % math.pi - math.pi / 2


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _ee_yaw(q):
    from roboclock.physics.geometry import rotation_matrix

    r = rotation_matrix(q)
    return math.atan2(-r[1, 0], -r[0, 0])


def _in_own_slot(task, state, name):
    info = task.layout.info["c2"]
    position = state.objects[name].position
    return any(
        kind == info["item_types"][name]
        and np.all(
            np.abs(position[:2] - task.layout.targets[slot].position[:2])
            <= task.layout.targets[slot].half_extents[:2]
        )
        for slot, kind in info["slot_types"].items()
    )


def _free_slot(task, state, name, slots, avoid=()):
    info = task.layout.info["c2"]
    kind = info["item_types"][name]
    options = [s for s in slots if info["slot_types"][s] == kind and s not in avoid]
    designated = task.layout.assignments.get(name)
    options.sort(key=lambda s: s != designated)
    for slot in options:
        target = task.layout.targets[slot]
        busy = any(
            other != name
            and np.all(np.abs(o.position[:2] - target.position[:2]) <= target.half_extents[:2])
            for other, o in state.objects.items()
        )
        if not busy:
            return slot
    return None


def _reachable_yaw(yaw, point, base):
    """The half-turn-equivalent gripper yaw nearest the bearing from the arm base.

    The X5 wrist cannot reach every yaw: Lula IK fails when the yaw is more than about
    100 degrees from the bearing of the target, so the symmetric gripper uses yaw or yaw + pi.
    """
    bearing = math.atan2(point[1] - base[1], point[0] - base[0])
    yaw = _wrap_half(yaw)
    return min((yaw - math.pi, yaw, yaw + math.pi), key=lambda v: abs(v - bearing))


def _turn(planner, target, rate=3.0):
    planner.yaw += float(np.clip(target - planner.yaw, -rate * planner.dt, rate * planner.dt))


def command(planner, task, state):
    """Left arm: the basin's -y row into the CASE 1 tray."""
    return _drive(planner, task, state, 0)


def command_secondary(planner, task, state):
    """Right arm: the basin's +y row into the CASE 2 tray (zones never overlap, so no locking)."""
    return _drive(planner, task, state, 1)


def _drive(planner, task, state, side):
    info = task.layout.info["c2"]
    arm = info["arms"][str(side)]
    now = task.time
    if getattr(planner, "_c2_task", None) is not task:
        planner._c2_task = task
        planner.c2 = SimpleNamespace(
            phase="select", item=None, slot=None, t=now, retries={}, released={}, avoid=set()
        )
        planner.c2.base = np.asarray(info["bases"][side], dtype=float)
    c = planner.c2
    planner.gentle = True
    planner.close_width = 0.0
    ee = getattr(planner, "ee_position", None)
    ee = np.asarray(state.ee_position if ee is None else ee, dtype=float)
    if side == 0:
        ee_yaw = _ee_yaw(state.ee_orientation)
    else:
        ee_yaw = float(state.controls.get("c2_right_ee_yaw", planner.yaw))
    basin = info["basin"]
    deck = max(t["deck_z"] for t in info["trays"])
    carry_z = max(basin["rim_z"], deck + SLOT_WALL_H) + 0.075

    def go(phase):
        c.phase, c.t = phase, now
        task.event_log.append(
            {
                "time": round(now, 3),
                "type": "c2_phase",
                "arm": side,
                "phase": phase,
                "object": c.item,
                "ee_yaw": round(ee_yaw, 3),
                "cmd_yaw": round(planner.yaw, 3),
            }
        )

    if c.phase == "select":
        planner.open_width = OPEN
        names = [
            n
            for n in arm["items"]
            if n not in task.correct
            and n not in task.dropped
            and c.retries.get(n, 0) < 3
            and not (task.resting.get(n) or "").startswith("slot_")
            and now - c.released.get(n, -9.0) > 2.0  # let a placed item settle and score
            and not _in_own_slot(task, state, n)
        ]
        names.sort(key=lambda n: (np.linalg.norm(state.objects[n].position[:2] - ee[:2]), n))
        for name in names:
            slot = _free_slot(task, state, name, arm["slots"], c.avoid)
            if slot is not None:
                c.item, c.slot = name, slot
                go("approach")
                break
        else:
            # Nothing to do (yet): hold the last commanded point at carry height. Chasing the
            # measured tool point let the idle hand sag back onto its own mount column.
            hold = planner.last_goal if planner.last_goal is not None else ee
            return planner.finish(state, np.array([*hold[:2], carry_z]), False, FAST)

    name, slot = c.item, c.slot
    obj = state.objects[name]
    kind = info["item_types"][name]
    hz = HALF[kind][2]
    obj_yaw = _yaw(obj.orientation)
    axis = np.array([math.cos(obj_yaw), math.sin(obj_yaw), 0.0])
    pick = obj.position + axis * info["grasp_x"][kind] + np.array([0.0, 0.0, GRIP_LIFT])
    target = task.layout.targets[slot]
    close, speed = False, FAST
    planner.open_width = OPEN
    # The contact-based grasp flag flickers on thin instruments: the grip counts as lost only once
    # the instrument has actually slipped away from where it was held (for 0.15 s).
    slipped = c.phase in ("lift", "transfer", "place") and np.linalg.norm(
        ee - obj.position - getattr(c, "offset", np.zeros(3))
    ) > 0.02
    c.slip = (getattr(c, "slip", 0.0) + planner.dt) if slipped else 0.0
    lost = c.slip >= 0.15

    if c.phase in ("approach", "descend", "grasp"):
        _turn(planner, _reachable_yaw(obj_yaw, pick, c.base))
    if c.phase == "approach":
        goal = pick + np.array([0.0, 0.0, 0.07])
        if np.linalg.norm(ee[:2] - goal[:2]) > 0.06:
            goal[2] = carry_z  # travel above the basin rim and the tray walls, then come down
        yaw_err = abs(_wrap_half(ee_yaw - obj_yaw))
        if (
            np.linalg.norm(ee[:2] - goal[:2]) < 0.006
            and abs(ee[2] - goal[2]) < 0.02
            and yaw_err < 0.06
        ):
            go("descend")
    elif c.phase == "descend":
        goal, speed = pick, DESCEND
        if np.linalg.norm(ee[:2] - pick[:2]) > 0.012:
            # Pushed off the item (e.g. it lies on a slot wall): give up after a few tries.
            c.flaps = getattr(c, "flaps", {})
            c.flaps[name] = c.flaps.get(name, 0) + 1
            if c.flaps[name] >= 4:
                c.retries[name] = 3
                go("select")
            else:
                go("approach")
        elif np.linalg.norm(ee - pick) < 0.005:
            go("grasp")
    elif c.phase == "grasp":
        goal, close = pick, True
        if obj.grasped and now - c.t > 0.30:  # let the gentle squeeze build before lifting
            c.offset = ee - obj.position
            go("lift")
        elif now - c.t > 1.4:
            c.retries[name] = c.retries.get(name, 0) + 1
            task.event_log.append({"time": now, "type": "c2_grasp_failed", "object": name})
            go("approach" if c.retries[name] < 3 else "select")
            close = False
    elif c.phase == "lift":
        # A fast lift shakes thin instruments out of the fingertips.
        close, speed = True, 0.25
        goal = np.array([ee[0], ee[1], carry_z])
        if lost:
            c.retries[name] = c.retries.get(name, 0) + 1
            go("select")
        elif obj.position[2] - hz > basin["rim_z"] + 0.02:
            go("transfer")
    elif c.phase in ("transfer", "place"):
        close = True
        held = ee - obj.position
        yaw_err = _wrap_half(math.pi / 2 - obj_yaw)
        xy = target.position[:2] + held[:2]
        _turn(planner, _reachable_yaw(ee_yaw + yaw_err, xy, c.base))
        xy_err = np.linalg.norm(obj.position[:2] - target.position[:2])
        if c.phase == "transfer":
            goal = np.array([*xy, carry_z])
            if xy_err < 0.006 and abs(yaw_err) < 0.05 and abs(ee[2] - carry_z) < 0.03:
                go("place")
        else:
            speed = PLACE
            # Fingertips must stay above the silicone walls (the closed hand is as wide as a
            # narrow slot), so the instrument is let go about 1 cm above the slot floor.
            bottom_goal = target.rim_z + 0.0015 + FINGERTIP_BELOW - GRIP_LIFT - hz
            goal = np.array([*xy, bottom_goal + hz + held[2]])
            bottom = obj.position[2] - hz
            if now - c.t > 2.5:
                task.event_log.append(
                    {
                        "time": now,
                        "type": "c2_place_timeout",
                        "arm": side,
                        "object": name,
                        "obj": [round(float(v), 4) for v in obj.position],
                        "ee": [round(float(v), 4) for v in ee],
                        "goal": [round(float(v), 4) for v in goal],
                        "slot": [round(float(v), 4) for v in target.position],
                    }
                )
            if (bottom - bottom_goal < 0.003 and xy_err < 0.0035) or now - c.t > 2.5:
                c.release_at = ee.copy()
                go("release")
        if lost and c.phase in ("transfer", "place"):
            c.retries[name] = c.retries.get(name, 0) + 1
            go("select")
    elif c.phase == "release":
        planner.open_width = RELEASE_OPEN
        goal = c.release_at
        if now - c.t > 0.25:
            c.released[name] = now
            go("retreat")
    else:  # retreat
        planner.open_width = RELEASE_OPEN
        goal = np.array([ee[0], ee[1], carry_z])
        if ee[2] > target.rim_z + 0.035:
            go("select")
    if c.phase != "select" and now - c.t > 6.0:
        c.retries[name] = c.retries.get(name, 0) + 1
        task.event_log.append(
            {"time": now, "type": "c2_timeout", "arm": side, "object": name, "phase": c.phase}
        )
        go("select")
    return planner.finish(state, np.asarray(goal, dtype=float), close, speed)
