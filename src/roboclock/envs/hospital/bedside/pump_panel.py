"""B6 infusion pump on an IV pole at the bedside equipment pendant, with contact-driven keys.

Placement (scene-02 preview left bay, assets/hospital/ward_02_preview_left.usdc): the robot is a
wheeled dual-arm base parked beside the bed in the bay's instrument-trolley spot (the trolley is
hidden), facing the window with the bed on its left. Ahead stands the ceiling equipment pendant with
its glass device shelves. The existing infusion pump (CC-BY-4.0, Ethan Cragun) is clamped to the
existing IV stand (CC-BY-4.0, teeart) in front of the pendant, screen toward the robot's right,
its line running to the bed. The response dial stands on the pendant's middle glass shelf.

The instruction is on screen from the first frame: the pump's own display shows the alarm and the
key to press (e.g. "OCCLUSION ALARM: PRESS STOP"), and the order card under the response dial
repeats it. The response clock starts at the instruction time (tasks.B6.instruction_time, just
after the scene settles), when the dial turns red and starts its countdown; the display turns
green when the pump answers the key.

The robot starts from rest, its pressing hand folded near its own mounting column (TUCK), and
reaches to the pump only after the instruction. The pump keypad is the original top-facing training
adapter (unchanged key geometry and springs): the six-axis X5 makes one downward press. The pump
body is the invisible support collider (layout work_surface), so the arm cannot pass through the
housing.

Public helpers reused by other tasks (keep their signatures): `add_keypad`, `author_keys`,
`measure_keys`, `dress_keypad` (optional textured caps and key surrounds), plus the module hooks
`build_layout`, `author`, `measure` and `command`.

Surfaces use the CC0 PBR sets of hospital/assets/textures (`look.textured`): tinted glossy
plastic key caps in molded key surrounds on a light plastic keypad bezel, a brushed aluminum pole
clamp, a brushed stainless IV stand on rubber casters, a paper order card in an aluminum clip and
a mobile base of matte plastic shells, a powder-coated skirt and rubber casters. The screens, the
key legends and the label images stay flat.
"""

import math

import numpy as np

from roboclock.envs.hospital.bedside.ward import furniture
from roboclock.envs.hospital.tasks.bedside.pump_panel import INSTRUCTION_TEXT, MODES, PRESS, STROKE

WARD_USD = "assets/hospital/ward_02_preview_left.usdc"
# Robot origin in the ward file (meters) and the ward yaw that turns the robot toward the window:
# yaw 90 maps ward (x, y) to robot-frame (ROBOT_Y - y, x - ROBOT_X); the seed shift moves the whole
# ward with the apparatus, so the pump keeps its place at the pendant.
# WARD_Z lifts the ward's floor slab 2 mm above the physics ground plane (no z-fighting).
ROBOT_X, ROBOT_Y, WARD_Z = 0.31, -0.95, 0.002
WARD_YAW = 90.0
# The common rig (hospital/rig.py) stands the idle arm 0.60 m to +y of the driving arm, over the
# bed's near side (robot y 0.46..1.46): the bed is pushed 0.35 m further from the robot (+y), and the
# ultrasound cart behind it 0.33 m toward the wall counter so the two still stand apart.
WARD_MOVE = {
    **{name: (0.0, 0.35, 0.0) for name in furniture("bed")},
    **{name: (0.0, 0.33, 0.0) for name in furniture("ultrasound_cart")},
}
# The bay's stainless 2-tier instrument trolley (ward x 0.26-0.70, y -1.56..-0.91) stands where the
# robot parks; it is scenery, so it is hidden for this task.
TROLLEY = [
    f"Obj24of_591669_{token}"
    for token in (
        "100_899 101_471 102_620 103_848 104_456 21_459 22_190 23_738 24_278 25_900 32_038 33_963 "
        "34_199 35_846 36_354 37_314 38_097 39_878 40_299 41_971 48_789 49_845 50_990 51_414 "
        "52_284 53_709 54_797 55_325 56_270 57_350 58_548 59_216 60_220 61_647 62_186 63_318 "
        "64_417 65_380 66_113 67_847 68_291 69_865 70_879 71_927 72_086 73_101 74_406 75_085 "
        "76_686 83_813 84_369 85_452 86_672 87_529 88_491 89_707 90_978 91_012 92_779 93_184 "
        "94_955 95_015 96_433 97_930 98_439 99_636"
    ).split()
]
SHELF_TOP = 0.894  # the pendant's middle glass shelf (ward file), front edge at ward y -1.625
DIAL_WARD = (0.33, -1.665)  # free front corner of that shelf, beside the shelf's device box

KEY_X, KEY_Y = 0.48, -0.04  # key row center (robot frame, before the seed shift)
PUMP_SIZE = (0.229, 0.24, 0.2235)  # infusion_pump model, native
PUMP_YAW = -90.0  # the pump's screen faces -y (the camera side), its pole clamp +y
POLE_AXIS = (0.014, 0.0265)  # pole axis in the iv_stand model's own frame (measured from the mesh)
POLE_RADIUS = 0.019  # measured from the iv_stand mesh
STAND_YAW = -54.0  # legs clear of the robot base and bed; the bag hook reaches to the front right
HOOK_REACH, HOOK_Z = 0.211, 1.70  # the stand's +x hook end from the pole axis, and its low point
# Camera a little behind and left of the earlier (0.30, -1.05, 1.70): the near robot column, its
# plinth, the saline bag and the whole IV stand stay in frame.
CAMERA = {"focal_mm": 30.0, "position": [0.24, -1.20, 1.78], "look_at_dz": 0.20}
KEY_COLORS = ((0.80, 0.12, 0.10), (0.90, 0.61, 0.07), (0.08, 0.43, 0.68))
KEY_PITCH = 0.075
# Rest pose of the pressing (left) hand from its mount: top-down, the arm folded at the column's
# outer (-y, camera) side, out of the way. The reach to a key turns the base joint ~1.3 rad. The
# mount stays 0.25-0.43 m behind the keys: the top-down X5's IK fails ~0.5 m out at key height.
TUCK = (0.0, -0.16, 0.10)
# Oracle reach: Cartesian speed limits (m/s), acceleration (m/s^2) and the hover above the key.
REACH_SPEED, PRESS_SPEED, REACH_ACCEL = 0.55, 0.25, 3.0
HOVER = 0.045
# Native X5 fingertips project 12.6 mm below the grasp-center frame: aim the frame 10 mm above the
# key's center, so the closed fingertips push the cap through its stroke.
PRESS_OFFSET = 0.010


def _ward_to_robot(x, y, shift):
    return np.array([ROBOT_Y - y, x - ROBOT_X]) + shift


def add_keypad(b, x, y, top, keys=MODES, prefix="pump_", pitch=KEY_PITCH, colors=KEY_COLORS):
    """The top-facing training keypad: a rubber base plate and one spring key item per entry.

    `x, y` (before the seed shift) is the key row's center, `top` the plate's upper face plus the
    key cap's rest; keys run along +x. Returns {key: center} (world, after the shift). Author the
    springs with `author_keys` and read them with `measure_keys`.
    """
    b.fixture(f"{prefix}panel_base", (x, y), (0.222, 0.13, 0.014), "rubber", z=top - 0.028)
    centers = {}
    for i, key in enumerate(keys):
        name = f"{prefix}{key}"
        b.item(
            name, "pump_key", (x + (i - (len(keys) - 1) / 2) * pitch, y), top - 0.006 - 0.002,
            color=colors[i % len(colors)], yaw=0.0,
        )
        centers[key] = b.objects[name].position.tolist()
    return centers


def rest_pose(config, task_id):
    """The pressing hand's tucked rest pose (robot frame): TUCK from arm 0's mount (rig right)."""
    from roboclock.envs.lab.robot import mount_positions

    return (np.asarray(mount_positions(config, task_id)[0], dtype=float) + TUCK).tolist()


def build_layout(b):
    x, y, top = KEY_X, KEY_Y, b.z + 0.255
    base = b.z + 0.025  # pump housing bottom (the pole clamp holds it here)
    panel_z = top - 0.028
    centers = add_keypad(b, x, y, top)
    pump_xy = b.xy(x, y + 0.025)
    # The pole passes the clamp on the pump's back (+y) face.
    pole_xy = pump_xy + [0.046, PUMP_SIZE[0] / 2 + POLE_RADIUS]
    # Response dial on the pendant's middle glass shelf (it moves with the ward).
    dial_xy = _ward_to_robot(*DIAL_WARD, b.shift)
    shelf = SHELF_TOP + WARD_Z
    # The dial's order card repeats the pump display's instruction.
    b.indicator("indicator", dial_xy - b.shift, INSTRUCTION_TEXT[b.task.mode], height=shelf + 0.045 - b.z)
    camera = {
        "position": list(CAMERA["position"]),
        "look_at": [float(pump_xy[0] + 0.04), float(pump_xy[1] + 0.095), float(b.z + CAMERA["look_at_dz"])],
        "focal_mm": CAMERA["focal_mm"],
    }
    body_h = panel_z - base
    b.info.update(
        bedside_task="pump_panel",
        pump_panel={
            "centers": centers,
            "pump_origin": [*pump_xy.tolist(), base],
            "pump_yaw": PUMP_YAW,
            "pole_xy": pole_xy.tolist(),
            "dial_xy": dial_xy.tolist(),
            "camera": camera,
            "placement": "IV pole at the bedside equipment pendant, robot beside the bed",
        },
        ward_pose={
            "usd": WARD_USD,
            "translate": [ROBOT_Y + float(b.shift[0]), -ROBOT_X + float(b.shift[1]), WARD_Z],
            "yaw": WARD_YAW,
            "hide": list(TROLLEY),
            "move": dict(WARD_MOVE),
        },
        hide_bench=True,
        # The keypad's support is the pump housing itself, held on the pole.
        work_surface={
            "center": [*pump_xy.tolist(), base + body_h / 2],
            "size": [PUMP_SIZE[1], PUMP_SIZE[0], body_h],
        },
        # At rest, folded near its own column: the reach to the pump is part of the response.
        initial_ee_position=rest_pose(b.config, b.task.task_id),
        initial_ee_yaw=0.0,
        initial_gripper_width=0.0,
    )
    return b


# --- scene authoring (before physics starts) ---------------------------------------------------


def author(backend):
    from roboclock.envs.hospital.props import add_model, model_size

    stage, look = backend.stage, backend.appearance
    info = backend.layout.info["pump_panel"]
    if model_size("infusion_pump") is None:
        raise FileNotFoundError("Required existing hospital infusion_pump USD is missing")
    add_model(
        stage, "infusion_pump", "/World/Bedside/PumpHousing", translate=info["pump_origin"],
        yaw=math.radians(info.get("pump_yaw", 0.0)),
    )
    # Top-facing key overlay keeps the retained six-axis robot's ordinary downward
    # press feasible. It is an explicit training adapter, not an exact pump replica.
    author_keys(stage, info["centers"], look.config["visuals"]["labels"])
    if hasattr(backend, "world"):  # the Isaac scene; plain-USD unit stages get the keys only
        _dress(backend)


def author_keys(stage, centers, labels=True, prefix="pump_"):
    """Spring-return prismatic joints (6 mm stroke along -z) for the keys of `add_keypad`."""
    from pxr import Gf, Sdf, UsdPhysics

    for key, origin in centers.items():
        name, path = f"{prefix}{key}", f"/World/Objects/{prefix}{key}"
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/World/Joints/{name}")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*origin))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLowerLimitAttr(-STROKE)
        joint.CreateUpperLimitAttr(0.0)
        joint.CreateCollisionEnabledAttr(False)
        spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        spring.CreateTypeAttr("force")
        spring.CreateStiffnessAttr(700.0)
        spring.CreateDampingAttr(2.0)
        spring.CreateTargetPositionAttr(0.0)
        spring.CreateTargetVelocityAttr(0.0)
        spring.CreateMaxForceAttr(6.0)
        if labels:
            # Printed on the physical cap (it moves with the rigid key), read from the camera side.
            _flat_text(stage, path + "/KeyName", key.upper(), [0, 0, 0.0065], (0.052, 0.026))


def _bright(color):
    """A tint for the mid-grey plastic_white maps: the hue of `color`, its largest channel at 1."""
    color = np.asarray(color, dtype=float)
    return tuple(float(c) for c in color / max(float(color.max()), 1e-6))


def dress_keypad(
    look, stage, centers, prefix="pump_", colors=KEY_COLORS, bezel=(0.93, 0.93, 0.92), plate=None
):
    """Textured key caps, one molded surround per key and a light plastic bezel (visual only).

    Optional companion of `add_keypad` / `author_keys`: call it from `author` after the keys exist.
    Each cap (and its catalog band) becomes glossy plastic_white tinted with its key color; under
    each cap sits a darker surround of the same hue that the cap sinks into on a press; the base
    plate fixture (`plate`, default f"{prefix}panel_base") becomes a light grey plastic bezel
    (`bezel` tint). No collider or joint changes.
    """
    from pxr import Usd, UsdGeom, UsdShade

    def bind(path, material):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material)

    plate_path = f"/World/Fixtures/{plate or prefix + 'panel_base'}_Visual"
    bind(plate_path, look.textured(f"{prefix}keypad_bezel", "plastic_white", color=bezel, roughness=0.45))
    plate_prim = stage.GetPrimAtPath(plate_path)
    plate_top = None
    if plate_prim.IsValid():
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        plate_top = float(bounds.ComputeWorldBound(plate_prim).ComputeAlignedRange().GetMax()[2])
    for i, (key, center) in enumerate(centers.items()):
        # Tints multiply the map's linear albedo: a saturated, slightly dark cap keeps its hue
        # under the bright ward exposure (the full-brightness tint read as pastel).
        tint = _bright(colors[i % len(colors)])
        cap = look.textured(
            f"{prefix}key_cap_{key}", "plastic_white", color=[0.8 * c for c in tint], roughness=0.28
        )
        rim = look.textured(
            f"{prefix}key_surround_{key}", "plastic_white", color=[0.45 * c for c in tint], roughness=0.5
        )
        for part in ("Shell", "Band"):
            bind(f"/World/Objects/{prefix}{key}/{part}", cap)
        cx, cy, cz = (float(v) for v in center)
        # add_keypad rests the cap 8 mm above the plate (6 mm half height + 2 mm clearance).
        top = cz - 0.014 if plate_top is None else plate_top
        look.rounded(
            f"/World/Bedside/{prefix}KeySurround_{key}", [cx, cy, top + 0.0015],
            [0.066, 0.046, 0.003], rim, 0.0012,
        )


def measure_keys(state, centers, prefix="pump_"):
    """Measured travel, speed, robot contact and axis alignment of each key (the task's inputs)."""
    buttons = {}
    for key, origin in centers.items():
        obj = state.objects[f"{prefix}{key}"]
        origin = np.asarray(origin)
        buttons[key] = {
            "depression_m": float(origin[2] - obj.position[2]),
            "speed_m_s": float(obj.velocity[2]),
            "contact": bool(obj.robot_contact),
            "aligned": bool(np.linalg.norm(obj.position[:2] - origin[:2]) < 0.001),
        }
    return buttons


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_a4_camera_for", None) is not backend.task:
        backend._a4_camera_for = backend.task
        _camera(backend)
    buttons = measure_keys(state, backend.layout.info["pump_panel"]["centers"])
    _pump_screen(backend, buttons)
    return {"pump_panel": buttons}


def press_goal(planner, state, center, since=None, time=None, speed_limit=REACH_SPEED):
    """One reach-and-press Cartesian command toward a key `center` (closed hand, top-down).

    The hand travels to HOVER above the key, its speed ramped up from rest (REACH_ACCEL from the
    reach's start `since` to now, `time`) to `speed_limit` and down into the hover point; once the
    fingertips are over the cap it presses straight down at PRESS_SPEED and stays there. Call it
    every control step until the press is scored.
    """
    target = np.asarray(center, dtype=float) + [0.0, 0.0, PRESS_OFFSET]
    hover = target + [0.0, 0.0, HOVER]
    ee = np.asarray(state.ee_position)
    if np.linalg.norm(ee[:2] - target[:2]) < 0.008 and ee[2] < hover[2] + 0.015:
        return planner.finish(state, target, True, speed=PRESS_SPEED)
    remaining = float(np.linalg.norm(hover - planner.last_goal))
    speed = min(speed_limit, np.sqrt(2.0 * REACH_ACCEL * remaining) + 0.05)
    if since is not None and time is not None:
        speed = min(speed, REACH_ACCEL * max(time - since, 0.0) + 0.05)
    return planner.finish(state, hover, True, speed=speed)


def command(planner, task, state):
    initial = np.asarray(task.layout.info["initial_ee_position"])
    planner.close_width = 0.0
    planner.gentle = False
    if task.event_time is None:  # at rest until the instruction
        return planner.finish(state, initial, True, speed=0.20)
    center = task.layout.info["pump_panel"]["centers"][task.mode]
    if task.goal_achieved:  # lift off the key
        lift = np.asarray(center) + [0.0, 0.0, PRESS_OFFSET + HOVER]
        return planner.finish(state, lift, True, speed=PRESS_SPEED)
    return press_goal(planner, state, center, since=task.event_time, time=task.time)


# --- bedside dressing (visual only) --------------------------------------------------------------


def _text_texture(text, fg, bg, size=(512, 192)):
    """A cached label image: bold lines of text ("\n"), `bg` None for a transparent background."""
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr(("a4", text, fg, bg, size)).encode()).hexdigest()[:20]
    path = cache / f"a4_{key}.png"
    if path.exists():
        return path
    color = (0, 0, 0, 0) if bg is None else tuple(round(255 * c) for c in bg) + (255,)
    image = Image.new("RGBA", size, color)
    draw = ImageDraw.Draw(image)
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    lines = text.split("\n")
    font_size = int(size[1] / len(lines) * 0.72)

    def font(n):
        try:
            return ImageFont.truetype(bold, n)
        except OSError:
            return ImageFont.load_default(size=n)

    while max(font(font_size).getlength(line) for line in lines) > size[0] * 0.92 and font_size > 10:
        font_size -= 4
    for i, line in enumerate(lines):
        draw.text(
            (size[0] / 2, size[1] * (i + 0.5) / len(lines)), line,
            fill=tuple(round(255 * c) for c in fg) + (255,), font=font(font_size), anchor="mm",
        )
    image.save(path)
    return path


def _texture_material(stage, path, texture, emissive=False, cutout=False):
    """A UsdPreviewSurface material showing `texture` through the mesh's "st" coordinates."""
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    tex = UsdShade.Shader.Define(stage, path + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, path + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    if emissive:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
    if cutout:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            tex.ConnectableAPI(), "a"
        )
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _quad(stage, path, texture, center, size, rotations=(), emissive=False, cutout=False):
    """A textured card in its local XZ plane, read from local +y; `rotations` are (axis, deg)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    material = _texture_material(stage, path + "_material", texture, emissive, cutout)
    mesh = UsdGeom.Mesh.Define(stage, path)
    w, h = size
    mesh.CreatePointsAttr(
        [Gf.Vec3f(-w / 2, 0, -h / 2), Gf.Vec3f(-w / 2, 0, h / 2),
         Gf.Vec3f(w / 2, 0, h / 2), Gf.Vec3f(w / 2, 0, -h / 2)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    # Image right edge at local -x: the card reads left to right from its +y side.
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(1, 0), (1, 1), (0, 1), (0, 0)])
    xform = UsdGeom.Xformable(mesh)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    for axis, degrees in rotations:
        {"x": xform.AddRotateXOp, "y": xform.AddRotateYOp, "z": xform.AddRotateZOp}[axis]().Set(
            float(degrees)
        )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _flat_text(stage, path, text, center, size):
    """White text printed flat on a key cap, upright for the camera on the robot's right (-y)."""
    texture = _text_texture(text, (1.0, 1.0, 1.0), None, (512, 256))
    # Card +y (its reading side) turned up, its text-up axis (+z) to +y, its reading line to +x.
    return _quad(stage, path, texture, center, size, rotations=[("z", 180.0), ("x", 90.0)], cutout=True)


def _dress(backend):
    look = backend.appearance
    info = backend.layout.info["pump_panel"]
    _iv_stand(backend, info)
    dress_keypad(look, backend.stage, info["centers"])
    _keypad_legend(backend, info)
    _line(look, info, backend.layout.table_height)
    _robot_base(backend)
    from roboclock.envs.hospital.rig import dial_target

    _face_dial(backend, dial_target(backend, info["camera"]["position"]))  # the policy's head camera
    _screens(backend, info)


def _iv_stand(backend, info):
    """The floor IV stand model at the pump's back, the pole clamp and a hanging saline bag."""
    from roboclock.envs.hospital.props import add_model, model_size

    look, stage = backend.appearance, backend.stage
    px, py = info["pole_xy"]
    yaw = math.radians(STAND_YAW)
    c, s = math.cos(yaw), math.sin(yaw)
    origin = (px - (c * POLE_AXIS[0] - s * POLE_AXIS[1]), py - (s * POLE_AXIS[0] + c * POLE_AXIS[1]), 0.0)
    if model_size("iv_stand") is not None:
        add_model(stage, "iv_stand", "/World/Bedside/IVStand", origin, yaw=yaw)
        # The model's own atlas maps the pole, legs and casters to near-uniform greys (it renders
        # flat): bind brushed stainless to the stand (part_004) and rubber to its four casters.
        from pxr import UsdShade

        for part, material in (
            ("part_004", look.textured("b7_iv_stand_steel", "stainless_brushed", repeats=10.0)),
            *((f"part_00{k}", look.material("rubber")) for k in range(4)),
        ):
            prim = stage.GetPrimAtPath(f"/World/Bedside/IVStand/{part}")
            if prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material)
    else:  # missing model: a plain pole on a cross base
        look.cylinder("/World/Bedside/StandPole", (px, py, 0.9), 0.0125, 1.62, "stainless")
        for k in range(2):
            look.rounded(
                f"/World/Bedside/StandFoot_{k}", (px, py, 0.06),
                (0.50, 0.04, 0.03) if k == 0 else (0.04, 0.50, 0.03), "stainless", 0.01,
            )
    base = info["pump_origin"][2]
    # Pole clamp on the pump's back face (the pump's own clamp knob sits on that face).
    clamp_z = base + 0.137
    clamp = look.textured("b7_pole_clamp", "aluminum_brushed", repeats=14.0)
    look.rounded("/World/Bedside/PoleClamp", (px, py - 0.008, clamp_z), (0.056, 0.054, 0.050), clamp, 0.006)
    look.cylinder("/World/Bedside/PoleClampKnob", (px + 0.038, py, clamp_z), 0.010, 0.022, "rubber", axis="X")


def _keypad_legend(backend, info):
    """Key legend cards standing on the keypad's front edge, readable from the camera side."""
    if not backend.config["visuals"]["labels"]:
        return
    stage = backend.stage
    for i, key in enumerate(MODES):
        cx, cy, cz = info["centers"][key]
        color = ((0.80, 0.12, 0.10), (0.90, 0.61, 0.07), (0.08, 0.43, 0.68))[i]
        texture = _text_texture(key.upper(), (1.0, 1.0, 1.0), color, (512, 160))
        # A legend card standing at the keypad's front edge, facing the camera side (-y).
        _quad(stage, f"/World/Bedside/KeyLegend_{key}", texture, (cx, cy - 0.060, cz - 0.0045), (0.062, 0.019),
              rotations=[("z", 180.0)])


def _line(look, info, table_z):
    """Saline bag on the stand's hook; tubing down into the pump channel and on over the rail to the bed."""
    px, py = info["pole_xy"]
    ox, oy, base = info["pump_origin"]
    tubing = look.material("a4_tubing", color=(0.90, 0.95, 0.97), roughness=0.25)
    yaw = math.radians(STAND_YAW)
    hx, hy = px + HOOK_REACH * math.cos(yaw), py + HOOK_REACH * math.sin(yaw)
    bag = (hx, hy, HOOK_Z - 0.035 - 0.10)  # hanger tab, then the 0.20 m bag body
    look.rounded("/World/Bedside/SalineHanger", (hx, hy, HOOK_Z - 0.02), (0.016, 0.004, 0.035), "iv_bag", 0.002)
    look.rounded("/World/Bedside/SalineBag", bag, (0.12, 0.022, 0.20), "iv_bag", 0.010)
    look.cylinder("/World/Bedside/SalinePort", (hx, hy, bag[2] - 0.112), 0.005, 0.026, "cap_white")
    look.cylinder("/World/Bedside/DripChamber", (hx, hy, bag[2] - 0.15), 0.009, 0.05, look.glass())
    if look.config["visuals"]["labels"]:
        card = look.label(
            "/World/Bedside/SalineLabel", "0.9% SALINE", (hx, hy - 0.0115, bag[2] + 0.025), (0.09, 0.045)
        )
        if card is not None:
            card.AddRotateZOp().Set(180.0)  # a label card reads from +y; this face looks -y
    channel_x, channel_y = ox + 0.115, oy - 0.05  # the pump channel module at the +x end
    top = (hx, hy, bag[2] - 0.176)
    inlet = (channel_x + 0.003, channel_y, base + 0.15)
    # Below the drip chamber the line swings over the pump's channel at once and hangs straight
    # down into it, so it passes left of the order card on the shelf behind (camera side).
    control = [
        top, (hx, hy, top[2] - 0.05), (inlet[0] + 0.004, inlet[1] - 0.004, top[2] - 0.19),
        (inlet[0] + 0.006, inlet[1], inlet[2] + 0.06), inlet,
    ]
    _tube(look, "/World/Bedside/LineIn", control, tubing)
    outlet = (channel_x - 0.01, channel_y, base + 0.005)
    rail = (ox + 0.18, oy + 0.50, table_z + 0.13)
    control = [
        outlet, (outlet[0] + 0.01, outlet[1], base - 0.12), (outlet[0] + 0.08, outlet[1] + 0.20, table_z - 0.10),
        (rail[0] - 0.02, rail[1] - 0.06, rail[2] - 0.06), rail, (rail[0] + 0.12, rail[1] + 0.22, table_z + 0.11),
    ]
    _tube(look, "/World/Bedside/LineOut", control, tubing)


def _tube(look, path, control, material):
    from pxr import UsdShade

    from roboclock.envs.hospital.scene_meshes import tube_arrays
    from roboclock.simulators.isaac.visual_geometry import author_mesh

    points, uv, indices, normals = tube_arrays(control, 0.002, sides=8)
    mesh = author_mesh(look.stage, path, (points, uv, indices), (0, 0, 0), normals)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material.material)
    return mesh


def _robot_base(backend):
    """Both X5 columns stand on one wheeled base parked beside the bed (the columns above the
    table height are the physics mounts)."""
    from roboclock.envs.lab.robot import mount_positions

    look, top = backend.appearance, backend.layout.table_height
    mounts = np.array(mount_positions(backend.config, backend.task.task_id))[:, :2]
    lo, hi = mounts.min(0) - 0.07, mounts.max(0) + 0.07
    lo[0] -= 0.10  # the base extends behind the columns
    center, size = (lo + hi) / 2, hi - lo
    # One mobile robot: matte light grey plastic shells (the columns match the textured arm
    # mounts above them), a powder-coated skirt and column plinths, brushed aluminum collars and
    # hubs, rubber casters.
    shell = look.textured("b7_base_shell", "plastic_matte", color=(0.84, 0.86, 0.88), roughness=0.5)
    coat = look.textured("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90))
    dark = look.textured("b7_base_skirt", "powder_coat_dark", repeats=8.0)
    trim = look.textured("b7_base_trim", "aluminum_brushed", repeats=12.0)
    look.rounded("/World/Bedside/RobotBase", [*center, 0.10], [*size, 0.14], shell, 0.02)
    look.rounded("/World/Bedside/RobotBumper", [*center, 0.035], [*(size + 0.012), 0.03], dark, 0.01)
    look.rounded("/World/Bedside/RobotDeckTrim", [*center, 0.1705], [*(size - 0.03), 0.003], trim, 0.001)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        wheel = [center[0] + sx * (size[0] / 2 - 0.05), center[1] + sy * (size[1] / 2 - 0.05), 0.03]
        look.cylinder(f"/World/Bedside/RobotWheel_{i}", wheel, 0.03, 0.025, "rubber", axis="Y")
        hub = [wheel[0], wheel[1] + sy * 0.0135, wheel[2]]
        look.cylinder(f"/World/Bedside/RobotHub_{i}", hub, 0.014, 0.003, trim, axis="Y")
    for i, m in enumerate(mounts):
        look.rounded(f"/World/Bedside/RobotColumn_{i}", [*m, (0.17 + top) / 2], [0.12, 0.12, top - 0.17], coat, 0.004)
        look.rounded(f"/World/Bedside/RobotPlinth_{i}", [*m, 0.19], [0.15, 0.15, 0.04], dark, 0.006)
        # A collar where the physics mount column (above the table height) continues the column.
        look.rounded(f"/World/Bedside/RobotCollar_{i}", [*m, top], [0.136, 0.136, 0.03], trim, 0.006)


def _face_dial(backend, toward):
    """The shared dial faces +x (the window here): turn it, its housing and label to the camera."""
    from pxr import Gf, UsdGeom

    from roboclock.envs.hospital.backend import INDICATOR_SIZE

    stage, look = backend.stage, backend.appearance
    indicator = next((i for i in backend.layout.indicators if i.name == "indicator"), None)
    dials = stage.GetPrimAtPath("/World/Dials")
    if indicator is None or not dials.IsValid():
        return
    x, y, z = (float(v) for v in indicator.position)
    yaw = math.degrees(math.atan2(toward[1] - y, toward[0] - x))
    # Only this task's dial lives under /World/Dials: pivot the whole group about its axis.
    xform = UsdGeom.Xformable(dials)
    xform.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(x, y, 0.0))
    xform.AddRotateZOp().Set(yaw)
    xform.AddTranslateOp(opSuffix="unpivot").Set(Gf.Vec3d(-x, -y, 0.0))
    for path in ("/World/Equipment/indicator_Housing_Visual", "/World/Labels/indicator_indicator"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    group = look.group("/World/Bedside/DialHousing", (x, y, z + 0.005), rz=yaw)
    look.rounded(group + "/Body", (0.0, 0.0, 0.0), INDICATOR_SIZE, "cabinet_white", 0.004)
    if backend.config["visuals"]["labels"]:
        # The order card clipped to the housing's lower face: the instruction, split at its colon.
        # It sits ORDER_CARD["shift"] along its reading line (group +y), clear of the IV line.
        text = indicator.label.replace(": ", ":\n")
        texture = _text_texture(text, *ALERT, ORDER_CARD["pixels"])
        front, dy, dz = INDICATOR_SIZE[0] / 2, ORDER_CARD["shift"], ORDER_CARD["z"]
        w, h = ORDER_CARD["size"]
        # A paper card stock behind the printed face, held by an aluminum spring clip on top.
        look.rounded(group + "/CardStock", (front + 0.0008, dy, dz), (0.0012, w + 0.006, h + 0.006),
                     look.textured("b7_order_card", "paper", repeats=24.0), 0.0004)
        look.rounded(group + "/CardClip", (front + 0.0016, dy, dz + h / 2 + 0.001), (0.004, 0.034, 0.010),
                     look.textured("b7_card_clip", "aluminum_brushed", repeats=20.0), 0.0012)
        # The card reads from its +y; turned to the dial's facing (+x of the group).
        _quad(stage, group + "/Label", texture, (front + 0.0018, dy, dz), (w, h), rotations=[("z", -90.0)])


SCREEN = {"offset": (-0.040, 0.138), "size": (0.076, 0.102)}  # the pump model's display, from its front
# The order card on the dial housing's face (housing 0.05 x 0.07 x 0.10 m): its center height from
# the housing's center, its size (m) and its texture (px).
ORDER_CARD = {"z": -0.022, "size": (0.150, 0.054), "pixels": (900, 324), "shift": 0.045}
ALERT = ((1.0, 1.0, 1.0), (0.78, 0.06, 0.05))  # white on red
SCREEN_TEXT = {
    # The instruction (INSTRUCTION_TEXT), on screen from the first frame.
    "cue": {
        "stop": ("OCCLUSION\nALARM\nPRESS STOP", *ALERT),
        "mute": ("SILENCE\nALARM\nPRESS MUTE", *ALERT),
        "bolus": ("BOLUS\nHOLD BOLUS\nKEY", *ALERT),
    },
    "done": {
        "stop": ("PUMP\nSTOPPED", (1.0, 1.0, 1.0), (0.05, 0.52, 0.20)),
        "mute": ("ALARM\nSILENCED", (1.0, 1.0, 1.0), (0.05, 0.52, 0.20)),
        "bolus": ("BOLUS\nRUNNING", (1.0, 1.0, 1.0), (0.05, 0.52, 0.20)),
    },
}


def _screens(backend, info):
    """The pump's own display: the instruction from the first frame, then the device's response.

    One screen mesh whose material is swapped: RTX captures show a material rebinding on the next
    frame, while a visibility toggle of two stacked cards blends both on the episode's last frame.
    """
    ox, oy, base = info["pump_origin"]
    mode = backend.task.mode
    center = (ox + SCREEN["offset"][0], oy - PUMP_SIZE[0] / 2 - 0.0015, base + SCREEN["offset"][1])
    materials = {}
    for state, texts in SCREEN_TEXT.items():
        text, fg, bg = texts[mode]
        texture = _text_texture(text, fg, bg, (384, 512))
        materials[state] = _texture_material(
            backend.stage, f"/World/Bedside/PumpScreen_{state}_material", texture, emissive=True
        )
    path = "/World/Bedside/PumpScreen"
    _quad(backend.stage, path, _text_texture(*SCREEN_TEXT["cue"][mode], (384, 512)), center, SCREEN["size"],
          rotations=[("z", 180.0)], emissive=True)
    backend._a4_screen = (backend.task, "cue")
    backend._a4_screen_materials = (path, materials)
    _bind_screen(backend, "cue")


def _bind_screen(backend, state):
    from pxr import UsdShade

    path, materials = backend._a4_screen_materials
    UsdShade.MaterialBindingAPI.Apply(backend.stage.GetPrimAtPath(path)).Bind(materials[state])


def _pump_screen(backend, buttons):
    """Show the display for the device's progress (visual only; the task scores the keys).

    The pump answers its requested key as soon as it is held down (BOLUS after its hold), so the
    response also shows on the final frame of an episode that ends at the scored press.
    """
    task = getattr(backend, "task", None)
    current = getattr(backend, "_a4_screen", None)
    if task is None or current is None or current[0] is not task:
        return
    key = buttons[task.mode]
    held = key["contact"] and key["depression_m"] >= PRESS
    if task.mode == "bolus":
        dt = backend.config["sim"]["physics_dt"]
        held = held and task.pressed_for + dt + 1e-9 >= task.hold_required
    state = "done" if task.event_time is not None and (task.goal_achieved or held) else "cue"
    if state == current[1]:
        return
    _bind_screen(backend, state)
    backend._a4_screen = (task, state)


def _camera(backend):
    """The robot's right side: hand, keypad, pump screen, pendant dial and the bed behind."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["pump_panel"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    camera.set_focal_length(view["focal_mm"] / 1000)
    for _ in range(3):
        backend.world.render()
