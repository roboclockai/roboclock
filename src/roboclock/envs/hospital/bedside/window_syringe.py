"""A3 syringe pump volume cutoff on the bedside prep trolley, with one physical STOP key.

Placement (scene-02 preview left bay, assets/hospital/ward_02_preview_left.usdc): the ward's own
stainless two-tier trolley stands beside the bed. The robot is a wheeled dual-arm base parked on its
aisle side, facing the bed across the trolley. On the trolley's top tier a syringe pump drives a
50 mL syringe (horizontal, plunger toward the robot) whose extension line runs over into a narrow
graduated measuring tube on a printed level scale: 0-8 mL, a green target band 4.5-5.5 mL and a red
over-dose zone. The plunger and its pusher block advance, the tube fills and a bar fills the pump
screen's 0-8 mL gauge (target band marked), all from task.quantity(); the screen's status card turns
from INFUSING to STOPPED and the stream into the tube stops when the key acts. The shared countdown
dial stands on the trolley's far side, behind the pump's screen, turned toward the camera.

The one action is a downward press of the pump's top-facing red STOP key (a sprung prismatic body,
6 mm stroke). device_moved: the key left rest under robot contact (> 1.5 mm). device_actuated: a
measured stroke of at least 4 mm with robot contact, held for 40 ms; it latches, and the pump stops.
The pump body is the invisible support collider (layout work_surface), so the arm cannot pass
through the housing. Everything else here is visual (no colliders) and away from the arm's path.

Mechanics copied (not imported) from bedside/pump_panel.py, which another session may rewrite.
"""

import math

import numpy as np

WARD_USD = "assets/hospital/ward_02_preview_left.usdc"
# The robot origin in the ward file (meters, yaw 0: robot +x is ward +x, toward the bed). The seed
# shift moves the whole ward with the apparatus, so the trolley keeps its place under the pump.
ROBOT_WARD = (-0.02, -1.20)
WARD_Z = 0.002  # lifts the ward 2 mm above the physics ground plane (no z-fighting)
# The trolley's top tray (ward x 0.261-0.667, y -1.539..-0.912, surface z 0.689; its push-handle
# rails rise to 0.80), in the robot frame.
TROLLEY_TOP = 0.689 + WARD_Z
TROLLEY_X = (0.261 - ROBOT_WARD[0], 0.667 - ROBOT_WARD[0])
TROLLEY_Y = (-1.539 - ROBOT_WARD[1], -0.912 - ROBOT_WARD[1])

# Syringe pump body: the syringe axis runs along robot x (plunger toward the robot).
PUMP_XY = (0.45, -0.02)
PUMP_SIZE = (0.30, 0.13, 0.072)
PUMP_TOP = TROLLEY_TOP + PUMP_SIZE[2]
KEY_UV = (-0.095, -0.030)  # STOP key center on the pump's top, from the body center
KEY_SIZE = (0.056, 0.036, 0.012)  # catalog pump_key box
KEY_LIFT = 0.014  # key center above the housing top: 2 mm clear of it at full stroke
STROKE = 0.006
PRESS = 0.004  # measured travel that counts as a press
HOLD = 0.040  # seconds the press must be held
MOVED = 0.0015  # travel under contact that counts as the key having moved

# 50 mL syringe (ID 26.7 mm: 1.79 mm of plunger travel per mL) and the dispensing path. It was a
# 10 mL syringe (OD 17.0, ID 14.5 mm, zero mark at 0.060) until 2026-11-16: the task's 720-1800 mL/h
# are bolus rates a syringe pump reaches only with a 50/60 mL syringe (B. Braun Perfusor Space:
# 1800 mL/h; 10 mL syringe 500 mL/h). Visual only; the cradle, flange, clamp and thumb follow the OD.
SYRINGE_V = 0.032  # syringe axis, across the pump top (toward the camera, +y)
SYRINGE_OD, SYRINGE_ID = 0.0300, 0.0267
SYRINGE_FULL = 50.0  # mL in the syringe at quantity 0
SYRINGE_ZERO_U = 0.080  # inner end of the barrel (the 0 mL mark), from the body center
MM_PER_ML_SYRINGE = 1e-6 / (math.pi * (SYRINGE_ID / 2) ** 2)  # meters of plunger travel per mL
STOPPER = 0.008
ROD = 0.075

# Graduated measuring tube on its level scale (ID 8.5 mm: 17.6 mm of level per mL).
TUBE_XY = (0.645, 0.012)
TUBE_OD, TUBE_ID = 0.0115, 0.0085
TUBE_BASE = 0.012  # holder block under the tube and the scale card
TUBE_BOTTOM = TROLLEY_TOP + TUBE_BASE + 0.012  # inner floor (above a 12 mm solid glass foot): 0 mL
TUBE_HEIGHT = 0.170  # glass, from the holder block up
M_PER_ML_TUBE = 1e-6 / (math.pi * (TUBE_ID / 2) ** 2)
SCALE_MAX = 8.0
CARD_SIZE = (0.066, 0.180)  # level scale card behind the tube (width, height)
CARD_BEHIND = 0.016  # card face behind the tube axis, away from the camera
CARD_Z0 = TROLLEY_TOP + TUBE_BASE + 0.002  # card bottom edge, standing in the holder block

# The shared countdown dial stands on the trolley's +y end, left of the pump in the rig's head
# camera view. RoboDojo rig (2026-11-04): on the far -y side, behind the pump's display, arm 0's
# upper arm stood between it and the head camera for the whole episode.
DIAL_XY = (0.55, 0.20)
CAMERA = {"position": [0.55, 0.45, 1.06], "look_at": [0.46, -0.04, 0.81], "focal_mm": 28.0}


def _ward_translate(shift):
    return [-ROBOT_WARD[0] + float(shift[0]), -ROBOT_WARD[1] + float(shift[1]), WARD_Z]


def press_target(info):
    """Grasp-center target of the oracle press: the X5 fingertips project 12.6 mm below it."""
    return np.asarray(info["key_center"]) + [0.0, 0.0, 0.010]


def build_layout(b):
    task = b.task
    pump = b.xy(*PUMP_XY)
    key_xy = (PUMP_XY[0] + KEY_UV[0], PUMP_XY[1] + KEY_UV[1])
    key_z = PUMP_TOP + KEY_LIFT
    # The item's center is support + half height + 2 mm: put it exactly at key_z.
    b.item(
        "pump_stop",
        "pump_key",
        key_xy,
        key_z - KEY_SIZE[2] / 2 - 0.002,
        color=(0.82, 0.10, 0.08),
        yaw=0.0,
    )
    key_center = b.objects["pump_stop"].position.tolist()
    b.indicator("indicator", DIAL_XY, "STOP WINDOW", height=TROLLEY_TOP + 0.045 - b.z)
    tube = b.xy(*TUBE_XY)
    camera_xy = np.asarray(CAMERA["position"][:2]) + b.shift
    toward = camera_xy - tube
    if "rig" in b.config.get("robot", {}):  # the level scale card faces the policy's head camera
        from roboclock.envs.hospital.rig import head_camera_pose

        toward = head_camera_pose(b.config, b.task.task_id)[0][:2] - tube
    card_yaw = math.degrees(math.atan2(toward[1], toward[0])) - 90.0  # card +y faces the camera
    unit = toward / np.linalg.norm(toward)
    card_xy = tube - unit * CARD_BEHIND
    look_at = np.asarray(CAMERA["look_at"]) + [*b.shift, 0.0]
    camera = {
        "position": [float(camera_xy[0]), float(camera_xy[1]), CAMERA["position"][2]],
        "look_at": look_at.tolist(),
        "focal_mm": CAMERA["focal_mm"],
    }
    b.info.update(
        bedside_task="window_syringe",
        window_syringe={
            "key_center": key_center,
            "pump_center": [*pump.tolist(), TROLLEY_TOP + PUMP_SIZE[2] / 2],
            "pump_size": list(PUMP_SIZE),
            "pump_top": PUMP_TOP,
            "tube_xy": tube.tolist(),
            "card_xy": card_xy.tolist(),
            "card_yaw": card_yaw,
            "camera": camera,
            "band": list(task.band),
            "unit": task.unit,
            "placement": "syringe pump on the bedside prep trolley, robot in the aisle beside the bed",
        },
        ward_pose={"usd": WARD_USD, "translate": _ward_translate(b.shift), "yaw": 0.0, "hide": []},
        hide_bench=True,
        # The key's support is the pump housing itself, standing on the trolley's top tier.
        work_surface={
            "center": [*pump.tolist(), TROLLEY_TOP + PUMP_SIZE[2] / 2],
            "size": list(PUMP_SIZE),
        },
        initial_ee_position=(np.asarray(key_center) + [0.0, 0.0, 0.038]).tolist(),
        initial_ee_yaw=0.0,
        initial_gripper_width=0.0,
    )
    return b


# --- measured key state ---------------------------------------------------------------------------


class KeyMonitor:
    """Travel, contact and the continuous hold of the STOP key, one update per physics tick."""

    def __init__(self):
        self.held_for = 0.0
        self.moved = False
        self.actuated = False
        self.actuated_time = None
        self.last_time = None

    def update(self, time, dt, depression, contact, aligned):
        if self.last_time is not None and time <= self.last_time + 1e-9:
            return  # no physics tick since the last reading (the reset measurement)
        self.last_time = time
        if contact and depression > MOVED:
            self.moved = True
        valid = contact and aligned and PRESS <= depression <= STROKE + 0.0006
        self.held_for = self.held_for + dt if valid else 0.0
        if not self.actuated and self.held_for + 1e-9 >= HOLD:
            self.actuated, self.actuated_time = True, time


def _monitor(backend):
    owner = getattr(backend, "task", None) or backend.layout
    current = getattr(backend, "_a3w_key", None)
    if current is None or current[0] is not owner:
        current = (owner, KeyMonitor())
        backend._a3w_key = current
    return current[1]


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_a3w_camera_for", None) is not backend.task:
        backend._a3w_camera_for = backend.task
        _camera(backend)
    info = backend.layout.info["window_syringe"]
    key = state.objects["pump_stop"]
    origin = np.asarray(info["key_center"])
    depression = float(origin[2] - key.position[2])
    contact = bool(key.robot_contact)
    aligned = bool(np.linalg.norm(np.asarray(key.position[:2]) - origin[:2]) < 0.001)
    monitor = _monitor(backend)
    task = getattr(backend, "task", None)
    time = float(task.time) if task is not None else 0.0
    dt = float(backend.config["sim"]["physics_dt"]) if hasattr(backend, "config") else 1 / 120
    monitor.update(time, dt, depression, contact, aligned)
    if getattr(backend, "_a3w_visual", None) is not None:
        _update_visuals(backend)
    return {
        "device_moved": monitor.moved,
        "device_actuated": monitor.actuated,
        "stop_key": {
            "depression_m": depression,
            "contact": contact,
            "aligned": aligned,
            "held_s": monitor.held_for,
            "actuated_time": monitor.actuated_time,
        },
    }


# --- privileged diagnostic oracle -----------------------------------------------------------------

# Measured from the hover pose (fingertips 19 mm above the cap): the time from the start of the
# descent to the end of the 40 ms hold, 0.158-0.167 s in Isaac Sim (3 levels x 5 seeds). The oracle
# starts its press this long before the middle of the window, so the scored action lands at 49-51 %
# of the window on every level (A3 oracle_summary.json of the 2026-11-04 recordings, commit 17295d4).
PRESS_LEAD = 0.17
PRESS_SPEED = 0.20


def command(planner, task, state):
    info = task.layout.info["window_syringe"]
    hover = np.asarray(task.layout.info["initial_ee_position"])
    planner.close_width, planner.open_width, planner.gentle = 0.0, 0.0, False
    if task.action_time is not None:
        return planner.finish(state, hover, True, speed=PRESS_SPEED)  # lift off the stopped pump
    middle = 0.5 * (task.window_open + task.window_close)
    start = middle - PRESS_LEAD
    if task.time + 1e-9 < start:
        return planner.finish(state, hover, True, speed=PRESS_SPEED)
    if getattr(planner, "_a3w_press_for", None) is not task:
        planner._a3w_press_for = task
        task.event_log.append({"time": task.time, "type": "oracle_press_start"})
    return planner.finish(state, press_target(info), True, speed=PRESS_SPEED)


# --- scene authoring (before physics starts) ------------------------------------------------------


def author(backend):
    from pxr import Gf, Sdf, UsdPhysics

    stage = backend.stage
    info = backend.layout.info["window_syringe"]
    origin = info["key_center"]
    path = "/World/Objects/pump_stop"
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/pump_stop")
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
    if backend.config["visuals"]["labels"]:
        # Printed on the physical cap (it moves with the rigid key), read from the camera side (+y).
        texture = _text_texture("STOP", (1.0, 1.0, 1.0), None, (512, 256))
        _quad(
            stage,
            path + "/KeyName",
            texture,
            [0, 0, KEY_SIZE[2] / 2 + 0.0005],
            (0.050, 0.025),
            rotations=[("x", 90.0)],
            cutout=True,
        )
    if hasattr(backend, "world"):  # the Isaac scene; plain-USD unit stages get the key only
        _dress(backend)


# --- dressing (visual only) -----------------------------------------------------------------------


def _text_texture(text, fg, bg, size=(512, 192)):
    """A cached label image: bold lines of text ("\n"), `bg` None for a transparent background."""
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr(("a3w", text, fg, bg, size)).encode()).hexdigest()[:20]
    path = cache / f"a3w_{key}.png"
    if path.exists():
        return path
    color = (0, 0, 0, 0) if bg is None else tuple(round(255 * c) for c in bg) + (255,)
    image = Image.new("RGBA", size, color)
    draw = ImageDraw.Draw(image)
    lines = text.split("\n")
    font_size = int(size[1] / len(lines) * 0.72)
    while (
        max(_font(font_size).getlength(line) for line in lines) > size[0] * 0.92 and font_size > 10
    ):
        font_size -= 4
    for i, line in enumerate(lines):
        draw.text(
            (size[0] / 2, size[1] * (i + 0.5) / len(lines)),
            line,
            fill=tuple(round(255 * c) for c in fg) + (255,),
            font=_font(font_size),
            anchor="mm",
        )
    image.save(path)
    return path


def _font(size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def scale_rows(height_px, z0, z1, band, top=SCALE_MAX):
    """Pixel rows of the scale card for levels 0..top mL (row 0 at the card's top edge z1)."""

    def row(ml):
        z = TUBE_BOTTOM + ml * M_PER_ML_TUBE
        return (z1 - z) / (z1 - z0) * height_px

    return {"row": row, "band": (row(band[1]), row(band[0])), "over": (row(top), row(band[1]))}


def _scale_texture(band):
    """The printed level scale: 0.5 mL ticks, numbers each mL, green band and red over zone."""
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw

    width_px = 420
    height_px = round(width_px * CARD_SIZE[1] / CARD_SIZE[0])
    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        repr(("a3w_scale_v3", band, CARD_SIZE, TUBE_ID, round(TUBE_BOTTOM - CARD_Z0, 5))).encode()
    ).hexdigest()[:20]
    path = cache / f"a3w_scale_{key}.png"
    if path.exists():
        return path
    z0, z1 = CARD_Z0, CARD_Z0 + CARD_SIZE[1]
    rows = scale_rows(height_px, z0, z1, band)
    image = Image.new("RGBA", (width_px, height_px), (250, 250, 247, 255))
    draw = ImageDraw.Draw(image)
    over = rows["over"]
    draw.rectangle([0, over[0], width_px, over[1]], fill=(244, 170, 160, 255))
    hi, lo = rows["band"]
    draw.rectangle([0, hi, width_px, lo], fill=(60, 190, 90, 255))
    tube_px = width_px * (TUBE_OD + 0.004) / CARD_SIZE[0]
    left_edge, right_edge = width_px / 2 - tube_px / 2, width_px / 2 + tube_px / 2
    for k in range(int(SCALE_MAX * 2) + 1):
        ml = k / 2
        y = rows["row"](ml)
        major = k % 2 == 0
        length = width_px * (0.16 if major else 0.09)
        thick = 7 if major else 4
        draw.rectangle(
            [left_edge - length, y - thick / 2, left_edge, y + thick / 2], fill=(20, 20, 24, 255)
        )
        draw.rectangle(
            [right_edge, y - thick / 2, right_edge + length, y + thick / 2], fill=(20, 20, 24, 255)
        )
        if major:
            draw.text(
                (left_edge - length - 10, y),
                f"{int(ml)}",
                fill=(15, 15, 20, 255),
                font=_font(76),
                anchor="rm",
            )
    draw.text(
        (width_px / 2, rows["row"](SCALE_MAX) - 40),
        "mL",
        fill=(15, 15, 20, 255),
        font=_font(56),
        anchor="mm",
    )
    mid = (hi + lo) / 2
    draw.text((right_edge + 14, mid), "OK", fill=(255, 255, 255, 255), font=_font(54), anchor="lm")
    draw.text(
        (right_edge + 14, (over[0] + over[1]) / 2),
        "HIGH",
        fill=(150, 20, 20, 255),
        font=_font(44),
        anchor="lm",
    )
    image.save(path)
    return path


def _quad(stage, path, texture, center, size, rotations=(), emissive=False, cutout=False):
    """A textured card in its local XZ plane, read from local +y; `rotations` are (axis, deg)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    material = UsdShade.Material.Define(stage, path + "_material")
    shader = UsdShade.Shader.Define(stage, path + "_material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
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
    if cutout:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
            tex.ConnectableAPI(), "a"
        )
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, path)
    w, h = size
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-w / 2, 0, -h / 2),
            Gf.Vec3f(-w / 2, 0, h / 2),
            Gf.Vec3f(w / 2, 0, h / 2),
            Gf.Vec3f(w / 2, 0, -h / 2),
        ]
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


def _movable(stage, path, shape, material, size, axis="Z"):
    """A unit cylinder or cube with translate/scale ops, for parts that move or grow each tick."""
    from pxr import Gf, UsdGeom, UsdShade

    if shape == "cylinder":
        prim = UsdGeom.Cylinder.Define(stage, path)
        prim.CreateRadiusAttr(float(size))
        prim.CreateHeightAttr(1.0)
        prim.CreateAxisAttr(axis)
    else:
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(1.0)
    translate = prim.AddTranslateOp()
    scale = prim.AddScaleOp()
    translate.Set(Gf.Vec3d(0, 0, 0))
    scale.Set(Gf.Vec3f(1, 1, 1))
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material.material)
    return prim, translate, scale


def _material(look, name, color, roughness=0.4, emissive=0.0, opacity=None):
    """An OmniPBR material, optionally glowing a little or see-through (clear plastic)."""
    from pxr import Gf, Sdf

    first = name not in look.materials
    material = look.material(name, color=color, roughness=roughness)
    if first:
        shader = material.shaders_list[0]
        if emissive:
            shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(emissive))
        if opacity is not None:
            shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))
    return material


class _Visuals:
    """Handles to the parts that follow the process each tick."""

    def __init__(self, task):
        self.task = task
        self.quantity = None
        self.stopped = None


def _dress(backend):
    look, stage = backend.appearance, backend.stage
    info = backend.layout.info["window_syringe"]
    labels = backend.config["visuals"]["labels"]
    visual = _Visuals(backend.task)
    px, py, _ = info["pump_center"]
    top = info["pump_top"]
    sx, sy, sz = PUMP_SIZE
    housing = look.material("a3w_housing", color=(0.90, 0.91, 0.92), roughness=0.45)
    trim = look.material("a3w_trim", color=(0.16, 0.36, 0.62), roughness=0.4)
    dark = look.material("a3w_dark", color=(0.10, 0.11, 0.12), roughness=0.5)
    gray = look.material("a3w_gray", color=(0.55, 0.57, 0.60), roughness=0.4)
    rod_white = look.material("a3w_rod", color=(0.96, 0.96, 0.95), roughness=0.35)
    rubber = look.material("a3w_rubber", color=(0.06, 0.06, 0.07), roughness=0.7)
    liquid = _material(look, "a3w_liquid", (0.08, 0.40, 0.95), roughness=0.15, emissive=2.5)
    glass = _material(look, "a3w_clear", (0.94, 0.97, 1.0), roughness=0.08, opacity=0.22)

    # Pump housing on rubber feet, a blue front trim band and the syringe cradle along +x.
    base = top - sz
    look.rounded(
        "/World/A3W/Pump/Body",
        (px, py, base + sz / 2 + 0.004),
        (sx, sy, sz - 0.008),
        housing,
        0.012,
    )
    for i, (fx, fy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        look.cylinder(
            f"/World/A3W/Pump/Foot_{i}",
            (px + fx * (sx / 2 - 0.03), py + fy * (sy / 2 - 0.025), base + 0.0045),
            0.010,
            0.009,
            rubber,
        )
    look.rounded(
        "/World/A3W/Pump/Trim",
        (px, py + sy / 2 + 0.0005, base + 0.022),
        (sx - 0.02, 0.003, 0.016),
        trim,
        0.001,
    )
    sv = py + SYRINGE_V
    axis_z = top + 0.004 + SYRINGE_OD / 2  # the barrel rests 3 mm deep in the cradle
    cradle = (px - 0.080, px + SYRINGE_ZERO_U + 0.010)  # from under the plunger rod to the luer tip
    look.rounded(
        "/World/A3W/Pump/Cradle",
        ((cradle[0] + cradle[1]) / 2, sv, top + 0.003),
        (cradle[1] - cradle[0], SYRINGE_OD + 0.007, 0.008),
        gray,
        0.002,
    )
    # Lead screw and guide rod under the pusher block, from the drive end to the barrel flange.
    look.cylinder(
        "/World/A3W/Pump/LeadScrew",
        (px - 0.105, sv, top + 0.004),
        0.003,
        0.085,
        look.material("stainless"),
        axis="X",
    )
    look.rounded(
        "/World/A3W/Pump/DriveCap",
        (px - 0.140, sv, top + 0.012),
        (0.016, 0.030, 0.024),
        housing,
        0.004,
    )

    # Syringe barrel (clear), flange, luer tip, and the barrel clamp arm.
    zero_u = px + SYRINGE_ZERO_U
    barrel_len = SYRINGE_FULL * MM_PER_ML_SYRINGE + 0.012
    barrel_c = zero_u - barrel_len / 2 + 0.002
    look.cylinder(
        "/World/A3W/Syringe/Barrel",
        (barrel_c, sv, axis_z),
        SYRINGE_OD / 2,
        barrel_len,
        glass,
        axis="X",
    )
    flange_u = zero_u - barrel_len + 0.002
    look.rounded(
        "/World/A3W/Syringe/Flange",
        (flange_u, sv, axis_z),
        (0.003, SYRINGE_OD + 0.017, SYRINGE_OD + 0.003),
        rod_white,
        0.001,
    )
    look.cylinder(
        "/World/A3W/Syringe/Tip", (zero_u + 0.008, sv, axis_z), 0.0028, 0.016, rod_white, axis="X"
    )
    look.rounded(
        "/World/A3W/Pump/ClampArm",
        (px + 0.020, sv, axis_z + SYRINGE_OD / 2 + 0.003),
        (0.018, SYRINGE_OD + 0.009, 0.005),
        gray,
        0.002,
    )
    look.rounded(
        "/World/A3W/Pump/ClampPost",
        (px + 0.020, sv - SYRINGE_OD / 2 - 0.0075, top + (SYRINGE_OD + 0.007) / 2),
        (0.012, 0.006, SYRINGE_OD + 0.007),
        gray,
        0.002,
    )

    # The moving plunger assembly (stopper, rod, thumb press, pusher block), placed for quantity 0.
    from pxr import Gf, UsdGeom

    group = UsdGeom.Xform.Define(stage, "/World/A3W/Plunger")
    visual.plunger = group.AddTranslateOp()
    visual.plunger.Set(Gf.Vec3d(0, 0, 0))
    stopper_front = zero_u - SYRINGE_FULL * MM_PER_ML_SYRINGE
    look.cylinder(
        "/World/A3W/Plunger/Stopper",
        (stopper_front - STOPPER / 2, sv, axis_z),
        SYRINGE_ID / 2,
        STOPPER,
        rubber,
        axis="X",
    )
    look.rounded(
        "/World/A3W/Plunger/Rod",
        (stopper_front - STOPPER - ROD / 2, sv, axis_z),
        (ROD, 0.62 * SYRINGE_ID, 0.62 * SYRINGE_ID),
        rod_white,
        0.001,
    )
    thumb = stopper_front - STOPPER - ROD
    look.cylinder(
        "/World/A3W/Plunger/Thumb",
        (thumb - 0.0015, sv, axis_z),
        SYRINGE_OD / 2 + 0.0025,
        0.003,
        rod_white,
        axis="X",
    )
    look.rounded(
        "/World/A3W/Plunger/Pusher",
        (thumb - 0.003 - 0.011, sv, top + 0.019),
        (0.022, 0.040, 0.038),
        housing,
        0.004,
    )
    look.rounded(
        "/World/A3W/Plunger/PusherBand",
        (thumb - 0.003 - 0.011, sv + 0.0205, top + 0.024),
        (0.018, 0.002, 0.012),
        trim,
        0.0008,
    )
    # Medication left in the barrel: shrinks from the stopper toward the tip.
    _, visual.barrel_t, visual.barrel_s = _movable(
        stage, "/World/A3W/Syringe/Liquid", "cylinder", liquid, SYRINGE_ID / 2 - 0.0004, axis="X"
    )
    visual.syringe = (zero_u, sv, axis_z)

    # Display console on the back half of the top, tilted toward the camera, and the key boot.
    _console(backend, visual, info, labels)
    kx, ky, kz = info["key_center"]
    look.rounded(
        "/World/A3W/Pump/KeyBoot",
        (kx, ky, top + 0.004),
        (KEY_SIZE[0] - 0.006, KEY_SIZE[1] - 0.006, 0.008),
        rubber,
        0.002,
    )
    look.rounded(
        "/World/A3W/Pump/KeyBezel",
        (kx, ky, top + 0.0012),
        (KEY_SIZE[0] + 0.012, KEY_SIZE[1] + 0.012, 0.0024),
        dark,
        0.002,
    )

    _tube(backend, visual, info, labels, liquid, glass)
    _line(backend, info, visual)
    _robot_base(backend)
    from roboclock.envs.hospital.rig import dial_target

    _face_dial(backend, dial_target(backend, info["camera"]["position"]))  # the policy's head camera
    dial = getattr(backend, "timer_bars", {}).get("indicator")
    if dial is not None:  # frame 0 already shows the waiting state the task sends on its first tick
        dial.set(1.0, "amber")
    backend._a3w_visual = visual
    _update_visuals(backend)


# The pump screen's volume gauge, in screen text coordinates (right, up), meters.
GAUGE = {"x0": -0.058, "x1": 0.058, "up": -0.005, "height": 0.0065}
GAUGE_CARD = {"center": (0.0, -0.008), "size": (0.128, 0.024)}


def _gauge_texture(band):
    """The screen's printed gauge: track outline, 0-8 mL ticks and numbers, green band brackets."""
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw

    width_px = 1024
    height_px = round(width_px * GAUGE_CARD["size"][1] / GAUGE_CARD["size"][0])
    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr(("a3w_gauge_v2", band, GAUGE, GAUGE_CARD)).encode()).hexdigest()[:20]
    path = cache / f"a3w_gauge_{key}.png"
    if path.exists():
        return path
    (cx, cy), (w, h) = GAUGE_CARD["center"], GAUGE_CARD["size"]

    def col(x):
        return (x - (cx - w / 2)) / w * width_px

    def row(up):
        return ((cy + h / 2) - up) / h * height_px

    image = Image.new("RGBA", (width_px, height_px), (5, 8, 10, 255))
    draw = ImageDraw.Draw(image)
    top, bottom = row(GAUGE["up"] + GAUGE["height"] / 2), row(GAUGE["up"] - GAUGE["height"] / 2)
    x0, x1 = col(GAUGE["x0"]), col(GAUGE["x1"])
    gauge = lambda ml: x0 + (x1 - x0) * ml / SCALE_MAX  # noqa: E731
    lo, hi = gauge(band[0]), gauge(band[1])
    draw.rectangle(
        [lo, top - 22, hi, bottom + 22], fill=(40, 190, 90, 255)
    )  # band, around the track
    draw.rectangle(
        [x0 - 6, top - 6, x1 + 6, bottom + 6],
        fill=(5, 8, 10, 255),
        outline=(150, 160, 165, 255),
        width=4,
    )
    for k in range(int(SCALE_MAX) + 1):
        x = gauge(k)
        draw.rectangle([x - 3, bottom + 8, x + 3, bottom + 30], fill=(200, 210, 215, 255))
        if k % 2 == 0:
            draw.text(
                (x, bottom + 34), f"{k}", fill=(220, 230, 235, 255), font=_font(52), anchor="mt"
            )
    image.save(path)
    return path


def _console(backend, visual, info, labels):
    """A tilted screen: status (INFUSING or STOPPED), the volume as a filling bar on a 0-8 mL gauge
    with the target band, and the target card on the pump's front.

    The switching parts are world-level prims whose only op is a translation (or translation and
    scale): RTX in Isaac Sim 6.0.1 kept showing small parts switched off by visibility, material or
    a short move (a seven-segment readout read 8.8), so the volume is a continuously growing bar,
    like the tube's level, and a status card that switches off drops far out of sight.
    """
    from pxr import Gf, UsdGeom, UsdShade

    from roboclock.simulators.isaac.visual_geometry import author_mesh, rounded_box

    look, stage = backend.appearance, backend.stage
    px, py, _ = info["pump_center"]
    top = info["pump_top"]
    tilt = -50.0  # local +z (the screen normal) turns toward +y (the camera side) and up
    center = np.array([px + 0.050, py - 0.034, top + 0.020])
    group = look.group("/World/A3W/Console", center, rx=tilt)
    c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def local(right, up, z):  # screen text coordinates (read from the camera side) to console local
        return np.array([-right, -up, z])

    def world(point):
        return Gf.Vec3d(*(center + rotation @ np.asarray(point, dtype=float)).tolist())

    housing = look.material("a3w_housing")
    screen = look.material("a3w_screen", color=(0.02, 0.03, 0.04), roughness=0.2)
    look.rounded(group + "/Frame", (0, 0, 0), (0.150, 0.058, 0.014), housing, 0.004)
    look.rounded(group + "/Screen", (0, 0, 0.0072), (0.136, 0.046, 0.0008), screen, 0.0004)
    # The console stands on a wedge so it does not float above the housing.
    look.rounded(
        "/World/A3W/Console_Riser",
        (center[0], center[1] - 0.006, top + 0.008),
        (0.150, 0.040, 0.016),
        housing,
        0.003,
    )
    surface = 0.0080
    # The fill bar: a box mesh with the tilt baked into its points, from its origin at the gauge's
    # 0 mL end toward text-right (-x); its x scale is the bar's length.
    fill = _material(look, "a3w_gauge_fill", (0.25, 0.70, 1.0), roughness=0.3, emissive=6.0)
    points, normals, uv, indices = rounded_box((1.0, GAUGE["height"] * 0.7, 0.001), 0.0002)
    points = points - [0.5, 0.0, 0.0]  # x in [-1, 0]
    start = world(local(GAUGE["x0"], GAUGE["up"], surface + 0.0003))
    mesh = author_mesh(
        stage,
        "/World/A3W/Readout/GaugeFill",
        ((rotation @ points.T).T, uv, indices),
        start,
        (rotation @ normals.T).T,
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(fill.material)
    visual.gauge = mesh.AddScaleOp()
    visual.gauge_length = GAUGE["x1"] - GAUGE["x0"]
    visual.status = {}
    if labels:
        (gx, gy), size = GAUGE_CARD["center"], GAUGE_CARD["size"]
        _quad(
            stage,
            group + "/Gauge",
            _gauge_texture(tuple(info["band"])),
            local(gx, gy, surface),
            size,
            rotations=[("x", 90.0)],
            emissive=True,
        )
        title = _text_texture("VOLUME mL", (0.75, 0.85, 0.90), None, (512, 128))
        _quad(
            stage,
            group + "/Title",
            title,
            local(-0.040, 0.012, surface + 0.0002),
            (0.046, 0.0115),
            rotations=[("x", 90.0)],
            emissive=True,
            cutout=True,
        )
        for state, text, fg, bg in (
            ("running", "INFUSING", (0.05, 0.05, 0.05), (0.30, 0.95, 0.45)),
            ("stopped", "STOPPED", (1.0, 1.0, 1.0), (0.85, 0.10, 0.08)),
        ):
            texture = _text_texture(text, fg, bg, (512, 160))
            shown = world(local(0.038, 0.012, surface + 0.0002))
            card = _quad(
                stage,
                f"/World/A3W/Readout/Status_{state}",
                texture,
                shown,
                (0.046, 0.0145),
                rotations=[("x", tilt + 90.0)],
                emissive=True,
            )
            visual.status[state] = (UsdGeom.Xformable(card).GetOrderedXformOps()[0], shown)
        lo, hi = info["band"]
        target = _text_texture(
            f"STOP AT {lo:g}-{hi:g} mL", (1.0, 1.0, 1.0), (0.16, 0.36, 0.62), (768, 120)
        )
        _quad(
            stage,
            "/World/A3W/Pump/TargetCard",
            target,
            (px - 0.010, py + PUMP_SIZE[1] / 2 + 0.0025, top - 0.020),
            (0.140, 0.022),
        )


def _tube(backend, visual, info, labels, liquid, glass):
    """The graduated measuring tube on its foot, the level-scale card behind it, and the level."""
    look, stage = backend.appearance, backend.stage
    tx, ty = info["tube_xy"]
    gray = look.material("a3w_gray")
    # Holder block (turned to the camera) with the scale card standing in its back edge.
    cx, cy = info["card_xy"]
    group = look.group("/World/A3W/Scale", (cx, cy, TROLLEY_TOP), rz=info["card_yaw"])
    look.rounded(
        group + "/Base",
        (0, 0.006, TUBE_BASE / 2),
        (CARD_SIZE[0] + 0.008, 0.040, TUBE_BASE),
        gray,
        0.003,
    )
    card_z = CARD_Z0 - TROLLEY_TOP + CARD_SIZE[1] / 2
    white = look.material("a3w_card_back", color=(0.93, 0.94, 0.95), roughness=0.5)
    look.rounded(
        group + "/Backing",
        (0, -0.0022, card_z - 0.001),
        (CARD_SIZE[0] + 0.006, 0.003, CARD_SIZE[1] + 0.004),
        white,
        0.002,
    )
    _quad(stage, group + "/Face", _scale_texture(tuple(info["band"])), (0, 0, card_z), CARD_SIZE)
    # The tube: a collar in the holder block and the clear wall with a pour lip.
    glass_base = TROLLEY_TOP + TUBE_BASE
    look.cylinder(
        "/World/A3W/Tube/Collar", (tx, ty, glass_base + 0.006), TUBE_OD / 2 + 0.003, 0.012, gray
    )
    look.cylinder(
        "/World/A3W/Tube/Glass",
        (tx, ty, glass_base + TUBE_HEIGHT / 2),
        TUBE_OD / 2,
        TUBE_HEIGHT,
        glass,
    )
    look.cylinder(
        "/World/A3W/Tube/Lip",
        (tx, ty, glass_base + TUBE_HEIGHT - 0.002),
        TUBE_OD / 2 + 0.0012,
        0.004,
        glass,
    )
    _, visual.level_t, visual.level_s = _movable(
        stage, "/World/A3W/Tube/Liquid", "cylinder", liquid, TUBE_ID / 2 - 0.0003
    )
    visual.tube = (tx, ty)
    _, visual.stream_t, visual.stream_s = _movable(
        stage, "/World/A3W/Tube/Stream", "cylinder", liquid, 0.0008
    )


def _line(backend, info, visual):
    """The extension line from the syringe's luer tip over into the top of the measuring tube."""
    from pxr import UsdShade

    from roboclock.envs.hospital.scene_meshes import tube_arrays
    from roboclock.simulators.isaac.visual_geometry import author_mesh

    look = backend.appearance
    zero_u, sv, axis_z = visual.syringe
    tx, ty = info["tube_xy"]
    rim = TROLLEY_TOP + TUBE_BASE + TUBE_HEIGHT
    outlet = (tx, ty, rim - 0.012)
    control = [
        (zero_u + 0.016, sv, axis_z),
        (zero_u + 0.040, sv, axis_z + 0.004),
        (tx - 0.030, (sv + ty) / 2, rim + 0.030),
        (tx, ty, rim + 0.022),
        outlet,
    ]
    points, uv, indices, normals = tube_arrays(control, 0.0016, sides=8)
    tubing = look.material("a3w_tubing", color=(0.88, 0.93, 0.96), roughness=0.25)
    mesh = author_mesh(look.stage, "/World/A3W/Line", (points, uv, indices), (0, 0, 0), normals)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(tubing.material)
    visual.outlet_z = outlet[2]


FAR = 1.0  # a switched-off readout part drops this far, out of sight below the floor


def _update_visuals(backend):
    """Plunger travel, the barrel, tube and gauge levels, the falling stream and the status card.

    Continuous parts move every tick; the stream and the status card that switch off drop FAR below
    their place instead of being hidden (see _console).
    """
    from pxr import Gf

    visual = backend._a3w_visual
    task = backend.task
    if visual.task is not task:
        return
    quantity = float(task.quantity())
    key = getattr(backend, "_a3w_key", None)
    stopped = task.action_time is not None or (
        key is not None and key[0] is task and key[1].actuated
    )
    if (
        visual.quantity is not None
        and abs(quantity - visual.quantity) < 0.004
        and stopped == visual.stopped
    ):
        return
    visual.quantity = quantity
    q = min(max(quantity, 0.0), SCALE_MAX + 0.5)
    visual.plunger.Set(Gf.Vec3d(q * MM_PER_ML_SYRINGE, 0.0, 0.0))
    zero_u, sv, axis_z = visual.syringe
    left = max(SYRINGE_FULL - q, 0.0) * MM_PER_ML_SYRINGE
    visual.barrel_t.Set(Gf.Vec3d(zero_u - left / 2, sv, axis_z))
    visual.barrel_s.Set(Gf.Vec3f(max(left, 1e-4), 1.0, 1.0))
    level = max(q * M_PER_ML_TUBE, 1e-4)
    tx, ty = visual.tube
    visual.level_t.Set(Gf.Vec3d(tx, ty, TUBE_BOTTOM + level / 2))
    visual.level_s.Set(Gf.Vec3f(1.0, 1.0, level))
    fall = max(visual.outlet_z - (TUBE_BOTTOM + level), 1e-3)
    visual.stream_t.Set(
        Gf.Vec3d(tx, ty, TUBE_BOTTOM + level + fall / 2 - (FAR if stopped else 0.0))
    )
    visual.stream_s.Set(Gf.Vec3f(1.0, 1.0, fall))
    visual.gauge.Set(Gf.Vec3f(max(min(q / SCALE_MAX, 1.0) * visual.gauge_length, 1e-4), 1.0, 1.0))
    if stopped != visual.stopped:
        visual.stopped = stopped
        for name, (move, shown) in visual.status.items():
            move.Set(shown if (name == "stopped") == stopped else shown - Gf.Vec3d(0, 0, FAR))


def _robot_base(backend):
    """Both X5 columns stand on one wheeled base parked in the aisle (the columns above the
    table height are the physics mounts)."""
    from roboclock.envs.lab.robot import mount_positions

    look, top = backend.appearance, backend.layout.table_height
    mounts = np.array(mount_positions(backend.config, backend.task.task_id))[:, :2]
    lo, hi = mounts.min(0) - 0.07, mounts.max(0) + 0.07
    lo[0] -= 0.10  # the base extends behind the columns
    center, size = (lo + hi) / 2, hi - lo
    shell = look.material("a3w_base_shell", color=(0.88, 0.90, 0.91), roughness=0.45)
    dark = look.material("a3w_base_dark", color=(0.24, 0.26, 0.28), roughness=0.5)
    look.rounded("/World/A3W/RobotBase", [*center, 0.10], [*size, 0.14], shell, 0.02)
    look.rounded("/World/A3W/RobotBumper", [*center, 0.035], [*(size + 0.012), 0.03], dark, 0.01)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        wheel = [center[0] + sx * (size[0] / 2 - 0.05), center[1] + sy * (size[1] / 2 - 0.05), 0.03]
        look.cylinder(f"/World/A3W/RobotWheel_{i}", wheel, 0.03, 0.025, dark, axis="Y")
    for i, m in enumerate(mounts):
        look.rounded(
            f"/World/A3W/RobotColumn_{i}",
            [*m, (0.17 + top) / 2],
            [0.12, 0.12, top - 0.17],
            dark,
            0.004,
        )
        look.rounded(f"/World/A3W/RobotCollar_{i}", [*m, top], [0.136, 0.136, 0.03], shell, 0.006)


def _face_dial(backend, toward):
    """The shared dial faces +x: turn it, its housing and label toward the camera."""
    from pxr import Gf, UsdGeom

    from roboclock.envs.hospital.backend import INDICATOR_SIZE

    stage, look = backend.stage, backend.appearance
    indicator = next((i for i in backend.layout.indicators if i.name == "indicator"), None)
    dials = stage.GetPrimAtPath("/World/Dials")
    if indicator is None or not dials.IsValid():
        return
    x, y, z = (float(v) for v in indicator.position)
    yaw = math.degrees(math.atan2(toward[1] - y, toward[0] - x))
    xform = UsdGeom.Xformable(dials)
    xform.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(x, y, 0.0))
    xform.AddRotateZOp().Set(yaw)
    xform.AddTranslateOp(opSuffix="unpivot").Set(Gf.Vec3d(-x, -y, 0.0))
    for path in ("/World/Equipment/indicator_Housing_Visual", "/World/Labels/indicator_indicator"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    group = look.group("/World/A3W/DialHousing", (x, y, z + 0.005), rz=yaw)
    look.rounded(group + "/Body", (0.0, 0.0, 0.0), INDICATOR_SIZE, "cabinet_white", 0.004)
    # No label card: the housing's front is hidden behind the pump from the task camera.


def _camera(backend):
    """From the robot's left, over the bed's foot: hand, key, syringe, tube scale and the bay."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["window_syringe"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    camera.set_focal_length(view["focal_mm"] / 1000)
    for _ in range(3):
        backend.world.render()
