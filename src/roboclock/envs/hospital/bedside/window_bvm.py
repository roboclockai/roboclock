"""A3 bag-valve ventilation of an intubated manikin, one physical squeeze of the bag per breath.

Placement (scene-02 preview left bay, assets/hospital/ward_02_preview_left.usdc): the robot, a
wheeled dual-arm base, is parked in the aisle and faces the bed across the ward's two-tier
bedside trolley (the placement of the A1/A2 window tasks). A head-and-torso training manikin lies
supine on the mattress, head toward the bed's head end. A tracheal tube with its 15 mm connector
stands in its mouth; corrugated extension tubing runs from the connector over the bed rail to the
patient valve of a self-inflating resuscitator bag that lies on a non-slip mat on the trolley,
bag axis toward the bed. A vital-signs monitor on a roll stand beside the trolley shows the VENT
TIMER: a bar that sweeps once per 6 s breath cycle over a scale with the due band marked, and a
status card (WAIT / SQUEEZE). The suite's countdown dial stands on the trolley.

Mechanism. The bag's upper wall is a rigid pad on a sprung, damped prismatic joint (40 mm stroke,
the bag re-expands by itself). A breath is delivered when the pad has been pushed DELIVER (28 mm)
down under robot contact; the bag must then come back above RELEASE (10 mm) before another
breath can count. The scanned bag mesh is deformed every tick to follow the pad (its top wall
goes down with it, the sides bulge), and the manikin's chest rises with the compression and falls
again when the bag is let go. device_moved: the pad pushed more than 3 mm under contact.
Everything except the pad and the trolley-top support collider is visual.

Models (assets/models/credits.json, all CC BY 4.0, imported by
scripts/hospital/import_bvm_assets.py): bvm_resuscitator, bvm_manikin, bvm_tube,
patient_monitor. The mat, the extension tubing, the monitor's screen content and the dial are
built here.
"""

import math

import numpy as np

from roboclock.envs.hospital.bedside.window_syringe import (
    FAR,
    TROLLEY_TOP,
    WARD_USD,
    WARD_Z,
    _face_dial,
    _font,
    _material,
    _quad,
    _robot_base,
    _text_texture,
    _ward_translate,
)

MATTRESS_TOP = 0.665 + WARD_Z  # the bed's flat mattress section (ward y -1.90..-0.67)

# Resuscitator: model origin under the middle of the bag, bag axis along robot +x (toward the bed).
BAG_XY = (0.43, -0.02)
MAT = 0.004  # non-slip mat under the bag
BAG_BASE = TROLLEY_TOP + MAT
PAD_SIZE = (0.10, 0.08, 0.012)  # catalog bvm_pad box: the bag's upper wall under the fingers
STROKE = 0.040
DELIVER = 0.028  # pad travel that delivers a breath (about 70 % of the stroke)
RELEASE = 0.010  # the bag has re-expanded above this travel: the next breath can count
MOVED = 0.003
SPRING = {"stiffness": 150.0, "damping": 40.0, "max_force": 12.0, "preload": 0.004}

# Manikin on the mattress: model origin under the middle of the body, head toward -y.
MANIKIN_XY = (1.13, 0.10)
CHEST = {"center": (0.0, -0.03), "sigma": (0.15, 0.14), "rise": 0.030, "widen": 0.06}
LUNG_FILL, LUNG_EMPTY = 0.15, 0.45  # time constants (s) of the chest rising and falling

# Vital-signs monitor on its roll stand, on the floor beside the trolley.
MONITOR_XY = (0.52, 0.46)
SCREEN = (0.222, 0.150)  # live screen content (width, height), over the monitor head's front
GAUGE = {"y0": -0.090, "y1": 0.090, "z": -0.012, "height": 0.034}
GAUGE_CARD = {"center": (0.0, -0.026), "size": (0.214, 0.092)}

DIAL_XY = (0.56, 0.21)
CAMERA = {"position": [0.10, 0.95, 1.45], "look_at": [0.75, 0.0, 0.80], "focal_mm": 24.0}

# Rest pose of the pressing hand from its mount: top-down, the arm folded at its column's outer
# (-y) side (the pose the bedside reaction tasks start from).
TUCK = (0.0, -0.16, 0.10)
FINGERTIP = 0.01257  # the X5 fingertips project this far below the tool point
HOVER_CLEAR = 0.025  # fingertips above the bag at the hover pose
PRESS_DEPTH = 0.034  # commanded fingertip travel into the bag (past DELIVER, short of STROKE)


def _model(name):
    from roboclock.envs.hospital.props import model_info

    info = model_info(name)
    if info is None:
        raise FileNotFoundError(
            f"hospital model {name} is not installed: run scripts/hospital/import_bvm_assets.py"
        )
    return info


def rest_pose(config, task_id):
    """The pressing hand's tucked rest pose (robot frame): TUCK from arm 0's mount (rig right)."""
    from roboclock.envs.lab.robot import mount_positions

    return (np.asarray(mount_positions(config, task_id)[0], dtype=float) + TUCK).tolist()


def hover_target(info):
    return np.asarray(info["pad_center"]) + [0.0, 0.0, PAD_SIZE[2] / 2 + FINGERTIP + HOVER_CLEAR]


def press_target(info):
    """Grasp-center target of the oracle squeeze: fingertips PRESS_DEPTH below the bag's top."""
    return np.asarray(info["pad_center"]) + [0.0, 0.0, PAD_SIZE[2] / 2 + FINGERTIP - PRESS_DEPTH]


def build_layout(b):
    task = b.task
    bvm = _model("bvm_resuscitator")
    manikin = _model("bvm_manikin")
    monitor = _model("patient_monitor")
    bag_top = BAG_BASE + bvm["bag_top_z_m"]
    # The item's center is support + half height + 2 mm: put the pad's top at the bag's top.
    b.item(
        "bvm_bag",
        "bvm_pad",
        BAG_XY,
        bag_top - PAD_SIZE[2] - 0.002,
        color=(0.20, 0.30, 0.55),
        yaw=0.0,
    )
    pad_center = b.objects["bvm_bag"].position.tolist()
    b.indicator("indicator", DIAL_XY, "BREATH WINDOW", height=TROLLEY_TOP + 0.045 - b.z)
    bag = b.xy(*BAG_XY)
    body = b.xy(*MANIKIN_XY)
    stand = b.xy(*MONITOR_XY)
    camera_xy = np.asarray(CAMERA["position"][:2]) + b.shift
    toward = camera_xy - stand
    if "rig" in b.config.get("robot", {}):  # the screen faces the policy's head camera
        from roboclock.envs.hospital.rig import head_camera_pose

        toward = head_camera_pose(b.config, b.task.task_id)[0][:2] - stand
    monitor_yaw = math.degrees(math.atan2(toward[1], toward[0]))
    face = manikin["face_top_m"]
    port = bvm["patient_port_m"]
    look_at = np.asarray(CAMERA["look_at"]) + [*b.shift, 0.0]
    b.info.update(
        bedside_task="window_bvm",
        window_bvm={
            "pad_center": pad_center,
            "bag_xy": bag.tolist(),
            "bag_base": BAG_BASE,
            "bag_top": bag_top,
            "patient_port": [bag[0] + port[0], bag[1] + port[1], BAG_BASE + port[2]],
            "manikin_xy": body.tolist(),
            "manikin_top": MATTRESS_TOP + manikin["top_z_m"],
            # The mouth: 35 mm toward the chest from the nose tip, 20 mm below it.
            "mouth": [body[0] + face[0], body[1] + face[1] + 0.035, MATTRESS_TOP + face[2] - 0.020],
            "monitor_xy": stand.tolist(),
            "monitor_yaw": monitor_yaw,
            "screen_center": list(monitor["screen_center_m"]),
            "camera": {
                "position": [float(camera_xy[0]), float(camera_xy[1]), CAMERA["position"][2]],
                "look_at": look_at.tolist(),
                "focal_mm": CAMERA["focal_mm"],
            },
            "period": task.period,
            "band": list(task.band()),
            "breaths": len(task.windows),
            "placement": "resuscitator bag on the bedside trolley, manikin on the bed, robot in the aisle",
        },
        ward_pose={"usd": WARD_USD, "translate": _ward_translate(b.shift), "yaw": 0.0, "hide": []},
        hide_bench=True,
        # The support collider is the trolley's top tier under the bag (the pad hangs on its joint).
        work_surface={
            "center": [float(bag[0]), float(bag[1]), TROLLEY_TOP - 0.02],
            "size": [0.40, 0.30, 0.04],
        },
        # At rest, folded near its own column: the reach to the bag is part of the episode.
        initial_ee_position=rest_pose(b.config, b.task.task_id),
        initial_ee_yaw=0.0,
        initial_gripper_width=0.0,
    )
    return b


# --- measured bag state ---------------------------------------------------------------------------


class BagMonitor:
    """Compression, contact, delivered breaths and the lung volume, one update per physics tick."""

    def __init__(self):
        self.moved = False
        self.armed = True  # the bag is expanded: the next full squeeze delivers a breath
        self.breath_times = []
        self.compression = 0.0
        self.lung = 0.0  # 0..1, follows the compression: the chest rise
        self.last_time = None

    def update(self, time, dt, compression, contact, aligned):
        if self.last_time is not None and time <= self.last_time + 1e-9:
            return  # no physics tick since the last reading (the reset measurement)
        self.last_time = time
        self.compression = compression
        if contact and compression > MOVED:
            self.moved = True
        if compression < RELEASE:
            self.armed = True
        if self.armed and contact and aligned and compression >= DELIVER:
            self.armed = False
            self.breath_times.append(time)
        target = min(max(compression, 0.0) / DELIVER, 1.0)
        tau = LUNG_FILL if target > self.lung else LUNG_EMPTY
        self.lung += (target - self.lung) * min(dt / tau, 1.0)


def _monitor(backend):
    owner = getattr(backend, "task", None) or backend.layout
    current = getattr(backend, "_a3v_bag", None)
    if current is None or current[0] is not owner:
        current = (owner, BagMonitor())
        backend._a3v_bag = current
    return current[1]


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_a3v_camera_for", None) is not backend.task:
        backend._a3v_camera_for = backend.task
        _camera(backend)
    info = backend.layout.info["window_bvm"]
    pad = state.objects["bvm_bag"]
    origin = np.asarray(info["pad_center"])
    compression = float(origin[2] - pad.position[2])
    contact = bool(pad.robot_contact)
    aligned = bool(np.linalg.norm(np.asarray(pad.position[:2]) - origin[:2]) < 0.001)
    monitor = _monitor(backend)
    task = getattr(backend, "task", None)
    time = float(task.time) if task is not None else 0.0
    dt = float(backend.config["sim"]["physics_dt"]) if hasattr(backend, "config") else 1 / 250
    monitor.update(time, dt, compression, contact, aligned)
    if getattr(backend, "_a3v_visual", None) is not None:
        _update_visuals(backend)
    return {
        "device_moved": monitor.moved,
        "breath_times": tuple(monitor.breath_times),
        "bag": {
            "compression_m": compression,
            "contact": contact,
            "aligned": aligned,
            "armed": monitor.armed,
            "lung_fraction": monitor.lung,
            "breaths": len(monitor.breath_times),
        },
    }


# --- privileged diagnostic oracle -----------------------------------------------------------------

# Measured in Isaac Sim from the hover pose (fingertips 25 mm above the bag): the time from the
# start of the descent to the pad reaching DELIVER. The oracle starts each squeeze this long
# before the beat, so the breath is delivered on the beat.
PRESS_LEAD = 0.66
PRESS_SPEED = 0.08  # m/s: the squeeze takes about a second, like a bagged breath
RISE_SPEED = 0.20
REACH_SPEED = 0.30  # from the rest pose to the hover over the bag
HOLD = 0.25  # seconds the bag is held squeezed after the breath


def command(planner, task, state):
    info = task.layout.info["window_bvm"]
    hover = hover_target(info)
    planner.close_width, planner.open_width, planner.gentle = 0.0, 0.0, False
    owed = [
        k
        for k, (_, end) in enumerate(task.windows)
        if k not in task.served and task.time <= end + 1e-9
    ]
    last = task.breath_times[-1] if task.breath_times else None
    if last is not None and task.time < last + HOLD:
        return planner.finish(state, press_target(info), True, speed=PRESS_SPEED)
    if not owed or task.time + 1e-9 < task.beats[owed[0]] - PRESS_LEAD:
        speed = RISE_SPEED if task.breath_times else REACH_SPEED
        return planner.finish(state, hover, True, speed=speed)
    if getattr(planner, "_a3v_press_for", None) != (id(task), owed[0]):
        planner._a3v_press_for = (id(task), owed[0])
        task.event_log.append({"time": task.time, "type": "oracle_press_start", "breath": owed[0]})
    return planner.finish(state, press_target(info), True, speed=PRESS_SPEED)


# --- scene authoring (before physics starts) ------------------------------------------------------


def author(backend):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = backend.stage
    info = backend.layout.info["window_bvm"]
    origin = info["pad_center"]
    path = "/World/Objects/bvm_bag"
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/bvm_bag")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*origin))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLowerLimitAttr(-STROKE)
    joint.CreateUpperLimitAttr(0.0)
    joint.CreateCollisionEnabledAttr(False)
    spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    spring.CreateTypeAttr("force")
    spring.CreateStiffnessAttr(SPRING["stiffness"])
    spring.CreateDampingAttr(SPRING["damping"])
    spring.CreateTargetPositionAttr(SPRING["preload"])  # rests against the upper stop
    spring.CreateTargetVelocityAttr(0.0)
    spring.CreateMaxForceAttr(SPRING["max_force"])
    # The pad is the bag's upper wall: the scanned bag mesh is its visual, not the catalog box.
    for name in ("Shell", "Band"):
        prim = stage.GetPrimAtPath(f"{path}/{name}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    if hasattr(backend, "world"):  # the Isaac scene; plain-USD unit stages get the pad only
        _dress(backend)


# --- dressing (visual only) -----------------------------------------------------------------------


def bag_weights(points, span, top):
    """Per-vertex (weight along the bag, height fraction) for the squeeze deformation."""
    x0, x1 = span
    u = np.clip((points[:, 0] - x0) / (x1 - x0), 0.0, 1.0)
    return np.sin(np.pi * u) ** 2, np.clip(points[:, 2] / top, 0.0, 1.2)


def squeezed(points, weights, compression, top):
    """The bag's points with its upper wall pushed `compression` down: the middle of the top wall
    moves by exactly that (it stays under the pad), the bottom stays on the mat, the sides bulge."""
    along, height = weights
    out = points.copy()
    out[:, 2] -= compression * along * height
    out[:, 1] *= 1.0 + 0.30 * along * compression / top
    return out


def chest_weights(points, top):
    (cx, cy), (sx, sy) = CHEST["center"], CHEST["sigma"]
    bump = np.exp(-(((points[:, 0] - cx) / sx) ** 2 + ((points[:, 1] - cy) / sy) ** 2))
    return bump * np.clip((points[:, 2] / top - 0.35) / 0.4, 0.0, 1.0)


def inflated(points, bump, rise):
    """The manikin's points with the chest `rise` meters up; the rib cage also widens a little."""
    out = points.copy()
    out[:, 2] += bump * rise
    out[:, 0] += (points[:, 0] - CHEST["center"][0]) * bump * CHEST["widen"] * rise / CHEST["rise"]
    return out


class _Deform:
    """A referenced mesh whose points follow a scalar (rewritten only when it visibly changes)."""

    def __init__(self, prim, shape):
        from pxr import UsdGeom

        self.attr = UsdGeom.Mesh(prim).GetPointsAttr()
        self.base = np.asarray(self.attr.Get(), dtype=np.float64)
        self.shape = shape
        self.value = 0.0

    def set(self, value, tolerance):
        from pxr import Vt

        if abs(value - self.value) < tolerance:
            return
        self.value = value
        self.attr.Set(Vt.Vec3fArray.FromNumpy(self.shape(self.base, value).astype(np.float32)))


def _gauge_texture(band, period):
    """The timer's printed scale: track outline, second ticks and numbers, the green due band."""
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw

    width_px = 1024
    (cx, cz), (w, h) = GAUGE_CARD["center"], GAUGE_CARD["size"]
    height_px = round(width_px * h / w)
    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr(("a3v_gauge_v1", band, period, GAUGE, GAUGE_CARD)).encode()).hexdigest()
    path = cache / f"a3v_gauge_{key[:20]}.png"
    if path.exists():
        return path

    def col(y):
        return (y - (cx - w / 2)) / w * width_px

    def row(z):
        return ((cz + h / 2) - z) / h * height_px

    image = Image.new("RGBA", (width_px, height_px), (5, 8, 10, 255))
    draw = ImageDraw.Draw(image)
    top, bottom = row(GAUGE["z"] + GAUGE["height"] / 2), row(GAUGE["z"] - GAUGE["height"] / 2)
    x0, x1 = col(GAUGE["y0"]), col(GAUGE["y1"])
    at = lambda seconds: x0 + (x1 - x0) * seconds / period  # noqa: E731
    draw.rectangle([at(band[0]), top - 34, at(band[1]), bottom + 34], fill=(40, 200, 90, 255))
    draw.rectangle(
        [x0 - 8, top - 8, x1 + 8, bottom + 8], fill=(5, 8, 10, 255), outline=(170, 180, 185, 255), width=6
    )
    for k in range(int(period) + 1):
        x = at(k)
        draw.rectangle([x - 4, bottom + 12, x + 4, bottom + 44], fill=(210, 220, 225, 255))
        if k % 2 == 0:
            draw.text((x, bottom + 50), f"{k}", fill=(225, 235, 240, 255), font=_font(84), anchor="mt")
    draw.text((x1 + 14, bottom + 50), "s", fill=(225, 235, 240, 255), font=_font(72), anchor="lt")
    image.save(path)
    return path


class _Visuals:
    def __init__(self, task):
        self.task = task
        self.state = None
        self.bag, self.chest = [], None
        self.status = {}
        self.count = {}


def _dress(backend):
    from pxr import Gf, UsdGeom, UsdShade

    from roboclock.envs.hospital.props import MODELS, add_model
    from roboclock.envs.hospital.rig import dial_target
    from roboclock.envs.hospital.scene_meshes import tube_arrays
    from roboclock.simulators.isaac.visual_geometry import author_mesh

    look, stage = backend.appearance, backend.stage
    info = backend.layout.info["window_bvm"]
    labels = backend.config["visuals"]["labels"]
    visual = _Visuals(backend.task)
    bvm, manikin = _model("bvm_resuscitator"), _model("bvm_manikin")

    # Non-slip mat and the resuscitator on it.
    bx, by = info["bag_xy"]
    mat = look.material("a3v_mat", color=(0.16, 0.30, 0.34), roughness=0.8)
    look.rounded("/World/A3V/Mat", (bx + 0.02, by, TROLLEY_TOP + MAT / 2), (0.36, 0.20, MAT), mat, 0.002)
    add_model(stage, "bvm_resuscitator", "/World/A3V/Bvm", (bx, by, info["bag_base"]))
    top = bvm["bag_top_z_m"]
    for prim in stage.GetPrimAtPath("/World/A3V/Bvm").GetChildren():
        if prim.GetName().startswith("bag_"):
            deform = _Deform(prim, None)
            weights = bag_weights(deform.base, bvm["bag_span_x_m"], top)
            deform.shape = lambda base, d, w=weights: squeezed(base, w, d, top)
            visual.bag.append(deform)

    # The manikin on the mattress, the tracheal tube in its mouth and the extension tubing.
    mx, my = info["manikin_xy"]
    add_model(stage, "bvm_manikin", "/World/A3V/Manikin", (mx, my, MATTRESS_TOP))
    body = stage.GetPrimAtPath("/World/A3V/Manikin/body")
    if body.IsValid():
        visual.chest = _Deform(body, None)
        bump = chest_weights(visual.chest.base, manikin["top_z_m"])
        visual.chest.shape = lambda base, rise, w=bump: inflated(base, w, rise)
    mouth = np.asarray(info["mouth"])
    connector = mouth + [0.0, -0.012, 0.050]  # the 15 mm connector, above the lips
    tube = UsdGeom.Xform.Define(stage, "/World/A3V/Tube")
    # Model: connector at the origin, the tube leaves it in its xz plane. Here: down into the
    # mouth and toward the chest (model x -> +y, model z -> -z).
    tube.AddTransformOp().Set(
        Gf.Matrix4d(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, *[float(v) for v in connector], 1)
    )
    tube.GetPrim().GetReferences().AddReference(str(MODELS / "bvm_tube/bvm_tube.usd"))
    port = np.asarray(info["patient_port"])
    control = [
        port,
        port + [0.0, 0.0, 0.035],
        port + [0.10, -0.01, 0.085],
        [(port[0] + connector[0]) / 2 + 0.03, (port[1] + connector[1]) / 2, connector[2] + 0.075],
        connector + [-0.05, 0.0, 0.055],
        connector + [0.0, 0.0, 0.012],
    ]
    points, uv, indices, normals = tube_arrays(control, 0.0105, sides=10, ribs=0.0018)
    hose = _material(look, "a3v_hose", (0.80, 0.88, 0.93), roughness=0.3)
    mesh = author_mesh(stage, "/World/A3V/Hose", (points, uv, indices), (0, 0, 0), normals)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(hose.material)
    collar = look.material("a3v_collar", color=(0.30, 0.45, 0.75), roughness=0.4)
    look.cylinder("/World/A3V/HoseCollarBag", port + [0.0, 0.0, 0.012], 0.0135, 0.024, collar)
    look.cylinder("/World/A3V/HoseCollarTube", connector + [0.0, 0.0, 0.016], 0.0125, 0.020, collar)

    _screen(backend, visual, info, labels)
    _robot_base(backend)
    _face_dial(backend, dial_target(backend, info["camera"]["position"]))
    dial = getattr(backend, "timer_bars", {}).get("indicator")
    if dial is not None:  # frame 0 already shows the waiting state the task sends on its first tick
        dial.set(1.0, "amber")
    backend._a3v_visual = visual
    _update_visuals(backend)


def _screen(backend, visual, info, labels):
    """The monitor on its roll stand and the live VENT TIMER on its screen.

    Local screen frame: +x out of the screen (toward the reader), +y to the reader's right, +z up.
    Parts that switch are world-level prims that drop FAR out of sight (RTX in Isaac Sim 6.0.1
    kept showing small parts switched by visibility); the sweep is a continuously growing bar.
    """
    from pxr import Gf, UsdGeom, UsdShade

    from roboclock.envs.hospital.props import add_model

    look, stage = backend.appearance, backend.stage
    sx, sy = info["monitor_xy"]
    yaw = info["monitor_yaw"]
    add_model(stage, "patient_monitor", "/World/A3V/Monitor", (sx, sy, WARD_Z), yaw=math.radians(yaw))
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    center = np.array([sx, sy, WARD_Z]) + rotation @ (np.asarray(info["screen_center"]) + [0.003, 0.0, 0.0])

    def world(out, right, up):
        return center + rotation @ np.array([out, right, up])

    group = look.group("/World/A3V/Screen", center, rz=yaw)
    glass = _material(look, "a3v_screen", (0.02, 0.03, 0.04), roughness=0.25)
    look.rounded(group + "/Glass", (0.0, 0.0, 0.0), (0.002, SCREEN[0], SCREEN[1]), glass, 0.0008)
    fill = _material(look, "a3v_sweep", (0.25, 0.75, 1.0), roughness=0.3, emissive=6.0)
    bar = UsdGeom.Cube.Define(stage, "/World/A3V/Readout/Sweep")
    bar.CreateSizeAttr(1.0)
    visual.sweep_t = bar.AddTranslateOp()
    bar.AddRotateZOp().Set(float(yaw))
    visual.sweep_s = bar.AddScaleOp()
    UsdShade.MaterialBindingAPI.Apply(bar.GetPrim()).Bind(fill.material)
    visual.sweep = lambda length: (  # noqa: E731
        Gf.Vec3d(*world(0.0022, GAUGE["y0"] + length / 2, GAUGE["z"]).tolist()),
        Gf.Vec3f(0.0012, max(length, 1e-4), GAUGE["height"] * 0.72),
    )
    visual.sweep_length = GAUGE["y1"] - GAUGE["y0"]
    if not labels:
        return
    turn = [("z", yaw - 90.0)]  # a card is read from its local +y: turn that to the screen's +x
    (gy, gz), size = GAUGE_CARD["center"], GAUGE_CARD["size"]
    _quad(
        stage,
        "/World/A3V/Readout/Gauge",
        _gauge_texture(tuple(info["band"]), info["period"]),
        world(0.0014, gy, gz),
        size,
        rotations=turn,
        emissive=True,
    )
    title = _text_texture("VENT TIMER", (0.80, 0.90, 0.95), None, (512, 128))
    _quad(
        stage,
        "/World/A3V/Readout/Title",
        title,
        world(0.0016, -0.056, 0.059),
        (0.096, 0.024),
        rotations=turn,
        emissive=True,
        cutout=True,
    )
    # Breath counter under the title: one card per count, the others dropped out of sight.
    for count in range(info["breaths"] + 1):
        shown = world(0.0016, -0.056, 0.035)
        card = _quad(
            stage,
            f"/World/A3V/Readout/Count_{count}",
            _text_texture(f"BREATHS {count}/{info['breaths']}", (0.95, 0.95, 0.95), (0.10, 0.14, 0.18), (512, 128)),
            shown,
            (0.096, 0.024),
            rotations=turn,
            emissive=True,
        )
        visual.count[count] = (UsdGeom.Xformable(card).GetOrderedXformOps()[0], Gf.Vec3d(*shown.tolist()))
    for state, text, fg, bg in (
        ("wait", "WAIT", (0.05, 0.05, 0.05), (1.0, 0.70, 0.10)),
        ("due", "SQUEEZE", (0.03, 0.08, 0.03), (0.30, 0.95, 0.45)),
        ("done", "DONE", (1.0, 1.0, 1.0), (0.20, 0.35, 0.60)),
    ):
        shown = world(0.0016, 0.054, 0.047)
        card = _quad(
            stage,
            f"/World/A3V/Readout/Status_{state}",
            _text_texture(text, fg, bg, (512, 224)),
            shown,
            (0.104, 0.046),
            rotations=turn,
            emissive=True,
        )
        visual.status[state] = (UsdGeom.Xformable(card).GetOrderedXformOps()[0], Gf.Vec3d(*shown.tolist()))


def display_state(task, breaths=None):
    """(seconds into the breath cycle, status, breaths counted) the monitor shows.

    The counter shows the breaths the bag has delivered (measured), capped at the number asked."""
    index, seconds = task.cycle()
    count = len(task.breath_times) if breaths is None else breaths
    count = min(count, len(task.windows))
    if task.settled_time is not None:
        return task.period, "done", count
    window = task.current_window()
    due = window is not None and window not in task.served
    return seconds, "due" if due else "wait", count


def _update_visuals(backend):
    """The bag follows the pad, the chest the lung volume, the timer the task clock."""
    from pxr import Gf

    visual = backend._a3v_visual
    task = backend.task
    if visual.task is not task:
        return
    monitor = _monitor(backend)
    for deform in visual.bag:
        deform.set(min(max(monitor.compression, 0.0), STROKE), 0.0004)
    if visual.chest is not None:
        visual.chest.set(CHEST["rise"] * monitor.lung, 0.0003)
    seconds, status, count = display_state(task, len(monitor.breath_times))
    state = (round(seconds, 2), status, count)
    if state == visual.state:
        return
    visual.state = state
    position, scale = visual.sweep(visual.sweep_length * min(seconds / task.period, 1.0))
    visual.sweep_t.Set(position)
    visual.sweep_s.Set(scale)
    for name, (move, shown) in visual.status.items():
        move.Set(shown if name == status else shown - Gf.Vec3d(0, 0, FAR))
    for number, (move, shown) in visual.count.items():
        move.Set(shown if number == count else shown - Gf.Vec3d(0, 0, FAR))


def _camera(backend):
    """Third-person recording view from the robot's left: bag, manikin, hose and monitor."""
    camera = getattr(backend, "task_camera", None)
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["window_bvm"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    camera.set_focal_length(view["focal_mm"] / 1000)
    for _ in range(3):
        backend.world.render()
