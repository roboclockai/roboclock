"""A2 sequential IV bag switch: two color-coded saline bags, one selector, one timed quarter turn.

What it stands for (2026-11-16): two small 100 mL bags (the header label says so) run wide open by
gravity; once bag 1 is empty its drip chamber runs dry within seconds, which is the 3-8 s window.
A 3-7 mL chamber drains in 0.75-12 s at gravity free-flow rates of 35-240 mL/min. The drawn chamber
(CHAMBER_RADIUS, CHAMBER_FULL) and pouch (BAG_SIZE) are still larger than such a set.

Apparatus: a disconnected infusion-line demonstrator beside the bed's foot-left corner in the
scene-02 left bay (assets/hospital/ward_02_preview_left.usdc, ward turned 180 degrees). An IV stand
(local CC-BY-4.0 "IV Stand" by teeart, assets/models/credits.json; telescoped low) stands beside
the instrument trolley and carries BAG 1 (blue label and tint) and BAG 2 (amber) on its two hooks.
Each bag feeds its own supply (drip) chamber and line into a 3-way selector valve on a tray clamped
to the stand at the robot's hand height: BAG 1 enters the +x port, BAG 2 the +y port, and the
outlet (-y) runs under the tray to a small collection bag low on the stand. The SWITCH WINDOW dial
is clamped to the pole between the two lines. The dual-arm robot is a wheeled base with its back
to the bed; internal arm 0 (the rig's right hand) makes the one action and arm 1 stays parked. Bags, chambers,
lines and labels are clean primitives (visual only, no colliders).

Process (q = % of bag 1 delivered = `task.quantity()`; the band 95-100 is the switch window):
    q < 95         bag 1 drips and its fluid level falls toward the green END mark,
    95 <= q < 100  bag 1 has reached END and its drip chamber runs dry (switch now),
    q >= 100       chamber 1 is empty and air (red) advances down bag 1's line into the outlet.
After the switch the process freezes: bag 1's flow stops (an early switch leaves fluid above END)
and bag 2's chamber drips. Fluid is visual only: every level is a function of the task's quantity.

Mechanics: the selector handle copies the B4 stopcock handle (the laboratory valve's passive
bearing; copied, not imported): a free revolute joint (0-90 degrees) with a viscous bearing and no
motor. The robot pinches the hub and turns the black pointer a quarter turn counterclockwise, from
the BAG 1 port (+x) to the BAG 2 port (+y). `device_actuated` latches when the measured angle enters
the BAG 2 detent after robot contact turned the handle; `device_moved` is any measured turn.
"""

import math
from pathlib import Path

import numpy as np

NAME = "bag_selector"
ROOT = f"/World/Objects/{NAME}"
JOINT_PATH = "/World/Joints/A2BagSelectorBearing"
BASE = "/World/Bedside/A2"
TARGET_ANGLE = math.pi / 2
DETENT_TOLERANCE = math.radians(4.0)
MOTION_THRESHOLD = math.radians(3.0)

# --- placement (robot frame, x away from the robot, before the seed shift) ----------------------
WARD_USD = "assets/hospital/ward_02_preview_left.usdc"
# The robot stands at the foot-left corner of the bed with its back to the bed, facing the open floor
# (ward -x) where the demonstrator's IV stand stands beside the instrument trolley. Yaw 180 maps ward
# (x, y) to robot (ROBOT_WARD[0] - x, ROBOT_WARD[1] - y). The task camera looks back past the stand
# toward the robot, the bed, the ultrasound cart and the wall counter (the ward's walled side).
# On the common rig (hospital/rig.py) the idle arm stands 0.60 m to the robot's left (+y), where
# the two-tier supply trolley beside the bed stood (robot y 0.33-0.97): the robot and the
# demonstrator moved 0.45 m along the bed (ward +y: -0.55 -> -0.10), so the trolley and the bed's
# side stand 0.45 m further to the robot's left, clear of the idle arm and the wheeled base.
ROBOT_WARD = (0.72, -0.10)
WARD_YAW = 180.0
WARD_HIDE = []
PIVOT_XY = (0.44, 0.0)
SHELF_X = (0.37, 0.555)
SHELF_Y = (-0.085, 0.085)
SHELF_THICK = 0.012
POLE_XY = (0.60, 0.34)  # IV stand pole axis, to the robot's left of the tray (clear of the view)
POLE_RADIUS = 0.019
POLE_AXIS_MODEL = (0.014, 0.0265)  # the pole axis in the iv_stand model frame (measured)
HOOK_REACH = (-0.195, 0.197)  # hook loop centers along the model's x from the pole axis (measured)
HOOK_Z_MODEL = 1.722  # top of the hook loops' bottom wire in the model (measured)
STAND_SCALE_Z = 0.80  # the stand's telescoping pole is set low (hooks at 1.36 m)
# The SWITCH WINDOW dial is clamped to the IV pole, facing the camera, below the supply chambers.
INDICATOR_XY = (0.657, 0.34)
INDICATOR_BASE = 0.665  # housing bottom (m); the dial face spans about 0.77-0.92 m
CAMERA = {"position": [0.93, -0.25, 1.31], "look_at": [0.46, 0.20, 1.03], "focal_mm": 21.0}

# --- oracle --------------------------------------------------------------------------------------
GRIP_HEIGHT = 0.014  # X5 fingertips project 12.57 mm below grasp_center; keep them above the wings
HOVER_HEIGHT = 0.09  # after the detent the hand lets go and lifts clear, showing the pointer
READY_WIDTH = 0.0145  # finger joint position; the 24 mm hub surface is at about 0.0122
SQUEEZE_WIDTH = 0.008
TURN_RATE = 2.5  # commanded yaw rate (rad/s); the wrist itself tracks at about 1.1 rad/s
TURN_LEAD = 0.16  # the commanded yaw never leads the measured handle by more than this
# Measured from the pinch command to the detent entry (oracle runs, 3 levels x 5 seeds).
ACTUATION_DURATION = 1.35

# --- process visuals -----------------------------------------------------------------------------
ENDPOINT = 95.0  # % of bag 1 delivered at the END mark (window opens)
EMPTY = 100.0  # bag 1's supply chamber is empty (window closes)
BAG_SIZE = (0.14, 0.036, 0.25)  # pouch width, depth, height
BAG_PRINT = "0.9% NaCl 100 mL"  # small print under BAG 1 / BAG 2 on each header label
BAG_HEADER = 0.055  # printed label header above the fluid area
END_HEIGHT = 0.022  # END mark above the pouch's bottom seam (residual below it)
LEVEL_PER_PERCENT = 0.0125  # bag 1 fluid height above END per % still to deliver
CHAMBER_RADIUS, CHAMBER_HEIGHT, CHAMBER_FULL = 0.019, 0.095, 0.070
AIR_SPEED = 0.16  # m/s: the air front's travel down the line once chamber 1 is empty
BAG2_DRAIN = 0.004  # m/s: bag 2's level fall once it feeds the line
DRIP_PERIOD = 0.55
LINE_RADIUS = 0.0032
AIR_RADIUS = 0.0048  # the red air segment is drawn over the line, thicker than it


def ward_translate(shift):
    """Translation that puts ROBOT_WARD at the robot origin after the ward's 180 degree yaw."""
    return [ROBOT_WARD[0] + float(shift[0]), ROBOT_WARD[1] + float(shift[1]), 0.0]


def stand_hooks(pole_xy, hook_z):
    """World positions of the two hook ends (BAG 1, BAG 2) of the IV stand turned 90 degrees."""
    x, y = pole_xy
    return [np.array([x, y + reach, hook_z]) for reach in HOOK_REACH]


def build_layout(b):
    z = b.z
    shelf_top = z + SHELF_THICK
    sx, sy = (SHELF_X[0] + SHELF_X[1]) / 2, (SHELF_Y[0] + SHELF_Y[1]) / 2
    size = (SHELF_X[1] - SHELF_X[0], SHELF_Y[1] - SHELF_Y[0], SHELF_THICK)
    b.fixture("a2_selector_tray", (sx, sy), size, "stainless", collision=False)
    b.fixture("a2_selector_mount", PIVOT_XY, (0.042, 0.042, 0.060), "cabinet_white", z=shelf_top)
    b.item(NAME, "stopcock_handle", PIVOT_XY, z + 0.074, color=(0.18, 0.20, 0.23), yaw=0.0)
    b.indicator(
        "indicator", INDICATOR_XY, "SWITCH WINDOW", height=INDICATOR_BASE + 0.045 - z
    )
    pivot = b.objects[NAME].position
    center = b.xy(sx, sy)
    hook_z = HOOK_Z_MODEL * STAND_SCALE_Z
    pole = b.xy(*POLE_XY)
    camera = {
        "position": [float(v) for v in (*b.xy(*CAMERA["position"][:2]), CAMERA["position"][2])],
        "look_at": [float(v) for v in (*b.xy(*CAMERA["look_at"][:2]), CAMERA["look_at"][2])],
        "focal_mm": CAMERA["focal_mm"],
    }
    b.info.update(
        bedside_task="window_bags",
        a2_bags={
            "pivot": pivot.tolist(),
            "target_angle_rad": TARGET_ANGLE,
            "shelf": {
                "x": [float(b.xy(x, 0)[0]) for x in SHELF_X],
                "y": [float(b.xy(0, y)[1]) for y in SHELF_Y],
                "top": float(shelf_top),
            },
            "pole": [float(v) for v in pole],
            "hooks": [h.tolist() for h in stand_hooks(pole, hook_z)],
            "stand_scale_z": STAND_SCALE_Z,
            "indicator": [float(v) for v in b.xy(*INDICATOR_XY)],
            "table_height": float(z),
            "camera": camera,
            "fluid_simulated": False,
        },
        ward_pose={"usd": WARD_USD, "translate": ward_translate(b.shift), "yaw": WARD_YAW,
                   "hide": list(WARD_HIDE)},
        hide_bench=True,
        work_surface={"center": [*map(float, center), z + SHELF_THICK / 2], "size": list(size)},
        initial_ee_position=(pivot + [0.0, 0.0, GRIP_HEIGHT]).tolist(),
        initial_ee_yaw=0.0,
        initial_gripper_width=READY_WIDTH,
    )
    return b


# --- process state (pure; unit tested without the simulator) -------------------------------------


def fluid_state(quantity, *, rate, switched, since_close=0.0, since_switch=0.0, time=0.0):
    """Visual state of both bags for a quantity (percent of bag 1 delivered).

    rate: % per second of the running process; switched: the selector has reached BAG 2;
    since_close / since_switch: seconds past the chamber-empty instant / past the switch.
    Returns bag fluid heights above the END mark (m), chamber fills (0..1), the air front's distance
    down bag 1's line (m) and which chambers drip.
    """
    q = float(quantity)
    full = BAG_SIZE[2] - BAG_HEADER - END_HEIGHT - 0.012  # fluid height above END when full
    bag1 = min(full, max(0.0, (ENDPOINT - q) * LEVEL_PER_PERCENT))
    if q <= ENDPOINT:
        chamber1 = 1.0
    elif q >= EMPTY:
        chamber1 = 0.0
    else:
        chamber1 = (EMPTY - q) / (EMPTY - ENDPOINT)
    air = 0.0
    if q > EMPTY:
        air = (q - EMPTY) / max(rate, 1e-9) * AIR_SPEED
    bag2 = full - (BAG2_DRAIN * max(since_switch, 0.0) if switched else 0.0)
    phase = (float(time) % DRIP_PERIOD) / DRIP_PERIOD
    return {
        "bag1_level": bag1,
        "chamber1": chamber1,
        "air1": air,
        "drip1": (not switched) and q < ENDPOINT,
        "bag2_level": max(bag2, 0.0),
        "chamber2": 1.0,
        "drip2": bool(switched),
        "drip_phase": phase,
    }


def selector_state(previous, angle, robot_contact, grasped, axis_error=0.0, pivot_error=0.0):
    """Latched selector measurements: moved, contact-driven turn, BAG 2 detent reached.

    previous: the dict this function returned last step (or None at reset).
    """
    state = dict(previous or {"moved": False, "contact_turn": False, "actuated": False})
    finite = math.isfinite(angle)
    turned = finite and abs(angle) > MOTION_THRESHOLD
    state["moved"] = state["moved"] or turned
    state["contact_turn"] = state["contact_turn"] or (turned and (robot_contact or grasped))
    aligned = (
        finite
        and angle >= TARGET_ANGLE - DETENT_TOLERANCE
        and axis_error <= 0.05
        and pivot_error <= 0.004
    )
    state["actuated"] = state["actuated"] or (aligned and state["contact_turn"])
    return state


def switch_start_time(task, duration=ACTUATION_DURATION):
    """When the oracle starts its pinch so the detent lands at the middle of the window."""
    middle = 0.5 * (task.window_open + task.window_close)
    return max(0.0, middle - duration)


# --- authoring -----------------------------------------------------------------------------------


def _collision(backend, prim):
    from pxr import UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    if hasattr(backend, "_collision_settings"):
        backend._collision_settings(prim)
    material = backend.stage.GetPrimAtPath(f"/World/Materials/{NAME}")
    if material.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(material), materialPurpose="physics"
        )


def _quat_to(u):
    from pxr import Gf

    axis = np.cross([0.0, 0.0, 1.0], u)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0) if u[2] > 0 else Gf.Quatf(0.0, 1.0, 0.0, 0.0)
    angle = math.atan2(s, float(u[2]))
    axis = axis / s * math.sin(angle / 2)
    return Gf.Quatf(math.cos(angle / 2), *[float(v) for v in axis])


def _tube(look, path, points, radius, material, spheres=True):
    """A visual tube through `points`: cylinders between them (and spheres at the bends)."""
    from pxr import Gf, UsdGeom, UsdShade

    stage = look.stage
    UsdGeom.Xform.Define(stage, path)
    visual = look.material(material) if isinstance(material, str) else material
    points = [np.asarray(p, dtype=float) for p in points]
    prims = []
    for i, (a, c) in enumerate(zip(points[:-1], points[1:])):
        d = c - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        seg = UsdGeom.Cylinder.Define(stage, f"{path}/Seg{i}")
        seg.CreateRadiusAttr(float(radius))
        seg.CreateHeightAttr(length)
        seg.CreateAxisAttr("Z")
        seg.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in (a + c) / 2]))
        seg.AddOrientOp().Set(_quat_to(d / length))
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(visual.material)
        prims.append(seg.GetPrim())
    if spheres:
        for i, p in enumerate(points[1:-1]):
            prims.append(look.sphere(f"{path}/Bend{i}", p, radius, visual).GetPrim())
    return prims


def _spline(points, n=6):
    """Catmull-Rom curve through `points` (n samples per span), for draped tubing."""
    pts = [np.asarray(p, dtype=float) for p in points]
    pts = [2 * pts[0] - pts[1], *pts, 2 * pts[-1] - pts[-2]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1 : i + 3]
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            out.append(
                0.5
                * (
                    2 * p1
                    + (p2 - p0) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                    + (3 * p1 - p0 - 3 * p2 + p3) * t**3
                )
            )
    out.append(pts[-2])
    return out


def _resample(points, step):
    """Points evenly spaced by `step` along a polyline (keeps both ends)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    lengths = [float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])]
    total = sum(lengths)
    count = max(2, int(math.ceil(total / step)) + 1)
    out, cumulative = [], np.concatenate([[0.0], np.cumsum(lengths)])
    for s in np.linspace(0.0, total, count):
        k = int(np.clip(np.searchsorted(cumulative, s, side="right") - 1, 0, len(lengths) - 1))
        t = 0.0 if lengths[k] < 1e-9 else (s - cumulative[k]) / lengths[k]
        out.append(pts[k] + (pts[k + 1] - pts[k]) * t)
    return out, total


def author(backend):
    """Selector handle (passive revolute joint) plus the visual bags, chambers and lines."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage, look = backend.stage, backend.appearance
    root = stage.GetPrimAtPath(ROOT)
    for child in list(root.GetChildren()):
        stage.RemovePrim(child.GetPath())
    body = look.material("A2SelectorBody", color=[0.92, 0.93, 0.95], roughness=0.30)
    handle = look.material("A2SelectorHandle", color=[0.86, 0.88, 0.90], roughness=0.35)
    pointer = look.material("A2SelectorPointer", color=[0.04, 0.04, 0.05], roughness=0.4)
    clear = look.material("A2SelectorClear", color=[0.80, 0.88, 0.92], roughness=0.18)
    # The narrow raised hub gives the parallel fingertips a friction grip; the wings stay below it
    # so the open hand can descend without striking them (the B4 handle geometry).
    hub = look.cylinder(ROOT + "/Hub", [0, 0, 0.003], 0.012, 0.020, handle)
    _collision(backend, hub.GetPrim())
    for name, size in (("WingX", [0.052, 0.008, 0.010]), ("WingY", [0.008, 0.052, 0.010])):
        look.rounded(ROOT + "/" + name, [0, 0, -0.007], size, handle, 0.002)
        col = UsdGeom.Cube.Define(stage, ROOT + "/" + name + "Collider")
        col.CreateSizeAttr(1.0)
        col.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.007))
        col.AddScaleOp().Set(Gf.Vec3f(*size))
        _collision(backend, col.GetPrim())
        UsdGeom.Imageable(col).MakeInvisible()
    # The pointer: a white arrow along the +x wing (the BAG 1 port at the start, the BAG 2 port
    # after the quarter turn), repeated on the hub top so it reads from any camera elevation.
    # A bold black arrow (longer than the wing; visual only, below the fingertips).
    look.rounded(ROOT + "/Pointer", [0.026, 0, -0.0008], [0.034, 0.009, 0.0030], pointer, 0.0008)
    look.rounded(ROOT + "/PointerTip", [0.044, 0, -0.0008], [0.007, 0.020, 0.0030], pointer, 0.0008)
    look.rounded(ROOT + "/HubArrow", [0.0055, 0, 0.0134], [0.013, 0.0045, 0.0012], pointer, 0.0005)
    info = backend.layout.info["a2_bags"]
    pivot = np.asarray(info["pivot"])
    x, y, z = map(float, pivot)
    housing = look.cylinder(BASE + "/Housing", [x, y, z - 0.024], 0.017, 0.016, body)
    _collision(backend, housing.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, JOINT_PATH)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ROOT)])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(90.0)
    # The laboratory's passive viscous bearing: no position target, spring, motor or task force.
    bearing = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    bearing.CreateTypeAttr("force")
    bearing.CreateStiffnessAttr(0.0)
    bearing.CreateDampingAttr(0.008)
    bearing.CreateTargetVelocityAttr(0.0)
    bearing.CreateMaxForceAttr(0.015)
    backend._a2w_visuals = _dress(backend, info, pivot, clear)
    task = getattr(backend, "task", None)
    if task is not None and getattr(task, "layout", None) is not None:
        apply_visuals(backend, task)  # pose the fluid for the reset frame, not only from step 1
    return joint


# Colors: BAG 1 blue, BAG 2 amber, air red.
BLUE = (0.10, 0.42, 0.92)
AMBER = (0.96, 0.58, 0.08)
FLUID1 = (0.30, 0.62, 1.00)
FLUID2 = (1.00, 0.68, 0.20)


def _film(look, name, tint, opacity=0.38):
    """Clear PVC film: an OmniPBR surface with fractional opacity (the renderer enables
    fractional cutout opacity), so the fluid inside stays visible."""
    from pxr import Sdf

    key = f"A2Film_{name}"
    first = key not in look.materials
    material = look.material(key, color=list(tint), roughness=0.12)
    if first:
        shader = material.shaders_list[0]
        shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))
    return material


class _Box:
    """A unit cube posed by a translate and a scale op (a fluid column whose height changes)."""

    def __init__(self, stage, path, material, rz=0.0):
        from pxr import UsdGeom, UsdShade

        self.cube = UsdGeom.Cube.Define(stage, path)
        self.cube.CreateSizeAttr(1.0)
        self.t = self.cube.AddTranslateOp()
        self.r = self.cube.AddRotateZOp()
        self.r.Set(float(rz))
        self.s = self.cube.AddScaleOp()
        UsdShade.MaterialBindingAPI.Apply(self.cube.GetPrim()).Bind(material.material)
        # Hidden until the first set(): an unscaled unit cube (1 m) filled the head camera's
        # reset frame with flat blue/amber until 2026-11-19.
        UsdGeom.Imageable(self.cube).MakeInvisible()

    def set(self, center, size):
        from pxr import Gf, UsdGeom

        if min(size) <= 1e-5:
            UsdGeom.Imageable(self.cube).MakeInvisible()
            return
        UsdGeom.Imageable(self.cube).MakeVisible()
        self.t.Set(Gf.Vec3d(*[float(v) for v in center]))
        self.s.Set(Gf.Vec3f(*[float(v) for v in size]))


class _Column:
    """A vertical cylinder whose height changes (chamber fluid)."""

    def __init__(self, stage, path, radius, material):
        from pxr import UsdGeom, UsdShade

        self.cyl = UsdGeom.Cylinder.Define(stage, path)
        self.cyl.CreateRadiusAttr(float(radius))
        self.cyl.CreateAxisAttr("Z")
        self.h = self.cyl.CreateHeightAttr(0.01)
        self.t = self.cyl.AddTranslateOp()
        UsdShade.MaterialBindingAPI.Apply(self.cyl.GetPrim()).Bind(material.material)

    def set(self, bottom, height):
        from pxr import Gf, UsdGeom

        if height < 1e-4:
            UsdGeom.Imageable(self.cyl).MakeInvisible()
            return
        UsdGeom.Imageable(self.cyl).MakeVisible()
        self.h.Set(float(height))
        self.t.Set(Gf.Vec3d(float(bottom[0]), float(bottom[1]), float(bottom[2] + height / 2)))


def _camera_yaw(point, camera):
    """Yaw (degrees) that turns a card's +y normal toward the camera."""
    return math.degrees(math.atan2(-(camera[0] - point[0]), camera[1] - point[1]))


def _dress(backend, info, pivot, clear):
    """Visual-only apparatus: IV stand, bags, chambers, lines, tray, labels and the mobile base."""
    from pxr import UsdGeom

    look = backend.appearance
    config = look.config
    shelf = info["shelf"]
    cam = np.asarray(info["camera"]["position"], dtype=float)
    holder = look.material("A2HolderGrey", color=[0.80, 0.83, 0.86], roughness=0.45)
    fluid = [
        look.material("A2Fluid1", color=FLUID1, roughness=0.15),
        look.material("A2Fluid2", color=FLUID2, roughness=0.15),
    ]
    tint = [
        look.material("A2Tube1", color=(0.45, 0.70, 1.0), roughness=0.2),
        look.material("A2Tube2", color=(1.0, 0.76, 0.38), roughness=0.2),
    ]
    air = look.material("A2Air", color=(1.0, 0.04, 0.02), roughness=0.3, emissive=True)
    _iv_stand(look, info)
    pole = np.asarray(info["pole"], dtype=float)
    # Pole clamp and arm carrying the selector tray.
    tray_z = shelf["top"] - SHELF_THICK / 2
    look.cylinder(BASE + "/PoleClamp", [pole[0], pole[1], tray_z], POLE_RADIUS + 0.008, 0.05, holder)
    anchor = np.array([shelf["x"][1] - 0.03, shelf["y"][1] - 0.025])
    arm = pole - anchor
    length = float(np.linalg.norm(arm))
    group = look.group(BASE + "/ClampArm", [*(anchor + arm / 2), tray_z],
                       rz=math.degrees(math.atan2(arm[1], arm[0])))
    look.rounded(group + "/Bar", [0, 0, 0], [length, 0.026, 0.014], holder, 0.003)
    look.rounded(
        BASE + "/ClampScrew", [pole[0], pole[1] + POLE_RADIUS + 0.014, tray_z],
        [0.010, 0.022, 0.010], "rubber", 0.003,
    )
    # Ports on the valve body: BAG 1 (+x, toward the camera), BAG 2 (+y, toward the bags), outlet
    # (-y). The pointer turns a quarter turn counterclockwise from BAG 1 to BAG 2.
    ports = {"bag1": np.array([1.0, 0.0, 0.0]), "bag2": np.array([0.0, 1.0, 0.0]),
             "out": np.array([0.0, -1.0, 0.0])}
    ends = {}
    for name, d in ports.items():
        axis = "X" if abs(d[0]) > 0 else "Y"
        center = pivot + 0.027 * d + [0, 0, -0.024]
        look.cylinder(f"{BASE}/{name}Luer", center, 0.007, 0.025, clear, axis)
        for k in range(3):
            ring = center + (k - 1) * 0.003 * d
            look.cylinder(f"{BASE}/{name}Thread{k}", ring, 0.008, 0.0014, "paper", axis)
        ends[name] = pivot + 0.040 * d + [0, 0, -0.024]
    # Detent collars on the port stubs: blue BAG 1 (start), amber BAG 2 (target).
    look.cylinder(BASE + "/Bag1Mark", ends["bag1"] + [0.004, 0, 0], 0.0052, 0.010,
                  look.material("A2Blue", color=BLUE, roughness=0.4), "X")
    look.cylinder(BASE + "/Bag2Mark", ends["bag2"] + [0, 0.004, 0], 0.0052, 0.010,
                  look.material("A2Amber", color=AMBER, roughness=0.4), "Y")
    hooks = [np.asarray(h, dtype=float) for h in info["hooks"]]
    visuals = {"bags": [], "chambers": [], "drops": [], "air": [], "line_lengths": []}
    for i, hook in enumerate(hooks):
        bag = _bag(look, i, hook, cam, fluid[i], visuals)
        chamber_top = bag["outlet"] - [0, 0, 0.030]
        _chamber(look, i, chamber_top, fluid[i], clear, visuals)
        # Line from the chamber's outlet to the selector port.
        start = chamber_top - [0, 0, CHAMBER_HEIGHT + 0.012]
        port = ends["bag1" if i == 0 else "bag2"]
        d = ports["bag1" if i == 0 else "bag2"]
        drop = min(0.20, (start[2] - port[2]) * 0.6)
        control = [
            start,
            start - [0, 0, drop * 0.5],
            [0.5 * (start[0] + port[0]) + 0.03 * d[0], 0.5 * (start[1] + port[1]) + 0.03 * d[1],
             port[2] + 0.10],
            port + 0.035 * d + [0, 0, 0.018],
            port + 0.012 * d,
            port,
        ]
        line, length = _resample(_spline(control, 6), 0.012)
        _tube(look, f"{BASE}/Line{i + 1}", line, LINE_RADIUS, tint[i])
        visuals["line_lengths"].append(length)
        if i == 0:
            visuals["line1"] = line
    # Outlet: over the tray's edge, under the tray to the pole and down to the demonstrator's
    # collection bag hanging low on the stand (the line is not connected to a patient).
    out = ends["out"]
    sy0 = shelf["y"][0]
    under = shelf["top"] - SHELF_THICK - 0.05
    bag_top = 0.42
    outlet = _spline(
        [
            out,
            [out[0], sy0 - 0.008, out[2] - 0.006],
            [out[0] - 0.006, sy0 - 0.018, under + 0.02],
            [out[0] + 0.02, sy0 + 0.04, under - 0.03],
            [pole[0] - 0.03, pole[1] - 0.10, under - 0.08],
            [pole[0] + 0.035, pole[1] - 0.02, bag_top + 0.10],
            [pole[0] + 0.045, pole[1], bag_top + 0.02],
            [pole[0] + 0.045, pole[1], bag_top],
        ],
        6,
    )
    outlet, _ = _resample(outlet, 0.012)
    _tube(look, BASE + "/Outlet", outlet, LINE_RADIUS, tint[0])
    _collection_bag(look, np.array([pole[0] + 0.045, pole[1], bag_top]), cam, holder)
    # Its hanger: a collar on the pole with a short hook arm over the bag's top port.
    look.cylinder(BASE + "/CollectClamp", [pole[0], pole[1], bag_top + 0.012], POLE_RADIUS + 0.006,
                  0.022, holder)
    look.rounded(BASE + "/CollectHook", [pole[0] + 0.030, pole[1], bag_top + 0.012],
                 [0.036, 0.010, 0.008], holder, 0.002)
    # The SWITCH WINDOW dial's clamp on the pole.
    ind = np.asarray(info["indicator"], dtype=float)
    look.cylinder(BASE + "/DialClamp", [pole[0], pole[1], INDICATOR_BASE + 0.025], POLE_RADIUS + 0.007,
                  0.04, holder)
    look.rounded(BASE + "/DialBracket", [(pole[0] + ind[0]) / 2, ind[1], INDICATOR_BASE + 0.025],
                 [abs(ind[0] - pole[0]), 0.03, 0.03], holder, 0.003)
    # Air overlay along bag 1's line, then (past the valve body) along the outlet: hidden until
    # the air front passes. Each entry: (prims, distance of the segment's end from chamber 1).
    offset = 0.0
    for name, path in (("Line1", visuals["line1"]), ("Outlet", list(outlet))):
        for k, (a, c) in enumerate(zip(path[:-1], path[1:])):
            prims = _tube(look, f"{BASE}/Air{name}/Seg{k}", [a, c], AIR_RADIUS, air, spheres=False)
            for prim in prims:
                UsdGeom.Imageable(prim).MakeInvisible()
            offset += float(np.linalg.norm(c - a))
            visuals["air"].append((prims, offset))
        offset += 0.03  # through the valve body
    if config["visuals"].get("labels", True):
        _labels(look, info, pivot, cam)
    _mobile_base(look, config, info)
    visuals["state"] = None
    return visuals


def _bag(look, i, hook, cam, fluid, visuals):
    """A hanging saline bag: frosted film pouch, fluid column, printed header, END mark."""
    w, d, h = BAG_SIZE
    top = hook[2] - 0.020  # the eyelet's top bar rests on the hook loop's bottom wire
    center = np.array([hook[0], hook[1], top - h / 2])
    yaw = _camera_yaw(center, cam)
    group = look.group(f"{BASE}/Bag{i + 1}", center.tolist(), rz=yaw)
    film = _film(look, f"bag{i + 1}", (0.97, 0.985, 1.0))
    look.rounded(group + "/Pouch", [0, 0, 0], [w, d, h], film, 0.012)
    band = BLUE if i == 0 else AMBER
    seam = look.material("A2Seam", color=(0.93, 0.95, 0.97), roughness=0.4)
    look.rounded(group + "/SeamTop", [0, 0, h / 2 - 0.006], [w + 0.002, d * 0.4, 0.012], seam, 0.002)
    look.rounded(group + "/SeamBottom", [0, 0, -h / 2 + 0.005], [w + 0.002, d * 0.4, 0.010], seam, 0.002)
    for k, side in enumerate((-1, 1)):
        look.rounded(group + f"/SeamSide{k}", [side * (w / 2 - 0.002), 0, 0], [0.006, d * 0.5, h - 0.004],
                     seam, 0.002)
    look.ring(group + "/Eyelet", [0, 0, h / 2 + 0.014], 0.026, 0.024, 0.006, seam, depth=0.004)
    look.cylinder(group + "/Port", [0, 0, -h / 2 - 0.012], 0.0055, 0.024, seam)
    # Printed header facing the camera (+y of the group): the color-coded BAG label.
    header_z = h / 2 - 0.012 - BAG_HEADER / 2
    if look.config["visuals"].get("labels", True):
        text = "BAG 1" if i == 0 else "BAG 2"
        _card(look.stage, group + "/Header", _label_texture(text, band, sub=BAG_PRINT),
              (w - 0.012, BAG_HEADER - 0.006), [0, d / 2 + 0.0015, header_z])
    # Fluid: a column inside the pouch from the bottom seam up to its level.
    bottom = -h / 2 + 0.010
    box = _Box(look.stage, group + "/Fluid", fluid)
    end_z = bottom + END_HEIGHT
    # END mark across the front (bag 1: the endpoint; bag 2 the same print).
    mark = look.material("A2EndMark", color=(0.05, 0.62, 0.22), roughness=0.4, emissive=True)
    look.rounded(group + "/EndMark", [0, d / 2 + 0.0012, end_z], [w - 0.01, 0.0015, 0.004], mark, 0.0006)
    if look.config["visuals"].get("labels", True):
        _card(look.stage, group + "/EndTag", _tag_texture("END", (0.05, 0.62, 0.22)),
              (0.034, 0.012), [w / 2 - 0.021, d / 2 + 0.0022, end_z + 0.009])
    visuals["bags"].append({"box": box, "bottom": bottom, "end": end_z, "inner": (w - 0.014, d - 0.012)})
    outlet = center + [0, 0, -h / 2 - 0.024]
    return {"outlet": outlet}


def _collection_bag(look, port, cam, holder):
    """The demonstrator's small collection bag on a hook low on the IV pole (visual)."""
    w, d, h = 0.10, 0.025, 0.14
    center = port - [0, 0, 0.006 + h / 2]
    group = look.group(BASE + "/CollectionBag", center.tolist(), rz=_camera_yaw(center, cam))
    look.rounded(group + "/Pouch", [0, 0, 0], [w, d, h], _film(look, "collect", (0.97, 0.985, 1.0)), 0.01)
    look.rounded(group + "/Fluid", [0, 0, -h / 2 + 0.03], [w - 0.016, d - 0.01, 0.045],
                 look.material("A2Fluid1"), 0.004)
    look.rounded(group + "/Seam", [0, 0, h / 2 - 0.005], [w + 0.002, d * 0.4, 0.010], "paper", 0.002)
    if look.config["visuals"].get("labels", True):
        _card(look.stage, group + "/Tag", _tag_texture("DEMO LINE", (0.45, 0.48, 0.52)),
              (w - 0.02, 0.022), [0, d / 2 + 0.0015, h / 2 - 0.03])


def _chamber(look, i, top, fluid, clear, visuals):
    """Supply (drip) chamber under a bag: clear barrel, spike cap, fluid column and a drop."""
    glass = _film(look, f"chamber{i + 1}", (0.93, 0.96, 1.0), opacity=0.32)
    r, h = CHAMBER_RADIUS, CHAMBER_HEIGHT
    base = f"{BASE}/Chamber{i + 1}"
    look.cylinder(base + "/Spike", top + [0, 0, 0.012], 0.0045, 0.024, "paper")
    look.cylinder(base + "/Cap", top - [0, 0, 0.005], r + 0.0015, 0.010, "paper")
    look.cylinder(base + "/Barrel", top - [0, 0, 0.010 + h / 2], r, h, glass)
    look.cylinder(base + "/Foot", top - [0, 0, 0.010 + h + 0.003], 0.006, 0.008, "paper")
    bottom = top - [0, 0, 0.010 + h - 0.002]
    column = _Column(look.stage, base + "/Fluid", r - 0.0025, fluid)
    drop = look.sphere(base + "/Drop", top - [0, 0, 0.016], 0.0032, fluid)
    from pxr import UsdGeom

    UsdGeom.Imageable(drop).MakeInvisible()
    visuals["chambers"].append({"column": column, "bottom": bottom, "top": top - [0, 0, 0.012]})
    visuals["drops"].append(drop)


def apply_visuals(backend, task):
    """Pose the fluid from the task's quantity (called from measure; visual only)."""
    from pxr import Gf, UsdGeom

    vis = getattr(backend, "_a2w_visuals", None)
    if vis is None or task is None:
        return
    switched = task.action_time is not None
    q = task.quantity()
    state = fluid_state(
        q,
        rate=task.rate,
        switched=switched,
        since_close=task.time - task.window_close,
        since_switch=(task.time - task.action_time) if switched else 0.0,
        time=task.time,
    )
    key = tuple(round(float(v), 4) if not isinstance(v, bool) else v for v in state.values())
    if key == vis["state"]:
        return
    vis["state"] = key
    for i, bag in enumerate(vis["bags"]):
        level = bag["end"] - bag["bottom"] + (state["bag1_level"] if i == 0 else state["bag2_level"])
        iw, idp = bag["inner"]
        bag["box"].set([0.0, 0.0, bag["bottom"] + level / 2], [iw, idp, level])
    for i, chamber in enumerate(vis["chambers"]):
        fill = state["chamber1"] if i == 0 else state["chamber2"]
        chamber["column"].set(chamber["bottom"], CHAMBER_FULL * fill)
        drop = vis["drops"][i]
        drips = state["drip1"] if i == 0 else state["drip2"]
        if drips:
            surface = chamber["bottom"][2] + CHAMBER_FULL * fill
            z = chamber["top"][2] - (chamber["top"][2] - surface) * state["drip_phase"] ** 2
            drop.GetPrim().GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(float(chamber["top"][0]), float(chamber["top"][1]), float(z))
            )
            UsdGeom.Imageable(drop).MakeVisible()
        else:
            UsdGeom.Imageable(drop).MakeInvisible()
    front = state["air1"]
    for prims, end in vis["air"]:
        for prim in prims:
            if end <= front + 1e-6:
                UsdGeom.Imageable(prim).MakeVisible()
            else:
                UsdGeom.Imageable(prim).MakeInvisible()


# --- printed labels ------------------------------------------------------------------------------


def _font(size):
    from PIL import ImageFont

    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
    try:
        import matplotlib

        candidates.append(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
    except ImportError:
        pass
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _cache(key):
    import hashlib
    import os

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"a2w_{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.png"


def _label_texture(text, band, sub=None, size=(640, 200)):
    """A printed bag/port label: colored band at the left, bold dark text as large as fits."""
    from PIL import Image, ImageDraw

    path = _cache((text, band, sub, size, 3))
    if path.exists():
        return path
    w, h = size
    image = Image.new("RGB", size, (250, 250, 247))
    draw = ImageDraw.Draw(image)
    stripe = int(h * 0.36)
    draw.rectangle([0, 0, stripe, h], fill=tuple(round(255 * c) for c in band))
    draw.rectangle([0, 0, w - 1, h - 1], outline=tuple(round(200 * c) for c in band), width=8)
    main_h = h * (0.62 if sub else 0.9)
    font_size = int(main_h * 0.82)
    while font_size > 12 and _font(font_size).getlength(text) > (w - stripe) * 0.9:
        font_size -= 2
    cy = h * (0.38 if sub else 0.5)
    draw.text(((w + stripe) / 2, cy), text, fill=(14, 18, 24), font=_font(font_size), anchor="mm")
    if sub:
        small = int(h * 0.24)
        draw.text(((w + stripe) / 2, h * 0.80), sub, fill=(60, 64, 70), font=_font(small), anchor="mm")
    image.save(path)
    return path


def _tag_texture(text, color, size=(300, 110)):
    from PIL import Image, ImageDraw

    path = _cache((text, color, size, 1))
    if path.exists():
        return path
    image = Image.new("RGB", size, tuple(round(255 * c) for c in color))
    draw = ImageDraw.Draw(image)
    font_size = int(size[1] * 0.8)
    while font_size > 10 and _font(font_size).getlength(text) > size[0] * 0.86:
        font_size -= 2
    draw.text((size[0] / 2, size[1] / 2), text, fill=(255, 255, 255), font=_font(font_size), anchor="mm")
    image.save(path)
    return path


def _card(stage, path, texture, size, offset=(0.0, 0.0, 0.0)):
    """A textured quad in its parent's xz plane that reads correctly from +y."""
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
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, path)
    w, h = size
    ox, oy, oz = offset
    mesh.CreatePointsAttr(
        [Gf.Vec3f(ox + w / 2, oy, oz - h / 2), Gf.Vec3f(ox + w / 2, oy, oz + h / 2),
         Gf.Vec3f(ox - w / 2, oy, oz + h / 2), Gf.Vec3f(ox - w / 2, oy, oz - h / 2)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(0, 0), (0, 1), (1, 1), (1, 0)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


PORT_LABELS = (  # text, band color, offset from the pivot, width
    ("BAG 1", BLUE, (0.098, -0.022), 0.064),
    ("BAG 2", AMBER, (-0.018, 0.074), 0.064),
    ("OUT", (0.55, 0.58, 0.62), (-0.020, -0.068), 0.050),
)


def _labels(look, info, pivot, cam):
    """Port labels standing on the selector tray, turned toward the task camera."""
    x, y, _ = map(float, pivot)
    top = info["shelf"]["top"]
    for k, (text, band, (ox, oy), width) in enumerate(PORT_LABELS):
        lx, ly = x + ox, y + oy
        yaw = _camera_yaw((lx, ly), cam)
        group = look.group(f"{BASE}/Tags/Tag_{k}", [lx, ly, top + 0.0135], rz=yaw)
        _card(look.stage, group + "/Label", _label_texture(text, band), (width, width * 0.3125))
        look.rounded(group + "/Back", [0.0, -0.0022, -0.0005], [width + 0.004, 0.003, width * 0.3125 + 0.004], "steel", 0.001)


def _iv_stand(look, info):
    """The bedside IV stand (local CC-BY-4.0 model), turned so its hooks run along y."""
    from roboclock.envs.hospital.props import add_model, model_size

    px, py = info["pole"]
    s = float(info["stand_scale_z"])
    if model_size("iv_stand") is not None:
        # yaw 90: model (mx, my) -> world (-my, mx); put the model's pole axis on (px, py).
        tx, ty = px + POLE_AXIS_MODEL[1], py - POLE_AXIS_MODEL[0]
        add_model(look.stage, "iv_stand", BASE + "/IVStand", (tx, ty, 0.0), (1.0, 1.0, s),
                  yaw=math.pi / 2)
    else:
        look.cylinder(BASE + "/IVStandPole", [px, py, 0.85 * s], POLE_RADIUS, 1.7 * s, "stainless")


def _mobile_base(look, config, info):
    """The X5 columns stand on one wheeled base on the ward floor (visual; the robot is fixed)."""
    from roboclock.envs.lab.robot import mount_positions

    z = info["table_height"]
    mounts = [np.asarray(m[:2], dtype=float) for m in mount_positions(config, "A2")]
    lo, hi = np.min(mounts, axis=0) - 0.11, np.max(mounts, axis=0) + 0.11
    center, size = (lo + hi) / 2, hi - lo
    shell = look.material("A2BaseShell", color=(0.86, 0.88, 0.90), roughness=0.45)
    dark = look.material("A2BaseDark", color=(0.20, 0.22, 0.24), roughness=0.55)
    look.rounded(BASE + "/RobotBase", [*center, 0.075], [*size, 0.11], shell, 0.02)
    look.rounded(BASE + "/RobotBumper", [*center, 0.030], [*(size + 0.012), 0.03], dark, 0.01)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        wheel = [center[0] + sx * (size[0] / 2 - 0.05), center[1] + sy * (size[1] / 2 - 0.05), 0.03]
        look.cylinder(f"{BASE}/RobotWheel_{i}", wheel, 0.03, 0.025, dark, axis="Y")
    for i, m in enumerate(mounts):
        look.rounded(f"{BASE}/RobotColumn_{i}", [*m, (0.13 + z) / 2], [0.12, 0.12, z - 0.13], dark, 0.006)


# --- runtime -------------------------------------------------------------------------------------


def _camera(backend):
    """The task camera: both bags, the chambers, the selector, the hand and the bed; once per reset."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["a2_bags"]["camera"]
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


def measure(backend, state):
    """Privileged PhysX measurements of the selector; also poses the (visual) fluid."""
    task = getattr(backend, "task", None)
    if getattr(backend, "_a2w_for", None) is not task:
        backend._a2w_for = task
        backend._a2w_selector = None
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_a2w_camera_for", None) is not task:
        backend._a2w_camera_for = task
        _camera(backend)
    obj = state.objects[NAME]
    quaternion = np.asarray(obj.orientation, dtype=float)
    valid = bool(np.all(np.isfinite(quaternion)) and abs(np.linalg.norm(quaternion) - 1.0) < 0.01)
    w, x, y, z = quaternion
    angle = (
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)) if valid else float("nan")
    )
    axis_error = (
        math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))) if valid else float("inf")
    )
    pivot = np.asarray(backend.layout.info["a2_bags"]["pivot"])
    pivot_error = float(np.linalg.norm(obj.position - pivot))
    selector = selector_state(
        backend._a2w_selector, angle, bool(obj.robot_contact), bool(obj.grasped), axis_error,
        pivot_error,
    )
    backend._a2w_selector = selector
    if task is not None and getattr(task, "layout", None) is not None:
        apply_visuals(backend, task)
    return {
        "selector_angle_rad": float(angle),
        "selector_axis_error_rad": axis_error,
        "selector_pivot_error_m": pivot_error,
        "selector_robot_contact": bool(obj.robot_contact),
        "selector_grasped": bool(obj.grasped),
        "selector_contact_turn": selector["contact_turn"],
        "device_moved": selector["moved"],
        "device_actuated": selector["actuated"],
    }


def command(planner, task, state):
    """Prepared straddle; at the planned instant, pinch and make one measured quarter turn.

    Privileged: reads the window and starts the pinch ACTUATION_DURATION before its middle, so the
    detent lands near the middle on every level. The commanded yaw is capped at the detent (it never
    drives past the hard stop); after the detent the fingers open and the hand lifts clear.
    """
    if planner.last_goal is None:
        planner.last_goal = state.ee_position.copy()
        planner.yaw = 0.0
    obj = state.objects[NAME]
    angle = float(state.controls.get("selector_angle_rad", 0.0))
    if not math.isfinite(angle):
        angle = 0.0
    grip = obj.position + np.array([0.0, 0.0, GRIP_HEIGHT])
    planner.open_width, planner.close_width = READY_WIDTH, SQUEEZE_WIDTH
    planner.gentle = False
    if task.goal_achieved:
        planner.phase = "release"
        return planner.finish(state, grip + [0.0, 0.0, HOVER_HEIGHT], False, 0.12)
    if task.time + 1e-9 < switch_start_time(task):
        planner.phase = "wait_window"
        return planner.finish(state, grip, False, 0.12)
    if getattr(planner, "a2w_started", None) is not task:
        planner.a2w_started = task
        task.event_log.append({"time": task.time, "type": "oracle_switch_start"})
    near = np.linalg.norm(state.ee_position - grip) < 0.009
    close = near or planner.phase in ("pinch_selector", "turn_selector")
    if not close:
        planner.phase = "approach_selector"
        planner.yaw = angle
    elif not obj.grasped:
        planner.phase = "pinch_selector"
    else:
        planner.phase = "turn_selector"
        desired = min(TARGET_ANGLE, angle + TURN_LEAD)
        planner.yaw += float(np.clip(desired - planner.yaw, 0.0, TURN_RATE * planner.dt))
    return planner.finish(state, grip, close, 0.20)
