"""Physical B4 line switch: a MAINTENANCE roller clamp and a three-way stopcock on one rail holder.

The order "SWITCH INFUSION: 1 CLOSE MAINTENANCE CLAMP  2 TURN STOPCOCK TO REQUESTED" is shown on a
bedside order screen under the countdown dial from frame 0 and issued right after settling.

Mechanics (composed, not new): the stopcock handle reuses the laboratory valve's passive bearing
(roboclock/envs/lab/liquid_scene.py::LiquidScene.valve) with real hub and wing colliders; the roller
clamp is the B5 passive prismatic roller (hospital/bedside/roller_clamp.py: viscous slide, joint
end stops, conservative box collider, colliding organizer tray), copied here and turned 90 degrees
so it slides along the line. Tubing is visual only; connectivity is symbolic and inferred from the
measured roller travel and handle angle.

Placement (scene-02 left bay, assets/hospital/ward_02_preview_left.usdc): one line holder hooks over
the flat top of the bed's patient-right foot rail and hangs on its mattress side. The running
MAINTENANCE line comes from the upper pump on the IV stand at the foot end, over the rail, through
the roller clamp and into the stopcock's upstream port; the PATIENT port continues along the line
toward the head end and drops onto the mattress toward the patient's arm; the REQUESTED side port
faces the rail and is connected to the primed second line from the lower pump. The roller closes
toward the patient (downstream), the clinical convention. The robot is a wheeled dual-arm base in
the aisle between the bed and the wall counter, starting tucked at its base about 0.46 m (arm-0
mount to devices, horizontally) from the line, so it clearly reaches over the rail. The top-down
X5 hand reaches at most about 0.53 m horizontally from its mount at this height (link2-link4 0.51
m), so the seeded jitter (+-25 mm) keeps every waypoint at or inside 0.50 m. The ward turns 180
degrees about the robot side and shifts with the seeded jitter, so the holder always hangs on the
rail.
"""

import math
from pathlib import Path

import numpy as np

from roboclock.envs.hospital.bedside.ward import ULTRASOUND_CART
from roboclock.envs.hospital.tasks.bedside.stopcock import (
    CLAMP,
    CLAMP_END_TOLERANCE,
    CLAMP_TRAVEL,
    HANDLE,
    TARGET_ANGLE,
)

NAME = HANDLE
ROOT = f"/World/Objects/{NAME}"
JOINT_PATH = "/World/Joints/StopcockBearing"
ROLLER_NAME = CLAMP
ROLLER_ROOT = f"/World/Objects/{ROLLER_NAME}"
ROLLER_ASSET = "iv_roller"
ROLLER_JOINT_PATH = "/World/Joints/MaintenanceRollerSlide"
# X5 fingertips project 12.57 mm below grasp_center; keep them above the wings.
GRIP_HEIGHT = 0.014
HOVER_HEIGHT = 0.015
READY_WIDTH = 0.0145  # finger joint position (half opening); the hub surface is at about 0.0122
SQUEEZE_WIDTH = 0.005  # a firm pinch: the extended arm's wrist turn slipped on the hub at 0.008
GENTLE_CLOSE = False  # the jointed handle cannot be knocked away, so close at the gripper's speed
TURN_RATE = 2.5  # commanded yaw rate (rad/s); the wrist itself tracks at about 1.1 rad/s
TURN_LEAD = 0.16  # the commanded yaw never leads the measured handle by more than this
# The commanded turn may pass the 90-degree detent by this much: the fingertips slip a few degrees
# on the hub, so the hand turns on until the handle itself sits at its hard stop.
TURN_OVERSHOOT = 0.2
# Roller push, B5 geometry turned 90 degrees: with the hand at yaw -90 degrees the closed fingers
# separate along x and their edges push the roller's upstream face along +y. The hub is round, so
# the same yaw grasps it; the quarter turn then runs from -90 to 0 degrees.
HAND_YAW = -math.pi / 2
PUSH_WIDTH = 0.004
PUSH_HEIGHT = 0.009
READY_OFFSET = 0.028
PUSH_OFFSET = 0.020
APPROACH_LIFT = 0.060  # the reach ends this high above the push start, then descends
TRANSFER_LIFT = 0.025  # hand height above the grip point when crossing from roller to hub
REACH_SPEED = 0.85
DESCEND_SPEED = 0.45
PUSH_SPEED = 0.30
TRANSFER_SPEED = 0.55
# Tucked start: in front of the left mount at about mount height, fingers closed, wrist at HAND_YAW.
TUCK_OFFSET = (0.24, 0.0, 0.06)

# Ward geometry (ward file frame, meters, measured from the mesh): the foot section of the bed's
# patient-right side rail is a 30 mm panel at x 1.757-1.787 with a flat top at z 0.878 for
# y -1.45..-1.05; the mattress top is about 0.668 there and the floor is z 0.
WARD_USD = "assets/hospital/ward_02_preview_left.usdc"
RAIL_WARD_X = 1.787  # the rail's outer (robot-side) face
RAIL_THICK = 0.030
RAIL_TOP = 0.878
MATTRESS_TOP = 0.668  # quilted top; lines rest on the crests
MATTRESS_EDGE = 0.046  # mattress side, beyond the rail's outer face
RAIL_FLAT_WARD_Y = (-1.45, -1.05)
# Robot frame (x away from the robot, +y toward the bed head). The rail's outer face is at x 0.335;
# the driving arm-0 mount stands at the origin (robot.rig.tasks.B4), so the line set at
# x 0.46 is 0.46 m out and 0.095 m beyond the rail's inner face.
RAIL_OUT = 0.335
RAIL_IN = RAIL_OUT + RAIL_THICK
LINE_X = 0.46
PIVOT_XY = (LINE_X, 0.045)
CLAMP_GAP = 0.070  # roller centre at CLOSED to the stopcock pivot
CLAMP_OPEN_XY = (LINE_X, PIVOT_XY[1] - CLAMP_GAP - CLAMP_TRAVEL)
GROUP_WARD_Y = -1.215  # ward y of robot-frame y = 0: holder and terminal clip on the rail's flat top
WARD_TRANSLATE = (RAIL_OUT + RAIL_WARD_X, GROUP_WARD_Y, 0.0)
# Rail-hung holder shelf on the mattress side, 9 cm above the mattress.
SHELF_X = (RAIL_IN + 0.006, 0.575)
SHELF_Y = (-0.130, 0.125)
SHELF_THICK = 0.012
STRAP_Y = (-0.090, 0.085)
# Roller organizer (B5 tray turned along y), standing on the shelf.
TRAY_SIZE = (0.052, 0.094, 0.010)
TRACK_HALF_GAP = 0.018
# Order terminal clipped on the rail beside the holder: the countdown dial stands on it and the
# order screen fills its bed-side face.
DIAL_Y = 0.205
TERMINAL_SIZE = (0.050, 0.210, 0.125)
SCREEN_SIZE = (0.200, 0.115)
# IV stand with the two pumps (upper MAINTENANCE, lower REQUESTED), on the floor at the foot end.
POLE_XY = (-0.12, -0.72)
# The common rig (hospital/rig.py) puts the idle arm 0.60 m to +y of the driving arm, where the
# ward's ultrasound cart stands (robot x -0.42..0.25, y 0.49..1.38): the cart is pushed 0.50 m
# further along the aisle toward the bed head and 0.12 m back, clear of the base and the arm.
WARD_MOVE = {name: (-0.12, 0.50, 0.0) for name in ULTRASOUND_CART}
CAMERA = {"position": [1.00, -0.07, 1.55], "look_at": [0.30, 0.06, 1.08], "focal_mm": 22.0}
# The order screen: header, then numbered steps ("|" breaks a step's text into lines).
ORDER_LINES = (
    ("SWITCH INFUSION", None),
    ("1", "CLOSE MAINTENANCE|CLAMP"),
    ("2", "TURN STOPCOCK|TO REQUESTED"),
)


def _mount(config):
    from roboclock.envs.lab.robot import mount_positions

    return np.asarray(mount_positions(config, "B4")[0], dtype=float)


def build_layout(b):
    """The roller clamp and the stopcock on one holder hung on the bed rail beside the patient."""
    z = b.z
    shelf_top = z + SHELF_THICK
    sx, sy = (SHELF_X[0] + SHELF_X[1]) / 2, (SHELF_Y[0] + SHELF_Y[1]) / 2
    size = (SHELF_X[1] - SHELF_X[0], SHELF_Y[1] - SHELF_Y[0], SHELF_THICK)
    # Visual shelf; its collider is the task's work surface (below), sized to match.
    b.fixture("stopcock_rail_shelf", (sx, sy), size, "stainless", collision=False)
    # Step 2: the stopcock on its mount block.
    b.fixture("stopcock_mount", PIVOT_XY, (0.042, 0.042, 0.060), "cabinet_white", z=shelf_top)
    b.item(NAME, NAME, PIVOT_XY, z + 0.074, color=(0.05, 0.35, 0.85), yaw=0.0)
    # Step 1: the roller in its organizer tray (the B5 tray, turned to run along y).
    cx, oy = CLAMP_OPEN_XY
    center_z = shelf_top + 0.042  # roller axle height: the tray base is 42 mm below it
    b.item(
        ROLLER_NAME, ROLLER_ASSET, (cx, oy), center_z - 0.011 - 0.002,
        color=(0.15, 0.48, 0.80), yaw=math.pi / 2,
    )
    mid = oy + CLAMP_TRAVEL / 2
    b.fixture("clamp_tray", (cx, mid - 0.003), TRAY_SIZE, "cabinet_white", z=shelf_top)
    for sign, suffix in ((-1, "rail_side"), (1, "bed_side")):
        b.fixture(
            f"clamp_track_{suffix}", (cx + sign * TRACK_HALF_GAP, mid), (0.005, 0.074, 0.019),
            "cabinet_white", z=center_z - 0.029,
        )
    for dy, suffix in ((-0.018, "open"), (CLAMP_TRAVEL + 0.018, "closed")):
        b.fixture(
            f"clamp_stop_{suffix}", (cx, oy + dy), (0.041, 0.007, 0.019), "cabinet_white",
            z=center_z - 0.029,
        )
    # The order terminal on the rail clip: invisible collider matching its visual box (the
    # backend's dial housing is narrower), with the countdown dial standing on top.
    clip = RAIL_TOP + 0.008
    rail_mid = RAIL_OUT + RAIL_THICK / 2
    b.fixture(
        "order_terminal", (rail_mid, DIAL_Y), TERMINAL_SIZE, "cabinet_white", z=clip, visible=False
    )
    # The backend's dial housing (0.10 m tall) sits inside the terminal, flush with its top.
    housing_center = clip + TERMINAL_SIZE[2] - 0.05
    b.indicator("indicator", (rail_mid, DIAL_Y), "SWITCH ORDER", height=housing_center - 0.005 - z)
    pivot = b.objects[NAME].position
    origin = b.objects[ROLLER_NAME].position.copy()  # roller at OPEN; +y runs to CLOSED
    shift = [float(v) for v in b.shift]
    translate = [WARD_TRANSLATE[0] + shift[0], WARD_TRANSLATE[1] + shift[1], WARD_TRANSLATE[2]]
    center = b.xy(sx, sy)
    mount = _mount(b.config)
    b.info.update(
        bedside_task="stopcock",
        stopcock={"pivot": pivot.tolist(), "target_angle_rad": TARGET_ANGLE},
        clamp={"origin": origin.tolist(), "axis": [0.0, 1.0, 0.0], "travel": CLAMP_TRAVEL},
        stopcock_scene={
            "shift": shift,
            "table_height": float(z),
            "shelf": {
                "x": [float(b.xy(x, 0)[0]) for x in SHELF_X],
                "y": [float(b.xy(0, y)[1]) for y in SHELF_Y],
                "top": float(shelf_top),
            },
            "rail": {
                "out_x": float(b.xy(RAIL_OUT, 0)[0]),
                "in_x": float(b.xy(RAIL_IN, 0)[0]),
                "thick": RAIL_THICK,
                "top": RAIL_TOP,
                "mattress_top": MATTRESS_TOP,
                "mattress_x": float(b.xy(RAIL_OUT + MATTRESS_EDGE, 0)[0]),
            },
            "straps_y": [float(b.xy(0, y)[1]) for y in STRAP_Y],
            "dial": [float(v) for v in b.xy(rail_mid, DIAL_Y)],
            "terminal": {"size": list(TERMINAL_SIZE), "bottom": float(clip)},
            "pole": [float(v) for v in b.xy(*POLE_XY)],
            "camera": dict(CAMERA),
            "order": [list(line) for line in ORDER_LINES],
        },
        ward_pose={"usd": WARD_USD, "translate": translate, "yaw": 180.0, "move": dict(WARD_MOVE)},
        hide_bench=True,
        work_surface={"center": [*map(float, center), z + SHELF_THICK / 2], "size": list(size)},
        # The robot starts from rest, tucked in front of its own base (no pre-positioning).
        initial_ee_position=(mount + np.asarray(TUCK_OFFSET)).tolist(),
        initial_ee_yaw=HAND_YAW,
        initial_gripper_width=PUSH_WIDTH,
    )
    return b


def _collision(backend, prim):
    """Apply the same contact material to the replacement handle colliders."""
    from pxr import UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    if hasattr(backend, "_collision_settings"):
        backend._collision_settings(prim)
    material = backend.stage.GetPrimAtPath(f"/World/Materials/{NAME}")
    if material.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(material), materialPurpose="physics"
        )


def _tube(look, path, points, radius, material):
    """A visual tube through `points`: cylinders between them, spheres at the bends."""
    from pxr import Gf, UsdGeom, UsdShade

    stage = look.stage
    UsdGeom.Xform.Define(stage, path)
    visual = look.material(material) if isinstance(material, str) else material
    points = [np.asarray(p, dtype=float) for p in points]
    for i, (a, c) in enumerate(zip(points[:-1], points[1:])):
        d = c - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        u = d / length
        axis = np.cross([0.0, 0.0, 1.0], u)
        s = float(np.linalg.norm(axis))
        if s < 1e-9:
            q = Gf.Quatf(1.0, 0.0, 0.0, 0.0) if u[2] > 0 else Gf.Quatf(0.0, 1.0, 0.0, 0.0)
        else:
            angle = math.atan2(s, float(u[2]))
            axis = axis / s * math.sin(angle / 2)
            q = Gf.Quatf(math.cos(angle / 2), *[float(v) for v in axis])
        seg = UsdGeom.Cylinder.Define(stage, f"{path}/Seg{i}")
        seg.CreateRadiusAttr(float(radius))
        seg.CreateHeightAttr(length)
        seg.CreateAxisAttr("Z")
        seg.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in (a + c) / 2]))
        seg.AddOrientOp().Set(q)
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(visual.material)
    for i, p in enumerate(points[1:-1]):
        look.sphere(f"{path}/Bend{i}", p, radius, visual)


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


# Stopcock ports (unit xy direction from the pivot): the line runs along +y, the side port faces
# the rail. The OFF arm starts on REQUESTED and a +90-degree (counterclockwise) turn puts it on
# MAINTENANCE, so REQUESTED then connects to PATIENT.
PORTS = (("Maintenance", (0.0, -1.0)), ("Patient", (0.0, 1.0)), ("Requested", (-1.0, 0.0)))


def _skins(look):
    """Textured PBR materials (CC0 sets in hospital/assets/textures) for the B4 props.

    Color-coded parts keep their color as the tint of a molded-plastic map. The tubing itself, the
    lit order screen, the detent collars and the printed labels stay flat.
    """
    t = look.textured
    return {
        # Rail-hung holder: anodized aluminum straps and hooks (the shelf and braces are stainless).
        "holder": t("B4HolderAluminum", "aluminum_brushed"),
        # Molded white parts: stopcock mount block, roller-clamp organizer, OFF arrow, luer collars.
        "molded": t("B4MoldedWhite", "plastic_white", color=(0.95, 0.96, 0.97), roughness=0.3),
        # Clear polycarbonate stopcock body and luer connectors (an opaque, slightly tinted proxy).
        "clear": t("B4ClearPolycarbonate", "plastic_white", color=(0.78, 0.89, 0.94), roughness=0.14),
        "handle": t("B4StopcockHandle", "plastic_white", color=(0.05, 0.31, 0.82), roughness=0.26),
        "shell": t("B4TerminalShell", "plastic_white", color=(0.90, 0.91, 0.92), roughness=0.4),
        "bezel": t("B4TerminalBezel", "plastic_matte", color=(0.22, 0.23, 0.25)),
        "dressing": t("B4Dressing", "fabric_wrap", color=(0.98, 0.98, 0.96)),
        # One mobile robot look: the ARXMount shells' own light grey matte plastic, a dark
        # powder-coated frame and skirt, rubber casters and brushed aluminum trim.
        "robot_shell": t("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90)),
        "robot_frame": t("robot_frame_dark", "powder_coat_dark"),
    }


def _rebind(stage, path, material):
    """Bind a material to a layout fixture's visual (`<path>_Visual` for colliding fixtures)."""
    from pxr import UsdShade

    for candidate in (path + "_Visual", path):
        prim = stage.GetPrimAtPath(candidate)
        if prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material)
            return prim
    return None


def author(backend):
    """Replace visual/proxy children before reset; never move a body at runtime."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage, look = backend.stage, backend.appearance
    root = stage.GetPrimAtPath(ROOT)
    for child in list(root.GetChildren()):
        stage.RemovePrim(child.GetPath())
    skins = _skins(look)
    blue, molded = skins["handle"], skins["molded"]
    # The tubing stays a flat clear PVC; the stopcock body and luers get the molded clear skin.
    clear = look.material("StopcockClear", color=[0.74, 0.87, 0.92], roughness=0.18)
    body = skins["clear"]
    for name in ("stopcock_mount", "clamp_tray", "clamp_track_rail_side", "clamp_track_bed_side",
                 "clamp_stop_open", "clamp_stop_closed"):
        _rebind(stage, f"/World/Fixtures/{name}", molded)
    # The narrow raised hub gives the parallel fingertips a real friction grip;
    # wings remain below it so the open hand can descend without striking them.
    hub = look.cylinder(ROOT + "/Hub", [0, 0, 0.003], 0.012, 0.020, blue)
    _collision(backend, hub.GetPrim())
    for name, size in (("WingX", [0.052, 0.008, 0.010]), ("WingY", [0.008, 0.052, 0.010])):
        look.rounded(ROOT + "/" + name, [0, 0, -0.007], size, blue, 0.002)
        col = UsdGeom.Cube.Define(stage, ROOT + "/" + name + "Collider")
        col.CreateSizeAttr(1.0)
        col.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.007))
        col.AddScaleOp().Set(Gf.Vec3f(*size))
        _collision(backend, col.GetPrim())
        UsdGeom.Imageable(col).MakeInvisible()
    # A white OFF pointer along one wing disambiguates the otherwise symmetric cross: it starts on
    # the REQUESTED side port (-x) and must end on MAINTENANCE (-y). Visual only.
    look.rounded(ROOT + "/OffPointer", [-0.0205, 0, -0.0008], [0.021, 0.007, 0.0025], molded, 0.0008)
    # The same arrow molded on the hub top, visible from any camera elevation.
    look.rounded(ROOT + "/HubArrow", [-0.0055, 0, 0.0134], [0.011, 0.0035, 0.0012], molded, 0.0005)
    pivot = np.asarray(backend.layout.info["stopcock"]["pivot"])
    x, y, z = map(float, pivot)
    base = "/World/Bedside/Stopcock"
    housing = look.cylinder(base + "/Housing", [x, y, z - 0.024], 0.017, 0.016, body)
    _collision(backend, housing.GetPrim())
    # Three visible luer connectors and short line stubs, kept below the handle sweep and
    # fingertips. These passive tubes do not score the task.
    for name, (dx, dy) in PORTS:
        d = np.array([dx, dy, 0.0])
        axis = "X" if dx else "Y"
        center = pivot + 0.027 * d + [0, 0, -0.024]
        look.cylinder(base + "/" + name + "Luer", center, 0.007, 0.025, body, axis)
        for index in range(3):
            ring = center + (index - 1) * 0.003 * d
            look.cylinder(base + "/" + name + f"Thread{index}", ring, 0.008, 0.0014, molded, axis)
        line = pivot + 0.0485 * d + [0, 0, -0.024]
        look.cylinder(base + "/" + name + "Tube", line, 0.0025, 0.019, clear, axis)
    # Detent marks: colored collars on the line stubs, clear of the handle sweep and visible from
    # above. Blue: start (the pointer's REQUESTED port); green: the 90-degree target (MAINTENANCE).
    look.cylinder(base + "/StartMark", [x - 0.047, y, z - 0.024], 0.0048, 0.012, "aed_blue", "X")
    look.cylinder(
        base + "/TargetMark", [x, y - 0.047, z - 0.024], 0.0048, 0.012, "monitor_green", "Y"
    )
    joint = UsdPhysics.RevoluteJoint.Define(stage, JOINT_PATH)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ROOT)])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(90.0)
    # Reuse the laboratory's passive viscous bearing: no position target,
    # spring, scripted motor, attachment to the tool, or task-state force.
    bearing = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    bearing.CreateTypeAttr("force")
    bearing.CreateStiffnessAttr(0.0)
    bearing.CreateDampingAttr(0.008)
    bearing.CreateTargetVelocityAttr(0.0)
    bearing.CreateMaxForceAttr(0.015)
    if "clamp" in backend.layout.info:
        author_roller(backend)
    scene = backend.layout.info.get("stopcock_scene")
    if scene is not None:
        _dress(backend, scene, pivot, clear, skins)
    return joint


def author_roller(backend):
    """The B5 passive roller, turned to slide along +y (OPEN at 0, CLOSED at CLAMP_TRAVEL).

    The body's own frame is yawed 90 degrees, so its local x (the B5 slide axis) is world +y and
    the B5 wheel visuals are reused unchanged in body coordinates.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage, look = backend.stage, backend.appearance
    root = stage.GetPrimAtPath(ROLLER_ROOT)
    if not root.IsValid():
        raise ValueError("The B4 roller rigid body must exist before author()")
    for child in root.GetChildren():
        # Keep the backend's conservative box collider and remove its generic box appearance.
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            UsdGeom.Imageable(child).MakeInvisible()
    look.cylinder(ROLLER_ROOT + "/KnurledWheel", (0.0, 0.0, 0.0), 0.011, 0.020, "cap_blue", axis="Y")
    for sign in (-1, 1):
        look.cylinder(
            ROLLER_ROOT + f"/Axle_{'L' if sign < 0 else 'R'}", (0, sign * 0.011, 0), 0.003, 0.003,
            "cap_white", axis="Y",
        )
    for index in range(24):
        theta = 2 * np.pi * index / 24
        offset = [0.0108 * np.sin(theta), 0.0, 0.0108 * np.cos(theta)]
        tooth = look.rounded(
            ROLLER_ROOT + f"/Rib_{index:02d}", offset, [0.0012, 0.020, 0.0014], "cap_blue", 0.00025
        )
        UsdGeom.Xformable(tooth.GetPrim()).AddRotateYOp().Set(float(theta * 180 / np.pi))
    origin = np.asarray(backend.layout.info["clamp"]["origin"], dtype=float)
    joint = UsdPhysics.PrismaticJoint.Define(stage, ROLLER_JOINT_PATH)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ROLLER_ROOT)])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(CLAMP_TRAVEL)
    half = math.sqrt(0.5)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*origin.tolist()))
    joint.CreateLocalRot0Attr(Gf.Quatf(half, 0.0, 0.0, half))  # joint x = world +y
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.6)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(0.35)
    # Horizontal placement avoids gravity-driven travel. Viscous drag represents the
    # restrained roller mechanism; actual released endpoint retention is still required.
    joint.GetPrim().SetCustomDataByKey("mechanism", "passive_stabilized_IV_roller")
    return joint


def _dress(backend, scene, pivot, clear, skins=None):
    """Visual-only bedside dressing: rail holder, lines, labels, order terminal, IV stand, base."""
    look = backend.appearance
    config = look.config
    skins = skins or _skins(look)
    x, y, z = map(float, pivot)
    base = "/World/Bedside/Stopcock"
    rail, shelf = scene["rail"], scene["shelf"]
    out, inner, top = rail["out_x"], rail["in_x"], rail["top"]
    holder = skins["holder"]
    # Rail-hung holder: two straps down the rail's mattress-side face, hooks over its top bar with a
    # lip on the outer face, and braces under the shelf (the shelf is the layout fixture).
    h0 = shelf["top"] - 0.07
    for k, sy in enumerate(scene["straps_y"]):
        look.rounded(
            f"{base}/Strap{k}", [inner + 0.003, sy, (h0 + top + 0.012) / 2],
            [0.006, 0.026, top + 0.012 - h0], holder, 0.002,
        )
        look.rounded(
            f"{base}/Hook{k}", [(out + inner) / 2, sy, top + 0.006],
            [inner - out + 0.012, 0.026, 0.012], holder, 0.003,
        )
        look.rounded(
            f"{base}/HookLip{k}", [out - 0.003, sy, top - 0.012], [0.006, 0.026, 0.036], holder, 0.002
        )
        _tube(
            look, f"{base}/Brace{k}",
            [[inner + 0.006, sy, h0 + 0.008], [shelf["x"][1] - 0.03, sy, shelf["top"] - SHELF_THICK]],
            0.005, "stainless",
        )
    zt = z - 0.024
    clamp = backend.layout.info.get("clamp")
    # MAINTENANCE line: from the upper pump, over the rail, through the roller clamp (under the
    # roller) and into the stopcock's upstream port.
    px, py = scene["pole"]
    pump_out = [px + 0.06, py + 0.03, 0.93]
    if clamp is not None:
        ox, oy, oz = clamp["origin"]
        tube_z = oz - 0.019
        ym = shelf["y"][0] + 0.012  # the line comes over the rail beside the holder's upstream end
        maint = _spline(
            [
                pump_out,
                [pump_out[0] + 0.03, pump_out[1] + 0.06, pump_out[2] - 0.12],
                [(out + pump_out[0]) / 2, (ym + pump_out[1]) / 2, top - 0.24],
                [out - 0.03, ym - 0.03, top - 0.12],
                [out - 0.006, ym, top - 0.02],
                [(out + inner) / 2, ym, top + 0.0105],
                [inner + 0.006, ym + 0.004, top - 0.012],
                [inner + 0.02, ym + 0.012, shelf["top"] + 0.03],
                [ox - 0.004, oy - 0.072, tube_z + 0.004],
                [ox, oy - 0.047, tube_z],
            ],
            5,
        )
        # Straight through the organizer, under the roller, then up into the MAINTENANCE luer.
        maint += [
            np.array([ox, oy + CLAMP_TRAVEL + 0.028, tube_z]),
            np.array([x, y - 0.075, zt - 0.006]),
            np.array([x, y - 0.058, zt]),
        ]
        _tube(look, f"{base}/MaintenanceLine", maint, 0.0025, clear)
        _organizer_plates(look, config, clamp)
    # PATIENT line: along the holder toward the head end, then onto the mattress (patient's arm).
    mz = rail["mattress_top"] + 0.006
    edge = shelf["y"][1]
    patient = _spline(
        [
            [x, y + 0.058, zt],
            [x + 0.005, edge - 0.01, zt],
            [x + 0.03, edge + 0.035, zt - 0.03],
            [x + 0.06, edge + 0.065, mz + 0.04],
            [x + 0.10, edge + 0.10, mz],
            [x + 0.20, edge + 0.20, mz],
            [x + 0.26, edge + 0.36, mz],
        ]
    )
    _tube(look, f"{base}/PatientLine", patient, 0.0025, clear)
    look.rounded(
        f"{base}/PatientDressing", [x + 0.262, edge + 0.40, mz - 0.002], [0.05, 0.07, 0.006],
        skins["dressing"], 0.002,
    )
    # REQUESTED line: the primed second infusion, from the side port back over the rail to the
    # lower pump.
    pump2_out = [px + 0.06, py + 0.03, 0.74]
    req = _spline(
        [
            [x - 0.058, y, zt],
            [inner + 0.03, y + 0.002, zt + 0.006],
            [inner + 0.006, y + 0.004, top - 0.008],
            [(out + inner) / 2, y + 0.002, top + 0.0105],
            [out - 0.006, y - 0.004, top - 0.02],
            [out - 0.035, y - 0.03, top - 0.16],
            [(out + pump2_out[0]) / 2, (y + pump2_out[1]) / 2, top - 0.30],
            [pump2_out[0] + 0.03, pump2_out[1] + 0.06, pump2_out[2] - 0.10],
            pump2_out,
        ],
        5,
    )
    _tube(look, f"{base}/RequestedLine", req, 0.0025, clear)
    _labels(look, config, scene, pivot, clamp)
    _order_terminal(backend, scene, skins)
    _iv_stand(look, scene)
    _mobile_base(look, config, scene, skins)


LABELS = (  # text, band color (matching the detent collars), anchor, offset, width
    ("MAINTENANCE", (0.18, 0.66, 0.30), "clamp", (0.066, -0.034), 0.094),
    ("TURN 90\u00b0 \u21ba", (0.95, 0.55, 0.10), "pivot", (0.072, -0.012), 0.084),
    ("PATIENT", (0.55, 0.58, 0.62), "pivot", (0.052, 0.074), 0.070),
    ("REQUESTED", (0.15, 0.40, 0.90), "pivot", (-0.050, 0.060), 0.084),
)


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


def _cache_dir():
    import os

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _label_texture(text, band, size=(640, 160)):
    """A printed line label: colored band at the left, bold dark text as large as fits."""
    import hashlib

    from PIL import Image, ImageDraw

    key = hashlib.sha256(repr((text, band, size, 2)).encode()).hexdigest()[:20]
    path = _cache_dir() / f"a1_{key}.png"
    if path.exists():
        return path
    w, h = size
    image = Image.new("RGB", size, (248, 248, 244))
    draw = ImageDraw.Draw(image)
    stripe = int(h * 0.42)
    draw.rectangle([0, 0, stripe, h], fill=tuple(round(255 * c) for c in band))
    draw.rectangle([0, 0, w - 1, h - 1], outline=(60, 64, 70), width=5)
    font_size = int(h * 0.62)
    while font_size > 12 and _font(font_size).getlength(text) > (w - stripe) * 0.92:
        font_size -= 2
    draw.text(((w + stripe) / 2, h / 2), text, fill=(16, 20, 26), font=_font(font_size), anchor="mm")
    image.save(path)
    return path


def _plate_texture(text, color, size=(400, 160)):
    """A colored plate with white bold text (the organizer's OPEN / CLOSED end plates)."""
    import hashlib

    from PIL import Image, ImageDraw

    key = hashlib.sha256(repr(("plate", text, color, size)).encode()).hexdigest()[:20]
    path = _cache_dir() / f"b4_{key}.png"
    if path.exists():
        return path
    image = Image.new("RGB", size, tuple(round(255 * c) for c in color))
    draw = ImageDraw.Draw(image)
    font_size = int(size[1] * 0.72)
    while font_size > 12 and _font(font_size).getlength("CLOSED") > size[0] * 0.88:
        font_size -= 2
    draw.text((size[0] / 2, size[1] / 2), text, fill=(255, 255, 255), font=_font(font_size), anchor="mm")
    image.save(path)
    return path


def _order_texture(lines, size=(1056, 600)):
    """The bedside order screen: a dark display with an amber header and two numbered steps."""
    import hashlib

    from PIL import Image, ImageDraw

    key = hashlib.sha256(repr(("order", lines, size, 4)).encode()).hexdigest()[:20]
    path = _cache_dir() / f"b4_{key}.png"
    if path.exists():
        return path
    w, h = size
    image = Image.new("RGB", size, (14, 20, 30))
    draw = ImageDraw.Draw(image)
    header_h = int(h * 0.27)
    draw.rectangle([0, 0, w, header_h], fill=(236, 140, 24))
    header = lines[0][0]
    font_size = int(header_h * 0.70)
    while font_size > 12 and _font(font_size).getlength(header) > w * 0.92:
        font_size -= 2
    draw.text((w / 2, header_h / 2), header, fill=(16, 18, 22), font=_font(font_size), anchor="mm")
    chips = ((46, 168, 76), (38, 102, 230))
    row_h = (h - header_h) / max(1, len(lines) - 1)
    chip = int(row_h * 0.62)
    text_x = int(w * 0.04) + chip + int(w * 0.035)
    for k, (number, text) in enumerate(lines[1:]):
        y0 = header_h + k * row_h
        cy = y0 + row_h / 2
        x0 = int(w * 0.04)
        draw.rounded_rectangle(
            [x0, cy - chip / 2, x0 + chip, cy + chip / 2], radius=chip // 5, fill=chips[k % 2]
        )
        draw.text((x0 + chip / 2, cy), number, fill=(255, 255, 255), font=_font(int(chip * 0.75)), anchor="mm")
        wrapped = text.split("|")
        size_t = int(row_h * 0.40)
        while size_t > 12 and max(_font(size_t).getlength(s) for s in wrapped) > (w - text_x) * 0.96:
            size_t -= 2
        step = size_t * 1.12
        for i, line in enumerate(wrapped):
            ty = cy + (i - (len(wrapped) - 1) / 2) * step
            draw.text((text_x, ty), line, fill=(244, 246, 248), font=_font(size_t), anchor="lm")
        if k:
            draw.line([(x0, y0), (w - x0, y0)], fill=(60, 72, 90), width=3)
    image.save(path)
    return path


def _card(stage, path, texture, size, emissive=0.0):
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
    if emissive:
        # A lit display: the same texture glows, so it reads at any ward exposure.
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.0, 1.0, 1.0, 1.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
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
    ).Set([(0, 0), (0, 1), (1, 1), (1, 0)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _labels(look, config, scene, pivot, clamp=None):
    """Line labels standing on the holder shelf, turned toward the task camera."""
    if not config["visuals"].get("labels", True):
        return
    anchors = {"pivot": np.asarray(pivot, dtype=float)}
    if clamp is not None:
        anchors["clamp"] = np.asarray(clamp["origin"], dtype=float)
    top = scene["shelf"]["top"]
    cam = np.asarray(scene["camera"]["position"], dtype=float)
    for k, (text, band, anchor, (ox, oy), width) in enumerate(LABELS):
        if anchor not in anchors:
            continue
        lx, ly = anchors[anchor][0] + ox, anchors[anchor][1] + oy
        # A card reads correctly from +y of its own frame: turn that toward the camera.
        yaw = math.degrees(math.atan2(-(cam[0] - lx), cam[1] - ly))
        group = look.group(f"/World/Bedside/StopcockTags/Tag_{k}", [lx, ly, top + 0.0135], rz=yaw)
        _card(look.stage, group + "/Label", _label_texture(text, band), (width, width / 4))
        look.rounded(group + "/Back", [0.0, -0.0022, -0.0005], [width + 0.004, 0.003, width / 4 + 0.004], "steel", 0.001)


def _organizer_plates(look, config, clamp):
    """OPEN and CLOSED plates on the organizer's bed-side face, at the two ends of the slot."""
    if not config["visuals"].get("labels", True):
        return
    ox, oy, oz = clamp["origin"]
    face = ox + TRACK_HALF_GAP + 0.0026
    for text, dy, color in (
        ("OPEN", -0.004, (0.10, 0.48, 0.25)),
        ("CLOSED", CLAMP_TRAVEL + 0.004, (0.72, 0.16, 0.14)),
    ):
        group = look.group(f"/World/Bedside/ClampPlates/{text}", [face + 0.0008, oy + dy, oz - 0.0195], rz=-90.0)
        _card(look.stage, group + "/Plate", _plate_texture(text, color), (0.034, 0.0136))


def _order_terminal(backend, scene, skins=None):
    """The order terminal on the rail: dial on top, the lit order screen on its bed-side face.

    The backend's dial housing and its small side label are hidden; the terminal box replaces them
    (its collider is the layout's invisible `order_terminal` fixture).
    """
    from pxr import UsdGeom

    stage, look = backend.stage, backend.appearance
    skins = skins or _skins(look)
    for path in ("/World/Labels/indicator_indicator", "/World/Equipment/indicator_Housing_Visual"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    dx, dy = scene["dial"]
    size = np.asarray(scene["terminal"]["size"], dtype=float)
    bottom = scene["terminal"]["bottom"]
    rail = scene["rail"]
    center = np.array([dx, dy, bottom + size[2] / 2])
    look.rounded("/World/Bedside/OrderTerminal", center, size, skins["shell"], 0.006)
    look.rounded(
        "/World/Bedside/OrderTerminalClip", [dx, dy, rail["top"] + 0.002],
        [rail["in_x"] - rail["out_x"] + 0.016, 0.060, 0.014], "rubber", 0.003,
    )
    look.rounded(
        "/World/Bedside/OrderBezel", center + [size[0] / 2 + 0.001, 0, 0],
        [0.003, SCREEN_SIZE[0] + 0.006, SCREEN_SIZE[1] + 0.006], skins["bezel"], 0.001,
    )
    if not look.config["visuals"].get("labels", True):
        return
    lines = tuple(tuple(line) for line in scene.get("order", ORDER_LINES))
    group = look.group(
        "/World/Bedside/OrderScreen", center + [size[0] / 2 + 0.0028, 0, 0], rz=-90.0
    )
    _card(stage, group + "/Screen", _order_texture(lines), SCREEN_SIZE, emissive=1.0)


def _iv_stand(look, scene):
    """The bedside IV stand with the two pumps feeding MAINTENANCE and REQUESTED (visual)."""
    from pxr import UsdShade

    from roboclock.envs.hospital.props import add_model, model_size

    px, py = scene["pole"]
    if model_size("iv_stand") is not None:
        add_model(look.stage, "iv_stand", "/World/Bedside/IVStand", (px, py, 0.0))
        # The model's own maps are a near-uniform grey bake (pole, legs: part_004) and a rust-stained
        # bake on the four casters (part_000..003) that renders as flat grey/black at this distance:
        # bind brushed stainless and rubber instead. The infusion pumps' maps (keypad, display,
        # labels) do show and are kept.
        for k in range(5):
            prim = look.stage.GetPrimAtPath(f"/World/Bedside/IVStand/part_{k:03d}")
            if prim.IsValid():
                skin = look.material("stainless" if k == 4 else "rubber")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    skin.material, UsdShade.Tokens.strongerThanDescendants
                )
    else:
        look.cylinder("/World/Bedside/IVStandPole", [px, py, 0.9], 0.012, 1.8, "stainless")
    if model_size("infusion_pump") is not None:
        for name, z in (("InfusionPump", 0.99), ("RequestedPump", 0.80)):
            add_model(
                look.stage, "infusion_pump", f"/World/Bedside/{name}",
                (px + 0.02, py + 0.10, z), (0.8, 0.8, 0.8), yaw=math.pi / 2,
            )


def _mobile_base(look, config, scene, skins=None):
    """The X5 columns stand on one wheeled base on the ward floor (visual; the robot is fixed).

    Light grey matte shells (the ARXMount shells' own material, so each pedestal reads as one
    column up to the arm), a dark powder-coated skirt, rubber casters and brushed aluminum trim.
    """
    from roboclock.envs.lab.robot import mount_positions

    skins = skins or _skins(look)
    shell, frame = skins["robot_shell"], skins["robot_frame"]
    z = scene["table_height"]
    mounts = [np.asarray(m[:2], dtype=float) for m in mount_positions(config, "B4")]
    lo, hi = np.min(mounts, axis=0) - 0.11, np.max(mounts, axis=0) + 0.11
    center, size = (lo + hi) / 2, hi - lo
    look.rounded("/World/Bedside/RobotBase", [*center, 0.075], [*size, 0.11], shell, 0.02)
    look.rounded("/World/Bedside/RobotBumper", [*center, 0.030], [*(size + 0.012), 0.03], frame, 0.01)
    look.rounded("/World/Bedside/RobotTrim", [*center, 0.128], [*(size + 0.004), 0.008], "aluminum", 0.003)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        wheel = [center[0] + sx * (size[0] / 2 - 0.05), center[1] + sy * (size[1] / 2 - 0.05), 0.03]
        look.cylinder(f"/World/Bedside/RobotWheel_{i}", wheel, 0.03, 0.025, "rubber", axis="Y")
    for i, m in enumerate(mounts):
        look.rounded(
            f"/World/Bedside/RobotColumn_{i}", [*m, (0.13 + z) / 2], [0.12, 0.12, z - 0.13], shell, 0.006
        )
        # Aluminum collar where the column meets the arm's mounting column.
        look.rounded(f"/World/Bedside/RobotCollar_{i}", [*m, z], [0.126, 0.126, 0.014], "aluminum", 0.003)


def _camera(backend):
    """A bedside view from over the mattress: devices, order terminal, the reach and the robot."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["stopcock_scene"]["camera"]
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


def _quaternion_angle(quaternion):
    """(yaw about z, tilt of the body's z axis) from a w-x-y-z quaternion; NaN/inf if invalid."""
    quaternion = np.asarray(quaternion, dtype=float)
    valid = bool(np.all(np.isfinite(quaternion)) and abs(np.linalg.norm(quaternion) - 1.0) < 0.01)
    if not valid:
        return float("nan"), float("inf")
    w, x, y, z = quaternion
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    tilt = math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))
    return yaw, tilt


def measure(backend, state):
    """Privileged PhysX measurements; no command or event state can set angle or travel."""
    camera = getattr(backend, "camera", None)
    if (
        camera is not None
        and "stopcock_scene" in backend.layout.info
        and getattr(backend, "_a1_camera_for", None) is not backend.task
    ):
        backend._a1_camera_for = backend.task
        _camera(backend)
    obj = state.objects[NAME]
    angle, axis_error = _quaternion_angle(obj.orientation)
    pivot = np.asarray(backend.layout.info["stopcock"]["pivot"])
    speed = backend.objects[NAME].get_angular_velocity()
    result = {
        "stopcock_angle_rad": float(angle),
        "stopcock_speed_rad_s": float(speed[2]),
        "stopcock_axis_error_rad": axis_error,
        "stopcock_pivot_error_m": float(np.linalg.norm(obj.position - pivot)),
        "stopcock_robot_contact": bool(obj.robot_contact),
        "stopcock_grasped": bool(obj.grasped),
    }
    clamp = backend.layout.info.get("clamp")
    roller = state.objects.get(ROLLER_NAME)
    if clamp is not None and roller is not None:
        origin = np.asarray(clamp["origin"], dtype=float)
        axis = np.asarray(clamp.get("axis", (0.0, 1.0, 0.0)), dtype=float)
        delta = np.asarray(roller.position, dtype=float) - origin
        travel = float(delta @ axis)
        result.update(
            clamp_position_m=travel,
            clamp_speed_m_s=float(np.linalg.norm(roller.velocity)),
            clamp_axis_error_m=float(np.linalg.norm(delta - travel * axis)),
            clamp_contact=bool(roller.robot_contact or roller.grasped),
            clamp_grasped=bool(roller.grasped),
            clamp_closed_fraction=float(np.clip(travel / CLAMP_TRAVEL, 0.0, 1.0)),
        )
    return result


def _waypoints(info):
    """Oracle waypoints (world): push start/end on the roller, the hub grip point."""
    origin = np.asarray(info["clamp"]["origin"], dtype=float)
    pivot = np.asarray(info["stopcock"]["pivot"], dtype=float)
    start = origin + np.array([0.0, -READY_OFFSET, PUSH_HEIGHT])
    end = origin + np.array([0.0, CLAMP_TRAVEL - PUSH_OFFSET, PUSH_HEIGHT])
    grip = pivot + np.array([0.0, 0.0, GRIP_HEIGHT])
    return start, end, grip


def command(planner, task, state):
    """From the tucked pose: push the clamp closed, release it, then pinch and turn the stopcock.

    One arm, in the clinical order. The reach comes over the rail and descends beside the roller's
    upstream face; the closed fingers' edges push it to its CLOSED stop; the hand lifts off (the
    clamp must be released to count), crosses to the hub with open fingers, descends around it and
    pinches. The turn starts only after the evaluator has confirmed the retained clamp closure.
    The commanded yaw never leads the measured handle by more than TURN_LEAD and may pass the
    90-degree detent by TURN_OVERSHOOT only, because the fingertips slip a few degrees on the hub
    and the handle must reach its hard stop. The fingers open only after the detent retention is
    confirmed.
    """
    if planner.last_goal is None:
        planner.last_goal = state.ee_position.copy()
    info = task.layout.info
    start, end, grip = _waypoints(info)
    ee = np.asarray(state.ee_position, dtype=float)
    obj = state.objects[NAME]
    controls = state.controls
    angle = float(controls.get("stopcock_angle_rad", 0.0))
    if not math.isfinite(angle):
        angle = 0.0
    travel = float(controls.get("clamp_position_m", 0.0))
    planner.gentle = GENTLE_CLOSE
    if task.event_time is None:
        # At rest, tucked at the base, until the order is issued.
        planner.phase = "wait_order"
        planner.yaw = HAND_YAW
        planner.turn = 0.0
        planner.open_width = planner.close_width = PUSH_WIDTH
        return planner.finish(state, np.asarray(info["initial_ee_position"]), False, 0.12)
    if task.goal_achieved:
        planner.phase = "retain_detent"
        planner.open_width = READY_WIDTH
        return planner.finish(state, grip + [0.0, 0.0, HOVER_HEIGHT + 0.03], False, 0.15)
    step1 = planner.phase in ("wait_order", "reach_clamp", "lower_clamp", "push_clamp")
    if step1 and travel < CLAMP_TRAVEL - CLAMP_END_TOLERANCE:
        planner.yaw = HAND_YAW
        planner.open_width = planner.close_width = PUSH_WIDTH
        if planner.phase in ("wait_order", "reach_clamp"):
            planner.phase = "reach_clamp"
            if np.linalg.norm(ee[:2] - start[:2]) > 0.012:
                return planner.finish(state, start + [0.0, 0.0, APPROACH_LIFT], False, REACH_SPEED)
            planner.phase = "lower_clamp"
        if planner.phase == "lower_clamp":
            if np.linalg.norm(ee - start) > 0.006:
                return planner.finish(state, start, False, DESCEND_SPEED)
            planner.phase = "push_clamp"
        return planner.finish(state, end, False, PUSH_SPEED)
    if step1:
        planner.phase = "lift_clamp"
    planner.open_width, planner.close_width = READY_WIDTH, SQUEEZE_WIDTH
    hover = grip + np.array([0.0, 0.0, TRANSFER_LIFT])
    if planner.phase == "lift_clamp":
        # Lift straight off the roller (it must be released to count), then cross to the hub.
        if ee[2] < end[2] + 0.018:
            planner.yaw = HAND_YAW
            back = np.array([end[0], end[1] - 0.004, end[2] + 0.04])
            return planner.finish(state, back, False, TRANSFER_SPEED)
        planner.phase = "to_stopcock"
    if planner.phase == "to_stopcock":
        planner.yaw = HAND_YAW
        if np.linalg.norm(ee[:2] - hover[:2]) > 0.006:
            return planner.finish(state, hover, False, TRANSFER_SPEED)
        planner.phase = "lower_stopcock"
    near = np.linalg.norm(ee - grip) < 0.009
    if planner.phase == "lower_stopcock" and not near:
        planner.yaw = HAND_YAW + angle  # re-grip at the handle's current angle
        planner.turn = angle
        return planner.finish(state, grip, False, DESCEND_SPEED)
    if not obj.grasped:
        if planner.phase == "turn_stopcock" and not near:
            planner.phase = "lower_stopcock"  # lost the hub: re-approach at its current angle
            return planner.finish(state, grip, False, DESCEND_SPEED)
        planner.phase = "pinch_stopcock"
        planner.turn = angle
        planner.yaw = HAND_YAW + angle
        return planner.finish(state, grip, True, 0.20)
    if task.clamp_time is None:
        # Grip held, but the stopcock is never turned before the clamp closure is confirmed.
        planner.phase = "pinch_stopcock"
        planner.yaw = HAND_YAW + getattr(planner, "turn", angle)
        return planner.finish(state, grip, True, 0.20)
    planner.phase = "turn_stopcock"
    # Limit the commanded yaw lead so loss of friction cannot accumulate an
    # unreachable target; only finger contact can change the actual handle.
    turn = getattr(planner, "turn", angle)
    desired = min(TARGET_ANGLE + TURN_OVERSHOOT, angle + TURN_LEAD)
    turn += float(np.clip(desired - turn, 0.0, TURN_RATE * planner.dt))
    planner.turn = turn
    planner.yaw = HAND_YAW + turn
    return planner.finish(state, grip, True, 0.20)
