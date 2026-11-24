"""B1 oxygen desaturation: bedside monitor, wall oxygen flowmeter with a contact-turned knob.

Placement (scene-02 preview left bay, assets/hospital/ward_02_preview_left.usdc; the B6 pose): the
robot is a wheeled dual-arm base parked beside the bed, facing the bay's ceiling equipment pendant
with the bed on its left. A medical-gas service panel hangs on the pendant's column in front of the
robot. On it, in front of the driving arm (internal arm 0, the rig's right hand), an oxygen outlet
with a plugged-in Thorpe-tube flowmeter; beside it, in front of the head camera, the bedside
monitor's display head, tilted up toward the robot. Oxygen tubing runs from the flowmeter's barbed
outlet to the mask of the patient lying in the bed.

Downloaded models (CC BY 4.0, assets/models/credits.json, scripts/hospital/import_oxygen_assets.py):
`oxygen_flowmeter` (outlet plate, valve body, glass tube; its ball float and control knob are
separate files so they can move), `patient_monitor_head`, `patient_supine`, `oxygen_mask`. Custom:
the service panel, the monitor's live screen and alarm lamp, the band marks and scale card, the
tubing, the order card and the mobile base.

Mechanics: the knob is the catalog's `stopcock_handle` body re-authored as a horizontal 24 mm
cylinder on a free revolute joint about the wall normal (a viscous bearing, no motor, spring or
task force), in front of the valve body as on front-knob flowmeters. Only finger contact turns it.
`measure` reports its angle (counterclockwise seen from the robot, unwrapped) and contact; the task
turns the measured angle into flow, float height and SpO2 (tasks/reactive/oxygen_desaturation.py)
and `apply_visuals` poses the ball float, the monitor's number and the alarm lamp from the task.
The cue prims are switched by scale and moved by translation only: RTX shows transform changes on
the same frame, material and visibility edits several frames late.

Oracle: at rest, tucked at its base (tool pointing down), until the alarm. Then the hand swings to
a forward-pointing pose in front of the knob (one joint-space move), advances onto the knob,
pinches it and rolls the wrist counterclockwise until the measured flow is the ordered flow, holds
until the task confirms the float settled in the band, lets go and backs off.
"""

import math
from pathlib import Path

import numpy as np

from roboclock.envs.hospital.bedside import pump_panel as bay
from roboclock.envs.hospital.tasks.reactive.oxygen_desaturation import (
    ALARM_THRESHOLD,
    KNOB,
    MAX_FLOW,
    angle_for_flow,
)

NAME = KNOB
ROOT = f"/World/Objects/{NAME}"
JOINT_PATH = "/World/Joints/OxygenKnobBearing"
BASE = "/World/Bedside/B1"
MODELS = Path(__file__).resolve().parents[1] / "assets" / "models"

# --- flowmeter model (assets/models/oxygen_flowmeter, measured by the import script) ---------------
# Model frame: +x out of the wall (plate back at x = 0), z up, origin on the outlet axis.
TUBE_AXIS_X = 0.0967
TUBE_Z = (0.0248, 0.1345)
TUBE_OUTER_RADIUS = 0.0154
BALL_REST_Z = 0.0301  # float centre at zero flow
BALL_TRAVEL = 0.094  # float rise from 0 to MAX_FLOW L/min
BALL_SCALE = 1.4  # the 6.8 mm float drawn 9.5 mm across (the tube bore is 10 mm)
VALVE_FRONT_X = 0.1143
BARB_TIP = (0.0967, -0.0016, -0.0962)
KNOB_RADIUS = 0.012
KNOB_HALF_LENGTH = 0.0137
KNOB_X = VALVE_FRONT_X + 0.004 + KNOB_HALF_LENGTH  # knob centre: 4 mm proud of the valve front
# Knob travel (degrees, counterclockwise seen from the robot = opening): from fully closed
# (BASELINE_FLOW / FLOW_PER_RAD below the start) to the top of the scale.
KNOB_CLOSE_DEG = 36.0
KNOB_OPEN_DEG = 234.0

# --- placement (layout frame: x away from the robot, y to its left, before the seed shift) --------
# The knob stands KNOB_OFFSET from the driving arm's mount (rig origin robot.rig.tasks.B1): 0.44 m
# out and 0.22 m up, where the forward-pointing X5 has IK solutions with margin (probe 2026-11-19:
# forward tool solvable from 0.20 m above the mount for 0.25-0.55 m out), and the tube's top stays
# inside the head camera's image.
KNOB_OFFSET = (0.44, 0.08, 0.22)
PANEL_Y = (-0.24, 0.56)  # the service panel's span
PANEL_Z = (-0.22, 0.27)  # from the knob height
PANEL_THICK = 0.024
MONITOR_OFFSET = (-0.045, 0.285, 0.04)  # screen centre from the knob
MONITOR_TILT = 28.0  # degrees the screen leans back (up toward the head camera)
SCREEN_SIZE = (0.250, 0.170)
LEDGE_FRONT = 0.055  # the instrument ledge under the panel reaches this far in front of the knob
LEDGE_THICK = 0.020
DIAL_OFFSET = (0.035, -0.170, PANEL_Z[0])  # countdown dial housing (bottom centre) from the knob
# The bed (ward file): mattress top 0.67 m, flat up to 0.08 m beyond the patient's hips, then the
# raised backrest climbs 13 degrees to 0.86 m at the head end. The rigid supine model is pitched
# with the backrest about its hips; its legs, which the pitch sinks into the mattress, lie under
# a blanket. The ward's own plump pillow (0.2 m thick, it would swallow the head) is replaced by
# a flat one.
PATIENT_XY = (0.87, 1.335)  # the patient's hips (model centre) on the bed, head toward +x
PATIENT_Z = 0.672
PATIENT_PITCH = -12.0  # degrees about the model's x axis: head up
WARD_PILLOW = "KR_012"
BLANKET = {"beyond_feet": 0.06, "above_hips": 0.16, "half_width": 0.43, "thick": 0.22}
CAMERA = {"position": [-0.25, -0.75, 1.75], "look_at": [0.60, 0.25, 1.35], "focal_mm": 22.0}

# --- oracle --------------------------------------------------------------------------------------
TUCK = (0.0, -0.16, 0.10)  # rest pose of the hand from its mount, tool pointing down
FORWARD = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)  # tool tilt: pointing +x
READY_WIDTH = 0.0150  # finger joint position (half opening); the knob surface is at 0.012
SQUEEZE_WIDTH = 0.005
GRIP_BACK = 0.004  # the grasp centre stops this far in front of the knob's centre
STANDOFF = 0.075  # the swing ends this far in front of the grasp point
HAND_YAW = -0.75  # wrist roll at the pinch, so the turn stays inside the last joint's range
ADVANCE_SPEED = 0.30
TURN_RATE = 2.5  # commanded roll rate (rad/s); the wrist itself tracks at about 1.1 rad/s
TURN_LEAD = 0.16  # the commanded roll never leads the measured knob by more than this
FLOW_TOLERANCE = 0.25  # L/min: the oracle stops turning this close to the ordered flow

SPO2_VALUES = tuple(range(78, 101))


def _mount(config):
    from roboclock.envs.lab.robot import mount_positions

    return np.asarray(mount_positions(config, "B1")[0], dtype=float)


def geometry(config, shift=(0.0, 0.0)):
    """World positions of the apparatus (pure; used by the layout, the tests and the oracle)."""
    shift = np.array([float(shift[0]), float(shift[1]), 0.0])
    knob = _mount(config) + np.asarray(KNOB_OFFSET) + shift
    origin = knob + [KNOB_X, 0.0, 0.0]  # flowmeter model origin; the model is turned 180 degrees

    def model(x, y, z):
        return origin + [-x, -y, z]

    screen = knob + np.asarray(MONITOR_OFFSET)
    return {
        "knob": knob,
        "origin": origin,
        "tube_bottom": model(TUBE_AXIS_X, 0.0, TUBE_Z[0]),
        "tube_top": model(TUBE_AXIS_X, 0.0, TUBE_Z[1]),
        "ball_rest": model(TUBE_AXIS_X, 0.0, BALL_REST_Z),
        "barb": model(*BARB_TIP),
        "panel_x": float(origin[0]),
        "screen": screen,
        "dial": knob + np.asarray(DIAL_OFFSET),
    }


def ball_height(flow):
    """Float rise (m) above its rest for a flow in L/min."""
    return BALL_TRAVEL * min(max(float(flow), 0.0), MAX_FLOW) / MAX_FLOW


def screen_yaw(screen, head):
    """Yaw (degrees) that turns the monitor's +x screen normal toward the head camera."""
    return math.degrees(math.atan2(head[1] - screen[1], head[0] - screen[0]))


def build_layout(b):
    config, task = b.config, b.task
    g = geometry(config, b.shift)
    knob = g["knob"]
    unshift = knob[:2] - b.shift
    # The knob body (catalog stopcock_handle: 18 g, friction >= 1.2): its origin on the joint axis.
    b.item(NAME, "stopcock_handle", unshift, knob[2] - 0.013 - 0.002, color=(0.10, 0.45, 0.20), yaw=0.0)
    # Invisible colliders: the valve body with its tube, and the monitor head.
    valve_x = g["origin"][0] - VALVE_FRONT_X / 2 - b.shift[0]
    b.fixture(
        "oxygen_valve", (valve_x, unshift[1]), (VALVE_FRONT_X, 0.040, 0.054), "cabinet_white",
        z=knob[2] - 0.027, visible=False,
    )
    b.fixture(
        "oxygen_tube", (g["tube_top"][0] - b.shift[0], unshift[1]),
        (2 * TUBE_OUTER_RADIUS, 2 * TUBE_OUTER_RADIUS, TUBE_Z[1] - TUBE_Z[0] + 0.008),
        "cabinet_white", z=knob[2] + TUBE_Z[0], visible=False,
    )
    screen = g["screen"]
    b.fixture(
        "monitor_head", (screen[0] + 0.07 - b.shift[0], screen[1] - b.shift[1]), (0.14, 0.34, 0.23),
        "cabinet_white", z=screen[2] - 0.115, visible=False,
    )
    dial = g["dial"]
    lo, hi = task.band
    label = f"IF SpO2 < {ALARM_THRESHOLD:g} %: O2 TO {task.ordered_flow:g} L/MIN"
    b.indicator("indicator", dial[:2] - b.shift, label, height=dial[2] + 0.045 - b.z)
    panel_center = [
        g["panel_x"] + PANEL_THICK / 2,
        float(b.xy(0.0, (PANEL_Y[0] + PANEL_Y[1]) / 2)[1]),
        float(knob[2] + (PANEL_Z[0] + PANEL_Z[1]) / 2),
    ]
    panel_size = [PANEL_THICK, PANEL_Y[1] - PANEL_Y[0], PANEL_Z[1] - PANEL_Z[0]]
    b.fixture(
        "gas_panel", (panel_center[0] - b.shift[0], panel_center[1] - b.shift[1]), panel_size,
        "cabinet_white", z=panel_center[2] - panel_size[2] / 2, visible=False,
    )
    # The instrument ledge at the panel's foot carries the dial; it is the task's support surface.
    ledge_x = (float(knob[0]) - LEDGE_FRONT, g["panel_x"] + PANEL_THICK)
    ledge_center = [
        (ledge_x[0] + ledge_x[1]) / 2, panel_center[1], float(knob[2] + PANEL_Z[0] - LEDGE_THICK / 2)
    ]
    ledge_size = [ledge_x[1] - ledge_x[0], panel_size[1], LEDGE_THICK]
    camera = {
        "position": [float(v) for v in (*b.xy(*CAMERA["position"][:2]), CAMERA["position"][2])],
        "look_at": [float(v) for v in (*b.xy(*CAMERA["look_at"][:2]), CAMERA["look_at"][2])],
        "focal_mm": CAMERA["focal_mm"],
    }
    b.info.update(
        bedside_task="oxygen_flow",
        oxygen_flow={
            "knob": knob.tolist(),
            "origin": g["origin"].tolist(),
            "tube_bottom": g["tube_bottom"].tolist(),
            "tube_top": g["tube_top"].tolist(),
            "ball_rest": g["ball_rest"].tolist(),
            "barb": g["barb"].tolist(),
            "screen": screen.tolist(),
            "dial": dial.tolist(),
            "panel": {"center": panel_center, "size": panel_size},
            "ledge": {"center": ledge_center, "size": ledge_size},
            "band_l_min": [lo, hi],
            "ordered_flow_l_min": task.ordered_flow,
            "patient": [*map(float, b.xy(*PATIENT_XY)), PATIENT_Z],
            "camera": camera,
            "placement": "gas service panel on the bedside equipment pendant, robot beside the bed",
        },
        ward_pose={
            "usd": bay.WARD_USD,
            "translate": [bay.ROBOT_Y + float(b.shift[0]), -bay.ROBOT_X + float(b.shift[1]), bay.WARD_Z],
            "yaw": bay.WARD_YAW,
            "hide": [*bay.TROLLEY, WARD_PILLOW],
            "move": dict(bay.WARD_MOVE),
        },
        hide_bench=True,
        # The support collider is the ledge; the panel, valve and monitor have their own colliders.
        work_surface={"center": ledge_center, "size": ledge_size},
        # At rest, folded near its own column: the reach to the flowmeter is part of the response.
        initial_ee_position=(_mount(config) + np.asarray(TUCK)).tolist(),
        initial_ee_yaw=0.0,
        initial_gripper_width=READY_WIDTH,
    )
    return b


# --- process state (pure; unit tested without the simulator) -------------------------------------


def knob_angle(quaternion, previous=None):
    """(angle, axis error): the knob's turn about the wall normal, counterclockwise seen from the
    robot (rad, unwrapped against `previous`), and the tilt of its axis off the wall normal."""
    q = np.asarray(quaternion, dtype=float)
    if not (np.all(np.isfinite(q)) and abs(np.linalg.norm(q) - 1.0) < 0.01):
        return float("nan"), float("inf")
    w, x, y, z = q
    # Positive rotation about world +x (pointing away from the robot) is clockwise from the robot.
    angle = -math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    if previous is not None and math.isfinite(previous):
        angle += 2.0 * math.pi * round((previous - angle) / (2.0 * math.pi))
    axis_error = math.acos(float(np.clip(1.0 - 2.0 * (y * y + z * z), -1.0, 1.0)))
    return angle, axis_error


def monitor_state(task):
    """What the monitor shows: (whole-number SpO2 clamped to the drawn range, alarm lamp lit)."""
    value = int(min(max(task.displayed_spo2(), SPO2_VALUES[0]), SPO2_VALUES[-1]))
    lamp = task.alarm_active and int(task.time / 0.25) % 2 == 0
    return value, bool(lamp)


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


def _reference(stage, path, usd, translate=(0.0, 0.0, 0.0), rotations=(), scale=1.0):
    """Reference a model USD at `path`; `rotations` are (axis, degrees), applied last to first."""
    from pxr import Gf, UsdGeom

    node = UsdGeom.Xform.Define(stage, path)
    node.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translate]))
    for axis, degrees in rotations:
        {"x": node.AddRotateXOp, "y": node.AddRotateYOp, "z": node.AddRotateZOp}[axis]().Set(float(degrees))
    if scale != 1.0:
        node.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
    node.GetPrim().GetReferences().AddReference(str(usd))
    return node


def _glass(look, name, tint):
    """Thin-walled tinted glass (OmniGlass): what is behind and inside it stays visible. An
    OmniPBR surface with fractional opacity renders as a cutout here: below about one half it
    vanishes altogether (the tube did), above it is opaque."""
    from isaacsim.core.api.materials.visual_material import VisualMaterial
    from pxr import Gf, Sdf, UsdShade

    if name not in look.materials:
        path = f"/World/Materials/{name}"
        material = UsdShade.Material.Define(look.stage, path)
        shader = UsdShade.Shader.Define(look.stage, path + "/Shader")
        shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.5)
        shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.004)
        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.12)
        out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mdl").ConnectToSource(out)
        look.materials[name] = VisualMaterial(
            name=name, prim_path=path, prim=material.GetPrim(), shaders_list=[shader], material=material
        )
    return look.materials[name]


def author(backend):
    """The knob (passive revolute joint) and, in the Isaac scene, the dressed bedside."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = backend.stage
    root = stage.GetPrimAtPath(ROOT)
    for child in list(root.GetChildren()):
        stage.RemovePrim(child.GetPath())
    info = backend.layout.info["oxygen_flow"]
    x, y, z = (float(v) for v in info["knob"])
    collider = UsdGeom.Cylinder.Define(stage, ROOT + "/Collider")
    collider.CreateRadiusAttr(KNOB_RADIUS)
    collider.CreateHeightAttr(2 * KNOB_HALF_LENGTH)
    collider.CreateAxisAttr("X")
    _collision(backend, collider.GetPrim())
    UsdGeom.Imageable(collider).MakeInvisible()
    # The downloaded knob (cap and stem), turned with the model: its +x (out of the wall) to -x.
    _reference(stage, ROOT + "/Knob", MODELS / "oxygen_flowmeter" / "oxygen_flowmeter_knob.usd",
               rotations=[("z", 180.0)])
    joint = UsdPhysics.RevoluteJoint.Define(stage, JOINT_PATH)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ROOT)])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    # Counterclockwise from the robot is a negative rotation about world +x.
    joint.CreateLowerLimitAttr(-KNOB_OPEN_DEG)
    joint.CreateUpperLimitAttr(KNOB_CLOSE_DEG)
    # A needle valve's packing: viscous friction only, no position target, spring or motor.
    bearing = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    bearing.CreateTypeAttr("force")
    bearing.CreateStiffnessAttr(0.0)
    bearing.CreateDampingAttr(0.008)
    bearing.CreateTargetVelocityAttr(0.0)
    bearing.CreateMaxForceAttr(0.015)
    if hasattr(backend, "world"):  # the Isaac scene; plain-USD unit stages get the knob only
        backend._b1_visuals = _dress(backend, info)
        task = getattr(backend, "task", None)
        if task is not None and getattr(task, "layout", None) is not None:
            apply_visuals(backend, task)  # pose the cues for the reset frame
    return joint


def _dress(backend, info):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    from roboclock.envs.hospital.rig import dial_target

    stage, look = backend.stage, backend.appearance
    labels = look.config["visuals"].get("labels", True)
    origin = np.asarray(info["origin"], dtype=float)
    head = np.asarray(dial_target(backend, info["camera"]["position"]), dtype=float)
    # The colliders' own box visuals are replaced by the models and the dressed panel below.
    for name in ("oxygen_valve", "oxygen_tube", "monitor_head", "gas_panel"):
        prim = stage.GetPrimAtPath(f"/World/Fixtures/{name}_Visual")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    # --- service panel on the pendant column ---------------------------------------------------
    panel = info["panel"]
    shell = look.textured("b1_panel", "plastic_matte", color=(0.90, 0.92, 0.93), roughness=0.5)
    trim = look.textured("b1_panel_trim", "aluminum_brushed", repeats=10.0)
    look.rounded(BASE + "/Panel", panel["center"], panel["size"], shell, 0.006)
    cx, cy, cz = panel["center"]
    sx, sy, sz = panel["size"]
    for k, dz in enumerate((-sz / 2 + 0.012, sz / 2 - 0.012)):
        look.rounded(f"{BASE}/PanelRail_{k}", [cx - sx / 2 - 0.004, cy, cz + dz], [0.010, sy - 0.02, 0.016], trim, 0.003)
    ledge = info["ledge"]
    look.rounded(BASE + "/Ledge", ledge["center"], ledge["size"], shell, 0.004)
    look.rounded(BASE + "/LedgeLip", [ledge["center"][0] - ledge["size"][0] / 2 + 0.004, cy,
                                      ledge["center"][2] + 0.002], [0.008, sy - 0.01, LEDGE_THICK + 0.004], trim, 0.003)
    # --- flowmeter (downloaded shell, float and knob) --------------------------------------------
    flow = _reference(stage, BASE + "/Flowmeter", MODELS / "oxygen_flowmeter" / "oxygen_flowmeter.usd",
                      translate=origin, rotations=[("z", 180.0)])
    # Tinted: clear glass in front of the white panel would not read at all.
    glass = _glass(look, "b1_tube_glass", (0.55, 0.78, 0.92))
    for name in ("cone01_Glass_0", "valve03_Glass_0", "innerTube_Glass_0"):
        prim = stage.GetPrimAtPath(f"{flow.GetPath()}/{name}")
        if prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                glass.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    rest = np.asarray(info["ball_rest"], dtype=float)
    ball = _reference(stage, BASE + "/Float", MODELS / "oxygen_flowmeter" / "oxygen_flowmeter_ball.usd",
                      translate=rest, scale=BALL_SCALE)
    ball_op = ball.GetOrderedXformOps()[0]
    # The float is drawn signal orange (the model's is black) so it reads against the band.
    orange = look.material("b1_float", color=(1.0, 0.30, 0.02), roughness=0.35, emissive=True)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{ball.GetPath()}/ball")).Bind(
        orange.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    # --- band marks and scale card (custom) ------------------------------------------------------
    lo, hi = info["band_l_min"]
    green = look.material("b1_band_green", color=(0.05, 0.70, 0.22), roughness=0.4, emissive=True)
    tube_x, tube_y = float(rest[0]), float(rest[1])
    for k, flow_value in enumerate((lo, hi)):
        look.cylinder(f"{BASE}/BandRing_{k}", [tube_x, tube_y, rest[2] + ball_height(flow_value)],
                      TUBE_OUTER_RADIUS + 0.0012, 0.0024, green)
    mid = rest[2] + ball_height((lo + hi) / 2)
    span = ball_height(hi) - ball_height(lo)
    # A green backing strip behind the tube between the two rings (on the plate side).
    look.rounded(f"{BASE}/BandBack", [tube_x + TUBE_OUTER_RADIUS + 0.004, tube_y, mid],
                 [0.002, 0.036, span], green, 0.0008)
    if labels:
        card_y = tube_y - 0.058
        card_x = float(origin[0]) - 0.012
        yaw = math.degrees(math.atan2(head[1] - card_y, head[0] - card_x))
        group = _xform(stage, BASE + "/ScaleCard", [card_x - 0.004, card_y, rest[2] + BALL_TRAVEL / 2],
                       [("z", yaw)])
        height = BALL_TRAVEL + 0.03
        look.rounded(group + "/Back", [-0.002, 0.0, 0.0], [0.003, 0.052, height + 0.006], shell, 0.001)
        bay._quad(stage, group + "/Label", _scale_texture(lo, hi), (0.0, 0.0, 0.0), (0.048, height),
                  rotations=[("z", -90.0)])
    # --- monitor head (downloaded) with the live screen (custom) ---------------------------------
    screen = np.asarray(info["screen"], dtype=float)
    yaw = screen_yaw(screen, head)
    monitor = _xform(stage, BASE + "/Monitor", screen, [("z", yaw), ("y", -MONITOR_TILT)])
    _reference(stage, monitor + "/Head", MODELS / "patient_monitor_head" / "patient_monitor_head.usd")
    # Bracket from the panel to the monitor's back.
    look.rounded(BASE + "/MonitorArm", [(screen[0] + 0.12 + cx) / 2, screen[1], screen[2] - 0.02],
                 [max(cx - screen[0] - 0.12, 0.02), 0.05, 0.05], trim, 0.006)
    dark = look.material("b1_screen_dark", color=(0.01, 0.012, 0.015), roughness=0.2)
    w, h = SCREEN_SIZE
    look.rounded(monitor + "/ScreenGlass", [0.0015, 0.0, 0.0], [0.002, w + 0.012, h + 0.012], dark, 0.001)
    digits = {}
    for value in SPO2_VALUES:
        mesh = bay._quad(stage, f"{monitor}/Screen_{value}", _screen_texture(value), (0.0032, 0.0, 0.0),
                         (w, h), rotations=[("z", -90.0)], emissive=True)
        scale = UsdGeom.Xformable(mesh).AddScaleOp()
        scale.Set(Gf.Vec3f(1e-4, 1e-4, 1e-4))
        digits[value] = scale
    # Alarm lamp on the monitor's top edge: a grey dome with a red emissive dome switched by scale.
    lamp_at = [-0.02, 0.0, 0.114 + 0.012]
    look.rounded(monitor + "/LampBase", [lamp_at[0], 0.0, 0.114 + 0.003], [0.04, 0.10, 0.006], shell, 0.002)
    look.sphere(monitor + "/LampOff", lamp_at, 0.014, look.material("b1_lamp_off", color=(0.55, 0.50, 0.50), roughness=0.3))
    red = look.material("b1_lamp_red", color=(1.0, 0.03, 0.02), roughness=0.3, emissive=True)
    lamp = look.sphere(monitor + "/LampOn", lamp_at, 0.0155, red)
    lamp_scale = lamp.AddScaleOp()
    lamp_scale.Set(Gf.Vec3f(1e-4, 1e-4, 1e-4))
    # --- patient, mask and oxygen tubing ---------------------------------------------------------
    px, py, pz = info["patient"]
    _reference(stage, BASE + "/Patient", MODELS / "patient_supine" / "patient_supine.usd",
               translate=[px, py, pz], rotations=[("z", 90.0), ("x", PATIENT_PITCH)])

    def on_patient(point):
        """World position of a point of the patient model (pitched about x, then turned 90 degrees)."""
        x, y, z = (float(v) for v in point)
        a = math.radians(PATIENT_PITCH)
        y, z = y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)
        return np.array([px - y, py + x, pz + z])

    credits = _credits().get("patient_supine", {})
    nose = np.asarray(credits.get("nose_tip_m", (0.0, -0.73, 0.26)), dtype=float)
    mask_at = on_patient(nose + [0.0, 0.02, 0.012])
    _reference(stage, BASE + "/Mask", MODELS / "oxygen_mask" / "oxygen_mask.usd", translate=mask_at,
               rotations=[("z", 90.0), ("x", 90.0 + PATIENT_PITCH)])
    clear = _glass(look, "b1_mask_clear", (0.60, 0.92, 0.78))
    for part in ("cup", "tubes"):
        prim = stage.GetPrimAtPath(f"{BASE}/Mask/{part}")
        if prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                clear.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
    linen = look.material("b1_linen", color=(0.90, 0.92, 0.94), roughness=0.85)
    blanket = look.material("b1_blanket", color=(0.50, 0.70, 0.86), roughness=0.9)
    crown = on_patient([0.0, -0.80, 0.0])
    pillow = _xform(stage, BASE + "/Pillow", crown + [0.0, 0.0, -0.005], [("y", PATIENT_PITCH)])
    look.rounded(pillow + "/Body", [0.0, 0.0, 0.0], [0.34, 0.56, 0.07], linen, 0.03)
    foot_end = on_patient([0.0, 0.93, 0.0])[0] - BLANKET["beyond_feet"]
    waist_end = px + BLANKET["above_hips"]
    look.rounded(BASE + "/Blanket", [(foot_end + waist_end) / 2, py, pz - 0.02 + BLANKET["thick"] / 2],
                 [waist_end - foot_end, 2 * BLANKET["half_width"], BLANKET["thick"]], blanket, 0.05)
    barb = np.asarray(info["barb"], dtype=float)
    tubing = look.material("b1_o2_tubing", color=(0.62, 0.90, 0.70), roughness=0.3)
    bottom = cz - sz / 2
    chest = on_patient([0.05, -0.45, 0.27])
    control = [
        barb,
        barb + [0.0, 0.01, -0.05],
        [barb[0] + 0.01, barb[1] + 0.16, bottom - 0.05],
        [barb[0] + 0.03, cy + sy / 2 + 0.03, bottom - 0.08],
        [(barb[0] + chest[0]) / 2, py - 0.50, pz + 0.30],
        chest + [-0.12, -0.16, 0.01],
        chest,
        mask_at + [-0.05, 0.0, 0.035],
    ]
    bay._tube(look, BASE + "/Tubing", control, tubing)
    bay._robot_base(backend)
    bay._face_dial(backend, head)
    return {"ball": ball_op, "rest": rest, "digits": digits, "lamp": lamp_scale, "state": None}


def _xform(stage, path, translate, rotations=()):
    """A posed parent prim; `rotations` are (axis, degrees) applied last to first."""
    from pxr import Gf, UsdGeom

    node = UsdGeom.Xform.Define(stage, path)
    node.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translate]))
    for axis, degrees in rotations:
        {"x": node.AddRotateXOp, "y": node.AddRotateYOp, "z": node.AddRotateZOp}[axis]().Set(float(degrees))
    return path


def _credits():
    import json

    path = MODELS / "credits.json"
    return json.loads(path.read_text()) if path.exists() else {}


def apply_visuals(backend, task):
    """Pose the float, the monitor's number and the alarm lamp from the task (visual only)."""
    from pxr import Gf

    vis = getattr(backend, "_b1_visuals", None)
    if vis is None or task is None:
        return
    value, lamp = monitor_state(task)
    rise = round(ball_height(task.float_flow), 5)
    key = (value, lamp, rise)
    if key == vis["state"]:
        return
    previous = vis["state"] or (None, None, None)
    vis["state"] = key
    if rise != previous[2]:
        rest = vis["rest"]
        vis["ball"].Set(Gf.Vec3d(float(rest[0]), float(rest[1]), float(rest[2] + rise)))
    if value != previous[0]:
        for shown, scale in vis["digits"].items():
            s = 1.0 if shown == value else 1e-4
            scale.Set(Gf.Vec3f(s, s, s))
    if lamp != previous[1]:
        s = 1.0 if lamp else 1e-4
        vis["lamp"].Set(Gf.Vec3f(s, s, s))


# --- printed and displayed images ----------------------------------------------------------------


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
    return cache / f"b1o2_{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.png"


def _screen_texture(value, size=(600, 408)):
    """The monitor's screen for one SpO2 reading: cyan when normal, white on red when in alarm."""
    from PIL import Image, ImageDraw

    path = _cache(("screen", value, size, 3))
    if path.exists():
        return path
    w, h = size
    alarm = value < ALARM_THRESHOLD
    image = Image.new("RGB", size, (4, 6, 10))
    draw = ImageDraw.Draw(image)
    cyan, green, white, red = (40, 220, 255), (60, 230, 90), (255, 255, 255), (225, 20, 15)
    if alarm:
        draw.rectangle([0, 0, w, int(h * 0.27)], fill=red)
        draw.text((w / 2, h * 0.135), "LOW SpO2", fill=white, font=_font(int(h * 0.21)), anchor="mm")
        draw.rectangle([int(w * 0.30), int(h * 0.30), w - 8, h - 8], fill=red)
    else:
        draw.text((w * 0.03, h * 0.135), "HR 88", fill=green, font=_font(int(h * 0.15)), anchor="lm")
        draw.text((w * 0.97, h * 0.135), "RR 18", fill=(240, 220, 60), font=_font(int(h * 0.15)), anchor="rm")
    draw.text((w * 0.15, h * 0.50), "SpO2", fill=white if alarm else cyan, font=_font(int(h * 0.13)), anchor="mm")
    draw.text((w * 0.15, h * 0.68), "%", fill=white if alarm else cyan, font=_font(int(h * 0.16)), anchor="mm")
    draw.text((w * 0.645, h * 0.635), str(value), fill=white if alarm else cyan,
              font=_font(int(h * 0.62)), anchor="mm")
    image.save(path)
    return path


def _scale_texture(lo, hi, size=(192, 496)):
    """The flowmeter's scale card: L/min ticks from 0 to MAX_FLOW with the ordered band in green."""
    from PIL import Image, ImageDraw

    path = _cache(("scale", lo, hi, size, 3))
    if path.exists():
        return path
    w, h = size
    image = Image.new("RGB", size, (250, 250, 247))
    draw = ImageDraw.Draw(image)
    height = BALL_TRAVEL + 0.03  # card height (m); the scale's zero sits 0.015 m above its bottom

    def row(flow):
        return h - (0.015 + ball_height(flow)) / height * h

    draw.rectangle([0, row(hi), w, row(lo)], fill=(70, 200, 100))
    for flow in range(0, int(MAX_FLOW) + 1):
        major = flow % 5 == 0
        draw.line([0, row(flow), w * (0.42 if major else 0.25), row(flow)], fill=(20, 20, 20), width=5 if major else 3)
        if major:
            draw.text((w * 0.72, row(flow)), str(flow), fill=(15, 15, 15), font=_font(int(w * 0.30)), anchor="mm")
    draw.text((w * 0.5, h - 10), "L/min", fill=(15, 15, 15), font=_font(int(w * 0.17)), anchor="mb")
    image.save(path)
    return path


# --- runtime -------------------------------------------------------------------------------------


def _camera(backend):
    """The third-person recording view: robot, panel, monitor, flowmeter and the bed behind."""
    camera = getattr(backend, "task_camera", None)
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["oxygen_flow"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    camera.set_focal_length(view["focal_mm"] / 1000)
    for _ in range(3):
        backend.world.render()


def measure(backend, state):
    """Privileged PhysX measurements of the knob; also poses the (visual) float and monitor."""
    task = getattr(backend, "task", None)
    if getattr(backend, "_b1_for", None) is not task:
        backend._b1_for = task
        backend._b1_angle = 0.0
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_b1_camera_for", None) is not task:
        backend._b1_camera_for = task
        _camera(backend)
    obj = state.objects[NAME]
    angle, axis_error = knob_angle(obj.orientation, backend._b1_angle)
    if math.isfinite(angle):
        backend._b1_angle = angle
    pivot = np.asarray(backend.layout.info["oxygen_flow"]["knob"])
    if task is not None and getattr(task, "layout", None) is not None:
        apply_visuals(backend, task)
    return {
        "knob_angle_rad": float(angle),
        "knob_axis_error_rad": axis_error,
        "knob_pivot_error_m": float(np.linalg.norm(obj.position - pivot)),
        "knob_contact": bool(obj.robot_contact or obj.grasped),
        "knob_grasped": bool(obj.grasped),
    }


def command(planner, task, state):
    """Tucked until the alarm; then swing forward, pinch the knob and roll it to the ordered flow."""
    from roboclock.envs.lab.policies.trajectories import CartesianCommand

    info = task.layout.info
    knob = np.asarray(info["oxygen_flow"]["knob"], dtype=float)
    grip = knob - [GRIP_BACK, 0.0, 0.0]
    pre = grip - [STANDOFF, 0.0, 0.0]
    ee = np.asarray(state.ee_position, dtype=float)
    obj = state.objects[NAME]
    angle = float(state.controls.get("knob_angle_rad", 0.0))
    if not math.isfinite(angle):
        angle = 0.0
    planner.open_width, planner.close_width = READY_WIDTH, SQUEEZE_WIDTH
    planner.gentle = False
    if task.event_time is None:
        planner.phase = "wait_alarm"
        planner.tool_tilt, planner.sync_joints = None, False
        planner.yaw = 0.0
        return planner.finish(state, np.asarray(info["initial_ee_position"]), False, 0.12)
    planner.tool_tilt, planner.sync_joints = np.asarray(FORWARD), True
    if task.goal_achieved or task.physical_failures:
        # Let go and back straight off the knob.
        planner.phase = "release"
        return planner.finish(state, pre, False, ADVANCE_SPEED)
    if planner.phase == "wait_alarm":
        planner.phase = "swing"
        planner.turn = 0.0
        task.event_log.append({"time": task.time, "type": "oracle_reach_start"})
    if planner.phase == "swing":
        # One joint-space move (sync_joints) to the forward-pointing pose in front of the knob.
        planner.yaw = HAND_YAW
        planner.last_goal = pre.copy()
        if np.linalg.norm(ee - pre) > 0.012:
            return CartesianCommand(pre.copy(), False, planner.yaw)
        planner.phase = "advance"
    if planner.phase == "advance":
        planner.yaw = HAND_YAW
        if np.linalg.norm(ee - grip) > 0.005:
            return planner.finish(state, grip, False, ADVANCE_SPEED)
        planner.phase = "pinch"
    if planner.phase == "pinch":
        planner.yaw = HAND_YAW
        if not obj.grasped:
            return planner.finish(state, grip, True, 0.20)
        planner.phase = "turn"
        planner.turn = 0.0
        planner.grip_angle = angle
    # Roll the wrist: the commanded roll follows the measured knob toward the ordered flow and
    # never leads it by more than TURN_LEAD (finger slip cannot wind up an unreachable target).
    target = angle_for_flow(task.ordered_flow)
    remaining = target - angle
    done = abs(remaining) <= FLOW_TOLERANCE * math.radians(18.0)
    if not done:
        turned = angle - planner.grip_angle
        desired = min(planner.turn + remaining, turned + TURN_LEAD)
        planner.turn += float(np.clip(desired - planner.turn, -TURN_RATE * planner.dt, TURN_RATE * planner.dt))
    planner.yaw = HAND_YAW + planner.turn
    return planner.finish(state, grip, True, 0.20)
