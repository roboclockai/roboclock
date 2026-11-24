"""B5 "anaphylaxis": give the adrenaline auto-injector into the patient's outer thigh.

Scene (scene-02 left bay, the B4 bed). A patient on an intravenous infusion lies supine in the
bed, head toward +y; the robot is parked beside the thighs. An over-bed table stands across the
shins with the anaphylaxis kit tray on it: the armed auto-injector trainer stands tip down in its
holder (its blue safety release lies beside it). A training injection pad is strapped over the
outer middle of the near thigh. The bedside monitor stands beyond the bed and faces the robot's
head camera; the infusion stand is at the far side of the chest.

Mechanism. The injector is two rigid bodies: the barrel (`epi_pen`, the body the fingers pinch)
and the orange needle-end tip (`epi_tip`) on a prismatic joint along the barrel, pushed out by a
spring against its stop. Nothing is attached or triggered: the tip is pushed in only by pressing
it on something, and `geometry` measures the tip's travel in the barrel's own frame, the barrel's
tilt and whether the tip's end is on the pad. The task (tasks/bedside/autoinjector.py) decides
from those measurements when the injector fires and whether the hold was kept.

What is not modelled: the needle, the drug, the patient's physiology (blood pressure and heart
rate on the monitor are a time-compressed process model), the safety release (it is already
off), the clothing. The pad and the legs are rigid.

Assets. Downloaded models (CC BY 4.0, scripts/hospital/import_b5_assets.py, credits in
assets/models/credits.json): epi_autoinjector (+ _tip, epi_safety_cap), overbed_table, iv_stand;
reused: patient_supine, patient_monitor_head (B1), hospital_bed_flat. Authored here because they
are the measured mechanism or its cue: the tip joint and colliders, the holder, the pad, the
monitor's live screen and lamp, the hives.

All heights follow the configured arm-0 mount (robot.rig.tasks.B5): the mattress top is
MATTRESS_BELOW_MOUNT below it.
"""

import math
from pathlib import Path

import numpy as np

from roboclock.envs.hospital.tasks.bedside.autoinjector import (
    ACTIVATE_TRAVEL,
    ALARM_SYSTOLIC,
    AXIS_TOLERANCE,
    DIASTOLIC_RATIO,
    PEN,
    SYSTOLIC_FLOOR,
    SYSTOLIC_RANGE,
    TIP,
    TIP_TRAVEL,
    heart_rate,
)
from roboclock.physics.geometry import rotation_matrix

BASE = "/World/Bedside/B5"
MODELS = Path(__file__).resolve().parents[1] / "assets" / "models"
INSTRUCTION = "ANAPHYLAXIS: GIVE THE ADRENALINE AUTO-INJECTOR INTO THE OUTER THIGH, HOLD 3 S"
PAD_CONTACT = "fixture:thigh_pad"

# Injector geometry (barrel frame: origin at the barrel's centre, tip toward -z).
PEN_HALF = 0.070  # half the barrel's length
TIP_LENGTH = 0.030  # the tip body's collider; its lower face is the tip's end
TIP_REST = -PEN_HALF - TIP_TRAVEL + TIP_LENGTH / 2  # tip centre in the barrel frame, at rest
SPRING = {"stiffness": 400.0, "damping": 6.0, "max_force": 15.0, "preload": 0.004}
GRASP_UP = 0.040  # the pinch is this far above the barrel's centre

# Patient frame (patient_supine.usd turned head toward +y: world = origin + (-x, -y, z)).
PAD_ON_PATIENT = (0.145, 0.26)  # pad centre: outer middle of the near thigh
PAD_SIZE = (0.090, 0.140, 0.012)
PAD_TOP = 0.250  # above the mattress
PAD_MARGIN = 0.008  # the tip's end must be this far inside the pad's edge
SITE_HEIGHT = 0.005  # ... and within this of its surface
MATTRESS_BELOW_MOUNT = 0.60

# Robot frame before the seed's common shift; arm 0's mount (the rig's right arm) is at the origin.
PAD_XY = (0.31, 0.27)  # nominal pad centre; the seeded offsets below move it
BED_SCALE_Y = 0.50
TABLE_XY = (0.52, -0.07)  # over-bed table top centre, across the shins
TABLE_TOP = 0.30  # above the mattress
TRAY_XY = (0.43, -0.02)
TRAY_SIZE = (0.24, 0.17, 0.012)
HOLDER_XY = (0.39, 0.0)
WELL = (0.0304, 0.0204)  # inner opening: the barrel plus 2 mm each side
WELL_WALL = 0.008
WELL_DEPTH = 0.045
CAP_XY = (0.47, -0.06)
MONITOR_XY = (0.95, 0.44)
MONITOR_BELOW_MOUNT = 0.06
MONITOR_TILT = 24.0
MONITOR_SCALE = 1.3  # the B1 monitor head enlarged: it stands 1 m from the head camera
SCREEN_SIZE = (0.250, 0.170)
HEADER_SHARE = 0.28  # of the screen's height: the alarm banner
IV_XY = (1.02, 0.98)
INDICATOR_XY = (0.68, -0.10)
TUCK = (0.20, -0.04, 0.02)  # from the arm-0 mount
APPROACH_WIDTH = 0.020  # per finger: 40 mm across the 16 mm barrel
CAMERA = {"position": [0.55, -1.25, 1.95], "look_at": [0.50, 0.30, 0.92], "focal_mm": 20.0}
WARD_TRANSLATE = (1.74, -1.19)  # the B4 pose moved with the bed: the robot stands at the thighs
SYSTOLIC_STEP = 4  # the monitor's cards: one per 4 mmHg (a card change takes a few frames to settle)
HIVES = (  # patient frame: x, y, z on the near forearm and hand, radius
    (0.262, 0.045, 0.092, 0.010),
    (0.270, 0.010, 0.100, 0.008),
    (0.258, -0.030, 0.106, 0.011),
    (0.275, -0.065, 0.110, 0.009),
    (0.263, -0.100, 0.112, 0.010),
    (0.250, 0.075, 0.085, 0.007),
)


# Seeded per-episode placement (tasks.B5.spatial in configs/benchmark.yaml gives the ranges, m).
# Drawn from the layout's generator, so a seed has the same placement at every level.
SPATIAL_KEYS = ("tray_dx", "tray_dy", "pad_across", "pad_along", "patient_dx", "patient_dy")
SPATIAL_DEFAULT = {
    "tray_dx": (-0.03, 0.03),  # kit tray (holder, injector, safety release) on the table, robot x
    "tray_dy": (-0.04, 0.04),  # ... and along the table's width, robot y
    "pad_across": (-0.02, 0.01),  # pad around the thigh: + toward the outer side (patient frame x)
    "pad_along": (-0.03, 0.03),  # pad along the thigh: + toward the knee (patient frame y)
    "patient_dx": (-0.015, 0.015),  # where the patient lies on the mattress, robot x
    "patient_dy": (-0.025, 0.025),  # ... and along the bed, robot y
}


def sample_spatial(config, task_id, rng):
    """The episode's placement offsets (m). `tasks.<id>.spatial_force` = {key: u in [0, 1]} pins an
    offset at that fraction of its range (diagnostic runs at the extremes); the generator is
    advanced all the same, so the rest of the layout does not change."""
    task = config["tasks"].get(task_id, {})
    ranges = {**SPATIAL_DEFAULT, **{k: tuple(v) for k, v in (task.get("spatial") or {}).items()}}
    forced = task.get("spatial_force") or {}
    values = {}
    for key in SPATIAL_KEYS:
        lo, hi = (float(v) for v in ranges[key])
        u = float(rng.uniform(0.0, 1.0))
        if key in forced:
            u = float(forced[key])
        values[key] = lo + (hi - lo) * u
    return values, {key: [float(v) for v in ranges[key]] for key in SPATIAL_KEYS}


def mount_height(config, task_id="B5"):
    from roboclock.envs.lab.robot import mount_positions

    return float(mount_positions(config, task_id)[0][2])


def build_layout(builder):
    from roboclock.envs.hospital.bedside.ventilator import WARD_POSE
    from roboclock.envs.hospital.bedside.ward import ULTRASOUND_CART, furniture

    b = builder
    task_id = getattr(b.task, "task_id", "B5")
    mount_z = mount_height(b.config, task_id)
    mattress = mount_z - MATTRESS_BELOW_MOUNT
    b.z = mattress
    pad_top = mattress + PAD_TOP
    table_top = mattress + TABLE_TOP
    floor = table_top + TRAY_SIZE[2]  # the tray's surface: the injector's tip stands on it
    spatial, ranges = sample_spatial(b.config, task_id, b.rng)
    # The patient (head toward +y: patient frame x, y = -robot x, -robot y) and the pad on the thigh.
    nominal_origin = (PAD_XY[0] + PAD_ON_PATIENT[0], PAD_XY[1] + PAD_ON_PATIENT[1])
    origin_xy = (nominal_origin[0] + spatial["patient_dx"], nominal_origin[1] + spatial["patient_dy"])
    pad_on_patient = (PAD_ON_PATIENT[0] + spatial["pad_across"], PAD_ON_PATIENT[1] + spatial["pad_along"])
    pad_xy = (origin_xy[0] - pad_on_patient[0], origin_xy[1] - pad_on_patient[1])
    tray = (spatial["tray_dx"], spatial["tray_dy"])
    tray_xy = (TRAY_XY[0] + tray[0], TRAY_XY[1] + tray[1])
    holder_xy = (HOLDER_XY[0] + tray[0], HOLDER_XY[1] + tray[1])
    cap_xy = (CAP_XY[0] + tray[0], CAP_XY[1] + tray[1])

    # Colliders (all drawn by _dress): the injection pad, the patient under it, the table top,
    # the tray and the four walls of the holder's well.
    b.fixture("thigh_pad", pad_xy, PAD_SIZE, "cap_white", z=pad_top - PAD_SIZE[2], visible=False)
    b.fixture(
        "patient_legs", (origin_xy[0] - 0.025, origin_xy[1] - 0.47), (0.38, 0.92, 0.20), "skin",
        z=mattress, visible=False,
    )
    b.fixture(
        "patient_trunk", (origin_xy[0], origin_xy[1] + 0.45), (0.60, 0.90, 0.24), "skin",
        z=mattress, visible=False,
    )
    b.fixture("overbed_top", TABLE_XY, (0.983, 0.394, 0.030), "cabinet_white", z=table_top - 0.030, visible=False)
    b.fixture("kit_tray", tray_xy, TRAY_SIZE, "cap_white", z=table_top, visible=False)
    wx, wy = WELL[0] / 2 + WELL_WALL / 2, WELL[1] / 2 + WELL_WALL / 2
    outer = (WELL[0] + 2 * WELL_WALL, WELL[1] + 2 * WELL_WALL)
    for name, (dx, dy), size in (
        ("well_xn", (-wx, 0.0), (WELL_WALL, outer[1], WELL_DEPTH)),
        ("well_xp", (wx, 0.0), (WELL_WALL, outer[1], WELL_DEPTH)),
        ("well_yn", (0.0, -wy), (WELL[0], WELL_WALL, WELL_DEPTH)),
        ("well_yp", (0.0, wy), (WELL[0], WELL_WALL, WELL_DEPTH)),
    ):
        b.fixture(name, (holder_xy[0] + dx, holder_xy[1] + dy), size, "cap_white", z=floor, visible=False)
    # The injector stands on its tip in the well: the tip's end 0.5 mm above the tray, the barrel
    # TIP_TRAVEL above that. (An item's centre is support + half height + 2 mm + lift.)
    b.item(PEN, PEN, holder_xy, floor, color=(0.96, 0.86, 0.20), yaw=0.0, lift=TIP_TRAVEL - 0.0015)
    b.item(TIP, TIP, holder_xy, floor, color=(0.98, 0.42, 0.05), yaw=0.0, lift=-0.0015)
    b.indicator("indicator", INDICATOR_XY, "ADRENALINE", height=TABLE_TOP + 0.045)

    pad = np.array([*b.xy(*pad_xy), pad_top])
    origin = np.array([*b.xy(*origin_xy), mattress])
    bed_center = b.xy(nominal_origin[0] - 0.005, nominal_origin[1])  # the bed does not move with the patient
    tuck = np.array([TUCK[0], TUCK[1], mount_z + TUCK[2]])  # robot frame: the robot never shifts
    hide = ULTRASOUND_CART + furniture("bed") + ["iMac_5k_Monitor054"]
    b.info.update(
        {
            "bedside_task": "autoinjector",
            "instruction": INSTRUCTION,
            "attached": [(PEN, TIP)],  # the tip slides inside the barrel: one device, two bodies
            "initial_ee_position": tuck.tolist(),
            "initial_ee_yaw": 0.0,
            "initial_gripper_width": APPROACH_WIDTH,
            "ward_pose": {
                **WARD_POSE,
                "translate": [WARD_TRANSLATE[0] + float(b.shift[0]), WARD_TRANSLATE[1] + float(b.shift[1]), 0.0],
                "hide": hide,
            },
            "hide_bench": True,
            # The mattress is the support surface: nothing falls through the bed.
            "work_surface": {"center": [*bed_center.tolist(), mattress - 0.05], "size": [0.75, 2.0, 0.10]},
            "autoinjector": {
                "mattress_top": mattress,
                "pad": {"center": pad.tolist(), "size": list(PAD_SIZE)},
                "patient_origin": origin.tolist(),
                "bed": {"center": bed_center.tolist(), "scale_y": BED_SCALE_Y},
                "table": {"center": b.xy(*TABLE_XY).tolist(), "top": table_top},
                "tray": {"center": [*b.xy(*tray_xy).tolist(), table_top], "size": list(TRAY_SIZE)},
                "holder": [*b.xy(*holder_xy).tolist(), floor],
                "cap": [*b.xy(*cap_xy).tolist(), floor],
                "spatial": spatial,
                "spatial_ranges": ranges,
                "pad_on_patient": list(pad_on_patient),
                "monitor": [*b.xy(*MONITOR_XY).tolist(), mount_z - MONITOR_BELOW_MOUNT],
                "iv_stand": b.xy(*IV_XY).tolist(),
                "camera": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in CAMERA.items()},
                "assumptions": [
                    "two rigid bodies: barrel and a sprung prismatic tip (10 mm travel); no needle or drug",
                    "the safety release is already off: the trainer stands armed in its holder",
                    "rigid injection pad and legs; injection through the pad stands for through clothing",
                    "blood pressure and heart rate are time-compressed process models",
                ],
            },
        }
    )


# --- measurement ---------------------------------------------------------------------------------


def geometry(layout, pen, tip):
    """Tip travel and where the tip's end is, measured in the barrel's own frame."""
    info = layout.info["autoinjector"]
    rotation = rotation_matrix(pen.orientation)
    axis = rotation[:, 2]  # barrel axis in the world; the tip is toward -axis
    offset = float(np.dot(np.asarray(tip.position) - np.asarray(pen.position), axis))
    travel = float(np.clip(offset - TIP_REST, -0.002, TIP_TRAVEL + 0.002))
    tilt = float(np.arccos(np.clip(axis[2], -1.0, 1.0)))
    end = np.asarray(tip.position) - axis * (TIP_LENGTH / 2)  # the tip's end face centre
    blunt = np.asarray(pen.position) + axis * PEN_HALF  # the barrel's other end
    pad = info["pad"]
    centre, size = np.asarray(pad["center"]), np.asarray(pad["size"])

    def over_pad(point, margin):
        return bool(np.all(np.abs(point[:2] - centre[:2]) <= size[:2] / 2 - margin))

    on_site = over_pad(end, PAD_MARGIN) and abs(float(end[2] - centre[2])) <= SITE_HEIGHT and tilt < math.pi / 2
    blunt_on_site = (
        tilt > math.pi / 2 and over_pad(blunt, 0.0) and abs(float(blunt[2] - centre[2])) <= SITE_HEIGHT
    )
    return {
        "travel": travel,
        "tilt": tilt,
        "aligned": bool(tilt <= AXIS_TOLERANCE),
        "on_site": bool(on_site),
        "blunt_on_site": bool(blunt_on_site),
        "pad_contact": bool(PAD_CONTACT in tip.fixture_contacts),
        "tip_end": end.tolist(),
        "pressed": bool(travel >= ACTIVATE_TRAVEL),
    }


def monitor_state(task):
    """What the monitor shows: (systolic card value, alarm lamp lit)."""
    systolic = task.displayed()[0]
    lo = int(SYSTOLIC_FLOOR)
    hi = int(SYSTOLIC_RANGE[1])
    value = min(max(systolic - systolic % SYSTOLIC_STEP, lo), hi)
    lamp = task.alarm_active and int(task.time / 0.25) % 2 == 0
    return int(value), bool(lamp)


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_b5_camera_for", None) is not backend.task:
        backend._b5_camera_for = backend.task  # the first measurement of a reset aims the view
        _camera(backend)
    measured = geometry(backend.layout, state.objects[PEN], state.objects[TIP])
    _animate(backend)
    return {"autoinjector": measured}


# --- privileged diagnostic oracle ----------------------------------------------------------------

REACH_SPEED, CARRY_SPEED = 0.55, 0.40
HOVER = 0.030  # above the pinch point before the final vertical descent
LIFT, LIFT_SPEED = 0.060, 0.30  # out of the holder's well before moving sideways
CLEAR = 0.012  # tip end above the pad while the injector is brought over it
TRACK_STEP = 0.006
PRESS_STEP, PRESS_SPEED = 0.004, 0.10
PRESS_TRAVEL = 0.009  # tip travel the oracle holds (the barrel 1 mm off the pad)
GRIP_WIDTH = 0.002  # per-finger target on the 16 mm barrel
TAPER = 4.0


def command(planner, task, state):
    """Privileged diagnostic controller; every waypoint becomes bounded articulation commands."""
    info, pen, tip = task.layout.info["autoinjector"], state.objects[PEN], state.objects[TIP]
    now = task.time
    if planner.last_goal is None:
        planner.last_goal = state.ee_position.copy()
    if getattr(planner, "_b5_task", None) is not task:
        planner._b5_task = task
        planner.b5_phase, planner.b5_phase_time = "wait", now
        planner.b5_offset = np.zeros(3)
    phase = planner.b5_phase
    planner.open_width, planner.close_width, planner.gentle = APPROACH_WIDTH, GRIP_WIDTH, True
    planner.step_yaw(0.0)
    close, speed = False, REACH_SPEED
    tuck = np.asarray(task.layout.info["initial_ee_position"])
    goal = tuck.copy()
    measured = geometry(task.layout, pen, tip)
    pad = np.asarray(info["pad"]["center"])

    def transition(value):
        planner.b5_phase, planner.b5_phase_time = value, now
        task.event_log.append({"time": now, "type": "autoinjector_planner_phase", "phase": value})

    def tapered(target, vmax):
        return float(np.clip(TAPER * np.linalg.norm(target - planner.last_goal), 0.05, vmax))

    def pen_goal(end_height):
        """Barrel centre that puts the resting tip's end `end_height` above the pad, upright."""
        return pad + [0.0, 0.0, end_height + TIP_TRAVEL + PEN_HALF]

    def track(target, step=TRACK_STEP):
        # A bounded integral correction on the measured barrel removes IK and load bias.
        correction = np.clip(0.25 * (target - pen.position), -step, step)
        return np.clip(planner.last_goal + correction, state.ee_position - 0.020, state.ee_position + 0.020)

    pinch = pen.position + [0.0, 0.0, GRASP_UP]
    if phase == "wait":
        # At rest in the tucked pose; nothing is touched before the alarm.
        if task.event_time is not None:
            transition("reach")
    elif phase == "reach":
        goal = pinch + [0.0, 0.0, HOVER]
        speed = tapered(goal, REACH_SPEED)
        if (
            np.linalg.norm(state.ee_position[:2] - pinch[:2]) < 0.006
            and state.ee_position[2] - pinch[2] < HOVER + 0.012
        ):
            planner.b5_grasp = pinch.copy()
            transition("descend")
    elif phase == "descend":
        goal, speed = planner.b5_grasp, 0.30
        if abs(state.ee_position[2] - planner.b5_grasp[2]) < 0.006:
            transition("grasp")
    elif phase == "grasp":
        goal, close = planner.b5_grasp, True
        if pen.grasped:
            planner.b5_offset = state.ee_position - pen.position
            planner.b5_start = pen.position.copy()
            transition("lift")
        elif now - planner.b5_phase_time > 1.5:
            transition("reach")  # a missed pinch: open and approach again
    elif phase == "lift":
        goal = planner.b5_start + planner.b5_offset + [0.0, 0.0, LIFT + 0.01]
        close, speed = True, LIFT_SPEED
        if pen.position[2] >= planner.b5_start[2] + LIFT:
            transition("carry")
        elif not pen.grasped and now - planner.b5_phase_time > 0.3:
            transition("reach")
    elif phase == "carry":
        # Across at the lifted height (the tray is higher than the pad), then down over the pad.
        desired = pen_goal(CLEAR)
        if np.linalg.norm(pen.position[:2] - desired[:2]) > 0.015:
            desired[2] = max(desired[2], planner.b5_start[2] + LIFT)
        goal, close = desired + planner.b5_offset, True
        speed = tapered(goal, CARRY_SPEED)
        if np.linalg.norm(pen.position - desired) < 0.010:
            transition("press")
        elif not pen.grasped and now - planner.b5_phase_time > 0.3:
            transition("reach")  # the injector slipped out of the pinch
    elif phase == "press":
        # Push until the tip is PRESS_TRAVEL in: the barrel's end stops 1 mm above the pad.
        desired = pen_goal(-PRESS_TRAVEL)
        goal, close, speed = track(desired, PRESS_STEP), True, PRESS_SPEED
        if getattr(task, "fire_time", None) is not None and measured["travel"] >= PRESS_TRAVEL - 0.0015:
            planner.b5_hold = planner.last_goal.copy()
            transition("hold")
        elif not pen.grasped and now - planner.b5_phase_time > 0.3:
            transition("reach")
    elif phase == "hold":
        # Keep the measured travel while the hold runs; lift off once the dose is delivered.
        desired = pen_goal(-PRESS_TRAVEL)
        goal, close, speed = track(desired, 0.002), True, 0.05
        if getattr(task, "delivered_time", None) is not None:
            planner.b5_done = state.ee_position.copy()
            transition("withdraw")
    elif phase == "withdraw":
        goal = planner.b5_done + [0.0, 0.0, 0.06]
        close, speed = True, 0.20
    return planner.finish(state, np.asarray(goal, dtype=float), close, speed)


# --- authoring -----------------------------------------------------------------------------------


def author(backend):
    """The tip's prismatic joint and spring, then every visual of the bedside."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = backend.stage
    pen_path, tip_path = f"/World/Objects/{PEN}", f"/World/Objects/{TIP}"
    if not stage.GetPrimAtPath(pen_path).IsValid() or not stage.GetPrimAtPath(tip_path).IsValid():
        raise ValueError("Author the injector's two rigid bodies before their joint")
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/epi_tip")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(pen_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tip_path)])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, TIP_REST))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateLowerLimitAttr(0.0)  # at rest against the stop; pushed in = toward +z of the barrel
    joint.CreateUpperLimitAttr(TIP_TRAVEL)
    joint.CreateCollisionEnabledAttr(False)
    spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    spring.CreateTypeAttr("force")
    spring.CreateStiffnessAttr(SPRING["stiffness"])
    spring.CreateDampingAttr(SPRING["damping"])
    spring.CreateTargetPositionAttr(-SPRING["preload"])  # rests against the stop with a preload
    spring.CreateTargetVelocityAttr(0.0)
    spring.CreateMaxForceAttr(SPRING["max_force"])
    # The downloaded barrel and tip are the bodies' visuals, not the catalog boxes.
    for path in (pen_path, tip_path):
        for name in ("Shell", "Band"):
            prim = stage.GetPrimAtPath(f"{path}/{name}")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
    if hasattr(backend, "world"):  # the Isaac scene; plain-USD unit stages get the joint only
        _dress(backend)


def _reference(stage, path, name, translate=(0.0, 0.0, 0.0), rotations=(), scale=None):
    """Reference a converted model at `path`; `rotations` are (axis, degrees), applied last to first."""
    from pxr import Gf, UsdGeom

    node = UsdGeom.Xform.Define(stage, path)
    node.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translate]))
    for axis, degrees in rotations:
        {"x": node.AddRotateXOp, "y": node.AddRotateYOp, "z": node.AddRotateZOp}[axis]().Set(float(degrees))
    if scale is not None:
        node.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))
    node.GetPrim().GetReferences().AddReference(str(MODELS / name / f"{name}.usd"))
    return node


def _installed(name):
    return (MODELS / name / f"{name}.usd").exists()


def _dress(backend):
    from pxr import Gf, UsdGeom

    from roboclock.envs.hospital.bedside.iv_bag import _face_dial, _materials, _robot_base
    from roboclock.envs.hospital.bedside.pump_panel import _tube
    from roboclock.envs.hospital.bedside.ventilator import _bed
    from roboclock.envs.hospital.rig import dial_target

    look, layout, stage = backend.appearance, backend.layout, backend.stage
    info = layout.info["autoinjector"]
    labels = backend.config["visuals"].get("labels", True)
    for fixture in layout.fixtures:
        prim = stage.GetPrimAtPath(f"/World/Fixtures/{fixture.name}_Visual")
        if not fixture.visible and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    mats = _materials(look)
    _robot_base(backend, mats)
    _bed(backend, info, mats)
    head = np.asarray(dial_target(backend, info["camera"]["position"]), dtype=float)
    top = info["mattress_top"]
    ox, oy, oz = info["patient_origin"]

    def on_patient(x, y, z):
        """World position of a point of the patient model (turned head toward +y)."""
        return np.array([ox - x, oy - y, oz + z])

    # --- patient, pillow, blanket over the shins and feet ----------------------------------------
    linen = look.material("b5_linen", color=(0.90, 0.92, 0.94), roughness=0.85)
    blanket = look.material("b5_blanket", color=(0.50, 0.70, 0.86), roughness=0.9)
    if _installed("patient_supine"):
        from pxr import UsdShade

        _reference(stage, BASE + "/Patient", "patient_supine", (ox, oy, oz), [("z", 180.0)])
        # The model's arm, leg and foot skin textures render black in this RTX mode: flat skin.
        skin = look.material("b5_skin", color=(0.78, 0.58, 0.47), roughness=0.6)
        for part in ("part_007", "part_008", "part_009"):
            prim = stage.GetPrimAtPath(f"{BASE}/Patient/{part}")
            if prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    skin.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )
    else:
        skin = look.material("b5_skin", color=(0.84, 0.64, 0.52), roughness=0.55)
        look.rounded(BASE + "/Legs", on_patient(0.02, 0.45, 0.10), (0.36, 0.90, 0.20), skin, 0.08)
        look.rounded(BASE + "/Trunk", on_patient(0.0, -0.40, 0.12), (0.40, 0.80, 0.24), skin, 0.08)
    look.rounded(BASE + "/Pillow", on_patient(0.0, -0.80, 0.03), (0.50, 0.30, 0.06), linen, 0.025)
    look.rounded(BASE + "/Blanket", on_patient(0.02, 0.72, 0.125), (0.66, 0.42, 0.25), blanket, 0.09)

    # --- injection pad strapped on the thigh -------------------------------------------------------
    pad, size = np.asarray(info["pad"]["center"]), info["pad"]["size"]
    pad_blue = look.material("b5_pad", color=(0.30, 0.62, 0.92), roughness=0.75)
    strap = look.material("b5_strap", color=(0.16, 0.20, 0.30), roughness=0.9)
    look.rounded(BASE + "/Pad", pad - [0.0, 0.0, size[2] / 2], size, pad_blue, 0.005)
    for k, dy in enumerate((-0.045, 0.045)):
        look.rounded(f"{BASE}/PadStrap_{k}", pad + [0.035, dy, -0.012], (0.20, 0.018, 0.016), strap, 0.006)
    if labels:
        ring = look.material("b5_pad_ring", color=(0.97, 0.98, 1.0), roughness=0.6)
        look.cylinder(BASE + "/PadMark", pad + [0.0, 0.0, 0.0006], 0.022, 0.0012, ring)
        look.cylinder(BASE + "/PadMarkCore", pad + [0.0, 0.0, 0.0010], 0.015, 0.0014, pad_blue)

    # --- over-bed table, kit tray and the injector's holder ----------------------------------------
    tx, ty = info["table"]["center"]
    table_top = info["table"]["top"]
    if _installed("overbed_table"):
        from roboclock.envs.hospital.props import model_info

        native = model_info("overbed_table")["top_z"]
        _reference(stage, BASE + "/Table", "overbed_table", (tx, ty, 0.0), scale=(1.0, 1.0, table_top / native))
    laminate = look.textured("b5_table_top", "plastic_white", color=(0.93, 0.92, 0.88), roughness=0.4)
    look.rounded(BASE + "/TableTop", (tx, ty, table_top - 0.014), (0.985, 0.396, 0.030), laminate, 0.006)
    tray_c, tray_s = np.asarray(info["tray"]["center"]), info["tray"]["size"]
    red = look.textured("b5_tray", "plastic_white", color=(0.80, 0.10, 0.12), roughness=0.4)
    white = look.textured("b5_holder", "plastic_white", color=(0.96, 0.96, 0.95), roughness=0.4)
    look.rounded(BASE + "/Tray", tray_c + [0.0, 0.0, tray_s[2] / 2], tray_s, red, 0.004)
    for k, (dx, dy, sx, sy) in enumerate(
        (
            (0.0, -tray_s[1] / 2 + 0.004, tray_s[0], 0.008),
            (0.0, tray_s[1] / 2 - 0.004, tray_s[0], 0.008),
            (-tray_s[0] / 2 + 0.004, 0.0, 0.008, tray_s[1]),
            (tray_s[0] / 2 - 0.004, 0.0, 0.008, tray_s[1]),
        )
    ):
        look.rounded(f"{BASE}/TrayRim_{k}", tray_c + [dx, dy, tray_s[2] + 0.006], (sx, sy, 0.012), red, 0.003)
    hx, hy, floor = info["holder"]
    for fixture in layout.fixtures:
        if fixture.name.startswith("well_"):
            look.rounded(f"{BASE}/Holder_{fixture.name}", fixture.position, fixture.size, white, 0.002)
    if _installed("epi_safety_cap"):
        cx, cy, cz = info["cap"]
        _reference(stage, BASE + "/SafetyCap", "epi_safety_cap", (cx, cy, cz))
    # --- the injector's downloaded barrel and tip (children of the rigid bodies) -------------------
    if _installed("epi_autoinjector"):
        _reference(stage, f"/World/Objects/{PEN}/Model", "epi_autoinjector")
        _reference(stage, f"/World/Objects/{TIP}/Model", "epi_autoinjector_tip")
    else:
        yellow = look.material("b5_pen", color=(0.96, 0.86, 0.20), roughness=0.4)
        orange = look.material("b5_tip", color=(0.98, 0.42, 0.05), roughness=0.4)
        look.rounded(f"/World/Objects/{PEN}/Body", (0, 0, 0), (0.0264, 0.0164, 2 * PEN_HALF), yellow, 0.006)
        look.rounded(f"/World/Objects/{TIP}/Body", (0, 0, 0), (0.0205, 0.0128, TIP_LENGTH), orange, 0.004)

    # --- infusion stand and line to the far forearm ------------------------------------------------
    ix, iy = info["iv_stand"]
    if _installed("iv_stand"):
        from roboclock.envs.hospital.props import model_info

        _reference(stage, BASE + "/IVStand", "iv_stand", (ix, iy, 0.0))
        bag = np.asarray(model_info("iv_stand").get("bag_bottom_m", (0.07, 0.0, 1.70))) + [ix, iy, 0.0]
        line = look.material("b5_iv_line", color=(0.86, 0.92, 0.96), roughness=0.3)
        arm = on_patient(-0.265, 0.02, 0.105)
        _tube(look, BASE + "/IVLine", [bag, bag + [0.0, 0.0, -0.25], [bag[0] - 0.04, bag[1] - 0.15, top + 0.45],
                                       arm + [0.10, 0.20, 0.20], arm + [0.02, 0.05, 0.03], arm], line)
        tape = look.material("b5_iv_tape", color=(0.97, 0.97, 0.95), roughness=0.8)
        look.rounded(BASE + "/IVTape", arm + [0.0, 0.0, 0.004], (0.035, 0.045, 0.004), tape, 0.0015)

    # --- bedside monitor on a pole beyond the bed, turned and leaned toward the head camera --------
    mx, my, mz = info["monitor"]
    yaw = math.degrees(math.atan2(head[1] - my, head[0] - mx))
    frame = look.tint("b5_monitor_stand", (0.72, 0.74, 0.77), 0.35)
    look.cylinder(BASE + "/MonitorPole", (mx + 0.10, my, (mz - 0.05) / 2), 0.016, mz - 0.05, frame)
    look.cylinder(BASE + "/MonitorBase", (mx + 0.10, my, 0.012), 0.18, 0.024, frame)
    monitor = UsdGeom.Xform.Define(stage, BASE + "/Monitor")
    monitor.AddTranslateOp().Set(Gf.Vec3d(float(mx), float(my), float(mz)))
    monitor.AddRotateZOp().Set(float(yaw))
    monitor.AddRotateYOp().Set(-MONITOR_TILT)
    monitor.AddScaleOp().Set(Gf.Vec3f(MONITOR_SCALE, MONITOR_SCALE, MONITOR_SCALE))
    mpath = str(monitor.GetPath())
    if _installed("patient_monitor_head"):
        _reference(stage, mpath + "/Head", "patient_monitor_head")
    dark = look.material("b5_screen_dark", color=(0.01, 0.012, 0.015), roughness=0.2)
    w, h = SCREEN_SIZE
    look.rounded(mpath + "/ScreenGlass", (0.0015, 0.0, 0.0), (0.002, w + 0.012, h + 0.012), dark, 0.001)
    cards, lamp_scale = {}, None
    if labels:
        from roboclock.envs.hospital.bedside.iv_bag import _quad

        # The header (two states) and the numbers (one card per SYSTOLIC_STEP) are separate
        # cards, so the LOW BP banner stays sharp while a number card is changing.
        hh = h * HEADER_SHARE
        for alarm in (False, True):
            mesh = _quad(stage, f"{mpath}/Header_{int(alarm)}", _header_texture(alarm), (0.0032, 0.0, (h - hh) / 2),
                         (w, hh), rotations=[("z", -90.0)], emissive=True)
            scale = UsdGeom.Xformable(mesh).AddScaleOp()
            scale.Set(Gf.Vec3f(1e-4, 1e-4, 1e-4))
            cards[alarm] = scale
        for value in range(int(SYSTOLIC_FLOOR), int(SYSTOLIC_RANGE[1]) + 1, SYSTOLIC_STEP):
            mesh = _quad(stage, f"{mpath}/Screen_{value}", _screen_texture(value), (0.0032, 0.0, -hh / 2), (w, h - hh),
                         rotations=[("z", -90.0)], emissive=True)
            scale = UsdGeom.Xformable(mesh).AddScaleOp()
            scale.Set(Gf.Vec3f(1e-4, 1e-4, 1e-4))
            cards[value] = scale
        shell = look.material("b5_lamp_base", color=(0.90, 0.92, 0.93), roughness=0.5)
        lamp_at = (-0.02, 0.0, 0.114 + 0.012)
        look.rounded(mpath + "/LampBase", (lamp_at[0], 0.0, 0.114 + 0.003), (0.04, 0.10, 0.006), shell, 0.002)
        look.sphere(mpath + "/LampOff", lamp_at, 0.014, look.material("b5_lamp_off", color=(0.55, 0.50, 0.50), roughness=0.3))
        lamp = look.sphere(mpath + "/LampOn", lamp_at, 0.0155,
                           look.material("b5_lamp_red", color=(1.0, 0.03, 0.02), roughness=0.3, emissive=True))
        lamp_scale = lamp.AddScaleOp()
        lamp_scale.Set(Gf.Vec3f(1e-4, 1e-4, 1e-4))

    # --- hives on the near forearm: shown from the alarm on (switched by scale) --------------------
    rash = look.material("b5_hives", color=(0.85, 0.22, 0.24), roughness=0.7)
    hives = UsdGeom.Xform.Define(stage, BASE + "/Hives")
    hives_scale = hives.AddScaleOp()
    for k, (x, y, z, radius) in enumerate(HIVES):
        look.cylinder(f"{BASE}/Hives/Spot_{k}", on_patient(x, y, z), radius, 0.002, rash)
    hives_scale.Set(Gf.Vec3f(1e-4, 1e-4, 1e-4))  # shrunk to nothing until the alarm

    _face_dial(backend, head)
    x, y, z = next(i for i in layout.indicators if i.name == "indicator").position
    look.rounded(BASE + "/DialFoot", (x, y, table_top + 0.004), (0.085, 0.085, 0.008), mats["bracket"], 0.003)
    backend._b5_visual = {"task": backend.task, "cards": cards, "lamp": lamp_scale, "hives": hives_scale, "state": None}
    _animate(backend)


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


def _cached(key):
    import hashlib
    import os

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"b5bp_{hashlib.sha256(repr(key).encode()).hexdigest()[:20]}.png"


def _header_texture(alarm, size=(600, 114)):
    """The monitor's top strip: parameter names when normal, the red LOW BP banner in alarm."""
    from PIL import Image, ImageDraw

    path = _cached(("header", alarm, size, 1))
    if path.exists():
        return path
    w, h = size
    image = Image.new("RGB", size, (225, 20, 15) if alarm else (4, 6, 10))
    draw = ImageDraw.Draw(image)
    if alarm:
        draw.text((w / 2, h / 2), "LOW BP", fill=(255, 255, 255), font=_font(int(h * 0.80)), anchor="mm")
    else:
        draw.text((w * 0.03, h / 2), "BP mmHg", fill=(40, 220, 255), font=_font(int(h * 0.48)), anchor="lm")
        draw.text((w * 0.97, h / 2), "SpO2 97", fill=(240, 220, 60), font=_font(int(h * 0.48)), anchor="rm")
    image.save(path)
    return path


def _screen_texture(systolic, size=(600, 294)):
    """The monitor's numbers for one systolic reading: cyan when normal, white on red in alarm."""
    from PIL import Image, ImageDraw

    path = _cached(("numbers", systolic, size, 2))
    if path.exists():
        return path
    w, h = size
    alarm = systolic < ALARM_SYSTOLIC
    diastolic = int(round(systolic * DIASTOLIC_RATIO))
    rate = heart_rate(systolic)
    image = Image.new("RGB", size, (4, 6, 10))
    draw = ImageDraw.Draw(image)
    cyan, green, white, red = (40, 220, 255), (60, 230, 90), (255, 255, 255), (225, 20, 15)
    if alarm:
        draw.rectangle([8, 6, w - 8, int(h * 0.62)], fill=red)
    draw.text((w * 0.5, h * 0.32), f"{systolic}/{diastolic}", fill=white if alarm else cyan,
              font=_font(int(h * 0.50)), anchor="mm")
    draw.text((w * 0.5, h * 0.82), f"HR {rate}", fill=red if alarm else green, font=_font(int(h * 0.27)), anchor="mm")
    image.save(path)
    return path


def _animate(backend):
    """Visual consequences of the task state: the monitor's card, the alarm lamp, the hives."""
    from pxr import Gf

    visual = getattr(backend, "_b5_visual", None)
    task = getattr(backend, "task", None)
    if visual is None or task is None or visual["task"] is not task or getattr(backend, "no_render", False):
        return
    value, lamp = monitor_state(task)
    hives = task.event_time is not None
    key = (value, lamp, hives)
    if key == visual["state"]:
        return
    previous = visual["state"] or (None, None, None)
    visual["state"] = key
    if value != previous[0]:
        header = value < ALARM_SYSTOLIC
        for shown, scale in visual["cards"].items():
            s = 1.0 if shown == (header if isinstance(shown, bool) else value) else 1e-4
            scale.Set(Gf.Vec3f(s, s, s))
    if lamp != previous[1] and visual["lamp"] is not None:
        s = 1.0 if lamp else 1e-4
        visual["lamp"].Set(Gf.Vec3f(s, s, s))
    if hives != previous[2]:
        visual["hives"].Set(Gf.Vec3f(1.0, 1.0, 1.0) if hives else Gf.Vec3f(1e-4, 1e-4, 1e-4))


def _camera(backend):
    """Frame the robot, the patient's legs, the table, the injector and the monitor, once per reset."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["autoinjector"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1))
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    if "focal_mm" in view:
        camera.set_focal_length(view["focal_mm"] / 1000)
    for _ in range(3):
        backend.world.render()
