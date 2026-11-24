"""B6 "massive limb haemorrhage": tighten the windlass tourniquet on the bleeding thigh.

Scene (trauma bay, scene-02 left bay). A bleeding-control training manikin lies supine on the
bed, draped except for its injured leg (the one on the robot's side). A windlass tourniquet is
already strapped high on that thigh: a 38 mm band, a stabilization plate on top of the thigh, the
windlass rod over it and the retaining clip at the plate's proximal end. The wound is on the
outer thigh below the band, over an absorbent pad. The trainer's display stands beyond the bed,
facing the robot's head camera, with a blood-loss column beside its screen.

Event. At a seeded time the wound starts to bleed: a streak runs from the wound to the pad, a
pool grows on the pad (drawn from the task's lost volume) and the display turns red.

Mechanism. The rod is the only moving body: a rigid bar on a revolute joint about the vertical
through the plate, turned only by finger contact. A sprung, damped joint drive stands for the
twisted strap: it unwinds a released rod back to its start. The retaining clip is a latch on that
drive: `measure` reads the rod's angle and, once it is within the clip's capture range of the
half-turn (tasks/bedside/tourniquet.py::clip_engaged), retargets the drive to the clip so the
released rod stays there; turned back out of the range it unwinds again. No command, phase or
timer sets the angle. The clip's hook is drawn, not simulated.

Assets. Downloaded (CC BY 4.0, credits in assets/models/credits.json): patient_manikin and
hospital_bed_flat (B4's), trauma_kit_pouch, bandage_roll, gauze_pack, trauma_forceps
(scripts/hospital/import_b6_assets.py). Authored here because they are the measured mechanism,
a process-driven cue or not available under a free licence: the tourniquet, the wound, streak and
pool, the pad, the drapes and the display.

Seeded placement (tasks.B6.placement in configs/benchmark.yaml, drawn from the layout rng, so the
same at every level of a seed): the band with its windlass and the wound slide along the thigh,
the rod starts a few degrees off the limb axis, the clip sits a few degrees off it and the leg
(with the patient) is yawed a few degrees about the windlass. The turn from the rod's start to the
clip is therefore 180 + clip - start degrees. Everything is in layout.info["tourniquet"]
(placement, start_yaw_rad, clip_yaw_rad, turn_required_rad) and the oracle reads it from there.

All heights follow the configured arm-0 mount (robot.rig.tasks.B6): the mattress top is
MATTRESS_BELOW_MOUNT below it.
"""

import math

import numpy as np

from roboclock.envs.hospital.tasks.bedside.tourniquet import (
    INSTRUCTION,
    OVERTURN,
    ROD,
    TURN_ANGLE,
    clip_engaged,
    pool_radius,
)

NAME = ROD
ROOT = f"/World/Objects/{NAME}"
JOINT_PATH = "/World/Joints/WindlassBearing"
BASE = "/World/Tourniquet"

# --- the tourniquet (metres) ----------------------------------------------------------------------
ROD_LENGTH, ROD_WIDTH, ROD_HEIGHT = 0.120, 0.014, 0.012
ROD_ABOVE_PLATE = 0.014  # rod axis above the stabilization plate: room for the fingertips
PLATE = (0.052, 0.150, 0.006)  # stabilization plate on the thigh (x across the limb, y along it)
BAND_WIDTH = 0.038
START_YAW = -math.pi / 2  # nominal: the rod lies along the limb, its red tip toward the knee (-y)
# Joint drive (USD angular units: degrees). Unwinding strap: a spring to the start whose torque
# saturates at max_force; viscous enough that a released rod swings back in about a second. The
# retaining clip is the same drive retargeted to the clip's angle. The joint angle is the rod's
# world yaw (PhysX limits a revolute joint to +-180 degrees, so the turn runs from about -90 to
# about +90 degrees of joint angle, not from 0 to 180); the targets and limits are the episode's
# (`drive_values`, `joint_limits`).
START_DEG = math.degrees(START_YAW)
STRAP = {"stiffness": 0.0006, "damping": 0.0003, "max_force": 0.030, "target": START_DEG}
CLIP = {"stiffness": 0.004, "damping": 0.0004, "max_force": 0.080, "target": START_DEG + math.degrees(TURN_ANGLE)}
LOWER_LIMIT_DEG = START_DEG - 3.0
UPPER_LIMIT_DEG = START_DEG + math.degrees(TURN_ANGLE + OVERTURN) + 5.0
# Seeded placement (configs/benchmark.yaml tasks.B6.placement; zero ranges when the block is absent).
PLACEMENT_KEYS = ("band_shift_m", "rod_start_deg", "clip_deg", "leg_yaw_deg")
# How the thigh changes along its length (measured on the mesh between y = -0.36 and -0.30):
# per metre toward the hip its top rises 0.40 m, its centreline moves 0.13 m toward the midline.
THIGH_TOP_SLOPE, THIGH_X_SLOPE, THIGH_AXIS_SLOPE = 0.40, 0.13, 0.20

# --- manikin frame (patient_manikin.usd: supine, head toward +y, back on z = 0) -----------------
# The injured leg is the one at -x (the robot's side). Measured on the mesh: at y = -0.33 the
# thigh spans x -0.153..-0.047 and its top is 0.190 above the mattress.
THIGH = (-0.100, -0.330, 0.190)  # top of the thigh under the stabilization plate
THIGH_AXIS_Z = 0.127  # height of the thigh's axis there
THIGH_RADIUS = 0.064  # the band is drawn as a ring of this radius about the axis
WOUND = (-0.150, -0.435, 0.120)  # on the outer thigh, 0.10 m below the band toward the knee
POOL = (-0.215, -0.435)  # where the blood reaches the pad
PAD = {"center": (-0.150, -0.50), "size": (0.36, 0.56, 0.008)}
MATTRESS_BELOW_MOUNT = 0.52

# --- robot frame before the seed's common shift; arm 0's mount (the rig's right arm) is at the origin
PIVOT_XY = (0.40, 0.05)
BED_DX = 0.02  # bed centre from the manikin's centreline
BED_SCALE_Y = 0.50  # the 1.49 m wide model slimmed to a 0.74 m trauma bed
DISPLAY_XY = (0.99, 0.10)
DISPLAY_BELOW_MOUNT = 0.10
INDICATOR_XY = (0.16, 0.42)
KIT_TRAY = (0.235, -0.36)  # instrument tray on the mattress beside the injured shin
TUCK = (0.20, -0.04, 0.02)  # from the arm-0 mount
CAMERA = {"position": [0.55, -1.25, 1.90], "look_at": [0.50, 0.05, 0.85], "focal_mm": 20.0}
# The ward turns 180 degrees about the robot and shifts so its bed bay lands on this bed (B4's
# pose, moved with the manikin); the ward's own bed and the ultrasound cart are hidden.
WARD_POSE = {"usd": "assets/hospital/ward_02_preview_left.usdc", "yaw": 180.0}
WARD_FROM_MANIKIN = (1.34, -1.755)  # ward translate minus the manikin origin (B4: 1.74-0.40, -2.22+0.465)

# --- oracle ---------------------------------------------------------------------------------------
HAND_YAW = START_YAW  # nominal; the hand's yaw is the rod's (the fingers separate across the rod)
READY_WIDTH = 0.0150  # finger joint position (half opening); the rod's side is at 0.007
SQUEEZE_WIDTH = 0.003
GRIP_HEIGHT = 0.003  # grasp centre above the rod's axis (fingertips reach 12.6 mm below it)
HOVER = 0.035
REACH_SPEED, DESCEND_SPEED = 0.60, 0.30
TURN_RATE = 2.5  # commanded yaw rate (rad/s); the wrist itself tracks at about 1.1 rad/s
TURN_LEAD = 0.16  # the commanded yaw never leads the measured rod by more than this
TURN_PAST = math.radians(2.0)  # the commanded turn ends this far past the half-turn
RELEASE_BEFORE = math.radians(4.0)  # the fingers open once the rod is this close to the clip angle
RELEASE_ANGLE = None  # diagnostics only: open the fingers at this turn instead


def mount_height(config, task_id="B6"):
    from roboclock.envs.lab.robot import mount_positions

    return float(mount_positions(config, task_id)[0][2])


def sample_placement(config, rng, task_id="B6"):
    """The episode's seeded placement, drawn from the layout rng (the same at every level of a
    seed): the band's shift along the thigh (m, + toward the hip; the wound moves with it), the
    rod's start angle off the limb axis, the clip's angle off the limb axis and the leg's yaw
    (degrees, counterclockwise seen from above)."""
    ranges = config["tasks"].get(task_id, {}).get("placement") or {}
    values = {}
    for key in PLACEMENT_KEYS:
        lo, hi = ranges.get(key, (0.0, 0.0))
        values[key] = float(rng.uniform(float(lo), float(hi))) if hi > lo else float(lo)
    return values


def turn_angles(placement):
    """(start yaw, clip yaw, turn required), rad, world frame, counterclockwise from above."""
    leg = math.radians(placement.get("leg_yaw_deg", 0.0))
    start = START_YAW + leg + math.radians(placement.get("rod_start_deg", 0.0))
    clip = START_YAW + TURN_ANGLE + leg + math.radians(placement.get("clip_deg", 0.0))
    return start, clip, clip - start


def geometry(config, shift=(0.0, 0.0), task_id="B6", placement=None):
    """Positions of the apparatus (pure; used by the layout, the tests and the oracle). World
    positions, and under "limb" the same patient parts in the limb frame: origin at the pivot's
    xy, axes turned leg_yaw about the vertical (the patient group is authored in that frame)."""
    placement = placement or dict.fromkeys(PLACEMENT_KEYS, 0.0)
    s = float(placement.get("band_shift_m", 0.0))
    leg = math.radians(placement.get("leg_yaw_deg", 0.0))
    shift = np.array([float(shift[0]), float(shift[1])])
    mattress = mount_height(config, task_id) - MATTRESS_BELOW_MOUNT
    thigh = np.array([THIGH[0] + THIGH_X_SLOPE * s, THIGH[1] + s, THIGH[2] + THIGH_TOP_SLOPE * s])
    # The patient lies where it lies; the band (and so the pivot) moves along the thigh.
    pivot_xy = np.asarray(PIVOT_XY) + shift + [THIGH_X_SLOPE * s, s]
    plate_top = mattress + thigh[2] + 0.004 + PLATE[2]  # 4 mm of band under the plate
    origin = np.array([-thigh[0], -thigh[1], mattress])  # manikin origin, limb frame
    limb = {
        "manikin_origin": origin,
        "thigh_axis": origin + [thigh[0], thigh[1], THIGH_AXIS_Z + THIGH_AXIS_SLOPE * s],
        "wound": origin + [WOUND[0], WOUND[1] + s, WOUND[2]],
        "pool": origin + [POOL[0], POOL[1] + s, PAD["size"][2] + 0.0015],
        "pad": origin + [*PAD["center"], PAD["size"][2] / 2],
    }
    c, sn = math.cos(leg), math.sin(leg)

    def world(point):
        x, y, z = point
        return np.array([pivot_xy[0] + c * x - sn * y, pivot_xy[1] + sn * x + c * y, z])

    nominal = np.asarray(PIVOT_XY) + shift - [THIGH[0], THIGH[1]]  # the unshifted manikin origin
    return {
        "mattress": mattress,
        "origin": world(origin),
        "pivot": np.array([*pivot_xy, plate_top + ROD_ABOVE_PLATE]),
        "plate_top": plate_top,
        "thigh_axis": world(limb["thigh_axis"]),
        "wound": world(limb["wound"]),
        "pool": world(limb["pool"]),
        "pad": world(limb["pad"]),
        "limb": limb,
        "bed": np.array([nominal[0] + BED_DX, nominal[1] - 0.035]),
        "ward_origin": nominal,
        "display": np.array([*(np.asarray(DISPLAY_XY) + shift), mount_height(config, task_id) - DISPLAY_BELOW_MOUNT]),
    }


def build_layout(builder):
    from roboclock.envs.hospital.bedside.ward import ULTRASOUND_CART, furniture

    b = builder
    task_id = getattr(b.task, "task_id", "B6")
    placement = sample_placement(b.config, b.rng, task_id)
    start_yaw, clip_yaw, turn = turn_angles(placement)
    g = geometry(b.config, b.shift, task_id, placement)
    mount_z = mount_height(b.config, task_id)
    mattress = g["mattress"]
    b.z = mattress
    pivot = g["pivot"]
    unshift = pivot[:2] - b.shift
    # Colliders (drawn by _dress, axis-aligned: the leg's yaw of a few degrees is not applied to
    # them): the stabilization plate, the thigh under it and the leg.
    b.fixture("tq_plate", unshift, PLATE, "cap_white", z=g["plate_top"] - PLATE[2], visible=False)
    b.fixture(
        "patient_leg", (unshift[0], unshift[1] - 0.25), (0.12, 0.80, g["plate_top"] - mattress - 0.030), "skin",
        z=mattress, visible=False,
    )
    half = b.catalog[NAME]["size"][2] / 2
    b.item(NAME, NAME, unshift, pivot[2] - half - 0.002, color=(0.08, 0.08, 0.09), yaw=start_yaw)
    b.indicator("indicator", INDICATOR_XY, "BLEEDING")
    tuck = np.array([TUCK[0], TUCK[1], mount_z + TUCK[2]])  # robot frame: the robot never shifts
    hide = ULTRASOUND_CART + furniture("bed") + ["iMac_5k_Monitor054"]
    ward = g["ward_origin"]
    b.info.update(
        {
            "bedside_task": "tourniquet",
            "instruction": INSTRUCTION,
            "initial_ee_position": tuck.tolist(),
            "initial_ee_yaw": float(start_yaw),
            "initial_gripper_width": READY_WIDTH,
            "ward_pose": {
                **WARD_POSE,
                "translate": [ward[0] + WARD_FROM_MANIKIN[0], ward[1] + WARD_FROM_MANIKIN[1], 0.0],
                "hide": hide,
            },
            "hide_bench": True,
            # The mattress is the support surface.
            "work_surface": {
                "center": [*g["bed"].tolist(), mattress - 0.05],
                "size": [0.74, 2.0, 0.10],
            },
            "tourniquet": {
                "rod": NAME,
                "placement": placement,
                "start_yaw_rad": float(start_yaw),
                "clip_yaw_rad": float(clip_yaw),
                "turn_required_rad": float(turn),
                "leg_yaw_deg": placement["leg_yaw_deg"],
                "pivot": pivot.tolist(),
                "plate_top": float(g["plate_top"]),
                "mattress_top": float(mattress),
                "manikin_origin": g["origin"].tolist(),
                "thigh_axis": g["thigh_axis"].tolist(),
                "wound": g["wound"].tolist(),
                "pool": g["pool"].tolist(),
                "pad": {"center": g["pad"].tolist(), "size": list(PAD["size"])},
                "limb": {key: value.tolist() for key, value in g["limb"].items()},
                "bed": {"center": g["bed"].tolist(), "scale_y": BED_SCALE_Y},
                "display": g["display"].tolist(),
                "kit_tray": [*b.xy(*KIT_TRAY).tolist(), float(mattress)],
                "camera": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in CAMERA.items()},
                "assumptions": [
                    "the band is already placed and strapped; only the windlass is turned",
                    "one windlass turn of about 180 degrees (seeded): the clip captures the rod within 12 degrees of its angle",
                    "strap tension = sprung, damped joint drive; clip = latch on that drive engaged by the measured angle",
                    "bleeding, streak and pool are process models of the measured angle; no physiology",
                ],
            },
        }
    )


# --- measurement ---------------------------------------------------------------------------------


def rod_angle(quaternion, previous=None, start_yaw=START_YAW):
    """(angle, tilt): the rod's turn about the vertical from its start (world yaw `start_yaw`),
    counterclockwise seen from above (rad, unwrapped against `previous`), and the tilt of its
    axis off the vertical."""
    q = np.asarray(quaternion, dtype=float)
    if not (np.all(np.isfinite(q)) and abs(np.linalg.norm(q) - 1.0) < 0.01):
        return float("nan"), float("inf")
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    angle = yaw - start_yaw
    reference = TURN_ANGLE / 2 if previous is None or not math.isfinite(previous) else previous
    angle += 2.0 * math.pi * round((reference - angle) / (2.0 * math.pi))
    tilt = math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))
    return angle, tilt


def drive_values(info, mode):
    """The joint drive of the episode: the unwinding strap (target: the rod's start) or the
    retaining clip (target: the clip's angle), in the joint's own angle (world yaw, degrees)."""
    values = dict(CLIP if mode == "clip" else STRAP)
    key = "clip_yaw_rad" if mode == "clip" else "start_yaw_rad"
    fallback = START_YAW + TURN_ANGLE if mode == "clip" else START_YAW
    values["target"] = math.degrees(float(info.get(key, fallback)))
    return values


def joint_limits(info):
    """(lower, upper) joint limits in degrees: just below the start to past the over-turn."""
    start = math.degrees(float(info.get("start_yaw_rad", START_YAW)))
    clip = math.degrees(float(info.get("clip_yaw_rad", START_YAW + TURN_ANGLE)))
    return start - 3.0, clip + math.degrees(OVERTURN) + 5.0


def _set_drive(backend, mode):
    """Retarget the joint drive: the unwinding strap or the retaining clip."""
    from pxr import UsdPhysics

    values = drive_values(backend.layout.info["tourniquet"], mode)
    drive = UsdPhysics.DriveAPI(backend.stage.GetPrimAtPath(JOINT_PATH), "angular")
    drive.GetStiffnessAttr().Set(values["stiffness"])
    drive.GetDampingAttr().Set(values["damping"])
    drive.GetMaxForceAttr().Set(values["max_force"])
    drive.GetTargetPositionAttr().Set(values["target"])


def measure(backend, state):
    """Privileged PhysX measurements of the rod; engages the clip latch; poses the visual cues."""
    task = getattr(backend, "task", None)
    if getattr(backend, "_tq_for", None) is not task:
        backend._tq_for = task
        backend._tq_angle = 0.0
        backend._tq_mode = "strap"
        _set_drive(backend, "strap")
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_tq_camera_for", None) is not task:
        backend._tq_camera_for = task
        _camera(backend)
    obj = state.objects[NAME]
    info = backend.layout.info["tourniquet"]
    turn = float(info.get("turn_required_rad", TURN_ANGLE))
    angle, tilt = rod_angle(obj.orientation, backend._tq_angle, float(info.get("start_yaw_rad", START_YAW)))
    if math.isfinite(angle):
        backend._tq_angle = angle
    mode = "clip" if clip_engaged(angle, turn) else "strap"
    if mode != backend._tq_mode:
        backend._tq_mode = mode
        _set_drive(backend, mode)
    speed = backend.objects[NAME].get_angular_velocity()
    pivot = np.asarray(info["pivot"])
    measured = {
        "angle_rad": float(angle),
        "speed_rad_s": float(speed[2]),
        "axis_error_rad": tilt,
        "pivot_error_m": float(np.linalg.norm(obj.position - pivot)),
        "contact": bool(obj.robot_contact or obj.grasped),
        "grasped": bool(obj.grasped),
        "clip_engaged": mode == "clip",
    }
    _animate(backend)
    return {"tourniquet": measured}


# --- privileged diagnostic oracle ----------------------------------------------------------------


def command(planner, task, state):
    """Tucked until the bleeding starts; then reach over the rod, pinch it across its width, roll
    the wrist through the episode's turn (the commanded yaw follows the measured rod), let go in
    the clip. The rod's start yaw and the turn to the clip are read from the layout."""
    info = task.layout.info
    pivot = np.asarray(info["tourniquet"]["pivot"], dtype=float)
    hand_yaw = float(info["tourniquet"].get("start_yaw_rad", START_YAW))
    turn_required = float(info["tourniquet"].get("turn_required_rad", TURN_ANGLE))
    release_at = turn_required - RELEASE_BEFORE if RELEASE_ANGLE is None else RELEASE_ANGLE
    grip = pivot + [0.0, 0.0, GRIP_HEIGHT]
    hover = grip + [0.0, 0.0, HOVER]
    ee = np.asarray(state.ee_position, dtype=float)
    obj = state.objects[NAME]
    controls = state.controls.get("tourniquet") or {}
    angle = float(controls.get("angle_rad", 0.0))
    if not math.isfinite(angle):
        angle = 0.0
    planner.open_width, planner.close_width = READY_WIDTH, SQUEEZE_WIDTH
    planner.gentle = False
    if getattr(planner, "_tq_task", None) is not task:
        planner._tq_task = task
        planner.phase = "wait"
        planner.turn = 0.0
    if task.event_time is None:
        planner.phase = "wait"
        planner.yaw = hand_yaw
        return planner.finish(state, np.asarray(info["initial_ee_position"]), False, 0.12)
    if planner.phase == "wait":
        planner.phase = "reach"
        task.event_log.append({"time": task.time, "type": "oracle_reach_start"})
    secured = angle >= release_at
    if planner.phase in ("release", "clear") or (planner.phase == "turn" and secured):
        # Open the fingers at the clip, then lift clear so the rod is out of contact.
        if planner.phase == "turn":
            planner.phase = "release"
            planner.release_time = task.time
            planner.release_yaw = planner.yaw
        planner.yaw = planner.release_yaw
        planner.open_width = 0.030
        if planner.phase == "release" and not obj.grasped and task.time - planner.release_time > 0.15:
            planner.phase = "clear"
        if planner.phase == "release":
            return planner.finish(state, grip, False, 0.20)
        return planner.finish(state, grip + [0.0, 0.0, 0.07], False, 0.30)
    if planner.phase == "reach":
        planner.yaw = hand_yaw + angle
        if np.linalg.norm(ee[:2] - hover[:2]) > 0.008 or ee[2] - hover[2] > 0.02:
            return planner.finish(state, hover, False, REACH_SPEED)
        planner.phase = "descend"
    if planner.phase == "descend":
        planner.yaw = hand_yaw + angle
        if np.linalg.norm(ee - grip) > 0.005:
            return planner.finish(state, grip, False, DESCEND_SPEED)
        planner.phase = "pinch"
    if planner.phase == "pinch":
        planner.yaw = hand_yaw + angle
        if not obj.grasped:
            return planner.finish(state, grip, True, 0.20)
        planner.phase = "turn"
        planner.turn = angle
    # Roll the wrist: the commanded turn follows the measured rod toward the clip and never leads
    # it by more than TURN_LEAD (finger slip cannot wind up an unreachable target).
    if not obj.grasped and np.linalg.norm(ee - grip) > 0.012:
        planner.phase = "reach"  # lost the rod: approach again at its current angle
        return planner.finish(state, hover, False, REACH_SPEED)
    desired = min(turn_required + TURN_PAST, angle + TURN_LEAD)
    planner.turn += float(np.clip(desired - planner.turn, 0.0, TURN_RATE * planner.dt))
    planner.yaw = hand_yaw + planner.turn
    return planner.finish(state, grip, True, 0.20)


# --- authoring -----------------------------------------------------------------------------------

SCREEN_TEXT = {
    "ok": ("BLEEDING TRAINER\nNO BLEEDING\nTOURNIQUET LOOSE", (0.45, 1.0, 0.60), (0.02, 0.06, 0.08)),
    "alarm": ("MASSIVE BLEEDING\nTIGHTEN THE\nTOURNIQUET", (1.0, 1.0, 1.0), (0.85, 0.05, 0.04)),
    "stopped": ("BLEEDING STOPPED\nSECURE THE ROD\nIN THE CLIP", (0.05, 0.05, 0.05), (1.0, 0.72, 0.10)),
    "secured": ("TOURNIQUET SECURED\nBLEEDING\nCONTROLLED", (1.0, 1.0, 1.0), (0.05, 0.50, 0.20)),
}
LOSS_SCALE_ML = 100.0  # full height of the display's blood-loss column
LOSS_BAR_HEIGHT = 0.170


def _collision(backend, prim):
    from pxr import UsdPhysics, UsdShade

    UsdPhysics.CollisionAPI.Apply(prim)
    if hasattr(backend, "_collision_settings"):
        backend._collision_settings(prim)
    material = backend.stage.GetPrimAtPath(f"/World/Materials/{NAME}")
    if material.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(material), materialPurpose="physics")


def author(backend):
    """The rod's colliders and its sprung revolute joint; in the Isaac scene, every visual."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = backend.stage
    root = stage.GetPrimAtPath(ROOT)
    if not root.IsValid():
        raise ValueError("Author the windlass rod's rigid body before its joint")
    for child in list(root.GetChildren()):
        stage.RemovePrim(child.GetPath())
    bar = UsdGeom.Cube.Define(stage, ROOT + "/Collider")
    bar.CreateSizeAttr(1.0)
    bar.AddScaleOp().Set(Gf.Vec3f(ROD_LENGTH, ROD_WIDTH, ROD_HEIGHT))
    _collision(backend, bar.GetPrim())
    UsdGeom.Imageable(bar).MakeInvisible()
    info = backend.layout.info["tourniquet"]
    x, y, z = (float(v) for v in info["pivot"])
    joint = UsdPhysics.RevoluteJoint.Define(stage, JOINT_PATH)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ROOT)])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, z))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    lower, upper = joint_limits(info)
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    strap = drive_values(info, "strap")
    drive.CreateStiffnessAttr(strap["stiffness"])
    drive.CreateDampingAttr(strap["damping"])
    drive.CreateTargetPositionAttr(strap["target"])
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(strap["max_force"])
    if hasattr(backend, "world"):  # the Isaac scene; plain-USD unit stages get the mechanism only
        _dress(backend)
    return joint


def _dress(backend):
    from pxr import UsdGeom

    from roboclock.envs.hospital.bedside.iv_bag import _face_dial, _materials, _robot_base
    from roboclock.envs.hospital.bedside.ventilator import _bed, _dial_clamp
    from roboclock.envs.hospital.rig import dial_target

    layout, stage = backend.layout, backend.stage
    info = layout.info["tourniquet"]
    for fixture in layout.fixtures:
        prim = stage.GetPrimAtPath(f"/World/Fixtures/{fixture.name}_Visual")
        if not fixture.visible and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    mats = _materials(backend.appearance)
    _robot_base(backend, mats)
    _bed(backend, info, mats)
    # The patient, the band and the bleeding are authored in the limb frame: a group at the
    # pivot, turned by the episode's leg yaw.
    px, py, pz = info["pivot"]
    limb = backend.appearance.group(f"{BASE}/Limb", (px, py, 0.0), rz=float(info.get("leg_yaw_deg", 0.0)))
    local = {
        **info, **info["limb"], "pivot": [0.0, 0.0, pz],
        "pad": {"center": info["limb"]["pad"], "size": info["pad"]["size"]},
    }
    _patient(backend, local, limb)
    _tourniquet(backend, local, limb)
    _rod_visual(backend)
    cues = _bleeding(backend, local, limb)
    screens, lamps, bar = _display(backend, info)
    _kit(backend, info)
    _face_dial(backend, dial_target(backend, info["camera"]["position"]))
    _dial_clamp(backend, mats)
    backend._tq_visual = {
        "task": backend.task, "tick": -1, "screen": None, "lamp": None, "pool": None, "bar": None,
        "streak": None, "screens": screens, "lamps": lamps, "loss_bar": bar, **cues,
    }
    _animate(backend, force=True)


def _patient(backend, info, base):
    """The manikin with its injured leg bare; drapes over the trunk and the other leg; a pillow;
    the absorbent pad under the injured leg."""
    from pxr import Usd, UsdGeom, UsdShade

    from roboclock.envs.hospital.props import add_model, model_size

    look, stage = backend.appearance, backend.stage
    ox, oy, oz = info["manikin_origin"]
    skin = look.material("b6_manikin_skin", color=(0.84, 0.64, 0.52), roughness=0.55)
    drape = look.material("b6_drape", color=(0.36, 0.60, 0.66), roughness=0.9)
    linen = look.material("b6_pillow", color=(0.92, 0.93, 0.95), roughness=0.9)
    pad_blue = look.material("b6_pad_blue", color=(0.55, 0.74, 0.90), roughness=0.85)
    pad_white = look.material("b6_pad_white", color=(0.95, 0.96, 0.96), roughness=0.95)
    look.rounded(f"{base}/Pillow", (ox, oy + 0.80, oz + 0.022), (0.46, 0.30, 0.050), linen, 0.022)
    pad = info["pad"]
    px, py, pz = pad["center"]
    sx, sy, sz = pad["size"]
    look.rounded(f"{base}/PadBacking", (px, py, pz - 0.001), (sx + 0.03, sy + 0.03, sz - 0.002), pad_blue, 0.003)
    look.rounded(f"{base}/Pad", (px, py, pz), (sx, sy, sz), pad_white, 0.003)
    # Trunk drape from the groin to the collarbones (over the arms), and a drape over the far leg.
    look.rounded(f"{base}/DrapeTrunk", (ox, oy + 0.14, oz + 0.135), (0.70, 0.76, 0.29), drape, 0.11)
    look.rounded(f"{base}/DrapeLeg", (ox + 0.185, oy - 0.56, oz + 0.11), (0.33, 0.70, 0.25), drape, 0.10)
    if model_size("patient_manikin") is None:
        look.rounded(f"{base}/Thigh", (ox + THIGH[0], oy - 0.35, oz + 0.12), (0.11, 0.36, 0.14), skin, 0.05)
        look.sphere(f"{base}/Head", (ox, oy + 0.75, oz + 0.12), 0.10, skin)
        return
    add_model(stage, "patient_manikin", f"{base}/Manikin", (ox, oy, oz))
    for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{base}/Manikin")):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(skin.material, UsdShade.Tokens.strongerThanDescendants)


def _tourniquet(backend, info, base):
    """Band around the thigh, stabilization plate, twisted strap under the rod, retaining clip
    with its white securing strap (all static visuals; the plate also has a collider)."""
    look = backend.appearance
    nylon = look.material("b6_tq_nylon", color=(0.035, 0.035, 0.04), roughness=0.85)
    plastic = look.material("b6_tq_plastic", color=(0.07, 0.07, 0.08), roughness=0.4)
    white = look.material("b6_tq_time_strap", color=(0.93, 0.93, 0.90), roughness=0.8)
    red = look.material("b6_tq_red", color=(0.85, 0.05, 0.04), roughness=0.5)
    px, py, pz = info["pivot"]
    top = info["plate_top"]
    ax, ay, az = info["thigh_axis"]
    look.cylinder(f"{base}/Band", (ax, ay, az), THIGH_RADIUS + 0.004, BAND_WIDTH, nylon, axis="Y")
    look.rounded(f"{base}/Plate", (px, py, top - PLATE[2] / 2), PLATE, plastic, 0.003)
    look.cylinder(f"{base}/StrapTwist", (px, py, top + (ROD_ABOVE_PLATE - ROD_HEIGHT / 2) / 2),
                  0.008, ROD_ABOVE_PLATE - ROD_HEIGHT / 2, nylon)
    # Retaining clip at the proximal end, at the episode's clip angle off the limb axis: two horns
    # either side of the rod's end position on a red mark, in a group turned about the pivot. The
    # white securing strap lies across the plate's proximal end.
    clip = look.group(f"{base}/Clip", (px, py, 0.0), rz=float(info.get("placement", {}).get("clip_deg", 0.0)))
    clip_y = ROD_LENGTH / 2 - 0.012
    horn = (0.006, 0.020, ROD_ABOVE_PLATE + ROD_HEIGHT / 2 + 0.004)
    for k, side in enumerate((-1, 1)):
        look.rounded(f"{clip}/ClipHorn_{k}", (side * (ROD_WIDTH / 2 + 0.006), clip_y, top + horn[2] / 2),
                     horn, plastic, 0.002)
    look.rounded(f"{clip}/ClipBase", (0.0, clip_y, top + 0.0015), (ROD_WIDTH + 0.024, 0.024, 0.003), plastic, 0.001)
    look.rounded(f"{base}/TimeStrap", (px, py + PLATE[1] / 2 - 0.006, top + 0.002), (PLATE[0] - 0.006, 0.012, 0.004),
                 white, 0.001)
    look.rounded(f"{clip}/ClipMark", (0.0, clip_y, top + 0.0032), (ROD_WIDTH + 0.022, 0.010, 0.0008), red, 0.0003)


def _rod_visual(backend):
    """The windlass rod (children of the rigid body): a black bar with a red tip at its +x end."""
    look = backend.appearance
    black = look.material("b6_rod_black", color=(0.05, 0.05, 0.06), roughness=0.35)
    red = look.material("b6_rod_red", color=(0.95, 0.06, 0.04), roughness=0.4, emissive=True)
    tip = 0.028
    look.rounded(ROOT + "/Bar", (-tip / 2, 0.0, 0.0), (ROD_LENGTH - tip, ROD_WIDTH, ROD_HEIGHT), black, 0.004)
    look.rounded(ROOT + "/Tip", (ROD_LENGTH / 2 - tip / 2, 0.0, 0.0), (tip, ROD_WIDTH + 0.001, ROD_HEIGHT + 0.001), red, 0.004)
    look.cylinder(ROOT + "/Slot", (0.0, 0.0, 0.0), 0.0045, ROD_HEIGHT + 0.0012, look.material(
        "b6_rod_slot", color=(0.30, 0.30, 0.32), roughness=0.5))


def _bleeding(backend, info, base):
    """The wound (always visible, dry until the event), the streak to the pad and the pool."""
    from pxr import Gf

    look = backend.appearance
    dark = look.material("b6_wound", color=(0.30, 0.02, 0.03), roughness=0.5)
    blood = look.material("b6_blood", color=(0.55, 0.01, 0.02), roughness=0.12)
    wx, wy, wz = info["wound"]
    wound = look.sphere(f"{base}/Wound", (wx, wy, wz), 0.010, dark)
    wound.AddScaleOp().Set(Gf.Vec3f(0.6, 2.6, 1.4))
    px, py, pz = info["pool"]
    # The streak runs from the wound down the outer thigh to the pad; shown from the event on.
    from pxr import UsdGeom

    from roboclock.envs.hospital.scene_meshes import tube_arrays

    ax, _, az = info["thigh_axis"]
    radius = math.hypot(wx - ax, wz - az) + 0.002
    start = math.atan2(wz - az, wx - ax)  # angle of the wound about the thigh's axis (x-z plane)
    path = [
        [ax + radius * math.cos(a), wy, az + radius * math.sin(a)]
        for a in np.linspace(start, start + math.radians(35.0), 5)
    ]
    path += [[path[-1][0] - 0.004, wy, pz + 0.03], [px + 0.02, wy, pz + 0.004], [px, py, pz + 0.002]]
    look.mesh(f"{base}/Streak", tube_arrays(path, 0.0045, sides=8), (0, 0, 0), blood)
    streak_scale = UsdGeom.Xformable(backend.stage.GetPrimAtPath(f"{base}/Streak")).AddScaleOp(opSuffix="shown")
    streak_scale.Set(Gf.Vec3f(1e-4, 1e-4, 1e-4))
    pool = look.cylinder(f"{base}/Pool", (px, py, pz), 0.010, 0.0016, blood)
    pool_scale = pool.AddScaleOp(opSuffix="grown")
    pool_scale.Set(Gf.Vec3f(1e-4, 1e-4, 1.0))
    return {"streak_op": streak_scale, "pool_op": pool_scale}


def _display(backend, info):
    """The bleeding trainer's display on a pole stand beyond the bed, turned and leaned back toward
    the robot's head camera: status screen, alarm bar and a blood-loss column."""
    from pxr import Gf, UsdGeom

    from roboclock.envs.hospital.bedside.iv_bag import _quad, _text_texture
    from roboclock.envs.hospital.rig import dial_target

    look, stage = backend.appearance, backend.stage
    dx, dy, dz = info["display"]
    toward = dial_target(backend, info["camera"]["position"])
    yaw = math.degrees(math.atan2(toward[1] - dy, toward[0] - dx))
    lean = math.degrees(math.atan2(toward[2] - dz, math.hypot(toward[0] - dx, toward[1] - dy)))
    frame = look.tint("b6_display_stand", (0.72, 0.74, 0.77), 0.35)
    look.cylinder(f"{BASE}/DisplayPole", (dx + 0.03, dy, (dz - 0.06) / 2), 0.014, dz - 0.06, frame)
    look.cylinder(f"{BASE}/DisplayBase", (dx + 0.03, dy, 0.012), 0.16, 0.024, frame)
    group = look.group(f"{BASE}/Display", (dx, dy, dz), rz=yaw - 90.0)
    panel = look.group(group + "/Lean", (0.0, 0.0, 0.0), rx=min(lean, 60.0))
    look.rounded(panel + "/Housing", (0.03, -0.014, 0.0), (0.40, 0.024, 0.235), "cabinet_white", 0.008)
    look.rounded(panel + "/Bezel", (0.0, -0.0025, -0.012), (0.29, 0.004, 0.185), "screen_dark", 0.003)
    look.rounded(panel + "/AlarmBarSeat", (0.0, -0.008, 0.101), (0.25, 0.018, 0.026), "cabinet_white", 0.006)
    # Blood-loss column right of the screen: a dark well with a red bar that grows from its foot.
    well_x = 0.190
    look.rounded(panel + "/LossWell", (well_x, -0.0025, -0.012), (0.036, 0.004, LOSS_BAR_HEIGHT + 0.012), "screen_dark", 0.003)
    red = look.material("b6_loss_red", color=(1.0, 0.05, 0.03), roughness=0.3, emissive=True)
    foot = look.group(panel + "/LossFoot", (well_x, 0.0008, -0.012 - LOSS_BAR_HEIGHT / 2))
    bar = look.rounded(foot + "/Bar", (0.0, 0.0, 0.5), (0.026, 0.002, 1.0), red, 0.0004)
    bar_scale = UsdGeom.Xformable(stage.GetPrimAtPath(foot)).AddScaleOp(opSuffix="level")
    bar_scale.Set(Gf.Vec3f(1.0, 1.0, 1e-4))
    if not backend.config["visuals"]["labels"]:
        return None, None, bar_scale

    def switchable(prim):
        return UsdGeom.Xformable(prim).AddScaleOp(opSuffix="shown")

    screens = {}
    for key, (text, fg, bg) in SCREEN_TEXT.items():
        card = _quad(stage, f"{panel}/Screen_{key}", _text_texture(text, fg, bg, (640, 400)),
                     (0.0, 0.0005, -0.012), (0.28, 0.175), emissive=True)
        screens[key] = switchable(card)
    _quad(stage, f"{panel}/LossLabel", _text_texture("BLOOD\nLOSS", (0.1, 0.1, 0.1), (0.93, 0.93, 0.93), (256, 128)),
          (well_x, 0.0005, 0.097), (0.056, 0.028))
    lamps = {}
    for on, material in (
        (True, look.material("b6_alarm_on", color=(1.0, 0.06, 0.03), roughness=0.3, emissive=True)),
        (False, look.material("b6_alarm_off", color=(0.42, 0.10, 0.09), roughness=0.4)),
    ):
        name = f"{panel}/AlarmBar_{'on' if on else 'off'}"
        look.rounded(name, (0.0, 0.002, 0.101), (0.24, 0.012, 0.022), material, 0.005)
        lamps[on] = switchable(stage.GetPrimAtPath(name))
    return screens, lamps, bar_scale


def _kit(backend, info):
    """Bleeding-control kit laid out on a tray beside the injured shin (downloaded props)."""
    from roboclock.envs.hospital.props import add_model, model_size

    look, stage = backend.appearance, backend.stage
    kx, ky, top = info["kit_tray"]
    tray = look.material("b6_kit_tray", color=(0.78, 0.80, 0.82), roughness=0.35)
    look.rounded(f"{BASE}/KitTray", (kx, ky, top + 0.006), (0.15, 0.30, 0.012), tray, 0.006)
    z = top + 0.012
    for name, (dx, dy, yaw) in {
        "trauma_kit_pouch": (0.0, 0.075, math.pi / 2),
        "bandage_roll": (-0.03, -0.045, 0.0),
        "gauze_pack": (0.035, -0.055, 0.3),
        "trauma_forceps": (0.0, -0.115, math.pi / 2),
    }.items():
        if model_size(name) is not None:
            add_model(stage, name, f"{BASE}/Kit_{name}", (kx + dx, ky + dy, z), yaw=yaw)


def display_state(task):
    """Which screen the trainer shows: ok, alarm (bleeding), stopped (not yet secured), secured."""
    if task.event_time is None:
        return "ok"
    if task.secure_time is not None and not task.physical_failures:
        return "secured"
    return "alarm" if task.rate > 0.0 else "stopped"


def _show(ops, key):
    from pxr import Gf

    for name, op in ops.items():
        scale = 1.0 if name == key else 1e-4
        op.Set(Gf.Vec3f(scale, scale, scale))


def _animate(backend, force=False):
    """Visual consequences of the task's state (25 Hz): display, alarm bar, loss column, streak, pool.
    Cues are switched by scale: RTX shows transform changes on the next frame."""
    visual = getattr(backend, "_tq_visual", None)
    task = getattr(backend, "task", None)
    if visual is None or task is None or getattr(backend, "no_render", False):
        return
    if visual["task"] is not task:
        visual.update(task=task, tick=-1, screen=None, lamp=None, pool=None, bar=None, streak=None)
    tick = int(round(task.time / 0.04))
    if tick == visual["tick"] and not force:
        return
    visual["tick"] = tick
    from pxr import Gf

    state = display_state(task)
    if visual["screens"] is not None and state != visual["screen"]:
        _show(visual["screens"], state)
        visual["screen"] = state
    lamp = state == "alarm" and int(task.time / 0.25) % 2 == 0
    if visual["lamps"] is not None and lamp != visual["lamp"]:
        _show(visual["lamps"], lamp)
        visual["lamp"] = lamp
    streak = task.event_time is not None
    if streak != visual["streak"]:
        s = 1.0 if streak else 1e-4
        visual["streak_op"].Set(Gf.Vec3f(s, s, s))
        visual["streak"] = streak
    radius = round(pool_radius(task.blood_lost), 4)
    if radius != visual["pool"]:
        s = max(radius / 0.010, 1e-4)
        visual["pool_op"].Set(Gf.Vec3f(s, s, 1.0))
        visual["pool"] = radius
    level = round(min(task.blood_lost / LOSS_SCALE_ML, 1.0) * LOSS_BAR_HEIGHT, 4)
    if level != visual["bar"]:
        visual["loss_bar"].Set(Gf.Vec3f(1.0, 1.0, max(level, 1e-4)))
        visual["bar"] = level


def _camera(backend):
    """Frame the robot, the injured leg, the tourniquet and the display, once per reset."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["tourniquet"]["camera"]
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
