"""B4 "ventilator circuit disconnection": put the breathing circuit back on the patient's airway.

Scene (ICU bedside, scene-02 left bay). A ventilated training manikin lies supine on a bed with a
tracheostomy tube in its neck: only the neck flange and the tube's standard 15 mm connector show.
The ventilator's breathing circuit ends in a catheter-mount elbow whose 22 mm socket sits over
that connector. The ventilator stands on the far side of the bed; its display and alarm bar face
the robot's head camera. While the circuit is connected the manikin's chest rises and falls.

Event. At a seeded time the circuit comes off the connector, as it does when a patient coughs or
the circuit's weight drags on the tube: the elbow lifts off and comes to rest on a folded towel on
the upper chest, 0.145 m from the connector, held upright by its hose. The chest stops moving, the
display turns red (DISCONNECT / LOW PRESSURE) and the alarm bar flashes. The pop-off itself is a
scripted exogenous event: for EJECT_TIME the elbow's velocity is driven along a short arc; from
then on it is an ordinary free rigid body again.

Response. The robot starts tucked at rest. It has to pick the elbow up and push its socket back
over the connector. Nothing attaches: the elbow is a free body before and after the event, and it
is "reconnected" when the measured geometry says so and it stays there by itself after release:
the connector inside the socket by at least SEAT_DEPTH of its 18 mm, the two axes within
AXIS_TOLERANCE, the elbow released, out of robot contact, at rest and in contact with the tube.
The alarm clears and the chest motion resumes from that measurement only.

What is not modelled: gas flow and airway pressure (the alarm is a symbolic consequence of the
measured geometry), the taper's friction fit (the reseated elbow is held by gravity; the hose is
a visual that follows the elbow and carries no force), the patient's physiology.

Assets. Downloaded models (CC BY 4.0, scripts/hospital/import_b4_assets.py, credits in
assets/models/credits.json): icu_ventilator, patient_manikin, tracheostomy_tube (the bedside
spare) and the repository's hospital_bed_flat. Authored here because they are the measured
mechanism or its cue: the connector and flange, the elbow, the hose, the towel, the display.

All heights follow the configured arm-0 mount (robot.rig.tasks.B4): the mattress top is
MATTRESS_BELOW_MOUNT below it.
"""

import math
from pathlib import Path

import numpy as np

from roboclock.physics.geometry import rotation_matrix

ELBOW = "vent_elbow"
CONNECTOR_CONTACTS = ("fixture:trach_connector", "fixture:trach_flange")
INSTRUCTION = "VENTILATOR DISCONNECTED: PUT THE CIRCUIT BACK ON THE AIRWAY CONNECTOR"

# Elbow geometry in its own frame (origin = center of the catalog grip body, 24 x 30 x 30 mm).
GRIP_HALF_Z = 0.015
SOCKET_LENGTH = 0.022  # socket depth below the grip body
SOCKET_INNER_HALF = 0.0095  # 19 mm square bore (collision); drawn as a round 22 mm socket
SOCKET_WALL = 0.004
RIM_Z = -GRIP_HALF_Z - SOCKET_LENGTH  # socket rim plane, elbow frame
STUB = (0.047, 0.0, -0.002)  # where the hose leaves the elbow (elbow frame, +x)
# Tracheostomy tube: a 14 mm square collision post drawn as the round 15 mm connector.
POST_HALF = 0.007
POST_HEIGHT = 0.018
FLANGE = (0.060, 0.036, 0.006)
SEAT_DEPTH = 0.014  # connector length inside the socket that counts as reseated
AXIS_TOLERANCE = 0.15  # rad between the socket and connector axes
REST_SPEED = 0.05  # m/s: "at rest" for the released, reseated elbow
SEAT_DWELL = 0.20  # s the released elbow must stay seated

# Manikin frame (patient_manikin.usd: supine, head toward +y, back on z = 0): the stoma.
STOMA = (0.0, 0.585, 0.180)
MATTRESS_BELOW_MOUNT = 0.52
PAD_TOP = 0.268  # towel top above the mattress
PAD_SIZE = (0.11, 0.10, 0.030)
PARK_DY = -0.145  # towel center from the stoma, toward the feet

# Robot frame before the seed's common shift; arm 0's mount (the rig's right arm) is at the origin.
STOMA_XY = (0.40, 0.12)
BED_XY = (0.45, STOMA_XY[1] - 0.62)  # bed model center; its head end is toward +y
BED_SCALE_Y = 0.50  # the 1.49 m wide model (rails out) slimmed to a 0.74 m ICU bed
VENT_XY = (1.22, 0.42)  # ventilator trolley, beyond the bed
VENT_YAW = 200.0
DISPLAY_XY = (0.93, 0.10)
DISPLAY_BELOW_MOUNT = 0.10
HOSE_ANCHOR = (0.80, 0.34, 0.33)  # circuit hanger: x, y, height above the mattress
INDICATOR_XY = (0.15, 0.40)
SPARE_XY = (0.63, 0.36)
TUCK = (0.20, -0.04, 0.02)  # from the arm-0 mount
APPROACH_WIDTH = 0.022  # per finger: 44 mm across the 30 mm grip body
EJECT_TIME = 0.45  # s the scripted pop-off lasts
BREATH_PERIOD = 3.75  # s, 16 breaths a minute
CHEST_RISE = 0.014  # m at the sternum
CAMERA = {"position": [0.62, -1.10, 1.90], "look_at": [0.50, 0.12, 0.88], "focal_mm": 20.0}

# The ward turns 180 degrees about the robot and shifts so its bed bay lands on this bed; the
# ward's own bed and the ultrasound cart (which stands where the robot goes) are hidden.
WARD_POSE = {
    "usd": "assets/hospital/ward_02_preview_left.usdc",
    "translate": [1.74, -2.22, 0.0],
    "yaw": 180.0,
}


def mount_height(config, task_id="B4"):
    from roboclock.envs.lab.robot import mount_positions

    return float(mount_positions(config, task_id)[0][2])


def build_layout(builder):
    from roboclock.envs.hospital.bedside.ward import ULTRASOUND_CART, furniture

    b = builder
    task_id = getattr(b.task, "task_id", "B4")
    mount_z = mount_height(b.config, task_id)
    mattress = mount_z - MATTRESS_BELOW_MOUNT
    b.z = mattress
    flange_z = mattress + STOMA[2]  # the neck surface at the stoma
    flange_top = flange_z + FLANGE[2]
    post_top = flange_top + POST_HEIGHT
    pad_top = mattress + PAD_TOP
    park_xy = (STOMA_XY[0], STOMA_XY[1] + PARK_DY)

    # Colliders (all drawn by _dress): the neck flange, the 15 mm connector, the towel on the
    # chest and the chest itself, so a dropped elbow lands on the patient, not inside it.
    b.fixture("trach_flange", STOMA_XY, FLANGE, "cap_white", z=flange_z, visible=False)
    b.fixture(
        "trach_connector", STOMA_XY, (2 * POST_HALF, 2 * POST_HALF, POST_HEIGHT), "cap_white",
        z=flange_top, visible=False,
    )
    b.fixture("chest_pad", park_xy, PAD_SIZE, "tissue_white", z=pad_top - PAD_SIZE[2], visible=False)
    b.fixture(
        "patient_chest", (STOMA_XY[0], STOMA_XY[1] - 0.29), (0.34, 0.44, 0.232), "skin",
        z=mattress, visible=False,
    )
    # Seated: the socket rim rests on the flange with the whole connector inside the socket.
    seated_center = flange_top - RIM_Z
    half = b.catalog[ELBOW]["size"][2] / 2
    b.item(
        ELBOW, ELBOW, STOMA_XY, seated_center - half, color=(0.16, 0.42, 0.80), yaw=0.0,
        lift=-0.0015,
    )
    b.indicator("indicator", INDICATOR_XY, "VENT ALARM")

    stoma = np.array([*b.xy(*STOMA_XY), flange_z])
    bed_center = b.xy(*BED_XY)
    tuck = np.array([TUCK[0], TUCK[1], mount_z + TUCK[2]])  # robot frame: the robot never shifts
    hide = ULTRASOUND_CART + furniture("bed") + ["iMac_5k_Monitor054"]
    b.info.update(
        {
            "bedside_task": "ventilator",
            "instruction": INSTRUCTION,
            "initial_ee_position": tuck.tolist(),
            "initial_ee_yaw": 0.0,
            "initial_gripper_width": APPROACH_WIDTH,
            "ward_pose": {**WARD_POSE, "hide": hide},
            "hide_bench": True,
            # The mattress is the support surface: nothing falls through the bed.
            "work_surface": {
                "center": [*bed_center.tolist(), mattress - 0.05],
                "size": [2.2 * BED_SCALE_Y * 1.49 / 2.2, 2.0, 0.10],
            },
            "ventilator": {
                "elbow": ELBOW,
                "mattress_top": mattress,
                "stoma": stoma.tolist(),
                "post_top": [stoma[0], stoma[1], post_top],
                "post_base": [stoma[0], stoma[1], flange_top],
                "seated_center": [stoma[0], stoma[1], seated_center],
                "park_center": [*b.xy(*park_xy).tolist(), pad_top - RIM_Z + 0.001],
                "pad": {"center": [*b.xy(*park_xy).tolist(), pad_top - PAD_SIZE[2] / 2], "size": list(PAD_SIZE)},
                "manikin_origin": [stoma[0] - STOMA[0], stoma[1] - STOMA[1], mattress],
                "bed": {"center": bed_center.tolist(), "scale_y": BED_SCALE_Y},
                "vent": {"xy": b.xy(*VENT_XY).tolist(), "yaw": VENT_YAW},
                "display": [*b.xy(*DISPLAY_XY).tolist(), mount_z - DISPLAY_BELOW_MOUNT],
                "hose_anchor": [*b.xy(*HOSE_ANCHOR[:2]).tolist(), mattress + HOSE_ANCHOR[2]],
                "spare_xy": b.xy(*SPARE_XY).tolist(),
                "camera": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in CAMERA.items()},
                "eject_time": EJECT_TIME,
                "seat_depth": SEAT_DEPTH,
                "assumptions": [
                    "scripted pop-off: the elbow's velocity is driven for 0.45 s, then it is a free body",
                    "rigid 19 mm square socket over a 14 mm square post (drawn round: 22 mm over 15 mm)",
                    "no taper friction fit: the reseated elbow is held by gravity",
                    "the hose is a visual that follows the elbow and carries no force",
                    "symbolic ventilator state; no gas flow, pressure or physiology",
                ],
            },
        }
    )


# --- measurement ---------------------------------------------------------------------------------


def geometry(layout, obj):
    """Where the connector is inside the elbow's socket, measured in the elbow's own frame."""
    info = layout.info["ventilator"]
    rotation = rotation_matrix(obj.orientation)
    top = rotation.T @ (np.asarray(info["post_top"]) - obj.position)
    base = rotation.T @ (np.asarray(info["post_base"]) - obj.position)
    axis = top - base
    axis /= np.linalg.norm(axis)
    tilt = float(np.arccos(np.clip(axis[2], -1.0, 1.0)))
    depth = float(top[2] - RIM_Z)  # connector length above the socket's rim plane
    # The connector where it crosses the rim plane must fit the bore as well as its top does.
    crossing = top[:2] - axis[:2] * depth / max(axis[2], 1e-12)
    lateral = float(max(np.max(np.abs(top[:2])), np.max(np.abs(crossing))))
    aligned = tilt <= AXIS_TOLERANCE and lateral + POST_HALF <= SOCKET_INNER_HALF + 0.0005
    contact = any(name in obj.fixture_contacts for name in CONNECTOR_CONTACTS)
    seated = aligned and depth >= SEAT_DEPTH
    return {
        "depth": depth,
        "lateral_error": lateral,
        "axis_error": tilt,
        "aligned": bool(aligned),
        "seated": bool(seated),
        "tube_contact": bool(contact),
        "retained": bool(
            seated
            and contact
            and not obj.grasped
            and not obj.robot_contact
            and np.linalg.norm(obj.velocity) < REST_SPEED
        ),
    }


def eject_reference(start, park, s):
    """Elbow center along the scripted pop-off, s in [0, 1]: straight up off the connector first,
    then across to the towel in a low arc."""
    s = float(np.clip(s, 0.0, 1.0))
    start, park = np.asarray(start, dtype=float), np.asarray(park, dtype=float)

    def smooth(u):
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3 - 2 * u)

    # Up first: clear of the connector before any sideways motion and above the towel before
    # reaching it; the last 12 mm down onto the towel come at the end.
    hover = 0.012
    across = smooth((s - 0.25) / 0.75)
    z = start[2] + (park[2] + hover - start[2]) * smooth(s / 0.45) - hover * smooth((s - 0.8) / 0.2)
    xy = start[:2] + (park[:2] - start[:2]) * across
    return np.array([xy[0], xy[1], z])


def _eject(backend, state):
    """The scripted event: drive the elbow's velocity along the pop-off arc for EJECT_TIME."""
    task = backend.task
    record = getattr(backend, "_vent_eject", None)
    if record is None or record["task"] is not task:
        record = backend._vent_eject = {"task": task, "phase": "connected", "start": None, "t0": None}
    if record["phase"] == "connected":
        if task.event_time is None:
            return record["phase"]
        record.update(phase="ejecting", t0=task.time, start=state.objects[ELBOW].position.copy())
    if record["phase"] == "ejecting":
        info = backend.layout.info["ventilator"]
        dt = backend.config["sim"]["physics_dt"]
        s = (task.time - record["t0"]) / EJECT_TIME
        body = backend.objects[ELBOW]
        if s >= 1.0:
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))
            record["phase"] = "free"
            task.event_log.append({"time": task.time, "type": "circuit_at_rest_on_chest"})
        else:
            target = eject_reference(record["start"], info["park_center"], s + dt / EJECT_TIME)
            velocity = (target - state.objects[ELBOW].position) / dt
            speed = float(np.linalg.norm(velocity))
            if speed > 1.5:
                velocity *= 1.5 / speed
            body.set_linear_velocity(velocity)
            body.set_angular_velocity(np.zeros(3))
    return record["phase"]


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_vent_camera_for", None) is not backend.task:
        backend._vent_camera_for = backend.task  # the first measurement of a reset aims the view
        _camera(backend)
    phase = _eject(backend, state)
    measured = geometry(backend.layout, state.objects[ELBOW])
    measured["circuit_free"] = phase == "free"
    measured["ejecting"] = phase == "ejecting"
    _animate(backend, state, measured)
    return {"ventilator": measured}


# --- privileged diagnostic oracle ----------------------------------------------------------------

REACH_SPEED, CARRY_SPEED = 0.55, 0.40
HOVER = 0.030  # above the grip body before the final vertical descent
LIFT, LIFT_SPEED = 0.012, 0.25  # off the towel before moving sideways
CLEAR = 0.010  # socket rim above the connector top while the elbow is brought over it
TRACK_STEP = 0.006
ALIGN_SPEED = 0.12
INSERT_STEP, INSERT_SPEED = 0.004, 0.08
GRIP_WIDTH = 0.006  # per-finger target on the 30 mm grip body
TAPER = 4.0


def command(planner, task, state):
    """Privileged diagnostic controller; every waypoint becomes bounded articulation commands."""
    info, obj = task.layout.info["ventilator"], state.objects[ELBOW]
    now = task.time
    if planner.last_goal is None:
        planner.last_goal = state.ee_position.copy()
    if getattr(planner, "_vent_task", None) is not task:
        planner._vent_task = task
        planner.vent_phase, planner.vent_phase_time = "wait", now
        planner.vent_offset = np.zeros(3)
    phase = planner.vent_phase
    planner.open_width, planner.close_width, planner.gentle = APPROACH_WIDTH, GRIP_WIDTH, True
    planner.step_yaw(0.0)
    close, speed = False, REACH_SPEED
    tuck = np.asarray(task.layout.info["initial_ee_position"])
    goal = tuck.copy()
    measured = geometry(task.layout, obj)
    post_top = np.asarray(info["post_top"])
    park = np.asarray(info["park_center"])

    def transition(value):
        planner.vent_phase, planner.vent_phase_time = value, now
        task.event_log.append({"time": now, "type": "ventilator_planner_phase", "phase": value})

    def tapered(target, vmax):
        return float(np.clip(TAPER * np.linalg.norm(target - planner.last_goal), 0.05, vmax))

    def elbow_goal(depth):
        """Elbow center that puts the connector `depth` inside the socket, axes aligned."""
        return post_top - [0.0, 0.0, RIM_Z + depth]

    def track(elbow_target, step=TRACK_STEP):
        # A bounded integral correction on the measured elbow removes IK and load bias.
        correction = np.clip(0.25 * (elbow_target - obj.position), -step, step)
        return np.clip(
            planner.last_goal + correction, state.ee_position - 0.020, state.ee_position + 0.020
        )

    if phase == "wait":
        # At rest in the tucked pose; nothing is touched before the circuit comes off.
        if task.event_time is not None:
            transition("reach")
    elif phase == "reach":
        # Head for where the elbow comes to rest (the towel); follow the elbow once it is there.
        target = obj.position if getattr(task, "circuit_free", False) else park
        goal = target + [0.0, 0.0, HOVER]
        speed = tapered(goal, REACH_SPEED)
        if (
            getattr(task, "circuit_free", False)
            and np.linalg.norm(state.ee_position[:2] - obj.position[:2]) < 0.008
            and state.ee_position[2] - obj.position[2] < HOVER + 0.012
            and np.linalg.norm(obj.velocity) < 0.05
        ):
            planner.vent_grasp = obj.position.copy()
            transition("descend")
    elif phase == "descend":
        goal, speed = planner.vent_grasp, 0.25
        if abs(state.ee_position[2] - planner.vent_grasp[2]) < 0.006:
            transition("grasp")
    elif phase == "grasp":
        goal, close = planner.vent_grasp, True
        if obj.grasped:
            planner.vent_offset = state.ee_position - obj.position
            transition("lift")
        elif now - planner.vent_phase_time > 1.5:
            transition("reach")  # a missed pinch: open and approach again
    elif phase == "lift":
        goal = planner.vent_grasp + planner.vent_offset + [0.0, 0.0, LIFT + 0.01]
        close, speed = True, LIFT_SPEED
        if obj.position[2] >= planner.vent_grasp[2] + LIFT:
            transition("carry")
    elif phase == "carry":
        # Across at the lifted height (the towel is higher than the connector), then down.
        desired = elbow_goal(-CLEAR)
        if np.linalg.norm(obj.position[:2] - desired[:2]) > 0.015:
            desired[2] = max(desired[2], planner.vent_grasp[2] + LIFT)
        goal, close = desired + planner.vent_offset, True
        speed = tapered(goal, CARRY_SPEED)
        if np.linalg.norm(obj.position - desired) < 0.010:
            transition("align")
        elif not obj.grasped and now - planner.vent_phase_time > 0.3:
            transition("reach")  # the elbow slipped out of the pinch
    elif phase == "align":
        desired = elbow_goal(-CLEAR)
        goal, close, speed = track(desired), True, ALIGN_SPEED
        if (
            np.linalg.norm(obj.position[:2] - desired[:2]) < 0.0025
            and abs(obj.position[2] - desired[2]) < 0.004
            and np.linalg.norm(obj.velocity) < 0.06
        ):
            transition("insert")
        elif not obj.grasped and now - planner.vent_phase_time > 0.3:
            transition("reach")
    elif phase == "insert":
        # Aim 1.5 mm past the seat: the flange stops the rim, so contact arrives during the motion.
        desired = elbow_goal(POST_HEIGHT + 0.0015)
        goal, close, speed = track(desired, INSERT_STEP), True, INSERT_SPEED
        if measured["seated"] and measured["tube_contact"] and measured["depth"] >= POST_HEIGHT - 0.002:
            planner.vent_release = state.ee_position.copy()
            transition("release")
        elif not obj.grasped and now - planner.vent_phase_time > 0.3:
            transition("reach")  # the elbow slipped out of the pinch
    elif phase == "release":
        planner.open_width = 0.04
        goal = planner.vent_release.copy()
        if not obj.grasped and now - planner.vent_phase_time > 0.18:
            transition("clear")
    elif phase == "clear":
        planner.open_width = 0.04
        goal = planner.vent_release + [0.0, 0.0, 0.07]
        speed = 0.30
        if getattr(task, "reconnect_time", None) is None and (
            now - planner.vent_phase_time > 1.0 and not measured["seated"]
        ):
            transition("reach")  # it did not stay on: pick it up again
    return planner.finish(state, np.asarray(goal, dtype=float), close, speed)


# --- dressing (visual only, except the socket walls on the elbow) --------------------------------

SCREEN_TEXT = {
    "ok": ("VENTILATING\nRR 16   PEEP 5\nCIRCUIT OK", (0.45, 1.0, 0.60), (0.02, 0.06, 0.08)),
    "alarm": ("DISCONNECT\nLOW PRESSURE\nAPNEA", (1.0, 1.0, 1.0), (0.85, 0.05, 0.04)),
    "restored": ("VENTILATING\nALARM CLEARED\nCIRCUIT OK", (1.0, 1.0, 1.0), (0.05, 0.50, 0.20)),
}


def author(backend):
    """The elbow's socket walls on its rigid body, then every visual of the bedside."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    stage, look = backend.stage, backend.appearance
    path = f"/World/Objects/{ELBOW}"
    body = stage.GetPrimAtPath(path)
    if not body.IsValid():
        raise ValueError("Author the elbow's rigid body before its socket")
    outer = SOCKET_INNER_HALF + SOCKET_WALL
    center_z = RIM_Z + SOCKET_LENGTH / 2
    physics_material = UsdShade.Material.Get(stage, f"/World/Materials/{ELBOW}")
    for axis in (0, 1):
        for sign in (-1, 1):
            center = [0.0, 0.0, center_z]
            center[axis] = sign * (SOCKET_INNER_HALF + SOCKET_WALL / 2)
            size = [2 * outer, 2 * outer, SOCKET_LENGTH]
            size[axis] = SOCKET_WALL
            cube = UsdGeom.Cube.Define(stage, f"{path}/Socket_{axis}_{'neg' if sign < 0 else 'pos'}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
            if physics_material:
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                    physics_material, materialPurpose="physics"
                )
            if hasattr(backend, "_collision_settings"):
                backend._collision_settings(cube.GetPrim())
    try:
        from pxr import PhysxSchema
    except ImportError:  # plain usd-core (unit tests) has no PhysX schema
        PhysxSchema = None
    if PhysxSchema is not None and body.HasAPI(UsdPhysics.RigidBodyAPI):
        # The hose damps the elbow's rocking; a bare rigid proxy rattles on the connector.
        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateAngularDampingAttr(3.0)
    if hasattr(backend, "world"):  # the Isaac scene; plain-USD unit stages get the socket only
        _dress(backend)


def _dress(backend):
    from pxr import UsdGeom

    from roboclock.envs.hospital.bedside.iv_bag import _face_dial, _materials, _robot_base
    from roboclock.envs.hospital.rig import dial_target

    look, layout, stage = backend.appearance, backend.layout, backend.stage
    info = layout.info["ventilator"]
    for fixture in layout.fixtures:
        prim = stage.GetPrimAtPath(f"/World/Fixtures/{fixture.name}_Visual")
        if not fixture.visible and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    mats = _materials(look)
    _robot_base(backend, mats)
    _bed(backend, info, mats)
    _patient(backend, info)
    _airway(backend, info)
    _elbow_visual(backend)
    _ventilator(backend, info, mats)
    _hose(backend, info)
    _face_dial(backend, dial_target(backend, info["camera"]["position"]))
    _dial_clamp(backend, mats)
    backend._vent_visual = {"task": backend.task, "screen": None, "lamp": None, "tick": -1}
    _set_screen(backend, "ok")
    _set_lamp(backend, False)


def _bed(backend, info, mats):
    from roboclock.envs.hospital.props import add_model, model_info

    look, stage = backend.appearance, backend.stage
    (cx, cy), top = info["bed"]["center"], info["mattress_top"]
    model = model_info("hospital_bed_flat")
    if model is not None:
        # The real bed with its backrest laid flat, head end toward +y, slimmed to ICU width and
        # stretched so its mattress top meets the patient's back.
        native_top = model["top_z"] + 0.02
        add_model(
            stage, "hospital_bed_flat", "/World/Vent/Bed", (cx, cy, 0.0),
            scale=(0.95, info["bed"]["scale_y"], top / native_top), yaw=math.pi / 2,
        )
    else:
        look.rounded("/World/Vent/Mattress", (cx, cy, top - 0.07), (0.74, 2.0, 0.14), "tissue_white", 0.03)
        look.rounded("/World/Vent/BedFrame", (cx, cy, (top - 0.14) / 2), (0.70, 1.96, top - 0.14), "hosp_base", 0.02)
    sheet = look.material("b4_sheet", color=(0.90, 0.93, 0.95), roughness=0.9)
    look.rounded("/World/Vent/Sheet", (cx, cy + 0.02, top - 0.004), (0.70, 1.92, 0.012), sheet, 0.005)


def _patient(backend, info):
    """The manikin, a pillow, a blanket to the lower ribs, the towel on the chest, and the chest
    shell that rises and falls (a copy of the manikin's own chest faces, moved by `_animate`)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    from roboclock.envs.hospital.props import MODELS, add_model, model_size

    look, stage = backend.appearance, backend.stage
    ox, oy, oz = info["manikin_origin"]
    skin = look.material("b4_manikin_skin", color=(0.84, 0.64, 0.52), roughness=0.55)
    blanket = look.material("b4_blanket", color=(0.42, 0.62, 0.78), roughness=0.9)
    towel = look.material("b4_towel", color=(0.95, 0.95, 0.92), roughness=0.95)
    pillow = look.material("b4_pillow", color=(0.92, 0.93, 0.95), roughness=0.9)
    look.rounded("/World/Vent/Pillow", (ox, oy + 0.80, oz + 0.022), (0.46, 0.30, 0.050), pillow, 0.022)
    # Blanket from the feet to the lower ribs, over the arms.
    look.rounded("/World/Vent/Blanket", (ox, oy - 0.36, oz + 0.13), (0.72, 1.12, 0.30), blanket, 0.12)
    look.rounded("/World/Vent/BlanketFold", (ox, oy + 0.20, oz + 0.262), (0.66, 0.09, 0.030), blanket, 0.014)
    pad = info["pad"]
    look.rounded("/World/Vent/Towel", pad["center"], pad["size"], towel, 0.010)
    look.rounded(
        "/World/Vent/TowelFold", np.asarray(pad["center"]) + [0, 0, pad["size"][2] / 2 - 0.003],
        (pad["size"][0] + 0.004, pad["size"][1] * 0.5, 0.008), towel, 0.004,
    )
    backend._vent_chest = None
    if model_size("patient_manikin") is None:
        look.rounded("/World/Vent/Torso", (ox, oy + 0.33, oz + 0.12), (0.36, 0.50, 0.24), skin, 0.08)
        look.sphere("/World/Vent/Head", (ox, oy + 0.75, oz + 0.12), 0.10, skin)
        look.cylinder("/World/Vent/Neck", (ox, oy + 0.60, oz + 0.12), 0.055, 0.12, skin, axis="Y")
        return
    add_model(stage, "patient_manikin", "/World/Vent/Manikin", (ox, oy, oz))
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Vent/Manikin")):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                skin.material, UsdShade.Tokens.strongerThanDescendants
            )
    source = Usd.Stage.Open(str(MODELS / "patient_manikin" / "patient_manikin.usd"))
    mesh = next(UsdGeom.Mesh(p) for p in source.Traverse() if p.IsA(UsdGeom.Mesh))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3)
    x, y, z = points.T

    def ramp(value, lo, hi):
        u = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
        return u * u * (3 - 2 * u)

    weight = (
        ramp(y, 0.20, 0.30) * ramp(-y, -0.565, -0.50) * ramp(-np.abs(x), -0.19, -0.11) * ramp(z, 0.13, 0.19)
    )
    keep = np.all(weight[faces] > 0.0, axis=1) | (np.sum(weight[faces] > 0.0, axis=1) >= 2)
    used = np.unique(faces[keep])
    remap = -np.ones(len(points), dtype=int)
    remap[used] = np.arange(len(used))
    shell = UsdGeom.Mesh.Define(stage, "/World/Vent/ChestShell")
    base = points[used] + [ox, oy, oz + 0.0008]
    shell.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(base.astype(np.float32)))
    shell.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(int(keep.sum()), 3, dtype=np.int32)))
    shell.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(remap[faces[keep]].ravel().astype(np.int32)))
    normals = mesh.GetNormalsAttr().Get()
    if normals is not None and len(normals) == len(points):
        shell.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)[used]))
        shell.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    shell.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(shell.GetPrim()).Bind(skin.material)
    backend._vent_chest = (shell, base, weight[used])
    del Gf, Sdf


def _airway(backend, info):
    """Tracheostomy tube as it shows at the neck: flange with its ties, the 15 mm connector."""
    from roboclock.envs.hospital.props import add_model, model_size

    look, stage = backend.appearance, backend.stage
    sx, sy, sz = info["stoma"]
    white = look.textured("b4_tube_white", "plastic_white", color=(0.97, 0.97, 0.98), roughness=0.3)
    tie = look.material("b4_trach_tie", color=(0.25, 0.50, 0.78), roughness=0.9)
    look.rounded("/World/Vent/Flange", (sx, sy, sz + FLANGE[2] / 2), FLANGE, white, 0.003)
    look.cylinder("/World/Vent/Connector", (sx, sy, sz + FLANGE[2] + POST_HEIGHT / 2), 0.0075, POST_HEIGHT, white)
    look.cylinder("/World/Vent/ConnectorCollar", (sx, sy, sz + FLANGE[2] + 0.0015), 0.0105, 0.003, white)
    for k, side in enumerate((-1, 1)):
        look.rounded(
            f"/World/Vent/Tie_{k}", (sx + side * 0.052, sy, sz - 0.004), (0.050, 0.014, 0.004), tie, 0.0015,
        )
    if model_size("tracheostomy_tube") is not None and "spare_xy" in info:
        # The spare tube every tracheostomy bedside keeps, in its tray beside the pillow.
        px, py = info["spare_xy"]
        top = info["mattress_top"]
        tray = look.textured("b4_spare_tray", "plastic_white", color=(0.80, 0.90, 0.93))
        look.rounded("/World/Vent/SpareTray", (px, py, top + 0.008), (0.10, 0.15, 0.012), tray, 0.004)
        add_model(stage, "tracheostomy_tube", "/World/Vent/SpareTube", (px, py, top + 0.014), yaw=math.pi / 2)


def _elbow_visual(backend):
    """Catheter-mount swivel elbow: blue grip body, round 22 mm socket with a collar, hose stub
    toward +x and the capped suction port on top (children of the rigid body)."""
    from pxr import UsdGeom

    look, stage = backend.appearance, backend.stage
    path = f"/World/Objects/{ELBOW}"
    for name in ("Shell", "Band"):
        prim = stage.GetPrimAtPath(f"{path}/{name}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    blue = look.textured("b4_elbow_blue", "plastic_white", color=(0.16, 0.45, 0.85), roughness=0.3)
    clear = look.textured("b4_elbow_white", "plastic_white", color=(0.97, 0.98, 0.99), roughness=0.25)
    look.rounded(path + "/GripBody", (0.0, 0.0, 0.0), (0.024, 0.030, 2 * GRIP_HALF_Z), blue, 0.006)
    look.cylinder(path + "/SocketShell", (0.0, 0.0, RIM_Z + SOCKET_LENGTH / 2), 0.0125, SOCKET_LENGTH, clear)
    look.cylinder(path + "/SocketCollar", (0.0, 0.0, RIM_Z + 0.002), 0.0145, 0.004, blue)
    look.cylinder(path + "/Stub", (0.012 + (STUB[0] - 0.012) / 2, 0.0, STUB[2]), 0.0085, STUB[0] - 0.012, clear, axis="X")
    look.cylinder(path + "/StubCuff", (STUB[0] - 0.004, 0.0, STUB[2]), 0.0105, 0.008, blue, axis="X")
    look.cylinder(path + "/PortCap", (0.0, 0.0, GRIP_HALF_Z + 0.003), 0.005, 0.006, clear)


def _ventilator(backend, info, mats):
    """The ventilator on its trolley beyond the bed, and its display on a pole stand at the bed's
    far edge, turned and leaned back toward the robot's head camera."""
    from roboclock.envs.hospital.bedside.iv_bag import _quad, _text_texture, _texture_material
    from roboclock.envs.hospital.props import add_model, model_size
    from roboclock.envs.hospital.rig import dial_target

    look, stage = backend.appearance, backend.stage
    vx, vy = info["vent"]["xy"]
    if model_size("icu_ventilator") is not None:
        add_model(stage, "icu_ventilator", "/World/Vent/Ventilator", (vx, vy, 0.0), yaw=math.radians(info["vent"]["yaw"]))
    else:
        look.rounded("/World/Vent/VentilatorBody", (vx, vy, 0.55), (0.36, 0.36, 1.10), "cabinet_white", 0.03)
    dx, dy, dz = info["display"]
    toward = dial_target(backend, info["camera"]["position"])
    yaw = math.degrees(math.atan2(toward[1] - dy, toward[0] - dx))
    lean = math.degrees(math.atan2(toward[2] - dz, math.hypot(toward[0] - dx, toward[1] - dy)))
    frame = look.tint("b4_display_stand", (0.72, 0.74, 0.77), 0.35)
    look.cylinder("/World/Vent/DisplayPole", (dx + 0.03, dy, (dz - 0.06) / 2), 0.014, dz - 0.06, frame)
    look.cylinder("/World/Vent/DisplayBase", (dx + 0.03, dy, 0.012), 0.16, 0.024, frame)
    # The group's local +y faces the camera; the panel leans back to meet its line of sight.
    group = look.group("/World/Vent/Display", (dx, dy, dz), rz=yaw - 90.0)
    panel = look.group(group + "/Lean", (0.0, 0.0, 0.0), rx=min(lean, 60.0))
    look.rounded(panel + "/Housing", (0.0, -0.014, 0.0), (0.32, 0.024, 0.235), "cabinet_white", 0.008)
    look.rounded(panel + "/Bezel", (0.0, -0.0025, -0.012), (0.29, 0.004, 0.185), "screen_dark", 0.003)
    look.rounded(panel + "/AlarmBarSeat", (0.0, -0.008, 0.101), (0.25, 0.018, 0.026), "cabinet_white", 0.006)
    if not backend.config["visuals"]["labels"]:
        backend._vent_screen = backend._vent_lamp = None
        return
    from pxr import Gf, UsdGeom

    def switchable(prim):
        """A scale op that shows (1) or hides (1e-4) the prim. RTX takes a geometry change on the
        next frame; a material rebound on a static card fades in over a second."""
        return UsdGeom.Xformable(prim).AddScaleOp(opSuffix="shown")

    # One card per display state and one bar per lamp state, stacked; only one of each is shown.
    screens = {}
    for key, (text, fg, bg) in SCREEN_TEXT.items():
        card = _quad(
            stage, f"{panel}/Screen_{key}", _text_texture(text, fg, bg, (640, 400)),
            (0.0, 0.0005, -0.012), (0.28, 0.175), emissive=True,
        )
        screens[key] = switchable(card)
    lamps = {}
    for on, material in (
        (True, look.material("b4_alarm_on", color=(1.0, 0.06, 0.03), roughness=0.3, emissive=True)),
        (False, look.material("b4_alarm_off", color=(0.42, 0.10, 0.09), roughness=0.4)),
    ):
        bar = look.rounded(f"{panel}/AlarmBar_{'on' if on else 'off'}", (0.0, 0.002, 0.101), (0.24, 0.012, 0.022), material, 0.005)
        prim = bar.GetPrim() if hasattr(bar, "GetPrim") else stage.GetPrimAtPath(f"{panel}/AlarmBar_{'on' if on else 'off'}")
        lamps[on] = switchable(prim)
    backend._vent_screen, backend._vent_lamp = screens, lamps
    del Gf, _texture_material


def _hose_points(info, position, rotation):
    """The circuit from the elbow's stub to its hanger: out along the stub, a sag, up to the arm."""
    stub = np.asarray(position) + rotation @ np.asarray(STUB)
    out = rotation @ np.array([1.0, 0.0, 0.0])
    anchor = np.asarray(info["hose_anchor"], dtype=float)
    span = anchor - stub
    return [
        stub,
        stub + 0.05 * out,
        stub + 0.12 * out + 0.30 * span * [0, 1, 0] + [0.0, 0.0, 0.03],
        stub + 0.62 * span + [0.0, 0.0, -0.035],
        anchor + [-0.05, 0.0, -0.01],
        anchor,
    ]


def _hose(backend, info):
    from pxr import UsdShade

    from roboclock.envs.hospital.scene_meshes import tube_arrays
    from roboclock.simulators.isaac.visual_geometry import author_mesh

    look, stage = backend.appearance, backend.stage
    hose = look.textured("b4_hose", "plastic_white", color=(0.90, 0.95, 0.98), roughness=0.35)
    start = backend.layout.objects[ELBOW].position
    points, uv, indices, normals = tube_arrays(_hose_points(info, start, np.eye(3)), 0.011, sides=10, ribs=0.10)
    mesh = author_mesh(stage, "/World/Vent/Hose", (points, uv, indices), (0, 0, 0), normals)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(hose.material)
    backend._vent_hose = (mesh, np.asarray(start, dtype=float).copy(), np.eye(3))
    # The circuit's hanger and the limb that runs on to the ventilator (static).
    ax, ay, az = info["hose_anchor"]
    vx, vy = info["vent"]["xy"]
    frame = look.tint("b4_display_stand", (0.72, 0.74, 0.77), 0.35)
    look.cylinder("/World/Vent/HangerClip", (ax, ay, az), 0.016, 0.03, frame, axis="X")
    control = [
        [ax, ay, az], [ax + 0.08, ay + 0.01, az - 0.01],
        [(ax + vx) / 2 + 0.05, (ay + vy) / 2, az - 0.10], [vx - 0.12, vy - 0.06, az - 0.02],
    ]
    look.mesh("/World/Vent/HoseLimb", tube_arrays(control, 0.011, sides=10, ribs=0.10), (0, 0, 0), hose)
    look.cylinder("/World/Vent/HangerPole", (ax + 0.02, ay + 0.03, az / 2), 0.010, az, frame)
    look.cylinder("/World/Vent/HangerBase", (ax + 0.02, ay + 0.03, 0.010), 0.12, 0.020, frame)


def _dial_clamp(backend, mats):
    """The response dial stands on a rail clamp at the bed's near edge."""
    indicator = next((i for i in backend.layout.indicators if i.name == "indicator"), None)
    if indicator is None:
        return
    x, y, z = indicator.position
    top = backend.layout.table_height
    backend.appearance.rounded("/World/Vent/DialClamp", (x, y, top + 0.004), (0.085, 0.085, 0.010), mats["bracket"], 0.003)


def _show(ops, key):
    from pxr import Gf

    for name, op in ops.items():
        scale = 1.0 if name == key else 1e-4
        op.Set(Gf.Vec3f(scale, scale, scale))


def _set_screen(backend, state):
    visual = backend._vent_visual
    if visual["screen"] == state or getattr(backend, "_vent_screen", None) is None:
        return
    _show(backend._vent_screen, state)
    visual["screen"] = state


def _set_lamp(backend, on):
    visual = backend._vent_visual
    if visual["lamp"] == on or getattr(backend, "_vent_lamp", None) is None:
        return
    _show(backend._vent_lamp, on)
    visual["lamp"] = on


def breath(phase_time):
    """Chest excursion (0..1) of a volume-controlled breath: 1.2 s in, then a passive decay."""
    t = phase_time % BREATH_PERIOD
    if t < 1.2:
        return 0.5 - 0.5 * math.cos(math.pi * t / 1.2)
    return math.exp(-(t - 1.2) / 0.45)


def _animate(backend, state, measured):
    """Visual consequences of the measured state (25 Hz): display, alarm bar, chest, hose."""
    visual = getattr(backend, "_vent_visual", None)
    task = backend.task
    if visual is None or visual["task"] is not task or getattr(backend, "no_render", False):
        return
    tick = int(round(task.time / 0.04))
    if tick == visual["tick"]:
        return
    visual["tick"] = tick
    from pxr import Vt

    alarm = task.event_time is not None and getattr(task, "reconnect_time", None) is None
    restored = getattr(task, "reconnect_time", None) is not None
    _set_screen(backend, "alarm" if alarm else "restored" if restored else "ok")
    _set_lamp(backend, alarm and int(task.time / 0.25) % 2 == 0)
    chest = getattr(backend, "_vent_chest", None)
    if chest is not None:
        shell, base, weight = chest
        # Ventilated breaths while connected; the chest falls passively once the circuit is off.
        if alarm:
            since = task.time - task.event_time
            level = visual.get("level_at_event", 0.0) * math.exp(-since / 0.45)
        else:
            origin = task.reconnect_time if restored else 0.0
            level = breath(task.time - origin)
            visual["level_at_event"] = level
        if abs(level - visual.get("level", -1.0)) > 0.01:
            visual["level"] = level
            points = base.copy()
            points[:, 2] += CHEST_RISE * level * weight
            shell.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    hose = getattr(backend, "_vent_hose", None)
    if hose is not None:
        mesh, last_position, last_rotation = hose
        obj = state.objects[ELBOW]
        rotation = rotation_matrix(obj.orientation)
        if (
            np.linalg.norm(obj.position - last_position) > 0.0008
            or np.abs(rotation - last_rotation).max() > 0.01
        ):
            from roboclock.envs.hospital.scene_meshes import tube_arrays

            points, _, _, normals = tube_arrays(
                _hose_points(backend.layout.info["ventilator"], obj.position, rotation), 0.011, sides=10, ribs=0.10
            )
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)))
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)))
            backend._vent_hose = (mesh, obj.position.copy(), rotation)


def _camera(backend):
    """Frame the robot, the patient's head and chest, the circuit and the display, once per reset."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["ventilator"]["camera"]
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


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "models"
