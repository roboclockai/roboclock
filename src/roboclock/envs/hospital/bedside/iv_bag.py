"""A4 "start the new infusion": spike and hang a prepared saline bag, then press START on its pump.

Step 1 (spike and hang). The cap is removed before the episode. A clamp on the IV pole holds the
spike pointing up, so one grasp can lower the pouch's trainer sleeve over it and release the eyelet
onto a clamp-on hook near the top of the pole. This is an insertion surrogate, not a model of
membrane puncture, flexible tubing or fluid flow. The spike never attaches to the bag; the hook must
physically support the released bag.

Step 2 (press START). The existing infusion pump (CC-BY-4.0, Ethan Cragun) is clamped to the same
pole below the bag, on the pole's robot side: its rotatable pole clamp grips the pole at the
pump's side, so its screen faces the foot of the bed. A top-facing
training adapter carries one physical START key, the same kind of key as B6's keypad
(spring-returned prismatic key, 6 mm stroke, pressed by robot contact), so the top-down X5 hand
makes one downward press. (With the hand pointing down, the X5's wrist and forearm sit about
0.09-0.2 m radially beyond the tool point: a key under the hanging bag would put the wrist into the
bag, so the key sits at a different bearing from the robot than the bag.)

Ward placement (scene-02 left bay, assets/hospital/ward_02_preview_left.usdc): the mobile dual-arm
robot stands on the bed's open side, facing the bed at the patient's shoulder. Between it and the
bed stand the IV pole (local CC-BY-4.0 "IV Stand" by teeart, see assets/models/credits.json) with
the hook, spike clamp and pump, and a narrow stainless prep cart pushed in under the pump, whose top
holds the bag cradle, the response dial and the order card (the invisible support collider is
sized to that top). The ward's ultrasound cart, which stands on this spot, is hidden. The IV line
is visual only.

All heights follow the configured arm-0 mount (the rig height, robot.rig.tasks.A4): the cart top is
CART_BELOW_MOUNT below it and the hook HOOK_ABOVE_MOUNT above it, so the relative geometry the
oracle and the scoring use does not change with the mount height. The top-down X5 hand reaches at
most about 0.15-0.18 m above its mount at these distances (joint 4 limit), which sets how high the
hook can go: with the mount at 1.50 m the hook is at 1.55 m, 0.15 m below the stand's crown.
"""

import math

import numpy as np

from roboclock.physics.geometry import rotation_matrix

BAG = "bag_0"
HOOK_CONTACT = "fixture:iv_hook_peg"
PORT_LENGTH = 0.030
PORT_INNER_HALF = 0.010
PORT_WALL = 0.004
SPIKE_HALF = 0.003
INSERT_DEPTH = 0.018
AXIS_TOLERANCE = 0.12
BAG_ANGULAR_DAMPING = 4.0  # 1/s, the saline pouch's own damping of its swing

# START key: the same physical key as B6's keypad (see bedside/pump_panel.py), copied here.
START_KEY = "pump_start"
START_STROKE = 0.006  # key travel to its end stop
START_PRESS = 0.004  # travel that actuates the key
START_HOLD = 0.04  # seconds the actuated key must stay down with robot contact
INSTRUCTION = "START INFUSION: 1 SPIKE + HANG SALINE  2 PRESS PUMP START"

# Last round's bench rule (bench = mount - 0.30), kept for code that imports bench_height().
MOUNT_ABOVE_BENCH = 0.30
CART_BELOW_MOUNT = 0.55  # A4 prep-cart top below the left-arm mount (0.95 m for a 1.50 m mount)
HOOK_ABOVE_MOUNT = 0.05  # A4 hook peg center above the left-arm mount (1.55 m)
PUMP_LIFT = 0.005  # the pump hangs on the pole just above the cart's back end
CRADLE_HEIGHT = 0.10  # the bag cradle's shoulders on the cart top

# Scene-02's ultrasound cart stands on the bed's open side at the head end, where the robot goes.
ULTRASOUND_CART = [
    f"Obj24of_591669_{k}_{suffix}"
    for k, suffix in (
        (4, "386"), (5, "498"), (6, "853"), (7, "182"), (8, "786"), (9, "616"),
        (10, "520"), (11, "051"), (12, "779"), (13, "830"), (14, "113"),
    )
]
# The ward turns 180 degrees about the robot, then shifts: the bed's open long side (ward x 1.79)
# lands at robot x 0.67 and the patient's shoulders (ward y -2.10) at robot y 0.24, by the pole.
WARD_POSE = {
    "usd": "assets/hospital/ward_02_preview_left.usdc",
    "translate": [2.46, -1.86, 0.0],
    "yaw": 180.0,
    # The patient monitor on the ceiling boom hangs just behind the IV pole's top in the view.
    "hide": ULTRASOUND_CART + ["iMac_5k_Monitor054"],
}
# Robot frame, before the seed's common shift. Arm 0's mount (the rig's right arm) is at the origin.
POLE_XY = (0.36, 0.20)
HANG_XY = (0.36, 0.06)  # the bag hangs 0.14 m in front of the pole on the hook's free arm
PICK_XY = (0.21, -0.12)  # the cradle on the prep cart
# Pump: native size (x, y, z) of the infusion_pump model, whose screen faces its +x. Turned -90
# degrees the screen faces the foot of the bed (-y); its side clamp grips the pole at its +x end.
PUMP_SIZE = (0.229, 0.24, 0.2235)
PUMP_YAW = -90.0
POLE_RADIUS = 0.022  # measured from the iv_stand mesh
PUMP_XY = (POLE_XY[0] - POLE_RADIUS - 0.002 - PUMP_SIZE[1] / 2, POLE_XY[1])
# START sits on the main unit's top at its robot end, about 45 degrees off the bag's bearing.
START_XY = (PUMP_XY[0] - 0.040, PUMP_XY[1] - 0.010)
# Prep cart top: its top is the task bench height. The cart is pushed in under the pump between
# the robot and the pole, so its top (the support collider) spans the cradle, the dial, the order
# card and the START key's footprint; the pole stands just beyond its bed-side edge.
CART_X = (0.085, POLE_XY[0] - POLE_RADIUS - 0.008)
CART_Y = (-0.39, START_XY[1] + 0.035)
CART_CENTER = ((CART_X[0] + CART_X[1]) / 2, (CART_Y[0] + CART_Y[1]) / 2)
CART_SIZE = (CART_X[1] - CART_X[0], CART_Y[1] - CART_Y[0])
INDICATOR_XY = (0.135, -0.33)
ORDER_CARD_XY = (0.255, -0.345)
# Tucked start: the hand folded in near the robot's own column, gripper pre-opened, top down.
TUCK = (0.20, -0.04, 0.02)  # from the left-arm mount
APPROACH_WIDTH = 0.022  # per finger: 44 mm across the 30 mm grasp tab
# From the bed's foot-side corner: robot, cradle, dial and order card, hook, bag, spike and pump.
CAMERA = {"position": [0.70, -0.85, 1.72], "look_at": [0.22, 0.03, 1.31], "focal_mm": 20.0}


def bench_height(config, task_id="A4"):
    """Last round's bench rule: the configured left-arm mount minus a 0.30 m column."""
    from roboclock.envs.lab.robot import mount_positions

    return float(mount_positions(config, task_id)[0][2]) - MOUNT_ABOVE_BENCH


def mount_height(config, task_id="A4"):
    from roboclock.envs.lab.robot import mount_positions

    return float(mount_positions(config, task_id)[0][2])


def cart_top(config, task_id="A4"):
    """A4's bench: the prep-cart top, CART_BELOW_MOUNT below the configured left-arm mount."""
    return mount_height(config, task_id) - CART_BELOW_MOUNT


def build_layout(builder):
    b = builder
    task_id = getattr(b.task, "task_id", "A4")
    mount_z = mount_height(b.config, task_id)
    b.z = mount_z - CART_BELOW_MOUNT
    spec = b.catalog["blood_bag"]
    bx, by, bz = spec["size"]
    eye = spec["eyelet"]
    # The peg's top is the eyelet's upper inner surface when the bag hangs.
    peg_z = mount_z + HOOK_ABOVE_MOUNT
    target = np.array([*b.xy(*HANG_XY), peg_z - (bz / 2 + eye["height"] - eye["bar"] - 0.006)])
    # Two shoulders support the bag body, leaving the trainer port clear underneath.
    support_z = b.z + CRADLE_HEIGHT
    for side in (-1, 1):
        b.fixture(
            f"iv_cradle_{'left' if side < 0 else 'right'}",
            (PICK_XY[0] + side * 0.036, PICK_XY[1]),
            (0.022, 0.040, CRADLE_HEIGHT),
            "polymer",
        )
    b.item(
        BAG,
        "blood_bag",
        PICK_XY,
        support_z - 0.002,
        color=(0.86, 0.93, 0.96),
        label="SALINE",
        yaw=0.0,
    )

    # Floor-standing IV pole beside the bed (the model is drawn in author); its collider is the
    # pole tube. A clamp-on hook near the top, a spike clamp and the pump hang off the pole.
    hook_y = POLE_XY[1] - 0.10
    b.fixture(
        "iv_stand_pole",
        POLE_XY,
        (0.032, 0.032, 1.69 - 0.10),
        "stainless",
        z=0.10,
        visible=False,
    )
    b.fixture("iv_hook_peg", (HANG_XY[0], hook_y), (0.012, 0.20, 0.012), "stainless", z=peg_z - 0.006)
    # The spike has 8 mm clearance to the solid body when the eyelet rests on the peg.
    tip = target + np.array([0.0, 0.0, -bz / 2 - 0.008])
    spike_height = 0.055
    b.fixture(
        "iv_spike_holder",
        HANG_XY,
        (0.055, 0.055, 0.014),
        "polymer",
        z=tip[2] - spike_height - 0.014,
    )
    reach = POLE_XY[1] - HANG_XY[1]
    b.fixture(
        "iv_spike_support",
        (HANG_XY[0], HANG_XY[1] + reach / 2),
        (0.014, reach, 0.012),
        "stainless",
        z=tip[2] - spike_height - 0.012,
    )
    b.fixture(
        "iv_spike",
        HANG_XY,
        (2 * SPIKE_HALF, 2 * SPIKE_HALF, spike_height),
        "polymer",
        z=tip[2] - spike_height,
    )

    # The pump housing (an invisible collider around the model) and its START training adapter.
    pump_base = b.z + PUMP_LIFT
    pump_top = pump_base + PUMP_SIZE[2]
    b.fixture(
        "iv_pump_housing",
        PUMP_XY,
        (PUMP_SIZE[1], PUMP_SIZE[0], PUMP_SIZE[2]),  # turned -90 degrees: native x along y
        "cabinet_white",
        z=pump_base,
        visible=False,
    )
    b.fixture("pump_start_base", START_XY, (0.080, 0.060, 0.014), "rubber", z=pump_top)
    # Key center 28 mm above the plate's bottom: 8 mm free travel above the plate, 6 mm stroke.
    b.item(
        START_KEY,
        "pump_key",
        START_XY,
        pump_top + 0.028 - 0.006 - 0.002,
        color=(0.10, 0.62, 0.25),
        yaw=0.0,
    )
    b.indicator("indicator", INDICATOR_XY, "START INFUSION")

    grasp_dz = bz / 2 + eye["height"] + 0.022
    # A visible trainer tab above the eyelet permits a central, torque-free pinch.
    grasp_offset = np.array([0.0, 0.0, grasp_dz])
    cart_center = b.xy(*CART_CENTER)
    tuck = np.array([TUCK[0], TUCK[1], mount_z + TUCK[2]])  # robot frame: the robot never shifts
    b.info.update(
        {
            "bedside_task": "iv_bag",
            "instruction": INSTRUCTION,
            "initial_ee_position": tuck.tolist(),
            "initial_ee_yaw": 0.0,
            "initial_gripper_width": APPROACH_WIDTH,
            # Bedside placement: the ward around the robot, the cart top as the only bench.
            "ward_pose": {**WARD_POSE, "hide": list(WARD_POSE["hide"])},
            "hide_bench": True,
            "work_surface": {
                "center": [*cart_center.tolist(), b.z - 0.05],
                "size": [*CART_SIZE, 0.10],
            },
            "iv_bag": {
                "bag": BAG,
                "body_size": list(spec["size"]),
                "eyelet": dict(eye),
                "port_origin": [0.0, 0.0, -bz / 2 - PORT_LENGTH],
                "spike_tip": tip.tolist(),
                "spike_base": (tip - [0.0, 0.0, spike_height]).tolist(),
                "hook_center": [target[0], b.xy(HANG_XY[0], hook_y)[1], peg_z],
                "hook_half_length": 0.10,
                "hook_half_thickness": 0.006,
                "hang_position": target.tolist(),
                "grasp_offset": grasp_offset.tolist(),
                "pole_xy": b.xy(*POLE_XY).tolist(),
                "cart": {"center": cart_center.tolist(), "size": list(CART_SIZE), "top": b.z},
                "pump": {
                    "origin": [*b.xy(*PUMP_XY).tolist(), pump_base],
                    "yaw": PUMP_YAW,
                    "top": pump_top,
                },
                "start_key": {
                    "name": START_KEY,
                    "origin": b.objects[START_KEY].position.tolist(),
                    "stroke": START_STROKE,
                    "press": START_PRESS,
                    "hold": START_HOLD,
                },
                "order_card_xy": b.xy(*ORDER_CARD_XY).tolist(),
                "camera": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in CAMERA.items()},
                "assumptions": [
                    "cap pre-removed",
                    "fixture-held spike",
                    "reinforced grasp tab above eyelet",
                    "rigid open trainer port",
                    "no membrane, fluid or flexible tubing simulation",
                    "clamp-on training hook near the top of the IV pole",
                    "top-facing START training key on the pump (same key as B6)",
                    "symbolic pump state; no infusion model",
                ],
                "visual_asset": "src/roboclock/envs/hospital/assets/models/blood_bag/blood_bag.usd",
            },
        }
    )


def author(backend):
    """Add the hollow training port to the existing dynamic pouch, without any joint, and the
    START key's spring-returned prismatic joint."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage, look = backend.stage, backend.appearance
    info = backend.layout.info["iv_bag"]
    path = f"/World/Objects/{BAG}"
    body = stage.GetPrimAtPath(path)
    if not body.IsValid():
        raise ValueError("Author the dynamic IV pouch before its trainer port")
    origin = np.array(info["port_origin"])
    center_z = origin[2] + PORT_LENGTH / 2
    outer = PORT_INNER_HALF + PORT_WALL
    for axis in (0, 1):
        for sign in (-1, 1):
            center = [0.0, 0.0, center_z]
            center[axis] = sign * (PORT_INNER_HALF + PORT_WALL / 2)
            size = [2 * outer, 2 * outer, PORT_LENGTH]
            size[axis] = PORT_WALL
            cube = UsdGeom.Cube.Define(
                stage, f"{path}/TrainerPort_{axis}_{'neg' if sign < 0 else 'pos'}"
            )
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                look.material("cap_blue").material
            )
            if hasattr(backend, "_collision_settings"):
                backend._collision_settings(cube.GetPrim())
    # Reinforced trainer tab: one visible collider on the same free pouch rigid body.
    # The central pinch keeps the pouch upright and leaves the hanging eyelet unobstructed.
    eye_top = info["body_size"][2] / 2 + info["eyelet"]["height"]
    tab = UsdGeom.Cube.Define(stage, f"{path}/TrainerGripTab")
    tab.CreateSizeAttr(1.0)
    tab.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, eye_top + 0.015))
    tab.AddScaleOp().Set(Gf.Vec3f(0.024, 0.030, 0.030))
    UsdPhysics.CollisionAPI.Apply(tab.GetPrim())
    binding = UsdShade.MaterialBindingAPI.Apply(tab.GetPrim())
    binding.Bind(look.material("cap_blue").material)
    physics_material = UsdShade.Material.Get(stage, f"/World/Materials/{BAG}")
    if physics_material:
        binding.Bind(physics_material, materialPurpose="physics")
    if hasattr(backend, "_collision_settings"):
        backend._collision_settings(tab.GetPrim())
    # A liquid-filled pouch damps its own swing; the rigid proxy otherwise rocks on the hook and
    # spike for seconds after release (PhysX default 0.05/s).
    try:
        from pxr import PhysxSchema
    except ImportError:  # plain usd-core (unit tests) has no PhysX schema
        PhysxSchema = None
    if PhysxSchema is not None and body.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateAngularDampingAttr(BAG_ANGULAR_DAMPING)
    # Preserve the supplied mesh's silhouette, replace the blood texture with saline plastic.
    saline = look.material("iv_bag", color=(0.86, 0.93, 0.96), roughness=0.18).material
    for prim in Usd.PrimRange(body):
        if prim.IsA(UsdGeom.Mesh) or prim.GetName() in ("Pouch", "WeldTop", "WeldBottom"):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                saline, UsdShade.Tokens.strongerThanDescendants
            )
    label = stage.GetPrimAtPath(path + "/BagLabel")
    if label.IsValid():
        UsdGeom.Imageable(label).MakeInvisible()
    labels = backend.config["visuals"]["labels"]
    if labels:
        by = info["body_size"][1]
        for side, suffix in ((-1, "Back"), (1, "Front")):
            card = look.label(  # 2:1 like the label texture; short enough for its 512 px width
                path + "/SalineLabel" + suffix,
                "SALINE 0.9%",
                [0.0, side * (by / 2 + 0.001), 0.0],
                [0.084, 0.042],
                (0.15, 0.42, 0.65),
            )
            if side < 0 and card is not None:
                card.AddRotateZOp().Set(180.0)  # a label card reads from +y; this face looks -y
    # START: a spring-returned prismatic key with end stops, as on B6's keypad.
    key = info["start_key"]
    key_path = f"/World/Objects/{key['name']}"
    joint = UsdPhysics.PrismaticJoint.Define(stage, f"/World/Joints/{key['name']}")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(key_path)])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*key["origin"]))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLowerLimitAttr(-key["stroke"])
    joint.CreateUpperLimitAttr(0.0)
    joint.CreateCollisionEnabledAttr(False)
    spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    spring.CreateTypeAttr("force")
    spring.CreateStiffnessAttr(700.0)
    spring.CreateDampingAttr(2.0)
    spring.CreateTargetPositionAttr(0.0)
    spring.CreateTargetVelocityAttr(0.0)
    spring.CreateMaxForceAttr(6.0)
    if hasattr(backend, "world"):  # the Isaac scene; plain-USD unit stages get the port only
        if labels and stage.GetPrimAtPath(key_path).IsValid():
            _flat_text(stage, key_path + "/KeyName", "START", [0, 0, 0.0065], (0.050, 0.025), KEY_TEXT_YAW)
        _dress(backend)


# --- measurement ---------------------------------------------------------------------------------


def start_key_state(layout, obj):
    """Measured START key travel, speed, robot contact and lateral alignment (B6's key measure)."""
    origin = np.asarray(layout.info["iv_bag"]["start_key"]["origin"])
    return {
        "depression_m": float(origin[2] - obj.position[2]),
        "speed_m_s": float(obj.velocity[2]),
        "contact": bool(obj.robot_contact),
        "aligned": bool(np.linalg.norm(obj.position[:2] - origin[:2]) < 0.001),
    }


def geometry(layout, obj):
    """Measure the complete spike envelope in the port frame and eyelet on the peg."""
    info = layout.info["iv_bag"]
    rotation = rotation_matrix(obj.orientation)
    local_tip = rotation.T @ (np.asarray(info["spike_tip"]) - obj.position)
    local_base = rotation.T @ (np.asarray(info["spike_base"]) - obj.position)
    port = np.asarray(info["port_origin"])
    axis = local_tip - local_base
    axis /= np.linalg.norm(axis)
    tilt = float(np.arccos(np.clip(axis[2], -1.0, 1.0)))
    depth = float(local_tip[2] - port[2])
    # The shaft where it crosses the opening must also fit; checking its tip alone permits tilt.
    crossing = local_tip[:2] - axis[:2] * depth / max(axis[2], 1e-12)
    lateral = float(max(np.max(np.abs(local_tip[:2])), np.max(np.abs(crossing))))
    aligned = tilt <= AXIS_TOLERANCE and lateral + SPIKE_HALF <= PORT_INNER_HALF - 0.001
    contained = aligned and INSERT_DEPTH <= depth <= PORT_LENGTH - 0.004

    hook = np.asarray(info["hook_center"])
    # Closest point on the real peg segment, with room for the entire eyelet thickness.
    hook_y = float(
        np.clip(
            obj.position[1], hook[1] - info["hook_half_length"], hook[1] + info["hook_half_length"]
        )
    )
    local_hook = rotation.T @ (np.array([hook[0], hook_y, hook[2]]) - obj.position)
    eye = info["eyelet"]
    half_peg = info["hook_half_thickness"]
    upper_inner = info["body_size"][2] / 2 + eye["height"] - eye["bar"]
    peg_overlap = abs(obj.position[1] - hook[1]) + eye["thickness"] / 2 < info["hook_half_length"]
    eye_supported = (
        peg_overlap
        and tilt <= AXIS_TOLERANCE
        and abs(rotation[1, 1]) >= np.cos(AXIS_TOLERANCE)
        and abs(local_hook[0]) + half_peg < eye["width"] / 2 - eye["bar"]
        and abs(local_hook[2] + half_peg - upper_inner) <= 0.004
        and HOOK_CONTACT in obj.fixture_contacts
        and not obj.grasped
        and not obj.robot_contact
        and np.linalg.norm(obj.velocity) < 0.06
    )
    return {
        "depth": depth,
        "lateral_error": lateral,
        "axis_error": tilt,
        "aligned": bool(aligned),
        "inserted": bool(contained),
        "hanging": bool(eye_supported),
        "hook_contact": HOOK_CONTACT in obj.fixture_contacts,
    }


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_iv_bag_camera_for", None) is not backend.task:
        backend._iv_bag_camera_for = backend.task  # the first measurement of a reset aims the view
        _camera(backend)
    key = start_key_state(backend.layout, state.objects[START_KEY])
    _pump_screen(backend, key)
    return {"iv_bag": geometry(backend.layout, state.objects[BAG]), START_KEY: key}


# --- privileged diagnostic oracle ----------------------------------------------------------------

# Oracle rates. Near the spike the bag is tracked with a bounded integral correction of at most
# TRACK_STEP per 33 ms control step (last round's speed-ups: 6 mm tracking steps, 0.09 m/s
# alignment, 1.5 mm insertion overshoot, kept). The gross moves (reach to the tab, lift and raise
# with the bag, the move to START) command the tool directly at the speeds below.
TRACK_STEP, FREE_SPEED = 0.006, 0.22
LIFT, LIFT_SPEED = 0.04, 0.30  # straight up off the cradle: the 30 mm port clears its shoulders
RAISE_SPEED = 0.36
RAISE_BACK = 0.06  # the rising bag stays this far short of the entry until the palm clears
RAISE_SHORT = 0.015  # the raise ends this far short of the entry along the hook (no overshoot)
TAPER = 4.0  # gross moves slow to TAPER x the remaining distance (1/s) near their goals
TRANSFER_STEP, TRANSFER_SPEED = TRACK_STEP, FREE_SPEED  # threading (faster swung the eyelet onto the hook)
PALM_CLEAR = -0.035  # tool height above the hook peg at which the palm's underside clears it
ALIGN_SPEED = 0.09
ENTRY_TILT = 0.10  # rad: the held pouch should hang this upright before it is threaded
GRIP_WIDTH = 0.006  # per-finger target on the 30 mm tab: a firm pinch the pouch cannot pivot in
INSERT_STEP, INSERT_SPEED = 0.003, 0.07
REACH_SPEED, PRESS_SPEED = 0.55, 0.20
HOVER = 0.030  # above the tab before the final vertical descent
KEY_HOVER = 0.030  # above the START key before the press
KEY_HIGH = 0.17  # the move to START passes over the key this high, then descends vertically
CLEAR_X = 0.05  # after release the open fingers back off the 24 mm tab toward the robot


def command(planner, task, state):
    """Privileged diagnostic controller; every waypoint becomes bounded articulation commands."""
    info, obj = task.layout.info["iv_bag"], state.objects[BAG]
    now = task.time
    if planner.last_goal is None:
        planner.last_goal = state.ee_position.copy()
    if getattr(planner, "_iv_bag_task", None) is not task:
        planner._iv_bag_task = task
        planner.iv_bag_phase, planner.iv_bag_phase_time = "wait", now
        planner.iv_bag_offset = np.asarray(info["grasp_offset"])
    phase = planner.iv_bag_phase
    planner.open_width, planner.close_width, planner.gentle = APPROACH_WIDTH, GRIP_WIDTH, True
    planner.step_yaw(0.0)
    close, speed = False, FREE_SPEED
    # Holding a fixed goal prevents contact/IK error from accumulating as drift.
    goal = np.asarray(task.layout.info["initial_ee_position"]).copy()
    measured = geometry(task.layout, obj)
    rotation = rotation_matrix(obj.orientation)
    held_offset = state.ee_position - obj.position
    source = task.layout.objects[BAG].position
    grasp = source + np.asarray(info["grasp_offset"])
    key = np.asarray(info["start_key"]["origin"])

    def transition(value):
        planner.iv_bag_phase, planner.iv_bag_phase_time = value, now
        task.event_log.append({"time": now, "type": "iv_bag_planner_phase", "phase": value})

    def sleeve_goal(depth, step=TRACK_STEP):
        # Align the measured sleeve axis, accounting for the small natural tilt of a held pouch.
        local_tip = np.asarray(info["port_origin"]) + [0.0, 0.0, depth]
        bag_goal = np.asarray(info["spike_tip"]) - rotation @ local_tip
        return track_bag(bag_goal, step), bag_goal

    def tapered(target, vmax):
        # Constant speed far away, then an exponential approach: no overshoot or swing at the end.
        return float(np.clip(TAPER * np.linalg.norm(target - planner.last_goal), 0.05, vmax))

    def track_bag(bag_goal, step=TRACK_STEP):
        # Accumulate a bounded Cartesian correction to remove static IK/load bias.
        # The command remains within 20 mm of the measured tool and uses normal motor limits.
        correction = np.clip(0.25 * (bag_goal - obj.position), -step, step)
        return np.clip(
            planner.last_goal + correction, state.ee_position - 0.020, state.ee_position + 0.020
        )

    if phase == "wait":
        # At rest in the tucked pose until the instruction; nothing is touched before it.
        if task.event_time is not None:
            transition("reach")
    elif phase == "reach":
        goal = grasp + [0.0, 0.0, HOVER]
        speed = tapered(goal, REACH_SPEED)
        if np.linalg.norm(state.ee_position[:2] - grasp[:2]) < 0.008 and (
            state.ee_position[2] - grasp[2] < HOVER + 0.012
        ):
            transition("descend")
    elif phase == "descend":
        goal, speed = grasp, 0.25
        # The fingers start closing over the last few millimeters of the descent.
        if abs(state.ee_position[2] - grasp[2]) < 0.006:
            transition("grasp")
    elif phase == "grasp":
        goal, close = grasp, True
        if obj.grasped:
            planner.iv_bag_offset = held_offset
            transition("lift")
    elif phase == "lift":
        # Straight up until the port is out of the cradle, then the raise may move sideways.
        goal = source + planner.iv_bag_offset + [0.0, 0.0, LIFT + 0.01]
        close, speed = True, LIFT_SPEED
        if obj.position[2] >= source[2] + LIFT:
            transition("raise")
    elif phase == "raise":
        # Carry the bag up quickly toward the entry with a direct tool goal. The 0.17 m wide palm
        # (across the fingers) must rise above the hook before it passes over the hook's free end,
        # so the bag stays RAISE_BACK short of the entry until the palm is clear. The entry
        # alignment below removes the remaining error with the integral tracking.
        _, desired = sleeve_goal(-0.004)
        desired[1] -= 0.115 + RAISE_SHORT
        if state.ee_position[2] < info["hook_center"][2] + PALM_CLEAR:
            desired[1] -= RAISE_BACK
        goal, close = desired + planner.iv_bag_offset, True
        speed = tapered(goal, RAISE_SPEED)
        # The direct goal leaves the hanging pouch's small static offset; the alignment's integral
        # tracking removes it once the hand has arrived.
        arrived = np.linalg.norm(planner.last_goal - goal) < 0.003
        if (arrived and np.linalg.norm(state.ee_position - goal) < 0.012) or np.linalg.norm(
            obj.position - desired
        ) < 0.008:
            transition("align_entry")
    elif phase == "align_entry":
        _, desired = sleeve_goal(-0.004)
        desired[1] -= 0.115
        goal, close, speed = track_bag(desired), True, ALIGN_SPEED
        # Thread a settled, nearly upright pouch: one swung about the pinch knocks the hook. A
        # held pouch keeps a small static tilt in the pinch, so the tilt gate only waits a while.
        upright = measured["axis_error"] < ENTRY_TILT or now - planner.iv_bag_phase_time > 0.5
        if (
            np.linalg.norm(obj.position - desired) < 0.003
            and np.linalg.norm(obj.velocity) < 0.06
            and upright
        ):
            transition("transfer")
    elif phase == "transfer":
        # Move gently through the open eyelet while the entire spike remains below its sleeve.
        goal, desired = sleeve_goal(-0.004, TRANSFER_STEP)
        close, speed = True, TRANSFER_SPEED
        if (
            measured["aligned"]
            and measured["depth"] <= 0.0
            and np.linalg.norm(obj.position - desired) < 0.004
            and np.linalg.norm(obj.velocity) < 0.06
        ):
            transition("insert")
    elif phase == "insert":
        # Aim 1.5 mm past the 22 mm resting depth: the peg stops the eyelet there, so hook contact
        # arrives during the motion instead of at the end of an asymptotic approach. (3 mm pressed
        # the held bag onto the peg hard enough to tilt it off the spike at release.)
        goal, _ = sleeve_goal(0.0235, INSERT_STEP)
        close, speed = True, INSERT_SPEED
        if task.insertion_time is not None and measured["inserted"] and measured["hook_contact"]:
            planner.iv_bag_release_goal = state.ee_position.copy()
            transition("release")
    elif phase == "release":
        planner.open_width = 0.04
        goal = planner.iv_bag_release_goal.copy()
        if not obj.grasped and now - planner.iv_bag_phase_time > 0.18:
            transition("clear")
    elif phase == "clear":
        # Back the open fingers off the tab toward the robot, above the bag, then go down to START.
        planner.open_width = 0.04
        goal = planner.iv_bag_release_goal + [-CLEAR_X, 0.0, 0.02]
        speed = REACH_SPEED
        if state.ee_position[0] <= planner.iv_bag_release_goal[0] - CLEAR_X + 0.006:
            transition("to_start")
    elif phase == "to_start":
        # Close the fingers into one pressing tip on the way (B6's press posture). Swing over the
        # key high first, then descend: one long diagonal move lets the per-joint rate limits pull
        # the hand in toward the robot's own column.
        planner.close_width, planner.gentle = 0.0, False
        goal, close = key + [0.0, 0.0, KEY_HOVER], True
        if np.linalg.norm(planner.last_goal[:2] - key[:2]) > 0.01:
            goal = key + [0.0, 0.0, max(KEY_HIGH, min(state.ee_position[2] - key[2], KEY_HIGH + 0.1))]
        speed = tapered(goal, REACH_SPEED)
        # START counts only after the hang is retained: never press before the verification.
        if (
            np.linalg.norm(state.ee_position[:2] - goal[:2]) < 0.006
            and abs(state.ee_position[2] - goal[2]) < 0.015
            and task.hanging_time is not None
        ):
            transition("press")
    elif phase == "press":
        planner.close_width, planner.gentle = 0.0, False
        # Native X5 fingertips project 12.6 mm below the grasp-center frame.
        goal, close, speed = key + [0.0, 0.0, 0.010], True, PRESS_SPEED
        if getattr(task, "start_time", None) is not None:
            transition("done")
    else:
        planner.close_width, planner.gentle = 0.0, False
        goal, close = key + [0.0, 0.0, KEY_HOVER], True
    # A failed grasp remains an actual controller failure; never attach or relocate the pouch.
    return planner.finish(state, np.asarray(goal, dtype=float), close, speed)


# --- bedside dressing (visual only; the colliders are the layout's fixtures) --------------------

STAND_YAW = 0.0  # the crown's bar runs along x (robot to bed); the legs clear base, cart and bed
STAND_SCALE = (0.75, 0.75, 1.0)  # the wheeled base scaled to fit between robot, cart and bed
POLE_AXIS = (0.014, 0.0265)  # pole axis in the iv_stand model's own frame (measured from the mesh)
SCREEN = {"offset": (-0.040, 0.138), "size": (0.076, 0.102)}  # the pump model's display, from its front
KEY_TEXT_YAW = 180.0  # the START cap reads from the camera's side (-y)
SCREEN_TEXT = {
    "idle": ("STANDBY", (0.35, 1.0, 0.55), (0.02, 0.05, 0.08)),
    "cue": ("PRESS\nSTART", (1.0, 1.0, 1.0), (0.80, 0.45, 0.02)),
    "done": ("INFUSING", (1.0, 1.0, 1.0), (0.05, 0.52, 0.20)),
}


def _dress(backend):
    from pxr import UsdGeom

    look, layout = backend.appearance, backend.layout
    info = layout.info["iv_bag"]
    # An invisible collider fixture still authors a "_Visual" box: the models stand in for them.
    for fixture in layout.fixtures:
        prim = backend.stage.GetPrimAtPath(f"/World/Fixtures/{fixture.name}_Visual")
        if not fixture.visible and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    mats = _materials(look)
    _robot_base(backend, mats)
    _cart(look, info["cart"], mats)
    _iv_stand(backend, info, mats)
    _pump(backend, info, mats)
    _line(look, info)
    from roboclock.envs.hospital.rig import dial_target

    _face_dial(backend, dial_target(backend, info["camera"]["position"]))  # the policy's head camera
    _order_card(backend, info, mats)
    bag = layout.objects[BAG].position
    # The cradle's two shoulders stand on one alignment base plate: a molded white bag holder on
    # a brushed aluminum plate, and the spike sits in a white holder on an aluminum arm (these
    # fixtures were "polymer", which the backend draws as blue sterile wrap).
    look.rounded(
        "/World/IVBag/CradleBase", (bag[0], bag[1], layout.table_height + 0.004),
        (0.12, 0.07, 0.008), mats["bracket"], 0.003,
    )
    for name, key in (
        ("iv_cradle_left", "holder"), ("iv_cradle_right", "holder"), ("iv_spike_holder", "holder"),
        ("iv_spike", "spike"), ("iv_spike_support", "bracket"),
    ):
        _bind(backend.stage, f"/World/Fixtures/{name}_Visual", mats[key])
    # START: a molded green key cap (the backend draws the item's shell in a flat color).
    _bind(backend.stage, f"/World/Objects/{START_KEY}/Shell", mats["key"])


def _materials(look):
    """Textured PBR materials (CC0 sets in hospital/assets/textures) for A4's own parts."""
    return {
        "holder": look.textured("b6_holder_white", "plastic_white", color=(1.0, 1.0, 1.0)),
        "spike": look.textured("b6_spike_white", "plastic_white", color=(1.0, 1.0, 1.0), roughness=0.25),
        "bracket": look.textured("b6_bracket_aluminum", "aluminum_brushed"),
        "knob": look.textured("b6_knob_black", "plastic_matte", color=(0.10, 0.10, 0.11)),
        "key": look.textured("b6_start_key", "plastic_white", color=(0.16, 0.95, 0.38)),
        # The robot's shells match the central mount-column coat (appearance.texture_robot_mounts).
        "shell": look.textured("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90)),
        "skirt": look.textured("b6_robot_skirt", "powder_coat_dark"),
        "trim": look.textured("b6_robot_trim", "aluminum_brushed"),
        "card": look.textured("b6_card_paper", "paper"),
        "band": look.textured("b6_card_band", "paper", color=(0.30, 0.64, 0.68)),
        "acrylic": look.textured("b6_card_foot", "plastic_white", color=(1.0, 1.0, 1.0), roughness=0.2),
        "carton": look.textured("b6_supply_carton", "paper", color=(0.95, 0.93, 0.88)),
    }


def _bind(stage, path, material):
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material)


def _face_dial(backend, toward):
    """The shared dial faces +x (the bed here): turn it, its housing and label toward the camera."""
    from pxr import Gf, UsdGeom

    from roboclock.envs.hospital.backend import INDICATOR_SIZE

    stage, look = backend.stage, backend.appearance
    indicator = next((i for i in backend.layout.indicators if i.name == "indicator"), None)
    dials = stage.GetPrimAtPath("/World/Dials")
    if indicator is None or not dials.IsValid():
        return
    x, y, z = indicator.position
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
    width = INDICATOR_SIZE[0]
    group = look.group("/World/IVBag/DialHousing", (x, y, z + 0.005), rz=yaw)
    look.rounded(group + "/Body", (0.0, 0.0, 0.0), INDICATOR_SIZE, "cabinet_white", 0.004)
    if backend.config["visuals"]["labels"]:
        texture = _text_texture(indicator.label, (1.0, 1.0, 1.0), (0.05, 0.40, 0.20), (768, 160))
        # The card reads from its +y; turned to the dial's facing (+x of the group).
        _quad(stage, group + "/Label", texture, (width / 2 + 0.0015, 0.0, -0.022), (0.066, 0.016),
              rotations=[("z", -90.0)])


def _order_card(backend, info, mats):
    """The order card beside the dial: the instruction, legible from frame 0, facing the camera."""
    if not backend.config["visuals"]["labels"]:
        return
    look, stage = backend.appearance, backend.stage
    x, y = info["order_card_xy"]
    top = info["cart"]["top"]
    toward = info["camera"]["position"]
    yaw = math.degrees(math.atan2(toward[1] - y, toward[0] - x))
    # The group's local +y faces the camera; the card leans back 15 degrees on an acrylic foot.
    group = look.group("/World/IVBag/OrderCard", (x, y, top), rz=yaw - 90.0)
    look.rounded(group + "/Foot", (0.0, -0.010, 0.004), (0.20, 0.05, 0.008), mats["acrylic"], 0.003)
    card = look.group(group + "/Lean", (0.0, 0.0, 0.008), rx=15.0)
    look.rounded(card + "/Board", (0.0, -0.004, 0.058), (0.204, 0.006, 0.112), mats["card"], 0.003)
    text = "START INFUSION\n1  SPIKE + HANG SALINE\n2  PRESS PUMP START"
    texture = _text_texture(text, (0.05, 0.10, 0.20), (1.0, 0.97, 0.80), (1024, 512))
    _quad(stage, card + "/Face", texture, (0.0, -0.0005, 0.058), (0.196, 0.098))
    # A green band on top ties the card to the green request on the dial.
    look.rounded(card + "/Band", (0.0, -0.004, 0.117), (0.204, 0.007, 0.010), mats["band"], 0.002)


def _caster(look, path, x, y, radius=0.03, fork="steel"):
    look.cylinder(path + "_Wheel", (x, y, radius), radius, 0.022, "rubber", axis="Y")
    look.rounded(path + "_Fork", (x, y, 2 * radius + 0.012), (0.03, 0.034, 0.026), fork, 0.003)


def _robot_base(backend, mats):
    """Both X5 columns stand on one wheeled base, parked beside the bed (the columns are physics)."""
    from roboclock.envs.lab.robot import mount_positions

    look, top = backend.appearance, backend.layout.table_height
    mounts = np.array(mount_positions(backend.config, backend.task.task_id))[:, :2]
    lo, hi = mounts.min(0) - 0.075, mounts.max(0) + 0.075
    lo[0] -= 0.10  # the base extends behind the columns, away from the bed
    center, size = ((lo + hi) / 2).tolist(), (hi - lo).tolist()
    # Light grey molded shells like the mount columns, a dark powder-coated skirt with a rubber
    # bumper strip, an aluminum top trim and rubber casters on dark forks.
    look.rounded(
        "/World/IVBag/RobotBase", (*center, (0.08 + top) / 2), (*size, top - 0.08),
        mats["shell"], 0.02,
    )
    look.rounded(
        "/World/IVBag/RobotBumper", (*center, 0.10), (size[0] + 0.02, size[1] + 0.02, 0.05),
        mats["skirt"], 0.012,
    )
    look.rounded(
        "/World/IVBag/RobotBumperStrip", (*center, 0.10), (size[0] + 0.026, size[1] + 0.026, 0.016),
        "rubber", 0.006,
    )
    look.rounded(
        "/World/IVBag/RobotTop", (*center, top - 0.004), (size[0] + 0.006, size[1] + 0.006, 0.008),
        mats["trim"], 0.003,
    )
    # A status strip on the base's bed-facing front.
    look.rounded(
        "/World/IVBag/RobotStrip", (hi[0] + 0.001, center[1], top - 0.09), (0.004, size[1] - 0.08, 0.02),
        "rail", 0.002,
    )
    for k, (x, y) in enumerate(
        ((lo[0] + 0.05, lo[1] + 0.05), (hi[0] - 0.05, lo[1] + 0.05),
         (lo[0] + 0.05, hi[1] - 0.05), (hi[0] - 0.05, hi[1] - 0.05))
    ):
        _caster(look, f"/World/IVBag/RobotCaster_{k}", x, y, 0.035, mats["skirt"])


def _cart(look, cart, mats):
    """A small two-tier stainless prep cart; its top is the support collider's top. The "stainless"
    and "steel" palette names are brushed stainless (appearance.PALETTE_TEXTURES)."""
    (cx, cy), (sx, sy), top = cart["center"], cart["size"], cart["top"]
    look.rounded("/World/IVBag/CartTop", (cx, cy, top - 0.011), (sx, sy, 0.022), "stainless", 0.004)
    # A low lip on the far, near and robot sides (the bag side toward the pole stays open).
    for name, center, size in (
        ("LipFoot", (cx, cy - sy / 2 + 0.004, top + 0.006), (sx, 0.008, 0.012)),
        ("LipBed", (cx + sx / 2 - 0.004, cy, top + 0.006), (0.008, sy, 0.012)),
        ("LipRobot", (cx - sx / 2 + 0.004, cy, top + 0.006), (0.008, sy, 0.012)),
    ):
        look.rounded(f"/World/IVBag/Cart{name}", center, size, "stainless", 0.003)
    shelf_z = 0.34
    look.rounded("/World/IVBag/CartShelf", (cx, cy, shelf_z), (sx - 0.01, sy - 0.01, 0.016), "stainless", 0.004)
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        x, y = cx + dx * (sx / 2 - 0.022), cy + dy * (sy / 2 - 0.022)
        look.cylinder(f"/World/IVBag/CartLeg_{k}", (x, y, (0.085 + top - 0.02) / 2), 0.011, top - 0.105, "steel")
        _caster(look, f"/World/IVBag/CartCaster_{k}", x, y, 0.028)
    # Push handle at the foot end.
    hy = cy - sy / 2 - 0.035
    for k, dx in enumerate((-1, 1)):
        x = cx + dx * (sx / 2 - 0.05)
        look.cylinder(f"/World/IVBag/CartHandlePost_{k}", (x, cy - sy / 2 - 0.018, top - 0.01), 0.008, 0.04, "steel", axis="Y")
    look.cylinder("/World/IVBag/CartHandle", (cx, hy, top - 0.01), 0.011, sx - 0.06, "steel", axis="X")
    # Supplies on the lower shelf: a box of IV administration sets and a spare saline bag.
    # Both stay inside the shelf's footprint (the box used to overhang it toward the robot base).
    look.rounded("/World/IVBag/SupplyBox", (cx - 0.035, cy + 0.05, shelf_z + 0.05), (0.16, 0.12, 0.084), mats["carton"], 0.004)
    look.rounded("/World/IVBag/SpareBag", (cx + 0.06, cy - 0.06, shelf_z + 0.018), (0.12, 0.18, 0.02), "iv_bag", 0.008)


def _iv_stand(backend, info, mats):
    """The floor IV stand model around the pole collider, plus the clamps that carry hook and spike."""
    from roboclock.envs.hospital.props import add_model, model_size

    look, stage = backend.appearance, backend.stage
    px, py = info["pole_xy"]
    c, s = math.cos(STAND_YAW), math.sin(STAND_YAW)
    ax, ay = POLE_AXIS[0] * STAND_SCALE[0], POLE_AXIS[1] * STAND_SCALE[1]
    origin = (px - (c * ax - s * ay), py - (s * ax + c * ay), 0.0)
    if model_size("iv_stand") is not None:
        add_model(stage, "iv_stand", "/World/IVBag/Stand", origin, scale=STAND_SCALE, yaw=STAND_YAW)
    else:  # missing model: a plain pole on a cross base
        look.cylinder("/World/IVBag/StandPole", (px, py, 0.9), 0.016, 1.62, "stainless")
        for k in range(2):
            look.rounded(
                f"/World/IVBag/StandFoot_{k}", (px, py, 0.06),
                (0.50, 0.04, 0.03) if k == 0 else (0.04, 0.50, 0.03), "stainless", 0.01,
            )
    hook = info["hook_center"]
    peg_z = hook[2]
    support_z = info["spike_base"][2] - 0.006
    # Clamp-on brackets: brushed aluminum blocks with black plastic thumb screws.
    for name, z in (("HookClamp", peg_z), ("SpikeClamp", support_z)):
        look.rounded(f"/World/IVBag/{name}", (px, py, z), (0.05, 0.05, 0.036), mats["bracket"], 0.006)
        look.cylinder(f"/World/IVBag/{name}Knob", (px + 0.032, py, z), 0.009, 0.018, mats["knob"], axis="X")
    # The hook's free end turns up 12 mm (inside the eyelet's opening while it is threaded on).
    free_y = hook[1] - info["hook_half_length"]
    look.cylinder("/World/IVBag/HookTip", (hook[0], free_y + 0.004, peg_z + 0.009), 0.0045, 0.018, "stainless")


def _pump(backend, info, mats):
    """The infusion pump on the pole: model, pole clamp, START adapter legend and its screen."""
    from roboclock.envs.hospital.props import add_model, model_size

    look, stage = backend.appearance, backend.stage
    pump = info["pump"]
    ox, oy, base = pump["origin"]
    if model_size("infusion_pump") is None:
        raise FileNotFoundError("Required existing hospital infusion_pump USD is missing")
    add_model(stage, "infusion_pump", "/World/IVBag/Pump", translate=pump["origin"], yaw=math.radians(pump["yaw"]))
    px, py = info["pole_xy"]
    clamp_z = base + 0.137
    # The rotatable pole clamp, turned to grip the pole at the pump's side.
    look.rounded("/World/IVBag/PumpPoleClamp", (px - 0.012, py, clamp_z), (0.044, 0.056, 0.050), mats["bracket"], 0.006)
    look.cylinder("/World/IVBag/PumpPoleClampKnob", (px, py + 0.038, clamp_z), 0.010, 0.022, mats["knob"], axis="Y")
    labels = backend.config["visuals"]["labels"]
    if labels:
        kx, ky, kz = info["start_key"]["origin"]
        texture = _text_texture("START", (1.0, 1.0, 1.0), (0.10, 0.55, 0.22), (512, 160))
        # A legend card standing at the adapter's front edge, facing the foot of the bed (-y).
        _quad(stage, "/World/IVBag/StartLegend", texture, (kx, ky - 0.0315, kz - 0.010), (0.056, 0.0175),
              rotations=[("z", 180.0)])
    # The pump's own display, one card whose material is swapped (RTX shows a rebinding on the
    # next frame; two stacked cards toggled by visibility blend on an episode's last frame):
    # standby, READY / PRESS START from the instruction, then INFUSING once START is held.
    # The model's display sits 0.04 m off its center line, 0.138 m up its +x face.
    c, s = math.cos(math.radians(pump["yaw"])), math.sin(math.radians(pump["yaw"]))
    front, side = PUMP_SIZE[0] / 2 + 0.0015, -0.040
    center = (ox + c * front - s * side, oy + s * front + c * side, base + SCREEN["offset"][1])
    materials = {
        state: _texture_material(stage, f"/World/IVBag/PumpScreen_{state}_material",
                                 _text_texture(text, fg, bg, (384, 512)), emissive=True)
        for state, (text, fg, bg) in SCREEN_TEXT.items()
    }
    path = "/World/IVBag/PumpScreen"
    # The card reads from its +y; turned to face the pump's front.
    _quad(stage, path, _text_texture(*SCREEN_TEXT["idle"], (384, 512)), center, SCREEN["size"],
          rotations=[("z", pump["yaw"] - 90.0)], emissive=True)
    backend._iv_bag_screen = (backend.task, "idle")
    backend._iv_bag_screen_materials = (path, materials)
    _bind_screen(backend, "idle")


def _bind_screen(backend, state):
    from pxr import UsdShade

    path, materials = backend._iv_bag_screen_materials
    UsdShade.MaterialBindingAPI.Apply(backend.stage.GetPrimAtPath(path)).Bind(materials[state])


def _pump_screen(backend, key=None):
    """Show the display for the device's progress (visual only; the task scores the key).

    The pump answers START as soon as it is held down after the bag hangs, so the response also
    shows on the final frame of an episode that ends at the scored press.
    """
    task = getattr(backend, "task", None)
    current = getattr(backend, "_iv_bag_screen", None)
    if task is None or current is None or current[0] is not task:
        return
    held = (
        key is not None
        and key["contact"]
        and key["depression_m"] >= START_PRESS
        and getattr(task, "hanging_time", None) is not None
        and not task.physical_failures
    )
    if task.event_time is None:
        state = "idle"
    else:
        state = "done" if getattr(task, "start_time", None) is not None or held else "cue"
    if state == current[1]:
        return
    _bind_screen(backend, state)
    backend._iv_bag_screen = (task, state)


def _line(look, info):
    """Drip chamber under the spike clamp, the line down into the pump and on over the rail to the bed."""
    from roboclock.envs.hospital.scene_meshes import tube_arrays

    sx, sy, base_z = info["spike_base"]
    holder_bottom = base_z - 0.014
    chamber = np.array([sx, sy, holder_bottom - 0.032])
    look.cylinder("/World/IVBag/DripChamber", chamber, 0.011, 0.052, look.glass())
    look.cylinder("/World/IVBag/DripFluid", chamber + [0, 0, -0.014], 0.009, 0.020, "liquid")
    look.cylinder("/World/IVBag/DripCap", chamber + [0, 0, -0.029], 0.0075, 0.008, "cap_white")
    ox, oy, base = info["pump"]["origin"]
    top = info["pump"]["top"]
    # The channel module is the pump's pole-side (+x) end: the line drops into its top and leaves
    # its front face near the bottom, then runs past the pole and over the bed rail.
    channel_x, channel_y = ox + 0.082, oy - 0.035
    start = chamber + [0, 0, -0.033]
    inlet = np.array([channel_x, channel_y, top + 0.004])
    control = [start, start + [0.0, 0.0, -0.025], [(sx + channel_x) / 2, (sy + channel_y) / 2, top + 0.045],
               inlet + [0.0, 0.0, 0.02], inlet]
    look.mesh("/World/IVBag/LineIn", tube_arrays(control, 0.0024, sides=8), (0, 0, 0), "liquid")
    front = oy - PUMP_SIZE[0] / 2 - 0.003
    outlet = np.array([channel_x + 0.01, front, base + 0.03])
    control = [
        outlet,
        [outlet[0] + 0.03, front - 0.02, base + 0.025],
        [0.42, front - 0.02, base + 0.01],
        [0.55, front - 0.01, base - 0.06],
        [0.66, front + 0.02, 0.95],
        [0.72, front + 0.04, 1.00],
        [0.80, front + 0.06, 0.93],
        [0.92, front + 0.10, 0.90],
        [1.04, front + 0.12, 0.90],
    ]
    look.mesh("/World/IVBag/Line", tube_arrays(control, 0.0024, sides=8), (0, 0, 0), "liquid")


def _texture_material(stage, path, texture, emissive=False, cutout=False):
    """A UsdPreviewSurface material showing `texture` through the mesh's "st" coordinates.
    (Copied from bedside/pump_panel.py, which another task owns.)"""
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


def _text_texture(text, fg, bg, size=(512, 192)):
    """A cached label image: bold lines of text ("\n"), `bg` None for a transparent background.
    (Copied from bedside/pump_panel.py, which another task owns.)"""
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr(("b6", text, fg, bg, size)).encode()).hexdigest()[:20]
    path = cache / f"b6_{key}.png"
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


def _quad(stage, path, texture, center, size, rotations=(), emissive=False, cutout=False):
    """A textured card in its local XZ plane, read from local +y; `rotations` are (axis, deg).
    (Copied from bedside/pump_panel.py, which another task owns.)"""
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


def _flat_text(stage, path, text, center, size, yaw=180.0):
    """White text printed flat on a key cap. Card +y (its reading side) turned up: with yaw 180 the
    text is upright for a reader on the -y side, with yaw 90 for a reader on the -x side."""
    texture = _text_texture(text, (1.0, 1.0, 1.0), None, (512, 256))
    return _quad(stage, path, texture, center, size, rotations=[("z", yaw), ("x", 90.0)], cutout=True)


def _camera(backend):
    """Frame the robot hand, bag, spike, hook, pole, pump, dial and order card, once per reset."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["iv_bag"]["camera"]
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
