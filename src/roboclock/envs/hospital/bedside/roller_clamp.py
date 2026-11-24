"""Physical B5: hang the prepared saline bag on the IV stand, then open the line's roller clamp.

Ward placement (scene-02 left bay, assets/hospital/ward_02_preview_left.usdc): the robot is a
wheeled dual-arm base parked in the aisle between the bed and the wall counter, facing the bed's
side. The ward turns 180 degrees about the robot side, so the bed's foot-section side rail runs
along +y about 0.60 m in front of arm 0's mount (the rig's right arm). Along that side of the bed stand:
- a bedside cabinet (procedural) to the robot's front right; the prepared 0.9% saline bag (already
  spiked, drip chamber primed, line attached) stands upright in a cradle on its top;
- the IV stand (local CC-BY `iv_stand` model) to the front left, its pole about 0.56 m from the
  mount, with a clamp-on hook arm at a height the hand reaches while carrying the bag (the X5's
  top-down hand cannot reach the stand's 1.7 m top hooks);
- a line organizer clipped onto the side rail straight ahead. The line from the bag's drip chamber
  runs through the organizer's roller clamp, then over the rail onto the bed. The roller starts
  CLOSED (toward the robot) and opens by sliding toward the bed (+x).
The robot starts tucked at its own column and reaches out only after the instruction.

Mechanisms are composed from existing ones: the roller is the B5 passive prismatic slide with end
stops (viscous drive, released retention), and the bag is the shared rigid `blood_bag` pouch with a
four-bar eyelet hung on a horizontal peg, with A4's reinforced grasp tab above the eyelet (copied
here from bedside/iv_bag.py). The line itself is visual; its slack span from the drip chamber to
the organizer follows the measured bag pose. No tubing, fluid or flow physics is simulated.

The X5's top-down tool reaches at most ~0.53 m horizontally from its mount (Lula IK scan, hand
0.30 m below to 0.03 m above the mount); the hook and roller hand poses sit 0.46-0.50 m out with an
IK margin for every seed's shift (the radial shift is scaled by 0.4), while the IV pole with the
hook arm's clamp stands 0.56-0.58 m and the rail 0.59-0.60 m away. The arm mount is raised to
1.22 m (the rig height, robot.rig.tasks.B5) so the hook and the rail roller both fall in that band.

Public helpers (reused by other tasks; keep their names and signatures): `add_roller`,
`author_roller`, `roller_controls`, `push_roller`, `text_panel`, `backend_labels`, `hang_geometry`,
plus the module hooks `build_layout`, `author`, `measure` and `command`.
"""

import math

import numpy as np

from roboclock.envs.hospital.bedside.ward import furniture
from roboclock.envs.hospital.tasks.bedside.roller_clamp import (
    END_TOLERANCE,
    INSTRUCTION_TEXT,
    TRAVEL,
)

ROLLER_NAME = "roller_0"
ROLLER_ASSET = "iv_roller"
BAG_NAME = "bag_0"
BAG_ASSET = "blood_bag"
# The exported X5 fingertips extend 12.57 mm below grasp_center. The pushing pose
# keeps their lower edge above the holder rails while contacting the roller face.
PUSH_HEIGHT = 0.009
READY_OFFSET = 0.028
PUSH_OFFSET = 0.020

# --- ward placement (robot frame, before the seed's common shift) ------------------------------
WARD_USD = "assets/hospital/ward_02_preview_left.usdc"
# World = Rz(180) * ward + translate. The bed's high-x side rail (ward x 1.757-1.787) lands at
# x 0.600-0.630; its foot section (ward y -1.52..-0.97, top bar z 0.84-0.878) at y -0.08..0.47;
# the bed's foot end at y -0.43, the ultrasound cart at y >= 0.65 and the wall counter's front at
# x -0.49 behind the robot. The ward sits 3 mm high (its floor slab is hidden, see ward.py).
WARD_TRANSLATE = (2.387, -1.05, 0.003)
WARD_YAW = 180.0
# The common rig (hospital/rig.py) stands the idle arm 0.60 m to +y of the driving arm, against
# the ward's ultrasound cart (robot y 0.63..1.52, screen up to 1.55 m): the cart is pushed 0.30 m
# further along the bed (+y), clear of the wheeled base and the raised idle arm.
WARD_MOVE = {name: (0.0, 0.30, 0.0) for name in furniture("ultrasound_cart")}
RAIL_X = (0.600, 0.630)  # outer and inner face of the bed's side rail
RAIL_TOP = 0.878 + WARD_TRANSLATE[2]
MATTRESS_TOP = 0.862 + WARD_TRANSLATE[2]
FLOOR_Z = WARD_TRANSLATE[2]
# The radial (x) seed shift is reduced: the hook and roller hand poses sit near the X5's reach.
SHIFT_SCALE = (0.4, 1.0)

# Roller organizer on the rail: roller centre at CLOSED, centre height.
ROLLER_XY = (0.450, 0.0)
ROLLER_Z = 0.875
# Bedside cabinet (procedural): its top is the task bench (b.z) and the support collider.
CABINET_CENTER, CABINET_SIZE = (0.415, -0.16), (0.33, 0.44)
BAG_XY = (0.35, -0.15)  # bag cradle on the cabinet
CRADLE_HEIGHT = 0.075  # cradle shoulders: the drip chamber hangs between them
# IV stand pole and the clamp-on hook arm (a horizontal peg along -y from the pole).
POLE_XY = (0.45, 0.35)
PEG_Z = 1.125  # peg centre height (arm-0 mount at 1.22: robot.rig.tasks.B5)
PEG_HALF_LENGTH, PEG_HALF = 0.11, 0.006
HANG_Y = 0.19  # bag centre on the peg (0.06 m in from the peg's free end at y 0.13)
# Visual drip set below the bag body (bag frame, metres below the body's bottom face).
PORT_LENGTH, CHAMBER_LENGTH = 0.014, 0.040
OUTLET_DEPTH = PORT_LENGTH + 0.012 + CHAMBER_LENGTH + 0.008  # body bottom to the chamber outlet
LINE_LENGTH = 0.42  # slack span from the chamber outlet to the organizer inlet
LINE_SEGMENTS = 18
# Bedside order console on the cabinet top's front edge: the response dial and the instruction
# screen beside it (both below the camera's line of sight to the bag and clear of the hand).
# RoboDojo rig (2026-11-04): the dial moved off the cabinet onto its own floor stand on the bag's
# +y side, beside the IV stand: anywhere on the cabinet front the tucked arm 0 stands between it
# and the rig's head camera (70-80 % of the episode, from reset).
DIAL_XY = (0.32, 0.25)
DISPLAY_XY, DISPLAY_SIZE, DISPLAY_TILT = (0.475, -0.335), (0.20, 0.12), 18.0
# IV stand model (0.60 x 0.64 x 1.80 m): base scaled to 0.45 m, turned so its legs clear the
# cabinet, the robot base and the bed's frame.
STAND_SCALE = (0.75, 0.75, 1.0)
STAND_YAW = np.radians(20.0)
STAND_POLE = (0.014, 0.027)  # pole axis in model coordinates
# Tucked rest pose of the left hand, from its mount: folded at its own column.
TUCK = (0.14, 0.02, 0.03)
# From the bed's foot end: robot, cabinet and bag, the roller organizer, the hook and the bed.
CAMERA = {"position": [0.95, -0.75, 1.60], "look_at": [0.40, 0.02, 1.0], "focal_length": 22.0}


def _mount(config, task_id="B5"):
    from roboclock.envs.lab.robot import mount_positions

    return np.asarray(mount_positions(config, task_id)[0], dtype=float)


# --- roller helpers (reusable) -------------------------------------------------------------------


def add_roller(b, x, y, center_z, start=0.0, name=ROLLER_NAME, color=(0.15, 0.48, 0.80)):
    """Declare one passive roller and its slotted holder tray; returns the CLOSED-end origin.

    The roller slides along +x from `x` (position 0) to `x + TRAVEL`; `start` is its initial
    position along the slot. Coordinates are robot frame before the seed shift.
    """
    b.item(name, ROLLER_ASSET, (x + start, y), center_z - 0.011 - 0.002, color=color, yaw=0.0)
    origin = b.objects[name].position.copy()
    origin[0] -= start
    base_z = center_z - 0.042
    b.fixture(
        f"{name}_holder_base" if name != ROLLER_NAME else "roller_holder_base",
        (x + TRAVEL / 2, y),
        (0.100, 0.052, 0.010),
        material="cabinet_white",
        z=base_z,
    )
    prefix = "roller" if name == ROLLER_NAME else name
    for sign, suffix in ((-1, "front"), (1, "back")):
        b.fixture(
            f"{prefix}_track_{suffix}",
            (x + TRAVEL / 2, y + sign * 0.018),
            (0.074, 0.005, 0.019),
            material="cabinet_white",
            z=center_z - 0.029,
        )
    # The physical joint limits are the end stops; caps make the slot readable.
    for dx, suffix in ((-0.018, "closed"), (TRAVEL + 0.018, "open")):
        b.fixture(
            f"{prefix}_stop_{suffix}",
            (x + dx, y),
            (0.007, 0.041, 0.019),
            material="cabinet_white",
            z=center_z - 0.029,
        )
    return origin


def author_roller(backend, origin, name=ROLLER_NAME, joint_path="/World/Joints/iv_roller_slide"):
    """Knurled wheel visuals and the passive prismatic slide (viscous drive, physical end stops)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage, look = backend.stage, backend.appearance
    path = f"/World/Objects/{name}"
    root = stage.GetPrimAtPath(path)
    if not root.IsValid():
        raise ValueError("The roller rigid body must exist before author_roller()")
    for child in root.GetChildren():
        # Keep the backend's conservative box collider and remove its generic box appearance.
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            UsdGeom.Imageable(child).MakeInvisible()
    look.cylinder(path + "/KnurledWheel", (0.0, 0.0, 0.0), 0.011, 0.020, "cap_blue", axis="Y")
    for sign in (-1, 1):
        look.cylinder(
            path + f"/Axle_{'L' if sign < 0 else 'R'}",
            (0, sign * 0.011, 0),
            0.003,
            0.003,
            "cap_white",
            axis="Y",
        )
    for index in range(24):
        theta = 2 * np.pi * index / 24
        offset = [0.0108 * np.sin(theta), 0.0, 0.0108 * np.cos(theta)]
        tooth = look.rounded(
            path + f"/Rib_{index:02d}", offset, [0.0012, 0.020, 0.0014], "cap_blue", 0.00025
        )
        UsdGeom.Xformable(tooth.GetPrim()).AddRotateYOp().Set(float(theta * 180 / np.pi))
    origin = np.asarray(origin, dtype=float)
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(TRAVEL)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*origin.tolist()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
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


def roller_controls(origin, roller):
    """Opening and track validity derived only from the observed rigid roller pose."""
    delta = roller.position - np.asarray(origin, dtype=float)
    position = float(delta[0])
    return {
        "roller_position_m": position,
        "roller_speed_m_s": float(np.linalg.norm(roller.velocity)),
        "roller_axis_error_m": float(np.linalg.norm(delta[1:])),
        "roller_contact": bool(roller.robot_contact or roller.grasped),
        "roller_grasped": bool(roller.grasped),
        "roller_open_fraction": float(np.clip(position / TRAVEL, 0.0, 1.0)),
    }


def push_roller(planner, state, origin, position, target, speed=0.24):
    """Fingertip push of the roller's end face toward `target`, then a withdrawal once there.

    The holder stabilizes the line, so a push is sufficient in either direction (pinching the
    miniature wheel also pinched its track in the initial PhysX diagnostic).
    """
    origin = np.asarray(origin, dtype=float)
    direction = 1.0 if target >= position else -1.0
    planner.open_width = planner.close_width = 0.004
    planner.gentle = True
    if abs(position - target) <= END_TOLERANCE:
        goal = origin + [target - direction * (READY_OFFSET + 0.012), 0.0, PUSH_HEIGHT + 0.025]
        return planner.finish(state, goal, False, speed=0.20)
    goal = origin + np.array([target - direction * PUSH_OFFSET, 0.0, PUSH_HEIGHT])
    return planner.finish(state, goal, False, speed=speed)


# --- bag helpers --------------------------------------------------------------------------------


def hang_geometry(layout, obj):
    """Measured support of the released bag by its eyelet on the hook peg."""
    from roboclock.physics.geometry import rotation_matrix, upright_error

    info = layout.info["roller_clamp"]
    hook, bag = info["hook"], info["bag"]
    rotation = rotation_matrix(obj.orientation)
    center = np.asarray(hook["center"])
    half_length, half_peg = hook["half_length"], hook["half_thickness"]
    hook_y = float(np.clip(obj.position[1], center[1] - half_length, center[1] + half_length))
    local = rotation.T @ (np.array([center[0], hook_y, center[2]]) - obj.position)
    eye = bag["eyelet"]
    upper_inner = bag["body_size"][2] / 2 + eye["height"] - eye["bar"]
    tilt = float(upright_error(obj.orientation))
    on_peg = abs(obj.position[1] - center[1]) + eye["thickness"] / 2 < half_length
    through = abs(local[0]) + half_peg < eye["width"] / 2 - eye["bar"]
    seated = abs(local[2] + half_peg - upper_inner) <= 0.005
    contact = HOOK_CONTACT in obj.fixture_contacts
    speed = float(np.linalg.norm(obj.velocity))
    hanging = bool(
        on_peg
        and through
        and seated
        and tilt <= HANG_TILT
        and abs(rotation[1, 1]) >= math.cos(HANG_TILT)
        and contact
        and not obj.grasped
        and not obj.robot_contact
        and speed < HANG_SPEED
    )
    return {
        "on_peg": bool(on_peg and through),
        "seated": bool(seated),
        "hook_contact": bool(contact),
        "tilt_rad": tilt,
        "speed_m_s": speed,
        "hanging": hanging,
    }


HOOK_CONTACT = "fixture:iv_hook_peg"
HANG_TILT = 0.15
HANG_SPEED = 0.06


def build_layout(builder):
    """Declare the bag in its cradle, the hook peg and pole, and the roller on its rail organizer."""
    b = builder
    b.shift = b.shift * np.asarray(SHIFT_SCALE)
    shift = [float(v) for v in b.shift]
    # Bedside cabinet top = the task bench.
    top = b.z
    spec = b.catalog[BAG_ASSET]
    bx, by, bz = spec["size"]
    eye = spec["eyelet"]
    # 1. The prepared bag stands in a two-shoulder cradle; its drip set hangs between them.
    for side in (-1, 1):
        b.fixture(
            f"bag_cradle_{'left' if side < 0 else 'right'}",
            (BAG_XY[0] + side * 0.036, BAG_XY[1]),
            (0.022, 0.040, CRADLE_HEIGHT),
            "polymer",
        )
    b.item(
        BAG_NAME,
        BAG_ASSET,
        BAG_XY,
        top + CRADLE_HEIGHT - 0.002,
        color=(0.86, 0.93, 0.96),
        label="SALINE",
        yaw=0.0,
    )
    # 2. The IV stand: an invisible pole collider and the clamp-on hook peg along -y.
    b.fixture(
        "iv_stand_pole", POLE_XY, (0.032, 0.032, 1.62), "stainless", z=FLOOR_Z + 0.10, visible=False
    )
    peg_xy = (POLE_XY[0], POLE_XY[1] - PEG_HALF_LENGTH)
    b.fixture(
        "iv_hook_peg",
        peg_xy,
        (2 * PEG_HALF, 2 * PEG_HALF_LENGTH, 2 * PEG_HALF),
        "stainless",
        z=PEG_Z - PEG_HALF,
        visible=False,
    )
    # 3. The roller organizer clipped on the bed rail, the roller at CLOSED.
    x, y = ROLLER_XY
    origin = add_roller(b, x, y, ROLLER_Z, 0.0)
    base_z = ROLLER_Z - 0.042
    tray_end = x + TRAVEL / 2 + 0.050
    jaw = 0.008
    b.fixture(
        "roller_holder_arm",
        ((tray_end + RAIL_X[0] - jaw) / 2, y),
        (RAIL_X[0] - jaw - tray_end, 0.030, 0.010),
        material="cabinet_white",
        z=base_z,
    )
    b.fixture(
        "roller_clamp_outer",
        (RAIL_X[0] - jaw / 2, y),
        (jaw, 0.040, RAIL_TOP + 0.012 - base_z),
        material="cabinet_white",
        z=base_z,
    )
    b.fixture(
        "roller_clamp_top",
        ((RAIL_X[0] + RAIL_X[1]) / 2, y),
        (RAIL_X[1] - RAIL_X[0] + 2 * jaw, 0.040, 0.012),
        material="cabinet_white",
        z=RAIL_TOP,
    )
    b.fixture(
        "roller_clamp_inner",
        (RAIL_X[1] + jaw / 2, y),
        (jaw, 0.040, 0.030),
        material="cabinet_white",
        z=RAIL_TOP + 0.012 - 0.030,
    )
    # 4. The response dial on the cabinet; the instruction display stands beside it.
    b.indicator("indicator", DIAL_XY, "B5")
    # Hanging: the eyelet's top bar rests on the peg's top face.
    hang = np.array(
        [*b.xy(POLE_XY[0], HANG_Y), PEG_Z + PEG_HALF - (bz / 2 + eye["height"] - eye["bar"])]
    )
    grasp_dz = bz / 2 + eye["height"] + 0.022  # A4's reinforced tab above the eyelet
    mount = _mount(b.config, getattr(b.task, "task_id", "B5"))
    camera = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in CAMERA.items()}
    b.info["bedside_task"] = "roller_clamp"
    b.info["roller_clamp"] = {
        "origin": origin,
        "travel": TRAVEL,
        "mode": "open",
        "shift": shift,
        "camera": camera,
        "instruction": INSTRUCTION_TEXT,
        "bag": {
            "name": BAG_NAME,
            "body_size": [bx, by, bz],
            "eyelet": dict(eye),
            "grasp_offset": [0.0, 0.0, grasp_dz],
            "outlet": [0.0, 0.0, -bz / 2 - OUTLET_DEPTH],
            "cradle_top": top + CRADLE_HEIGHT,
            "cradle_xy": [*map(float, b.xy(*BAG_XY))],
        },
        "hook": {
            "center": [*map(float, b.xy(*peg_xy)), PEG_Z],
            "half_length": PEG_HALF_LENGTH,
            "half_thickness": PEG_HALF,
            "tip_y": float(b.xy(*peg_xy)[1] - PEG_HALF_LENGTH),
            "hang_position": hang.tolist(),
            "pole_xy": [*map(float, b.xy(*POLE_XY))],
        },
        "inlet": (origin + [-0.047, 0.0, -0.019]).tolist(),
        "cabinet": {
            "center": [*map(float, b.xy(*CABINET_CENTER))],
            "size": list(CABINET_SIZE),
            "top": top,
        },
        "display": [*map(float, b.xy(*DISPLAY_XY)), top + DISPLAY_SIZE[1] / 2 + 0.012],
    }
    # The ward moves with the seed's shift so the organizer stays clamped on the rail.
    b.info["ward_pose"] = {
        "usd": WARD_USD,
        "translate": [
            WARD_TRANSLATE[0] + shift[0],
            WARD_TRANSLATE[1] + shift[1],
            WARD_TRANSLATE[2],
        ],
        "yaw": WARD_YAW,
        "move": dict(WARD_MOVE),
    }
    b.info["hide_bench"] = True
    # The invisible support collider is the bedside cabinet's top.
    b.info["work_surface"] = {
        "center": [*map(float, b.xy(*CABINET_CENTER)), top - 0.05],
        "size": [*CABINET_SIZE, 0.10],
    }
    # At rest: the hand folded at its own column; it reaches out only after the instruction.
    b.info["initial_ee_position"] = (mount + np.asarray(TUCK)).tolist()
    b.info["initial_ee_yaw"] = 0.0
    b.info["initial_gripper_width"] = TAB_OPEN


TAB_OPEN, TAB_CLOSE = 0.024, 0.009  # finger targets around the 30 mm grasp tab


# --- authoring ----------------------------------------------------------------------------------


def author(backend):
    """Roller slide, bag grasp tab and drip set, line, organizer, stand, cabinet and display."""
    info = backend.layout.info["roller_clamp"]
    origin = np.asarray(info["origin"])
    mats = _materials(backend.appearance)
    joint = author_roller(backend, origin)
    _author_bag(backend, info, mats)
    shift = np.array([*info.get("shift", (0.0, 0.0)), 0.0])
    _author_line(backend, origin, shift)
    _organizer_labels(backend, origin)
    _dress_ward(backend, info, mats)
    _display(backend, info, mats)
    _texture_fixtures(backend, mats)
    return joint


def _materials(look):
    """Textured PBR materials (CC0 sets in hospital/assets/textures) for B5's own parts.

    Shared palette names ("stainless", "steel", "rubber", "cabinet_white", ...) are textured
    centrally; these cover the parts whose look is specific to B5.
    """
    return {
        # Bedside cabinet: matte white laminate carcass, light blue fronts like the ward counter.
        "cabinet": look.textured("b5_cabinet_laminate", "plastic_matte", color=(1.0, 1.0, 0.99)),
        "cabinet_top": look.textured(
            "b5_cabinet_top", "plastic_matte", color=(0.93, 0.94, 0.95), roughness=0.45
        ),
        "cabinet_front": look.textured(
            "b5_cabinet_front", "plastic_matte", color=(0.62, 0.80, 0.90), roughness=0.45
        ),
        "handle": look.textured("b5_cabinet_handle", "aluminum_brushed"),
        # Saline bag: glossy pale blue PVC pouch (see _author_bag).
        "saline": _glossy(
            look.material("b5_saline_pvc", color=[0.74, 0.85, 0.93], roughness=0.06), 0.9
        ),
        # Bag cradle: molded white plastic shoulders on blue silicone pads. The plastic_white map
        # is light grey (mean linear albedo ~0.39) and the silicone map dark (~0.033): the tints
        # scale them to the target colors (white; a mid blue).
        "cradle": look.textured("b5_cradle_plastic", "plastic_white", color=(2.25, 2.2, 2.1)),
        "cradle_pad": look.textured(
            "b5_cradle_pad", "silicone", color=(2.6, 6.5, 14.0), roughness=0.6
        ),
        # Clamp-on hook arm: aluminum clamp block on the pole.
        "clamp": look.textured("b5_hook_clamp", "aluminum_brushed"),
        # Line organizer on the rail: white molded tray, a grey clip around the rail.
        "organizer": look.textured("b5_organizer_white", "plastic_white", color=(2.25, 2.2, 2.1)),
        "rail_clip": look.textured("b5_rail_clip", "plastic_matte", color=(0.55, 0.58, 0.62)),
        # Order screen: powder-coated housing.
        "housing": look.textured("b5_screen_housing", "powder_coat_dark"),
        # Robot base: shells match the central mount-column coat (texture_robot_mounts).
        "shell": look.textured("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90)),
        "cover": look.textured("b5_robot_cover", "plastic_matte", color=(0.70, 0.73, 0.76)),
        "skirt": look.textured("b5_robot_skirt", "powder_coat_dark"),
        "trim": look.textured("b5_robot_trim", "aluminum_brushed"),
        # E-stop: red mushroom on a yellow collar (tint = target linear color / map mean).
        "estop": look.textured("b5_estop_red", "plastic_matte", color=(0.75, 0.008, 0.006)),
        "estop_collar": look.textured("b5_estop_yellow", "plastic_matte", color=(1.13, 0.73, 0.012)),
    }


def _glossy(material, specular_level):
    """Raise an OmniPBR material's specular level (wet, glossy plastic)."""
    shaders = getattr(material, "shaders_list", None)
    if shaders:
        from pxr import Sdf

        shaders[0].CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(specular_level)
    return material


def _bind(stage, path, material, stronger=False):
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return False
    strength = (
        UsdShade.Tokens.strongerThanDescendants if stronger else UsdShade.Tokens.weakerThanDescendants
    )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, strength)
    return True


def _texture_fixtures(backend, mats):
    """The backend draws fixtures with palette materials; give B5's visible ones their own."""
    stage = backend.stage
    for name, key in (
        ("bag_cradle_left", "cradle"),
        ("bag_cradle_right", "cradle"),
        ("roller_holder_base", "organizer"),
        ("roller_track_front", "organizer"),
        ("roller_track_back", "organizer"),
        ("roller_stop_closed", "organizer"),
        ("roller_stop_open", "organizer"),
        ("roller_holder_arm", "organizer"),
        ("roller_clamp_outer", "rail_clip"),
        ("roller_clamp_top", "rail_clip"),
        ("roller_clamp_inner", "rail_clip"),
    ):
        _bind(stage, f"/World/Fixtures/{name}_Visual", mats[key])
    _bind(stage, "/World/Bedside/RollerLine/PlateBacker", mats["organizer"])


def _author_bag(backend, info, mats):
    """A4's reinforced grasp tab (one collider on the free pouch), saline look and a drip set."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage, look = backend.stage, backend.appearance
    path = f"/World/Objects/{BAG_NAME}"
    body = stage.GetPrimAtPath(path)
    if not body.IsValid():
        raise ValueError("Author the dynamic IV bag before its grasp tab")
    bag = info["bag"]
    bx, by, bz = bag["body_size"]
    eye_top = bz / 2 + bag["eyelet"]["height"]
    tab = UsdGeom.Cube.Define(stage, f"{path}/TrainerGripTab")
    tab.CreateSizeAttr(1.0)
    tab.AddTranslateOp().Set((0.0, 0.0, eye_top + 0.015))
    tab.AddScaleOp().Set((0.024, 0.030, 0.030))
    UsdPhysics.CollisionAPI.Apply(tab.GetPrim())
    binding = UsdShade.MaterialBindingAPI.Apply(tab.GetPrim())
    binding.Bind(look.material("cap_blue").material)
    physics_material = UsdShade.Material.Get(stage, f"/World/Materials/{BAG_NAME}")
    if physics_material:
        binding.Bind(physics_material, materialPurpose="physics")
    if hasattr(backend, "_collision_settings"):
        backend._collision_settings(tab.GetPrim())
    # Keep the supplied mesh's silhouette, replace the blood texture with saline plastic: a glossy,
    # pale blue PVC pouch full of clear fluid. A see-through OmniGlass film was tried, but the
    # real-time renderer composites it over the printed label in front of it (the label vanished)
    # and fractional opacity renders as a cutout, so the pouch is opaque with strong reflections.
    saline = mats["saline"].material
    for prim in Usd.PrimRange(body):
        if prim.IsA(UsdGeom.Mesh) or prim.GetName() in ("Pouch", "WeldTop", "WeldBottom"):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                saline, UsdShade.Tokens.strongerThanDescendants
            )
    label = stage.GetPrimAtPath(path + "/BagLabel")
    if label.IsValid():
        UsdGeom.Imageable(label).MakeInvisible()
    if backend_labels(backend):
        for side, suffix in ((-1, "Back"), (1, "Front")):
            card = look.label(
                path + "/SalineLabel" + suffix,
                "SALINE 0.9%",
                [0.0, side * (by / 2 + 0.001), 0.0],
                [0.084, 0.042],
                (0.15, 0.42, 0.65),
            )
            if side < 0 and card is not None:
                card.AddRotateZOp().Set(180.0)  # a label card reads from +y; this face looks -y
    # Spiked outlet port, spike and primed drip chamber hang under the body (visual only).
    bottom = -bz / 2
    look.cylinder(
        path + "/OutletPort", (0, 0, bottom - PORT_LENGTH / 2), 0.0055, PORT_LENGTH, "cap_white"
    )
    look.cylinder(
        path + "/SpikeHub", (0, 0, bottom - PORT_LENGTH - 0.006), 0.0075, 0.012, "cap_white"
    )
    chamber = bottom - PORT_LENGTH - 0.012 - CHAMBER_LENGTH / 2
    look.cylinder(path + "/DripChamber", (0, 0, chamber), 0.0105, CHAMBER_LENGTH, look.glass())
    look.cylinder(path + "/DripFluid", (0, 0, chamber - 0.010), 0.0085, 0.018, "liquid")
    look.cylinder(
        path + "/ChamberCap",
        (0, 0, chamber - CHAMBER_LENGTH / 2 - 0.004),
        0.0075,
        0.008,
        "cap_white",
    )


class _SlackLine:
    """The line's slack span from the bag's chamber outlet to the organizer inlet.

    Visual only: unit cylinders re-posed from the measured bag pose (a Bezier curve that leaves the
    chamber downward, sags with the spare length and enters the organizer along +x).
    """

    def __init__(self, stage, look, path, inlet, radius=0.0018, count=LINE_SEGMENTS):
        from pxr import UsdGeom, UsdShade

        self.inlet = np.asarray(inlet, dtype=float)
        self.ops = []
        material = look.material("iv_bag")
        for index in range(count):
            shape = UsdGeom.Cylinder.Define(stage, f"{path}/Slack_{index:02d}")
            shape.CreateRadiusAttr(float(radius))
            shape.CreateHeightAttr(1.0)
            shape.CreateAxisAttr("Z")
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material.material)
            self.ops.append((shape.AddTranslateOp(), shape.AddOrientOp(), shape.AddScaleOp()))
        self.last = None

    def points(self, outlet, down, count):
        start, end = np.asarray(outlet, dtype=float), self.inlet
        chord = float(np.linalg.norm(end - start))
        sag = 0.5 * math.sqrt(max(LINE_LENGTH**2 - chord**2, 0.0))
        p1 = start + down * max(0.05, 0.35 * chord) + [0.0, 0.0, -0.6 * sag]
        p2 = end - np.array([max(0.05, 0.35 * chord), 0.0, 0.0]) + [0.0, 0.0, -0.6 * sag]
        s = np.linspace(0.0, 1.0, count + 1)[:, None]
        return (
            (1 - s) ** 3 * start + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s**2 * p2 + s**3 * end
        )

    def update(self, position, orientation):
        from pxr import Gf

        from roboclock.physics.geometry import rotation_matrix

        pose = np.concatenate([position, orientation])
        if self.last is not None and np.max(np.abs(pose - self.last)) < 5e-4:
            return
        self.last = pose
        rotation = rotation_matrix(orientation)
        outlet = position + rotation @ self.outlet_local
        points = self.points(outlet, -rotation[:, 2], len(self.ops))
        for (translate, orient, scale), a, b in zip(self.ops, points[:-1], points[1:]):
            delta = b - a
            length = float(np.linalg.norm(delta))
            quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*delta.tolist())).GetQuat()
            translate.Set(Gf.Vec3d(*((a + b) / 2).tolist()))
            orient.Set(Gf.Quatf(quat))
            scale.Set(Gf.Vec3f(1.0, 1.0, max(length, 1e-4) * 1.04))


def _spline(points, samples=6):
    """Catmull-Rom points through the given control points (ends repeated)."""
    p = [np.asarray(points[0])] + [np.asarray(q) for q in points] + [np.asarray(points[-1])]
    out = [p[1]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for k in range(1, samples + 1):
            t = k / samples
            out.append(
                0.5
                * (
                    2 * p1
                    + (p2 - p0) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                    + (3 * p1 - p0 - 3 * p2 + p3) * t**3
                )
            )
    return out


def _author_line(backend, origin, shift=np.zeros(3)):
    """The line: slack span from the bag, through the organizer, over the rail onto the bed."""
    from pxr import Gf, UsdGeom

    look = backend.appearance
    prefix = "/World/Bedside/RollerLine"
    info = backend.layout.info["roller_clamp"]

    def segment(name, first, second, radius=0.0018, material="iv_bag"):
        first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        delta = second - first
        part = look.cylinder(
            prefix + "/" + name, (first + second) / 2, radius, np.linalg.norm(delta), material
        )
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*delta.tolist())).GetQuat()
        UsdGeom.Xformable(part.GetPrim()).AddOrientOp().Set(Gf.Quatf(rotation))

    x, y, z = origin
    # Slack span from the drip chamber: re-posed from the measured bag pose (measure()).
    slack = _SlackLine(backend.stage, look, prefix, info["inlet"])
    slack.outlet_local = np.asarray(info["bag"]["outlet"], dtype=float)
    bag = backend.layout.objects[BAG_NAME]
    slack.update(np.asarray(bag.position, dtype=float), np.asarray(bag.orientation, dtype=float))
    backend._b5_slack_line = slack
    # Through the organizer under the roller, out of its open end, over the rail onto the mattress.
    rail_in, rail_out = RAIL_X[0] + shift[0], RAIL_X[1] + shift[0]
    downstream = [
        [x - 0.047, y, z - 0.019],
        [x + TRAVEL + 0.036, y, z - 0.019],
        [rail_in - 0.020, y, RAIL_TOP + 0.006],
        [rail_in - 0.006, y, RAIL_TOP + 0.020],
        [(rail_in + rail_out) / 2, y + 0.002, RAIL_TOP + 0.022],
        [rail_out + 0.012, y + 0.006, RAIL_TOP + 0.018],
        [rail_out + 0.040, y + 0.020, MATTRESS_TOP + 0.008],
        [rail_out + 0.110, y + 0.090, MATTRESS_TOP + 0.002],
        [rail_out + 0.200, y + 0.240, MATTRESS_TOP + 0.002],
    ]
    # The primed line lies in one loose loop on the mattress and ends in a capped connector.
    turn = np.asarray(downstream[-1])
    loop_center = turn + np.array([0.075, 0.040, 0.0])
    start = np.arctan2(*(turn - loop_center)[1::-1])
    for k in range(1, 15):
        angle = start + 1.7 * np.pi * k / 14
        radius = 0.075 - 0.012 * k / 14
        downstream.append(
            [
                loop_center[0] + radius * np.cos(angle),
                loop_center[1] + radius * np.sin(angle),
                MATTRESS_TOP + 0.002,
            ]
        )
    points = [np.asarray(downstream[0]), np.asarray(downstream[1])] + _spline(downstream[1:], 4)[1:]
    for index, (first, second) in enumerate(zip(points, points[1:])):
        segment(f"Tube_{index}", first, second)
    end, before = np.asarray(points[-1]), np.asarray(points[-2])
    axis = (end - before) / np.linalg.norm(end - before)
    segment("Connector", end, end + axis * 0.022, radius=0.0035, material="cap_white")
    segment(
        "ConnectorCap", end + axis * 0.022, end + axis * 0.034, radius=0.0042, material="cap_blue"
    )


def backend_labels(backend):
    config = getattr(backend, "config", None) or getattr(backend.appearance, "config", {})
    return bool(config.get("visuals", {}).get("labels", True))


def _stand_origin(pole_xy):
    """The iv_stand model origin that puts its pole axis at `pole_xy`."""
    c, s = np.cos(STAND_YAW), np.sin(STAND_YAW)
    mx, my = STAND_POLE[0] * STAND_SCALE[0], STAND_POLE[1] * STAND_SCALE[1]
    return np.asarray(pole_xy) - [c * mx - s * my, s * mx + c * my]


def _dress_ward(backend, info, mats):
    """IV stand with its hook arm, bedside cabinet with the cradle, and the robot's wheeled base."""
    from roboclock.envs.hospital.props import add_model, model_size

    look = backend.appearance
    config = getattr(backend, "config", None) or look.config
    hook = info["hook"]
    px, py = hook["pole_xy"]
    if model_size("iv_stand") is not None:
        ox, oy = _stand_origin((px, py))
        add_model(
            backend.stage,
            "iv_stand",
            "/World/Bedside/IVStand",
            (ox, oy, FLOOR_Z),
            STAND_SCALE,
            yaw=STAND_YAW,
        )
    else:  # missing model: a plain pole and base
        look.cylinder("/World/Bedside/IVPole", [px, py, 0.90], 0.012, 1.80, "stainless")
        look.cylinder("/World/Bedside/IVBase", [px, py, 0.02], 0.18, 0.03, "stainless")
    # Clamp-on hook arm: a clamp block on the pole, a round peg toward -y with an upturned tip.
    hx, hy, hz = hook["center"]
    look.rounded(
        "/World/Bedside/HookClamp", (px, py, hz), (0.046, 0.046, 0.040), mats["clamp"], 0.006
    )
    look.cylinder(
        "/World/Bedside/HookClampKnob", (px + 0.030, py, hz), 0.009, 0.020, "rubber", axis="X"
    )
    look.cylinder(
        "/World/Bedside/HookArm",
        (hx, hy, hz),
        PEG_HALF,
        2 * hook["half_length"],
        "stainless",
        axis="Y",
    )
    tip = hy - hook["half_length"]
    look.cylinder(
        "/World/Bedside/HookTip", (hx, tip + 0.004, hz + 0.010), 0.0045, 0.022, "stainless"
    )
    look.sphere("/World/Bedside/HookTipBall", (hx, tip + 0.004, hz + 0.022), 0.0055, "stainless")
    _cabinet(look, info, mats)
    _robot_base(backend, config, mats)


def _cabinet(look, info, mats):
    """A white bedside cabinet with a drawer and a door; its top is the support collider's top."""
    cab = info["cabinet"]
    (cx, cy), (sx, sy), top = cab["center"], cab["size"], cab["top"]
    body, front = mats["cabinet"], mats["cabinet_front"]
    look.rounded(
        "/World/Bedside/CabinetTop",
        (cx, cy, top - 0.012),
        (sx + 0.01, sy + 0.01, 0.024),
        mats["cabinet_top"],
        0.006,
    )
    carcass_h = top - 0.024 - FLOOR_Z - 0.06
    look.rounded(
        "/World/Bedside/CabinetBody",
        (cx, cy, FLOOR_Z + 0.06 + carcass_h / 2),
        (sx, sy, carcass_h),
        body,
        0.01,
    )
    # Fronts face the robot (-x side).
    fx = cx - sx / 2 - 0.004
    look.rounded(
        "/World/Bedside/CabinetDrawer", (fx, cy, top - 0.10), (0.012, sy - 0.03, 0.12), front, 0.004
    )
    look.rounded(
        "/World/Bedside/CabinetDoor",
        (fx, cy, FLOOR_Z + 0.06 + (top - 0.19 - FLOOR_Z - 0.06) / 2 + 0.01),
        (0.012, sy - 0.03, top - 0.19 - FLOOR_Z - 0.08),
        front,
        0.004,
    )
    for k, z in enumerate((top - 0.10, top - 0.30)):
        look.rounded(
            f"/World/Bedside/CabinetHandle_{k}",
            (fx - 0.010, cy, z),
            (0.010, 0.12, 0.014),
            mats["handle"],
            0.003,
        )
    look.rounded(
        "/World/Bedside/CabinetPlinth",
        (cx, cy, FLOOR_Z + 0.03),
        (sx - 0.03, sy - 0.03, 0.06),
        "rubber",
        0.006,
    )
    # Cradle base: a pad under each shoulder and a back bar, open under the hanging drip set.
    bx, by = info["bag"]["cradle_xy"]
    for side in (-1, 1):
        look.rounded(
            f"/World/Bedside/CradlePad_{side + 1}",
            (bx + side * 0.036, by, top + 0.004),
            (0.040, 0.070, 0.008),
            mats["cradle_pad"],
            0.003,
        )
    look.rounded(
        "/World/Bedside/CradleBar",
        (bx, by + 0.029, top + 0.004),
        (0.112, 0.012, 0.008),
        mats["cradle_pad"],
        0.003,
    )


def _robot_base(backend, config, mats):
    """Mobile base: one wheeled chassis under both X5 columns (the robot itself is fixed).

    Molded light grey shell like the mount columns over a dark powder-coated skirt with rubber
    bumper strips, an inset service cover, side vent slats, an e-stop and a status strip; dark
    powder-coated lower columns with aluminum collars under the mount columns; rubber casters.
    """
    from roboclock.envs.lab.robot import mount_positions

    look = backend.appearance
    z = backend.layout.table_height
    mounts = [np.asarray(m[:2], dtype=float) for m in mount_positions(config, "B5")]
    lo, hi = np.min(mounts, axis=0) - 0.13, np.max(mounts, axis=0) + 0.13
    lo[0] -= 0.06
    center, size = (lo + hi) / 2, hi - lo
    p = "/World/Bedside/Robot"
    # Skirt band (0.025-0.075 m) under the molded shell (0.06-0.13 m), cover plate on top.
    look.rounded(p + "Bumper", [*center, FLOOR_Z + 0.05], [*(size + 0.012), 0.05], mats["skirt"], 0.014)
    look.rounded(p + "Base", [*center, FLOOR_Z + 0.095], [*size, 0.07], mats["shell"], 0.025)
    look.rounded(
        p + "Cover", [center[0] - 0.01, center[1], FLOOR_Z + 0.1315], [size[0] - 0.08, size[1] - 0.07, 0.005],
        mats["cover"], 0.012,
    )
    for k, sign in enumerate((-1, 1)):  # rubber bumper strips on the front and back
        look.rounded(
            p + f"BumperStrip_{k}", [center[0] + sign * (size[0] / 2 + 0.008), center[1], FLOOR_Z + 0.05],
            [0.012, size[1] - 0.06, 0.024], "rubber", 0.005,
        )
    for k, sign in enumerate((-1, 1)):  # vent slats on both sides of the shell
        for j in range(3):
            look.rounded(
                p + f"Vent_{k}_{j}", [center[0], center[1] + sign * (size[1] / 2 + 0.0005), FLOOR_Z + 0.08 + 0.014 * j],
                [0.14, 0.003, 0.005], mats["skirt"], 0.0012,
            )
    look.rounded(  # status strip on the front, facing the bed
        p + "StatusStrip", [hi[0] + 0.0005, center[1], FLOOR_Z + 0.105], [0.003, 0.16, 0.010],
        look.material("b5_status_led", color=[0.2, 0.75, 1.0], roughness=0.3, emissive=True), 0.002,
    )
    stop = (lo[0] + 0.05, lo[1] + 0.06)  # e-stop at the rear corner of the cover
    look.cylinder(p + "EStopCollar", (*stop, FLOOR_Z + 0.137), 0.022, 0.008, mats["estop_collar"])
    look.cylinder(p + "EStopStem", (*stop, FLOOR_Z + 0.145), 0.008, 0.010, mats["skirt"])
    look.cylinder(p + "EStop", (*stop, FLOOR_Z + 0.154), 0.016, 0.010, mats["estop"])
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        wheel = [
            center[0] + sx * (size[0] / 2 - 0.05),
            center[1] + sy * (size[1] / 2 - 0.05),
            FLOOR_Z + 0.03,
        ]
        look.cylinder(f"/World/Bedside/RobotWheel_{i}", wheel, 0.03, 0.025, "rubber", axis="Y")
        look.cylinder(f"/World/Bedside/RobotWheelHub_{i}", wheel, 0.014, 0.027, "aluminum", axis="Y")
    for i, m in enumerate(mounts):
        look.rounded(
            f"/World/Bedside/RobotColumn_{i}",
            [*m, (FLOOR_Z + 0.13 + z) / 2],
            [0.12, 0.12, z - FLOOR_Z - 0.13],
            mats["skirt"],
            0.006,
        )
        # Aluminum collars where the column leaves the base and meets the mount column.
        for k, zc in enumerate((FLOOR_Z + 0.145, z - 0.012)):
            look.rounded(
                f"/World/Bedside/RobotCollar_{i}_{k}", [*m, zc], [0.128, 0.128, 0.024], mats["trim"], 0.004
            )


def _display(backend, info, mats):
    """Turn the dial toward the camera and stand the instruction console beside it.

    Both sit on the cabinet top's front (-y) edge, below the camera's line of sight to the bag.
    The backend's housing collider stays as built; only its visual and side label are replaced.
    The text never changes during the episode, so there is one static card (no swapping).
    """
    from pxr import Gf, UsdGeom

    stage, look = backend.stage, backend.appearance
    indicators = getattr(backend.layout, "indicators", ())
    if not indicators:
        return
    center = np.asarray(indicators[0].position, dtype=float) + [0.0, 0.0, 0.005]
    shift = np.array([*info.get("shift", (0.0, 0.0)), 0.0])
    camera = np.asarray(info["camera"]["position"]) + shift
    from roboclock.envs.hospital.rig import dial_target

    head = dial_target(backend, camera)  # the dial turns to the policy's head camera on the rig
    yaw = math.degrees(math.atan2(head[1] - center[1], head[0] - center[0]))
    dials = stage.GetPrimAtPath("/World/Dials")
    if dials.IsValid():
        xform = UsdGeom.Xformable(dials)
        xform.ClearXformOpOrder()
        pivot = Gf.Vec3d(*center.tolist())
        xform.AddTranslateOp(opSuffix="pivot").Set(pivot)
        xform.AddRotateZOp().Set(yaw)
        xform.AddTranslateOp(opSuffix="pivot", isInverseOp=True)
    for path in ("/World/Labels/indicator_indicator", "/World/Equipment/indicator_Housing_Visual"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    from roboclock.envs.hospital.backend import INDICATOR_SIZE

    group = look.group("/World/Bedside/DialHousing", center, rz=yaw)
    look.rounded(group + "/Body", (0.0, 0.0, 0.0), INDICATOR_SIZE, "cabinet_white", 0.006)
    # Off the cabinet the housing stands on a slim floor stand (visual only).
    bottom = float(center[2]) - INDICATOR_SIZE[2] / 2
    look.cylinder("/World/Bedside/DialStand", [center[0], center[1], bottom / 2], 0.012, bottom, "stainless")
    look.cylinder("/World/Bedside/DialStandBase", [center[0], center[1], 0.008], 0.09, 0.016, "stainless")
    # Instruction console: a screen leaning back on a wedge foot, turned toward the camera.
    dx, dy, dz = info["display"]
    w, h = DISPLAY_SIZE
    to_cam = math.degrees(math.atan2(camera[1] - dy, camera[0] - dx))
    outer = look.group("/World/Bedside/InstructionDisplay", (dx, dy, dz), rz=to_cam + 90.0)
    look.rounded(
        outer + "/Foot", (0.0, 0.025, -h / 2 + 0.004), (w * 0.7, 0.06, 0.012), "rubber", 0.004
    )
    screen = look.group(outer + "/Screen", (0.0, 0.0, 0.0), rx=-DISPLAY_TILT)
    look.rounded(
        screen + "/Bezel", (0.0, 0.006, 0.0), (w + 0.016, 0.010, h + 0.016), mats["housing"], 0.006
    )
    look.rounded(  # the housing's rear shell behind the bezel
        screen + "/Back", (0.0, 0.015, 0.0), (w * 0.72, 0.010, h * 0.70), mats["housing"], 0.004
    )
    if backend_labels(backend):
        text_panel(
            stage,
            screen + "/Text",
            [("HANG SALINE,", 1.0), ("THEN", 0.62), ("OPEN CLAMP", 1.0)],
            (1.0, 1.0, 1.0),
            (0.06, 0.30, 0.62),
            (0.0, -0.0005, 0.0),
            (w, h),
        )


def _organizer_labels(backend, origin):
    """CLOSED and OPEN plates under the two ends of the organizer, readable from the foot of the bed."""
    if not backend_labels(backend):
        return
    x, y, z = origin
    plate_z = z - 0.058  # hangs from the tray's front edge (tray top at z - 0.032)
    for text, cx, color in (
        ("CLOSED", x - 0.012, (0.72, 0.16, 0.14)),
        ("OPEN", x + TRAVEL + 0.012, (0.10, 0.48, 0.25)),
    ):
        text_panel(
            backend.stage,
            f"/World/Bedside/RollerLine/Plate_{text}",
            [(text, 1.0)],
            (1.0, 1.0, 1.0),
            color,
            (cx, y - 0.0275, plate_z),
            (0.056, 0.026),
            fit="CLOSED",
        )
    backend.appearance.rounded(
        "/World/Bedside/RollerLine/PlateBacker",
        (x + TRAVEL / 2, y - 0.0245, z - 0.0545),
        (0.124, 0.004, 0.045),
        "cabinet_white",
        0.0015,
    )


def _text_texture(lines, fg, bg, aspect, fit=None):
    """A cached PNG with centered bold lines of text filling a panel of the given aspect.

    `fit` sizes the font as if every line read `fit` (keeps sibling plates at one letter size).
    """
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    height = 256
    size = (max(64, int(round(height * aspect))), height)
    key = hashlib.sha256(repr((lines, fg, bg, size, fit)).encode()).hexdigest()[:20]
    path = cache / f"a2_{key}.png"
    if path.exists():
        return path
    image = Image.new("RGB", size, tuple(round(255 * c) for c in bg))
    draw = ImageDraw.Draw(image)
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    total = sum(scale for _, scale in lines)
    y = 0.0
    for text, scale in lines:
        band = size[1] * scale / total
        font_size = int(band * 0.78)
        while True:
            try:
                font = ImageFont.truetype(bold, font_size)
            except OSError:
                font = ImageFont.load_default(size=font_size)
            if font.getlength(fit or text) <= size[0] * 0.9 or font_size <= 10:
                break
            font_size -= 4
        draw.text(
            (size[0] / 2, y + band / 2),
            text,
            fill=tuple(round(255 * c) for c in fg),
            font=font,
            anchor="mm",
        )
        y += band
    image.save(path)
    return path


def text_panel(stage, path, lines, fg, bg, center, size, fit=None):
    """A lettered quad in the x-z plane that reads correctly from -y (in its parent's frame)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    w, h = size
    texture = _text_texture(lines, fg, bg, w / h, fit)
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
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-w / 2, 0, -h / 2),
            Gf.Vec3f(w / 2, 0, -h / 2),
            Gf.Vec3f(w / 2, 0, h / 2),
            Gf.Vec3f(-w / 2, 0, h / 2),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    mesh.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _camera(backend):
    """Foot-of-bed view: hand, cabinet and bag, organizer, hook, stand, display and the bed."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["roller_clamp"]["camera"]
    shift = np.array([*backend.layout.info["roller_clamp"].get("shift", (0.0, 0.0)), 0.0])
    position = np.asarray(view["position"]) + shift + backend.layout.camera_jitter
    target = np.asarray(view["look_at"]) + shift
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*target.tolist()), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    if view.get("focal_length"):
        camera.set_focal_length(float(view["focal_length"]) / 1000)
    for _ in range(3):
        backend.world.render()


# --- measurement ---------------------------------------------------------------------------------


def measure(backend, state):
    """Roller opening and bag support, derived only from the observed rigid bodies."""
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_a2_camera_for", None) is not backend.task:
        backend._a2_camera_for = backend.task
        _camera(backend)
    info = backend.layout.info["roller_clamp"]
    controls = {}
    roller = state.objects.get(ROLLER_NAME)
    if roller is not None:
        controls.update(roller_controls(info["origin"], roller))
    bag = state.objects.get(BAG_NAME)
    if bag is not None:
        hang = hang_geometry(backend.layout, bag)
        controls.update(
            bag_hanging=hang["hanging"],
            bag_on_peg=hang["on_peg"],
            bag_hook_contact=hang["hook_contact"],
            bag_tilt_rad=hang["tilt_rad"],
            bag_displacement_m=float(
                np.linalg.norm(bag.position - backend.layout.objects[BAG_NAME].position)
            ),
        )
        slack = getattr(backend, "_b5_slack_line", None)
        if slack is not None:
            slack.update(
                np.asarray(bag.position, dtype=float), np.asarray(bag.orientation, dtype=float)
            )
    return controls


# --- privileged oracle ----------------------------------------------------------------------------

# Free hand moves; bag tracking per 33 ms control step while carrying and threading onto the peg.
FREE_SPEED, CARRY_SPEED, THREAD_SPEED = 0.35, 0.30, 0.20
TRACK_STEP, THREAD_STEP = 0.010, 0.007
LIFT = 0.022  # bag body clear of the cradle shoulders; the y-led carry keeps the drip set clear
CLEAR_UP = 0.032  # open fingertips above the grasp tab's top before moving sideways
ENTRY_GAP = 0.030  # eyelet face to the peg's free end before threading
ROLLER_HOVER = 0.014  # fast approach to this height above the push pose, then a settled descent
PUSH_LIFT = 0.003  # fingertips a further 3 mm above the holder's end caps while pushing


def command(planner, task, state):
    """Hang the bag (lift, carry, thread over the peg, lower, release), then push the roller open."""
    info = task.layout.info["roller_clamp"]
    bag_info, hook = info["bag"], info["hook"]
    obj = state.objects[BAG_NAME]
    roller = state.objects[ROLLER_NAME]
    origin = np.asarray(info["origin"])
    now = task.time
    if planner.last_goal is None:
        planner.last_goal = state.ee_position.copy()
    if getattr(planner, "_b5_task", None) is not task:
        planner._b5_task = task
        planner.b5_phase, planner.b5_time = "wait", now
        planner.b5_bias = np.zeros(3)
        planner.b5_offset = np.asarray(bag_info["grasp_offset"], dtype=float)
        planner.b5_last_ee = state.ee_position.copy()
    planner.yaw = 0.0
    planner.gentle = True
    planner.open_width, planner.close_width = TAB_OPEN, TAB_CLOSE
    phase = planner.b5_phase
    moving = float(np.linalg.norm(state.ee_position - planner.b5_last_ee))
    planner.b5_last_ee = state.ee_position.copy()
    rest = np.asarray(task.layout.info["initial_ee_position"], dtype=float)
    start = np.asarray(task.layout.objects[BAG_NAME].position, dtype=float)
    grasp = start + np.asarray(bag_info["grasp_offset"])
    hang = np.asarray(hook["hang_position"], dtype=float)
    eye = bag_info["eyelet"]
    # Bag centre while threading: the peg passes midway through the eyelet's opening.
    bz = bag_info["body_size"][2]
    thread = hang.copy()
    thread[2] = hook["center"][2] - (bz / 2 + (eye["height"] - eye["bar"]) / 2)
    entry = thread.copy()
    entry[1] = hook["tip_y"] - eye["thickness"] / 2 - ENTRY_GAP

    def transition(value):
        planner.b5_phase, planner.b5_time = value, now
        task.event_log.append({"time": now, "type": "b5_planner_phase", "phase": value})

    def track(bag_goal, step=TRACK_STEP):
        # Bounded Cartesian correction from the measured bag error (removes IK and load bias).
        correction = np.clip(0.4 * (bag_goal - obj.position), -step, step)
        return np.clip(
            planner.last_goal + correction, state.ee_position - 0.03, state.ee_position + 0.03
        )

    close, speed, goal = False, FREE_SPEED, rest.copy()
    if phase == "wait":
        # Nothing moves before the instruction: the hand holds its tucked rest pose.
        if task.event_time is not None:
            transition("approach")
            phase = "approach"
    if phase == "approach":
        goal = grasp + [0.0, 0.0, 0.035]
        if np.linalg.norm(state.ee_position - goal) < 0.015:
            transition("descend")
    elif phase == "descend":
        goal, speed = grasp, 0.25
        if np.linalg.norm(state.ee_position - grasp) < 0.005:
            transition("grasp")
    elif phase == "grasp":
        goal, close = grasp, True
        if obj.grasped and now - planner.b5_time > 0.05:
            planner.b5_offset = state.ee_position - obj.position
            transition("lift")
        elif now - planner.b5_time > 1.5:
            transition("approach")  # a failed pinch is a real failure; retry from above
    elif phase == "lift":
        close, speed = True, CARRY_SPEED
        desired = start + [0.0, 0.0, LIFT]
        goal = track(desired)
        if obj.position[2] >= desired[2] - 0.008:
            transition("carry")
    elif phase == "carry":
        close, speed = True, CARRY_SPEED
        goal = track(entry)
        # Thread only once the carried bag has stopped swinging in line with the peg.
        if np.linalg.norm(obj.position - entry) < 0.007 and np.linalg.norm(obj.velocity) < 0.05:
            transition("thread")
    elif phase == "thread":
        close, speed = True, THREAD_SPEED
        desired = thread.copy()
        off_peg = obj.position[1] + eye["thickness"] / 2 < hook["tip_y"] + 0.004
        misaligned = (
            abs(obj.position[0] - thread[0]) > 0.010 or abs(obj.position[2] - thread[2]) > 0.008
        )
        if off_peg and misaligned:
            desired[1] = min(obj.position[1], entry[1])  # re-align before the eyelet meets the tip
        goal = track(desired, THREAD_STEP)
        if np.linalg.norm(obj.position - thread) < 0.005:
            transition("lower")
    elif phase == "lower":
        close, speed = True, 0.08
        goal = track(hang - [0.0, 0.0, 0.003], 0.003)
        seated = obj.position[2] <= hang[2] + 0.002 and HOOK_CONTACT in obj.fixture_contacts
        if seated or now - planner.b5_time > 0.8:
            planner.b5_release = state.ee_position.copy()
            transition("release")
    elif phase == "release":
        goal = planner.b5_release.copy()
        if not obj.grasped and now - planner.b5_time > 0.10:
            transition("clear")
    elif phase == "clear":
        # Straight up until the open fingers are above the grasp tab (they straddle it along y),
        # then the move to the roller leads with -y, away from the hanging bag.
        goal = planner.b5_release + [0.0, 0.0, CLEAR_UP + 0.01]
        if state.ee_position[2] > planner.b5_release[2] + CLEAR_UP:
            transition("to_roller")
    if phase in ("to_roller", "roller_descend", "push", "retreat"):
        planner.open_width = planner.close_width = 0.004
        position = float(state.controls.get("roller_position_m", roller.position[0] - origin[0]))
        ready = origin + np.array([-READY_OFFSET, 0.0, PUSH_HEIGHT + PUSH_LIFT])
        if phase == "to_roller":
            goal = ready + [0.0, 0.0, ROLLER_HOVER]
            if state.ee_position[1] > hang[1] - 0.075:  # still beside the hanging bag: stay high
                goal[2] = max(goal[2], planner.b5_release[2] + CLEAR_UP)
            if np.linalg.norm(state.ee_position - goal) < 0.006:
                transition("roller_descend")
            return planner.finish(state, goal, False, speed=FREE_SPEED)
        if phase == "roller_descend":
            # A slow, settled final descent: the fingertips clear the holder's rails by ~6 mm. The
            # bounded correction removes the arm's static offset at this low, extended pose.
            if np.linalg.norm(state.ee_position - ready) < 0.02:
                error = ready - state.ee_position
                planner.b5_bias = np.clip(
                    planner.b5_bias + np.clip(0.3 * error, -0.001, 0.001), -0.012, 0.012
                )
            settled = moving < 0.0012 and np.linalg.norm(state.ee_position - ready) < 0.0035
            if settled and task.hang_retained:
                transition("push")
            return planner.finish(state, ready + planner.b5_bias, False, speed=0.06)
        if task.goal_achieved or abs(position - TRAVEL) <= END_TOLERANCE:
            if phase != "retreat":
                transition("retreat")
        lift = np.array([0.0, 0.0, PUSH_LIFT])
        return push_roller(planner, state, origin + lift + planner.b5_bias, position, TRAVEL)
    return planner.finish(state, np.asarray(goal, dtype=float), close, speed)
