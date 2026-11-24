"""A1 drip chamber level: open the roller clamp, then slide it shut while the level is in its band.

Ward placement (scene-02 left bay, assets/hospital/ward_02_preview_left.usdc): the bed head by the
window. The wheeled dual-arm base parks in the gap between the bed's head end and the window wall,
on the pendant side, facing along the window toward the counter side. A small white treatment table
stands in front of the base, straight ahead of the rig's 60-degree-down head camera. On it: a
tabletop IV stand with the saline bag and the drip chamber under it (left of the image center), the
line organizer with the roller clamp (right of it, toward the driving arm, so the hand and forearm
never cover the chamber) and the countdown dial. The line runs from the chamber through the
organizer, off the table and onto the mattress beside the pillow. A white convector cover is added
under the window (the ward's bare wall there renders near-black).

The saline bag (local CC-BY-4.0 "Blood Bag V2" model recolored as a clear 0.9 % NaCl bag) turns its
labelled face to the head camera; a large clear trainer drip chamber below it shows the fluid level,
a green target band, a red overfill zone and falling drops. The level is the task's quantity
(``task.quantity()``, percent of the chamber height): still while the clamp is closed, rising from
the measured opening, frozen at the measured closing. It is a visual of the scored process, not a
fluid simulation.

The functional part is the roller: a free rigid body on a passive prismatic joint in the organizer
tray (horizontal, so gravity cannot drive it; viscous drag stands in for the restrained wheel). It
starts at its CLOSED end stop. The robot straddles it with the gripper open across the slide (fingers
on either side, no squeeze), pulls it to the OPEN end, waits, and pushes it back to CLOSED.
``device_opened`` (OPEN end reached after robot contact) and ``device_actuated`` (CLOSED end reached
again after that) come only from the measured roller pose; nothing is commanded on the joint.
"""

import numpy as np

from roboclock.envs.hospital.bedside.ward import furniture

ROLLER_NAME = "roller_0"
ROLLER_ASSET = "iv_roller"
TRAVEL = 0.036  # OPEN (joint 0, near the robot) to CLOSED (joint TRAVEL, toward the stand), along +x
ROLLER_HALF_X = 0.012  # catalog iv_roller box, along the slide
END_TOLERANCE = 0.0015
MOTION_THRESHOLD = 0.0015
# The exported X5 fingertips extend 12.57 mm below grasp_center: the straddling pose keeps their
# lower edge above the organizer's rails and end stops while their faces meet the roller's.
PUSH_HEIGHT = 0.009
HOVER = 0.035  # the start pose: straddle height + this, over the CLOSED roller
HAND_YAW = np.pi / 2  # the fingers open along the slide (x), one on each side of the roller
# The gripper width command is each finger's travel: measured in Isaac Sim (2026-11-20, width 0.040)
# the inner finger faces stood 38 mm either side of the tool point, i.e. 0.95 x the width.
FINGER_GAP_PER_WIDTH = 0.95
GRIP_WIDTH = 0.021  # inner faces ~20 mm either side: 8 mm clear of each roller face, a fork
FINGER_HALF_GAP = FINGER_GAP_PER_WIDTH * GRIP_WIDTH
SLACK = FINGER_HALF_GAP - ROLLER_HALF_X  # hand travel before a finger meets the roller
OVERDRIVE = 0.002  # the hand goal past the end stop, so the finger holds the roller there
PUSH_SPEED = 0.14  # m/s, commanded Cartesian speed of both slides
OPEN_AT = 1.2  # s, the oracle starts the opening slide once the hand has settled on the roller
PUSH_LEAD = 0.53  # s from the oracle starting the closing slide to the measured CLOSED end (0.528 s in Isaac Sim)

# --- the drip chamber (a trainer chamber, oversized for legibility) ---------------------------
# The 60-degree-down head camera halves vertical sizes: the chamber is 15 cm tall and the band 30 %
# of it (4.5 cm, ~12 px in the 640x480 head image; 9 cm / 15 % gave ~3 px, unreadable).
CHAMBER_RADIUS = 0.026
CHAMBER_HEIGHT = 0.150  # clear wall between the caps
CAP_HEIGHT = 0.008
LIQUID_RADIUS = 0.0235
# Saturated blue column, lit only a little (a strong glow blew it out to pale cyan on camera), with
# a bright surface disk so the level line reads from above.
LIQUID_COLOR = (0.04, 0.30, 0.95)
LIQUID_GLOW = 3.0
SURFACE_COLOR = (0.75, 0.90, 1.00)
SURFACE_GLOW = 4.0
RING_LOW_COLOR = (0.05, 0.75, 0.25)  # the band's lower edge
RING_HIGH_COLOR = (0.90, 0.10, 0.08)  # the band's upper edge: above it is OVER
CARD_WIDTH = 0.120  # printed scale behind the chamber (its height spans the chamber and both caps)
CARD_OFFSET_X = 0.030  # to the camera's left: the labels stand beside the chamber, not behind it
DROP_RADIUS = 0.0045
FEED_GAP = 0.070  # bag bottom to the chamber's top
DROP_PERIOD = 0.55  # seconds per drop: hangs at the orifice, then falls
DROP_FALL = 0.14  # seconds of the period spent falling

# --- placement (robot frame, before the seed's common shift) -----------------------------------
WARD_USD = "assets/hospital/ward_02_preview_left.usdc"
# Yaw 0: robot +x is ward +x. The robot origin sits at ward (0.25, -2.95), in the gap between the
# bed head (ward y -2.78) and the window wall (y -3.55), at the bay's pendant-side corner.
WARD_POSE = {"translate": [-0.25, 2.95, 0.0], "yaw": 0.0, "hide": []}
# The common rig (hospital/rig.py) stands its idle arm 0.60 m to +y of the driving arm, i.e. on
# the bed side; the 0.77 m gap between window wall and bed head cannot hold the 0.86 m wide base. The
# bed is pushed 0.65 m away from the window (robot +y) so the base and the idle arm stand clear of
# its head end; the line to the pillow follows it (LINE_TO_BED).
BED_MOVE = (0.0, 0.65, 0.0)
FLOOR_Z = 0.0
# The head camera sits at (0.04, 0.30, 1.59) looking down 60 degrees; its image's upper edge ray
# falls ~20 degrees below horizontal, so the bag's top must stay below ~1.40 m at x 0.48. Bag and
# chamber hang 0.10 m left of the camera's center line, the organizer 0.14 m right of it.
ROLLER_XY = (0.34, 0.16)  # the OPEN end (joint zero); the roller starts TRAVEL further, CLOSED
ROLLER_Z = 0.90
TABLE_TOP = ROLLER_Z - 0.047  # the organizer tray rests on it
TABLE_X = (0.22, 0.70)
TABLE_Y = (-0.06, 0.56)
HOOK_XY = (0.42, 0.40)  # the bag hangs here; the drip chamber is right below it
HOOK_Z = 1.40
POLE_XY = (0.55, 0.40)  # the tabletop stand's pole, behind the bag; its arm reaches the hook
BAG_SIZE = (0.150, 0.075, 0.220)  # the recolored model (a 500 mL bag), hanging loop included
BAG_HANG = 0.018  # the bag's top edge above the hook (the hanging hole sits below it)
INDICATOR_XY = (0.64, 0.18)  # the countdown dial on the table, right of the bag (not hidden by it)
INDICATOR_TOP = 0.97
DIAL_SCALE = 0.70
# Convector cover along the window wall: x range, y range (robot frame) and top height.
SILL_CABINET = ((0.30, 1.46), (-0.60, -0.44), 0.88)
# From the organizer's OPEN end the line runs along the table's near edge, off its +y side in front
# of the idle arm, outside the bed's far head rail (ward x 0.76-0.91, y -2.39..-2.13 before
# BED_MOVE: robot x 0.51-0.66, y 0.56-0.82) and, past the rail, over the mattress edge onto the
# mattress (top z 0.86) beside the pillow (robot frame, before the shift; bed points include BED_MOVE).
LINE_TO_BED = [
    (0.25, 0.22, TABLE_TOP + 0.003),
    (0.25, 0.48, TABLE_TOP + 0.003),
    (0.32, 0.60, TABLE_TOP - 0.04),
    (0.47, 0.72, 0.86),
    (0.485, 0.60 + BED_MOVE[1], 0.93),
    (0.50, 0.86 + BED_MOVE[1], 0.915),
    (0.575, 0.92 + BED_MOVE[1], 0.885),
    (0.66, 0.96 + BED_MOVE[1], 0.867),
    (0.76, 0.99 + BED_MOVE[1], 0.866),
]
# Third-person recording view (not the policy's camera): from the bed side toward the window.
CAMERA = {"position": [0.62, 1.05, 1.30], "look_at": [0.44, 0.26, 0.98], "focal_mm": 30.0}

# --- pure helpers (no simulator) -----------------------------------------------------------------


def level_fraction(quantity):
    """Fill fraction of the chamber for a quantity in percent of its height."""
    return float(np.clip(float(quantity) / 100.0, 0.0, 1.0))


def liquid_height(quantity):
    """Height of the liquid column (m) inside the chamber for the given quantity (%)."""
    return level_fraction(quantity) * CHAMBER_HEIGHT


def band_marks(band):
    """Heights above the chamber's inner floor of the band edges; everything above is OVER."""
    lo, hi = band
    return {"band_low": liquid_height(lo), "band_high": liquid_height(hi)}


def scale_rows(band, rows):
    """Pixel rows (top = 0) of the band's top and bottom on a scale image `rows` tall.

    The image spans the chamber plus one cap height above and below (the card's full height).
    """
    total = CHAMBER_HEIGHT + 2 * CAP_HEIGHT

    def row(fraction):
        return int(round(rows * (1.0 - (CAP_HEIGHT + fraction * CHAMBER_HEIGHT) / total)))

    return row(level_fraction(band[1])), row(level_fraction(band[0])), row(0.0), row(1.0)


def drop_state(time, surface, orifice):
    """(z, scale) of the one visible drop at `time` while the line flows.

    A drop grows at the orifice for most of a period, then falls to the liquid surface.
    """
    phase = (float(time) % DROP_PERIOD) / DROP_PERIOD
    hang = 1.0 - DROP_FALL / DROP_PERIOD
    if phase < hang:
        return orifice - DROP_RADIUS * (0.4 + 0.6 * phase / hang), 0.35 + 0.65 * phase / hang
    fall = (phase - hang) / (1.0 - hang)
    z = orifice - DROP_RADIUS - fall * fall * max(orifice - DROP_RADIUS - surface, 0.0)
    return z, 1.0


def slide_start(window_open, window_close, lead=None):
    """When the oracle starts the closing slide so the clamp closes at the window's middle."""
    return 0.5 * (window_open + window_close) - (PUSH_LEAD if lead is None else lead)


def hand_goals(origin):
    """Hand (tool point) positions of the oracle: the straddle over the CLOSED roller, the end of the
    opening pull (front finger holds the roller at the OPEN stop), the straddle over the OPEN roller
    and the end of the closing push (rear finger holds it at the CLOSED stop)."""
    origin = np.asarray(origin, dtype=float)
    z = np.array([0.0, 0.0, PUSH_HEIGHT])
    along = np.array([1.0, 0.0, 0.0])
    return {
        "closed": origin + TRAVEL * along + z,
        "pulled": origin + (-SLACK - OVERDRIVE) * along + z,
        "open": origin + z,
        "pushed": origin + (TRAVEL + SLACK + OVERDRIVE) * along + z,
    }


# --- layout --------------------------------------------------------------------------------------


def build_layout(builder):
    """Declare the roller (CLOSED), its organizer tray on the table, the cue and the ward placement."""
    b = builder
    shift = np.array([float(v) for v in b.shift])
    x, y, center_z = ROLLER_XY[0], ROLLER_XY[1], ROLLER_Z
    b.item(
        ROLLER_NAME,
        ROLLER_ASSET,
        (x + TRAVEL, y),
        center_z - 0.011 - 0.002,
        color=(0.15, 0.48, 0.80),
        yaw=0.0,
    )
    # The OPEN end is the joint's zero; the roller starts TRAVEL beyond it, at the CLOSED stop.
    origin = b.objects[ROLLER_NAME].position.copy() - np.array([TRAVEL, 0.0, 0.0])
    # Organizer tray: the roller protrudes above it, between two rails, with end caps as stops.
    base_z = center_z - 0.042
    b.fixture(
        "roller_holder_base",
        (x + TRAVEL / 2, y),
        (0.100, 0.052, 0.010),
        material="cabinet_white",
        z=base_z,
    )
    for sign, suffix in ((-1, "right"), (1, "left")):
        b.fixture(
            f"roller_track_{suffix}",
            (x + TRAVEL / 2, y + sign * 0.018),
            (0.074, 0.005, 0.019),
            material="cabinet_white",
            z=center_z - 0.029,
        )
    for dx, suffix in ((-0.018, "open"), (TRAVEL + 0.018, "closed")):
        b.fixture(
            f"roller_stop_{suffix}",
            (x + dx, y),
            (0.007, 0.041, 0.019),
            material="cabinet_white",
            z=center_z - 0.029,
        )
    hook = np.array(HOOK_XY) + shift
    # The countdown cue: its housing is the backend's indicator body; the dial stands on top.
    b.indicator("indicator", INDICATOR_XY, "DRIP", height=INDICATOR_TOP - 0.105 - b.z)
    b.info["bedside_task"] = "window_drip"
    b.info["drip"] = {
        "origin": origin,
        "travel": TRAVEL,
        "shift": shift.tolist(),
        "hook": [*hook.tolist(), HOOK_Z],
        "pole": (np.array(POLE_XY) + shift).tolist(),
        "table": {
            "x": [TABLE_X[0] + shift[0], TABLE_X[1] + shift[0]],
            "y": [TABLE_Y[0] + shift[1], TABLE_Y[1] + shift[1]],
            "top": TABLE_TOP,
        },
        "band": list(b.task.band),
        "camera": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in CAMERA.items()},
    }
    b.info["ward_pose"] = {
        "usd": WARD_USD,
        "translate": [
            WARD_POSE["translate"][0] + shift[0],
            WARD_POSE["translate"][1] + shift[1],
            WARD_POSE["translate"][2],
        ],
        "yaw": WARD_POSE["yaw"],
        "hide": list(WARD_POSE.get("hide", ())),
        "move": {name: list(BED_MOVE) for name in furniture("bed")},
    }
    b.info["hide_bench"] = True
    # The invisible support collider is the organizer tray the roller runs in.
    b.info["work_surface"] = {
        "center": [*map(float, b.xy(x + TRAVEL / 2, y)), base_z + 0.005],
        "size": [0.100, 0.052, 0.010],
    }
    b.info["initial_ee_position"] = hand_goals(origin)["closed"] + np.array([0.0, 0.0, HOVER])
    b.info["initial_ee_yaw"] = float(HAND_YAW)
    b.info["initial_gripper_width"] = GRIP_WIDTH


# --- authoring -----------------------------------------------------------------------------------


def author(backend):
    """Joint the existing roller body to its organizer and dress the apparatus (visual only)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage, look = backend.stage, backend.appearance
    info = backend.layout.info["drip"]
    path = f"/World/Objects/{ROLLER_NAME}"
    root = stage.GetPrimAtPath(path)
    if not root.IsValid():
        raise ValueError("The A1 roller rigid body must exist before author()")
    for child in root.GetChildren():
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            UsdGeom.Imageable(child).MakeInvisible()
    _roller_visual(look, path)
    origin = np.asarray(info["origin"])
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/drip_roller_slide")
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
    joint.GetPrim().SetCustomDataByKey("mechanism", "passive_stabilized_IV_roller")
    _organizer_dress(backend, origin)
    _table(backend, info)
    _stand_and_bag(backend, info)
    backend._a1_drip_visual = _chamber(backend, info)
    _line(backend, info, origin)
    _robot_base(backend, info)
    _sill_cabinet(backend, info)
    _cue(backend)
    return joint


def _material(look, name, color, roughness=0.4, emissive=0.0, opacity=None):
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


def _clear_wall(look):
    """The chamber wall: a fully clear shell, read by its caps, band rings and silhouette edges.

    OmniGlass under RTX real-time lighting darkened the empty part of the chamber (two refractions
    through the thin walls) and streaked the liquid with reflections; a cut-out shell keeps the
    level, the band and the printed scale behind it undistorted.
    """
    return _material(look, "a1_chamber_shell", (0.95, 0.97, 1.0), roughness=0.1, opacity=0.15)


def _roller_visual(look, path):
    from pxr import UsdGeom

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


def _organizer_dress(backend, origin):
    """The organizer's OPEN / CLOSED end plates (the tray itself rests on the table)."""
    x, y, z = origin
    if _labels(backend):
        for text, cx, color in (
            ("OPEN", x - 0.010, (0.10, 0.48, 0.25)),
            ("CLOSED", x + TRAVEL + 0.010, (0.72, 0.16, 0.14)),
        ):
            text_panel(
                backend.stage,
                f"/World/Bedside/Drip/Plate_{text}",
                [(text, 1.0)],
                (1.0, 1.0, 1.0),
                color,
                (cx, y + 0.0275, z - 0.050),
                (0.052, 0.022),
                fit="CLOSED",
                yaw=180.0,
            )


def _bar(look, path, first, second, radius, material):
    from pxr import Gf, UsdGeom

    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    delta = second - first
    part = look.cylinder(path, (first + second) / 2, radius, float(np.linalg.norm(delta)), material)
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*delta.tolist())).GetQuat()
    UsdGeom.Xformable(part.GetPrim()).AddOrientOp().Set(Gf.Quatf(rotation))
    return part


def _table(backend, info):
    """A small white treatment table under the apparatus (visual only; the tray is the collider)."""
    look = backend.appearance
    table = info["table"]
    (x0, x1), (y0, y1), top = table["x"], table["y"], table["top"]
    look.rounded(
        "/World/Bedside/Drip/TableTop",
        [(x0 + x1) / 2, (y0 + y1) / 2, top - 0.012],
        [x1 - x0, y1 - y0, 0.024],
        "cabinet_white",
        0.006,
    )
    legs = _material(look, "a1_table_legs", (0.72, 0.75, 0.78), roughness=0.35)
    for i, (lx, ly) in enumerate(((x0, y0), (x0, y1), (x1, y0), (x1, y1))):
        cx, cy = lx + np.sign((x0 + x1) / 2 - lx) * 0.03, ly + np.sign((y0 + y1) / 2 - ly) * 0.03
        look.cylinder(
            f"/World/Bedside/Drip/TableLeg_{i}", [cx, cy, (FLOOR_Z + top) / 2], 0.014, top, legs
        )
    look.rounded(
        "/World/Bedside/Drip/TableShelf",
        [(x0 + x1) / 2, (y0 + y1) / 2, FLOOR_Z + 0.22],
        [x1 - x0 - 0.04, y1 - y0 - 0.04, 0.014],
        "cabinet_white",
        0.004,
    )


def _stand_and_bag(backend, info):
    """The tabletop IV stand and the saline bag (CC-BY blood-bag model, recolored)."""
    from pxr import UsdShade

    from roboclock.envs.hospital.props import add_model, model_size

    stage, look = backend.stage, backend.appearance
    top = info["table"]["top"]
    pole, hook = np.asarray(info["pole"]), np.asarray(info["hook"])
    pole_top = hook[2] + 0.045
    look.cylinder("/World/Bedside/Drip/StandBase", [*pole, top + 0.008], 0.060, 0.016, "cabinet_white")
    look.cylinder(
        "/World/Bedside/Drip/StandPole", [*pole, (top + pole_top) / 2], 0.009, pole_top - top,
        "stainless",
    )
    arm_z = hook[2] + 0.030
    _bar(look, "/World/Bedside/Drip/StandArm", [*pole, arm_z], [*hook[:2], arm_z], 0.006, "stainless")
    _bar(look, "/World/Bedside/Drip/StandHook", [*hook[:2], arm_z], hook, 0.0025, "stainless")
    film = _material(look, "a1_saline_film", (0.86, 0.93, 0.97), roughness=0.2)
    bag_top = hook[2] + BAG_HANG
    # The bag's broad face (model -y/+y) and its label turn toward the head camera.
    toward, yaw = _face_camera(backend, info, hook[:2])
    if model_size("blood_bag") is not None:
        native = model_size("blood_bag")
        scale = tuple(a / b for a, b in zip(BAG_SIZE, native))
        node = add_model(
            stage,
            "blood_bag",
            "/World/Bedside/Drip/Bag",
            (hook[0], hook[1], bag_top - BAG_SIZE[2]),
            scale,
            yaw=np.radians(yaw),
        )
        UsdShade.MaterialBindingAPI.Apply(node.GetPrim()).Bind(
            film.material, UsdShade.Tokens.strongerThanDescendants
        )
    else:
        size = BAG_SIZE if abs(toward[1]) >= abs(toward[0]) else BAG_SIZE[1::-1] + BAG_SIZE[2:]
        look.rounded(
            "/World/Bedside/Drip/Bag",
            [hook[0], hook[1], bag_top - BAG_SIZE[2] / 2],
            size,
            film,
            0.010,
        )
    if _labels(backend):
        text_panel(
            stage,
            "/World/Bedside/Drip/BagLabel",
            [("0.9% NaCl", 1.0), ("500 mL", 0.7)],
            (0.08, 0.22, 0.55),
            (0.97, 0.98, 0.99),
            (*(hook[:2] + toward * (BAG_SIZE[1] / 2 + 0.002)), bag_top - 0.62 * BAG_SIZE[2]),
            (0.100, 0.070),
            yaw=yaw,
        )


def _face_camera(backend, info, point):
    """(unit xy vector from `point` toward the head camera, snapped to the nearest axis; the yaw in
    degrees that turns a quad readable from -y, or a model's -y face, toward it)."""
    from roboclock.envs.hospital.rig import dial_target

    fallback = np.asarray(info["camera"]["position"]) + np.array([*info["shift"], 0.0])
    d = np.asarray(dial_target(backend, fallback), dtype=float)[:2] - np.asarray(point)[:2]
    toward = (
        np.array([np.sign(d[0]), 0.0]) if abs(d[0]) > abs(d[1]) else np.array([0.0, np.sign(d[1])])
    )
    return toward, float(np.degrees(np.arctan2(toward[0], -toward[1])))


def chamber_frame(info):
    """Inner floor z, orifice z and xy axis of the drip chamber below the bag."""
    hook = np.asarray(info["hook"])
    bag_bottom = hook[2] + BAG_HANG - BAG_SIZE[2]
    # Bag port, a short feed tube and the spike above the chamber's top cap. 7 cm keeps the bag's
    # near lower edge from covering the chamber's top (OVER zone) in the 60-degree head view.
    top = bag_bottom - FEED_GAP
    floor = top - CAP_HEIGHT - CHAMBER_HEIGHT
    return {
        "xy": hook[:2],
        "floor": floor,
        "orifice": top - CAP_HEIGHT - 0.006,
        "top": top,
        "bag_bottom": bag_bottom,
    }


def _chamber(backend, info):
    """Clear chamber, liquid column, green band rings, the printed scale card and a drop."""
    from pxr import UsdGeom, UsdShade

    stage, look = backend.stage, backend.appearance
    frame = chamber_frame(info)
    (cx, cy), floor, top = frame["xy"], frame["floor"], frame["top"]
    prefix = "/World/Bedside/Drip/Chamber"
    look.cylinder(
        prefix + "/Port", [cx, cy, frame["bag_bottom"] - 0.008], 0.006, 0.018, "cap_white"
    )
    look.cylinder(prefix + "/Spike", [cx, cy, top - 0.006 + 0.0], 0.0045, 0.014, "cap_white")
    feed_top, feed_bottom = frame["bag_bottom"] - 0.016, top + 0.0005
    if feed_top > feed_bottom:
        look.cylinder(
            prefix + "/Feed",
            [cx, cy, (feed_top + feed_bottom) / 2],
            0.0028,
            feed_top - feed_bottom,
            "iv_bag",
        )
    look.cylinder(
        prefix + "/TopCap",
        [cx, cy, top - CAP_HEIGHT / 2],
        CHAMBER_RADIUS + 0.001,
        CAP_HEIGHT,
        "cap_white",
    )
    look.cylinder(
        prefix + "/BottomCap",
        [cx, cy, floor - CAP_HEIGHT / 2],
        CHAMBER_RADIUS + 0.001,
        CAP_HEIGHT,
        "cap_white",
    )
    look.cylinder(
        prefix + "/Outlet", [cx, cy, floor - CAP_HEIGHT - 0.006], 0.004, 0.012, "cap_white"
    )
    look.cylinder(
        prefix + "/Wall",
        [cx, cy, floor + CHAMBER_HEIGHT / 2],
        CHAMBER_RADIUS,
        CHAMBER_HEIGHT,
        _clear_wall(look),
    )
    toward, yaw = _face_camera(backend, info, (cx, cy))
    side = np.array([toward[1], -toward[0]])  # image left, seen from the camera
    edge = _material(look, "a1_wall_edge", (0.80, 0.86, 0.92), roughness=0.15)
    for k, sign in enumerate((-1.0, 1.0)):
        ex, ey = np.array([cx, cy]) + sign * (CHAMBER_RADIUS - 0.0006) * side
        look.cylinder(
            prefix + f"/WallEdge_{k}",
            [ex, ey, floor + CHAMBER_HEIGHT / 2],
            0.0009,
            CHAMBER_HEIGHT,
            edge,
        )
    liquid_mat = _material(look, "a1_saline", LIQUID_COLOR, roughness=0.55, emissive=LIQUID_GLOW)
    liquid = UsdGeom.Cylinder.Define(stage, prefix + "/Liquid")
    liquid.CreateRadiusAttr(LIQUID_RADIUS)
    liquid.CreateHeightAttr(1.0)
    liquid.CreateAxisAttr("Z")
    move = liquid.AddTranslateOp()
    size = liquid.AddScaleOp()
    UsdShade.MaterialBindingAPI.Apply(liquid.GetPrim()).Bind(liquid_mat.material)
    drop = UsdGeom.Sphere.Define(stage, prefix + "/Drop")
    drop.CreateRadiusAttr(DROP_RADIUS)
    drop_move = drop.AddTranslateOp()
    drop_size = drop.AddScaleOp()
    UsdShade.MaterialBindingAPI.Apply(drop.GetPrim()).Bind(liquid_mat.material)
    surface_mat = _material(look, "a1_surface", SURFACE_COLOR, roughness=0.3, emissive=SURFACE_GLOW)
    surface = UsdGeom.Cylinder.Define(stage, prefix + "/Surface")
    surface.CreateRadiusAttr(LIQUID_RADIUS + 0.0004)
    surface.CreateHeightAttr(0.003)
    surface.CreateAxisAttr("Z")
    surface_move = surface.AddTranslateOp()
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(surface_mat.material)
    marks = band_marks(info["band"])
    ring_r = CHAMBER_RADIUS + 0.0030
    for key, color in (("band_low", RING_LOW_COLOR), ("band_high", RING_HIGH_COLOR)):
        ring = _material(look, f"a1_{key}_ring", color, roughness=0.4, emissive=2.0)
        look.cylinder(prefix + f"/Mark_{key}", [cx, cy, floor + marks[key]], ring_r, 0.0035, ring)
    # Printed level scale behind the chamber (away from the camera): LOW / TARGET band / OVER zone.
    card_w = CARD_WIDTH
    behind = np.array([cx, cy]) - (CHAMBER_RADIUS + 0.004) * toward + CARD_OFFSET_X * side
    center = [*behind, floor + CHAMBER_HEIGHT / 2]
    textured_quad(
        stage,
        prefix + "/Card",
        scale_texture(info["band"]),
        center,
        (card_w, CHAMBER_HEIGHT + 2 * CAP_HEIGHT),
        yaw=yaw,
        emissive=0.0,
    )
    backer = np.abs(side) * (card_w + 0.004) + np.abs(toward) * 0.0024
    look.rounded(
        prefix + "/CardBacker",
        [*(behind - 0.0016 * toward), center[2]],
        [*backer, CHAMBER_HEIGHT + 2 * CAP_HEIGHT + 0.004],
        "cabinet_white",
        0.0008,
    )
    visual = {
        "liquid": (liquid.GetPrim(), move, size),
        "drop": (drop.GetPrim(), drop_move, drop_size),
        "surface": surface_move,
        "frame": frame,
        "last": None,
    }
    _set_level(visual, backend.task.initial_quantity(), 0.0, flowing=True)
    return visual


def _set_level(visual, quantity, time, flowing):
    from pxr import Gf, UsdGeom

    frame = visual["frame"]
    (cx, cy), floor = frame["xy"], frame["floor"]
    height = max(liquid_height(quantity), 1e-4)
    key = (round(height, 5), round(float(time), 3) if flowing else None)
    if key == visual["last"]:
        return
    visual["last"] = key
    prim, move, size = visual["liquid"]
    move.Set(Gf.Vec3d(float(cx), float(cy), float(floor + height / 2)))
    size.Set(Gf.Vec3f(1.0, 1.0, float(height)))
    visual["surface"].Set(Gf.Vec3d(float(cx), float(cy), float(floor + height - 0.0012)))
    drop, drop_move, drop_size = visual["drop"]
    if flowing:
        z, scale = drop_state(time, floor + height, frame["orifice"])
        drop_move.Set(Gf.Vec3d(float(cx), float(cy), float(z)))
        drop_size.Set(
            Gf.Vec3f(float(scale), float(scale), float(scale) * (1.15 if scale >= 1.0 else 1.0))
        )
        UsdGeom.Imageable(drop).MakeVisible()
    else:
        UsdGeom.Imageable(drop).MakeInvisible()


def _line(backend, info, origin):
    """One continuous IV line: chamber outlet, through the organizer, down and onto the bed.

    The chamber hangs beyond the organizer's CLOSED end (clear of the arm), so the line enters
    there, runs under the roller and leaves at the OPEN end, below the hand, toward the patient.
    """
    look = backend.appearance
    prefix = "/World/Bedside/Drip/Line"
    frame = chamber_frame(info)
    cx, cy = frame["xy"]
    shift = np.array([*info["shift"], 0.0])
    outlet = np.array([cx, cy, frame["floor"] - CAP_HEIGHT - 0.012])
    x, y, z = origin
    enter = np.array([x + TRAVEL + 0.050, y, z - 0.019])
    leave = np.array([x - 0.050, y, z - 0.019])
    upstream = [outlet, outlet + [0.0, 0.0, -0.035], enter + [0.022, 0.0, 0.012], enter]
    downstream = [leave, leave + [-0.022, 0.0, -0.018]] + [
        np.asarray(point) + shift for point in LINE_TO_BED
    ]
    points = _spline(upstream, 8)[:-1] + [enter, leave] + _spline(downstream, 6)[1:]
    for index, (first, second) in enumerate(zip(points, points[1:])):
        if np.linalg.norm(np.asarray(second) - np.asarray(first)) > 1e-5:
            _bar(look, f"{prefix}/Tube_{index:03d}", first, second, 0.0018, "iv_bag")
    end, before = np.asarray(points[-1]), np.asarray(points[-2])
    axis = (end - before) / np.linalg.norm(end - before)
    _bar(look, prefix + "/Connector", end, end + axis * 0.022, 0.0035, "cap_white")
    _bar(look, prefix + "/ConnectorCap", end + axis * 0.022, end + axis * 0.034, 0.0042, "cap_blue")


def _spline(points, samples=6):
    """Catmull-Rom points through the given control points (ends repeated)."""
    p = (
        [np.asarray(points[0], dtype=float)]
        + [np.asarray(q, dtype=float) for q in points]
        + [np.asarray(points[-1], dtype=float)]
    )
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


def _robot_base(backend, info):
    """The robot's wheeled base under both X5 columns (visual only; the robot itself is fixed)."""
    from roboclock.envs.lab.robot import mount_positions

    look = backend.appearance
    config = getattr(backend, "config", None) or look.config
    z = backend.layout.table_height
    mounts = [np.asarray(m[:2], dtype=float) for m in mount_positions(config, "A1")]
    lo, hi = np.min(mounts, axis=0) - 0.13, np.max(mounts, axis=0) + 0.13
    lo[0] -= 0.06
    center, size = (lo + hi) / 2, hi - lo
    shell = _material(look, "a1_base_shell", (0.86, 0.88, 0.90), roughness=0.45)
    dark = _material(look, "a1_base_dark", (0.10, 0.11, 0.12), roughness=0.6)
    look.rounded("/World/Bedside/RobotBase", [*center, FLOOR_Z + 0.075], [*size, 0.11], shell, 0.02)
    look.rounded(
        "/World/Bedside/RobotBumper",
        [*center, FLOOR_Z + 0.030],
        [*(size + 0.012), 0.03],
        dark,
        0.01,
    )
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        wheel = [
            center[0] + sx * (size[0] / 2 - 0.05),
            center[1] + sy * (size[1] / 2 - 0.05),
            FLOOR_Z + 0.03,
        ]
        look.cylinder(f"/World/Bedside/RobotWheel_{i}", wheel, 0.03, 0.025, dark, axis="Y")
    for i, m in enumerate(mounts):
        look.rounded(
            f"/World/Bedside/RobotColumn_{i}",
            [*m, (FLOOR_Z + 0.13 + z) / 2],
            [0.12, 0.12, z - FLOOR_Z - 0.13],
            dark,
            0.006,
        )


def _sill_cabinet(backend, info):
    """A white convector cover under the window (the ward's bare wall there renders near-black)."""
    if SILL_CABINET is None:
        return
    look = backend.appearance
    shift = np.asarray(info["shift"])
    (x0, x1), (y0, y1), top = SILL_CABINET
    center = [(x0 + x1) / 2 + shift[0], (y0 + y1) / 2 + shift[1], (FLOOR_Z + top) / 2]
    look.rounded(
        "/World/Bedside/Drip/SillCabinet",
        center,
        [x1 - x0, y1 - y0, top - FLOOR_Z],
        "cabinet_white",
        0.01,
    )
    grille = _material(look, "a1_grille", (0.62, 0.66, 0.70), roughness=0.5)
    for k in range(3):
        z = top - 0.06 - 0.025 * k
        look.rounded(
            f"/World/Bedside/Drip/SillGrille_{k}",
            [center[0], y1 + shift[1] + 0.001, z],
            [x1 - x0 - 0.06, 0.004, 0.008],
            grille,
            0.002,
        )
    look.rounded(
        "/World/Bedside/Drip/SillPlinth",
        [center[0], center[1], FLOOR_Z + 0.03],
        [x1 - x0 - 0.01, y1 - y0 + 0.004, 0.06],
        grille,
        0.004,
    )


def _cue(backend):
    """The countdown dial on a box on a post on the table, right of the bag, turned to the camera."""
    from pxr import Gf, UsdGeom

    stage, look = backend.stage, backend.appearance
    indicators = getattr(backend.layout, "indicators", ())
    if not indicators:
        return
    info = backend.layout.info["drip"]
    center = np.asarray(indicators[0].position, dtype=float) + [0.0, 0.0, 0.005]  # housing center
    contact = center + [0.0, 0.0, 0.050 + 0.003]  # where the dial's bezel meets the housing
    from roboclock.envs.hospital.rig import dial_target

    camera = np.asarray(info["camera"]["position"]) + np.array([*info["shift"], 0.0])
    camera = dial_target(backend, camera)  # the policy's head camera on the rig
    facing = float(np.degrees(np.arctan2(camera[1] - center[1], camera[0] - center[0])))
    dials = stage.GetPrimAtPath("/World/Dials")
    if dials.IsValid():
        xform = UsdGeom.Xformable(dials)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(*contact.tolist()))
        xform.AddRotateZOp().Set(facing)
        xform.AddScaleOp().Set(Gf.Vec3f(DIAL_SCALE, DIAL_SCALE, DIAL_SCALE))
        xform.AddTranslateOp(opSuffix="pivot", isInverseOp=True)
    for path in ("/World/Labels/indicator_indicator", "/World/Equipment/indicator_Housing_Visual"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
    # A slim monitor box under the dial (inside the housing's collider) on a post from the table.
    look.rounded(
        "/World/Bedside/Drip/CueBox",
        center + [0.0, 0.0, 0.0145],
        (0.044, 0.050, 0.081),
        "cabinet_white",
        0.006,
    )
    bottom, top = center[2] + 0.0145 - 0.0405, info["table"]["top"]
    if bottom > top + 0.002:
        look.cylinder(
            "/World/Bedside/Drip/CuePost",
            [center[0], center[1], (top + bottom) / 2],
            0.008,
            bottom - top,
            "stainless",
        )
    if _labels(backend):
        toward, yaw = _face_camera(backend, info, center[:2])
        text_panel(
            stage,
            "/World/Bedside/Drip/CueText",
            [("DRIP", 1.0)],
            (1.0, 1.0, 1.0),
            (0.10, 0.36, 0.62),
            [*(center[:2] + toward * 0.0262), center[2] + 0.022],
            (0.038, 0.016),
            yaw=yaw,
        )


def _labels(backend):
    config = getattr(backend, "config", None) or getattr(backend.appearance, "config", {})
    return bool(config.get("visuals", {}).get("labels", True))


def _text_texture(lines, fg, bg, aspect, fit=None):
    """A cached PNG with centered bold lines of text filling a panel of the given aspect."""
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    height = 256
    size = (max(64, int(round(height * aspect))), height)
    key = hashlib.sha256(repr((lines, fg, bg, size, fit)).encode()).hexdigest()[:20]
    path = cache / f"a1drip_{key}.png"
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


def scale_texture(band):
    """The printed level scale (cached PNG): LOW, a green TARGET band, a red line and OVER zone.

    The image spans the chamber plus a cap height above and below. Seen from the camera (the card
    is turned to face +y) its left part carries the words and the right part sits behind the
    chamber, where tick marks every 10 % run along the chamber's edge.
    """
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    width = 468
    rows = int(round(width * (CHAMBER_HEIGHT + 2 * CAP_HEIGHT) / CARD_WIDTH))
    key = hashlib.sha256(
        repr(("a1scale", tuple(band), width, rows, CARD_OFFSET_X, CHAMBER_RADIUS)).encode()
    ).hexdigest()[:20]
    path = cache / f"a1drip_scale_{key}.png"
    if path.exists():
        return path
    top, bottom, zero, full = scale_rows(band, rows)
    image = Image.new("RGB", (width, rows), (246, 247, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, top), fill=(226, 72, 60))  # OVER zone
    draw.rectangle((0, top, width, bottom), fill=(40, 190, 90))  # TARGET band
    draw.rectangle((0, bottom, width, rows), fill=(214, 218, 222))  # LOW zone
    draw.rectangle((0, top - 6, width, top + 6), fill=(150, 14, 10))  # band top: overfill line
    draw.rectangle((0, bottom - 6, width, bottom + 6), fill=(10, 110, 45))  # band bottom
    # Where the chamber's edge falls on the card (u of the +x edge, seen from +y).
    edge = int(round(width * (0.5 + (CARD_OFFSET_X - CHAMBER_RADIUS - 0.002) / CARD_WIDTH)))
    for k in range(11):
        row = int(round(zero + (full - zero) * k / 10))
        length = 30 if k % 5 == 0 else 18
        draw.rectangle((edge - length, row - 2, edge, row + 2), fill=(40, 46, 52))
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    def word(text, center_row, color, height):
        size = int(height)
        while True:
            try:
                font = ImageFont.truetype(bold, size)
            except OSError:
                font = ImageFont.load_default(size=size)
            if font.getlength(text) <= (edge - 40) * 0.92 or size <= 10:
                break
            size -= 2
        draw.text(((edge - 34) / 2, center_row), text, fill=color, font=font, anchor="mm")

    word("OVER", (top - 8) / 2, (255, 255, 255), min(80, top * 0.6))
    word("TARGET", (top + bottom) / 2, (255, 255, 255), (bottom - top) * 0.62)
    word("LOW", (bottom + rows) / 2, (70, 76, 82), min(90, (rows - bottom) * 0.4))
    image.save(path)
    return path


def textured_quad(stage, path, texture, center, size, yaw=0.0, emissive=0.0):
    """A quad in the x-z plane showing `texture`, readable from -y, turned by `yaw` degrees about z."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    w, h = size
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
    if yaw:
        mesh.AddRotateZOp().Set(float(yaw))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def text_panel(stage, path, lines, fg, bg, center, size, fit=None, yaw=0.0):
    """Centered bold lines of text on a quad in the x-z plane (see textured_quad)."""
    texture = _text_texture(lines, fg, bg, size[0] / size[1], fit)
    return textured_quad(stage, path, texture, center, size, yaw)


# --- camera and measurement ----------------------------------------------------------------------


def _camera(backend):
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    info = backend.layout.info["drip"]
    view = info["camera"]
    shift = np.array([*info["shift"], 0.0])
    position = np.asarray(view["position"]) + shift + backend.layout.camera_jitter
    target = np.asarray(view["look_at"]) + shift
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*target.tolist()), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    if view.get("focal_mm"):
        camera.set_focal_length(float(view["focal_mm"]) / 1000)
    for _ in range(3):
        backend.world.render()


def _episode(backend):
    """Latched per-episode measurements, kept on the backend and reset with each new task."""
    task = getattr(backend, "task", None)
    memo = getattr(backend, "_a1_drip_episode", None)
    if memo is None or memo["task"] is not task:
        memo = {
            "task": task,
            "contact_seen": False,
            "moved": False,
            "opened": False,
            "actuated": False,
        }
        backend._a1_drip_episode = memo
    return memo


def roller_controls(origin, roller, memo):
    """Device controls from the measured roller pose and contacts (updates the latches in `memo`).

    The roller starts at the CLOSED end. Opened: it reached the OPEN end after robot contact.
    Actuated (the scored closing): it is back at the CLOSED end after having been opened.
    """
    delta = np.asarray(roller.position, dtype=float) - np.asarray(origin, dtype=float)
    position = float(delta[0])
    contact = bool(roller.robot_contact or roller.grasped)
    memo["contact_seen"] = memo["contact_seen"] or contact
    memo["moved"] = memo["moved"] or position < TRAVEL - MOTION_THRESHOLD
    at_open = position <= END_TOLERANCE
    at_closed = position >= TRAVEL - END_TOLERANCE
    memo["opened"] = memo["opened"] or (at_open and memo["contact_seen"])
    memo["actuated"] = memo["actuated"] or (at_closed and memo["opened"])
    return {
        "roller_position_m": position,
        "roller_closed_fraction": float(np.clip(position / TRAVEL, 0.0, 1.0)),
        "roller_speed_m_s": float(np.linalg.norm(roller.velocity)),
        "roller_axis_error_m": float(np.linalg.norm(delta[1:])),
        "roller_contact": contact,
        "roller_contact_seen": memo["contact_seen"],
        "roller_at_open_end": bool(at_open),
        "roller_at_closed_end": bool(at_closed),
        "device_moved": memo["moved"],
        "device_opened": memo["opened"],
        "device_actuated": memo["actuated"],
    }


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_a1_camera_for", None) is not backend.task:
        backend._a1_camera_for = backend.task
        _camera(backend)
    roller = state.objects.get(ROLLER_NAME)
    if roller is None:
        return {}
    memo = _episode(backend)
    controls = roller_controls(backend.layout.info["drip"]["origin"], roller, memo)
    task = getattr(backend, "task", None)
    visual = getattr(backend, "_a1_drip_visual", None)
    if task is not None and hasattr(task, "quantity"):
        controls["chamber_level_pct"] = float(task.quantity())
        if visual is not None:
            flowing = (
                controls["device_opened"]
                and task.action_time is None
                and not controls["device_actuated"]
            )
            _set_level(visual, task.quantity(), task.time, flowing)
    return controls


# --- privileged diagnostic oracle ----------------------------------------------------------------


def command(planner, task, state):
    """Settle on the CLOSED roller, pull it OPEN, hold, and push it CLOSED at the window's middle."""
    info = task.layout.info
    goals = hand_goals(info["drip"]["origin"])
    if planner.last_goal is None:
        planner.last_goal = state.ee_position.copy()
    planner.yaw = float(HAND_YAW)
    planner.open_width = planner.close_width = GRIP_WIDTH
    planner.gentle = True
    controls = state.controls
    lift = np.array([0.0, 0.0, HOVER])
    if controls.get("device_actuated") or task.action_time is not None:
        return planner.finish(state, goals["pushed"] + lift, False, speed=0.15)
    opened = bool(controls.get("device_opened")) or getattr(task, "open_time", None) is not None
    if not opened:
        if task.time < OPEN_AT:  # down onto the straddle over the CLOSED roller
            return planner.finish(state, goals["closed"], False, speed=0.08)
        if getattr(planner, "a1_open_for", None) is not task:
            planner.a1_open_for = task
            task.event_log.append({"time": task.time, "type": "oracle_open_start"})
        return planner.finish(state, goals["pulled"], False, speed=PUSH_SPEED)
    start = slide_start(task.window_open, task.window_close)
    if task.time < start:  # back off the stop to the straddle over the OPEN roller, no contact
        return planner.finish(state, goals["open"], False, speed=0.05)
    if getattr(planner, "a1_slide_for", None) is not task:
        planner.a1_slide_for = task
        task.event_log.append({"time": task.time, "type": "oracle_slide_start"})
    return planner.finish(state, goals["pushed"], False, speed=PUSH_SPEED)
