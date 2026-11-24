"""Detailed V5-derived rooms with functional sports equipment."""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind, pbr
from roboclock.envs.sports.arena.geometry import cup, ellipsoid, ring, rod


def spring(b, name, position, axis, travel, stiffness):
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, UsdGeom, UsdPhysics

    path = f"/World/Controls/{name}"
    UsdGeom.Xform.Define(b.stage, path)
    obj = b.world.scene.add(
        SingleRigidPrim(path, name=name, position=np.asarray(position), mass=0.18)
    )
    joint = UsdPhysics.PrismaticJoint.Define(b.stage, path + "/Spring")
    joint.CreateBody1Rel().SetTargets([path])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*position))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(-travel if axis == "Z" else 0.0)
    joint.CreateUpperLimitAttr(0.0 if axis == "Z" else travel)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(12.0)
    drive.CreateTargetPositionAttr(0.0)
    b.pads[name] = obj
    return path


def build_scene(b):
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, UsdGeom, UsdLux

    from roboclock.envs.sports.arcade.classic import build_room_classic
    from roboclock.resources import resolve_asset

    table_center = [0.42, 0, 0.72] if b.kind == "A3" else [0.49, 0, 0.72]
    table_size = [1.04, 1.0, 0.06] if b.kind == "A3" else [0.64, 1.05, 0.06]
    if b.kind == "B2":
        table_center, table_size = [0, 0, 0.12], [0.5, 0.6, 0.24]
    b.world.scene.add(
        FixedCuboid(
            "/World/Table",
            name="table",
            position=np.array(table_center),
            scale=np.array(table_size),
        )
    )
    b.appearance = a = build_room_classic(b, "arcade")
    UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/defaultGroundPlane")).MakeInvisible()
    UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/Table")).MakeInvisible()
    for prefix in ["Worktop"]:
        UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/Room/" + prefix)).MakeInvisible()
    for prim in list(b.stage.Traverse()):
        if str(prim.GetPath()).startswith(
            ("/World/Refined/Pedestal", "/World/Refined/TableEdge", "/World/Refined/Vent")
        ):
            UsdGeom.Imageable(prim).MakeInvisible()
    if b.kind != "B2":
        a.rounded("/World/Equipment/Tabletop", table_center, table_size, "ink", 0.015)
        for x in [0.22, 0.76]:
            for y in [-0.42, 0.42]:
                a.rounded(
                    f"/World/Equipment/Leg{x}_{y}".replace(".", "_").replace("-", "n"),
                    [x, y, 0.36],
                    [0.05, 0.05, 0.7],
                    "steel",
                    0.008,
                )
        for y in [-0.51, 0.51]:
            rod(
                b,
                f"/World/Equipment/Edge{int(y * 100) + 100}",
                [0.18, y, 0.742],
                [0.8, y, 0.742],
                0.006,
                "brass",
            )
    else:
        from roboclock.envs.sports.arena.boxing_ring import build_ring

        build_ring(b)
    dome = UsdLux.DomeLight.Define(b.stage, "/World/Lights/Dome")
    dome.CreateTextureFileAttr(str(resolve_asset(b.config["visuals"]["hdr"])))
    dome.CreateIntensityAttr(650)
    fill = UsdLux.RectLight.Define(b.stage, "/World/Lights/Fill")
    fill.AddTranslateOp().Set(Gf.Vec3d(1.5, -0.5, 2.4))
    fill.CreateIntensityAttr(850)
    fill.CreateWidthAttr(1.2)
    fill.CreateHeightAttr(0.8)
    if b.kind == "A3":
        basketball(b)
    elif b.kind == "B2":
        boxing(b)
    else:
        cups(b)
    cp = np.array(
        [0.06, -2.55, 1.88]
        if b.kind == "A3"
        else [1.85, -2.65, 1.82]
        if b.kind == "B2"
        else [1.4, -1.7, 1.8]
    )
    target = np.array(
        [0.28, 0.05, 1.03]
        if b.kind == "B2"
        else [0.42, 0, 1.02]
        if b.kind == "A3"
        else [0.32, 0.01, 0.92]
    )
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*cp), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    cq = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
    b.camera_position, b.camera_orientation = cp, cq
    b.camera = Camera(
        "/World/Camera",
        position=cp,
        orientation=cq,
        resolution=tuple(b.config["sim"]["camera_resolution"]),
    )


def basketball(b):
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import PhysxSchema, UsdGeom

    a = b.appearance
    pbr(a, "basket_leather", [0.68, 0.21, 0.025], texture="leather_white", uv=(7, 7))
    # The complete support fixture moves with the ball only during scene reset.
    x, y = b.config.get("basketball", {}).get("pickup_xy", [0.32, -0.22])
    b.ball_radius = 0.029
    path = "/World/Objects/ball"
    UsdGeom.Xform.Define(b.stage, path)
    shape, _ = b.collider(path + "/Surface", "sphere", [0, 0, 0], [b.ball_radius], [1, 1, 1])
    bind(b, path + "/Surface", a.material("basket_leather"))
    for axis in ["xy", "yz"]:
        ring(
            b,
            path + "/Seam" + axis,
            [0, 0, 0],
            b.ball_radius + 0.0001,
            0.0006,
            "ink",
            False,
            64,
            axis,
        )
    # An additional curved seam gives the miniature ball a basketball silhouette.
    ring(
        b,
        path + "/SeamOffset",
        [0, 0, 0.011],
        np.sqrt(b.ball_radius**2 - 0.011**2),
        0.0005,
        "ink",
        False,
        48,
    )
    b.objects["ball"] = b.world.scene.add(
        SingleRigidPrim(path, name="ball", position=np.array([x, y, 0.806]), mass=0.035)
    )
    PhysxSchema.PhysxRigidBodyAPI.Apply(b.objects["ball"].prim).CreateEnableCCDAttr(True)
    b.collider(
        "/World/BallRack/Base", "cylinder", [x, y, 0.762], [0.055, 0.024], [0.04, 0.05, 0.08]
    )
    ring(b, "/World/BallRack/Cradle", [x, y, 0.780], 0.021, 0.003, "brass", True, 24)
    b.hoop = np.array([0.78, 0.06, 1.045])
    b.hoop_radius = 0.085
    rod(b, "/World/Basket/Post", [0.98, 0.06, 0.75], [0.98, 0.06, 1.5], 0.023, "steel", True)
    a.rounded("/World/Basket/Backboard", [0.97, 0.06, 1.23], [0.025, 0.47, 0.37], "cream", 0.014)
    b.collider(
        "/World/Basket/BoardCollider",
        "cube",
        [0.98, 0.06, 1.23],
        [0.025, 0.47, 0.37],
        [0.7, 0.7, 0.7],
    )
    for y in [-0.055, 0.175]:
        rod(
            b,
            "/World/Basket/MarkV" + str(int(y * 100) + 100),
            [0.949, y, 1.10],
            [0.949, y, 1.23],
            0.003,
            "red_lacquer",
        )
    for z in [1.1, 1.23]:
        rod(
            b,
            "/World/Basket/MarkH" + str(int(z * 100)),
            [0.949, -0.055, z],
            [0.949, 0.175, z],
            0.003,
            "red_lacquer",
        )
    ring(b, "/World/Basket/Rim", b.hoop, b.hoop_radius, 0.004, "red_lacquer", True, 48)
    rod(b, "/World/Basket/Mount", b.hoop + [0.085, 0, 0], [0.96, 0.06, 1.045], 0.009, "steel", True)
    # Decorative net follows the open scoring aperture; rigid rim remains authoritative.
    for i in range(16):
        theta = 2 * np.pi * i / 16
        for sign in [-1, 1]:
            for j in range(4):
                t1 = theta + sign * j * 0.13
                t2 = theta + sign * (j + 1) * 0.13
                r1 = 0.082 - j * 0.006
                r2 = 0.082 - (j + 1) * 0.006
                p = b.hoop + [r1 * np.cos(t1), r1 * np.sin(t1), -j * 0.035]
                q = b.hoop + [r2 * np.cos(t2), r2 * np.sin(t2), -(j + 1) * 0.035]
                rod(b, f"/World/Basket/Net{i}_{sign + 1}_{j}", p, q, 0.001, "cream")
    tray_position, tray_size = [0.73, 0.06, 0.765], [0.44, 0.38, 0.026]
    a.rounded("/World/Basket/ReturnTray", tray_position, tray_size, "teal_lacquer", 0.014)
    # Rounded appearance meshes are visual-only. Support the falling ball at
    # the tray surface, rather than the table 28 mm below it.
    tray_collider, _ = b.collider(
        "/World/Basket/ReturnTrayCollider", "cube", tray_position, tray_size, [1, 1, 1]
    )
    UsdGeom.Imageable(tray_collider).MakeInvisible()
    for y in [-0.14, 0.26]:
        rod(
            b,
            f"/World/Basket/Rail{int(y * 100) + 100}",
            [0.51, y, 0.80],
            [1.0, y, 0.80],
            0.008,
            "steel",
            True,
        )
    equipment_panel(
        b,
        a,
        "/World/Basket/Header",
        [0.70, 0.55, 1.53],
        [0.72, 0.30],
        "BUZZER SHOT",
        "RELEASE BEFORE THE BUZZER",
        accent=(241, 159, 67),
    )
    # Shot clock (the A3 cue): status text on a dark case, authored where the old
    # status label hung above the hoop; head_view.DISPLAYS moves the group low onto
    # the table, facing the rig's head camera.
    from roboclock.envs.sports.arcade.scenes import vertical_label

    a.rounded("/World/ShotClock/Case", [0.70, 0.553, 1.445], [0.68, 0.012, 0.10], "ink", 0.01)
    vertical_label(b, "/World/ShotClock/Status", "READY", [0.70, 0.543, 1.445], [0.64, 0.065])


def boxing(b):
    from pxr import UsdGeom

    from roboclock.envs.sports.arena.geometry import add_glove_mass
    from roboclock.envs.sports.arena.tasks import PAD_X, pad_center, pad_offset

    a = b.appearance
    for side in ("left", "right"):
        color = "glove_red" if side == "left" else "glove_blue"
        root = b.tool_paths[side]
        # Explicit wrist-mounted boxing tool, carried by the articulation link.
        ellipsoid(b, root + "/Glove/Knuckles", [0, 0, 0.174], [0.052, 0.048, 0.067], color, True)
        ellipsoid(
            b, root + "/Glove/Thumb", [0.042, -0.01, 0.155], [0.022, 0.023, 0.042], color, True
        )
        a.rounded(root + "/Glove/Cuff", [0, 0, 0.110], [0.080, 0.066, 0.045], "ink", 0.012)
        a.rounded(root + "/Glove/Strap", [0, -0.034, 0.112], [0.073, 0.004, 0.018], "cream", 0.003)
        # Added tool mass is included explicitly on its owning wrist link.
        from scipy.spatial.transform import Rotation

        UsdGeom.Xform.Define(b.stage, root + "/Glove").AddRotateYOp().Set(90)
        tool_rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        b.glove_rotation = tool_rotation
        add_glove_mass(b.stage.GetPrimAtPath(root), tool_rotation)
        pos = pad_center(b.config["workflow"], side).tolist()
        path = spring(b, "pad_" + side, pos, "X", 0.035, 380)
        ellipsoid(b, path + "/Face", [0, 0, 0], [0.030, 0.073, 0.096], "pad_leather", True)
        ring(b, path + "/Target", [-0.0305, 0, 0], 0.041, 0.0025, color, False, 48, "yz")
        for j in range(24):
            theta = j * np.pi / 12
            ellipsoid(
                b,
                path + f"/Stitch{j}",
                [-0.015, 0.069 * np.cos(theta), 0.090 * np.sin(theta)],
                [0.001, 0.0015, 0.002],
                "cream",
            )
        # The trainer frame moves with its pads (seeded ``workflow.pad_offset``).
        px, py = pos[0] - PAD_X, pos[1]
        rod(
            b,
            "/World/Trainer/Stem" + side,
            [0.71 + px, py, 0.20],
            [0.71 + px, py, 1.10],
            0.018,
            "steel",
            True,
        )
        rod(
            b,
            "/World/Trainer/Back" + side,
            [0.64 + px, py, 1.06],
            [0.73 + px, py, 1.06],
            0.015,
            "steel",
        )
    dx, dy = pad_offset(b.config["workflow"])
    a.rounded("/World/Trainer/Base", [0.71 + dx, dy, 0.10], [0.28, 0.85, 0.20], "ink", 0.04)
    from roboclock.envs.sports.arena import coach_display

    # Coach screen inside the RoboDojo head camera's view (arena/coach_display.py),
    # low on the robot's left beside the pad trainer. Its post is visual only (the
    # earlier post was a static collider far behind the trainer, out of reach).
    (sx, sy), bottom = coach_display.stand()
    rod(
        b,
        "/World/Trainer/DisplayPost",
        [sx, sy, 0.10],
        [sx, sy, bottom + 0.02],
        coach_display.STAND_RADIUS,
        "steel",
        False,
    )
    foot = a.rounded("/World/Trainer/DisplayFoot", [sx, sy, 0.05], [0.24, 0.22, 0.10], "ink", 0.02)
    foot.AddRotateZOp().Set(float(coach_display.yaw_deg()))
    screen = coach_display.frame(b)
    equipment_panel(
        b,
        a,
        screen + "/Header",
        list(coach_display.DESIGN_ORIGIN),
        [0.90, 0.43],
        "COMBO TRAINER",
        "STRIKE / RETRACT / GUARD",
        accent=(237, 159, 103),
    )
    from roboclock.envs.sports.arcade.scenes import vertical_label

    vertical_label(b, screen + "/WorkflowStatus", "READY", [0.65, 0.58, 1.36], [0.68, 0.07])


def cups(b):
    from pxr import Gf, UsdGeom

    from roboclock.envs.sports.arcade.scenes import vertical_label
    from roboclock.envs.sports.arena.cup_patterns import icons, layout

    a = b.appearance
    a.rounded("/World/Cups/Mat", [0.43, 0, 0.7501], [0.48, 0.90, 0.0003], "felt", 0.012)
    b.cup_sources = {}
    b.cup_targets, supports = layout(0)
    colors = [[0.70, 0.075, 0.035], [0.93, 0.57, 0.065], [0.025, 0.43, 0.50]]
    rng = np.random.default_rng(b.config["seed"])
    if b.config["seed"]:
        rng.shuffle(colors)
    # Seeded start offsets (``workflow.cup_source_offsets``, ``sports/spatial.py``, since
    # 2026-11-20: row U(+/-8 mm) per side plus U(+/-1 mm) x, U(+/-2 mm) y per cup, every seed).
    # Until then each cup moved U(+/-0.8 mm) on the colour stream, and not at seed 0.
    offsets = b.config["workflow"].get("cup_source_offsets")
    for side, sign in [("left", 1), ("right", -1)]:
        for i in range(3):
            name = f"{side}_{i}"
            p = np.array([0.21 + i * 0.07, sign * 0.34, 0.786])
            if offsets is not None:
                p[:2] += offsets[name]
            elif b.config["seed"]:
                p[:2] += rng.uniform(-0.0008, 0.0008, 2)
            b.cup_sources[name] = p
            b.objects[name] = cup(b, name, p, colors[i])
            ring(
                b,
                f"/World/Cups/SourceMark{name}",
                [p[0], p[1], 0.755],
                0.036,
                0.0008,
                "brass",
                False,
                32,
            )
    for name, target in b.cup_targets.items():
        path = f"/World/Cups/TargetMark{name}"
        marker = UsdGeom.Xform.Define(b.stage, path)
        marker.AddTranslateOp().Set(Gf.Vec3d(float(target[0]), float(target[1]), 0.755))
        ring(b, path, [0, 0, 0], 0.034, 0.0008, "cream", False, 32)
        if supports[name] != ["Table"]:
            UsdGeom.Imageable(marker).MakeInvisible()
    bellpos = [0.29, -0.075, 0.778]
    path = spring(b, "bell", bellpos, "Z", 0.006, 550)
    ellipsoid(b, path + "/Dome", [0, 0, 0], [0.029, 0.029, 0.015], "brass", True)
    a.rounded(
        "/World/Cups/BellBase", [bellpos[0], bellpos[1], 0.762], [0.078, 0.078, 0.017], "ink", 0.015
    )
    rod(b, "/World/Cups/ScreenStand", [0.63, 0.48, 0.75], [0.63, 0.48, 1.45], 0.014, "steel")
    a.rounded("/World/Cups/ScreenHousing", [0.63, 0.46, 1.28], [0.50, 0.035, 0.36], "ink", 0.016)
    equipment_panel(
        b,
        a,
        "/World/Cups/Card",
        [0.63, 0.438, 1.28],
        [0.46, 0.31],
        "SPEED CUPS",
        "MATCH BOTH PATTERNS / RING THE BELL",
        accent=(104, 207, 184),
    )
    # Public card icons show the actual first target pattern.
    for side, dx in [("left", -0.11), ("right", 0.11)]:
        for i in range(3):
            ix, iz = icons(0, side)[i]
            x, z = 0.63 + dx + ix, 1.20 + iz
            a.rounded(
                f"/World/Cups/Icon{side}{i}",
                [x, 0.432, z],
                [0.035, 0.004, 0.031],
                a.material(f"{side}_{i}"),
                0.004,
            )
        vertical_label(
            b, f"/World/Cups/Side{side}", side.upper(), [0.63 + dx, 0.432, 1.154], [0.16, 0.023]
        )
    # Live status text in its own group (re-created on every change), moved with the
    # screen by head_view.DISPLAYS.
    vertical_label(b, "/World/CupScreen/Status", "READY", [0.63, 0.433, 1.40], [0.42, 0.035])


def equipment_panel(b, a, path, position, size, title, subtitle="", **kwargs):
    """Readable task artwork with space for live cues and target diagrams."""
    from roboclock.envs.sports.arcade.scenes import vertical_label

    x, y, z = position
    w, h = size
    a.rounded(path + "/Case", [x, y + 0.012, z], [w + 0.025, 0.024, h + 0.025], "ink", 0.012)
    a.rounded(path + "/Face", [x, y + 0.002, z], [w, 0.004, h], "navy", 0.004)
    for dz in [-h / 2 + 0.006, h / 2 - 0.006]:
        rod(
            b,
            path + f"/Edge{int((dz + 1) * 1000)}",
            [x - w / 2 + 0.012, y - 0.001, z + dz],
            [x + w / 2 - 0.012, y - 0.001, z + dz],
            0.0018,
            "brass",
        )
    vertical_label(b, path + "/Title", title, [x, y - 0.003, z + h * 0.24], [w * 0.90, h * 0.20])
    if subtitle:
        vertical_label(
            b, path + "/Subtitle", subtitle, [x, y - 0.003, z + h * 0.01], [w * 0.90, h * 0.085]
        )
