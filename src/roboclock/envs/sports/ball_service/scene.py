"""Tennis club dressing and contact-valid dual-basket / gated-magazine equipment.

Reuse existing HDR, scanned material maps, X5 manufacturer meshes, and the
legacy gravity-gate design. Machine casing, baskets and court are authored.
"""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind, panel, pbr
from roboclock.envs.sports.arena.geometry import ring
from roboclock.envs.sports._shared.workflow.geometry import rod
from roboclock.envs.sports._shared.workflow.mobile_scenes import supported
from roboclock.envs.sports._shared.workflow.scenes import display, rigid, room, solid


def dressing(b):
    from pxr import UsdGeom

    a = b.appearance
    for path in (
        "/World/Room/BackWall",
        "/World/Room/SideWall",
        "/World/Studio/Brand",
        "/World/Refined/Plant",
    ):
        prim = b.stage.GetPrimAtPath(path)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()
    pbr(a, "club_wall", [0.72, 0.75, 0.67], 0.84)
    pbr(a, "club_navy", [0.035, 0.10, 0.15], 0.5)
    pbr(a, "ball_felt", [0.65, 0.80, 0.055], 0.96, texture="leather_white", uv=(4, 4))
    pbr(a, "machine_white", [0.79, 0.82, 0.75], 0.32, metallic=0.15)
    solid(b, "/World/Club/SideWall", [-2.35, 0, 1.6], [0.08, 5.0, 3.2], "club_wall")
    solid(b, "/World/Club/Wall", [0.4, 1.85, 1.6], [5.4, 0.08, 3.2], "club_wall")
    solid(b, "/World/Club/Wainscot", [0.4, 1.79, 0.5], [5.4, 0.04, 1.0], "club_navy")
    solid(b, "/World/Club/Rail", [0.4, 1.75, 1.02], [5.4, 0.035, 0.025], "brass")
    solid(b, "/World/Court/Surface", [0.7, 0.15, 0.008], [4.8, 3.2, 0.012], "court_green")
    for j, y in enumerate((-1.25, 1.35)):
        solid(b, f"/World/Court/Line{j}", [0.7, y, 0.016], [4.3, 0.025, 0.003], "cream", 0.001)
    for j, x in enumerate((-1.35, 1.4, 2.8)):
        solid(b, f"/World/Court/Cross{j}", [x, 0.05, 0.016], [0.025, 2.6, 0.003], "cream", 0.001)
    # Full net and posts in background; no isolated cut-off room wall.
    for j, x in enumerate((-0.8, 1.45)):
        rod(b, f"/World/Net/Post{j}", [x, 1.30, 0.02], [x, 1.30, 1.02], 0.025, "steel")
    solid(b, "/World/Net/Tape", [0.325, 1.30, 0.94], [2.25, 0.025, 0.055], "cream", 0.004)
    for j in range(46):
        x = -0.8 + j * 0.05
        rod(b, f"/World/Net/V{j}", [x, 1.3, 0.13], [x, 1.3, 0.91], 0.0014, "ink")
    for j in range(17):
        z = 0.13 + j * 0.047
        rod(b, f"/World/Net/H{j}", [-0.8, 1.3, z], [1.45, 1.3, z], 0.0014, "ink")
    panel(
        b,
        a,
        "/World/Club/Identity",
        [0.28, 1.73, 2.03],
        [1.75, 0.46],
        "TENNIS CLUB",
        "CONTINUOUS BALL SERVICE",
        style="title",
        accent=(209, 192, 120),
    )
    # Reuse the legacy racket construction with proportionate strings and grip.
    for j, x in enumerate((-1.05, 1.55)):
        center = np.array([x, 1.72, 1.61])
        for i in range(48):
            angles = np.array([i, i + 1]) * 2 * np.pi / 48
            pts = [center + [0.12 * np.cos(q), 0, 0.165 * np.sin(q)] for q in angles]
            rod(b, f"/World/Rackets/R{j}/Rim{i}", *pts, 0.008, "teal_lacquer")
        for i, v in enumerate(np.linspace(-0.85, 0.85, 11)):
            half = np.sqrt(1 - v * v)
            rod(
                b,
                f"/World/Rackets/R{j}/H{i}",
                center + [-0.12 * half, 0, 0.165 * v],
                center + [0.12 * half, 0, 0.165 * v],
                0.0007,
                "cream",
            )
            rod(
                b,
                f"/World/Rackets/R{j}/V{i}",
                center + [0.12 * v, 0, -0.165 * half],
                center + [0.12 * v, 0, 0.165 * half],
                0.0007,
                "cream",
            )
        rod(
            b,
            f"/World/Rackets/R{j}/Shaft",
            center + [0, 0, -0.15],
            center + [0, 0, -0.35],
            0.009,
            "steel",
        )
        rod(
            b,
            f"/World/Rackets/R{j}/Grip",
            center + [0, 0, -0.31],
            center + [0, 0, -0.44],
            0.014,
            "rubber",
        )


def equipment(b):
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    c = b.config["service"]
    a = b.appearance
    b.ball_sources = c["sources"]
    b.hopper_cells = [np.array(p) for p in c["cells"]]
    b.gate_drives = {}
    for side, sign in [("Left", 1), ("Right", -1)]:
        root = "/World/Supply" + side
        supported(b, root)
        solid(
            b,
            root + "/Deck",
            [0.35, sign * 0.36, 0.811],
            [0.41, 0.23, 0.03],
            "clean_wood",
            0.01,
            True,
        )
        for j, x in enumerate((0.16, 0.54)):
            solid(
                b,
                root + f"/End{j}",
                [x, sign * 0.36, 0.85],
                [0.015, 0.23, 0.075],
                "machine_white",
                0.006,
                True,
            )
        for j, dy in enumerate((-0.108, 0.108)):
            rod(
                b,
                root + f"/Rim{j}",
                [0.15, sign * 0.36 + dy, 0.872],
                [0.55, sign * 0.36 + dy, 0.872],
                0.007,
                "steel",
                True,
            )
            for k in range(14):
                x = 0.16 + k * 0.029
                rod(
                    b,
                    root + f"/Wire{j}_{k}",
                    [x, sign * 0.36 + dy, 0.83],
                    [x, sign * 0.36 + dy, 0.87],
                    0.0018,
                    "steel",
                )
        for j, x in enumerate((0.20, 0.50)):
            rod(
                b, root + f"/Leg{j}", [x, sign * 0.36, 0.04], [x, sign * 0.36, 0.79], 0.017, "steel"
            )
        # Low separators stop stock drifting but leave opposing fingers clear.
        for j, x in enumerate((0.29, 0.41)):
            solid(
                b,
                root + f"/Separator{j}",
                [x, sign * 0.36, 0.832],
                [0.005, 0.16, 0.012],
                "rubber",
                0.001,
                True,
            )
    for name, pos in c["sources"].items():
        root = "/World/Objects/" + name
        UsdGeom.Xform.Define(b.stage, root)
        b.collider(root + "/Body", "sphere", [0, 0, 0], [c["radius"]], [0.65, 0.8, 0.05])
        bind(b, root + "/Body", a.material("ball_felt"))
        ring(b, root + "/Seam", [0, 0, 0], c["radius"] + 0.0003, 0.0006, "cream", False, 48, "yz")
        rigid(b, name, pos, c["mass"])
    supported(b, "/World/Hopper")
    for i, (x, y, z) in enumerate(c["cells"]):
        h = c["cell_half_width"]
        for j, (dx, dy, sx, sy) in enumerate(
            (
                (-h - 0.004, 0, 0.008, 2 * h + 0.016),
                (h + 0.004, 0, 0.008, 2 * h + 0.016),
                (0, -h - 0.004, 2 * h, 0.008),
                (0, h + 0.004, 2 * h, 0.008),
            )
        ):
            solid(
                b,
                f"/World/Hopper/Cell{i}Wall{j}",
                [x + dx, y + dy, 0.889],
                [sx, sy, 0.122],
                "machine_white",
                0.003,
                True,
            )
        root = f"/World/Controls/gate{i}"
        UsdGeom.Xform.Define(b.stage, root)
        b.collider(root + "/Floor", "cube", [0, 0, 0], [2 * h, 2 * h, 0.010], [0.10, 0.28, 0.28])
        gate = b.world.scene.add(
            SingleRigidPrim(root, name=f"gate{i}", position=np.array([x, y, 0.821]), mass=0.15)
        )
        PhysxSchema.PhysxRigidBodyAPI.Apply(gate.prim).CreateSolverPositionIterationCountAttr(32)
        b.pads[f"gate{i}"] = gate
        joint = UsdPhysics.PrismaticJoint.Define(b.stage, root + "/Slide")
        joint.CreateBody1Rel().SetTargets([root])
        joint.CreateAxisAttr("Y")
        joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, 0.821))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLowerLimitAttr(-0.14)
        joint.CreateUpperLimitAttr(0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateStiffnessAttr(3000)
        drive.CreateDampingAttr(100)
        drive.CreateMaxForceAttr(200)
        drive.CreateTargetPositionAttr(0)
        b.gate_drives[i] = drive
    # Open equipment housing: no solid box hiding the actual physical outlet.
    for i, y in enumerate((-0.24, 0.24)):
        solid(
            b,
            f"/World/Machine/Side{i}Lower",
            [0.52, y, 0.28],
            [0.40, 0.035, 0.25],
            "teal_lacquer",
            0.018,
        )
        solid(
            b,
            f"/World/Machine/Side{i}Upper",
            [0.52, y, 0.77],
            [0.40, 0.035, 0.09],
            "teal_lacquer",
            0.014,
        )
        for j, x in enumerate((0.33, 0.71)):
            solid(
                b,
                f"/World/Machine/Side{i}Pillar{j}",
                [x, y, 0.56],
                [0.025, 0.035, 0.36],
                "teal_lacquer",
                0.008,
            )
        rod(
            b,
            f"/World/Machine/Wheel{i}",
            [0.69, y - 0.025, 0.13],
            [0.69, y + 0.025, 0.13],
            0.085,
            "rubber",
        )
        rod(
            b,
            f"/World/Machine/Hub{i}",
            [0.69, y - 0.028, 0.13],
            [0.69, y + 0.028, 0.13],
            0.035,
            "steel",
        )
    solid(b, "/World/Machine/Chassis", [0.52, 0, 0.17], [0.41, 0.46, 0.045], "ink", 0.008)
    rod(b, "/World/Machine/Axle", [0.69, -0.25, 0.13], [0.69, 0.25, 0.13], 0.012, "steel")
    for i, y in enumerate((-0.18, 0.18)):
        rod(b, f"/World/Machine/Foot{i}", [0.37, y, 0.02], [0.37, y, 0.18], 0.016, "steel")
        solid(
            b, f"/World/Machine/FootPad{i}", [0.37, y, 0.018], [0.07, 0.065, 0.025], "rubber", 0.008
        )
    solid(b, "/World/Machine/Header", [0.76, 0, 0.80], [0.045, 0.50, 0.13], "machine_white", 0.018)
    solid(b, "/World/Machine/Handle", [0.81, 0, 0.97], [0.025, 0.40, 0.025], "rubber", 0.009)
    for i, y in enumerate((-0.18, 0.18)):
        rod(b, f"/World/Machine/HandlePost{i}", [0.77, y, 0.82], [0.81, y, 0.97], 0.009, "steel")
    for i in range(8):
        solid(
            b,
            f"/World/Machine/Vent{i}",
            [0.43 + i * 0.035, -0.26, 0.28],
            [0.018, 0.004, 0.10],
            "ink",
            0.003,
        )
    # Real collector below the magazine: discharged balls remain physical.
    supported(b, "/World/Outlet")
    solid(b, "/World/Outlet/Floor", [0.51, 0, 0.47], [0.39, 0.43, 0.025], "rubber", 0.008, True)
    for i, y in enumerate((-0.215, 0.215)):
        solid(
            b,
            f"/World/Outlet/Side{i}",
            [0.51, y, 0.52],
            [0.39, 0.012, 0.10],
            "machine_white",
            0.005,
            True,
        )
    for i, x in enumerate((0.315, 0.705)):
        solid(
            b,
            f"/World/Outlet/End{i}",
            [x, 0, 0.52],
            [0.012, 0.43, 0.10],
            "machine_white",
            0.005,
            True,
        )
    for root in ("/World/Machine", "/World/Outlet"):
        UsdGeom.Xformable(b.stage.GetPrimAtPath(root)).AddTranslateOp().Set(Gf.Vec3d(-0.08, 0, 0))
    display(b, [0.58, 0.62, 1.48], 0.87)
    for i, x in enumerate((0.20, 0.95)):
        rod(b, f"/World/Display/Stand{i}", [x, 0.66, 0.02], [x, 0.66, 1.38], 0.016, "steel")
    b.support_paths = [
        "/World/SupplyLeft",
        "/World/SupplyRight",
        "/World/Hopper",
        "/World/Outlet",
        "/World/Table",
    ] + [p.prim_path for p in b.pads.values()]


def build_scene(b):
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.sensors.camera import Camera

    from roboclock.simulators.isaac.embodiments.views import look_pose

    b.world.scene.add(
        FixedCuboid(
            "/World/Table",
            name="table",
            position=np.array([-1, -1, -0.2]),
            scale=np.array([0.1, 0.1, 0.1]),
        )
    )
    room(b)
    dressing(b)
    equipment(b)
    profile = b.config["visuals"]["presentation_camera"]
    b.camera_position, b.camera_orientation = look_pose(
        profile["eye"], profile["target"], [0, 0, 1]
    )
    b.camera = Camera(
        "/World/Camera",
        position=b.camera_position,
        orientation=b.camera_orientation,
        resolution=tuple(b.config["sim"]["camera_resolution"]),
    )
