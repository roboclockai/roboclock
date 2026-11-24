"""Physical tools, carrier, divided hopper and bright mobile sports settings."""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind, pbr
from roboclock.envs.sports.arena.geometry import ring

from .geometry import rod
from .scenes import display, rigid, solid


def supported(b, path):
    from pxr import UsdGeom, UsdPhysics

    prim = UsdGeom.Xform.Define(b.stage, path)
    UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim()).CreateKinematicEnabledAttr(True)


def freeze(b):
    from pxr import UsdGeom

    a = b.appearance
    solid(b, "/World/Course/Surface", [1.7, 0, 0.003], [4.8, 1.24, 0.004], "court_green")
    for i, y in enumerate([-0.64, 0.64]):
        solid(b, f"/World/Course/Line{i}", [1.7, y, 0.007], [4.8, 0.018, 0.003], "cream", 0.001)
        solid(
            b,
            f"/World/Course/Barrier{i}",
            [1.7, y, 0.075],
            [4.8, 0.04, 0.13],
            "clean_wood",
            0.01,
            True,
        )
    b.goal_x = 3.4
    for i, y in enumerate([-0.34, 0.34]):
        rod(b, f"/World/Goal/Post{i}", [3.4, y, 0.01], [3.4, y, 0.95], 0.018, "cream", True)
        rod(b, f"/World/Goal/Side{i}", [3.4, y, 0.95], [3.72, y, 0.95], 0.014, "cream", True)
        rod(b, f"/World/Goal/Back{i}", [3.72, y, 0.01], [3.72, y, 0.95], 0.014, "cream", True)
    rod(b, "/World/Goal/Top", [3.4, -0.34, 0.95], [3.4, 0.34, 0.95], 0.018, "cream", True)
    for i in range(12):
        y = -0.33 + i * 0.06
        rod(b, f"/World/Goal/NetV{i}", [3.72, y, 0.02], [3.72, y, 0.93], 0.0012, "cream")
        z = 0.02 + i * 0.06
        rod(b, f"/World/Goal/NetH{i}", [3.72, -0.34, z], [3.72, 0.34, z], 0.0012, "cream")
    for side, y in enumerate([-0.34, 0.34]):
        for i in range(6):
            x = 3.42 + i * 0.06
            rod(b, f"/World/Goal/SideNet{side}_{i}", [x, y, 0.02], [x, y, 0.93], 0.001, "cream")
    b.stick_home = np.array([0.83, 0, 0.023])
    b.stick_grip = 0.55
    root = "/World/Objects/stick"
    UsdGeom.Xform.Define(b.stage, root)
    rod(b, root + "/Shaft", [0, 0, 0.025], [0, 0, 0.575], 0.011, "clean_wood", True)
    solid(b, root + "/Grip", [0, 0, 0.5475], [0.028, 0.028, 0.055], "rubber", 0.004, True)
    solid(b, root + "/Blade", [0.06, 0, 0], [0.15, 0.11, 0.035], "teal_lacquer", 0.008, True)
    rigid(b, "stick", b.stick_home, 0.12)
    # Broad blade supports the upright stick; no fixture obstructs the mobile lane.
    supported(b, "/World/StickRack")
    solid(
        b,
        "/World/StickRack/SpareHolder",
        [-0.7, 0.55, 0.06],
        [0.16, 0.10, 0.12],
        "brass",
        0.01,
        True,
    )
    root = "/World/Objects/puck"
    UsdGeom.Xform.Define(b.stage, root)
    b.collider(root + "/Body", "cylinder", [0, 0, 0], [0.04, 0.032], [0.7, 0.2, 0.02])
    bind(b, root + "/Body", a.material("plate_red"))
    ring(b, root + "/Edge", [0, 0, 0.012], 0.037, 0.002, "cream")
    rigid(b, "puck", [1.015, 0, 0.020], 0.09)
    b.support_paths = [
        "/World/defaultGroundPlane/GroundPlane/CollisionPlane",
        "/World/StickRack",
        "/World/Table",
    ]
    display(b, [2.55, 0.92, 1.0], 1.35)
    rod(b, "/World/Signal/Post", [2.55, 0.92, 0], [2.55, 0.92, 0.84], 0.025, "steel")
    solid(b, "/World/Signal/Housing", [3.48, 0.8, 1.03], [0.20, 0.12, 0.53], "ink", 0.035)
    rod(b, "/World/Signal/Stand", [3.48, 0.8, 0.02], [3.48, 0.8, 0.80], 0.02, "steel")
    for name, z in [("Red", 1.17), ("Green", 0.9)]:
        b.collider(
            "/World/Signal/" + name, "sphere", [3.48, 0.717, z], [0.065], [0.10, 0.10, 0.10], False
        )
        color = [1.0, 0.03, 0.01] if name == "Red" else [0.02, 0.85, 0.15]
        pbr(a, "signal_" + name + "_on", color, 0.22, emission=300)
        pbr(a, "signal_" + name + "_off", [c * 0.1 for c in color], 0.28)
    for x in [1.0, 1.8, 2.6, 3.4]:
        solid(
            b,
            "/World/Course/Mark" + str(int(x * 10)),
            [x, 0, 0.009],
            [0.025, 1.20, 0.003],
            "cream",
            0.001,
        )
    b.view = ([3.3, -6.4, 3.5], [1.65, 0, 0.70])


def basket(b):
    from pxr import UsdGeom

    root = "/World/Objects/carrier"
    UsdGeom.Xform.Define(b.stage, root)
    solid(b, root + "/Floor", [0, 0, 0], [0.22, 0.26, 0.016], "teal_lacquer", 0.006, True)
    for i, (center, size) in enumerate(
        [
            ([-0.11, 0, 0.0225], [0.012, 0.26, 0.045]),
            ([0.11, 0, 0.0225], [0.012, 0.26, 0.045]),
            ([0, -0.13, 0.0225], [0.22, 0.012, 0.045]),
            ([0, 0.13, 0.0225], [0.22, 0.012, 0.045]),
        ]
    ):
        solid(b, root + f"/Wall{i}", center, size, "teal_lacquer", 0.005, True)
    for x in [-0.085, 0.085]:
        rod(
            b,
            root + f"/HandlePost{int(x * 100) + 100}",
            [x, 0.13, 0.045],
            [x, 0.17, 0.19],
            0.007,
            "steel",
            True,
        )
    rod(b, root + "/Handle", [-0.085, 0.17, 0.19], [0.085, 0.17, 0.19], 0.012, "rubber", True)
    solid(b, root + "/GripBlock", [0, 0.17, 0.19], [0.08, 0.04, 0.05], "rubber", 0.005, True)
    for i in range(7):
        solid(
            b,
            root + f"/Vent{i}",
            [-0.085 + i * 0.028, -0.136, 0.026],
            [0.012, 0.001, 0.023],
            "ink",
            0.003,
        )
    # Shallow dividers locate the load without enclosing the gripper's fingers.
    for slot, x in enumerate([-0.045, 0.045]):
        for j, (dx, dy, sx, sy) in enumerate(
            [
                (-0.037, 0, 0.004, 0.078),
                (0.037, 0, 0.004, 0.078),
                (0, -0.037, 0.078, 0.004),
                (0, 0.037, 0.078, 0.004),
            ]
        ):
            solid(
                b,
                root + f"/Locator{slot}_{j}",
                [x + dx, -0.03 + dy, 0.020],
                [sx, sy, 0.024],
                "rubber",
                0.001,
                True,
            )
    rigid(b, "carrier", [0.83, 0.23, 0.648], 0.13)
    b.carrier_grip = 0.19
    b.carrier_handle = np.array([0, 0.17, 0.19])


def service(b):
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    a = b.appearance
    pbr(a, "tennis_yellow", [0.63, 0.77, 0.10], roughness=0.95)
    # Wall-mounted training rackets with real frame thickness and crossed strings.
    for racket, x in enumerate([0.25, 0.65]):
        center = np.array([x, 1.68, 1.45])
        for i in range(48):
            angles = np.array([i, i + 1]) * 2 * np.pi / 48
            points = [center + np.array([0.13 * np.cos(q), 0, 0.18 * np.sin(q)]) for q in angles]
            rod(
                b,
                f"/World/Tennis/Racket{racket}/Frame{i}",
                *points,
                0.009,
                "plate_blue" if racket else "plate_red",
            )
        for i, v in enumerate(np.linspace(-0.85, 0.85, 11)):
            half = np.sqrt(1 - v * v)
            rod(
                b,
                f"/World/Tennis/Racket{racket}/StringH{i}",
                center + [-0.13 * half, 0, 0.18 * v],
                center + [0.13 * half, 0, 0.18 * v],
                0.0007,
                "cream",
            )
            rod(
                b,
                f"/World/Tennis/Racket{racket}/StringV{i}",
                center + [0.13 * v, 0, -0.18 * half],
                center + [0.13 * v, 0, 0.18 * half],
                0.0007,
                "cream",
            )
        rod(
            b,
            f"/World/Tennis/Racket{racket}/Stem",
            center + [0, 0, -0.18],
            center + [0, 0, -0.32],
            0.008,
            "steel",
        )
        rod(
            b,
            f"/World/Tennis/Racket{racket}/Grip",
            center + [0, 0, -0.32],
            center + [0, 0, -0.48],
            0.014,
            "rubber",
        )
    solid(b, "/World/Court/Surface", [1.7, 0, 0.003], [5.2, 2.2, 0.004], "court_green")
    for i, y in enumerate([-0.92, 0.92]):
        solid(b, f"/World/Court/Line{i}", [1.7, y, 0.007], [5.2, 0.023, 0.004], "cream", 0.001)
    b.supply_x = 0.83
    b.delivery_x = 2.83
    b.delivery_base = 2.15
    for station, x in [("Supply", 0.83), ("Service", 2.89)]:
        supported(b, "/World/" + station)
        solid(
            b,
            "/World/" + station + "/Top",
            [x, 0.08, 0.625],
            [0.52, 0.74, 0.03],
            "clean_wood",
            0.012,
            True,
        )
        for i, (dx, y) in enumerate([(-0.2, -0.20), (-0.2, 0.35), (0.2, -0.20), (0.2, 0.35)]):
            rod(b, f"/World/{station}/Leg{i}", [x + dx, y, 0], [x + dx, y, 0.61], 0.024, "steel")
    basket(b)
    b.ball_sources = {}
    rng = np.random.default_rng(b.config["seed"])
    for i in range(8):
        # Finger clearance between rows; source rack is separate from the carrier.
        x = 0.59 + (i % 4) * 0.115
        y = 0.55 + (i // 4) * 0.14
        if b.config["seed"]:
            jitter = rng.uniform(-0.004, 0.004, 2) * b.config["scene"].get("randomness_scale", 1.0)
            x += float(jitter[0])
            y += float(jitter[1])
        b.ball_sources[f"ball{i}"] = [x, y, 0.669]
        root = f"/World/Objects/ball{i}"
        UsdGeom.Xform.Define(b.stage, root)
        b.collider(root + "/Body", "sphere", [0, 0, 0], [0.027], [0.6, 0.8, 0.1])
        bind(b, root + "/Body", a.material("tennis_yellow"))
        ring(b, root + "/Seam", [0, 0, 0], 0.0272, 0.0006, "cream", False, 32, "yz")
        rigid(b, f"ball{i}", [x, y, 0.669], 0.035)
    # Extend the stock rack beneath its last row, with real supports.
    solid(
        b, "/World/Supply/Shelf", [0.76, 0.65, 0.625], [0.48, 0.24, 0.03], "clean_wood", 0.01, True
    )
    b.hopper_cells = []
    b.gate_drives = {}
    supported(b, "/World/Hopper")
    for i, (x, y) in enumerate([(2.77, 0.60), (2.86, 0.60), (2.95, 0.60), (3.04, 0.60)]):
        b.hopper_cells.append(np.array([x, y, 0.68]))
        for j, (cx, cy, sx, sy) in enumerate(
            [
                (x - 0.036, y, 0.008, 0.078),
                (x + 0.036, y, 0.008, 0.078),
                (x, y - 0.037, 0.078, 0.008),
                (x, y + 0.037, 0.078, 0.008),
            ]
        ):
            solid(
                b,
                f"/World/Hopper/Cell{i}Wall{j}",
                [cx, cy, 0.70],
                [sx, sy, 0.18],
                "cream",
                0.003,
                True,
            )
        name = f"gate{i}"
        path = "/World/Controls/" + name
        UsdGeom.Xform.Define(b.stage, path)
        b.collider(path + "/Floor", "cube", [0, 0, 0], [0.069, 0.069, 0.01], [0.06, 0.3, 0.3])
        gate = b.world.scene.add(
            SingleRigidPrim(path, name=name, position=np.array([x, y, 0.604]), mass=0.15)
        )
        gate_api = PhysxSchema.PhysxRigidBodyAPI.Apply(gate.prim)
        gate_api.CreateSolverPositionIterationCountAttr(32)
        gate_api.CreateSolverVelocityIterationCountAttr(8)
        b.pads[name] = gate
        joint = UsdPhysics.PrismaticJoint.Define(b.stage, path + "/Slide")
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateAxisAttr("Y")
        joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, 0.604))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLowerLimitAttr(-0.12)
        joint.CreateUpperLimitAttr(0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateStiffnessAttr(3000)
        drive.CreateDampingAttr(100)
        drive.CreateMaxForceAttr(200)
        drive.CreateTargetPositionAttr(0)
        b.gate_drives[i] = drive
    # Service table does not cover the hopper outlet; cells are on the far edge.
    # Replace its wide collision box by a carrier-only surface.
    collider = b.stage.GetPrimAtPath("/World/Service/TopCollider")
    UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr(False)
    UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/Service/Top")).MakeInvisible()
    solid(
        b,
        "/World/Service/CarrierSupport",
        [2.83, 0.23, 0.625],
        [0.40, 0.36, 0.03],
        "clean_wood",
        0.008,
        True,
    )
    for i, x in enumerate([2.65, 3.17]):
        solid(b, f"/World/Machine/Side{i}", [x, 0.67, 0.34], [0.045, 0.36, 0.48], "ink", 0.015)
    solid(b, "/World/Machine/Back", [2.91, 0.85, 0.34], [0.52, 0.04, 0.48], "ink", 0.015)
    solid(b, "/World/Machine/Header", [2.91, 0.56, 0.55], [0.50, 0.04, 0.06], "teal_lacquer", 0.01)
    supported(b, "/World/Outlet")
    solid(
        b,
        "/World/Outlet/Collection",
        [2.91, 0.60, 0.20],
        [0.46, 0.16, 0.025],
        "rubber",
        0.005,
        True,
    )
    for i, y in enumerate([0.515, 0.685]):
        solid(
            b, f"/World/Outlet/Side{i}", [2.91, y, 0.24], [0.46, 0.012, 0.08], "cream", 0.004, True
        )
    for i, x in enumerate([2.665, 3.15]):
        solid(
            b, f"/World/Outlet/End{i}", [x, 0.60, 0.24], [0.015, 0.18, 0.08], "cream", 0.004, True
        )
    for i in range(8):
        solid(
            b,
            f"/World/Machine/Vent{i}",
            [2.735, 0.58, 0.32 + i * 0.022],
            [0.003, 0.16, 0.007],
            "steel",
            0.002,
        )
    display(b, [2.40, 1.0, 1.18], 1.35)
    b.support_paths = [
        "/World/Supply",
        "/World/Service",
        "/World/Hopper",
        "/World/Outlet",
        "/World/Table",
    ] + [p.prim_path for p in b.pads.values()]
    b.view = ([3.3, -6.0, 3.6], [1.5, 0.20, 0.80])


def build(b):
    if b.kind == "B4":
        freeze(b)
    else:
        service(b)
