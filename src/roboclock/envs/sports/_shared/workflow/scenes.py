"""Reused V5 materials and furniture, with explicitly authored sports equipment."""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind, panel, pbr
from roboclock.envs.sports.arcade.scenes import vertical_label

from .geometry import rod


def solid(b, path, center, size, material, radius=0.008, collision=False):
    b.appearance.rounded(path, center, size, material, radius)
    if collision:
        from pxr import UsdGeom

        shape, _ = b.collider(path + "Collider", "cube", center, size, [1, 1, 1])
        UsdGeom.Imageable(shape).MakeInvisible()


def rigid(b, name, position, mass):
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import PhysxSchema

    obj = b.world.scene.add(
        SingleRigidPrim("/World/Objects/" + name, name=name, position=np.array(position), mass=mass)
    )
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(obj.prim)
    api.CreateEnableCCDAttr(True)
    api.CreateSolverPositionIterationCountAttr(16)
    api.CreateSolverVelocityIterationCountAttr(8)
    b.objects[name] = obj
    return obj


def room(b):
    from pxr import Gf, UsdGeom, UsdLux

    from roboclock.envs.sports.arcade.classic import build_room_classic
    from roboclock.resources import resolve_asset

    a = b.appearance = build_room_classic(b, "arcade")
    # Reuse the room's material library and architecture, not its arcade machines.
    for p in list(b.stage.Traverse()):
        path = str(p.GetPath())
        if any(
            path.startswith("/World/" + x)
            for x in (
                "Room/Cabinet",
                "Room/Screen",
                "Room/GameTitle",
                "Room/Marquee",
                "Room/ControlShelf",
                "Room/Edge",
                "Room/Worktop",
                "Room/Sign",
                "Refined/Pedestal",
                "Refined/TableEdge",
                "Refined/Vent",
                "Refined/Arcade",
                "Room/Chair",
            )
        ):
            UsdGeom.Imageable(p).MakeInvisible()
    for path in ["/World/defaultGroundPlane", "/World/Table"]:
        if b.stage.GetPrimAtPath(path):
            UsdGeom.Imageable(b.stage.GetPrimAtPath(path)).MakeInvisible()
    pbr(a, "studio_wall", [0.43, 0.49, 0.48], roughness=0.84)
    bind(b, "/World/Room/BackWall", a.material("studio_wall"))
    bind(b, "/World/Room/SideWall", a.material("studio_wall"))
    pbr(a, "rubber", [0.075, 0.085, 0.09], roughness=0.85)
    pbr(a, "plate_red", [0.56, 0.065, 0.04], roughness=0.38)
    pbr(a, "plate_blue", [0.025, 0.21, 0.45], roughness=0.38)
    pbr(a, "court_green", [0.13, 0.32, 0.27], roughness=0.92)
    pbr(a, "pool_cloth", [0.04, 0.30, 0.25], roughness=0.96, texture="leather_white", uv=(5, 5))
    dome = UsdLux.DomeLight.Define(b.stage, "/World/Lights/Dome")
    dome.CreateTextureFileAttr(str(resolve_asset(b.config["visuals"]["hdr"])))
    dome.CreateIntensityAttr(800)
    for i, pos in enumerate(([0.8, -1.2, 2.8], [-1.2, -0.5, 2.8], [2.2, 0.2, 2.8])):
        light = UsdLux.RectLight.Define(b.stage, f"/World/Lights/Studio{i}")
        light.AddTranslateOp().Set(Gf.Vec3d(*pos))
        light.CreateWidthAttr(1.5)
        light.CreateHeightAttr(1.0)
        light.CreateIntensityAttr(900)
    # Properly proportioned benches and complete sourced chairs.
    for x in [-1.25, 1.55]:
        tag = "Left" if x < 0 else "Right"
        bench_y = 1.48 if b.kind == "C3" else 1.25
        solid(
            b, "/World/Dressing/Bench" + tag, [x, bench_y, 0.46], [0.85, 0.34, 0.07], "clean_wood"
        )
        for dx in [-0.32, 0.32]:
            rod(
                b,
                "/World/Dressing/Leg" + tag + str(int(dx * 100) + 100),
                [x + dx, bench_y, 0.04],
                [x + dx, bench_y, 0.44],
                0.022,
                "steel",
            )
    title = {
        "B3": "STRENGTH / CONTROL",
        "B4": "STOP / GO ARENA",
        "C1": "THE POOL ROOM",
        "C3": "TENNIS TRAINING CLUB",
    }[b.kind]
    panel(
        b,
        a,
        "/World/Studio/Brand",
        [0.3, 1.67, 2.10],
        [2.2, 0.45],
        title,
        "PRECISION IN MOTION",
        style="title",
        accent=(222, 177, 102),
    )


def display(b, center, width=0.85):
    b.display_center = np.array(center)
    b.display_width = width
    solid(b, "/World/Display/Case", center, [width + 0.06, 0.06, 0.29], "ink", 0.015)
    vertical_label(
        b,
        "/World/Display/Title",
        b.config["task_id"]
        + " / "
        + {"B3": "LIFT", "B4": "FREEZE", "C1": "POOL", "C3": "BALL SERVICE"}[b.kind],
        np.array(center) + [0, -0.033, 0.087],
        [width, 0.046],
    )


def lift(b):
    from pxr import UsdGeom, UsdPhysics

    solid(b, "/World/Equipment/Platform", [0.10, 0, 0.035], [1.65, 1.5, 0.07], "clean_wood")
    solid(b, "/World/Equipment/Mat", [0.36, 0, 0.079], [0.8, 1.28, 0.018], "rubber")
    solid(b, "/World/Equipment/RobotPlinth", [0, 0, 0.225], [0.34, 0.5, 0.45], "ink")
    b.bar_home = np.array([0.36, 0, 0.84])
    b.grip_y = 0.23
    # Cradle is one static rigid body containing two U-shaped supports.
    cradle = UsdGeom.Xform.Define(b.stage, "/World/Cradle")
    UsdPhysics.RigidBodyAPI.Apply(cradle.GetPrim()).CreateKinematicEnabledAttr(True)
    for i, y in enumerate([-0.35, 0.35]):
        rod(b, f"/World/Cradle/Post{i}", [0.36, y, 0.10], [0.36, y, 0.796], 0.02, "steel", True)
        solid(
            b, f"/World/Cradle/Foot{i}", [0.36, y, 0.12], [0.28, 0.12, 0.05], "ink", collision=True
        )
        solid(
            b,
            f"/World/Cradle/Seat{i}",
            [0.36, y, 0.817],
            [0.065, 0.06, 0.016],
            "rubber",
            collision=True,
        )
        for j, x in enumerate([0.326, 0.394]):
            solid(
                b,
                f"/World/Cradle/Lip{i}{j}",
                [x, y, 0.84],
                [0.012, 0.06, 0.055],
                "steel",
                collision=True,
            )
    path = "/World/Objects/bar"
    UsdGeom.Xform.Define(b.stage, path)
    rod(b, path + "/Shaft", [0, -0.45, 0], [0, 0.45, 0], 0.015, "steel", True)
    for i, y in enumerate([-0.23, 0.23]):
        rod(b, path + f"/Grip{i}", [0, y - 0.045, 0], [0, y + 0.045, 0], 0.0158, "rubber", True)
    for i, y in enumerate([-0.44, 0.44]):
        rod(b, path + f"/Plate{i}", [0, y - 0.022, 0], [0, y + 0.022, 0], 0.085, "plate_red", True)
        rod(b, path + f"/Collar{i}", [0, y - 0.035, 0], [0, y - 0.025, 0], 0.025, "brass", True)
    # Neutral-grip training handles: real welded frame and graspable crossbars.
    for i, y in enumerate([-0.23, 0.23]):
        for j, x in enumerate([-0.055, 0.055]):
            rod(b, path + f"/HandleSupport{i}{j}", [x, y, 0], [x, y, 0.078], 0.007, "steel", True)
            rod(b, path + f"/HandleFoot{i}{j}", [0, y, 0], [x, y, 0], 0.007, "steel", True)
        rod(
            b,
            path + f"/NeutralGrip{i}",
            [-0.055, y, 0.078],
            [0.055, y, 0.078],
            0.012,
            "rubber",
            True,
        )
        if b.config["workflow"].get("flat_lift_handles", False):
            solid(b, path + f"/GripFlat{i}", [0, y, 0.078], [0.085, 0.026, 0.026], "rubber", 0.003, True)
    rigid(b, "bar", b.bar_home + np.asarray(b.config["workflow"].get("initial_bar_offset", [0, 0, 0])), 0.28)
    if b.config["workflow"].get("flat_lift_handles", False):
        # Physical anti-roll saddles support the neutral-handle frame after release.
        for i, y in enumerate([-0.23, 0.23]):
            for j, x in enumerate([0.315, 0.405]):
                rod(b, f"/World/Cradle/FramePost{i}{j}", [x, y, 0.12], [x, y, 0.815], 0.011, "steel", True)
                solid(b, f"/World/Cradle/FrameSeat{i}{j}", [x, y, 0.824], [0.032, 0.04, 0.018], "rubber", 0.003, True)
    b.support_paths = ["/World/Cradle", "/World/Table"]
    # Distinct plate storage and rack hardware give the studio a recognizable setting.
    for i in range(3):
        y = 0.96
        rod(
            b,
            f"/World/Storage/Peg{i}",
            [-0.95 + i * 0.22, y, 0.6],
            [-0.95 + i * 0.22, y - 0.18, 0.6],
            0.014,
            "steel",
        )
        rod(
            b,
            f"/World/Storage/Plate{i}",
            [-0.95 + i * 0.22, y - 0.10, 0.6],
            [-0.95 + i * 0.22, y - 0.14, 0.6],
            0.13,
            ["plate_blue", "plate_red", "rubber"][i],
        )
    solid(b, "/World/Storage/Rack", [-0.73, 1.03, 0.50], [0.80, 0.12, 0.12], "ink")
    display(b, [0.92, 0.67, 1.29], 0.86)
    b.view = ([2.0, -3.2, 2.1], [0.30, 0, 1.02])


def build_scene(b):
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from isaacsim.sensors.camera import Camera
    from pxr import Gf

    datum = [0, 0, 0.10]
    b.world.scene.add(
        FixedCuboid(
            "/World/Table", name="table", position=np.array(datum), scale=np.array([0.3, 0.4, 0.2])
        )
    )
    room(b)
    if b.kind == "B3":
        if b.config["workflow"].get("bench_press"):
            from roboclock.envs.sports.bench_press.scene import build
            build(b)
        else:
            lift(b)
    elif b.kind == "C1":
        if b.config["workflow"].get("pool_racking"):
            from roboclock.envs.sports.pool_rack.scene import build
        elif b.config["workflow"].get("pool_mode") == "continuous_clearance":
            from roboclock.envs.sports._shared.precision.pool_scene import build
        else:
            from roboclock.envs.sports._shared.workflow.pool_scene import build

        build(b)
    else:
        from roboclock.envs.sports._shared.workflow.mobile_scenes import build

        build(b)
    cp, target = (np.array(x, float) for x in b.view)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*cp), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    cq = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
    b.camera_position, b.camera_orientation = cp, cq
    b.camera = Camera(
        "/World/Camera",
        position=cp,
        orientation=cq,
        resolution=tuple(
            int(n * b.config["visuals"].get("render_scale", 1))
            for n in b.config["sim"]["camera_resolution"]
        ),
    )


def update_cues(b, task):
    if b.kind == "B4":
        red = task.red() is not None
        if red != getattr(b, "signal_red", None):
            b.signal_red = red
            for name, active in [("Red", red), ("Green", not red)]:
                bind(
                    b,
                    "/World/Signal/" + name,
                    b.appearance.material("signal_" + name + ("_on" if active else "_off")),
                )
    label = task.display()
    if label == getattr(b, "cue_label", None):
        return
    b.cue_label = label
    for i, text in enumerate(label):
        vertical_label(
            b,
            f"/World/Display/Line{i}",
            text,
            b.display_center + [0, -0.034, 0.014 - i * 0.069],
            [b.display_width, 0.050],
        )
