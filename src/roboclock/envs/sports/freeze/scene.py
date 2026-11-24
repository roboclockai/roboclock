"""Bright game lounge, detailed table and independently graspable pushers.

Native X5 CAD, HDR, scanned wood and existing room assets are reused. The
specialized table and pushers are authored, explicitly documented proxies.
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
    for path in ("/World/Room/BackWall", "/World/Room/SideWall", "/World/Studio/Brand"):
        prim = b.stage.GetPrimAtPath(path)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()
    pbr(a, "lounge_cream", [0.78, 0.76, 0.68], 0.80)
    pbr(a, "lounge_blue", [0.045, 0.16, 0.23], 0.60)
    pbr(a, "rink_white", [0.79, 0.85, 0.82], 0.34)
    pbr(a, "pusher_coral", [0.76, 0.16, 0.065], 0.32)
    pbr(a, "pusher_teal", [0.02, 0.40, 0.42], 0.32)
    solid(b, "/World/Lounge/Wall", [0.4, 1.85, 1.6], [5.4, 0.08, 3.2], "lounge_cream")
    solid(b, "/World/Lounge/Side", [-2.35, 0, 1.6], [0.08, 5, 3.2], "lounge_cream")
    solid(b, "/World/Lounge/Panel", [0.4, 1.79, 0.59], [5.4, 0.04, 1.18], "lounge_blue")
    solid(b, "/World/Lounge/Rail", [0.4, 1.75, 1.20], [5.4, 0.04, 0.035], "brass")
    for j in range(10):
        solid(
            b,
            f"/World/Lounge/Slat{j}",
            [-1.55 + j * 0.08, 1.73, 1.9],
            [0.03, 0.05, 1.2],
            "clean_wood",
            0.004,
        )
    panel(
        b,
        a,
        "/World/Lounge/Identity",
        [0.35, 1.72, 2.05],
        [1.55, 0.42],
        "THE PLAY CLUB",
        "TABLETOP / PRECISION GAMES",
        style="title",
        accent=(211, 182, 117),
    )
    # Framed sport graphic, full benches, and a storage cabinet ground the room.
    panel(
        b,
        a,
        "/World/Lounge/Poster",
        [1.65, 1.72, 1.73],
        [0.60, 0.75],
        "PLAY",
        "MOVE / STOP / CONTROL",
        style="title",
        accent=(98, 181, 175),
    )
    solid(b, "/World/Lounge/Cabinet", [-1.35, 0.95, 0.4], [0.70, 0.42, 0.70], "clean_wood", 0.02)
    for j in range(3):
        solid(
            b,
            f"/World/Lounge/Drawer{j}",
            [-1.35, 0.733, 0.20 + j * 0.22],
            [0.64, 0.018, 0.19],
            "lounge_blue",
            0.008,
        )
        rod(
            b,
            f"/World/Lounge/Pull{j}",
            [-1.45, 0.705, 0.20 + j * 0.22],
            [-1.25, 0.705, 0.20 + j * 0.22],
            0.008,
            "brass",
        )


def equipment(b):
    from isaacsim.core.api.materials import PhysicsMaterial
    from pxr import PhysxSchema, UsdGeom, UsdShade

    c = b.config["freeze"]
    supported(b, "/World/Rink")
    solid(b, "/World/Rink/Cabinet", [0.44, 0, 0.735], [0.64, 1.12, 0.18], "clean_wood", 0.025)
    solid(b, "/World/Rink/Trim", [0.44, 0, 0.814], [0.66, 1.14, 0.022], "steel", 0.009)
    solid(b, "/World/Rink/Bed", [0.44, 0, 0.825], [0.60, 1.08, 0.030], "rink_white", 0.008, True)
    for j, x in enumerate((0.135, 0.745)):
        solid(
            b,
            f"/World/Rink/Side{j}",
            [x, 0, 0.861],
            [0.028, 1.10, 0.042],
            "lounge_blue",
            0.008,
            True,
        )
        solid(b, f"/World/Rink/SideCap{j}", [x, 0, 0.884], [0.034, 1.10, 0.008], "steel", 0.003)
    for j, y in enumerate((-0.55, 0.55)):
        solid(
            b,
            f"/World/Rink/End{j}",
            [0.44, y, 0.861],
            [0.64, 0.028, 0.042],
            "lounge_blue",
            0.008,
            True,
        )
    for j, (x, y) in enumerate((x, y) for x in (0.21, 0.67) for y in (-0.43, 0.43)):
        rod(b, f"/World/Rink/Leg{j}", [x, y, 0.04], [x, y, 0.66], 0.026, "steel")
        solid(b, f"/World/Rink/Foot{j}", [x, y, 0.025], [0.085, 0.085, 0.035], "rubber", 0.01)
    x = c["lane_x"]
    for j, yy in enumerate((-0.27, 0, 0.27)):
        ring(
            b,
            f"/World/Rink/Ring{j}",
            [x, yy, 0.8406],
            0.069,
            0.0018,
            "pusher_teal" if j else "pusher_coral",
            False,
            80,
        )
    solid(b, "/World/Rink/Midline", [0.44, 0, 0.8405], [0.56, 0.002, 0.0006], "lounge_blue", 0.0001)
    for j, xx in enumerate((x - 0.085, x + 0.085)):
        for k in range(20):
            solid(
                b,
                f"/World/Rink/Dash{j}_{k}",
                [xx, -0.37 + k * 0.039, 0.8405],
                [0.0018, 0.018, 0.0006],
                "pusher_teal",
                0.0001,
            )
    for j, g in enumerate(c["goals"]):
        # Goal boundary marks whole-puck containment, not a decorative hole.
        hx, hy = c["goal_half_size"]
        for k, (dx, dy, sx, sy) in enumerate(
            (
                (-hx, 0, 0.002, 2 * hy),
                (hx, 0, 0.002, 2 * hy),
                (0, -hy, 2 * hx, 0.002),
                (0, hy, 2 * hx, 0.002),
            )
        ):
            solid(
                b,
                f"/World/Rink/Goal{j}_{k}",
                [g[0] + dx, g[1] + dy, 0.8407],
                [sx, sy, 0.0007],
                "pusher_coral" if j == 0 else "pusher_teal",
                0.0001,
            )
    low_friction = PhysicsMaterial(
        "/World/Materials/Slide", static_friction=0.22, dynamic_friction=0.18, restitution=0.03
    )
    bed = b.stage.GetPrimAtPath("/World/Rink/BedCollider")
    UsdShade.MaterialBindingAPI.Apply(bed).Bind(low_friction.material, materialPurpose="physics")
    root = "/World/Objects/puck"
    UsdGeom.Xform.Define(b.stage, root)
    rod(b, root + "/Disk", [0, 0, -0.009], [0, 0, 0.009], c["puck_radius"], "ink", True)
    ring(b, root + "/TopStripe", [0, 0, 0.0095], 0.019, 0.0014, "brass", False, 64)
    rigid(b, "puck", c["puck_start"], 0.045)
    for prim in (b.stage.GetPrimAtPath(root + "/Disk"),):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            low_friction.material, materialPurpose="physics"
        )
    for side, pos in c["tool_home"].items():
        root = "/World/Objects/" + side + "_pusher"
        UsdGeom.Xform.Define(b.stage, root)
        mat = "pusher_teal" if side == "left" else "pusher_coral"
        rod(b, root + "/Foot", [0, 0, -0.007], [0, 0, 0.007], c["tool_radius"], mat, True)
        rod(b, root + "/Stem", [0, 0, 0.007], [0, 0, 0.065], 0.012, "steel", True)
        solid(b, root + "/Grip", [0, 0, 0.105], [0.028, 0.036, 0.080], "rubber", 0.004, True)
        # Mechanical sleeve end caps are real colliders, not grasp constraints.
        # They fit through the open jaws and resist axial slip after closure.
        for j, z in enumerate((0.060, 0.150)):
            solid(b, root + f"/GripCap{j}", [0, 0, z], [0.048, 0.052, 0.008], mat, 0.003, True)
        for j in range(6):
            ring(
                b, root + f"/GripBand{j}", [0, 0, 0.074 + j * 0.012], 0.0185, 0.0007, mat, False, 40
            )
        obj = rigid(b, side + "_pusher", pos, c["tool_mass"])
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(obj.prim)
        api.CreateSolverPositionIterationCountAttr(64)
        api.CreateSolverVelocityIterationCountAttr(32)
    # Status board; head_view.DISPLAYS turns it toward the head camera at reset.
    display(b, [0.52, 0.74, 1.42], 0.94)
    rod(b, "/World/Display/Post", [0.52, 0.78, 0.03], [0.52, 0.78, 1.32], 0.021, "steel")
    solid(b, "/World/Display/Base", [0.52, 0.78, 0.025], [0.30, 0.24, 0.035], "lounge_blue", 0.012)
    # Stop/go signal beyond the far rink end, facing the rig's head camera (and,
    # through the rear lenses, the presentation view). Visual only, no collider.
    # It stood behind the rig at (-0.06, 0.70) until 2026-11-03 and then 0.17 m
    # higher; this height keeps the whole housing inside the RoboDojo 60 deg head view.
    sx, sz = 0.92, 1.045
    rod(b, "/World/Signal/Post", [sx, 0, 0.03], [sx, 0, sz - 0.145], 0.021, "steel")
    solid(b, "/World/Signal/Base", [sx, 0, 0.025], [0.24, 0.24, 0.035], "lounge_blue", 0.012)
    solid(b, "/World/Signal/Housing", [sx, 0, sz], [0.075, 0.12, 0.29], "ink", 0.02)
    for name, z, color in (("Red", sz + 0.07, [0.90, 0.035, 0.02]), ("Green", sz - 0.06, [0.03, 0.80, 0.23])):
        for on in (True, False):
            pbr(
                b.appearance,
                "signal_" + name + ("_on" if on else "_off"),
                np.array(color) * (1 if on else 0.08),
                0.30,
            )
        rod(
            b,
            "/World/Signal/" + name,
            [sx - 0.0435, 0, z],
            [sx + 0.0435, 0, z],
            0.037,
            "signal_" + name + "_off",
        )
    b.support_paths = ["/World/Rink", "/World/Table"]


def update_cues(b, task):
    from roboclock.envs.sports._shared.workflow.scenes import update_cues as base

    # The bevelled lens is a separate visual mesh from its cylinder proxy.
    red = task.red() is not None
    for name, active in (("Red", red), ("Green", not red)):
        bind(
            b,
            "/World/Signal/" + name + "Visual",
            b.appearance.material("signal_" + name + ("_on" if active else "_off")),
        )
    base(b, task)


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
    p = b.config["visuals"]["presentation_camera"]
    b.camera_position, b.camera_orientation = look_pose(p["eye"], p["target"], [0, 0, 1])
    b.camera = Camera(
        "/World/Camera",
        position=b.camera_position,
        orientation=b.camera_orientation,
        resolution=tuple(b.config["sim"]["camera_resolution"]),
    )
