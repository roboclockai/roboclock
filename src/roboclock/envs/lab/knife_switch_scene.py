"""Passive hinged disconnect using the bundled CC-BY-4.0 Sam Feng asset."""

import numpy as np

from roboclock.resources import resolve_asset

GRIP_RADIUS = 0.1478361236913488
PIVOT_XY = (0.30, 0.12)
PIVOT_HEIGHT = 0.075
OPEN_ANGLE = np.deg2rad(35)


def handle_geometry(appearance, path):
    from pxr import Gf, UsdGeom, UsdPhysics

    visual = UsdGeom.Xform.Define(appearance.stage, path + "/Imported")
    visual.GetPrim().GetReferences().AddReference(
        str(resolve_asset("lab://knife_switch/handle.usdc"))
    )
    # Upper insulated handle: robot pads contact this capsule-like box.
    for name, pos, size in [
        ("Grip", [0, 0, 0], [0.061, 0.020, 0.021]),
        ("BladeLeft", [-0.095, -0.041, 0], [0.105, 0.009, 0.012]),
        ("BladeRight", [-0.095, 0.041, 0], [0.105, 0.009, 0.012]),
    ]:
        shape = UsdGeom.Cube.Define(appearance.stage, path + "/" + name)
        shape.CreateSizeAttr(1)
        shape.AddTranslateOp().Set(Gf.Vec3d(*pos))
        shape.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        UsdGeom.Imageable(shape).MakeInvisible()


def build_disconnect(scene, source):
    """Disconnect around the handle at nominal `source` (see ReactiveEquipmentScene)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    pivot = np.asarray(source) - [GRIP_RADIUS, 0, 0]
    scene.pivot_switch = pivot
    scene.cube("DisconnectStand", [*pivot[:2], scene.z + 0.025], [0.24, 0.16, 0.05], "rubber", True)
    panel = UsdGeom.Xform.Define(scene.stage, "/World/FaultRig/DisconnectPanel")
    panel.AddTranslateOp().Set(Gf.Vec3d(*map(float, pivot)))
    panel.GetPrim().GetReferences().AddReference(
        str(resolve_asset("lab://knife_switch/panel.usdc"))
    )
    joint = UsdPhysics.RevoluteJoint.Define(scene.stage, "/World/FaultRig/DisconnectHinge")
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Objects/sample_0")])
    joint.CreateAxisAttr("Y")
    # `source` is nominal (prims here sit under /World/FaultRig, which carries the seed's
    # apparatus placement); the world-attached hinge frame needs the placed pivot.
    frame = getattr(scene, "frame", None)
    world_pivot = pivot if frame is None else frame.point(pivot)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_pivot)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-GRIP_RADIUS, 0, 0))
    joint.CreateLowerLimitAttr(-72.0)
    joint.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.025)
    drive.CreateTargetVelocityAttr(0.0)
    scene.switch_drive = drive
    # Two poles interrupt both supply and return. The four sockets are distinct.
    scene.switch_terminals = {}
    brass = scene.a.material("TerminalBrass", color=[0.58, 0.34, 0.09], metallic=0.9, roughness=0.3)
    for label, x in [("IN", pivot[0] - 0.08), ("OUT", pivot[0] + 0.09)]:
        for i, dy in enumerate([-0.05, 0.05]):
            key = label + ("+" if i == 0 else "-")
            point = np.array([x, pivot[1] + dy, scene.z + 0.061])
            scene.switch_terminals[key] = point
            scene.cylinder(
                "Socket_" + label + str(i),
                point - [0, 0, 0.007],
                0.008,
                0.014,
                scene.red if i == 0 else "rubber",
            )
            scene.cylinder("Screw_" + label + str(i), point, 0.004, 0.003, brass)
            scene.cube(
                "Busbar_" + label + str(i),
                [x + (0.035 if label == "IN" else -0.005), point[1], point[2] - 0.004],
                [0.065 if label == "IN" else 0.024, 0.009, 0.004],
                brass,
            )
    scene.a.label(
        "/World/FaultRig/DisconnectLabel",
        "PULL TO ISOLATE",
        [pivot[0], pivot[1] + 0.081, scene.z + 0.026],
        [0.22, 0.032],
    )
