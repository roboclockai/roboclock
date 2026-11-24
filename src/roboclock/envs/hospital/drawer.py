"""Author a sliding drawer: one rigid body on a prismatic joint to the world, with viscous damping.

Only USD physics schemas are used, so this builds in a plain USD stage as well as in Isaac Sim.
The fridge cabinet stays fixed; the drawer slides along +y from its closed pose, up to `travel`.
"""

import numpy as np


def author_drawer(stage, appearance, drawer, collision_settings=None, damping=25.0):
    """Create the drawer body, its colliders and visuals, and the joint. Returns the body path."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    root = f"/World/Drawers/{drawer.name}"
    xform = UsdGeom.Xform.Define(stage, root)
    xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in drawer.position]))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(float(drawer.mass))
    for part in drawer.parts:
        offset = [float(v) for v in part.offset]
        if part.collision:
            cube = UsdGeom.Cube.Define(stage, f"{root}/{part.name}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*offset))
            cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in part.size]))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            if collision_settings is not None:
                collision_settings(cube.GetPrim())
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        appearance.rounded(f"{root}/{part.name}_Visual", offset, part.size, part.material, 0.002)
    joint = UsdPhysics.PrismaticJoint.Define(stage, f"/World/Joints/{drawer.name}")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(root)])  # body0 left unset: the world
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(float(drawer.travel))
    joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in drawer.position]))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)  # no spring: it stays where it is left
    drive.CreateDampingAttr(float(damping))  # viscous drag, so it does not slam or drift
    drive.CreateMaxForceAttr(1.0e4)
    return root


def drawer_collider_boxes(drawer):
    """(center, size) of each colliding part in the drawer's frame, for clearance checks."""
    return [(np.asarray(p.offset), np.asarray(p.size)) for p in drawer.parts if p.collision]
