"""Explicit authored sports fixtures and hollow collision geometry."""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind


def add_glove_mass(prim, tool_rotation=None):
    """Combine the authored wrist inertia with a 120 g ellipsoidal tool model."""
    from pxr import Gf, UsdPhysics
    from scipy.spatial.transform import Rotation

    api = UsdPhysics.MassAPI.Apply(prim)
    old_mass = float(api.GetMassAttr().Get())
    old_center = np.asarray(api.GetCenterOfMassAttr().Get(), float)
    axes = api.GetPrincipalAxesAttr().Get()
    rotation = Rotation.from_quat([*axes.GetImaginary(), axes.GetReal()]).as_matrix()
    old_inertia = rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T
    tool_mass = 0.12
    tool_center = np.array([0, 0, 0.174])
    tool_rotation = np.eye(3) if tool_rotation is None else np.asarray(tool_rotation)
    tool_center = tool_rotation @ tool_center
    center = (old_mass * old_center + tool_mass * tool_center) / (old_mass + tool_mass)
    radii = np.array([0.052, 0.048, 0.067])
    tool_inertia = np.diag(tool_mass / 5 * (np.sum(radii**2) - radii**2))
    inertia = old_inertia + tool_rotation @ tool_inertia @ tool_rotation.T
    for mass, offset in [(old_mass, old_center - center), (tool_mass, tool_center - center)]:
        inertia += mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    diagonal, principal = np.linalg.eigh(inertia)
    if np.linalg.det(principal) < 0:
        principal[:, 0] *= -1
    q = Rotation.from_matrix(principal).as_quat()
    api.CreateMassAttr(old_mass + tool_mass)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, center)))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, diagonal)))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))


def rod(b, path, start, end, radius, material="steel", collision=False):
    from pxr import Gf, UsdGeom

    p, q = np.asarray(start, float), np.asarray(end, float)
    v = q - p
    shape, _ = b.collider(
        path, "cylinder", (p + q) / 2, [radius, float(np.linalg.norm(v))], [1, 1, 1], collision
    )
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, v / np.linalg.norm(v)))).GetQuat()
    UsdGeom.Xformable(shape).AddOrientOp().Set(
        Gf.Quatf(rot.GetReal(), Gf.Vec3f(rot.GetImaginary()))
    )
    bind(b, path, b.appearance.material(material))
    return shape


def ring(b, path, center, radius, tube, material="steel", collision=False, segments=64, plane="xy"):
    c = np.asarray(center)
    for i in range(segments):
        a, d = 2 * np.pi * i / segments, 2 * np.pi * (i + 1) / segments

        def pos(t):
            return (
                c
                + (
                    np.array([np.cos(t), np.sin(t), 0])
                    if plane == "xy"
                    else np.array([0, np.cos(t), np.sin(t)])
                )
                * radius
            )

        rod(b, f"{path}/Segment{i}", pos(a), pos(d), tube, material, collision)


def ellipsoid(b, path, center, scale, material, collision=False):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    from scipy.spatial import ConvexHull

    shape, _ = b.collider(path, "sphere", center, [1], [1, 1, 1], False)
    UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    bind(b, path, b.appearance.material(material))
    if collision:
        # PhysX spheres do not support nonuniform scaling. Bake an explicit
        # convex ellipsoid instead of silently inflating to the largest radius.
        vertices = []
        for theta in np.linspace(0.01, np.pi - 0.01, 12):
            for phi in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                vertices.append(
                    np.asarray(center)
                    + np.asarray(scale)
                    * [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
                )
        vertices = np.asarray(vertices)
        hull = ConvexHull(vertices)
        mesh = UsdGeom.Mesh.Define(b.stage, path + "Collision")
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
        mesh.CreateFaceVertexIndicesAttr(hull.simplices.flatten().tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        api.CreateContactOffsetAttr(0.0005)
        api.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            b.material.material, materialPurpose="physics"
        )
        UsdGeom.Imageable(mesh).MakeInvisible()
    return shape


def cup(b, name, position, color):
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

    path = f"/World/Objects/{name}"
    UsdGeom.Xform.Define(b.stage, path)
    # Inverted competition cup: open wide base, closed narrow top, actual hollow interior.
    profile = [
        (0.032, -0.036),
        (0.0325, -0.034),
        (0.024, 0.034),
        (0.023, 0.036),
        (0.0, 0.036),
        (0.0, 0.034),
        (0.021, 0.034),
        (0.030, -0.034),
        (0.030, -0.036),
    ]
    mesh = author_mesh(b.stage, path + "/Shell", revolved(profile))
    mat = b.appearance.material(name, color=color, roughness=0.29)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat.material)
    # Separate convex wall wedges preserve the cavity and rim contact.
    for i in range(24):
        a, d = 2 * np.pi * i / 24, 2 * np.pi * (i + 1) / 24
        verts = []
        for z, ri, ro in [(-0.036, 0.030, 0.032), (0.034, 0.021, 0.024)]:
            for r, t in [(ri, a), (ro, a), (ro, d), (ri, d)]:
                verts.append((r * np.cos(t), r * np.sin(t), z))
        faces = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
        m = UsdGeom.Mesh.Define(b.stage, path + f"/Wall{i}")
        m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in verts])
        m.CreateFaceVertexCountsAttr([4] * 6)
        m.CreateFaceVertexIndicesAttr(faces)
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim()).CreateContactOffsetAttr(0.0005)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
            b.material.material, materialPurpose="physics"
        )
        UsdGeom.Imageable(m).MakeInvisible()
    cap, _ = b.collider(path + "/Cap", "cylinder", [0, 0, 0.035], [0.023, 0.002], [1, 1, 1])
    UsdGeom.Imageable(cap).MakeInvisible()
    ring(b, path + "/Rim", [0, 0, -0.0345], 0.0316, 0.001, "cream", False, 32)
    b.label(path + "/Brand", "PLAY", [0, 0, 0.0362], [0.027, 0.009], False)
    obj = b.world.scene.add(
        SingleRigidPrim(path, name=name, position=np.asarray(position), mass=0.028)
    )
    PhysxSchema.PhysxRigidBodyAPI.Apply(obj.prim).CreateEnableCCDAttr(True)
    return obj
