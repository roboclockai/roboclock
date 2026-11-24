"""Authored bolted extrusion fixture; manufacturer arm meshes remain untouched.

This fixture is benchmark equipment, not an official ARX torso or product CAD.
The original support collision envelopes are retained beneath detailed visuals.
"""

import itertools

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def build_mount(world, stage, path, position, yaw):
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import Gf, UsdGeom, UsdLux, UsdShade

    from .arx_appearance import material
    from .views import look_pose

    root = UsdGeom.Xform.Define(stage, path + "/Detail")
    root.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    root.AddRotateZOp().Set(float(np.degrees(yaw)))
    root.GetPrim().SetCustomDataByKey(
        "benchmark:provenance", "Authored fixed extrusion fixture; not official ARX CAD"
    )
    mats = {
        "alloy": material(stage, path + "/Alloy", (0.31, 0.34, 0.37), 0.3, 0.8),
        "dark": material(stage, path + "/Coating", (0.045, 0.055, 0.067), 0.4, 0.25),
        "slot": material(stage, path + "/Recess", (0.012, 0.017, 0.022), 0.5, 0.1),
        "steel": material(stage, path + "/Steel", (0.48, 0.5, 0.52), 0.25, 0.9),
    }

    def box(name, center, size, kind="alloy", bevel=0.002):
        # A 24-vertex chamfered box, preserving the original part envelope.
        half = np.asarray(size) / 2
        b = min(bevel, float(half.min()) * 0.45)
        points = []
        for signs in itertools.product((-1, 1), repeat=3):
            for axis in range(3):
                v = half - b
                v[axis] = half[axis]
                points.append(v * np.array(signs))
        points = np.asarray(points)
        hull = ConvexHull(points)
        faces = hull.simplices.copy()
        for i, face in enumerate(faces):
            a, c, d = points[face]
            if np.dot(np.cross(c - a, d - a), hull.equations[i, :3]) < 0:
                faces[i] = face[::-1]
        mesh = UsdGeom.Mesh.Define(stage, path + "/Detail/" + name)
        mesh.CreatePointsAttr(points.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[kind])
        return mesh

    def bolt(name, center, radius=0.004, height=0.003, axis="Z"):
        prim = UsdGeom.Cylinder.Define(stage, path + "/Detail/" + name)
        prim.CreateRadiusAttr(radius)
        prim.CreateHeightAttr(height)
        prim.CreateAxisAttr(axis)
        prim.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mats["steel"])

    r = Rotation.from_euler("z", yaw)
    q = np.roll(r.as_quat(), 1)
    height = float(position[2])
    pieces = [("Crossbeam", [0, 0, -0.045], [0.11, 0.74, 0.07], "alloy")]
    for i, y in enumerate([-0.30, 0.30]):
        pieces += [
            (f"Column{i}", [0, y, -height / 2], [0.075, 0.075, height - 0.10], "alloy"),
            (f"Foot{i}", [0, y, 0.025 - height], [0.24, 0.18, 0.05], "dark"),
            (f"Cap{i}", [0, y, -0.009], [0.12, 0.12, 0.018], "dark"),
        ]
    for name, center, size, kind in pieces:
        world.scene.add(
            FixedCuboid(
                path + "/" + name,
                name=path.replace("/", "_") + name,
                position=np.asarray(position) + r.apply(center),
                orientation=q,
                scale=np.asarray(size),
                visible=False,
            )
        )
        box(name, center, size, kind)
    # T-slot channels, cap seams and socket-head fasteners remain visual details.
    for i, y in enumerate([-0.30, 0.30]):
        for face, x in enumerate([-0.0376, 0.0376]):
            box(
                f"ColumnSlot{i}_{face}",
                [x, y, -height / 2],
                [0.0005, 0.007, height - 0.12],
                "slot",
                0.0001,
            )
        for j, dy in enumerate([-0.0376, 0.0376]):
            box(
                f"ColumnSideSlot{i}_{j}",
                [0, y + dy, -height / 2],
                [0.007, 0.0005, height - 0.12],
                "slot",
                0.0001,
            )
        for j, (x, dy) in enumerate(itertools.product([-0.047, 0.047], repeat=2)):
            bolt(f"PlateBolt{i}_{j}", [x, y + dy, 0.0015])
        for j, (x, dy) in enumerate(itertools.product([-0.088, 0.088], [-0.058, 0.058])):
            bolt(f"AnchorBolt{i}_{j}", [x, y + dy, 0.051 - height], 0.006, 0.004)
        for j, sign in enumerate([-1, 1]):
            box(f"BeamEnd{i}_{j}", [sign * 0.056, y, -0.045], [0.004, 0.095, 0.055], "dark")
            bolt(f"BeamBolt{i}_{j}", [sign * 0.059, y, -0.045], 0.005, 0.003, "X")
    for i, x in enumerate([-0.0551, 0.0551]):
        box(f"BeamSlot{i}", [x, 0, -0.045], [0.0005, 0.72, 0.008], "slot", 0.0001)
    # Soft local fill reveals black anodized CAD detail without changing room style.
    eye = np.asarray(position) + r.apply([0.35, -1.0, 1.4])
    target = np.asarray(position) + r.apply([0, 0, 0.30])
    light = UsdLux.RectLight.Define(stage, path + "/RobotFill")
    light.CreateIntensityAttr(130)
    light.CreateWidthAttr(1.0)
    light.CreateHeightAttr(0.7)
    light.CreateColorAttr(Gf.Vec3f(0.88, 0.93, 1.0))
    pos, orient = look_pose(eye, target, [0, 0, 1])
    xf = UsdGeom.Xformable(light)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    xf.AddOrientOp().Set(Gf.Quatf(float(orient[0]), Gf.Vec3f(*map(float, orient[1:]))))


def close_room(stage):
    """Close the previously off-camera room side for robot-mounted cameras."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    path = "/World/RobotViewBackdrop"
    if stage.GetPrimAtPath(path):
        return
    wall = UsdGeom.Cube.Define(stage, path)
    wall.CreateSizeAttr(1)
    wall.AddTranslateOp().Set(Gf.Vec3d(2.4, 0, 1.6))
    wall.AddScaleOp().Set(Gf.Vec3f(0.08, 4.5, 3.2))
    material = UsdShade.Material.Define(stage, path + "/Material")
    shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.065, 0.085, 0.085)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(wall.GetPrim()).Bind(material)
