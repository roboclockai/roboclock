"""Render the B4 liquid model, without moving the simulated glass or robot."""

import numpy as np


def liquid_mesh(appearance, path):
    from pxr import UsdGeom, UsdShade

    from roboclock.envs.lab.tube_liquid import TubeLiquid

    model = TubeLiquid()
    segments, rings = 64, 12
    angles = np.arange(segments) * 2 * np.pi / segments
    radii = np.linspace(0, model.radius, rings)
    xy = (radii[:, None, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)).reshape(-1, 2)
    faces = []
    count = len(xy)
    for r in range(rings - 1):
        for j in range(segments):
            a, b = r * segments + j, r * segments + (j + 1) % segments
            faces.append([a, a + segments, b + segments, b])
            faces.append([count + b, count + b + segments, count + a + segments, count + a])
    for j in range(segments):
        a, b = (rings - 1) * segments + j, (rings - 1) * segments + (j + 1) % segments
        faces.append([a, count + a, count + b, b])
    mesh = UsdGeom.Mesh.Define(appearance.stage, path + "/BlueSample")
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.array(faces).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    # Clear blue solution: refractive volume and a readable smooth meniscus.
    mat = appearance.glass("TubeSolution", color=[0.12, 0.60, 0.94], ior=1.333, roughness=0.035)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat.material)
    mesh.CreatePointsAttr(
        np.c_[np.tile(xy, (2, 1)), np.r_[np.full(count, 0.064), model.bottom(xy)]].tolist()
    )
    return mesh, xy


class TubeLiquidVisuals:
    def __init__(self, backend):
        from pxr import UsdGeom, UsdShade

        self.backend = backend
        self.mesh, self.xy = liquid_mesh(backend.appearance, "/World/Objects/sample_0")
        self.drops = UsdGeom.Points.Define(backend.stage, "/World/TubeDrops")
        self.drops.CreateWidthsAttr([0.00284])  # 0.012 mL spherical drops
        self.drops.SetWidthsInterpolation("constant")
        self.drops.CreatePointsAttr([])
        UsdShade.MaterialBindingAPI.Apply(self.drops.GetPrim()).Bind(
            backend.appearance.material(
                "BlueDrops", color=[0.015, 0.28, 0.65], roughness=0.08
            ).material
        )
        self.puddle = UsdGeom.Sphere.Define(backend.stage, "/World/TubeSpill")
        self.puddle.CreateRadiusAttr(1)
        self.puddle_pos = self.puddle.AddTranslateOp()
        self.puddle_scale = self.puddle.AddScaleOp()
        UsdShade.MaterialBindingAPI.Apply(self.puddle.GetPrim()).Bind(
            backend.appearance.material(
                "BlueSpill", color=[0.015, 0.28, 0.65], roughness=0.08
            ).material
        )
        self.puddle.MakeInvisible()

    def update(self):
        from pxr import Gf, Vt

        model = self.backend.task.liquid
        points = np.c_[
            np.tile(self.xy, (2, 1)), np.r_[model.heights(self.xy), model.bottom(self.xy)]
        ]
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        self.drops.GetPointsAttr().Set([Gf.Vec3f(*d["p"]) for d in model.drops])
        if model.puddle_ml > 0:
            self.puddle.MakeVisible()
            radius = np.sqrt(model.puddle_ml * 1e-6 / (np.pi * 0.0006))
            self.puddle_pos.Set(Gf.Vec3d(*model.puddle_position))
            self.puddle_scale.Set(Gf.Vec3f(float(radius), float(radius), 0.00045))


def closeup_camera(backend):
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    target = backend.layout.objects["sample_0"].position + [0.055, 0, 0.10]
    position = target + [0.32, 0.52, 0.24]
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    backend.camera.set_world_pose(
        position, gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat()), camera_axes="usd"
    )
    backend.camera.set_focal_length(0.035)
