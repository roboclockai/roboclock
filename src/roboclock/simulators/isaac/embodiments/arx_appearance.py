"""Photo-referenced surface finishes for unmodified manufacturer CAD meshes.

STL has no appearance data. These are authored finishes, not manufacturer PBR maps.
Reference: ARX5_beta/images/image_6.png at 6768e886935c7a8a368e4dc29adfe10a8b7ca473.
"""

import numpy as np


def material(stage, path, color, roughness=0.34, metallic=0.0):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def apply_surfaces(stage, root):
    from pxr import Usd, UsdGeom, UsdShade

    path = str(root.GetPath()) + "/SurfaceFinishes"
    mats = {
        "shell": material(stage, path + "/GraphiteAnodized", (0.052, 0.059, 0.067), 0.32, 0.35),
        "panel": material(stage, path + "/PearlAluminium", (0.68, 0.68, 0.65), 0.29, 0.25),
        "rail": material(stage, path + "/MachinedRail", (0.42, 0.45, 0.48), 0.23, 0.85),
        "jaw": material(stage, path + "/GripperGraphite", (0.033, 0.038, 0.043), 0.38, 0.25),
    }
    links = [
        p
        for p in Usd.PrimRange(root)
        if p.GetName()
        in [
            s + "_" + n
            for s in ("left", "right")
            for n in ["base_link"] + [f"link{i}" for i in range(1, 9)]
        ]
    ]
    for link in links:
        name = link.GetName().split("_", 1)[1]
        visual = stage.GetPrimAtPath(str(link.GetPath()) + "/" + name)
        if not visual:
            raise ValueError(f"Missing manufacturer visual: {link.GetPath()}/{name}")
        # Only visual instances are expanded; collision instances stay untouched.
        visual.SetInstanceable(False)
        for p in Usd.PrimRange(visual):
            if not p.IsA(UsdGeom.Mesh):
                continue
            kind = (
                "panel" if name == "link3" else ("jaw" if name in ("link7", "link8") else "shell")
            )
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mats[kind])
            if name in ("link7", "link8"):
                mesh = UsdGeom.Mesh(p)
                points = np.asarray(mesh.GetPointsAttr().Get())
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                starts = np.r_[0, np.cumsum(counts)]
                rail = [
                    i
                    for i in range(len(counts))
                    if points[indices[starts[i] : starts[i + 1]], 0].mean() < -0.002
                ]
                subset = UsdGeom.Subset.CreateGeomSubset(
                    mesh, "MachinedSlide", UsdGeom.Tokens.face, rail, "materialBind"
                )
                UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mats["rail"])
    root.SetCustomDataByKey("benchmark:assetAppearance", "official-x5a-photo-finishes-v1")
