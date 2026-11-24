"""Task-specific rooms with sourced furniture and explicit authored fixtures."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent / "assets"


def imported(stage, asset, path, position, size=None, yaw=0):
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom

    item = json.loads((ROOT / "props.json").read_text())[asset]
    prim = add_reference_to_stage(str((ROOT / item["file"]).resolve()), path)
    low, high = np.array(item["min"]), np.array(item["max"])
    scale = np.ones(3) if size is None else np.asarray(size) / (high - low)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    xf.AddRotateZOp().Set(yaw)
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    xf.AddTranslateOp(opSuffix="center").Set(
        Gf.Vec3d(*map(float, -np.r_[(low[:2] + high[:2]) / 2, low[2]]))
    )
    bind_isaac_materials(stage, prim)
    return prim


def vertical_label(b, path, text, position, size):
    from pxr import Gf, UsdGeom

    b.label(path, text, [0, 0, 0], size, False)
    xf = UsdGeom.Xformable(b.stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    xf.AddRotateXOp().Set(90)


def build_room(b, theme):
    if b.config["scene"].get("room_style") == "v5":
        from roboclock.envs.sports.arcade.classic import build_room_classic

        return build_room_classic(b, "chess" if theme == "chess" else "arcade")
    from pxr import Gf, UsdGeom, UsdLux

    from roboclock.envs.lab.appearance import LabAppearance

    a = LabAppearance(b.stage, b.world, b.config)
    for path in ("/World/Backdrop", "/World/TableSurface", "/World/Title"):
        prim = b.stage.GetPrimAtPath(path)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()
    a.material("navy", color=[0.025, 0.045, 0.065], roughness=0.65)
    a.material("walnut", color=[0.16, 0.07, 0.025], roughness=0.35)
    a.material("trim", color=[0.70, 0.43, 0.13], roughness=0.27, metallic=0.75)
    a.material("cyan", color=[0.05, 0.55, 0.75], emissive=True)
    a.material("coral", color=[0.75, 0.08, 0.06], emissive=True)
    a.rounded("/World/Room/Floor", [0.4, 0, -0.025], [7, 7, 0.05], "floor")
    a.rounded("/World/Room/BackWall", [0.4, 1.85, 1.6], [5.6, 0.08, 3.2], "wall")
    a.rounded("/World/Room/SideWall", [-2.35, 0, 1.6], [0.08, 3.8, 3.2], "wall")
    for z in (0.08, 1.02):
        a.rounded(
            f"/World/Room/Trim{int(z * 100)}",
            [0.4, 1.795, z],
            [5.6, 0.035, 0.06],
            "walnut" if theme == "chess" else "navy",
        )
    if theme == "chess":
        UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/Table")).MakeInvisible()
        imported(
            b.stage, "side_table_01", "/World/Room/PlayingTable", [0.40, 0, 0], [1.05, 0.95, 0.75]
        )
        for i, xy in enumerate(([-0.48, 0.98], [1.05, 1.02])):
            imported(
                b.stage, "dining_chair_02", f"/World/Room/Chair{i}", [*xy, 0], yaw=25 if i else -25
            )
        imported(
            b.stage, "side_table_01", "/World/Room/SideTable", [1.25, 1.30, 0], [0.5, 0.45, 0.64]
        )
        vertical_label(b, "/World/Room/Sign", "CHESS CLUB", [0.30, 1.795, 2.08], [1.8, 0.28])
        for i in range(3):
            x = -0.6 + i * 0.62
            a.rounded(f"/World/Room/Panel{i}", [x, 1.79, 1.48], [0.47, 0.04, 0.53], "walnut")
            vertical_label(
                b,
                f"/World/Room/PanelText{i}",
                ("THINK", "MOVE", "PLAY")[i],
                [x, 1.762, 1.48],
                [0.40, 0.12],
            )
        for j in range(6):
            a.rounded(
                f"/World/Room/WindowSlat{j}",
                [-2.29, 0.2 + j * 0.20, 1.85],
                [0.03, 0.035, 1.2],
                "walnut",
            )
    else:
        a.rounded("/World/Room/Worktop", [0.43, 0, 0.724], [1.05, 1.0, 0.05], "navy", radius=0.009)
        vertical_label(
            b, "/World/Room/Sign", "PLAY / RHYTHM / REACT", [0.3, 1.795, 2.22], [2.4, 0.24]
        )
        # Purpose-built background arcade fixtures, explicitly authored proxies.
        for i, x in enumerate((-0.9, 0.05, 1.0)):
            y = 1.25
            a.rounded(
                f"/World/Room/Cabinet{i}", [x, y, 0.64], [0.63, 0.55, 1.28], "navy", radius=0.04
            )
            a.rounded(
                f"/World/Room/Screen{i}",
                [x, y - 0.286, 1.25],
                [0.53, 0.02, 0.43],
                "polymer",
                radius=0.008,
            )
            vertical_label(
                b,
                f"/World/Room/GameTitle{i}",
                ("RHYTHM", "HIGH SCORE", "ARCADE")[i],
                [x, y - 0.300, 1.27],
                [0.47, 0.12],
            )
            a.rounded(
                f"/World/Room/Marquee{i}",
                [x, y, 1.67],
                [0.66, 0.58, 0.17],
                "coral" if i % 2 else "cyan",
                radius=0.025,
            )
            a.rounded(
                f"/World/Room/ControlShelf{i}",
                [x, y - 0.30, 0.87],
                [0.61, 0.20, 0.07],
                "steel",
                radius=0.01,
            )
            for side in (-1, 1):
                a.rounded(
                    f"/World/Room/Edge{i}_{side}".replace("-", "n"),
                    [x + side * 0.29, y - 0.282, 0.46],
                    [0.018, 0.014, 0.63],
                    "cyan",
                )
        imported(b.stage, "dining_chair_02", "/World/Room/Chair", [1.35, 0.3, 0], yaw=70)
    # Architectural ceiling fixtures provide a room-scale lighting context.
    for i, x in enumerate((-0.5, 0.8)):
        a.rounded(f"/World/Room/LightHousing{i}", [x, 0.8, 2.65], [0.65, 0.35, 0.06], "steel")
        light = UsdLux.RectLight.Define(b.stage, f"/World/Room/CeilingLight{i}")
        light.AddTranslateOp().Set(Gf.Vec3d(x, 0.8, 2.60))
        light.CreateWidthAttr(0.58)
        light.CreateHeightAttr(0.28)
        light.CreateIntensityAttr(450)
        light.CreateColorAttr(
            Gf.Vec3f(1, 0.88, 0.72) if theme == "chess" else Gf.Vec3f(0.78, 0.88, 1)
        )
    from roboclock.envs.sports.arcade.refinement import finish_room

    finish_room(b, a, theme)
    return a


def bind_isaac_materials(stage, root):
    """Translate source PreviewSurface maps to explicit Isaac-compatible MDL.

    Preserve mesh UVs and the original PBR texture files. MaterialX graphs in
    these Blender-exported USDs are not rendered correctly by this Kit build.
    """
    from isaacsim.core.api.materials import OmniPBR
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    cache = {}
    for prim in list(Usd.PrimRange(root)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        original, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not original:
            continue
        key = str(original.GetPath())
        if key not in cache:
            preview = next(
                (
                    UsdShade.Shader(p)
                    for p in Usd.PrimRange(original.GetPrim())
                    if p.IsA(UsdShade.Shader)
                    and UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"
                ),
                None,
            )
            if preview is None:
                continue
            material = OmniPBR(
                str(root.GetPath()) + "/IsaacMaterials/" + original.GetPrim().GetName(),
                color=np.ones(3),
            )
            material.set_project_uvw(False)
            material.set_reflection_roughness(0.45)
            material.set_metallic_constant(0)
            shader = material.shaders_list[0]
            for source, target, space in [
                ("diffuseColor", "diffuse_texture", "sRGB"),
                ("roughness", "reflectionroughness_texture", "raw"),
                ("normal", "normalmap_texture", "raw"),
            ]:
                connections = preview.GetInput(source).GetAttr().GetConnections()
                if not connections:
                    continue
                tex = UsdShade.Shader(stage.GetPrimAtPath(connections[0].GetPrimPath()))
                asset = tex.GetInput("file").Get()
                if asset:
                    path = asset.resolvedPath or asset.path
                    if not Path(path).is_file():
                        # Some source USD PreviewSurface graphs request EXR while
                        # the publisher's USD manifest supplies the same map as JPG.
                        matches = list(ROOT.rglob(Path(path).stem + ".*"))
                        if len(matches) != 1:
                            raise FileNotFoundError(path)
                        path = str(matches[0].resolve())
                    param = shader.CreateInput(target, Sdf.ValueTypeNames.Asset)
                    param.Set(Sdf.AssetPath(path))
                    param.GetAttr().SetMetadata("colorSpace", space)
            shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
            shader.CreateInput(
                "reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float
            ).Set(1)
            shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.25)
            shader.CreateInput("flip_tangent_v", Sdf.ValueTypeNames.Bool).Set(False)
            cache[key] = material
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(cache[key].material)
