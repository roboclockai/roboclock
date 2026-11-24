"""PBR materials, HDR lighting, reusable laboratory equipment, and visible labels.

Appearance geometry is kept separate from collision fixtures. Labels contain only
sample identities and workflow descriptions, never hidden event schedules.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

from roboclock.resources import resolve_asset

PALETTE = {
    "worktop": ([0.15, 0.19, 0.20], 0.38, 0.0),
    "steel": ([0.52, 0.57, 0.61], 0.30, 0.92),
    "aluminum": ([0.67, 0.70, 0.73], 0.24, 0.88),
    "polymer": ([0.57, 0.61, 0.62], 0.46, 0.0),
    "rubber": ([0.024, 0.029, 0.035], 0.78, 0.0),
    "ceramic": ([0.05, 0.065, 0.08], 0.18, 0.05),
    "paper": ([0.92, 0.94, 0.94], 0.78, 0.0),
    "wall": ([0.55, 0.60, 0.64], 0.85, 0.0),
}


def label_texture(text, color=(0.1, 0.35, 0.5), *, printed=False, serial=True) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    suffix = "" if serial else "|plain"  # keeps existing cache keys for the default labels
    key_text = text + repr(tuple(color)) + str(printed) + suffix
    key = hashlib.sha256(key_text.encode()).hexdigest()[:20]
    path = cache / f"{key}.png"
    if not path.exists():
        image = Image.new(
            "RGBA" if printed else "RGB", (512, 256), (0, 0, 0, 0) if printed else (238, 243, 245)
        )
        draw = ImageDraw.Draw(image)
        rgb = tuple(round(255 * x) for x in color)
        if not printed:
            draw.rectangle((0, 0, 512, 28), fill=rgb)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = (
            ImageFont.truetype(font_path, 172 if printed else 72)
            if Path(font_path).exists()
            else ImageFont.load_default(size=72)
        )
        draw.text(
            (256, 128 if printed else 116),
            text,
            fill=rgb if printed else (24, 37, 48),
            font=font,
            anchor="mm",
        )
        # A visual serial pattern, not a machine-readable simulator ID channel.
        for i, byte in enumerate(
            [] if printed or not serial else hashlib.sha256(text.encode()).digest()[:24]
        ):
            if byte & 1:
                draw.rectangle((28 + 18 * i, 202, 34 + 18 * i, 239), fill=(39, 50, 56))
        image.save(path)
    return path


def sign_texture(text, color=(0.1, 0.35, 0.5), aspect=4.0) -> Path:
    """A sign whose bold text fills the label (capitals about 60 % of its height), for labels a
    640x480 policy camera has to read: `label_texture` letters are a fifth of the label's
    height. `aspect` is the label's width / height, so the text is not stretched."""
    from PIL import Image, ImageDraw, ImageFont

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    height = 256
    width = int(np.clip(round(height * aspect), height, 4096))
    key = hashlib.sha256(f"{text}{tuple(color)!r}|sign|{width}".encode()).hexdigest()[:20]
    path = cache / f"{key}.png"
    if not path.exists():
        image = Image.new("RGB", (width, height), (238, 243, 245))
        draw = ImageDraw.Draw(image)
        rgb = tuple(round(255 * x) for x in color)
        draw.rectangle((0, 0, width - 1, height - 1), outline=rgb, width=10)
        bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        size = 216  # capitals 0.73 em: 158 px of 256
        while True:
            font = (
                ImageFont.truetype(str(bold), size)
                if bold.exists()
                else ImageFont.load_default(size=size)
            )
            if draw.textlength(text, font=font) <= width - 48 or size <= 24:
                break
            size -= 4
        draw.text((width / 2, height / 2), text, fill=(24, 37, 48), font=font, anchor="mm")
        image.save(path)
    return path


class LabAppearance:
    serial_pattern = True  # decorative barcode strip on labels; domains may turn it off

    def __init__(self, stage, world, config):
        self.stage, self.world, self.config = stage, world, config
        self.materials = {}

    def material(self, name, *, color=None, roughness=None, metallic=None, emissive=False):
        from isaacsim.core.api.materials import OmniPBR
        from pxr import Gf, Sdf

        if name not in self.materials:
            c, r, m = PALETTE.get(name, ([0.3, 0.6, 0.7], 0.35, 0.0))
            specification = self.config["visuals"].get("materials", {}).get(name, {})
            material = OmniPBR(
                f"/World/Materials/{name}", color=np.array(c if color is None else color)
            )
            material.set_reflection_roughness(r if roughness is None else roughness)
            material.set_metallic_constant(m if metallic is None else metallic)
            shader = material.shaders_list[0]
            texture_set = specification.get("texture_set")
            if texture_set:
                for suffix, parameter in [
                    ("diff", "diffuse_texture"),
                    ("rough", "reflectionroughness_texture"),
                    ("nor_gl", "normalmap_texture"),
                ]:
                    matches = list(
                        resolve_asset("env://textures", environment="lab").glob(
                            f"{texture_set}_{suffix}_1k.*"
                        )
                    )
                    if len(matches) != 1:
                        raise ValueError(
                            f"Missing or ambiguous texture map: {texture_set}/{suffix}"
                        )
                    texture_input = shader.CreateInput(parameter, Sdf.ValueTypeNames.Asset)
                    texture_input.Set(Sdf.AssetPath(str(matches[0])))
                    texture_input.GetAttr().SetMetadata(
                        "colorSpace", "sRGB" if suffix == "diff" else "raw"
                    )
                shader.CreateInput(
                    "reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float
                ).Set(specification.get("roughness_map_weight", 0.65))
                shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(
                    specification.get("normal_strength", 0.25)
                )
                shader.CreateInput("flip_tangent_v", Sdf.ValueTypeNames.Bool).Set(False)
                shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*specification.get("tint", [1.0, 1.0, 1.0]))
                )
                material.set_texture_scale(*specification.get("uv_scale", [1.0, 1.0]))
                shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
            if emissive:
                shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
                shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*(c if color is None else color))
                )
                shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(12.0)
            self.materials[name] = material
        return self.materials[name]

    def label_image(self, text, color):
        return label_texture(text, color, serial=self.serial_pattern)

    def glass(self, name="glass", *, color=None, ior=1.47, roughness=0.06):
        from isaacsim.core.api.materials.visual_material import VisualMaterial
        from pxr import Gf, Sdf, UsdShade

        if name not in self.materials:
            path = "/World/Materials/" + ("BorosilicateGlass" if name == "glass" else name)
            material = UsdShade.Material.Define(self.stage, path)
            shader = UsdShade.Shader.Define(self.stage, path + "/Shader")
            shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*(color if color is not None else [0.93, 0.98, 1.0]))
            )
            shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(ior)
            shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.025)
            shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
            shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput("mdl").ConnectToSource(out)
            self.materials[name] = VisualMaterial(
                name="borosilicate",
                prim_path=path,
                prim=material.GetPrim(),
                shaders_list=[shader],
                material=material,
            )
        return self.materials[name]

    def rounded(self, path, position, size, material="polymer", radius=0.003):
        from pxr import UsdShade

        from roboclock.simulators.isaac.visual_geometry import author_mesh, rounded_box

        points, normals, uv, indices = rounded_box(size, radius)
        mesh = author_mesh(self.stage, path, (points, uv, indices), position, normals)
        visual = self.material(material) if isinstance(material, str) else material
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual.material)
        return mesh

    def _world_pose(self, path, position):
        """(world position, world (w, x, y, z) or None) of `position` given in the frame of the
        nearest existing ancestor of `path`."""
        from pxr import Gf, Sdf, UsdGeom

        parent_path = Sdf.Path(path).GetParentPath()
        parent = self.stage.GetPrimAtPath(parent_path)
        while not parent.IsValid() and parent_path != Sdf.Path.absoluteRootPath:
            parent_path = parent_path.GetParentPath()
            parent = self.stage.GetPrimAtPath(parent_path)
        if not parent.IsValid() or parent.IsPseudoRoot():
            return np.array(position, dtype=float), None
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
        if np.allclose(np.array(matrix), np.eye(4), atol=1e-12):
            return np.array(position, dtype=float), None
        world = matrix.Transform(Gf.Vec3d(*map(float, position)))
        rotation = matrix.ExtractRotationQuat()
        return np.array(world), np.array([rotation.GetReal(), *rotation.GetImaginary()])

    def cube(self, path, position, size, material="polymer", collision=False):
        if not collision:
            return self.rounded(path, position, size, material)
        from isaacsim.core.api.objects import FixedCuboid
        from pxr import UsdGeom

        # FixedCuboid takes a world pose, whereas `position` is in the parent's frame, as for
        # every other helper here: the A1-A6 station roots are placed by a seeded
        # station_frame.StationFrame. Under an untransformed parent this changes nothing.
        world, orientation = self._world_pose(path, position)
        body = self.world.scene.add(
            FixedCuboid(
                prim_path=path,
                name=path.replace("/", "_"),
                position=world,
                orientation=orientation,
                scale=np.array(size),
            )
        )
        UsdGeom.Imageable(body.prim).MakeInvisible()
        self.rounded(path + "_Visual", position, size, material)
        return body

    def vial(self, path, radius, height, cap_material):
        from pxr import UsdShade

        from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

        bottom, top = -height / 2, height / 2
        outer = [
            (radius * 0.08, bottom + 0.001),
            (radius * 0.85, bottom + 0.001),
            (radius * 0.97, bottom + 0.004),
            (radius, bottom + 0.008),
            (radius, top - 0.020),
            (radius * 0.92, top - 0.015),
            (radius * 0.73, top - 0.011),
            (radius * 0.73, top - 0.008),
        ]
        inner = [(r - 0.0015, z if i else z + 0.002) for i, (r, z) in enumerate(outer[1:])]
        profile = outer + inner[::-1]
        points, uv, indices = revolved(profile)
        mesh = author_mesh(self.stage, path + "/GlassBody", (points, uv, indices))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.glass().material)
        # Cap skirt, crown and 48 small grip flutes within the original envelope.
        self.cylinder_child(path, "CapSkirt", radius * 0.91, 0.008, top - 0.005, cap_material)
        self.cylinder_child(path, "CapCrown", radius * 0.87, 0.0014, top - 0.0007, cap_material)
        for i in range(48):
            angle = 2 * np.pi * i / 48
            self.rounded(
                path + f"/CapFlute_{i}",
                [radius * 0.90 * np.cos(angle), radius * 0.90 * np.sin(angle), top - 0.005],
                [0.0012, 0.0012, 0.006],
                cap_material,
                0.0003,
            )
        self.cylinder_child(path, "TamperRing", radius * 0.86, 0.001, top - 0.010, cap_material)

    def label(
        self,
        path,
        text,
        center,
        size,
        color=(0.1, 0.35, 0.5),
        *,
        cylindrical_radius=None,
        printed=False,
        sign=False,
    ):
        """A textured quad in the local xz plane, read from +y (text runs toward -x, up is +z).
        `sign=True` uses `sign_texture`: bold text filling the label, for policy cameras."""
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        if sign:
            texture = sign_texture(text, color, size[0] / size[1])
        else:
            texture = (
                label_texture(text, color, printed=True)
                if printed
                else self.label_image(text, color)
            )
        material = UsdShade.Material.Define(self.stage, path + "_material")
        shader = UsdShade.Shader.Define(self.stage, path + "_material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        tex = UsdShade.Shader.Define(self.stage, path + "_material/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        reader = UsdShade.Shader.Define(self.stage, path + "_material/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        if printed:
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(
                tex.ConnectableAPI(), "a"
            )
            shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        w, h = size
        segments = 20 if cylindrical_radius else 1
        points, uv, indices = [], [], []
        for i in range(segments + 1):
            u = i / segments
            if cylindrical_radius:
                angle = (u - 0.5) * w / cylindrical_radius
                x, y = cylindrical_radius * np.sin(angle), cylindrical_radius * np.cos(angle)
                points.extend([(x, y, -h / 2), (x, y, h / 2)])
            else:
                points.extend([((u - 0.5) * w, 0.0, -h / 2), ((u - 0.5) * w, 0.0, h / 2)])
            uv.extend([(1 - u, 0.0), (1 - u, 1.0)])
            if i < segments:
                indices.extend([2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2])
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        mesh.CreateFaceVertexCountsAttr([4] * segments)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(uv)
        mesh.AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in center]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        return mesh

    def lights(self, layout):
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        spec = self.config.get("visuals", {})
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lighting/HDR")
        hdr = resolve_asset(
            spec.get("hdr_path", "shared://hdr/studio_small_09_1k.hdr"), environment="lab"
        )
        if not hdr.exists():
            raise ValueError(f"HDR environment is missing: {hdr}")
        dome.CreateTextureFileAttr(Sdf.AssetPath(str(hdr)))
        dome.CreateTextureFormatAttr("latlong")
        dome.CreateIntensityAttr(spec.get("hdr_intensity", 450.0) * layout.light_intensity / 2200)
        UsdGeom.Xformable(dome).AddRotateZOp().Set(spec.get("hdr_rotation", 35.0))
        for name, pos, intensity, scale in (
            ("Key", [0.45, -0.25, 2.0], spec.get("key_intensity", 1500.0), [1.2, 0.65]),
            ("Fill", [0.75, 0.55, 1.95], spec.get("fill_intensity", 450.0), [0.8, 0.8]),
        ):
            light = UsdLux.RectLight.Define(self.stage, f"/World/Lighting/{name}")
            light.CreateIntensityAttr(intensity)
            light.CreateWidthAttr(scale[0])
            light.CreateHeightAttr(scale[1])
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(5400.0)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*pos))

    def workbench(self, layout):
        from pxr import UsdShade

        z = layout.table_height
        # Decorations sit outside the robot's manipulated sample/target regions.
        for i, (x, y) in enumerate(((-0.005, -0.48), (0.90, -0.48), (-0.005, 0.48), (0.90, 0.48))):
            self.cube(f"/World/Bench/Leg_{i}", [x, y, 0.34], [0.045, 0.045, 0.68], "steel")
        self.cube("/World/Bench/Apron", [0.45, 0.0, 0.66], [0.94, 1.02, 0.06], "steel")
        self.cube("/World/Bench/Mount", [0.0, 0.0, z + 0.003], [0.14, 0.16, 0.006], "aluminum")
        self.cube("/World/Room/BackWall", [0.4, -0.84, 1.1], [2.5, 0.06, 2.2], "wall")
        self.cube("/World/Room/Baseboard", [0.4, -0.797, 0.10], [2.5, 0.025, 0.12], "steel")
        # Storage at the back of the laboratory introduces plausible depth and
        # material variation without entering the robot's manipulation region.
        for i, x in enumerate([-0.30, 0.12, 0.54, 0.96]):
            self.cube(f"/World/Room/Cabinet_{i}", [x, -0.91, 1.49], [0.40, 0.18, 0.47], "polymer")
            self.cube(
                f"/World/Room/CabinetDoor_{i}", [x, -0.81, 1.49], [0.386, 0.016, 0.448], "paper"
            )
            self.cube(
                f"/World/Room/CabinetHandle_{i}",
                [x + 0.125, -0.792, 1.46],
                [0.012, 0.018, 0.105],
                "steel",
            )
        self.cube("/World/Room/BackCounter", [0.40, -0.66, 0.74], [1.8, 0.25, 0.055], "worktop")
        for i, x in enumerate([-0.16, 0.28, 0.72, 1.16]):
            self.cube(
                f"/World/Room/LowerCabinet_{i}", [x, -0.72, 0.38], [0.425, 0.26, 0.66], "polymer"
            )
            for row in range(3):
                self.cube(
                    f"/World/Room/Drawer_{i}_{row}",
                    [x, -0.58, 0.18 + row * 0.20],
                    [0.407, 0.015, 0.185],
                    "paper",
                )
                self.cube(
                    f"/World/Room/DrawerHandle_{i}_{row}",
                    [x, -0.561, 0.19 + row * 0.20],
                    [0.12, 0.020, 0.010],
                    "steel",
                )
        # Recessed service panel and outlets give the room a laboratory scale.
        self.cube("/World/Room/ServicePanel", [0.4, -0.80, 1.02], [1.70, 0.016, 0.08], "aluminum")
        for i, x in enumerate([-0.20, 0.05, 0.30, 0.55, 0.80, 1.05]):
            self.cube(f"/World/Room/Socket_{i}", [x, -0.786, 1.02], [0.062, 0.008, 0.060], "paper")
            for j, delta in enumerate([-0.012, 0.012]):
                self.cube(
                    f"/World/Room/SocketHole_{i}_{j}",
                    [x + delta, -0.780, 1.024],
                    [0.004, 0.003, 0.014],
                    "rubber",
                )
        # Reusable instrument silhouette: enclosure, lid, control face, vent slots.
        self.cube("/World/Equipment/Analyzer", [0.86, -0.32, z + 0.085], [0.15, 0.21, 0.17])
        self.cube(
            "/World/Equipment/AnalyzerLid", [0.86, -0.32, z + 0.177], [0.155, 0.21, 0.014], "paper"
        )
        self.cube(
            "/World/Equipment/Display", [0.86, -0.211, z + 0.115], [0.085, 0.004, 0.036], "ceramic"
        )
        self.label("/World/Equipment/Nameplate", "ASSAY", [0.86, -0.208, z + 0.055], [0.08, 0.028])
        for i in range(5):
            self.cube(
                f"/World/Equipment/Vent_{i}",
                [0.94, -0.38 + i * 0.022, z + 0.10],
                [0.001, 0.012, 0.050],
                "rubber",
            )
        self.cube(
            "/World/Equipment/AnalyzerGasket",
            [0.86, -0.32, z + 0.170],
            [0.151, 0.211, 0.003],
            "rubber",
        )
        for i, x in enumerate([0.802, 0.918]):
            for j, y in enumerate([-0.39, -0.25]):
                self.cube(
                    f"/World/Equipment/AnalyzerFoot_{i}_{j}",
                    [x, y, z + 0.007],
                    [0.017, 0.025, 0.014],
                    "rubber",
                )
        self.cube(
            "/World/Equipment/DisplayBezel",
            [0.86, -0.214, z + 0.115],
            [0.102, 0.010, 0.053],
            "rubber",
        )
        self.cube(
            "/World/Equipment/DisplayFace",
            [0.86, -0.208, z + 0.115],
            [0.085, 0.002, 0.036],
            "ceramic",
        )
        self.label(
            "/World/Equipment/DisplayReadout",
            "STANDBY",
            [0.86, -0.2065, z + 0.115],
            [0.076, 0.026],
            [0.05, 0.45, 0.45],
        )
        for i, x in enumerate([0.80, 0.92]):
            self.cube(
                f"/World/Equipment/PanelScrew_{i}",
                [x, -0.211, z + 0.040],
                [0.004, 0.002, 0.004],
                "steel",
            )
        self.cube(
            "/World/Equipment/ControlButton",
            [0.912, -0.208, z + 0.075],
            [0.018, 0.008, 0.015],
            "rubber",
        )
        # Vial caps and steel probe handles use the same material palette in all tasks.
        self.label("/World/Room/LabSign", "LAB 01", [0.90, -0.795, 1.15], [0.22, 0.08])
        stage = self.stage
        from pxr import Usd, UsdGeom

        ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
        if ground.IsValid():
            for child in Usd.PrimRange(ground):
                if child.IsA(UsdGeom.Gprim):
                    UsdShade.MaterialBindingAPI.Apply(child).Bind(self.material("floor").material)

    def cylinder_child(self, parent, name, radius, height, z, material, collision=False):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        shape = UsdGeom.Cylinder.Define(self.stage, f"{parent}/{name}")
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(height))
        shape.CreateAxisAttr("Z")
        shape.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(z)))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material.material)
        if collision:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        return shape.GetPrim()
