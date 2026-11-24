"""Static factory preview tools; no robot, process or contact-success logic."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roboclock.envs.rates import PHYSICS_HZ, VIDEO_FPS

from .paths import asset_path


class IndustrialCell:

    def __init__(self, output, resolution=(1600, 1000)):
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Sdf, UsdLux

        from .appearance import IndustryAppearance
        self.output = Path(output)
        self.world = World(stage_units_in_meters=1, physics_dt=1 / PHYSICS_HZ, rendering_dt=1 / VIDEO_FPS)
        self.stage = self.world.stage
        self.appearance = IndustryAppearance(self.stage, self.world, {'visuals': {'materials': {'floor': {'texture_set': 'epoxy', 'uv_scale': [9, 8], 'normal_strength': 0.14}, 'steel': {'texture_set': 'brushed_steel', 'uv_scale': [2, 2], 'normal_strength': 0.12}, 'wall': {'texture_set': 'painted_plaster_wall', 'uv_scale': [5, 2], 'normal_strength': 0.04}}}})
        for name, color, rough, metal in [('floor', [0.68, 0.75, 0.73], 0.65, 0), ('wall', [0.65, 0.69, 0.68], 0.85, 0), ('frame', [0.1, 0.16, 0.18], 0.45, 0.45), ('steel', [0.64, 0.69, 0.71], 0.31, 0.85), ('black', [0.025, 0.032, 0.035], 0.65, 0.1), ('yellow', [0.98, 0.64, 0.055], 0.45, 0.05), ('green', [0.055, 0.55, 0.26], 0.5, 0), ('red', [0.78, 0.035, 0.025], 0.45, 0.05), ('white', [0.82, 0.87, 0.87], 0.5, 0.1), ('glue', [0.76, 0.53, 0.16], 0.26, 0), ('brass', [0.48, 0.31, 0.1], 0.35, 0.8), ('water', [0.12, 0.45, 0.62], 0.22, 0.05)]:
            self.appearance.material(name, color=color, roughness=rough, metallic=metal)
        self.assets = []
        self.box('Building/Floor', [0, 1, -0.04], [12, 10, 0.08], 'floor')
        self.box('Building/BackWall', [0, 3.9, 1.8], [12, 0.12, 3.6], 'wall')
        self.box('Building/SideWall', [-3.7, 1, 1.8], [0.12, 6, 3.6], 'wall')
        for i, x in enumerate([-3.4, -1.4, 0.6, 2.6, 4.6]):
            self.box(f'Building/Column{i}', [x, 3.8, 1.75], [0.14, 0.16, 3.5], 'frame')
        self.box('Building/Bumper', [0, 3.78, 0.25], [10, 0.08, 0.3], 'frame')
        for i, x in enumerate([-1.75, 1.75]):
            self.box(f'Floor/Boundary{i}', [x, 0.65, 0.003], [0.045, 3, 0.004], 'yellow')
        self.box('Floor/FrontBoundary', [0, -0.8, 0.003], [3.5, 0.045, 0.004], 'yellow')
        dome = UsdLux.DomeLight.Define(self.stage, '/World/Lighting/HDR')
        dome.CreateTextureFileAttr(Sdf.AssetPath(asset_path('/opt/industry/isaac601-test/robogym/src/robogym/assets/shared/hdr/studio_small_09_1k.hdr')))
        dome.CreateIntensityAttr(850)
        for i, (pos, shape, intensity) in enumerate([([0, -1, 3.1], [3, 2], 2200), ([-2, 2, 3], [2, 2], 1500), ([2, 2.8, 3], [2, 2], 1600)]):
            light = UsdLux.RectLight.Define(self.stage, f'/World/Lighting/Area{i}')
            light.CreateWidthAttr(shape[0])
            light.CreateHeightAttr(shape[1])
            light.CreateIntensityAttr(intensity)
            light.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        self.camera = self.world.scene.add(Camera('/World/ReviewCamera', name='factory_review', resolution=resolution))

    def box(self, name, position, size, material='frame', radius=0.004):
        return self.appearance.rounded('/World/' + name, position, size, material, radius=min(radius, min(size) / 5))

    def pipe(self, name, start, end, radius=0.02, material='steel', inner_radius=0):
        """Declared engineering pipe/fixture,48-sided; not a new task product."""
        from pxr import UsdGeom, UsdShade
        start, end = (np.asarray(start, float), np.asarray(end, float))
        direction = end - start
        length = np.linalg.norm(direction)
        z = direction / length
        x = np.cross(z, [0, 0, 1])
        if np.linalg.norm(x) < 1e-05:
            x = np.cross(z, [0, 1, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        assert np.linalg.det(R) > 0.99
        n = 48
        points = [((start + end) / 2 + R @ [radius * np.cos(2 * np.pi * i / n), radius * np.sin(2 * np.pi * i / n), h]).tolist() for h in [-length / 2, length / 2] for i in range(n)]
        faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        if inner_radius:
            assert 0 < inner_radius < radius
            points.extend([((start + end) / 2 + R @ [inner_radius * np.cos(2 * np.pi * i / n), inner_radius * np.sin(2 * np.pi * i / n), h]).tolist() for h in [-length / 2, length / 2] for i in range(n)])
            faces += [[2 * n + i, 3 * n + i, 3 * n + (i + 1) % n, 2 * n + (i + 1) % n] for i in range(n)]
            faces += [[i, 2 * n + i, 2 * n + (i + 1) % n, (i + 1) % n] for i in range(n)]
            faces += [[n + i, n + (i + 1) % n, 3 * n + (i + 1) % n, 3 * n + i] for i in range(n)]
        else:
            faces += [list(reversed(range(n))), list(range(n, 2 * n))]
        mesh = UsdGeom.Mesh.Define(self.stage, '/World/' + name)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.appearance.material(material).material)
        return mesh

    def elbow(self, name, center, bend_radius, start_deg, end_deg, radius=0.027, inner_radius=0.022):
        from pxr import UsdGeom, UsdShade
        n, count = (48, 25)
        points = []
        for r in [radius, inner_radius]:
            for t in np.deg2rad(np.linspace(start_deg, end_deg, count)):
                radial = np.array([np.cos(t), np.sin(t), 0.0])
                base = np.asarray(center) + bend_radius * radial
                points.extend(((base + r * (np.cos(phi) * radial + np.array([0, 0, np.sin(phi)]))).tolist() for phi in np.linspace(0, 2 * np.pi, n, endpoint=False)))
        offset = count * n
        faces = []
        for i in range(count - 1):
            for j in range(n):
                face = [i * n + j, (i + 1) * n + j, (i + 1) * n + (j + 1) % n, i * n + (j + 1) % n]
                faces.append(face)
                faces.append([v + offset for v in reversed(face)])
        for j in range(n):
            faces.append([j, (j + 1) % n, offset + (j + 1) % n, offset + j])
            last = (count - 1) * n
            faces.append([last + j, offset + last + j, offset + last + (j + 1) % n, last + (j + 1) % n])
        mesh = UsdGeom.Mesh.Define(self.stage, '/World/' + name)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
        mesh.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.appearance.material('steel').material)
        return mesh

    def asset(self, name, file, center, size=None, material=None, yaw=0):
        import hashlib

        from pxr import Sdf, Usd, UsdGeom, UsdShade

        from .imports import reference_fixture
        file = Path(file)
        if not file.is_file():
            raise FileNotFoundError(file)
        q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        body, r = reference_fixture(SimpleNamespace(stage=self.stage, world=self.world), '/World/' + name, file, np.asarray(center, float), size, orientation=q, collision=False)
        if material:
            for prim in Usd.PrimRange(body.prim):
                if prim.IsA(UsdGeom.Gprim):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.appearance.material(material).material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                if material == 'steel' and prim.IsA(UsdGeom.Mesh):
                    uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
                    if not uv or not uv.HasValue():
                        mesh = UsdGeom.Mesh(prim)
                        points = np.asarray(mesh.GetPointsAttr().Get())
                        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                        counts = mesh.GetFaceVertexCountsAttr().Get()
                        if np.all(np.asarray(counts) == 3):
                            polygons = points[indices.reshape(-1, 3)]
                            normals = np.cross(polygons[:, 1] - polygons[:, 0], polygons[:, 2] - polygons[:, 0])
                            major = np.argmax(np.abs(normals), axis=1)
                            axes = np.array([[1, 2], [0, 2], [0, 1]])[major]
                            coordinates = (np.take_along_axis(polygons, axes[:, None, :], axis=2).reshape(-1, 2) / 0.15).tolist()
                        else:
                            coordinates = []
                            cursor = 0
                            for count in counts:
                                polygon = points[indices[cursor:cursor + count]]
                                normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
                                axes = [axis for axis in range(3) if axis != int(np.argmax(np.abs(normal)))]
                                coordinates.extend((polygon[:, axes] / 0.15).tolist())
                                cursor += count
                        UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(coordinates)
        self.assets.append(dict(path=body.prim_path, **r, usd_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), native_status='Static reference import; collision disabled; review render pending', material_override=material, uv_adaptation='Original UVs retained when present; missing CAD steel UVs get face-planar0.15m mapping, shader scale2' if material == 'steel' else None))
        return body

    def sign(self, name, text, position, size, *, background_rgb=None):
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Sdf, UsdGeom, UsdShade
        path = '/World/Signs/' + name
        color = (0.075, 0.13, 0.16) if background_rgb is None else tuple((v / 255 for v in background_rgb))
        self.appearance.label(path, text, position, size, color=color)
        folder = self.output / 'labels'
        folder.mkdir(exist_ok=True)
        width = 1536
        height = max(96, round(width * size[1] / size[0]))
        im = Image.new('RGB', (width, height), background_rgb or (23, 43, 49))
        draw = ImageDraw.Draw(im)
        draw.rectangle((3, 3, width - 4, height - 4), outline=(160, 190, 191), width=3)
        fontsize = round(height * 0.62)
        while True:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', fontsize)
            b = draw.textbbox((0, 0), text, font=font)
            if b[2] - b[0] < width * 0.94:
                break
            fontsize -= 1
        draw.text(((width - b[2] + b[0]) / 2, (height - b[3] + b[1]) / 2 - b[1]), text, font=font, fill=(225, 237, 237))
        file = folder / (name + '.png')
        im.save(file)
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(path + '_material/Texture'))
        shader.GetInput('file').Set(Sdf.AssetPath(str(file)))
        uv = UsdGeom.PrimvarsAPI(self.stage.GetPrimAtPath(path)).GetPrimvar('st')
        uv.Set([(1 - u, v) for u, v in uv.Get()])
