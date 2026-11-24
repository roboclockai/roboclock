"""User-approved Task41 plating environment; static visuals before validation."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roboclock.envs.rates import PHYSICS_HZ, VIDEO_FPS

from .paths import asset_path


# Loading pad on the drip bridge (top face at 0.933 m).
LOAD_PAD_CENTER = [0.0, 0.43, 0.931]
LOAD_PAD_SIZE = [0.1, 0.1, 0.004]
LOAD_PAD_TOP_Z = LOAD_PAD_CENTER[2] + LOAD_PAD_SIZE[2] / 2


def build_plating_environment(assets, output, resolution=(1600, 1000)):
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    from .appearance import IndustryAppearance
    from .imports import reference_fixture
    world = World(stage_units_in_meters=1, physics_dt=1 / PHYSICS_HZ, rendering_dt=1 / VIDEO_FPS)
    stage = world.stage
    config = {'visuals': {'materials': {'floor': {'texture_set': 'epoxy', 'uv_scale': [9, 8], 'normal_strength': 0.14}, 'steel': {'texture_set': 'brushed_steel', 'uv_scale': [2, 2], 'normal_strength': 0.12}, 'wall': {'texture_set': 'painted_plaster_wall', 'uv_scale': [5, 2], 'normal_strength': 0.04}}}}
    a = IndustryAppearance(stage, world, config)
    for name, color, r, m in [('floor', [0.68, 0.75, 0.73], 0.65, 0), ('frame', [0.11, 0.17, 0.18], 0.45, 0.5), ('rail', [0.92, 0.64, 0.08], 0.38, 0.35), ('wall', [0.65, 0.69, 0.68], 0.85, 0), ('pipe', [0.23, 0.36, 0.44], 0.4, 0.35), ('black', [0.022, 0.029, 0.032], 0.7, 0), ('bath', [0.075, 0.24, 0.2], 0.13, 0.15), ('rinse', [0.13, 0.28, 0.32], 0.09, 0.1), ('screen', [0.018, 0.055, 0.065], 0.3, 0), ('light', [0.8, 0.9, 1], 0.3, 0)]:
        a.material(name, color=color, roughness=r, metallic=m)
    backend = SimpleNamespace(stage=stage, world=world)
    records = []

    def imported(path, file, center, size=None):
        _, r = reference_fixture(backend, path, file, center, size, collision=False)
        records.append(dict(path=path, **r))
        return stage.GetPrimAtPath(path)

    def coat(path, mat):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(a.material(mat).material)

    def cube(name, position, size, material):
        return a.rounded('/World/' + name, position, size, material, radius=min(0.006, min(size) / 5))

    def pipe(name, start, end, radius, material='pipe'):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        c = UsdGeom.Cylinder.Define(stage, '/World/' + name)
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(float(np.linalg.norm(delta)))
        c.AddTranslateOp().Set(Gf.Vec3d(*(start + end) / 2))
        c.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*delta)).GetQuat()))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(a.material(material).material)

    def label(name, text, position, size):
        a.label('/World/Signs/' + name, text, position, size, color=(0.075, 0.13, 0.16))
        directory = output / 'labels'
        directory.mkdir(exist_ok=True)
        width, height = (1024, max(64, round(1024 * size[1] / size[0])))
        image = Image.new('RGB', (width, height), (26, 49, 57))
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 3, width - 4, height - 4), outline=(170, 189, 190), width=3)
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        font_size = round(height * 0.62)
        while True:
            font = ImageFont.truetype(font_path, font_size) if Path(font_path).exists() else ImageFont.load_default(size=font_size)
            if draw.textbbox((0, 0), text, font=font)[2] < width - 50 or font_size <= 12:
                break
            font_size -= 2
        draw.text((width / 2, height / 2), text, font=font, fill=(230, 238, 235), anchor='mm')
        texture = directory / (name + '.png')
        image.save(texture)
        node = UsdShade.Shader(stage.GetPrimAtPath('/World/Signs/' + name + '_material/Texture'))
        node.GetInput('file').Set(Sdf.AssetPath(str(texture)))
        uv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('/World/Signs/' + name)).GetPrimvar('st')
        uv.Set([(1 - u, v) for u, v in uv.Get()])
    cube('Building/Floor', [0, 1, -0.04], [12, 10, 0.08], 'floor')
    cube('Building/BackWall', [0, 4.1, 1.8], [12, 0.13, 3.6], 'wall')
    cube('Building/SideWall', [-4.0, 1.0, 1.8], [0.13, 6.2, 3.6], 'wall')
    cube('Building/Ceiling', [0, 1, 3.65], [12, 10, 0.1], 'wall')
    for x in [-3.5, -1.6, 0.3, 2.2, 4.1]:
        cube('Building/Column' + str(x).replace('.', '_').replace('-', 'N'), [x, 3.97, 1.7], [0.13, 0.16, 3.4], 'frame')
    cube('Building/WallBumper', [0, 3.98, 0.25], [10, 0.08, 0.32], 'frame')
    for i, x in enumerate([-2.0, 2.0]):
        cube(f'Markings/Aisle{i}', [x, 0.2, 0.002], [0.045, 5.1, 0.003], 'rail')
    cube('Markings/RearBoundary', [0, 1.35, 0.002], [3.95, 0.045, 0.003], 'rail')
    for i in range(12):
        cube(f'Drain/Slot{i}', [-0.65 + i * 0.12, 1.15, 0.006], [0.065, 0.21, 0.009], 'black')
    imported('/World/Services/WaterTreatment', assets / 'water_treatment/equipment.usdc', [1.85, 2.8, 1.05])
    imported('/World/Services/Pump', assets / 'pump_module/equipment.usdc', [-1.8, 1.65, 0.525])
    for i, z in enumerate([1.75, 2.05]):
        pipe(f'Services/WallPipe{i}', [-3.2, 3.75, z], [3.8, 3.75, z], 0.045)
        for j, x in enumerate([-2.4, 0, 2.5]):
            cube(f'Services/Bracket{i}_{j}', [x, 3.84, z], [0.045, 0.19, 0.15], 'steel')
    pipe('Services/RinseDrop', [0.65, 3.75, 1.75], [0.65, 0.9, 1.75], 0.021)
    pipe('Services/RinseDown', [0.65, 0.9, 1.75], [0.65, 0.9, 1.11], 0.021)
    pipe('Services/Return', [0.82, 0.9, 0.28], [0.82, 2.9, 0.28], 0.027)
    pipe('Services/DrainDrop', [0.34, 0.67, 0.7], [0.34, 0.67, 0.28], 0.027)
    pipe('Services/DrainAcross', [0.34, 0.67, 0.28], [0.82, 0.67, 0.28], 0.027)
    pipe('Services/DrainElbow', [0.82, 0.67, 0.28], [0.82, 0.9, 0.28], 0.027)
    tray = Path(asset_path('/opt/industry/assets/v2/task46/prepared/tray.usd'))
    for index, (x, mat, text) in enumerate([(-0.34, 'bath', '01 / ZINCATE DIP'), (0.34, 'rinse', '02 / RINSE')]):
        path = f'/World/Station/Bath{index}'
        imported(path, tray, [x, 0.43, 0.8], [0.48, 0.46, 0.26])
        coat(path, 'steel')
        cube(f'Station/Liquid{index}', [x, 0.43, 0.885], [0.433, 0.413, 0.003], mat)
        for leg, y in enumerate([0.235, 0.625]):
            for side, xx in enumerate([x - 0.21, x + 0.21]):
                cube(f'Station/Leg{index}_{leg}_{side}', [xx, y, 0.34], [0.045, 0.045, 0.68], 'frame')
                cube(f'Station/Foot{index}_{leg}_{side}', [xx, y, 0.024], [0.083, 0.083, 0.035], 'black')
        cube(f'Station/FrontBeam{index}', [x, 0.2, 0.59], [0.5, 0.045, 0.075], 'frame')
        cube(f'Station/Panel{index}', [x, 0.191, 0.79], [0.4, 0.014, 0.19], 'frame')
        label('Bath' + str(index), text, [x, 0.181, 0.81], [0.35, 0.065])
        label('Sub' + str(index), 'PROCESS BATH' if index == 0 else 'FRESH WATER', [x, 0.18, 0.745], [0.27, 0.032])
    cube('Station/DripBridge', [0, 0.43, 0.921], [0.16, 0.46, 0.012], 'steel')
    for i in range(7):
        cube(f'Station/DripSlot{i}', [0, 0.27 + i * 0.05, 0.928], [0.105, 0.009, 0.003], 'black')
    # Loading pad: the dry part waits here, between the baths, until the arm dips it. It covers
    # the drip slots (a 60 mm part would rock on one 9 mm slot bar).
    a.material('load_pad', color=[0.86, 0.62, 0.1], roughness=0.8, metallic=0)
    cube('Station/LoadPad', LOAD_PAD_CENTER, LOAD_PAD_SIZE, 'load_pad')
    cube('Extraction/Plenum', [0, 0.86, 1.04], [1.27, 0.17, 0.21], 'steel')
    cube('Extraction/Slot', [0, 0.769, 1.035], [1.14, 0.009, 0.045], 'black')
    pipe('Extraction/Riser', [0, 0.88, 1.13], [0, 0.88, 2.3], 0.085, 'steel')
    pipe('Extraction/BackDuct', [0, 0.88, 2.3], [0, 3.85, 2.3], 0.085, 'steel')
    for i, z in enumerate([1.35, 1.9, 2.21]):
        pipe(f'Extraction/Band{i}', [0, 0.88, z - 0.009], [0, 0.88, z + 0.009], 0.092, 'frame')
    cube('Electrical/Rectifier', [-1.15, 0.49, 0.55], [0.45, 0.47, 1.1], 'frame')
    cube('Electrical/Door', [-1.15, 0.247, 0.56], [0.4, 0.014, 0.99], 'steel')
    cube('Electrical/Display', [-1.15, 0.233, 0.83], [0.29, 0.012, 0.15], 'screen')
    # A zincate dip draws no plating current: the cabinet is the bath's heater/agitation control.
    label('Rectifier', 'BATH CONTROL', [-1.15, 0.223, 0.83], [0.255, 0.052])
    for i in range(6):
        cube(f'Electrical/Vent{i}', [-1.15, 0.232, 0.3 + i * 0.031], [0.29, 0.01, 0.01], 'black')
    pipe('Electrical/Feed', [-0.93, 0.66, 0.77], [-0.58, 0.66, 0.77], 0.013, 'black')
    for i, x in enumerate([-0.38, 0.38]):
        cube(f'Dock/Line{i}', [x, -0.39, 0.002], [0.018, 0.55, 0.003], 'rail')
    cube('Dock/Front', [0, -0.66, 0.002], [0.77, 0.018, 0.003], 'rail')
    label('Cell', 'SURFACE FINISHING / CELL 41', [0, 3.82, 2.89], [2.8, 0.22])
    label('Water', 'WATER TREATMENT', [1.85, 3.82, 2.43], [1.4, 0.12])
    dome = UsdLux.DomeLight.Define(stage, '/World/Lighting/HDR')
    hdr = Path(asset_path('/opt/industry/isaac601-test/robogym/src/robogym/assets/shared/hdr/studio_small_09_1k.hdr'))
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(hdr)))
    dome.CreateIntensityAttr(850)
    for i, (pos, size, intensity) in enumerate([([0, -1, 3.1], [3, 2], 1600), ([-2, 2, 3], [2, 2], 1100), ([2, 2.8, 3], [2, 2], 1300)]):
        light = UsdLux.RectLight.Define(stage, f'/World/Lighting/Area{i}')
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        light.CreateIntensityAttr(intensity * 1.35)
        light.AddTranslateOp().Set(Gf.Vec3d(*pos))
    front = UsdLux.RectLight.Define(stage, '/World/Lighting/FrontFill')
    front.CreateWidthAttr(2.0)
    front.CreateHeightAttr(1.0)
    front.CreateIntensityAttr(1200)
    front.AddTranslateOp().Set(Gf.Vec3d(0, -1.5, 2.4))
    front.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(0, 2.0, -1.5)).GetQuat()))
    camera = world.scene.add(Camera('/World/ReviewCamera', name='environment_review', resolution=resolution))
    return SimpleNamespace(world=world, stage=stage, appearance=a, camera=camera, asset_records=records, assets=assets, output=output)
