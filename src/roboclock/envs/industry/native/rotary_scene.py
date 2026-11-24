"""Task42 factory environment preview, independent of robot/control/physics."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roboclock.envs.rates import PHYSICS_HZ, VIDEO_FPS

from .paths import asset_path, portable_json_loads


def build_rotary_environment(assets, output, resolution=(1600, 1000), sector_deg=(-120.0, -60.0), speed_deg_s=5.0):
    """The static cell. ``sector_deg`` is the scored loading sector (the difficulty level
    narrows it from its entry edge, -60 deg); the signs show its width and window.

    The green 60 deg ring and its posts are colliders (everything under /World/LoadingZone),
    so they never change; a narrower sector covers its closed part with a visual-only mask.
    """
    sector_deg = [float(v) for v in sector_deg]
    width_deg = sector_deg[1] - sector_deg[0]
    window_s = width_deg / speed_deg_s
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
    for name, color, rough, metal in [('floor', [0.68, 0.75, 0.73], 0.65, 0), ('wall', [0.65, 0.69, 0.68], 0.85, 0), ('frame', [0.1, 0.16, 0.18], 0.45, 0.45), ('steel', [0.64, 0.69, 0.71], 0.31, 0.85), ('black', [0.025, 0.032, 0.035], 0.65, 0.1), ('yellow', [0.98, 0.64, 0.055], 0.45, 0.05), ('green', [0.055, 0.55, 0.26], 0.5, 0), ('screen', [0.018, 0.055, 0.065], 0.4, 0), ('white', [0.82, 0.87, 0.87], 0.5, 0.1), ('fixture', [0.12, 0.16, 0.18], 0.55, 0.25), ('cast_metal', [0.53, 0.57, 0.6], 0.42, 0.8)]:
        a.material(name, color=color, roughness=rough, metallic=metal)
    backend = SimpleNamespace(world=world, stage=stage)
    records = []

    def box(name, position, size, material='frame', radius=0.004):
        return a.rounded('/World/' + name, position, size, material, radius=min(radius, min(size) / 5))

    def cylinder(name, position, radius, height, material='steel'):
        shape = UsdGeom.Cylinder.Define(stage, '/World/' + name)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(a.material(material).material)
        return shape

    def import_asset(name, file, center, size=None, instance=False):
        body, record = reference_fixture(backend, '/World/' + name, file, center, size, collision=False)
        if name.startswith('Products/'):
            UsdShade.MaterialBindingAPI.Apply(body.prim).Bind(a.material('cast_metal').material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            record['scene_material_adaptation'] = 'Declared cast-aluminium PBR; source geometry/UVs retained, not scanned material calibration'
        if instance:
            stage.GetPrimAtPath(body.prim_path + '/Offset/Asset').SetInstanceable(True)
        records.append(dict(path=body.prim_path, **record))
        return body

    def sign(name, text, position, size):
        a.label('/World/Signs/' + name, text, position, size, color=(0.075, 0.13, 0.16))
        folder = output / 'labels'
        folder.mkdir(exist_ok=True)
        width = 1536
        height = max(96, round(width * size[1] / size[0]))
        im = Image.new('RGB', (width, height), (23, 43, 49))
        draw = ImageDraw.Draw(im)
        draw.rectangle((3, 3, width - 4, height - 4), outline=(160, 190, 191), width=3)
        font_size = round(height * 0.62)
        while True:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
            bounds = draw.textbbox((0, 0), text, font=font)
            if bounds[2] - bounds[0] < width * 0.94:
                break
            font_size -= 1
        draw.text(((width - bounds[2] + bounds[0]) / 2, (height - bounds[3] + bounds[1]) / 2 - bounds[1]), text, font=font, fill=(225, 237, 237))
        path = folder / (name + '.png')
        im.save(path)
        shader = UsdShade.Shader(stage.GetPrimAtPath('/World/Signs/' + name + '_material/Texture'))
        shader.GetInput('file').Set(Sdf.AssetPath(str(path)))
        uv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('/World/Signs/' + name)).GetPrimvar('st')
        uv.Set([(1 - u, v) for u, v in uv.Get()])
    box('Building/Floor', [0, 1, -0.04], [12, 10, 0.08], 'floor')
    box('Building/BackWall', [0, 3.9, 1.8], [12, 0.12, 3.6], 'wall')
    box('Building/SideWall', [-3.7, 1, 1.8], [0.12, 6, 3.6], 'wall')
    for i, x in enumerate([-3.4, -1.4, 0.6, 2.6, 4.6]):
        box(f'Building/Column{i}', [x, 3.8, 1.75], [0.14, 0.16, 3.5])
    box('Building/Bumper', [0, 3.78, 0.25], [10, 0.08, 0.3])
    for i, x in enumerate([-1.65, 1.45]):
        box(f'Markings/Lane{i}', [x, 0.05, 0.003], [0.045, 3.2, 0.004], 'yellow')
    box('Markings/CellFront', [-0.1, -0.8, 0.003], [3.1, 0.045, 0.004], 'yellow')
    box('Markings/CellRear', [-0.1, 1.45, 0.003], [3.1, 0.045, 0.004], 'yellow')
    box('Machine/Plinth', [0, 0.64, 0.41], [1.18, 1.22, 0.66], 'white', 0.015)
    box('Machine/ToeKick', [0, 0.64, 0.1], [1.08, 1.12, 0.13], 'frame')
    for i, x in enumerate([-0.29, 0.29]):
        box(f'Machine/ServiceDoor{i}', [x, 0.023, 0.43], [0.55, 0.018, 0.5], 'white')
        box(f'Machine/DoorHandle{i}', [x + 0.18, 0.006, 0.49], [0.022, 0.025, 0.14], 'frame')
    for i, x in enumerate([-0.48, 0.48]):
        for j, y in enumerate([0.14, 1.12]):
            cylinder(f'Machine/Foot{i}{j}', [x, y, 0.045], 0.038, 0.09, 'black')
    for i, x in enumerate([-0.53, 0.53]):
        box(f'Machine/DeckSide{i}', [x, 0.64, 0.815], [0.12, 1.22, 0.035], 'steel')
    box('Machine/DeckRear', [0, 1.13, 0.815], [0.95, 0.24, 0.035], 'steel')
    box('Machine/DeckFront', [0, 0.065, 0.815], [0.95, 0.07, 0.035], 'steel')
    box('Process/RearPanel', [0, 1.24, 1.16], [1.18, 0.045, 0.7], 'white')
    for i, x in enumerate([-0.565, 0.565]):
        box(f'Process/SidePanel{i}', [x, 0.94, 1.16], [0.045, 0.58, 0.7], 'white')
        box(f'Process/PortalPost{i}', [x, 0.64, 1.16], [0.055, 0.055, 0.7], 'frame')
    box('Process/Roof', [0, 0.94, 1.535], [1.2, 0.66, 0.06], 'white')
    box('Process/PortalHeader', [0, 0.63, 1.46], [1.19, 0.075, 0.15], 'frame')
    box('Process/InternalRail', [0, 1.18, 1.27], [0.86, 0.055, 0.08], 'steel')
    box('Process/CameraArm', [0, 1.01, 1.28], [0.055, 0.34, 0.06], 'frame')
    box('Process/InspectionCamera', [0, 0.83, 1.22], [0.1, 0.09, 0.08], 'frame')
    cylinder('Process/Lens', [0, 0.83, 1.165], 0.028, 0.035, 'black')
    cylinder('Process/RingLight', [0, 0.83, 1.142], 0.04, 0.012, 'white')
    sign('Process', 'PUMP HOUSING / VISION INSPECTION', [0, 0.588, 1.46], [0.98, 0.065])
    conveyor_path = Path(asset_path('/opt/industry/assets/v2/task42/conveyor/conveyor-local.usd'))
    for name, center in [('Infeed', [-1.115, 0.25, 0.815]), ('Outfeed', [1.115, 0.55, 0.855])]:
        conveyor, record = reference_fixture(backend, f'/World/Flow/{name}', conveyor_path, center, [1.05, 0.22, 0.06], collision=False)
        for prim in Usd.PrimRange(conveyor.prim):
            if prim.IsA(UsdGeom.Gprim):
                rubber = '/Belt/' in str(prim.GetPath()) or 'Rubberband' in prim.GetName()
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(a.material('black' if rubber else 'steel').material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            if prim.GetTypeName() == 'OmniGraph':
                prim.SetActive(False)
        records.append(dict(path=conveyor.prim_path, **record, adaptation='Existing A04 geometry/UVs; resized compact line, steel frame/dark belt; static only'))
        for i, x in enumerate([center[0] - 0.4, center[0] + 0.4]):
            box(f'Flow/{name}Leg{i}', [x, center[1], center[2] / 2 - 0.015], [0.055, 0.18, center[2] - 0.03], 'frame')
            box(f'Flow/{name}Foot{i}', [x, center[1], 0.018], [0.16, 0.3, 0.035], 'frame')
        for i, y in enumerate([center[1] - 0.12, center[1] + 0.12]):
            box(f'Flow/{name}Guide{i}', [center[0], y, center[2] + 0.065], [1.05, 0.018, 0.035], 'steel')
    for i, x in enumerate([0.38, 0.78]):
        box(f'Unload/Post{i}', [x, 0.69, 1.04], [0.035, 0.035, 0.39], 'frame')
    box('Unload/LinearSlide', [0.58, 0.69, 1.25], [0.48, 0.07, 0.055], 'steel')
    box('Unload/Carriage', [0.7, 0.69, 1.22], [0.09, 0.1, 0.07], 'frame')
    box('Unload/VerticalAxis', [0.7, 0.69, 1.11], [0.04, 0.045, 0.18], 'steel')
    box('Unload/GripperBody', [0.7, 0.58, 1.02], [0.075, 0.24, 0.045], 'frame')
    for i, x in enumerate([0.655, 0.745]):
        box(f'Unload/OpenJaw{i}', [x, 0.55, 0.97], [0.015, 0.04, 0.08], 'steel')
    cylinder('Turntable/DriveHousing', [0, 0.55, 0.76], 0.205, 0.12, 'frame')
    cylinder('Turntable/DriveRim', [0, 0.55, 0.818], 0.24, 0.018, 'black')
    n = 128
    points = [[0.42 * np.cos(2 * np.pi * i / n), 0.55 + 0.42 * np.sin(2 * np.pi * i / n), z] for z in (0.828, 0.856) for i in range(n)]
    points += [[0, 0.55, 0.828], [0, 0.55, 0.856]]
    faces = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    faces += [[2 * n, (i + 1) % n, i] for i in range(n)]
    faces += [[2 * n + 1, n + i, n + (i + 1) % n] for i in range(n)]
    disc = UsdGeom.Mesh.Define(stage, '/World/Turntable/Disc')
    disc.CreatePointsAttr(points)
    disc.CreateFaceVertexCountsAttr([len(f) for f in faces])
    disc.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    disc.CreateSubdivisionSchemeAttr('none')
    UsdShade.MaterialBindingAPI.Apply(disc.GetPrim()).Bind(a.material('steel').material)
    vertices = []
    for t in np.linspace(0, 2 * np.pi, 97):
        for z, r in [(0.83, 0.429), (0.83, 0.465), (0.856, 0.465), (0.856, 0.429)]:
            vertices.append([float(r * np.cos(t)), float(0.55 + r * np.sin(t)), z])
    rail = UsdGeom.Mesh.Define(stage, '/World/LoadingZone/FixedRail')
    rail.CreatePointsAttr(vertices)
    rail.CreateFaceVertexCountsAttr([4] * (96 * 4))
    rail.CreateFaceVertexIndicesAttr([v for i in range(96) for j in range(4) for v in [4 * i + j, 4 * (i + 1) + j, 4 * (i + 1) + (j + 1) % 4, 4 * i + (j + 1) % 4]])
    rail.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(rail.GetPrim()).Bind(a.material('frame').material)
    for i, angle in enumerate([0, 90, 180, 270]):
        t = np.deg2rad(angle)
        cylinder(f'LoadingZone/RailSupport{i}', [0.45 * np.cos(t), 0.55 + 0.45 * np.sin(t), 0.765], 0.012, 0.13, 'steel')
    cylinder('Turntable/Hub', [0, 0.55, 0.862], 0.065, 0.012, 'frame')
    for i in range(8):
        theta = i * np.pi / 4
        cylinder(f'Turntable/HubBolt{i}', [0.047 * np.cos(theta), 0.55 + 0.047 * np.sin(theta), 0.87], 0.004, 0.006)
    carrier_records = []
    pump_dimensions = portable_json_loads((assets / 'pump/manifest.json').read_text())['dimensions_m']
    for i, angle in enumerate([-90, 0, 90, 180]):
        theta = np.deg2rad(angle)
        cx = 0.28 * np.cos(theta)
        cy = 0.55 + 0.28 * np.sin(theta)
        box(f'Carriers/C{i}/Plate', [cx, cy, 0.868], [0.16, 0.16, 0.014], 'fixture')
        if i == 0:
            for j, (offset, size) in enumerate([([0, -0.082, 0], [0.174, 0.008, 0.006]), ([0, 0.082, 0], [0.174, 0.008, 0.006]), ([-0.082, 0, 0], [0.008, 0.164, 0.006]), ([0.082, 0, 0], [0.008, 0.164, 0.006])]):
                box(f'Carriers/C{i}/TargetBorder{j}', np.array([cx, cy, 0.88]) + offset, size, 'yellow', 0.001)
        for j, (dx, dy) in enumerate([[-0.036, -0.033], [0.036, -0.033], [0, 0.036]]):
            cylinder(f'Carriers/C{i}/Support{j}', [cx + dx, cy + dy, 0.88], 0.009, 0.01, 'steel')
        for j, dx in enumerate([-0.068, 0.068]):
            cylinder(f'Carriers/C{i}/Locator{j}', [cx + dx, cy + 0.045, 0.886], 0.004, 0.022, 'steel')
        loaded = i in (1, 2)
        if loaded:
            import_asset(f'Products/Loaded{i}', assets / 'pump/pump_casing.usd', [cx, cy, 0.886 + pump_dimensions[2] / 2], instance=True)
        carrier_records.append(dict(id='A' + str(i + 1), angle_deg=angle, center_m=[float(cx), float(cy), 0.868], loaded=loaded, designated=i == 0))
    r0, r1 = (0.432, 0.457)
    theta = np.linspace(np.deg2rad(-120), np.deg2rad(-60), 49)
    points = [[float(r * np.cos(t)), float(0.55 + r * np.sin(t)), 0.858] for t in theta for r in (r0, r1)]
    ring = UsdGeom.Mesh.Define(stage, '/World/LoadingZone/FixedSector')
    ring.CreatePointsAttr(points)
    ring.CreateFaceVertexCountsAttr([4] * 48)
    ring.CreateFaceVertexIndicesAttr([v for i in range(48) for v in [2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2]])
    ring.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(ring.GetPrim()).Bind(a.material('green').material)
    for i, deg in enumerate([-120, -90, -60]):
        t = np.deg2rad(deg)
        cylinder(f'LoadingZone/Boundary{i}', [0.447 * np.cos(t), 0.55 + 0.447 * np.sin(t), 0.865], 0.006, 0.008, 'white')
    if sector_deg[1] < -60 - 1e-6:
        # Visual only (outside /World/LoadingZone, so no collider): the closed part
        # [sector entry, -60] in black and a white post at the level's entry edge.
        theta = np.linspace(np.deg2rad(sector_deg[1]), np.deg2rad(-60), 25)
        points = [[float(r * np.cos(t)), float(0.55 + r * np.sin(t)), 0.8585] for t in theta for r in (r0, r1)]
        mask = UsdGeom.Mesh.Define(stage, '/World/SectorMask/Closed')
        mask.CreatePointsAttr(points)
        mask.CreateFaceVertexCountsAttr([4] * 24)
        mask.CreateFaceVertexIndicesAttr([v for i in range(24) for v in [2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2]])
        mask.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mask.GetPrim()).Bind(a.material('black').material)
        t = np.deg2rad(sector_deg[1])
        cylinder('SectorMask/Edge', [0.447 * np.cos(t), 0.55 + 0.447 * np.sin(t), 0.865], 0.006, 0.008, 'white')
    for i, x in enumerate([-1.52, -1.28, -1.04, -0.8]):
        y = 0.25
        box(f'Feed/Support{i}', [x, y, 0.851], [0.15, 0.15, 0.014], 'fixture')
        import_asset(f'Products/WIP{i}', assets / 'pump/pump_casing.usd', [x, y, 0.86 + pump_dimensions[2] / 2], instance=True)
    for i, x in enumerate([0.94, 1.2, 1.46]):
        box(f'Output/Support{i}', [x, 0.55, 0.891], [0.15, 0.15, 0.014], 'fixture')
        import_asset(f'Products/Output{i}', assets / 'pump/pump_casing.usd', [x, 0.55, 0.9 + pump_dimensions[2] / 2], instance=True)
    box('Control/HMIMount', [0.45, 0.54, 1.31], [0.21, 0.08, 0.18], 'frame')
    sign('Recipe', f'LOAD / {window_s:g} s', [0.45, 0.495, 1.32], [0.19, 0.08])
    cylinder('Control/BeaconPole', [0.46, 1.08, 1.63], 0.012, 0.13, 'steel')
    cylinder('Control/Beacon', [0.46, 1.08, 1.73], 0.028, 0.065, 'green')
    sign('Cell', 'PUMP HOUSING / ROTARY INSPECTION CELL', [0, 3.72, 2.6], [3.65, 0.22])
    sign('Supply', '01  INFEED', [-1.12, 0.107, 0.75], [0.58, 0.075])
    sign('Loading', f'02  LOAD / {window_s:g} s', [0, 0.009, 0.69], [0.66, 0.075])
    sign('Table', f'{width_deg:g} deg WINDOW / {speed_deg_s:g} deg per s', [0, 0.009, 0.59], [0.85, 0.055])
    sign('Output', '04  INSPECTED / OUTFEED', [1.12, 0.407, 0.79], [0.94, 0.075])
    dome = UsdLux.DomeLight.Define(stage, '/World/Lighting/HDR')
    hdr = Path(asset_path('/opt/industry/isaac601-test/robogym/src/robogym/assets/shared/hdr/studio_small_09_1k.hdr'))
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(hdr)))
    dome.CreateIntensityAttr(850)
    for i, (position, size, intensity) in enumerate([([0, -1, 3.1], [3, 2], 2200), ([-2, 2, 3.0], [2, 2], 1500), ([2, 2.8, 3], [2, 2], 1600)]):
        light = UsdLux.RectLight.Define(stage, f'/World/Lighting/Area{i}')
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        light.CreateIntensityAttr(intensity)
        light.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    camera = world.scene.add(Camera('/World/ReviewCamera', name='rotary_environment_review', resolution=resolution))
    return SimpleNamespace(world=world, stage=stage, camera=camera, appearance=a, asset_records=records, carriers=carrier_records, metadata=dict(task_id=42, scope='Static environment review only; no robot, rotation dynamics, contact or task validation', temporal_category='A', loading_sector_deg=sector_deg, rotation_deg_s=speed_deg_s, window_s=window_s, revision='v4 integrated rotary inspection cell; v3 detached-table layout rejected by reviewer', workflow='Infeed -> robot loading sector -> rear optical inspection -> proposed linear unload -> outfeed', workflow_status='Static staging only; inspection and unloading are proposed visual context, not simulated or scored', authored_fixtures='Shared machine plinth/deck/enclosure, inspection head, proposed unload slide, turntable/carriers/fixed sector, conveyor supports and HMI', fixture_status='Engineering preview dimensions; no fit/support/contact validation', materials='Pump cast-aluminium override; original geometry/UVs; local steel/dark conveyor finish; existing epoxy/brushed-steel/plaster textures and CC0 studio_small_09 HDR'))
