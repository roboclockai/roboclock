"""Final 45–50 static factory proposals; products are existing imported assets.

No robot, active process, mobile base or task-success validation is implied.
Engineering nests, guards, signs and camera housings are declared fixtures.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .industrial_workcell import IndustrialCell
from .paths import asset_path, portable_json_loads

ROOT = Path(asset_path('/opt/industry/assets/v2'))
SHARED = ROOT / 'shared'
RACK = SHARED / 'nvidia-industrial-10012-v1/Assets/ArchVis/Industrial/Shelves/RackSmallEmpty_A1.usd'
BOX = SHARED / 'nvidia-industrial-10012-v1/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_A1.usd'
TABLE = SHARED / 'factory-v1/industrial_table.usd'
TRAY = ROOT / 'task46/prepared/tray.usd'

def build_remaining_environment(task, output, resolution=(1600, 1000)):
    c = IndustrialCell(output, resolution)
    category = 'B' if task <= 46 else 'C'
    requirements = {45: 'Trigger stop button within3s after motor-test warning', 46: 'Arrest bearing inside safety zone within3s of entry; then recover to tray', 47: 'Pack bearing, bracket and pin; deliver at least2 complete kits in90s', 48: 'Correctly shelve at least3 automotive-parts boxes in90s', 49: 'Inspect3 sides and sort at least3 metal parts in90s', 50: 'Assemble and deliver at least2 four-color palettes in90s'}
    titles = {45: 'MOTOR END-OF-LINE / EMERGENCY STOP', 46: 'BEARING RETURN / SAFETY RECOVERY', 47: 'MAINTENANCE KITTING / LINE-SIDE DELIVERY', 48: 'AUTOMOTIVE PARTS / INBOUND PUTAWAY', 49: 'MACHINED PARTS / THREE-SIDE INSPECTION', 50: 'COSMETICS / FOUR-COLOR PALETTE ASSEMBLY'}
    c.sign('Cell', f'CELL {task} / ' + titles[task], [0, 3.72, 2.6], [4.5, 0.23])

    def table(name, x=0, y=0.55, width=1.65, depth=0.85):
        c.asset(name, TABLE, [x, y, 0.43], [width, depth, 0.86])
        c.box(name + 'Top', [x, y, 0.875], [width, depth, 0.025], 'steel')

    def tray(name, xy, size=(0.28, 0.22, 0.04), label=None):
        c.asset(name, TRAY, [*xy, 0.8875 + size[2] / 2], list(size), material='steel')
        if label:
            c.sign(name.replace('/', '_') + 'Label', label, [xy[0], xy[1] - size[1] / 2 - 0.008, 0.87], [size[0], 0.045])

    def rack(name, x, y):
        c.asset(name, RACK, [x, y, 1.05], [1.1, 0.55, 2.1])

    def cabinet(name, x, y):
        c.box(name, [x, y, 0.65], [0.56, 0.48, 1.3], 'white')
        c.box(name + 'Door', [x, y - 0.247, 0.68], [0.49, 0.015, 1.12], 'steel')
        c.pipe(name + 'Handle', [x + 0.18, y - 0.27, 0.6], [x + 0.18, y - 0.27, 0.8], 0.009, 'black')
    if task == 45:
        table('TestBench')
        rack('InstrumentRack', 1.6, 2.2)
        cabinet('DriveCabinet', -0.95, 1.6)
        manifest = portable_json_loads((ROOT / 'task45/asset-manifest.json').read_text())
        bounds = np.asarray(manifest['prepared']['housing']['bounds_m'])
        offset = np.array([0, 0.62, 0.94 - bounds[0, 2]])
        c.box('Rig/MotorMount', [0, 0.62, 0.915], [0.42, 0.34, 0.055], 'frame')
        for name in ['housing', 'rotor']:
            b = np.asarray(manifest['prepared'][name]['bounds_m'])
            c.asset('Rig/' + name, ROOT / 'task45/prepared-v4' / (name + '.usd'), offset + b.mean(0) * [-1, -1, 1], yaw=np.pi)
        from pxr import Usd, UsdGeom
        for name in ['housing', 'cap']:
            file = ROOT / 'task45/button-prepared-v2' / (name + '.usd')
            st = Usd.Stage.Open(str(file))
            box = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(st.GetDefaultPrim()).ComputeAlignedBox()
            mid = (np.asarray(box.GetMin()) + np.asarray(box.GetMax())) / 2
            c.asset('Stop/Stop_' + name, file, np.array([-0.48, 0.3, 1.04]) + [-mid[1], mid[0], mid[2]], yaw=np.pi / 2)
        c.box('Stop/Mount', [-0.48, 0.437, 1.04], [0.2, 0.08, 0.16], 'yellow')
        c.sign('StopText', 'EMERGENCY STOP <=3 s', [-0.48, 0.273, 0.91], [0.42, 0.055])
        # The HMI (and the fault panel MinimalContext puts on it) stands behind the stop
        # button so the rig head camera sees it; recorded at x=0.55, outside that view.
        # robodojo_rig head view (60 deg down): the HMI sits on the bench top behind the
        # stop button, its fault panel centred at 0.985 m (was 1.19 m, above the head image).
        c.box('HMI', [-0.55, 0.80, 1.03], [0.42, 0.1, 0.29], 'frame')
        c.sign('HMI', 'MOTOR TEST / NORMAL', [-0.55, 0.743, 0.98], [0.39, 0.17])
        from pxr import Gf, UsdGeom, UsdShade
        for i, (p, size) in enumerate([([-0.26, 0.62, 1.13], [0.004, 0.32, 0.35]), ([0, 0.46, 1.13], [0.52, 0.004, 0.35]), ([0, 0.62, 1.31], [0.52, 0.32, 0.004])]):
            pane = UsdGeom.Cube.Define(c.stage, f'/World/Guard/Pane{i}')
            pane.CreateSizeAttr(1)
            pane.AddTranslateOp().Set(Gf.Vec3d(*p))
            pane.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdShade.MaterialBindingAPI.Apply(pane.GetPrim()).Bind(c.appearance.glass().material)
        views = [('01_test_bay', [2.3, -2.8, 2.25], [0, 0.9, 1.03]), ('02_motor_and_stop', [-1.3, -1.3, 1.8], [0, 0.55, 1.05]), ('03_control_station', [-0.95, -0.7, 1.5], [-0.13, 0.43, 1.03])]
    elif task == 46:
        table('RecoveryBench')
        rack('BearingStock', 1.7, 2.2)
        c.asset('Return/Chute', ROOT / 'task46/prepared/return_surface.usd', [-0.1, 0.64, 1.01], [1.1, 0.21, 0.001], material='steel')
        c.box('Return/SupportPlate', [-0.1, 0.64, 0.996], [1.1, 0.21, 0.026], 'frame')
        for j, x in enumerate([-0.54, 0.35]):
            c.box(f'Return/Foot{j}', [x, 0.64, 0.935], [0.055, 0.16, 0.095], 'frame')
        for i, x in enumerate([-0.37, -0.17, 0.03]):
            c.asset(f'Queue/Bearing{i}', ROOT / 'task46/prepared/bearing.usd', [x, 0.64, 1.045])
        c.box('Safety/Zone', [0.32, 0.64, 1.0105], [0.24, 0.22, 0.001], 'yellow')
        c.box('Safety/Edge', [0.48, 0.64, 1.01], [0.025, 0.25, 0.012], 'red')
        tray('Recovery/Tray', [0.27, 0.25], label='RECOVERED BEARINGS')
        c.sign('SafetySign', 'ARREST BEFORE EDGE / <=3 s', [0.1, 0.485, 0.9], [0.72, 0.07])
        c.sign('ChuteSign', 'GRINDING LINE / RETURN', [-0.31, 0.87, 1.24], [0.58, 0.09])
        views = [('01_recovery_bay', [2, -2.5, 2.1], [0, 0.9, 1]), ('02_chute_and_tray', [0.9, -0.9, 1.85], [0, 0.53, 1]), ('03_safety_zone', [0.8, -0.35, 1.5], [0.18, 0.57, 1.01])]
    elif task == 47:
        table('KittingBench', 0, 0.65, 1.75, 0.9)
        rack('ServiceInventory', -1.9, 2.1)
        products = [('bearing', ROOT / 'task46/prepared/bearing.usd'), ('bracket', ROOT / 'task47/prepared/bracket.usda'), ('pin', ROOT / 'task47/prepared/pin.usda')]
        for i, (name, path) in enumerate(products):
            x = -0.56 + i * 0.54
            tray('Feed/' + name, [x, 0.82], (0.45, 0.28, 0.035), name.upper())
            for j in range(3):
                h = {'bearing': 0.07, 'bracket': 0.05, 'pin': 0.016}[name]
                c.asset(f'Parts/{name}{j}', path, [x + (j - 1) * 0.11, 0.82, 0.89785 + h / 2], material='steel')
        tray('Kit/Active', [0, 0.32], (0.38, 0.25, 0.035), 'KIT: BEARING + BRACKET + PIN')
        for i, x in enumerate([-0.5, 0.5]):
            tray(f'Kit/Empty{i}', [x, 0.31], (0.34, 0.24, 0.035), 'EMPTY KIT')
        table('DeliveryStation', 2.25, 1.9, 0.85, 0.65)
        c.sign('Delivery', 'LINE-SIDE / COMPLETE KITS', [2.25, 1.54, 1.02], [0.95, 0.1])
        c.box('Route/Aisle', [1.25, 0.8, 0.003], [0.055, 2.7, 0.004], 'yellow')
        c.sign('Order', 'ORDER: 2 COMPLETE KITS / 90 s', [0, 1.23, 1.28], [1.2, 0.12])
        views = [('01_kitting_and_delivery', [3.1, -3.5, 2.8], [0.65, 1.1, 0.95]), ('02_kitting_station', [1.4, -1.45, 1.9], [0, 0.61, 1]), ('03_components', [-0.75, -0.35, 1.65], [-0.14, 0.69, 0.95])]
    elif task == 48:
        rack('Shelves/A', -0.9, 2.25)
        rack('Shelves/B', 0.5, 2.25)
        rack('Shelves/C', 1.9, 2.25)
        c.asset('Inbound/Pallet', ROOT / 'task48/recovered/pallet.usd', [-0.6, -0.1, 0.1], [1.1, 0.8, 0.2])
        for i in range(6):
            x = -0.91 + i % 3 * 0.31
            y = -0.28 + i // 3 % 2 * 0.37
            c.asset(f'Inbound/Box{i}', BOX, [x, y, 0.36], [0.28, 0.33, 0.32])
            c.sign(f'SKU{i}', ['BRAKE KITS / A', 'FILTERS / B', 'SENSORS / C'][i % 3], [x, y - 0.171, 0.36], [0.26, 0.06])
        for i, x in enumerate([-0.9, 0.5, 1.9]):
            c.sign(f'Shelf{i}', f'BAY {chr(65 + i)} / ' + ['BRAKES', 'FILTERS', 'SENSORS'][i], [x, 1.958, 1.7], [1.0, 0.13])
        for i, y in enumerate([0.56, 1.5]):
            c.box(f'Route/Edge{i}', [0.6, y, 0.004], [3.9, 0.035, 0.004], 'yellow')
        c.sign('Dispatch', 'INBOUND -> SKU MATCH -> SHELF / 3 BOXES IN90 s', [0.8, 3.72, 2.12], [3.6, 0.18])
        views = [('01_putaway_bay', [3.8, -4.1, 3.05], [0.4, 1.0, 0.9]), ('02_pallet_to_shelves', [2.8, -2.8, 2.4], [0.15, 1, 0.85]), ('03_inbound_products', [1.0, -1.9, 1.6], [-0.55, -0.02, 0.4])]
    elif task == 49:
        table('InspectionBench', 0, 0.65, 1.85, 0.9)
        rack('MachinedStock', -1.7, 2.0)
        path = ROOT / 'task49/prepared-v3/port_block.usda'
        tray('Input/Tray', [-0.6, 0.6], (0.42, 0.37, 0.035), 'MACHINED / INPUT')
        for i in range(3):
            c.asset(f'Input/Part{i}', path, [-0.72 + i * 0.11, 0.6, 0.91885], material='steel')
        c.box('Inspection/Nest', [0, 0.63, 0.923], [0.2, 0.2, 0.055], 'frame')
        c.asset('Inspection/Part', path, [0, 0.63, 0.9715], material='steel')
        c.box('Camera/Post', [0, 1.0, 1.22], [0.05, 0.06, 0.67], 'frame')
        c.box('Camera/Housing', [0, 0.95, 1.35], [0.13, 0.12, 0.13], 'black')
        c.pipe('Camera/Lens', [0, 0.89, 1.35], [0, 0.84, 1.35], 0.036, 'black')
        c.pipe('Camera/IlluminationRing', [0, 0.84, 1.35], [0, 0.831, 1.35], 0.06, 'white', inner_radius=0.04)
        tray('Output/Pass', [0.58, 0.82], (0.36, 0.27, 0.035), 'PASS')
        tray('Output/Reject', [0.58, 0.35], (0.36, 0.27, 0.035), 'REJECT')
        c.sign('Recipe', 'CAMERA RECIPE: FACE1 / FACE2 / FACE3', [0, 1.17, 1.65], [1.3, 0.1])
        views = [('01_inspection_bay', [2.4, -2.8, 2.25], [0, 0.9, 1.04]), ('02_three_side_station', [1.3, -1.1, 1.85], [0, 0.65, 1.04]), ('03_camera_and_part', [0.5, -0.15, 1.42], [0, 0.67, 1.1])]
    elif task == 50:
        table('AssemblyBench', 0, 0.65, 1.85, 0.9)
        cabinet('MaterialCabinet', -1.45, 2.0)
        m = portable_json_loads((ROOT / 'task50/asset-manifest.json').read_text())
        p = ROOT / 'task50/prepared-v2'
        for i, shade in enumerate(['CHAMPAGNE', 'ROSE', 'COPPER', 'PLUM']):
            x = -0.66 + i * 0.44
            tray(f'Feed/Color{i}', [x, 0.86], (0.35, 0.22, 0.025), shade)
            for j in range(6):
                c.asset(f'Feed/Pan{i}_{j}', p / f'pan_{i}.usd', [x + (j % 3 - 1) * 0.072, 0.82 + j // 3 * 0.075, 0.89617])
        tray('Empty/Cases', [-0.62, 0.32], (0.4, 0.29, 0.025), 'EMPTY CASES')
        case_bounds = np.asarray(m['case_bounds_m'])
        h = case_bounds[1, 2] - case_bounds[0, 2]
        for i, x in enumerate([-0.73, -0.53]):
            c.asset(f'Cases/Empty{i}', p / 'case.usd', [x, 0.32, 0.8949 + h / 2])
        c.box('Nest/Base', [0, 0.35, 0.916], [0.17, 0.24, 0.035], 'frame')
        c.asset('Assembly/Case', p / 'case.usd', [0, 0.35, 0.9335 + h / 2])
        tray('Output/Tray', [0.6, 0.33], (0.38, 0.3, 0.025), 'FINISHED PALETTES')
        c.sign('Recipe', 'RECIPE  1:CHAMPAGNE  2:ROSE  3:COPPER  4:PLUM', [0, 1.16, 1.29], [1.75, 0.11])
        views = [('01_cosmetics_cell', [2.4, -2.6, 2.3], [0, 0.85, 1.05]), ('02_four_feed_assembly', [1.1, -1.15, 1.78], [0, 0.61, 1.02]), ('03_palette_and_recipe', [0.4, -0.35, 1.48], [0, 0.48, 0.97])]
    else:
        raise ValueError(task)
    return SimpleNamespace(**vars(c), cell=c, views=views, metadata=dict(task_id=task, temporal_category=category, time_requirement=requirements[task], scene=titles[task], scope='Actual static RGB review only; collisions disabled. No robot, fluid/process, control, native contact or timing success validation.', mobile_requirement='Retained. Mobile base undecided; aisle and delivery/shelf destinations are layout proposals, not validated travel.' if task in [47, 48] else None, authored_fixtures='Support nests, signs, piping/guards and camera enclosure only; products reuse existing imported assets.', approval='Pending reviewer environment review before robot integration', randomization='Preview seed0 fixed layout. Planned independent bounded seeded product-pose, temporal and physical-property streams; not validated by a static preview.'))
