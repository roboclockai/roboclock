"""Task44 environment proposal: cooling-water leak isolation utility skid."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .industrial_workcell import IndustrialCell
from .paths import asset_path, portable_json_loads


def build_utility_environment(output, resolution=(1600, 1000), line_offset_xy=(0.0, 0.0)):
    """``line_offset_xy`` moves the whole valve line (skid, piping, flanges, valve and leak
    nipple) together in the floor plane; the plant, tray, HMI and signs stay put."""
    c = IndustrialCell(output, resolution=resolution)
    root = Path(asset_path('/opt/industry/assets/v2/task44'))
    manifest = portable_json_loads((root / 'asset-manifest.json').read_text())
    plant = Path(asset_path('/opt/industry/arx-x5/environment-preview-v1/assets/water_treatment/equipment.usdc'))
    c.asset('Plant/WaterTreatment', plant, [0, 2.05, 1.05])
    ox, oy = map(float, line_offset_xy)
    y = 0.5 + oy
    z = 1.04 - 0.07783249
    pivot = np.array([ox, y, 1.04])
    for i, x in enumerate(np.array([-0.9, 0.85]) + ox):
        c.box(f'Skid/Support{i}', [x, y, 0.46], [0.075, 0.075, 0.9], 'frame')
        c.box(f'Skid/Foot{i}', [x, y, 0.025], [0.28, 0.32, 0.05], 'frame')
        c.box(f'Skid/Saddle{i}', [x, y, z - 0.027 - 0.04], [0.16, 0.16, 0.08], 'frame')
    c.box('Skid/TieRail', [-0.025 + ox, y, 0.32], [1.82, 0.06, 0.07], 'frame')
    for name, a, b in [('Inlet', [-1.01 + ox, y, z], [-0.078 + ox, y, z]), ('Outlet', [0.078 + ox, y, z], [0.86 + ox, y, z]), ('SupplyReturn', [-1.1 + ox, y + 0.09, z], [-1.1 + ox, 1.34, z]), ('DownstreamReturn', [0.95 + ox, y + 0.09, z], [0.95 + ox, 1.34, z])]:
        c.pipe('Piping/' + name, a, b, 0.027, 'steel', inner_radius=0.022)
    c.elbow('Piping/InletElbow', [-1.01 + ox, y + 0.09, z], 0.09, 180, 270)
    c.elbow('Piping/OutletElbow', [0.86 + ox, y + 0.09, z], 0.09, 270, 360)
    for i, x in enumerate(np.array([-0.083728975, 0.083728975, 0.44]) + ox):
        c.pipe(f'Piping/Flange{i}', [x - 0.012, y, z], [x + 0.012, y, z], 0.052, 'steel', inner_radius=0.027)
        for j, t in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            cy = y + 0.041 * np.cos(t)
            cz = z + 0.041 * np.sin(t)
            c.pipe(f'Piping/Bolt{i}_{j}', [x - 0.018, cy, cz], [x + 0.018, cy, cz], 0.004, 'frame')
    for name in ['housing', 'handle']:
        bounds = np.array(manifest['prepared'][name]['bounds_m'])
        c.asset('Valve/' + name, root / 'prepared-v1' / (name + '.usd'), pivot + bounds.mean(0))
    c.pipe('Leak/TestNipple', [0.44 + ox, y, z], [0.44 + ox, y - 0.075, z], 0.008, 'steel')
    c.pipe('Leak/OutletLip', [0.44 + ox, y - 0.075, z], [0.44 + ox, y - 0.087, z], 0.013, 'brass')
    c.box('Containment/Support', [0.44, 0.35, 0.7], [0.3, 0.3, 0.055], 'frame')
    c.box('Containment/Bracket', [0.7, 0.48, 0.7], [0.4, 0.22, 0.08], 'frame')
    c.asset('Containment/OpenTray', Path(asset_path('/opt/industry/assets/v2/task46/prepared/tray.usd')), [0.44, 0.32, 0.755], [0.34, 0.28, 0.05], material='steel')
    from pxr import Gf, UsdGeom, UsdShade
    for i, t in enumerate(np.linspace(0, 1, 11)):
        s = UsdGeom.Sphere.Define(c.stage, f'/World/Leak/Drop{i}')
        s.CreateRadiusAttr(0.0035 + 0.001 * t)
        s.AddTranslateOp().Set(Gf.Vec3d(0.44 + ox, float(y - 0.087 - 0.05 * t), float(z - 0.18 * t * t)))
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(c.appearance.material('water').material)
    c.box('Control/Post', [0.89, 0.96, 1.04], [0.055, 0.055, 0.48], 'frame')
    c.box('Control/HMI', [0.89, 0.9, 1.34], [0.4, 0.12, 0.26], 'frame')
    c.sign('Warning', 'LEAK DETECTED / CLOSE <=5 s', [0.89, 0.833, 1.35], [0.375, 0.12])
    # robodojo_rig head view (60 deg down): the beacon stands low (was 1.67-1.74 m).
    c.pipe('Control/BeaconPole', [0.62, 0.75, 0.0], [0.62, 0.75, 1.01], 0.012, 'steel')
    c.pipe('Control/Beacon', [0.62, 0.75, 1.01], [0.62, 0.75, 1.08], 0.027, 'red')
    c.sign('Cell', 'COOLING WATER / LEAK ISOLATION / CELL44', [0, 3.72, 2.75], [3.7, 0.22])
    c.box('Signs/ValvePlacardBack', [0, 0.22, 0.79], [0.45, 0.025, 0.12], 'frame')
    c.sign('Valve', 'ISOLATION VALVE', [0, 0.204, 0.8], [0.43, 0.075])
    c.box('Signs/LeakPlacardBack', [0.46, 0.13, 0.62], [0.53, 0.025, 0.12], 'frame')
    c.sign('Leak', 'DOWNSTREAM LEAK / CAPTURE', [0.46, 0.114, 0.63], [0.51, 0.075])
    c.sign('Supply', 'SUPPLY', [-0.79, 0.44, 0.88], [0.25, 0.05])
    c.sign('Return', 'PROCESS LINE', [0.72, 0.44, 0.88], [0.3, 0.05])
    return SimpleNamespace(**vars(c), cell=c, views=[('01_utility_bay', [2.7, -3.0, 2.5], [0, 1.45, 1.15]), ('02_isolation_station', [0.95, -1.15, 1.75], [0.1, 0.65, 1.04]), ('03_valve_detail', [-0.32, -0.02, 1.4], [0.04, 0.51, 1.04])], metadata=dict(task_id=44, temporal_category='B', time_requirement='Fully close isolation valve within5s after leak onset', original_requirement='Close an isolation valve after a pipe starts leaking', scene='Cooling-water utility skid: imported water-treatment plant, original ball valve, downstream leak test flange and open capture tray', product_source=dict(url=manifest['source_url'], license=manifest['license'], prepared=manifest['prepared']), reviewer_equipment='Intact previously converted selection from industry_3/4/4.obj; redistribution unverified, local only', authored_equipment='Service routing/couplings/supports and HMI; illustrative leak snapshot is not fluid dynamics', scope='Static environment only; no robot, passive valve joint, contact closure, response-time or randomized validation in this preview', approval='Pending reviewer environment review; do not add robot/control before review', planned_randomization='Independent bounded seeded valve/layout, leak-onset and physical-property streams; not tested by this static preview'))
