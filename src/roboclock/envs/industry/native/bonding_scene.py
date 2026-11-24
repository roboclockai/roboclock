"""Task43 environment proposal: timed industrial enclosure-cover bonding."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .industrial_workcell import IndustrialCell
from .paths import asset_path, portable_json_loads


def build_bonding_environment(output, resolution=(1600, 1000)):
    c = IndustrialCell(output, resolution=resolution)
    root = Path(asset_path('/opt/industry/assets/v2/task43/prepared-knob40-v2'))
    manifest = portable_json_loads((root / 'asset-manifest.json').read_text())
    bounds = {name: np.array(manifest['prepared'][name]['bounds_m']) for name in ['housing', 'lid']}
    table = Path(asset_path('/opt/industry/assets/v2/shared/factory-v1/industrial_table.usd'))
    c.asset('Station/AssemblyBench', table, [0, 0.47, 0.39], [1.5, 0.9, 0.78])
    c.box('Station/SteelWorktop', [0, 0.47, 0.785], [1.52, 0.92, 0.05], 'steel')
    c.box('Station/FrontFascia', [0, 0.007, 0.665], [1.5, 0.025, 0.22], 'frame')
    c.box('Station/LowerShelf', [0, 0.55, 0.22], [1.32, 0.68, 0.04], 'frame')
    for i, x in enumerate([-0.64, 0.64]):
        c.box(f'Gantry/Column{i}', [x, 0.77, 1.13], [0.065, 0.065, 0.64], 'frame')
        c.box(f'Gantry/Mount{i}', [x, 0.77, 0.825], [0.16, 0.16, 0.03], 'steel')
    c.box('Gantry/TopRail', [0, 0.77, 1.43], [1.38, 0.095, 0.095], 'steel')
    c.box('Gantry/SlideCover', [0, 0.711, 1.43], [1.29, 0.025, 0.045], 'black')
    c.box('Dispenser/Carriage', [0.03, 0.7, 1.38], [0.18, 0.16, 0.14], 'white')
    c.box('Dispenser/YRail', [0.03, 0.55, 1.31], [0.055, 0.34, 0.07], 'steel')
    c.box('Dispenser/Head', [0.0465, 0.4, 1.16], [0.11, 0.1, 0.25], 'white')
    c.pipe('Dispenser/Nozzle', [0.0465, 0.4, 1.04], [0.0465, 0.4, 0.862], 0.0035, 'steel')
    c.box('Dispenser/FeedController', [-0.38, 0.78, 1.14], [0.28, 0.16, 0.43], 'white')
    c.sign('MixUnit', 'ADHESIVE DISPENSER', [-0.38, 0.692, 1.17], [0.25, 0.075])
    c.pipe('Dispenser/SupplyLineA', [-0.3, 0.75, 1.37], [-0.3, 0.75, 1.55], 0.008, 'black')
    c.pipe('Dispenser/SupplyLineB', [-0.3, 0.75, 1.55], [0.03, 0.75, 1.55], 0.008, 'black')
    c.pipe('Dispenser/SupplyLineC', [0.03, 0.75, 1.55], [0.03, 0.55, 1.38], 0.008, 'black')
    floor = 0.824
    c.box('Fixture/Nest', [0, 0.4, 0.815], [0.22, 0.27, 0.018], 'frame')
    c.asset('Products/ActiveHousing', root / 'housing.usd', np.array([0, 0.4, floor]) + bounds['housing'].mean(0), material='steel')
    for i, x in enumerate([-0.088, 0.088]):
        c.box(f'Fixture/Stop{i}', [x, 0.4, 0.84], [0.022, 0.16, 0.04], 'frame')
        c.pipe(f'Fixture/Screw{i}', [x, 0.4, 0.86], [x, 0.4, 0.875], 0.006, 'steel')
    path = [[-0.0465, 0.3285], [-0.0465, 0.4715], [0.0465, 0.4715], [0.0465, 0.3285], [-0.0465, 0.3285]]
    for i, (p, q) in enumerate(zip(path[:-1], path[1:], strict=True)):
        c.pipe(f'Process/AdhesiveBead{i}', [*p, floor + 0.0315], [*q, floor + 0.0315], 0.0007, 'glue')
    for i, (x, y) in enumerate([[-0.51, 0.25], [-0.51, 0.52], [-0.27, 0.25]]):
        c.box(f'Input/LidPallet{i}', [x, y, 0.818], [0.17, 0.21, 0.014], 'frame')
        center = np.array([x, y, 0.826]) + bounds['lid'].mean(0) - [0, 0, bounds['lid'][0, 2]]
        c.asset(f'Products/InputCover{i}', root / 'lid.usd', center, material='steel')
    for i, (x, y) in enumerate([[0.4, 0.26], [0.4, 0.61]]):
        c.box(f'Output/Pallet{i}', [x, y, 0.818], [0.19, 0.23, 0.014], 'frame')
        origin = np.array([x, y, 0.826])
        c.asset(f'Products/FinishedHousing{i}', root / 'housing.usd', origin + bounds['housing'].mean(0), material='steel')
        c.asset(f'Products/FinishedCover{i}', root / 'lid.usd', origin + bounds['lid'].mean(0), material='steel')
    c.box('Control/Stand', [0.72, 0.63, 1.02], [0.05, 0.05, 0.43], 'frame')
    c.box('Control/Display', [0.72, 0.58, 1.25], [0.32, 0.1, 0.24], 'frame')
    c.sign('Recipe', 'BOND WINDOW / 2-14 s', [0.72, 0.524, 1.26], [0.295, 0.11])
    # robodojo_rig head view (60 deg down): the process beacon stands low beside the bench
    # (was on a pole at 1.63-1.70 m, above the head image).
    c.pipe('Control/BeaconPole', [0.45, 0.82, 0.81], [0.45, 0.82, 0.97], 0.012, 'steel')
    c.pipe('Control/Beacon', [0.45, 0.82, 0.97], [0.45, 0.82, 1.04], 0.025, 'green')
    c.sign('Cell', 'ENCLOSURE COVER / ADHESIVE BONDING / CELL43', [0, 0.692, 1.43], [1.24, 0.065])
    c.sign('Input', 'MATCHING COVERS / INPUT', [-0.43, -0.012, 0.7], [0.55, 0.065])
    c.sign('Assembly', 'DISPENSE -> PLACE COVER', [0, -0.012, 0.64], [0.6, 0.065])
    c.sign('Output', 'BONDED ENCLOSURES / OUTPUT', [0.44, -0.012, 0.7], [0.56, 0.065])
    c.sign('Stud', 'COVERS USE REMOVABLE HANDLING STUDS', [-0.41, -0.012, 0.58], [0.49, 0.045])
    return SimpleNamespace(**vars(c), cell=c, views=[('01_factory_cell', [1.65, -1.75, 1.95], [0, 0.57, 0.98]), ('02_bonding_station', [-0.2, -1.2, 1.75], [0, 0.43, 0.99]), ('03_fixture_detail', [0.36, -0.02, 1.24], [0, 0.4, 0.86])], metadata=dict(task_id=43, temporal_category='A', time_requirement='Place cover2–14s after adhesive application', original_requirement='Place a lid onto a housing after glue is applied; fixture support allowed', scene='Industrial matched-enclosure cover bonding cell with dispensing gantry, cover input and assembled output', product_source=manifest['source_cad'], process_fixture=manifest.get('process_fixture'), material_notes='Declared brushed-aluminium shell adaptation of existing printable design; original shell geometry retained. Amber seal line is illustrative.', authored_equipment='Gantry/nozzle, nest/clamps, supply controller, HMI, pallet fixtures and adhesive-line visual', scope='Static environment only; no robot, adhesive chemistry, gantry dynamics, seating/contact, timing-success or randomized validation', approval='Pending reviewer environment review; no robot/control authorized before review', planned_randomization='Independent bounded seeded part/fixture poses, temporal glue-event phase and physical properties; not implemented/tested by this static preview'))
