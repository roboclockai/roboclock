"""Review-requested46/49 factory proposals; no robot or physical pass claimed."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .industrial_workcell import IndustrialCell
from .paths import asset_path
from .remaining_environments import RACK, ROOT, TABLE, TRAY, build_remaining_environment


def build_factory_revision(task, output, resolution=(1600, 1000)):
    from pxr import Gf, UsdGeom
    if task == 49:
        scene = build_remaining_environment(49, output, resolution)
        c = scene.cell
        for name in ['Housing', 'Lens', 'IlluminationRing']:
            c.stage.RemovePrim('/World/Camera/' + name)
        model = ROOT / 'task49/camera-source/rsd455.usd'
        c.asset('Camera/RealSenseD455', model, [0, 0.92, 1.35], yaw=-np.pi / 2)
        c.box('Camera/MountingBracket', [0, 0.963, 1.35], [0.1, 0.035, 0.024], 'steel')
        c.pipe('Camera/PowerCable', [0.045, 0.97, 1.35], [0.07, 1.02, 1.17], 0.004, 'black')
        c.pipe('Camera/RingLight', [0, 0.884, 1.35], [0, 0.874, 1.35], 0.082, 'white', inner_radius=0.066)
        for name, text, y, material in [('Pass', 'PASS / ACCEPTED', 0.82, 'green'), ('Reject', 'REJECT / REWORK', 0.35, 'red')]:
            old = 'Output_' + name + 'Label'
            c.stage.RemovePrim('/World/Signs/' + old)
            c.box('Output/' + name + 'LabelBack', [0.58, y + 0.16, 1.005], [0.36, 0.018, 0.125], material)
            c.sign(name + 'TrayLabel', text, [0.58, y + 0.148, 1.008], [0.335, 0.085])
            c.box('Output/' + name + 'ColorStrip', [0.58, y - 0.139, 0.908], [0.3, 0.012, 0.009], material)
        scene.views = [('01_inspection_bay', [2.4, -2.8, 2.25], [0, 0.9, 1.04]), ('02_camera_and_sorting', [1.15, -0.95, 1.75], [0.05, 0.67, 1.06]), ('03_real_camera_detail', [0.32, 0.39, 1.51], [0, 0.92, 1.35])]
        scene.metadata.update(revision='Review, October21: original official D455 camera replaces proxy; large separate green PASS/red REJECT tray signs', camera_source='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Sensors/RealSense/D455/rsd455.usd', camera_documentation='https://docs.isaacsim.omniverse.nvidia.com/6.0.1/sensors/isaacsim_sensors_camera.html', camera_scope='Complete original sensor model and optical cameras imported at1:1. Sensor calibration/defect classifier not validated by static review.', authored_fixtures='Camera mounting bracket, cable, ring illumination and sign supports; product and camera geometry imported unchanged')
        return scene
    if task != 46:
        raise ValueError(task)
    c = IndustrialCell(output, resolution)
    c.asset('Production/CNCLathe', Path(asset_path('/opt/industry/arx-x5/task42-environment-v1/assets/cnc_lathe/equipment.usdc')), [-1.3, 1.95, 0.825])
    c.asset('Return/QCConveyor', ROOT / 'task42/conveyor/conveyor-local.usd', [-0.28, 0.91, 1.0315], [0.92, 0.24, 0.065], material='steel')
    c.asset('Recovery/Bench', TABLE, [0.65, 0.4, 0.43], [1.3, 0.55, 0.86])
    c.box('Recovery/Top', [0.65, 0.4, 0.875], [1.3, 0.55, 0.025], 'steel')
    c.asset('Inventory/Rack', RACK, [2.15, 2.75, 1.05], [1.1, 0.55, 2.1])
    slope = np.deg2rad(8.2)
    rotation = Rotation.from_euler('y', slope).as_quat()
    q = np.r_[rotation[3], rotation[:3]]
    center = np.array([0.7, 0.91, 0.985])
    chute = c.asset('Return/GravityLane', ROOT / 'task46/prepared/return_surface.usd', center, [1.1, 0.22, 2.3280419814142306e-09], material='steel')
    chute.set_world_pose(position=center, orientation=q)
    chute.set_default_state(position=center, orientation=q)
    deck = c.box('Return/Deck', center + [0, 0, -0.018], [1.1, 0.22, 0.034], 'steel')
    UsdGeom.Xformable(deck.GetPrim()).AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
    for j, y in enumerate([0.79, 1.03]):
        rail = c.box(f'Return/Guide{j}', [0.7, y, 1.005], [1.1, 0.014, 0.048], 'frame')
        UsdGeom.Xformable(rail.GetPrim()).AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
    for i, x in enumerate([0.28, 0.46, 0.64]):
        z = center[2] - (x - center[0]) * np.tan(slope) + 0.035
        c.asset(f'Queue/Bearing{i}', ROOT / 'task46/prepared/bearing.usd', [x, 0.91, z])
    for i, x in enumerate([0.31, 1.12]):
        z = center[2] - (x - center[0]) * np.tan(slope) - 0.035
        c.box(f'Return/Support{i}', [x, 0.91, z / 2], [0.06, 0.22, z], 'frame')
        c.box(f'Return/Foot{i}', [x, 0.91, 0.025], [0.22, 0.38, 0.05], 'frame')
    for j, y in enumerate([0.795, 1.025]):
        c.box(f'Safety/Boundary{j}', [1.05, y, 0.987 - (1.05 - 0.7) * np.tan(slope)], [0.3, 0.018, 0.008], 'yellow')
    c.box('Safety/EndMarker', [1.27, 0.91, 0.905], [0.018, 0.25, 0.012], 'red')
    c.asset('Recovery/ReworkTray', TRAY, [0.78, 0.39, 0.9075], [0.4, 0.29, 0.04], material='steel')
    c.asset('Collection/KLTTote', ROOT / 'task46/collection-source/small_KLT_visual.usd', [1.36, 0.91, 0.78])
    c.box('Collection/TrolleyTop', [1.36, 0.91, 0.6968], [0.3, 0.4, 0.02], 'steel')
    for i, x in enumerate([1.25, 1.47]):
        for j, y in enumerate([0.76, 1.06]):
            c.box(f'Collection/TrolleyLeg{i}{j}', [x, y, 0.3434], [0.028, 0.028, 0.6868], 'frame')
    for i, y in enumerate([0.755, 1.065]):
        c.box(f'Inspection/Portal{i}', [-0.35, y, 1.14], [0.09, 0.035, 0.2], 'frame')
    c.box('Inspection/PortalTop', [-0.35, 0.91, 1.24], [0.09, 0.345, 0.035], 'frame')
    c.pipe('Inspection/Sensor', [-0.35, 0.91, 1.22], [-0.35, 0.91, 1.19], 0.018, 'black')
    c.sign('Inspection', 'QC / MACHINE OUTFEED', [-0.35, 1.13, 1.38], [0.72, 0.085])
    c.sign('ReturnFlow', 'GRAVITY DISCHARGE -> COLLECTION', [0.65, 1.13, 1.24], [0.9, 0.085])
    c.sign('Rework', 'RECOVERED / REWORK', [0.78, 0.226, 0.945], [0.42, 0.07])
    c.sign('Batch', 'BATCH COLLECTION', [1.36, 0.705, 0.77], [0.34, 0.07])
    c.sign('Safety', 'RECOVERY ZONE / ARREST <=3 s', [0.97, 0.773, 0.9], [0.69, 0.065])
    c.sign('Cell', 'BEARING FINISHING / QC RETURN / CELL46', [0.1, 3.72, 2.65], [4.0, 0.23])
    for i, x in enumerate([-0.65, 0.05]):
        c.box(f'Conveyor/Leg{i}', [x, 0.91, 0.48], [0.06, 0.15, 0.95], 'frame')
    return SimpleNamespace(**vars(c), cell=c, views=[('01_factory_return_corner', [3.3, -3.4, 2.6], [0.1, 1.5, 1.04]), ('02_qc_return_and_recovery', [2.4, -1.45, 1.95], [0.63, 0.89, 0.96]), ('03_zone_and_trays', [2.02, -0.15, 1.62], [1.03, 0.82, 0.88])], metadata=dict(task_id=46, temporal_category='B', time_requirement='Arrest bearing inside safety zone within3s after zone entry; then recover to rework tray', original_requirement='Stop a rolling bearing inside safety area, then put it in tray', scene='Machine outfeed inspection portal feeds gravity discharge chute into original KLT batch tote; separate side bench holds recovered/rework bearings', revision='reviewer rejected unexplained slope/end: inlet height now matches belt top, discharge end reaches above original collection tote, bench moved alongside lane', collection_asset='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Props/KLT_Bin/small_KLT_visual.usd', collection_status='Original KLT visual USD plus all3 original texture dependencies downloaded; static import only, native collection not validated', reviewer_equipment='industry_2/4/4.obj intact machine; local-only redistribution unverified', authored_fixtures='Ordinary gravity-lane deck/guides, supports, markings and tray labels; no new scored products', scope='Static GPU environment only; no robot, machining, belt transport, native rolling/arrest/contact or timing validation', approval='Awaiting reviewer revised-environment review', planned_randomization='Independent seeded bearing release/layout, zone-entry timing and rolling/friction properties; untested by static preview'))
