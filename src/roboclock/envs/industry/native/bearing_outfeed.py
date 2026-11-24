"""Task46 physical machine outfeed and non-physical red/green process cue.

Existing conveyor and original machine/bearing meshes; no scored-body writes.
The native metering gate releases only after real upstream transport/support.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from .arx_cell_task import remove_preview, set_visual_target
from .paths import asset_path


class BearingOutfeed:

    def __init__(self, scene, cell, contact_material):
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import XFormPrim
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        from .remaining_environments import ROOT
        s = scene
        self.s = s
        self.c = cell
        self.rows = []
        self.ready = False
        cfg = s.resolved['machine_outfeed']
        self.junction = cfg['junction_x_m']
        self.top = s.surface_z(self.junction)
        from pathlib import Path
        remove_preview(cell, '/World/Production/CNCLathe')
        cell.asset('Production/CNCLathe', Path(asset_path('/opt/industry/arx-x5/task42-environment-v1/assets/cnc_lathe/equipment.usdc')), [-1.5, 0.61, 0.825], yaw=np.pi / 2)
        bbox = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(s.stage.GetPrimAtPath('/World/Production/CNCLathe')).ComputeAlignedBox()
        self.machine_exit_x = float(bbox.GetMax()[0])
        start = cfg['belt_start_x_m']
        end = self.junction
        remove_preview(cell, '/World/Return/GravityLane')
        for path in ['/World/Return/Deck', '/World/Return/Guide0', '/World/Return/Guide1']:
            s.stage.RemovePrim(path)
        center = np.array([(end + 1.25) / 2, s.resolved['lane_y_m'], s.surface_z((end + 1.25) / 2)])
        slope = Rotation.from_euler('y', s.resolved['slope_deg'], degrees=True).as_quat()
        q = np.r_[slope[3], slope[:3]]
        lane = cell.asset('Return/GravityLane', ROOT / 'task46/prepared/return_surface.usd', center, [1.25 - end, 0.22, 2.3280419814142306e-09], material='steel')
        lane.set_world_pose(position=center, orientation=q)
        lane.set_default_state(position=center, orientation=q)
        deck = cell.box('Return/Deck', center + [0, 0, -0.018], [1.25 - end, 0.22, 0.034], 'steel')
        UsdGeom.Xformable(deck.GetPrim()).AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
        for j, dy in enumerate([-0.12, 0.12]):
            rail = cell.box(f'Return/Guide{j}', center + [0, dy, 0.02], [1.25 - end, 0.014, 0.048], 'frame')
            UsdGeom.Xformable(rail.GetPrim()).AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
        remove_preview(cell, '/World/Return/QCConveyor')
        cell.asset('Return/QCConveyor', ROOT / 'task42/conveyor/conveyor-local.usd', [(start + end) / 2, s.resolved['lane_y_m'], self.top - 0.0325], [end - start, 0.24, 0.065], material='steel')
        body = s.world.scene.add(FixedCuboid('/World/MachineOutfeed/Belt', name='machine_outfeed_belt', position=np.array([(start + end) / 2, s.resolved['lane_y_m'], self.top - 0.003]), scale=np.array([end - start, 0.22, 0.006]), color=np.array([0.055, 0.067, 0.07])))
        UsdPhysics.RigidBodyAPI.Apply(body.prim).CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(body.prim).CreateContactOffsetAttr(0.0003)
        PhysxSchema.PhysxCollisionAPI.Apply(body.prim).CreateRestOffsetAttr(0)
        self.belt_api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(body.prim)
        self.belt_api.CreateSurfaceVelocityLocalSpaceAttr(False)
        self.belt_api.CreateSurfaceVelocityAttr(Gf.Vec3f(cfg['surface_speed_m_s'], 0, 0))
        UsdShade.MaterialBindingAPI.Apply(body.prim).Bind(contact_material.material, materialPurpose='physics')
        for j, dy in enumerate([-0.022, 0.022]):
            cell.box(f'MachineOutfeed/Guide{j}', [(start + end) / 2, s.resolved['lane_y_m'] + dy, self.top + 0.02], [end - start, 0.01, 0.048], 'frame')
            gc = np.array([(0.78 + 0.94) / 2, s.resolved['lane_y_m'] + dy, s.surface_z((0.78 + 0.94) / 2) + 0.02])
            rail = cell.box(f'MachineOutfeed/RampGuide{j}', gc, [0.94 - 0.78, 0.01, 0.048], 'frame')
            UsdGeom.Xformable(rail.GetPrim()).AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))
        for path in ['/World/Conveyor/Leg0', '/World/Conveyor/Leg1', '/World/Return/Support0', '/World/Return/Foot0']:
            s.stage.RemovePrim(path)
        for i, x in enumerate([-0.95, -0.1, 0.55]):
            cell.box(f'MachineOutfeed/Leg{i}', [x, s.resolved['lane_y_m'], (self.top - 0.065) / 2], [0.045, 0.15, self.top - 0.065], 'frame')
        # robodojo_rig head view (60 deg down): the arrest/recovery lamp stands low beyond
        # the collection tote (was a 1.43 m post at x=1.5, above and right of the head image).
        cell.pipe('RecoverySignal/Post', [1.35, 1.25, 0], [1.35, 1.25, 0.94], 0.018, 'frame')
        cell.box('RecoverySignal/Housing', [1.35, 1.25, 1.0], [0.13, 0.12, 0.13], 'black')
        self.front = np.array([1.35, 1.182, 1.0])
        self.lenses = {}
        self.state = 'green'
        for color, rgb in [('green', [0.03, 0.8, 0.12]), ('red', [1.0, 0.015, 0.008])]:
            name = 'bearing_signal_' + color
            mat = cell.appearance.material(name, color=rgb, roughness=0.2, metallic=0, emissive=True)
            from pxr import Sdf
            mat.shaders_list[0].CreateInput('emissive_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            mat.shaders_list[0].GetInput('emissive_intensity').Set(3.0)
            pos = self.front + [0, 0 if color == 'green' else 0.08, 0]
            root = UsdGeom.Xform.Define(s.stage, '/World/RecoverySignal/' + color)
            root.AddTranslateOp().Set(Gf.Vec3d(*pos))
            lens = UsdGeom.Sphere.Define(s.stage, str(root.GetPath()) + '/Lens')
            lens.CreateRadiusAttr(0.042)
            lens.AddScaleOp().Set(Gf.Vec3f(1, 0.32, 1))
            UsdShade.MaterialBindingAPI.Apply(lens.GetPrim()).Bind(mat.material)
            self.lenses[color] = s.world.scene.add(XFormPrim(str(root.GetPath()), name='recovery_signal_' + color))
        cell.sign('RecoverySignal', 'RED = ARREST / GREEN = NORMAL', [1.35, 1.17, 0.90], [0.55, 0.065])
        s.stage.RemovePrim('/World/Signs/Inspection')
        cell.sign('Inspection', 'MACHINE OUTFEED / QC', [-0.65, 1.13, 1.21], [0.82, 0.08])
        # The policy camera is the rig head camera (rig.py); it no longer follows the bearing.
        self.camera_eye = None
        self.camera_target = None
        self.metadata = dict(source='Existing supplied original machine and existing Isaac conveyor; declared flat conservative conveyor sheet and process beacon.', original_machine_outlet_bound_x_m=self.machine_exit_x, belt_start_x_m=start, belt_end_x_m=end, belt_top_z_m=self.top, surface_speed_m_s=cfg['surface_speed_m_s'], junction='Continuous level-belt-to-original8.2degree lane. Capture-zone plane and source bearing unchanged.', metering='Original downstream native gate opens only after measured whole-part arrival, native gate support and a stable0.5s pose window; minimum temporal start remains independent.', cue='Green during outfeed, red at actual safety-zone entry until recovered; first contact cannot precede entry. Process lenses move via Fabric, never product pose writes.', camera='Fixed rig head camera (rig.py); the earlier review camera that followed the bearing was removed with the common rig.')

    def signal(self, state, t):
        if state != self.state:
            for name, body in self.lenses.items():
                set_visual_target(body, self.front + [0, 0 if name == state else 0.08, 0])
            self.state = state
            self.s.events['indicator_' + state + '_s'] = float(t)

    def step(self, o):
        s = self.s
        t = o['time_s']
        p = np.array(o['bearing_position'])
        if o['bounds_min_m'][0] >= self.machine_exit_x:
            s.events.setdefault('whole_machine_exit_s', t)
        at_gate = 0.795 < p[0] < 0.84 and o['gate_support_N'] > 0.05
        if not self.ready:
            if at_gate:
                self.rows.append((t, p.copy(), np.array(o['bearing_rotation'])))
            else:
                self.rows.clear()
            while len(self.rows) > 1 and self.rows[1][0] <= t - 0.5:
                self.rows.pop(0)
            if len(self.rows) >= round(0.5 * CONTROL_HZ) + 1 and t - self.rows[0][0] >= 0.5 - 1e-06:
                span = float(np.linalg.norm(np.ptp([x[1] for x in self.rows], axis=0)))
                angle = float(Rotation.from_matrix(np.array([x[2] for x in self.rows]) @ self.rows[0][2].T).magnitude().max())
                if span < 0.0002 and angle < np.deg2rad(0.2):
                    self.ready = True
                    s.events['machine_outfeed_ready_s'] = t
                    s.events['machine_outfeed_ready_window'] = dict(time_s=t, samples=len(self.rows), position_span_m=span, orientation_from_first_rad=angle)
        if 'released_tray_retention_s' in s.events:
            self.signal('green', t)
        elif s.entry_at is not None:
            self.signal('red', t)
