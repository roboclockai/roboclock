"""Build the static Dual-X5 initial scenes: shared kitchen/island/robot + each task's scene.py.

Outputs data/household/x5_initial/<ID>/initial_scene.usda and initial_state.json. Asset paths
inside the USD layers are relative, so the repository can be moved. No controller, physics
episode or process clock runs here.

Run with an OpenUSD Python (e.g. /opt/household/usd_static/bin/python):
    PYTHONPATH=src python -m roboclock.envs.household.x5.build [--tasks A1 B3]
"""

import argparse
import importlib
import json
from types import SimpleNamespace

from pxr import Gf, Usd, UsdGeom

from roboclock.envs.household.x5 import rig
from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import rc

ALL = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
T, K = L.TEXTURES, L.KITCHEN


def build(task):
    rig.use(task)  # this task's arm base layout (shared BASES)
    folder = L.OUTPUT/task
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder/'initial_scene.usda'
    if dest.exists():
        dest.unlink()
    L.SCENE_DIR = folder
    s = Usd.Stage.CreateNew(str(dest))
    s.SetDefaultPrim(UsdGeom.Xform.Define(s, '/World').GetPrim())
    UsdGeom.SetStageMetersPerUnit(s, 1.)
    UsdGeom.SetStageUpAxis(s, 'Z')
    background = L.kitchen(s)
    island = L.put(s, 'Island', K/'Kitchen_InsularShelf/Kitchen_InsularShelf.usd', [.30, 0], 0.)
    lo, hi = L.bound(island)
    op = UsdGeom.Xformable(island).GetOrderedXformOps()[0]
    v = op.Get()
    op.Set(Gf.Vec3d(v[0], v[1], v[2]+H-hi[2]))
    steel = L.material(s, 'MountSteel', [.26, .28, .29], .28, .85)
    for name, (x, y) in L.BASES.items():
        L.shape(s, 'Mount'+name, [x, y, H+.006], [.115, .12, .012], steel)
    emb = L.robot(s)
    audit, points, joints, notes, interior = [], [], {}, [], []

    def obj(name, source, xy, z, scale=1., rot=(0, 0, 0), prim=None, hide=(), override=None, role='', native=False):
        p = L.put(s, name, source, xy, z, scale, rot, prim, hide, override)
        lo, hi = L.bound(p)
        audit.append(dict(name=name, role=role, prim=str(p.GetPath()), source=L.repo_rel(source), source_prim=prim,
                          source_sha256=L.sha(source), uniform_scale=round(float(scale), 5), rotation_xyz_deg=list(rot),
                          bbox_min_m=lo.round(4).tolist(), bbox_max_m=hi.round(4).tolist(), size_m=(hi-lo).round(4).tolist()))
        if native:
            joints[name] = L.native_joints(source)
        return p

    def part_center(prim_path):
        lo, hi = L.bound(s.GetPrimAtPath(prim_path))
        return ((lo+hi)/2).round(4).tolist(), lo.round(4).tolist(), hi.round(4).tolist()

    def point(name, xyz, side, approaches, purpose):
        points.append(dict(name=name, xyz=[round(float(v), 4) for v in xyz], side=side, approaches=approaches, purpose=purpose))

    def authored(name, prim, role, source=None):
        lo, hi = L.bound(prim)
        row = dict(name=name, role=role, prim=str(prim.GetPath()), source=f'authored in {task.lower()}_x5/scene.py (disclosed proxy)')
        if source is not None:
            row.update(source=L.repo_rel(source), source_sha256=L.sha(source))
        audit.append(dict(row, bbox_min_m=lo.round(4).tolist(), bbox_max_m=hi.round(4).tolist(), size_m=(hi-lo).round(4).tolist()))

    def tex(name, file, rough=.6):
        return L.material(s, name, [1, 1, 1], rough, texture=T/file)

    def cooktop():
        return obj('Cooktop', K/'Stovetop012/Stovetop012.usd', [.30, .10], H, role='fixture: 5-burner gas cooktop', native=True)

    def flame_ring(name, xy, z, size):
        m = L.alpha_material(s, name+'Mat', T/'gas_flame_ring.png', emissive=True)
        L.card(s, name, [xy[0], xy[1], z], size, size, m)

    ctx = SimpleNamespace(s=s, obj=obj, point=point, authored=authored, part_center=part_center, tex=tex,
                          cooktop=cooktop, flame_ring=flame_ring, island=island, notes=notes, interior=interior, rc=rc,
                          wood=L.material(s, 'Wood', [.29, .18, .095], .62),
                          rubber=L.material(s, 'Rubber', [.03, .035, .035], .7))
    state, agent = importlib.import_module(f'roboclock.envs.household.{task.lower()}_x5.scene').build(ctx)

    L.lights(s, interior)
    cams = [L.camera(s, '/World/Camera', [.90, -2.12, 2.18], [.30, .27, 1.04], 34.)]
    if rig.on_common_rig(task):  # head camera fixed to the rig (scene.py's own view is not used)
        head = rig.head_camera(L.BASES)
        cams.append(dict(L.camera_at(s, '/World/AgentCamera', head['matrix'], head['focal'], head['h_ap'], head['v_ap'],
                                     head['clip']),
                         eye_m=head['eye'], look_at_m=head['look'],
                         mount=f'RoboDojo head camera (roboclock.envs.robodojo_rig): base midpoint + {rig.HEAD_FORWARD} m '
                               f'forward, {rig.HEAD_HEIGHT} m above the mount tops, pitched {rig.HEAD_PITCH_DEG:g} deg down, '
                               f'Gemini 345Lg {head["resolution"][0]}x{head["resolution"][1]}'))
    elif agent:
        cams.append(L.camera(s, '/World/AgentCamera', agent[0], agent[1], agent[2]))
    for item in audit:
        pp = s.GetPrimAtPath(item['prim'])
        if pp:
            lo, hi = L.bound(pp)
            item.update(bbox_min_m=lo.round(4).tolist(), bbox_max_m=hi.round(4).tolist(), size_m=(hi-lo).round(4).tolist())
    s.GetRootLayer().Save()
    info = dict(task=task, revision='R2 (2026-10-21)', scope='STATIC INITIAL STATE ONLY',
                world_frame='meters; z up; island counter top z=0.8584; +y toward the back wall; robots face +y',
                scene_file='initial_scene.usda', images=dict(overview='overview.png', agent_view='agent_view.png'),
                task_module=f'roboclock.envs.household.{task.lower()}_x5.task',
                task_clock_s=0., controller_executed=False, process_clock_advanced=False, physics_steps=0,
                embodiment=emb, background=background, objects=audit, native_joints=joints,
                operation_points=points, initial_state=state, cameras=cams,
                notes=notes+['Static stage: source joints and rigid-body APIs are deactivated by local overrides; '
                             're-enable them (or rebuild with physics) in the simulation stage.'])
    (folder/'initial_state.json').write_text(json.dumps(info, indent=2)+'\n')
    print('BUILT', task, len(audit), 'objects', len(points), 'operation points', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', choices=ALL, default=ALL)
    for t in ap.parse_args().tasks:
        build(t)
