"""Verify the Dual-X5 initial states and write data/household/x5_initial/MANIFEST.json (static checks only).

Checks per task: required files, image size/contrast, no absolute paths in the scene layer,
all USD/texture dependencies resolve inside the package (renderer MDL modules excepted),
no active rigid bodies or joints in the composed static stage, paused-capture clock records.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdPhysics, UsdUtils

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO/'data/household/x5_initial'
PKG = REPO  # every dependency must resolve inside the repository (kitchen via link_x5_kitchen.sh)
SRC = REPO/'src/roboclock/envs/household'
BUILTIN_MDL = {'OmniPBR.mdl', 'OmniGlass.mdl', 'OmniSurface.mdl', 'OmniSurfaceBase.mdl'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


tasks, problems = [], []
for d in sorted(OUT.iterdir()):
    if not (d/'initial_scene.usda').exists():
        continue
    row = dict(task=d.name, files={})
    for f in ['initial_scene.usda', 'initial_state.json', 'overview.png', 'agent_view.png']:
        p = d/f
        if not p.exists():
            problems.append(f'{d.name}: missing {f}')
            continue
        row['files'][f] = sha(p)
    for img in ['overview.png', 'agent_view.png']:
        a = np.asarray(Image.open(d/img).convert('RGB'))
        if a.shape != (960, 1280, 3) or a.std() < 5:
            problems.append(f'{d.name}: bad image {img}')
    text = (d/'initial_scene.usda').read_text()
    import re
    abs_assets = [m for m in re.findall(r'@([^@]+)@', text) if m.startswith('/')]
    if abs_assets:
        problems.append(f'{d.name}: absolute asset paths in scene layer: {abs_assets[:3]}')
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(d/'initial_scene.usda'))
    kitchen = (SRC/'assets/x5/lightwheel_kitchen/Collected_KitchenRoom').resolve()

    def inside(a):
        return str(Path(a).resolve()).startswith((str(PKG.resolve()), str(kitchen)))
    outside = [str(a) for a in list(assets)+[layer.realPath for layer in layers] if a and not inside(a)]
    unresolved = [Path(u).name for u in unresolved if Path(u).name not in BUILTIN_MDL]
    row['dependencies'] = dict(layers=len(layers), assets=len(assets), outside_package=outside, unresolved=unresolved)
    if outside:
        problems.append(f'{d.name}: {len(outside)} dependencies outside package')
    stage = Usd.Stage.Open(str(d/'initial_scene.usda'))
    rigid = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    row['static_stage'] = dict(active_rigid_bodies=len(rigid), active_joints=len(joints))
    if rigid or joints:
        problems.append(f'{d.name}: active physics in static stage')
    info = json.loads((d/'initial_state.json').read_text())
    r = info.get('render', {})
    shots = r.get('shots', [])
    row['render'] = dict(simulator=r.get('simulator'), renderer=r.get('renderer'),
                         clock_held=all(s['timeline_s'] == 0 and s['world_time_s'] == shots[0]['world_time_s'] for s in shots) if shots else False,
                         image_sha256_match={s['image']: s['sha256'] == row['files'].get(s['image']) for s in shots})
    if not shots or not all(row['render']['image_sha256_match'].values()):
        problems.append(f'{d.name}: render record does not match delivered images')
    row['operation_points'] = len(info.get('operation_points', []))
    tasks.append(row)
reach = json.loads((OUT/'REACH_CHECK.json').read_text())
reach_summary = {t: f"{sum(r['reachable'] for r in rows)}/{len(rows)}" for t, rows in reach['tasks'].items()}
src = {str(p.relative_to(REPO)): sha(p) for p in sorted([*SRC.glob('x5/*.py'), *SRC.glob('*_x5/*.py')])}
manifest = dict(package='household Dual-X5 initial states, revision R2', date='2026-10-21', tasks=tasks, reach_check=reach_summary,
                source_sha256=src, problems=problems,
                scope='static initial states and task contracts; no process model, controller, physics episode or video executed')
(OUT/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('TASKS', len(tasks), 'PROBLEMS', problems or 'none', 'REACH', reach_summary)
