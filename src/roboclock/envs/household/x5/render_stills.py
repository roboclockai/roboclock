# ruff: noqa: E402  (Isaac Sim modules must be imported after SimulationApp starts)
"""Capture Isaac Sim RTX still images from cameras authored in a static USD scene.

Usage: python render_stills.py --scene S.usda --shot /World/Camera=out.png [--shot ...]
The timeline is never advanced during capture; passive warm-up render calls only.
Scenes are static appearance stages (no rigid bodies, joints, controllers or processes).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--scene', required=True)
p.add_argument('--shot', action='append', required=True, help='camera_prim=output.png')
p.add_argument('--width', type=int, default=1280)
p.add_argument('--height', type=int, default=960)
p.add_argument('--subframes', type=int, default=32)
p.add_argument('--warmup', type=float, default=5., help='seconds of passive render warm-up (material/texture streaming)')
p.add_argument('--cache', default=os.environ.get('ROBOCLOCK_RENDER_CACHE', '/opt/household/household_current/render_cache'))
a = p.parse_args()
cache = Path(a.cache)
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
os.environ['OMNI_KIT_ALLOW_ROOT'] = '1'
os.environ['XDG_CACHE_HOME'] = str(cache/'xdg')
os.environ['TMPDIR'] = str(cache/'tmp')
Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
try:
    from graphics_diagnostics import select_headless_driver
    select_headless_driver()
except ImportError:
    pass
from isaacsim import SimulationApp

app = SimulationApp({'headless': True, 'renderer': 'RealTimePathTracing', 'width': a.width, 'height': a.height,
                     'multi_gpu': False, 'limit_cpu_threads': 4, 'sync_loads': False, 'disable_viewport_updates': True,
                     'create_new_stage': False,
                     'extra_args': ['--allow-root', '--portable-root', str(cache/'kit'), '--/app/renderer/waitIdle=false']})
import importlib.metadata

import carb
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from PIL import Image

from roboclock.envs.rates import PHYSICS_HZ

version = importlib.metadata.version('isaacsim')
settings = carb.settings.get_settings()
settings.set('/omni/replicator/captureOnPlay', False)
settings.set('/app/asyncRendering', False)
for k, v in {'/rtx/post/histogram/enabled': False, '/rtx/ecoMode/enabled': False,
             '/rtx/post/aa/op': 4, '/rtx/post/dlss/execMode': 2,
             '/rtx/raytracing/fractionalCutoutOpacity': True, '/rtx/pathtracing/fractionalCutoutOpacity': True}.items():
    settings.set(k, v)
omni.usd.get_context().open_stage(str(Path(a.scene).resolve()))
timeline = omni.timeline.get_timeline_interface()
timeline.stop()
timeline.set_current_time(0.)
world = World(stage_units_in_meters=1., physics_dt=1/PHYSICS_HZ, rendering_dt=1/PHYSICS_HZ)
shots = [s.split('=', 1) for s in a.shot]
cams = {}
for prim, _ in shots:
    cams[prim] = Camera(prim, resolution=(a.width, a.height))
world.reset()
world.pause()
for c in cams.values():
    c.initialize()
deadline = time.monotonic()+a.warmup
while time.monotonic() < deadline:
    world.render()
    time.sleep(.02)
results = []
for prim, out in shots:
    cam = cams[prim]
    before, wbefore = timeline.get_current_time(), float(world.current_time)
    rgb = np.empty(0)
    deadline = time.monotonic()+20.
    while time.monotonic() < deadline:
        rep.orchestrator.step(delta_time=0., rt_subframes=a.subframes, pause_timeline=False)
        rgb = np.asarray(cam.get_rgba())
        if rgb.ndim == 3 and rgb[:, :, :3].std() >= 5:
            break
        world.render()
        time.sleep(.2)
    if rgb.ndim != 3 or rgb[:, :, :3].std() < 5:
        raise RuntimeError('Camera RGB unavailable: '+prim)
    after, wafter = timeline.get_current_time(), float(world.current_time)
    if abs(after-before) > 1e-9 or abs(wafter-wbefore) > 1e-9:
        raise RuntimeError('Capture advanced the clock')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(out)
    results.append(dict(camera=prim, image=str(out), sha256=hashlib.sha256(Path(out).read_bytes()).hexdigest(),
                        rgb_std=float(rgb[:, :, :3].std()), timeline_s=after, world_time_s=wafter))
    print('RENDERED', prim, out, flush=True)
print('RENDER_RESULT', json.dumps(dict(scene=a.scene, simulator=version, renderer='RealTimePathTracing',
                                       resolution=[a.width, a.height], shots=results)), flush=True)
world.clear_instance()
app.close()
