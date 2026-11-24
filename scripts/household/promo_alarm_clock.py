# ruff: noqa: E402  (Isaac Sim modules must be imported after SimulationApp starts)
"""Render the alarm-clock promo clip: a twin-bell alarm clock rings, the X5 presses its stop button, it falls silent.

    LD_LIBRARY_PATH=/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu PATH=<ffmpeg dir>:$PATH \
        ISAAC_PYTHON scripts/household/promo_alarm_clock.py --out OUT_DIR [--ring 2.0] [--tail 2.5]

Not a benchmark task: the scene is ``roboclock.envs.household.alarm_x5`` (clock near the counter's front edge, 0.40 m
left of the right arm) and the motion is its privileged oracle on real physics (250 Hz, control 50 Hz), exactly as
the household recorder runs an episode. One camera, level with the counter and facing the dial, clock in the left
foreground and the right arm on the right; the left arm is hidden from the picture (it stays at its ready pose).
Writes OUT_DIR/alarm_clock_promo.mp4 (1920x1080, 25 fps, real time, no overlays), first.png / ring.png / press.png /
last.png stills and result.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--ring', type=float, default=2.0, help='seconds before the alarm starts')
p.add_argument('--tail', type=float, default=2.5, help='seconds kept after the arm is back')
p.add_argument('--eye', type=float, nargs=3, default=None, help='camera position (world m)')
p.add_argument('--look', type=float, nargs=3, default=None, help='camera target (world m)')
p.add_argument('--focal', type=float, default=34., help='focal length (mm) on a 36 x 20.25 mm aperture')
p.add_argument('--fstop', type=float, default=0., help='lens aperture for depth of field, focused on the clock (0: everything sharp)')
p.add_argument('--subframes', type=int, default=4, help='RTX subframes per video frame')
p.add_argument('--still-only', action='store_true', help='render first.png only (framing check)')
p.add_argument('--cache', default=os.environ.get('ROBOCLOCK_RENDER_CACHE', '/opt/household/household_current/render_cache'))
a = p.parse_args()
OUT = Path(a.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
cache = Path(a.cache)
os.environ.update(OMNI_KIT_ACCEPT_EULA='YES', OMNI_KIT_ALLOW_ROOT='1', XDG_CACHE_HOME=str(cache/'xdg'), TMPDIR=str(cache/'tmp'))
Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
from roboclock.envs.household.x5.graphics_diagnostics import select_headless_driver

select_headless_driver()
from isaacsim import SimulationApp

W, HGT, FPS = 1920, 1080, 25
app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'width': W, 'height': HGT, 'multi_gpu': False,
                     'limit_cpu_threads': 8, 'sync_loads': False, 'disable_viewport_updates': True, 'create_new_stage': False,
                     'extra_args': ['--allow-root', '--portable-root', str(cache/'kit'), '--/app/renderer/waitIdle=false',
                                    '--/persistent/rtx/modes/rt/enabled=true', '--/persistent/rtx/modes/rt2/enabled=true',
                                    '--/persistent/rtx/modes/pt/enabled=true']})
import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from PIL import Image
from pxr import UsdGeom

from roboclock.envs.household.alarm_x5 import sim as module
from roboclock.envs.household.alarm_x5.scene import CLOCK_XY
from roboclock.envs.household.alarm_x5.task import AlarmClock, AlarmSpec
from roboclock.envs.household.x5 import rig
from roboclock.envs.household.x5 import robodojo as RDH
from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.scenes import build_initial_scene
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import Servo
from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ

DT, CONTROL_EVERY, VIDEO_EVERY = 1/PHYSICS_HZ, PHYSICS_HZ//CONTROL_HZ, PHYSICS_HZ//FPS
settings = carb.settings.get_settings()
for k, v in {'/omni/replicator/captureOnPlay': False, '/app/asyncRendering': False, '/rtx/post/histogram/enabled': False,
             '/rtx/ecoMode/enabled': False}.items():
    settings.set(k, v)
RDH.apply_renderer()

task = AlarmClock(AlarmSpec.sample('ALARM', 'easy', 0, ring_time=a.ring))
rig.use('ALARM')
# Scene files three folders below a directory directly under the repository (relative asset paths resolve there).
initial = build_initial_scene('ALARM', REPO/'results/household/x5_initial')
scene_path, _ = P.overlay(initial, OUT, module.build)
omni.usd.get_context().open_stage(str(scene_path))
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1., physics_dt=DT, rendering_dt=DT)
world.get_physics_context().enable_fabric(False)  # physics results written to USD, so the renderer always sees them
settings.set('/physics/updateToUsd', True)
robots = {side: world.scene.add(SingleArticulation(f'{P.ARMS}/{side}', name=side, reset_xform_properties=False))
          for side in ('Left', 'Right')}
events = []
oracle = module.Oracle(task, lambda name, t, **kw: events.append(dict(event=name, t_s=float(t), **kw)))
oracle.setup(world, stage)
# Camera level with the clock, facing its dial; the clock a third of the way in from the left, the right arm on the right.
eye = a.eye or [CLOCK_XY[0]+.17, CLOCK_XY[1]-.80, H+.12]
look = a.look or [eye[0], CLOCK_XY[1], H+.17]
L.camera(stage, '/World/PromoCamera', eye, look, a.focal, 36., 20.25)
if a.fstop > 0:
    lens = UsdGeom.Camera(stage.GetPrimAtPath('/World/PromoCamera'))
    lens.CreateFStopAttr(a.fstop)
    lens.CreateFocusDistanceAttr(float(np.linalg.norm(np.array(eye)-[CLOCK_XY[0], CLOCK_XY[1]-.04, H+.065])))
    settings.set('/rtx/post/dof/enabled', True)
for path in (f'{P.ARMS}/Left', '/World/Objects/MountLeft'):  # single-arm picture
    UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
cam = Camera('/World/PromoCamera', resolution=(W, HGT))
world.reset()
world.pause()
cam.initialize()
for side, robot in robots.items():
    q = robot.get_joint_positions()
    names = list(robot.dof_names)
    for i in range(6):
        q[names.index(f'joint{i+1}')] = READY_Q[i]
    for name in ('jawP_joint', 'jawN_joint'):
        q[names.index(name)] = 0.
    robot.set_joint_positions(q)
    robot.set_joint_velocities(np.zeros_like(q))
    robot.apply_action(ArticulationAction(joint_positions=q))
servos = {side: Servo(robot, side, module.CONTROL.get(side.lower(), 'ik'), control_dt=DT*CONTROL_EVERY) for side, robot in robots.items()}
oracle.servos = servos
world.play()
for _ in range(int(3.*PHYSICS_HZ)):  # passive settling before t = 0
    world.step(render=False)
for s in servos.values():
    s.reset_target()


def shot(subframes):
    before = float(world.current_time)
    rep.orchestrator.step(delta_time=0., rt_subframes=subframes, pause_timeline=False)
    if float(world.current_time) != before:
        raise RuntimeError('capture advanced physics')
    return np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8)


deadline = time.monotonic()+900.  # first frames: RTX pipeline compile
while True:
    world.render()
    rgb = shot(8)
    if (rgb.ndim == 3 and rgb.std() > 5) or time.monotonic() > deadline:
        break
    time.sleep(.2)
RDH.apply_renderer()  # opening the stage can reset RTX settings during the first frames
for _ in range(4):
    rgb = shot(16)
if rgb.shape != (HGT, W, 3) or rgb.std() < 5:
    raise SystemExit('promo camera returned no image')
Image.fromarray(shot(64)).save(OUT/'first.png')
if a.still_only:
    print('RESULT', json.dumps(dict(still=str(OUT/'first.png'), eye=eye, look=look, focal=a.focal)), flush=True)
    app.close()
    sys.exit(0)

video = OUT/'alarm_clock_promo.mp4'
encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{HGT}',
                            '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE, stderr=(OUT/'encoder.log').open('w'))
start, frames, done_at, failures, stills = float(world.current_time), 0, None, [], {}
try:
    for tick in range(int(30*PHYSICS_HZ)):
        t = float(world.current_time)-start
        oracle.appliance_step(stage)
        task.update(oracle.measure(t, DT))
        if tick:
            task.advance(DT)
        if tick % VIDEO_EVERY == 0:
            oracle.update_cues()
        if tick % CONTROL_EVERY == 0:
            oracle.command(t)
        if oracle.failures:
            failures.extend(oracle.failures)
            break
        if oracle.done and done_at is None:
            done_at = t
        if done_at is not None and t-done_at >= a.tail:
            break
        world.step(render=False)
        if tick % VIDEO_EVERY == 0:
            rgb = shot(a.subframes)
            encoder.stdin.write(rgb.tobytes())
            frames += 1
            for name, when in (('ring', a.ring+.5), ('press', task.events.get('silenced', 1e9)+.04)):
                if name not in stills and t >= when:
                    stills[name] = t
                    Image.fromarray(rgb).save(OUT/f'{name}.png')
    else:
        failures.append('time_budget_exhausted')
finally:
    encoder.stdin.close()
    encoder.wait(timeout=600)
Image.fromarray(shot(64)).save(OUT/'last.png')
m = task.metrics()
result = dict(success=bool(m['task_success'] and not failures), failures=failures, frames=frames, seconds=frames/FPS, video=str(video),
              size=[W, HGT], fps=FPS, camera=dict(eye=eye, look=look, focal_mm=a.focal, aperture_mm=[36., 20.25]),
              ring_s=m['ring_time'], silenced_s=m['silenced_time'], events=events, stills=stills,
              clock_max_tilt_deg=m.get('clock_tilt_deg'), physical_failures=m['physical_failures'])
(OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print('RESULT', json.dumps({k: result[k] for k in ('success', 'failures', 'frames', 'seconds', 'ring_s', 'silenced_s')}), flush=True)
app.close()
