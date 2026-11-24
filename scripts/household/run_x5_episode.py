# ruff: noqa: E402  (Isaac Sim modules must be imported after SimulationApp starts)
"""Run one Dual-X5 household physics episode with the privileged oracle controller.

    ISAAC_PYTHON scripts/household/run_x5_episode.py --task A1 --out RUN_DIR [--record]

Without --record the episode runs physics only (no camera, no render call): this is the
ground-truth validation pass. With --record the same frozen configuration is re-run with RTX
rendering (RoboDojo rig spec: RaytracedLighting + DLAA + Isaac Lab's quality preset) and four
videos are written, frame-synchronized with equal frame counts: the policy cameras head.mp4,
left_wrist.mp4, right_wrist.mp4 (640x480, exactly what the policy observes) and the labelled
third-person overview <TASK>.mp4 (1280x960, policy views as insets). Physics 250 Hz, control
50 Hz, video 25 Hz (one render per 25 Hz policy observation; while the oracle only waits on the
process, 1 frame in TIMELAPSE is kept in all four videos, and frames.json lists each frame's
simulation time). The rig is measured and checked at reset (roboclock.envs.robodojo_rig.check_rig,
wrist mounts on link6). Tool poses are written only at reset; no attachment or teleport is used.

The task outcome comes from the R2 task model (``<id>_x5/task.py``) driven by measured signals.
"""
import argparse
import faulthandler
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
p = argparse.ArgumentParser()
p.add_argument('--task', required=True)
p.add_argument('--out', required=True)
p.add_argument('--difficulty', default='medium')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--seconds', type=float, default=None, help='episode budget (default: task max_time)')
p.add_argument('--record', action='store_true')
p.add_argument('--spec', action='append', default=[], help='diagnostic spec override key=value (not for reported runs)')
p.add_argument('--oracle-param', action='append', default=[],
               help='override a module-level oracle constant NAME=VALUE (e.g. A2 B_RETRIEVE=.85 for a late-retrieval clip; not for reported runs)')
p.add_argument('--initial', default=None, help='initial_scene.usda (default: data/household/x5_initial/<task>/)')
p.add_argument('--settle', type=float, default=3.0, help='seconds of passive settling after reset, before t = 0')
p.add_argument('--cache', default=os.environ.get('ROBOCLOCK_RENDER_CACHE', '/opt/household/household_current/render_cache'))
a = p.parse_args()
OUT = Path(a.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
if a.record and ((OUT/f'{a.task}.mp4').exists() or (OUT/'head.mp4').exists()):
    raise SystemExit(f'{OUT} already holds a recording; choose a fresh output directory')
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
cache = Path(a.cache)
os.environ.update(OMNI_KIT_ACCEPT_EULA='YES', OMNI_KIT_ALLOW_ROOT='1', XDG_CACHE_HOME=str(cache/'xdg'), TMPDIR=str(cache/'tmp'))
Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)


def _fail(kind, value, tb):
    sys.__excepthook__(kind, value, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _fail
from roboclock.envs.household.x5.graphics_diagnostics import select_headless_driver

select_headless_driver()
from isaacsim import SimulationApp

W, HGT = 1280, 960
app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'width': W, 'height': HGT, 'multi_gpu': False,
                     'limit_cpu_threads': 4, 'sync_loads': False, 'disable_viewport_updates': True, 'create_new_stage': False,
                     'extra_args': ['--allow-root', '--portable-root', str(cache/'kit'), '--/app/renderer/waitIdle=false',
                                    # Kit 110 can omit inactive RTX modes at startup (black buffers if selected later)
                                    '--/persistent/rtx/modes/rt/enabled=true', '--/persistent/rtx/modes/rt2/enabled=true',
                                    '--/persistent/rtx/modes/pt/enabled=true']})
import importlib.metadata

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from PIL import Image, ImageDraw

from roboclock.envs.household.tasks.registry import make_task
from roboclock.envs import robodojo_rig as RD
from roboclock.envs.household.x5 import rig
from roboclock.envs.household.x5 import robodojo as RDH
from roboclock.envs.household.x5.scene_lib import READY_Q
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import Servo
from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ, VIDEO_FPS

CONTROL_EVERY, VIDEO_EVERY, TRACE_EVERY = PHYSICS_HZ//CONTROL_HZ, PHYSICS_HZ//VIDEO_FPS, PHYSICS_HZ//VIDEO_FPS
TAIL = 2.5
TIMELAPSE = 8  # while the oracle only waits on the process, video keeps 1 frame in TIMELAPSE (labelled)
STILL = .03  # rad/s (m/s for the jaws): faster joints keep every frame, a moving arm would appear to jump
DT = 1/PHYSICS_HZ
settings = carb.settings.get_settings()
for k, v in {'/omni/replicator/captureOnPlay': False, '/app/asyncRendering': False, '/rtx/post/histogram/enabled': False,
             '/rtx/ecoMode/enabled': False,
             '/rtx/raytracing/fractionalCutoutOpacity': True, '/rtx/pathtracing/fractionalCutoutOpacity': True}.items():
    settings.set(k, v)
renderer = RDH.apply_renderer() if a.record else None  # RaytracedLighting + DLAA + quality preset

overrides = {k: float(v) for k, v in (x.split('=', 1) for x in a.spec)}
task = make_task(a.task, a.difficulty, a.seed, **overrides)
rig.use(a.task)  # this task's arm base layout (shared BASES), before the overlay and servos
module = importlib.import_module(os.environ.get('X5_SIM_MODULE') or f'roboclock.envs.household.{a.task.lower()}_x5.sim')  # override: diagnostics only
oracle_params = {k: float(v) for k, v in (x.split('=', 1) for x in a.oracle_param)}
for k, v in oracle_params.items():
    if not isinstance(getattr(module, k, None), (int, float)):
        raise SystemExit(f'--oracle-param: {module.__name__} has no numeric constant {k}')
    setattr(module, k, v)
initial = Path(a.initial) if a.initial else REPO/'data/household/x5_initial'/a.task/'initial_scene.usda'
scene_path, setup = P.overlay(initial, OUT, module.build)
omni.usd.get_context().open_stage(str(scene_path))
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1., physics_dt=DT, rendering_dt=DT)
if a.record:
    # Write physics results to USD so the RTX renderer always sees them. With the Fabric-only path some
    # recordings rendered physics-driven prims frozen at reset while plain USD edits (cue colours) showed.
    world.get_physics_context().enable_fabric(False)
    settings.set('/physics/updateToUsd', True)
robots = {side: world.scene.add(SingleArticulation(f'{P.ARMS}/{side}', name=side, reset_xform_properties=False)) for side in ['Left', 'Right']}
events = []


def event(name, t, **kw):
    row = dict(event=name, t_s=float(t), **kw)
    events.append(row)
    print('EVENT', json.dumps(row), flush=True)


oracle = module.Oracle(task, event)
oracle.setup(world, stage)  # task rigid-body/contact views, before the first reset
cams, policy_cams, wrist_paths, intrinsics = {}, {}, {}, None
if a.record:  # RoboDojo policy cameras (head + a D435 on each link6) and the third-person overview
    cams = {'overview': Camera('/World/Camera', resolution=(W, HGT))}
    wrist_paths = RDH.add_wrist_cameras(stage, P.ARMS)
    policy_cams = RDH.make_cameras(wrist_paths)
world.reset()
world.pause()
for c in cams.values():
    c.initialize()
if policy_cams:
    intrinsics = RDH.configure_cameras(policy_cams)
for side, robot in robots.items():
    q = robot.get_joint_positions()
    names = list(robot.dof_names)
    for i in range(6):
        q[names.index(f'joint{i+1}')] = READY_Q[i]
    for name in ['jawP_joint', 'jawN_joint']:
        q[names.index(name)] = 0.
    robot.set_joint_positions(q)
    robot.set_joint_velocities(np.zeros_like(q))
    robot.apply_action(ArticulationAction(joint_positions=q))
    print('ARTICULATION', side, names, flush=True)
servos = {side: Servo(robot, side, getattr(module, 'CONTROL', {}).get(side.lower(), 'ik'), control_dt=DT*CONTROL_EVERY)
          for side, robot in robots.items()}
oracle.servos = servos
world.play()
for i in range(int(a.settle*PHYSICS_HZ)):  # passive settling (task clock not started)
    world.step(render=False)
rig_check = RDH.json_ready(RDH.rig_check(stage, robots, wrist_paths))  # raises if the rig is off spec
for s in servos.values():
    s.reset_target()


def arms_still():
    return all(np.abs(robot.get_joint_velocities()).max() < STILL for robot in robots.values())


def capture(path):
    """Still capture of the overview camera without advancing physics."""
    if not cams:
        return None
    before = float(world.current_time)
    rgb = np.empty(0)
    for _ in range(40):
        rep.orchestrator.step(delta_time=0., rt_subframes=32, pause_timeline=False)
        rgb = np.asarray(cams['overview'].get_rgba())
        if rgb.ndim == 3 and rgb[:, :, :3].std() > 5:
            break
        time.sleep(.2)
    if abs(float(world.current_time)-before) > 1e-9:
        raise RuntimeError('still capture advanced physics')
    if rgb.ndim == 3 and rgb[:, :, :3].std() > 5:
        Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(path)
        for view, cam in policy_cams.items():  # the policy views of the same instant
            img = RDH.frame_rgb(cam)
            if img is not None:
                Image.fromarray(img).save(path.with_name(f'{path.stem}_{RDH.VIEWS[view]}.png'))
        return str(path)
    return None


render_check = {}


def frame(t, speedup=1):
    big = np.asarray(cams['overview'].get_rgba())[:, :, :3].astype(np.uint8)
    # guard: once an arm has moved > 10 cm the overview must differ from the first frame
    # (checked from t = 2 s: the path tracer needs a few frames before it shows motion started at t = 0)
    tcp = np.concatenate([s.pose()[0] for s in servos.values()])
    if 'tcp' not in render_check:
        render_check.update(tcp=tcp, img=big[::8, ::8].astype(float))
    elif t >= 2. and np.abs(tcp-render_check['tcp']).max() > .10 and not render_check.get('ok'):
        if np.abs(big[::8, ::8]-render_check['img']).mean() < .5:
            raise RuntimeError('rendered frames do not follow physics (overview unchanged after arm motion)')
        render_check['ok'] = True
    if big.shape != (HGT, W, 3) or big.std() < 5:
        raise RuntimeError('invalid overview RGB')
    views = {}
    for view, cam in policy_cams.items():
        rgb = RDH.frame_rgb(cam)
        if rgb is None or rgb.shape != (*RDH.image_shape(view), 3):
            raise RuntimeError(f'invalid {RDH.VIEWS[view]} RGB')
        views[view] = rgb
    img = Image.fromarray(big)
    # policy views as insets: head (W/3) above the two wrist cameras (W/6 each)
    img.paste(Image.fromarray(views['head']).resize((W//3, HGT//3)), (W-W//3-12, 40))
    for k, side in enumerate(('Left', 'Right')):
        img.paste(Image.fromarray(views[side]).resize((W//6, HGT//6)), (W-W//3-12+k*(W//6), 44+HGT//3))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 30], fill=(18, 18, 18))
    label = f'{a.task} | {task.name} | {a.difficulty} seed {a.seed} | simulation time {t:6.2f} s'
    d.text((12, 9), label+(f'   [waiting: {speedup}x time-lapse]' if speedup > 1 else ''), fill=(235, 235, 235))
    d.text((W-W//3-6, 44), 'head camera (rgb)', fill=(250, 250, 250))
    d.text((W-W//3-6, 48+HGT//3), 'left wrist', fill=(250, 250, 250))
    d.text((W-W//3-6+W//6, 48+HGT//3), 'right wrist', fill=(250, 250, 250))
    for view, enc in policy_encoders.items():  # the raw policy images, same frame
        enc.stdin.write(views[view].tobytes())
    if hasattr(oracle, 'video_overlay'):  # viewer-only annotation (e.g. A2 browning window); never in the agent image
        oracle.video_overlay(img, d, t)
    return np.asarray(img)


encoder = None
policy_encoders = {}
frames = 0
frame_times = []
POLICY_VIDEOS = {'head': 'head.mp4', 'Left': 'left_wrist.mp4', 'Right': 'right_wrist.mp4'}


def video_encoder(path, width, height, crf):
    return subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{width}x{height}', '-r', str(PHYSICS_HZ//VIDEO_EVERY), '-i', '-', '-an', '-c:v', 'libx264',
                             '-preset', 'fast', '-threads', '2', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                             str(path)], stdin=subprocess.PIPE, stderr=(OUT/f'encoder_{path.stem}.log').open('w'))


if a.record:
    world.pause()
    for _ in range(3):
        world.render()
    deadline = time.monotonic()+5.
    while time.monotonic() < deadline:
        world.render()
        time.sleep(.02)
    RDH.apply_renderer()  # opening the stage can reset RTX settings during the first frames
    for _ in range(3):
        world.render()
    renderer = RDH.renderer_readback()  # in effect after the first frames
    capture(OUT/'initial.png')
    world.play()
    encoder = video_encoder(OUT/'episode_partial.mp4', W, HGT, 20)
    for view, name in POLICY_VIDEOS.items():
        height, width = RDH.image_shape(view)
        policy_encoders[view] = video_encoder(OUT/f'partial_{name}', width, height, 18)
budget = a.seconds or task.spec.max_time
if getattr(module, 'FIXED_HORIZON', False):  # a last scored event at the horizon itself (C2's 360 s pickup) must be
    budget += .5                             # reached; summed steps end a hair short of it. The oracle stops the run.
start = float(world.current_time)
trace, failures, wall = [], [], time.monotonic()
stop = {'flag': False}
signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
after_done = None
try:
    for tick in range(int(budget*PHYSICS_HZ)+1):
        if stop['flag']:
            failures.append('interrupted')
            break
        t = float(world.current_time)-start
        if hasattr(oracle, 'appliance_step'):  # appliance mechanisms (e.g. A2 door-rack linkage), not control
            oracle.appliance_step(stage)
        signals = oracle.measure(t, DT)
        task.update(signals)
        if tick:
            task.advance(DT)
        if hasattr(oracle, 'update_cues') and tick % VIDEO_EVERY == 0:
            oracle.update_cues()
        if tick % CONTROL_EVERY == 0:
            oracle.command(t)
        if tick % TRACE_EVERY == 0:
            trace.append(dict(t=t, signals=signals, process=task.cues(), **oracle.trace(),
                              tcp={k: dict(position=v.pose()[0].tolist(), quaternion_xyzw=v.pose()[1].as_quat().tolist())
                                   for k, v in servos.items()},
                              joints={k: r.get_joint_positions().tolist() for k, r in robots.items()}))
            if tick % (PHYSICS_HZ*2) == 0:
                print('STATE', json.dumps(dict(t=round(t, 2), phase=trace[-1].get('phase'), process=task.cues(),
                                               wall_s=round(time.monotonic()-wall, 1))), flush=True)
            if tick % (PHYSICS_HZ*10) == 0:  # checkpoint: a native crash still leaves the samples so far
                (OUT/'trajectory.partial.json').write_text(json.dumps(trace[-150:]))
        if oracle.failures:
            failures.extend(oracle.failures)
            break
        if task.terminated and after_done is None:
            after_done = t
            event('task_terminated', t, metrics_success=task.joint_success)
        if getattr(oracle, 'finished', False):  # fixed-horizon tasks: last scored event passed, robot parked
            break
        if after_done is not None and ((oracle.done and t-after_done >= TAIL) or t-after_done > 8.):
            break  # short settle tail after termination: released objects stay at rest on camera
        every = VIDEO_EVERY*(TIMELAPSE if a.record and getattr(oracle, 'idle', False) and arms_still() else 1)
        render = a.record and tick % every == 0
        world.step(render=False)
        if render:
            # A rendered world.step() leaves the camera on an older image (the picture changed only every third
            # video frame). This blocking zero-time capture shows the current state and does not step physics.
            before = float(world.current_time)
            rep.orchestrator.step(delta_time=0., rt_subframes=getattr(oracle, 'subframes', 1), pause_timeline=False)
            if float(world.current_time) != before:
                raise RuntimeError('video capture advanced physics')
            encoder.stdin.write(frame(t, every//VIDEO_EVERY).tobytes())
            frame_times.append(round(float(world.current_time)-start, 6))
            frames += 1
    else:
        if not getattr(module, 'FIXED_HORIZON', False):
            failures.append('episode_budget_exhausted')
except Exception as exc:  # keep the trace of a failed attempt
    failures.append(f'{type(exc).__name__}: {exc}')
    import traceback
    traceback.print_exc()
finally:
    for enc in ([encoder] if encoder else [])+list(policy_encoders.values()):
        enc.stdin.close()
        enc.wait(timeout=300)
        if enc.returncode:
            failures.append('encoder_failed')
    world.pause()
    final = capture(OUT/'final.png') if a.record else None
    metrics = task.metrics()
    ok = bool(metrics['joint_success'] and not failures and not metrics['physical_failures'])
    code = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__), Path(module.__file__), Path(P.__file__), REPO/'src/roboclock/envs/household/x5/sim/servo.py',
             REPO/f'src/roboclock/envs/household/{a.task.lower()}_x5/task.py']}
    result = dict(task=a.task, difficulty=a.difficulty, seed=a.seed, spec_overrides=overrides, oracle_params=oracle_params, recorded=a.record, success=ok, metrics=metrics,
                  oracle_failures=failures, oracle_phases=getattr(oracle, 'phase', None), events=events,
                  sim_duration_s=float(world.current_time)-start, wall_s=time.monotonic()-wall, video_frames=frames,
                  rates_hz=dict(physics=PHYSICS_HZ, control=PHYSICS_HZ//CONTROL_EVERY, video=PHYSICS_HZ//VIDEO_EVERY, trace=PHYSICS_HZ//TRACE_EVERY,
                                policy=RD.POLICY_HZ, render=RD.RENDER_HZ if a.record else 0),
                  robodojo_rig=RD.describe(), robodojo_rig_check=rig_check, renderer=renderer, camera_intrinsics=intrinsics,
                  camera_intrinsics_rendered=None if intrinsics is None else RDH.rendered_intrinsics_report(),
                  videos=(dict(overview=f'{a.task}.mp4', **{RDH.VIEWS[v]: n for v, n in POLICY_VIDEOS.items()},
                               frames='frames.json', overview_size=[W, HGT],
                               policy_size=list(RD.HEAD_CAMERA['resolution']), fps=VIDEO_FPS) if a.record else None),
                  simulator=importlib.metadata.version('isaacsim'), setup=setup, code_sha256=code,
                  rig=dict(bases_xy_m={k: list(v) for k, v in rig.BASES.items()}, base_yaw_deg=rig.BASE_YAW_DEG,
                           mount_top_z_m=rig.MOUNT_TOP, common_rig=rig.on_common_rig(a.task),
                           head_camera={k: v for k, v in rig.head_camera(rig.BASES).items() if k != 'matrix'} if rig.on_common_rig(a.task) else None),
                  initial_scene_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(), final_image=final,
                  scope='privileged oracle controller; fixed initial scene; single episode')
    (OUT/'trajectory.json').write_text(json.dumps(trace))
    (OUT/'result.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    if encoder and encoder.returncode == 0:
        (OUT/'episode_partial.mp4').replace(OUT/(f'{a.task}.mp4' if ok else f'failed_{a.task}.mp4'))
        for view, name in POLICY_VIDEOS.items():
            (OUT/f'partial_{name}').replace(OUT/(name if ok else f'failed_{name}'))
        (OUT/'frames.json').write_text(json.dumps(dict(fps=VIDEO_FPS, frames=frames, sim_time_s=frame_times)))
    print('RESULT', json.dumps(dict(success=ok, failures=failures, metrics={k: metrics[k] for k in metrics if k != 'episode_spec'}), default=str), flush=True)
app.close()
