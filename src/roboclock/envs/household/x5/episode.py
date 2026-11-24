"""Isaac Sim side of the Dual-X5 household environment; import only after SimulationApp starts.

One ``X5Episode`` is one opened simulation stage, built exactly as the standalone recorder
``scripts/household/run_x5_episode.py`` built it: the task's initial scene, the shared physics
overlay (``x5/sim/physics.py``: articulated X5 arms, island collider, contact material) plus the
task's own colliders and rigid bodies (``<id>_x5/sim.py:build``), arms set to the ready pose,
then 3 s of passive settling before the task clock starts.

The task module's ``Oracle`` class is used here only as the scene probe: ``measure()`` turns
measured physics (rigid-body poses, contact forces, joint and tool poses) into the task's
signals, ``appliance_step()`` runs appliance mechanisms (oven rack linkage, appliance
buttons, customer pickups) and ``update_cues()`` drives the visible process cues. The task model
scores from those signals only. The same object's ``command()`` is the privileged controller,
which only the diagnostic oracle policy (``x5/oracle.py``) calls.

Per physics tick, in the recorder's order: physics step, appliance mechanisms, measurement,
``task.update(signals)``, ``task.advance(dt)``, cue update every ``cue_interval_ticks``.

Cameras and renderer follow the RoboDojo rig spec (``x5/robodojo.py``): head ``rgb`` and the two
wrist cameras on ``link6`` (``left_wrist_rgb``, ``right_wrist_rgb``), all 640x480, RaytracedLighting +
DLAA. Physics ticks never render: an observation's images render when a policy reads them
(``roboclock.envs.observation``), once per model query. The rig is measured and checked at every reset
(``self.rig``). With ``sim.no_render`` no camera is created and nothing is rendered; observations
carry zero images of the same shapes.
"""

import importlib
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.x5.interface import (
    ACTION_HIGH,
    ACTION_LOW,
    DOF_ORDER,
    MAX_GRIPPER_WIDTH,
    SIDES,
    action_to_native,
    native_to_action,
    ready_action,
    start_joints,
)

# Lab ee_pose convention: tool frame +Z along the approach axis (link6 +X); quaternion w, x, y, z.
TOOL_FRAME = R.from_euler("y", 90, degrees=True)
PRIME_TIMEOUT_S = 1800.0  # first frame of a new stage; a cold RTX pipeline compile takes minutes
# Optional progress line on stderr every N simulated seconds (long C1-C3 runs print nothing else).
PROGRESS_S = float(os.environ.get("ROBOCLOCK_HOUSEHOLD_PROGRESS", "0") or 0)


def sim_module(task_id):
    return importlib.import_module(f"roboclock.envs.household.{task_id.lower()}_x5.sim")


class X5Episode:
    def __init__(self, task, config, folder, initial_scene, *, depth=False):
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction

        from roboclock.envs.household.x5 import rig
        from roboclock.envs.household.x5 import robodojo as RDH
        from roboclock.envs.household.x5.sim import physics as P
        from roboclock.envs.household.x5.sim.servo import Servo

        rig.use(task.task_id)  # this task's arm base layout, before the overlay and servos
        sim = config["sim"]
        self.task = task
        self.dt = float(sim["physics_dt"])
        self.cue_interval = int(sim["cue_interval_ticks"])
        self.depth = depth
        self.no_render = bool(sim.get("no_render", False))
        self.events = []  # probe/controller diagnostics (phases, latch, door, pickups); not scored
        self.world = None
        self.cameras = {}  # view (head, Left, Right) -> isaacsim Camera; empty under sim.no_render
        self.images_rendered = None  # images of the latest render
        self.image_tick = None  # tick of the latest render
        self.renderer = self.intrinsics = None
        self._action = ArticulationAction
        self.module = sim_module(task.task_id)
        geometry = getattr(self.module, "G", None)
        if isinstance(geometry, dict):
            geometry.clear()  # filled by build() and the first measurement of this episode
        self.scene_path, self.setup = P.overlay(initial_scene, folder, self.module.build)

        World.clear_instance()
        context = omni.usd.get_context()
        if not context.open_stage(str(self.scene_path)):
            raise RuntimeError(f"Isaac Sim could not open {self.scene_path}")
        self.stage = context.get_stage()
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.dt, rendering_dt=self.dt)
        import carb

        # Recorder's record mode: physics results written to USD, Fabric off (process-wide).
        physics_to_usd = bool(sim.get("physics_to_usd", False))
        self.world.get_physics_context().enable_fabric(not physics_to_usd)
        carb.settings.get_settings().set("/physics/updateToUsd", physics_to_usd)
        self.robots = {
            side: self.world.scene.add(
                SingleArticulation(f"{P.ARMS}/{side}", name=side, reset_xform_properties=False)
            )
            for side in SIDES
        }
        self.probe = self.module.Oracle(task, self._event)
        self.probe.setup(self.world, self.stage)  # task rigid-body/contact views, before reset
        self.camera_path = sim["camera"]
        if self.camera_path != RDH.HEAD_PATH or not self.stage.GetPrimAtPath(self.camera_path).IsValid():
            raise RuntimeError(f"{task.task_id} scene has no head camera {RDH.HEAD_PATH}")
        self.wrist_paths = {}
        if not self.no_render:
            self.renderer = RDH.apply_renderer()
            self.wrist_paths = RDH.add_wrist_cameras(self.stage, P.ARMS)
            self.cameras = RDH.make_cameras(self.wrist_paths)
        self.world.reset()
        self.world.pause()
        if self.cameras:
            self.intrinsics = RDH.configure_cameras(self.cameras, depth)

        self.dof_index = {}
        # READY_Q, or joint 0 under robot.start_pose: zero (interface.start_joints).
        start_q = start_joints(config)
        for side, robot in self.robots.items():
            names = list(robot.dof_names)
            self.dof_index[side] = np.array([names.index(name) for name in DOF_ORDER])
            q = robot.get_joint_positions()
            for i in range(6):
                q[names.index(f"joint{i + 1}")] = start_q[i]
            for name in ("jawP_joint", "jawN_joint"):
                q[names.index(name)] = 0.0
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros_like(q))
            robot.apply_action(ArticulationAction(joint_positions=q))
        self.commanded = ready_action(start_q)  # 16 targets now active, action convention
        self.rewrite = set()  # arms to write even if unchanged (privileged oracle, next apply)
        self._native = {
            side: action_to_native(self.commanded[8 * k : 8 * k + 8])
            for k, side in enumerate(SIDES)
        }
        control = getattr(self.module, "CONTROL", {})
        self.control_modes = {side: control.get(side.lower(), "ik") for side in SIDES}
        self.control_dt = self.dt * int(sim["control_decimation"])
        # Measurement servos: the probe reads tool poses through them; they never actuate.
        self.servos = {
            side: Servo(robot, side, self.control_modes[side], control_dt=self.control_dt)
            for side, robot in self.robots.items()
        }
        self.probe.servos = self.servos
        self.world.play()
        self.settle_ticks = int(round(float(sim["settle_seconds"]) / self.dt))
        for _ in range(self.settle_ticks):  # passive settling; the task clock has not started
            self.world.step(render=False)
        self.start = float(self.world.current_time)
        self.ticks = 0
        # RoboDojo rig: bases, head camera, wrist mounts and link6, measured at every reset.
        self.rig = RDH.json_ready(RDH.rig_check(self.stage, self.robots, self.wrist_paths))
        self.render_interval = round(1 / (self.dt * RDH.RD.RENDER_HZ))  # 10 ticks at 250 Hz
        self._sense(advance=False)
        if self.no_render:
            self.images_rendered = RDH.blank_observations(depth)
        else:
            self._prime_camera()
            # Opening a stage can reset RTX settings during the first frames: apply again, read back.
            RDH.apply_renderer()
            self._prime_camera()
            self.renderer = RDH.renderer_readback()  # in effect after the first frames
            self._capture()
            self.image_tick = self.ticks

    # ---------------------------------------------------------------- simulation
    def _event(self, name, t, **data):
        self.events.append(dict(event=name, t_s=float(t), **data))

    def clock(self):
        """Seconds of physics since t = 0 (the recorder's controller and probe time base)."""
        return float(self.world.current_time) - self.start

    def _sense(self, advance):
        probe = self.probe
        if hasattr(probe, "appliance_step"):
            probe.appliance_step(self.stage)
        signals = probe.measure(self.clock(), self.dt)
        self.task.update(signals)
        if advance:
            self.task.advance(self.dt)
        if hasattr(probe, "update_cues") and self.ticks % self.cue_interval == 0:
            probe.update_cues()

    def tick(self):
        """One physics tick; nothing renders (observations render when read, ``images``)."""
        self.world.step(render=False)
        self.ticks += 1
        self._sense(advance=True)
        if PROGRESS_S > 0 and self.ticks % max(1, round(PROGRESS_S / self.dt)) == 0:
            print(
                f"household {self.task.task_id}: t={self.task.time:.1f} s",
                file=sys.stderr,
                flush=True,
            )

    def apply(self, action):
        """Set the 16 joint-position targets (action convention); they stay active until replaced.

        An arm whose 8 targets are unchanged is not written again: a PhysX target write also
        wakes a sleeping articulation, and re-sending a resting arm's targets every step made
        trajectories depart from the recorder's (which writes only what its controller commands)
        at the 1e-7 rad level, enough to change marginal grasps in C2/C3. ``rewrite`` (set by the
        privileged oracle only) names arms its controller wrote this tick even with unchanged
        values, so the oracle reproduces the recorder's writes exactly.
        """
        action = np.asarray(action, dtype=np.float64)
        rewrite, self.rewrite = self.rewrite, set()
        for k, side in enumerate(SIDES):
            native = action_to_native(action[8 * k : 8 * k + 8])
            if np.array_equal(native, self._native[side]) and side not in rewrite:
                continue
            self.robots[side].apply_action(
                self._action(joint_positions=native, joint_indices=self.dof_index[side])
            )
            self._native[side] = native
        self.commanded = action.copy()

    # --------------------------------------------------------------- observation
    def _prime_camera(self):
        """Render the first observation without advancing physics (world.render steps no physics)."""
        import time

        from roboclock.envs.household.x5 import robodojo as RDH

        before = float(self.world.current_time)
        # The first RTX frame of a new stage can take long (shader compilation, a shared GPU).
        deadline, renders = time.monotonic() + PRIME_TIMEOUT_S, 0
        while True:
            self.world.render()
            renders += 1
            ready = True
            for cam in self.cameras.values():
                rgb = RDH.frame_rgb(cam)
                ready = ready and rgb is not None and rgb.std() > 1
                if ready and self.depth:
                    ready = cam.get_depth() is not None
            if (ready and renders >= 3) or time.monotonic() > deadline:
                break
            if renders > 3:
                time.sleep(0.05)
        if abs(float(self.world.current_time) - before) > 1e-9:
            raise RuntimeError("Camera priming advanced physics")

    def images(self):
        """Policy camera images of the current tick: rendered now (no physics step) unless
        this tick was already rendered."""
        if not self.no_render and self.image_tick != self.ticks:
            from roboclock.simulators.isaac.rendering import capture_camera

            before = float(self.world.current_time)
            self.world.render()
            # world.render() alone leaves the annotators one frame behind (the previous
            # render); the Replicator step delivers this frame, as in the other domains.
            capture_camera(1)
            if abs(float(self.world.current_time) - before) > 1e-12:
                raise RuntimeError("Rendering advanced physics")
            self._capture()
            self.image_tick = self.ticks
        return {key: value.copy() for key, value in self.images_rendered.items()}

    def _capture(self):
        from roboclock.envs.household.x5 import robodojo as RDH

        images = {}
        for view, cam in self.cameras.items():
            rgb = RDH.frame_rgb(cam)
            if rgb is None:
                raise RuntimeError(f"{RDH.VIEWS[view]} has no RGB frame; check RTX rendering and camera setup")
            images[RDH.RGB_KEYS[view]] = rgb
            if self.depth:
                depth = cam.get_depth()
                if depth is None:
                    raise RuntimeError(f"{RDH.VIEWS[view]} has no depth frame")
                images[RDH.DEPTH_KEYS[view]] = np.asarray(depth, dtype=np.float32).copy()
        self.images_rendered = images

    def joint_actions(self):
        """Measured joints in the action convention (16 values, unclipped)."""
        rows = []
        for side in SIDES:
            q = np.asarray(self.robots[side].get_joint_positions(), dtype=np.float64)
            rows.append(native_to_action(q[self.dof_index[side]]))
        return np.concatenate(rows)

    def observation(self, instruction, sim_time, images=True):
        """Proprioception, preceded by the camera images (``images()``) unless `images` is false."""
        # Reported within the action limits, so a measured value can be sent back as a target.
        q = np.clip(self.joint_actions(), ACTION_LOW, ACTION_HIGH)
        ee = []
        for side in SIDES:
            tcp, rot = self.servos[side].pose()
            x, y, z, w = (rot * TOOL_FRAME).as_quat()
            ee += [*tcp, w, x, y, z]
        width = [np.clip(q[8 * k + 6] + q[8 * k + 7], 0.0, MAX_GRIPPER_WIDTH) for k in range(2)]
        obs = {
            **(self.images() if images else {}),  # rgb, wrist rgb (+ depth)
            "joint_positions": q.astype(np.float32),
            "ee_pose": np.asarray(ee, dtype=np.float32),
            "gripper_width": np.asarray(width, dtype=np.float32),
            "sim_time": np.array([sim_time], dtype=np.float32),
            "instruction": instruction,
        }
        return obs

    def close(self):
        from isaacsim.core.api import World

        for cam in self.cameras.values():
            cam.destroy()
        self.cameras = {}
        if self.world is not None:
            self.world.stop()
            self.world = None
        World.clear_instance()
