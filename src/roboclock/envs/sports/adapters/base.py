"""One sub-package's backend, native evaluator and privileged controller behind the env.

An adapter owns everything a task needs inside the shared SimulationApp: its
scene (a fresh USD stage per episode), the native task object that scores
measured physical state, environment-side mechanisms (opponents, machines,
humans) and, on request, the task's original privileged controller. The
controller returns full native X5 command vectors; the environment converts
them to and from the shared 16-dim layout (``roboclock.envs.sports.joints``).
"""

import numpy as np

from roboclock.envs import rates, robodojo_rig
from roboclock.envs.sports.joints import SIDES, JointMap


def ensure_extensions():
    """Enable the Isaac extensions the arcade/chess backends import.

    ``isaacsim.robot.manipulators(.examples)`` is enabled at startup by the shared
    runtime for Isaac Sim 6.2, where the sports recordings were made, but not for
    6.0.x. Enabling it here keeps the extension set (and so the recorded runtime)
    unchanged; the X5 backends no longer import its arm wrappers.
    """
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    for name in ("isaacsim.robot.manipulators", "isaacsim.robot.manipulators.examples"):
        if not manager.is_extension_enabled(name):
            manager.set_extension_enabled_immediate(name, True)


def enforce_finger_velocity(rig):
    """Apply the manufacturer finger speed limit (URDF ``velocity``, m/s) to the X5 fingers.

    The converted X5A USD stores the finger joints' 0.15 m/s URDF limit as
    ``physxJoint:maxJointVelocity = 8.594`` (an angular deg/s conversion). Isaac
    Sim 6.2.0, used for the recordings, applied it as 0.15 m/s; Isaac Sim 6.0.1
    applies 8.594 m/s, so the jaws close in ~25 ms instead of ~160 ms and, e.g.,
    strike the falling B3 bar. The limit is set here in SI units through the physics
    view, independent of how a simulator version parses the USD attribute.
    """
    names = list(rig.robot.dof_names)
    view = rig.robot._articulation_view
    imported = np.asarray(view.get_joint_max_velocities()).reshape(-1)
    ids, limits = [], []
    for joint_names in rig.grip_names.values():
        for name in joint_names:
            prim = rig.stage.GetPrimAtPath(rig.joints[name])
            limit = prim.GetAttribute("urdf:limit:velocity").Get() if prim else None
            if limit is None:
                continue
            ids.append(names.index(name))
            limits.append(float(limit))
    if ids:
        view.set_max_joint_velocities(
            np.asarray([limits], dtype=np.float32), joint_indices=np.asarray(ids)
        )
    applied = np.asarray(view.get_joint_max_velocities()).reshape(-1)
    return {names[i]: dict(imported=float(imported[i]), applied_m_s=float(applied[i])) for i in ids}


class Adapter:
    family = ""

    def __init__(self, runtime, task, settings):
        self.runtime = runtime
        self.task = task
        self.spec = task.spec
        self.c = task.native_config
        self.settings = settings
        # sim.no_render: physics and oracle only; no cameras and no render call.
        self.no_render = bool(settings["sim"].get("no_render", False))
        self.cams = None
        self.ticks = 0
        self.controller = None
        self.backend = None

    # -- implemented per sub-package -------------------------------------------------
    def build(self):
        raise NotImplementedError

    @property
    def rig(self):
        """The policy-controlled Dual X5 rig."""
        return self.backend.rig

    @property
    def time(self):
        raise NotImplementedError

    @property
    def done(self):
        raise NotImplementedError

    def command(self):
        """The native command vector most recently applied to the controlled rig."""
        return np.array(self.rig.command, copy=True)

    def apply(self, native):
        raise NotImplementedError

    def tick(self):
        raise NotImplementedError

    def after_control(self):
        """Displays and cues after one control period; visual only."""

    def result(self):
        raise NotImplementedError

    def events(self):
        raise NotImplementedError

    def joint_success(self):
        return bool(self.result().get("success"))

    def state(self):
        return self.backend.last_state

    def make_controller(self):
        raise NotImplementedError

    def before_capture(self):
        """Visual-only updates the recording driver performs before a frame."""

    # -- shared --------------------------------------------------------------------
    def rigs(self):
        """Every Dual X5 rig in the scene (the controlled one first)."""
        return [self.rig]

    def after_build(self):
        from roboclock.envs.sports import head_view, rig_cameras

        # Cue displays turned toward the head camera before anything renders.
        self.displays = head_view.apply(self.backend, self.spec.task_id)
        self.key_objects = head_view.check(self.backend, self.spec.task_id, self.c)

        # RoboDojo rig (roboclock.envs.robodojo_rig): head + two wrist cameras on the
        # policy rig, measured and checked at every reset. sim.no_render builds no
        # camera and renders nothing; the arm frames are still measured and checked.
        if self.no_render:
            self.cams = None
            self.rig_check = rig_cameras.measure_and_check(self.rig, self.c, cameras=False)
            self.renderer = None
        else:
            self.renderer = rig_cameras.apply_renderer()
            self.cams = rig_cameras.ensure(self.backend)
            self.rig_check = self.cams.measured
            # The presentation camera is only for recordings: stop rendering it.
            presentation = getattr(self.backend, "camera", None)
            self.presentation_paused = bool(
                presentation is not None and rig_cameras.disable_updates(presentation)
            )
        self.control_steps = 0
        self._images = None
        self._image_tick = None
        self.physics_scene = measure_physics_rate(self.backend)
        self.joints = JointMap(self.rig.robot.dof_names)
        self.decimation = int(self.c["sim"]["control_decimation"])
        self.physics_dt = float(self.c["sim"]["physics_dt"])
        self.finger_velocity = {}
        if self.settings.get("robot", {}).get("enforce_urdf_finger_velocity", True):
            for rig in self.rigs():
                self.finger_velocity[rig.path] = enforce_finger_velocity(rig)

    def control_step(self):
        """One control period of physics; the native evaluator sees every tick."""
        for _ in range(self.decimation):
            self.tick()
            if self.done:
                break
        self.control_steps += 1
        self.after_control()

    def advance(self, count):
        """Physics ticks under the previous command (injected policy latency)."""
        for _ in range(count):
            if self.done:
                break
            self.tick()
        self.after_control()

    @property
    def world(self):
        return self.backend.world

    @property
    def camera(self):
        """The head camera (``rgb``) of the RoboDojo rig (``rig_cameras``)."""
        if self.cams is None:
            raise RuntimeError("sim.no_render is set: this run has no cameras")
        return self.cams.cameras["cam_head"]

    def render_due(self):
        """Whether the current physics tick has not been rendered yet."""
        return self._images is None or self._image_tick != self.ticks

    def capture_images(self, depth=False):
        """Head and wrist images (and depth) of the current physics tick.

        Called when a policy reads an observation (roboclock.envs.observation), so once per
        model query; a second call at the same tick returns the same frames. Physics never
        advances here.
        """
        from roboclock.envs.sports import rig_cameras
        from roboclock.simulators.isaac.rendering import capture_camera

        if self.no_render:
            # Physics-only iteration: shape-valid zero images, nothing rendered.
            return rig_cameras.blank_images(depth)
        if not self.render_due():
            return {key: value.copy() for key, value in self._images.items()}
        observation = self.settings["observation"]
        self.before_capture()
        before = float(self.world.current_time)
        for _ in range(int(observation["render_updates"])):
            self.world.render()
        capture_camera(subframes=int(observation["rt_subframes"]))
        if abs(float(self.world.current_time) - before) > 1e-9:
            raise RuntimeError("Observation capture advanced physics time")
        self._images = self.cams.images(depth)
        self._image_tick = self.ticks
        self.renders = getattr(self, "renders", 0) + 1
        return {key: value.copy() for key, value in self._images.items()}

    def capture_rgb(self):
        return self.capture_images()["rgb"]

    def enable_depth(self):
        self.cams.enable_depth()

    def observation(self, depth=False, images=True):
        """Proprioception, preceded by the camera images unless `images` is false."""
        images = self.capture_images(depth) if images else {}
        rig = self.rig
        poses = []
        for side in SIDES:
            position, orientation = rig.pose(side)
            poses.extend([*np.asarray(position, float), *np.asarray(orientation, float)])
        result = {
            **images,
            "joint_positions": self.joints.observe(rig.robot.get_joint_positions()),
            "ee_pose": np.asarray(poses, dtype=np.float32),
            "gripper_width": np.array(
                [np.clip(rig.width(side), 0.0, 0.088) for side in SIDES], dtype=np.float32
            ),
            "sim_time": np.array([self.time], dtype=np.float32),
            "instruction": self.spec.instruction,
        }
        return result

    def metadata(self):
        """Episode identity, mechanisms and robot mapping for the result row."""
        c = self.c
        sections = {
            key: c[key]
            for key in (
                "embodiment",
                "randomization",
                "workflow",
                "freeze",
                "rack",
                "service",
                "duel",
            )
            if key in c
        }
        from roboclock.envs.sports import head_view, rig_cameras

        renderer = rig_cameras.renderer_state() if not self.no_render else None
        return dict(
            robodojo_rig=rig_cameras.describe(
                self.rig_check,
                self.cams.intrinsics() if self.cams is not None else None,
                renderer,
            ),
            head_view=dict(
                displays_moved=self.displays,
                key_objects=self.key_objects,
                margin_required_px=head_view.MARGIN_PX,
            ),
            observation_rendered=not self.no_render,
            render=None if self.no_render else robodojo_rig.POLICY_RENDER,
            renders=getattr(self, "renders", 0),
            physics_scene=self.physics_scene,
            native_joint_order=list(self.joints.native_names),
            joint_mapping=self.joints.describe(),
            finger_velocity_limits=self.finger_velocity,
            native_configuration=sections,
        )

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None


def measure_physics_rate(backend):
    """Physics rate as the simulator runs it: World physics dt and the PhysX scene."""
    from pxr import PhysxSchema, UsdPhysics

    world = backend.world
    dt = float(world.get_physics_dt())
    scenes = {}
    for prim in backend.stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            api = PhysxSchema.PhysxSceneAPI(prim)
            attr = api.GetTimeStepsPerSecondAttr() if api else None
            scenes[str(prim.GetPath())] = attr.Get() if attr and attr.HasAuthoredValue() else None
    hz = 1.0 / dt
    if abs(hz - rates.PHYSICS_HZ) > 1e-6 or any(
        v is not None and abs(v - rates.PHYSICS_HZ) > 1e-6 for v in scenes.values()
    ):
        raise ValueError(f"Sports physics runs at {hz:g} Hz {scenes}, not {rates.PHYSICS_HZ} Hz")
    return dict(physics_dt=dt, physics_hz=hz, physx_time_steps_per_second=scenes)


def episode_rates(adapter):
    return dict(
        physics_hz=round(1 / adapter.physics_dt),
        control_hz=round(1 / (adapter.physics_dt * adapter.decimation)),
    )
