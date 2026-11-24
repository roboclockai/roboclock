"""Industry BenchmarkEnv: the native Dual ARX X5 workcells behind joint-position actions.

Each task builds its native scene (``native/catalog.py``) in the shared Isaac Sim
process and is stepped exactly like its recorded runner: physics at the task's
native rate (250/500/1000 Hz), 50 Hz control, ``scene.observe()`` on every physics
step or on control ticks as recorded, scene processes (lamps, dispenser, machine
outfeed and gate) on control ticks. Scoring comes from ``scoring.py`` monitors over
measured scene state. An episode lasts ``spec.max_time``: A/B tasks end earlier once the
monitor measures completion plus the recorded hold, C tasks (horizon H) once all their jobs
are done and settled or the stall stop fires (``scoring.ThroughputMonitor``).

Actions are 16 joint-position targets ``[left joint1..8, right joint1..8]`` in the
native articulation's units (joint1-6 radians, joint7/8 finger-slider metres),
written to ``DualARX.commands`` and applied on every physics step of the control
period, as the native controllers' commands were.
"""

import math
import string
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs import robodojo_rig as spec
from roboclock.envs.base import BenchmarkEnv
from roboclock.envs.industry.config import load_config
from roboclock.envs.industry.tasks import make_task
from roboclock.envs.lab.robot import JOINT_HIGH, JOINT_LOW, ROBOT_ID
from roboclock.envs.observation import Observation, image_keys
from roboclock.envs.rates import CONTROL_HZ
from roboclock.envs.start_pose import ARM_DOF, zero_start
from roboclock.simulators.isaac.runtime import simulator_version

SIDES = ("left", "right")
JOINT_ORDER = [f"{side}/joint{i}" for side in SIDES for i in range(1, 9)]


def tool_quaternion(R):
    """Lab ``ee_pose`` convention: wxyz of link6 turned so +Z points to the fingertips."""
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w - y, x - z, w + y, x + z]) / np.sqrt(2.0)


class IndustryEnv(BenchmarkEnv):
    environment_id = "industry"
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(
        self,
        task_id="B2",
        difficulty="reviewed_nominal",
        *,
        runtime=None,
        headless=True,
        config_path=None,
        asset_catalog=None,
        depth=False,
        oracle=False,
        start_pose=None,
    ):
        from roboclock.envs.start_pose import set_start_pose
        from roboclock.simulators.isaac.runtime import IsaacRuntime

        self.config = load_config(config_path)
        # robot.start_pose (roboclock.envs.start_pose): "zero" starts both arms at joint 0.
        set_start_pose(self.config, start_pose)
        # sim.no_render (robodojo_rig spec): no cameras and no render calls; images are zeros.
        self.no_render = bool(self.config["sim"].get("no_render", False))
        self.runtime = runtime or IsaacRuntime(headless=headless, renderer=spec.RENDER_MODE)
        self.owns_runtime = runtime is None
        if asset_catalog is not None:
            raise ValueError("Industry scenes use their native asset manifests; --asset-catalog is unsupported")
        self.task_id, self.difficulty = task_id, difficulty
        self.depth, self.oracle = depth, oracle
        self.task = self.scene = self.native = self.monitor = None
        self.native_controller = None
        self.native_observation = None
        self.done = False
        self.workdir = Path(tempfile.mkdtemp(prefix="roboclock-industry-"))
        self._configure_renderer()
        self.action_space = gym.spaces.Box(
            np.tile(JOINT_LOW, 2), np.tile(JOINT_HIGH, 2), dtype=np.float32
        )
        templates = {"cam_head": spec.HEAD_CAMERA, "cam_left_wrist": spec.WRIST_CAMERA, "cam_right_wrist": spec.WRIST_CAMERA}
        self.image_shapes = {
            spec.OBSERVATION_KEYS[name]: (t["resolution"][1], t["resolution"][0]) for name, t in templates.items()
        }
        spaces = {
            **{key: gym.spaces.Box(0, 255, (*shape, 3), dtype=np.uint8) for key, shape in self.image_shapes.items()},
            "joint_positions": gym.spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32),
            "ee_pose": gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
            "gripper_width": gym.spaces.Box(0.0, 0.088, (2,), dtype=np.float32),
            "sim_time": gym.spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
            "instruction": gym.spaces.Text(1024, charset=string.printable + "–°×≤≥"),
        }
        if depth:
            for name, key in spec.DEPTH_KEYS.items():
                shape = self.image_shapes[spec.OBSERVATION_KEYS[name]]
                spaces[key] = gym.spaces.Box(0, np.inf, shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def _configure_renderer(self):
        import omni.replicator.core as rep

        from roboclock.envs.industry.native import rig

        # robodojo_rig renderer: RaytracedLighting + DLAA + the quality preset; read back.
        self.renderer = None if self.no_render else rig.apply_renderer()
        # Captures are explicit and never advance the task clock.
        rep.orchestrator.set_capture_on_play(False)

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        actual_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.task = make_task(
            options.get("task_id", self.task_id),
            options.get("difficulty", self.difficulty),
            actual_seed,
            self.config,
        )
        self._close_scene()
        self._build_scene()
        self.done = False
        # The recorded runners observed after the first physics step, before control.
        self._physics_step()
        self._advance(0)
        return self.get_obs(), self._info()

    def _build_scene(self):
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage

        from roboclock.envs.industry.native.catalog import TASKS
        from roboclock.envs.industry.native.minimal_context import MinimalContext
        from roboclock.envs.industry.scoring import MONITORS

        number = self.task.legacy_task_id
        self.native = TASKS[number]
        resolved = self.task.spec.resolved
        from roboclock.envs.industry.native import rig

        World.clear_instance()
        create_new_stage()
        rig.configure(no_render=self.no_render, overview=False)
        output = self.workdir / f"task{number}-seed{self.task.spec.seed}"
        output.mkdir(parents=True, exist_ok=True)
        self.scene = self.native.make_scene(output, resolved)
        hz = round(1 / self.scene.world.get_physics_dt())
        configured = self.config["tasks"].get(self.task.task_id, {}).get("physics_hz", hz)
        if not hz == self.native.physics_hz == configured:
            raise RuntimeError(
                f"Task {self.task.task_id} scene runs at {hz} Hz (native {self.native.physics_hz} Hz, "
                f"configured {configured} Hz); Industry physics rates are fixed by the native scenes"
            )
        self.config["sim"]["physics_dt"] = 1.0 / hz
        self.config["sim"]["control_decimation"] = self.native.control_stride
        if zero_start(self.config):
            # After every task's own posture writes, before the controller, cache and monitor
            # read the arms.
            self._zero_start_pose()
        # The privileged diagnostic controller must exist from time 0 like the
        # recorded runner's; it only acts when the oracle policy calls it.
        self.native_controller = self.native.load()[1](self.scene, "nominal") if self.oracle else None
        MinimalContext(self.scene, number, output)
        if self.native.readback_cache:
            self.scene.dual.native_readback_cache_enabled = True
        self.monitor = MONITORS[number](self.scene, resolved, self.native)
        self.physics_steps = 0
        self.max_physics_steps = round(self.task.spec.max_time * hz)
        self._cue = getattr(self.scene, "cue_state", None)
        self._warmup = True
        self._images = None
        self._rendered_step = None
        self._depth_attached = False
        if self.no_render:
            # No cameras: the rig check measures the bases and uses the spec head pose.
            self.rig_check = rig.measure(self.scene)
            self.scene.rig["measured"] = self.rig_check
        self._map_joints()

    def _zero_start_pose(self):
        """robot.start_pose: zero. Joints 1-6 of both arms at 0 rad as positions, default
        state and PD targets; the jaws keep the task's start width."""
        dual = self.scene.dual
        for side, robot in dual.robots.items():
            names = dual.names[side]
            target = np.asarray(dual.commands[side], dtype=float).copy()
            target[[i for i, n in enumerate(names) if int(n[5:]) <= ARM_DOF]] = 0.0
            robot.set_joints_default_state(positions=target, velocities=np.zeros(len(names)))
            robot.set_joint_positions(target)
            robot.set_joint_velocities(np.zeros(len(names)))
            dual.commands[side] = target.copy()

    def _map_joints(self):
        dual = self.scene.dual
        self.joint_index = {side: [dual.names[side].index(f"joint{i}") for i in range(1, 9)] for side in SIDES}

    def _close_scene(self):
        if self.scene is None:
            return
        scene = self.scene
        for holder in (scene, getattr(scene, "contacts", None)):
            subscription = getattr(holder, "subscription", None)
            if subscription is not None and hasattr(subscription, "unsubscribe"):
                subscription.unsubscribe()
            if holder is not None and hasattr(holder, "subscription"):
                holder.subscription = None
        for camera in [scene.camera, *getattr(scene, "wrist_cameras", {}).values()]:
            try:
                camera.destroy()
            except Exception:  # noqa: BLE001 - a failed teardown must not mask the next reset
                pass
        scene.world.stop()
        scene.world.clear()
        self.scene = self.native_controller = self.monitor = self.native_observation = None

    # ------------------------------------------------------------------ physics
    def _physics_step(self):
        """One native physics step with the recorded runner's bookkeeping."""
        native, scene = self.native, self.scene
        native.interval(scene).clear()
        scene.dual.apply()
        scene.world.step(render=False)
        k = self.physics_steps
        self.physics_steps += 1
        tick = k % native.control_stride == 0
        if native.observe_every_step or tick:
            o = scene.observe()
            self.native_observation = o
            self.monitor.physics(o)
        if tick:
            o = self.native_observation
            process = getattr(scene, "process_step", None)
            if process is not None:
                process(o)
            self.monitor.tick(o)
        self.task.time = float(scene.world.current_time)
        self.task.steps = self.physics_steps
        if self.monitor.finished(self.task.time):
            self.task.terminated = True

    def _advance(self, count):
        for _ in range(count):
            if self.task.terminated or self.physics_steps >= self.max_physics_steps:
                break
            self._physics_step()
        out_of_time = self.physics_steps >= self.max_physics_steps
        if out_of_time and not self.task.terminated:
            # C tasks report why they ended: the monitor's all_done/stall, else the horizon.
            self.monitor.end("horizon")
        if (self.task.terminated or out_of_time) and self.native.final_observe:
            # Throughput runners observed once more after the loop, before scoring.
            self.native_observation = self.scene.observe()
        self.task.record(self.monitor.outcome(self.native_observation))
        self.task.event_log = [
            dict(time=v["time_s"] if isinstance(v, dict) else v, type=k)
            for k, v in sorted(
                self.monitor.events.items(),
                key=lambda item: item[1]["time_s"] if isinstance(item[1], dict) else item[1],
            )
            if isinstance(v, (int, float, dict))
        ]
        self.done = self.task.terminated or out_of_time

    def _require_running(self):
        if self.task is None:
            raise RuntimeError("Call reset() before step()")
        if self.done:
            raise RuntimeError("Episode has ended; call reset()")

    def apply_action(self, action):
        dual = self.scene.dual
        for side, values in zip(SIDES, (action[:8], action[8:]), strict=True):
            command = dual.commands[side]
            command[self.joint_index[side]] = values.astype(np.float64)

    def set_action(self, action):
        """Make `action` the joint targets in effect from the current physics tick on."""
        self._require_running()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain 16 finite joint targets (left then right)")
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("Action exceeds ARX X5 joint limits")
        self.apply_action(action)
        self.task.action_count += 1

    def step(self, action):
        self.set_action(action)
        self._advance(self.native.control_stride)
        terminated = self.task.terminated
        truncated = not terminated and self.done
        return self.get_obs(), float(self.task.joint_success), terminated, truncated, self._info()

    def advance_latency(self, seconds: float):
        """Keep the previous joint targets while a policy computes its next action."""
        self._require_running()
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Latency must be finite and nonnegative")
        self._advance(math.ceil(seconds / self.config["sim"]["physics_dt"] - 1e-12))

    # ------------------------------------------------------------------ sensing
    def _render(self):
        """One render of the head and wrist cameras at the current physics step, made when a
        policy reads an observation (roboclock.envs.observation): once per model query."""
        from roboclock.envs.industry.native import rig
        from roboclock.simulators.isaac.rendering import capture_camera

        scene = self.scene
        if self._images is not None and self._rendered_step == self.physics_steps:
            return  # this step is already rendered
        before = float(scene.world.current_time)
        cameras = rig.policy_cameras(scene)
        cue = getattr(scene, "cue_state", None)
        if self._warmup:
            # A new render product needs a few frames before the annotator has data
            # (the recorded runner's initial capture used the same 6 x 3 subframes).
            for _ in range(6):
                scene.world.render()
                capture_camera(3)
            self._warmup = False
            self.rig_check = rig.measure(scene)
            scene.rig["measured"] = self.rig_check
        elif cue != self._cue:
            # Kit 6 can keep stale signal materials in an existing render product; the
            # recorded runner refreshed it the same way (no physics).
            for _, camera in cameras:
                camera.destroy()
                camera.initialize()
                if self.depth:
                    camera.add_distance_to_image_plane_to_frame()
            rig.reapply_intrinsics(scene)
            for _ in range(6):
                scene.world.render()
                capture_camera(1)
            self._cue = cue
        scene.world.render()
        capture_camera(int(self.config["sim"].get("render_subframes", 2)))
        if float(scene.world.current_time) != before:
            raise RuntimeError("Rendering advanced physics")
        images = {}
        for key, camera in cameras:
            rgba = camera.get_rgba()
            if rgba is None or np.asarray(rgba).ndim != 3:
                raise RuntimeError(f"{key} camera has no RGB frame. Check RTX rendering and camera initialization.")
            images[key] = np.asarray(rgba)[:, :, :3].astype(np.uint8)
            if self.depth:
                name = next(n for n, k in spec.OBSERVATION_KEYS.items() if k == key)
                images[spec.DEPTH_KEYS[name]] = np.asarray(camera.get_depth(), dtype=np.float32).copy()
        self._images = images
        self._rendered_step = self.physics_steps

    def images(self):
        """Policy camera images of the current physics step (rendered now unless this step
        already was); zero images under sim.no_render."""
        if self.no_render:
            images = {key: np.zeros((*shape, 3), dtype=np.uint8) for key, shape in self.image_shapes.items()}
            if self.depth:
                for name, key in spec.DEPTH_KEYS.items():
                    images[key] = np.zeros(self.image_shapes[spec.OBSERVATION_KEYS[name]], dtype=np.float32)
            return images
        if self.depth and not self._depth_attached:
            from roboclock.envs.industry.native import rig

            for _, camera in rig.policy_cameras(self.scene):
                camera.add_distance_to_image_plane_to_frame()
            self._depth_attached = True
        self._render()
        return {key: value.copy() for key, value in self._images.items()}

    def get_obs(self):
        """Proprioception now; camera images when first read (roboclock.envs.observation)."""
        if self.task is None:
            raise RuntimeError("Call reset() before observing")
        scene = self.scene
        joints, poses, widths = [], [], []
        for side in SIDES:
            q = np.asarray(scene.dual.robots[side].get_joint_positions())[self.joint_index[side]]
            joints.extend(q)
            p, R = scene.dual.tool_world(side)
            poses.extend([*p, *tool_quaternion(R)])
            widths.append(np.clip(q[6] + q[7], 0.0, 0.088))
        # Readings can sit a solver tolerance past a drive limit (a closed slider reads
        # about -1e-5 m); report them inside the action bounds so they are valid targets.
        joints = np.clip(np.asarray(joints, dtype=np.float32), self.action_space.low, self.action_space.high)
        state = {
            "joint_positions": joints,
            "ee_pose": np.asarray(poses, dtype=np.float32),
            "gripper_width": np.asarray(widths, dtype=np.float32),
            "sim_time": np.array([self.task.time], dtype=np.float32),
            "instruction": self.task.gen_instruction(),
        }
        return Observation(state, image_keys(self.observation_space), self.images, lambda: self.task.time)

    def _info(self):
        if not self.oracle:
            return {}
        return {
            "metrics": self.task.metrics(),
            "native_observation": self.native_observation,
            "episode_spec": self.task.spec,
        }

    def render(self):
        return self.get_obs()["rgb"]

    # ------------------------------------------------------------------ metadata
    def benchmark_metadata(self):
        from roboclock.envs.industry.native.paths import asset_root, robot_manifest

        return {
            "environment": self.environment_id,
            "simulator": "isaacsim==" + simulator_version(),
            "robot": ROBOT_ID,
            "asset_catalog": str(asset_root() / "manifest.json"),
            "robot_manifest": str(robot_manifest()),
        }

    def episode_metadata(self):
        number = self.task.legacy_task_id
        return {
            "scene_id": f"industry_task{number}_{self.task.name}",
            "robot": ROBOT_ID,
            "joint_order": JOINT_ORDER,
            "native_joint_names": {side: list(self.scene.dual.names[side]) for side in SIDES},
            "robot_mounts": {side: list(map(float, v)) for side, v in self.scene.dual.mounts.items()},
            # robodojo_rig.describe(), this task's rig placement, the measured rig check
            # and the camera intrinsics reported by the renderer.
            "rig": getattr(self.scene, "rig", None),
            "renderer": self.renderer,
            "no_render": self.no_render,
            "control_mode": "joint_position",
            "physics_hz": self.native.physics_hz,
            "control_hz": CONTROL_HZ,
            "legacy_task_id": number,
            "episode_spec_sha256": self.task.spec.resolved.get("sha256"),
        }

    def close(self):
        self._close_scene()
        if self.owns_runtime:
            self.runtime.close()
