"""Gymnasium-compatible laboratory environment with simulation-time latency injection.

Arm convention of the interface (shared with hospital, which subclasses LabEnv): actions,
`joint_positions`, `ee_pose` and `gripper_width` list the rig's LEFT arm (+y in the rig frame,
the arm `left_wrist_rgb` rides on) first, then the rig's right arm. The backend keeps its
internal side order (side 0 = /World/ARXLeft = rig right); `step` and `get_obs` convert once,
here (`external_arm_order`).
"""

import math
import string

import gymnasium as gym
import numpy as np

from roboclock.config import load_config
from roboclock.envs.base import BenchmarkEnv
from roboclock.envs.lab.robot import (
    EXTERNAL_ARMS,
    EXTERNAL_JOINT_ORDER,
    JOINT_HIGH,
    JOINT_LOW,
    ROBOT_ID,
    arms_to_internal,
    observation_to_external,
)
from roboclock.envs.lab.tasks.registry import make_task
from roboclock.envs.observation import Observation, image_keys
from roboclock.envs.rates import CONTROL_HZ
from roboclock.simulators.isaac.runtime import simulator_version


class LabEnv(BenchmarkEnv):
    environment_id = "lab"
    scene_id = "laboratory_tabletop_v1"
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": CONTROL_HZ}
    # Internal side (robot.ROOTS index) of each interface arm: rig left (+y) first.
    external_arm_order = EXTERNAL_ARMS

    def make_backend(self, runtime, config, asset_catalog):
        from roboclock.envs.lab.backend import IsaacBackend

        return IsaacBackend(runtime, config, asset_catalog)

    def make_task(self, task_id, difficulty, seed, config):
        return make_task(task_id, difficulty, seed, config)

    def __init__(
        self,
        task_id="A1",
        difficulty="easy",
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

        self.config = load_config(config_path, environment=self.environment_id)
        # robot.start_pose (roboclock.envs.start_pose): "zero" starts both arms at joint 0.
        set_start_pose(self.config, start_pose)
        render_profile = (
            self.config.get("liquid_visuals", {})
            if task_id
            in (
                "A1",
                "A2",
                "A3",
                "A4",
                "B1",
                "B2",
                "B3",
                "B4",
                "thermal_short_circuit",
                "motor_stall",
                "test_tube_rescue",
                "hotplate_isolation",
                "alarm_transfer",
            )
            else {}
        )
        self.runtime = runtime or IsaacRuntime(
            headless=headless,
            renderer=render_profile.get(
                "renderer", self.config["visuals"].get("renderer", "RaytracedLighting")
            ),
        )
        self.owns_runtime = runtime is None
        # Delayed until SimulationApp has been constructed.
        self.backend = self._create_backend(asset_catalog)
        self.task_id, self.difficulty = task_id, difficulty
        self.depth, self.oracle = depth, oracle
        self.backend.serve_depth = depth  # depth annotators only when observed
        self.task = None
        self.done = False
        self.action_space = gym.spaces.Box(
            np.tile(JOINT_LOW, 2), np.tile(JOINT_HIGH, 2), dtype=np.float32
        )
        width, height = self.config["sim"]["camera_resolution"]
        spaces = {
            "rgb": gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8),
            "joint_positions": gym.spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32),
            "ee_pose": gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
            "gripper_width": gym.spaces.Box(0.0, 0.088, (2,), dtype=np.float32),
            "sim_time": gym.spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
            "instruction": gym.spaces.Text(512, charset=string.printable + "–°"),
        }
        if depth:
            spaces["depth"] = gym.spaces.Box(0, np.inf, (height, width), dtype=np.float32)
        if self.robodojo_cameras:
            from roboclock.envs.lab.rig_cameras import observation_spaces

            # RoboDojo cam_head / cam_left_wrist / cam_right_wrist (640x480).
            spaces.update(observation_spaces(depth))
        self.observation_space = gym.spaces.Dict(spaces)

    @property
    def robodojo_cameras(self):
        """Head + wrist RoboDojo cameras, 250 Hz physics, rendered when an observation is read."""
        return bool(getattr(self.backend, "robodojo_cameras", False))

    # Hook names used by the hospital domain; they forward to make_backend/make_task.
    def _create_backend(self, asset_catalog):
        return self.make_backend(self.runtime, self.config, asset_catalog)

    def _make_task(self, task_id, difficulty, seed):
        return self.make_task(task_id, difficulty, seed, self.config)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        actual_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.task = self._make_task(
            options.get("task_id", self.task_id),
            options.get("difficulty", self.difficulty),
            actual_seed,
        )
        self.backend.reset(self.task)
        arms = 2
        self.action_space = gym.spaces.Box(
            np.tile(JOINT_LOW, arms), np.tile(JOINT_HIGH, arms), dtype=np.float32
        )
        for key, count in (("joint_positions", 8), ("ee_pose", 7)):
            self.observation_space.spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, (count * arms,), dtype=np.float32
            )
        self.observation_space.spaces["gripper_width"] = gym.spaces.Box(
            0.0, 0.088, (arms,), dtype=np.float32
        )
        self.done = False
        return self.get_obs(), self._info()

    def get_obs(self):
        """Proprioception now; camera images when first read (roboclock.envs.observation)."""
        if self.task is None:
            raise RuntimeError("Call reset() before observing")
        state = self.external_observation(self.backend.observation(depth=self.depth, images=False))
        return Observation(
            state,
            image_keys(self.observation_space),
            lambda: self.backend.images(depth=self.depth),
            lambda: self.task.time,
        )

    def external_observation(self, observation):
        """A backend observation with its per-arm vectors in interface order (rig left first)."""
        return observation_to_external(observation, self.external_arm_order)

    def internal_action(self, action):
        """An interface action (rig left arm first) in the backend's internal side order."""
        return arms_to_internal(action, self.external_arm_order)

    def joint_order(self):
        """Labels of the 16 action / joint_positions entries; left/right = physical rig side."""
        if tuple(self.external_arm_order) == EXTERNAL_ARMS:
            return list(EXTERNAL_JOINT_ORDER)
        # Legacy envs (hospital, household) keep internal side order: side 0 is "arm0".
        return [f"arm{side}/joint{i}" for side in (0, 1) for i in range(1, 9)]

    def _info(self):
        # Default info exposes neither evaluator state nor hidden dynamics to policies.
        if not self.oracle:
            return {}
        return {
            "metrics": self.task.metrics(),
            "physical_state": self.backend.state(),
            "episode_spec": self.task.spec,
            "layout": self.backend.layout,
        }

    def _ticks(self, count, *, render=True):
        # Lab: physics never renders; observations render when read (rig_cameras.read).
        render = render and not self.robodojo_cameras
        dt = self.config["sim"]["physics_dt"]
        remaining = max(0, math.ceil((self.task.spec.max_time - self.task.time - 1e-9) / dt))
        for i in range(min(count, remaining)):
            for command in self.task.advance(dt):
                self.backend.command(command)
            self.backend.tick(render=render and i == min(count, remaining) - 1)
            self.task.observe(self.backend.state(), dt)
            self.task.collision_count = self.backend.collision_count
            if getattr(self.task, "safety_task", False):
                from roboclock.envs.lab.mdp.events import SceneCommand

                self.backend.command(
                    SceneCommand("safety_visual", "equipment", self.task.visual_state())
                )
            if getattr(self.task, "bath_task", False):
                from roboclock.envs.lab.mdp.events import SceneCommand

                self.backend.command(
                    SceneCommand("bath_visual", "equipment", self.task.visual_state())
                )
            if self.task.terminated:
                if getattr(self.task, "laboratory_v2", False):
                    from roboclock.envs.lab.mdp.events import GREEN, RED, SceneCommand

                    self.backend.command(
                        SceneCommand(
                            "indicator", "indicator", GREEN if self.task.joint_success else RED
                        )
                    )
                # A terminal substep may precede the requested render interval.
                if render:
                    self.backend.world.render()
                break
        self.done = self.task.terminated or self.task.time >= self.task.spec.max_time - 1e-9

    def advance_latency(self, seconds: float):
        """Keep the previous motor command active while a policy computes its next action."""
        self._require_running()
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Latency must be finite and nonnegative")
        dt = self.config["sim"]["physics_dt"]
        self._ticks(math.ceil(seconds / dt - 1e-12))

    def _require_running(self):
        if self.task is None:
            raise RuntimeError("Call reset() before step()")
        if self.done:
            raise RuntimeError("Episode has ended; call reset()")

    def set_action(self, action):
        """Make `action` the joint targets in effect from the current physics tick on."""
        self._require_running()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain finite joint targets for every configured arm")
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("Action exceeds ARX X5 joint limits")
        self.backend.apply_action(self.internal_action(action))
        self.task.action_count += 1

    def step(self, action):
        self.set_action(action)
        self._ticks(self.config["sim"]["control_decimation"])
        terminated = self.task.terminated
        truncated = not terminated and self.task.time >= self.task.spec.max_time - 1e-9
        return self.get_obs(), float(self.task.joint_success), terminated, truncated, self._info()

    def render(self):
        return self.get_obs()["rgb"]

    def benchmark_metadata(self):
        return {
            "environment": self.environment_id,
            "simulator": "isaacsim==" + simulator_version(),
            "robot": ROBOT_ID,
            "asset_catalog": self.backend.catalog,
        }

    def episode_metadata(self):
        from roboclock.envs.lab.layout import apparatus_frame

        layout = self.task.layout
        return {
            "scene_id": "laboratory_liquid_process_v2"
            if getattr(self.task, "liquid_process", False)
            else "laboratory_hotplate_isolation_v2"
            if getattr(self.task, "hotplate_task", False)
            else "laboratory_equipment_faults_v2"
            if getattr(self.task, "laboratory_v2", False)
            else self.scene_id,
            "robot": ROBOT_ID,
            "joint_order": self.joint_order(),
            "robot_configuration": self.config.get("robot", {"model": ROBOT_ID}),
            "control_mode": "joint_position",
            **(self.robodojo_metadata() if self.robodojo_cameras else {}),
            "clutter_initial_state": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "position": item.position.tolist(),
                    "size": item.size.tolist(),
                    "mass": item.mass,
                    "label": item.label,
                }
                for item in getattr(self.backend, "clutter_specs", [])
            ],
            "clutter_robot_contact_count": getattr(self.backend, "clutter_robot_contact_count", 0),
            # Per-seed rigid placement of the apparatus (B2/B3/B5-B7, C1 racks; layout.py).
            "apparatus_placement": apparatus_frame(layout).info(),
            "initial_state": {
                name: {
                    "asset": obj.asset,
                    "position": obj.position.tolist(),
                    "orientation": obj.orientation.tolist(),
                    "color": obj.color.tolist(),
                    "mass": obj.mass,
                    "friction": obj.friction,
                    "target": layout.assignments[name],
                }
                for name, obj in layout.objects.items()
            },
            "targets": {
                name: {
                    "position": target.position.tolist(),
                    "orientation": target.orientation.tolist(),
                    "half_extents": target.half_extents.tolist(),
                    "kind": target.kind,
                }
                for name, target in layout.targets.items()
            },
        }

    def robodojo_metadata(self):
        """RoboDojo rig spec, measured rig check, renderer read-back and rates."""
        from roboclock.envs import robodojo_rig

        b = self.backend
        context = b.world.get_physics_context() if b.world is not None else None
        physics_dt = context.get_physics_dt() if context is not None else None
        return {
            **b.rig_cameras.metadata(),
            "observation_cameras": {
                "rgb": "cam_head",
                "left_wrist_rgb": "cam_left_wrist",
                "right_wrist_rgb": "cam_right_wrist",
            },
            "rates": {
                "physics_hz": None if physics_dt is None else round(1 / physics_dt, 6),
                "control_hz": round(
                    1 / (self.config["sim"]["physics_dt"] * self.config["sim"]["control_decimation"]),
                    6,
                ),
                "render": robodojo_rig.POLICY_RENDER,
                "policy_hz": robodojo_rig.POLICY_HZ,
            },
        }

    def close(self):
        self.backend.close()
        if self.owns_runtime:
            self.runtime.close()
