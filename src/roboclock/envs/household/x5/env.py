"""Gymnasium environment for the ten current Dual-X5 household tasks (A1-A4, B1-B3, C1-C3).

``roboclock eval --env household`` builds this class (``spec.py``). Each ``reset()`` builds the
requested task's initial scene in code (``x5/scenes.py``), adds the physics overlay and opens it
as a new stage in the running SimulationApp, so one environment serves a whole evaluation plan.
``step()`` applies 16 joint-position targets (``x5/interface.py``) for one control period
(5 physics ticks: 250 Hz physics, 50 Hz control, ``roboclock.envs.rates``). The task model
(``<id>_x5/task.py``) is driven only by signals measured from the simulated physics.

Rig, cameras and renderer follow the RoboDojo rig spec (``roboclock.envs.robodojo_rig``): head
``rgb`` and wrist ``left_wrist_rgb`` / ``right_wrist_rgb`` at 640x480 (plus ``depth``,
``left_wrist_depth``, ``right_wrist_depth`` with ``depth=True``), rendered when a policy reads an
observation (``roboclock.envs.observation``); ``sim.no_render: true`` (``configs/no_render.yaml``) renders nothing and returns zero
images of the same shapes. Every reset measures the rig (``episode_metadata()["robodojo_rig_check"]``).
"""

import json
import math
import shutil
import string

import gymnasium as gym
import numpy as np

from roboclock.config import load_config
from roboclock.envs import robodojo_rig as RD
from roboclock.envs.base import BenchmarkEnv
from roboclock.envs.household.tasks.registry import make_task
from roboclock.envs.household.x5.config import env_config
from roboclock.envs.household.x5.interface import (
    ACTION_HIGH,
    ACTION_LOW,
    JOINT_ORDER,
    MAX_GRIPPER_WIDTH,
    ROBOT_ID,
    robot_configuration,
)
from roboclock.envs.household.x5.robodojo import (
    DEPTH_KEYS,
    RGB_KEYS,
    image_shape,
    rendered_intrinsics_report,
)
from roboclock.envs.observation import Observation, image_keys
from roboclock.envs.rates import CONTROL_HZ

SCENE_REVISION = "R2"


class HouseholdX5Env(BenchmarkEnv):
    environment_id = "household"
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(
        self,
        task_id="A1",
        difficulty="medium",
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

        if asset_catalog is not None:
            raise ValueError(
                "The Dual-X5 household scenes take no asset catalog; assets are in assets/x5"
            )
        self.config = env_config(load_config(config_path, environment=self.environment_id))
        set_start_pose(self.config, start_pose)
        self.runtime = runtime or IsaacRuntime(
            headless=headless, renderer=self.config["visuals"]["renderer"]
        )
        self.owns_runtime = runtime is None
        self.task_id, self.difficulty = task_id, difficulty
        self.depth, self.oracle = depth, oracle
        self.task = None
        self.x5 = None  # the open X5Episode (Isaac stage, robots, probe, camera)
        self.done = False
        self.number_of_actions = 0
        self._workdir = None
        self._initial = {}  # task id -> built initial scene
        self._episodes = 0
        self._kitchen_sha = None
        self.action_space = gym.spaces.Box(ACTION_LOW.copy(), ACTION_HIGH.copy(), dtype=np.float32)
        if depth and self.config["sim"]["no_render"]:
            raise ValueError("depth needs rendered observations; sim.no_render is set")
        spaces = {
            **{
                key: gym.spaces.Box(0, 255, (*image_shape(view), 3), dtype=np.uint8)
                for view, key in RGB_KEYS.items()
            },
            "joint_positions": gym.spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32),
            "ee_pose": gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
            "gripper_width": gym.spaces.Box(0.0, MAX_GRIPPER_WIDTH, (2,), dtype=np.float32),
            "sim_time": gym.spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
            "instruction": gym.spaces.Text(512, charset=string.printable + "–°"),
        }
        if depth:
            for view, key in DEPTH_KEYS.items():
                spaces[key] = gym.spaces.Box(0, np.inf, image_shape(view), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    # -------------------------------------------------------------------- scenes
    def _initial_scene(self, task_id):
        from roboclock.envs.household.x5.scenes import build_initial_scene, make_workdir

        if self._workdir is None:
            self._workdir = make_workdir(self.config["scene"]["scene_dir"])
        if task_id not in self._initial:
            self._initial[task_id] = build_initial_scene(task_id, self._workdir)
        return self._initial[task_id]

    # ----------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        from roboclock.envs.household.x5.episode import X5Episode

        super().reset(seed=seed)
        options = options or {}
        actual_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        task = make_task(
            options.get("task_id", self.task_id),
            options.get("difficulty", self.difficulty),
            actual_seed,
        )
        if self.x5 is not None:
            self.x5.close()
            self.x5 = None
        self.task = task
        initial = self._initial_scene(task.task_id)
        folder = self._workdir / "episodes" / f"{self._episodes:04d}_{task.task_id}"
        self._episodes += 1
        self.x5 = X5Episode(task, self.config, folder, initial, depth=self.depth)
        self.done = False
        self.number_of_actions = 0
        return self.get_obs(), self._info()

    def get_obs(self):
        """Proprioception now; camera images when first read (roboclock.envs.observation)."""
        if self.task is None:
            raise RuntimeError("Call reset() before observing")
        x5 = self.x5
        state = x5.observation(self.task.gen_instruction(), self.task.time, images=False)
        return Observation(state, image_keys(self.observation_space), x5.images, lambda: self.task.time)

    def _info(self):
        # Default info exposes neither evaluator state nor hidden process parameters.
        if not self.oracle:
            return {}
        return {
            "metrics": self.task.metrics(),
            "physical_state": getattr(self.x5.probe, "row", {}),
            "episode_spec": self.task.spec,
            "probe_events": self.x5.events,
        }

    def _expired(self):
        # The task clock sums physics steps (as the recorder does) and can end a hair short of a
        # whole-second horizon, so the episode runs until the clock reaches max_time: possibly
        # one tick past round(max_time / dt), as the recorder does. Otherwise a pickup scheduled
        # at max_time itself, like C2's at 6:00, could go unscored.
        return self.task.time >= self.task.spec.max_time

    def _ticks(self, count):
        # Physics only: observations render when a policy reads them (X5Episode.images).
        for _ in range(count):
            if self.task.terminated or self._expired():
                break
            self.x5.tick()
        self.done = bool(self.task.terminated) or self._expired()

    def advance_latency(self, seconds: float):
        """Keep the previous joint targets active while a policy computes its next action."""
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
        targets = np.asarray(action, dtype=np.float64)
        if targets.shape != self.action_space.shape or not np.all(np.isfinite(targets)):
            raise ValueError("Action must contain 16 finite joint targets (left then right arm)")
        # Bounds are checked at the action space's float32 precision; targets apply unrounded.
        checked = targets.astype(np.float32)
        if np.any(checked < self.action_space.low) or np.any(checked > self.action_space.high):
            raise ValueError("Action exceeds the Dual-X5 household joint limits")
        self.x5.apply(targets)
        self.number_of_actions += 1

    def step(self, action):
        self.set_action(action)
        self._ticks(self.config["sim"]["control_decimation"])
        terminated = bool(self.task.terminated)
        truncated = not terminated and self._expired()
        return self.get_obs(), float(self.task.joint_success), terminated, truncated, self._info()

    def render(self):
        return self.get_obs()["rgb"]

    # ---------------------------------------------------------------- metadata
    def benchmark_metadata(self):
        from roboclock.envs.household.x5 import scene_lib as L
        from roboclock.envs.household.x5.scenes import sha256
        from roboclock.simulators.isaac.runtime import simulator_version

        kitchen = L.KITCHEN / "KitchenRoom_A1.usdc"
        if self._kitchen_sha is None and kitchen.exists():
            self._kitchen_sha = sha256(kitchen)
        return {
            "environment": self.environment_id,
            "simulator": "isaacsim==" + simulator_version(),
            "robot": ROBOT_ID,
            "embodiment": "Dual ARX X5, stationary island mount (household Dual-X5 revision R2)",
            "assets": {
                "root": L.repo_rel(L.ASSETS),
                "kitchen_layer": L.repo_rel(kitchen),
                "kitchen_layer_sha256": self._kitchen_sha,
                "kitchen_note": "Lightwheel KitchenRoom (CC BY-NC 4.0) is linked, not in git: "
                "scripts/household/link_x5_kitchen.sh",
            },
            "rates_hz": {
                "physics": round(1 / self.config["sim"]["physics_dt"]),
                "control": round(
                    1
                    / (self.config["sim"]["physics_dt"] * self.config["sim"]["control_decimation"])
                ),
                "policy": RD.POLICY_HZ,
                "render": 0 if self.config["sim"]["no_render"] else RD.POLICY_RENDER,
            },
            "robodojo_rig": RD.describe(),
            "no_render": self.config["sim"]["no_render"],
        }

    def episode_metadata(self):
        from roboclock.envs.household.x5.scenes import recorded_initial_sha, sha256

        task_id = self.task.task_id
        initial = self._initial[task_id]
        initial_sha = sha256(initial)
        recorded = recorded_initial_sha(task_id)
        state = json.loads((initial.parent / "initial_state.json").read_text())
        return {
            "scene_id": f"household_x5_{SCENE_REVISION.lower()}_{task_id.lower()}",
            "robot": ROBOT_ID,
            "joint_order": list(JOINT_ORDER),
            "robot_configuration": robot_configuration(self.config),
            "control_mode": "joint_position",
            "rates_hz": {
                "physics": round(1 / self.x5.dt),
                "control": round(1 / self.x5.control_dt),
                "cue_updates": round(1 / (self.x5.dt * self.x5.cue_interval)),
                "render": 0 if self.x5.no_render else RD.POLICY_RENDER,
            },
            "robodojo_rig": RD.describe(),
            "robodojo_rig_check": self.x5.rig,
            "renderer": self.x5.renderer,  # read back from carb settings; None under sim.no_render
            "camera_intrinsics": self.x5.intrinsics,  # as get_intrinsics_matrix() reports them
            # what RTX renders (square pixels); use this one for projection and depth unprojection
            "camera_intrinsics_rendered": None if self.x5.intrinsics is None else rendered_intrinsics_report(),
            "observation_cameras": {key: RD.OBSERVATION_KEYS[key] for key in RD.OBSERVATION_KEYS},
            "settle_seconds": self.x5.settle_ticks * self.x5.dt,
            "camera": self.x5.camera_path,
            "number_of_actions": self.number_of_actions,
            "initial_scene": {
                "builder": "roboclock.envs.household.x5.build (+ "
                f"roboclock.envs.household.{task_id.lower()}_x5.scene)",
                "sha256": initial_sha,
                "recorded_demo_sha256": recorded,
                "matches_recorded_demo": None if recorded is None else initial_sha == recorded,
            },
            "simulation_scene": {
                "module": self.x5.module.__name__,
                **{k: v for k, v in self.x5.setup.items() if k not in ("initial_scene", "arms")},
            },
            "initial_state": state.get("initial_state", {}),
            "objects": [
                {k: row.get(k) for k in ("name", "role", "source", "bbox_min_m", "bbox_max_m")}
                for row in state.get("objects", [])
            ],
        }

    def close(self):
        if self.x5 is not None:
            self.x5.close()
            self.x5 = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None
            self._initial.clear()
        if self.owns_runtime:
            self.runtime.close()
