"""Unified Gymnasium environment for the ten current Dual ARX X5 sports tasks.

One environment instance owns one SimulationApp and dispatches each reset to the
task's runtime adapter (``roboclock.envs.sports.adapters``), which builds that
sub-package's scene on a fresh stage. Every task shares the 16-dim joint
position action, the RoboDojo rig cameras (``rgb`` head, ``left_wrist_rgb``,
``right_wrist_rgb``, 640x480, rendered when a policy reads an observation;
``rig_cameras``, ``roboclock.envs.observation``), exactly 250 Hz physics and 50 Hz control
(``roboclock.envs.robodojo_rig``); scenes, environment mechanisms and scoring stay
those of the recordings.
"""

import math
import string
from dataclasses import asdict

import gymnasium as gym
import numpy as np

from roboclock.envs.base import BenchmarkEnv
from roboclock.envs.observation import Observation, image_keys
from roboclock.envs.rates import POLICY_HZ
from roboclock.envs.sports.joints import JOINT_ORDER, action_bounds

ROBOT_ID = "dual_arx_x5"


class SportsBenchmarkEnv(BenchmarkEnv):
    environment_id = "sports"
    metadata = {"render_modes": ["rgb_array"], "render_fps": POLICY_HZ}

    def __init__(
        self,
        task_id="A2",
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
        from roboclock.envs.sports.config import load_config, with_task_rates
        from roboclock.envs.sports.tasks import SOURCES
        from roboclock.envs.start_pose import set_start_pose

        if asset_catalog is not None:
            raise ValueError(
                "Sports scenes use their packaged assets; --asset-catalog is unsupported"
            )
        if str(task_id) not in SOURCES:
            raise ValueError(f"Unknown sports task {task_id!r}; choose one of {', '.join(SOURCES)}")
        self.settings = load_config(config_path)
        # robot.start_pose (roboclock.envs.start_pose): "zero" starts both arms at joint 0.
        set_start_pose(self.settings, start_pose)
        self.config = with_task_rates(self.settings, str(task_id))
        if runtime is None:
            from roboclock.simulators.isaac.runtime import IsaacRuntime

            runtime = IsaacRuntime(headless=headless, renderer=self.config["visuals"]["renderer"])
            self.owns_runtime = True
        else:
            self.owns_runtime = False
        self.runtime = runtime
        self.task_id, self.difficulty = str(task_id), difficulty
        self.depth, self.oracle = depth, oracle
        self.task = None
        self.adapter = None
        self.done = False
        low, high = action_bounds()
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        from roboclock.envs.sports.rig_cameras import observation_shapes

        # RoboDojo rig cameras (roboclock.envs.robodojo_rig): head ``rgb`` and both wrists,
        # 640x480 each (+ depth with --depth); zero images under sim.no_render.
        spaces = {
            key: gym.spaces.Box(0, 255, shape, dtype=np.uint8)
            if len(shape) == 3
            else gym.spaces.Box(0, np.inf, shape, dtype=np.float32)
            for key, shape in observation_shapes(depth).items()
        }
        spaces.update(
            joint_positions=gym.spaces.Box(-np.inf, np.inf, (16,), dtype=np.float32),
            ee_pose=gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32),
            gripper_width=gym.spaces.Box(0.0, 0.088, (2,), dtype=np.float32),
            sim_time=gym.spaces.Box(0.0, np.inf, (1,), dtype=np.float32),
            instruction=gym.spaces.Text(512, charset=string.printable + "–°"),
        )
        self.observation_space = gym.spaces.Dict(spaces)

    @property
    def backend(self):
        """The active task's native backend (stage, world, camera)."""
        return None if self.adapter is None else self.adapter.backend

    def reset(self, *, seed=None, options=None):
        from roboclock.envs.sports.adapters import adapter_class
        from roboclock.envs.sports.config import with_task_rates
        from roboclock.envs.sports.tasks import make_task

        super().reset(seed=seed)
        options = options or {}
        actual_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        task_id = str(options.get("task_id", self.task_id))
        difficulty = options.get("difficulty", self.difficulty)
        task = make_task(task_id, difficulty, actual_seed, self.settings)
        # One backend at a time: each task builds its own stage in this SimulationApp.
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        self.task = None
        self.config = with_task_rates(self.settings, task_id)
        adapter = adapter_class(task_id)(self.runtime, task, self.settings)
        try:
            adapter.build()
        except BaseException:
            adapter.close()
            raise
        task.bind(adapter)
        if self.depth and not adapter.no_render:
            adapter.enable_depth()
        self.adapter, self.task = adapter, task
        self.done = False
        return self.get_obs(), self._info()

    def get_obs(self):
        """Proprioception now; camera images when first read (roboclock.envs.observation)."""
        if self.adapter is None:
            raise RuntimeError("Call reset() before observing")
        adapter, depth = self.adapter, self.depth
        return Observation(
            adapter.observation(depth=depth, images=False),
            image_keys(self.observation_space),
            lambda: adapter.capture_images(depth),
            lambda: adapter.time,
        )

    def _info(self):
        if not self.oracle:
            return {}
        return {
            "metrics": self.task.metrics(),
            "physical_state": self.adapter.state(),
            "episode_spec": self.task.spec,
        }

    def _require_running(self):
        if self.task is None:
            raise RuntimeError("Call reset() before step()")
        if self.done:
            raise RuntimeError("Episode has ended; call reset()")

    def _update_done(self):
        self.done = bool(self.adapter.done or self.adapter.time >= self.task.spec.max_time - 1e-9)

    def set_action(self, action):
        """Make `action` the joint targets in effect from the current physics tick on."""
        self._require_running()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain 16 finite Dual X5 joint targets")
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("Action exceeds ARX X5 joint limits")
        adapter = self.adapter
        adapter.apply(adapter.joints.to_native(action, adapter.command()))
        self.task.action_count += 1

    def step(self, action):
        self.set_action(action)
        adapter = self.adapter
        adapter.control_step()
        self._update_done()
        self._report_progress()
        terminated = bool(adapter.done)
        truncated = not terminated and self.done
        return self.get_obs(), float(self.task.joint_success), terminated, truncated, self._info()

    def _report_progress(self):
        interval = self.settings.get("progress_seconds")
        if not interval:
            return
        last = getattr(self, "_last_progress", None)
        if last is None or self.adapter.time < last or self.adapter.time - last >= interval:
            self._last_progress = self.adapter.time
            print(
                f"SPORTS {self.task.task_id} sim={self.adapter.time:.2f}s "
                f"ticks={self.adapter.ticks} done={self.done}",
                flush=True,
            )

    def advance_latency(self, seconds: float):
        """Advance physics while the previous joint targets stay active."""
        self._require_running()
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Latency must be finite and nonnegative")
        dt = self.adapter.physics_dt
        remaining = max(0, math.ceil((self.task.spec.max_time - self.adapter.time - 1e-9) / dt))
        self.adapter.advance(min(math.ceil(seconds / dt - 1e-12), remaining))
        self._update_done()

    def render(self):
        return self.get_obs()["rgb"]

    def benchmark_metadata(self):
        import json
        from pathlib import Path

        from roboclock.envs import robodojo_rig
        from roboclock.envs.sports.tasks import CATALOG
        from roboclock.simulators.isaac.runtime import simulator_version

        manifest = Path(__file__).parent / "common/assets/dual_arx_x5/manifest.json"
        return {
            "environment": self.environment_id,
            "simulator": "isaacsim==" + simulator_version(),
            "robot": ROBOT_ID,
            "joint_order": list(JOINT_ORDER),
            "control_mode": "joint_position",
            "asset_catalog": {
                "robot": json.loads(manifest.read_text()) if manifest.exists() else None,
                "scenes": "assets under each sub-package of src/roboclock/envs/sports",
            },
            "tasks": {
                task_id: dict(
                    name=name, discipline=discipline, source=f"roboclock.envs.sports.{source}"
                )
                for task_id, name, discipline, source in CATALOG
            },
            "task_rates": self.settings["tasks"],
            "robodojo_rig": robodojo_rig.describe(),
        }

    def episode_metadata(self):
        from roboclock.envs.sports.tasks import jsonable

        task, adapter = self.task, self.adapter
        return jsonable(
            {
                "scene_id": f"sports_{task.task_id}_{task.source}",
                "robot": ROBOT_ID,
                "joint_order": list(JOINT_ORDER),
                "control_mode": "joint_position",
                "episode_spec": asdict(task.spec),
                **adapter.metadata(),
            }
        )

    def close(self):
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        if self.owns_runtime:
            self.runtime.close()
