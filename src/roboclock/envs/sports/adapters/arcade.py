"""B5 two-stick taiko (arcade kind A2) and B1 nine-hole moles (``arcade/``).

Mirrors ``arcade/demo.py``: the X5 runs both tasks on ``DualTaikoBackend``
(two physically grasped tools), the original ``TaikoTask``/``MalletMoles``
evaluator observes every tick, and the joint command is applied once per
control period with gravity feed-forward (``RigView.apply_action``); the
drives then hold that target. Drum springs and mole drives are task machinery
inside the backend tick.
"""

from types import SimpleNamespace

import numpy as np

from roboclock.envs.sports.adapters.base import Adapter, ensure_extensions, episode_rates


class ArcadeAdapter(Adapter):
    family = "arcade"

    def build(self):
        ensure_extensions()
        from pxr import UsdGeom

        from roboclock.envs.sports.arcade.dual_taiko import DualTaikoBackend
        from roboclock.envs.sports.arcade.tasks import MalletMoles, TaikoTask
        from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_camera

        c = self.c
        if self.spec.task_id == "B5" and not c["v2"].get("dual_sticks"):
            raise ValueError("B5 (taiko) requires the dual-stick X5 configuration")
        if self.spec.task_id == "B1" and not c["v2"].get("dual_mallets"):
            raise ValueError("B1 requires the dual-mallet X5 configuration")
        cls = TaikoTask if self.spec.task_id == "B5" else MalletMoles
        self.native = cls(self.spec.difficulty, self.spec.seed, c)
        b = DualTaikoBackend(self.runtime, c)
        self.backend = b
        b.reset(self.native)
        apply_presentation_camera(b)
        title = b.stage.GetPrimAtPath("/World/Title")
        if title:
            UsdGeom.Imageable(title).MakeInvisible()
        self.after_build()

    @property
    def time(self):
        return float(self.native.time)

    @property
    def done(self):
        return bool(self.native.terminated or self.native.time >= self.native.spec.max_time - 1e-9)

    def apply(self, native):
        self.backend.apply_action(native)
        self.native.action_count += 1

    def tick(self):
        b, task = self.backend, self.native
        b.tick(render=False)
        task.observe(b.state(), self.physics_dt)
        task.collision_count = b.collision_count
        self.ticks += 1

    def after_control(self):
        self.backend.update_cues()

    def state(self):
        return self.backend.state()

    def events(self):
        return self.native.event_log

    def result(self):
        task = self.native
        return dict(
            task=self.spec.task_id,
            variant="basic",
            seed=self.spec.seed,
            difficulty=self.spec.difficulty,
            metrics=task.metrics(),
            events=task.event_log,
            controller="privileged bounded native robot IK; not a learned visual policy",
            full_episode=True,
            evaluation_end=task.time,
            embodiment=self.c["embodiment"],
            **episode_rates(self),
        )

    def joint_success(self):
        return bool(self.native.joint_success)

    def make_controller(self):
        if self.spec.task_id == "B5":
            from roboclock.envs.sports.arcade.dual_taiko import DualToolOracle as cls
        else:
            from roboclock.envs.sports.arcade.dual_moles import DualMoleOracle as cls
        env = SimpleNamespace(task=self.native, backend=self.backend, config=self.c)
        self.controller = ArcadeController(self, cls(env))
        return self.controller


class ArcadeController:
    def __init__(self, adapter, policy):
        self.adapter, self.policy = adapter, policy
        if hasattr(policy, "reset"):
            policy.reset(self.observation())

    @property
    def phase(self):
        return getattr(self.policy, "phase", None)

    def observation(self):
        return {"joint_positions": self.adapter.backend.robot.get_joint_positions().copy()}

    def act(self):
        return np.asarray(self.policy.act(self.observation())).copy()
