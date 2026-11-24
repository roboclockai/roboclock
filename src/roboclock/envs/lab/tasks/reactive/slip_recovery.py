import numpy as np

from roboclock.envs.lab.mdp.events import SceneCommand
from roboclock.envs.lab.mdp.geometry import safe_workspace
from roboclock.envs.lab.tasks.base import ReactiveTask


class SlipRecovery(ReactiveTask):
    task_id = "B2"
    name = "slip_recovery"
    discipline = "biology"
    instruction = (
        "Recover or block the slipping sample after release, before it falls off the workbench."
    )

    def __init__(self, spec):
        super().__init__(spec)
        rng = np.random.default_rng(np.random.SeedSequence([spec.seed, 202]))
        self.sliding_observed = False
        self.slip_velocity = np.array([0.0, -float(rng.uniform(0.45, 0.65)), 0.0])

    def process(self, dt):
        before = self.event_time
        commands = super().process(dt)
        if before is None and self.event_time is not None:
            commands.append(SceneCommand("release", "sample_0", self.slip_velocity))
        return commands

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        if self.event_time is None:
            if obj.grasped:
                self.record_reaction()
            return
        if np.linalg.norm(obj.velocity[:2]) > 0.12 or obj.release_speed > 0.12:
            self.sliding_observed = True
        if not safe_workspace(self, "sample_0", obj):
            if "left_safe_workspace" not in self.physical_failures:
                self.physical_failures.append("left_safe_workspace")
            return
        recovered = self.sliding_observed and (
            obj.grasped
            or (obj.robot_contact and obj.table_contact and np.linalg.norm(obj.velocity) < 0.08)
        )
        self.dwell["recover"] = self.dwell.get("recover", 0) + dt if recovered else 0
        if recovered and self.action_time is None:
            self.record_reaction()
        if (
            self.dwell["recover"] >= self.recovery_dwell
            and obj.position[2] >= self.layout.table_height
        ):
            self.goal_achieved = True
            self.correct.add("sample_0")
            self.attempted.add("sample_0")

    @property
    def terminated(self):
        return super().terminated or bool(self.physical_failures)
