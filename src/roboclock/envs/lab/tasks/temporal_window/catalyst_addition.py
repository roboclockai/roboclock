import numpy as np

from roboclock.envs.lab.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.lab.tasks.base import WindowTask


class CatalystAddition(WindowTask):
    task_id = "A3"
    name = "catalyst_addition"
    discipline = "chemistry"
    asset = "pellet"
    instruction = (
        "Release exactly one catalyst pellet into the vessel while the reaction indicator is green."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.additions = {}

    def process(self, dt):
        self.progress = float(
            np.clip(
                0.45
                + 0.15
                * (self.time - self.spec.valid_start)
                / (self.spec.valid_end - self.spec.valid_start),
                0,
                1,
            )
        )
        if self.time > self.spec.valid_end and self.action_time is None:
            self.violation("missed_window", self.time - self.spec.valid_end)
        color = BLUE if self.progress < 0.45 else GREEN if self.progress <= 0.60 else RED
        return [
            SceneCommand("indicator", "indicator", color),
            SceneCommand("target_color", "destination", color),
        ]

    def evaluate(self, state, dt):
        for name, obj in state.objects.items():
            if name in self.insertion_times and not obj.grasped and name in self.moved:
                if name not in self.additions:
                    self.additions[name] = self.insertion_times[name]
                    self.record_action(self.additions[name])
                    self.check_window(self.additions[name])
            if name in self.additions and self.placed(name, state, dt):
                self.correct.add(name)
                self.attempted.add(name)
        if len(self.additions) > 1:
            self.violation("multiple_pellets")
        self.goal_achieved = len(self.additions) == 1 and len(self.correct) == 1
