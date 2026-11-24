from dataclasses import replace

from roboclock.envs.lab.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.lab.tasks.base import WindowTask


class TemperatureRemoval(WindowTask):
    task_id = "A1"
    name = "temperature_removal"
    discipline = "chemistry"
    asset = "bottle"
    instruction = "Grasp the heated bottle while the indicator is green (55–65 °C), then place it in the tray."

    def __init__(self, spec):
        width = spec.valid_end - spec.valid_start
        self.heat_rate = 10 / width
        super().__init__(
            replace(spec, valid_start=30 / self.heat_rate, valid_end=40 / self.heat_rate)
        )

    def process(self, dt):
        in_heater = self.previous is None or self.layout.heating_zone.contains(
            self.previous.objects["sample_0"].position
        )
        if in_heater:
            self.temperature += self.heat_rate * dt
        color = BLUE if self.temperature < 55 else GREEN if self.temperature <= 65 else RED
        if self.temperature > 65 + 1e-8 and self.action_time is None:
            self.violation("missed_window", (self.temperature - 65) / self.heat_rate)
        return [SceneCommand("indicator", "indicator", color)]

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        if obj.grasped and self.action_time is None:
            self.record_action()
            if self.temperature < 55 - 1e-8:
                self.violation("early_action", (55 - self.temperature) / self.heat_rate)
            elif self.temperature > 65 + 1e-8:
                self.violation("late_action", (self.temperature - 65) / self.heat_rate)
        if not self.layout.heating_zone.contains(obj.position) and self.action_time is None:
            self.violation("removed_without_grasp")
        if self.placed("sample_0", state, dt):
            self.correct.add("sample_0")
            self.attempted.add("sample_0")
            self.goal_achieved = True
