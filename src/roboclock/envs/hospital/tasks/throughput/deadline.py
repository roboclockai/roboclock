from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, ThroughputTask


class DeadlineThroughput(ThroughputTask):
    """Every item in its container before the countdown (cycle_time) runs out; the episode ends at the last item or the deadline."""

    def __init__(self, spec):
        super().__init__(spec)
        self.done_time = None

    def budget_text(self):
        return f"{self.spec.cycle_time:.1f} seconds (the deadline)"

    def process(self, dt):
        cycle = self.spec.cycle_time
        if self.done_time is not None:
            light = GREEN
        elif self.time + EPS >= cycle:
            light = RED
        else:
            light = BLUE
        commands = [SceneCommand("indicator", "indicator", light)]
        if self.done_time is not None:
            commands += self.timer("indicator", 1.0, "green")
        else:
            commands += self.timer("indicator", (cycle - self.time) / cycle)
        return commands

    def evaluate(self, state, dt):
        super().evaluate(state, dt)
        if self.done_time is None and self.goal_achieved:
            self.done_time = self.time
            self.record_action(self.time)
            if self.time + EPS >= self.spec.cycle_time:
                self.violation("late_restock", self.time - self.spec.cycle_time)
        if self.done_time is None and self.time + EPS >= self.spec.cycle_time:
            self.violation("late_restock", self.time - self.spec.cycle_time)

    @property
    def temporal_success(self):
        return self.done_time is not None and self.done_time + EPS < self.spec.cycle_time

    @property
    def physical_success(self):
        return self.goal_achieved and super().physical_success

    @property
    def terminated(self):
        return self.goal_achieved or self.time + EPS >= self.spec.cycle_time

    def metrics(self):
        result = super().metrics()
        result["restock_time"] = self.done_time
        return result
