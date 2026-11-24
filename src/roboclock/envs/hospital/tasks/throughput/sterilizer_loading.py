import numpy as np

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, ThroughputTask

CYCLE_SHOW = 5.0  # seconds the cycle runs on the dial after START, so the run is visible before the episode ends


class SterilizerLoading(ThroughputTask):
    """The next case needs these instruments: load them all into the sterilizer, then press START before the cutoff.

    The countdown is the time until the cycle must start (cycle_time). The START press only counts once every
    instrument is in the basket; t_start < cycle_time, strict.
    """

    task_id = "C3"
    name = "sterilizer_loading"
    discipline = "sterile_processing"

    def __init__(self, spec):
        super().__init__(spec)
        self.start_time = None

    def bind(self, layout, placement_dwell=0.2):
        super().bind(layout, placement_dwell)
        self.button = np.array(layout.info["sterilizer"]["start"])

    @property
    def press_ready(self):
        return len(self.correct) == len(self.layout.objects)

    @property
    def press_done(self):
        return self.start_time is not None

    def budget_text(self):
        return f"{self.spec.cycle_time:.1f} seconds (the cycle must start by then)"

    def gen_instruction(self):
        return (
            "The next case needs these instruments clean. Load every used instrument into the sterilizer basket, then "
            f"press the green START button within {self.spec.cycle_time:.0f} seconds."
        )

    def process(self, dt):
        cycle = self.spec.cycle_time
        if self.start_time is not None:
            light = GREEN
        elif self.time + EPS >= cycle:
            light = RED
        else:
            light = BLUE
        commands = [SceneCommand("indicator", "indicator", light)]
        if self.start_time is not None:
            left = max(0.0, 1.0 - (self.time - self.start_time) / CYCLE_SHOW)
            commands += self.timer("indicator", left, "green")
        else:
            commands += self.timer("indicator", (cycle - self.time) / cycle)
        return commands

    def accepts(self, name, container):
        return True

    def _pressing(self, state):
        # Either arm may press START.
        return any(
            float(np.linalg.norm(ee[:2] - self.button[:2])) < 0.016
            and self.button[2] - 0.02 < ee[2] < self.button[2] + 0.012
            for ee in state.tool_points()
        )

    def evaluate(self, state, dt):
        super().evaluate(state, dt)
        if self.start_time is None and self.press_ready:
            key = "start"
            self.dwell[key] = self.dwell.get(key, 0.0) + dt if self._pressing(state) else 0.0
            if self.dwell[key] + EPS >= self.placement_dwell:
                self.start_time = self.time
                self.record_action(self.time)
                self.event_log.append({"time": self.time, "type": "cycle_started"})
        if self.start_time is None and self.time + EPS >= self.spec.cycle_time:
            self.violation("late_start", self.time - self.spec.cycle_time)
        self.goal_achieved = self.press_ready and self.start_time is not None

    @property
    def physical_success(self):
        return self.goal_achieved and super().physical_success

    @property
    def temporal_success(self):
        return self.start_time is not None and self.start_time + EPS < self.spec.cycle_time

    @property
    def terminated(self):
        if self.start_time is not None:
            return self.time - self.start_time + EPS >= CYCLE_SHOW
        return self.time + EPS >= self.spec.cycle_time

    def metrics(self):
        result = super().metrics()
        result["start_time"] = self.start_time
        return result
