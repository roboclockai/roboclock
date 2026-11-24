import numpy as np

from roboclock.envs.lab.mdp.events import GREEN, RED, SceneCommand
from roboclock.envs.lab.mdp.geometry import bottom_tip, fits_xy
from roboclock.envs.lab.mdp.state import upright_error
from roboclock.envs.lab.tasks.base import WindowTask


class ProbeAlignment(WindowTask):
    task_id = "A4"
    name = "probe_alignment"
    discipline = "physics"
    asset = "probe"
    instruction = "Insert the upright probe into the moving slot when it aligns with the green reference marker."

    amplitude = 0.08
    alignment_tolerance = 0.012

    def process(self, dt):
        self.omega = np.pi / (self.spec.valid_end - self.spec.valid_start)
        self.fixture_offset = self.amplitude * np.sin(self.omega * self.time + self.spec.phase)
        aligned = abs(self.fixture_offset) < self.alignment_tolerance
        return [
            SceneCommand("move_target", "destination", self.fixture_offset),
            SceneCommand("indicator", "indicator", GREEN if aligned else RED),
        ]

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        target = self.layout.targets["destination"]
        tip = bottom_tip(self, "sample_0", obj)
        inserted = (
            "sample_0" in self.insertion_times
            and obj.grasped
            and fits_xy(self, "sample_0", obj, target, self.fixture_offset)
            and target.support_z - 0.003 <= tip[2] <= target.rim_z - target.insertion_depth
            and upright_error(obj.orientation) < self.spec.orientation_tolerance
        )
        if inserted and self.action_time is None:
            self.record_action(self.insertion_times["sample_0"])
            if (
                abs(self.insertion_offsets.get("sample_0", self.fixture_offset))
                >= self.alignment_tolerance
            ):
                phase = self.omega * self.time + self.spec.phase
                nearest = abs((phase + np.pi / 2) % np.pi - np.pi / 2)
                error = (
                    max(0, nearest - np.arcsin(self.alignment_tolerance / self.amplitude))
                    / self.omega
                )
                self.violation("invalid_alignment", error)
        self.dwell["insertion"] = self.dwell.get("insertion", 0) + dt if inserted else 0
        if self.dwell["insertion"] >= 0.04:
            self.correct.add("sample_0")
            self.attempted.add("sample_0")
            self.goal_achieved = True

    def metrics(self):
        result = super().metrics()
        result["alignment_spec"] = {
            "amplitude": self.amplitude,
            "tolerance": self.alignment_tolerance,
            "omega": np.pi / (self.spec.valid_end - self.spec.valid_start),
            "phase": self.spec.phase,
        }
        return result
