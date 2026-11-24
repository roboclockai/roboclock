"""B1: isolate a nearly dry beaker after a scripted hotplate over-temperature alarm.

The response timestamp is a retained lift, not the start of an arm motion or a
grasp on the plate. Thermal values are scenario cues, not a glass fracture model.
"""

import numpy as np

from roboclock.envs.lab.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.lab.mdp.geometry import object_extents, safe_workspace
from roboclock.envs.lab.mdp.state import upright_error
from roboclock.envs.lab.tasks.base import ReactiveTask


class AlarmTransfer(ReactiveTask):
    task_id = "B1"
    name = "hotplate_isolation"
    discipline = "chemistry"
    asset = "beaker"
    hotplate_task = True
    laboratory_v2 = True
    isolation_clearance = 0.020
    isolation_dwell = 0.050
    instruction = (
        "When the hotplate display flashes red, grasp the beaker and lift its entire base "
        "at least 2 cm above the hotplate. Keep it upright and place it in the safety tray."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.grasp_time = None
        self.isolation_elapsed = 0.0
        self.temperature = 85.0

    def gen_instruction(self):
        return (
            self.instruction
            + f" Complete and retain the lift for 0.05 s within {self.spec.max_latency:.2f} s "
            f"of the alarm; finish placement within {self.spec.transfer_deadline:.2f} s. "
            "Do not grasp or remove the beaker before the alarm."
        )

    def process(self, dt):
        super().process(dt)
        self.alarm = self.event_time is not None and self.action_time is None
        if self.event_time is not None:
            exposed = (self.action_time or self.time) - self.event_time
            cooling = max(0.0, self.time - self.action_time) if self.action_time else 0.0
            self.temperature = max(25.0, min(250.0, 85.0 + 18.0 * exposed) - 8.0 * cooling)
        color = (
            RED
            if self.violations
            else GREEN
            if self.action_time is not None
            else (RED if self.alarm else BLUE)
        )
        return [
            SceneCommand("indicator", "indicator", color),
            SceneCommand("hotplate_visual", "hotplate", self.visual_state()),
        ]

    def visual_state(self):
        return {
            "temperature_c": self.temperature,
            "alarm": self.event_time is not None and self.action_time is None,
            "isolated": self.action_time is not None,
            "failed": bool(self.violations),
            "time": self.time,
            "alarm_elapsed": 0.0 if self.event_time is None else self.time - self.event_time,
        }

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        extent = object_extents(self, "sample_0", obj)
        heater = self.layout.heating_zone
        bottom = obj.position[2] - extent[2]
        if obj.grasped and self.grasp_time is None:
            self.grasp_time = self.time
            self.event_log.append({"time": self.time, "type": "beaker_grasped"})
        if self.event_time is None:
            if (
                obj.grasped
                or np.linalg.norm(obj.position - self.layout.objects["sample_0"].position) > 0.025
            ):
                self.violation("pre_event_action")
            return
        lifted = (
            obj.grasped
            and bottom >= heater.support_z + self.isolation_clearance
            and "heater" not in obj.fixture_contacts
            and upright_error(obj.orientation) < 0.30
            and safe_workspace(self, "sample_0", obj)
        )
        if self.action_time is None:
            self.isolation_elapsed = self.isolation_elapsed + dt if lifted else 0.0
            if self.isolation_elapsed + 1e-9 >= self.isolation_dwell:
                self.record_reaction()
                if self.time - self.event_time >= self.spec.max_latency - 1e-9:
                    self.violation(
                        "late_reaction",
                        max(0.0, self.time - self.event_time - self.spec.max_latency),
                    )
                self.event_log.append({"time": self.time, "type": "beaker_isolated"})
            elif self.time - self.event_time >= self.spec.max_latency - 1e-9:
                self.violation("late_reaction", self.time - self.event_time - self.spec.max_latency)
        elif (
            np.all(
                np.abs(obj.position[:2] - heater.position[:2])
                < heater.half_extents[:2] + extent[:2]
            )
            and bottom < heater.support_z + 0.010
        ):
            self.violation("returned_to_hotplate")
        if (
            self.event_time is not None
            and self.time - self.event_time > self.spec.transfer_deadline
            and not self.goal_achieved
        ):
            self.violation(
                "transfer_deadline", self.time - self.event_time - self.spec.transfer_deadline
            )
        if self.action_time is not None and self.placed("sample_0", state, dt):
            self.goal_achieved = True
            self.correct.add("sample_0")
            self.attempted.add("sample_0")

    def metrics(self):
        result = super().metrics()
        result.update(
            grasp_time=self.grasp_time,
            isolation_time=self.action_time,
            isolation_window=self.spec.max_latency,
            overheat_failure="late_reaction" in self.violations,
            scenario_version="hotplate_isolation_v2",
        )
        return result
