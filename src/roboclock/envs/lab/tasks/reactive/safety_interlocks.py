"""B5–B7: measured door, vent and hose-clamp emergency responses.

Heat, pressure and liquid are explicit scenario models, not multiphysics solvers.
Deadlines measure completed intervention and are strict; failures are terminal.
"""

import numpy as np

from roboclock.envs.lab.mdp.events import GREEN, RED, SceneCommand
from roboclock.envs.lab.tasks.base import ReactiveTask
from roboclock.physics.geometry import orientation_error


class SafetyInterlock(ReactiveTask):
    laboratory_v2 = True
    safety_task = True
    discipline = "chemistry"
    asset = "safety_actuator"
    failure_kind = "safety_failure"
    success_kind = "intervention_complete"

    def __init__(self, spec):
        super().__init__(spec)
        self.failed = False
        self.secured = False
        self.contact_seen = False
        self.grasp_seen = False
        self.safe_dwell = 0.0
        self.controls = {}
        self.volume_ml = 9990.0
        self.fault_release_time = None
        self.initial_robot_state = None
        self.pre_event_max_ee_drift = 0.0
        self.pre_event_max_orientation_drift = 0.0

    def process(self, dt):
        commands = []
        if self.task_id == "B5":
            if self.fault_release_time is None and self.time >= self.spec.event_time:
                self.fault_release_time = self.time
                self.event_log.append({"time": self.time, "type": "internal_catch_released"})
                commands.append(SceneCommand("safety_fault_release", "equipment", None))
        else:
            self.trigger_event()
        # Evaluated after the physics tick, so the final measured state is used.
        return commands + [SceneCommand("safety_visual", "equipment", self.visual_state())]

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        self.controls = dict(state.controls)
        self.secured = bool(self.controls.get("secured", False))
        if self.initial_robot_state is None:
            self.initial_robot_state = self.previous or state
        if self.event_time is None:
            self.pre_event_max_ee_drift = max(
                self.pre_event_max_ee_drift,
                float(np.linalg.norm(state.ee_position - self.initial_robot_state.ee_position)),
            )
            self.pre_event_max_orientation_drift = max(
                self.pre_event_max_orientation_drift,
                float(
                    orientation_error(state.ee_orientation, self.initial_robot_state.ee_orientation)
                ),
            )
            # Moving before the alarm (e.g. hovering over the actuator) is allowed; only touching
            # or securing the actuator early is a violation. The drift is kept as a metric.
            if obj.robot_contact or self.secured:
                self.violation("pre_event_action")
            if (
                self.task_id == "B5"
                and self.fault_release_time is not None
                and self.controls.get("opening_reached", False)
            ):
                self.event_time = self.time
                self.event_log.append({"time": self.time, "type": "door_open_alarm"})
            else:
                return
        self.contact_seen |= obj.robot_contact
        self.grasp_seen |= obj.grasped
        if self.failed:
            return
        elapsed = self.time - self.event_time
        if self.task_id == "B7" and not self.secured:
            # Ten mL of headroom remains at the IR trigger. Level is mass balance.
            self.volume_ml = min(10001.0, self.volume_ml + 10.0 / self.spec.max_latency * dt)
        valid = self.secured and self.contact_seen
        if self.task_id == "B5":
            valid &= self.grasp_seen
        if self.action_time is None and elapsed >= self.spec.max_latency - 1e-9:
            self.failed = True
            self.violation("late_reaction", max(0.0, elapsed - self.spec.max_latency))
            self.physical_failures.append(self.failure_kind)
            self.event_log.append({"time": self.time, "type": self.failure_kind})
            if self.task_id == "B7":
                self.volume_ml = 10001.0
            return
        if valid and self.action_time is None:
            self.record_reaction()
            self.event_log.append({"time": self.time, "type": self.success_kind})
        if self.action_time is not None and not valid:
            self.failed = True
            self.violation("intervention_reversed")
            self.physical_failures.append("intervention_reversed")
        self.safe_dwell = self.safe_dwell + dt if valid else 0.0
        if self.action_time is not None and self.safe_dwell >= 0.20 - 1e-9:
            self.goal_achieved = True
            self.correct.add("sample_0")
            self.attempted.add("sample_0")

    def visual_state(self):
        elapsed = 0.0 if self.event_time is None else self.time - self.event_time
        exposed = elapsed if self.action_time is None else self.action_time - self.event_time
        return {
            "time": self.time,
            "fault_release_time": self.fault_release_time,
            "alarm": self.event_time is not None and self.action_time is None and not self.failed,
            "failed": self.failed,
            "secured": self.secured,
            "completed": self.goal_achieved,
            "window_remaining": max(0.0, self.spec.max_latency - elapsed),
            # A door alarm is immediate; the chamber readout drifts slowly.
            # The benchmark deadline remains separate from any thaw prediction.
            "temperature_c": -80.0 + 0.12 * max(0.0, exposed),
            "pressure_kpa": 101.3
            if self.secured
            else 20.0 + (8.0 * np.sin(elapsed * 30.0) if self.event_time is not None else 0.0),
            "volume_ml": self.volume_ml,
            "flow_ml_s": 0.0 if self.secured else 10.0 / self.spec.max_latency,
            "failure_kind": self.failure_kind if self.failed else None,
        }

    def metrics(self):
        return dict(
            super().metrics(),
            intervention_secured=self.secured,
            robot_contact_observed=self.contact_seen,
            bilateral_grasp_observed=self.grasp_seen,
            secured_dwell=self.safe_dwell,
            final_controls=self.controls,
            volume_ml=self.volume_ml if self.task_id == "B7" else None,
            scenario_version=self.name + "_v4_internal_catch",
            process_model="symbolic thermal/pressure and volume balance; rigid-body PhysX actuation",
            fault_release_time=self.fault_release_time,
            pre_event_max_ee_drift_m=self.pre_event_max_ee_drift,
            pre_event_max_orientation_drift_rad=self.pre_event_max_orientation_drift,
        )

    @property
    def terminated(self):
        return super().terminated or self.failed

    def terminal_commands(self):
        return [
            SceneCommand("safety_visual", "equipment", self.visual_state()),
            SceneCommand("indicator", "indicator", GREEN if self.joint_success else RED),
        ]


class FreezerDoor(SafetyInterlock):
    task_id = "B5"
    name = "freezer_door_rescue"
    failure_kind = "sample_thawed_scenario"
    success_kind = "freezer_door_latched"
    instruction = (
        "When the freezer door-alarm LED flashes red beside its temperature readout and beeps, grasp the top "
        "handle and close the door. It starts closed; the internal catch releases and the hinge "
        "springs open before the alarm. Hold the initial robot posture until the alarm. "
        "The deadline is a compressed scenario, not a thaw prediction."
    )


class VacuumVent(SafetyInterlock):
    task_id = "B6"
    name = "vacuum_bumping_relief"
    failure_kind = "vacuum_bumping"
    success_kind = "vacuum_breaker_opened"
    instruction = (
        "When the vacuum pressure needle oscillates and its red pressure lamp and amber beacon blink, "
        "press the large VENT breaker plunger fully down to admit air through the restricted "
        "vent and isolate the vacuum pump. Complete the stroke before the bumping deadline."
    )


class WasteClamp(SafetyInterlock):
    task_id = "B7"
    name = "waste_carboy_overflow"
    failure_kind = "hazardous_waste_overflow"
    success_kind = "waste_hose_clamped"
    instruction = (
        "When the LEVEL lamp on the bench level-alarm console, wired to the waste carboy's IR "
        "level sensor, changes from green to red and beeps, "
        "press the hose pinch clamp fully down to stop the incoming pumped waste. "
        "Complete closure before the remaining capacity is exhausted."
    )
