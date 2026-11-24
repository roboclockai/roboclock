"""V2 electrical faults and unstable glassware; deadlines use measured responses.

Voltage, temperature and spill are benchmark scenario models. Disconnect angle,
robot contacts, rotor speed and test-tube motion come from PhysX.
"""

import numpy as np

from roboclock.envs.lab.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.lab.mdp.geometry import object_extents, safe_workspace
from roboclock.envs.lab.mdp.state import upright_error
from roboclock.envs.lab.tasks.base import ReactiveTask


class PowerIsolation(ReactiveTask):
    laboratory_v2 = True
    power_task = True
    asset = "knife_switch"
    discipline = "electrical_engineering"

    def __init__(self, spec):
        super().__init__(spec)
        self.fault_time = None
        self.power_on = True
        self.switch_angle = 0.0
        self.wheel_speed = 0.0
        self.rotating_seen = False
        self.isolated_dwell = 0.0

    def process(self, dt):
        if self.fault_time is None and self.time >= self.spec.event_time:
            self.fault_time = self.time
            self.event_log.append(
                {
                    "time": self.time,
                    "type": "insulation_overheat"
                    if self.task_id == "B2"
                    else "internal_motor_stall",
                }
            )
            if self.task_id == "B2":
                self.event_time = self.time
                self.event_log.append({"time": self.time, "type": "event_triggered"})
        if self.event_time is not None and self.action_time is None:
            if self.time - self.event_time >= self.spec.max_latency - 1e-9:
                self.violation(
                    "late_reaction", max(0, self.time - self.event_time - self.spec.max_latency)
                )
        self.update_fault_model(dt)
        color = (
            RED
            if self.violations
            else GREEN
            if not self.power_on
            else BLUE
            if self.task_id == "B2" and self.event_time is not None
            else RED
            if self.event_time is not None
            else BLUE
        )
        return [
            SceneCommand("indicator", "indicator", color),
            SceneCommand("equipment_visual", "equipment", self.visual_state()),
        ]

    def update_fault_model(self, dt):
        """Task-specific simulated electrical/thermal consequences."""

    def evaluate(self, state, dt):
        controls = state.controls
        self.power_on = bool(controls.get("power_on", True))
        self.switch_angle = float(controls.get("switch_angle_rad", 0))
        self.wheel_speed = float(controls.get("wheel_speed_rad_s", 0))
        self.rotating_seen |= abs(self.wheel_speed) > 2.0
        if (
            self.task_id == "B3"
            and self.fault_time is not None
            and self.event_time is None
            and self.rotating_seen
            and self.power_on
            and abs(self.wheel_speed) < 0.35
        ):
            self.event_time = self.time
            self.event_log.append({"time": self.time, "type": "motor_stalled"})
        if self.event_time is None and state.objects["sample_0"].grasped:
            self.violation("pre_event_action")
        if self.action_time is not None and self.power_on:
            self.violation("power_reconnected")
            self.isolated_dwell = 0.0
        if not self.power_on:
            if self.event_time is None:
                self.violation("pre_event_action")
            elif self.action_time is None:
                self.record_reaction()
                if self.time - self.event_time >= self.spec.max_latency - 1e-9:
                    self.violation("late_reaction")
                self.event_log.append(
                    {"time": self.time, "type": "power_isolated", "opening_rad": self.switch_angle}
                )
            self.isolated_dwell += dt
            if self.action_time is not None and self.isolated_dwell >= 0.6:
                self.goal_achieved = True
                self.correct.add("sample_0")
                self.attempted.add("sample_0")

    def visual_state(self):
        elapsed = 0 if self.event_time is None else self.time - self.event_time
        return {
            "time": self.time,
            "fault": self.fault_time is not None,
            "alarm": self.event_time is not None and self.power_on,
            "power_on": self.power_on,
            "failed": bool(self.violations),
            "window_remaining": max(0, self.spec.max_latency - elapsed),
            "switch_angle_rad": self.switch_angle,
            "wheel_speed_rad_s": self.wheel_speed,
            "voltage_v": (18 if self.fault_time is not None else 12) if self.power_on else 0,
            "thermal_damage": "late_reaction" in self.violations,
        }

    def metrics(self):
        return dict(
            super().metrics(),
            fault_time=self.fault_time,
            power_isolated=not self.power_on,
            switch_angle_rad=self.switch_angle,
            rotor_previously_rotating=self.rotating_seen,
            thermal_damage="late_reaction" in self.violations,
            scenario_version=self.name + "_v2",
        )


class ThermalShortCircuit(PowerIsolation):
    task_id = "B2"
    name = "thermal_short_circuit"
    instruction = "Keep the knife switch closed while the electrical load's round status LED is green. When that same LED turns red, begin reaching and open the switch by at least 35 degrees within the reaction window. Timely isolation extinguishes the LED; a missed deadline causes a simulated short circuit and the same LED flashes red."

    def __init__(self, spec):
        super().__init__(spec)
        self.short_circuit = False
        self.insulation_damage = 0.0
        self.insulation_temperature = 35.0

    def update_fault_model(self, dt):
        if self.fault_time is not None and self.power_on:
            elapsed = self.time - self.fault_time
            self.insulation_damage = max(
                self.insulation_damage, min(1.0, elapsed / self.spec.max_latency)
            )
            self.insulation_temperature = 90.0 + 70.0 * self.insulation_damage
            if elapsed >= self.spec.max_latency - 1e-9 and not self.short_circuit:
                self.short_circuit = True
                self.physical_failures.append("short_circuit")
                self.violation("short_circuit")
        elif not self.power_on:
            self.insulation_temperature = max(25.0, self.insulation_temperature - 60.0 * dt)

    def visual_state(self):
        data = super().visual_state()
        data.update(
            phase="short_circuit"
            if self.short_circuit
            else "isolated"
            if not self.power_on
            else "overheating"
            if self.fault_time is not None
            else "normal",
            voltage_v=12 if self.power_on and not self.short_circuit else 0,
            short_circuit=self.short_circuit,
            warning_red=self.fault_time is not None and self.power_on and not self.short_circuit,
            short_indicator_red=self.short_circuit,
            status_indicator="flashing_red"
            if self.short_circuit
            else "off"
            if not self.power_on
            else "red"
            if self.fault_time is not None
            else "green",
            insulation_damage=self.insulation_damage,
            insulation_temperature_c=self.insulation_temperature,
        )
        return data

    def metrics(self):
        return dict(
            super().metrics(),
            short_circuit=self.short_circuit,
            insulation_damage=self.insulation_damage,
            insulation_temperature_c=self.insulation_temperature,
        )

    @property
    def terminated(self):
        return super().terminated or self.short_circuit


class MotorStall(PowerIsolation):
    task_id = "B3"
    name = "motor_stall"
    instruction = "Keep the robot in its initial posture while the wheel rotates. Only after detecting that the wheel has stopped, immediately move to and grasp the insulated disconnect handle and pull the knife switch open by at least 35 degrees to disconnect the DC motor supply before overheating. The STATE lamp on the bench turns from blue to red when the wheel has stopped. Do not cut power before the stall."


class UnstableTestTube(ReactiveTask):
    task_id = "B4"
    name = "test_tube_rescue"
    discipline = "chemistry"
    laboratory_v2 = True
    tube_task = True
    asset = "test_tube"
    rack_support = 0.035
    rack_rim = 0.080
    warning_tilt = np.deg2rad(32.0)
    instruction = "Watch the initially upright, half-filled test tube. A loose support slowly tilts it. Begin reaching and grasping only when its measured tilt reaches 32 degrees: the STATE lamp on the bench turns from blue to red at that moment. Then lift it clear and retain it upright before it falls or loses 0.75 mL of sample."

    def __init__(self, spec):
        super().__init__(spec)
        self.waiting_ee = None
        self.maximum_pre_event_ee_displacement = 0.0
        self.disturbance_time = None
        self.first_grasp_time = None
        self.grasp_tilt = None
        self.retained = 0.0
        self.tilt = 0.0
        self.max_tilt = 0.0
        from roboclock.envs.lab.tube_liquid import TubeLiquid

        self.liquid = TubeLiquid()

    def trigger_event(self):
        # The reaction window starts at measured near-fall tilt, not fault onset.
        if (
            self.event_time is None
            and self.disturbance_time is not None
            and self.tilt >= self.warning_tilt
        ):
            self.event_time = self.time
            self.event_log.append(
                {
                    "time": self.time,
                    "type": "event_triggered",
                    "tilt_degrees": float(np.rad2deg(self.tilt)),
                }
            )
            return True
        return False

    def process(self, dt):
        commands = super().process(dt)
        if self.disturbance_time is None and self.time >= self.spec.event_time:
            self.disturbance_time = self.time
            self.event_log.append({"time": self.time, "type": "support_started_tilting"})
            commands.append(SceneCommand("disturb_tube", "sample_0", None))
        commands.append(SceneCommand("equipment_visual", "equipment", self.visual_state()))
        return commands

    def evaluate(self, state, dt):
        obj = state.objects["sample_0"]
        prior_spill = self.liquid.spilled_ml
        self.liquid.step(obj.position, obj.orientation, obj.velocity, dt, self.layout.table_height)
        if prior_spill < 0.012 <= self.liquid.spilled_ml:
            self.event_log.append({"time": self.time, "type": "first_liquid_drop"})
        self.tilt = upright_error(obj.orientation)
        self.max_tilt = max(self.max_tilt, self.tilt)
        self.trigger_event()
        if self.event_time is None:
            if self.waiting_ee is None:
                self.waiting_ee = state.ee_position.copy()
            displacement = float(np.linalg.norm(state.ee_position - self.waiting_ee))
            self.maximum_pre_event_ee_displacement = max(
                self.maximum_pre_event_ee_displacement, displacement
            )
            # Reaching toward the tube before the warning tilt is allowed; grasping or touching it is not.
            if obj.grasped or obj.robot_contact:
                self.violation("pre_event_action")
            return
        if (
            self.liquid.spilled_ml >= self.liquid.spill_limit_ml
            and "sample_spilled" not in self.physical_failures
        ):
            self.physical_failures.append("sample_spilled")
            self.event_log.append({"time": self.time, "type": "sample_spilled"})
        if obj.grasped and self.first_grasp_time is None:
            self.first_grasp_time = self.time
            self.grasp_tilt = float(np.rad2deg(self.tilt))
            self.event_log.append(
                {"time": self.time, "type": "tube_grasped", "tilt_degrees": self.grasp_tilt}
            )
        bottom = obj.position[2] - object_extents(self, "sample_0", obj)[2]
        safe = (
            obj.grasped
            and self.tilt < 0.30
            and bottom >= self.layout.table_height + self.rack_rim + 0.020
            and not any(c.startswith("rack") for c in obj.fixture_contacts)
            and safe_workspace(self, "sample_0", obj)
        )
        self.retained = self.retained + dt if safe else 0.0
        if self.action_time is None and self.retained + 1e-9 >= 0.15:
            self.record_reaction()
            if self.time - self.event_time >= self.spec.max_latency - 1e-9:
                self.violation("late_reaction")
            self.event_log.append({"time": self.time, "type": "tube_rescued"})
            self.goal_achieved = True
            self.correct.add("sample_0")
            self.attempted.add("sample_0")

    def visual_state(self):
        return {
            **self.liquid.state(),
            "time": self.time,
            "disturbance_time": self.disturbance_time,
            "warning_tilt_degrees": float(np.rad2deg(self.warning_tilt)),
            "alarm": self.event_time is not None and self.action_time is None,
            "failed": bool(self.violations or self.physical_failures),
            "rescued": self.action_time is not None,
            "tilt_degrees": float(np.rad2deg(self.tilt)),
            # Legacy flag means excessive sample loss; small drops are reported in mL.
            "liquid_spilled": "sample_spilled" in self.physical_failures,
        }

    def metrics(self):
        return dict(
            super().metrics(),
            maximum_tilt_degrees=float(np.rad2deg(self.max_tilt)),
            retained_clearance_dwell=self.retained,
            maximum_pre_event_ee_displacement=self.maximum_pre_event_ee_displacement,
            disturbance_time=self.disturbance_time,
            first_grasp_time=self.first_grasp_time,
            grasp_tilt_degrees=self.grasp_tilt,
            **self.liquid.state(),
            scenario_version=self.name + "_v2",
        )
