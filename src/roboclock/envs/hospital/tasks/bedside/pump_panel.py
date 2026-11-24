"""B6 (B7 until 2026-11-19): press one requested pump key, read from the pump display, using measured travel.

The instruction is on the pump's own display and the order card at the response dial from the
first frame; the response clock starts at the instruction time (just after the scene settles) and
runs to the end of the required press. The robot starts from rest, folded near its base, so the
measured response includes the visible reach to the pump.

This is a device-interaction benchmark. Mute acknowledges an alarm; it never
clears an occlusion or restarts infusion. Bolus is an armed training mode with
no drug, dose or physiological model. Deadlines are benchmark assumptions.
"""

import numpy as np

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask

MODES = ("stop", "mute", "bolus")
STROKE = 0.006
PRESS = 0.004
# What the pump display and the order card at the dial say, from the first frame.
INSTRUCTION_TEXT = {
    "stop": "OCCLUSION ALARM: PRESS STOP",
    "mute": "SILENCE ALARM: PRESS MUTE",
    "bolus": "BOLUS: HOLD BOLUS KEY",
}


class PumpPanel(HospitalTask):
    task_id = "B6"
    name = "pump_panel_response"
    category = "event_reactive"
    discipline = "bedside_infusion"
    bedside_module = "pump_panel"
    instruction = (
        "Read the pump display and press the requested physical key before the response deadline."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.mode = MODES[spec.seed % len(MODES)]
        self.hold_required = 0.35 if self.mode == "bolus" else 0.04
        self.pressed_for = 0.0
        self.trigger_time = None
        self.key_moved_time = None  # the requested key first pressed in under contact
        self.device_state = {
            "alarm_active": False,
            "alarm_silenced": False,
            "pump_running": False,
            "bolus_active": False,
        }

    def gen_instruction(self):
        action = {
            "stop": "Press STOP to stop the simulated pump",
            "mute": "Press MUTE to silence the alarm; the occlusion remains",
            "bolus": "Hold the armed BOLUS key for 0.35 seconds",
        }[self.mode]
        return (
            f"The pump display reads {INSTRUCTION_TEXT[self.mode]}. {action} within "
            f"{self.spec.response_limit:g} seconds of the instruction (the dial starts its countdown)."
        )

    def process(self, dt):
        if self.event_time is None and self.time + EPS >= self.spec.event_time:
            self.event_time = self.time
            self.device_state["alarm_active"] = self.mode != "bolus"
            self.device_state["pump_running"] = self.mode == "stop"
            self.event_log.append({"time": self.time, "type": "pump_request", "mode": self.mode})
        if self.event_time is not None and not self.goal_achieved:
            lateness = self.time - self.event_time - self.spec.response_limit
            if lateness > EPS:
                self.violation("late_response", lateness)
        active = self.event_time is not None
        color = GREEN if self.goal_achieved else (RED if active else BLUE)
        commands = [SceneCommand("indicator", "indicator", color)]
        if active:  # the countdown ring drains from the instruction to the deadline
            commands += self.timer(
                "indicator",
                1.0 - (self.time - self.event_time) / self.spec.response_limit,
                "green" if self.goal_achieved else None,
            )
        return commands

    def evaluate(self, state, dt):
        buttons = state.controls.get("pump_panel", {})
        for key in MODES:
            measured = buttons.get(key, {})
            travel = float(measured.get("depression_m", 0.0))
            aligned = bool(measured.get("aligned", False))
            contact = bool(measured.get("contact", False))
            if not np.isfinite(travel):
                continue
            if travel > 0.0015 and contact and self.event_time is None:
                self.violation("early_action")
            if travel >= PRESS and contact and aligned and key != self.mode:
                self.physical_failure("wrong_button")
            if key != self.mode:
                continue
            if self.event_time is not None and contact and travel > 0.0015:
                if self.key_moved_time is None:
                    self.key_moved_time = self.time
            valid = (
                self.event_time is not None
                and contact
                and aligned
                and PRESS <= travel <= STROKE + 0.0006
            )
            self.pressed_for = self.pressed_for + dt if valid else 0.0
            if self.pressed_for + EPS >= self.hold_required and not self.goal_achieved:
                self.trigger_time = self.time
                self.record_action()
                self.attempted.add(key)
                self.correct.add(key)
                self.goal_achieved = True
                if self.time - self.event_time > self.spec.response_limit + EPS:
                    self.violation(
                        "late_response", self.time - self.event_time - self.spec.response_limit
                    )
                if self.mode == "stop":
                    self.device_state["pump_running"] = False
                elif self.mode == "mute":
                    self.device_state["alarm_silenced"] = True
                else:
                    self.device_state["bolus_active"] = True
                self.event_log.append(
                    {"time": self.time, "type": "physical_key_triggered", "key": key}
                )

    def metrics(self):
        return {
            **super().metrics(),
            "requested_key": self.mode,
            "key_moved_time": self.key_moved_time,
            "button_hold_required_s": self.hold_required,
            "device_state": dict(self.device_state),
            "response_limit_s": self.spec.response_limit,
            "simulation_scope": "physical pump key and symbolic device state; no infusion model",
        }
