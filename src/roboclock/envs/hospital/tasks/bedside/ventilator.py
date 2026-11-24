"""B4 (since 2026-11-19): the ventilator's breathing circuit comes off the airway; put it back.

An intubated training manikin is ventilated through a tracheostomy tube. At a seeded time
(`spec.event_time`, 4-10 s) the circuit's catheter-mount elbow comes off the tube's 15 mm
connector and comes to rest on a towel on the chest; the chest stops moving and the ventilator
shows a red DISCONNECT / LOW PRESSURE alarm. The robot starts at rest, tucked near its own base,
and has to pick the elbow up and push its socket back over the connector.

Reconnection is measured, not triggered (bedside/ventilator.py::geometry): the connector at least
14 mm inside the socket with the axes within 0.15 rad, the elbow released, out of robot contact,
at rest and touching the tube, for 0.20 s. It is scored at the first instant of that dwell, and
the response limit runs from the disconnection to that instant. Touching or moving the circuit
before the disconnection is an early action. No gas flow, airway pressure or physiology is
simulated; the alarm state is a symbolic consequence of the measured geometry.

Until 2026-11-19 B4 was the infusion-line switch (tasks/bedside/stopcock.py, `line_switch`),
which left the suite; results recorded for it keep its name (results.py).
"""

import numpy as np

from roboclock.envs.hospital.bedside.ventilator import (
    ELBOW,
    INSTRUCTION,
    SEAT_DEPTH,
    SEAT_DWELL,
    geometry,
)
from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask


class VentilatorReconnect(HospitalTask):
    task_id = "B4"
    name = "ventilator_reconnect"
    category = "event_reactive"
    discipline = "respiratory_care"
    bedside_task = "ventilator"
    bedside_module = "ventilator"
    elbow = ELBOW
    seat_dwell = SEAT_DWELL
    hold_after = 1.0  # seconds shown after the reconnection: alarm cleared, chest moving again

    def __init__(self, spec):
        super().__init__(spec)
        self.circuit_free = False  # the scripted pop-off is over: the elbow is a free body again
        self.grasp_time = self.entry_time = self.seat_time = None
        self.reconnect_time = None  # first released, retained instant (verified by the dwell)
        self._retained_since = None
        self._prior = None
        self.device_state = {"circuit_connected": True, "alarm_active": False, "ventilating": True}

    def gen_instruction(self):
        return (
            "ICU bedside: a training manikin is ventilated through a tracheostomy tube. If the "
            "breathing circuit comes off (its blue elbow ends up on the towel on the chest, the "
            "chest stops moving, the ventilator display turns red: DISCONNECT), pick the elbow "
            "up, push its socket straight down over the white 15 mm connector at the neck until "
            "seated, and let go. The circuit must be back on, released and at rest within "
            f"{self.spec.response_limit:.1f} seconds of the disconnection. Do not touch the "
            "circuit before it comes off."
        )

    # --- simulation loop -------------------------------------------------------------------

    def process(self, dt):
        if self.event_time is None and self.time + EPS >= self.spec.event_time:
            self.event_time = self.time
            self.device_state.update(circuit_connected=False, alarm_active=True, ventilating=False)
            self.event_log.append({"time": self.time, "type": "circuit_disconnected"})
        if self.event_time is None:
            return [SceneCommand("indicator", "indicator", BLUE)]
        deadline = self.event_time + self.spec.response_limit
        if self.time > deadline + EPS and self.reconnect_time is None:
            self.violation("late_response", self.time - deadline)
        done = self.reconnect_time is not None
        color = GREEN if done and not self.violations else RED
        commands = [SceneCommand("indicator", "indicator", color)]
        if not done:
            commands += self.timer("indicator", (deadline - self.time) / self.spec.response_limit)
        return commands

    def evaluate(self, state, dt):
        obj = state.objects[ELBOW]
        measured = state.controls.get("ventilator") or geometry(self.layout, obj)
        if self.event_time is None:
            # The robot starts at rest: any contact with or motion of the circuit is early.
            if obj.robot_contact or (
                np.linalg.norm(obj.position - self.layout.objects[ELBOW].position) > 0.008
            ):
                self.violation("early_action", max(0.0, self.spec.event_time - self.time))
            self._prior = measured
            return
        if not self.circuit_free:
            if measured.get("circuit_free", True):
                self.circuit_free = True
            else:
                if obj.robot_contact:  # caught in mid-air: the event has not finished
                    self.attempted.add("elbow")
                self._prior = measured
                return
        if obj.grasped and self.grasp_time is None:
            self.grasp_time = self.time
            self.attempted.add("elbow")
            self.event_log.append({"time": self.time, "type": "elbow_grasped"})
        prior = self._prior
        if (
            self.entry_time is None
            and prior is not None
            and prior["depth"] <= 0.0 < measured["depth"]
            and measured["lateral_error"] < 0.020
        ):
            self.attempted.add("connector")
            if measured["aligned"]:
                self.entry_time = self.time
                self.event_log.append({"time": self.time, "type": "connector_entered_socket"})
        if self.seat_time is None and measured["seated"] and self.entry_time is not None:
            self.seat_time = self.time
            self.event_log.append(
                {"time": self.time, "type": "connector_seated", "depth": measured["depth"]}
            )
        if measured["retained"] and self.seat_time is not None:
            if self._retained_since is None:
                self._retained_since = self.time
            self.dwell["seat"] = self.dwell.get("seat", 0.0) + dt
            if self.dwell["seat"] + EPS >= self.seat_dwell and self.reconnect_time is None:
                self.reconnect_time = self._retained_since
                self.record_action(self.reconnect_time)
                self.correct.update(("elbow", "connector"))
                self.device_state.update(
                    circuit_connected=True, alarm_active=False, ventilating=True
                )
                self.event_log.append({"time": self.reconnect_time, "type": "circuit_reconnected"})
                limit = self.event_time + self.spec.response_limit
                if self.reconnect_time > limit + EPS:
                    self.violation("late_response", self.reconnect_time - limit)
        else:
            self._retained_since = None
            self.dwell["seat"] = 0.0
        # Reconnected for as long as the released elbow stays on the connector.
        connected = self.reconnect_time is not None and measured["depth"] >= SEAT_DEPTH - 0.003
        if self.reconnect_time is not None and not connected:
            self.physical_failure("circuit_came_off_again")
        self.goal_achieved = connected
        self._prior = measured

    # --- outcomes ----------------------------------------------------------------------------

    @property
    def cue_observed(self):
        return self.event_time is not None

    @property
    def temporal_success(self):
        return self.reconnect_time is not None and not self.violations

    @property
    def terminated(self):
        if bool(self.dropped):
            return True
        if self.completion_time is not None:
            return self.time + EPS >= self.completion_time + self.hold_after
        return False

    def metrics(self):
        result = super().metrics()

        def after_event(value):
            return None if value is None or self.event_time is None else value - self.event_time

        result.update(
            {
                "instruction": INSTRUCTION,
                "disconnect_time": self.event_time,
                "grasp_time": self.grasp_time,
                "socket_entry_time": self.entry_time,
                "seat_time": self.seat_time,
                "reconnect_time": self.reconnect_time,
                "response_limit": self.spec.response_limit,
                "response_latency": after_event(self.reconnect_time),
                "grasp_latency": after_event(self.grasp_time),
                "seat_depth_required": SEAT_DEPTH,
                "seat_dwell_required_s": self.seat_dwell,
                "device_state": dict(self.device_state),
                "physical_model": "free_rigid_elbow_socket_over_fixed_connector_scripted_pop_off",
            }
        )
        return result
