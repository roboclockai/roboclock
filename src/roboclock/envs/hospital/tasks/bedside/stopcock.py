"""B4 line switch: close the running MAINTENANCE line's roller clamp, then turn the stopcock.

"Switch the infusion to the requested line", in the clinical order: first slide the MAINTENANCE
line's roller clamp to CLOSED (released and retained at its end stop), then turn the three-way
stopcock 90 degrees so REQUESTED connects to PATIENT (retained in its detent). Both devices are
measured rigid mechanisms on one bed-rail line holder; flow is symbolic, never simulated.

Step 2 counts only once step 1 is complete and retained, and only while the clamp is still closed.
Turning the stopcock more than 3 degrees before the clamp is complete is a wrong-order failure.
The response limit runs from the instruction to the completion of step 2, i.e. the end of its
80 ms detent retention (inclusive). Device motion before the instruction is an early action.
"""

import math

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask

# Stopcock (step 2): passive revolute handle, 0 -> 90 degrees counterclockwise.
TARGET_ANGLE = math.pi / 2
ANGLE_TOLERANCE = math.radians(3.0)
MOTION_THRESHOLD = math.radians(3.0)
RETENTION_SECONDS = 0.08
MAX_RETAIN_SPEED = 0.4

# Roller clamp (step 1): the B5 prismatic roller and its predicates (copied, not imported, so the
# B5 module can change independently). Travel runs from OPEN (0) to CLOSED (CLAMP_TRAVEL).
CLAMP_TRAVEL = 0.036
CLAMP_END_TOLERANCE = 0.0015
CLAMP_MOTION_THRESHOLD = 0.0015
CLAMP_RETENTION = 0.060
CLAMP_MAX_REST_SPEED = 0.012
CLAMP_MAX_AXIS_ERROR = 0.003

HANDLE = "stopcock_handle"
CLAMP = "maintenance_roller"
ORDER_TEXT = "SWITCH INFUSION: 1 CLOSE MAINTENANCE CLAMP  2 TURN STOPCOCK TO REQUESTED"


class StopcockSwitch(HospitalTask):
    task_id = "B4"
    name = "line_switch"
    category = "event_reactive"
    bedside_module = "stopcock"
    discipline = "bedside_infusion"
    instruction = (
        f"{ORDER_TEXT}. First slide the MAINTENANCE line's roller clamp fully to CLOSED and "
        "release it. Then pinch the blue stopcock handle and turn it 90 degrees counterclockwise "
        "to the green detent so REQUESTED connects to PATIENT. Never turn the stopcock while the "
        "maintenance clamp is still open."
    )

    def __init__(self, spec):
        super().__init__(spec)
        # Step 1: roller clamp.
        self.clamp_position = 0.0
        self.clamp_contact_seen = False
        self.clamp_motion_seen = False
        self.clamp_dwell = 0.0
        self.clamp_closed_now = False
        self.clamp_time = None  # retained, released closure (latched)
        # Step 2: stopcock.
        self.first_target_entry = None
        self.target_entry = None
        self.target_dwell = 0.0
        self.grasp_observed = False
        self.contact_motion_observed = False
        self.angle = 0.0
        self.angular_speed = 0.0
        self.wrong_order = False

    def gen_instruction(self):
        return (
            f"{self.instruction} Complete both steps within {self.spec.response_limit:g} seconds "
            "of the order."
        )

    def process(self, dt):
        if self.event_time is None and self.time + EPS >= self.spec.event_time:
            self.event_time = self.time
            self.event_log.append({"time": self.time, "type": "switch_order"})
        active = self.event_time is not None
        color = GREEN if self.goal_achieved else RED if active else BLUE
        commands = [SceneCommand("indicator", "indicator", color)]
        if active:
            elapsed = self.time - self.event_time
            commands += self.timer(
                "indicator",
                1.0 - elapsed / self.spec.response_limit,
                "green" if self.goal_achieved else None,
            )
        return commands

    # --- step 1 ------------------------------------------------------------------------------

    def _clamp(self, controls, dt):
        if "clamp_position_m" not in controls:
            self.clamp_closed_now = False
            return
        position = float(controls["clamp_position_m"])
        speed = float(controls.get("clamp_speed_m_s", math.inf))
        axis_error = float(controls.get("clamp_axis_error_m", math.inf))
        if not all(math.isfinite(value) for value in (position, speed, axis_error)):
            self.physical_failure("invalid_clamp_measurement")
            self.clamp_closed_now = False
            return
        self.clamp_position = position
        if axis_error > CLAMP_MAX_AXIS_ERROR or not (
            -CLAMP_MAX_AXIS_ERROR <= position <= CLAMP_TRAVEL + CLAMP_MAX_AXIS_ERROR
        ):
            self.physical_failure("clamp_out_of_track")
        contact = bool(controls.get("clamp_contact", False))
        self.clamp_contact_seen |= contact
        if position > CLAMP_MOTION_THRESHOLD:
            self.clamp_motion_seen = True
            self.attempted.add(CLAMP)
            if self.first_motion_time is None:
                self.first_motion_time = self.time
            if self.event_time is None:
                self.violation("early_action", max(0.0, self.spec.event_time - self.time))
        self.clamp_closed_now = (
            abs(position - CLAMP_TRAVEL) <= CLAMP_END_TOLERANCE and axis_error <= CLAMP_MAX_AXIS_ERROR
        )
        if self.clamp_time is not None:
            return
        released = (
            self.clamp_closed_now
            and self.clamp_motion_seen
            and self.clamp_contact_seen
            and not controls.get("clamp_grasped", False)
            and not contact
            and abs(speed) <= CLAMP_MAX_REST_SPEED
        )
        self.clamp_dwell = self.clamp_dwell + dt if released else 0.0
        if self.clamp_dwell + EPS >= CLAMP_RETENTION:
            self.clamp_time = self.time
            self.correct.add(CLAMP)
            self.event_log.append({"time": self.time, "type": "clamp_closed"})

    # --- step 2 ------------------------------------------------------------------------------

    def evaluate(self, state, dt):
        controls = state.controls
        self._clamp(controls, dt)
        # Missing measurements must never look like an aligned or stationary device.
        previous_angle = self.angle
        self.angle = float(controls.get("stopcock_angle_rad", float("nan")))
        self.angular_speed = float(controls.get("stopcock_speed_rad_s", float("inf")))
        contact = bool(controls.get("stopcock_robot_contact", False))
        grasped = bool(controls.get("stopcock_grasped", False))
        self.grasp_observed |= grasped
        moved = math.isfinite(self.angle) and abs(self.angle) > MOTION_THRESHOLD
        if moved:
            self.attempted.add(HANDLE)
            if self.event_time is None:
                self.violation("early_action", max(0.0, self.spec.event_time - self.time))
            if self.clamp_time is None and not self.wrong_order:
                # Clinically the running line is clamped before the stopcock switches lines.
                self.wrong_order = True
                self.physical_failure("wrong_order")
            rotating = abs(self.angle - previous_angle) > 1e-4 or abs(self.angular_speed) > 0.02
            self.contact_motion_observed |= contact and self.grasp_observed and rotating
        aligned = (
            self.clamp_time is not None
            and self.clamp_closed_now
            and math.isfinite(self.angle)
            and abs(self.angle - TARGET_ANGLE) <= ANGLE_TOLERANCE
            and abs(self.angular_speed) <= MAX_RETAIN_SPEED
            and float(controls.get("stopcock_axis_error_rad", float("inf"))) <= 0.05
            and float(controls.get("stopcock_pivot_error_m", float("inf"))) <= 0.003
            and self.grasp_observed
            and self.contact_motion_observed
        )
        if self.goal_achieved:
            pass
        elif aligned:
            if self.target_entry is None:
                self.target_entry = self.time
                if self.first_target_entry is None:
                    self.first_target_entry = self.time
                    self.event_log.append({"time": self.time, "type": "first_detent_entry"})
                self.target_dwell = 0.0
            else:
                self.target_dwell += dt
            if self.target_dwell + EPS >= RETENTION_SECONDS:
                # Step 2 completes when its detent retention completes.
                self.record_action()
                self.correct.add(HANDLE)
                self.goal_achieved = True
                self.event_log.append({"time": self.time, "type": "stopcock_switched"})
        else:
            self.target_entry = None
            self.target_dwell = 0.0
        # Inclusive deadline on the completion of step 2 (after its retention).
        if self.event_time is not None:
            deadline = self.event_time + self.spec.response_limit
            if self.action_time is not None and self.action_time > deadline + EPS:
                self.violation("late_action", self.action_time - deadline)
            elif self.action_time is None and self.time > deadline + EPS:
                self.violation("late_action", self.time - deadline)

    def _since_order(self, timestamp):
        if timestamp is None or self.event_time is None:
            return None
        return timestamp - self.event_time

    def metrics(self):
        result = super().metrics()
        result.update(
            order_text=ORDER_TEXT,
            step_order=[CLAMP, HANDLE],
            clamp_closed=self.clamp_time is not None,
            clamp_closed_time=self.clamp_time,
            clamp_closed_after_order_s=self._since_order(self.clamp_time),
            clamp_travel_m=self.clamp_position,
            clamp_travel_target_m=CLAMP_TRAVEL,
            clamp_contact_observed=self.clamp_contact_seen,
            stopcock_angle_deg=math.degrees(self.angle),
            target_angle_deg=90.0,
            angle_tolerance_deg=math.degrees(ANGLE_TOLERANCE),
            first_target_entry=self.first_target_entry,
            switched_time=self.action_time,
            switched_after_order_s=self._since_order(self.action_time),
            target_retention_s=self.target_dwell,
            grasp_observed=self.grasp_observed,
            contact_motion_observed=self.contact_motion_observed,
            wrong_order=self.wrong_order,
            maintenance_line_clamped=self.clamp_time is not None and self.clamp_closed_now,
            maintenance_channel_closed=self.goal_achieved,
            requested_channel_connected=self.goal_achieved,
            deadline_event="stopcock_detent_retained_after_clamp",
            deadline_inclusive=True,
            fluid_simulated=False,
        )
        return result
