"""B5: hang the prepared saline bag on the IV stand, then open the line's roller clamp.

Two steps in order, measured from the instruction (issued at the start of the episode):
1. the spiked, primed saline bag is lifted from its cradle on the bedside cabinet and hung by its
   eyelet on the IV stand's hook; it counts once it hangs released, supported by the hook and
   still for HANG_DWELL;
2. the roller clamp on the line below is slid fully OPEN and released; it counts once the roller is
   retained at the open stop for RETENTION_DWELL while the bag still hangs.
Moving the roller before the bag hangs is a wrong-order failure. The inclusive response limit runs
from the instruction to the completion of step 2, including the roller's released retention.

Flow is a symbolic open/closed state derived from the rigid roller's position; no patient
physiology, fluid or flexible-tubing claim is made.
"""

import math

import numpy as np

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask

# Roller mechanism (shared with other tasks that reuse the roller; keep these names).
TRAVEL = 0.036
END_TOLERANCE = 0.0015
MOTION_THRESHOLD = 0.0015
RETENTION_DWELL = 0.060
MAX_REST_SPEED = 0.012
MAX_AXIS_ERROR = 0.003
# Bag hanging.
HANG_DWELL = 0.20
BAG_MOTION_THRESHOLD = 0.008  # bag displacement from its cradle that counts as handling it
INSTRUCTION_TEXT = "HANG SALINE, THEN OPEN CLAMP"


def episode_mode(seed):
    """Historical single-slide direction by seed parity (kept for tasks reusing the roller).

    B5 itself always opens the clamp: the line starts CLOSED below the bag to be hung.
    """
    return "open" if int(seed) % 2 == 0 else "close"


class RollerClamp(HospitalTask):
    task_id = "B5"
    name = "hang_bag_open_clamp"
    category = "event_reactive"
    bedside_module = "roller_clamp"
    discipline = "bedside_infusion"
    instruction = (
        "Hang the prepared saline bag on the IV stand's hook, then slide the line's roller clamp "
        "open and release it."
    )
    roller = "roller_0"
    bag = "bag_0"

    def __init__(self, spec):
        super().__init__(spec)
        self.mode = "open"
        self.initial_position = 0.0
        self.target_position = TRAVEL
        self.retention_time = 0.0
        self.contact_seen = False
        self.motion_seen = False
        self.endpoint_time = None
        self.measured_position = self.initial_position
        self.measured_open_fraction = 0.0
        # Step 1 (hang): first released support of the retained hanging, and its dwell.
        self.hang_time = None
        self.hang_dwell = 0.0
        self._hanging_since = None
        self.bag_hanging = False
        self.bag_handled_time = None
        self.order_violation_time = None

    def gen_instruction(self):
        return (
            f"{INSTRUCTION_TEXT}. Pick up the prepared saline bag (spiked, line attached) from its "
            "cradle on the bedside cabinet and hang it by its eyelet on the IV stand's hook; then "
            "slide the line's roller clamp fully OPEN and release it. Both steps, in this order, "
            f"within {self.spec.response_limit:.2f} seconds of the instruction. The bag must hang "
            f"released and still for {HANG_DWELL:.2f} s before the clamp moves, and the roller "
            f"must stay at the open stop for {RETENTION_DWELL:.2f} s after release."
        )

    @property
    def hang_retained(self):
        return self.hang_time is not None

    def process(self, dt):
        if self.event_time is None and self.time + EPS >= self.spec.event_time:
            self.event_time = self.time
            self.event_log.append(
                {"time": self.time, "type": "instruction", "text": INSTRUCTION_TEXT}
            )
        if self.event_time is None:
            return [SceneCommand("indicator", "indicator", BLUE)]
        elapsed = self.time - self.event_time
        if self.action_time is None and elapsed > self.spec.response_limit + EPS:
            self.violation("late_response", elapsed - self.spec.response_limit)
        color = GREEN if elapsed <= self.spec.response_limit + EPS else RED
        return [
            SceneCommand("indicator", "indicator", color),
            *self.timer("indicator", 1.0 - elapsed / self.spec.response_limit),
        ]

    # --- step 1: the bag ------------------------------------------------------------------------

    def _evaluate_bag(self, state, dt):
        controls = state.controls
        obj = state.objects.get(self.bag)
        if obj is None:
            return
        initial = self.layout.objects[self.bag].position
        displaced = float(np.linalg.norm(obj.position - initial)) > BAG_MOTION_THRESHOLD
        handled = bool(obj.grasped or obj.robot_contact)
        if (displaced or handled) and self.bag_handled_time is None:
            self.bag_handled_time = self.time
            self.attempted.add(self.bag)
        if (displaced or handled) and (
            self.event_time is None or self.time < self.event_time - EPS
        ):
            self.violation("early_action", max(0.0, self.spec.event_time - self.time))
        hanging = bool(controls.get("bag_hanging", False))
        self.bag_hanging = hanging
        if hanging:
            if self._hanging_since is None:
                self._hanging_since = self.time
            self.hang_dwell += dt
            if self.hang_dwell + EPS >= HANG_DWELL and self.hang_time is None:
                self.hang_time = self._hanging_since
                self.correct.add(self.bag)
                self.event_log.append({"time": self.hang_time, "type": "bag_hung"})
                if self.event_time is None or self.hang_time < self.event_time - EPS:
                    self.violation("early_action", max(0.0, self.spec.event_time - self.time))
        else:
            self._hanging_since = None
            self.hang_dwell = 0.0
            if self.hang_time is not None and self.bag in self.correct:
                # The bag came off the hook again: the hang no longer holds.
                self.correct.discard(self.bag)
                self.event_log.append({"time": self.time, "type": "bag_unhung"})

    # --- step 2: the roller ---------------------------------------------------------------------

    def evaluate(self, state, dt):
        self._evaluate_bag(state, dt)
        controls = state.controls
        if "roller_position_m" not in controls:
            return
        position = float(controls["roller_position_m"])
        speed = float(controls.get("roller_speed_m_s", math.inf))
        axis_error = float(controls.get("roller_axis_error_m", math.inf))
        if not all(math.isfinite(value) for value in (position, speed, axis_error)):
            self.physical_failure("invalid_roller_measurement")
            return
        self.measured_position = position
        self.measured_open_fraction = min(1.0, max(0.0, position / TRAVEL))
        if (
            axis_error > MAX_AXIS_ERROR
            or not -MAX_AXIS_ERROR <= position <= TRAVEL + MAX_AXIS_ERROR
        ):
            self.physical_failure("roller_out_of_track")
        contact = bool(controls.get("roller_contact", False))
        if contact and (self.event_time is None or self.time < self.event_time - EPS):
            self.violation("early_action", max(0.0, self.spec.event_time - self.time))
        self.contact_seen = self.contact_seen or contact
        moved = abs(position - self.initial_position) > MOTION_THRESHOLD
        if moved:
            self.motion_seen = True
            self.attempted.add(self.roller)
            if self.first_motion_time is None:
                self.first_motion_time = self.time
            if self.event_time is None or self.time < self.event_time - EPS:
                self.violation("early_action", max(0.0, self.spec.event_time - self.time))
            if not self.hang_retained and self.order_violation_time is None:
                # Opening the clamp before the bag hangs runs the line from a bag on the cabinet.
                self.order_violation_time = self.time
                self.physical_failure("wrong_order")
        endpoint = abs(position - self.target_position) <= END_TOLERANCE
        valid = (
            endpoint
            and self.motion_seen
            and self.contact_seen
            and axis_error <= MAX_AXIS_ERROR
            and not controls.get("roller_grasped", False)
            and not contact
            and abs(speed) <= MAX_REST_SPEED
        )
        if valid:
            if self.endpoint_time is None:
                self.endpoint_time = self.time
            self.retention_time += dt
        else:
            self.endpoint_time = None
            self.retention_time = 0.0
        retained = self.retention_time + EPS >= RETENTION_DWELL
        if retained:
            self.correct.add(self.roller)
        else:
            self.correct.discard(self.roller)
        # Both steps hold now: the roller retained open below a bag that still hangs.
        self.goal_achieved = retained and self.hang_retained and self.bag_hanging
        if not self.goal_achieved or self.action_time is not None:
            return
        self.record_action()
        self.event_log.append({"time": self.time, "type": "roller_retained", "mode": self.mode})
        if self.event_time is None:
            self.violation("early_action", max(0.0, self.spec.event_time - self.time))
        elif self.time - self.event_time > self.spec.response_limit + EPS:
            self.violation("late_response", self.time - self.event_time - self.spec.response_limit)

    @property
    def temporal_success(self):
        return self.event_time is not None and self.action_time is not None and not self.violations

    def metrics(self):
        metrics = super().metrics()
        metrics.update(
            requested_mode=self.mode,
            instruction_text=INSTRUCTION_TEXT,
            steps=["hang_bag", "open_roller_clamp"],
            hang_time=self.hang_time,
            hang_latency=None
            if self.hang_time is None or self.event_time is None
            else self.hang_time - self.event_time,
            bag_hanging=self.bag_hanging,
            roller_travel_m=self.measured_position,
            roller_open_fraction=self.measured_open_fraction,
            released_retention_s=self.retention_time,
            contact_observed=self.contact_seen,
            wrong_order="wrong_order" in self.physical_failures,
            deadline_inclusive=True,
            deadline_event="bag_hung_then_released_open_roller_retained",
            deadline_missed="late_response" in self.violations,
            fluid_model="symbolic_measured_roller_position",
        )
        return metrics
