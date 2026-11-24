"""B1: the patient desaturates; raise the oxygen flow into the ordered band after the SpO2 alarm.

A patient on low-flow oxygen (2 L/min through a mask) lies in the bed; the bedside monitor shows
SpO2 and the wall oxygen flowmeter (a Thorpe tube: ball float in a tapered glass tube over a
needle valve) stands plugged into the outlet panel at the bed head. The standing order is on the
order card from the first frame: if SpO2 falls below 90 %, raise the oxygen to the ordered flow.

Event: SpO2 is already falling when the episode starts, at the real rate (about 0.12 %/s), and
crosses 90 % at the seeded event time; the monitor then alarms (red numeric, LOW SpO2 banner, alarm lamp). The robot starts
at rest, tucked at its base, and must turn the flowmeter's control knob counterclockwise until
the float stands inside the green band on the tube (ordered flow +- 1 L/min) and stays there for
DWELL seconds. The response limit runs from the alarm to the end of that dwell.

What is measured and what is modelled:
- The knob is a passive revolute joint turned only by finger contact; the apparatus module
  reports its measured angle (`knob_angle_rad`, counterclockwise from the start) and contact.
- Flow is a needle-valve characteristic of the measured angle (`flow_from_angle`); the float
  follows it with a short first-order lag. SpO2 is a process model (`spo2_step`): it keeps
  falling, more slowly, until the float is in or above the band, then recovers after a delay.
  Before the alarm SpO2 falls at the real rate (FALL_RATE); the rates after the alarm and of the
  recovery are design settings, and a monitor's averaging time and the minutes a real recovery
  takes are not modelled.
- Turning the knob before the alarm is an early action (a temporal violation). Opening the valve
  to OVERSHOOT L/min above the band, or closing it below the baseline, are physical failures.
"""

import math

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask

KNOB = "oxygen_knob"
BASELINE_FLOW = 2.0  # L/min at the start (knob angle 0)
FLOW_PER_RAD = 1.0 / math.radians(18.0)  # needle valve: 1 L/min per 18 degrees of knob turn
MAX_FLOW = 15.0  # top of the tube's scale
ORDERED_FLOWS = (5.0, 6.0, 7.0)  # L/min, by seed
BAND_HALF_WIDTH = 1.0  # L/min: the green band is the ordered flow +- 1
OVERSHOOT = 3.0  # L/min above the band's top: flooding the patient is a failure
REDUCED = 1.0  # L/min below the baseline: turned the wrong way
DWELL = 0.5  # s the float must stay inside the band
FLOAT_LAG = 0.12  # s, first-order lag of the ball float behind the valve flow
MOTION_THRESHOLD = math.radians(3.0)  # knob turn that counts as moved
ALARM_THRESHOLD = 90.0  # % SpO2
SPO2_RANGE = (95.0, 97.0)  # seeded baseline
# %/s before the alarm, at real scale since 2026-11-23 (user rule: no process is sped up): an apnoeic patient on room
# air desaturates to 90 % in 45-60 s (Weingart and Levitan 2012), about 6 % in 50 s. The episode starts mid-fall, 0.5-1.2 %
# above the alarm line (event_time 4-10 s), instead of showing the whole fall from the baseline. Was 2.0 %/s.
FALL_RATE = 0.12
FALL_RATE_ALARM = 0.6  # %/s after the alarm, until oxygen is raised
SPO2_FLOOR = 80.0
RECOVERY_DELAY = 0.8  # s from the float entering the band to the first rise of SpO2
RECOVERY_RATE = 2.5  # %/s
RECOVERY_TARGET = 95.0
HOLD_AFTER = 2.0  # s shown after the flow is settled, so the recovery reads on the monitor


def flow_from_angle(angle: float) -> float:
    """Valve flow (L/min) for a knob angle (rad, counterclockwise from the start)."""
    if not math.isfinite(angle):
        return BASELINE_FLOW
    return min(max(BASELINE_FLOW + angle * FLOW_PER_RAD, 0.0), MAX_FLOW)


def angle_for_flow(flow: float) -> float:
    return (flow - BASELINE_FLOW) / FLOW_PER_RAD


def float_step(shown: float, flow: float, dt: float) -> float:
    """The float's reading one step on: first-order lag behind the valve flow."""
    return flow + (shown - flow) * math.exp(-dt / FLOAT_LAG)


def spo2_before_alarm(time: float, event_time: float, baseline: float) -> float:
    """SpO2 up to the alarm: falling to cross 90 % at `event_time`, never above the baseline."""
    return min(baseline, ALARM_THRESHOLD + FALL_RATE * (event_time - time))


def spo2_step(spo2: float, dt: float, oxygen_since: float | None) -> float:
    """SpO2 one step on after the alarm; `oxygen_since` = seconds the float has been in or above
    the band (None while it is below)."""
    if oxygen_since is not None and oxygen_since + EPS >= RECOVERY_DELAY:
        return min(RECOVERY_TARGET, spo2 + RECOVERY_RATE * dt)
    return max(SPO2_FLOOR, spo2 - FALL_RATE_ALARM * dt)


class OxygenDesaturation(HospitalTask):
    task_id = "B1"
    name = "oxygen_desaturation"
    category = "event_reactive"
    discipline = "bedside_respiratory"
    bedside_module = "oxygen_flow"
    instruction = (
        "The patient is on oxygen; the bedside monitor shows SpO2. When SpO2 falls below 90 % (red "
        "number, LOW SpO2 alarm), raise the oxygen: pinch the flowmeter's control knob and turn it "
        "counterclockwise until the ball float is inside the green band on the tube"
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.ordered_flow = ORDERED_FLOWS[spec.seed % len(ORDERED_FLOWS)]
        self.band = (self.ordered_flow - BAND_HALF_WIDTH, self.ordered_flow + BAND_HALF_WIDTH)
        lo, hi = SPO2_RANGE
        self.baseline_spo2 = lo + (hi - lo) * ((spec.seed * 7) % 5) / 4.0
        self.spo2 = self.baseline_spo2
        self.min_spo2 = self.baseline_spo2
        self.knob_angle = 0.0
        self.flow = BASELINE_FLOW  # valve flow from the measured angle
        self.float_flow = BASELINE_FLOW  # what the ball float shows
        self.knob_moved_time = None  # first contact-driven turn after the alarm
        self.band_entry_time = None  # float first inside the band after the alarm
        self.in_band_for = 0.0
        self.oxygen_since = None  # time the float has been in or above the band
        self.settled_time = None

    # --- instruction -------------------------------------------------------------------------

    def gen_instruction(self):
        lo, hi = self.band
        return (
            f"{self.instruction} ({lo:g}-{hi:g} L/min; the order is {self.ordered_flow:g}) and keep "
            f"it there for {DWELL:g} s, within {self.spec.response_limit:g} s of the alarm. Do not "
            f"turn the knob before the alarm, do not open it more than {OVERSHOOT:g} L/min past the "
            "band and do not turn it down."
        )

    @property
    def alarm_active(self) -> bool:
        return self.event_time is not None and self.spo2 < ALARM_THRESHOLD

    def displayed_spo2(self) -> int:
        """The monitor's whole-number reading (rounded down, so the alarm starts at 89)."""
        return int(math.floor(self.spo2 + 1e-6))

    # --- simulation loop ---------------------------------------------------------------------

    def process(self, dt):
        if self.event_time is None:
            self.spo2 = spo2_before_alarm(self.time, self.spec.event_time, self.baseline_spo2)
            if self.time + EPS >= self.spec.event_time:
                self.event_time = self.time
                self.event_log.append({"time": self.time, "type": "spo2_alarm"})
        else:
            since = None if self.oxygen_since is None else self.time - self.oxygen_since
            self.spo2 = spo2_step(self.spo2, dt, since)
        self.min_spo2 = min(self.min_spo2, self.spo2)
        if self.event_time is not None and not self.goal_achieved:
            lateness = self.time - self.event_time - self.spec.response_limit
            if lateness > EPS:
                self.violation("late_response", lateness)
        active = self.event_time is not None
        color = GREEN if self.goal_achieved else (RED if active else BLUE)
        commands = [SceneCommand("indicator", "indicator", color)]
        if active:  # the countdown ring drains from the alarm to the deadline
            done = self.action_time if self.action_time is not None else self.time
            commands += self.timer(
                "indicator",
                1.0 - (done - self.event_time) / self.spec.response_limit,
                "green" if self.goal_achieved and not self.violations else None,
            )
        return commands

    def evaluate(self, state, dt):
        controls = state.controls
        angle = float(controls.get("knob_angle_rad", 0.0))
        contact = bool(controls.get("knob_contact", False))
        if math.isfinite(angle):
            self.knob_angle = angle
        self.flow = flow_from_angle(self.knob_angle)
        self.float_flow = float_step(self.float_flow, self.flow, dt)
        turned = abs(self.knob_angle) > MOTION_THRESHOLD
        if self.event_time is None:
            if turned:
                self.violation("early_action", self.spec.event_time - self.time)
                self.attempted.add(KNOB)
            return
        if turned and contact and self.knob_moved_time is None:
            self.knob_moved_time = self.time
            self.attempted.add(KNOB)
            self.event_log.append({"time": self.time, "type": "knob_turned"})
        lo, hi = self.band
        if self.float_flow > hi + OVERSHOOT:
            self.physical_failure("flow_overshoot")
        if self.float_flow < BASELINE_FLOW - REDUCED:
            self.physical_failure("flow_reduced")
        if self.float_flow + EPS >= lo:
            if self.oxygen_since is None:
                self.oxygen_since = self.time
        else:
            self.oxygen_since = None
        inside = lo - EPS <= self.float_flow <= hi + EPS and self.knob_moved_time is not None
        if inside and self.band_entry_time is None:
            self.band_entry_time = self.time
            self.event_log.append(
                {"time": self.time, "type": "float_in_band", "flow": self.float_flow}
            )
        self.in_band_for = self.in_band_for + dt if inside else 0.0
        if self.in_band_for + EPS >= DWELL and not self.goal_achieved:
            self.goal_achieved = True
            self.settled_time = self.time
            self.record_action()
            self.correct.add(KNOB)
            self.event_log.append(
                {"time": self.time, "type": "flow_settled", "flow": self.float_flow}
            )
            if self.time - self.event_time > self.spec.response_limit + EPS:
                self.violation(
                    "late_response", self.time - self.event_time - self.spec.response_limit
                )

    # --- outcomes ----------------------------------------------------------------------------

    @property
    def terminated(self) -> bool:
        if bool(self.dropped) or bool(self.physical_failures):
            return True
        if self.settled_time is not None:
            return self.time + EPS >= self.settled_time + HOLD_AFTER
        return False

    def metrics(self):
        return {
            **super().metrics(),
            "ordered_flow_l_min": self.ordered_flow,
            "band_l_min": list(self.band),
            "knob_angle_rad": self.knob_angle,
            "valve_flow_l_min": self.flow,
            "float_flow_l_min": self.float_flow,
            "knob_moved_time": self.knob_moved_time,
            "band_entry_time": self.band_entry_time,
            "settled_time": self.settled_time,
            "baseline_spo2": self.baseline_spo2,
            "final_spo2": self.spo2,
            "min_spo2": self.min_spo2,
            "response_limit_s": self.spec.response_limit,
            "dwell_s": DWELL,
            "simulation_scope": (
                "contact-turned knob; valve flow, float and SpO2 are process models of the "
                "measured knob angle"
            ),
        }
