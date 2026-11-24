"""B5 (since 2026-11-20): anaphylaxis at the bedside; give the adrenaline auto-injector.

A patient receiving an intravenous infusion lies in the bed; the bedside monitor shows blood
pressure and heart rate. An adrenaline auto-injector trainer stands armed (safety release off) in
the holder of the anaphylaxis kit tray on the over-bed table, and a training injection pad is
strapped over the outer middle of the patient's near thigh. The standing order is in the
instruction from the first frame.

Event: the blood pressure starts to fall a few seconds before the seeded event time
(`spec.event_time`, 4-10 s) and the systolic value crosses 90 mmHg at the event time; the monitor
then alarms (red LOW BP banner, red numbers, flashing lamp) and hives appear on the patient's
near forearm. The robot starts at rest, tucked near its own base.

Response: pick the auto-injector up, press its orange tip against the injection pad until the
spring-loaded tip is pushed in, and hold it there for HOLD seconds. Everything is measured, not
triggered (bedside/autoinjector.py::geometry): the tip is a second rigid body on a sprung
prismatic joint in the barrel; the injector "fires" when the measured tip travel reaches
ACTIVATE_TRAVEL with the tip end on the pad, the barrel within AXIS_TOLERANCE of the pad's normal
and the robot holding it, and the dose counts as delivered when the travel has then stayed at or
above HOLD_TRAVEL, on the pad, for HOLD seconds. The response limit runs from the alarm to the
end of that hold.

Scoring. Touching or moving the injector before the alarm is an early action (temporal). Physical
failures: the tip pushed in anywhere but on the pad (`wrong_site`: a real device fires once),
the injector lifted before the hold is complete (`released_early`: part of the dose is lost),
the barrel's blunt end pressed on the pad with the tip up (`wrong_end`), the injector dropped.

What is modelled: blood pressure and heart rate are process models (`systolic_step`), strongly
time-compressed; no drug, needle, dose or physiology is simulated. The hold time is the real one.

Until 2026-11-20 B5 was "hang the bag, open the clamp" (tasks/bedside/roller_clamp.py,
`hang_bag_open_clamp`), which left the suite; results recorded for it keep its name (results.py).
"""

import math

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask

PEN = "epi_pen"
TIP = "epi_tip"
HOLD = 3.0  # s the fired injector is held in place (the device's instructions for use)
TIP_TRAVEL = 0.010  # m the tip can be pushed into the barrel
ACTIVATE_TRAVEL = 0.008  # m of tip travel at which the injector fires
HOLD_TRAVEL = 0.006  # m of tip travel that must be kept during the hold
AXIS_TOLERANCE = 0.35  # rad between the barrel and the pad's normal when it fires
PRESS_DWELL = 0.04  # s the travel must persist before it counts (not an impact transient)
EARLY_MOVE = 0.008  # m the injector may shift before the alarm
ALARM_SYSTOLIC = 90.0  # mmHg
SYSTOLIC_RANGE = (112.0, 124.0)  # seeded baseline
DIASTOLIC_RATIO = 0.62
# mmHg/s before the alarm, at real scale since 2026-11-23 (user rule: no process is sped up): fatal reactions to injected
# drugs reached arrest after a median 5 min (Pumphrey 2000), so the 22-34 mmHg from the baseline to the alarm take minutes.
# The episode starts mid-fall, 0.4-1 mmHg above the alarm line (event_time 4-10 s). Was 8 mmHg/s.
FALL_RATE = 0.1
FALL_RATE_ALARM = 1.0  # mmHg/s after the alarm, until adrenaline is given
SYSTOLIC_FLOOR = 60.0
RECOVERY_DELAY = 1.0  # s from the completed injection to the first rise
RECOVERY_RATE = 12.0  # mmHg/s (time-compressed)
RECOVERY_TARGET = 110.0
HOLD_AFTER = 3.0  # s shown after delivery, so the recovery reads on the monitor


def systolic_before_alarm(time: float, event_time: float, baseline: float) -> float:
    """Systolic pressure up to the alarm: flat, then falling to cross 90 mmHg at `event_time`."""
    return min(baseline, ALARM_SYSTOLIC + FALL_RATE * (event_time - time))


def systolic_step(systolic: float, dt: float, since_delivery: float | None) -> float:
    """Systolic pressure one step on after the alarm."""
    if since_delivery is not None and since_delivery + EPS >= RECOVERY_DELAY:
        return min(RECOVERY_TARGET, systolic + RECOVERY_RATE * dt)
    return max(SYSTOLIC_FLOOR, systolic - FALL_RATE_ALARM * dt)


HEART_RATE_REST = 84  # /min while the systolic pressure is at or above HEART_RATE_KNEE
HEART_RATE_KNEE = 110.0  # mmHg: below every seeded baseline, so the rate is steady before the fall


def heart_rate(systolic: float, knee: float = HEART_RATE_KNEE) -> int:
    """Compensatory tachycardia: one beat per mmHg lost below the knee, from 84 /min."""
    return int(round(HEART_RATE_REST + max(0.0, knee - systolic)))


class AnaphylaxisAutoInjector(HospitalTask):
    task_id = "B5"
    name = "anaphylaxis_autoinjector"
    category = "event_reactive"
    discipline = "emergency_medication"
    bedside_task = "autoinjector"
    bedside_module = "autoinjector"
    pen, tip = PEN, TIP
    hold_required = HOLD

    def __init__(self, spec):
        super().__init__(spec)
        lo, hi = SYSTOLIC_RANGE
        self.baseline_systolic = lo + (hi - lo) * ((spec.seed * 7) % 5) / 4.0
        self.systolic = self.baseline_systolic
        self.min_systolic = self.baseline_systolic
        self.grasp_time = None  # injector first held after the alarm
        self.site_time = None  # tip first on the pad, injector held
        self.fire_time = None  # tip pushed in on the pad: the hold starts here
        self.delivered_time = None  # end of the completed hold
        self.held_for = 0.0
        self.max_travel = 0.0
        self._pressed_for = 0.0
        self._wrong_for = 0.0
        self._wrong_end_for = 0.0
        self.device_state = {"alarm_active": False, "injector_fired": False, "dose_delivered": False}

    # --- instruction -------------------------------------------------------------------------

    def gen_instruction(self):
        return (
            "The patient on an IV infusion may develop anaphylaxis: the monitor then alarms (red "
            "LOW BP) and hives appear on the arm. Then take the adrenaline auto-injector from its "
            "holder, press its orange tip straight down on the blue injection pad on the patient's "
            f"outer thigh until the tip is pushed in, and hold it there for {HOLD:g} s; the hold "
            f"must end within {self.spec.response_limit:.1f} s of the alarm. Do not touch the "
            "injector before the alarm, press its tip nowhere else, and do not lift it early."
        )

    @property
    def alarm_active(self) -> bool:
        return self.event_time is not None and self.systolic < ALARM_SYSTOLIC

    def displayed(self):
        """What the monitor shows: (systolic, diastolic, heart rate) as whole numbers."""
        systolic = int(math.floor(self.systolic + 1e-6))
        return systolic, int(round(systolic * DIASTOLIC_RATIO)), heart_rate(systolic)

    # --- simulation loop ---------------------------------------------------------------------

    def process(self, dt):
        if self.event_time is None:
            self.systolic = systolic_before_alarm(self.time, self.spec.event_time, self.baseline_systolic)
            if self.time + EPS >= self.spec.event_time:
                self.event_time = self.time
                self.device_state["alarm_active"] = True
                self.event_log.append({"time": self.time, "type": "anaphylaxis_alarm"})
        else:
            since = None if self.delivered_time is None else self.time - self.delivered_time
            self.systolic = systolic_step(self.systolic, dt, since)
            self.device_state["alarm_active"] = self.alarm_active
        self.min_systolic = min(self.min_systolic, self.systolic)
        if self.event_time is None:
            return [SceneCommand("indicator", "indicator", BLUE)]
        deadline = self.event_time + self.spec.response_limit
        if self.time > deadline + EPS and self.delivered_time is None:
            self.violation("late_response", self.time - deadline)
        done = self.delivered_time is not None
        color = GREEN if done and not self.violations else RED
        commands = [SceneCommand("indicator", "indicator", color)]
        if not done:
            commands += self.timer("indicator", (deadline - self.time) / self.spec.response_limit)
        return commands

    def _measured(self, state):
        measured = state.controls.get("autoinjector")
        if measured is None:
            from roboclock.envs.hospital.bedside.autoinjector import geometry

            measured = geometry(self.layout, state.objects[PEN], state.objects[TIP])
        return measured

    def evaluate(self, state, dt):
        pen = state.objects[PEN]
        measured = self._measured(state)
        travel = float(measured["travel"])
        held = bool(pen.grasped or pen.robot_contact or state.objects[TIP].robot_contact)
        if self.event_time is None:
            # The robot starts at rest: any contact with or motion of the injector is early.
            start = self.layout.objects[PEN].position
            if held or math.dist(pen.position, start) > EARLY_MOVE:
                self.violation("early_action", max(0.0, self.spec.event_time - self.time))
                self.attempted.add(PEN)
            return
        if pen.grasped and self.grasp_time is None:
            self.grasp_time = self.time
            self.attempted.add(PEN)
            self.event_log.append({"time": self.time, "type": "injector_grasped"})
        on_site = bool(measured["on_site"])
        if on_site and held and self.site_time is None:
            self.site_time = self.time
            self.attempted.add("site")
            self.event_log.append({"time": self.time, "type": "tip_on_pad"})
        self.max_travel = max(self.max_travel, travel)
        # The blunt end pressed on the pad with the tip up: the needle end points at the hand.
        wrong_end = bool(measured.get("blunt_on_site")) and held
        self._wrong_end_for = self._wrong_end_for + dt if wrong_end else 0.0
        if self._wrong_end_for + EPS >= 0.2:
            self.physical_failure("wrong_end")
        pressed = travel + 1e-9 >= ACTIVATE_TRAVEL
        if self.fire_time is None:
            good = pressed and on_site and bool(measured["aligned"]) and held
            self._pressed_for = self._pressed_for + dt if good else 0.0
            # Pushed in anywhere else: the device fires once, so the dose is lost.
            wrong = pressed and not on_site
            self._wrong_for = self._wrong_for + dt if wrong else 0.0
            if self._wrong_for + EPS >= PRESS_DWELL:
                self.physical_failure("wrong_site")
            if self._pressed_for + EPS >= PRESS_DWELL:
                self.fire_time = self.time - self._pressed_for + dt
                self.device_state["injector_fired"] = True
                self.attempted.add("hold")
                self.event_log.append({"time": self.fire_time, "type": "injector_fired", "travel": travel})
        if self.fire_time is not None and self.delivered_time is None and not self.physical_failures:
            if travel + 1e-9 >= HOLD_TRAVEL and on_site:
                self.held_for = self.time - self.fire_time
                if self.held_for + EPS >= HOLD:
                    self.delivered_time = self.fire_time + HOLD
                    self.record_action(self.delivered_time)
                    self.correct.update((PEN, "site", "hold"))
                    self.goal_achieved = True
                    self.device_state["dose_delivered"] = True
                    self.event_log.append({"time": self.delivered_time, "type": "dose_delivered"})
                    limit = self.event_time + self.spec.response_limit
                    if self.delivered_time > limit + EPS:
                        self.violation("late_response", self.delivered_time - limit)
            else:
                self.physical_failure("released_early")

    # --- outcomes ----------------------------------------------------------------------------

    @property
    def cue_observed(self):
        return self.event_time is not None

    @property
    def temporal_success(self):
        return self.delivered_time is not None and not self.violations

    @property
    def terminated(self):
        if bool(self.dropped) or bool(self.physical_failures):
            return True
        if self.delivered_time is not None:
            return self.time + EPS >= self.delivered_time + HOLD_AFTER
        return False

    def metrics(self):
        result = super().metrics()

        def after_event(value):
            return None if value is None or self.event_time is None else value - self.event_time

        result.update(
            {
                "alarm_time": self.event_time,
                "grasp_time": self.grasp_time,
                "site_time": self.site_time,
                "fire_time": self.fire_time,
                "delivered_time": self.delivered_time,
                "response_limit": self.spec.response_limit,
                "response_latency": after_event(self.delivered_time),
                "grasp_latency": after_event(self.grasp_time),
                "fire_latency": after_event(self.fire_time),
                "hold_required_s": HOLD,
                "held_for_s": self.held_for,
                "max_tip_travel_m": self.max_travel,
                "activate_travel_m": ACTIVATE_TRAVEL,
                "baseline_systolic": self.baseline_systolic,
                "min_systolic": self.min_systolic,
                "final_systolic": self.systolic,
                "device_state": dict(self.device_state),
                # Seeded placement offsets of the tray, the pad and the patient (m), see the layout.
                "spatial": dict((self.layout.info.get("autoinjector") or {}).get("spatial") or {}),
                "physical_model": (
                    "free rigid barrel with a sprung prismatic tip; blood pressure and heart rate "
                    "are time-compressed process models"
                ),
            }
        )
        return result
