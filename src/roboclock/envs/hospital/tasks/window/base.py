"""Process-window tasks A1-A3: act on a bedside device while a monitored quantity is in its band.

A process (a filling drip chamber, an emptying supply bag, a syringe pump's dispensed volume) runs
from the start of the episode. Its quantity enters the target band at the window opening t_o and
leaves it at t_c; the band is marked on the apparatus. One device action, measured by the
apparatus module (a clamp slid shut, a selector turned, a key pressed), freezes the process.
Success needs that action inside [t_o, t_c], inclusive: earlier leaves the quantity short of the
band, later overshoots it. Physical completion (the measured action) and timing are scored apart.

Apparatus modules report, in ``state.controls``:
    device_moved: bool     the device has left its rest state (contact-driven travel)
    device_actuated: bool  the action is complete (end of travel, detent, or held key press)
"""

from roboclock.envs.hospital.mdp.events import GREEN, RED, SceneCommand, window_color
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask


class ProcessWindowTask(HospitalTask):
    category = "temporal_window"
    discipline = "infusion"
    band = (0.0, 1.0)  # target band of the monitored quantity, in `unit`
    unit = ""
    hold_after = 1.5  # seconds shown after the action, so the frozen quantity is visible

    def __init__(self, spec):
        super().__init__(spec)
        ((self.window_open, self.window_close),) = spec.windows
        self.device_moved_time = None

    # --- the process ---------------------------------------------------------------------

    @property
    def rate(self) -> float:
        """Quantity units per second; the band takes exactly the window to cross."""
        return (self.band[1] - self.band[0]) / (self.window_close - self.window_open)

    def quantity(self, time: float | None = None) -> float:
        """The monitored quantity, frozen at the action time once the device has acted."""
        t = self.time if time is None else time
        if self.action_time is not None:
            t = min(t, self.action_time)
        return self.band[0] + (t - self.window_open) * self.rate

    def initial_quantity(self) -> float:
        return self.quantity(0.0)

    def outcome(self) -> str:
        if self.action_time is None:
            return "overshoot" if self.time > self.window_close + EPS else "no_action"
        if self.action_time < self.window_open - EPS:
            return "short"
        if self.action_time > self.window_close + EPS:
            return "overshoot"
        return "in_band"

    # --- simulation loop -------------------------------------------------------------------

    def process(self, dt):
        if self.event_time is None and self.time + EPS >= self.window_open:
            self.event_time = self.time
            self.event_log.append({"time": self.time, "type": "window_open"})
        if self.action_time is not None:
            color = GREEN if self.temporal_success else RED
            fraction, bar = 1.0, "green" if self.temporal_success else "red"
        else:
            color = window_color(self.time, self.window_open, self.window_close)
            if self.time < self.window_open:
                fraction, bar = 1.0, "amber"  # waiting: the band has not been reached
            else:
                left = (self.window_close - self.time) / (self.window_close - self.window_open)
                fraction, bar = max(left, 0.0), None
        return [SceneCommand("indicator", "indicator", color)] + self.timer(
            "indicator", fraction, bar
        )

    def evaluate(self, state, dt):
        controls = state.controls
        if bool(controls.get("device_moved", False)) and self.device_moved_time is None:
            self.device_moved_time = self.time
            self.attempted.add("device")
            self.event_log.append({"time": self.time, "type": "device_moved"})
        if bool(controls.get("device_actuated", False)) and self.action_time is None:
            self.record_action()
            self.goal_achieved = True
            self.event_log.append(
                {"time": self.time, "type": "device_actuated", "quantity": self.quantity()}
            )
            if self.time < self.window_open - EPS:
                self.violation("early_action", self.window_open - self.time)
            elif self.time > self.window_close + EPS:
                self.violation("late_action", self.time - self.window_close)
            else:
                self.correct.add("device")
        if self.action_time is None and self.time > self.window_close + EPS:
            self.violation("missed_window", self.time - self.window_close)

    # --- outcomes ----------------------------------------------------------------------------

    @property
    def temporal_success(self) -> bool:
        return (
            self.action_time is not None
            and self.window_open - EPS <= self.action_time <= self.window_close + EPS
            and not self.violations
        )

    @property
    def terminated(self) -> bool:
        if bool(self.dropped):
            return True
        if self.action_time is not None:
            return self.time + EPS >= self.action_time + self.hold_after
        return self.time + EPS >= self.spec.max_time

    def metrics(self):
        result = super().metrics()
        result.update(
            window_open=self.window_open,
            window_close=self.window_close,
            window_width=self.window_close - self.window_open,
            band=list(self.band),
            unit=self.unit,
            initial_quantity=self.initial_quantity(),
            final_quantity=self.quantity(),
            quantity_outcome=self.outcome(),
            device_moved_time=self.device_moved_time,
        )
        return result
