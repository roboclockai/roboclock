"""A3: ventilate an intubated manikin with a resuscitator bag, one breath every 6 seconds.

A self-inflating bag is connected to the manikin's tracheal tube. A ventilation timer on the
bedside monitor sweeps once per breath cycle (6 s: 10 breaths per minute, the rate for an adult
with an advanced airway) and marks the part of the cycle in which the breath is due. Each breath
is one full squeeze of the bag, measured by the apparatus module as compression travel under robot
contact (bedside/window_bvm.py); the bag must be let re-expand before the next one counts.

Scoring, physical and temporal apart:
- physical: all `num_items` breaths delivered (full squeezes), whenever they came.
- temporal: every breath window holds exactly one delivered breath and nothing else was
  delivered. A squeeze before its window opens is `early_breath` (hyperventilation), a second
  one inside a served window or between windows `extra_breath`, a window that closes unserved
  `missed_window` (hypoventilation).

The apparatus module reports, in ``state.controls``:
    device_moved: bool      the bag has been compressed under robot contact (> 3 mm)
    breath_times: tuple     the time of every delivered breath so far
"""

from roboclock.envs.hospital.mdp.events import BLUE, GREEN, RED, SceneCommand
from roboclock.envs.hospital.tasks.base import EPS, HospitalTask

AMBER = (1.0, 0.6, 0.05)


class BagValveVentilation(HospitalTask):
    task_id = "A3"
    name = "bvm_ventilation"
    category = "temporal_window"
    discipline = "airway"
    bedside_module = "window_bvm"
    period = 6.0  # seconds between breaths (config tasks.A3.beat_period)
    hold_after = 1.5  # seconds shown after the last window is settled
    instruction = (
        "An intubated manikin lies on the bed; the resuscitator bag on the trolley is connected "
        "to its tracheal tube. Ventilate at 10 breaths per minute, one breath every 6 seconds: "
        "the VENT TIMER on the monitor sweeps once per breath and its bar crosses the green "
        "band when the breath is due. For each breath press the bag down fully (the chest "
        "rises), then let it refill. Too early or an extra squeeze hyperventilates, too late "
        "or a missed breath hypoventilates."
    )

    def __init__(self, spec):
        super().__init__(spec)
        self.windows = tuple(spec.windows)
        self.beats = tuple((start + end) / 2 for start, end in self.windows)
        if len(self.beats) > 1:
            self.period = self.beats[1] - self.beats[0]
        self.served = {}  # window index -> delivery time
        self.missed = set()  # windows that closed without a breath
        self.breath_times = []  # every delivered breath, in or out of a window
        self.device_moved_time = None
        self.settled_time = None  # when the last window was served or missed

    def gen_instruction(self):
        width = self.windows[0][1] - self.windows[0][0]
        return (
            f"{self.instruction} Deliver {len(self.windows)} breaths; the green band is "
            f"{width:g} seconds wide."
        )

    # --- the breath cycle ------------------------------------------------------------------

    @property
    def window_open(self):
        return self.windows[0][0]

    @property
    def window_close(self):
        return self.windows[-1][1]

    def cycle(self, time=None):
        """(index, seconds into the cycle) of the breath cycle at `time`; each cycle is one
        period long and has its beat in the middle. Index -1 before the first cycle, and the
        last cycle keeps running to the end of the episode."""
        t = self.time if time is None else time
        start = self.beats[0] - self.period / 2
        if t < start:
            return -1, 0.0
        index = min(int((t - start) / self.period), len(self.beats) - 1)
        return index, t - (start + index * self.period)

    def band(self):
        """The due band inside a cycle, seconds from the cycle start."""
        half = (self.windows[0][1] - self.windows[0][0]) / 2
        return self.period / 2 - half, self.period / 2 + half

    def current_window(self, time=None):
        """Index of the window that is open at `time`, or None."""
        t = self.time if time is None else time
        for index, (start, end) in enumerate(self.windows):
            if start - EPS <= t <= end + EPS:
                return index
        return None

    # --- simulation loop -------------------------------------------------------------------

    def process(self, dt):
        if self.event_time is None and self.time + EPS >= self.window_open:
            self.event_time = self.time
            self.event_log.append({"time": self.time, "type": "window_open"})
        for index, (_, end) in enumerate(self.windows):
            if index not in self.served and index not in self.missed and self.time > end + EPS:
                self.missed.add(index)
                self.violation("missed_window", self.time - end)
        if self.settled_time is None and self._settled():
            self.settled_time = self.time
        index = self.current_window()
        if self.settled_time is not None:
            color = GREEN if self.temporal_success else RED
            fraction, bar = 1.0, "green" if self.temporal_success else "red"
        elif index is None:
            color, fraction, bar = BLUE, 1.0, "amber"  # waiting for the next breath
        elif index in self.served:
            color, fraction, bar = GREEN, 1.0, "green"
        else:
            start, end = self.windows[index]
            color, fraction, bar = GREEN, max((end - self.time) / (end - start), 0.0), None
        return [SceneCommand("indicator", "indicator", color)] + self.timer(
            "indicator", fraction, bar
        )

    def _settled(self):
        last = len(self.windows) - 1
        return last in self.served or self.time > self.windows[last][1] + EPS

    def evaluate(self, state, dt):
        controls = state.controls
        if bool(controls.get("device_moved", False)) and self.device_moved_time is None:
            self.device_moved_time = self.time
            self.attempted.add("bag")
            self.event_log.append({"time": self.time, "type": "device_moved"})
        delivered = tuple(controls.get("breath_times", ()))
        for _ in delivered[len(self.breath_times) :]:
            self._breath(self.time)
        self.goal_achieved = len(self.breath_times) >= len(self.windows)

    def _breath(self, time):
        """Score one delivered breath at `time` (the task clock when the squeeze completed)."""
        number = len(self.breath_times)
        self.breath_times.append(time)
        self.record_action()
        self.attempted.add(f"breath_{number}")
        index = self.current_window(time)
        entry = {"time": time, "type": "breath_delivered", "number": number, "window": index}
        self.event_log.append(entry)
        if index is not None and index not in self.served:
            self.served[index] = time
            self.correct.add(f"breath_{number}")
            return
        upcoming = [start for start, _ in self.windows if time < start - EPS]
        if index is None and upcoming:
            self.violation("early_breath", upcoming[0] - time)  # between windows: too soon
        else:
            self.violation("extra_breath")  # a second breath in a served window, or after the last

    # --- outcomes --------------------------------------------------------------------------

    @property
    def temporal_success(self) -> bool:
        return len(self.served) == len(self.windows) and not self.violations

    @property
    def terminated(self) -> bool:
        if bool(self.dropped):
            return True
        if self.settled_time is not None and self.time + EPS >= self.settled_time + self.hold_after:
            return True
        return self.time + EPS >= self.spec.max_time

    def outcome(self) -> str:
        if not self.breath_times:
            return "no_breath"
        if self.temporal_success:
            return "in_rhythm"
        if "extra_breath" in self.violations or "early_breath" in self.violations:
            return "hyperventilation"
        return "hypoventilation"

    def metrics(self):
        result = super().metrics()
        intervals = [b - a for a, b in zip(self.breath_times, self.breath_times[1:])]
        result.update(
            windows=[list(window) for window in self.windows],
            window_open=self.window_open,
            window_close=self.window_close,
            window_width=self.windows[0][1] - self.windows[0][0],
            beat_period=self.period,
            beats=list(self.beats),
            breath_times=list(self.breath_times),
            breaths_delivered=len(self.breath_times),
            breaths_in_window=len(self.served),
            breaths_required=len(self.windows),
            windows_missed=sorted(self.missed),
            beat_errors=[self.served[k] - self.beats[k] for k in sorted(self.served)],
            breath_intervals=intervals,
            rhythm_outcome=self.outcome(),
            device_moved_time=self.device_moved_time,
            first_breath_time=self.breath_times[0] if self.breath_times else None,
        )
        return result
