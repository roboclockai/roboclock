"""A1: open the roller clamp, then slide it shut while the drip chamber's level is inside its band.

A saline bag hangs from a tabletop stand over a trainer drip chamber whose printed scale reads LOW,
a green TARGET band (55-85 % of the chamber height) and a red OVER zone. The episode starts with
the roller clamp CLOSED and the level still, LOW. Two measured device actions (bedside/window_drip.py
reads both from the roller's rigid-body pose):

1. Open: the roller slid from its CLOSED end stop to the OPEN end. The bag starts to drip; the level
   rises linearly and reaches the band ``fill_lead`` seconds later (seeded, the same for every level
   of a seed), then crosses it in the window width (6 / 4.5 / 3 s).
2. Close: the roller slid back to the CLOSED end, which freezes the level. Success needs it inside
   the band, i.e. inside [t_open + fill_lead, t_open + fill_lead + width]; earlier leaves the level
   LOW, later overfills it into OVER.

The spec's window is therefore relative to the opening: ``spec.windows == ((fill_lead, fill_lead +
width),)``. ``window_open`` / ``window_close`` are absolute episode times once the clamp is open and
infinite before. Never opening is ``no_action``.
"""

import math

from roboclock.envs.hospital.tasks.base import HospitalTask
from roboclock.envs.hospital.tasks.window.base import ProcessWindowTask


class DripChamberLevel(ProcessWindowTask):
    task_id = "A1"
    name = "drip_chamber_level"
    bedside_module = "window_drip"
    # Percent of the drip chamber's height. 30 % of the 15 cm chamber reads as ~12 px in the 60-degree
    # head view; at hard (3 s window, fill_lead up to 5 s) the level still starts at >= 5 %.
    band = (55.0, 85.0)
    unit = "% chamber"
    instruction = (
        "The roller clamp on the saline line is CLOSED. Slide it to OPEN so the saline bag drips "
        "into the drip chamber, then slide it back to CLOSED while the liquid level is inside the "
        "green TARGET band on the chamber: too early leaves it LOW, too late overfills it into the "
        "red OVER zone."
    )

    def __init__(self, spec):
        self.open_time = None
        super().__init__(spec)  # assigns the spec's relative window through the setters below

    # --- the window, relative to the opening -------------------------------------------------

    @property
    def fill_lead(self) -> float:
        """Seconds from the opening until the level reaches the band."""
        return self._relative[0]

    @property
    def window_width(self) -> float:
        return self._relative[1] - self._relative[0]

    @property
    def window_open(self) -> float:
        return math.inf if self.open_time is None else self.open_time + self._relative[0]

    @window_open.setter
    def window_open(self, value):
        self._relative = (float(value), getattr(self, "_relative", (0.0, 0.0))[1])

    @property
    def window_close(self) -> float:
        return math.inf if self.open_time is None else self.open_time + self._relative[1]

    @window_close.setter
    def window_close(self, value):
        self._relative = (self._relative[0], float(value))

    # --- the process -----------------------------------------------------------------------

    @property
    def rate(self) -> float:
        return (self.band[1] - self.band[0]) / self.window_width

    def initial_quantity(self) -> float:
        """The still level before the clamp opens: LOW, `fill_lead` seconds below the band."""
        return self.band[0] - self.fill_lead * self.rate

    def quantity(self, time: float | None = None) -> float:
        if self.open_time is None:
            return self.initial_quantity()
        t = self.time if time is None else time
        if self.action_time is not None:
            t = min(t, self.action_time)
        level = self.initial_quantity() + max(t - self.open_time, 0.0) * self.rate
        return min(level, 100.0)  # a missed window fills the chamber to the brim

    def outcome(self) -> str:
        if self.open_time is None:
            return "no_action"
        return super().outcome()

    def evaluate(self, state, dt):
        if bool(state.controls.get("device_opened", False)) and self.open_time is None:
            self.open_time = self.time
            self.attempted.add("open")
            self.event_log.append({"time": self.time, "type": "clamp_opened"})
        super().evaluate(state, dt)

    # --- outcomes ----------------------------------------------------------------------------

    def metrics(self):
        result = HospitalTask.metrics(self)
        opened = self.open_time is not None
        result.update(
            window_open=self.window_open if opened else None,
            window_close=self.window_close if opened else None,
            window_width=self.window_width,
            fill_lead=self.fill_lead,
            open_time=self.open_time,
            band=list(self.band),
            unit=self.unit,
            initial_quantity=self.initial_quantity(),
            final_quantity=self.quantity(),
            quantity_outcome=self.outcome(),
            device_moved_time=self.device_moved_time,
        )
        return result

    def gen_instruction(self):
        return (
            f"{self.instruction} After opening, the level reaches the band in about "
            f"{self.fill_lead:.1f} seconds and stays in it for {self.window_width:g} seconds."
        )

