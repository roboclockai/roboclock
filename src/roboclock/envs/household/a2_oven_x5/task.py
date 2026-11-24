"""A2: take broiling cheese toast out of the countertop oven when it is golden-bubbly."""

import math
from dataclasses import dataclass

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task, band_quality


@dataclass(frozen=True)
class A2Spec(X5Spec):
    max_time: float = 360.0
    t_melt: float = 60.0
    tau: float = 90.0
    band: tuple = (0.55, 0.75)
    zero: tuple = (0.35, 0.90)
    door_open_rate: float = 0.4
    carryover: float = 0.03
    carryover_seconds: float = 20.0
    release_dwell: float = 1.0
    max_tilt_deg: float = 20.0

    @classmethod
    def hidden_ranges(cls):
        return {"t_melt": (45.0, 75.0), "tau": (75.0, 105.0)}


class OvenCheeseToast(X5Task):
    task_id, name, category = "A2", "Take cheese toast out of the countertop oven at the right moment", "temporal_window"
    Spec = A2Spec
    instruction = ("Cheese toast is broiling in the countertop oven. Take the tray out once the cheese "
                   "is bubbling with golden spots, before it browns further and burns, and put it on the wooden trivet.")
    SIGNALS = {
        "door_angle_deg": ("measured drop-down door angle, 0 = closed", 0.0),
        "tray_in_cavity": ("any part of the tray inside the oven cavity", True),
        "tray_on_trivet_released": ("tray released and supported on the trivet", False),
        "food_on_tray": ("both toasts still supported by the tray", True),
        "tray_tilt_deg": ("tray tilt from horizontal", 0.0),
        "dropped": ("tray or food dropped", False),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.heat_time = 0.0
        self.out_time = 0.0
        self.b_at_retrieval = None
        self.release_hold = Hold()

    def _b(self, heat_time):
        x = max(0.0, heat_time * self.spec.time_scale - self.spec.t_melt)
        return 1 - math.exp(-x / self.spec.tau)

    @property
    def browning(self):
        if self.b_at_retrieval is None:
            return self._b(self.heat_time)
        frac = min(1.0, self.out_time * self.spec.time_scale / self.spec.carryover_seconds)
        return self.b_at_retrieval + self.spec.carryover * frac

    def process(self, dt):
        s = self.signals
        if s["door_angle_deg"] > 45:
            self.mark("door_opened")
        if self.b_at_retrieval is None:
            if s["tray_in_cavity"]:
                self.heat_time += dt * (self.spec.door_open_rate if s["door_angle_deg"] > 10 else 1.0)
            else:
                self.b_at_retrieval = self._b(self.heat_time)
                self.mark("retrieval", browning=self.b_at_retrieval)
        else:
            self.out_time += dt
        if s["dropped"] or not s["food_on_tray"]:
            self.failure("food_or_tray_dropped")
        if s["tray_tilt_deg"] > self.spec.max_tilt_deg:
            self.failure("tray_tilted")
        if self.release_hold.step(s["tray_on_trivet_released"], dt) >= self.spec.release_dwell:
            self.mark("tray_on_trivet")
        settled = "tray_on_trivet" in self.events and self.out_time * self.spec.time_scale >= self.spec.carryover_seconds
        self.done = bool(self.physical_failures) or (settled and self.physical_success)

    def quality(self):
        return band_quality(self.browning if self.b_at_retrieval is not None else None, *self.spec.band, *self.spec.zero)

    def cues(self):
        return {"cheese_browning": self.browning}

    @property
    def physical_success(self):
        return not self.physical_failures and all(k in self.events for k in ("door_opened", "retrieval", "tray_on_trivet"))

    @property
    def temporal_success(self):
        return self.quality() == 1.0

    def task_metrics(self):
        return {"retrieval_time": self.events.get("retrieval"), "browning_at_retrieval": self.b_at_retrieval,
                "final_browning": self.browning if self.b_at_retrieval is not None else None, "quality": self.quality()}


TASK = OvenCheeseToast
