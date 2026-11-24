"""A4: pour 300 +/- 10 g of water onto the scale and stop in time (115 / 64 / 36 s budget at easy / medium / hard)."""

import math
from collections import deque
from dataclasses import dataclass

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task

# Completion budget per tier (s from t = 0), recalibrated 2026-11-16 on the reference (oracle) controller: its slowest
# kettle-down time over seeds 0-9 is t = 28.69 s at every tier (the tier does not change the pour), so
# hard = max(t / 0.8, t + 0.3) = 35.86 -> 36 s. Easy (2026-11-23, user rule: a documented time where one exists, else
# 4 x t): no document gives a time, so 4 x 28.69 = 114.8 -> 115 s (was 60 s); medium = sqrt(115 x 36) = 64.3 -> 64 s
# (was 46 s).
BUDGET_S = {"easy": 115.0, "medium": 64.0, "hard": 36.0}


@dataclass(frozen=True)
class A4Spec(X5Spec):
    max_time: float = 75.0
    target_g: float = 300.0
    tolerance_g: float = 10.0
    budget_s: float = BUDGET_S["medium"]  # this tier's budget; sample() sets it from BUDGET_S
    kettle_ml: float = 1000.0
    kettle_capacity_ml: float = 1500.0
    flow_coefficient: float = 1.0  # multiplies the nominal tilt-flow curve
    max_flow_ml_s: float = 40.0
    tail_tau_s: float = 0.15
    display_lag_s: float = 0.3
    spill_limit_g: float = 5.0
    settle_s: float = 3.0

    # Per-seed placement (sim.py Oracle.setup): the kettle about its body axis. +-1.5 cm / +-5 deg only: both arms must
    # reach its handle and lid knob together, and hard leaves little time for a longer carry. The jug on the scale is
    # a fixed fixture (collider and water cues at JUG_XY).
    LAYOUT_JITTER = {"kettle": (0.015, 0.015, 5.0)}

    @classmethod
    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        overrides.setdefault("budget_s", BUDGET_S.get(difficulty, BUDGET_S["medium"]))
        # The episode must outlast the budget and the dose read (settle_s): 75 s unless the budget is longer.
        overrides.setdefault("max_time", max(cls.max_time, overrides["budget_s"] + 5.0))
        return super().sample(task_id, difficulty, seed, **overrides)

    @classmethod
    def hidden_ranges(cls):
        return {"kettle_ml": (700.0, 1100.0), "flow_coefficient": (0.85, 1.15)}


class PourToWeight(X5Task):
    task_id, name, category = "A4", "Pour 300 g of hot water and stop in time", "temporal_window"
    Spec = A4Spec
    # States the release, the spill rule and its 5 g limit, the two steps of the judging (the kettle standing
    # released within the budget, the dose read settle_s later) and the kettle's weight (0.8 kg shell + water).
    instruction = ("Pour 300 g (+/- 10 g) of water from the kettle into the glass jug on the scale, then set the "
                   "kettle down upright on the counter and let go. Water runs out once the kettle tilts roughly 40 "
                   "to 60 degrees toward its spout (further as it empties); what leaves while the spout's tip is "
                   "not above the jug's mouth is spilled, and 5 g spilled fails the task. The kettle with its water "
                   "weighs 1.5 to 1.9 kg. Someone is waiting: finish within 64 seconds, with the kettle standing "
                   "released; the dose is read 3 s later.")
    SIGNALS = {
        "kettle_tilt_deg": ("kettle tilt from upright toward the spout", 0.0),
        "spout_over_jug": ("stream lands inside the jug mouth", False),
        "kettle_upright_released": ("kettle released, upright and supported on the counter", True),
        "jug_on_scale": ("jug still supported by the scale", True),
        "dropped": ("kettle or jug dropped/tipped", False),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.kettle_ml = spec.kettle_ml
        self.jug_g = 0.0
        self.spill_g = 0.0
        self.flow = 0.0
        self.history = deque()
        self.release_hold = Hold()
        self.final_dose = None

    def gen_instruction(self):
        # The budget scored is this tier's budget_s (easy 115 s, medium 64 s, hard 36 s): say that one.
        return self.instruction.replace("64 seconds", f"{self.spec.budget_s:g} seconds")

    def theta0(self):
        return 20.0 + 70.0 * (1 - self.kettle_ml / self.spec.kettle_capacity_ml)

    def process(self, dt):
        s = self.signals
        if s["kettle_tilt_deg"] > 5 and not s["kettle_upright_released"]:
            self.mark("pour_started")
        excess = s["kettle_tilt_deg"] - self.theta0()
        c = self.spec.max_flow_ml_s / 35.0 ** 1.5 * self.spec.flow_coefficient
        if excess > 0:
            self.flow = min(self.spec.max_flow_ml_s, c * excess ** 1.5)
        else:
            self.flow *= math.exp(-dt / self.spec.tail_tau_s)  # tail flow after returning upright
            if self.flow < 0.05:
                self.flow = 0.0
        volume = min(self.kettle_ml, self.flow * dt)
        self.kettle_ml -= volume
        if s["spout_over_jug"]:
            self.jug_g += volume
        else:
            self.spill_g += volume
        self.history.append((self.time, self.jug_g))
        while self.history and self.history[0][0] < self.time - self.spec.display_lag_s - 1.0:
            self.history.popleft()
        if not s["jug_on_scale"] or s["dropped"]:
            self.failure("jug_or_kettle_moved_or_dropped")
        if self.spill_g >= self.spec.spill_limit_g:
            self.failure("spill")
        # The pour is over once the kettle stands released again after water has left it. A tilt
        # past 5 deg that pours nothing (the kettle bumped, or tipped short of its pouring angle)
        # is not a pour: it used to end the episode 3 s later as a success with 0 g.
        if "pour_started" in self.events and self.kettle_ml < self.spec.kettle_ml:
            held = self.release_hold.step(s["kettle_upright_released"] and self.flow == 0.0, dt)
            if held > 0 and "kettle_down" not in self.events:
                self.mark("kettle_down")
            if held == 0:
                self.events.pop("kettle_down", None)
            if held >= self.spec.settle_s:
                self.final_dose = self.jug_g
                self.mark("dose_final", dose_g=self.jug_g)
                if self.jug_g <= 0.0:
                    self.failure("no_water_in_jug")  # all of it went beside the jug
        if self.jug_g > self.spec.target_g + self.spec.tolerance_g:
            self.violation("overshoot")
        self.done = bool(self.physical_failures) or "dose_final" in self.events

    def scale_reading(self):
        lagged = [g for t, g in self.history if t <= self.time - self.spec.display_lag_s]
        return float(round(lagged[-1] if lagged else 0.0))

    def cues(self):
        return {"scale_reading_g": self.scale_reading(), "stream_ml_s": self.flow}

    @property
    def physical_success(self):
        return not self.physical_failures and "dose_final" in self.events

    @property
    def temporal_success(self):
        if self.final_dose is None:
            return False
        in_tol = abs(self.final_dose - self.spec.target_g) <= self.spec.tolerance_g
        return in_tol and self.events.get("kettle_down", math.inf) <= self.spec.budget_s

    def task_metrics(self):
        return {"final_dose_g": self.final_dose, "dose_error_g": None if self.final_dose is None else self.final_dose - self.spec.target_g,
                "spill_g": self.spill_g, "completion_time": self.events.get("kettle_down")}


TASK = PourToWeight
