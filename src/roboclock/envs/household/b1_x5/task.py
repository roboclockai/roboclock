"""B1: shut off the running faucet before the bowl under it overflows; spilled volume is the loss."""

from dataclasses import dataclass

import numpy as np

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task

# Inner profile of the bowl in the scene, from its mesh: water level above the bowl's inner floor, surface radius and
# volume held at that level. The last row is the rim.
BOWL_LEVEL_CM = (0.0, 0.56, 1.13, 1.69, 2.26, 2.82, 3.38, 3.95, 4.51, 5.08, 5.64, 6.2, 6.77, 7.33)
BOWL_RADIUS_CM = (0.13, 5.35, 6.37, 7.1, 7.55, 8.06, 8.34, 8.72, 9.05, 9.35, 9.72, 9.98, 10.28, 11.35)
BOWL_VOLUME_L = (0.0, 0.024, 0.083, 0.165, 0.258, 0.367, 0.488, 0.613, 0.754, 0.905, 1.062, 1.236, 1.419, 1.623)

# Abnormal surge flow models a supply fault, rather than a normal faucet rating.


@dataclass(frozen=True)
class B1Spec(X5Spec):
    max_time: float = 60.0
    headroom_l: float = 1.2
    inflow_l_s: float = 0.45  # abnormal full-open surge, before difficulty scaling
    normal_flow_l_s: float = 0.025
    surge_time: float = 2.0
    closed_flow_l_s: float = 0.005
    closure_dwell: float = 1.0
    fail_overflow_l: float = 0.5

    # Per-seed placement (sim.py Oracle.setup): the faucet (spout and lever, with the lever's hinge) slides on the sink
    # deck. Translation only: the lever's hinge axis is the world y axis the push is planned about, and the sink, the
    # basin and the bowl under the spout are fixed (the stream still lands in the 23.5 cm bowl).
    LAYOUT_JITTER = {"faucet": (0.015, 0.015, 0.0)}

    @classmethod
    def hidden_ranges(cls):
        return {"headroom_l": (0.8, 1.4), "surge_time": (1.8, 3.0)}


class FaucetOverflow(X5Task):
    task_id, name, category = "B1", "Shut off a sudden water surge before the bowl overflows", "event_reactive"
    Spec = B1Spec
    # States the overflow limit, how the lever moves (hinged about y: a sideways push of its top closes it,
    # pressing down does nothing; 49 of 59 calls of the 2026-11-10 sweep pressed down) and the release.
    instruction = 'The gentle faucet stream suddenly surges. Wait for the surge, then shut it off; half a litre of overflow fails. The lever is the short rod right of the faucet neck. It cannot be pressed down: push its top about 2 cm sideways toward the neck (to the left, seen from the robot) until it stands upright. Then let go: it must stay shut and untouched for 1 second.'
    SIGNALS = {
        "lever_opening": ("measured faucet valve opening, 1 = fully open", 1.0),
        "hand_on_lever": ("any robot contact with the faucet lever", False),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.stored_above_start_l = 0.0
        self.overflow_l = 0.0
        self.closed_hold = Hold()


    def inflow(self):
        rate = self.spec.inflow_l_s if self.time >= self.spec.surge_time else self.spec.normal_flow_l_s
        return rate * self.spec.time_scale * max(0.0, min(1.0, self.signals["lever_opening"]))

    def process(self, dt):
        s = self.signals
        if self.time >= self.spec.surge_time:
            self.mark("event")
        elif s["hand_on_lever"] or s["lever_opening"] < 0.9:
            self.failure("pre_event_action")
        q_in = self.inflow()
        self.stored_above_start_l += q_in * dt
        if self.stored_above_start_l > self.spec.headroom_l:
            self.overflow_l += self.stored_above_start_l - self.spec.headroom_l
            self.stored_above_start_l = self.spec.headroom_l
            self.mark("overflow_started")
        if s["hand_on_lever"]:
            self.mark("first_contact")
        closed = q_in < self.spec.closed_flow_l_s * self.spec.time_scale and not s["hand_on_lever"]
        if self.closed_hold.step(closed, dt) >= self.spec.closure_dwell and "first_contact" in self.events and "event" in self.events:
            self.mark("effective_closure", time_of_closure=self.time - self.spec.closure_dwell)
        if "effective_closure" in self.events and q_in > self.spec.closed_flow_l_s * self.spec.time_scale:
            self.failure("faucet_reopened")
        if self.overflow_l >= self.spec.fail_overflow_l:
            self.violation("overflow_limit")
        self.done = "effective_closure" in self.events or "overflow_limit" in self.violations

    def cues(self):
        volume = BOWL_VOLUME_L[-1] - self.spec.headroom_l + self.stored_above_start_l
        level = float(np.interp(volume, BOWL_VOLUME_L, BOWL_LEVEL_CM))
        return {"water_level_cm": level, "water_radius_cm": float(np.interp(level, BOWL_LEVEL_CM, BOWL_RADIUS_CM)),
                "water_below_rim_cm": BOWL_LEVEL_CM[-1] - level, "stream_l_s": self.inflow(),
                "surging": self.time >= self.spec.surge_time,
                "overflowing": self.stored_above_start_l >= self.spec.headroom_l, "spilled_l": self.overflow_l}

    @property
    def physical_success(self):
        return "effective_closure" in self.events and not self.physical_failures

    @property
    def temporal_success(self):
        return self.physical_success and self.overflow_l < self.spec.fail_overflow_l

    def task_metrics(self):
        closure = self.events.get("effective_closure")
        return {"reaction_latency": None if closure is None else closure - self.spec.closure_dwell - self.events["event"],
                "overflow_l": self.overflow_l, "loss_score": max(0.0, 1 - self.overflow_l / self.spec.fail_overflow_l)}


TASK = FaucetOverflow
