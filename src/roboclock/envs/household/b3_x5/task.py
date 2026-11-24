"""B3: wait for a randomly timed oil fire, then turn its burner off. Flame exposure is the loss.

Time scale: a freely burning pan of oil under wall cabinets (no range hood) puts the cabinets at risk after
as little as 40 s and can ignite them from 75 s (NBSIR 75-788); the flame here spreads at 64 / 48 / 32 s
nominal (easy / medium / hard). Abstraction: turning the burner off puts the flame out and no lid is used
(the lid step was removed on 2026-11-02); real guidance also covers the pan. See design.json.
"""

from dataclasses import dataclass

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task


@dataclass(frozen=True)
class B3Spec(X5Spec):
    max_time: float = 68.0  # allow the full response horizon after the latest ignition
    ignition_time: float = 5.0  # seed 0 nominal; other seeds sample 2-8 s after the first observation
    growth: float = 0.065  # 1/s, logistic, while the burner is on
    f0: float = 0.15
    spread: float = 0.8
    off_tau: float = 1.0  # s: with the gas off the flame dies within a few seconds
    off_angle_deg: float = 10.0
    out_threshold: float = 0.01
    observe_out_s: float = 3.0

    # Per-seed placement (sim.py Oracle.setup): the cooktop (with its knobs, the pot on its burner, the oil and the
    # flame) slides on the counter. Translation only: the oracle turns the knob with joint 6 alone and the turn
    # ends 2.5 deg short of the knob's stop at the joint-6 limit, so a turned cooktop would leave the knob short of
    # OFF (sim.py KNOB_JAW0). The glass of water is not touched by the task and stays.
    LAYOUT_JITTER = {"cooktop": (0.015, 0.015, 0.0)}

    @classmethod
    def hidden_ranges(cls):
        return {"growth": (0.05, 0.08), "f0": (0.12, 0.18), "ignition_time": (2.0, 8.0)}


class PotFire(X5Task):
    task_id, name, category = "B3", "Turn off the burner under a pot of burning oil", "event_reactive"
    Spec = B3Spec
    # States which knob, which way and how far it turns to OFF, the release, and the pot handle over the knob
    # strip (sim.py KNOB_JAW0: a hand whose jaws close front to back does not fit under it).
    instruction = ("The oil in the pot on the stove may catch fire at an unpredictable time. Wait for visible "
                   "flames, then turn off its burner before the fire spreads: "
                   "its knob is the middle one of the five and goes off with a quarter turn clockwise (seen from "
                   "above), to its stop. The gas is off only within 10 degrees of that stop, and it counts once no "
                   "gripper touches the knob, so let go. The pot's front handle overhangs the cooktop: its edge is "
                   "6 cm behind the knob's axis, 5 to 9 cm above the knob.")
    SIGNALS = {
        "knob_angle_deg": ("center burner knob angle, 0 = OFF", 90.0),
        "hand_on_knob": ("robot contact with the burner knob", False),
        "water_contact": ("water entered the pot", False),
        "pot_displacement": ("pot displacement from the burner (m)", 0.0),
        "wrong_knob_turned": ("another burner knob was turned", False),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.flame = 0.0
        self.exposure = 0.0
        self.out_hold = Hold()

    def process(self, dt):
        s, k = self.signals, self.spec.time_scale
        heat_on = s["knob_angle_deg"] > self.spec.off_angle_deg
        if s["pot_displacement"] > 0.03:
            self.failure("pot_moved")
        if s["wrong_knob_turned"]:
            self.violation("wrong_knob")
        if "event" not in self.events:
            if self.time < self.spec.ignition_time or not heat_on:
                self.done = bool(self.physical_failures)
                return
            self.flame = self.spec.f0
            self.mark("event")
            # Only integrate the part of this tick after ignition.
            dt = min(dt, max(0.0, self.time - self.spec.ignition_time))
        if s["water_contact"]:
            self.flame = 1.0
            self.failure("water_on_oil_fire")
        if heat_on:
            self.flame += self.spec.growth * k * self.flame * (1 - self.flame) * dt
        else:
            self.flame -= self.flame / self.spec.off_tau * dt
        if not heat_on and not s["hand_on_knob"]:
            self.mark("heat_off")
        self.flame = min(1.0, max(0.0, self.flame))
        self.exposure += self.flame * dt
        if self.flame >= self.spec.spread:
            self.violation("fire_spread")
        out = self.flame < self.spec.out_threshold
        if out:
            self.mark("flame_out")
        complete = "heat_off" in self.events and not heat_on and not s["hand_on_knob"] and out
        if self.out_hold.step(complete, dt) >= self.spec.observe_out_s:
            self.mark("secured")
        self.done = "secured" in self.events or "fire_spread" in self.violations or bool(self.physical_failures)

    def cues(self):
        return {"flame_intensity": self.flame}

    @property
    def physical_success(self):
        return "secured" in self.events and not self.physical_failures

    @property
    def temporal_success(self):
        return self.physical_success and "fire_spread" not in self.violations

    def task_metrics(self):
        return {"ignition_time": self.events.get("event"),
                "heat_off_time": self.events.get("heat_off"), "flame_out_time": self.events.get("flame_out"),
                "flame_exposure": self.exposure}


TASK = PotFire
