"""Alarm-clock promo scene (not a benchmark task; scripts/household/promo_alarm_clock.py renders it).

A wind-up alarm clock starts ringing at a hidden time; press its stop button down to silence it for good.

The clock is a round twin-bell alarm clock standing on the counter. Its stop button is a plunger on top of the
carrying handle: pushed down past its catch it latches down and the alarm stays off (a shut-off plunger, not a
snooze). The time the alarm rings before it is silenced is the loss; a tier limit on it decides the timing.
"""

from dataclasses import dataclass

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task

# Longest ringing allowed per tier (s from the first ring to the button passing its catch). Hard follows the
# benchmark rule max(t / 0.8, t + 0.3 s, 1 s) on the reference (oracle) controller's slowest silencing over seeds
# 0-9 (t = 1.10 s from the ready pose, 2026-11-22: a reference level, not an optimum); medium is the geometric mean
# of easy and hard. Easy is a design choice (no source found for how long such an alarm is left ringing).
LIMIT_S = {"easy": 15.0, "medium": 4.6, "hard": 1.4}


@dataclass(frozen=True)
class AlarmSpec(X5Spec):
    max_time: float = 26.0  # latest ring (8 s) + the easy limit + the release dwell and a margin
    ring_time: float = 5.0  # seed 0 nominal; other seeds sample 2-8 s after the first observation
    limit_s: float = LIMIT_S["medium"]  # this tier's limit; sample() sets it from LIMIT_S
    travel_mm: float = 6.0  # full plunger travel
    catch_mm: float = 4.0  # pressed this far, the plunger latches and the alarm is off
    release_dwell: float = 1.0
    tilt_limit_deg: float = 30.0
    move_limit_m: float = 0.05

    # Per-seed placement (sim.py Oracle.setup): the clock, with its button, slides and turns on the counter. Small:
    # the promo layout stands it 4 cm from the counter's front edge, 13 cm from the left arm's mount.
    LAYOUT_JITTER = {"clock": (0.02, 0.01, 8.0)}

    @classmethod
    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        overrides.setdefault("limit_s", LIMIT_S.get(difficulty, LIMIT_S["medium"]))
        return super().sample(task_id, difficulty, seed, **overrides)

    @classmethod
    def hidden_ranges(cls):
        return {"ring_time": (2.0, 8.0)}


class AlarmClock(X5Task):
    task_id, name, category = "ALARM", "Silence the ringing alarm clock", "promo"
    Spec = AlarmSpec
    instruction = ("The alarm clock on the counter will start ringing at an unpredictable time: its bells and hammer "
                   "shake and the whole clock rattles. Wait until it rings, then silence it within 4.6 seconds: "
                   "press the red button on top of its handle straight down, from above, about 5 mm, until it stays "
                   "down. Then let go: it counts once nothing touches the button for 1 second. Do not touch the "
                   "button before the alarm rings, and do not knock the clock over or push it more than 5 cm.")
    SIGNALS = {
        "button_travel_mm": ("measured stop-button travel, 0 = up", 0.0),
        "hand_on_button": ("any robot contact with the stop button", False),
        "clock_tilt_deg": ("tilt of the clock from upright", 0.0),
        "clock_displacement_m": ("horizontal displacement of the clock from its initial place", 0.0),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.ringing = False
        self.rang_s = 0.0
        self.released = Hold()

    def gen_instruction(self):
        return self.instruction.replace("4.6 seconds", f"{self.spec.limit_s:g} seconds")

    def process(self, dt):
        s, spec = self.signals, self.spec
        pressed = s["button_travel_mm"] >= spec.catch_mm
        if s["clock_tilt_deg"] > spec.tilt_limit_deg:
            self.failure("clock_knocked_over")
        if s["clock_displacement_m"] > spec.move_limit_m:
            self.failure("clock_pushed_away")
        if "event" not in self.events:
            if s["hand_on_button"] or pressed:  # switching the alarm off beforehand is not a reaction
                self.failure("pre_event_action")
            elif self.time >= spec.ring_time:
                self.mark("event")
                self.ringing = True
        if s["hand_on_button"] and "event" in self.events:
            self.mark("first_contact")
        if self.ringing:
            if pressed:
                self.ringing = False  # latched: it does not ring again
                self.mark("silenced")
            else:
                self.rang_s += dt
                if self.rang_s > spec.limit_s:
                    self.violation("rang_too_long")
        if "silenced" in self.events and self.released.step(not s["hand_on_button"], dt) >= spec.release_dwell:
            self.mark("secured")
        # A late silencing still completes the task physically, so the episode runs on past the limit.
        self.done = "secured" in self.events or bool(self.physical_failures)

    def cues(self):
        return {"ringing": self.ringing, "silenced": "silenced" in self.events,
                "ring_elapsed_s": self.time - self.events["event"] if self.ringing else 0.0}

    @property
    def physical_success(self):
        return "secured" in self.events and not self.physical_failures

    @property
    def temporal_success(self):
        return self.physical_success and "rang_too_long" not in self.violations

    def task_metrics(self):
        silenced = self.events.get("silenced")
        return {"ring_time": self.events.get("event"), "silenced_time": silenced,
                "reaction_latency": None if silenced is None else silenced - self.events["event"],
                "rang_s": self.rang_s, "limit_s": self.spec.limit_s,
                "loss_score": max(0.0, 1 - self.rang_s / self.spec.limit_s)}


TASK = AlarmClock
