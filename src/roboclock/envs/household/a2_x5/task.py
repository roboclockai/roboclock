"""A2: fill a glass about half full of ice at a refrigerator-door style ice dispenser.

The unit is described as a refrigerator-door style dispenser (2026-11-16; text only, the scene keeps the
countertop body around the native refrigerator-door paddle): such dispensers meter ice out in batches
while the paddle is held, one batch about every 1.5 s (Whirlpool, US patent 4,942,979). Here a batch is one
cube, every 2.0 / 1.5 / 1.3 s nominal (easy / medium / hard). The mass rate (4.3 / 5.7 / 6.5 g/s) has no source.

Release window (2026-11-17): depending on how the cubes pile, the 40-60 % band holds one to three cube counts
(measured in Isaac with the paddle held, seeds 0-9 at every tier). With a single count in the band, the release
must come after that cube and at least ``coast_s`` before the next one is due: one interval minus the coast. The
cadence is set so that this worst case is at least 1.0 s at every tier and seed (sub-second limits approach human
reaction time): hidden interval U[1.4, 1.7] s (was U[1.1, 1.6], nominal 1.35 s), per-cube jitter +-5 % (was 12 %),
and at hard the cube cadence runs at x1.15 instead of the household x1.5 process rate (spin-up and coast still
x1.5; compare B1's flow cap). Worst single-count window: 1.61 / 1.21 / 1.08 s at easy / medium / hard; before the
change it was 1.13 / 0.85 / 0.56 s, and hard seed 2 (band = 5 cubes only) measured 0.94 s.

The robot picks the glass up from the counter, holds it under the dispenser's outlet and presses the
dispenser paddle with its rim, as a person does. While the paddle is held in, the dispenser delivers
cubes one at a time: after a spin-up delay the auger pushes a cube into the chute every ``interval_s``
(with a small per-cube jitter). Releasing the paddle (moving the glass away) stops the auger, which
coasts for ``coast_s``: a cube due within that time still drops. Dispensing is one press: the first
release after dispensing has started ends it. The glass is then set back down on the counter; the
ice level in it, once it stands there released and nothing moves, is the result. Releasing too early
leaves too little ice, releasing too late gives too much.

The task model decides only *when* a cube is delivered into the chute (``cues()['cubes_dispensed']``);
the simulation drops a real rigid cube from the outlet and measures the ice level in the glass (the top
of the highest cube over the glass's inner height, signal ``ice_level``) and the cube counts.
"""

from dataclasses import dataclass, replace

import numpy as np

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task, band_quality

POOL = 16  # cubes in the dispenser's bin (the scene's rigid-cube pool)
MAX_CUBE_RATE = 1.15  # cube cadence cap over the nominal (medium) rate: hard runs x1.15, not x1.5 (see the docstring)


@dataclass(frozen=True)
class A2Spec(X5Spec):
    max_time: float = 40.0
    level_band: tuple = (0.40, 0.60)  # "about half full": top of the ice over the glass's inner height
    level_zero: tuple = (0.20, 0.80)  # quality falls to 0 at these levels
    spin_up_s: float = 0.9      # from the press to the first cube (process seconds)
    interval_s: float = 1.5     # nominal time between cubes (process seconds; sample() caps the hard cadence)
    jitter: float = 0.05        # per-cube interval varies by up to +-5 % (seeded; none at seed 0)
    coast_s: float = 0.12       # auger run-on after the release (process seconds)
    press_dwell_s: float = 0.25  # a press must hold this long before it counts as a dispensing press
    settle_s: float = 1.0       # glass released on the counter and no cube moving for this long -> final level
    max_tilt_deg: float = 45.0

    LAYOUT_JITTER = {"glass": (0.02, 0.02, 10.0)}  # tumbler on the counter (axisymmetric: the yaw only turns its mesh)

    @classmethod
    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        """Wall interval between cubes = interval_s / time_scale: x0.75 / x1 at easy / medium, x1.15 (not x1.5) at hard,
        so one interval minus the coast stays >= 1.0 s (the release window when the band holds a single cube count)."""
        spec = super().sample(task_id, difficulty, seed, **overrides)
        if spec.time_scale > MAX_CUBE_RATE and "interval_s" not in overrides:
            spec = replace(spec, interval_s=spec.interval_s * spec.time_scale / MAX_CUBE_RATE)
        return spec

    @classmethod
    def hidden_ranges(cls):
        return {"spin_up_s": (0.6, 1.3), "interval_s": (1.4, 1.7)}


class IceDispenser(X5Task):
    task_id, name, category = "A2", "Fill a glass about half full of ice at the refrigerator-door style ice dispenser", "temporal_window"
    Spec = A2Spec
    # States the judged band, the single press, the release, and the two failures (a cube outside, the tilt limit).
    instruction = ("Fill the glass about half full at the refrigerator-door style ice dispenser: 40 to 60 % of its "
                   "inner height, at the top of the highest cube. Take the glass off the counter, hold it under the "
                   "outlet and push the paddle in with its rim. Ice is metered out in batches of one cube while the "
                   "paddle is held in, during the first press only: pull the glass back when there is enough, then "
                   "set it down upright on the counter and let go. Every cube must end up in the glass, which must "
                   "never tilt more than 45 degrees.")
    SIGNALS = {
        "paddle_deg": ("measured dispenser paddle deflection, 0 = rest", 0.0),
        "ice_level": ("top of the highest resting cube in the glass over the glass's inner height (0 empty, 1 at the rim)", 0.0),
        "cubes_in_glass": ("cubes inside the glass (centre within the inner wall, below the rim)", 0),
        "cubes_outside": ("dispensed cubes that came to rest outside the glass", 0),
        "cubes_moving": ("any dispensed cube still moving faster than 2 cm/s", False),
        "glass_tilt_deg": ("glass tilt from upright", 0.0),
        "glass_on_counter_released": ("glass upright on the counter, no robot contact", True),
        "glass_dropped": ("glass fell (below the counter or on its side)", False),
    }
    PRESS_DEG, RELEASE_DEG = 5.0, 3.0  # paddle micro-switch, with hysteresis

    def __init__(self, spec):
        super().__init__(spec)
        self.pressed = False
        self.run_s = 0.0       # process seconds the auger has run during the press
        self.coast_left = 0.0
        self.dispensed = 0
        self.final_count = None
        self.final_level = None
        self.settle_hold = Hold()
        self.press_hold = Hold()
        rng = np.random.default_rng([int(spec.seed), 7302]) if spec.seed else None
        jit = rng.uniform(-spec.jitter, spec.jitter, POOL) if rng is not None else np.zeros(POOL)
        # process time of the k-th cube (k = 0 ..) measured from the start of the press
        self.due = np.cumsum([spec.spin_up_s] + [spec.interval_s * (1 + j) for j in jit[1:]]).tolist()

    @property
    def auger_running(self):
        return "released" not in self.events and self.pressed or self.coast_left > 0

    def process(self, dt):
        s = self.signals
        travel = s["paddle_deg"]
        if not self.pressed and travel >= self.PRESS_DEG:
            self.pressed = True
            self.mark("paddle_pressed")
        elif self.pressed and travel <= self.RELEASE_DEG:
            self.pressed = False
            if "dispense_started" in self.events and "released" not in self.events:
                self.mark("released", cubes_dispensed=self.dispensed)
                self.coast_left = self.spec.coast_s
        if self.press_hold.step(self.pressed, dt) >= self.spec.press_dwell_s:
            self.mark("dispense_started")
        if not s["glass_on_counter_released"]:
            self.mark("glass_picked_up")
        # the auger runs while the (first) press is held, then coasts briefly after the release
        step = dt * self.spec.time_scale
        if "released" in self.events:
            run = min(step, self.coast_left)
            self.coast_left = max(0.0, self.coast_left - step)
        else:
            run = step if self.pressed else 0.0
        self.run_s += run
        while self.dispensed < POOL and self.dispensed < len(self.due) and run > 0 and self.run_s >= self.due[self.dispensed]:
            self.dispensed += 1
            self.mark("first_cube")
            self.event_log.append(dict(time=self.time, type="cube_dispensed", n=self.dispensed))
        if s["glass_dropped"]:
            self.failure("glass_dropped")
        if s["glass_tilt_deg"] > self.spec.max_tilt_deg:
            self.failure("glass_tipped")
        if s["cubes_outside"] > 0:
            self.failure("cube_missed_glass")
        if "released" in self.events and self.coast_left <= 0:
            down = s["glass_on_counter_released"] and not s["cubes_moving"]
            if self.settle_hold.step(down, dt) >= self.spec.settle_s:
                self.final_count = int(s["cubes_in_glass"])
                self.final_level = float(s["ice_level"])
                self.mark("level_final", ice_level=self.final_level, cubes_in_glass=self.final_count)
        self.done = bool(self.physical_failures) or "level_final" in self.events

    def cues(self):
        return {"cubes_dispensed": self.dispensed, "auger_running": bool(self.auger_running),
                "ice_level": float(self.signals["ice_level"]), "cubes_in_glass": int(self.signals["cubes_in_glass"])}

    def quality(self):
        return band_quality(self.final_level, *self.spec.level_band, *self.spec.level_zero)

    @property
    def physical_success(self):
        return not self.physical_failures and "level_final" in self.events

    @property
    def temporal_success(self):
        lo, hi = self.spec.level_band
        return self.final_level is not None and lo <= self.final_level <= hi

    def task_metrics(self):
        t0 = self.events.get("paddle_pressed", 0.0)  # one press: the auger has run since then
        due_s = [t0 + d / self.spec.time_scale for d in self.due]  # cube delivery times if the press is held
        return {"release_time": self.events.get("released"), "cubes_dispensed": self.dispensed,
                "final_ice_level": self.final_level, "final_cubes_in_glass": self.final_count,
                "delivery_times_if_held_s": due_s[:10], "quality": self.quality()}


TASK = IceDispenser
