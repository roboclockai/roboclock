"""A3: flip one pancake with the turner when its underside is golden."""

import math
from dataclasses import dataclass

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task, band_quality


@dataclass(frozen=True)
class A3Spec(X5Spec):
    max_time: float = 150.0
    tau: float = 65.0
    t0: float = 20.0  # first-side age at t = 0
    band: tuple = (0.55, 0.75)
    zero: tuple = (0.35, 0.90)
    flip_angle_deg: float = 150.0
    release_dwell: float = 1.0

    # Per-seed placement (sim.py Oracle.setup): the turner on the counter strip in front of the cooktop, about its neck.
    # The oracle's flip is sensitive to small changes (the blade pushes the pancake up the pan wall until it wedges
    # under it, and the pancake lands about 2 R past where it then stands, within ~1 cm of the in-pan limit). Measured
    # through roboclock eval (float32 actions), 2026-11-20/21:
    # - turner +-2 cm / y -1.5..+0.4 cm / +-6 deg: A3 medium seed 3 landed on the wall; +1.3 cm toward the cooktop
    #   (+y) the lever failed outright (seeds 3, 31);
    # - pancake moved 0.8 cm off its poured spot: it landed past the rim; turned (any yaw, even in 15 deg steps of its
    #   24-sided collider) the flip failed at easy seeds 1 and 5. The pancake therefore stays as poured;
    # - a right-hand backstop that removed the push threw or missed the pancake (not kept).
    # - half that (+-1 cm / -0.75..+0.2 cm / +-3 deg): seed 5 (+0.59, +0.16 cm, +2.5 deg) failed at every tier, the blade
    #   pushing the pancake 10 cm before it jammed (the nominal layout passes that seed).
    # Range: a quarter of the first one, y one-sided.
    LAYOUT_JITTER = {"turner": (0.005, (-0.004, 0.001), 1.5)}

    @classmethod
    def hidden_ranges(cls):
        return {"tau": (50.0, 80.0), "t0": (15.0, 25.0)}


class PancakeFlip(X5Task):
    task_id, name, category = "A3", "Flip one pancake at the right moment", "temporal_window"
    Spec = A3Spec
    # The only cue of the browning is the underside's colour (the top texture does not change), and the turner
    # must be withdrawn for the flip to count (sim.py turner_clear): the instruction says both.
    instruction = ("A pancake is cooking in the frying pan. Flip it with the turner when its underside is golden, "
                   "and leave it cooking in the pan. The underside turns from pale to golden to dark as it cooks; "
                   "it cannot be seen from above, only where the pancake's edge is lifted. The flip counts once the "
                   "pancake lies turned over, wholly inside the pan and at rest, with the turner off it and out of "
                   "the pan: the blade's tip 3 cm outside the rim, or 12 cm above the pan's floor.")
    SIGNALS = {
        "pancake_on_pan": ("pancake underside in contact with the pan floor", True),
        "pancake_rotation_deg": ("accumulated rotation about a horizontal axis since t = 0", 0.0),
        "pancake_in_pan_released": ("pancake fully inside the pan, not touched, settled", True),
        "turner_clear": ("turner withdrawn from the pan area", True),
        "dropped": ("pancake left the pan", False),
        "pan_moved": ("pan moved off the burner", False),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.contact_time = spec.t0 / spec.time_scale
        self.first_side = None
        self.release_hold = Hold()

    @property
    def doneness(self):
        return 1 - math.exp(-self.contact_time * self.spec.time_scale / self.spec.tau)

    def process(self, dt):
        s = self.signals
        if self.first_side is None:
            if s["pancake_on_pan"]:
                self.contact_time += dt
            if s["pancake_rotation_deg"] >= self.spec.flip_angle_deg:
                self.first_side = self.doneness
                self.mark("flip", first_side=self.first_side)
        if s["dropped"]:
            self.failure("pancake_dropped")
        if s["pan_moved"]:
            self.failure("pan_moved")
        ok = "flip" in self.events and s["pancake_in_pan_released"] and s["turner_clear"]
        if self.release_hold.step(ok, dt) >= self.spec.release_dwell:
            self.mark("settled_after_flip")
        self.done = self.physical_success or bool(self.physical_failures)

    def quality(self):
        return band_quality(self.first_side, *self.spec.band, *self.spec.zero)

    def cues(self):
        # Top-surface bubbles track cooking progress; the underside is not directly visible.
        age = self.contact_time * self.spec.time_scale
        bubbles = min(1.0, max(0.0, (age - 0.4 * self.spec.tau) / (0.9 * self.spec.tau)))
        return {"bottom_doneness": self.doneness, "top_bubble_state": bubbles}

    @property
    def physical_success(self):
        return not self.physical_failures and "settled_after_flip" in self.events

    @property
    def temporal_success(self):
        return self.quality() == 1.0

    def task_metrics(self):
        return {"flip_time": self.events.get("flip"), "first_side_doneness": self.first_side, "quality": self.quality()}


TASK = PancakeFlip
