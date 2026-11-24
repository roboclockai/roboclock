"""A1: sweeten a glass of cold tea: pour sugar from an open jar, stop in the gram band, then stir two turns.

The band is the tier: 10-30 g easy, 10-25 g medium, 10-15 g hard (chosen by the user, 2026-11-19); the pour itself
is the same at every tier. The only cue is what can be seen: the sugar sinks and piles up in a white mound on the
bottom of the glass (sim.py MOUND), the stream while it runs and the sugar left in the glass jar. Sugar runs once
the jar tilts past an onset angle that depends on how full it is, at up to MAX_FLOW_G_S, so the hard band lasts at
least 5 g / (4 g/s x 1.15) = 1.09 s at the fastest pour (the 1 s floor of the time-limit rule).
"""

import math
from dataclasses import dataclass

import numpy as np

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task, band_quality

PINCH_N, PINCH_SLIP, PINCH_DWELL = 0.04, 0.10, 0.065  # N on each jaw, m/s of the tool in the hand, s
BANDS_G = {"easy": (10.0, 30.0), "medium": (10.0, 25.0), "hard": (10.0, 15.0)}
SUGAR_BULK_G_ML = 0.85    # granulated sugar, loose
REPOSE_DEG = 33.0         # mound slope (granulated sugar settled under water)
GLASS_R_IN = 0.0385       # m, inner radius of the glass (sim.py CUP_R)
JAR_CAPACITY_G = 94.0     # jar body (inner r 24.9 mm, 57 mm high) full of loose sugar


def mound_radius(grams):
    """Radius (m) of the cone a sugar mass forms on the glass bottom (capped at the glass wall)."""
    volume = max(0.0, grams) / SUGAR_BULK_G_ML * 1e-6
    r = (3 * volume / (math.pi * math.tan(math.radians(REPOSE_DEG)))) ** (1 / 3)
    return min(r, GLASS_R_IN - 0.0005)


class Pinch:
    """Whether one gripper holds one tool: both jaws press on it (more than PINCH_N each) and it moves slower
    than PINCH_SLIP in the hand's frame, for longer than PINCH_DWELL.

    The oracle's own test in sim.py (left hand on the jar, right hand on the spoon), kept here for the other
    hand of each tool: the judged signals (``jar_held``, the stir count) must not depend on which arm holds
    which tool."""

    def __init__(self):
        self.dwell, self.offset = 0.0, None

    def step(self, jaw_forces, offset, dt):
        """``jaw_forces``: the two jaws' contact forces on the tool (N); ``offset``: the tool's position in the
        hand's frame (m). True while the tool is held."""
        offset = np.asarray(offset, dtype=float)
        slip = 0.0 if self.offset is None else float(np.linalg.norm(offset - self.offset) / dt)
        self.offset = offset
        self.dwell = self.dwell + dt if min(jaw_forces) > PINCH_N and slip < PINCH_SLIP else 0.0
        return self.dwell > PINCH_DWELL


@dataclass(frozen=True)
class A1Spec(X5Spec):
    max_time: float = 90.0
    band: tuple = BANDS_G["medium"]          # g of sugar in the glass; sample() sets this tier's band
    jar_sugar_g: float = 75.0                # sugar in the jar at the start
    flow_coefficient: float = 1.0            # multiplies the nominal tilt-flow curve
    max_flow_g_s: float = 4.0
    flow_span_deg: float = 35.0              # tilt past the onset at which the flow is full
    tail_tau_s: float = 0.12                 # sugar in the air lands with this time constant
    spill_limit_g: float = 2.0
    required_turns: float = 2.0
    release_dwell: float = 1.0
    max_glass_shift: float = 0.02

    # Per-seed placement (sim.py Oracle.setup): the sugar jar, and the wire basket with the spoon in it as one unit.
    # The glass of tea is a fixed fixture (its colliders, the tea and the sugar cues are authored at CUP).
    LAYOUT_JITTER = {"jar": (0.02, 0.02, 10.0), "spoon_basket": (0.015, 0.015, 5.0)}

    @property
    def zero(self):
        """Quality falls linearly to 0 at 5 g and at 10 g above the band."""
        return (5.0, self.band[1] + 10.0)

    @classmethod
    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        overrides.setdefault("band", BANDS_G.get(difficulty, BANDS_G["medium"]))
        return super().sample(task_id, difficulty, seed, **overrides)

    @classmethod
    def hidden_ranges(cls):
        return {"jar_sugar_g": (60.0, 85.0), "flow_coefficient": (0.85, 1.15)}


class SugarPour(X5Task):
    task_id, name, category = "A1", "Sweeten a glass of tea with sugar from a jar and stir it", "temporal_window"
    Spec = A1Spec
    # States what is judged and cannot be seen: the gram band with size anchors for the mound (task.mound_radius), the
    # flow law in outline (onset tilt, maximum rate), the spill rule and its limit, how a stir turn counts, the releases.
    instruction = ("Add 10 to 25 g of sugar from the open jar to the glass of tea. Sugar runs once the jar tilts past "
                   "40-60 degrees (later when emptier), faster with more tilt (max 4 g/s), and settles as a white "
                   "mound: 10 g is 5.2 cm across, 25 g 7.0 cm. 2 g spilled outside the glass fails. Set the jar down "
                   "upright and let go. Then stir two full clockwise turns (seen from above), the spoon's whole head "
                   "under the surface, and put the spoon back in its basket, released. No stirring until the jar is "
                   "down.")
    SIGNALS = {
        "jar_tilt_deg": ("jar axis tilt from upright", 0.0),
        "mouth_over_glass": ("the stream from the jar's lowest mouth point falls inside the glass mouth", False),
        "jar_upright_released": ("jar released, upright and supported on the counter", True),
        "spoon_in_cup": ("spoon head inside the glass, under the tea surface", False),
        "stir_turns": ("net clockwise turns of the spoon head about the glass axis", 0.0),
        "spoon_in_basket_released": ("spoon released, supported, low speed in its basket", False),
        "glass_displacement": ("glass displacement from its initial pose (m)", 0.0),
        "dropped": ("jar, spoon or glass dropped / tipped", False),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.jar_g = spec.jar_sugar_g
        self.glass_g = self.spill_g = 0.0
        self.air_glass_g = self.air_spill_g = 0.0
        self.flow = 0.0
        self.sugar_at_set_down = None
        self.jar_hold, self.spoon_hold = Hold(), Hold()

    def gen_instruction(self):
        lo, hi = self.spec.band
        if (lo, hi) == BANDS_G["medium"]:
            return self.instruction
        return (self.instruction.replace("Add 10 to 25 g", f"Add {lo:g} to {hi:g} g")
                .replace("25 g 7.0 cm", f"{hi:g} g {200 * mound_radius(hi):.1f} cm"))

    def onset_deg(self):
        """Tilt at which sugar starts to run: 30 deg for a full jar, 110 deg for an empty one."""
        return 30.0 + 80.0 * (1.0 - self.jar_g / JAR_CAPACITY_G)

    def process(self, dt):
        s, sp = self.signals, self.spec
        excess = s["jar_tilt_deg"] - self.onset_deg()
        rate = 0.0
        if excess > 0 and self.jar_g > 0 and not s["jar_upright_released"]:
            rate = sp.max_flow_g_s * sp.flow_coefficient * min(1.0, excess / sp.flow_span_deg) ** 1.5
        out = min(self.jar_g, rate * dt)
        self.jar_g -= out
        self.flow = out / dt
        if out > 0:
            self.mark("pour_started")
            if s["mouth_over_glass"]:
                self.air_glass_g += out
            else:
                self.air_spill_g += out
        land = 1.0 - math.exp(-dt / sp.tail_tau_s)
        g, x = self.air_glass_g * land, self.air_spill_g * land
        self.air_glass_g -= g
        self.air_spill_g -= x
        self.glass_g += g
        self.spill_g += x
        if self.spill_g >= sp.spill_limit_g:
            self.failure("sugar_spilled")
        if "pour_started" in self.events and s["jar_upright_released"]:
            if self.jar_hold.step(True, dt) >= sp.release_dwell and "jar_set_down" not in self.events:
                self.sugar_at_set_down = self.glass_g
                self.mark("jar_set_down", sugar_g=self.glass_g)
        else:
            self.jar_hold.step(False, dt)
        if s["spoon_in_cup"] and "jar_set_down" not in self.events:
            self.violation("stir_before_jar_set_down")
        if s["dropped"]:
            self.failure("dropped")
        if s["glass_displacement"] > sp.max_glass_shift:
            self.failure("glass_displaced")
        if s["stir_turns"] >= sp.required_turns:
            self.mark("stirred")
        if "stirred" in self.events and self.spoon_hold.step(s["spoon_in_basket_released"], dt) >= sp.release_dwell:
            self.mark("spoon_returned")
        self.done = self.physical_success or bool(self.physical_failures)

    def quality(self):
        return band_quality(self.sugar_at_set_down, *self.spec.band, *self.spec.zero)

    def cues(self):
        # undissolved: stirring dissolves the mound, fully after the required turns (appearance only)
        stirred = min(1.0, float(self.signals["stir_turns"]) / self.spec.required_turns)
        return {"sugar_g": self.glass_g, "undissolved_g": self.glass_g * (1.0 - stirred), "jar_g": self.jar_g,
                "flow_g_s": self.flow, "in_air_g": self.air_glass_g + self.air_spill_g}

    @property
    def physical_success(self):
        return (not self.physical_failures and "stir_before_jar_set_down" not in self.violations
                and all(k in self.events for k in ("pour_started", "jar_set_down", "stirred", "spoon_returned")))

    @property
    def temporal_success(self):
        return self.quality() == 1.0

    def task_metrics(self):
        return {"pour_start_time": self.events.get("pour_started"), "set_down_time": self.events.get("jar_set_down"),
                "sugar_g": self.sugar_at_set_down, "spilled_g": self.spill_g, "target_band_g": list(self.spec.band),
                "quality": self.quality()}


TASK = SugarPour
