"""C2: two arms pour cola (a particle fluid) into six glasses in a row, all six as quickly as possible.

C-class judging (score_version 2, docs/c-class-timing.md): the six glasses are the Q = 6 jobs. The episode runs
once per seed to the horizon H = 5 t_o (roboclock.evaluation.throughput) and is scored at every time tier from
t_Q, the first time all six were served with both bottles stood back released (``all_served``, as in the
frozen t_o). Difficulty changes nothing in the episode; it only picks the tier that fills temporal_success.
"""

from dataclasses import dataclass, replace

import numpy as np

from roboclock.envs.household.x5.task_base import Hold, X5Spec, X5Task
from roboclock.evaluation import throughput as TP

GLASSES, BOTTLES = 6, 2
ORACLE_S = TP.oracle_time("household", "C2")  # t_o, frozen with its source in roboclock.evaluation.throughput


@dataclass(frozen=True)
class C2Spec(X5Spec):
    max_time: float = TP.horizon(ORACLE_S)  # H = 5 t_o at every difficulty
    target_ml: float = 90.0         # white line on each glass
    band_ml: tuple = (80.0, 105.0)
    spill_limit_ml: float = 15.0
    settle_s: float = 1.0           # a glass counts once its level has held in the band this long with nothing pouring
    end_hold_s: float = 1.0         # all six served and both bottles released this long: all_served (t_Q, as in t_o)

    # Per-seed placement (sim.py Oracle.setup): each bottle (with the cola in it) and each tumbler. Translation only:
    # both are bodies of revolution. Tumblers +-1 cm: they stand 10 cm apart (4.2 cm gaps) and a pour clears the
    # neighbouring rims by 12 mm.
    LAYOUT_JITTER = {"bottle_left": (0.015, 0.015, 0.0), "bottle_right": (0.015, 0.015, 0.0),
                     **{f"glass{j}": (0.01, 0.01, 0.0) for j in range(1, 7)}}

    @classmethod
    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        """Identical episode at every difficulty: no process scaling (time_scale 1) and the same horizon."""
        spec = super().sample(task_id, difficulty, seed, **overrides)
        return spec if "time_scale" in overrides else replace(spec, time_scale=1.0)


class ColaPour(X5Task):
    task_id, name, category = "C2", "Pour cola into six glasses with two arms", "throughput"
    Spec = C2Spec
    # States the judged band around the line, the release of the bottles and the failures with their limits.
    # No time is stated (docs/c-class-timing.md).
    instruction = ("Six guests are waiting. Pour cola from the two opened bottles into the six glasses in the row, "
                   "each up to the white line, without spilling, and stand both bottles back on the counter, "
                   "released. A glass counts with 80 to 105 ml in it (the line marks 90 ml: from 5 mm below it to 8 "
                   "mm above); above that it cannot be corrected. Spilling 15 ml, moving a glass more than 3 cm or "
                   "tipping it, or dropping a bottle fails the task. Fill all six glasses as quickly as possible.")
    SIGNALS = {
        "glass_ml": ("per-glass cola volume, counted from the fluid particles inside it", (0.0,) * GLASSES),
        "bottle_ml": ("per-bottle cola volume left inside it", (0.0,) * BOTTLES),
        "spilled_ml": ("cola on or below the counter outside the glasses", 0.0),
        "air_ml": ("cola in the air (streams and splashes)", 0.0),
        "bottle_released_upright": ("per-bottle: released, upright and standing on the counter", (True, True)),
        "bottle_dropped": ("per-bottle: dropped or tipped over", (False, False)),
        "glass_upright": ("per-glass: upright and in its place", (True,) * GLASSES),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.served_t = [None] * GLASSES
        self.holds = [Hold() for _ in range(GLASSES)]
        self.end_hold = Hold()
        self.peak = [0.0] * GLASSES
        self.last_job = None        # last time a glass became served (stall stop)

    def in_band(self, j):
        lo, hi = self.spec.band_ml
        return lo <= self.signals["glass_ml"][j] <= hi and self.signals["glass_upright"][j]

    def bottles_down(self):
        return all(self.signals["bottle_released_upright"])

    def process(self, dt):
        s, sp = self.signals, self.spec
        quiet = s["air_ml"] < 0.5
        for j in range(GLASSES):
            self.peak[j] = max(self.peak[j], s["glass_ml"][j])
            if s["glass_ml"][j] > 0.5:
                self.mark("pour_started")
            held = self.holds[j].step(self.in_band(j) and quiet, dt)
            if self.served_t[j] is None and held >= sp.settle_s:
                self.served_t[j] = self.last_job = self.time
                self.event_log.append(dict(time=self.time, type=f"glass{j + 1}_served", glass_ml=s["glass_ml"][j]))
            if self.served_t[j] is not None and not self.in_band(j):
                self.served_t[j] = None  # topped up past the band or knocked
            if not s["glass_upright"][j]:
                self.failure(f"glass{j + 1}_knocked")
            if s["glass_ml"][j] > sp.band_ml[1]:
                # Recorded only: the cola cannot leave a glass that stays upright, so an overfilled glass is never
                # back in the band and fails the final check of physical success.
                self.mark(f"glass{j + 1}_overfilled", glass_ml=s["glass_ml"][j])
        for b in range(BOTTLES):
            if s["bottle_dropped"][b]:
                self.failure(f"bottle{b + 1}_dropped")
        if s["spilled_ml"] >= sp.spill_limit_ml:
            self.failure("spill")
        down = self.bottles_down() and "pour_started" in self.events
        held = self.end_hold.step(down and self.served_count() == GLASSES, dt)
        if held >= sp.end_hold_s:
            self.mark("all_served")  # t_Q, measured as the frozen t_o (finish_time)
        if self.physical_failures:
            self.end("failure")
        elif held >= TP.SETTLE_S:
            self.end("all_done")
        elif self.served_count() < GLASSES and TP.stalled(self.time, self.last_job, ORACLE_S, steps=GLASSES):
            # Once all six are served only the bottles are left to set down (part of t_Q), so no stall then.
            self.end("stall")

    def served_count(self):
        return sum(t is not None for t in self.served_t)

    def cues(self):
        return {}  # the cola itself is the visible cue (rendered fluid)

    @property
    def physical_success(self):
        # All six served (in the band, upright, settled) and both bottles stood back released, at the end; nothing
        # dropped, knocked or spilled. Time plays no part.
        return not self.physical_failures and self.served_count() == GLASSES and self.bottles_down()

    @property
    def temporal_success(self):
        return self.judged()["temporal_success"]

    def judged(self):
        return TP.judge(self.served_t, GLASSES, self.physical_success, ORACLE_S, self.spec.difficulty,
                        t_q=self.events.get("all_served"))

    def task_metrics(self):
        s, n = self.signals, self.served_count()
        return {"served_glasses": n, "served_per_minute": 60 * n / max(self.time, 1e-9),
                "glass_ml": list(s["glass_ml"]), "glass_peak_ml": list(self.peak), "glass_served_s": list(self.served_t),
                "overfilled_glasses": [j + 1 for j in range(GLASSES) if f"glass{j + 1}_overfilled" in self.events],
                "spilled_ml": s["spilled_ml"], "bottle_left_ml": list(s["bottle_ml"]), "finish_time": self.events.get("all_served"),
                "mean_fill_error_ml": float(np.mean([abs(v - self.spec.target_ml) for v in s["glass_ml"]])),
                **self.judged(), "termination_reason": self.termination_reason}


TASK = ColaPour
