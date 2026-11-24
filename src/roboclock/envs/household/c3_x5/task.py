"""C3: cutlery organization. Sort the mixed spoons, forks and knives into their drawer compartments.

C-class judging (score_version 2, docs/c-class-timing.md): the six loose utensils (two of each kind) are the Q = 6
jobs. The episode runs once per seed to the horizon H = 5 t_o (roboclock.evaluation.throughput) and is scored at
every time tier from t_Q, the first time all six were sorted at once. Difficulty changes nothing in the episode;
it only picks the tier that fills temporal_success.
"""

from dataclasses import dataclass, replace

from roboclock.envs.household.x5.task_base import X5Spec, X5Task
from roboclock.evaluation import throughput as TP

ORACLE_S = TP.oracle_time("household", "C3")  # t_o, frozen with its source in roboclock.evaluation.throughput

# Utensil names (here rather than in scene.py, which needs pxr; scene.py and sim.py import them).
KINDS = ("spoon", "fork", "knife")
LOOSE_PER_KIND = 2  # on the table: two of each kind, Q = 6 (three of each until 2026-11-09)


def name(kind, k):
    return f"{kind.title()}{k+1}"


LOOSE = tuple(name(kd, k) for kd in KINDS for k in range(LOOSE_PER_KIND))  # Spoon1-2, Fork1-2, Knife1-2
REFERENCES = tuple(name(kd, LOOSE_PER_KIND) for kd in KINDS)  # Spoon3, Fork3, Knife3: one in each compartment


@dataclass(frozen=True)
class C3Spec(X5Spec):
    max_time: float = TP.horizon(ORACLE_S)  # H = 5 t_o at every difficulty
    # The table layout and the compartment order are hidden and drawn from the seed (c3_x5/scene.py:layout).

    @classmethod
    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        """Identical episode at every difficulty: no process scaling (time_scale 1) and the same horizon."""
        spec = super().sample(task_id, difficulty, seed, **overrides)
        return spec if "time_scale" in overrides else replace(spec, time_scale=1.0)


class CutleryOrganization(X5Task):
    task_id, name, category = "C3", "Cutlery organization: sort the utensils into the drawer", "throughput"
    Spec = C3Spec
    # States when a utensil counts (sim.py: at rest, released, its middle below the wall tops) and what fails.
    # No time is stated (docs/c-class-timing.md).
    instruction = ("Spoons, forks and knives are mixed on the table. Put each one into its compartment of the open "
                   "drawer (each compartment already holds one of its kind). A utensil counts while it lies at "
                   "rest, released, inside the compartment of its kind and below the tops of its walls: lying "
                   "across a wall does not count. At the end none may be in a wrong compartment or off the counter, "
                   "and the three utensils already in the drawer must stay in their compartments. Sort all of them "
                   "as quickly as possible.")
    SIGNALS = {
        "correct": (f"loose utensils resting, released, in the compartment of their kind (0-{len(LOOSE)})", 0),
        "wrong": (f"loose utensils resting, released, in another compartment (0-{len(LOOSE)})", 0),
        "placed": ("names of the loose utensils currently counted as correct", ()),
        "dropped": (f"loose utensils that fell off the counter (0-{len(LOOSE)})", 0),
        "references_moved": ("a reference utensil left its compartment", False),
    }

    def __init__(self, spec):
        super().__init__(spec)
        self.best = 0
        self.last_job = None    # last time a utensil was sorted for the first time, or more were sorted at once
        self.all_hold = 0.0     # seconds all utensils have stayed sorted (early end)

    def process(self, dt):
        s = self.signals
        for n in s["placed"]:
            if "sorted_"+n not in self.events:
                self.last_job = self.mark("sorted_"+n)
        if s["correct"] > self.best:
            self.best, self.last_job = s["correct"], self.time
        if s["correct"] >= len(LOOSE):
            self.mark("all_sorted")  # t_Q: the first time all six were sorted at once
        if s["dropped"]:
            self.failure("utensil_dropped")
        if s["references_moved"]:
            self.failure("reference_moved")
        # Every utensil sorted counts only while it rests released in its compartment, so this is "all done and
        # valid"; a dropped utensil or a moved reference does not end the episode (as before).
        self.all_hold = self.all_hold + dt if s["correct"] >= len(LOOSE) and not s["wrong"] else 0.0
        if self.all_hold >= TP.SETTLE_S:
            self.end("all_done")
        elif TP.stalled(self.time, self.last_job, ORACLE_S, steps=len(LOOSE)):
            self.end("stall")

    @property
    def physical_success(self):
        # All six in the compartment of their kind at the end, none in a wrong one, nothing dropped, references in
        # place. Time plays no part.
        s = self.signals
        return not self.physical_failures and s["correct"] >= len(LOOSE) and s["wrong"] == 0

    @property
    def temporal_success(self):
        return self.judged()["temporal_success"]

    def job_times(self):
        """Per loose utensil (Spoon1, Spoon2, Fork1, ...): the first time it was counted as sorted, or None."""
        return [self.events.get("sorted_"+n) for n in LOOSE]

    def judged(self):
        return TP.judge(self.job_times(), len(LOOSE), self.physical_success, ORACLE_S, self.spec.difficulty,
                        t_q=self.events.get("all_sorted"), misplaced=self.signals["wrong"])

    def task_metrics(self):
        s = self.signals
        times = sorted(t for k, t in self.events.items() if k.startswith("sorted_"))
        return {"correct": s["correct"], "wrong": s["wrong"], "dropped": s["dropped"],
                "organized_score": s["correct"]-s["wrong"], "loose_utensils": len(LOOSE),
                "utensils_per_minute": 60*s["correct"]/max(self.time, 1e-9),
                "first_sorted_times": times, **self.judged(), "termination_reason": self.termination_reason}


TASK = CutleryOrganization
