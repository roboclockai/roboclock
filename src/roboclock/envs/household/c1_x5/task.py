"""C1: serve the ordered meals at the pass (serving relay, single-plate pass).

R3 (2026-11-06) replaces the four-bowl microwave relay (R2.2, 600 process seconds). Its episodes ran
800 / 600 / 400 s, a quarter of the simulated time of the whole household suite; R3 keeps the bowls,
the staging mat and the serving board, drops the microwave and runs 60 process seconds.

Layout (scene.py): one meal is ready on the staging mat between the arms; the other three stand on
the counter in front of the left arm. The serving board, in front of the right arm, holds one bowl.
A waiter takes the bowl on the board at each pickup time, so the next one can be set down only after
the previous pickup. Neither arm reaches both the counter slots and the board: the left arm brings
bowls to the mat, the right arm takes them on to the board.

Per seed: the order in which the meals are wanted (the meal on the mat is always the first), which
meal stands in which counter slot, and small pose jitter. Seed 0 is the nominal layout.

C-class judging (score_version 2, docs/c-class-timing.md), a stream task: each order is a job whose
deadline is its pickup time and whose time is when its meal was set down on the board. Physical success
needs every ordered meal served by the end of the episode, late or not; temporal success needs each by its
pickup (roboclock.evaluation.throughput.judge_stream). A pickup that finds the board without its meal leaves
the order open: the waiter takes that meal as soon as it has rested on the board for LATE_TAKE_S, which
frees the board for the next one. The episode runs to 1.25 x the last pickup, or ends SETTLE_S after the
last pickup once every ordered meal is served.

Schedule (since 2026-11-09, tier factors since 2026-11-16): the same four orders at every difficulty. The
reference pickups (HARD_PICKUPS_S) are the slowest reference oracle's set-down time of each order; every tier
stretches that whole schedule, first pickup and intervals alike, by TP.MULTIPLIER (hard 1.25, i.e. 20 % headroom,
medium sqrt(5), easy 4). Nothing else scales with the difficulty (time_scale 1).
"""

from dataclasses import dataclass, replace

import numpy as np

from roboclock.envs.household.x5.task_base import X5Spec, X5Task
from roboclock.evaluation import throughput as TP

MEALS = ("curry", "rice", "soup", "pasta")  # MealBowl1..4 (food textures in scene.py)
LOCATIONS = ("counter", "held", "mat", "board", "taken", "dropped")

# Reference schedule (wall s; hard is 1.25 x it), frozen: a first pickup and a fixed interval, each rounded up to 0.01 s, the tightest such
# schedule the slowest oracle meets. Measured 2026-11-09 on c-class-unify (3d002b4 + the score_version 2 change),
# seeds 0-6 and 50-56: 14 oracle runs with the board freed 1 s after each set-down, so that no pickup held the oracle
# back, then 14 at this schedule; the 2026-11-16 matrix (seeds 0-9 at every tier) met it with >= 0.006 s; at the released 1.25 x hard schedule seed 3 stalls placing the third bowl (oracle manipulation, not timing). Slowest set-down per order over the 28 runs: 10.524 s (every seed; the first meal
# starts on the mat), 23.484 s (seed 2 at hard), 36.344 s and 49.204 s (seed 2, board freed). The second order
# binds: 10.53 + 12.96 >= 23.484. Orders 3 and 4 keep 0.11 and 0.21 s, which covers the shifts of up to 0.1 s in
# later set-downs when a pickup moves. At hard the next meal arrives beside the board about 8.8 s or more after
# each pickup, so the board never holds the oracle back. Never derive these at run time: retuning the oracle must
# not move the benchmark (re-measure deliberately with scripts/household/c1_oracle_free_run.py).
HARD_FIRST_PICKUP_S, HARD_INTERVAL_S = 10.53, 12.96
HARD_PICKUPS_S = tuple(round(HARD_FIRST_PICKUP_S + k*HARD_INTERVAL_S, 2) for k in range(4))  # 10.53 ... 49.41
LATE_TAKE_S = 1.0  # an overdue meal is taken once it has rested on the board this long
# Judged rules a policy cannot see, appended to the instruction: when a meal counts, what the waiter takes, and the
# tilt limit (sim.py reports a bowl past 40 deg as dropped whether or not a gripper holds it: its meal is spilled).
RULES = (" A meal is served once its bowl stands upright and released on the board; the waiter takes whatever bowl "
         "is there, and a late meal after it has stood there 1 second. A bowl tilted more than 40 degrees, even in "
         "a gripper, or off the counter counts as dropped and fails the task.")

# Counter slots of the three waiting meals (bowl centre x, y in m; body yaw in deg, handle B facing
# the left arm's base). Nearest the mat first.
SLOTS = ((.20, .16, 154.3), (.03, .22, 176.4), (-.14, .18, 197.8))
JITTER = (.012, .012, 8.)  # per-seed uniform +- (m, m, deg) on the counter slots


def layout(seed):
    """(order, slots): meal indices in pickup order (order[0] starts on the mat) and, per meal, its
    counter pose (x, y, yaw) or None for the meal on the mat. Seed 0 is nominal."""
    if not seed:
        order = (0, 1, 2, 3)
        slot_of = {1: 0, 2: 1, 3: 2}
        jit = np.zeros((3, 3))
    else:
        rng = np.random.default_rng([int(seed), 101])
        order = tuple(int(i) for i in rng.permutation(4))
        waiting = list(order[1:])
        perm = rng.permutation(3)
        slot_of = {meal: int(perm[k]) for k, meal in enumerate(waiting)}
        jit = rng.uniform(-1, 1, (3, 3))*JITTER
    slots = [None]*4
    for meal, k in slot_of.items():
        x, y, yaw = np.add(SLOTS[k], jit[k])
        slots[meal] = (float(x), float(y), float(yaw))
    return order, tuple(slots)


def pickups(difficulty):
    """Wall seconds of the four pickups at a difficulty: TP.MULTIPLIER[difficulty] x HARD_PICKUPS_S."""
    if difficulty not in TP.MULTIPLIER:
        raise ValueError(f"Unknown difficulty {difficulty!r}")
    return tuple(round(TP.MULTIPLIER[difficulty]*p, 6) for p in HARD_PICKUPS_S)


@dataclass(frozen=True)
class C1Spec(X5Spec):
    max_time: float = round(TP.HORIZON_FACTOR*pickups("medium")[-1], 6)
    pickups_s: tuple = pickups("medium")  # wall seconds (before 2026-11-09: process seconds / time_scale)
    order: tuple = (0, 1, 2, 3)   # meal index wanted at each pickup (len = number of orders)
    slots: tuple = (None, (.20, .16, 154.3), (.03, .22, 176.4), (-.14, .18, 197.8))

    @classmethod
    def sample(cls, task_id, difficulty="medium", seed=0, **overrides):
        """Four orders at every difficulty, picked up at pickups(difficulty); no process scaling (time_scale 1).
        max_time = H = 1.25 x the last pickup: easy 247.05 s, medium 123.525 s, hard 61.7625 s."""
        order, slots = layout(seed)
        defaults = dict(order=order, slots=slots, pickups_s=pickups(difficulty), time_scale=1.0)
        spec = super().sample(task_id, difficulty, seed, **{**defaults, **overrides})
        spec = replace(spec, order=tuple(spec.order), slots=tuple(spec.slots), pickups_s=tuple(spec.pickups_s))
        if "max_time" not in overrides:
            spec = replace(spec, max_time=round(TP.HORIZON_FACTOR*spec.pickup_times[-1], 6))
        return spec

    @property
    def pickup_times(self):
        """Wall (simulation) seconds of the pickups: pickups_s (the deadlines of the orders)."""
        return tuple(self.pickups_s)


def _join(words):
    return words[0] if len(words) == 1 else ", ".join(words[:-1])+" and "+words[-1]


def _seconds(t):
    """A pickup time as the instruction states it: to 0.01 s, without trailing zeros (21.06, 42.1, 50)."""
    return f"{t:.2f}".rstrip("0").rstrip(".")


class PassService(X5Task):
    task_id, name, category = "C1", "Serve the ordered meals at the pass", "throughput"
    Spec = C1Spec
    instruction = ("Serve the meals on the serving board in the order given. The board holds one bowl; a waiter "
                   "takes it at each pickup time. Bring the waiting bowls to the mat, then on to the board." + RULES)
    SIGNALS = {
        "bowl_location": ("per-bowl location, one of " + "/".join(LOCATIONS), ("mat",)+("counter",)*3),
        "board_bowl": ("index of the bowl resting released and upright on the board; -1 when none", -1),
    }

    def gen_instruction(self):
        s = self.spec
        meals = _join([MEALS[i] for i in s.order])
        times = _join([_seconds(t) for t in s.pickup_times])
        return (f"Serve the {meals} on the serving board, in that order. The board holds one bowl; a waiter takes "
                f"it at {times} seconds. Bring the waiting bowls to the mat, then on to the board." + RULES)

    def __init__(self, spec):
        super().__init__(spec)
        # Time each bowl was set down on the board (resting, released, upright). It stays while the bowl rests
        # there so (``board_bowl``) or once the waiter has taken it; a bowl lifted off the board, or tipped past
        # the upright limit, before that loses it.
        self.served = [None]*4
        self.taken = [None]*4        # order index the waiter took each bowl for
        self.qualified = [False]*len(spec.order)
        self.board_at_pickup = [None]*len(spec.order)
        # The waiter's collections in order, for the simulation (sim.py moves each bowl off the counter):
        # dict(pickup=order index, bowl=bowl index or None for an empty board, late=bool, time=s).
        self.waiter = []
        self.resting = (-1, 0.0)     # (bowl on the board, since when)
        self.end_hold = 0.0

    def process(self, dt):
        s, sp = self.signals, self.spec
        loc = list(s["bowl_location"])
        board = int(s["board_bowl"])
        for b in range(4):
            if loc[b] == "dropped":
                self.failure(f"bowl{b+1}_dropped")
            if self.served[b] is not None and self.taken[b] is None and board != b:
                # lifted off the board again, or no longer resting upright and released there, before the waiter
                # took it (the waiter only takes the board bowl, so a tipped bowl would never be delivered)
                self.served[b] = None
        if board >= 0 and self.taken[board] is not None:
            board = -1  # already taken (the simulation moves it off on its next tick)
        if board != self.resting[0]:
            self.resting = (board, self.time)
        if board >= 0 and self.served[board] is None:
            self.served[board] = self.time
            self.mark(f"bowl{board+1}_served", meal=MEALS[board])
        for i, wanted in enumerate(sp.order):
            if self.time >= sp.pickup_times[i] and f"pickup{i+1}" not in self.events:
                on_board = board if board >= 0 else None
                self.board_at_pickup[i] = on_board
                self.qualified[i] = on_board == wanted
                if on_board is not None:
                    self.taken[on_board] = i
                    board = -1
                    if on_board != wanted:
                        self.violation("wrong_meal_served")
                self.waiter.append(dict(pickup=i, bowl=on_board, late=False, time=self.time))
                self.mark(f"pickup{i+1}", wanted=MEALS[wanted], served=None if on_board is None else MEALS[on_board],
                          qualified=self.qualified[i])
        # Overdue orders: once the pickup has passed without its meal, the waiter takes that meal as soon as it has
        # rested on the board for LATE_TAKE_S. It counts as served from when it was set down (late), and the board
        # is free again for the next order.
        for i, wanted in enumerate(sp.order):
            if (board == wanted and self.taken[wanted] is None and f"pickup{i+1}" in self.events
                    and self.time - self.resting[1] >= LATE_TAKE_S - 1e-9):
                self.taken[wanted] = i
                board = -1
                self.waiter.append(dict(pickup=i, bowl=wanted, late=True, time=self.time))
                self.mark(f"late_pickup{i+1}", meal=MEALS[wanted], served_s=self.served[wanted])
        # Early end: every order done (job_times) and the last pickup passed, for SETTLE_S.
        complete = self.time >= sp.pickup_times[-1] and all(t is not None for t in self.job_times())
        self.end_hold = self.end_hold + dt if complete else 0.0
        if self.end_hold >= TP.SETTLE_S:
            self.end("all_done")

    @property
    def physical_success(self):
        # Nothing dropped and every order done by the end of the episode, late service included (job_times): its
        # meal set down on the board and still there, or taken by the waiter for that order. A meal the waiter took
        # for another order is a wrong delivery and leaves its own order undone. When is temporal's concern.
        # (Before 2026-11-08 the served check was missing, so a robot that never moved passed.)
        return not self.physical_failures and all(t is not None for t in self.job_times())

    @property
    def temporal_success(self):
        return self.judged()["temporal_success"]

    def job_times(self):
        """Per order: when its meal was set down on the board, if the waiter took it for this order or it still
        rests there untaken; None otherwise (not served, or taken for another order)."""
        return [self.served[b] if self.taken[b] in (None, i) else None for i, b in enumerate(self.spec.order)]

    def judged(self):
        return TP.judge_stream(self.job_times(), self.spec.pickup_times, self.physical_success)

    def task_metrics(self):
        n = sum(self.qualified)
        order = self.spec.order
        served = self.job_times()
        return {"orders": len(order), "qualified_meals": n, "qualified_per_minute": 60*n/max(self.time, 1e-9),
                "served_s": served,
                "slack_s": [None if t is None else p-t for t, p in zip(served, self.spec.pickup_times)],
                "wrong_meals": sum(b is not None and b != w for b, w in zip(self.board_at_pickup, order)),
                "late_pickups": [w["pickup"]+1 for w in self.waiter if w["late"]],
                "order": [MEALS[i] for i in order],
                **self.judged(), "horizon_s": self.spec.max_time, "termination_reason": self.termination_reason}


TASK = PassService
