"""Independent physical release, rack legality and batch-throughput evaluation.

C-class judging (score_version 2, docs/c-class-timing.md): the 15 balls are the jobs; a job is
done when its ball is first accepted (lifted, released and settled in a rack slot). The rack
counts as finished (t_Q, ``completed_at``) when it is legal with all 15 accepted and both
hands have stayed clear of the slots for SETTLE_S, which also ends the episode (all_done).
Otherwise the episode ends at the stall stop or at the horizon H = 5 t_o. Physical success is
judged at the end without time terms: all 15 accepted, the rack legal, no failure.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.sports import c_class

BALLS = 15
# Judged distances the instruction states (``roboclock.envs.sports.tasks``); metres.
# A ball counts only after it was lifted this far above its caddy cell, and the rack is handed
# over once both tool points are this far from every slot.
LIFT_HEIGHT = 0.05
WITHDRAW_DISTANCE = 0.12


class Task:
    def __init__(self, b):
        self.b, self.c = b, b.config["rack"]
        self.difficulty = b.config.get("difficulty", "easy")
        self.done = False
        self.termination_reason = None
        self.events, self.failures = [], []
        self.grasp_since, self.offsets, self.grasped, self.lifted = {}, {}, {}, set()
        self.stable_since, self.accepted = {}, {}
        self.current, self.complete_since, self.completed_at = {}, None, None
        self.legal = False
        self.slots = np.asarray(self.c["slots"])

    def finish(self, reason):
        self.done, self.termination_reason = True, reason

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)
            self.events.append(dict(event="failure", reason=reason, time=self.b.time))

    def placement(self, name, o):
        p = np.asarray(o["position"])
        distances = np.linalg.norm(self.slots[:, :2] - p[:2], axis=1)
        slot = int(np.argmin(distances))
        supported = (
            sum(o["support_contacts"].get(k, 0.0) for k in ("PoolBed", "RackTemplate")) > 0.30
        )
        valid = (
            name in self.lifted
            and distances[slot] < self.c["placement_tolerance"]
            and abs(p[2] - self.slots[slot, 2]) < 0.003
            and max(o["forces"][:4]) < 0.05
            and supported
            and np.linalg.norm(o["velocity"][:2]) < 0.008
            and abs(o["velocity"][2]) < 0.03
        )
        return slot if valid else None

    def legality(self, state, occupants):
        if len(occupants) != 15 or len(set(occupants.values())) != 15:
            return False
        by_slot = {v: k for k, v in occupants.items()}
        if by_slot[4] != "ball08":
            return False
        corners = [int(by_slot[i][4:]) for i in (10, 14)]
        if not (min(corners) < 8 < max(corners)):
            return False
        for i, a in enumerate(self.slots):
            for j, other in enumerate(self.slots[:i]):
                if np.linalg.norm(a - other) < 2 * self.c["radius"] + 0.001:
                    p = np.asarray(state["objects"][by_slot[i]]["position"])
                    q = np.asarray(state["objects"][by_slot[j]]["position"])
                    gap = np.linalg.norm(p - q) - 2 * self.c["radius"]
                    if not -0.002 <= gap <= self.c["neighbor_gap_tolerance"]:
                        return False
        return True

    def update(self, s):
        if self.done:
            return
        now = s["time"]
        if now > self.c["horizon"] + 1e-8:
            self.finish("horizon")
            return
        if c_class.arm_collision(s):
            self.fail("arm_collision")
        white = s["objects"]["cue_ball"]
        if max(white["forces"][:4]) > 0.05:
            self.fail("cue_ball_touched")
        occupants = {}
        for name in self.c["sources"]:
            o = s["objects"][name]
            p = np.asarray(o["position"])
            if p[2] < 0.70:
                self.fail("ball_dropped_" + name)
            for side, start in (("left", 0), ("right", 2)):
                key = (name, side)
                hand = s["hands"][side]
                r = Rotation.from_quat(np.roll(hand["orientation"], -1)).as_matrix()
                offset = r.T @ (p - np.asarray(hand["position"]))
                if min(o["forces"][start : start + 2]) > 0.08:
                    if (
                        key not in self.grasp_since
                        or np.linalg.norm(offset - self.offsets[key]) > 0.004
                    ):
                        self.grasp_since[key], self.offsets[key] = now, offset.copy()
                    if (
                        now - self.grasp_since[key] >= self.c["grasp_duration"]
                        and name not in self.grasped
                    ):
                        self.grasped[name] = side
                        self.events.append(dict(event="grasp", ball=name, side=side, time=now))
                    if (
                        name in self.grasped
                        and p[2] > self.c["sources"][name][2] + LIFT_HEIGHT
                        and name not in self.lifted
                    ):
                        self.lifted.add(name)
                        self.events.append(dict(event="lift", ball=name, side=side, time=now))
                else:
                    self.grasp_since.pop(key, None)
            slot = self.placement(name, o)
            if slot is None:
                self.stable_since.pop(name, None)
            else:
                occupants[name] = slot
                self.stable_since.setdefault(name, now)
                if (
                    now - self.stable_since[name] >= self.c["settle_duration"]
                    and name not in self.accepted
                ):
                    self.accepted[name] = dict(time=now, slot=slot)
                    self.events.append(dict(event="placed", ball=name, slot=slot, time=now))
        self.current = occupants
        self.legal = self.legality(s, occupants) and len(self.accepted) == BALLS
        withdrawn = all(
            min(np.linalg.norm(np.asarray(h["position"]) - p) for p in self.slots)
            > WITHDRAW_DISTANCE
            for h in s["hands"].values()
        )
        if self.legal and withdrawn:
            self.complete_since = now if self.complete_since is None else self.complete_since
            if now - self.complete_since >= c_class.SETTLE_S:
                self.completed_at = now
                self.events.append(dict(event="rack_ready", time=now))
                self.finish("all_done")
                return
        else:
            self.complete_since = None
        if self.complete_since is None and c_class.stalled("C1", now, self.last_job_time()):
            self.finish("stall")  # a rack that is settling is never cut short
        elif now >= self.c["horizon"] - 1e-8:
            self.finish("horizon")

    def last_job_time(self):
        return max((a["time"] for a in self.accepted.values()), default=None)

    def job_times(self):
        """Acceptance time of each ball (ball01 ... ball15), None while not accepted."""
        return [
            self.accepted[name]["time"] if name in self.accepted else None
            for name in sorted(self.c["sources"])
        ]

    def display(self):
        return (
            f"{len(self.current):02d}/15 BALLS",
            "RACK READY" if self.completed_at else "8-BALL / PREPARE THE BREAK",
        )

    def result(self):
        racked = len(self.accepted) == BALLS and self.legal
        physical = racked and not self.failures
        judged = c_class.judge(
            "C1",
            self.job_times(),
            BALLS,
            physical,
            self.difficulty,
            start=0.0,
            t_q=self.completed_at,
            termination_reason=self.termination_reason,
        )
        failures = self.failures.copy()
        if self.done and not racked:
            failures.append("incomplete_rack_at_deadline")
        elif self.done and self.completed_at is None:
            failures.append("rack_not_handed_over")  # legal rack, hands never withdrew
        budget = judged["budget_s_by_tier"][self.difficulty]
        return dict(
            task="C1",
            variant="eight_ball_racking",
            primary_temporal_category="throughput",
            physical_success=physical,
            temporal_success=judged["temporal_success"],
            success=judged["joint_success"],
            unique_placements=len(self.accepted),
            final_valid_positions=len(self.current),
            final_rack_legal=self.legal,
            total_balls=BALLS,
            completed_at=self.completed_at,
            budget=budget,
            horizon=self.c["horizon"],
            balls_per_budget_minute=60 * len(self.current) / budget,
            required_balls_per_minute=60 * BALLS / budget,
            termination_reason=self.termination_reason,
            throughput=judged,
            events=self.events,
            failures=failures,
            accepted=self.accepted,
            final_occupancy=self.current,
            controller="privileged scripted bimanual feedback baseline",
        )
