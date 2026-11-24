"""Contact-derived physical outcomes and independent workflow timing predicates.

C2 speed cups is a C-class batch task (score_version 2, docs/c-class-timing.md): the jobs are
the ``cups_required`` cards, each done at its valid bell press after the preparation. The
episode ends SETTLE_S after the last required card (all_done), at the stall stop, or at the
horizon (preparation + 5 t_o). Physical success needs every card, no failure and no arm
collision; the time tiers are judged from the last card's time, counted from the end of
the preparation.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.sports import c_class

# Judged distances the instructions state (``roboclock.envs.sports.tasks``); metres.
# A3: a release closer than this to the hoop's axis is a dunk, and a release counts only
# with the ball above RELEASE_MIN_HEIGHT.
NO_DUNK_RADIUS = 0.12
RELEASE_MIN_HEIGHT = 0.90
# B2: guard is the glove (tool point) near GUARD_X, +/-GUARD_Y, GUARD_Z: back in guard
# within GUARD_RETURN, still in guard during HOLD within GUARD_HOLD, and the hand that does
# not strike within GUARD_OTHER at the hit.
GUARD_X, GUARD_Y, GUARD_Z = 0.35, 0.23, 1.06
GUARD_RETURN, GUARD_HOLD, GUARD_OTHER = 0.025, 0.035, 0.05
# C2: a cup counts within this of its place on the card (horizontal).
CUP_TOLERANCE = 0.009
# C2: the hand that rang the bell must lift above this before the bell can ring again.
BELL_CLEAR_HEIGHT = 0.83


def guard(side):
    """Guard position of a hand's glove (B2)."""
    return np.array([GUARD_X, GUARD_Y if side == "left" else -GUARD_Y, GUARD_Z])


# B2 pad trainer: nominal rest centre of each spring-mounted pad (x, +/-y, z) and how far in
# front of a pad's centre the glove's tool point is when its knuckles press the face. The
# whole trainer moves by the seeded ``workflow.pad_offset`` [dx, dy] (``sports/spatial.py``);
# guard stays fixed to the robot.
PAD_X, PAD_Y, PAD_Z = 0.60, 0.23, 1.06
STRIKE_DEPTH = 0.077


def pad_offset(w):
    """The trainer's seeded [dx, dy] (m) from the workflow config; zero without."""
    dx, dy = w.get("pad_offset", (0.0, 0.0))
    return float(dx), float(dy)


def pad_center(w, side):
    """Rest centre of a pad (B2): the scene, the evaluator and the oracle all use this."""
    dx, dy = pad_offset(w)
    return np.array([PAD_X + dx, (PAD_Y if side == "left" else -PAD_Y) + dy, PAD_Z])


def retained(o, hand_index=0):
    f = np.asarray(o["forces"])
    return bool(f[2 * hand_index] > 0.025 and f[2 * hand_index + 1] > 0.025)


def cup_valid(o, target, support_force, held):
    up = Rotation.from_quat(np.roll(o["orientation"], -1)).apply([0, 0, 1])
    return bool(
        np.linalg.norm(np.asarray(o["position"])[:2] - target[:2]) < CUP_TOLERANCE
        and abs(o["position"][2] - target[2]) < 0.006
        and up[2] > 0.985
        and np.linalg.norm(o["velocity"]) < 0.035
        and np.linalg.norm(o["angular_velocity"]) < 0.25
        and support_force > 0.015
        and not held
    )


class Task:
    def __init__(self, b):
        self.b = b
        self.c = b.config["workflow"]
        self.kind = b.kind
        self.time = 0.0
        self.events = []
        self.failures = []
        self.physical = False
        self.temporal = False
        self.done = False
        self.termination_reason = None
        self.previous = None
        self.held = False
        self.release = None
        self.release_candidate = None
        self.crossing = False
        self.basket = False
        self.card = 0
        self.card_times = []
        self.cups_placed = set()
        self.stable_since = None
        self.bell_down = False
        self.bell_hand = "right"
        self.hit_on = {"pad_left": False, "pad_right": False}
        self.hits = []
        self.guards = []
        self.pending_guard = {}
        self.bell_rest = (
            float(b.last_state["objects"]["bell"]["position"][2]) if self.kind == "C2" else None
        )
        self.arm_contact_peak = 0.0
        self.arm_collision_steps = 0
        from roboclock.envs.sports.arena.boxing import schedule

        self.commands = schedule(self.c, b.config["seed"])
        self.reactions = []
        self.cue_events = set()
        self.cup_targets = b.cup_targets if self.kind == "C2" else {}

    def event(self, kind, **fields):
        self.events.append(dict(time=self.time, kind=kind, **fields))

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)
            self.event("failure", reason=reason)

    def update(self, s):
        if self.done and self.kind == "C2":
            return  # the C2 verdict is frozen at its end
        self.time = s["time"]
        peak = s.get("arm_contact_peak", 0.0)
        self.arm_contact_peak = max(self.arm_contact_peak, peak)
        # C2 uses the C-class arm-collision rule (c_class.ARM_COLLISION_N); A3 and B2 keep 2 N.
        limit = c_class.ARM_COLLISION_N if self.kind == "C2" else 2.0
        self.arm_collision_steps += int(peak > limit)
        if self.kind == "A3":
            self.shot(s)
        elif self.kind == "B2":
            self.box(s)
        else:
            self.cups(s)
        self.previous = s

    def shot(self, s):
        o = s["objects"]["ball"]
        p = np.asarray(o["position"])
        v = np.asarray(o["velocity"])
        grip = retained(o)
        if grip and not self.held:
            self.held = True
            self.event("grasp")
        # Timestamp the first contact-free tick, then confirm physical separation.
        # Confirmation latency must not shift a legal release across the buzzer.
        if self.held and self.release is None:
            if np.max(o["forces"][:3]) < 0.02 and p[2] > RELEASE_MIN_HEIGHT:
                if self.release_candidate is None:
                    self.release_candidate = (self.time, p.copy(), v.copy())
            else:
                self.release_candidate = None
        if (
            self.release_candidate is not None
            and self.release is None
            and np.linalg.norm(p - s["hands"]["main"]["position"]) > 0.04
        ):
            self.release, release_position, release_velocity = self.release_candidate
            self.temporal = bool(
                self.c["shot_start"] <= self.release < self.c["shot_start"] + self.c["shot_clock"]
            )
            self.event(
                "release",
                release_time=self.release,
                on_time=self.temporal,
                position=release_position.tolist(),
                velocity=release_velocity.tolist(),
            )
            if np.linalg.norm(release_position[:2] - self.b.hoop[:2]) < NO_DUNK_RADIUS:
                self.fail("release_inside_no_dunk_region")
            if not self.temporal:
                self.fail("release_outside_round")
        if self.release is not None and np.max(o["forces"][:3]) > 0.05:
            self.fail("ball_recontact_after_release")
        if self.previous and self.release is not None:
            old = np.asarray(self.previous["objects"]["ball"]["position"])
            z = self.b.hoop[2]
            if old[2] > z and p[2] <= z and v[2] < 0:
                alpha = (old[2] - z) / max(old[2] - p[2], 1e-9)
                cross = old + alpha * (p - old)
                self.crossing = bool(
                    np.linalg.norm(cross[:2] - self.b.hoop[:2]) + self.b.ball_radius
                    < self.b.hoop_radius - 0.004
                )
                if self.crossing:
                    self.event("hoop_entry")
            if (
                self.crossing
                and p[2] + self.b.ball_radius < z
                and np.linalg.norm(p[:2] - self.b.hoop[:2]) + self.b.ball_radius
                < self.b.hoop_radius - 0.004
            ):
                self.basket = True
                self.physical = True
                self.event("basket")
                self.done = True
        if self.time > self.c["shot_start"] + self.c["shot_clock"] + self.c["flight_grace"]:
            if not self.held:
                self.fail("no_grasp")
            elif self.release is None:
                self.fail("no_release")
            elif not self.basket:
                self.fail("missed_basket")
            self.done = True

    def box(self, s):
        from roboclock.envs.sports.arena.boxing import active_command, next_strike

        for i, command in enumerate(self.commands):
            for j, cue in enumerate(command["cues"]):
                if self.time >= cue and (i, j) not in self.cue_events:
                    self.cue_events.add((i, j))
                    self.event(
                        "cue",
                        command=i,
                        strike=j,
                        hand=command["hands"][j],
                        cue_time=cue,
                        deadline=cue + self.c["cue_response"],
                    )
        current = active_command(self.commands, self.time)
        if current is not None and not self.commands[current]["hands"]:
            for side in ("left", "right"):
                if np.linalg.norm(s["hands"][side]["position"] - guard(side)) > GUARD_HOLD:
                    self.fail("moved_during_hold")
        for side, pending in list(self.pending_guard.items()):
            if np.linalg.norm(s["hands"][side]["position"] - guard(side)) < GUARD_RETURN:
                latency = self.time - pending
                self.guards.append(dict(side=side, time=self.time, latency=latency))
                self.event("guard", side=side, latency=latency)
                if latency > self.c["guard_deadline"]:
                    self.fail("late_guard")
                del self.pending_guard[side]
            elif self.time - pending > self.c["guard_deadline"]:
                self.fail("late_guard")
        expected = next_strike(self, self.time)
        if expected is not None:
            index, count, hand, cue = expected
            if (
                not any(r["command"] == index and r["strike"] == count for r in self.reactions)
                and np.linalg.norm(s["hands"][hand]["position"] - guard(hand)) > GUARD_RETURN
            ):
                self.reactions.append(
                    dict(
                        command=index,
                        strike=count,
                        hand=hand,
                        time=self.time,
                        latency=self.time - cue,
                    )
                )
                self.event("reaction", command=index, strike=count, latency=self.time - cue)
        for side in ["left", "right"]:
            o = s["objects"]["pad_" + side]
            forces = o["glove_forces"]
            rest = pad_center(self.c, side)
            active = bool(max(forces) > 0.10 and o["position"][0] - rest[0] > 0.0004)
            if active and not self.hit_on["pad_" + side]:
                hand = ["left", "right"][int(np.argmax(forces))]
                correct = expected is not None and hand == side and hand == expected[2]
                if not correct:
                    self.fail("early_wrong_or_extra_strike")
                else:
                    index, count, hand, cue = expected
                    other = "right" if hand == "left" else "left"
                    if np.linalg.norm(s["hands"][other]["position"] - guard(other)) > GUARD_OTHER:
                        self.fail("non_striking_hand_out_of_guard")
                    latency = self.time - cue
                    timed = latency <= self.c["cue_response"]
                    self.hits.append(
                        dict(
                            command=index,
                            strike=count,
                            hand=hand,
                            time=self.time,
                            cue_time=cue,
                            latency=latency,
                            on_time=timed,
                            force=float(max(forces)),
                        )
                    )
                    self.event(
                        "strike",
                        command=index,
                        strike=count,
                        hand=hand,
                        latency=latency,
                        deadline=cue + self.c["cue_response"],
                        on_time=timed,
                    )
                    self.pending_guard[hand] = self.time
                    if not timed:
                        self.fail("late_strike")
            if active:
                self.hit_on["pad_" + side] = True
            elif s["hands"][side]["position"][0] < rest[0] - 0.15:
                self.hit_on["pad_" + side] = False
        if self.time >= self.commands[-1]["end"]:
            required = sum(len(c["hands"]) for c in self.commands)
            self.physical = (
                len(self.hits) == required
                and len(self.guards) == required
                and "moved_during_hold" not in self.failures
            )
            self.temporal = (
                self.physical
                and all(h["on_time"] for h in self.hits)
                and "late_guard" not in self.failures
            )
            if not self.physical:
                self.fail("incomplete_combinations")
            self.done = True

    def cups(self, s):
        from roboclock.envs.sports.arena.cup_patterns import layout

        _, required_supports = layout(self.card)
        good = []
        for name, target in self.cup_targets.items():
            o = s["objects"][name]
            held = retained(o, 0) or retained(o, 1)
            support = o["support_force_z"]
            contacts = o["support_contacts"]
            supporters = required_supports[name]
            topology = all(contacts.get(n, 0) > 0.015 for n in supporters)
            valid = cup_valid(o, target, support, held) and topology
            good.append(valid)
            if valid and (self.card, name) not in self.cups_placed:
                # Measured placement (stage score): the cup first rests released on its
                # card target with the card's support topology.
                self.cups_placed.add((self.card, name))
                self.event("cup_placed", card=self.card, cup=name)
            if o["position"][2] < 0.70:
                self.fail("dropped_cup")
        if all(good):
            if self.stable_since is None:
                self.stable_since = self.time
        else:
            self.stable_since = None
        bell = s["objects"]["bell"]
        down = bool(self.bell_rest - bell["position"][2] > 0.001 and max(bell["forces"][:4]) > 0.05)
        if down and not self.bell_down:
            # Either hand may ring (finger forces: left pair, then right pair); the bell
            # re-arms once the hand that rang has lifted, not the right hand in particular.
            fingers = bell["forces"][:4]
            self.bell_hand = "left" if max(fingers[:2]) > max(fingers[2:]) else "right"
            valid = (
                self.stable_since is not None
                and self.time - self.stable_since >= self.c["support_dwell"]
            )
            if (
                valid
                and self.c["preparation"]
                <= self.time
                <= self.c["preparation"] + self.c["cups_horizon"]
            ):
                self.card_times.append(self.time)
                self.event("card_complete", card=self.card)
                self.card += 1
                self.stable_since = None
                self.cup_targets, _ = layout(self.card)
            else:
                self.fail("early_or_incorrect_bell")
        if down:
            self.bell_down = True
        elif (
            s["hands"][self.bell_hand]["position"][2] > BELL_CLEAR_HEIGHT
            and abs(self.bell_rest - bell["position"][2]) < 0.001
            and max(bell["forces"][:4]) < 0.02
        ):
            # Spring rebound while the same hand is still pressing is one activation.
            self.bell_down = False
        required = self.c["cups_required"]
        start = self.c["preparation"]
        if len(self.card_times) >= required:
            if self.time - self.card_times[required - 1] >= c_class.SETTLE_S:
                self.finish_cups("all_done")
        elif c_class.stalled("C2", self.time, self.last_card_time(), start):
            self.finish_cups("stall")
        if not self.done and self.time >= start + self.c["cups_horizon"] - 1e-8:
            self.finish_cups("horizon")

    def last_card_time(self):
        return self.card_times[-1] if self.card_times else None

    def finish_cups(self, reason):
        if len(self.card_times) < self.c["cups_required"]:
            self.fail("incomplete_cards")
        self.done, self.termination_reason = True, reason

    def cups_result(self):
        """C2 verdicts: physical at the end without time terms, tiers from the last card."""
        required = self.c["cups_required"]
        failures = list(self.failures)
        if self.arm_collision_steps:
            failures.append("arm_collision")
        complete = len(self.card_times) >= required
        physical = complete and not failures
        jobs = [self.card_times[i] if i < len(self.card_times) else None for i in range(required)]
        difficulty = self.b.config.get("difficulty", "easy")
        judged = c_class.judge(
            "C2",
            jobs,
            required,
            physical,
            difficulty,
            start=self.c["preparation"],
            t_q=self.card_times[required - 1] if complete else None,
            termination_reason=self.termination_reason,
        )
        budget = judged["budget_s_by_tier"][difficulty]
        return dict(
            physical_success=physical,
            temporal_success=judged["temporal_success"],
            success=judged["joint_success"],
            failures=failures,
            card_times=list(self.card_times),
            completed_at=judged["t_q_s"],
            cards_per_minute=len(self.card_times) * 60 / budget,
            termination_reason=self.termination_reason,
            throughput=judged,
        )

    def result(self):
        result = dict(
            task=self.kind,
            randomization=self.b.config.get("randomization"),
            physical_success=self.physical,
            temporal_success=self.temporal,
            success=self.physical
            and self.temporal
            and not self.failures
            and self.arm_collision_steps == 0,
            failures=self.failures,
            sim_time=self.time,
            release_time=self.release,
            baskets=int(self.basket),
            hits=self.hits,
            reactions=self.reactions,
            boxing_schedule=self.commands if self.kind == "B2" else None,
            guard_returns=self.guards,
            cards=len(self.card_times),
            cards_per_minute=None,
            events=self.events,
            controller="privileged scripted feedback baseline; not trained visual AI",
            arm_contact_peak=self.arm_contact_peak,
            arm_collision_steps=self.arm_collision_steps,
        )
        if self.kind == "C2":
            result.update(self.cups_result())
        return result
