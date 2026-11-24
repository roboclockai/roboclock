"""Independent measured-contact, reaction and red-hold evaluator."""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


# Path-length window of the stop measurement (s).
SPEED_WINDOW = 0.032
# Checkpoint of a round, where the indicated hand changes (the instruction states both,
# ``roboclock.envs.sports.tasks``): the puck centre this far past the midline toward the goal,
# within this distance of the lane's centre line; metres.
CHECKPOINT_PAST_MIDLINE = 0.060
CHECKPOINT_LANE_HALF_WIDTH = 0.040


def angle(a, b):
    return float(
        (Rotation.from_quat(np.roll(a, -1)) * Rotation.from_quat(np.roll(b, -1)).inv()).magnitude()
    )


def retained(obj, side):
    i = 0 if side == "left" else 2
    return min(obj["forces"][i : i + 2]) > 0.025


def inside_goal(position, goal, c):
    return bool(
        np.all(
            np.abs(np.asarray(position)[:2] - np.asarray(goal)[:2]) + c["puck_radius"]
            <= np.asarray(c["goal_half_size"])
        )
    )


def deadline_met(latency, limit):
    return latency < limit  # Strict application deadline, including equality failure.


class Task:
    def __init__(self, b):
        self.b, self.c = b, b.config["freeze"]
        self.done = False
        self.events = []
        self.failures = []
        self.round = 0
        self.checkpoint = False
        self.grasped = set()
        self.grasp_since = {}
        self.grasp_offsets = {}
        self.last_contacts = {}
        self.goal_since = None
        self.previous = None
        self.active_red = None
        self.reactions = []
        self.still_since = None
        self.hold_reference = None
        self.peak_arm_contact = 0.0
        self.travel = 0.0
        self.maximum_lift = 0.0
        self.current_speeds = {}
        self.speed_intervals = round(SPEED_WINDOW / b.config["sim"]["physics_dt"])
        self.motion_history = deque(maxlen=self.speed_intervals + 1)

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)

    def emit(self, event, **kw):
        self.events.append(dict(event=event, time=self.b.time, **kw))

    def red(self):
        return next(
            (i for i, (a, z) in enumerate(self.c["red_windows"]) if a <= self.b.time < z), None
        )

    @property
    def active_side(self):
        pair = ("left", "right") if self.round % 2 == 0 else ("right", "left")
        return pair[int(self.checkpoint)]

    def bodies(self, s):
        return {
            **{n: o for n, o in s["objects"].items()},
            **{"hand_" + n: o for n, o in s["hands"].items()},
        }

    def stopped(self, s):
        # PhysX constraint impulses can report nonzero end-step velocities while
        # solved poses remain stationary. Measure path length over the speed
        # window instead. This adds up to 32 ms latency, never subtracts it,
        # and catches oscillatory motion rather than cancelling endpoint drift.
        self.motion_history.append(s)
        if len(self.motion_history) <= self.speed_intervals:
            return False
        history = list(self.motion_history)
        dt = history[-1]["time"] - history[0]["time"]
        okay = True
        for n in self.bodies(s):
            samples = [self.bodies(row)[n] for row in history]
            positions = np.asarray([o["position"] for o in samples])
            orientations = np.asarray([o["orientation"] for o in samples])
            v = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum() / dt
            rotations = Rotation.from_quat(np.roll(orientations, -1, axis=1))
            w = (rotations[1:] * rotations[:-1].inv()).magnitude().sum() / dt
            prefix = "hand" if n.startswith("hand_") else "body"
            self.current_speeds[n] = dict(linear=float(v), angular=float(w))
            okay = (
                okay
                and v < self.c[prefix + "_linear_limit"]
                and w < self.c[prefix + "_angular_limit"]
            )
        return bool(okay)

    def update(self, s):
        if self.done:
            return
        now = s["time"]
        c = self.c
        puck = s["objects"]["puck"]
        p = np.asarray(puck["position"])
        supported = puck.get("support_contacts", {}).get("Rink", 0) > 0.10
        self.peak_arm_contact = max(self.peak_arm_contact, s["arm_contact_peak"])
        if s["arm_contact_peak"] > 8:
            self.fail("arm_collision")
        if self.previous is not None:
            self.travel += float(
                np.linalg.norm(p[:2] - self.previous["objects"]["puck"]["position"][:2])
            )
        self.maximum_lift = max(
            self.maximum_lift, float(p[2] - c["surface_z"] - c["puck_height"] / 2)
        )
        if self.maximum_lift > 0.025:
            self.fail("puck_lifted")
        if p[2] < c["surface_z"] - 0.03 or not (0.12 < p[0] < 0.76 and abs(p[1]) < 0.56):
            self.fail("puck_out_of_bounds")
        for side in ("left", "right"):
            name = side + "_pusher"
            tool = s["objects"][name]
            offset = np.asarray(tool["position"]) - s["hands"][side]["position"]
            stable = retained(tool, side) and (
                side not in self.grasp_offsets
                or np.linalg.norm(offset - self.grasp_offsets[side]) < 0.008
            )
            if stable:
                self.grasp_since.setdefault(side, now)
                self.grasp_offsets.setdefault(side, offset.copy())
                if now - self.grasp_since[side] >= c["grasp_dwell"] and side not in self.grasped:
                    self.grasped.add(side)
                    self.emit("retained_tool", side=side)
            else:
                self.grasp_since.pop(side, None)
                self.grasp_offsets.pop(side, None)
            idx = self.b.filter_paths.index("/World/Objects/" + name)
            contact = float(puck["forces"][idx])
            if contact > 0.015 and retained(tool, side):
                self.last_contacts[side] = now
            if tool["position"][2] < 0.78:
                self.fail("dropped_tool_" + side)
        red = self.red()
        stopped = self.stopped(s)
        if red != self.active_red:
            if self.active_red is not None:
                record = self.reactions[-1]
                if record["stop_time"] is None:
                    self.fail("no_confirmed_stop")
                self.emit("green", index=self.active_red)
            self.active_red = red
            self.still_since = None
            self.hold_reference = None
            if red is not None:
                record = dict(
                    index=red,
                    event_time=now,
                    scheduled_time=c["red_windows"][red][0],
                    stop_time=None,
                    latency=None,
                    within_deadline=False,
                    hold_ok=True,
                    moving_at_event=not stopped,
                )
                self.reactions.append(record)
                self.emit("red", index=red, moving_at_event=not stopped)
        if red is not None:
            record = self.reactions[-1]
            if stopped:
                if self.still_since is None:
                    self.still_since = now
                if record["stop_time"] is None and now - self.still_since >= c["stable_dwell"]:
                    latency = self.still_since - record["event_time"]
                    record.update(
                        stop_time=self.still_since,
                        confirmation_time=now,
                        latency=latency,
                        within_deadline=deadline_met(latency, c["response_limit"]),
                    )
                    self.hold_reference = {
                        n: (np.array(o["position"]), np.array(o["orientation"]))
                        for n, o in self.bodies(s).items()
                    }
                    self.emit(
                        "confirmed_stop",
                        index=red,
                        latency=latency,
                        within_deadline=record["within_deadline"],
                    )
                    if not record["within_deadline"]:
                        self.fail("late_stop")
            else:
                self.still_since = None
                if record["stop_time"] is not None:
                    record["hold_ok"] = False
                    self.fail("red_motion")
            if self.hold_reference:
                for n, o in self.bodies(s).items():
                    pp, qq = self.hold_reference[n]
                    if (
                        np.linalg.norm(np.asarray(o["position"]) - pp) > c["hold_translation"]
                        or angle(o["orientation"], qq) > c["hold_rotation"]
                    ):
                        record["hold_ok"] = False
                        self.fail("red_drift")
            if (
                record["stop_time"] is None
                and now - record["event_time"] >= c["response_limit"] + c["stable_dwell"]
            ):
                self.fail("late_stop")
        # Goal and checkpoint progress is allowed only in green with actual supported push contacts.
        if self.round < c["rounds"] and red is None:
            side = self.active_side
            touched = now - self.last_contacts.get(side, -100) < 0.25
            if (
                not self.checkpoint
                and (-1 if self.round % 2 == 0 else 1) * p[1] >= CHECKPOINT_PAST_MIDLINE
                and abs(p[0] - c["lane_x"]) < CHECKPOINT_LANE_HALF_WIDTH
                and supported
                and touched
            ):
                self.checkpoint = True
                self.last_contacts.clear()
                self.emit("checkpoint", round=self.round, side=side)
            valid = (
                self.checkpoint
                and inside_goal(p, c["goals"][self.round], c)
                and supported
                and side in self.last_contacts
                and np.linalg.norm(puck["velocity"]) < 0.018
            )
            if valid:
                if self.goal_since is None:
                    self.goal_since = now
                if now - self.goal_since >= c["goal_dwell"]:
                    self.emit("goal", round=self.round, side=side)
                    self.round += 1
                    self.checkpoint = False
                    self.goal_since = None
                    self.last_contacts.clear()
            else:
                self.goal_since = None
        if self.round >= c["rounds"] and len(self.reactions) >= 3 and red is None:
            self.done = True
        if now >= c["horizon"]:
            self.done = True
            if self.round < c["rounds"]:
                self.fail("incomplete_route")
        self.previous = s

    def display(self):
        if self.done:
            return (
                "COMPLETE" if self.result()["success"] else "RUN FINISHED",
                f"ROUNDS {self.round} / {self.c['rounds']}"
                + (" / CHECK HOLD" if "red_motion" in self.failures else ""),
            )
        red = self.red()
        if red is not None:
            r = self.reactions[-1] if self.reactions else None
            value = "BRAKING" if not r or r["latency"] is None else f"STOP {r['latency']:.2f}s"
            if r and not r["hold_ok"]:
                value = "HOLD VIOLATION"
            return ("RED / HOLD STILL", f"{value} / LIMIT {self.c['response_limit']:.2f}s")
        return (
            "GREEN / MOVE",
            f"ROUND {min(self.round + 1, self.c['rounds'])} / {self.c['rounds']}   {self.active_side.upper()} HAND",
        )

    def result(self):
        physical = self.round == self.c["rounds"] and not any(
            x in self.failures
            for x in (
                "puck_lifted",
                "puck_out_of_bounds",
                "arm_collision",
                "dropped_tool_left",
                "dropped_tool_right",
            )
        )
        temporal = len(self.reactions) >= 3 and all(
            r["within_deadline"] and r["hold_ok"] for r in self.reactions
        )
        return dict(
            task="B4",
            revision=self.b.config["revision"],
            primary_temporal_category="event_reactive",
            success=physical and temporal and not self.failures,
            physical_success=physical,
            temporal_success=temporal,
            rounds_completed=self.round,
            reactions=self.reactions,
            failures=self.failures,
            events=self.events,
            puck_travel_m=self.travel,
            max_puck_lift_m=self.maximum_lift,
            peak_arm_contact_n=self.peak_arm_contact,
            controller="Privileged state-feedback scripted baseline; not a trained visual agent",
            limitations="Authored tabletop game; no air cushion; fixed physical friction. Hands and tools must both stop.",
        )
