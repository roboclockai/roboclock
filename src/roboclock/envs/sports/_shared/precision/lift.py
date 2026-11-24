"""Evaluation from measured contacts and poses, independent of oracle phases."""

import numpy as np
from scipy.spatial.transform import Rotation


class Lift:
    def __init__(self, b):
        self.b = b
        self.c = b.config["workflow"]
        self.done = False
        self.events = []
        self.failures = []
        self.rounds = []
        self.phase = "ACQUIRE"
        self.stable_start = None
        self.down = None
        self.down_height = None
        self.lower = None
        self.return_start = None
        self.release_start = None
        self.had_grasp = False
        self.peak_arm_contact = 0.0
        self.last_grip = 0.0
        self.grasp_dwell = None
        self.hold_resets = 0
        self.filtered_angular_velocity = np.zeros(3)
        self.previous_time = 0.0
        self.previous_rotation = None
        self.hold_orientation = None
        self.reminder_shown = False
        self.delays = np.random.default_rng(b.config["seed"]).uniform(
            0.9, 1.8, self.c["lift_rounds"]
        )

    @property
    def round_index(self):
        return min(len(self.rounds), self.c["lift_rounds"] - 1)

    @property
    def target_height(self):
        return self.c["lift_heights"][self.round_index]

    @property
    def hold_duration(self):
        return self.c["lift_holds"][self.round_index]

    def fail(self, reason):
        if reason not in self.failures:
            self.failures.append(reason)
            self.events.append(dict(time=self.b.time, event=reason))

    def update(self, s):
        t = s["time"]
        o = s["objects"]["bar"]
        p = np.asarray(o["position"])
        forces = np.asarray(o["forces"])
        grasp = all(forces[i] > 0.025 for i in range(4))
        self.peak_arm_contact = max(self.peak_arm_contact, s["arm_contact_peak"])
        if s["arm_contact_peak"] > 5.0:
            self.fail("arm_collision")
        if grasp:
            self.last_grip = t
            if self.grasp_dwell is None:
                self.grasp_dwell = t
            if t - self.grasp_dwell > 0.15:
                self.had_grasp = True
        else:
            self.grasp_dwell = None
        axis = Rotation.from_quat(np.roll(o["orientation"], -1)).apply([0, 1, 0])
        rotation = Rotation.from_quat(np.roll(o["orientation"], -1))
        dt = max(0., t - self.previous_time)
        alpha = 1.0 - np.exp(-dt / 0.04)
        # Contact solvers can report a bias velocity while the measured pose is
        # stationary. Estimate actual rotation from consecutive measured poses.
        measured_spin = np.zeros(3) if self.previous_rotation is None or dt == 0 else (self.previous_rotation.inv() * rotation).as_rotvec() / dt
        self.filtered_angular_velocity += alpha * (measured_spin - self.filtered_angular_velocity)
        self.previous_rotation = rotation
        self.previous_time = t
        orientation_stable = self.hold_orientation is None or (self.hold_orientation.inv() * rotation).magnitude() < 0.025
        stable = (
            abs(p[2] - self.target_height) < 0.015
            and abs(axis[2]) < 0.035
            and np.linalg.norm(o["velocity"]) < 0.035
            and np.linalg.norm(self.filtered_angular_velocity) < 0.15
            and orientation_stable
            and t - self.last_grip < 0.15
            and o.get("support_contacts", {}).get("Cradle", 0) < 0.05
        )
        if self.had_grasp and p[2] > 0.9 and t - self.last_grip > 0.4:
            self.fail("lost_grasp")
        if self.phase in ("ACQUIRE", "HOLD"):
            if stable:
                if self.stable_start is None:
                    self.stable_start = t
                    self.hold_orientation = rotation
                    self.phase = "HOLD"
                if not self.reminder_shown and t - self.stable_start >= self.hold_duration:
                    self.reminder_shown = True
                    self.events.append(dict(time=t, event="hold_reminder", round=self.round_index))
                if t - self.stable_start >= self.hold_duration + self.delays[len(self.rounds)]:
                    self.down = t
                    self.down_height = float(p[2])
                    self.phase = "DOWN"
                    self.events.append(
                        dict(
                            time=t,
                            event="down",
                            round=len(self.rounds),
                            hold_start=self.stable_start,
                            continuous_hold=t - self.stable_start,
                            measured_height=self.down_height,
                        )
                    )
            else:
                if self.stable_start is not None:
                    self.hold_resets += 1
                if self.phase == "HOLD" and p[2] < self.target_height - 0.04:
                    self.fail("early_lowering")
                self.stable_start = None
                self.hold_orientation = None
        if self.phase == "DOWN":
            if (
                self.lower is None
                and p[2] < (self.down_height if self.down_height is not None else self.target_height) - 0.012
                and o["velocity"][2] < -0.025
            ):
                self.lower = t
                self.events.append(dict(time=t, event="lowering", latency=t - self.down))
                if t - self.down > self.c["lower_response"]:
                    self.fail("late_lowering")
            if self.lower is None and t - self.down > self.c["lower_response"]:
                self.fail("late_lowering")
            support = o.get("support_contacts", {}).get("Cradle", 0) > 0.15
            seated = (
                np.linalg.norm(p - self.b.bar_home) < 0.022
                and np.linalg.norm(o["velocity"]) < 0.025
            )
            released = np.max(forces[:4]) < 0.025
            if self.lower is not None and support and seated and released:
                if self.release_start is None:
                    self.release_start = t
                if t - self.release_start > 0.3:
                    self.rounds.append(
                        dict(
                            target_height=self.target_height,
                            required_hold=self.hold_duration,
                            down=self.down,
                            lower=self.lower,
                            latency=self.lower - self.down,
                            complete=t,
                        )
                    )
                    self.phase = "REST"
                    self.return_start = t
                    self.events.append(dict(time=t, event="round_complete", round=len(self.rounds)))
            else:
                self.release_start = None
            if t - self.down > 16:
                self.fail("return_timeout")
                self.done = True
        if self.phase == "REST" and t - self.return_start > 1.0:
            if len(self.rounds) >= self.c["lift_rounds"]:
                self.done = True
                self.phase = "COMPLETE"
            else:
                self.reminder_shown = False
                self.phase = "ACQUIRE"
                self.down = self.lower = self.stable_start = self.release_start = None
                self.hold_orientation = None
        if p[2] < 0.45:
            self.fail("dropped_bar")
            self.done = True
        if t >= self.c["lift_timeout"]:
            self.fail("episode_timeout")
            self.done = True

    def display(self):
        progress = f"R{self.round_index + 1} / {self.target_height:.2f}m / {self.hold_duration:.1f}s"
        if self.phase == "HOLD":
            elapsed = 0.0 if self.stable_start is None else self.b.time - self.stable_start
            reminder = self.hold_duration <= elapsed < self.hold_duration + self.c["hold_reminder_duration"]
            return ("HOLD / NOT YET" if reminder else progress, f"HOLD {elapsed:03.1f}s / WAIT FOR DOWN")
        if self.phase == "DOWN":
            return ("DOWN / LOWER NOW", f"REACT WITHIN {self.c['lower_response']:.2f}s")
        return (progress, self.phase + " / KEEP BAR LEVEL")

    def result(self):
        physical = (
            len(self.rounds) == self.c["lift_rounds"]
            and self.had_grasp
            and not any(
                x in self.failures
                for x in ["dropped_bar", "return_timeout", "lost_grasp", "arm_collision"]
            )
        )
        temporal = physical and not any(
            x in self.failures for x in ["early_lowering", "late_lowering", "episode_timeout"]
        )
        return dict(
            task="B3",
            primary_temporal_category="event_reactive",
            success=physical and temporal,
            physical_success=physical,
            temporal_success=temporal,
            rounds=self.rounds,
            failures=self.failures,
            events=self.events,
            peak_arm_contact=self.peak_arm_contact,
            hold_resets=self.hold_resets,
            controller="privileged scripted feedback baseline",
        )


class LiftOracle:
    def __init__(self, b, t):
        self.b = b
        self.task = t
        self.phase = "approach"
        self.start = 0.0
        self.offset = np.zeros(3)

    def transition(self, name):
        self.phase = name
        self.start = self.b.time

    def act(self):
        b, t = self.b, self.task
        o = b.last_state["objects"]["bar"]
        p = np.asarray(o["position"])
        z = p[2] + 0.15
        x = 0.36
        grip = 0.038
        if self.phase == "approach":
            z = p[2] + 0.20
            x = p[0]
        elif self.phase == "descend":
            z = p[2] + 0.079
            x = p[0]
        elif self.phase == "grasp":
            z = p[2] + 0.078
            x = p[0]
            grip = 0
        elif self.phase in ("lift", "hold"):
            z = t.target_height - self.offset[2]
            x = 0.36 - self.offset[0]
            grip = 0
        elif self.phase == "lower":
            z = 0.841 - self.offset[2]
            x = 0.36 - self.offset[0]
            grip = 0
        elif self.phase == "release":
            z = 0.841 - self.offset[2]
            x = 0.36 - self.offset[0]
        elif self.phase == "retreat":
            z = 1.07
        # Rotate around Z so the jaws close across, rather than along, the shaft.
        r = None
        errors = []
        for side, sign in [("left", 1), ("right", -1)]:
            r = Rotation.from_euler("y", 180, degrees=True).as_matrix()
            y = sign * b.grip_y + (p[1] if self.phase in ("approach", "descend", "grasp") else 0.)
            errors.append(b.move(side, [x, y, z], r, grip=grip, speed=0.16))
        elapsed = b.time - self.start
        if self.phase == "approach" and max(errors) < 0.008:
            self.transition("descend")
        elif self.phase == "descend" and max(errors) < 0.004:
            self.transition("grasp")
        elif self.phase == "grasp" and elapsed > 0.7 and t.grasp_dwell is not None and b.time - t.grasp_dwell > 0.15:
            midpoint = np.mean(
                [b.last_state["hands"][side]["position"] for side in ["left", "right"]], axis=0
            )
            self.offset = p - midpoint
            self.transition("lift")
        elif self.phase == "lift" and abs(p[2] - t.target_height) < 0.015:
            self.transition("hold")
        elif self.phase == "hold" and t.phase == "DOWN":
            self.transition("lower")
        elif self.phase == "lower" and abs(p[2] - 0.84) < 0.005 and elapsed > 1:
            self.transition("release")
        elif self.phase == "release" and elapsed > 0.7:
            self.transition("retreat")
        elif self.phase == "retreat" and t.phase == "ACQUIRE":
            self.transition("approach")
