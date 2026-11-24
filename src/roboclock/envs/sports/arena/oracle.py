"""Bounded feedback controllers; task objects are never attached or teleported."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.sports.arena.tasks import retained

# The launch jaw delay was tuned at 30 Hz, where it took effect on the next control
# tick: 0.12 s meant 0.133 s. Opening earlier leaves the ball between the fingers.
SHOT_TUNED_HZ = 30


class Oracle:
    def __init__(self, b, task):
        self.b, self.task = b, task
        self.phase = "approach"
        self.start = 0.0
        self.offset = None
        self.side_phase = {s: "approach" for s in ["left", "right"]}
        self.side_start = {s: 0.0 for s in ["left", "right"]}
        self.item = {s: 0 for s in ["left", "right"]}
        self.cup_offset = {}
        self.card = 0
        self.box_phase = {s: "guard" for s in ["left", "right"]}
        self.box_index = -1
        self.bell_phase = "lift"

    def transition(self, phase):
        self.phase = phase
        self.start = self.b.time

    def act(self):
        if self.b.kind == "A3":
            self.shot()
        elif self.b.kind == "B2":
            self.box()
        else:
            self.cups()

    def box(self):
        b, t = self.b, self.task
        from roboclock.envs.sports.arena.boxing import next_strike
        from roboclock.envs.sports.arena.tasks import STRIKE_DEPTH, guard, pad_center

        expected = next_strike(t, b.time)
        self.phase = "guard"
        for side in ("left", "right"):
            target = guard(side)
            speed = 0.45
            if expected is not None and not t.pending_guard:
                hand = expected[2]
                if side == hand:
                    # Straight punch at the (seeded) pad centre, knuckles into its face.
                    target = pad_center(t.c, side) - [STRIKE_DEPTH, 0.0, 0.0]
                    target[2] = guard(side)[2]
                    speed = 0.70
                    self.phase = "punch_" + side
            b.move(
                side,
                target,
                Rotation.from_euler("y", 90, degrees=True).as_matrix(),
                grip=0,
                speed=speed,
            )

    def shot(self):
        b = self.b
        s = b.last_state
        o = s["objects"]["ball"]
        p = np.asarray(o["position"])
        ee = s["hands"]["main"]["position"]
        target = ee.copy()
        grip = 0.04
        speed = 0.24
        r = Rotation.from_euler("y", 180, degrees=True).as_matrix()
        native_shot = b.config.get("basketball", {}).get("native_shot", {})
        if b.time < self.task.c["shot_start"]:
            self.phase = "approach"
            return
        if self.phase == "approach":
            target = np.r_[p[:2], 1.03]
            if np.linalg.norm(target - ee) < 0.008:
                self.transition("descend")
        elif self.phase == "descend":
            target = p
            if np.linalg.norm(target - ee) < 0.004:
                self.home = p.copy()
                self.transition("grasp")
        elif self.phase == "grasp":
            target = self.home
            grip = 0
            if retained(o):
                self.offset = p - ee
                self.transition("lift")
        elif self.phase == "lift":
            grip = 0
            target = np.array([0.34, -0.08, 1.07])
            if np.linalg.norm(target - ee) < 0.012:
                self.transition("aim")
        elif self.phase == "aim":
            grip = 0
            angle = max(110.0, 180.0 - 65.0 * (b.time - self.start))
            r = Rotation.from_euler("y", angle, degrees=True).as_matrix()
            target = np.array(native_shot.get("aim_position", [0.43, 0.06, 1.14]))
            if np.linalg.norm(target - ee) < 0.008 and b.time - self.start > 1.8:
                self.launch_start = target.copy()
                self.transition("launch")
        elif self.phase == "launch":
            # Accelerating arm supplies the launch velocity; release is jaw opening only.
            elapsed = b.time - self.start
            speed = 1.1
            grip = 0
            r = Rotation.from_euler("y", 110.0, degrees=True).as_matrix()
            target = np.array(native_shot.get("follow_through", [0.64, 0.06, 1.34]))
            open_after = native_shot.get("open_after", 0.40)
            if elapsed > (math.floor(open_after * SHOT_TUNED_HZ) + 1) / SHOT_TUNED_HZ:
                grip = 0.04
            if elapsed > 0.75:
                self.transition("follow")
        else:
            target = np.array([0.51, 0.06, 1.20])
            r = Rotation.from_euler("y", 110.0, degrees=True).as_matrix()
            grip = 0.04
        b.move("main", target, r, grip=grip, speed=speed)

    def cups(self):
        b, t = self.b, self.task
        s = b.last_state
        if b.time < t.c["preparation"]:
            self.phase = "ready"
            return
        if t.card != self.card:
            self.card = t.card
            self.item = {s: 0 for s in ["left", "right"]}
            self.side_phase = {s: "approach" for s in ["left", "right"]}
            self.bell_phase = "retract"
        if self.bell_phase == "retract":
            self.phase = "bell_retract"
            ee = s["hands"]["right"]["position"]
            b.move("right", [0.29, -0.075, 1.00], grip=0, speed=0.18)
            if ee[2] > 0.98:
                self.bell_phase = "lift"
            return
        if t.card >= 2:
            self.phase = "round_wait"
            for side, y in [("left", 0.30), ("right", -0.30)]:
                b.move(side, [0.24, y, 0.95], grip=0.038)
            return
        if all(i >= 3 for i in self.item.values()):
            self.phase = "bell_" + self.bell_phase
            for side, y in [("left", 0.3), ("right", -0.3)]:
                target = np.array([0.25, y, 1.00])
                if side == "right":
                    ee = s["hands"][side]["position"]
                    if self.bell_phase == "lift":
                        target = np.array([ee[0], ee[1], 1.00])
                        if ee[2] > 0.99 and t.stable_since is not None:
                            self.bell_phase = "above"
                    elif self.bell_phase == "above":
                        target = np.array([0.29, -0.075, 1.00])
                        if np.linalg.norm(target - ee) < 0.008:
                            self.bell_phase = "press"
                    else:
                        target = np.array([0.29, -0.075, 0.773])
                b.move(side, target, grip=0 if side == "right" else 0.038, speed=0.15)
            return
        for side, y in [("left", 0.23), ("right", -0.23)]:
            index = self.item[side]
            if index >= 3:
                b.move(side, [0.26, y, 0.95], grip=0.038)
                continue
            # Unstack the top first for the second card.
            order = [0, 1, 2] if t.card % 2 == 0 else [2, 0, 1]
            name = f"{side}_{order[index]}"
            o = s["objects"][name]
            p = np.asarray(o["position"])
            ee = s["hands"][side]["position"]
            phase = self.side_phase[side]
            dest = t.cup_targets[name]
            target = ee.copy()
            grip = 0.038
            w = t.c
            speed = w.get("cups_speed", 0.18)

            def advance(next_phase):
                self.side_phase[side] = next_phase
                self.side_start[side] = b.time

            if phase == "approach":
                target = np.r_[p[:2], max(0.94, p[2] + 0.10)]
                if np.linalg.norm(target - ee) < 0.008:
                    advance("descend")
            elif phase == "descend":
                target = p + [0, 0, 0.005]
                speed = w.get("cups_descend_speed", 0.10)
                if np.linalg.norm(target - ee) < 0.005:
                    advance("grasp")
            elif phase == "grasp":
                target = p + [0, 0, 0.005]
                grip = 0
                if retained(o, 0 if side == "left" else 1):
                    self.cup_offset[side] = p - ee
                    advance("lift")
            elif phase == "lift":
                target = np.r_[p[:2], 0.965]
                grip = 0
                if ee[2] > 0.95:
                    advance("transfer")
            elif phase == "transfer":
                target = np.r_[dest[:2], 0.965] - np.r_[self.cup_offset[side][:2], 0]
                grip = 0
                if np.linalg.norm(target - ee) < 0.008:
                    advance("lower")
            elif phase == "lower":
                target = ee + (dest - p) + [0, 0, 0.001]
                self.cup_offset[side] = p - ee
                grip = 0
                # Brake for the last centimetres above the support.
                speed = (
                    w.get("cups_lower_speed", 0.06)
                    if p[2] - dest[2] > w.get("cups_brake_height", 0.0)
                    else w.get("cups_brake_speed", 0.06)
                )
                if np.linalg.norm(p - dest) < 0.006:
                    advance("release")
            elif phase == "release":
                target = dest - self.cup_offset[side] + [0, 0, 0.001]
                if b.time - self.side_start[side] > w.get("cups_release_time", 0.45):
                    advance("retreat")
            elif phase == "retreat":
                target = np.r_[dest[:2], 0.965]
                if ee[2] > 0.95:
                    self.item[side] += 1
                    advance("approach")
            b.move(side, target, grip=grip, speed=speed)
        self.phase = "/".join(self.side_phase.values())
