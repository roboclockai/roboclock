"""Parallel source pickup; exclusive shared-rack placement with measured release."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

# Time constant of the held-ball offset re-estimate (s) and the ball speed that
# counts as settled before release (m/s).
OFFSET_TIME_CONSTANT = 0.158
SETTLED_BALL_SPEED = 0.009


class Oracle:
    def __init__(self, b, t):
        self.b, self.t = b, t
        self.lock = None
        self.transitions = []
        self.dt = b.config["sim"]["physics_dt"] * b.config["sim"]["control_decimation"]
        self.arms = {s: dict(phase="select", since=0.0, index=None) for s in ("left", "right")}

    @property
    def phase(self):
        return "/".join(s + ":" + a["phase"] for s, a in self.arms.items())

    def transition(self, side, phase):
        self.arms[side].update(phase=phase, since=self.b.time)
        self.transitions.append(
            dict(side=side, phase=phase, time=self.b.time, index=self.arms[side]["index"])
        )

    @staticmethod
    def aligned(hand, rotation):
        actual = Rotation.from_quat(np.roll(hand["orientation"], -1)).as_matrix()
        return Rotation.from_matrix(actual.T @ rotation).magnitude() < 0.035

    def act(self):
        b, t, c = self.b, self.t, self.b.config["rack"]
        for side, a in self.arms.items():
            phase, elapsed = a["phase"], b.time - a["since"]
            hand = b.last_state["hands"][side]
            hp = np.asarray(hand["position"])
            rotation = Rotation.from_euler("y", 180, degrees=True).as_matrix()
            target, grip, speed = np.array(c["home"][side]), c["open_grip"], 0.32
            if phase == "select":
                todo = [
                    i
                    for i, s in enumerate(c["sides"])
                    if s == side and f"ball{c['order'][i]:02d}" not in t.accepted
                ]
                if todo and not t.done:
                    a["index"] = todo[0]
                    self.transition(side, "approach")
                else:
                    self.transition(side, "idle")
            elif phase != "idle":
                i = a["index"]
                name = f"ball{c['order'][i]:02d}"
                p = np.asarray(b.last_state["objects"][name]["position"])
                if phase in ("approach", "descend", "grasp", "lift", "wait"):
                    rotation = Rotation.from_euler("y", c["grasp_pitch"], degrees=True).as_matrix()
                if phase == "approach":
                    target = p + [0, 0, 0.12]
                    if np.linalg.norm(hp - target) < 0.008:
                        self.transition(side, "descend")
                elif phase == "descend":
                    target, speed = p + [0, 0, c["grasp_height"]], 0.10
                    if np.linalg.norm(hp - target) < 0.004 and self.aligned(hand, rotation):
                        a["hold"] = target.copy()
                        self.transition(side, "grasp")
                elif phase == "grasp":
                    target, grip = a["hold"], c["closed_grip"]
                    if t.grasped.get(name) == side:
                        hr = Rotation.from_quat(np.roll(hand["orientation"], -1)).as_matrix()
                        a["offset"] = hr.T @ (p - hp)
                        a["held_xy"] = p[:2].copy()
                        self.transition(side, "lift")
                elif phase == "lift":
                    speed = 0.12
                    target = (
                        np.array([*a["held_xy"], c["transfer_height"]]) - rotation @ a["offset"]
                    )
                    grip = c["closed_grip"]
                    if np.linalg.norm(hp - target) < 0.008 and name in t.lifted:
                        self.transition(side, "wait")
                elif phase == "wait":
                    target = (
                        np.array([*a["held_xy"], c["transfer_height"]]) - rotation @ a["offset"]
                    )
                    grip = c["closed_grip"]
                    first = next(
                        (j for j, n in enumerate(c["order"]) if f"ball{n:02d}" not in t.accepted),
                        None,
                    )
                    if self.lock is None and first == i:
                        self.lock = side
                        self.transition(side, "transfer")
                elif phase in ("transfer", "lower", "release", "retreat"):
                    rotation = (
                        Rotation.from_euler(
                            "z", -90 if side == "left" else 90, degrees=True
                        ).as_matrix()
                        @ Rotation.from_euler("y", 180, degrees=True).as_matrix()
                    )
                    dest = np.array(c["slots"][i])
                    ball_goal = dest.copy()
                    ball_goal[2] = (
                        c["transfer_height"]
                        if phase in ("transfer", "retreat")
                        else dest[2] + c["release_clearance"]
                    )
                    target = ball_goal - rotation @ a["offset"]
                    if phase == "lower":
                        # Re-estimate the held-ball offset from measured geometry.
                        hr = Rotation.from_quat(np.roll(hand["orientation"], -1)).as_matrix()
                        blend = 1 - math.exp(-self.dt / OFFSET_TIME_CONSTANT)
                        a["offset"] += blend * (hr.T @ (p - hp) - a["offset"])
                        target = ball_goal - rotation @ a["offset"]
                    grip = c["closed_grip"]
                    if phase == "transfer" and np.linalg.norm(hp - target) < 0.006:
                        self.transition(side, "lower")
                    elif phase == "lower":
                        speed = 0.12
                        if (
                            np.linalg.norm(p - ball_goal) < 0.0015
                            and np.linalg.norm(p - a.get("previous_ball", p + 1))
                            < SETTLED_BALL_SPEED * self.dt
                            and self.aligned(hand, rotation)
                        ):
                            a.setdefault("settled", b.time)
                        else:
                            a.pop("settled", None)
                        if b.time - a.get("settled", b.time) > 0.35:
                            a["release_target"] = target.copy()
                            a["release_joints"] = b.robot.get_joint_positions()[
                                b.rig.grip_ids[side]
                            ].copy()
                            self.transition(side, "release")
                    elif phase == "release":
                        target = a["release_target"]
                        grip = c["closed_grip"] + (0.030 - c["closed_grip"]) * min(
                            1.0, elapsed / 0.45
                        )
                        if (
                            elapsed > 0.7
                            and max(b.last_state["objects"][name]["forces"][:4]) < 0.05
                        ):
                            self.transition(side, "retreat")
                    elif phase == "retreat":
                        grip = 0.030 if hp[2] < dest[2] + 0.085 else c["open_grip"]
                        if np.linalg.norm(hp - target) < 0.008 and name in t.accepted:
                            self.transition(side, "home")
                elif phase == "home":
                    if np.linalg.norm(hp - target) < 0.015:
                        self.lock = None
                        self.transition(side, "select")
                if elapsed > 12.0 and phase not in ("wait", "idle"):
                    t.fail("controller_timeout_" + side + "_" + phase)
                    self.transition(side, "idle")
            if a["index"] is not None:
                a["previous_ball"] = np.array(
                    b.last_state["objects"][f"ball{c['order'][a['index']]:02d}"]["position"]
                )
            b.move(side, target, rotation, grip=grip, speed=speed)
            if phase == "release" or (phase == "retreat" and hp[2] < dest[2] + 0.085):
                # Retract each jaw from its measured contact position. Equal absolute
                # targets can otherwise close the wider jaw and push the supported ball.
                fraction = min(1.0, elapsed / 0.35) if phase == "release" else 1.0
                b.command[b.rig.grip_ids[side]] = np.minimum(
                    0.044, a["release_joints"] + 0.004 * fraction
                )
