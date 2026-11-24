"""Real opposing-finger grasp, contact pushing and sequential dual-arm handoff.

Only current cue is read; future red timestamps are never used by the policy.
"""

import numpy as np


class Oracle:
    def __init__(self, b, task):
        self.b, self.t = b, task
        self.arms = {s: dict(phase="approach", hold=None) for s in ("left", "right")}
        self.owner = None
        self.red_hold = False
        self.last_round = 0

    @property
    def phase(self):
        return "/".join(s + ":" + a["phase"] for s, a in self.arms.items())

    def hold_grips(self):
        # A saturated close command allows the handle to rock between the jaws.
        # Once opposing contact has settled, hold each measured jaw separately
        # with a small finite preload, preserving an asymmetric physical grasp.
        for side, arm in self.arms.items():
            if "grip_targets" in arm:
                self.b.command[self.b.rig.grip_ids[side]] = arm["grip_targets"]

    def act(self):
        b, t = self.b, self.t
        c = b.config["freeze"]
        s = b.last_state
        if t.red() is not None or t.done:
            if not self.red_hold:
                # Stop issuing moving waypoints. Position-drive hold is a robot action.
                for ids in b.rig.arm_ids.values():
                    b.command[ids] = s["joints"][ids]
                self.red_targets = {}
                if not t.done:
                    direction = -1 if t.round % 2 == 0 else 1
                    for side, arm in self.arms.items():
                        if arm["phase"] in ("push", "goal_hold"):
                            back, up = c.get("red_backoff", (0.012, 0.008))
                            self.red_targets[side] = np.asarray(s["hands"][side]["position"]) + [
                                0,
                                -direction * back,
                                up,
                            ]
                self.red_hold = True
            for side, target in self.red_targets.items():
                b.move(side, target, grip=0.008, speed=c.get("red_backoff_speed", 0.14))
            self.hold_grips()
            return
        self.red_hold = False
        desired = t.active_side
        if t.round != self.last_round:
            self.owner = None
            self.last_round = t.round
            for a in self.arms.values():
                a["phase"] = "retreat"
        if self.owner is not None and self.owner != desired:
            self.arms[self.owner]["phase"] = "retreat"
            self.owner = None
        all_ready = all(side in t.grasped for side in self.arms)
        for side, a in self.arms.items():
            sign = 1 if side == "left" else -1
            hand = np.asarray(s["hands"][side]["position"])
            tool = np.asarray(s["objects"][side + "_pusher"]["position"])
            park = np.array([0.30, sign * 0.37, 1.08])
            target = park.copy()
            grip = 0.044
            speed = 0.22
            phase = a["phase"]
            if phase == "approach":
                target = tool + [0, 0, c["tool_grip_offset"] + 0.12]
                if np.linalg.norm(target - hand) < 0.012:
                    a["phase"] = "descend"
            elif phase == "descend":
                target = tool + [0, 0, c["tool_grip_offset"]]
                speed = 0.09
                if np.linalg.norm(target - hand) < 0.007:
                    a["hold"] = target.copy()
                    a["phase"] = "grasp"
            elif phase == "grasp":
                target = a["hold"]
                grip = 0.008
                if side in t.grasped:
                    a["offset"] = tool - hand
                    a["grip_targets"] = np.clip(
                        s["joints"][b.rig.grip_ids[side]] - c["grip_preload"], 0, 0.044
                    )
                    a["phase"] = "retreat"
            elif phase == "retreat":
                grip = 0.008
                target = hand.copy()
                target[2] = 1.08
                if hand[2] > 1.065:
                    a["phase"] = "park"
            elif phase == "park":
                grip = 0.008
                if np.linalg.norm(target - hand) < 0.02:
                    other = "right" if side == "left" else "left"
                    if (
                        all_ready
                        and side == desired
                        and self.owner is None
                        and self.arms[other]["phase"] == "park"
                        and t.round < c["rounds"]
                    ):
                        self.owner = side
                        a["phase"] = "align"
            elif phase in ("align", "lower", "push", "goal_hold"):
                grip = 0.008
                puck = np.asarray(s["objects"]["puck"]["position"])
                direction = -1 if t.round % 2 == 0 else 1
                center = puck.copy()
                center[0] += np.clip(1.5 * (puck[0] - c["lane_x"]), -0.020, 0.020)
                center[1] -= direction * (c["puck_radius"] + c["tool_radius"] + 0.008)
                center[2] = c["surface_z"] + 0.016
                target = center - (tool - hand)
                if phase == "align":
                    target[2] = 1.08
                    if np.linalg.norm(target - hand) < 0.012:
                        a["phase"] = "lower"
                elif phase == "lower":
                    speed = 0.08
                    if np.linalg.norm(target - hand) < 0.008:
                        a["phase"] = "push"
                elif phase == "push":
                    target[1] += direction * 0.017
                    speed = c["push_speed"]
                    if t.checkpoint and abs(puck[1] - c["goals"][t.round][1]) < 0.008:
                        a["hold"] = hand.copy()
                        a["phase"] = "goal_hold"
                else:
                    target = a["hold"]
            b.move(side, target, grip=grip, speed=speed)
        self.hold_grips()
