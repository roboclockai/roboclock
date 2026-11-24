"""Two independent grasp cycles sharing an exclusive hopper approach region."""

import numpy as np


class Oracle:
    def __init__(self, b, task):
        self.b = b
        self.t = task
        self.lock = None
        self.arms = {
            side: dict(phase="select", since=0, ball=None, cell=None, hold=None)
            for side in ("left", "right")
        }

    @property
    def phase(self):
        return "/".join(s + ":" + a["phase"] for s, a in self.arms.items())

    def transition(self, a, phase):
        a.update(phase=phase, since=self.b.time)

    def act(self):
        b = self.b
        t = self.t
        c = b.config["service"]
        s = b.last_state
        for side, a in self.arms.items():
            sign = 1 if side == "left" else -1
            target = np.array([0.30, sign * 0.36, 1.12])
            grip = 0.044
            speed = 0.28
            phase = a["phase"]
            elapsed = b.time - a["since"]
            if phase == "select":
                names = [
                    n
                    for n, p in c["sources"].items()
                    if p[1] * sign > 0 and n not in t.accepted and n not in t.grasped
                ]
                if names and not t.done:
                    a["ball"] = names[0]
                    self.transition(a, "approach")
                else:
                    self.transition(a, "idle")
            elif phase == "idle":
                pass
            else:
                name = a["ball"]
                o = s["objects"][name]
                p = np.array(o["position"])
                hand = s["hands"][side]["position"]
                if phase == "approach":
                    target = p + [0, 0, 0.17]
                    if np.linalg.norm(target - hand) < 0.012:
                        self.transition(a, "descend")
                elif phase == "descend":
                    target = p + [0, 0, 0.002]
                    speed = 0.10
                    if np.linalg.norm(target - hand) < 0.007:
                        a["hold"] = target.copy()
                        self.transition(a, "grasp")
                elif phase == "grasp":
                    target = a["hold"]
                    grip = 0.022
                    speed = 0.08
                    if name in t.grasped:
                        a["offset"] = p - np.array(hand)
                        self.transition(a, "lift")
                elif phase == "lift":
                    target = np.array([a["hold"][0], a["hold"][1], 1.14])
                    grip = 0.022
                    speed = 0.18
                    if np.linalg.norm(target - hand) < 0.012:
                        self.transition(a, "wait")
                elif phase == "wait":
                    target = np.array([a["hold"][0], sign * 0.36, 1.14])
                    grip = 0.022
                    used = set(t.occupants(s).values())
                    if t.pending:
                        used.add(t.pending[1])
                    free = [
                        i for i, p0 in enumerate(c["cells"]) if p0[1] * sign > 0 and i not in used
                    ]
                    if free and self.lock is None:
                        a["cell"] = free[0]
                        self.lock = side
                        self.transition(a, "transfer")
                elif phase in ("transfer", "release", "retreat"):
                    center = np.array(c["cells"][a["cell"]])
                    target = center.copy()
                    target[2] = 1.08
                    target -= a.get("offset", np.zeros(3))
                    if phase == "transfer":
                        grip = 0.022
                        speed = 0.20
                        if np.linalg.norm(target - hand) < 0.008:
                            self.transition(a, "release")
                    elif phase == "release":
                        grip = 0.044
                        if name in t.accepted:
                            self.transition(a, "retreat")
                    else:
                        target[2] = 1.17
                        if np.linalg.norm(target - hand) < 0.015:
                            if self.lock == side:
                                self.lock = None
                            self.transition(a, "select")
                if elapsed > 12 and phase not in ("wait", "idle"):
                    t.fail("controller_timeout_" + side + "_" + phase)
            b.move(side, target, grip=grip, speed=speed)
