"""Explicit bounded human-support wrench, not a bar pose animation.

The skinned character visualizes this reduced-order actuator model. It is not
a musculoskeletal simulation. Robot retention and rack support use contacts.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u * u * (3. - 2. * u)


def eccentric_drop(u, c):
    """Distance (m) the exhausted lifter has let the bar sink u s after the stall ends.

    A failed rep is lowered slowly under control (2026-11-17): the sink speed ramps linearly
    to ``fatigue_descent_speed`` over ``fatigue_descent_ramp`` and then stays constant.
    """
    if u <= 0.:
        return 0.
    v, ramp = c["fatigue_descent_speed"], c["fatigue_descent_ramp"]
    return v * u * u / (2. * ramp) if u < ramp else v * (u - ramp / 2.)


class HumanEffort:
    def __init__(self, b):
        self.b, self.c = b, b.config["workflow"]
        self.released = False
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.phase = "UNRACK"
        self.fatigue_at = 3. + (self.c["fatigue_rep"] - 1) * 5. + 2. + 2. * self.c["fatigue_progress"]

    def target(self, t):
        c = self.c
        x, top, bottom = c["bench_x"], c["bench_top"], c["bench_bottom"]
        if t < 1.:
            self.phase = "UNRACK"
            return np.array([c["rack_x"], 0., c["rack_height"] + (top - c["rack_height"]) * smooth(t)])
        if t < 2.:
            return np.array([c["rack_x"] + (x - c["rack_x"]) * smooth(t - 1), 0., top])
        if t < 3.:
            return np.array([x, 0., top])
        if t >= self.fatigue_at:
            self.phase = "STALL" if t < self.fatigue_at + c["fatigue_stall"] else "EXHAUSTED"
            z = bottom + (top - bottom) * smooth(c["fatigue_progress"])
            return np.array([x, 0., z - eccentric_drop(t - self.fatigue_at - c["fatigue_stall"], c)])
        u = (t - 3.) % 5.
        self.phase = "LOWER" if u < 2. else "PRESS" if u < 4. else "LOCKOUT"
        z = top - (top - bottom) * smooth(u / 2.) if u < 2. else bottom + (top - bottom) * smooth((u - 2.) / 2.)
        return np.array([x, 0., z])

    def apply(self):
        b, c = self.b, self.c
        o = b.last_state["objects"]["bar"]
        p, v = np.asarray(o["position"]), np.asarray(o["velocity"])
        target = self.target(b.time)
        weight = c["bench_mass"] * 9.81
        f = 120. * (target - p) - 14. * v + [0., 0., weight]
        capacity = 1.9 * weight
        if self.phase == "EXHAUSTED":
            # Exhausted: the lifter can no longer press the bar up (support at most its weight)
            # and lowers it along the slow eccentric target (``eccentric_drop``). Until
            # 2026-11-17 the support instead decayed to 0.97 x weight, a 0.29 m/s^2 fall that
            # reached the danger height 1.0-1.3 s after the descent was detected, before a
            # catch deadline of 1 s or more could expire.
            capacity = weight
        f[:2] = np.clip(f[:2], -4., 4.)
        f[2] = np.clip(f[2], 0., capacity)
        rot = Rotation.from_quat(np.roll(o["orientation"], -1))
        torque = np.clip(-2.5 * rot.as_rotvec() - 0.30 * np.asarray(o["angular_velocity"]), -0.60, 0.60)
        if self.released:
            f[:] = 0.
            torque[:] = 0.
            self.phase = "RELEASED"
        self.force, self.torque = f, torque
        row = b.contact_rows["bar"]
        b.contacts.apply_forces_and_torques_at_pos(
            forces=np.asarray([f], np.float32), torques=np.asarray([torque], np.float32),
            indices=np.asarray([row]), is_global=True,
        )

    def state(self):
        return dict(model="bounded bilateral human wrench", force=self.force.copy(),
                    torque=self.torque.copy(), phase=self.phase, released=self.released)
