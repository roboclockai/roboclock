"""Stationary standby, event-triggered approach and supported reracking."""
import numpy as np
from scipy.spatial.transform import Rotation


def grip_rotation(side):
    return Rotation.from_euler("z", 90 if side == "left" else -90, degrees=True).as_matrix() @ Rotation.from_euler("y", 180, degrees=True).as_matrix()


class BenchOracle:
    def __init__(self, b, task):
        self.b, self.task = b, task
        self.phase = "wait"
        self.start = 0.
        self.grip_offsets = {}

    def transition(self, phase):
        self.phase, self.start = phase, self.b.time

    def act(self):
        b, t, c = self.b, self.task, self.b.config["workflow"]
        o = b.last_state["objects"]["bar"]
        p, v = np.asarray(o["position"]), np.asarray(o["velocity"])
        bar_rotation = Rotation.from_quat(np.roll(o["orientation"], -1)).as_matrix()
        target = p.copy()
        gap = 0.038
        if self.phase == "wait":
            target = np.asarray(c["standby_position"])
            gap = c["standby_grip"]
            if t.event_time is not None and b.time-t.event_time >= c.get("controller_delay", 0.):
                self.transition("approach")
        if self.phase == "approach":
            # Close toward the bar while approaching: only the last few mm of jaw
            # travel remain at capture (the X5 fingers close at 0.15 m/s).
            gap = c.get("approach_grip", 0.038)
            target = p + np.clip(v * 0.07, -0.025, 0.025)
            grasp_points = p + (bar_rotation @ np.array([[0.,c["bench_grip_y"],0.],[0.,-c["bench_grip_y"],0.]]).T).T
            hands = np.array([b.last_state["hands"][side]["position"] for side in ("left", "right")])
            if np.max(np.linalg.norm(hands-grasp_points, axis=1)) < 0.018 and np.min(b.last_state["finger_gaps"]) > c.get("capture_gap_min", .065):
                # Account for finite native finger closing time from measured velocity.
                horizon = c.get("capture_prediction", 0.04)
                limit = c.get("capture_prediction_limit", 0.008)
                self.capture_target = p + np.clip(v * horizon, -limit, limit)
                self.capture_rotation = bar_rotation.copy()
                self.transition("capture")
        if self.phase == "capture":
            gap = 0.
            # Hold the interception frame while closing: chasing the falling
            # bar after contact would move the fingers away from their load.
            target = self.capture_target
            if t.catch_time is not None: self.transition("raise")
        if self.phase == "raise":
            gap = 0.
            target = np.array([c["bench_x"], 0., c["bench_top"] + 0.035])
            if abs(p[2]-target[2]) < 0.012: self.transition("transfer")
        if self.phase == "transfer":
            gap = 0.
            target = np.array([c["rack_x"], 0., c["bench_top"] + 0.035])
            arrived = np.linalg.norm(p-target) < 0.015
            if b.config.get("embodiment", {}).get("model") == "dual_arx_x5":
                # Native fingers can retain the bar above their nominal TCP.
                # Require rack alignment and safe vertical clearance, not an
                # exact transport height. Seating still requires both supports.
                arrived = (np.linalg.norm(p[:2]-target[:2]) < 0.010
                           and c["rack_height"] + 0.025 < p[2] < c["bench_top"] + 0.09)
            if arrived: self.transition("seat")
        if self.phase == "seat":
            gap = 0.
            target = np.array([c["rack_x"], 0., c["rack_height"] - 0.012])
            support = o.get("support_contacts", {})
            if all(support.get(name,0.) > 0.2 for name in ("RackLeft", "RackRight")):
                self.transition("release")
        if self.phase in ("release", "retreat"):
            target = np.array([c["rack_x"], 0., c["rack_height"] + (0.10 if self.phase == "retreat" else -0.012)])
            if b.time-self.start > 0.8 and self.phase == "release": self.transition("retreat")
        for side, sign in (("left",1),("right",-1)):
            rotation = bar_rotation if self.phase == "approach" else self.capture_rotation if self.phase in ("capture", "raise") else np.eye(3)
            if self.phase == "transfer":
                rotation = Rotation.from_rotvec(Rotation.from_matrix(self.capture_rotation).as_rotvec() * max(0., 1.-(b.time-self.start)/1.5)).as_matrix()
            lever = np.array([0., sign*c["bench_grip_y"], 0.])
            hand_target = target + rotation @ lever
            if b.config.get("embodiment", {}).get("model") == "dual_arx_x5":
                if self.phase in ("raise", "transfer", "seat"):
                    # Track the actual grasp offset: native fingers can hold
                    # the bar above the nominal TCP. Do not leave it hovering
                    # over the hooks while the hand reaches its own waypoint.
                    actual_hand = np.asarray(b.last_state["hands"][side]["position"])
                    self.grip_offsets[side] = p + bar_rotation @ lever - actual_hand
                if self.phase in ("raise", "transfer", "seat", "release", "retreat"):
                    hand_target -= self.grip_offsets.get(side, np.zeros(3))
            b.move(side, hand_target, grip_rotation(side), grip=gap,
                   speed=c["approach_speed"] if self.phase in ("approach","capture") else 0.10)
