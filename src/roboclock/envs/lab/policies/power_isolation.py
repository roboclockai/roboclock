"""Contact-driven knife-switch pull; every command goes through robot motors."""

import numpy as np

from roboclock.envs.lab.knife_switch_scene import GRIP_RADIUS
from roboclock.envs.lab.policies.trajectories import CartesianCommand, limit_step

# B2 reacts after a perception delay (control.cue_reaction_delay) and still has
# to open the switch within the hard deadline (1.88 s since the 2026-11-16 oracle
# calibration; 2.0 s before), so it approaches and pulls faster. B3 uses the same
# rates: at 0.4 m/s / 0.75 rad/s it isolated 2.1 s after the stall, missing the
# then 2.0 s hard deadline.
SPEED = {"B2": 0.9, "B3": 0.65}
PULL_RATE = {"B2": 3.0, "B3": 1.5}
# Hover height above the handle before the final descent. B2 hovers lower so the
# descent is short; B3 keeps the original 12 cm.
HOVER = {"B2": 0.05, "B3": 0.12}
# How far the commanded pull angle may lead the measured switch angle (rad).
PULL_LEAD = {"B2": 0.2, "B3": 0.06}


class PowerIsolationTrajectory:
    def __init__(self, dt):
        self.dt = dt
        self.last_goal = None
        self.phase = "approach"
        self.angle = 0.0

    def command(self, task, state):
        if task.event_time is None:
            if self.last_goal is None:
                self.last_goal = state.ee_position.copy()
            return CartesianCommand(
                self.last_goal.copy(), False, orientation=state.ee_orientation.copy()
            )
        initial = task.layout.objects["sample_0"].position
        pivot = initial - [GRIP_RADIUS, 0, 0]
        obj = state.objects["sample_0"]
        goal = initial + [0, 0, HOVER[task.task_id]]
        close = False
        if self.phase == "approach":
            if np.linalg.norm(state.ee_position - goal) < 0.02:
                self.phase = "descend"
        elif self.phase == "descend":
            goal = obj.position.copy()
            if task.event_time is not None and np.linalg.norm(state.ee_position - goal) < 0.012:
                self.phase = "grasp"
                # Close as soon as the handle is reached after the observed B3 stop.
                close = task.task_id == "B3"
        elif self.phase == "grasp":
            goal = obj.position.copy()
            close = True
            if obj.grasped:
                self.phase = "pull"
        elif self.phase in ("pull", "hold"):
            close = True
            measured = max(0, float(state.controls.get("switch_angle_rad", 0)))
            self.angle = min(
                np.deg2rad(65), self.angle + PULL_RATE[task.task_id] * self.dt, measured + PULL_LEAD[task.task_id]
            )
            goal = pivot + [GRIP_RADIUS * np.cos(self.angle), 0, GRIP_RADIUS * np.sin(self.angle)]
            # Broad parallel pads retain the handle while it rotates through the pinch.
            if self.angle >= np.deg2rad(64):
                self.phase = "hold"
        if self.last_goal is None:
            self.last_goal = state.ee_position.copy()
        self.last_goal = limit_step(self.last_goal, goal, SPEED[task.task_id], self.dt)
        return CartesianCommand(self.last_goal.copy(), close)
