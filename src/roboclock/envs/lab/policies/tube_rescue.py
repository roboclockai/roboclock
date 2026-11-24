"""Reach after the near-fall cue with open fingers; grasp, lift, then upright."""

import numpy as np

from roboclock.envs.lab.mdp.state import orientation_error
from roboclock.envs.lab.policies.trajectories import CartesianCommand, limit_step
from roboclock.physics.geometry import rotation_matrix


class TubeRescueTrajectory:
    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self):
        self.phase = "align"
        self.last_goal = None
        self.hold = None
        self.orientation = None
        self.hand_pitch = 0.0

    def command(self, task, state):
        if task.event_time is None:
            return CartesianCommand(
                state.ee_position.copy(), False, orientation=state.ee_orientation.copy()
            )
        obj = state.objects["sample_0"]
        axis = rotation_matrix(obj.orientation)[:, 2]
        theta = float(np.arctan2(axis[0], axis[2]))
        # The symmetric pinch is unchanged by a half turn around the approach
        # axis; use the orientation within the X5 wrist's range.
        aligned = np.array([0.0, 0.0, 1.0, 0.0])
        pick = obj.position + axis * 0.0525 + obj.velocity * 0.025
        goal = pick.copy()
        target_orientation = aligned
        close = False
        if self.phase == "align":
            if self.hold is None:
                self.hold = state.ee_position.copy()
            goal = self.hold
            if orientation_error(state.ee_orientation, aligned) < 0.15:
                self.phase = "hover"
        elif self.phase == "hover":
            goal = goal + np.array([0, 0, 0.08])
            if (
                np.linalg.norm(state.ee_position - goal) < 0.02
                and orientation_error(state.ee_orientation, aligned) < 0.20
            ):
                self.phase = "reach"
        if self.phase == "reach":
            goal = pick.copy()
            if (
                np.linalg.norm(state.ee_position - goal) < 0.018
                and orientation_error(state.ee_orientation, aligned) < 0.23
            ):
                self.phase = "grasp"
        if self.phase == "grasp":
            close = True
            if obj.grasped:
                self.hold = state.ee_position.copy()
                self.hold[2] = max(self.hold[2] + 0.16, task.layout.table_height + 0.29)
                self.grasp_orientation = state.ee_orientation.copy()
                self.phase = "lift"
        if self.phase == "lift":
            close = True
            goal = self.hold
            target_orientation = self.grasp_orientation
            if state.ee_position[2] > task.layout.table_height + 0.24:
                self.phase = "upright"
        if self.phase == "upright":
            close = True
            goal = self.hold
            self.hand_pitch -= np.clip(theta, -0.6, 0.6) * 2.5 * self.dt
            self.hand_pitch = float(np.clip(self.hand_pitch, -1.2, 1.2))
            target_orientation = np.array(
                [-np.sin(self.hand_pitch / 2), 0, np.cos(self.hand_pitch / 2), 0]
            )
        if self.last_goal is None:
            self.last_goal = state.ee_position.copy()
            self.orientation = state.ee_orientation.copy()
        self.last_goal = limit_step(self.last_goal, goal, 0.65, self.dt)
        if np.dot(self.orientation, target_orientation) < 0:
            target_orientation = -target_orientation
        angle = orientation_error(self.orientation, target_orientation)
        mix = min(1, 2.0 * self.dt / max(angle, 1e-9))
        self.orientation = (1 - mix) * self.orientation + mix * target_orientation
        self.orientation /= np.linalg.norm(self.orientation)
        return CartesianCommand(self.last_goal.copy(), close, orientation=self.orientation.copy())
