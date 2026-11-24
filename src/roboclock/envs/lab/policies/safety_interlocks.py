"""Privileged measured-contact diagnostic; all motion uses the ARX X5 joint motors."""

import numpy as np

from roboclock.envs.lab.layout import apparatus_frame
from roboclock.envs.lab.policies.trajectories import CartesianCommand, limit_step
from roboclock.envs.lab.safety_geometry import body_yaw, grip_point, mechanism
from roboclock.physics.geometry import orientation_error


class SafetyTrajectory:
    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self):
        self.phase = "ready"
        self.last_goal = None
        self.angle = None

    def command(self, task, state):
        if task.event_time is None:
            return CartesianCommand(
                state.ee_position.copy(),
                not state.gripper_open,
                orientation=state.ee_orientation.copy(),
            )
        obj = state.objects["sample_0"]
        spec = mechanism(task.task_id, task.layout.table_height, apparatus_frame(task.layout))
        start = task.layout.objects["sample_0"].position
        close = False
        # B5: the door's opening angle; the jaws follow the door's world yaw.
        angle = float(state.controls.get("angle_rad", spec["angle"])) if task.task_id == "B5" else 0.0
        yaw = body_yaw(spec, angle) if task.task_id == "B5" else 0.0
        goal = start + [0, 0, 0.08]
        if task.event_time is not None:
            if task.task_id == "B5":
                # X5 pads are wider than the gap between the handle supports.
                # Pinch the exposed upper edge, keeping the tips above the mounts.
                grip = obj.position + [0, 0, 0.018]
                if self.phase == "ready":
                    self.phase = "approach"
                if self.phase == "approach":
                    goal = obj.position + [0, 0, 0.08]
                    desired = np.array([0.0, -np.sin(yaw / 2), np.cos(yaw / 2), 0.0])
                    if (
                        np.linalg.norm(state.ee_position - goal) < 0.018
                        and orientation_error(state.ee_orientation, desired) < 0.12
                    ):
                        self.phase = "descend"
                elif self.phase == "descend":
                    goal = grip.copy()
                    if np.linalg.norm(state.ee_position - goal) < 0.012:
                        self.phase = "grasp"
                elif self.phase == "grasp":
                    goal, close = grip.copy(), True
                    if obj.grasped:
                        self.phase, self.angle = "close", angle
                elif self.phase == "close":
                    close = True
                    self.angle = max(-0.025, self.angle - 0.90 * self.dt, angle - 0.10)
                    goal = grip_point(spec, self.angle) + [0, 0, 0.018]
                    yaw = body_yaw(spec, self.angle)
            else:
                close = True
                goal = start + [0, 0, -0.028]
        if self.last_goal is None:
            self.last_goal = state.ee_position.copy()
        # B6/B7 press a valve/clamp with the shortest deadlines (1.0 s hard): approach faster.
        speed = 0.9 if task.task_id in ("B6", "B7") else 0.65
        self.last_goal = limit_step(self.last_goal, goal, speed, self.dt)
        # Align the X5's symmetric jaws without a half-turn beyond its wrist range.
        return CartesianCommand(self.last_goal.copy(), close, yaw)
