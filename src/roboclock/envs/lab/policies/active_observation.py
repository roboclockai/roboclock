"""Move the otherwise idle second arm to a task-detail inspection pose.

The observer is internal side 1 (/World/ARXRight), which stands on the rig's LEFT (+y);
`apply` edits an internal side-order action (backend level, see lab/videos.py).

This is a privileged, calibrated inspection policy, not image-based control.
The operating arm and cooperative task planners retain priority.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.lab.robot import JOINT_HIGH, JOINT_LOW


def inspection_focus(backend):
    from roboclock.envs.lab.layout import apparatus_frame

    task = backend.task
    z = task.layout.table_height
    focus = {
        "A1": [0.286, 0.05, z + 0.075],
        "A2": [0.50, 0.13, z + 0.43],
        "A3": [0.48, 0.21, z + 0.10],
        "A4": [0.50, 0.15, z + 0.13],
        "A5": [0.48, 0.0, z + 0.12],
        "A6": [0.48, 0.0, z + 0.12],
        "C1": [0.48, 0.0, z + 0.10],
    }.get(task.task_id)
    if focus is not None:
        from roboclock.envs.lab.station_frame import station_of

        # Nominal station coordinates: follow the episode's seeded station frame.
        focus = station_of(task).point(focus)
    if focus is None:
        focus = task.layout.objects["sample_0"].position
    if getattr(task, "power_task", False):
        from roboclock.envs.lab.reactive_scene import cue_position

        # Frame the fault cue (LED or wheel) together with the disconnect handle.
        focus = (focus + cue_position(task.task_id, z, apparatus_frame(task.layout))) / 2
    if task.task_id == "B7":
        from roboclock.envs.lab.safety_instruments import level_lamp_position

        # Frame the level lamp together with the hose clamp.
        focus = (focus + level_lamp_position(z, apparatus_frame(task.layout))) / 2
    if task.task_id == "B1":
        # Watch the receiving tray from its open side; the operating forearm
        # occludes the original hotplate once the beaker is transferred.
        focus = task.layout.targets["destination"].position + [0, 0, 0.08]
    return np.asarray(focus, dtype=float)


def inspection_orientation(position, target, roll=0.0):
    """Aim tool +Z toward the detail while keeping the wrist camera upright."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    x = -(up - np.dot(up, forward) * forward)
    x /= np.linalg.norm(x)
    y = np.cross(forward, x)
    rotation = np.column_stack((x, y, forward)) @ Rotation.from_rotvec([0, 0, roll]).as_matrix()
    return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]


class ActiveObservation:
    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self.focus = inspection_focus(env.backend)
        self.goal = None
        self.goal_position = None
        self.goal_orientation = None
        self.status = "waiting for task cue"
        self.enabled = policy.bimanual is None
        self.started_at = None
        self.arrived_at = None
        self.rejected_candidates = 0
        self.commanded = env.backend.secondary_robot.get_joint_positions().copy()
        # Reactive episodes end soon after the action (B6/B7 0.5 s after the event,
        # B5 1.7 s), so their observer moves faster to arrive before the end.
        self.speed = (
            3.0
            if env.task.task_id in {"B6", "B7"}
            else 2.0
            if env.task.task_id == "B5"
            else 1.2
            if env.task.task_id.startswith("B")
            else 0.8
        )
        if not self.enabled:
            self.status = "cooperative task arm"
            return
        # Inspect from the observer arm's own side (rig left), above the operating hand.
        # Candidate IK is computed once; every move uses bounded joint targets.
        base, _ = env.backend.secondary_robot.get_world_pose()
        radial = self.focus[:2] - base[:2]
        radial /= np.linalg.norm(radial)
        radial_yaw = np.arctan2(radial[1], radial[0])
        distances, heights = (0.22, 0.28, 0.34), (0.20, 0.14, 0.28)
        if env.task.task_id == "A1":
            distances = (0.16, 0.18, 0.20, 0.22)
        if env.task.task_id in {"A2", "B5"}:
            heights = (0.08, 0.0, -0.08, 0.16)
        if env.task.task_id in {"A5", "A6", "C1"}:
            # Keep the observing fingers above the transfer/placement corridor.
            distances, heights = (0.28, 0.34, 0.40), (0.38, 0.32, 0.44)
        if env.task.task_id == "C1":
            # The final destination columns need the full overhead corridor.
            distances, heights = (0.46, 0.50, 0.54), (0.50, 0.44, 0.56)
        if env.task.task_id == "B1":
            distances, heights = (0.40, 0.46, 0.34), (0.38, 0.44, 0.32)
        # Approach from the mounting base's side and point along that same radial
        # plane. Large wrist yaw can self-contact the X5's offset wrist housing.
        offsets = [[*(-radial * distance), height] for height in heights for distance in distances]
        for offset in offsets:
            position = self.focus + offset
            for roll in (0.0, np.pi / 2, -np.pi / 2, np.pi):
                orientation = inspection_orientation(position, self.focus, roll)
                action, solved = policy.secondary_solver.compute_inverse_kinematics(
                    target_position=position, target_orientation=orientation
                )
                if not solved:
                    self.rejected_candidates += 1
                    continue
                q = env.backend.secondary_robot.get_joint_positions().copy()
                indices = action.joint_indices
                if indices is None:
                    indices = range(len(action.joint_positions))
                for index, value in zip(indices, action.joint_positions):
                    if value is not None:
                        q[index] = value
                q[6:] = 0.04
                if np.any(q < JOINT_LOW) or np.any(q > JOINT_HIGH):
                    self.rejected_candidates += 1
                    continue
                # Leave room around the physical elbow stop and wrist housing.
                yaw_error = (q[0] - radial_yaw + np.pi) % (2 * np.pi) - np.pi
                if q[2] < 0.12 or abs(q[4]) > 0.45 or abs(yaw_error) > 0.35:
                    self.rejected_candidates += 1
                    continue
                self.goal, self.goal_position, self.goal_orientation = q, position, orientation
                return
        self.status = "holding: no reachable inspection pose"

    def apply(self, action):
        if not self.enabled or self.goal is None:
            return action
        task = self.env.task
        if task.task_id.startswith("B") and task.event_time is None:
            return action
        if self.started_at is None:
            self.started_at = task.time
        action = np.asarray(action).copy()
        measured = self.env.backend.secondary_robot.get_joint_positions()
        # Deliberately slower than the operating hand; no teleport or joint reset.
        step = self.speed * self.policy.dt
        self.commanded = np.clip(
            self.commanded + np.clip(self.goal - self.commanded, -step, step),
            JOINT_LOW,
            JOINT_HIGH,
        )
        action[8:] = self.commanded
        self.status = "moving to inspect"
        if np.max(np.abs(measured[:6] - self.goal[:6])) < 0.04:
            self.status = "observing detail"
            if self.arrived_at is None:
                self.arrived_at = task.time
        return action

    def metadata(self):
        return {
            "enabled": self.enabled,
            # Physical rig side of internal side 1 (/World/ARXRight); recordings made before
            # the 2026-11-04 arm-order alignment say "right" here.
            "observer_arm": "left" if self.enabled else None,
            "controller": "privileged_calibrated_inspection_pose; joint_motor_control",
            "focus_position": self.focus.tolist(),
            "goal_position": None if self.goal_position is None else self.goal_position.tolist(),
            "goal_orientation_wxyz": (
                None if self.goal_orientation is None else self.goal_orientation.tolist()
            ),
            "goal_joints": None if self.goal is None else self.goal.tolist(),
            "final_joint_error": (
                None
                if self.goal is None
                else float(
                    np.max(
                        np.abs(
                            self.env.backend.secondary_robot.get_joint_positions()[:6]
                            - self.goal[:6]
                        )
                    )
                )
            ),
            "started_at": self.started_at,
            "arrived_at": self.arrived_at,
            "status": self.status,
            "rejected_ik_candidates": self.rejected_candidates,
            "cooperative_task_priority": True,
            "maximum_command_speed_rad_s": self.speed,
        }
