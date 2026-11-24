"""Privileged hospital trajectory diagnostic using Isaac Sim Lula IK and real motor control.

The planner drives internal arm 0 (/World/ARXLeft, which stands on the rig's RIGHT, -y). A ward
task whose bedside module defines ``command_secondary(planner, task, state)`` also drives internal
arm 1 (/World/ARXRight, the rig's LEFT, +y) with its own planner; otherwise arm 1 holds the joints
it started the episode with. Each planner carries ``side`` (internal arm index: 0 = rig right,
1 = rig left), ``partner`` (the other arm's planner, for sharing work and keeping clear),
``ee_position`` (its own tool point, refreshed every step) and ``held`` (objects its hand grasps).

`reset` / `act` speak the env interface (rig-left arm first, LabEnv.external_arm_order);
`reset_internal` / `act_internal` use internal arm order, for scripts that drive `env.backend`.
"""

import numpy as np

from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory, ramp_finger
from roboclock.envs.lab.policies.trajectories import multiply as quat_multiply
from roboclock.envs.lab.robot import (
    ARM_DOF,
    DOF,
    EXTERNAL_ARMS,
    JOINT_HIGH,
    JOINT_LOW,
    VELOCITIES,
    arms_to_external,
    observation_to_internal,
)


class OraclePickPlace:
    def __init__(self, env):
        from roboclock.envs.lab.robot import make_solver

        self.env = env
        self.dt = env.config["sim"]["physics_dt"] * env.config["sim"]["control_decimation"]
        self.solver = make_solver(env.backend.robot)
        self.secondary_solver = make_solver(env.backend.secondary_robot)
        self.planner = HospitalTrajectory(env.config["control"], self.dt)
        self.secondary_planner = HospitalTrajectory(env.config["control"], self.dt)
        self._link_planners()
        self.waiting_joints = env.backend.joint_positions().copy()

    def _link_planners(self):
        for side, (planner, partner) in enumerate(
            ((self.planner, self.secondary_planner), (self.secondary_planner, self.planner))
        ):
            planner.side, planner.partner = side, partner
            planner.ee_position, planner.ee_orientation, planner.held = None, None, ()
            planner.joint_target = None

    @property
    def arm_order(self):
        """Internal arm of each interface arm (LabEnv.external_arm_order)."""
        return getattr(self.env, "external_arm_order", EXTERNAL_ARMS)

    def reset(self, observation):
        """Reset from an env observation (interface order)."""
        self.reset_internal(observation_to_internal(observation, self.arm_order))

    def act(self, observation):
        """Interface-order action (rig-left arm first) for `env.step`."""
        internal = observation_to_internal(observation, self.arm_order)
        return arms_to_external(self.act_internal(internal), self.arm_order)

    def reset_internal(self, observation):
        self.planner.reset()
        self.secondary_planner.reset()
        self._link_planners()
        self.waiting_joints = observation["joint_positions"].copy()
        self.finger = float(np.mean(observation["joint_positions"][6:DOF]))
        self.secondary_finger = float(np.mean(observation["joint_positions"][DOF + 6 :]))

    @property
    def bimanual(self):
        from roboclock.envs.hospital.bedside import module_for

        module = module_for(self.env.task)
        return module if module is not None and hasattr(module, "command_secondary") else None

    def act_internal(self, observation):
        """Internal arm-order action from internal arm-order joint_positions."""
        backend, task = self.env.backend, self.env.task
        state = backend.state()
        owners = getattr(backend, "grasp_owners", {})
        self.planner.ee_position = state.ee_position.copy()
        self.planner.ee_orientation = np.asarray(state.ee_orientation, dtype=float).copy()
        self.planner.held = tuple(name for name, side in owners.items() if side == 0)
        joints = observation["joint_positions"]
        module = self.bimanual
        if module is not None:
            position, orientation = backend._tool_pose(backend.secondary_robot)
            self.secondary_planner.ee_position = np.asarray(position, dtype=float).copy()
            self.secondary_planner.ee_orientation = np.asarray(orientation, dtype=float).copy()
            self.secondary_planner.held = tuple(n for n, side in owners.items() if side == 1)
        command = self.planner.command(task, state)
        if command is None:
            # B1-B3 before their cue: hold the joints the episode started with (the env's start pose).
            primary = self.waiting_joints[:DOF].copy()
        else:
            self.finger = self._finger(self.planner, command, self.finger)
            primary = self._solve(command, joints[:DOF], self.solver, self.finger, self.planner)
        secondary = self.waiting_joints[DOF:]
        if module is not None:
            planner = self.secondary_planner
            if planner.last_goal is None:
                planner.last_goal = planner.ee_position.copy()
                planner.yaw = float(task.layout.info.get("initial_secondary_ee_yaw", 0.0))
            command2 = module.command_secondary(planner, task, state)
            if command2 is not None:
                self.secondary_finger = self._finger(planner, command2, self.secondary_finger)
                secondary = self._solve(
                    command2, joints[DOF:], self.secondary_solver, self.secondary_finger, planner
                )
        return np.concatenate([primary, secondary])

    def _finger(self, planner, command, finger):
        target = planner.close_width if command.close else planner.open_width
        return ramp_finger(finger, target, self.dt) if planner.gentle else target

    def _solve(self, command, joints, solver, finger, planner=None):
        """Bounded joint targets toward the IK solution of the command's tool pose.

        A planner may opt into two refinements (C4 does, the other tasks keep the per-joint steps):
        ``sync_joints`` scales the whole step so every joint arrives together (a joint-space straight
        line, so a pointing-down tool stays pointing down on the way), and ``tool_tilt``, a
        world-frame quaternion applied on top of the pointing-down orientation (to stand up an item
        that has pivoted between the fingers). With ``sync_joints`` an unsolved pose also holds the
        last joint targets instead of the measured joints, so the arm does not creep under gravity.
        """
        from isaacsim.core.utils.rotations import euler_angles_to_quat

        joints = joints.copy()
        orientation = euler_angles_to_quat(np.array([0.0, np.pi, command.yaw]))
        tilt = getattr(planner, "tool_tilt", None)
        if tilt is not None:
            orientation = quat_multiply(np.asarray(tilt, dtype=float), orientation)
        action, solved = solver.compute_inverse_kinematics(
            target_position=command.position,
            target_orientation=orientation,
        )
        if solved:
            indices = (
                action.joint_indices
                if action.joint_indices is not None
                else range(len(action.joint_positions))
            )
            speed = self.env.config["control"]["joint_target_speed"]
            targets = {
                index: value
                for index, value in zip(indices, action.joint_positions)
                if value is not None
            }
            limits = {index: min(speed, VELOCITIES[index]) * self.dt for index in targets}
            joints = joint_step(joints, targets, limits, getattr(planner, "sync_joints", False))
        elif (
            getattr(planner, "sync_joints", False)
            and getattr(planner, "joint_target", None) is not None
        ):
            joints[:ARM_DOF] = planner.joint_target[:ARM_DOF]
        joints[6:] = finger
        joints = np.clip(joints, JOINT_LOW, JOINT_HIGH)
        if planner is not None:
            planner.joint_target = joints.copy()
        return joints


def joint_step(joints, targets, limits, sync=False):
    """Joint targets one control period on from `joints` toward `targets` ({index: value}), each joint
    moving at most limits[index]: clipped joint by joint, or with `sync` the whole step scaled down
    so every joint arrives together (a straight line in joint space)."""
    joints = joints.copy()
    deltas = {index: value - joints[index] for index, value in targets.items()}
    if sync:
        scale = min([1.0] + [limits[i] / abs(delta) for i, delta in deltas.items() if delta])
        for index, delta in deltas.items():
            joints[index] += delta * scale
    else:
        for index, delta in deltas.items():
            joints[index] += np.clip(delta, -limits[index], limits[index])
    return joints
