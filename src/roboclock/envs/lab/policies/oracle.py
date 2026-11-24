"""Privileged task-specific trajectory diagnostic using Isaac Sim Lula IK and real motor control.

`reset` / `act` speak the env interface (rig-left arm first, LabEnv.external_arm_order).
The planners work in the backend's internal side order (side 0 = /World/ARXLeft = rig right,
the single-arm worker); `reset_internal` / `act_internal` are that order, for scripts that
drive `env.backend` directly. `ik_failures` and `diagnostics()["arms"]` are internal order.
"""

import numpy as np

from roboclock.envs.lab.env import JOINT_HIGH, JOINT_LOW
from roboclock.envs.lab.policies.trajectories import LabTrajectory
from roboclock.envs.lab.robot import (
    EXTERNAL_ARMS,
    RIG_LEFT_ARM,
    arms_to_external,
    observation_to_internal,
)


GRIP_SQUEEZE = 0.0085  # m inside the measured fingers: 8.5 N from the 1000 N/m drives


class OraclePickPlace:
    def __init__(self, env):
        from roboclock.envs.lab.robot import make_solver

        self.env = env
        self.waiting_joints = env.backend.joint_positions().copy()
        self.last_primary = self.waiting_joints[:8].copy()
        self.last_commands = self.waiting_joints.reshape(2, 8).copy()
        self.ik_failures = [0, 0]
        self.dt = env.config["sim"]["physics_dt"] * env.config["sim"]["control_decimation"]
        self.solver = make_solver(env.backend.robot)
        self.secondary_solver = make_solver(env.backend.secondary_robot)
        if getattr(env.task, "liquid_process", False):
            from roboclock.envs.lab.policies.liquid import LiquidTrajectory

            self.planner = LiquidTrajectory(self.dt)
        elif getattr(env.task, "bath_task", False):
            from roboclock.envs.lab.policies.bath_transfer import BathTrajectory

            self.planner = BathTrajectory(self.dt)
        elif getattr(env.task, "safety_task", False):
            from roboclock.envs.lab.policies.safety_interlocks import SafetyTrajectory

            self.planner = SafetyTrajectory(self.dt)
        elif getattr(env.task, "tube_task", False):
            from roboclock.envs.lab.policies.tube_rescue import TubeRescueTrajectory

            self.planner = TubeRescueTrajectory(self.dt)
        elif getattr(env.task, "power_task", False):
            from roboclock.envs.lab.policies.power_isolation import PowerIsolationTrajectory

            self.planner = PowerIsolationTrajectory(self.dt)
        elif getattr(env.task, "rack_organization", False):
            from roboclock.envs.lab.policies.rack_organization import RackOrganizationTrajectory

            self.planner = RackOrganizationTrajectory(env.config["control"], self.dt)
        else:
            self.planner = LabTrajectory(env.config["control"], self.dt)
        if isinstance(self.planner, LabTrajectory):
            from roboclock.envs.lab.robot import mount_positions

            # Reach-aware selection needs the arm's shoulder; the planner only sees poses.
            self.planner.base = np.asarray(
                mount_positions(env.config, env.task.task_id)[0], dtype=float
            )
        self.bimanual = None
        if env.task.task_id in {"C1", "C2", "C3"} and env.task.task_id in env.config.get(
            "robot", {}
        ).get("cooperative_tasks", ["C2", "C3"]):
            from roboclock.envs.lab.policies.bimanual import BimanualThroughput

            self.bimanual = BimanualThroughput(env, self.dt)
            self.bimanual.solvers = (self.solver, self.secondary_solver)

    @property
    def arm_order(self):
        """Internal side of each interface arm (LabEnv.external_arm_order)."""
        return getattr(self.env, "external_arm_order", EXTERNAL_ARMS)

    def reset(self, observation):
        """Reset from an env observation (interface order)."""
        self.reset_internal(observation_to_internal(observation, self.arm_order))

    def act(self, observation):
        """Interface-order action (rig-left arm first) for `env.step`."""
        internal = observation_to_internal(observation, self.arm_order)
        return arms_to_external(self.act_internal(internal), self.arm_order)

    def reset_internal(self, observation):
        self.waiting_joints = np.asarray(observation["joint_positions"]).copy()
        self.last_primary = self.waiting_joints[:8].copy()
        self.last_commands = self.waiting_joints.reshape(2, 8).copy()
        self.ik_failures = [0, 0]
        if getattr(self.env.task, "safety_task", False):
            target = getattr(getattr(self.env, "backend", None), "initial_hold_joints", None)
            if target is not None:
                self.waiting_joints = target.copy()
        if hasattr(self.planner, "reset"):
            self.planner.reset()
        if getattr(self, "bimanual", None) is not None:
            self.bimanual.reset()

    def act_internal(self, observation):
        """Internal side-order action from internal side-order joint_positions."""
        # Hold the original joints until the observed event, including the wrist
        # and fingers. Cartesian position holding can still rotate the arm via IK.
        # An optional per-task perception delay keeps the first motion after the
        # cue is visible, instead of on the same control tick as the fault.
        task = self.env.task
        reactive = task.task_id in ("B1", "B2", "B3", "B4") or getattr(task, "safety_task", False)
        if reactive and (
            task.event_time is None or task.time < task.event_time + self.cue_delay(task) - 1e-9
        ):
            if not hasattr(self, "waiting_joints"):
                self.reset_internal(observation)
            return self.waiting_joints.copy()
        # A cue that only appears during the episode (A4's first pink): same hold.
        waiting = getattr(self.planner, "waiting_for_cue", None)
        if waiting is not None and waiting(task):
            return self.waiting_joints.copy()

        from isaacsim.core.utils.rotations import euler_angles_to_quat

        if getattr(self, "bimanual", None) is not None:
            commands = self.bimanual.commands()
            self.last_cartesian = [np.asarray(c.position, dtype=float).copy() for c in commands]
            return np.concatenate(
                [
                    self._solve_command(
                        command,
                        observation["joint_positions"][8 * side : 8 * (side + 1)],
                        solver,
                        side,
                    )
                    for side, (command, solver) in enumerate(
                        zip(commands, (self.solver, self.secondary_solver))
                    )
                ]
            )

        state = self.env.backend.state()
        previous_goal = getattr(self.planner, "last_goal", None)
        previous_goal = None if previous_goal is None else np.array(previous_goal, dtype=float)
        command = self.planner.command(self.env.task, state)
        if self.env.task.task_id == "C1" and self.env.config["control"].get(
            "c1_expert_grasp_assist", False
        ):
            from roboclock.envs.lab.rack_organization import update_expert_grasp

            update_expert_grasp(self.env.backend, self.planner.sample, command.close, state)
        open_width = (
            0.020
            if self.env.task.task_id == "B5"
            else 0.025
            if getattr(self.env.task, "rack_organization", False)
            else 0.04
        )
        joints = observation["joint_positions"][:8].copy()
        ik_position = command.position
        ik_orientation = (
            command.orientation
            if getattr(command, "orientation", None) is not None
            else euler_angles_to_quat(np.array([0.0, np.pi, command.yaw]))
        )
        action, solved = self.solver.compute_inverse_kinematics(
            target_position=ik_position,
            target_orientation=ik_orientation,
        )
        if not solved:
            self.ik_failures[0] += 1
            if previous_goal is not None and hasattr(self.planner, "last_goal"):
                # Hold the Cartesian goal with the arm: a goal that keeps advancing
                # while IK fails leaves the planner waiting at the wrong place.
                self.planner.last_goal = previous_goal
            joints = self.last_primary.copy()
            joints[6:] = (
                0.0
                if command.close
                else 0.027
                if self.env.task.task_id == "B3" and not getattr(self.env.task, "power_task", False)
                else open_width
            )
            return self._combine(np.clip(joints, JOINT_LOW, JOINT_HIGH))
        indices = (
            action.joint_indices
            if action.joint_indices is not None
            else range(len(action.joint_positions))
        )
        for index, value in zip(indices, action.joint_positions):
            if value is not None:
                step = self.joint_target_speed() * self.dt
                joints[index] += np.clip(value - joints[index], -step, step)
        joints[6:] = (
            0.0
            if command.close
            else 0.027
            if self.env.task.task_id == "B3" and not getattr(self.env.task, "power_task", False)
            else open_width
        )
        self.last_primary = np.clip(joints, JOINT_LOW, JOINT_HIGH)
        return self._combine(self.last_primary)

    def joint_target_speed(self):
        """Joint target lead rate; `control.task_joint_target_speed` overrides it per task."""
        control = self.env.config["control"]
        return control.get("task_joint_target_speed", {}).get(
            self.env.task.task_id, control["joint_target_speed"]
        )

    def cue_delay(self, task):
        return self.env.config["control"].get("cue_reaction_delay", {}).get(task.task_id, 0.0)

    def _solve_command(self, command, joints, solver, side=0):
        from isaacsim.core.utils.rotations import euler_angles_to_quat

        from roboclock.envs.lab.robot import VELOCITIES

        joints = joints.copy()
        fingers = float(np.mean(joints[6:8]))
        orientation = (
            command.orientation
            if command.orientation is not None
            else euler_angles_to_quat(np.array([0.0, np.pi, command.yaw]))
        )
        action, solved = solver.compute_inverse_kinematics(
            target_position=command.position,
            target_orientation=orientation,
        )
        if solved:
            indices = action.joint_indices
            if indices is None:
                indices = range(len(action.joint_positions))
            speed = self.env.config["control"]["joint_target_speed"]
            for index, value in zip(indices, action.joint_positions):
                if value is not None:
                    step = min(speed, VELOCITIES[index]) * self.dt
                    joints[index] += np.clip(value - joints[index], -step, step)
        else:
            self.ik_failures[side] += 1
            if self.bimanual is not None:
                self.bimanual.ik_failed(side)
            joints = self.last_commands[side].copy()
        # Narrow openings in dense grids: C1 tubes (28 mm), C2 vials (36 mm) and C3
        # blocks (40 mm) sit 60 mm apart, so a fully open X5 hand (80 mm) touches
        # the neighbours.
        width = getattr(command, "gripper", None)
        if command.close:
            # Drives pressing at their effort limit cancel and leave nothing to
            # recenter a held tube, which then slides along the jaws in transit
            # and escapes. Below the limit they act as a centering spring.
            joints[6:] = max(0.0, fingers - GRIP_SQUEEZE)
        elif width is not None:
            # Releasing: the fingers close independently, so a held object sits
            # off-centre. Commanding both to one width moves the wider finger
            # inward and pushes the object; open each finger, never close it.
            joints[6:] = np.maximum(joints[6:], width)
        else:
            joints[6:] = {"C1": 0.025, "C2": 0.028, "C3": 0.03}.get(self.env.task.task_id, 0.04)
        self.last_commands[side] = np.clip(joints, JOINT_LOW, JOINT_HIGH)
        return self.last_commands[side].copy()

    def external_ik_failures(self):
        """IK failure counts per arm in interface order (rig left first)."""
        return [int(self.ik_failures[side]) for side in self.arm_order]

    def diagnostics(self):
        """JSON-serializable planner state for recorded process traces.

        Per-arm lists are in interface order (rig left first); each arm names its rig side.
        Single-arm tasks plan only internal side 0, the rig's right arm.
        """
        planners = self.bimanual.planners if self.bimanual is not None else [self.planner]
        sides = [side for side in self.arm_order if side < len(planners)]
        yielding = list(self.bimanual.yielding) if self.bimanual is not None else []
        commanded = [p.tolist() for p in getattr(self, "last_cartesian", [])]
        return {
            "arms": [
                {
                    "rig_side": "left" if side == RIG_LEFT_ARM else "right",
                    "sample": getattr(planners[side], "sample", None),
                    "phase": getattr(planners[side], "phase", None),
                    "returning": bool(getattr(planners[side], "returning", False)),
                }
                for side in sides
            ],
            "ik_failures": self.external_ik_failures(),
            "yielding": [yielding[side] for side in self.arm_order] if yielding else [],
            "commanded": [commanded[side] for side in self.arm_order] if commanded else [],
        }

    def _combine(self, primary):
        return np.concatenate([primary, self.waiting_joints[8:]])
