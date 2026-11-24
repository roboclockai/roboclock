"""Privileged feedback controller; only articulation drives move objects."""

import numpy as np

from roboclock.envs.sports.joints import joint_step_speed

# Control rate the X5 joint speed limit was tuned at.
TUNED_HZ = 30


class RefinedChessOracle:
    def __init__(self, env):
        from roboclock.simulators.isaac.embodiments.solver import ChessIKSolver

        self.env = env
        self.solver = ChessIKSolver(env.backend.rig, env.backend.active_side)
        self.dt = env.config["sim"]["physics_dt"] * env.config["sim"]["control_decimation"]
        self.joint_speed = env.config["control"]["joint_target_speed"]
        self.joint_speed = joint_step_speed(self.joint_speed, TUNED_HZ, self.dt)

    def reset(self, observation):
        self.phase = "wait"
        self.index = 0
        self.phase_start = 0.0
        self.grip_offset = None
        self.commanded_target = None
        self.placement_ready_since = None
        self.diagnostics = {}
        self.phase_origin = self.env.backend.state()["tool_position"].copy()

    def transition(self, phase):
        self.phase = phase
        self.phase_start = self.env.task.time
        self.placement_ready_since = None
        self.phase_origin = self.env.backend.state()["tool_position"].copy()

    def act(self, observation):
        from isaacsim.core.utils.rotations import euler_angles_to_quat

        task = self.env.task
        state = self.env.backend.state()
        ee = state["tool_position"]
        z = self.env.config["control"]["transit_height"]
        close = False
        grasp_height = self.env.config["control"].get("grasp_height_offset", 0.004)
        target = ee.copy()
        if self.phase == "wait":
            if task.opponent_time is None:
                return observation["joint_positions"].copy()
            self.transition("approach")
        if self.index < len(task.operations):
            op = task.operations[self.index]
            obj = state["objects"][op.piece]
            p = obj["position"]
            dest = task.position(op.destination)
            if self.phase == "approach":
                target = np.array([p[0], p[1], z])
                if np.linalg.norm(target - ee) < 0.009:
                    self.transition("descend")
            elif self.phase == "descend":
                target = p + np.array([0, 0, grasp_height])
                if np.linalg.norm(target - ee) < 0.006:
                    self.transition("grasp")
            elif self.phase == "grasp":
                target = p + np.array([0, 0, grasp_height])
                close = True
                if obj["grasped"]:
                    self.grip_offset = p - ee
                    self.transition("lift")
                elif task.time - self.phase_start > 2.0:
                    self.transition("approach")
            elif self.phase == "lift":
                target = np.array([self.phase_origin[0], self.phase_origin[1], z])
                close = True
                if abs(ee[2] - z) < 0.009:
                    self.transition("transfer")
            elif self.phase == "transfer":
                target = np.array([dest[0] - self.grip_offset[0], dest[1] - self.grip_offset[1], z])
                close = True
                if np.linalg.norm(target - ee) < 0.009:
                    self.transition("lower")
            elif self.phase == "lower":
                if task.initial_symbols[op.piece].lower() != "p":
                    # Keep the established release trajectory for non-pawn meshes:
                    # their asymmetric profiles need different contact handling.
                    target = dest - self.grip_offset + np.array([0, 0, 0.001])
                    close = True
                    if np.linalg.norm(target - ee) < 0.004:
                        self.transition("release")
                else:
                    # Refresh the actual lever/offset while held, rather than trusting
                    # the offset measured during the first jaw contact.
                    self.grip_offset = p - ee
                    goal = dest + np.array([0, 0, 0.001])
                    target = goal - self.grip_offset
                    close = True
                    ready = (
                        np.linalg.norm(p[:2] - dest[:2]) < 0.004
                        and abs(p[2] - goal[2]) < 0.002
                        and np.linalg.norm(obj["velocity"]) < 0.020
                        and np.linalg.norm(obj["angular_velocity"]) < 0.25
                        and obj["supported"]
                    )
                    # A supported, released and fully valid placement is already
                    # sufficient. Do not deadlock chasing sub-IK-tolerance centering
                    # after the board has taken the load.
                    ready = ready or task.located(op.piece, obj)
                    if getattr(getattr(self.env.backend, "rig", None), "model", None) == "dual_arx_x5":
                        # Release a held pawn within 2 mm of its supported height.
                        # Requiring support before opening deadlocks a precise pinch
                        # that still carries the full weight. The evaluator still
                        # independently requires released, stable board support.
                        ready = ready or (
                            np.linalg.norm(p[:2] - dest[:2]) < 0.003
                            and abs(p[2] - dest[2]) < 0.002
                            and np.linalg.norm(target - ee) < 0.004
                        )
                    if ready:
                        if self.placement_ready_since is None:
                            self.placement_ready_since = task.time
                        elif task.time - self.placement_ready_since >= self.env.config["control"].get(
                            "placement_settle_chess", 0.10
                        ):
                            self.transition("release")
                    else:
                        self.placement_ready_since = None
            elif self.phase == "release":
                target = dest - self.grip_offset + np.array([0, 0, 0.001])
                if task.time - self.phase_start > self.env.config["control"].get("release_time_chess", 0.35):
                    self.transition("retreat")
            elif self.phase == "retreat":
                target = np.array([self.phase_origin[0], self.phase_origin[1], z])
                if abs(ee[2] - z) < 0.01:
                    if task.located(op.piece, obj):
                        self.index += 1
                    self.transition(
                        "approach" if self.index < len(task.operations) else "clock_approach"
                    )
        else:
            button = self.env.backend.button_rest["robot"]
            if self.phase == "clock_approach":
                target = np.array([button[0], button[1], z])
                close = True
                if task.placement_valid and np.linalg.norm(target - ee) < 0.009:
                    self.transition("press")
            elif self.phase == "press":
                target = button + np.array([0.0, 0.0, -0.010])
                close = True
        joints = observation["joint_positions"].copy()
        origin = ee if self.commanded_target is None else self.commanded_target
        delta = target - origin
        distance = np.linalg.norm(delta)
        control = self.env.config["control"]
        speed = control.get("cartesian_speed_chess", 0.22)
        if self.phase in ("descend", "grasp") or (self.phase == "press"):
            speed = control.get("vertical_speed_chess", speed)
        if self.phase == "lower" and task.initial_symbols[op.piece].lower() == "p":
            # Brake before board contact; retain the same actuator bounds.
            speed = (
                control.get("brake_speed_chess", 0.045)
                if p[2] - dest[2] < control.get("brake_height_chess", 0.035)
                else control.get("lower_speed_chess", 0.16)
            )
        max_step = speed * self.dt
        self.diagnostics = dict(operation_index=self.index, phase=self.phase, target=target.copy())
        if self.index < len(task.operations):
            self.diagnostics.update(
                piece=op.piece, destination=op.destination, position_error=(p - dest).copy()
            )
        target = target if distance <= max_step else origin + delta * max_step / distance
        self.commanded_target = target.copy()
        # Targets carry a 3.4 mm offset that ChessIKSolver removes for the X5 grasp frame.
        action, solved = self.solver.compute_inverse_kinematics(
            target_position=target + [0, 0, 0.0034],
            target_orientation=euler_angles_to_quat(
                np.array([0.0, np.pi, self.env.config["control"].get("tool_yaw", np.pi)])
            ),
        )
        if solved:
            indices = (
                action.joint_indices
                if action.joint_indices is not None
                else range(len(action.joint_positions))
            )
            limit = self.joint_speed * self.dt
            moves = [
                (i, value - joints[i])
                for i, value in zip(indices, action.joint_positions)
                if value is not None
            ]
            scale = min(1, limit / max(1e-12, max(abs(delta) for _, delta in moves)))
            for i, delta in moves:
                joints[i] += delta * scale
        rig = self.env.backend.rig
        rig.set_arm_command(self.env.backend.active_side, joints)
        rig.grip(
            self.env.backend.active_side,
            0.0 if close else self.env.config["control"].get("grasp_open_half_width", 0.019),
        )
        return rig.command.copy()
