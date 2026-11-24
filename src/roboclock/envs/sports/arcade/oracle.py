"""Feedback IK with a cross-handle pinch and wrist-driven striking arc."""

import math

import numpy as np

from roboclock.envs.sports.joints import joint_step_speed

# Control rate the X5 joint speed limits were tuned at.
TUNED_HZ = 30
# Time constant of the retained-head offset re-estimate after impacts (s).
OFFSET_TIME_CONSTANT = 0.149
# Tool height after the pickup (m): the lift ends here, above the tool's rest.
LIFT_HEIGHT = 1.12


def pitch_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def mole_target_expired(task, index, phase, start, aim_timeout):
    """Never let an unreachable or retracted mole monopolize an arm."""
    return not task.mole_active(index) or (phase == "aim" and task.time - start > aim_timeout)


class ToolOracle:
    def __init__(self, env):
        self.env = env
        self.phase, self.start = "approach", 0
        self.commanded = None
        self.pitch = 0.0
        self.offset = None
        self.event_index = None
        self.finished = set()
        self.home = None
        self.had_grasp = False

    def transition(self, phase):
        self.phase, self.start = phase, self.env.task.time

    def act(self, observation):
        from isaacsim.core.utils.rotations import euler_angles_to_quat

        from roboclock.physics.geometry import rotation_matrix

        b, t = self.env.backend, self.env.task
        s = b.state()
        ee = s["tool_position"]
        o = s["objects"]["mallet"]
        p = o["position"]
        dt = self.env.config["sim"]["physics_dt"] * self.env.config["sim"]["control_decimation"]
        grip = p + rotation_matrix(o["orientation"]) @ np.array([0, 0, 0.020])
        target, close, angle = ee.copy(), False, 0.0
        native_moles = t.task_id == "34"
        if native_moles and self.event_index is not None:
            i = self.event_index
            # An unreachable target must not block every later round.
            expired = mole_target_expired(
                t, i, self.phase, self.start, self.env.config["v2"].get("mole_aim_timeout", 1.6)
            )
            if expired:
                t.event(
                    "mallet_target_abandoned",
                    index=i,
                    side=b.active_side,
                    reason="event_expired" if not t.mole_active(i) else "aim_timeout",
                )
                self.finished.add(i)
                self.event_index = None
                self.transition("wait")
        native_dual = t.task_id == "32"
        ready_angle, hit_angle = (0.0, 0.0) if native_dual else (np.deg2rad(-25), np.deg2rad(22))
        if native_moles:
            ready_angle = np.deg2rad(self.env.config["v2"].get("mole_ready_angle", -25))
            hit_angle = np.deg2rad(self.env.config["v2"].get("mole_hit_angle", 22))
            if self.offset is not None and o["grasped"]:
                # Re-estimate the retained head offset after impacts; a fixed pickup
                # calibration misses by millimeters after the handle settles in the jaws.
                _, actual_q = b.rig.pose(b.active_side)
                local = rotation_matrix(actual_q).T @ (s["tool_head_position"] - ee)
                measured = pitch_matrix(np.pi) @ local
                self.offset += (1 - math.exp(-dt / OFFSET_TIME_CONSTANT)) * (measured - self.offset)
        if self.phase == "approach":
            target = np.r_[grip[:2], 1.08]
            if np.linalg.norm(target - ee) < 0.008:
                self.transition("descend")
        elif self.phase == "descend":
            target = grip
            if np.linalg.norm(target - ee) < 0.005:
                self.home = target.copy()
                self.transition("grasp")
        elif self.phase == "grasp":
            target, close = self.home.copy(), True
            if o["grasped"]:
                self.offset = s["tool_head_position"] - ee
                self.had_grasp = True
                self.transition("lift")
        elif self.phase == "lift":
            close = True
            target = np.r_[self.home[:2], LIFT_HEIGHT]
            if ee[2] > 1.105:
                self.transition("wait")
        else:
            close, angle = True, ready_angle
            if self.event_index is None:
                if t.task_id == "32":
                    self.event_index = next(
                        (
                            i
                            for i, e in enumerate(t.events)
                            if i not in self.finished
                            and e["time"] + t.limits["note_window"] >= t.time
                            and e["time"] - t.time <= 5
                        ),
                        None,
                    )
                else:
                    i = t.current_index
                    if i is not None and i not in self.finished and t.events[i]["valid"]:
                        self.event_index = i
                if self.event_index is not None:
                    self.transition("aim")
            if self.event_index is None:
                idle = getattr(b, "idle_target", [0.33, -0.08, 1.06])
                # No idle target: hold the pose the pickup ended in (above the tool's rest).
                target = np.r_[self.home[:2], LIFT_HEIGHT] if idle is None else np.array(idle)
            else:
                i = self.event_index
                e = t.events[i]
                # Shorter rebound for closely spaced roll notes; preserve separate contacts.
                if t.task_id == "32" and (
                    e.get("kind") == "roll"
                    or (i + 1 < len(t.events) and t.events[i + 1].get("kind") == "roll")
                ):
                    ready_angle = np.deg2rad(-12)
                    angle = ready_angle
                sp = t.layout.controls[e["control"]]
                contact = np.array(
                    [
                        *sp.position[:2],
                        sp.position[2]
                        + sp.height / 2
                        + b.head_radius
                        - (
                            self.env.config["v2"].get("mole_contact_depth", 0.005)
                            if native_moles
                            else 0.005
                        ),
                    ]
                )
                wrist = (
                    contact - (s["tool_head_position"] - ee)
                    if native_dual
                    else contact - pitch_matrix(hit_angle) @ self.offset
                )
                target = wrist.copy()
                if native_dual and self.phase != "strike":
                    target[2] += 0.05
                if self.phase == "aim":
                    ready = (
                        t.time >= e["time"] - 0.35 if t.task_id == "32" else i in t.mole_up_times
                    )
                    v2 = self.env.config["v2"]
                    aim_tolerance = v2.get("mole_aim_tolerance", 0.014) if native_moles else 0.014
                    pitch_tolerance = v2.get("mole_pitch_tolerance", 0.02) if native_moles else 0.02
                    if (
                        np.linalg.norm(target - ee) < aim_tolerance
                        and abs(self.pitch - ready_angle) < pitch_tolerance
                        and ready
                    ):
                        self.transition("strike")
                elif self.phase == "strike":
                    angle = hit_angle
                    if i in t.correct or t.time - self.start > 0.85:
                        self.finished.add(i)
                        if native_dual:
                            angle = ready_angle
                            target[2] += 0.05
                        self.transition("retract")
                elif self.phase == "retract":
                    angle = ready_angle
                    if abs(self.pitch - ready_angle) < 0.02:
                        self.event_index = None
                        self.transition("wait")
        # Coordinate a bounded wrist angle with translation. The tool pose remains
        # entirely governed by opposing jaw forces, friction, gravity and impacts.
        pitch_speed = 1.4 if native_dual else 2.3
        if native_moles:
            pitch_speed = self.env.config["v2"].get("mole_pitch_speed", pitch_speed)
        self.pitch += np.clip(angle - self.pitch, -pitch_speed * dt, pitch_speed * dt)
        origin = ee if self.commanded is None else self.commanded
        delta = target - origin
        speed = self.env.config["v2"].get("mole_cartesian_speed", 0.23) if native_moles else 0.23
        self.commanded = origin + delta * min(1, speed * dt / max(np.linalg.norm(delta), 1e-9))
        q = euler_angles_to_quat(np.array([0, np.pi + self.pitch, 0]))
        # Native robot IK targets its measured grasp frame directly.
        rig = b.rig
        angles, error = rig.kin[b.active_side].solve(
            self.commanded,
            rotation_matrix(q),
            seed=rig.robot.get_joint_positions()[rig.arm_ids[b.active_side]],
        )
        actual = rig.robot.get_joint_positions()
        ids = rig.arm_ids[b.active_side]
        joint_speed = joint_step_speed(
            self.env.config["v2"].get("mole_joint_speed", 1.7) if native_moles else 1.7,
            TUNED_HZ,
            dt,
        )
        rig.command[ids] = actual[ids] + np.clip(
            angles - actual[ids], -joint_speed * dt, joint_speed * dt
        )
        rig.grip(b.active_side, 0 if close else 0.04)
        self.diagnostics = dict(
            ik_solved=error < 0.02,
            ik_error=float(error),
            event_index=self.event_index,
            target=self.commanded.tolist(),
            pitch=float(self.pitch),
            phase=self.phase,
        )
        return rig.command.copy()
