"""Keyboard teleoperation of the Dual ARX X5 from a browser, as a roboclock policy.

Each control step `act` reads the keys the browser holds, moves the selected arm's tool-point
target in the world frame, solves the arm's joints for it (pure-Python kinematics, calibrated
from the first observation like the move_eef agent) and returns the joint targets. Camera
frames go to the browser every TELEOP_SPEED / TELEOP_FPS seconds of simulation time (TELEOP_FPS
frames per wall second when the simulator keeps pace); reading them renders that observation, and
rendering the rig's three cameras costs about 115 ms on an L40S (household A1), against about
22 ms of physics per control step.

Pacing. The operator's thinking time must not leave the task clock unfair, so the policy
reports zero inference latency and paces the simulation itself:

- realtime (speed 1.0): simulation time follows wall time, or falls behind it when the
  simulator cannot keep up (the status shows the achieved ratio);
- slow motion (speed < 1): one simulated second takes 1/speed wall seconds;
- frozen: the episode starts frozen and P freezes it at any time; while frozen no simulation
  time passes, and "." advances it by STEP_SECONDS.

Run with ``roboclock eval ... --policy roboclock.policies.teleop:make``.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.lab.robot import ARM_DOF, DOF, JOINT_HIGH, JOINT_LOW
from roboclock.envs.rates import CONTROL_DT
from roboclock.evaluation.evaluator import RestartEpisode
from roboclock.policies.teleop.server import get_server
from roboclock.policies.x5_kinematics import X5Kinematics

SIDES = ("left", "right")
FINGER_OPEN = float(JOINT_HIGH[ARM_DOF])  # m per finger, jaws fully open
GRIPPER_TIME = 0.5  # s for a full open or close
STEP_SECONDS = 0.1  # s of simulation one "." advances while frozen
SPEEDS = (0.1, 0.25, 0.5, 1.0)
IK_ITERATIONS = 20
TRACK_LIMIT = 0.02  # m the solved tool point may lag its target before the target is pulled back
BEHIND_RESET = 0.25  # s of wall time the simulation may lag the pace before the pace is reset

# KeyboardEvent.code -> (linear direction in the operator frame: forward, left, up) or
# (angular axis in the operator frame) for the selected arm(s).
LINEAR = {
    "KeyW": (1, 0, 0), "KeyS": (-1, 0, 0),
    "KeyA": (0, 1, 0), "KeyD": (0, -1, 0),
    "KeyR": (0, 0, 1), "KeyF": (0, 0, -1),
}
ANGULAR = {
    "KeyQ": (0, 0, 1), "KeyE": (0, 0, -1),  # yaw about world up
    "KeyI": (0, 1, 0), "KeyK": (0, -1, 0),  # pitch about the operator's left axis
    "KeyJ": (-1, 0, 0), "KeyL": (1, 0, 0),  # roll about the operator's forward axis
}
FINE = ("ShiftLeft", "ShiftRight")

_EPISODES = 0  # episodes this process has teleoperated (the evaluator makes a policy per episode)


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


class TeleopPolicy:
    def __init__(self):
        self.server = get_server(int(os.environ.get("TELEOP_PORT", "8765")))
        self.fps = env_float("TELEOP_FPS", 10.0)
        self.speed = env_float("TELEOP_SPEED", 1.0)
        self.linear_speed = env_float("TELEOP_LINEAR", 0.12)  # m/s, tool point
        self.angular_speed = env_float("TELEOP_ANGULAR", 0.8)  # rad/s
        self.fine_scale = env_float("TELEOP_FINE", 0.25)  # Shift held
        self.joint_speed = env_float("TELEOP_JOINT_SPEED", 0.6)  # rad/s, under every rig's cap
        log = os.environ.get("TELEOP_LOG")
        self.log_dir = Path(log) if log else None
        if not (0 < self.speed <= 1 and self.fps > 0):
            raise ValueError("TELEOP_SPEED must be in (0, 1] and TELEOP_FPS > 0")
        self.kinematics = X5Kinematics()
        self.finger_min, self.finger_max = 0.0, FINGER_OPEN
        # Operator inputs are not a model: nothing is charged, the policy paces the clock itself.
        self.last_inference_latency = 0.0
        self.queried = True

    def set_spaces(self, action_space, observation_space):
        low = np.asarray(action_space.low, dtype=float)
        high = np.asarray(action_space.high, dtype=float)
        self.finger_min = min(0.0, float(low[ARM_DOF]))  # closes as far as the object allows
        self.finger_max = float(high[ARM_DOF])

    # ------------------------------------------------------------------ episode
    def reset(self, observation):
        global _EPISODES
        _EPISODES += 1
        self.server.pop_commands()  # clicks from the previous attempt do not carry over
        self.episode = _EPISODES
        joints = np.asarray(observation["joint_positions"], dtype=float)
        measured = np.asarray(observation["ee_pose"], dtype=float)
        self.arms = [
            self.kinematics.calibrated(
                joints[DOF * i : DOF * i + ARM_DOF], measured[7 * i : 7 * i + 3], measured[7 * i + 3 : 7 * i + 7]
            )
            for i in range(2)
        ]
        forward = self.arms[0].base[:3, 0] + self.arms[1].base[:3, 0]
        forward[2] = 0.0
        forward /= max(np.linalg.norm(forward), 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        # Columns: the operator's forward, left and up axes in the world frame.
        self.frame = np.column_stack([forward, np.cross(up, forward), up])
        self.command = [joints[DOF * i : DOF * i + ARM_DOF].copy() for i in range(2)]
        self.home = [command.copy() for command in self.command]
        self.targets = [arm.tool_pose(command) for arm, command in zip(self.arms, self.command)]
        self.fingers = [float(joints[DOF * i + ARM_DOF]) for i in range(2)]
        self.gripper_goal = list(self.fingers)
        self.homing = [False, False]
        self.selected = (0,)
        self.instruction = str(observation["instruction"])
        self.started = False
        self.frozen = True
        self.step_until = None
        self.last_time = None
        self.last_frame = -math.inf
        self.pace = None  # (wall, sim) the pace is measured from
        self.ratio = None  # achieved simulated seconds per wall second
        self.ratio_mark = None
        self.log = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.log = (self.log_dir / f"episode{self.episode:03d}-{stamp}.jsonl").open("w", encoding="utf-8")
            self._write(event="reset", instruction=self.instruction)
        self.server.publish_status(
            episode=self.episode, instruction=self.instruction, sim_time=self._now(observation),
            state="waiting", speed=self.speed, selected=self._selected_name(), achieved=None,
            gripper=self._gripper_names(),
        )
        print(f"[teleop] episode {self.episode} ready; press Enter in the browser to start", flush=True)

    # ------------------------------------------------------------------ control
    def act(self, observation):
        now = self._now(observation)
        self._commands(now)
        if self.frozen and not self._stepping(now):
            self._publish(observation, now, force=True)
            self._wait_frozen(observation, now)
        dt = CONTROL_DT if self.last_time is None else min(max(now - self.last_time, 0.0), 0.1)
        self.last_time = now
        keys = self.server.held_keys()
        self._move(keys, dt)
        self._grippers(dt)
        action = self._action()
        if self.log is not None:
            self._write(
                t=round(now, 4), keys=sorted(keys), action=np.round(action, 5).tolist(),
                ee_pose=np.round(np.asarray(observation["ee_pose"], dtype=float), 5).tolist(),
                targets=[np.round(target[:3, 3], 5).tolist() for target in self.targets],
            )
        self._publish(observation, now)
        self._pace(now)
        return action

    def _move(self, keys, dt):
        scale = self.fine_scale if any(key in keys for key in FINE) else 1.0
        linear = sum((np.array(LINEAR[key], float) for key in keys if key in LINEAR), np.zeros(3))
        angular = sum((np.array(ANGULAR[key], float) for key in keys if key in ANGULAR), np.zeros(3))
        velocity = self.frame @ linear * self.linear_speed * scale
        spin = self.frame @ angular * self.angular_speed * scale
        limit = self.joint_speed * dt
        for i in range(2):
            if self.homing[i]:
                step = np.clip(self.home[i] - self.command[i], -limit, limit)
                self.command[i] = self.command[i] + step
                self.targets[i] = self.arms[i].tool_pose(self.command[i])
                self.homing[i] = bool(np.abs(self.home[i] - self.command[i]).max() > 1e-4)
                continue
            if i not in self.selected or not (velocity.any() or spin.any()):
                continue
            target = self.targets[i].copy()
            target[:3, 3] += velocity * dt
            if spin.any():
                target[:3, :3] = Rotation.from_rotvec(spin * dt).as_matrix() @ target[:3, :3]
            joints, _ = self.arms[i].solve(target, self.command[i], iterations=IK_ITERATIONS)
            joints = self.command[i] + np.clip(joints - self.command[i], -limit, limit)
            reached = self.arms[i].tool_pose(joints)
            # An unreachable (or joint-speed-limited) target is pulled back to where the arm
            # is, so releasing the key stops the arm instead of letting it chase a far target.
            if np.linalg.norm(reached[:3, 3] - target[:3, 3]) > TRACK_LIMIT:
                target = reached
            self.command[i], self.targets[i] = joints, target

    def _grippers(self, dt):
        rate = (self.finger_max - min(self.finger_min, 0.0)) / GRIPPER_TIME * dt
        for i in range(2):
            self.fingers[i] += float(np.clip(self.gripper_goal[i] - self.fingers[i], -rate, rate))

    def _action(self):
        out = np.zeros(2 * DOF, dtype=np.float32)
        for i in range(2):
            out[DOF * i : DOF * i + ARM_DOF] = np.clip(self.command[i], JOINT_LOW[:ARM_DOF], JOINT_HIGH[:ARM_DOF])
            out[DOF * i + ARM_DOF : DOF * (i + 1)] = min(max(self.fingers[i], self.finger_min), self.finger_max)
        return out

    # ------------------------------------------------------------------ operator commands
    def _commands(self, now):
        for message in self.server.pop_commands():
            command, value = message.get("command"), message.get("value")
            if command == "start":
                self.started, self.frozen = True, False
                self.pace = None
            elif command == "freeze" and self.started:
                self.frozen = not self.frozen
                self.pace = None
            elif command == "step" and self.started and self.frozen:
                self.step_until = now + STEP_SECONDS
            elif command == "speed":
                faster = value == "up"
                index = min(range(len(SPEEDS)), key=lambda k: abs(SPEEDS[k] - self.speed))
                self.speed = SPEEDS[max(0, min(len(SPEEDS) - 1, index + (1 if faster else -1)))]
                self.pace = None
            elif command == "arm" and value in ("left", "right", "both"):
                self.selected = (0, 1) if value == "both" else (SIDES.index(value),)
            elif command == "gripper":
                for i in self.selected:
                    closing = self.gripper_goal[i] > (self.finger_min + self.finger_max) / 2
                    self.gripper_goal[i] = self.finger_min if closing else self.finger_max
            elif command == "restart" and self.started:
                if self.log is not None:
                    self._write(t=round(now, 4), command=command, value=value)
                    self._write(event="restarted")
                    self.log.close()
                    self.log = None
                self.server.publish_status(state="restarting", result=None)
                raise RestartEpisode()
            elif command == "home":
                for i in self.selected:
                    self.homing[i] = True
            else:
                continue
            if self.log is not None:
                self._write(t=round(now, 4), command=command, value=value)
        self.server.publish_status(speed=self.speed, selected=self._selected_name(), gripper=self._gripper_names())

    def _stepping(self, now):
        if self.step_until is None:
            return False
        if now < self.step_until - 1e-9:
            return True
        self.step_until = None
        return False

    def _wait_frozen(self, observation, now):
        """Block (no simulation time passes) until the operator starts, unfreezes or steps."""
        self.server.publish_status(state="waiting" if not self.started else "frozen", sim_time=now)
        while self.frozen and self.step_until is None:
            time.sleep(0.02)
            self._commands(now)
        self.server.publish_status(state="running" if not self.frozen else "stepping")
        self.last_frame = -math.inf

    # ------------------------------------------------------------------ frames and pacing
    def _publish(self, observation, now, force=False):
        # Frames are due every speed / fps of simulation time (fps of wall time when the
        # simulator keeps pace). A wall-clock schedule would render on every step once a
        # step (with its render) took longer than 1 / fps, and the simulation would crawl.
        if not force and now - self.last_frame < self.speed / self.fps - 1e-9:
            return
        self.last_frame = now
        wall = time.monotonic()
        keys = getattr(observation, "image_keys", None) or [
            key for key in observation.keys() if key.endswith("rgb")
        ]
        images = {self._camera_name(key): observation[key] for key in keys if key.endswith("rgb")}
        self.server.publish_frames(images)
        if self.ratio_mark is not None and wall > self.ratio_mark[0]:
            ratio = (now - self.ratio_mark[1]) / (wall - self.ratio_mark[0])
            self.ratio = ratio if self.ratio is None else 0.7 * self.ratio + 0.3 * ratio
        self.ratio_mark = (wall, now)
        self.server.publish_status(
            sim_time=round(now, 3), achieved=None if self.ratio is None else round(self.ratio, 3),
            state="frozen" if self.frozen and self.step_until is None else "running",
        )

    def _pace(self, now):
        """Sleep so that simulation time runs at `speed` times wall time, never faster."""
        if self.frozen:
            return
        wall = time.monotonic()
        if self.pace is None:
            self.pace = (wall, now)
            self.ratio_mark = None
            return
        due = self.pace[0] + (now - self.pace[1]) / self.speed
        if due > wall:
            time.sleep(due - wall)
        elif wall - due > BEHIND_RESET:
            self.pace = (wall, now)  # the simulator is slower than the pace: do not catch up

    def episode_finished(self, row):
        """Called by the evaluator with the episode's result row: shown to the operator, who
        chooses to run the episode again or to continue. Returns "restart" or "continue"."""
        result = {
            key: row.get(key)
            for key in ("task_id", "difficulty", "seed", "task_success", "temporal_success",
                        "joint_success", "termination_reason", "score")
            if key in row
        }
        self.server.publish_status(state="finished", result=result)
        print(f"[teleop] episode {self.episode} finished; restart or continue in the browser", flush=True)
        # Wait for the operator: "restart" runs the same episode again (this row goes to
        # restarted_attempts.jsonl), "continue" keeps it and goes on (or ends the run).
        choice = None
        while choice is None:
            for message in self.server.pop_commands():
                if message.get("command") in ("restart", "continue"):
                    choice = message["command"]
            time.sleep(0.05)
        if self.log is not None:
            self._write(event="finished", result=result, choice=choice)
            self.log.close()
            self.log = None
        self.server.publish_status(state="restarting" if choice == "restart" else "continuing")
        return choice

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _now(observation):
        return float(np.ravel(observation["sim_time"])[0])

    @staticmethod
    def _camera_name(key):
        return "head" if key == "rgb" else key.removesuffix("_rgb")

    def _selected_name(self):
        return "both" if len(self.selected) == 2 else SIDES[self.selected[0]]

    def _gripper_names(self):
        middle = (self.finger_min + self.finger_max) / 2
        return {side: ("open" if self.gripper_goal[i] > middle else "closed") for i, side in enumerate(SIDES)}

    def _write(self, **row):
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()


def make():
    return TeleopPolicy()
