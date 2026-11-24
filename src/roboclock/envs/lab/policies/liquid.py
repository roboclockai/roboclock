"""Privileged feedback paths for physical valve handles and collector rails."""

import numpy as np

from roboclock.envs.lab.liquid_scene import CLOSE_ANGLE, LEVER_RADIUS, TUBE_SPACING
from roboclock.envs.lab.policies.trajectories import CartesianCommand, limit_step

CLOSE_RATE = 0.35  # rad/s commanded lever closing rate
# A4 starts from rest when the flask first shows pink, with about 2.3 mL left before the
# excess limit, so it reaches for the lever and closes it faster than the tasks that hold
# the lever while a visible level approaches its mark.
TASK_CLOSE_RATE = {"A4": 1.0}
A4_REACH_SPEED = 0.35  # m/s, control.max_cartesian_speed
A4_DESCEND_SPEED = 0.20
# Flow stops at 4 % opening; until then it falls linearly with the lever angle.
CLOSED_ANGLE = 0.96 * CLOSE_ANGLE
# Measured delay (s) between the closing command and the lever's first motion.
CLOSE_LAG = 0.1
# Grasp the lever this long before closing starts, so descent and grasp never
# add volume past the planned endpoint (descent and grasp take about 1 s).
GRASP_LEAD = 2.0
# Planned final volumes (mL): A2 leaves the last ~4 mL of the blue phase instead
# of a third of it; A4 stops at a pale pink, not midway to the excess colour.
STOP_MARGIN = {"A2": -4.0, "A4": 0.3}


def closure_volume(task, measured):
    """Volume still delivered while the lever closes from `measured` at the task's rate."""
    remaining = max(0.0, CLOSED_ANGLE - measured) / close_rate(task)
    flow = task.nominal_flow * max(0.0, 1 - measured / CLOSE_ANGLE)
    return flow * (remaining / 2 + CLOSE_LAG)


def close_rate(task):
    return TASK_CLOSE_RATE.get(task.task_id, CLOSE_RATE)


def cue_visible(task):
    """Whether the task's cue can be seen yet. Only A4 has a cue that appears: the flask
    is clear until its floor shows the first pink step (17.7 mL); the other liquid tasks
    show a level, boundary or band that approaches its mark from the start."""
    if task.task_id != "A4":
        return True
    from roboclock.envs.lab.titration_visuals import floor_step

    return floor_step(task.volume) >= 1


def stop_volume(task):
    margin = STOP_MARGIN[task.task_id]
    return (task.maximum_volume if margin < 0 else task.minimum_volume) + margin


class LiquidTrajectory:
    def __init__(self, dt):
        self.dt = dt
        self.phase = "approach"
        self.commanded = None
        self.angle = 0.0
        self.phase_time = 0.0
        self.grasp_time = None
        self.release_time = None
        self.closing = False
        self.hold_angle = None
        self.cue_seen = False

    def waiting_for_cue(self, task):
        """True until the cue is visible; the oracle holds the start pose meanwhile."""
        self.cue_seen = self.cue_seen or cue_visible(task)
        return not self.cue_seen

    def command(self, task, state):
        if task.task_id == "A1":
            return self.water_cycle(task, state)
        grip = state.objects["sample_0"].position
        if self.commanded is None:
            self.commanded = state.ee_position.copy()
        close = False
        yaw = 0.0
        if self.phase == "approach":
            target = grip + [0, 0, 0.11]
            if np.linalg.norm(state.ee_position - target) < 0.015:
                self.phase = "descend"
        elif self.phase == "descend":
            if task.task_id != "A3" and task.volume < (
                stop_volume(task) - closure_volume(task, 0.0) - task.nominal_flow * GRASP_LEAD
            ):
                target = grip + [0, 0, 0.11]
            else:
                target = grip.copy()
            if np.linalg.norm(state.ee_position - grip) < 0.009:
                self.phase = "grasp"
                self.grasp_time = task.time
        elif self.phase == "grasp":
            target = grip.copy()
            close = True
            if state.objects["sample_0"].grasped or task.time - self.grasp_time > 1.2:
                self.phase = "track"
        else:
            close = True
            if task.task_id == "A3":
                desired = 0
                for i, (start, end) in enumerate(task.intervals):
                    if start - 3.5 <= task.time <= end + 0.05:
                        desired = i + 1
                # The collector's start (waste under the outlet) is the layout's grip, which
                # follows the episode's seeded station frame; fraction i is i tubes along -x.
                target = task.layout.objects["sample_0"].position - [
                    desired * TUBE_SPACING,
                    0.0,
                    0.0,
                ]
                # Fractions use the full real collection interval, then go to waste.
                if task.goal_achieved:
                    close = False
                    target = grip + [0, 0, 0.10]
            else:
                # Flow is proportional to opening, so the closing motion itself
                # delivers more liquid: start closing once the predicted remainder
                # brings the delivered volume to the planned endpoint.
                measured = state.controls.get("valve_angle", 0.0)
                if self.hold_angle is None:
                    self.hold_angle = measured
                self.closing |= task.volume + closure_volume(task, measured) >= stop_volume(task)
                if self.closing:
                    # A fixed-rate ramp keeps the delivered volume predictable;
                    # restarting from a leading measured angle each tick sped the
                    # closure up to 0.53 rad/s.
                    self.angle = min(
                        CLOSE_ANGLE, self.angle + close_rate(task) * self.dt, measured + 0.10
                    )
                else:
                    # Hold the lever where it was grasped. Following its measured
                    # angle made the wrist yaw drag it shut (about 0.25 rad/s).
                    self.angle = self.hold_angle
                yaw = self.angle
                radius = 0.0 if task.task_id == "A1" else LEVER_RADIUS
                pivot = grip - np.array([radius * np.cos(measured), radius * np.sin(measured), 0])
                target = pivot + [
                    radius * np.cos(self.angle),
                    radius * np.sin(self.angle),
                    0,
                ]
                if task.stopped_at is not None:
                    if self.release_time is None:
                        self.release_time = task.time
                    if task.time - self.release_time > 0.60:
                        close = False
                        target = grip + [0, 0, 0.10]
        speed = 0.12 if self.phase in ("descend", "grasp") else 0.16
        if task.task_id == "A4" and task.stopped_at is None:
            speed = (
                A4_REACH_SPEED
                if self.phase == "approach"
                else A4_DESCEND_SPEED
                if self.phase == "descend"
                else speed
            )
        self.commanded = limit_step(self.commanded, np.asarray(target), speed, self.dt)
        return CartesianCommand(self.commanded.copy(), close, yaw=yaw)

    def water_cycle(self, task, state):
        """Turn the passive faucet with finger friction, then close and release it."""
        grip = state.objects["sample_0"].position
        measured = state.controls["valve_angle"]
        if self.commanded is None:
            self.commanded = state.ee_position.copy()
            self.angle = CLOSE_ANGLE
        target = grip.copy()
        close = self.phase not in ("approach", "descend", "retreat")
        if self.phase == "approach":
            target = grip + [0, 0, 0.11]
            if np.linalg.norm(state.ee_position - target) < 0.015:
                self.phase = "descend"
        elif self.phase == "descend":
            if np.linalg.norm(state.ee_position - grip) < 0.009:
                self.phase = "grasp"
                self.grasp_time = task.time
        elif self.phase == "grasp":
            if state.objects["sample_0"].grasped or task.time - self.grasp_time > 1.2:
                self.phase = "open"
        elif self.phase == "open":
            self.angle = max(0.0, min(self.angle, measured) - 0.35 * self.dt, measured - 0.10)
            if task.opening >= 0.96:
                self.phase = "collect"
                self.angle = 0.0
        elif self.phase == "collect":
            # The closing turn still delivers ~20 mL at full flow: start it once
            # the predicted remainder reaches the middle of the accepted band.
            stop = (task.minimum_volume + task.maximum_volume) / 2
            if task.volume + closure_volume(task, measured) >= stop:
                self.phase = "close"
                self.angle = measured
        elif self.phase == "close":
            # Fixed-rate ramp, as for A2/A4, so closure_volume holds.
            self.angle = min(CLOSE_ANGLE, self.angle + CLOSE_RATE * self.dt, measured + 0.10)
            if task.stopped_at is not None:
                self.phase = "release"
                self.release_time = task.time
        elif self.phase == "release":
            if task.time - self.release_time > 0.3:
                close = False
            if task.time - self.release_time > 0.6:
                self.phase = "retreat"
        elif self.phase == "retreat":
            target = grip + [0, 0, 0.11]
        self.commanded = limit_step(self.commanded, np.asarray(target), 0.12, self.dt)
        return CartesianCommand(self.commanded.copy(), close, yaw=self.angle - CLOSE_ANGLE)
