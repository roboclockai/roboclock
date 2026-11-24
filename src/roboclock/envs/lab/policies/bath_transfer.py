"""Contact-driven scheduled transfer; timing is measured from actual bath crossings."""

import numpy as np

from roboclock.envs.lab.bath_scene import FLOOR, bath_points
from roboclock.envs.lab.policies.trajectories import CartesianCommand, limit_step
from roboclock.envs.lab.station_frame import station_of
from roboclock.physics.geometry import orientation_error

# A5 leaves the reel in the stop bath until the countdown has ended, then carries
# it just above the tank rims. The fixer has no deadline of its own.
RIM_CLEARANCE = 0.03
SPEED = 0.35
INSERT_SPEED = 0.25
# The reel first moves about 0.03 s after the lift command (measured at 50 Hz
# control), so it stays still until the countdown has ended, and the 1 cm rise
# that scores the lift comes early in hard's 1 s late window.
LIFT_LAG = 0.03
# A6: nominal lift speed (m/s) and the X5's measured delay from the lift command
# to the sample moving at that speed (motor response plus the phase-change tick).
A6_LIFT_SPEED = 0.31
A6_LIFT_LAG = 0.056


class BathTrajectory:
    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.last_goal = None
        self.pick = None
        self.grasp_offset = 0.0
        self.lift_speed = A6_LIFT_SPEED

    def command(self, task, state):
        obj = state.objects["sample_0"]
        h = task.layout.dimensions["sample_0"][1]
        z = task.layout.table_height
        pick = obj.position + np.array([0, 0, h * 0.375])
        clearance = z + 0.36
        carry = clearance
        # Bath centres in the world: the episode's seeded station placement.
        _, source_xy, dest_xy = bath_points(station_of(task))
        if task.task_id == "A5":
            rim = task.layout.targets["destination"].rim_z
            carry = rim + RIM_CLEARANCE + self.grasp_offset + h / 2
        if task.task_id == "A5" and obj.grasped and self.phase in ("hold", "lift", "transfer"):
            # Re-seating on the source floor can slide the narrow spindle within
            # the fingers; use its measured offset before destination insertion.
            self.grasp_offset = float(state.ee_position[2] - obj.position[2])
        close = self.phase in (
            "grasp",
            "load_lift",
            "load_move",
            "load_insert",
            "hold",
            "lift",
            "transfer",
            "insert",
        )
        goal = pick.copy()
        if self.phase == "approach":
            goal[2] = clearance
            if (
                np.linalg.norm(state.ee_position - goal) < 0.015
                and orientation_error(state.ee_orientation, np.array([0.0, 0, 1.0, 0])) < 0.12
            ):
                self.phase = "descend"
        elif self.phase == "descend":
            if np.linalg.norm(state.ee_position - goal) < 0.006:
                self.phase = "grasp"
        elif self.phase == "grasp":
            if obj.grasped:
                self.pick = state.ee_position.copy()
                self.grasp_offset = float(state.ee_position[2] - obj.position[2])
                self.phase = "load_lift"
        elif self.phase == "load_lift":
            goal = self.pick.copy()
            goal[2] = clearance
            if state.ee_position[2] > clearance - 0.008:
                self.phase = "load_move"
        elif self.phase == "load_move":
            goal = np.array([*source_xy, clearance])
            if np.linalg.norm(state.ee_position - goal) < 0.012:
                self.phase = "load_insert"
        elif self.phase == "load_insert":
            goal = np.array([*source_xy, z + FLOOR + h / 2 + self.grasp_offset - 0.004])
            if (
                "source_floor" in obj.fixture_contacts
                and np.linalg.norm(state.ee_position - goal) < 0.012
            ):
                self.pick = state.ee_position.copy()
                self.phase = "hold"
        elif self.phase == "hold":
            goal = self.pick.copy()
            if task.deadline is None:
                start = float("inf")
            elif task.task_id == "A5":
                start = task.deadline - LIFT_LAG
            else:
                # Account for the X5's measured motor response and the control
                # tick that changes phase before the lift target starts moving.
                # Lift starts on a 20 ms control tick; the lift speed is set to cross
                # the surface 30 s after immersion, the centre of the +-1 s (hard) exit window.
                distance = state.controls["liquid_surface_z"] - state.controls["sample_bottom_z"]
                start = task.deadline - distance / A6_LIFT_SPEED - A6_LIFT_LAG
                if task.time >= start:
                    remaining = task.deadline - task.time - A6_LIFT_LAG
                    self.lift_speed = distance / max(remaining, 1e-3)
            if task.time >= start:
                self.phase = "lift"
        elif self.phase == "lift":
            goal = self.pick.copy()
            goal[2] = carry
            # A5 turns toward the stop bath once the reel is nearly clear of the rims.
            if state.ee_position[2] > carry - (0.015 if task.task_id == "A5" else 0.008):
                self.phase = "transfer"
        elif self.phase == "transfer":
            goal = np.array([*dest_xy, carry])
            if np.linalg.norm(state.ee_position - goal) < 0.012:
                self.phase = "insert"
        elif self.phase == "insert":
            goal = np.array([*dest_xy, z + FLOOR + h / 2 + self.grasp_offset - 0.004])
            if (
                "destination_floor" in obj.fixture_contacts
                and np.linalg.norm(state.ee_position - goal) < 0.012
            ):
                self.phase = "release"
        else:
            goal = np.array([*dest_xy, clearance])
            close = False
        if self.last_goal is None:
            self.last_goal = state.ee_position.copy()
        speed = (
            INSERT_SPEED
            if self.phase in ("insert", "load_insert")
            else self.lift_speed
            if task.task_id == "A6" and self.phase == "lift"
            else SPEED
        )
        self.last_goal = limit_step(self.last_goal, goal, speed, self.dt)
        return CartesianCommand(self.last_goal.copy(), close)
