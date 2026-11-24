"""Ground-truth color sorting with measured-pose, contact-driven motor control.

The planner reads the privileged object poses and computes every tool goal from
the measured tube-in-hand offset, so the tube (not the tool) is driven onto the
slot. Slots and tubes beyond the arm's reach are skipped, a slot that failed is
not chosen again, and a held tube is never released above the racks: a failed
transfer is re-targeted or carried back to its source slot.
"""

import numpy as np

from roboclock.envs.lab.mdp.state import Target, orientation_error, upright_error
from roboclock.envs.lab.policies.trajectories import (
    CARRYING_PHASES,
    CartesianCommand,
    LabTrajectory,
    multiply,
    reach_ratio,
)
from roboclock.envs.lab.rack_organization import rack_spec
from roboclock.physics.geometry import rotation_matrix

GRASP_OFFSET = 0.060  # Tool above the tube center: fingers stay above the rack walls.


def upright_correction(obj_orientation, ee_orientation):
    """Tool orientation that makes the held tube vertical while keeping the tool yaw.

    Forcing the tube's own yaw to zero (full inverse) demanded wrist rotations the
    X5 cannot reach and produced long IK failure streaks mid-transfer.
    """
    axis = rotation_matrix(obj_orientation)[:, 2]
    cross = np.cross(axis, [0.0, 0.0, 1.0])
    sine = np.linalg.norm(cross)
    if sine < 1e-6:
        return np.asarray(ee_orientation, dtype=float).copy()
    angle = float(np.arctan2(sine, axis[2]))
    fix = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * cross / sine)])
    return multiply(fix, ee_orientation)


class RackOrganizationTrajectory(LabTrajectory):
    """GT identifies tubes and free same-color holes; physics verifies every placement."""

    def reset(self):
        super().reset()
        self.orientation_goal = np.array([0.0, 0.0, 1.0, 0.0])
        self.release_pose = None
        self.aligned_since = None
        self.target_failures = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def tool_point(position):
        return np.asarray(position, dtype=float) + [0.0, 0.0, GRASP_OFFSET]

    def transit_height(self, task):
        return task.layout.table_height + 0.30

    def slot_reachable(self, task, slot):
        return self.reachable(
            *self.with_transit(task, self.tool_point(task.layout.targets[slot].position))
        )

    def can_handle(self, task, name, state=None):
        source = (state.objects[name] if state is not None else task.layout.objects[name]).position
        return self.reachable(*self.with_transit(task, self.tool_point(source))) and any(
            self.slot_reachable(task, slot) for slot in task.compatible_targets(name)
        )

    def handling_cost(self, task, name, state=None):
        source = (state.objects[name] if state is not None else task.layout.objects[name]).position
        slots = [task.layout.targets[slot].position for slot in task.compatible_targets(name)]
        nearest = min(slots, key=lambda p: reach_ratio(self.base, self.tool_point(p)))
        return reach_ratio(
            self.base,
            *self.with_transit(task, self.tool_point(source), self.tool_point(nearest)),
        )

    def candidate_targets(self, task, name, state, exclude=()):
        """Free same-color slots within reach, least-failed and nearest to the arm first."""
        anchor = None if self.base is None else self.base
        obj = state.objects[name]

        def key(slot):
            position = task.layout.targets[slot].position
            distance = np.linalg.norm(
                position[:2] - (anchor[:2] if anchor is not None else obj.position[:2])
            )
            return (self.target_failures.get(slot, 0), round(distance, 3), slot)

        return sorted(
            (
                slot
                for slot in task.free_targets(name, state)
                if slot not in exclude and self.slot_reachable(task, slot)
            ),
            key=key,
        )

    def source_target(self, task, name):
        """The tube's own source hole, used when a tube must be put back."""
        z = task.layout.table_height
        spec = rack_spec()
        return Target(
            task.layout.objects[name].position.copy(),
            np.array([0.019, 0.019, 0.075]),
            kind="slot",
            support_z=z + spec["support_z"],
            rim_z=z + spec["rim_z"],
        )

    def retarget(self, task, state):
        """After a failed transfer: blame the slot, pick another, or carry the tube home."""
        name = self.sample
        current = task.layout.assignments[name]
        if not self.returning:
            self.target_failures[current] = self.target_failures.get(current, 0) + 1
        self.retries[name] = self.retries.get(name, 0) + 1
        self.recoveries += 1
        alternatives = self.candidate_targets(task, name, state, exclude={current})
        if alternatives and not self.returning and self.retries[name] < 3 and self.recoveries < 3:
            task.layout.assignments[name] = alternatives[0]
            self.lift_again(task, state)
        elif self.phase == "return":
            self.abandon_point = self.abandon_goal(task, state, 0.14, GRASP_OFFSET)
            self.transition("abandon", task.time)
        elif self.phase == "abandon":
            self.release_pose = state.ee_position.copy()
            self.transition("release", task.time)
            return False
        else:
            self.give_up(task, state)
        self.aligned_since = None
        return True

    # ------------------------------------------------------------------ command
    def command(self, task, state):
        for name, obj in state.objects.items():
            # A tipped tube cannot be grasped from above; this includes the arm's own
            # sample once it has fallen out of the hand.
            if upright_error(obj.orientation) > 0.5 and not obj.grasped:
                self.retries[name] = 3
        if self.phase in {"select", "approach", "descend", "grasp"} and self.sample is not None:
            if self.retries.get(self.sample, 0) >= 3:
                self.transition("select", task.time)
        if self.phase == "select":
            self.returning = False
            self.recoveries = 0
            self.correction[:] = 0.0
            self.lost_since = None
            remaining = [
                name
                for name in task.active
                if name not in task.correct
                and name not in task.dropped
                and self.retries.get(name, 0) < 3
                and self.reachable(
                    *self.with_transit(task, self.tool_point(state.objects[name].position))
                )
                and self.candidate_targets(task, name, state)
            ]
            if not remaining:
                return CartesianCommand(
                    state.ee_position.copy(), False, orientation=self.orientation_goal
                )
            self.sample = min(
                remaining,
                key=lambda name: (
                    self.retries.get(name, 0),
                    np.linalg.norm(state.objects[name].position - state.ee_position),
                    name,
                ),
            )
            task.layout.assignments[self.sample] = self.candidate_targets(task, self.sample, state)[
                0
            ]
            self.transition("approach", task.time)
            self.orientation_goal = np.array([0.0, 0.0, 1.0, 0.0])
            self.aligned_since = None
        obj = state.objects[self.sample]
        target = (
            self.source_target(task, self.sample)
            if self.returning
            else task.layout.targets[task.layout.assignments[self.sample]]
        )
        pick = self.tool_point(obj.position)
        clearance = task.layout.table_height + 0.30
        goal = pick.copy()
        close = self.phase in CARRYING_PHASES or self.phase == "grasp"
        speed = 0.60
        orientation = self.orientation_goal.copy()
        held_offset = state.ee_position - obj.position
        if self.phase in CARRYING_PHASES and obj.grasped:
            # Feedback straightens the tube rather than assuming it has not
            # rotated within the pads. No object poses or scoring flags are set.
            desired = upright_correction(obj.orientation, state.ee_orientation)
            if np.dot(desired, self.orientation_goal) < 0:
                desired = -desired
            angle = orientation_error(desired, self.orientation_goal)
            mix = min(1.0, 0.8 * self.dt / max(angle, 1e-9))
            orientation = (1 - mix) * self.orientation_goal + mix * desired
            orientation /= np.linalg.norm(orientation)
            self.orientation_goal = orientation.copy()

        def near(position, tolerance=0.009):
            return np.linalg.norm(state.ee_position - position) < tolerance

        def integrate_correction():
            """Measured tube error at the slot, integrated to cancel steady-state tool offset."""
            error = target.position[:2] - obj.position[:2]
            if np.linalg.norm(error) < 0.03:
                self.correction[:2] = np.clip(
                    self.correction[:2] + 1.5 * error * self.dt, -0.02, 0.02
                )
            return self.correction

        if self.phase == "approach":
            goal[2] = clearance
            if near(goal, 0.012) and orientation_error(state.ee_orientation, orientation) < 0.07:
                self.transition("descend", task.time)
        elif self.phase == "descend":
            speed = 0.16
            if near(goal, 0.009) and orientation_error(state.ee_orientation, orientation) < 0.12:
                self.transition("grasp", task.time)
        elif self.phase == "grasp":
            speed = 0.06
            if obj.grasped:
                self.hold = state.ee_position.copy()
                self.hold[2] = clearance
                self.lost_since = None
                self.transition("lift", task.time)
            elif task.time - self.phase_time > 1.0:
                self.retries[self.sample] = self.retries.get(self.sample, 0) + 1
                self.transition("approach", task.time)
                close = False
        elif self.phase == "lift":
            goal = self.hold.copy()
            speed = 0.35
            if near(goal, 0.009):
                self.correction[:] = 0.0
                self.transition("return" if self.returning else "transfer", task.time)
        elif self.phase in {"transfer", "align", "return"}:
            goal = target.position + held_offset + integrate_correction()
            goal[2] = clearance
            aligned = (
                np.linalg.norm(obj.position[:2] - target.position[:2]) < 0.0025
                and upright_error(obj.orientation) < 0.045
                and near(goal, 0.012)
            )
            if aligned:
                if self.aligned_since is None:
                    self.aligned_since = task.time
                if task.time - self.aligned_since >= 0.08:
                    self.transition("insert", task.time)
            else:
                self.aligned_since = None
        elif self.phase == "insert":
            goal = target.position + held_offset + integrate_correction() + [0, 0, 0.001]
            speed = 0.065
            lateral = np.linalg.norm(obj.position[:2] - target.position[:2])
            scored = self.returning or self.sample in task.insertion_times
            # Stop descent while centering. A small lateral tracking error at
            # the lip otherwise misses the required downward insertion event.
            if not scored and (lateral > 0.0025 or upright_error(obj.orientation) > 0.025):
                goal[2] = max(goal[2], state.ee_position[2])
            if (
                lateral < 0.0035
                and abs(obj.position[2] - target.position[2]) < 0.004
                and upright_error(obj.orientation) < 0.06
            ):
                if not scored:
                    self.hold = state.ee_position.copy()
                    self.hold[2] = clearance
                    self.aligned_since = None
                    self.transition("lift", task.time)
                else:
                    self.release_pose = state.ee_position.copy()
                    self.transition("release", task.time)
                    close = False
        elif self.phase == "abandon":
            # Last resort: pull back toward the base and lower the tube onto
            # whatever is below before opening.
            goal = self.abandon_point.copy()
            speed = 0.065 if near(goal[:2].tolist() + [state.ee_position[2]], 0.01) else 0.30
            if near(goal, 0.01) or obj.table_contact or obj.fixture_contacts:
                self.release_pose = state.ee_position.copy()
                self.transition("release", task.time)
                close = False
        elif self.phase == "release":
            goal = self.release_pose.copy()
            close = False
            speed = 0.20
            if task.time - self.phase_time >= 0.25:
                self.transition("retreat", task.time)
        elif self.phase == "retreat":
            goal = self.release_pose.copy()
            goal[2] = clearance
            close = False
            speed = 0.20  # faster, the lagging arm drifts sideways into its column
            if near(goal, 0.015):
                if self.sample not in task.correct:
                    self.retries[self.sample] = self.retries.get(self.sample, 0) + 1
                self.returning = False
                self.transition("select", task.time)
        if (
            self.phase in CARRYING_PHASES
            and self.phase != "abandon"
            and self.lost(task, obj, state)
        ):
            # The tube left the hand: do not chase it with the in-hand offset.
            self.retries[self.sample] = self.retries.get(self.sample, 0) + 1
            self.returning = False
            self.transition("select", task.time)
            close = False
        stalled = self.phase not in ("select", "grasp", "release") and self.stalled(
            task, state, goal
        )
        if stalled or task.time - self.phase_time > 6.0:
            if self.phase in CARRYING_PHASES and obj.grasped:
                close = self.retarget(task, state)
            else:
                self.retries[self.sample] = 3 if stalled else self.retries.get(self.sample, 0) + 1
                self.returning = False
                self.transition("select", task.time)
                close = False
        self.last_goal = self.advance_goal(goal, speed, state)
        self.previous_ee = state.ee_position.copy()
        return CartesianCommand(self.last_goal.copy(), close, orientation=orientation)
