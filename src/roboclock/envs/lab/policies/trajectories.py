"""Task-specific Cartesian waypoints with contact-driven transitions.

Privileged diagnostic planner. This is a reproducible baseline, not an expert or
an agent observation. Its output always becomes articulation motor commands.

Placement closes the loop on the measured object pose: the tool goal is computed
from the ground-truth object-in-hand offset so the object, not the tool, reaches
the target. A held object is never released in mid-air; failed transfers are
retried from above, re-targeted, or carried back to their source.
"""

from dataclasses import dataclass

import numpy as np

from roboclock.envs.lab.mdp.state import orientation_error, upright_error
from roboclock.physics.geometry import rotation_matrix

# ARX X5 joint2 (shoulder pitch) is 0.1005 m above the mount flange (URDF).
SHOULDER_HEIGHT = 0.1005
# Measured Lula IK reach of the X5 with the tool pointing straight down
# (scratch probe, Isaac Sim 6.0.1): maximum horizontal tool distance from the
# mount axis as a function of tool height relative to the mount flange.
REACH_TABLE = (
    (-0.45, 0.395),
    (-0.40, 0.455),
    (-0.369, 0.475),
    (-0.33, 0.50),
    (-0.30, 0.51),
    (-0.265, 0.52),
    (-0.20, 0.53),
    (-0.10, 0.535),
    (0.0, 0.515),
)
GOAL_LEASH = 0.04  # max distance (m) the commanded goal may lead the measured tool
FINGERTIP_LENGTH = 0.0126  # X5 fingertips extend below the grasp center.
# Closer than this (horizontal, from the mount axis) the folded wrist housing
# swings into the arm's own mount column.
MIN_REACH = 0.15
BASE_TURN_RATE = 1.1  # rad/s: the arm's settled joint speed (control.joint_target_speed)
DEFAULT_REACH_MARGIN = 0.04
CARRYING_PHASES = ("lift", "transfer", "align", "insert", "return", "abandon")


@dataclass
class CartesianCommand:
    position: np.ndarray
    close: bool
    yaw: float = 0.0
    orientation: np.ndarray | None = None
    gripper: float | None = None  # open finger position (m) overriding the task default


def limit_step(previous, target, speed, dt):
    delta = np.asarray(target) - previous
    length = np.linalg.norm(delta)
    return previous + delta * min(1.0, speed * dt / max(length, 1e-12))


def multiply(a, b):
    """Hamilton product of scalar-first quaternions."""
    w, x, y, z = a
    v, i, j, k = b
    return np.array(
        [
            w * v - x * i - y * j - z * k,
            w * i + x * v + y * k - z * j,
            w * j - x * k + y * v + z * i,
            w * k + x * j - y * i + z * v,
        ]
    )


def conjugate(q):
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def wrap_angle(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def quaternion_yaw(q):
    """Yaw of an upright object frame (rotation about world z)."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tool_yaw(q):
    """Yaw of the downward-pointing tool; inverse of euler(0, pi, yaw) used for IK goals."""
    r = rotation_matrix(q)
    return float(np.arctan2(-r[1, 0], -r[0, 0]))


def tool_orientation(yaw):
    """Scalar-first quaternion of euler_angles_to_quat([0, pi, yaw])."""
    return np.array([0.0, -np.sin(yaw / 2), np.cos(yaw / 2), 0.0])


def shoulder_point(mount):
    return np.asarray(mount, dtype=float) + [0.0, 0.0, SHOULDER_HEIGHT]


def reach_limit(dz):
    """Maximum horizontal reach for a tool height `dz` relative to the mount flange."""
    heights, radii = zip(*REACH_TABLE)
    return float(np.interp(dz, heights, radii, left=0.0))


def reachable(base, margin, *points):
    """True when every downward-pointing tool goal lies inside the measured reach map."""
    if base is None:
        return True
    base = np.asarray(base, dtype=float)
    for point in points:
        point = np.asarray(point, dtype=float)
        horizontal = float(np.linalg.norm(point[:2] - base[:2]))
        if not MIN_REACH <= horizontal <= reach_limit(point[2] - base[2]) - margin:
            return False
    return True


def reach_ratio(base, *points):
    """Largest fraction of the measured reach used by any of `points` (lower is easier)."""
    base = np.asarray(base, dtype=float)
    ratios = []
    for point in points:
        point = np.asarray(point, dtype=float)
        limit = max(reach_limit(point[2] - base[2]), 1e-6)
        ratios.append(float(np.linalg.norm(point[:2] - base[:2])) / limit)
    return max(ratios)


def sorting_landing(task, sample, target):
    destination = target.position.copy()
    index = int(sample.split("_")[-1]) // 3
    if task.spec.num_samples > 36:
        # Four staggered columns and five rows provide 20 separate places per bin.
        pitch = 0.038
        column, row = index % 4, index // 4
        destination[:2] += [
            (column - 1.5) * np.sqrt(3) / 2 * pitch,
            (row - 2) * pitch + ((column % 2) - 0.5) * pitch / 2,
        ]
    else:
        destination[:2] += [((index % 3) - 1) * 0.040, ((index // 3) % 4 - 1.5) * 0.043]
    return destination


def grasp_height_offset(task, name):
    """Vertical tool offset above the object center used for picking."""
    _, height = task.layout.dimensions[name]
    if task.asset in ("probe", "test_tube"):
        return height * 0.375  # Upper-body grasp leaves fingers above rack walls.
    if task.asset in ("bottle", "container", "beaker"):
        return height * 0.20
    return 0.006


class LabTrajectory:
    def __init__(self, config, dt, base=None, keep_out=None):
        self.config, self.dt = config, dt
        self.base = None if base is None else np.asarray(base, dtype=float)
        # (xy, radius) region this arm's goals must stay out of: the other arm's
        # mount, whose shoulder and upper links occupy it even when parked.
        self.keep_out = keep_out
        self.reset()

    def reset(self):
        self.phase, self.sample = "select", None
        self.phase_time = 0.0
        self.retries = {}
        self.last_goal = None
        self.hold = None
        self.grasp_offset = np.zeros(3)
        self.yaw_offset = 0.0
        self.correction = np.zeros(3)
        self.previous_ee = None
        self.motion_anchor = None
        self.motion_time = 0.0
        self.lost_since = None
        self.returning = False
        self.recoveries = 0
        self.abandon_point = None
        self.marker_stage = "center"

    def transition(self, phase, time):
        self.phase, self.phase_time = phase, time
        self.motion_anchor = None
        if phase == "insert":
            self.marker_stage = "center"

    # ------------------------------------------------------------------ helpers
    @property
    def reach_margin(self):
        return self.config.get("reach_margin", DEFAULT_REACH_MARGIN)

    def reachable(self, *points):
        if self.keep_out is not None:
            center, radius = self.keep_out
            for point in points:
                if np.linalg.norm(np.asarray(point, dtype=float)[:2] - center[:2]) < radius:
                    return False
        return reachable(self.base, self.reach_margin, *points)

    def with_transit(self, task, *points):
        """Each tool goal plus the transit point directly above it."""
        result = []
        for point in points:
            above = np.array(point, dtype=float)
            above[2] = self.transit_height(task)
            result.extend([np.asarray(point, dtype=float), above])
        return result

    def task_points(self, task, name, state=None):
        """Tool goals one arm needs for `name`: pick, place and the transit above both."""
        return self.with_transit(
            task, self.pick_point(task, name, state), self.place_point(task, name)
        )

    def can_handle(self, task, name, state=None):
        return self.reachable(*self.task_points(task, name, state))

    def handling_cost(self, task, name, state=None):
        return reach_ratio(self.base, *self.task_points(task, name, state))

    def transit_height(self, task):
        heights = [h for _, h in task.layout.dimensions.values()]
        return task.layout.table_height + max(0.22, max(heights) + 0.12)

    def pick_point(self, task, name, state=None):
        position = (
            state.objects[name].position
            if state is not None
            else task.layout.objects[name].position
        )
        return position + [0.0, 0.0, grasp_height_offset(task, name)]

    def place_point(self, task, name):
        target = task.layout.targets[task.layout.assignments[name]]
        return target.position + [0.0, 0.0, grasp_height_offset(task, name)]

    def advance_goal(self, goal, speed, state, leash=True):
        """Rate-limited Cartesian goal that never runs more than GOAL_LEASH ahead.

        Joint1 cannot turn fast enough for 0.6 m/s close to the base. A goal that
        runs ahead lets the other joints reach their final values first, so the arm
        extends before it rotates and sweeps 20-30 cm off the commanded line, into
        the other arm's half. On a leash the tool follows the straight path.
        """
        if self.last_goal is None:
            self.last_goal = state.ee_position.copy()
        if self.base is not None:
            # Past this speed joint1 falls behind and the tool cuts inside MIN_REACH,
            # where the hand or held object hits the arm's own links.
            radius = float(np.linalg.norm(state.ee_position[:2] - self.base[:2]))
            speed = min(speed, max(BASE_TURN_RATE * radius, 0.05))
        goal = limit_step(self.last_goal, goal, speed, self.dt)
        if leash:
            lead = goal - state.ee_position
            distance = float(np.linalg.norm(lead))
            if distance > GOAL_LEASH:
                goal = state.ee_position + lead * (GOAL_LEASH / distance)
            if self.base is not None:
                radial = goal[:2] - self.base[:2]
                radius = float(np.linalg.norm(radial))
                if 1e-6 < radius < MIN_REACH:
                    goal[:2] = self.base[:2] + radial * (MIN_REACH / radius)
        return goal

    def ee_speed(self, state):
        if self.previous_ee is None:
            return 0.0
        return float(np.linalg.norm(state.ee_position - self.previous_ee) / self.dt)

    def stalled(self, task, state, goal, window=1.5, tolerance=0.02):
        """No tool motion for `window` seconds while the phase goal is still far away."""
        ee = state.ee_position
        if self.motion_anchor is None or np.linalg.norm(ee - self.motion_anchor) > 0.005:
            self.motion_anchor, self.motion_time = ee.copy(), task.time
            return False
        return task.time - self.motion_time > window and np.linalg.norm(ee - goal) > tolerance

    def lost(self, task, obj, state):
        """Grasp lost for longer than contact flicker, or object clearly out of hand."""
        if obj.grasped:
            # A measured grasp already implies the object is at the tool.
            self.lost_since = None
            return False
        if self.lost_since is None:
            self.lost_since = task.time
        _, height = task.layout.dimensions[self.sample]
        far = np.linalg.norm(obj.position - state.ee_position) > height * 0.5 + 0.05
        return far or task.time - self.lost_since > 0.5

    def lift_again(self, task, state):
        self.hold = state.ee_position.copy()
        _, height = task.layout.dimensions[self.sample]
        self.hold[2] = max(self.hold[2], task.layout.table_height + max(0.22, height + 0.12))
        self.correction[:] = 0.0
        self.transition("lift", task.time)

    def abandon_goal(self, task, state, height, offset):
        """Low drop-off point pulled back toward the arm's base, where lowering is feasible."""
        goal = state.ee_position.copy()
        if self.base is not None:
            goal[:2] = self.base[:2] + (goal[:2] - self.base[:2]) * 0.75
        goal[2] = task.layout.table_height + height + offset + 0.01
        return goal

    def give_up(self, task, state):
        """Carry the held object back to its source; never drop it from transit height."""
        self.retries[self.sample] = 3
        self.returning = True
        self.lift_again(task, state)

    def recover(self, task, state, obj, close, stalled):
        """Throughput recovery after a stall or timeout. Returns the gripper command."""
        name = self.sample
        if self.phase in CARRYING_PHASES and obj.grasped:
            self.recoveries += 1
            if self.phase == "abandon":
                # Cannot lower any further: open here, as close to the surface as possible.
                self.hold = state.ee_position.copy()
                self.transition("release", task.time)
                return False
            if self.phase == "return":
                _, height = task.layout.dimensions[name]
                self.abandon_point = self.abandon_goal(
                    task, state, height, grasp_height_offset(task, name)
                )
                self.transition("abandon", task.time)
                return True
            self.retries[name] = self.retries.get(name, 0) + 1
            if self.retries[name] < 3 and self.recoveries < 3:
                self.lift_again(task, state)
            else:
                self.give_up(task, state)
            return True
        # Nothing held: a stalled approach means the sample is out of reach.
        self.retries[name] = 3 if stalled else self.retries.get(name, 0) + 1
        self.returning = False
        self.transition("select", task.time)
        return False

    # ------------------------------------------------------------------ command
    def command(self, task, state):
        if task.task_id == "B1" and task.event_time is None:
            return CartesianCommand(
                state.ee_position.copy(), False, orientation=state.ee_orientation.copy()
            )
        throughput = task.category == "throughput"
        if self.phase == "select":
            self.returning = False
            self.recoveries = 0
            self.correction[:] = 0.0
            self.lost_since = None
            remaining = [
                n
                for n in task.active
                if n not in task.correct and n not in task.dropped and self.retries.get(n, 0) < 3
            ]
            if throughput:
                remaining = [
                    n
                    for n in remaining
                    if self.reachable(
                        *self.with_transit(
                            task, self.pick_point(task, n, state), self.place_point(task, n)
                        )
                    )
                ]
            if not remaining or task.goal_achieved:
                return CartesianCommand(state.ee_position.copy(), False)
            self.sample = min(
                remaining,
                key=lambda n: (np.linalg.norm(state.objects[n].position - state.ee_position), n),
            )
            self.transition("approach", task.time)
        obj = state.objects[self.sample]
        target_name = task.layout.assignments[self.sample]
        target = task.layout.targets[target_name]
        radius, height = task.layout.dimensions[self.sample]
        offset = grasp_height_offset(task, self.sample)
        pick = obj.position + np.array([0.0, 0.0, offset])
        if getattr(task, "tube_task", False):
            pick = obj.position + rotation_matrix(obj.orientation)[:, 2] * offset
        reactive = task.category == "event_reactive"
        catch = task.task_id == "B3" and not getattr(task, "power_task", False)
        slip = task.task_id == "B2" and not getattr(task, "power_task", False)
        if catch:
            pick = obj.position + np.array([0.0, 0.0, 0.010])
        if reactive and not catch and task.event_time is not None:
            pick += obj.velocity * min(0.02, self.dt)
            pick[2] = max(pick[2], task.layout.table_height + height / 2 + 0.002)
        destination = target.position.copy()
        if task.task_id == "A4":
            destination[0] += task.fixture_offset
        if getattr(task, "bin_sorting", False):
            destination = sorting_landing(task, self.sample, target)
        if self.returning:
            # Measured source pose is the only destination known to be reachable.
            destination = task.layout.objects[self.sample].position.copy()
        clearance = task.layout.table_height + max(0.22, height + 0.12)
        above = pick.copy()
        above[2] = max(pick[2] + 0.10, clearance)
        # Falling sample is suspended above the ordinary transit plane.
        if catch:
            above = task.layout.objects[self.sample].position.copy()
            above[2] += 0.010
        if slip and task.event_time is None:
            above = pick.copy()
        yaw = 0.0
        if task.task_id == "C3":
            yaw = quaternion_yaw(obj.orientation)
        ready_orientation = orientation_error(state.ee_orientation, tool_orientation(yaw)) < 0.10
        goal = above
        close = self.phase in CARRYING_PHASES or self.phase in ("grasp", "retain")

        def near(p, tol=0.012):
            return np.linalg.norm(state.ee_position - p) < tol

        if self.phase == "approach":
            if catch:
                if task.event_time is not None:
                    self.transition("grasp", task.time)
                    close = True
                    goal = pick
            elif slip:
                if task.event_time is not None:
                    self.transition("grasp", task.time)
                    close = True
                    goal = pick
            elif near(goal, 0.02) and ready_orientation:
                self.correction[:] = 0.0
                self.transition("descend", task.time)
        elif self.phase == "descend":
            goal = pick.copy()
            if throughput:
                # The arm lags a descending goal by a few cm in low postures. Close
                # the loop on the measured tool xy and stay above the object's top
                # until centred, so an open finger never lands on the object.
                error = pick[:2] - state.ee_position[:2]
                self.correction[:2] = np.clip(
                    self.correction[:2] + 2.0 * error * self.dt, -0.03, 0.03
                )
                goal[:2] += self.correction[:2]
                hover = obj.position[2] + height / 2 + FINGERTIP_LENGTH + 0.02
                if np.linalg.norm(error) > 0.005:
                    goal[2] = max(goal[2], min(hover, state.ee_position[2]))
            allowed = task.task_id != "A1" or 55 <= task.temperature <= 65
            allowed &= not reactive or task.event_time is not None
            at_pick = np.linalg.norm(state.ee_position - pick) < 0.012 if throughput else near(goal)
            if at_pick and allowed and ready_orientation:
                self.transition("grasp", task.time)
        elif self.phase == "grasp":
            goal = pick
            close = True
            if obj.grasped:
                self.grasp_offset = state.ee_position - obj.position
                # Ground-truth in-hand yaw: object yaw relative to the measured tool yaw.
                self.yaw_offset = wrap_angle(
                    quaternion_yaw(obj.orientation) - tool_yaw(state.ee_orientation)
                )
                self.hold = state.ee_position.copy()
                if not catch:
                    self.hold[2] = max(self.hold[2], clearance)
                self.lost_since = None
                self.transition("retain" if catch or slip else "lift", task.time)
            elif task.time - self.phase_time > 1.4 and not reactive:
                self.retries[self.sample] = self.retries.get(self.sample, 0) + 1
                self.transition("approach", task.time)
                close = False
        elif self.phase == "retain":
            # Hold the measured grasp pose until the evaluator verifies retention.
            goal = self.hold
        elif self.phase == "lift":
            goal = self.hold
            if near(goal, 0.02):
                self.transition("return" if self.returning else "transfer", task.time)
        elif self.phase == "transfer":
            goal = destination + self.grasp_offset
            goal[2] = max(clearance, goal[2] + 0.14)
            allowed = task.task_id not in ("A2", "A3") or task.time >= task.spec.valid_start - 0.12
            if task.task_id == "A4":
                # Align at the predicted penetration time rather than at descent start.
                travel = (
                    max(0.0, goal[2] - (target.rim_z + height / 2 + self.grasp_offset[2]))
                    / self.config["insertion_speed"]
                )
                phase = task.omega * (task.time + travel) + task.spec.phase
                allowed = abs(task.amplitude * np.sin(phase)) < task.alignment_tolerance * 0.65
            if near(goal, 0.02) and allowed:
                self.correction[:] = 0.0
                self.transition("insert", task.time)
        elif self.phase == "insert":
            # Close the loop on the held object, accounting for slip within the pads.
            held_offset = state.ee_position - obj.position
            goal = destination + held_offset
            if getattr(task, "hotplate_task", False):
                # The open safety tray admits the X5 fingers. Seat the beaker
                # before opening them so a long free fall cannot tip it over.
                goal[2] += 0.006
            elif target.kind == "slot" and task.task_id != "A4":
                # C2 rack: lower until the vial stands on the slot floor; the
                # fingertips stay ~2 cm above the 22 mm walls. A vial released
                # higher falls through the lip unscored when 5 mm off-centre.
                goal[2] += 0.002 if throughput else 0.009
            elif target.kind == "marker":
                # Flat marker without walls: centre at a fixed hover, lower vertically,
                # set the object down without pressing (pressing slid the hand along it).
                touch = target.rim_z + height / 2 + held_offset[2] + 0.001
                goal[2] = touch + (0.02 if self.marker_stage == "center" else 0.0)
            elif task.task_id != "A4":
                # Drop through the mouth while keeping fingertips outside the walls.
                # This avoids commanding an impossible gripper pose inside a narrow slot.
                goal[2] = target.rim_z + height / 2 + held_offset[2] + 0.012
            if throughput:
                # Integrate the measured object error: the IK solution alone leaves a
                # steady-state offset under load that proportional tracking never closes.
                # While lowering onto a marker the correction is frozen: once the block
                # rests on it, friction holds it and further correction slides the hand.
                error = destination[:2] - obj.position[:2]
                lateral = float(np.linalg.norm(error))
                if target.kind != "marker" or self.marker_stage == "center":
                    self.correction[:2] = np.clip(
                        self.correction[:2] + 1.5 * error * self.dt, -0.02, 0.02
                    )
                goal[:2] += self.correction[:2]
                if target.kind == "marker":
                    bottom_gap = obj.position[2] - height / 2 - target.rim_z
                    if (
                        self.marker_stage == "center"
                        and lateral < 0.0015
                        and abs(state.ee_position[2] - goal[2]) < 0.006
                        and self.ee_speed(state) < 0.02
                    ):
                        self.marker_stage = "lower"
                    elif self.marker_stage == "lower" and lateral > 0.0035:
                        # Drifted while lowering (or landed off-centre): back to hover.
                        self.marker_stage = "center"
                        self.last_goal = state.ee_position.copy()
                # Marker scoring needs the whole block footprint inside the marker:
                # 4.5 mm slack when square to it, less when rotated.
                centred = np.linalg.norm(obj.position[:2] - destination[:2]) < (
                    0.0025 if target.kind == "marker" else 0.006
                )
                if task.task_id == "C3":
                    centred &= (
                        abs(
                            wrap_angle(
                                quaternion_yaw(obj.orientation) - quaternion_yaw(target.orientation)
                            )
                        )
                        < 0.06
                    )
                landed = target.kind == "marker" and target_name in obj.fixture_contacts
                if target.kind == "slot":
                    # Scoring needs the vial to cross the lip while its footprint fits
                    # the slot: centre and straighten above the lip, then lower.
                    scored = self.sample in task.insertion_times
                    tilt = upright_error(obj.orientation)
                    if not scored and (lateral > 0.0025 or tilt > 0.025):
                        goal[2] = max(goal[2], state.ee_position[2])
                    seated = (
                        abs(obj.position[2] - destination[2]) < 0.004
                        and lateral < 0.0035
                        and tilt < 0.06
                        and self.ee_speed(state) < 0.03
                    )
                    if seated and not scored:
                        # Down without a scored insertion: lift out and insert again.
                        self.retries[self.sample] = self.retries.get(self.sample, 0) + 1
                        if self.retries[self.sample] < 3:
                            self.lift_again(task, state)
                        else:
                            self.give_up(task, state)
                    ready = seated and scored
                elif target.kind == "marker":
                    # Down on the marker, not merely at the hover goal above it.
                    landed = landed or bottom_gap < 0.003
                    ready = (
                        self.marker_stage == "lower"
                        and centred
                        and self.ee_speed(state) < 0.03
                        and landed
                    )
                else:
                    ready = centred and self.ee_speed(state) < 0.03 and near(goal, 0.009)
            else:
                ready = near(goal, 0.009)
            if ready:
                allowed = (
                    task.task_id not in ("A2", "A3")
                    or task.spec.valid_start - 0.10 <= task.time <= task.spec.valid_end
                )
                if task.task_id == "A4":
                    allowed = task.action_time is not None
                if allowed:
                    self.transition("release" if task.task_id != "A4" else "retain", task.time)
                    self.hold = state.ee_position.copy()
        elif self.phase == "return":
            held_offset = state.ee_position - obj.position
            error = destination[:2] - obj.position[:2]
            if np.linalg.norm(error) < 0.03:
                self.correction[:2] = np.clip(
                    self.correction[:2] + 1.5 * error * self.dt, -0.02, 0.02
                )
            goal = destination + held_offset
            goal[:2] += self.correction[:2]
            goal[2] += 0.003
            low = goal[2]
            lateral = np.linalg.norm(error)
            if lateral > 0.010:
                goal[2] = max(goal[2], state.ee_position[2])  # Center before lowering.
            # Release only once the object is centered AND down at the surface; the
            # centering tolerance is tighter than the lowering gate so a hovering
            # tool can never satisfy the release test.
            lowered = abs(state.ee_position[2] - low) < 0.008
            if lateral < 0.006 and (lowered or obj.table_contact or obj.fixture_contacts):
                self.hold = state.ee_position.copy()
                self.transition("release", task.time)
        elif self.phase == "abandon":
            # Last resort for an object that cannot be carried anywhere: pull back
            # toward the base and lower it to the surface, then open. Nothing is
            # released from transit height unless even lowering stalls.
            goal = self.abandon_point.copy()
            if near(goal, 0.01) or obj.table_contact or obj.fixture_contacts:
                self.hold = state.ee_position.copy()
                self.transition("release", task.time)
        elif self.phase == "release":
            goal = self.hold.copy()
            close = False
            # The X5 fingers open at 0.1 m/s: lifting before they clear the object
            # drags it several millimetres off the marker. Hold still, then rise.
            settle = 0.40 if throughput else 0.15
            if task.time - self.phase_time >= settle:
                goal[2] += 0.04
            if not obj.grasped and task.time - self.phase_time >= settle + 0.30:
                self.transition("retreat", task.time)
        elif self.phase == "retreat":
            goal = destination + self.grasp_offset
            if self.returning or throughput:
                # Straight up from the release pose: no sideways drag near the object.
                goal = self.hold.copy() if self.hold is not None else state.ee_position.copy()
            goal[2] = clearance
            close = False
            if near(goal, 0.025):
                self.retries[self.sample] = self.retries.get(self.sample, 0) + 1
                self.returning = False
                self.transition("select", task.time)
        if (
            throughput
            and self.phase in CARRYING_PHASES
            and self.phase != "abandon"
            and self.lost(task, obj, state)
        ):
            # The object is no longer in hand: chasing it with the in-hand offset
            # drives the tool into the table. Re-select and pick it up again.
            self.retries[self.sample] = self.retries.get(self.sample, 0) + 1
            self.returning = False
            self.transition("select", task.time)
            close = False
        waiting = self.phase in ("approach", "descend", "transfer") and (
            task.task_id in ("A1", "A2", "A3", "A4") or reactive
        )
        timed_out = task.time - self.phase_time > self.config["phase_timeout"] and not waiting
        stalled = (
            throughput
            and self.phase not in ("select", "grasp", "retain", "release")
            and self.stalled(task, state, goal)
        )
        if timed_out or stalled:
            if throughput:
                close = self.recover(task, state, obj, close, stalled)
            else:
                self.retries[self.sample] = self.retries.get(self.sample, 0) + 1
                self.transition("select", task.time)
                close = False
        speed = self.config["reactive_speed"] if reactive else self.config["max_cartesian_speed"]
        if self.phase in ("insert", "abandon") or (
            self.phase == "return" and np.linalg.norm(state.ee_position[:2] - goal[:2]) <= 0.006
        ):
            speed = self.config["insertion_speed"]
        if (
            throughput
            and self.phase == "insert"
            and target.kind == "marker"
            and obj.position[2] - height / 2 - target.rim_z > 0.03
        ):
            speed = 0.18  # No walls around a marker: descend quickly until the last 3 cm.
        if (
            throughput
            and self.phase == "insert"
            and target.kind == "slot"
            and obj.position[2] - height / 2 - target.rim_z > 0.03
        ):
            speed = 0.18  # Quickly down to 3 cm above the slot walls, then insert slowly.
        if throughput and self.phase == "insert" and target.kind == "marker":
            if self.marker_stage == "lower":
                speed = 0.05
        if throughput and self.phase == "descend" and state.ee_position[2] - pick[2] < 0.06:
            speed = min(speed, 0.12)  # Final approach: let the arm settle on the object.
        self.last_goal = self.advance_goal(goal, speed, state, leash=throughput)
        if task.task_id == "C3" and self.phase in CARRYING_PHASES + ("release", "retreat"):
            # Place the arrow along the marker axis. The ground-truth in-hand yaw is
            # re-measured every step so slip within the pads is corrected, and the
            # tool never tracks the object's own yaw (which would make it spin).
            desired = quaternion_yaw(
                task.layout.objects[self.sample].orientation
                if self.returning
                else target.orientation
            )
            in_hand = (
                wrap_angle(quaternion_yaw(obj.orientation) - tool_yaw(state.ee_orientation))
                if obj.grasped
                else self.yaw_offset
            )
            yaw = wrap_angle(desired - in_hand)
        self.previous_ee = state.ee_position.copy()
        gripper = None
        if throughput and target.kind == "marker" and self.phase in ("release", "retreat"):
            # Open each finger only a little past the block, and never move a finger
            # inward until the hand is clear of it (see OraclePickPlace).
            gripper = radius + (0.004 if self.phase == "release" else 0.01)
        return CartesianCommand(self.last_goal.copy(), close, yaw, gripper=gripper)
