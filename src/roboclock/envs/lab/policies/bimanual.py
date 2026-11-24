"""Independent sample queues, reserved destinations and arm separation for throughput tasks.

The two X5 arms stand side by side on the common rig (0.60 m apart, both facing
the work area). Each sample goes to an arm that can reach both its pick and its
place pose (measured reach map). At run time a separation guard keeps the arms'
footprints apart: when the next motion of one arm would bring it close to the
other, the arm doing less delicate work backs off toward its own mount and waits.
The guard also handles mounts that face each other (legacy configurations).
"""

from dataclasses import replace

import numpy as np

from roboclock.envs.lab.policies.rack_organization import RackOrganizationTrajectory
from roboclock.envs.lab.policies.trajectories import (
    GOAL_LEASH,
    SHOULDER_HEIGHT,
    LabTrajectory,
    limit_step,
)
from roboclock.envs.lab.robot import mount_positions, mount_yaws

# Minimum xy distance between the two arms' footprints (mount-to-tool segments).
# The X5 forearm and open gripper are ~0.09 m wide; 0.15 m leaves a margin for
# links that are not exactly above the segment.
ARM_CLEARANCE = 0.15  # between link-chain centerlines (3D); wrist housings are ~9 cm thick
CHAIN_FRAMES = ("link3", "link4", "link5", "link6")
# No goal of one arm within this xy radius of the other arm's mount: its shoulder
# and upper links stay there even when parked (a cross-over into the far rack
# pinned both arms in C1).
OTHER_MOUNT_KEEP_OUT = 0.25
RESUME_MARGIN = 0.05  # extra clearance required before a waiting arm resumes
LOOKAHEAD = 0.50  # seconds of commanded motion checked ahead (PD arms need time to stop)
MAX_LOOKAHEAD = 0.30  # m
YIELD_WATCHDOG = 5.0  # s of one-sided waiting before the roles swap
# Parking mirrors the start pose: tool retracted 15-18 cm, about 5 cm below the
# mount flange, turned 60-110 degrees to the side. Closer, the folded wrist hits
# the own column; higher or facing the other arm, the elbow and forearm rise into
# the middle where the other arm's upper links pass.
PARK_RADII = (0.15, 0.18)
PARK_MIN_RADIUS = 0.15
PARK_BELOW_MOUNT = 0.05
PARK_OFFSETS_DEG = (-60.0, 60.0, -90.0, 90.0, -110.0, 110.0)
PARK_SPEED = 0.35
# Rest point of an arm that has finished its queue: the start-pose tool position
# (X5 HOME puts the tool ~0.28 m ahead of and 0.10 m above the mount flange).
REST_FORWARD = 0.26
REST_ABOVE_MOUNT = 0.08
# Arms close to the table (grasping, inserting, releasing) never back off.
PRECISE_PHASES = ("descend", "grasp", "insert", "release", "return", "abandon")
CARRYING = ("lift", "transfer", "align")


class ArmTask:
    """Planner-only view; the real task remains the sole owner of scoring."""

    def __init__(self, task, names, reserved):
        self.task, self.names, self.reserved = task, names, reserved

    def __getattr__(self, key):
        return getattr(self.task, key)

    @property
    def active(self):
        return self.task.active.intersection(self.names)

    def free_targets(self, sample, state):
        return [t for t in self.task.free_targets(sample, state) if t not in self.reserved]


def partition_samples(task, planners):
    """Give each sample to an arm that can do it; balance the samples both arms can do.

    Samples only one arm can reach go to that arm. Shared samples are sorted by how
    much easier they are for side 0 (the rig's right arm) and split so the queues are balanced,
    which also keeps each queue spatially contiguous. Samples neither arm can reach
    go to the arm that needs the smaller fraction of its reach; the planner's own
    reach filter then decides whether it is attempted.
    """
    names = sorted(task.layout.objects)
    able = {n: [p.can_handle(task, n) for p in planners] for n in names}
    cost = {n: [p.handling_cost(task, n) for p in planners] for n in names}
    queues = (set(), set())
    shared = []
    for name in names:
        if able[name][0] and able[name][1]:
            shared.append(name)
        elif able[name][0] or able[name][1]:
            queues[0 if able[name][0] else 1].add(name)
        else:
            queues[int(cost[name][1] < cost[name][0])].add(name)
    shared.sort(key=lambda n: (cost[n][0] - cost[n][1], n))
    for name in shared:
        left_turn = len(queues[0]) < len(queues[1]) or (
            len(queues[0]) == len(queues[1]) and cost[name][0] <= cost[name][1]
        )
        queues[0 if left_turn else 1].add(name)
    return queues


def segment_distance_3d(p0, p1, q0, q1):
    """Minimum distance between two 3D segments (closest points, clamped)."""
    p0, p1, q0, q1 = (np.asarray(v, dtype=float) for v in (p0, p1, q0, q1))
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
        else:
            b = float(d1 @ d2)
            denominator = a * e - b * b
            s = (
                float(np.clip((b * f - c * e) / denominator, 0.0, 1.0))
                if denominator > 1e-12
                else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
            elif t > 1.0:
                s, t = float(np.clip((b - c) / a, 0.0, 1.0)), 1.0
    return float(np.linalg.norm((p0 + d1 * s) - (q0 + d2 * t)))


def segment_distance(a0, a1, b0, b1):
    """Minimum distance between two 2D segments."""
    a0, a1, b0, b1 = (np.asarray(v, dtype=float)[:2] for v in (a0, a1, b0, b1))

    def point_segment(p, s0, s1):
        d = s1 - s0
        t = np.clip(np.dot(p - s0, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(p - (s0 + t * d)))

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1, d2 = cross(b0, b1, a0), cross(b0, b1, a1)
    d3, d4 = cross(a0, a1, b0), cross(a0, a1, b1)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(
        point_segment(a0, b0, b1),
        point_segment(a1, b0, b1),
        point_segment(b0, a0, a1),
        point_segment(b1, a0, a1),
    )


class BimanualThroughput:
    def __init__(self, env, dt):
        self.env = env
        self.dt = dt
        task = env.task
        self.mounts = [
            np.asarray(m, dtype=float) for m in mount_positions(env.config, task.task_id)
        ]
        yaws = mount_yaws(env.config, task.task_id)
        # A side-1 arm mounted facing side 0 sees the task frame turned 180 deg.
        self.opposed = abs((yaws[1] - yaws[0] + np.pi) % (2 * np.pi) - np.pi) > np.pi / 2
        # Unit vector from the side-0 to the side-1 mount: "columns" of one arm's job
        # are measured along it when reserving space for the other arm.
        across = self.mounts[1][:2] - self.mounts[0][:2]
        self.across = across / max(np.linalg.norm(across), 1e-9)
        cls = RackOrganizationTrajectory if task.task_id == "C1" else LabTrajectory
        keep_out = env.config["control"].get("other_mount_keep_out", OTHER_MOUNT_KEEP_OUT)
        self.planners = [
            cls(
                env.config["control"],
                dt,
                base=self.mounts[side],
                keep_out=(self.mounts[1 - side][:2], keep_out),
            )
            for side in range(2)
        ]
        self.names = partition_samples(task, self.planners)
        self.initial_names = tuple(set(names) for names in self.names)
        self.handed_over = set()
        self.previous_goals = [None, None]
        self.yield_goals = [None, None]
        self.yielding = [False, False]
        self.yield_steps = [0, 0]
        self.freeze_anchors = [None, None]
        self.crossing_jobs = {}
        self.yielded_now = None
        self.prev_commands = [None, None]
        self.yield_since = 0.0
        self.rest_goals = [None, None]
        self.solvers = None  # (left, right) ArticulationKinematicsSolvers, set by the oracle

    def reset(self):
        for planner in self.planners:
            planner.reset()
        self.names = tuple(set(names) for names in self.initial_names)
        self.handed_over = set()
        self.previous_goals = [None, None]
        self.yield_goals = [None, None]
        self.yielding = [False, False]
        self.yield_steps = [0, 0]
        self.freeze_anchors = [None, None]
        self.crossing_jobs = {}
        self.prev_commands = [None, None]
        self.yield_since = 0.0
        self.rest_goals = [None, None]

    def ik_failed(self, side):
        """Keep the Cartesian goal from running ahead of an arm that could not follow."""
        if self.previous_goals[side] is not None:
            self.planners[side].last_goal = self.previous_goals[side].copy()

    def hand_over_abandoned(self):
        """A sample one arm gave up on moves to the other arm's queue, once."""
        task = self.env.task
        for side, planner in enumerate(self.planners):
            other = self.planners[1 - side]
            retries = getattr(planner, "retries", {})
            other_retries = getattr(other, "retries", {})
            can_handle = getattr(other, "can_handle", None)
            for name in list(self.names[side]):
                if (
                    retries.get(name, 0) >= 3
                    and other_retries.get(name, 0) < 3
                    and planner.sample != name
                    and name not in self.handed_over
                    and (can_handle is None or can_handle(task, name))
                ):
                    self.names[side].discard(name)
                    self.names[1 - side].add(name)
                    self.handed_over.add(name)

    def doable_now(self, side, name, state):
        """Can this arm pick `name` from where it is and place it in a free slot now?"""
        planner = self.planners[side]
        task = self.env.task
        view = ArmTask(task, {name}, set())
        if hasattr(planner, "candidate_targets"):
            source = state.objects[name].position
            return planner.reachable(
                *planner.with_transit(task, planner.tool_point(source))
            ) and bool(planner.candidate_targets(view, name, state))
        return planner.can_handle(task, name, state)

    def rebalance(self, state):
        """Move samples an arm cannot do any more (e.g. its reachable slots of that
        colour are full) to the other arm when that arm can do them now."""
        task = self.env.task
        if not all(hasattr(p, "can_handle") for p in self.planners):
            return
        if not any(p.phase == "select" for p in self.planners):
            return
        for side in range(2):
            other = 1 - side
            for name in sorted(self.names[side]):
                if (
                    name in task.correct
                    or name in task.dropped
                    or self.planners[side].sample == name
                ):
                    continue
                if getattr(self.planners[other], "retries", {}).get(name, 0) >= 3:
                    continue
                if not self.doable_now(side, name, state) and self.doable_now(other, name, state):
                    self.names[side].discard(name)
                    self.names[other].add(name)

    # ------------------------------------------------------------------ separation
    def priority(self, side, command):
        planner = self.planners[side]
        phase = getattr(planner, "phase", "select")
        if phase in PRECISE_PHASES:
            return 2
        if phase in CARRYING or command.close:
            return 1
        if phase == "select" or getattr(planner, "sample", None) is None:
            return -1  # an idle arm always makes room for a working one
        return 0

    def arm_chain(self, side, tool):
        """3D polyline through the arm: shoulder, elbow, forearm, wrist housing, tool.

        With Lula forward kinematics of the measured joints when available. The X5
        elbow can sit radially beyond the tool and the wrist housing sticks out 9 cm
        past it, so a mount-to-tool segment underestimates the arm's extent.
        """
        tool = np.asarray(tool, dtype=float)
        shoulder = self.mounts[side] + [0.0, 0.0, SHOULDER_HEIGHT]
        solver = None if self.solvers is None else self.solvers[side]
        if solver is None:
            return [shoulder, tool]
        backend = self.env.backend
        robot = backend.secondary_robot if side else backend.robot
        q = np.asarray(robot.get_joint_positions(), dtype=float)[:6]
        kinematics = solver.get_kinematics_solver()
        points = [shoulder]
        for frame in CHAIN_FRAMES:
            position, _ = kinematics.compute_forward_kinematics(frame, q)
            points.append(np.asarray(position, dtype=float))
        points.append(tool)
        return points

    def lookahead_chain(self, chain, tool, command, side=None):
        """The chain moved to where the tool will be after the look-ahead window.

        That is the commanded point plus the commanded velocity (one control step
        of goal motion) carried on for LOOKAHEAD seconds, capped at MAX_LOOKAHEAD.
        Extrapolating the command-to-tool gap instead inflated motion up to 0.6 m.
        """
        tool = np.asarray(tool, dtype=float)
        target = np.asarray(command.position, dtype=float)
        previous = None if side is None else self.prev_commands[side]
        velocity = np.zeros(3) if previous is None else (target - previous) / max(self.dt, 1e-9)
        travel = velocity * LOOKAHEAD
        length = float(np.linalg.norm(travel))
        if length > MAX_LOOKAHEAD:
            travel *= MAX_LOOKAHEAD / length
        delta = (target - tool) + travel
        n = len(chain)
        # The shoulder is fixed; links move progressively more toward the tool.
        return [p + delta * (i / (n - 1)) for i, p in enumerate(chain)]

    @staticmethod
    def chain_distance(a, b):
        return min(
            segment_distance_3d(a[i], a[i + 1], b[j], b[j + 1])
            for i in range(len(a) - 1)
            for j in range(len(b) - 1)
        )

    def clearance(self, tools, commands, chains=None):
        chains = chains or [self.arm_chain(s, tools[s]) for s in range(2)]
        ahead = [self.lookahead_chain(chains[s], tools[s], commands[s], s) for s in range(2)]
        pairs = [(ahead[0], ahead[1]), (ahead[0], chains[1]), (chains[0], ahead[1])]
        return min(self.chain_distance(a, b) for a, b in pairs)

    def conflict(self, tools, commands, chains=None):
        required = ARM_CLEARANCE + (RESUME_MARGIN if any(self.yielding) else 0.0)
        return self.clearance(tools, commands, chains) < required

    def park_point(self, side, tool, other_chain):
        """Retracted pose around the own mount that stays farthest from the other arm.

        Never farther out than the tool already is (parking must not move toward the
        other arm), and fanned around the mount: when the other arm reaches close to
        this mount, swinging sideways is the only way to make room.
        """
        mount = self.mounts[side]
        shoulder = mount + [0.0, 0.0, SHOULDER_HEIGHT]
        center = (self.mounts[0][:2] + self.mounts[1][:2]) / 2
        forward = center - mount[:2]
        base_angle = float(np.arctan2(forward[1], forward[0]))
        planner = self.planners[side]
        height = (
            planner.transit_height(self.env.task)
            if hasattr(planner, "transit_height")
            else float(tool[2])
        )
        height = max(height, mount[2] - PARK_BELOW_MOUNT)
        current = float(np.linalg.norm(np.asarray(tool[:2]) - mount[:2]))
        # Never farther out than now, never closer than the column allows (a held
        # C1 tube once swung into the column from 10 cm and slid out of the hand).
        radii = sorted({min(r, max(current, PARK_MIN_RADIUS)) for r in PARK_RADII})
        best, best_score = None, None
        for radius in radii:
            for offset in np.deg2rad(PARK_OFFSETS_DEG):
                angle = base_angle + offset
                point = np.array(
                    [
                        mount[0] + radius * np.cos(angle),
                        mount[1] + radius * np.sin(angle),
                        max(float(tool[2]), height),
                    ]
                )
                # The parked arm reaches out past its tool by about the wrist housing.
                reach = point + (point - shoulder) * np.array([1.0, 1.0, 0.0]) * 0.4
                distance = min(
                    self.chain_distance([shoulder, point, reach], other_chain),
                    ARM_CLEARANCE + 0.05,
                )
                # Prefer clearance, then little motion from where the tool is now.
                score = (round(distance, 2), -float(np.linalg.norm(point[:2] - tool[:2])))
                if best_score is None or score > best_score:
                    best, best_score = point, score
        return best

    def apply_yield(self, side, tool, command, other_chain):
        """Back off toward the own mount (or freeze while working at the table)."""
        planner = self.planners[side]
        task = self.env.task
        # Anchor where the wait began: re-reading the measured tool every step lets
        # the arm drift without bound (it walked 20 cm toward the other arm once).
        if self.freeze_anchors[side] is None:
            self.freeze_anchors[side] = np.asarray(tool, dtype=float).copy()
        anchor = self.freeze_anchors[side]
        if getattr(planner, "phase", "") in PRECISE_PHASES:
            goal = anchor
        else:
            goal = self.park_point(side, anchor, other_chain)
            if tool[2] < goal[2] - 0.02:
                # Rise first: moving inward while low swings the wrist into the column.
                goal = np.array([anchor[0], anchor[1], goal[2]])
        start = self.yield_goals[side] if self.yield_goals[side] is not None else tool
        step = limit_step(np.asarray(start, float), goal, PARK_SPEED, self.dt)
        lead = step - np.asarray(tool, dtype=float)
        if np.linalg.norm(lead) > GOAL_LEASH:  # keep the arm on the straight park path
            step = np.asarray(tool, dtype=float) + lead * (GOAL_LEASH / np.linalg.norm(lead))
        self.yield_goals[side] = step
        # Waiting must not count as a stall or a phase timeout, and the planner
        # resumes smoothly from where the arm actually is.
        planner.phase_time = getattr(planner, "phase_time", task.time) + self.dt
        planner.motion_anchor = None
        if hasattr(planner, "aligned_since"):
            planner.aligned_since = None
        planner.last_goal = np.asarray(tool, dtype=float).copy()
        self.yield_steps[side] += 1
        return replace(command, position=self.yield_goals[side].copy())

    def separate(self, tools, commands, chains=None, hold=None):
        """Resolve arm conflicts; `hold` is an arm that must stay parked regardless."""
        self.yielded_now = None
        chains = chains or [self.arm_chain(s, tools[s]) for s in range(2)]
        if not self.conflict(tools, commands, chains):
            if any(self.yielding):
                # Resume only once the other arm's footprint is clear by a margin.
                for side in range(2):
                    if side != hold:
                        self.yield_goals[side] = None
                        self.freeze_anchors[side] = None
                self.yielding = [side == hold for side in range(2)]
            return commands
        priorities = [self.priority(s, commands[s]) for s in range(2)]
        now = self.env.task.time
        if self.yielding[0] != self.yielding[1]:
            loser = 0 if self.yielding[0] else 1  # hysteresis: keep the same arm waiting
            if priorities[loser] > priorities[1 - loser]:
                loser = 1 - loser
            elif (
                now - self.yield_since > YIELD_WATCHDOG
                and priorities[1 - loser] < 2
                and priorities[loser] >= priorities[1 - loser]
            ):
                loser = 1 - loser  # a long one-sided wait is a deadlock: swap roles
        elif priorities[0] != priorities[1]:
            loser = 0 if priorities[0] < priorities[1] else 1
        else:
            # Equal work: the arm farther from its own mount retreats more easily.
            reach = [np.linalg.norm(tools[s][:2] - self.mounts[s][:2]) for s in range(2)]
            loser = int(reach[1] > reach[0])
        if not self.yielding[loser]:
            self.yield_since = now
        self.yielding = [side == loser for side in range(2)]
        self.yield_goals[1 - loser] = None
        self.freeze_anchors[1 - loser] = None
        winner = 1 - loser
        other = self.lookahead_chain(chains[winner], tools[winner], commands[winner], winner)
        commands = list(commands)
        commands[loser] = self.apply_yield(loser, tools[loser], commands[loser], other)
        self.yielded_now = loser
        return commands

    # ------------------------------------------------------------------ scheduling
    def job_footprint(self, side, view, name, state):
        """Mount, pick and nominal place of a candidate job (tool goals, xy)."""
        planner = self.planners[side]
        task = self.env.task
        source = state.objects[name].position
        if hasattr(planner, "candidate_targets"):
            slots = planner.candidate_targets(view, name, state)
            if not slots:
                return None
            target = task.layout.targets[slots[0]].position
        else:
            target = task.layout.targets[task.layout.assignments[name]].position
        return [self.mounts[side][:2], np.asarray(source[:2]), np.asarray(target[:2])]

    def active_footprint(self, side, tool, state):
        """What the busy arm still has to sweep: mount, tool, its object and its target."""
        planner = self.planners[side]
        task = self.env.task
        name = planner.sample
        if name is None or planner.phase == "select" or name in task.correct:
            return None
        target = (
            task.layout.objects[name].position
            if getattr(planner, "returning", False)
            else task.layout.targets[task.layout.assignments[name]].position
        )
        points = [self.mounts[side][:2], np.asarray(tool[:2])]
        if planner.phase in ("approach", "descend", "grasp"):
            points.append(np.asarray(state.objects[name].position[:2]))
        points.append(np.asarray(target[:2]))
        return points

    @staticmethod
    def polygon_distance(a, b):
        edges_a = [(a[i], a[(i + 1) % len(a)]) for i in range(len(a))]
        edges_b = [(b[i], b[(i + 1) % len(b)]) for i in range(len(b))]
        return min(segment_distance(*ea, *eb) for ea in edges_a for eb in edges_b)

    def crossing(self, side, points):
        """A job crosses over when any of its goals is nearer the other arm's mount."""
        own, other = self.mounts[side][:2], self.mounts[1 - side][:2]
        return any(
            np.linalg.norm(np.asarray(p[:2]) - other) < np.linalg.norm(np.asarray(p[:2]) - own)
            for p in points
        )

    def other_busy(self, side):
        planner = self.planners[1 - side]
        return (
            planner.sample is not None
            and planner.phase != "select"
            and planner.sample not in self.env.task.correct
        )

    def exclusive_job(self, state, tools):
        """(side, footprint) of a running cross-over job, which owns the shared space.

        It stays exclusive until its object is placed and the arm's tool is back on
        its own side, so the retreat out of the other arm's half is covered too.
        """
        task = self.env.task
        for side, planner in enumerate(self.planners):
            name = planner.sample
            if name is None or planner.phase == "select":
                continue
            tool = np.asarray(tools[side][:2], dtype=float)
            home = np.linalg.norm(tool - self.mounts[side][:2]) < np.linalg.norm(
                tool - self.mounts[1 - side][:2]
            )
            if name in task.correct and home:
                continue
            target = (
                task.layout.objects[name].position
                if getattr(planner, "returning", False)
                else task.layout.targets[task.layout.assignments[name]].position
            )
            source = task.layout.objects[name].position
            points = [self.mounts[side][:2], np.asarray(source[:2]), np.asarray(target[:2])]
            if name in self.crossing_jobs.get(side, set()) or self.crossing(side, points[1:]):
                return side, points
        return None

    def compatible_jobs(self, side, view, names, tool, other_tool, state):
        """Jobs whose swept area stays clear of the other arm's current job.

        Cross-over jobs (e.g. side 1 carrying a red tube to side 0's rack)
        start only while the other arm has no job; the other arm then parks out of
        the way (see `commands`). Other jobs must keep clear of the busy arm's area.
        """
        busy = self.active_footprint(1 - side, other_tool, state)
        keep = set()
        reasons = {}
        task = self.env.task
        # Queues keep finished samples; a placed sample looks like a local job (its
        # source is its slot) and would crowd out every real cross-over job.
        names = {n for n in names if n not in task.correct and n not in task.dropped}
        for name in names:
            job = self.job_footprint(side, view, name, state)
            if job is None:
                reasons[name] = "no slot"
                continue
            if self.crossing(side, job[1:]):
                if not self.other_busy(side):
                    keep.add(name)
                else:
                    reasons[name] = "crossing while other busy"
            elif busy is None or self.polygon_distance(job, busy) >= ARM_CLEARANCE:
                keep.add(name)
            else:
                reasons[name] = "overlaps busy arm"
        self.compat_reasons = reasons
        # Prefer jobs on the own side: cross-over jobs only when nothing else is left.
        local = {
            n for n in keep if not self.crossing(side, self.job_footprint(side, view, n, state)[1:])
        }
        return local or keep

    # ------------------------------------------------------------------ commands
    def commands(self):
        backend, task = self.env.backend, self.env.task
        self.hand_over_abandoned()
        global_state = backend.state()
        self.rebalance(global_state)
        commands = []
        tools = []
        states = []
        for side, (robot, planner) in enumerate(
            zip((backend.robot, backend.secondary_robot), self.planners)
        ):
            position, orientation = backend._tool_pose(robot)
            tools.append(np.asarray(position, dtype=float))
            # An opposed right mount faces the left mount. A 180-degree tool
            # rotation swaps identical fingers and avoids exhausting the wrist's
            # range. Present the equivalent frame to the planner, incl. feedback.
            if side and self.opposed:
                orientation = self.swap_fingers(orientation)
            q = robot.get_joint_positions()
            state = replace(
                global_state,
                objects={
                    name: replace(obj, grasped=backend.grasp_owners.get(name) == side)
                    for name, obj in global_state.objects.items()
                },
                ee_position=position,
                ee_orientation=orientation,
                gripper_open=bool(q[6] + q[7] > 0.065),
            )
            states.append(state)
            other = self.planners[1 - side]
            names = self.names[side]
            if (
                other.sample is not None
                and other.phase != "select"
                and other.sample not in task.correct
            ):
                # Reserve the other arm's source and destination columns (along
                # the mount-to-mount axis) until it releases. Independent columns
                # can still run concurrently.
                def along(position):
                    return float(np.asarray(position[:2]) @ self.across)

                source_c = along(task.layout.objects[other.sample].position)
                target_c = along(task.layout.targets[task.layout.assignments[other.sample]].position)
                names = {
                    name
                    for name in names
                    if abs(along(task.layout.objects[name].position) - source_c) >= 0.18
                    and abs(
                        along(task.layout.targets[task.layout.assignments[name]].position)
                        - target_c
                    )
                    >= 0.18
                }
            reserved = (
                {task.layout.assignments[other.sample]}
                if other.sample is not None and other.sample not in task.correct
                else set()
            )
            if side == 0:
                both_tools = [
                    np.asarray(backend._tool_pose(r)[0], dtype=float)
                    for r in (backend.robot, backend.secondary_robot)
                ]
            exclusive = self.exclusive_job(state, both_tools)
            if exclusive is not None and exclusive[0] != side and planner.phase == "select":
                names = set()  # the other arm owns the shared space; wait parked
            elif planner.phase == "select":
                other_robot = backend.secondary_robot if side == 0 else backend.robot
                other_tool = np.asarray(backend._tool_pose(other_robot)[0], dtype=float)
                view = ArmTask(task, names, reserved)
                names = self.compatible_jobs(side, view, names, position, other_tool, state)
            last_goal = getattr(planner, "last_goal", None)
            self.previous_goals[side] = None if last_goal is None else np.array(last_goal, float)
            commands.append(planner.command(ArmTask(task, names, reserved), state))
        for side, planner in enumerate(self.planners):
            # Remember which running jobs are cross-overs (targets can change later).
            if planner.phase == "approach" and planner.sample is not None:
                points = [
                    task.layout.objects[planner.sample].position,
                    task.layout.targets[task.layout.assignments[planner.sample]].position,
                ]
                if self.crossing(side, points):
                    self.crossing_jobs.setdefault(side, set()).add(planner.sample)
        commands = [self.rest_if_done(side, tools[side], c) for side, c in enumerate(commands)]
        exclusive = self.exclusive_job(global_state, tools)
        hold = None
        if exclusive is not None and self.planners[1 - exclusive[0]].phase == "select":
            hold = 1 - exclusive[0]
        intended = [np.asarray(c.position, dtype=float).copy() for c in commands]
        commands = self.separate(tools, commands, hold=hold)
        if hold is not None and self.yielded_now != hold:
            # Proactively clear the shared space for the cross-over job.
            owner, footprint = exclusive
            z = tools[owner][2]
            chain = [np.array([p[0], p[1], z]) for p in footprint]
            commands = list(commands)
            commands[hold] = self.apply_yield(hold, tools[hold], commands[hold], chain)
            self.yielding[hold] = True
        self.prev_commands = intended
        result = []
        for side, command in enumerate(commands):
            if side and self.opposed:
                command = (
                    replace(command, yaw=command.yaw + np.pi)
                    if command.orientation is None
                    else replace(command, orientation=self.swap_fingers(command.orientation))
                )
            if task.task_id == "C1" and self.env.config["control"].get(
                "c1_expert_grasp_assist", False
            ):
                from roboclock.envs.lab.rack_organization import update_expert_grasp

                planner = self.planners[side]
                update_expert_grasp(backend, planner.sample, command.close, states[side], side)
            result.append(command)
        return result

    def rest_point(self, side):
        """The arm's start-pose tool position: in front of its own mount, above the work.

        Holding the measured tool once an arm has no work left let it sag; on the
        side-by-side rig a retracted idle wrist then pressed link5 into its own
        mount column for the rest of the episode.
        """
        yaw = mount_yaws(self.env.config, self.env.task.task_id)[side]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return self.mounts[side] + forward * REST_FORWARD + [0.0, 0.0, REST_ABOVE_MOUNT]

    def rest_if_done(self, side, tool, command):
        """Send an arm whose queue is finished to its rest point (bounded, leashed)."""
        task, planner = self.env.task, self.planners[side]
        pending = [
            name
            for name in self.names[side]
            if name not in task.correct
            and name not in task.dropped
            and getattr(planner, "retries", {}).get(name, 0) < 3
        ]
        if pending or planner.phase != "select" or command.close:
            self.rest_goals[side] = None
            return command
        tool = np.asarray(tool, dtype=float)
        start = self.rest_goals[side] if self.rest_goals[side] is not None else tool
        goal = self.rest_point(side)
        if tool[2] < goal[2] - 0.02 and np.linalg.norm(tool[:2] - goal[:2]) > 0.02:
            goal = np.array([tool[0], tool[1], goal[2]])  # rise first, clear of the racks
        step = limit_step(np.asarray(start, float), goal, PARK_SPEED, self.dt)
        lead = step - tool
        if np.linalg.norm(lead) > GOAL_LEASH:
            step = tool + lead * (GOAL_LEASH / np.linalg.norm(lead))
        self.rest_goals[side] = step
        planner.last_goal = tool.copy()
        return replace(command, position=step.copy())

    @staticmethod
    def swap_fingers(quaternion):
        w, x, y, z = quaternion
        return np.array([-z, y, -x, w])
