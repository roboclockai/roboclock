"""The primitives themselves: closed-loop skills that turn one tool call into joint commands.

A skill is a generator. Each `yield` hands back control for one control step, during which the
policy sends the current command (`Skills.action`); the generator is resumed with the next
observation. It returns the text the model is told, or raises `Failed` with it.

Skills run on proprioception only: joint positions, the gripper opening and the clock. What they
report is what was measured (the tool point at the measured joints, the opening, contact stops,
times). They draw no conclusion from it: nothing says an object is held or in place.
"""

import copy
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from roboclock.envs.lab.robot import ARM_DOF, DOF, JOINT_HIGH, JOINT_LOW
from roboclock.policies.vision_llm import HEAD_SIZE, IK_WAYPOINTS, SIDES, interpolate, pixel_to_world
from roboclock.policies.vision_llm_primitives.geometry import (
    UP,
    about_axis,
    arm_capsules,
    capsules_gap,
    hand_rotation,
)
from roboclock.policies.vision_llm_primitives.perception import (
    PIXEL_MAX,
    mask_geometry,
    opening_under,
    pick_instance,
    point_request,
    rgb_at,
    segment_request,
)
from roboclock.policies.vision_llm_primitives.settings import (
    CALL,
    FINGER_CLOSED,
    FINGER_MAX,
    MAX_WAIT,
    defaults,
)
from roboclock.policies.x5_kinematics import transform

STALL_WINDOW = 0.1  # s over which a guarded hand must have advanced ...
STALL_FRACTION = 0.25  # ... at least this fraction of the commanded speed, or it has stalled
SETTLE_MAX = 0.03  # m of an offset still missing after arriving (beyond the arrival setting) that move_by sends again
STALL_STILL = 0.0002  # m in STALL_WINDOW under which a hand is not advancing, however slow its command
GUARD_START = 0.3  # s from the start of a guarded motion in which a trailing hand is not yet a stall (it is setting off)
RELAXED_ORIENTATION = 0.02  # IK orientation weight when the asked orientation is out of reach
ARRIVAL_JOINT = 0.03  # rad each measured joint may still be from its command when the hand has arrived
ARRIVAL_FINGER = 0.002  # m a finger may still be from its command after an open or close
ARRIVE_LIMIT = 20.0  # s a motion waits, at most, for an arm that is still getting nearer (industry's are slow)
ARRIVE_STILL = 0.3  # s without getting nearer after which an arm is taken to have stopped
ARRIVE_PROGRESS, ARRIVE_PROGRESS_JOINT = 0.001, 0.005  # m, rad closed over ARRIVE_STILL by an arm still arriving
CLOSE_LEAD = 0.004  # m the finger command leads the measured finger while closing (free tracking lag)
CLOSE_STILL = 0.0005  # m the opening may still change over STALL_WINDOW once the fingers have stopped
CLOSE_EXTRA = 2.0  # s a close may run past its stroke time waiting for the fingers to stop
WATCH_EVERY = 0.1  # s between colour checks of a watched pixel
CLEARANCE_SAMPLES = 12  # poses checked along a motion against the other arm
CLEARANCE_SLACK = 0.002  # m a motion may close an already small gap by before it is refused
MEASURED_GAP = 0.02  # m between measured and commanded tool point from which a report states both
PLACE_HEIGHT = {"on": 0.0, "into": 0.05, "drop": 0.10}
APPROACH_PITCH = {"top": math.pi / 2, "front": 0.0}  # rad below level the fingers point
ARC_STEP = 0.1  # rad between the poses a swing about a hinge is solved at
ORIENTATION_MISS = 0.09  # rad (5 deg) the solved hand may be off the asked orientation before it is said
# rad a commanded joint may be ahead of the measured one; beyond it the trajectory waits. A
# position drive follows at about (stiffness / damping) x this lead: 0.8 rad/s on industry's
# arms (700 / 70, capped at 0.70 to 0.75 rad/s), 2 rad/s on lab's (1500 / 60), 2.7 rad/s on
# household's (3000 / 90). Measured on industry B2's first move (2.2 rad of joint 4): 5.5 s at
# 0.05, 3.6 s at 0.08, 5.2 s at 0.15; at 0.3 and above the joint creeps and never arrives.
JOINT_LEAD = 0.08
OFFSET_MAX = 0.2  # rad of standing offset a trajectory takes over from its start
PACE_MIN_RATE = 0.3  # rad/s a joint must be commanded at for the pace to watch its lead (every rig's cap is above it)
PACE_PROGRESS = 0.02  # rad a waiting arm's joints must move in ARRIVE_STILL, or it is taken to be held off


class Pace:
    """Advances a trajectory no faster than the arm follows it.

    Every rig caps its joints' speed (industry at 0.70 to 0.75 rad/s), and a trajectory timed
    from a tool speed alone can ask for several times that. A command that runs far ahead of
    the arm takes the hand off its line and can leave a joint a radian behind for good. So the
    trajectory's phase advances only while that leaves no commanded joint more than JOINT_LEAD
    ahead of its measured one (or brings the command nearer the arm: a hand backing off from
    what held it): a motion asked faster than the arm can go runs at the arm's own limit, on
    its path.

    Only joints commanded faster than PACE_MIN_RATE are watched: below it no rig's cap is in
    play, and a lead is a load or a contact, which the guard and the arrival wait deal with.

    A joint that carries a load settles off its command by more than that and still follows
    (a wrist lifting its side of a carton). The lead is therefore counted from the joints'
    standing offset: what the arm stands off by when the trajectory starts, and, when the
    load comes on during it, what it stands at once the phase has waited ARRIVE_STILL without
    the arm moving (taken once). If the arm then moves on, it was a load. If it has not moved
    after a further ARRIVE_STILL, something holds it off: `advance` returns None (and
    otherwise the next phase)."""

    def __init__(self, skills, i, count, joints_at):
        self.skills, self.i, self.count, self.joints_at = skills, i, count, joints_at
        self.phase, self.waited, self.mark = 0.0, 0, None
        skills.waiting[i] = False
        # The joints' standing offset from their command. A trajectory starts from an arm at
        # rest, so what it stands off by then is one already: a load it holds, a press it keeps.
        self.offset = np.clip(skills.command[i] - skills.measured(i), -OFFSET_MAX, OFFSET_MAX)
        self.assumed = False  # an offset was taken and the arm has not moved since
        self.still = max(2, round(ARRIVE_STILL * skills.control_hz))

    def advance(self):
        skills, i = self.skills, self.i
        measured = skills.measured(i)
        step = min(1.0 / self.count, 1.0 - self.phase)
        ahead = self.joints_at(self.phase + step)
        # Only a joint that is being commanded fast can be outrun for its speed. One commanded
        # slowly that stands off its command does so for a load or a contact, which is not the
        # pace's business: a slow two-handed lift must rise as it is commanded, not in stops.
        fast = np.abs(ahead - skills.command[i]) * skills.control_hz > PACE_MIN_RATE
        if not fast.any():
            skills.waiting[i] = False
            if skills.waiting[1 - i]:
                return self.phase
            self.waited, self.mark, self.assumed = 0, None, False
            self.phase = 1.0 if self.phase + step > 1.0 - 1e-9 else self.phase + step
            return self.phase
        now = float(np.max(np.abs(skills.command[i] - measured - self.offset)[fast]))
        lead = float(np.max(np.abs(ahead - measured - self.offset)[fast]))
        if lead > max(JOINT_LEAD, now) and now < JOINT_LEAD:
            step *= (JOINT_LEAD - now) / (lead - now)  # as much of the step as the lead allows
            lead = JOINT_LEAD
        if self.mark is not None and float(np.max(np.abs(measured - self.mark))) >= PACE_PROGRESS:
            self.waited, self.mark, self.assumed = 0, None, False  # the arm moves: it follows
        if lead <= max(JOINT_LEAD, now):
            skills.waiting[i] = False
            if skills.waiting[1 - i]:
                # Two arms moving together keep in step: while the other one's trajectory waits
                # for its arm, this one waits too, or what they carry between them would turn.
                return self.phase
            if not self.assumed:
                self.waited, self.mark = 0, None
            self.phase = 1.0 if self.phase + step > 1.0 - 1e-9 else self.phase + step
            return self.phase
        skills.waiting[i] = True
        if self.mark is None:
            self.waited, self.mark = 0, measured.copy()
        self.waited += 1
        if self.waited > self.still:
            if self.assumed:
                skills.waiting[i] = False
                return None
            self.offset = skills.command[i] - measured
            self.assumed, self.waited, self.mark = True, 0, measured.copy()
        return self.phase


class Failed(Exception):
    """A call could not do what it was asked; the message goes to the model."""


class Skills:
    """Both arms' commanded state for one episode, and the skills that move it."""

    def __init__(self, arms, camera, joints, control_hz, finger_min=0.0, sam3_url=None, finger_max=FINGER_MAX):
        self.arms, self.camera = arms, camera
        self.home = [joints[DOF * i : DOF * i + ARM_DOF].copy() for i in range(2)]
        self.command = [joints[DOF * i : DOF * i + ARM_DOF].copy() for i in range(2)]
        self.fingers = [float(joints[DOF * i + ARM_DOF]) for i in range(2)]
        self.control_hz = control_hz
        # Lowest finger target (m per finger) the rig's action space accepts: 0 is the closed stop,
        # household takes down to -0.03, which lets a close squeeze a thin object.
        self.finger_min = finger_min
        # Highest finger target that still opens the jaws further: half the rig's widest opening.
        self.finger_max = finger_max
        self.sam3_url = sam3_url
        self.settings = defaults()  # the episode's settings; the configure tool changes them
        self.args = {}  # the running call's arguments
        self.view = None  # what the model saw when it answered: {"rgb", "depth"}
        self.obs, self.time = None, 0.0
        self.diagnostics = []  # guarded segments: largest lag and whether they stopped on contact
        self.short = [None, None]  # per arm: how far before its goal the last motion was held off (m)
        self.waiting = [False, False]  # per arm: its running trajectory is waiting for the arm (Pace)
        self.press = [None, None]  # per arm: (press vector, the command it holds at) after a guard stop

    def action(self):
        """The joint targets to send this control step."""
        out = np.zeros(2 * DOF, dtype=np.float32)
        for i in range(2):
            out[DOF * i : DOF * i + ARM_DOF] = np.clip(self.command[i], JOINT_LOW[:ARM_DOF], JOINT_HIGH[:ARM_DOF])
            out[DOF * i + ARM_DOF : DOF * (i + 1)] = min(max(self.fingers[i], self.finger_min), self.finger_max)
        return out

    def look(self, observation):
        """Keep the views a model query is answering from: pixel targets and watches refer to them."""
        self.view = {key: np.array(observation[key]) for key in ("rgb", "depth") if key in observation}

    def q(self, key):
        """A quantity for the running call: its own argument, else the episode's setting, else
        the argument's default."""
        if key in self.args:
            return self.args[key]
        if key in self.settings:
            return self.settings[key]
        return CALL[key].default

    # ------------------------------------------------------------------ one call
    def run(self, call, observation):
        """Generator for one call: each `yield` is one control step; it returns a report."""
        self.args = args = call["args"]
        notes = []
        start = float(np.ravel(observation["sim_time"])[0])
        self.obs, self.time = observation, start
        try:
            self.resolve(call, notes)
            if "when" in args:
                timeout = args["when"].get("timeout", self.settings["watch_timeout"])
                fired = yield from self.watch(args["when"]["watch"], timeout)
                if fired is None:
                    raise Failed(f"nothing changed at the watched pixel within {timeout:.1f} s")
                notes.append(f"triggered at t={fired['t']:.2f} s (colour {fired['before']} -> {fired['after']})")
            report = yield from getattr(self, "skill_" + call["name"])(call, notes)
        except Failed as error:
            # Keep what was measured before the failure: the model is told it after every call.
            notes.extend([str(error), f"took {self.time - start:.1f} s, now t={self.time:.2f} s"])
            raise Failed("; ".join(n for n in notes if n)) from None
        if call["name"] != "configure":
            notes.extend([report, f"took {self.time - start:.1f} s, now t={self.time:.2f} s"])
        else:
            notes.append(report)
        return "; ".join(n for n in notes if n)

    def run_pair(self, calls, observation):
        """Generator for two calls run at the same time, one per arm (the `together` tool).

        Each control step advances both skills, so both arms move in the same step. A skill keeps
        its own call arguments, clock and observation on a shallow copy of this object; the
        commanded joints, the fingers and the settings are shared, and each skill writes only
        its own arm's. Returns [(failed, text), (failed, text)] in the calls' order: one call
        failing does not stop the other."""
        skills = [copy.copy(self).run(call, observation) for call in calls]
        sides = [SIDES.index(call["args"]["arm"]) for call in calls]
        results = [None] * len(calls)
        fresh = True
        while True:
            for k, skill in enumerate(skills):
                if results[k] is None:
                    try:
                        skill.send(None if fresh else observation)
                    except StopIteration as done:
                        results[k] = (False, done.value or "done")
                    except Failed as error:
                        results[k] = (True, str(error))
                    if results[k] is not None:
                        self.waiting[sides[k]] = False  # a call that has ended holds the other up no longer
            if all(result is not None for result in results):
                return results
            fresh = False
            observation = yield
            self.obs = observation
            self.time = float(np.ravel(observation["sim_time"])[0])

    def resolve(self, call, notes):
        """World points for the call's targets, from the depth the model's views were taken with."""
        args = call["args"]
        call["points"] = {}
        refs = {key: args[key] for key in ("to", "object", "target") if key in args}
        if "pivot" in args.get("hinge", {}):
            refs["pivot"] = args["hinge"]["pivot"]
        for key, ref in refs.items():
            if "xyz" in ref:
                call["points"][key] = np.asarray(ref["xyz"], dtype=float)
                continue
            point, problem = pixel_to_world(
                self.view or {}, self.camera, ref["u"] / PIXEL_MAX * HEAD_SIZE[0], ref["v"] / PIXEL_MAX * HEAD_SIZE[1], HEAD_SIZE
            )
            if point is None:
                raise Failed(f"{key} {problem.replace('refused: ', '')}")
            call["points"][key] = point
            if call["name"] == "place" and args["mode"] == "into":
                radius = self.settings["rim_radius"]
                found = opening_under(self.view, self.camera, ref, point, radius)
                if found is not None:
                    rim, crossing = found
                    call["points"][key] = np.array([crossing[0], crossing[1], point[2]])
                    notes.append(
                        f"target pixel is the surface at {np.round(point, 3).tolist()}, inside an opening "
                        f"whose rim is at z={rim:.3f}; releasing over {np.round(crossing[:2], 3).tolist()}, "
                        "where the pixel's line of sight crosses the rim's plane"
                    )
                else:
                    notes.append(
                        f"target pixel is the surface at {np.round(point, 3).tolist()}; no level rim stands "
                        f"around it within the rim_radius setting {radius:g} (a larger one reaches further), "
                        "so the release is over that point"
                    )
                continue
            notes.append(f"{key} pixel is the surface at {np.round(point, 3).tolist()}")
            if self.sam3_url and ref.get("name"):
                call.setdefault("objects", {})[key] = self.segment(ref)

    def segment(self, ref):
        """SAM3 geometry of the named object under a clicked pixel, or a note why there is none."""
        rgb = self.view["rgb"]
        try:
            masks = segment_request(self.sam3_url, rgb, [ref["name"]])
        except RuntimeError as error:
            return f"segmenter failed ({error}); used the clicked point"
        picked, how = pick_instance(masks, ref["u"], ref["v"]), "by name"
        if picked is None:
            # Unusual objects score low by name; the click itself still says which object is meant.
            try:
                picked, how = point_request(self.sam3_url, rgb, ref["u"], ref["v"]), "under the click"
            except RuntimeError as error:
                return f"SAM3 found no '{ref['name']}' at the clicked pixel and {error}; used the clicked point"
        if picked is None:
            return f"SAM3 found no '{ref['name']}' at the clicked pixel ({len(masks)} elsewhere); used the clicked point"
        geometry = mask_geometry(picked[0], self.view["depth"], self.camera)
        if geometry is None:
            return f"SAM3 '{ref['name']}' has too little depth; used the clicked point"
        geometry.update(name=ref["name"], score=picked[1], how=how)
        return geometry

    # ------------------------------------------------------------------ motion
    # Every generator below keeps self.obs and self.time current.

    def step(self):
        observation = yield
        self.obs = observation
        self.time = float(np.ravel(observation["sim_time"])[0])
        return observation

    def hold(self, seconds):
        for _ in range(max(1, math.ceil(seconds * self.control_hz - 1e-9))):
            yield from self.step()

    def watch(self, pixel, timeout):
        """Hold until the head-image colour at `pixel` has moved by the watch_change setting from
        what the model saw; the event {"t", "before", "after"}, or None after `timeout`."""
        if not self.view or "rgb" not in self.view:
            raise Failed("no head image to watch")
        radius, change = self.settings["watch_radius"], self.settings["watch_change"]
        before = rgb_at(self.view["rgb"], pixel, radius)
        every = max(1, round(WATCH_EVERY * self.control_hz))
        steps = max(1, math.ceil(timeout * self.control_hz))
        for index in range(steps):
            observation = yield from self.step()
            if index % every == every - 1 or index == steps - 1:
                now = rgb_at(observation["rgb"], pixel, radius)
                if np.linalg.norm(now - before) >= change:
                    return {"t": self.time, "before": np.round(before).astype(int).tolist(), "after": np.round(now).astype(int).tolist()}
        return None

    def tool(self, i, joints=None):
        """The commanded tool pose of arm i (or the pose at `joints`)."""
        return self.arms[i].tool_pose(self.command[i] if joints is None else joints)

    def measured(self, i):
        joints = np.asarray(self.obs["joint_positions"], dtype=float)
        return joints[DOF * i : DOF * i + ARM_DOF]

    def actual(self, i):
        """The tool point at the measured joints: where the hand is, not where it was sent."""
        return self.arms[i].tool_pose(self.measured(i))[:3, 3]

    def where(self, i):
        """The measured tool point, and its distance from the commanded one when they differ."""
        actual, commanded = self.actual(i), self.tool(i)[:3, 3]
        text = f"measured tool point {np.round(actual, 3).tolist()}"
        gap = float(np.linalg.norm(actual - commanded))
        if self.short[i] is not None:
            text += f"; the arm stopped following {self.short[i]:.3f} m before where it was sent (something holds it off)"
        elif gap > MEASURED_GAP:
            text += f", {gap:.3f} m from the commanded {np.round(commanded, 3).tolist()}"
        return text

    def route(self, i, position, rotation=None):
        """Joint path for arm i's tool point going straight from its commanded pose to `position`
        (orientation `rotation`): (start pose, position, rotation, path, IK residual)."""
        start = self.tool(i)
        rotation = start[:3, :3] if rotation is None else rotation
        position = np.asarray(position, dtype=float)
        path, residual = self.ik_path(i, start, position, rotation)
        if residual > 0.01:
            # Out of reach with this orientation: let the hand tilt if that gets the point there.
            relaxed, loose = self.ik_path(i, start, position, rotation, RELAXED_ORIENTATION)
            if loose < residual - 0.005:
                path, residual = relaxed, loose
                rotation = self.arms[i].tool_pose(path[-1])[:3, :3]
        return start, position, rotation, path, residual

    def reach(self, i, position, rotation, what, clear=True):
        """The route to a pose, or Failed (before the arm moves) when the IK misses it by more
        than the call's tolerance, or (`clear`) the way there comes too near the other arm."""
        route = self.route(i, position, rotation)
        tolerance = self.q("tolerance")
        if route[-1] > tolerance:
            raise Failed(
                f"{what} is out of reach by {route[-1]:.3f} m (tolerance {tolerance:g} m); the arm did not move"
            )
        if clear:
            self.keep_clear(i, route[3], "the arm did not move")
        return route

    def keep_clear(self, i, path, then):
        """Failed when the joint `path` of arm i would close the gap to the other arm (which holds
        still) to under the call's arm_clearance. A path that only moves the arms apart, or does
        not bring them nearer than they already are, always passes."""
        need = self.q("arm_clearance")
        other = arm_capsules(self.arms[1 - i], self.measured(1 - i))
        start = capsules_gap(arm_capsules(self.arms[i], path[0]), other)
        for phase in np.linspace(0.0, 1.0, CLEARANCE_SAMPLES + 1)[1:]:
            gap = capsules_gap(arm_capsules(self.arms[i], interpolate(path, phase)), other)
            if gap < need and gap < start - CLEARANCE_SLACK:
                raise Failed(
                    f"the path would bring the {SIDES[i]} arm to {gap:.3f} m from the {SIDES[1 - i]} arm, "
                    f"which holds still (arm_clearance {need:g} m); {then}"
                )

    def travel(self, i, position, rotation=None, speed=None, fingers=None, guard=False, route=None):
        """Move arm i's tool point in a straight line to `position` (orientation `rotation`),
        fingers to `fingers`, from its commanded pose (or along a `route` already solved). A
        guarded move stops at contact. Returns (contact point or None, IK residual)."""
        if route is None:
            route = self.route(i, position, rotation)
            self.keep_clear(i, route[3], "the arm stopped before this part of the motion")
        start, position, rotation, path, residual = route
        speed = self.q("speed") if speed is None else speed
        contact_lag = self.q("contact")
        f0 = self.fingers[i]
        f1 = f0 if fingers is None else fingers
        turn = Rotation.from_matrix(rotation @ start[:3, :3].T).magnitude()
        seconds = max(
            np.linalg.norm(position - start[:3, 3]) / speed,
            turn / self.q("turn_speed"),
            abs(f1 - f0) / self.finger_max * self.q("gripper_time"),
        )
        count = max(1, math.ceil(seconds * self.control_hz - 1e-9))
        direction = position - start[:3, 3]
        direction /= max(np.linalg.norm(direction), 1e-9)
        window = max(2, round(STALL_WINDOW * self.control_hz))
        grace = round(GUARD_START * self.control_hz)
        trail, max_lag = [], 0.0
        pace, step, sent = Pace(self, i, count, lambda phase: interpolate(path, phase)), -1, []
        self.short[i] = None
        while pace.phase < 1.0:
            phase = pace.advance()
            if phase is None:
                # Held off: stop where the command is (twice JOINT_LEAD ahead of the arm at most),
                # so that what follows is planned from near where the hand really is.
                if guard:
                    actual = self.actual(i)
                    self.command[i] = self.measured(i).copy()
                    self.diagnostics.append({"t": self.time, "arm": i, "max_lag": max_lag, "contact": True})
                    self.waiting[i] = False
                    return actual, residual
                self.short[i] = float(np.linalg.norm(position - self.actual(i)))
                break
            step += 1
            self.command[i] = interpolate(path, phase)
            self.fingers[i] = f0 + (f1 - f0) * phase
            yield from self.step()
            if guard:
                actual = self.actual(i)
                trail.append(float(actual @ direction))
                sent.append(float(self.tool(i)[:3, 3] @ direction))
                lag = float(np.linalg.norm(self.tool(i)[:3, 3] - actual))
                max_lag = max(max_lag, lag)
                # Contact: the hand trails the command and has nearly stopped advancing. A hand
                # that only lags (the drives settle behind a moving target) still advances.
                # "Nearly stopped" is against how far the command itself went in the window:
                # a command the pace holds back to the joints' speed is slower than `speed`,
                # and a hand keeping up with it is not stalled.
                stalled = len(trail) > window and trail[-1] - trail[-1 - window] < max(
                    STALL_FRACTION * (sent[-1] - sent[-1 - window]), STALL_STILL
                )
                if lag > contact_lag and stalled and step >= grace:
                    # Stop pushing: hold where the hand is, a little into the contact.
                    self.command[i] = self.measured(i).copy()
                    self.diagnostics.append({"t": self.time, "arm": i, "max_lag": max_lag, "contact": True})
                    self.waiting[i] = False
                    return actual, residual
        self.waiting[i] = False
        contact = yield from self.arrive(i, guard, fingers=f1 != f0)
        if guard:
            self.diagnostics.append({"t": self.time, "arm": i, "max_lag": max_lag, "contact": contact is not None})
        return contact, residual

    def arrive(self, i, guard=False, fingers=False):
        """Hold arm i's command until the measured arm has caught up with it.

        The drives trail a moving command, by centimetres at the higher speeds a model may ask
        for, so the commanded trajectory can end long before the hand is there. Without this
        wait the next step (a close, a release, a press stroke, a report) happens on the way.
        The wait ends when the tool point is within the call's `arrival` of its command (and
        the joints, and after an open or close the fingers, are at theirs), when the arm has
        stopped getting nearer (it is held off, or settles with an offset), or after
        ARRIVE_LIMIT. A guarded motion that is held off further than `contact` away has met
        something: the command is set to where the hand is and the contact point is returned."""
        arrival = self.q("arrival")
        window = max(2, round(ARRIVE_STILL * self.control_hz))
        trail = []
        for _ in range(math.ceil(ARRIVE_LIMIT * self.control_hz)):
            actual = self.actual(i)
            gap = float(np.linalg.norm(self.tool(i)[:3, 3] - actual))
            joints = float(np.max(np.abs(self.command[i] - self.measured(i))))
            target = min(max(self.fingers[i], self.finger_min), self.finger_max)
            finger = abs(self.width(i) / 2 - target) if fingers else 0.0
            if gap <= arrival and joints <= ARRIVAL_JOINT and finger <= ARRIVAL_FINGER:
                return None
            trail.append((gap, joints, finger))
            if len(trail) > window:
                before = trail[-1 - window]
                nearer = (
                    before[0] - gap >= ARRIVE_PROGRESS
                    or before[1] - joints >= ARRIVE_PROGRESS_JOINT
                    or before[2] - finger >= ARRIVE_PROGRESS
                )
                if not nearer:
                    if guard and gap > self.q("contact"):
                        self.command[i] = self.measured(i).copy()
                        return actual
                    return None
            yield from self.step()
        return None

    def ik_path(self, i, start, position, rotation, orientation_weight=0.3):
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([start[:3, :3], rotation])))
        path, residual = [self.command[i].copy()], 0.0
        for phase in np.linspace(0.0, 1.0, IK_WAYPOINTS + 1)[1:]:
            waypoint = transform(slerp(phase).as_matrix(), start[:3, 3] + (position - start[:3, 3]) * phase)
            solved, residual = self.arms[i].solve(waypoint, path[-1], orientation_weight=orientation_weight)
            path.append(solved)
        return np.stack(path), residual

    def approach_frame(self, i, point, approach, yaw=None):
        """(rotation, unit vector the fingers move along) for approaching `point`."""
        current = self.tool(i)
        reach = point - self.arms[i].base[:3, 3]
        reach[2] = 0.0
        if np.linalg.norm(reach) < 1e-6:
            reach = self.arms[i].base[:3, 0].copy()
        reach /= np.linalg.norm(reach)
        tilt = self.args.get("tilt", {})
        if approach == "keep" and tilt:
            approach = "front"  # a tilt is measured from the presets; keep has none (move's default)
        if approach == "keep":
            return current[:3, :3], current[:3, 2].copy()
        # The presets are two points of one family: fingers along `heading` in the table plane,
        # `pitch` below level, jaws turned `roll` about the fingers from closing level.
        heading = math.atan2(reach[1], reach[0])
        if approach == "top" and yaw is not None:
            heading = yaw - math.pi / 2  # `yaw` names the jaws' closing direction of a top approach
        heading = tilt.get("heading", heading)
        pitch = tilt.get("pitch", APPROACH_PITCH[approach])
        roll = tilt.get("roll", 0.0)
        rotation = hand_rotation(heading, pitch, roll)
        # The jaws are symmetric: from above, take the half turn nearer the current wrist.
        if approach == "top" and "roll" not in tilt and rotation[:, 1] @ current[:3, 1] < 0:
            rotation = hand_rotation(heading, pitch, roll + math.pi)
        return rotation, rotation[:, 2].copy()

    @staticmethod
    def off_by(rotation, asked):
        """Angle (rad) between two orientations."""
        return float(Rotation.from_matrix(rotation @ np.asarray(asked).T).magnitude())

    def orientation_note(self, i, asked):
        """Says so when the commanded hand is not at the orientation that was asked for: the
        route gave it up to reach the point (`route`), or the joints could not turn that far."""
        off = self.off_by(self.tool(i)[:3, :3], asked)
        return f"the hand is {math.degrees(off):.0f} deg off the asked orientation (out of its joints' range there)" if off > ORIENTATION_MISS else ""

    def width(self, i):
        """The measured distance between arm i's jaws."""
        return float(np.asarray(self.obs["gripper_width"], dtype=float)[i])

    def reach_note(self, residual):
        return f"target out of reach by {residual:.3f} m" if residual > 0.01 else ""

    def set_fingers(self, i, finger):
        """Open or close arm i's gripper to a finger target without moving the arm."""
        yield from self.travel(i, self.tool(i)[:3, 3], fingers=finger)

    def close(self, i):
        """Run arm i's fingers in until they stop, then command each one `squeeze` past where it
        stopped. Returns (measured opening, squeeze applied, squeeze asked).

        The fingers are position drives, so the grip force is their stiffness times how far the
        command is past the finger. While closing, the command leads the measured finger by at
        most max(squeeze, CLOSE_LEAD): the fingers arrive with no more than the grip that was
        asked for, whatever the object's width."""
        squeeze = self.q("squeeze")
        lead = max(squeeze, CLOSE_LEAD)
        f0, low = self.fingers[i], self.finger_min
        stroke = max(1, math.ceil(abs(f0 - low) / self.finger_max * self.q("gripper_time") * self.control_hz))
        window = max(2, round(STALL_WINDOW * self.control_hz))
        trail = []
        for step in range(stroke + math.ceil(CLOSE_EXTRA * self.control_hz)):
            ramp = f0 + (low - f0) * min(1.0, (step + 1) / stroke)
            held_back = self.width(i) / 2 - lead
            self.fingers[i] = max(ramp, held_back, low)
            yield from self.step()
            trail.append(self.width(i))
            still = len(trail) > window and abs(trail[-1] - trail[-1 - window]) < CLOSE_STILL
            if still and (step >= stroke - 1 or ramp <= held_back):
                break
        yield from self.hold(self.q("settle"))
        stopped = self.width(i) / 2
        self.fingers[i] = max(stopped - squeeze, low)
        applied = stopped - self.fingers[i]
        yield from self.hold(STALL_WINDOW)
        return self.width(i), applied, squeeze

    def take(self, i, point, approach, depth, yaw, notes):
        """Open, approach `point`, go `depth` past it and close. Appends what was measured to
        `notes`; Failed when the pose is out of reach or the fingers close under min_opening."""
        rotation, along = self.approach_frame(i, point, approach, yaw)
        route = self.reach(i, point - along * self.q("standoff"), rotation, "the approach pose")
        yield from self.travel(i, None, fingers=self.args.get("opening", 2 * self.finger_max) / 2, route=route)
        contact, residual = yield from self.travel(i, point + along * depth, rotation, speed=self.q("approach_speed"), guard=True)
        if contact is not None:
            notes.append(f"stopped on contact {np.linalg.norm(point + along * depth - contact):.3f} m short")
        notes.append(self.reach_note(residual))
        notes.append(self.orientation_note(i, rotation))
        width, applied, asked = yield from self.close(i)
        limited = f" of the {asked:.3f} m asked (finger command limit {self.finger_min:.3f} m)" if applied < asked - 1e-4 else ""
        notes.append(f"closed: opening {width:.3f} m, squeezed {applied:.3f} m per finger{limited}")
        least = self.q("min_opening")
        if width < least:
            raise Failed(f"the fingers met: opening {width:.3f} m is under min_opening {least:g} m")

    def arm(self, call):
        return SIDES.index(call["args"]["arm"])

    def refined(self, call, key, approach, depth, notes):
        """(point, depth, yaw) for grasping the call's `key` target, from its SAM3 mask when there
        is one: the object's centre (top centre from above, mid-height from the front), the jaws
        across its narrow side, and a depth from its size unless the model gave one."""
        args, point = call["args"], call["points"][key]
        yaw = args.get("yaw")
        found = call.get("objects", {}).get(key)
        if found is None:
            return point, args.get("depth", depth), yaw
        if isinstance(found, str):
            notes.append(found)
            return point, args.get("depth", depth), yaw
        x, y = found["centre"]
        tall = found["top"] - found["bottom"]
        if approach == "front":
            target = np.array([x, y, found["bottom"] + 0.5 * tall])
            default = 0.5 * min(found["width"], found["length"])
        else:
            target = np.array([x, y, found["top"]])
            default = float(np.clip(0.5 * tall, 0.01, 0.03))
            yaw = found["narrow_yaw"] if yaw is None else yaw
        notes.append(
            f"SAM3 '{found['name']}' ({found.get('how', 'by name')}, score {found['score']:.2f}): top z={found['top']:.3f}, footprint "
            f"{found['length']:.3f} x {found['width']:.3f} m; grasping at {np.round(target, 3).tolist()}, "
            f"{np.linalg.norm(target - point):.3f} m from the clicked point"
        )
        if approach != "front" and found["width"] > 2 * self.finger_max - 0.008:
            notes.append(f"its narrow side ({found['width']:.3f} m) is about as wide as the open gripper ({2 * self.finger_max:.3f} m)")
        return target, args.get("depth", default), yaw

    # ------------------------------------------------------------------ the tools
    def gripper_target(self, args):
        """The finger target a move's `opening` or `gripper` argument asks for (None: keep)."""
        if "opening" in args:
            return args["opening"] / 2
        return {"open": self.finger_max, "close": FINGER_CLOSED}.get(args.get("gripper", "keep"))

    def skill_move(self, call, notes):
        i, args = self.arm(call), call["args"]
        goal = call["points"]["to"] + UP * args.get("height", 0.0)
        rotation, _ = self.approach_frame(i, goal, args.get("approach", "keep"), args.get("yaw"))
        route = self.reach(i, goal, rotation, "the target")
        _, residual = yield from self.travel(i, None, fingers=self.gripper_target(args), route=route)
        return "; ".join(n for n in [self.where(i), self.reach_note(residual), self.orientation_note(i, rotation)] if n)

    def pressed(self, i):
        """The press a guard stop left on arm i (the vector its command sits inside what the hand
        met), or None when the arm has been commanded elsewhere since."""
        press = self.press[i]
        if press is not None and np.array_equal(press[1], self.command[i]):
            return press[0]
        self.press[i] = None
        return None

    def skill_move_by(self, call, notes):
        i, args = self.arm(call), call["args"]
        by = np.asarray(args["by"], dtype=float)
        guard = bool(args.get("guard"))
        start = self.actual(i)
        # After a guard stop the command sits `contact` inside what the hand met: the press on
        # it. An offset that points away from the contact gives the press up first, or the hand
        # would only relax by it; any other keeps it (two hands carrying a thing between them
        # move sideways to their presses).
        press = self.pressed(i)
        keep = press is not None and by @ press >= 0
        goal = self.tool(i)[:3, 3] + by - (0.0 if press is None or keep else press)
        if guard:
            # A guarded offset is sent toward something it will meet on the way: its end point
            # may lie beyond the arm's reach (below a table), so only the other arm is checked.
            route = self.route(i, goal, None)
            self.keep_clear(i, route[3], "the arm did not move")
        else:
            route = self.reach(i, goal, None, "the target")
        speed = self.q("approach_speed") if guard and "speed" not in args else None
        contact, residual = yield from self.travel(
            i, None, speed=speed, fingers=self.gripper_target(args), guard=guard, route=route
        )
        met = ""
        if contact is not None:
            # travel() left the command where the hand stopped; press on by the contact
            # setting so that the hand stays loaded against what it met.
            short = float(np.linalg.norm(goal - contact))
            unit = by / max(np.linalg.norm(by), 1e-9)
            path, _ = self.ik_path(i, self.tool(i), self.tool(i)[:3, 3] + unit * self.q("contact"), self.tool(i)[:3, :3])
            self.command[i] = path[-1]
            self.press[i] = (unit * self.q("contact"), self.command[i].copy())
            met = f"stopped on contact {short:.3f} m short and stays pressed against it"
        elif keep:
            # Still pressed, at the new place. (No correction here: a hand that carries
            # something with the other one settles under the load, and each hand sending its
            # own shortfall again, at its own time, would tilt what they hold.)
            self.press[i] = (press, self.command[i].copy())
        elif not guard and self.short[i] is None:
            # The offset is the measured hand's. An arm settles a little off its command (it
            # sags under its own weight, by up to a centimetre at full reach, and a guard stop
            # takes that sag into the command): when more than the arrival setting is still
            # missing, send the command on by it, once.
            missing = start + by - self.actual(i)
            if self.q("arrival") < np.linalg.norm(missing) <= SETTLE_MAX:
                yield from self.travel(i, self.tool(i)[:3, 3] + missing, speed=self.q("approach_speed"))
        return "; ".join(n for n in [self.where(i), met, self.reach_note(residual)] if n)

    def skill_home(self, call, notes):
        i, args = self.arm(call), call["args"]
        fingers = self.gripper_target(args)
        start, f0 = self.command[i].copy(), self.fingers[i]
        f1 = f0 if fingers is None else fingers
        distance = np.linalg.norm(self.tool(i, self.home[i])[:3, 3] - self.tool(i)[:3, 3])
        seconds = max(distance / self.q("speed"), abs(f1 - f0) / self.finger_max * self.q("gripper_time"))
        self.keep_clear(i, np.stack([start, self.home[i]]), "the arm did not move")
        count = max(1, math.ceil(seconds * self.control_hz))
        pace = Pace(self, i, count, lambda phase: start + (self.home[i] - start) * phase)
        self.short[i] = None
        while pace.phase < 1.0:
            phase = pace.advance()
            if phase is None:
                self.short[i] = float(np.linalg.norm(self.tool(i, self.home[i])[:3, 3] - self.actual(i)))
                break
            self.command[i] = start + (self.home[i] - start) * phase
            self.fingers[i] = f0 + (f1 - f0) * phase
            yield from self.step()
        yield from self.arrive(i, fingers=f1 != f0)
        return f"sent home; {self.where(i)}"

    def skill_grasp(self, call, notes):
        i, args = self.arm(call), call["args"]
        approach = args.get("approach", "top")
        point, depth, yaw = self.refined(call, "object", approach, 0.02, notes)
        yield from self.take(i, point, approach, depth, yaw, notes)
        lift = self.q("lift")
        if lift <= 0:
            return ""
        yield from self.travel(i, self.tool(i)[:3, 3] + UP * lift)
        # Read again after the lift: what closed on the object may not have come up with the hand.
        return f"after the lift: {self.where(i)}, opening {self.width(i):.3f} m"

    def skill_place(self, call, notes):
        i, args = self.arm(call), call["args"]
        point, mode = call["points"]["target"], args["mode"]
        goal = point + UP * args.get("height", PLACE_HEIGHT[mode])
        here = self.tool(i)[:3, 3]
        carry = max(here[2], max(goal[2], point[2]) + self.q("clearance"))
        above = np.array([point[0], point[1], carry])
        rotation = None
        if "yaw" in args:
            # Turn about the vertical until the jaws close along `yaw`; they are symmetric, so
            # the nearer of the two half turns.
            jaws = self.tool(i)[:3, 1]
            delta = (args["yaw"] - math.atan2(jaws[1], jaws[0]) + math.pi / 2) % math.pi - math.pi / 2
            rotation = about_axis(UP, delta) @ self.tool(i)[:3, :3]
        # Reach only: the hand rises first, so the straight line from here is not its path.
        self.reach(i, above, rotation, "above the target", clear=False)
        yield from self.travel(i, [here[0], here[1], carry])
        yield from self.travel(i, above, rotation)
        out = []
        if mode == "drop":
            _, residual = yield from self.travel(i, goal)
            out.append(f"lowered to z={self.actual(i)[2]:.3f}")
        else:
            contact, residual = yield from self.travel(i, goal, speed=self.q("approach_speed"), guard=True)
            out.append(
                f"stopped on contact at z={contact[2]:.3f}, {max(contact[2] - goal[2], 0.0):.3f} m above the release point"
                if contact is not None
                else f"lowered to z={self.actual(i)[2]:.3f} without contact"
            )
        out.append(self.reach_note(residual))
        out.append(f"opening {self.width(i):.3f} m before the release")
        yield from self.set_fingers(i, min(args.get("opening", 2 * self.finger_max) / 2, self.finger_max))
        yield from self.travel(i, self.tool(i)[:3, 3] + UP * self.q("retreat"))
        out.append(f"opened to {self.width(i):.3f} m and backed off")
        return "; ".join(n for n in out if n)

    def skill_press(self, call, notes):
        i, args = self.arm(call), call["args"]
        point = call["points"].get("target")
        depth, standoff = args.get("depth", 0.015), self.q("press_standoff")
        if point is None:
            # From where the hand is (a hover), straight along its fingers.
            rotation = self.tool(i)[:3, :3]
            along = rotation[:, 2].copy()
            start = self.tool(i)[:3, 3]
            point = start + along * standoff
        else:
            rotation, along = self.approach_frame(i, point, args.get("approach", "top"), args.get("yaw"))
            start = point - along * standoff
        route = self.reach(i, start, rotation, "the press start")
        yield from self.travel(i, None, fingers=FINGER_CLOSED, route=route)
        _, residual = yield from self.travel(i, point + along * depth, rotation, speed=self.q("approach_speed"))
        yield from self.hold(self.q("hold"))
        reached = self.actual(i)
        yield from self.travel(i, start, rotation)
        short = float((point + along * depth - reached) @ along)
        report = f"pressed: the measured fingertips ended {short:.3f} m short of the commanded depth; backed off"
        return "; ".join(n for n in [report, self.reach_note(residual)] if n)

    def skill_push(self, call, notes):
        i, args = self.arm(call), call["args"]
        point, by = call["points"].get("target"), np.asarray(args["by"], dtype=float)
        if np.linalg.norm(by) < 1e-4:
            raise Failed("push needs a non-zero by")
        if point is None:
            report = yield from self.shift_hand(i, by)
            return report
        rotation, _ = self.approach_frame(i, point, args.get("approach", "top"), args.get("yaw"))
        unit = by / np.linalg.norm(by)
        offset = self.q("push_offset")
        behind = point - unit * offset
        route = self.reach(i, behind + UP * self.q("standoff"), rotation, "the push start")
        yield from self.travel(i, None, fingers=FINGER_CLOSED, route=route)
        contact, _ = yield from self.travel(i, behind, rotation, speed=self.q("approach_speed"), guard=True)
        yield from self.travel(i, self.tool(i)[:3, 3] + by + unit * offset, rotation, speed=self.q("approach_speed"))
        end = self.actual(i)
        yield from self.travel(i, self.tool(i)[:3, 3] + UP * self.q("retreat"))
        return f"pushed; the measured fingers ended at {np.round(end, 3).tolist()}" + (" (touched down early)" if contact is not None else "")

    def skill_pull(self, call, notes):
        i, args = self.arm(call), call["args"]
        by = np.asarray(args["by"], dtype=float)
        if np.linalg.norm(by) < 1e-4:
            raise Failed("pull needs a non-zero by")
        if "target" not in call["points"]:
            report = yield from self.shift_hand(i, by)
            return report
        approach = args.get("approach", "top")
        point, depth, yaw = self.refined(call, "target", approach, 0.01, notes)
        yield from self.take(i, point, approach, depth, yaw, notes)
        along = self.tool(i)[:3, 2].copy()
        yield from self.travel(i, self.tool(i)[:3, 3] + by, speed=self.q("approach_speed"))
        out = f"dragged; the measured hand ended at {np.round(self.actual(i), 3).tolist()}"
        report = yield from self.let_go(i, along, out)
        return report

    def swing(self, i, hinge, pivot, angle):
        """Swing arm i's hand by `angle` about the hinge's world line (through `pivot`, else the
        tool point), the hand turning with it. Returns the report. Failed, before moving, when
        any pose of the arc is out of reach or comes too near the other arm."""
        axis = np.asarray(hinge.get("axis", UP), dtype=float)
        axis = axis / np.linalg.norm(axis)
        start = self.tool(i)
        centre = start[:3, 3] if pivot is None else np.asarray(pivot, dtype=float)
        arm = start[:3, 3] - centre
        count = max(2, math.ceil(abs(angle) / ARC_STEP))
        path, residual, off = [self.command[i].copy()], 0.0, 0.0
        for step in range(1, count + 1):
            turn = about_axis(axis, angle * step / count)
            pose = transform(turn @ start[:3, :3], centre + turn @ arm)
            # The turn is the point of a swing: weigh orientation as much as position.
            solved, miss = self.arms[i].solve(pose, path[-1], orientation_weight=1.0)
            residual = max(residual, miss)
            off = max(off, self.off_by(self.arms[i].tool_pose(solved)[:3, :3], pose[:3, :3]))
            path.append(solved)
        path = np.stack(path)
        tolerance = self.q("tolerance")
        if residual > tolerance:
            raise Failed(f"the arc is out of reach by {residual:.3f} m (tolerance {tolerance:g} m); the arm did not swing")
        if off > ORIENTATION_MISS:
            raise Failed(
                f"the hand cannot turn that far there: its joints leave it {math.degrees(off):.0f} deg "
                "short of the asked turn; the arm did not swing"
            )
        self.keep_clear(i, path, "the arm did not swing")
        before = self.arms[i].tool_pose(self.measured(i))
        end = self.arms[i].tool_pose(path[-1])
        # travel() times a route by its straight length; an arc is longer than its chord.
        radius = float(np.linalg.norm(arm - axis * (arm @ axis)))
        speed = self.q("approach_speed") * max(np.linalg.norm(end[:3, 3] - start[:3, 3]), 1e-9) / max(radius * abs(angle), 1e-9)
        yield from self.travel(i, None, speed=min(speed, self.q("approach_speed")), route=(start, end[:3, 3], end[:3, :3], path, residual))
        after = self.arms[i].tool_pose(self.measured(i))
        swung = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec() @ axis
        where = f", {radius:.3f} m from its line" if pivot is not None else ""
        return (
            f"the measured hand swung {math.degrees(swung):.0f} deg about the hinge{where} "
            f"(asked {math.degrees(angle):.0f}) and ended at {np.round(after[:3, 3], 3).tolist()}"
        )

    def skill_turn(self, call, notes):
        i, args = self.arm(call), call["args"]
        hinge, pivot = args.get("hinge"), call["points"].get("pivot")
        if hinge is not None:
            along = None
            if "target" in call["points"]:
                approach = args.get("approach", "front")
                point, depth, yaw = self.refined(call, "target", approach, 0.01, notes)
                yield from self.take(i, point, approach, depth, yaw, notes)
                along = self.tool(i)[:3, 2].copy()
            out = yield from self.swing(i, hinge, pivot, args["angle"])
            if along is None:
                return f"{out}; gripper unchanged, opening {self.width(i):.3f} m"
            report = yield from self.let_go(i, self.tool(i)[:3, 2].copy(), out)
            return report
        if "target" not in call["points"]:
            start = self.arms[i].tool_pose(self.measured(i))
            rotation = self.tool(i)[:3, :3]
            turned = Rotation.from_rotvec(-rotation[:, 2] * args["angle"]).as_matrix() @ rotation
            yield from self.travel(i, self.tool(i)[:3, 3], turned)
            achieved = Rotation.from_matrix(self.arms[i].tool_pose(self.measured(i))[:3, :3] @ start[:3, :3].T)
            return f"the measured hand turned {math.degrees(achieved.magnitude()):.0f} deg; gripper unchanged, opening {self.width(i):.3f} m"
        approach = args.get("approach", "front")
        point, depth, yaw = self.refined(call, "target", approach, 0.01, notes)
        yield from self.take(i, point, approach, depth, yaw, notes)
        rotation = self.tool(i)[:3, :3]
        along = rotation[:, 2].copy()
        # Counterclockwise as seen looking along the fingers is a rotation about -along.
        turned = Rotation.from_rotvec(-along * args["angle"]).as_matrix() @ rotation
        yield from self.travel(i, self.tool(i)[:3, 3], turned)
        achieved = Rotation.from_matrix(self.arms[i].tool_pose(self.measured(i))[:3, :3] @ rotation.T)
        out = f"turned; the measured hand turned {math.degrees(achieved.magnitude()):.0f} deg"
        report = yield from self.let_go(i, along, out)
        return report

    def shift_hand(self, i, by):
        """Push or pull from the current hand pose, leaving the gripper as it is."""
        start = self.actual(i)
        yield from self.travel(i, self.tool(i)[:3, 3] + by, speed=self.q("approach_speed"))
        moved = self.actual(i) - start
        share = float(moved @ by / (by @ by))
        return f"the measured hand moved {np.round(moved, 3).tolist()} ({share:.0%} of by); gripper unchanged, opening {self.width(i):.3f} m"

    def let_go(self, i, along, out):
        """Open after a pull or a turn and back off along the fingers."""
        width = self.width(i)
        yield from self.set_fingers(i, self.finger_max)
        yield from self.travel(i, self.tool(i)[:3, 3] - along * self.q("retreat"))
        return f"{out}; opening {width:.3f} m before the release; opened and backed off"

    def skill_wait(self, call, notes):
        args = call["args"]
        if "watch" in args:
            fired = yield from self.watch(args["watch"], args.get("timeout", self.settings["watch_timeout"]))
            if fired is None:
                return "no change at the watched pixel"
            return f"changed at t={fired['t']:.2f} s (colour {fired['before']} -> {fired['after']})"
        seconds = min(max(args["until_t"] - self.time, 0.0), MAX_WAIT) if "until_t" in args else args["seconds"]
        if seconds > 0:
            yield from self.hold(seconds)
        return ""

    def skill_configure(self, call, notes):
        """Change episode settings. It takes no control step: `yield` never runs."""
        self.settings.update(call["args"])
        return "settings now: " + ", ".join(f"{key}={value:g}" for key, value in self.settings.items())
        yield  # pragma: no cover - makes this a generator like the other skills
