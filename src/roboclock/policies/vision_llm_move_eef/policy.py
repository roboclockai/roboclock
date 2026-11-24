"""The move_eef agent: the tool, what the model reads, and the playout of an accepted motion.

`act` is called every control step. While a motion plays it sends the next joint waypoint;
when the arms have come to rest it asks the model (a query step: the only steps charged
inference latency) and turns the answer into the next motion, or into a refusal the model
reads on its next turn.
"""

import json
import math
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from roboclock.envs.lab.robot import ARM_DOF, DOF, JOINT_HIGH, JOINT_LOW
from roboclock.envs.rates import CONTROL_HZ
from roboclock.policies.vision_llm import (
    GRIPPER_TIME,
    SIDES,
    ContentFiltered,
    endpoint_settings,
    request_chat,
    rig_camera,
)
from roboclock.policies.vision_llm_move_eef import pose
from roboclock.policies.vision_llm_primitives.client import assistant_message, parse_arguments, read_reply, tool_body, tool_message
from roboclock.policies.vision_llm_primitives.geometry import arm_capsules, capsules_gap
from roboclock.policies.vision_llm_primitives.policy import too_long
from roboclock.policies.vision_llm_primitives.prompt import HISTORY_TURNS, Conversation, observation_content, observation_views
from roboclock.policies.vision_llm_primitives.settings import OPEN_WIDTH, opening_range
from roboclock.policies.x5_kinematics import X5Kinematics, transform

MAX_QUERIES = 100  # model calls per episode (VISION_LLM_MAX_QUERIES; 0 = no cap), RoboDojo's budget
MAX_DURATION = 10.0  # s one motion may play; a longer one is refused and has to be split (RoboDojo's cap)
# rad/s no joint is commanded faster than. RoboDojo takes each joint's limit from the simulator's
# declaration; a roboclock policy sees only the observation, so this is one value under every rig's
# cap (industry's arms: 0.70 to 0.75 rad/s). VISION_LLM_EEF_JOINT_SPEED overrides it.
JOINT_SPEED = 0.6
IK_WAYPOINTS = 12  # poses solved along the tool point's line
GOAL_SEEDS = 12  # random joint seeds tried, after the arm's own, for a target solved by itself
GOAL_ITERATIONS = 300  # solver iterations for a target solved by itself, when the line cannot be followed
REACH_TOLERANCE = 0.01  # m the kinematics may miss a target's position; beyond it the pose is refused
ANGLE_TOLERANCE = 5.0  # deg it may miss the orientation
HOLD_GAP = 0.02  # m (and ANGLE_TOLERANCE) within which a measured arm stands where it was sent
ARM_CLEARANCE = 0.02  # m gap between the two arms' bodies a motion may not close
CLEARANCE_SAMPLES = 12
CLEARANCE_SLACK = 0.002  # m a motion may close an already small gap by before it is refused
SETTLE_WINDOW = 0.1  # s over which ...
SETTLE_STILL = 0.002  # ... no joint moves more than this (rad) once the arms are at rest
SETTLE_WIDTH = 0.0005  # ... and no jaw opening changes more than this (m)
SETTLE_LIMIT = 2.0  # s a motion waits for the arms to come to rest, at most
MIN_TURN = 0.1  # s of task time a turn without a motion lasts
MIN_WAIT = 0.1  # s the wait tool takes at least (at most the playout cap)

SYSTEM_PROMPT = """You are controlling a robot of two identical 6-DoF ARX X5 arms, prefixed left_ and right_, \
each with a parallel-jaw gripper, in a simulated workspace. You receive RGB camera images (a head \
camera and one wrist camera per arm), the task clock, the current world-frame tool-point state of \
both arms in the same 14 dimensions move_eef takes, arm joint angles as context you cannot \
command, and a task instruction. Move with move_eef by naming only the world-frame dimensions \
you want to change. Cartesian targets must be estimated from RGB; no depth or world-coordinate \
query is available. Respond with exactly one tool call per turn. After each motion the next \
observation reports how far the tool point ended up from what you asked for, so check it before \
assuming a motion landed. Nothing tells you whether an object is held or whether the task \
advanced: judge that from the views and the numbers.

Two budgets run down at once: {budget}, and the task's own time. The task clock runs while the \
arms move and while you think. A motion takes time in proportion to how far the joints travel, \
so a small correction is quick and a long reach is not.{waiting} The episode ends by itself when the task \
is judged; there is no tool for declaring it finished, and holding still never finishes it.

Embodiment notes:
World frame: metres, +z up. The left arm's base is at {left_base}, the right arm's at \
{right_base}, and both arms face along {forward} in the horizontal plane. Each tool point reaches \
about {reach:.2f} m from its own base. The head camera is at {camera}, looking along {looking}. \
"left" and "right" are fixed arm names and need not match image sides: identify each arm by its \
coordinates and its wrist view.
Every position, commanded and reported, is the tool point: the point between the jaws where an \
object is held. To grasp something, name that object's own position: no tool offset is yours to \
add.
Gripper 0 is fully closed, 1 is fully open ({open_width:.3f} m between the jaws), and commanding \
0 closes as far as the object allows and keeps squeezing.
A motion carries the tool point in a straight line from where the arm is to the target, turning \
evenly on the way; where the joints cannot follow that line they go directly to the target's \
pose and the hand swings on an arc. It does not avoid the scene: where something is in the way, go around it in \
several calls. A path that would bring the two arms' bodies within {clearance:g} m of each other \
is refused.
Each wrist camera rides its own arm, so which views you have is itself something you can \
command: when the head camera is blocked, move the idle arm until its wrist looks at the work.
Scale every motion to how close it is to touching something. Full-size moves belong in free \
space; once the jaws are within about a centimetre of the object, the surface, or wherever a \
held object is going down, keep corrections small and re-read the cameras between them.
The advertised per-dimension bounds only reject gross magnitude errors: a pose inside them can \
still be refused as unreachable, and that refusal spends a turn. A refusal is likelier the more \
one call asks for, so travel to the new position first and turn the tool in a later call rather \
than doing both at once; large rotations are more reliable in two or three steps than in one. \
Both arms named in one call are refused as one: a pose that cannot be reached for either arm \
leaves the other standing still as well.

TASK. {instruction}"""

WAITING = (
    " To let task time pass without moving (for a signal, a fill, a countdown), call wait: it holds "
    "both arms still for the seconds you name and costs one model call however long it is, so "
    "do not spend calls on motions that go nowhere."
)


class Refused(Exception):
    """A call that moves nothing; the message goes to the model."""


def vector(values, digits=3):
    return str(np.round(np.asarray(values, dtype=float), digits).tolist())


def tool_definitions(bases, max_duration=MAX_DURATION, wait=True):
    """The tools: `move_eef`, `wait` (not RoboDojo's; left out with wait=False) and `give_up`.
    There is deliberately no tool for declaring the task finished (the episode ends by itself
    when the task is judged)."""
    move = {
        "name": "move_eef",
        "description": (
            "Move to absolute world-frame tool-point targets. Name only the dimensions you intend "
            "to change; every unnamed dimension holds its value, so lowering one hand 2 cm is a "
            "single entry. Positions are metres. Orientation is three angles in degrees measured "
            "from a straight-down grasp, not absolute world angles: at "
            "pitch_deg=roll_deg=yaw_deg=0 the fingers point at world -z and the jaws close along "
            "world +x. pitch_deg turns about world +x (0 is straight down, +90 is horizontal "
            "pointing at world +y), roll_deg about world +y (+90 is horizontal pointing at world "
            "-x), and yaw_deg about world +z, which is the jaw rotation of a straight-down grasp. "
            "Naming dimensions of both arms moves them simultaneously along time-aligned paths. "
            "The motion plays at a fixed safe speed, so how long it takes follows from how far "
            f"it goes; a move needing over {max_duration:g}s is refused and has to be split. A "
            "gripper value named alongside a pose moves the jaw only after the arm has arrived, "
            "so naming both in one call is a safe way to grasp at a pose. Naming only values the "
            "arm already has waits one control step. "
            f"Per-dimension bounds: {pose.bounds_text(bases)}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Map of dimension name to value. Valid names: " + ", ".join(pose.labels()),
                },
                "note": {
                    "type": "string",
                    "description": "What you observe right now and why you chose this motion, in one or two plain sentences.",
                },
            },
            "required": ["targets", "note"],
        },
    }
    give_up = {
        "name": "give_up",
        "description": "End your part of the episode because the task cannot be completed: the arms hold still from here on.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    }
    pause = {
        "name": "wait",
        "description": (
            "Hold both arms and grippers still while task time passes, then observe again. Use it "
            "to wait for a signal or for something in the scene to finish; the next observation "
            "comes after the seconds you name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": f"Task time to let pass, {MIN_WAIT:g} to {max_duration:g}."},
                "note": {"type": "string", "description": "What you are waiting for, in one plain sentence."},
            },
            "required": ["seconds", "note"],
        },
    }
    functions = (move, pause, give_up) if wait else (move, give_up)
    return [{"type": "function", "function": function} for function in functions]


def parse_targets(raw, bases):
    """{arm index: {axis: value}} of a call's `targets`, or Refused with a message for the model."""
    if not isinstance(raw, dict) or not raw:
        raise Refused("move_eef.targets must be a non-empty object of dimension name to value")
    valid = pose.labels()
    by_arm = {}
    for label, value in raw.items():
        label = str(label)
        if label not in valid:
            raise Refused(f"unknown dimension {label!r}; valid names: {', '.join(valid)}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise Refused(f"value for {label!r} must be a finite number, got {value!r}")
        low, high = pose.bounds(label, bases)
        if not low <= value <= high:
            raise Refused(f"target for {label!r} is outside [{low:g}, {high:g}]")
        side, _, axis = label.partition("_")
        by_arm.setdefault(SIDES.index(side), {})[axis] = float(value)
    return by_arm


def merged(asked):
    """Several parallel move_eef calls folded into one arguments object, or None when they are
    not that (another tool, unreadable arguments, a dimension named twice)."""
    targets, notes = {}, []
    for call in asked:
        if call["name"] != "move_eef":
            return None
        try:
            arguments = parse_arguments(call["raw"])
        except ValueError:
            return None
        named = arguments.get("targets")
        if not isinstance(named, dict) or set(named) & set(targets):
            return None
        targets.update(named)
        notes.append(str(arguments.get("note", "")))
    return {"targets": targets, "note": " | ".join(note for note in notes if note)}


def resample(path, steps, limit):
    """`steps` joint waypoints along `path`, ending on its last point, spaced so that no joint
    moves more than `limit` per step. The path's first point is where the arm already is, so
    sampling starts one interval in."""
    travel = np.r_[0.0, np.cumsum(np.abs(np.diff(path, axis=0)).max(axis=1))]
    if travel[-1] <= 0.0:
        return np.repeat(path[-1:], steps, axis=0)
    at = np.linspace(0.0, travel[-1], steps + 1)[1:]
    return np.stack([np.interp(at, travel, path[:, joint]) for joint in range(path.shape[1])], axis=1)


class MoveEefPolicy:
    def __init__(self):
        endpoint_settings(self)
        self.last_output_tokens = None  # of the latest model query, thinking included (--inference-latency ideal)
        self.max_queries = int(os.environ.get("VISION_LLM_MAX_QUERIES", str(MAX_QUERIES)))
        self.history_turns = int(os.environ.get("VISION_LLM_HISTORY", str(HISTORY_TURNS)))
        self.joint_speed = float(os.environ.get("VISION_LLM_EEF_JOINT_SPEED", str(JOINT_SPEED)))
        self.max_duration = float(os.environ.get("VISION_LLM_EEF_MAX_DURATION", str(MAX_DURATION)))
        self.wait_tool = os.environ.get("VISION_LLM_EEF_WAIT", "1") != "0"  # 0: RoboDojo's two tools only
        if not (self.joint_speed > 0 and self.max_duration > 0):
            raise ValueError("VISION_LLM_EEF_JOINT_SPEED and VISION_LLM_EEF_MAX_DURATION must be > 0")
        self.control_hz = float(CONTROL_HZ)
        self.kinematics = X5Kinematics()
        self.finger_min, self.open_width = 0.0, OPEN_WIDTH

    def set_spaces(self, action_space, observation_space):
        """The environment's interface, told by the evaluator before `reset`: the lowest finger
        target (what gripper 0 commands) and how wide the jaws open (what gripper 1 is)."""
        low = np.asarray(action_space.low, dtype=float)
        high = np.asarray(action_space.high, dtype=float)
        spaces = getattr(observation_space, "spaces", None) or {}
        width = float(np.min(spaces["gripper_width"].high)) if "gripper_width" in spaces else None
        self.finger_min = min(0.0, float(low[ARM_DOF]))
        self.open_width = opening_range(2 * float(high[ARM_DOF]), width)

    @property
    def inference_model(self):
        """Served model name; --inference-latency ideal charges queries at its surveyed speed."""
        return self.model

    # ------------------------------------------------------------------ episode
    def reset(self, observation):
        joints = np.asarray(observation["joint_positions"], dtype=float)
        measured = np.asarray(observation["ee_pose"], dtype=float)
        self.arms = [
            self.kinematics.calibrated(
                joints[DOF * i : DOF * i + ARM_DOF], measured[7 * i : 7 * i + 3], measured[7 * i + 3 : 7 * i + 7]
            )
            for i in range(2)
        ]
        self.bases = [arm.base[:3, 3].copy() for arm in self.arms]
        self.command = [joints[DOF * i : DOF * i + ARM_DOF].copy() for i in range(2)]
        self.home = [command.copy() for command in self.command]  # a seed for solving a target
        self.fingers = [float(joints[DOF * i + ARM_DOF]) for i in range(2)]
        self.tools = tool_definitions(self.bases, self.max_duration, self.wait_tool)
        camera = rig_camera(self.arms[0].base, self.arms[1].base)
        forward = self.arms[0].base[:3, 0] + self.arms[1].base[:3, 0]
        self.instruction = str(observation["instruction"])
        budget = f"{self.max_queries} model calls, one per turn" if self.max_queries else "your model calls, one per turn"
        system = SYSTEM_PROMPT.format(
            budget=budget,
            waiting=WAITING if self.wait_tool else "",
            left_base=vector(self.bases[0]),
            right_base=vector(self.bases[1]),
            forward=vector(forward[:2] / max(np.linalg.norm(forward[:2]), 1e-9), 2),
            reach=pose.REACH,
            camera=vector(camera["pose"][:3, 3]),
            looking=vector(camera["pose"][:3, 2], 2),
            open_width=self.open_width,
            clearance=ARM_CLEARANCE,
            instruction=self.instruction,
        )
        self.conversation = Conversation(system, self.history_turns)
        self.queue = []  # the accepted motion's remaining control steps: ([joints, joints], [finger, finger])
        self.trail = None  # measurements since the motion's last step, while the arms come to rest
        self.sent = {}  # arm -> the tool pose the latest motion sent it to, until its arrival is reported
        self.turn = None  # the latest model turn, until the next one files it in the conversation
        self.notes = []  # what the harness has to tell the model on its next turn
        self.requests = 0
        self.previous_turn = None
        self.hold_until = -math.inf
        self.gave_up = False
        self.queried = True

    def action(self):
        """The joint targets to send this control step."""
        out = np.zeros(2 * DOF, dtype=np.float32)
        finger_max = self.open_width / 2
        for i in range(2):
            out[DOF * i : DOF * i + ARM_DOF] = np.clip(self.command[i], JOINT_LOW[:ARM_DOF], JOINT_HIGH[:ARM_DOF])
            out[DOF * i + ARM_DOF : DOF * (i + 1)] = min(max(self.fingers[i], self.finger_min), finger_max)
        return out

    def act(self, observation):
        self.queried = False
        now = float(np.ravel(observation["sim_time"])[0])
        if self.queue:
            self.command, self.fingers = self.queue.pop(0)
            if not self.queue:
                self.trail = []
            return self.action()
        if self.trail is not None:
            if not self.at_rest(observation):
                return self.action()
            self.trail = None
        if self.gave_up or now < self.hold_until or (self.max_queries and self.requests >= self.max_queries):
            return self.action()
        self.queried = True
        if not self.ask(observation):
            self.hold_until = max(self.hold_until, now + MIN_TURN)  # a wait has set its own
        return self.action()

    def at_rest(self, observation):
        """Whether the arms and jaws have stopped moving after a motion's last step (or
        SETTLE_LIMIT has passed). The drives trail a moving command, so the commanded path ends
        before the hand is there; the model's next views and its arrival check wait for it."""
        joints = np.asarray(observation["joint_positions"], dtype=float)
        self.trail.append(np.r_[joints, np.asarray(observation["gripper_width"], dtype=float) * (SETTLE_STILL / SETTLE_WIDTH)])
        window = max(2, round(SETTLE_WINDOW * self.control_hz))
        if len(self.trail) > SETTLE_LIMIT * self.control_hz:
            return True
        return len(self.trail) > window and float(np.max(np.abs(self.trail[-1] - self.trail[-1 - window]))) < SETTLE_STILL

    # ------------------------------------------------------------------ what the model reads
    def measured(self, observation, i):
        joints = np.asarray(observation["joint_positions"], dtype=float)
        return joints[DOF * i : DOF * i + ARM_DOF]

    def gripper(self, observation, i):
        """Arm i's measured jaw opening on the tool's scale: 0 closed, 1 fully open."""
        width = float(np.asarray(observation["gripper_width"], dtype=float)[i])
        return min(max(width / self.open_width, 0.0), 1.0)

    def state_text(self, observation, notes=()):
        """The turn's measurements: the clock, the 14 dimensions in the order and the names
        move_eef takes them, the joints as context, and the previous motion's arrival check."""
        now = float(np.ravel(observation["sim_time"])[0])
        before = "this is your first turn" if self.previous_turn is None else f"your previous turn was at {self.previous_turn:.2f} s"
        lines = [
            f"Time now: {now:.2f} s on the task clock ({before}).",
            "World-frame tool-point state, named as move_eef takes them (metres; degrees from the straight-down reference):",
        ]
        poses = [self.arms[i].tool_pose(self.measured(observation, i)) for i in range(2)]
        for i, side in enumerate(SIDES):
            values = pose.pose_to_values(poses[i])
            lines.extend(f"{side}_{axis}={values[axis]:.4f}" for axis in pose.POSITION_AXES)
            lines.extend(f"{side}_{axis}={values[axis]:.1f}" for axis in pose.ANGLE_AXES)
            lines.append(f"{side}_gripper={self.gripper(observation, i):.2f}")
        lines.append("Arm joint angles, radians, for context; not commandable:")
        for i, side in enumerate(SIDES):
            lines.append(f"{side}: {vector(self.measured(observation, i), 4)}")
        sent, self.sent = self.sent, {}
        if sent:
            parts = [
                f"{SIDES[i]} {np.linalg.norm(poses[i][:3, 3] - target[:3, 3]) * 1000:.0f} mm and "
                f"{pose.angle_between(poses[i][:3, :3], target[:3, :3]):.1f} deg away"
                for i, target in sorted(sent.items())
            ]
            lines.append("Arrival check for the previous move_eef target: " + "; ".join(parts) + ".")
        lines.extend(notes)
        return "\n".join(lines)

    # ------------------------------------------------------------------ model turn
    def ask(self, observation):
        """One model turn. True when it started a motion."""
        if self.turn is not None:
            self.conversation.add(self.turn)
            self.turn = None
        now = float(np.ravel(observation["sim_time"])[0])
        notes, self.notes = self.notes, []
        if str(observation["instruction"]) != self.instruction:
            self.instruction = str(observation["instruction"])
            notes.append(f"The task now reads: {self.instruction}")
        text = self.state_text(observation, notes)
        views = observation_views(observation)
        content = observation_content(text, views)
        messages = self.conversation.messages(content)
        row = dict(request=self.requests, sim_time=now, messages=len(messages))
        self.requests += 1
        self.previous_turn = now
        start = time.perf_counter()
        try:
            while True:
                body = tool_body(self, messages, self.tools)
                body["parallel_tool_calls"] = False  # one call per turn
                try:
                    reply = request_chat(
                        self.url, body, self.timeout, self.cookie,
                        api_key=self.api_key, retries=self.retries, retry_timeouts=self.retry_timeouts,
                    )
                    break
                except RuntimeError as error:
                    # The conversation outgrew the model's context: drop the older half and ask again.
                    if not (too_long(error) and self.conversation.forget()):
                        raise
                    messages = self.conversation.messages(content)
                    row["messages"] = len(messages)
        except ContentFiltered:
            self.last_output_tokens = None
            self.notes.append("Your previous turn got no answer: the model provider's content filter refused the request.")
            self.log(row, round_trip_ms=(time.perf_counter() - start) * 1000.0, parse_error="content_filter", calls=[], answer="", reasoning="")
            return False
        round_trip = time.perf_counter() - start
        usage = reply.get("usage") or {}
        self.last_output_tokens = usage.get("completion_tokens")
        answer, asked, finish_reason = read_reply(reply, self.requests - 1)
        self.turn = {
            "text": text,
            "head": views[0][1] if views and views[0][0] == "head" else None,
            "assistant": assistant_message(answer, asked),
            "tools": [],
        }
        result, moved = self.handle(asked, observation)
        if not asked:
            cut = " (the reply was cut off at the output-token limit)" if finish_reason == "length" else ""
            self.notes.append(f"Your previous reply called no tool{cut}. Reply with exactly one tool call.")
            result = f"no tool call (finish_reason {finish_reason})"
        for call in asked:
            self.turn["tools"].append(tool_message(call, result))
        message = reply["choices"][0].get("message") or {}
        self.log(
            row,
            round_trip_ms=round_trip * 1000.0,
            prompt_tokens=usage.get("prompt_tokens"),
            cached_prompt_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            cost_usd=reply.get("response_cost_usd"),
            finish_reason=finish_reason,
            parse_error=result if not asked or result.startswith("REFUSED") else None,
            calls=[f"{call['name']}({call['raw'][:400]})" for call in asked],
            result=result,
            answer=answer[:2000],
            reasoning=(message.get("reasoning") or message.get("reasoning_content") or "")[:2000],
        )
        return moved

    def handle(self, asked, observation):
        """Act on the turn's tool calls: (the result every call is answered with, whether a
        motion started)."""
        if not asked:
            return "", False
        try:
            if len(asked) > 1:
                arguments = merged(asked)  # several move_eef calls naming different dimensions are one motion
                if arguments is None:
                    raise Refused(f"{len(asked)} tool calls in one turn; reply with exactly one. Nothing moved.")
                name = "move_eef"
            else:
                name = asked[0]["name"]
                try:
                    arguments = parse_arguments(asked[0]["raw"])
                except ValueError as problem:
                    raise Refused(str(problem)) from None
            if name == "give_up":
                self.gave_up = True
                return "Acknowledged give_up: the arms hold still until the episode ends.", False
            if name == "wait" and self.wait_tool:
                return self.wait(arguments.get("seconds"), observation), False
            if name != "move_eef":
                raise Refused(f"unknown tool {name!r}")
            return self.move(parse_targets(arguments.get("targets"), self.bases), observation), True
        except Refused as refusal:
            return f"REFUSED: {refusal}", False

    def wait(self, seconds, observation):
        """Hold still for `seconds` of task time before the next turn; the text the call is answered with."""
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise Refused(f"wait.seconds must be a finite number, got {seconds!r}")
        if not MIN_WAIT <= seconds <= self.max_duration:
            raise Refused(f"wait.seconds is outside [{MIN_WAIT:g}, {self.max_duration:g}]")
        self.hold_until = float(np.ravel(observation["sim_time"])[0]) + float(seconds)
        return f"Accepted: holding still for {float(seconds):.2f}s."

    def log(self, row, **fields):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({**row, **fields}) + "\n")

    # ------------------------------------------------------------------ motion
    def standing(self, observation, i):
        """(joints, tool pose) arm i's next motion starts from.

        An arm that stands where it was sent (within HOLD_GAP: the drives' sag under gravity)
        starts from its command, so that the dimensions a call does not name stay where they
        were sent and do not sink by the sag on every call. One that is further off is held
        off by something: it starts from where it is measured, which lets go of what its
        command was pressing into."""
        measured = self.measured(observation, i)
        commanded, actual = self.arms[i].tool_pose(self.command[i]), self.arms[i].tool_pose(measured)
        near = (
            np.linalg.norm(commanded[:3, 3] - actual[:3, 3]) <= HOLD_GAP
            and pose.angle_between(commanded[:3, :3], actual[:3, :3]) <= ANGLE_TOLERANCE
        )
        return (self.command[i].copy(), commanded) if near else (measured.copy(), actual)

    def plan(self, i, joints, start, target):
        """Joint path of arm i from `joints` (tool pose `start`) to the tool pose `target`, or
        Refused when its kinematics cannot reach the target.

        The tool point goes in a straight line where the joints can follow one. Where they
        cannot (a joint limit lies across it), the target is solved for by itself from several
        seeds and the joints go straight to that solution: the hand then swings on an arc.
        RoboDojo's planner needs only the target to be reachable too, not the line to it."""
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]])))
        path = [joints]
        for phase in np.linspace(0.0, 1.0, IK_WAYPOINTS + 1)[1:]:
            waypoint = transform(slerp(phase).as_matrix(), start[:3, 3] + (target[:3, 3] - start[:3, 3]) * phase)
            path.append(self.arms[i].solve(waypoint, path[-1])[0])
        miss, turn = self.off(i, path[-1], target)
        if miss <= REACH_TOLERANCE and turn <= ANGLE_TOLERANCE:
            return np.stack(path)
        low, high = self.arms[i].low, self.arms[i].high
        random = np.random.default_rng(0).uniform(low, high, (GOAL_SEEDS, len(low)))  # the same seeds every call
        for seed in (path[-1], joints, self.home[i], (low + high) / 2, *random):
            solved = self.arms[i].solve(target, seed, iterations=GOAL_ITERATIONS)[0]
            near = self.off(i, solved, target)
            if near[0] <= REACH_TOLERANCE and near[1] <= ANGLE_TOLERANCE:
                return np.stack([joints, solved])
            miss, turn = min((miss, turn), near, key=lambda off: off[0] + off[1] / 100.0)
        raise Refused(
            f"{SIDES[i]} arm pose unreachable (the nearest pose its joints reach is {miss * 1000:.0f} mm and "
            f"{turn:.0f} deg from it). The robot did not move; choose another pose. Neither arm moved."
        )

    def off(self, i, joints, target):
        """(m, deg) arm i's tool pose at `joints` is from `target`."""
        reached = self.arms[i].tool_pose(joints)
        return float(np.linalg.norm(reached[:3, 3] - target[:3, 3])), pose.angle_between(reached[:3, :3], target[:3, :3])

    def move(self, by_arm, observation):
        """Queue the motion a checked `targets` asks for; the text the call is answered with."""
        starts = [self.standing(observation, i) for i in range(2)]
        paths, sent = {}, {}
        for i in sorted(by_arm):
            axes = {axis: value for axis, value in by_arm[i].items() if axis != "gripper"}
            if axes:
                target = pose.values_to_pose({**pose.pose_to_values(starts[i][1]), **axes})
                paths[i], sent[i] = self.plan(i, starts[i][0], starts[i][1], target), target
        self.keep_clear(paths, starts)
        finger_max = self.open_width / 2
        # Gripper 0 commands the lowest finger target the rig takes: it closes as far as the
        # object allows and the standing position error is the grip.
        fingers = {
            i: self.finger_min if named["gripper"] <= 0.0 else named["gripper"] * finger_max
            for i, named in by_arm.items()
            if "gripper" in named
        }
        limit = self.joint_speed / self.control_hz
        arm_steps = max(
            (math.ceil(np.abs(np.diff(path, axis=0)).max(axis=1).sum() / limit - 1e-9) for path in paths.values()), default=0
        )
        # The jaw moves only once the arm has arrived: closing it while the hand still travels
        # sweeps the object out of the way before the grasp.
        gripper_steps = max(
            (math.ceil(abs(goal - self.fingers[i]) / finger_max * GRIPPER_TIME * self.control_hz - 1e-9) for i, goal in fingers.items()),
            default=0,
        )
        steps = arm_steps + gripper_steps
        if steps > self.max_duration * self.control_hz:
            raise Refused(
                f"this motion needs {steps / self.control_hz:.1f}s at the safe speed, over the "
                f"{self.max_duration:g}s playout cap; split it into smaller motions. Nothing moved."
            )
        if steps == 0:
            arm_steps = steps = 1  # every named dimension already has its value: one step passes
        sampled = {i: resample(path, arm_steps, limit) for i, path in paths.items()}
        before = list(self.fingers)
        self.queue = []
        for index in range(steps):
            command = [sampled[i][min(index, arm_steps - 1)] if i in sampled else self.command[i].copy() for i in range(2)]
            jaws = list(before)
            for i, goal in fingers.items():
                if index >= arm_steps:
                    jaws[i] = before[i] + (goal - before[i]) * (index - arm_steps + 1) / gripper_steps
            self.queue.append((command, jaws))
        self.sent = sent
        text = f"Accepted: playing {steps} control steps ({steps / self.control_hz:.1f}s)."
        if gripper_steps:
            text += f" The last {gripper_steps} move the gripper, once the arm has arrived."
        if sent:
            text += " The next observation reports how close the tool point landed."
        return text

    def keep_clear(self, paths, starts):
        """Refused when the motion would close the gap between the two arms' bodies (capsules
        around the links and the hand, without anything held) to under ARM_CLEARANCE. Both paths
        are time-aligned; an arm without one stands still. A motion that does not bring the
        arms nearer than they already are always passes."""
        if not paths:
            return

        def at(i, phase):
            if i not in paths:
                return starts[i][0]
            position = phase * (len(paths[i]) - 1)
            index = min(int(position), len(paths[i]) - 2)
            return paths[i][index] + (paths[i][index + 1] - paths[i][index]) * (position - index)

        def gap(phase):
            return capsules_gap(arm_capsules(self.arms[0], at(0, phase)), arm_capsules(self.arms[1], at(1, phase)))

        first = gap(0.0)
        for phase in np.linspace(0.0, 1.0, CLEARANCE_SAMPLES + 1)[1:]:
            now = gap(phase)
            if now < ARM_CLEARANCE and now < first - CLEARANCE_SLACK:
                raise Refused(
                    f"the path would bring the two arms' bodies to {now:.3f} m from each other "
                    f"(at least {ARM_CLEARANCE:g} m is kept). Neither arm moved."
                )


def make():
    return MoveEefPolicy()
