"""What the model reads besides the tools: the system prompt, each turn's observation, and the
conversation that carries earlier turns.

The system prompt states the rules that hold for every tool. What each tool does is in its own
definition (tools.py), not here. The task's instruction closes the system prompt, so it stays
in view however much of the conversation is trimmed.

The conversation is the standard tool-calling one: the user message of a turn is the
observation (views and measurements), the assistant message is the model's tool calls, and
each call is answered by a tool message with what it measured. Past turns are kept so the
model can see what it did and what changed; to bound the request, only the latest
`keep` turns are sent, and views older than the previous turn's head view are dropped.
"""

import numpy as np

from roboclock.policies.vision_llm import SIDES, VIEWS, jpeg_data_url

HISTORY_TURNS = 6  # past turns sent with a request, at most (VISION_LLM_HISTORY)

SYSTEM_PROMPT = """You control two identical 6-DoF ARX X5 arms, "left" and "right", each with a \
parallel-jaw gripper, in a simulated workspace. You act only by calling the tools: each tool is \
one manipulation primitive, and its description says what it does and what it reports.

EACH TURN you get a head camera view and one wrist camera view per arm, the task clock at the \
moment the views were taken, and each arm's measured tool point and gripper opening. What your \
tool calls measured comes back in their results. Earlier turns stay in this conversation, with \
the previous turn's head view, so you can compare what changed.

Everything you are told is a measurement. Nothing tells you whether an object is held, has \
moved or is in place, or whether the task advanced: judge that yourself from the views and the \
numbers.

FRAMES. World coordinates are metres with +z up; both arm bases are mounted at z = {mount:.2f}. \
"left" and "right" are fixed arm names and need not match image sides: identify each arm by its \
coordinates and its wrist view.

TARGETS. A target is a head-image pixel {{"u": u, "v": v}}, with u from 0 (left edge) to 1000 \
(right edge) and v from 0 (top) to 1000 (bottom), meaning the surface point seen there as \
measured by the head depth camera when you answer; or a world point {{"xyz": [x, y, z]}} in \
metres. A result states the world point a pixel landed on.

APPROACH. "top": the fingers point straight down and the jaws close horizontally. "front": the \
fingers point horizontally, away from that arm's base toward the target. "keep": the hand keeps \
its current orientation. These are two points of one family, and "tilt" reaches the rest: \
"pitch" is how far below level the fingers point (front is 0, top is 1.57), "heading" the \
direction in the table plane they point toward, "roll" how far the jaws are turned about the \
fingers from closing level (1.57 on a level hand makes them close vertically, around a flat \
tab or a plate's edge).

WHEN. A tool that moves an arm can carry "when": {{"u": u, "v": v, "timeout": seconds}}. Both \
arms then hold until the mean colour of the patch at that head-image pixel has changed from the \
view you were shown, and the call runs at once: a reaction that does not wait for your next \
turn. If nothing changes within the timeout the call fails.

CALLS. Call 1 to {calls} tools per turn. They run in the order you give them, one arm at a \
time, while the other arm holds still; the "together" tool makes the two calls after it run at \
the same time, one per arm. When a call fails or is refused, the calls after it are not run. \
You see new views only on your next turn.

MOTION. Paths are straight lines: the arms do not avoid each other or the scene by themselves. \
A motion whose path would bring the moving arm closer to the other arm than the arm_clearance \
setting fails before it starts (each arm is modelled as capsules around its links and hand, \
without anything it holds). The arm's joints have a speed limit, and turning the hand (a change \
of approach above all) takes several seconds at it. A motion asked faster than the arm can \
follow is slowed to what it follows and stays on its line: every result states the time the \
call took, so plan with the times you have seen, not with the speed settings. A motion ends \
when the measured hand has arrived where it was sent, or when something holds it off (the \
result then says how far short it stopped).

GRIPPER. The jaws open to at most {open_width:.3f} m. The fingers are position controlled: a \
close runs them in until they stop (on something or on each other), then commands each finger \
`squeeze` metres past where it stopped. A finger cannot be commanded below {finger_min:.3f} m \
(0 is the closed stop), so on a thin object less squeeze than you asked for may be applied; the \
result states what was applied.

SETTINGS. Speeds, stand-offs, sensitivities and tolerances are settings with defaults. The \
configure tool changes them for the rest of the episode.

TIME. The task clock runs while the arms move and while a call waits, and it may also have \
advanced between your turns: compare the clock values you are given. The episode ends by itself \
when the task is judged; holding still never finishes it.

TASK. {instruction}"""


def system_prompt(instruction, mount, open_width, finger_min, calls):
    return SYSTEM_PROMPT.format(
        instruction=instruction, mount=mount, open_width=open_width, finger_min=finger_min, calls=calls
    )


def observation_text(observation, fingers, previous_turn, notes=()):
    """The turn's measurements as text. It has the same form in every domain and interprets
    nothing: no "holding", no task state."""
    pose = np.asarray(observation["ee_pose"], dtype=float)
    width = np.asarray(observation["gripper_width"], dtype=float)
    now = float(np.ravel(observation["sim_time"])[0])
    before = "this is your first turn" if previous_turn is None else f"your previous turn was at {previous_turn:.2f} s"
    lines = [f"Time now: {now:.2f} s on the task clock ({before})."]
    for i, side in enumerate(SIDES):
        lines.append(
            f"{side} arm: measured tool point {np.round(pose[7 * i : 7 * i + 3], 3).tolist()}, "
            f"measured gripper opening {width[i]:.3f} m (commanded {2 * fingers[i]:.3f} m)"
        )
    if "depth" not in observation:
        lines.append("No head depth image: pixel targets fail, use world points.")
    lines.extend(notes)
    return "\n".join(lines)


def observation_views(observation):
    """[(camera name, JPEG data URL)] of the views in the observation, head first."""
    return [(name, jpeg_data_url(observation[key], size)) for name, key, size in VIEWS if key in observation]


def observation_content(text, views):
    """The user message of the current turn: every view, then the measurements."""
    content = []
    for name, url in views:
        content.append({"type": "text", "text": f"{name} camera:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": text})
    return content


class Conversation:
    """The episode's messages: the system prompt and the finished turns still in view."""

    def __init__(self, system, keep=HISTORY_TURNS):
        self.system = system
        self.keep = max(0, int(keep))
        self.turns = []  # {"text", "head", "assistant", "tools"}, oldest first

    def add(self, turn):
        """Record a finished turn. Past `keep` turns the oldest half goes at once, not one per
        turn: the messages before the latest turns then stay byte-identical from request to
        request, which is what a provider's prompt cache needs."""
        self.turns.append(turn)
        if len(self.turns) > self.keep:
            del self.turns[: len(self.turns) - self.keep // 2]

    def forget(self):
        """Drop the older half of the past turns (at least one); False when none is left."""
        if not self.turns:
            return False
        del self.turns[: (len(self.turns) + 1) // 2]
        return True

    def messages(self, content):
        """The request's messages: system, the turns in view, then the current observation."""
        out = [{"role": "system", "content": self.system}]
        for index, turn in enumerate(self.turns):
            if index == len(self.turns) - 1 and turn["head"]:
                past = [
                    {"type": "text", "text": "head camera, as it was at that turn:"},
                    {"type": "image_url", "image_url": {"url": turn["head"]}},
                    {"type": "text", "text": turn["text"]},
                ]
            else:
                past = turn["text"] + "\n(The views of that turn are no longer shown.)"
            out.append({"role": "user", "content": past})
            out.append(turn["assistant"])
            out.extend(turn["tools"])
        out.append({"role": "user", "content": content})
        return out
