"""Every quantity the primitives use: its range, its default and what it means.

The model decides all of them. They come in two kinds:

- CALL quantities belong to one call: how deep a grasp goes, how far it lifts, how hard the
  fingers squeeze, how long a press holds. They are arguments of the tools.
- SETTINGS hold for the rest of the episode: speeds, stand-offs, sensitivities, tolerances. The
  `configure` tool changes them; until then they are at their defaults.

Keeping the settings out of the motion tools keeps each tool's signature to what a call is
about. A value outside its range is refused with a message naming the range; nothing is
clamped silently.
"""

import math
from dataclasses import dataclass

from roboclock.envs.lab.robot import ARM_DOF, JOINT_HIGH
from roboclock.policies.vision_llm import GRIPPER_TIME, TOOL_SPEED, TURN_SPEED

# m per finger, fully open. JOINT_HIGH is float32 (0.04399999976...): rounded, so that the range's
# upper end is the 0.088 m the model is told and a fully open request is not refused.
FINGER_MAX = round(float(JOINT_HIGH[ARM_DOF]), 6)
FINGER_CLOSED = 0.0  # m per finger: the closed stop
OPEN_WIDTH = 2 * FINGER_MAX
MAX_WAIT = 60.0  # s one wait, watch or `when` may hold


def opening_range(commandable, observed=None):
    """The widest the jaws open on a rig: what the action space can command (twice a finger's
    upper target), capped by the `gripper_width` observation's bound when the rig gives one.
    Rounded to the micrometre: the bounds are float32, and 0.0879999995 must read as 0.088."""
    width = commandable if observed is None else min(commandable, observed)
    return round(width, 6)


@dataclass(frozen=True)
class Quantity:
    low: float
    high: float
    default: float | None  # None: the tool's description states it (it differs per tool, or there is none)
    unit: str
    meaning: str

    def span(self):
        return f"{self.low:g} to {self.high:g}" if math.isfinite(self.high) else f"at least {self.low:g}"



CALL = {
    "height": Quantity(-0.1, 0.5, None, "m", "The target is raised by this."),
    "depth": Quantity(-0.05, 0.15, None, "m", "How far past the target point the approach goes."),
    "lift": Quantity(0.0, 0.4, 0.08, "m", "How far the hand rises after closing (0 = stay)."),
    "opening": Quantity(0.002, OPEN_WIDTH, None, "m", "Distance between the jaws the gripper opens to."),
    "squeeze": Quantity(
        0.0005, 0.03, 0.01, "m",
        "How far past where it stopped each finger is commanded after closing: the grip force, "
        "roughly 1 N per millimetre per finger up to the drive's limit of 10 to 20 N.",
    ),
    "hold": Quantity(0.0, 10.0, 0.3, "s", "How long the press stays down."),
    "yaw": Quantity(
        -2 * math.pi, 2 * math.pi, None, "rad",
        "Top approach only: the direction the jaws close in the table plane (0 = along world +x, "
        "1.57 = along +y). By default they close across the arm's reach direction.",
    ),
    "angle": Quantity(
        -math.pi, math.pi, None, "rad",
        "How far the hand turns. Without `axis` or `pivot` it turns about the fingers' axis, "
        "positive counterclockwise as seen from behind the hand, looking where the fingers "
        "point. With them it turns about that world axis, positive counterclockwise as seen "
        "looking back along the axis from its tip (right-hand rule).",
    ),
    "speed": Quantity(0.01, 1.0, None, "m/s", "Tool speed of this call's unguarded motion, at most, instead of the setting."),
    "seconds": Quantity(0.0, MAX_WAIT, None, "s", "How long to wait."),
    "until_t": Quantity(0.0, math.inf, None, "s", f"Wait until this time on the task clock (at most {MAX_WAIT:g} s per call)."),
    "timeout": Quantity(0.1, MAX_WAIT, None, "s", "Longest the watch holds, instead of the watch_timeout setting."),
}

# The fields of a call's `tilt` argument: the hand's orientation beyond the two presets. A tool
# takes them as one argument so that its signature stays short.
TILT = {
    "pitch": Quantity(
        -0.8, math.pi / 2, None, "rad",
        "How far below level the fingers point: 0 is level (approach front), 1.57 straight down "
        "(approach top); between them the hand is tilted, e.g. 0.35 to come in at 20 degrees "
        "under something.",
    ),
    "roll": Quantity(
        -math.pi, math.pi, None, "rad",
        "Turns the jaws about the fingers' axis from where the approach puts them (0: they close "
        "level, across the direction the fingers point). 1.57 makes a level hand's jaws close "
        "vertically.",
    ),
    "heading": Quantity(
        -2 * math.pi, 2 * math.pi, None, "rad",
        "The direction in the table plane the fingers point toward (0 = world +x, 1.57 = +y). By "
        "default they point from the arm's base toward the target.",
    ),
}

SETTINGS = {
    "speed": Quantity(0.01, 1.0, TOOL_SPEED, "m/s", "Tool speed of unguarded motion, at most: the joints' speed limit can make it slower."),
    "approach_speed": Quantity(
        0.005, 0.5, 0.10, "m/s", "Tool speed of guarded approaches, lowering, drags, pushes and press strokes."
    ),
    "turn_speed": Quantity(0.1, 4.0, TURN_SPEED, "rad/s", "How fast the hand reorients, at most: the joints' speed limit can make it slower."),
    "gripper_time": Quantity(0.1, 5.0, GRIPPER_TIME, "s", "Time of a full open or close stroke."),
    "standoff": Quantity(
        0.01, 0.3, 0.10, "m",
        "Where a grasp, pull or turn starts its approach, back from the target along the "
        "approach; for a push, how far above its start point it comes down from.",
    ),
    "press_standoff": Quantity(0.01, 0.3, 0.05, "m", "Where a press starts its stroke, back from the target."),
    "contact": Quantity(
        0.002, 0.1, 0.012, "m",
        "A guarded motion stops as contact once the measured hand trails its command by this "
        "much and has nearly stopped advancing (smaller = more sensitive).",
    ),
    "tolerance": Quantity(
        0.001, 0.2, 0.02, "m",
        "How far the arm's kinematics may miss a pose; beyond it the call fails before the arm moves.",
    ),
    "arrival": Quantity(
        0.001, 0.05, 0.005, "m",
        "A motion ends once the measured tool point is within this of where it was sent, or the "
        "arm has stopped getting nearer.",
    ),
    "arm_clearance": Quantity(
        -0.05, 0.2, 0.02, "m",
        "Smallest gap a motion may leave between the two arms' bodies; a motion that would close "
        "it further fails before it starts.",
    ),
    "settle": Quantity(0.05, 3.0, 0.3, "s", "Pause after closing before the opening is read."),
    "min_opening": Quantity(
        0.0001, OPEN_WIDTH, 0.003, "m",
        "A grasp, pull or turn whose opening after closing is under this fails: the fingers met.",
    ),
    "clearance": Quantity(0.01, 0.5, 0.12, "m", "How far above its release point a place carries the hand."),
    "retreat": Quantity(0.005, 0.3, 0.08, "m", "How far the hand backs off after a release or a push."),
    "push_offset": Quantity(0.005, 0.2, 0.04, "m", "How far behind the point a push starts."),
    "rim_radius": Quantity(
        5.0, 150.0, 80.0, "image units",
        "place into: how far around the pixel the opening's rim is looked for. It must reach the "
        "rim on every side of the pixel, so use at least the opening's whole width in the image; "
        "a larger value does no harm because the nearest rim in each direction is used.",
    ),
    "watch_timeout": Quantity(0.1, MAX_WAIT, 5.0, "s", "Longest a watch or a `when` holds."),
    "watch_change": Quantity(1.0, 441.0, 40.0, "RGB distance", "Colour change that fires a watch."),
    "watch_radius": Quantity(
        1.0, 50.0, 5.0, "image units",
        "Half-size of the patch around a watched pixel whose mean colour is compared.",
    ),
}
# A decoder that fills fields it does not mean writes 0 for them. Where 0 is outside the range it
# cannot be a value, so it is read as "not given".
ZERO_IS_UNSET = frozenset(
    key for table in (CALL, SETTINGS) for key, quantity in table.items() if quantity.low > 0
)


def in_range(what, key, value, table):
    """`value` as a finite number inside the quantity's range, or ValueError with a message for the model."""
    quantity = table[key]
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}.{key} must be a number") from None
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{what}.{key} must be a finite number")
    if not quantity.low <= number <= quantity.high:
        raise ValueError(f"{what}.{key} = {number:g} is outside its range ({quantity.span()} {quantity.unit})")
    return number


def defaults():
    """The settings an episode starts with."""
    return {key: quantity.default for key, quantity in SETTINGS.items()}
