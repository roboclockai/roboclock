"""The agent's tools: their definitions for the chat API, and the check of each call.

A tool is one manipulation primitive. The definitions are OpenAI function tools: a name, a
description of what the primitive does and returns, and a JSON schema of its arguments in which
every argument has a description and every number its range. The model calls them through the
API's native tool calling; nothing about the tools is described anywhere else.

`check_call` is the single gate between what a model wrote and what the arm does. It returns a
clean copy of the arguments or raises ValueError with a message for the model.
"""

import json
import math
from dataclasses import replace

from roboclock.policies.vision_llm import SIDES
from roboclock.policies.vision_llm_primitives.perception import PIXEL_MAX
from roboclock.policies.vision_llm_primitives.settings import (
    CALL,
    OPEN_WIDTH,
    SETTINGS,
    TILT,
    ZERO_IS_UNSET,
    in_range,
)

PLACE_MODES = ("on", "into", "drop")
APPROACHES = ("top", "front", "keep")
GRIPPER = ("open", "close", "keep")
REFS = ("to", "object", "target")  # arguments that name a point

# What each tool takes. Anything else is refused. `tilt` (the hand's pitch, roll and heading) and
# `hinge` (the line a turn swings about) are objects, so that a signature stays short.
SPEC = {
    "move": (("arm", "to"), ("height", "approach", "yaw", "tilt", "gripper", "opening", "speed", "when")),
    "move_by": (("arm", "by"), ("guard", "gripper", "opening", "speed", "when")),
    "home": (("arm",), ("gripper", "opening", "speed", "when")),
    "grasp": (("arm", "object"), ("approach", "yaw", "tilt", "depth", "opening", "squeeze", "lift", "when")),
    "place": (("arm", "target"), ("mode", "height", "yaw", "opening", "when")),
    "press": (("arm",), ("target", "approach", "yaw", "tilt", "depth", "hold", "when")),
    "push": (("arm", "by"), ("target", "approach", "yaw", "tilt", "when")),
    "pull": (("arm", "by"), ("target", "approach", "yaw", "tilt", "depth", "opening", "squeeze", "when")),
    "turn": (("arm", "angle"), ("target", "hinge", "approach", "yaw", "tilt", "depth", "opening", "squeeze", "when")),
    "together": ((), ()),
    "wait": ((), ("seconds", "until_t", "watch", "timeout")),
    "configure": ((), tuple(SETTINGS)),
}
TOOL_NAMES = tuple(SPEC)
ARM_TOOLS = tuple(name for name, (required, _) in SPEC.items() if "arm" in required)

DESCRIPTIONS = {
    "move": (
        "Move the point between an arm's jaws in a straight line to a target, with no intended "
        "contact. It fails without moving if the target is out of reach by more than the "
        "tolerance setting. Returns the measured tool point."
    ),
    "move_by": (
        "Move an arm's tool point in a straight line by an offset from where it is measured to "
        "be, keeping the hand's orientation. With `guard` it moves at the approach_speed "
        "setting and stops when it meets something, staying pressed against it (two hands "
        "closing in on a thing too wide for one gripper, each with guard and `together`, hold "
        "it between them; moved together by the same offset they carry it). A hand that is "
        "pressed against something stays pressed unless the offset points away from it. The "
        "offset is the measured hand's: what a free arm settles short of it is sent again once. "
        "Returns the measured tool point, and a contact stop if there was one."
    ),
    "home": "Return an arm to the pose it started the episode in. Returns the measured tool point.",
    "grasp": (
        "Close an arm's gripper on a target point and lift. The hand opens to `opening`, goes to "
        "the standoff setting before the point along the approach, advances under guard to "
        "`depth` past it (into the object: the jaws must close around it), runs the fingers in "
        "until they stop, waits the settle setting, squeezes, then rises by `lift`. Returns a "
        "contact stop if there was one, the opening after closing, the squeeze applied, and "
        "after the lift the measured tool point and the opening again. It fails if the opening "
        "after closing is under the min_opening setting (the fingers met)."
    ),
    "place": (
        "Carry what the hand holds to a target and release it. The hand rises to the clearance "
        "setting above the release point (the target raised by `height`), moves over it, "
        "lowers, opens to `opening` and rises by the retreat setting. Modes on and into lower under guard "
        "until contact or the release point; drop lowers to the release point without a guard. "
        "A pixel that looks into an opening (a well, a slot, a cell) lands on its inner wall or "
        "floor, beyond the place it points at: for mode into, when a level rim stands above that "
        "point all around the pixel (within the rim_radius setting), the release moves sideways "
        "to where the pixel's line of sight crosses the rim's plane. A world-point target is "
        "used as given. With `yaw` the hand first turns about the vertical so that its jaws "
        "close along that direction, and what it holds turns with it. Returns where the "
        "lowering ended, how far above the release point a contact stopped it, and the opening "
        "just before and after the release."
    ),
    "press": (
        "Press with closed fingers (a button, a key, a plunger). The gripper closes, comes to "
        "the press_standoff setting before the point, pushes at the approach_speed setting to "
        "`depth` past it along the approach, holds, and backs off to where the stroke started. "
        "Without `target` it presses from where the hand is, straight along its fingers by "
        "press_standoff plus depth. Returns how far short of the commanded depth the measured "
        "fingertips ended."
    ),
    "push": (
        "Slide something with closed fingers. With `target`: the closed fingers come down under "
        "guard, from the standoff setting above, to the push_offset setting behind the point "
        "(opposite to `by`), slide by `by` plus that offset at the approach_speed setting, and "
        "rise by the retreat setting. Without `target`: the hand moves straight by `by` from "
        "where it is and the gripper stays as it is. Returns where the measured fingers ended."
    ),
    "pull": (
        "Drag something by gripping it (a slider, a clamp, a drawer, a lever). With `target`: "
        "the gripper closes on the point as grasp does, drags the hand by `by` at the "
        "approach_speed setting, opens and backs off by the retreat setting along the fingers. "
        "Without `target`: the hand moves straight by `by` from where it is and the gripper "
        "stays as it is (close on the thing first with grasp and lift 0). Returns where the "
        "measured hand ended."
    ),
    "turn": (
        "Turn something by gripping it (a knob, a valve handle, a switch). With `target`: the "
        "gripper closes on the point as grasp does, turns the hand about the fingers' axis by "
        "`angle`, opens and backs off by the retreat setting. Without `target`: the hand turns "
        "by `angle` where it is and the gripper stays as it is. With `hinge` the hand does not "
        "spin about its fingers: it swings on an arc about the hinge's line, turning as it goes, "
        "which is how a lever, a crank or a door moves about its pivot; a hinge given by its "
        "axis alone passes through the tool point, which tilts what the hand holds (to pour, to "
        "flip). Returns how far the measured hand turned or swung."
    ),
    "together": (
        "Run the next two calls of this turn at the same time instead of one after the other. "
        "They must be arm tools and use one arm each (a two-handed catch, lift or carry; two "
        "hands closing in on something; each arm doing its own job). Both start at once; each "
        "ends when its own motion ends. Returns nothing itself: each of the two calls returns "
        "its own result. The arm_clearance check is made against where the other arm is when "
        "they start."
    ),
    "wait": (
        "Hold both arms: for `seconds`, or until `until_t` on the task clock, or until the "
        "colour at a watched head-image pixel changes from the view you were shown (or the "
        "timeout passes). You see nothing new while it holds. Give exactly one of seconds, "
        "until_t or watch."
    ),
    "configure": (
        "Change settings for the rest of the episode: speeds, stand-offs, sensitivities and "
        "tolerances that the other tools use. Give only the ones to change. It takes no time and "
        "returns every setting now in force."
    ),
}

# Defaults that depend on the tool, appended to the argument's description.
TOOL_DEFAULTS = {
    ("move", "height"): "Default 0.",
    ("place", "height"): "Default 0 for on, 0.05 for into, 0.10 for drop.",
    ("grasp", "depth"): "Default 0.02.",
    ("press", "depth"): "Default 0.015.",
    ("pull", "depth"): "Default 0.01.",
    ("turn", "depth"): "Default 0.01.",
    ("grasp", "opening"): "Default: fully open.",
    ("pull", "opening"): "Default: fully open.",
    ("turn", "opening"): "Default: fully open.",
    ("place", "opening"): "Here: what the release opens to (less than fully open beside a wall or a neighbour). Default: fully open.",
    ("move", "opening"): "Applied while moving; overrides `gripper`.",
    ("move_by", "opening"): "Applied while moving; overrides `gripper`.",
    ("home", "opening"): "Applied while moving; overrides `gripper`.",
    ("move", "approach"): "Default keep.",
    ("grasp", "approach"): "Default top.",
    ("press", "approach"): "Default top.",
    ("push", "approach"): "Default top.",
    ("pull", "approach"): "Default top.",
    ("turn", "approach"): "Default front.",
    ("place", "yaw"): "Here: the direction the jaws close along when the hand lets go. Default: as it carries.",
}
REF_MEANING = {
    ("move", "to"): "Where the tool point goes.",
    ("grasp", "object"): "The point to close on: the centre of what the jaws should close around.",
    ("place", "target"): "Where to release.",
    ("press", "target"): "The point to press.",
    ("push", "target"): "The point to push.",
    ("pull", "target"): "The point to grip and drag.",
    ("turn", "target"): "The point to grip and turn.",
}


def number_schema(quantity, note="", bounds=True):
    """A numeric argument. Its range goes into `minimum`/`maximum`, or into the description for a
    server whose guided decoding does not take those keywords (`bounds=False`)."""
    text = [quantity.meaning, f"In {quantity.unit}."]
    if not bounds:
        text.append(f"Range {quantity.span()}.")
    if quantity.default is not None:
        text.append(f"Default {quantity.default:g}.")
    schema = {"type": "number", "description": " ".join(filter(None, [*text, note]))}
    if bounds:
        schema["minimum"] = quantity.low
        if math.isfinite(quantity.high):
            schema["maximum"] = quantity.high
    return schema


def pixel_properties(bounds=True):
    limits = {"minimum": 0, "maximum": PIXEL_MAX} if bounds else {}
    return {"u": {"type": "number", **limits}, "v": {"type": "number", **limits}}


def ref_schema(meaning, segmenter, bounds=True):
    properties = pixel_properties(bounds)
    properties["xyz"] = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    if segmenter:
        properties["name"] = {
            "type": "string", "maxLength": 60,
            "description": "With a pixel: a short noun phrase naming the object there. The system then "
            "segments it and grasps it at its centre, jaws across its narrow side, and reports its size.",
        }
    return {
        "type": "object",
        "description": f"{meaning} A pixel {{u, v}} or a world point {{xyz}} (see TARGETS).",
        "properties": properties,
    }


def property_schema(tool, key, segmenter=False, bounds=True, open_width=OPEN_WIDTH):
    """The JSON schema of one argument of one tool, on a rig whose jaws open `open_width`."""
    note = TOOL_DEFAULTS.get((tool, key), "")
    if tool == "configure":
        return number_schema(rig_quantity(SETTINGS, key, open_width), bounds=bounds)
    if key in CALL:
        return number_schema(rig_quantity(CALL, key, open_width), note, bounds)
    if key == "arm":
        return {"type": "string", "enum": list(SIDES), "description": "The arm that acts."}
    if key in REFS:
        named = segmenter and key in ("object", "target") and tool in ("grasp", "pull", "turn")
        return ref_schema(REF_MEANING[(tool, key)], named, bounds)
    if key == "by":
        return {
            "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
            "description": "Offset [dx, dy, dz] in metres, in world axes.",
        }
    if key == "hinge":
        pivot = ref_schema("A point on the hinge's line (the lever's or door's pivot).", False, bounds)
        pivot["description"] += " Default: the tool point."
        return {
            "type": "object",
            "description": "The world line the hand swings about instead of spinning about its fingers.",
            "properties": {
                "pivot": pivot,
                "axis": {
                    "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                    "description": "The line's direction [x, y, z] in world axes. Default [0, 0, 1], a vertical hinge.",
                },
            },
        }
    if key == "tilt":
        return {
            "type": "object",
            "description": "The hand's orientation beyond approach top and front (see APPROACH). Give only "
            "the angles to change; it needs approach top or front.",
            "properties": {name: number_schema(quantity, bounds=bounds) for name, quantity in TILT.items()},
        }
    if key == "guard":
        return {"type": "boolean", "description": "Stop on contact and stay pressed against it. Default false."}
    if key == "approach":
        return {"type": "string", "enum": list(APPROACHES), "description": f"Hand orientation (see APPROACH). {note}"}
    if key == "gripper":
        return {
            "type": "string", "enum": list(GRIPPER),
            "description": "Applied while moving; close runs the fingers to the closed stop without a squeeze. Default keep.",
        }
    if key == "mode":
        return {"type": "string", "enum": list(PLACE_MODES), "description": "How the release is done. Default on."}
    if key == "watch":
        return {"type": "object", "description": "The head-image pixel to watch.", "properties": pixel_properties(bounds), "required": ["u", "v"]}
    if key == "when":
        timeout = {"type": "number", "description": "In s; default the watch_timeout setting."}
        if bounds:
            timeout.update(minimum=CALL["timeout"].low, maximum=CALL["timeout"].high)
        return {
            "type": "object",
            "description": "Run this call only once the colour at this head-image pixel changes (see WHEN).",
            "properties": {**pixel_properties(bounds), "timeout": timeout},
            "required": ["u", "v"],
        }
    raise KeyError((tool, key))


def rig_quantity(table, key, open_width):
    """table[key], with an opening's upper end set to how wide this rig's jaws open."""
    quantity = table[key]
    if key in ("opening", "min_opening"):
        quantity = replace(quantity, high=open_width)
    return quantity


def tool_definitions(segmenter=False, bounds=True, open_width=OPEN_WIDTH):
    """The tools as the chat API takes them (`tools=`), for a rig whose jaws open `open_width`.
    `bounds=False` writes the numeric ranges into the descriptions instead of `minimum`/`maximum`."""
    tools = []
    for name, (required, optional) in SPEC.items():
        parameters = {
            "type": "object",
            "properties": {
                key: property_schema(name, key, segmenter, bounds, open_width) for key in (*required, *optional)
            },
            "required": list(required),
        }
        tools.append({"type": "function", "function": {"name": name, "description": DESCRIPTIONS[name], "parameters": parameters}})
    return tools


# ------------------------------------------------------------------ checking a call
def finite(value):
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{value!r} is not a finite number")
    return number


def placeholder(value):
    """Whether `value` is what a decoder that fills every schema field writes for a field it does not
    mean: null, false, 0, "", or a list or object made only of those ([0, 0, 0], {}, {"u": 0, "v": 0})."""
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 0
    if isinstance(value, list):
        return all(placeholder(item) for item in value)
    if isinstance(value, dict):
        return all(placeholder(item) for item in value.values())
    return False


def tidy_ref(ref):
    """A point without placeholder parts. Given both a pixel and a world point, a zero one is
    dropped; given two real ones, the pixel wins (it is the main form). An empty name is dropped."""
    if not isinstance(ref, dict):
        return ref
    ref = dict(ref)
    if "name" in ref and placeholder(ref["name"]):
        del ref["name"]
    if "xyz" in ref and {"u", "v"} <= set(ref):
        pixel_zero = placeholder(ref["u"]) and placeholder(ref["v"])
        if placeholder(ref["xyz"]):
            del ref["xyz"]
            if pixel_zero:  # both halves placeholders: no point at all
                del ref["u"], ref["v"]
        elif pixel_zero:
            del ref["u"], ref["v"]
        else:
            del ref["xyz"]
    return ref


def check_pixel(ref, what):
    if not isinstance(ref, dict) or not {"u", "v"} <= set(ref):
        raise ValueError(f"{what} must be a head-image pixel {{u, v}}")
    u, v = finite(ref["u"]), finite(ref["v"])
    if not (0 <= u <= PIXEL_MAX and 0 <= v <= PIXEL_MAX):
        raise ValueError(f"{what}: u and v are in 0..{PIXEL_MAX:.0f}")
    return {"u": u, "v": v}


def check_ref(ref, what):
    """A pixel {u, v} (with an optional name) or a world point {xyz}."""
    if isinstance(ref, dict) and {"u", "v"} <= set(ref):
        out = check_pixel(ref, what)
        if isinstance(ref.get("name"), str) and ref["name"].strip():
            out["name"] = ref["name"].strip()[:60]
        return out
    if isinstance(ref, dict) and "xyz" in ref:
        xyz = ref["xyz"]
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError(f"{what}.xyz must be [x, y, z]")
        return {"xyz": [finite(x) for x in xyz]}
    raise ValueError(f"{what} must be a pixel {{u, v}} or a world point {{xyz}}")


def check_tilt(name, tilt):
    """The angles of a `tilt` argument, each inside its range; nulls are left out."""
    if not isinstance(tilt, dict):
        raise ValueError(f"{name}.tilt must be an object with pitch, roll and/or heading")
    extra = sorted(set(tilt) - set(TILT))
    if extra:
        raise ValueError(f"{name}.tilt: unexpected {extra}; it takes {list(TILT)}")
    return {key: in_range(f"{name}.tilt", key, value, TILT) for key, value in tilt.items() if value is not None}


def check_hinge(name, hinge):
    """A `hinge` argument: {"pivot": point, "axis": [x, y, z]}, either part optional."""
    if not isinstance(hinge, dict):
        raise ValueError(f"{name}.hinge must be an object with pivot and/or axis")
    extra = sorted(set(hinge) - {"pivot", "axis"})
    if extra:
        raise ValueError(f"{name}.hinge: unexpected {extra}; it takes ['pivot', 'axis']")
    out = {}
    pivot = tidy_ref(hinge.get("pivot"))
    if pivot not in (None, {}):
        out["pivot"] = check_ref(pivot, f"{name}.hinge.pivot")
    axis = hinge.get("axis")
    if axis is not None and not placeholder(axis):
        if not isinstance(axis, list) or len(axis) != 3:
            raise ValueError(f"{name}.hinge.axis must be [x, y, z]")
        out["axis"] = [finite(x) for x in axis]
    return out


def unset(key, value):
    """Whether an argument was written only to fill the schema: null, or 0 where 0 is no value."""
    if value is None:
        return True
    return key in ZERO_IS_UNSET and not isinstance(value, bool) and value == 0


def check_call(name, arguments, open_width=OPEN_WIDTH):
    """A validated copy of one tool call's arguments, or ValueError with a message for the model.

    Arguments written as null, a quantity written as 0 where 0 is outside its range, and empty
    or all-zero points are read as not given: decoders behind some gateways fill unused fields
    that way. A real value in a wrong place is refused.
    """
    if name not in SPEC:
        raise ValueError(f"unknown tool {name!r}; the tools are {list(SPEC)}")
    if not isinstance(arguments, dict):
        raise ValueError(f"{name}: the arguments must be a JSON object")
    required, optional = SPEC[name]
    args = {}
    for key, value in arguments.items():
        if key in (*REFS, "watch"):
            value = tidy_ref(value)
            if value == {}:
                continue
        if key == "when" and isinstance(value, dict) and placeholder(value):
            continue
        if key == "guard" and value is False:
            continue  # the default, written out
        if key in ("tilt", "hinge") and (value is None or value == {}):
            continue
        if not unset(key, value):
            args[key] = value
    unexpected = sorted(set(args) - set(required) - set(optional))
    if unexpected:
        raise ValueError(f"{name}: unexpected arguments {unexpected}; {name} takes {[*required, *optional]}")
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError(f"{name} needs {missing}")
    if "arm" in args and args["arm"] not in SIDES:
        raise ValueError(f'{name}.arm must be "left" or "right"')
    for key in REFS:
        if key in args:
            args[key] = check_ref(args[key], f"{name}.{key}")
    if "by" in args:
        by = args["by"]
        if not isinstance(by, list) or len(by) != 3:
            raise ValueError(f"{name}.by must be [dx, dy, dz]")
        args["by"] = [finite(x) for x in by]
    if "guard" in args and args["guard"] is not True:
        raise ValueError(f"{name}.guard must be true or false")
    if "tilt" in args:
        args["tilt"] = check_tilt(name, args["tilt"])
        if not args["tilt"]:
            del args["tilt"]
        elif args.get("approach") == "keep":
            raise ValueError(f"{name}.tilt needs approach top or front, not keep")
    if "hinge" in args:
        args["hinge"] = check_hinge(name, args["hinge"])
    for key, choices in (("approach", APPROACHES), ("gripper", GRIPPER), ("mode", PLACE_MODES)):
        if key in args and args[key] not in choices:
            raise ValueError(f"{name}.{key} must be one of {list(choices)}")
    table = SETTINGS if name == "configure" else CALL
    table = {key: rig_quantity(table, key, open_width) for key in table}
    for key in table:
        if key in args:
            args[key] = in_range(name, key, args[key], table)
    if name == "place":
        args.setdefault("mode", "on")
    if name == "wait":
        given = [key for key in ("seconds", "until_t", "watch") if key in args]
        if len(given) != 1:
            raise ValueError("wait takes exactly one of seconds, until_t or watch")
        if "timeout" in args and "watch" not in args:
            raise ValueError("wait.timeout goes with watch")
    if "watch" in args:
        args["watch"] = check_pixel(args["watch"], f"{name}.watch")
    if "when" in args:
        when = args["when"]
        if not isinstance(when, dict):
            raise ValueError(f"{name}.when must be {{u, v}}")
        extra = sorted(set(when) - {"u", "v", "timeout"})
        if extra:
            raise ValueError(f"{name}.when: unexpected {extra}; it takes ['u', 'v', 'timeout']")
        checked = {"watch": check_pixel(when, f"{name}.when")}
        if not unset("timeout", when.get("timeout")):
            checked["timeout"] = in_range(f"{name}.when", "timeout", when["timeout"], CALL)
        args["when"] = checked
    return args


def describe(name, args):
    """A call as one line, for the log."""
    inner = ", ".join(f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in args.items())
    return f"{name}({inner})"
