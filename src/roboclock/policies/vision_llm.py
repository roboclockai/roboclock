"""Vision LLM agent (e.g. Qwen3.8-27B on vLLM) commanding the dual-X5 tool points through local IK.

Works in every domain that uses the RoboDojo dual ARX X5 rig (roboclock.envs.robodojo_rig): lab,
hospital, household, sports and industry.

The design follows RoboDojo's LLM agents: the model sees the RGB views and the measured state,
and a local planner turns its answer into joint targets.
- Per arm, the answer is a short list of waypoints for the point between the jaws, so one
  answer can approach, grasp and lift.
- A waypoint is a head-camera pixel (lifted to world metres through the head depth image), a
  world point, or an offset from the previous waypoint.
- Waypoints run at a fixed tool speed, so an answer lasts as long as its motion plus a
  requested wait.

The agent gets what a calibrated RGB-D rig provides, never simulator object state.
- Custom policies receive only observations. At reset the agent therefore recovers each arm's
  base from the measured joints and tool pose (`X5Kinematics.calibrated`).
- The head camera's pose relative to the two bases, and its intrinsics, are fixed by the
  published RoboDojo rig spec. The agent derives the camera from the two bases with that spec.
- Run `roboclock eval --depth` so observations carry the head depth image. Without it, pixel
  waypoints are refused and the model is told why.

Any OpenAI-compatible chat endpoint works: a local vLLM server or a hosted API such as NVIDIA's
inference API (https://inference-api.example.com/v1). Custom policy factories take no arguments,
so the policy is configured by environment:
  VISION_LLM_URL              OpenAI-compatible base URL (default http://127.0.0.1:8000/v1)
  VISION_LLM_COOKIE           code-server-session cookie when the URL goes through code-server
  VISION_LLM_API_KEY          bearer token of a hosted API (sent as Authorization, never logged)
  VISION_LLM_MODEL            served model name (default Qwen/Qwen3.8-27B)
  VISION_LLM_DIALECT          vllm (default) sends vLLM's thinking controls (chat_template_kwargs,
                              thinking_token_budget); openai sends only standard OpenAI fields, for
                              hosted APIs (the model then reasons as its provider configures it)
  VISION_LLM_THINKING         0 answers without reasoning first (default 1)
  VISION_LLM_THINKING_BUDGET  most reasoning tokens per request (default 1024; needs a server
                              started with a reasoning parser, as serve_qwen.sh does)
  VISION_LLM_REASONING_EFFORT openai dialect with thinking: reasoning_effort to request (e.g. low,
                              medium, high; default: not sent)
  VISION_LLM_MAX_TOKENS       most output tokens per request, reasoning included (default 1024,
                              plus the thinking budget when thinking)
  VISION_LLM_MAX_STEP         longest single waypoint move in metres (default 0.6)
  VISION_LLM_TIMEOUT          request timeout in seconds (default 300)
  VISION_LLM_RETRIES          retries of a request that failed with HTTP 408/429/5xx or a dropped
                              connection, after 5, 10, 20, ... s (default 3; not after a timeout)
  VISION_LLM_LOG              optional JSONL file receiving one row per request

`roboclock eval --policy roboclock.policies.vision_llm:make --depth`. See docs/vision-llm-agent.md.
"""

import base64
import io
import json
import math
import os
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from roboclock.envs import robodojo_rig
from roboclock.envs.lab.robot import ARM_DOF, DOF, JOINT_HIGH, JOINT_LOW
from roboclock.envs.rates import CONTROL_HZ
from roboclock.policies.x5_kinematics import X5Kinematics, pose_matrix, transform

GRIPPER_OPEN, GRIPPER_CLOSED = 0.044, 0.0
TOOL_SPEED = 0.25  # m/s along a waypoint segment
TURN_SPEED = 1.5  # rad/s when a waypoint turns the tool down
GRIPPER_TIME = 0.5  # s for a full open or close
MAX_WAYPOINTS = 4
MAX_WAIT = 2.0  # s an answer may hold after its waypoints
ERROR_HOLD = 0.5  # s held after an invalid answer
DEPTH_PATCH = 2  # pixels around a pointed pixel whose median depth is used
TOOL_DOWN = np.diag([1.0, -1.0, -1.0])  # tool +Z (fingers) at world -Z
IK_WAYPOINTS = 8
HISTORY = 4
HEAD_SIZE, WRIST_SIZE = (640, 480), (320, 240)
SIDES = ("left", "right")
VIEWS = (
    ("head", "rgb", HEAD_SIZE),
    ("left wrist", "left_wrist_rgb", WRIST_SIZE),
    ("right wrist", "right_wrist_rgb", WRIST_SIZE),
)
REFERENCES = ("pixel", "world", "offset")
# USD camera axes (x right, y up, looking along -z) to optical axes (x right, y down, z forward).
USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])
# A calibration whose bases are this far from the spec separation is reported to the model.
RIG_TOLERANCE_M = 0.01

WAYPOINT_SCHEMA = {
    "type": "object",
    "properties": {
        "ref": {"enum": list(REFERENCES)},
        "value": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "tool": {"enum": ["keep", "down"]},
        "gripper": {"enum": ["open", "close"]},
    },
    "required": ["ref", "value", "tool", "gripper"],
    "additionalProperties": False,
}
ARM_SCHEMA = {"type": "array", "items": WAYPOINT_SCHEMA, "maxItems": MAX_WAYPOINTS}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
        "left": ARM_SCHEMA,
        "right": ARM_SCHEMA,
        "wait": {"type": "number"},
    },
    "required": ["plan", "left", "right", "wait"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = """You control two identical 6-DoF ARX X5 arms, "left" and "right", each with a \
parallel-jaw gripper, in a simulated workspace. You receive a head camera view and one wrist \
camera view per arm. World coordinates are metres with +z up; both arm bases are mounted at \
z = {mount:.2f}. "left" and "right" are fixed arm names and need not match image sides, so \
identify each arm by its coordinates and its wrist view.

Each turn, give each arm a list of up to {waypoints} waypoints for the point between its jaws. \
They run in order, both arms at once. Each waypoint has:
- "ref" and "value", one of:
  - "pixel": [u, v, h] is the surface seen at pixel (u, v) of the {width}x{height} head image, \
u counted right from its left edge and v down from its top edge, raised h metres. Use it to \
reach anything visible in the head image: h = 0 touches the visible surface, h = 0.1 hovers \
10 cm above it, and a negative h goes below it, e.g. into a container you grasp from the top.
  - "world": [x, y, z] is a world point.
  - "offset": [dx, dy, dz] moves from the previous waypoint, or from the current tool point for \
the first one; [0, 0, 0.05] lifts 5 cm.
- "tool": "down" turns the fingers toward the floor for top grasps; "keep" holds the orientation.
- "gripper": "open" or "close", applied while moving to the waypoint. To grasp, reach the \
object with the gripper open, then add an offset [0, 0, 0] waypoint with "close".
An empty list holds that arm still. Tools move at {speed:.2f} m/s, so a 0.3 m reach takes about \
{reach:.1f} s. A waypoint farther than {max_step:.2f} m from the previous one is shortened.

"wait" holds the arms for that many seconds (0 to {max_wait:.0f}) after the waypoints, before \
you see the next views. Keep it short while watching for an event.
"plan": one short sentence saying what these waypoints achieve and what comes next.

After each turn you are told where pixel waypoints landed in world coordinates and whether a \
waypoint was shortened, unreachable or refused. The task clock can advance while you think and \
while the arms move, so act promptly against the task's deadlines. Answer with the JSON object \
only."""


def jpeg_data_url(image, size):
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).resize(size).save(
        buffer, "JPEG", quality=90
    )
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_answer(text):
    """The model's motion, or None when it is not a valid answer."""
    try:
        answer = json.loads(text)
        float(answer["wait"])
        for side in SIDES:
            waypoints = answer[side]
            if not isinstance(waypoints, list) or len(waypoints) > MAX_WAYPOINTS:
                return None
            for waypoint in waypoints:
                value = np.asarray(waypoint["value"], dtype=float)
                if value.shape != (3,) or not np.all(np.isfinite(value)):
                    return None
                if (
                    waypoint["ref"] not in REFERENCES
                    or waypoint["tool"] not in ("keep", "down")
                    or waypoint["gripper"] not in ("open", "close")
                ):
                    return None
        if not isinstance(answer["plan"], str) or not math.isfinite(float(answer["wait"])):
            return None
        return answer
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def rig_camera(left_base, right_base):
    """Head camera of the RoboDojo rig whose arm bases (4x4 world poses, +x forward) are given.

    Returns the 3x3 intrinsics of the rendered head image, the 4x4 world pose of its optical
    frame (x right, y down, z forward), the rig's mount height and how far the measured base
    separation is from the spec's.
    """
    origin = (left_base[:3, 3] + right_base[:3, 3]) / 2.0
    forward = left_base[:3, 0] + right_base[:3, 0]
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    pose = robodojo_rig.head_camera_world(origin, yaw)
    pose[:3, :3] = pose[:3, :3] @ USD_TO_OPTICAL
    separation = float(np.linalg.norm(left_base[:3, 3] - right_base[:3, 3]))
    return {
        "intrinsics": robodojo_rig.intrinsics(robodojo_rig.HEAD_CAMERA),
        "resolution": tuple(robodojo_rig.HEAD_CAMERA["resolution"]),
        "pose": pose,
        "mount": float(origin[2]),
        "separation_error_m": abs(separation - robodojo_rig.ARM_SEPARATION_M),
    }


def pixel_to_world(observation, camera, u, v, grid):
    """World point seen at pixel (u, v) of a grid-sized head image, or (None, reason)."""
    depth = observation.get("depth")
    if depth is None or camera is None:
        return None, "refused: no head depth image (run eval with --depth)"
    depth = np.asarray(depth)
    height, width = depth.shape[:2]
    x, y = u * width / grid[0], v * height / grid[1]
    if not (0 <= x < width and 0 <= y < height):
        return None, f"refused: pixel ({u:.0f}, {v:.0f}) is outside the head image"
    row, column = int(y), int(x)
    patch = depth[
        max(0, row - DEPTH_PATCH) : row + DEPTH_PATCH + 1,
        max(0, column - DEPTH_PATCH) : column + DEPTH_PATCH + 1,
    ]
    distances = patch[np.isfinite(patch) & (patch > 0)]
    if not distances.size:
        return None, f"refused: no surface at pixel ({u:.0f}, {v:.0f})"
    distance = float(np.median(distances))
    # Intrinsics are those of the rendered image; scale them to the depth image's size.
    k = np.asarray(camera["intrinsics"], dtype=float).copy()
    k[0] *= width / camera["resolution"][0]
    k[1] *= height / camera["resolution"][1]
    point = np.array(
        [(x - k[0, 2]) * distance / k[0, 0], (y - k[1, 2]) * distance / k[1, 1], distance]
    )
    pose = np.asarray(camera["pose"], dtype=float)
    return pose[:3, :3] @ point + pose[:3, 3], None


DIALECTS = ("vllm", "openai")
# Worth retrying: the request timed out at a proxy, the provider throttled, or a server failed.
RETRY_CODES = frozenset({408, 429, 500, 502, 503, 504})
RETRY_WAIT = 5.0  # s before the first retry; doubles each time, at most RETRY_WAIT_MAX
RETRY_WAIT_MAX = 60.0


class ContentFiltered(RuntimeError):
    """The provider refused this request under its content policy (Azure OpenAI answers HTTP 400
    with code content_filter). It judges one request, images included, and is not a fault of the
    endpoint: the same episode's next request usually passes."""


def endpoint_settings(policy):
    """Read the endpoint settings both vision-LLM agents share (module docstring) into `policy`."""
    policy.url = os.environ.get("VISION_LLM_URL", "http://127.0.0.1:8000/v1")
    policy.cookie = os.environ.get("VISION_LLM_COOKIE")
    policy.api_key = os.environ.get("VISION_LLM_API_KEY") or None
    policy.model = os.environ.get("VISION_LLM_MODEL", "Qwen/Qwen3.8-27B")
    policy.dialect = os.environ.get("VISION_LLM_DIALECT", "vllm")
    if policy.dialect not in DIALECTS:
        raise ValueError(f"VISION_LLM_DIALECT must be one of {', '.join(DIALECTS)}")
    policy.thinking = os.environ.get("VISION_LLM_THINKING", "1") != "0"
    policy.thinking_budget = int(os.environ.get("VISION_LLM_THINKING_BUDGET", "1024"))
    policy.reasoning_effort = os.environ.get("VISION_LLM_REASONING_EFFORT") or None
    max_tokens = os.environ.get("VISION_LLM_MAX_TOKENS")
    policy.max_tokens = int(max_tokens) if max_tokens else None
    policy.timeout = float(os.environ.get("VISION_LLM_TIMEOUT", "300"))
    policy.retries = int(os.environ.get("VISION_LLM_RETRIES", "3"))
    # Whether a request that times out is sent again. Off for a local server, where a timeout
    # means it is overloaded and a second request would only lengthen its queue; on for a hosted
    # API, where it is a stalled connection (two episodes were lost to one, 2026-11-10).
    policy.retry_timeouts = os.environ.get("VISION_LLM_RETRY_TIMEOUTS", "0") == "1"
    policy.log_path = os.environ.get("VISION_LLM_LOG")
    # Names the episode in every logged row. A sweep that writes one log per episode does not
    # need it; an ablation that runs one condition into one log does, since each episode
    # restarts the request counter (docs/ablations.md).
    policy.episode_tag = os.environ.get("VISION_LLM_EPISODE") or None


def chat_body(policy, messages, schema_name, schema):
    """Chat request for `policy`'s model and dialect, answering in the JSON `schema`."""
    body = {
        "model": policy.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        },
    }
    if policy.dialect == "vllm":
        body["chat_template_kwargs"] = {"enable_thinking": policy.thinking}
    if policy.thinking:
        body["max_tokens"] += policy.thinking_budget
        if policy.dialect == "vllm":
            body["thinking_token_budget"] = policy.thinking_budget
        elif policy.reasoning_effort:
            body["reasoning_effort"] = policy.reasoning_effort
    if policy.max_tokens:
        body["max_tokens"] = policy.max_tokens
    return body


def request_chat(url, body, timeout, cookie=None, api_key=None, retries=0, retry_timeouts=False):
    """POST `body` to the chat endpoint at `url` and return the reply, retrying transient failures
    (and, with `retry_timeouts`, a request that timed out)."""
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = f"code-server-session={cookie}"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        wait = min(RETRY_WAIT * 2**attempt, RETRY_WAIT_MAX)
        request = urllib.request.Request(
            url.rstrip("/") + "/chat/completions", data=data, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                reply = json.load(response)
                # A LiteLLM gateway (e.g. NVIDIA's inference API) reports what the request was billed.
                cost = response.headers.get("x-litellm-response-cost")
                if cost is not None and isinstance(reply, dict):
                    try:
                        reply["response_cost_usd"] = float(cost)
                    except ValueError:
                        pass
                return reply
        except urllib.error.HTTPError as error:
            answer = error.read()
            detail = answer[:500]
            if error.code in RETRY_CODES and attempt < retries:
                time.sleep(wait)
                continue
            # Azure OpenAI words it two ways: code content_filter (the text or the answer), and
            # content_policy_violation ("Image processing blocked", sports B3 on 2026-11-11).
            if error.code == 400 and (b"content_filter" in answer or b"content_policy" in answer):
                raise ContentFiltered(f"Vision LLM server {error.code}: {answer[:2000]!r}") from error
            raise RuntimeError(f"Vision LLM server {error.code}: {detail!r}") from error
        except json.JSONDecodeError as error:
            # code-server redirects a missing or stale cookie to its HTML login page.
            raise RuntimeError(
                f"Vision LLM server at {url} did not answer with JSON; "
                "a code-server /proxy/ URL needs a valid VISION_LLM_COOKIE"
            ) from error
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            timed_out = isinstance(error, TimeoutError) or isinstance(
                getattr(error, "reason", None), TimeoutError
            )
            if (retry_timeouts or not timed_out) and attempt < retries:
                time.sleep(wait)
                continue
            raise RuntimeError(f"Vision LLM server unreachable at {url}: {error}") from error


class VisionLLMPolicy:
    def __init__(self):
        endpoint_settings(self)
        # Output tokens of the latest model query, thinking included (--inference-latency ideal).
        self.last_output_tokens = None
        self.max_step = float(os.environ.get("VISION_LLM_MAX_STEP", "0.6"))
        self.control_hz = float(CONTROL_HZ)
        self.kinematics = X5Kinematics()

    @property
    def inference_model(self):
        """Served model name; --inference-latency ideal charges queries at its surveyed speed."""
        return self.model

    def reset(self, observation):
        joints = np.asarray(observation["joint_positions"], dtype=float)
        pose = np.asarray(observation["ee_pose"], dtype=float)
        self.arms = [
            self.kinematics.calibrated(
                joints[DOF * index : DOF * index + ARM_DOF],
                pose[7 * index : 7 * index + 3],
                pose[7 * index + 3 : 7 * index + 7],
            )
            for index in range(2)
        ]
        self.camera = rig_camera(self.arms[0].base, self.arms[1].base)
        self.system = SYSTEM_PROMPT.format(
            mount=self.camera["mount"],
            waypoints=MAX_WAYPOINTS,
            width=HEAD_SIZE[0],
            height=HEAD_SIZE[1],
            speed=TOOL_SPEED,
            reach=0.3 / TOOL_SPEED,
            max_step=self.max_step,
            max_wait=MAX_WAIT,
        )
        # Commanded finger targets: holding the measured width of a grasped object would stop
        # squeezing it, so a still arm keeps its last command.
        self.grippers = [float(joints[DOF * index + ARM_DOF]) for index in range(2)]
        self.queue = []
        self.history = []
        self.feedback = ""
        self.requests = 0
        self.expected = None
        self.queried = True

    def act(self, observation):
        self.queried = not self.queue
        if self.queried:
            self.queue = self.plan(observation)
        return self.queue.pop(0)

    def camera_text(self, observation):
        if "depth" not in observation:
            return "Head camera: no depth image, so pixel waypoints are refused."
        pose = self.camera["pose"]
        focal = float(self.camera["intrinsics"][0, 0]) * HEAD_SIZE[0] / self.camera["resolution"][0]
        return (
            f"Head camera: at {np.round(pose[:3, 3], 3).tolist()}, looking along "
            f"{np.round(pose[:3, 2], 2).tolist()}, focal length {focal:.0f} px in the "
            f"{HEAD_SIZE[0]}x{HEAD_SIZE[1]} head image."
        )

    def prompt(self, observation):
        pose = np.asarray(observation["ee_pose"], dtype=float)
        width = np.asarray(observation["gripper_width"], dtype=float)
        lines = [f"Task: {observation['instruction']}"]
        lines.append(f"Time: {float(np.ravel(observation['sim_time'])[0]):.1f} s")
        for index, side in enumerate(SIDES):
            lines.append(
                f"{side} tool point {np.round(pose[7 * index : 7 * index + 3], 3).tolist()}, "
                f"gripper opening {width[index]:.3f} m"
            )
        lines.append(self.camera_text(observation))
        if self.feedback:
            lines.append(f"Last turn: {self.feedback}")
        if self.history:
            lines.append("Recent plans: " + " | ".join(self.history[-HISTORY:]))
        content = []
        for name, key, size in VIEWS:
            if key in observation:
                content.append({"type": "text", "text": f"{name} camera:"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": jpeg_data_url(observation[key], size)},
                    }
                )
        content.append({"type": "text", "text": "\n".join(lines)})
        return content

    def request_body(self, observation):
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.prompt(observation)},
        ]
        return chat_body(self, messages, "dual_arm_waypoints", ANSWER_SCHEMA)

    def chat(self, body):
        return request_chat(
            self.url, body, self.timeout, self.cookie, api_key=self.api_key, retries=self.retries
        )

    def plan(self, observation):
        # Reading the images renders them (roboclock.envs.observation); build the request before
        # timing, so round_trip_ms is the model server alone.
        body = self.request_body(observation)
        tracking = self.tracking_error(observation)
        start = time.perf_counter()
        reply = self.chat(body)
        round_trip = time.perf_counter() - start
        usage = reply.get("usage") or {}
        self.last_output_tokens = usage.get("completion_tokens")
        message = reply["choices"][0]["message"]
        text = message.get("content") or ""
        answer = parse_answer(text)
        # A malformed answer is the model's own failure: it costs a turn, not the run.
        steps, row = (
            self.motion(observation, answer) if answer else self.hold(observation, "invalid answer")
        )
        self.requests += 1
        if answer:
            self.history.append(answer["plan"][:160])
        if self.log_path:
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            row.update(
                request=self.requests - 1,
                sim_time=float(np.ravel(observation["sim_time"])[0]),
                round_trip_ms=round_trip * 1000.0,
                tracking_error_m=tracking,
                prompt_tokens=usage.get("prompt_tokens"),
                cached_prompt_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                cost_usd=reply.get("response_cost_usd"),
                rig_separation_error_m=self.camera["separation_error_m"],
                answer=text[:1500],
                reasoning=reasoning[:2000],
            )
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        return steps

    def tracking_error(self, observation):
        """Measured distance per arm from where the previous answer was meant to end."""
        if self.expected is None:
            return None
        pose = np.asarray(observation["ee_pose"], dtype=float)
        return [
            float(np.linalg.norm(pose[7 * index : 7 * index + 3] - self.expected[index]))
            for index in range(2)
        ]

    def hold(self, observation, reason):
        self.feedback = reason
        self.expected = None
        joints = np.asarray(observation["joint_positions"], dtype=np.float32).copy()
        for index, fingers in enumerate(self.grippers):
            joints[DOF * index + ARM_DOF : DOF * (index + 1)] = fingers
        count = max(1, round(ERROR_HOLD * self.control_hz))
        target = np.clip(joints, np.tile(JOINT_LOW, 2), np.tile(JOINT_HIGH, 2))
        return [target.copy() for _ in range(count)], {"parse_error": True}

    def motion(self, observation, answer):
        joints = np.asarray(observation["joint_positions"], dtype=float)
        pose = np.asarray(observation["ee_pose"], dtype=float)
        tracks, moved, notes, landed, residuals, shortened = [], [], [], [], [], []
        self.expected = []
        for index, side in enumerate(SIDES):
            current = pose_matrix(
                pose[7 * index : 7 * index + 3], pose[7 * index + 3 : 7 * index + 7]
            )
            track, end, report = self.arm_track(
                index, joints[DOF * index : DOF * (index + 1)], current, answer[side], observation
            )
            tracks.append(track)
            moved.append(report["moved"])
            self.expected.append(end)
            notes += [f"{side} {note}" for note in report["notes"]]
            landed.append(report["landed"])
            residuals.append(report["residual"])
            shortened.append(report["shortened"])
        wait = min(max(float(answer["wait"]), 0.0), MAX_WAIT)
        moving = max(len(track) if report else 0 for track, report in zip(tracks, moved))
        count = moving + round(wait * self.control_hz)
        steps = []
        for step in range(max(1, count)):
            action = np.concatenate([track[min(step, len(track) - 1)] for track in tracks])
            steps.append(np.clip(action, np.tile(JOINT_LOW, 2), np.tile(JOINT_HIGH, 2)))
        self.feedback = "; ".join(notes) or "reached as commanded"
        return steps, {
            "parse_error": False,
            "pixel_points": landed,
            "shortened": shortened,
            "ik_residual_m": residuals,
            "control_steps": len(steps),
            "wait_s": wait,
        }

    def arm_track(self, index, joints, current, waypoints, observation):
        """Per-step joint and gripper targets for one arm's waypoints, its final tool point,
        and what the model should be told about them."""
        kinematics = self.arms[index]
        rotation, position = current[:3, :3], current[:3, 3]
        arm, gripper = joints[:ARM_DOF], self.grippers[index]
        track = [np.concatenate([arm, [gripper, gripper]])]
        report = {"notes": [], "landed": [], "residual": 0.0, "shortened": False}
        for number, waypoint in enumerate(waypoints, start=1):
            value = np.asarray(waypoint["value"], dtype=float)
            if waypoint["ref"] == "pixel":
                point, problem = pixel_to_world(
                    observation, self.camera, value[0], value[1], HEAD_SIZE
                )
                if point is None:
                    # Later offsets would be relative to a point that does not exist.
                    report["notes"].append(f"waypoint {number} {problem}; the rest skipped")
                    break
                goal = point + [0.0, 0.0, value[2]]
                report["landed"].append(np.round(point, 3).tolist())
                report["notes"].append(
                    f"waypoint {number} pixel ({value[0]:.0f}, {value[1]:.0f}) is the surface "
                    f"at {np.round(point, 3).tolist()}"
                )
            elif waypoint["ref"] == "world":
                goal = value
            else:
                goal = position + value
            goal, was_shortened = self.bounded_target(position, goal)
            if was_shortened:
                report["shortened"] = True
                report["notes"].append(
                    f"waypoint {number} shortened to {self.max_step:.2f} m from the previous one"
                )
            end_rotation = TOOL_DOWN if waypoint["tool"] == "down" else rotation
            path, residual = self.arm_path(
                kinematics, arm, transform(rotation, position), goal, end_rotation
            )
            report["residual"] = max(report["residual"], residual)
            if residual > 0.01:
                report["notes"].append(f"waypoint {number} out of reach by {residual:.3f} m")
            closing = GRIPPER_OPEN if waypoint["gripper"] == "open" else GRIPPER_CLOSED
            turn = Rotation.from_matrix(end_rotation @ rotation.T).magnitude()
            seconds = max(
                np.linalg.norm(goal - position) / TOOL_SPEED,
                turn / TURN_SPEED,
                abs(closing - gripper) / GRIPPER_OPEN * GRIPPER_TIME,
            )
            count = max(1, math.ceil(seconds * self.control_hz - 1e-9))
            for step in range(count):
                phase = (step + 1) / count
                fingers = gripper + (closing - gripper) * phase
                track.append(np.concatenate([interpolate(path, phase), [fingers, fingers]]))
            achieved = kinematics.tool_pose(path[-1])
            arm, gripper = path[-1], closing
            rotation, position = achieved[:3, :3], goal
        self.grippers[index] = gripper
        report["moved"] = len(track) > 1
        # The first entry is the measured arm, so an arm without waypoints holds where it is.
        return np.asarray(track[1:] or track, dtype=np.float32), position, report

    def bounded_target(self, start, target):
        """`target` moved at most max_step from `start`. Scenes differ in their surfaces, so
        there is no floor clamp; IK and contact keep the jaws out of solid geometry."""
        delta = np.asarray(target, dtype=float) - start
        length = float(np.linalg.norm(delta))
        shortened = length > self.max_step
        if shortened:
            delta *= self.max_step / length
        return start + delta, shortened

    def arm_path(self, kinematics, joints, current, target, end_rotation):
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([current[:3, :3], end_rotation])))
        path = [joints]
        residual = 0.0
        for phase in np.linspace(0.0, 1.0, IK_WAYPOINTS + 1)[1:]:
            waypoint = transform(
                slerp(phase).as_matrix(), current[:3, 3] + (target - current[:3, 3]) * phase
            )
            solved, residual = kinematics.solve(waypoint, path[-1])
            path.append(solved)
        return np.stack(path), residual


def interpolate(path, phase):
    position = phase * (len(path) - 1)
    index = min(int(position), len(path) - 2)
    return path[index] + (path[index + 1] - path[index]) * (position - index)


def make():
    return VisionLLMPolicy()
