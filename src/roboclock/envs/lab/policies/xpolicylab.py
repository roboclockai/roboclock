"""Remote XPolicyLab policy server (e.g. RoboDojo pi0.5) driving the dual-X5 lab arms.

Speaks the XPolicyLab websocket protocol (msgpack frames: hello, reset, infer) and maps
between the lab observation/action and RoboDojo's ``arx_x5`` joint layout:
[left joints 1-6, left gripper, right joints 1-6, right gripper], with each gripper
normalised to [0, 1] over RoboDojo's X5 ``gripper_scale``. The lab interface lists the arm on
the rig's left (+y, the arm `left_wrist_rgb` rides on) first, as RoboDojo does, so the arms map
straight through (ARM_ORDER = (0, 1)). Chunks recorded at RoboDojo's 25 Hz are resampled onto
the lab control clock, so motions keep their trained speed.

Custom policy factories take no arguments, so the policy is configured by environment:
  XPOLICYLAB_URL         ws://host:port, or wss://<code-server host>/proxy/<port>/
  XPOLICYLAB_COOKIE      code-server-session cookie when the URL goes through code-server
  XPOLICYLAB_REPLAN      policy actions executed per chunk (default: the whole chunk, as RoboDojo)
  XPOLICYLAB_CONTROL_HZ  lab control rate (default: the shared 50 Hz control rate)
  XPOLICYLAB_LOG         optional JSONL file receiving one row per inference
  XPOLICYLAB_PROMPT_TIME 1: append the observation's env time to the instruction,
                         "<instruction> Time: <sim_time:.1f> s", as the roboclock-demo fine-tunes are
                         trained (docs/five-domain-training.md, rule 4); default 0 sends the
                         instruction unchanged (RoboDojo checkpoints). Read by every XPolicyLab client
                         (this one, xvla, openwam, internw0) through policy_instruction().

Under ``roboclock eval --time-model streaming`` the evaluator takes whole chunks (act_chunk) and
reports which action is in effect (on_action). See docs/xpolicylab-pi05.md.
"""

import asyncio
import io
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from roboclock.envs.lab.env import JOINT_HIGH, JOINT_LOW
from roboclock.envs.rates import CONTROL_HZ

# RoboDojo Assets/Robots/x5/robot_config.yml (joint7 drives, joint8 mimics 1:1).
GRIPPER_SCALE = (-0.01, 0.044)
# RoboDojo env_cfg/arx_x5.yml collect_freq: one chunk entry per 1/25 s.
POLICY_HZ = 25.0
LOW, HIGH = np.tile(JOINT_LOW, 2), np.tile(JOINT_HIGH, 2)
ARM_DIMS = np.r_[0:6, 8:14]
# XPolicyLab tags standard RGB JPEGs with this COM payload; untagged ones are read channel-swapped.
RGB_MARKER = b"XPL-RGB1"
# RoboDojo camera name -> lab observation key; Pi_05 also accepts cam_head for cam_high.
CAMERAS = (
    ("cam_head", "rgb"),
    ("cam_left_wrist", "left_wrist_rgb"),
    ("cam_right_wrist", "right_wrist_rgb"),
)
SIDES = ("left", "right")
# Interface arm index (8 joints each in the lab action/state) of RoboDojo's left and right arm.
# The lab / hospital interface lists the rig's left arm (+y, `left_wrist_rgb`) first, like
# RoboDojo, so this is a straight pass-through (LabEnv converts to its internal side order).
ARM_ORDER = (0, 1)


def encode_jpeg(image):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(
        buffer, "JPEG", quality=95
    )
    data = buffer.getvalue()
    # The COM segment goes after JFIF APP0, where XPolicyLab's encoder puts it.
    offset = 4 + int.from_bytes(data[4:6], "big") if data[2:4] == b"\xff\xe0" else 2
    segment = b"\xff\xfe" + (len(RGB_MARKER) + 2).to_bytes(2, "big") + RGB_MARKER
    return data[:offset] + segment + data[offset:]


def normalise_gripper(width):
    return (np.asarray(width, dtype=np.float32) - GRIPPER_SCALE[0]) / (
        GRIPPER_SCALE[1] - GRIPPER_SCALE[0]
    )


def robodojo_vector(action):
    """One XPolicyLab joint action dict as the packed 14-value RoboDojo vector."""
    return np.concatenate(
        [
            np.asarray(action[f"{side}_{part}_joint_state"], dtype=np.float32).reshape(-1)
            for side in SIDES
            for part in ("arm", "ee")
        ]
    )


def lab_action(vector):
    """Packed RoboDojo vectors (..., 14) as unclipped lab joint targets (..., 16)."""
    vector = np.asarray(vector, dtype=np.float32)
    action = np.empty(vector.shape[:-1] + (16,), dtype=np.float32)
    for arm, lab_arm in enumerate(ARM_ORDER):
        source = vector[..., 7 * arm : 7 * arm + 7]
        target = action[..., 8 * lab_arm : 8 * lab_arm + 8]
        target[..., :6] = source[..., :6]
        opening = np.clip(source[..., 6:7], 0.0, 1.0)
        target[..., 6:8] = GRIPPER_SCALE[0] + opening * (GRIPPER_SCALE[1] - GRIPPER_SCALE[0])
    return action


def prompt_with_time(instruction, sim_time_s):
    """Instruction plus env time, character for character the training prompt of the time-aware
    fine-tunes (there demo frame k is sim time k / 25 s; here it is the observation's)."""
    return f"{instruction} Time: {sim_time_s:.1f} s"


def observation_time(observation):
    return float(np.asarray(observation["sim_time"]).reshape(-1)[0])


def prompt_time_enabled():
    return os.environ.get("XPOLICYLAB_PROMPT_TIME", "0") not in ("", "0")


def policy_instruction(observation, with_time):
    """The instruction a client sends: the env's, plus the env time when the policy was trained so."""
    instruction = observation["instruction"]
    return prompt_with_time(instruction, observation_time(observation)) if with_time else instruction


def resample(chunk, source_hz, target_hz):
    """Replay a chunk recorded at source_hz on a target_hz clock over the same duration."""
    count = int(np.ceil(len(chunk) * target_hz / source_hz))
    times = np.arange(count) / target_hz
    source = np.arange(len(chunk)) / source_hz
    return np.stack([np.interp(times, source, column) for column in chunk.T], axis=1)


class Connection:
    """One websocket per server URL, shared by the per-episode policy objects.

    The socket lives on a daemon event-loop thread, as in XPolicyLab's WsModelClient:
    keepalive pings are answered while the simulator blocks, and an open connection
    never keeps the evaluator process alive at exit (websockets' sync client would).
    """

    def __init__(self, url, cookie=None, timeout=120.0):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="xpolicylab", daemon=True).start()
        self.timeout = timeout
        self.socket = self.run(self.open(url, cookie))
        self.evaluation_id = f"roboclock-{uuid.uuid4().hex[:12]}"
        self.server_instance = self.request("hello", {}).get("server_instance_id")

    def run(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    @staticmethod
    async def open(url, cookie):
        try:
            from websockets.asyncio.client import connect

            header_option = "additional_headers"
        except ImportError:  # websockets < 13, as shipped with Isaac Sim
            from websockets.legacy.client import connect

            header_option = "extra_headers"
        headers = {"Cookie": f"code-server-session={cookie}"} if cookie else None
        # Frames carry multi-camera observations; the server also disables compression.
        return await connect(
            url, max_size=None, compression=None, open_timeout=60, **{header_option: headers}
        )

    def request(self, message_type, payload, *, trial_id=None, step=0):
        import msgpack
        import msgpack_numpy

        message_id = uuid.uuid4().hex
        frame = {
            "message_type": message_type,
            "message_id": message_id,
            "evaluation_id": self.evaluation_id,
            "action_case_id": None,
            "trial_id": trial_id,
            "repeat_index": None,
            "step": step,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self.run(
            self.socket.send(msgpack.packb(frame, default=msgpack_numpy.encode, use_bin_type=True))
        )
        while True:
            reply = msgpack.unpackb(
                self.run(asyncio.wait_for(self.socket.recv(), self.timeout)),
                raw=False,
                strict_map_key=False,
                object_hook=msgpack_numpy.decode,
            )
            if reply.get("message_id") != message_id:
                continue
            body = reply.get("payload") or {}
            if reply.get("message_type") == "error":
                raise RuntimeError(
                    f"XPolicyLab server {body.get('code', 'error')}: {body.get('message')}"
                )
            return body


_CONNECTIONS = {}


def connection():
    url = os.environ.get("XPOLICYLAB_URL")
    if not url:
        raise ValueError("Set XPOLICYLAB_URL to the policy server, e.g. ws://10.0.0.5:6000")
    if url not in _CONNECTIONS:
        _CONNECTIONS[url] = Connection(url, os.environ.get("XPOLICYLAB_COOKIE"))
    return _CONNECTIONS[url]


class XPolicyLabPolicy:
    def __init__(self):
        self.connection = connection()
        self.replan = int(os.environ.get("XPOLICYLAB_REPLAN", "0")) or None
        self.control_hz = float(os.environ.get("XPOLICYLAB_CONTROL_HZ", CONTROL_HZ))
        self.log_path = os.environ.get("XPOLICYLAB_LOG")
        self.prompt_time = prompt_time_enabled()
        self.warned_cameras = set()

    def reset(self, observation):
        self.trial_id = uuid.uuid4().hex
        self.inferences = 0
        self.queue = []
        # Read by the evaluator after act(): whether it queried the server, and the server's
        # own compute time for that query (None: charge the round trip).
        self.queried = True
        self.last_inference_latency = None
        # Streaming mode: commands of the chunks returned by act_chunk, by query number.
        self.chunk_commands = {}
        # RoboDojo reports the last commanded opening; before any command, the measured one.
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        self.gripper = normalise_gripper(joints[[8 * arm + 6 for arm in ARM_ORDER]])
        self.connection.request("reset", {"trial_id": self.trial_id}, trial_id=self.trial_id)

    def act(self, observation):
        self.queried = not self.queue
        if self.queried:
            self.queue = self.infer(observation)
        command, target = self.queue.pop(0)
        self.gripper = command[[6, 13]]
        return target

    def act_chunk(self, observation):
        """Streaming mode: one query; the whole chunk of joint targets at the control rate."""
        pairs = self.infer(observation)
        self.chunk_commands[len(self.chunk_commands)] = [command for command, _ in pairs]
        return np.stack([target for _, target in pairs])

    def on_action(self, query, index):
        """Streaming mode: target `index` of the chunk from query `query` is now in effect, so
        the gripper state sent with the next query is its command (RoboDojo semantics)."""
        self.gripper = self.chunk_commands[query][index][[6, 13]]
        for old in [key for key in self.chunk_commands if key < query]:
            self.chunk_commands[old] = None  # superseded; keep the numbering

    def observation(self, observation):
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        head = observation["rgb"]
        vision = {}
        for camera, key in CAMERAS:
            image = observation.get(key)
            if image is None:
                # Pi_05 needs all three views; a black frame keeps the request valid.
                if camera not in self.warned_cameras:
                    print(f"[xpolicylab] observation has no {key!r}; sending black {camera}")
                    self.warned_cameras.add(camera)
                image = np.zeros_like(head)
            vision[camera] = {"color": encode_jpeg(image)}
        state = {}
        for arm, (side, lab_arm) in enumerate(zip(SIDES, ARM_ORDER)):
            state[f"{side}_arm_joint_state"] = joints[8 * lab_arm : 8 * lab_arm + 6]
            state[f"{side}_ee_joint_state"] = np.asarray(
                self.gripper[arm : arm + 1], dtype=np.float32
            )
        return {
            "instruction": policy_instruction(observation, self.prompt_time),
            "env_idx": 0,
            "vision": vision,
            "state": state,
        }

    def infer(self, observation):
        # Packing reads the camera images, which renders them (roboclock.envs.observation);
        # build the request first so round_trip_ms is transport plus server time only.
        request = {"observation": self.observation(observation)}
        start = time.perf_counter()
        reply = self.connection.request(
            "infer", request, trial_id=self.trial_id, step=self.inferences
        )
        round_trip = time.perf_counter() - start
        self.inferences += 1
        server_ms = reply.get("latency_ms")
        self.last_inference_latency = None if server_ms is None else float(server_ms) / 1000.0
        chunk = np.stack([robodojo_vector(action) for action in reply["actions"]])
        if self.replan:
            chunk = chunk[: self.replan]
        commands = resample(chunk, POLICY_HZ, self.control_hz).astype(np.float32)
        targets = lab_action(commands)
        clipped = (targets < LOW) | (targets > HIGH)
        if self.log_path:
            row = {
                "trial_id": self.trial_id,
                "inference": self.inferences - 1,
                "sim_time": float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0]),
                "chunk_actions": int(len(reply["actions"])),
                "control_steps": int(len(commands)),
                "server_ms": reply.get("latency_ms"),
                "round_trip_ms": round_trip * 1000.0,
                # Gripper targets below 0 m are RoboDojo's squeeze command, not an error.
                "clipped_arm_targets": int(clipped[:, ARM_DIMS].sum()),
            }
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        return list(zip(commands, np.clip(targets, LOW, HIGH)))


def make():
    return XPolicyLabPolicy()
