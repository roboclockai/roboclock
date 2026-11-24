"""Awomo-0.5 (RoboDojo checkpoint) over XPolicyLab, driving the dual-X5 arms.

The checkpoint ``Auwomo/Awomo-0.5-Robodojo`` (rev 5a3e043) is a FLUX.2-klein world-action
model with a Qwen3-VL prompt encoder, served by XPolicyLab's ``policy/Awomo05`` adapter. It
predicts 16 absolute RoboDojo joint targets ``[left j1-6, left gripper, right j1-6, right
gripper]`` (grippers in [0, 1]) and the adapter returns the first 8 (``replan_steps``).

What the client must reproduce (``policy/Awomo05/deploy.py``):

- **The history lives on the server and advances one frame per ``update_obs``.** The adapter
  pushes the head view of every observation into a per-``env_idx`` buffer and reads its 20
  history slots (1 s to 20 s back) and the 20 Qwen3-VL history frames off that buffer,
  assuming one entry per 25 Hz RoboDojo step. So every act() sends ``update_obs`` with the
  full observation, and the act() that empties the queue also sends ``get_action``: one
  observation per executed action, exactly as deploy.py's loop. Every frame is read by some
  later query (25 frames per second against queries every 8 frames covers every residue), so
  every frame renders.
- **One act() per RoboDojo step.** Run ``roboclock eval`` with ``--policy-hz 25`` and
  ``--inference-latency zero``: each action spans two 50 Hz control steps, and observations
  are exactly 1/25 s apart. The client raises an error at any other spacing, which would
  stretch the history timeline.
- **Requests use the ``call`` message** with the batch methods ``update_obs_batch`` and
  ``get_action_batch(env_idx_list)``, which is the path RoboDojo takes for this adapter
  (``eval_batch: true`` in deploy.yml, ``eval_one_episode_batch``), here with one
  environment. The plain ``infer`` message would pair one ``update_obs`` with every
  ``get_action`` and lose the frames in between.
- **The server keeps per-episode state, keyed by ``env_idx``.** ``reset`` clears the buffers
  of every ``env_idx`` (and the VL subtask-decode cadence, which only feeds logging). By
  default the client sends ``env_idx`` 0 and calls ``reset`` at each episode start, exactly as
  deploy.py, so one server serves one episode at a time. With ``AWOMO_SHARED_SERVER=1``
  episodes may share a server: each episode sends its own random ``env_idx`` (a fresh,
  empty buffer on the server) and skips ``reset``, which would clear the other episodes'
  buffers. The server serialises the calls under its model lock, and ``get_action_batch``
  names the ``env_idx``, so episodes cannot read each other's frames. Each finished episode
  leaves its buffers (about 0.7 GB of host memory) on the server until it restarts.
- **Colours.** Observations are RGB. The client sends RGB JPEGs tagged XPL-RGB1, the server
  decodes them to RGB (``decode_obs_images``), and the adapter itself converts to the BGR this
  checkpoint was trained on (``input_color_order: bgr`` in deploy.yml; the README forbids
  client-side swaps).

Configured by environment (custom policy factories take no arguments):
  XPOLICYLAB_URL       ws://host:port of the Awomo05 XPolicyLab server
  XPOLICYLAB_COOKIE    code-server-session cookie when the URL goes through code-server
  XPOLICYLAB_LOG       optional JSONL file receiving one row per get_action
  AWOMO_FRAME_DIR      optional directory receiving the head JPEG of every get_action
  AWOMO_SHARED_SERVER  1: a random env_idx per episode and no server reset (see above)

See docs/awomo.md.
"""

import json
import os
import time
import uuid

import numpy as np

from roboclock.envs.lab.policies.xpolicylab import (
    ARM_DIMS,
    ARM_ORDER,
    CAMERAS,
    HIGH,
    LOW,
    SIDES,
    encode_jpeg,
    lab_action,
    normalise_gripper,
    robodojo_vector,
)
from roboclock.envs.lab.policies.xpolicylab import connection as xpolicylab_connection

# policy/Awomo05/deploy.yml: 16 predicted steps (num_frames 17 - 1), 8 executed per query.
REPLAN_STEPS = 8
POLICY_HZ = 25.0
HEAD_KEY = dict(CAMERAS)["cam_head"]


class AwomoPolicy:
    def __init__(self):
        self.connection = xpolicylab_connection()
        self.log_path = os.environ.get("XPOLICYLAB_LOG")
        self.frame_dir = os.environ.get("AWOMO_FRAME_DIR")
        self.shared = os.environ.get("AWOMO_SHARED_SERVER") == "1"
        self.warned_cameras = set()

    def call(self, func_name, obs=None):
        """One XPolicyLab ``call``. Returns (result, server latency in s; it includes the wait
        for the server's model lock)."""
        payload = {"func_name": func_name}
        if obs is not None:
            payload["obs"] = obs
        reply = self.connection.request("call", payload, trial_id=self.trial_id, step=self.frame)
        latency = reply.get("latency_ms")
        return reply.get("result"), None if latency is None else float(latency) / 1000.0

    def reset(self, observation):
        self.trial_id = uuid.uuid4().hex
        self.frame = -1
        self.inferences = 0
        self.queue = []
        self.previous_time = None
        self.update_seconds = []  # update_obs latencies since the last get_action
        # Read by the evaluator after act().
        self.queried = True
        self.last_inference_latency = None
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        # RoboDojo reports the last commanded opening; before any command, the measured one.
        self.gripper = normalise_gripper(joints[[8 * arm + 6 for arm in ARM_ORDER]])
        if self.shared:
            # A key no other episode uses: the server starts an empty buffer for it.
            self.env_idx = 1 + int(self.trial_id[:7], 16)
        else:
            self.env_idx = 0
            # deploy.py: clears the frame history (all env_idx) and the VL decode cadence.
            self.call("reset")

    def check_rate(self, observation):
        """Observations must be one RoboDojo step apart: the server counts them as 1/25 s."""
        now = float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0])
        if self.previous_time is not None and np.isfinite(now):
            step = now - self.previous_time
            if abs(step - 1.0 / POLICY_HZ) > 0.25 / POLICY_HZ:
                raise RuntimeError(
                    f"Awomo expects one act() per 1/{POLICY_HZ:g} s (got {step:.4f} s): run "
                    "roboclock eval with --policy-hz 25 and --inference-latency zero"
                )
        self.previous_time = now

    def act(self, observation):
        self.frame += 1
        self.check_rate(observation)
        request = self.observation(observation)  # reads the images, which renders them
        _, update_latency = self.call("update_obs_batch", [request])
        self.update_seconds.append(update_latency or 0.0)
        self.queried = not self.queue
        if self.queried:
            self.queue = self.infer(observation, request)
        else:
            self.last_inference_latency = None
        command, target = self.queue.pop(0)
        # RoboDojo reports the commanded opening after clipping it to [0, 1].
        self.gripper = np.clip(command[[6, 13]], 0.0, 1.0)
        return target

    def observation(self, observation):
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        head = observation[HEAD_KEY]
        vision = {}
        for camera, key in CAMERAS:
            image = head if key == HEAD_KEY else observation.get(key)
            if image is None:
                # The adapter needs all three views; a black frame keeps the request valid.
                if camera not in self.warned_cameras:
                    print(f"[awomo] observation has no {key!r}; sending black {camera}")
                    self.warned_cameras.add(camera)
                image = np.zeros_like(head)
            vision[camera] = {"color": encode_jpeg(image)}
        state = {}
        for arm, (side, lab_arm) in enumerate(zip(SIDES, ARM_ORDER)):
            state[f"{side}_arm_joint_state"] = joints[8 * lab_arm : 8 * lab_arm + 6]
            state[f"{side}_ee_joint_state"] = np.asarray(self.gripper[arm : arm + 1], dtype=np.float32)
        return {
            "instruction": observation["instruction"],
            "env_idx": self.env_idx,
            "vision": vision,
            "state": state,
        }

    def infer(self, observation, request):
        if self.frame_dir:
            os.makedirs(self.frame_dir, exist_ok=True)
            with open(os.path.join(self.frame_dir, f"head_{self.frame:05d}.jpg"), "wb") as stream:
                stream.write(request["vision"]["cam_head"]["color"])
        start = time.perf_counter()
        batch, latency = self.call("get_action_batch", [self.env_idx])
        round_trip = time.perf_counter() - start
        actions = batch[0] if batch else None
        if not actions:
            raise RuntimeError("Awomo05 server returned no actions; see the server log")
        self.inferences += 1
        update_latency = self.update_seconds[-1]
        self.last_inference_latency = None if latency is None else update_latency + latency
        commands = np.stack([robodojo_vector(action) for action in actions]).astype(np.float32)
        targets = lab_action(commands)
        clipped = (targets < LOW) | (targets > HIGH)
        if self.log_path:
            joints = np.asarray(observation["joint_positions"], dtype=np.float32)
            arm_path = np.clip(targets, LOW, HIGH)[:, ARM_DIMS]
            row = {
                "trial_id": self.trial_id,
                "env_idx": self.env_idx,
                "inference": self.inferences - 1,
                "frame": self.frame,
                "sim_time": float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0]),
                "chunk_actions": int(len(actions)),
                "server_ms": None if latency is None else latency * 1000.0,
                "update_ms": update_latency * 1000.0,
                # update_obs calls of the frames the previous chunk played out.
                "mean_update_ms": float(np.mean(self.update_seconds)) * 1000.0,
                "round_trip_ms": round_trip * 1000.0,
                "gripper_state": [round(float(g), 3) for g in request["state"]["left_ee_joint_state"]]
                + [round(float(g), 3) for g in request["state"]["right_ee_joint_state"]],
                "gripper_commands": [[round(float(v), 3) for v in c[[6, 13]]] for c in commands],
                "clipped_arm_targets": int(clipped[:, ARM_DIMS].sum()),
                "start_jump_rad": float(np.abs(arm_path[0] - joints[ARM_DIMS]).max()),
                "chunk_travel_rad": float(np.abs(np.diff(arm_path, axis=0)).sum(axis=0).max()),
            }
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        self.update_seconds = []
        return list(zip(commands, np.clip(targets, LOW, HIGH)))


def make():
    return AwomoPolicy()
