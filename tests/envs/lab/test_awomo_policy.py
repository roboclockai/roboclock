"""Awomo-0.5 client: one observation per RoboDojo step, a query every 8, RGB JPEGs, env_idx keys."""

import io

import numpy as np
import pytest
from PIL import Image

from roboclock.envs.lab.policies import awomo, xpolicylab
from roboclock.envs.lab.robot import HOME


class FakeServer:
    """Answers reset/update_obs_batch/get_action_batch like the Awomo05 adapter: each env_idx
    has a frame buffer that advances one entry per observation, reset clears every buffer, and
    get_action_batch returns 8 joint actions for the newest observation of each env_idx."""

    def __init__(self, chunk=awomo.REPLAN_STEPS):
        self.chunk = chunk
        self.calls = []
        self.frames = {}  # env_idx -> head images since reset, one per update_obs
        self.queries = []  # (env_idx, frame index) of each get_action
        self.latest = {}

    def request(self, message_type, payload, **_):
        assert message_type == "call"
        name = payload["func_name"]
        self.calls.append(name)
        if name == "reset":
            assert "obs" not in payload
            self.frames, self.latest = {}, {}
            return {"ok": True, "result": None, "latency_ms": 0.1}
        if name == "update_obs_batch":
            for obs in payload["obs"]:
                assert set(obs["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
                self.frames.setdefault(obs["env_idx"], []).append(obs["vision"]["cam_head"]["color"])
                self.latest[obs["env_idx"]] = obs
            return {"ok": True, "result": None, "latency_ms": 20.0}
        assert name == "get_action_batch"
        result = []
        for env_idx in payload["obs"]:
            assert env_idx in self.latest, "get_action before update_obs"
            self.queries.append((env_idx, len(self.frames[env_idx]) - 1))
            state = self.latest[env_idx]["state"]
            # Step the left joint 1 by 0.01 rad per action; grippers to -0.2 and 0.75.
            actions = []
            for step in range(self.chunk):
                action = {key: np.asarray(value, dtype=np.float32).copy() for key, value in state.items()}
                action["left_arm_joint_state"][0] += 0.01 * (step + 1)
                action["left_ee_joint_state"][:] = -0.2
                action["right_ee_joint_state"][:] = 0.75
                actions.append(action)
            result.append(actions)
        return {"ok": True, "result": result, "latency_ms": 900.0}

    @property
    def last(self):
        return self.latest[max(self.latest, key=lambda key: len(self.frames[key]))]


def observation(step, joints=None):
    head = np.zeros((48, 64, 3), dtype=np.uint8)
    head[..., 0], head[..., 1], head[..., 2] = 200, 60, step % 50
    wrist = np.full((48, 64, 3), 90, dtype=np.uint8)
    return {
        "rgb": head,
        "left_wrist_rgb": wrist,
        "right_wrist_rgb": wrist,
        "joint_positions": np.tile(HOME, 2).astype(np.float32) if joints is None else joints,
        "instruction": "stack the bowls",
        "sim_time": np.array([step / 25.0], dtype=np.float32),
    }


def make_policy(monkeypatch, server):
    monkeypatch.setattr(awomo, "xpolicylab_connection", lambda: server)
    return awomo.make()


def test_one_update_per_step_and_a_query_every_8(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    steps = 41
    for step in range(steps):
        target = policy.act(observation(step))
        assert target.shape == (16,)
        assert policy.queried == (step % 8 == 0)
        assert policy.last_inference_latency == (pytest.approx(0.92) if step % 8 == 0 else None)
    assert server.calls[0] == "reset"
    assert server.calls.count("update_obs_batch") == steps
    assert list(server.frames) == [0] and len(server.frames[0]) == steps
    # get_action always follows the update_obs of the frame it acts on.
    assert server.queries == [(0, frame) for frame in range(0, steps, 8)]
    for index, name in enumerate(server.calls):
        if name == "get_action_batch":
            assert server.calls[index - 1] == "update_obs_batch"


def test_chunk_is_played_one_action_per_step(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    targets = [policy.act(observation(step)) for step in range(8)]
    for step, target in enumerate(targets):
        np.testing.assert_allclose(target[0], HOME[0] + 0.01 * (step + 1), atol=1e-6)
        np.testing.assert_allclose(target[1:6], HOME[1:6], atol=1e-6)
        np.testing.assert_allclose(target[8:14], HOME[:6], atol=1e-6)
    # Gripper commands map [0, 1] onto the finger travel; below 0 is clipped to closed, and the
    # lab joint limit then clips RoboDojo's squeeze (-0.01 m) to 0 m.
    np.testing.assert_allclose(targets[0][6:8], max(xpolicylab.GRIPPER_SCALE[0], xpolicylab.LOW[6]), atol=1e-6)
    expected = xpolicylab.GRIPPER_SCALE[0] + 0.75 * (xpolicylab.GRIPPER_SCALE[1] - xpolicylab.GRIPPER_SCALE[0])
    np.testing.assert_allclose(targets[0][14:16], expected, atol=1e-6)


def test_state_layout_and_last_commanded_gripper(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    joints = np.tile(HOME, 2).astype(np.float32)
    joints[8:14] += 0.1  # right arm
    joints[6:8] = 0.017  # left fingers half open
    policy.reset(observation(0, joints))
    policy.act(observation(0, joints))
    first = server.last["state"]
    np.testing.assert_allclose(first["left_arm_joint_state"], HOME[:6], atol=1e-6)
    np.testing.assert_allclose(first["right_arm_joint_state"], HOME[:6] + 0.1, atol=1e-6)
    # Before any command: the measured opening.
    np.testing.assert_allclose(first["left_ee_joint_state"], [0.5], atol=1e-3)
    policy.act(observation(1, joints))
    # Afterwards: the last command, clipped to [0, 1] as RoboDojo reports it.
    np.testing.assert_allclose(server.last["state"]["left_ee_joint_state"], [0.0])
    np.testing.assert_allclose(server.last["state"]["right_ee_joint_state"], [0.75])
    assert server.last["instruction"] == "stack the bowls"


def test_images_are_rgb_jpegs_without_a_channel_swap(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    policy.act(observation(0))
    data = server.last["vision"]["cam_head"]["color"]
    assert isinstance(data, bytes) and xpolicylab.RGB_MARKER in data[:64]
    pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).reshape(-1, 3).mean(0)
    np.testing.assert_allclose(pixels, [200, 60, 0], atol=3)
    process_data = pytest.importorskip(
        "XPolicyLab.utils.process_data", reason="needs the RoboDojo checkout on PYTHONPATH"
    )
    pytest.importorskip("cv2")
    # The server's own decoder reads the tag and returns RGB; the adapter swaps to BGR itself.
    decoded = process_data.decode_image_bit(data).reshape(-1, 3).mean(0)
    np.testing.assert_allclose(decoded, [200, 60, 0], atol=3)


def test_rejects_the_50_hz_control_rate(monkeypatch):
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    policy.act(observation(0))
    obs = observation(1)
    obs["sim_time"] = np.array([0.02], dtype=np.float32)
    with pytest.raises(RuntimeError, match="--policy-hz 25"):
        policy.act(obs)


def test_reset_starts_a_new_history(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    for step in range(5):
        policy.act(observation(step))
    policy.reset(observation(0))
    policy.act(observation(0))
    # The second episode queries on its first frame, against a fresh server buffer.
    assert server.calls[-3:] == ["reset", "update_obs_batch", "get_action_batch"]
    assert len(server.frames[0]) == 1 and policy.queried


def test_shared_server_episodes_keep_separate_buffers(monkeypatch):
    monkeypatch.setenv("AWOMO_SHARED_SERVER", "1")
    server = FakeServer()
    first, second = make_policy(monkeypatch, server), make_policy(monkeypatch, server)
    first.reset(observation(0))
    for step in range(10):
        first.act(observation(step))
    second.reset(observation(0))  # starts while the first episode runs
    for step in range(10, 30):
        first.act(observation(step))
        second.act(observation(step - 10))
    assert "reset" not in server.calls
    assert first.env_idx != second.env_idx and 0 not in (first.env_idx, second.env_idx)
    assert len(server.frames[first.env_idx]) == 30 and len(server.frames[second.env_idx]) == 20
    assert [f for key, f in server.queries if key == first.env_idx] == list(range(0, 30, 8))
    assert [f for key, f in server.queries if key == second.env_idx] == list(range(0, 20, 8))
