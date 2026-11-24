"""Observation/action mapping of the remote XPolicyLab (RoboDojo pi0.5) policy client."""

import io

import numpy as np
import pytest
from PIL import Image

from roboclock.envs.lab.policies import xpolicylab
from roboclock.envs.lab.robot import HOME, JOINT_HIGH, JOINT_LOW


class FakeConnection:
    """Answers infer with a fixed-length chunk that holds the observed state."""

    def __init__(self, chunk=50):
        self.chunk = chunk
        self.requests = []

    def request(self, message_type, payload, **_):
        self.requests.append((message_type, payload))
        if message_type != "infer":
            return {"ok": True}
        state = payload["observation"]["state"]
        action = {key: np.asarray(value, dtype=np.float32) for key, value in state.items()}
        return {"actions": [action] * self.chunk, "latency_ms": 12.0}


def observation(joints=None):
    return {
        "rgb": np.full((72, 96, 3), (200, 40, 10), dtype=np.uint8),
        "joint_positions": np.tile(HOME, 2).astype(np.float32) if joints is None else joints,
        "instruction": "stack the bowls",
        "sim_time": np.array([0.0], dtype=np.float32),
    }


def make_policy(monkeypatch, connection):
    monkeypatch.setattr(xpolicylab, "connection", lambda: connection)
    return xpolicylab.make()


def test_jpeg_carries_the_rgb_marker_after_jfif_and_keeps_channel_order():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 0] = 255
    data = xpolicylab.encode_jpeg(image)
    app0_end = 4 + int.from_bytes(data[4:6], "big")
    assert data[2:4] == b"\xff\xe0"
    assert data[app0_end : app0_end + 4] == b"\xff\xfe\x00\x0a"
    assert data[app0_end + 4 : app0_end + 12] == xpolicylab.RGB_MARKER
    decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    assert decoded[..., 0].mean() > 240 and decoded[..., 2].mean() < 15


def test_lab_action_orders_arms_and_maps_robodojo_gripper_scale():
    vector = np.r_[np.arange(6) * 0.1, 1.0, -np.arange(6) * 0.1, 0.0].astype(np.float32)
    action = xpolicylab.lab_action(vector)
    assert action.shape == (16,)
    # The lab interface lists the rig's left arm (+y) first, as RoboDojo: a pass-through.
    np.testing.assert_allclose(action[:6], vector[:6])
    np.testing.assert_allclose(action[8:14], vector[7:13])
    np.testing.assert_allclose(action[6:8], [0.044, 0.044], atol=1e-6)
    # Normalised 0 is RoboDojo's squeeze below the physical 0 m stop.
    np.testing.assert_allclose(action[14:16], [-0.01, -0.01], atol=1e-6)
    np.testing.assert_allclose(
        xpolicylab.normalise_gripper([-0.01, 0.044]), [0.0, 1.0], atol=1e-6
    )


def test_resample_keeps_the_chunk_duration_on_the_control_clock():
    chunk = np.linspace(0.0, 49.0, 50)[:, None]
    steps = xpolicylab.resample(chunk, 25.0, 30.0)
    assert len(steps) == 60
    assert steps[0, 0] == 0.0
    # Control step 30 is 1 s in, which is policy action 25.
    assert steps[30, 0] == pytest.approx(25.0)
    assert steps[-1, 0] == pytest.approx(49.0)


def test_policy_sends_robodojo_layout_and_holds_a_static_chunk(monkeypatch):
    connection = FakeConnection()
    policy = make_policy(monkeypatch, connection)
    obs = observation()
    policy.reset(obs)
    first = policy.act(obs)

    assert [kind for kind, _ in connection.requests] == ["reset", "infer"]
    sent = connection.requests[1][1]["observation"]
    assert set(sent["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    assert all(isinstance(camera["color"], bytes) for camera in sent["vision"].values())
    assert {key: np.shape(value) for key, value in sent["state"].items()} == {
        "left_arm_joint_state": (6,),
        "left_ee_joint_state": (1,),
        "right_arm_joint_state": (6,),
        "right_ee_joint_state": (1,),
    }
    np.testing.assert_allclose(sent["state"]["left_arm_joint_state"], obs["joint_positions"][:6])
    np.testing.assert_allclose(sent["state"]["right_arm_joint_state"], obs["joint_positions"][8:14])

    # A chunk holding the state reproduces the observed joints within lab limits.
    np.testing.assert_allclose(first, obs["joint_positions"], atol=1e-6)
    assert np.all(first >= np.tile(JOINT_LOW, 2)) and np.all(first <= np.tile(JOINT_HIGH, 2))
    for _ in range(59):
        policy.act(obs)
    assert [kind for kind, _ in connection.requests] == ["reset", "infer"]
    policy.act(obs)
    assert [kind for kind, _ in connection.requests][-1] == "infer"


def test_policy_reports_queries_and_the_server_compute_time(monkeypatch):
    connection = FakeConnection(chunk=2)  # 2 actions at 25 Hz span 4 control steps at 50 Hz
    policy = make_policy(monkeypatch, connection)
    obs = observation()
    policy.reset(obs)
    queried = []
    for _ in range(5):
        policy.act(obs)
        queried.append(policy.queried)
    assert queried == [True, False, False, False, True]
    # The evaluator charges the server's own 12 ms, not the round trip.
    assert policy.last_inference_latency == pytest.approx(0.012)


def test_policy_without_server_timing_leaves_the_round_trip_to_the_evaluator(monkeypatch):
    connection = FakeConnection(chunk=2)
    request = connection.request
    connection.request = lambda *a, **k: {
        key: value for key, value in request(*a, **k).items() if key != "latency_ms"
    }
    policy = make_policy(monkeypatch, connection)
    policy.reset(observation())
    policy.act(observation())
    assert policy.queried and policy.last_inference_latency is None


def test_gripper_state_reports_the_last_command(monkeypatch):
    connection = FakeConnection(chunk=2)
    policy = make_policy(monkeypatch, connection)
    obs = observation()
    policy.reset(obs)
    policy.gripper = np.array([0.25, 0.75], dtype=np.float32)
    policy.act(obs)
    sent = connection.requests[-1][1]["observation"]["state"]
    assert sent["left_ee_joint_state"][0] == pytest.approx(0.25)
    assert sent["right_ee_joint_state"][0] == pytest.approx(0.75)
    np.testing.assert_allclose(policy.gripper, [0.25, 0.75])


def test_replan_limits_the_executed_part_of_the_chunk(monkeypatch):
    monkeypatch.setenv("XPOLICYLAB_REPLAN", "10")
    connection = FakeConnection()
    policy = make_policy(monkeypatch, connection)
    obs = observation()
    policy.reset(obs)
    for _ in range(20):  # 10 actions at 25 Hz span 20 control steps at 50 Hz
        policy.act(obs)
    assert [kind for kind, _ in connection.requests] == ["reset", "infer"]
    policy.act(obs)
    assert [kind for kind, _ in connection.requests] == ["reset", "infer", "infer"]


class RampConnection(FakeConnection):
    """Answers infer with a chunk whose gripper openings ramp 0.0, 0.1, 0.2, ..."""

    def request(self, message_type, payload, **_):
        reply = super().request(message_type, payload)
        if message_type == "infer":
            reply["actions"] = [
                {**action, "left_ee_joint_state": np.float32([i / 10]), "right_ee_joint_state": np.float32([1 - i / 10])}
                for i, action in enumerate(reply["actions"])
            ]
        return reply


def test_streaming_chunk_and_the_gripper_state_of_the_action_in_effect(monkeypatch):
    connection = RampConnection(chunk=5)
    policy = make_policy(monkeypatch, connection)
    obs = observation()
    policy.reset(obs)
    chunk = policy.act_chunk(obs)
    # 5 actions at 25 Hz span 10 control steps at 50 Hz, clipped to the lab joint limits.
    assert chunk.shape == (10, 16) and chunk.dtype == np.float32
    assert np.all(chunk >= np.tile(JOINT_LOW, 2)) and np.all(chunk <= np.tile(JOINT_HIGH, 2))
    assert policy.last_inference_latency == pytest.approx(0.012)
    policy.on_action(0, 4)  # control step 4 = chunk entry 2 (opening 0.2 / 0.8)
    np.testing.assert_allclose(policy.gripper, [0.2, 0.8], atol=1e-6)
    policy.act_chunk(obs)
    sent = connection.requests[-1][1]["observation"]["state"]
    assert sent["left_ee_joint_state"][0] == pytest.approx(0.2)
    assert sent["right_ee_joint_state"][0] == pytest.approx(0.8)
    policy.on_action(1, 0)
    np.testing.assert_allclose(policy.gripper, [0.0, 1.0], atol=1e-6)
    assert policy.chunk_commands[0] is None  # superseded chunks are released


def load_replay_check():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts/xpolicylab/replay_check.py"
    spec = importlib.util.spec_from_file_location("replay_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_check_packs_a_demo_frame_like_robodojo(monkeypatch):
    """The roboclock path of scripts/xpolicylab/replay_check.py sends the stored RoboDojo state."""
    replay = load_replay_check()
    state = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.25, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.75]],
        dtype=np.float32,
    )
    episode = {
        "instruction": "stack the blocks",
        "hz": 25,
        "state": state,
        "jpeg": {camera: [b"jpeg"] for camera, _ in xpolicylab.CAMERAS},
        "decode": lambda data: np.zeros((480, 640, 3), dtype=np.uint8),
    }
    native = replay.native_request(episode, 0)
    connection = FakeConnection()
    policy = make_policy(monkeypatch, connection)
    observation = replay.roboclock_observation(episode, 0)
    policy.reset(observation)
    ours = policy.observation(observation)
    assert set(ours["state"]) == set(native["state"])
    for key, value in native["state"].items():
        np.testing.assert_allclose(ours["state"][key], value, atol=1e-6, err_msg=key)
    assert ours["instruction"] == native["instruction"]
    assert set(ours["vision"]) == set(native["vision"])
