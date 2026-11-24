"""X-VLA client: RoboDojo request layout, whole-chunk replay at 25 Hz, EE chunks solved to joints."""

import numpy as np
import pytest

from roboclock.envs.lab.policies import wallx, xvla
from roboclock.envs.lab.robot import HOME


class FakeServer:
    """Answers like the X_VLA adapter: call reset, then infer -> 30 RoboDojo EE action dicts."""

    def __init__(self, drift=0.0):
        self.messages = []
        self.requests = []
        self.drift = drift

    def request(self, message_type, payload, **_):
        self.messages.append((message_type, payload.get("func_name")))
        if message_type == "call":
            assert payload == {"func_name": "reset"}
            return {"ok": True, "result": None, "latency_ms": 0.1}
        assert message_type == "infer"
        obs = payload["observation"]
        self.requests.append(obs)
        actions = []
        for step in range(xvla.ACTION_CHUNK):
            action = {}
            for side in ("left", "right"):
                pose = np.asarray(obs["state"][f"{side}_ee_pose"], dtype=np.float32).copy()
                pose[2] += self.drift * (step + 1)  # rises drift metres per 25 Hz step
                action[f"{side}_ee_pose"] = pose
                action[f"{side}_ee_joint_state"] = np.array([0.5], dtype=np.float32)
            actions.append(action)
        return {"actions": actions, "latency_ms": 180.0}


class CountingObservation(dict):
    renders = 0

    def __getitem__(self, key):
        if key.endswith("rgb"):
            CountingObservation.renders += 1
        return super().__getitem__(key)


def observation(step, joints=None):
    image = np.full((48, 64, 3), step % 255, dtype=np.uint8)
    return CountingObservation(
        rgb=image,
        left_wrist_rgb=image,
        right_wrist_rgb=image,
        joint_positions=np.tile(HOME, 2).astype(np.float32) if joints is None else joints,
        instruction="stack the bowls",
        sim_time=np.array([step / 25.0], dtype=np.float32),
    )


def make_policy(monkeypatch, server):
    monkeypatch.setattr(xvla, "xpolicylab_connection", lambda: server)
    return xvla.make()


def test_proprio_matches_the_adapter_rot6d():
    # Identity rotation: first two matrix columns, row-major -> [1, 0, 0, 1, 0, 0].
    np.testing.assert_allclose(xvla.quat_to_rotate6d([1.0, 0.0, 0.0, 0.0]), [1, 0, 0, 1, 0, 0], atol=1e-7)
    # +90 deg about z: columns (0, 1, 0) and (-1, 0, 0).
    s = np.sqrt(0.5)
    np.testing.assert_allclose(xvla.quat_to_rotate6d([s, 0.0, 0.0, s]), [0, -1, 1, 0, 0, 0], atol=1e-7)
    state = {
        "left_ee_pose": wallx.link6_pose("left", HOME[:6]),
        "right_ee_pose": wallx.link6_pose("right", HOME[:6]),
        "left_ee_joint_state": np.array([0.25]),
        "right_ee_joint_state": np.array([1.0]),
    }
    vector = xvla.proprio(state)
    assert vector.shape == (20,)
    np.testing.assert_allclose(vector[[9, 19]], [0.25, 1.0])
    np.testing.assert_allclose(vector[:3], state["left_ee_pose"][:3])


def test_policy_replays_whole_chunks_with_the_robodojo_layout(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    CountingObservation.renders = 0
    policy.reset(observation(0))
    assert server.messages == [("call", "reset")]
    steps = 95
    for step in range(steps):
        target = policy.act(observation(step))
        assert target.shape == (16,)
        assert policy.queried == (step % 30 == 0)
        assert policy.last_inference_latency == (0.18 if step % 30 == 0 else None)
    assert len(server.requests) == 4
    first = server.requests[0]
    assert set(first["vision"]) == {"cam_head"}
    assert first["vision"]["cam_head"]["color"][:2] == b"\xff\xd8"
    assert first["env_idx"] == 0
    np.testing.assert_allclose(first["state"]["left_ee_pose"], wallx.link6_pose("left", HOME[:6]))
    np.testing.assert_allclose(first["state"]["right_ee_pose"], wallx.link6_pose("right", HOME[:6]))
    # A static chunk holds the arms where they are.
    np.testing.assert_allclose(target[:6], HOME[:6], atol=2e-3)
    np.testing.assert_allclose(target[8:14], HOME[:6], atol=2e-3)
    # The gripper sent next is the last commanded opening; before that, the measured one.
    np.testing.assert_allclose(
        first["state"]["left_ee_joint_state"], xvla.normalise_gripper([HOME[6]]), atol=1e-6
    )
    np.testing.assert_allclose(server.requests[-1]["state"]["left_ee_joint_state"], [0.5])
    # Images are read on request frames only.
    assert CountingObservation.renders == 4


def test_chunk_entries_are_25_hz_steps_solved_from_the_measured_joints(monkeypatch):
    server = FakeServer(drift=0.002)
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    targets = [policy.act(observation(step)) for step in range(3)]
    # Entry k lifts link6 by 2 mm x (k + 1); solve the targets back to poses.
    start = wallx.link6_pose("left", HOME[:6])
    for k, target in enumerate(targets):
        pose = wallx.link6_pose("left", target[:6])
        np.testing.assert_allclose(pose[:3] - start[:3], [0, 0, 0.002 * (k + 1)], atol=3e-4)


def test_replan_override(monkeypatch):
    monkeypatch.setenv("XVLA_REPLAN", "10")
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    for step in range(25):
        policy.act(observation(step))
    assert len(server.requests) == 3


def test_policy_rejects_the_50_hz_control_rate(monkeypatch):
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    policy.act(observation(0))  # a query: the next step may include charged latency
    policy.act(observation(1))
    obs = observation(1)
    obs["sim_time"] = np.array([1 / 25 + 0.02], dtype=np.float32)
    with pytest.raises(RuntimeError, match="--policy-hz 25"):
        policy.act(obs)
