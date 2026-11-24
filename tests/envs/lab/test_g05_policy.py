"""G0.5 client: RoboDojo request layout, 16-entry chunks at 25 Hz, and the commanded-gripper state."""

import numpy as np
import pytest

from roboclock.envs.lab.policies import g05, xpolicylab
from roboclock.envs.lab.robot import HOME


def chunk(observation, steps=g05.ACTION_STEPS, gripper=(0.5, 1.3)):
    """What policy/G05 returns: unpack_robot_state dicts, the arms moving 0.01 rad per step."""
    state = observation["state"]
    return [
        {
            "left_arm_joint_state": state["left_arm_joint_state"] + 0.01 * (t + 1),
            "left_ee_joint_state": np.array([gripper[0]], dtype=np.float32),
            "right_arm_joint_state": state["right_arm_joint_state"] - 0.01 * (t + 1),
            "right_ee_joint_state": np.array([gripper[1]], dtype=np.float32),
        }
        for t in range(steps)
    ]


class FakeServer:
    """Answers reset/infer like the G0.5 XPolicyLab server (update_obs + get_action)."""

    def __init__(self, steps=g05.ACTION_STEPS):
        self.steps = steps
        self.requests = []
        self.resets = 0

    def request(self, message_type, payload, **_):
        if message_type == "reset":
            self.resets += 1
            return {"ok": True}
        assert message_type == "infer"
        observation = payload["observation"]
        self.requests.append(observation)
        return {"actions": chunk(observation, self.steps), "latency_ms": 180.0}


class CountingObservation(dict):
    renders = 0

    def __getitem__(self, key):
        CountingObservation.renders += key == "rgb"
        return super().__getitem__(key)


def observation(step, gripper_width=0.03):
    image = np.full((48, 64, 3), step % 255, dtype=np.uint8)
    joints = np.tile(HOME, 2).astype(np.float32)
    joints[[6, 7, 14, 15]] = gripper_width
    return CountingObservation(
        rgb=image,
        left_wrist_rgb=image,
        right_wrist_rgb=image,
        joint_positions=joints,
        instruction="stack the bowls",
        sim_time=np.array([step / 25.0], dtype=np.float32),
    )


def make_policy(monkeypatch, server):
    monkeypatch.setattr(g05, "xpolicylab_connection", lambda: server)
    return g05.make()


def test_policy_requests_every_16_steps_with_the_robodojo_layout(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    CountingObservation.renders = 0
    policy.reset(observation(0))
    assert server.resets == 1
    steps = 64
    for step in range(steps):
        target = policy.act(observation(step))
        assert target.shape == (16,)
        assert policy.queried == (step % 16 == 0)
        assert policy.last_inference_latency == (0.18 if step % 16 == 0 else None)
    assert len(server.requests) == steps // 16
    # Images are rendered only on query frames.
    assert CountingObservation.renders == steps // 16
    first = server.requests[0]
    assert set(first["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    assert all(xpolicylab.RGB_MARKER in view["color"] for view in first["vision"].values())
    assert first["additional_info"] == {"frequency": 25.0}
    assert first["env_idx"] == 0 and first["instruction"] == "stack the bowls"
    np.testing.assert_allclose(first["state"]["left_arm_joint_state"], HOME[:6])
    np.testing.assert_allclose(first["state"]["right_arm_joint_state"], HOME[:6])
    # Before any command the state gripper is the measured opening.
    measured = xpolicylab.normalise_gripper(0.03)
    np.testing.assert_allclose(first["state"]["left_ee_joint_state"], [measured], rtol=1e-6)
    # Afterwards it is the last commanded opening, clipped to [0, 1] as RoboDojo reports it.
    np.testing.assert_allclose(server.requests[1]["state"]["left_ee_joint_state"], [0.5])
    np.testing.assert_allclose(server.requests[1]["state"]["right_ee_joint_state"], [1.0])


def test_chunk_entries_map_to_lab_targets_in_order(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    targets = np.stack([policy.act(observation(step)) for step in range(16)])
    # Entry t is the t-th RoboDojo step: left arm (lab arm 0) +0.01 (t + 1), right arm -0.01 (t + 1).
    expected_left = HOME[:6] + 0.01 * np.arange(1, 17)[:, None]
    expected_right = HOME[:6] - 0.01 * np.arange(1, 17)[:, None]
    np.testing.assert_allclose(targets[:, :6], np.clip(expected_left, g05.LOW[:6], g05.HIGH[:6]), atol=1e-6)
    np.testing.assert_allclose(targets[:, 8:14], np.clip(expected_right, g05.LOW[8:14], g05.HIGH[8:14]), atol=1e-6)
    # Gripper openings become finger positions over RoboDojo's gripper_scale.
    low, high = xpolicylab.GRIPPER_SCALE
    np.testing.assert_allclose(targets[:, 6:8], low + 0.5 * (high - low), atol=1e-6)
    np.testing.assert_allclose(targets[:, 14:16], high, atol=1e-6)


def test_replan_executes_a_prefix_of_the_chunk(monkeypatch):
    monkeypatch.setenv("G05_REPLAN", "8")
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    for step in range(32):
        policy.act(observation(step))
        assert policy.queried == (step % 8 == 0)
    assert len(server.requests) == 4


def test_policy_rejects_a_short_chunk(monkeypatch):
    policy = make_policy(monkeypatch, FakeServer(steps=32))
    policy.reset(observation(0))
    with pytest.raises(ValueError, match="expected 16"):
        policy.act(observation(0))


def test_policy_rejects_the_50_hz_control_rate(monkeypatch):
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    policy.act(observation(0))  # a query: the next step may include charged latency
    policy.act(observation(1))
    obs = observation(1)
    obs["sim_time"] = np.array([1 / 25 + 0.02], dtype=np.float32)
    with pytest.raises(RuntimeError, match="--policy-hz 25"):
        policy.act(obs)
