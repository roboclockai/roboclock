"""InternW0 client: RoboDojo's update_obs_batch / get_action_batch cadence and the frames read."""

import io

import numpy as np
import pytest

from roboclock.envs.lab.policies import internw0, xpolicylab
from roboclock.envs.lab.robot import HOME


def decode(data):
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


class Session:
    def __init__(self):
        self.pending = []
        self.step = 0
        self.frames = {}  # session step -> (image tag, state)
        self.last = None


class FakeServer:
    """The adapter's session rules (StatefulWAMAdapter + RobotWinPolicySession): one session
    per env_idx; update_obs_batch acknowledges one pending action and records the new step;
    get_action_batch refuses while actions are pending and reads the current, first and
    32-steps-earlier frames."""

    def __init__(self):
        self.sessions = {}
        self.resets = 0
        self.updates = 0
        self.requests = []  # (step, frames read)
        self.messages = []

    def request(self, message_type, payload, **_):
        self.messages.append((message_type, payload.get("func_name")))
        if message_type == "reset":
            self.resets += 1
            self.sessions.clear()
            return {"ok": True}
        assert message_type == "call"
        name = payload["func_name"]
        if name == "update_obs_batch":
            for obs in payload["obs"]:
                session = self.sessions.setdefault(obs["env_idx"], Session())
                tag = int(round(decode(obs["vision"]["cam_head"]["color"]).mean()))
                for camera in ("cam_left_wrist", "cam_right_wrist"):
                    assert int(round(decode(obs["vision"][camera]["color"]).mean())) == tag
                state = np.concatenate(
                    [
                        np.asarray(obs["state"][f"{side}_{part}_joint_state"]).reshape(-1)
                        for side in ("left", "right")
                        for part in ("arm", "ee")
                    ]
                )
                if session.pending:
                    session.pending.pop(0)
                    session.step += 1
                    session.frames[session.step] = (tag, state)
                session.last = (tag, state)
            self.updates += 1
            return {"ok": True, "result": None, "latency_ms": 20.0}
        assert name == "get_action_batch"
        assert payload["obs"] == [0]
        session = self.sessions[0]
        assert not session.pending, "a replan was requested before all actions were acknowledged"
        session.frames.setdefault(session.step, session.last)
        now = session.step
        read = {"current": session.last[0], "anchor": session.frames[0][0]}
        if now >= internw0.ACTION_HORIZON:
            read["recent"] = session.frames[now - internw0.ACTION_HORIZON][0]
        self.requests.append((now, read, session.last[1]))
        state = session.last[1]
        action = {
            "left_arm_joint_state": state[0:6],
            "left_ee_joint_state": np.array([0.5], dtype=np.float32),
            "right_arm_joint_state": state[7:13],
            "right_ee_joint_state": np.array([0.25], dtype=np.float32),
        }
        session.pending = [action] * internw0.REPLAN_STEPS
        return {"ok": True, "result": [list(session.pending)], "latency_ms": 900.0}


class CountingObservation(dict):
    renders = 0

    def __getitem__(self, key):
        if key.endswith("rgb"):
            CountingObservation.renders += key == "rgb"
        return super().__getitem__(key)


def tag(step):
    return (step % 50) * 5


def observation(step):
    image = np.full((48, 64, 3), tag(step), dtype=np.uint8)
    joints = np.tile(HOME, 2).astype(np.float32)
    joints[0] += 0.001 * step
    return CountingObservation(
        rgb=image,
        left_wrist_rgb=image,
        right_wrist_rgb=image,
        joint_positions=joints,
        instruction="stack the bowls",
        sim_time=np.array([step / 25.0], dtype=np.float32),
    )


def make_policy(monkeypatch, server):
    monkeypatch.setattr(internw0, "xpolicylab_connection", lambda: server)
    return internw0.make()


def test_frames_needed_are_requests_and_recent_frames():
    needed = [step for step in range(60) if internw0.frame_needed(step)]
    assert needed == [0, 8, 10, 18, 20, 28, 30, 38, 40, 48, 50, 58]


def test_policy_follows_the_robodojo_cadence(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    CountingObservation.renders = 0
    policy.reset(observation(0))
    assert server.messages == [("reset", None)]
    steps = 205
    for step in range(steps):
        target = policy.act(observation(step))
        assert target.shape == (16,)
        assert policy.queried == (step % 10 == 0)
        assert policy.last_inference_latency == (pytest.approx(0.92) if step % 10 == 0 else None)
    # One update per step, a request every 10 steps right after that step's update.
    assert server.updates == steps
    assert [now for now, _, _ in server.requests] == list(range(0, steps, 10))
    assert server.messages[1:4] == [
        ("call", "update_obs_batch"),
        ("call", "get_action_batch"),
        ("call", "update_obs_batch"),
    ]
    # Every frame the server reads is the real image of its step.
    for now, read, state in server.requests:
        assert read["current"] == tag(now)
        assert read["anchor"] == tag(0)
        if now >= 32:
            assert read["recent"] == tag(now - 32)
        # The state is always current; the gripper is the last command (0.5 / 0.25).
        np.testing.assert_allclose(state[0], HOME[0] + 0.001 * now, atol=1e-6)
        if now:
            np.testing.assert_allclose(state[[6, 13]], [0.5, 0.25])
    first_state = server.requests[0][2]
    np.testing.assert_allclose(
        first_state[[6, 13]], xpolicylab.normalise_gripper([HOME[6], HOME[6]]), atol=1e-6
    )
    # Holding chunk: the targets are the measured joints; grippers decode to finger travel.
    np.testing.assert_allclose(target[1:6], HOME[1:6], atol=1e-6)
    np.testing.assert_allclose(target[6], -0.01 + 0.5 * 0.054, atol=1e-6)
    # Only the frames the server reads are rendered: 2 of every 10.
    assert CountingObservation.renders == len([s for s in range(steps) if internw0.frame_needed(s)])
    assert CountingObservation.renders < 0.25 * steps


def test_a_new_episode_resets_the_session(monkeypatch):
    server = FakeServer()
    for _ in range(2):
        policy = make_policy(monkeypatch, server)
        policy.reset(observation(0))
        for step in range(25):
            policy.act(observation(step))
    assert server.resets == 2
    assert [now for now, _, _ in server.requests] == [0, 10, 20, 0, 10, 20]


def test_policy_rejects_the_50_hz_control_rate(monkeypatch):
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    policy.act(observation(0))  # a query: the next step may include charged latency
    policy.act(observation(1))
    obs = observation(1)
    obs["sim_time"] = np.array([1 / 25 + 0.02], dtype=np.float32)
    with pytest.raises(RuntimeError, match="--policy-hz 25"):
        policy.act(obs)


def test_render_all_renders_every_step(monkeypatch):
    monkeypatch.setenv("INTERNW0_RENDER_ALL", "1")
    policy = make_policy(monkeypatch, FakeServer())
    CountingObservation.renders = 0
    policy.reset(observation(0))
    for step in range(40):
        policy.act(observation(step))
    assert CountingObservation.renders == 40


def test_request_log_has_one_row_per_request(monkeypatch, tmp_path):
    import json

    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("XPOLICYLAB_LOG", str(log))
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    for step in range(30):
        policy.act(observation(step))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 10, 20]
    assert rows[1]["server_ms"] == 900.0 and rows[1]["update_ms"] == 20.0
    assert rows[1]["clipped_joints"] == []
