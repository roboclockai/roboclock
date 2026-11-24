"""VPP2 client: one recorded observation per RoboDojo step, sparse renders, EE16 chunks."""

import io
from collections import deque

import numpy as np
import pytest
from PIL import Image

from roboclock.envs.lab.policies import vpp2, wallx
from roboclock.envs.lab.robot import HOME

BUFFER = 177  # RoboDojoVPP2Policy.history_buffer_size


def select_history(history, anchor, current_step):
    """Copy of RoboDojoVPP2Policy._build_history_condition's step selection."""
    step_to_frame = dict(history)
    selected = [(0, anchor)]
    for index in range(vpp2.HISTORY_FRAMES):
        requested = current_step - (vpp2.HISTORY_FRAMES - 1 - index) * vpp2.HISTORY_STRIDE
        if requested <= 0:
            selected.append((0, anchor))
        elif requested in step_to_frame:
            selected.append((requested, step_to_frame[requested]))
        else:
            prior = [(step, frame) for step, frame in history if step <= requested]
            selected.append(prior[-1] if prior else (0, anchor))
    return selected


class FakeServer:
    """Answers like XPolicyLab's VPP2 adapter: one history per client id, every update_obs
    recorded, and requests conditioned on the anchor plus 8 frames 25 steps apart."""

    def __init__(self):
        self.histories = {}
        self.calls = []
        self.requests = []

    def request(self, message_type, payload, **_):
        assert message_type == "call"
        name = payload["func_name"]
        obs = payload.get("obs")
        self.calls.append(name)
        if name == "reset":
            assert obs is None
            return {"ok": True, "result": None, "latency_ms": 1.0}
        client = obs[vpp2.CLIENT_ID_FIELD]
        if name == "reset_client":
            self.histories.pop(client, None)
            return {"ok": True, "result": None, "latency_ms": 1.0}
        if name == "update_obs":
            assert set(obs["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
            # The test frames are uniform images whose value is the step they were rendered at.
            pixels = np.asarray(Image.open(io.BytesIO(obs["vision"]["cam_head"]["color"])))
            value = int(round(float(pixels.mean())))
            entry = self.histories.setdefault(client, {"steps": deque(maxlen=BUFFER), "step": -1})
            entry["step"] += 1
            if entry["step"] == 0:
                entry["anchor"] = value
            entry["steps"].append((entry["step"], value))
            entry["obs"] = obs
            return {"ok": True, "result": None, "latency_ms": 2.0}
        assert name == "get_action"
        entry = self.histories[client]
        now = entry["step"]
        selected = select_history(entry["steps"], entry["anchor"], now)
        for step, value in selected:
            assert value == step % 200, f"request at {now} read step {step} with a stale image ({value})"
        last = entry["obs"]
        self.requests.append((now, last, [step for step, _ in selected]))
        action = {
            f"{side}_ee_pose": last["state"][f"{side}_ee_pose"] for side in ("left", "right")
        } | {f"{side}_ee_joint_state": np.array([0.5], dtype=np.float32) for side in ("left", "right")}
        return {"ok": True, "latency_ms": 250.0, "result": [action] * vpp2.REPLAN_STEPS}


class CountingObservation(dict):
    renders = 0

    def __getitem__(self, key):
        if key.endswith("rgb"):
            CountingObservation.renders += key == "rgb"
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


def observation(step):
    image = np.full((48, 64, 3), step % 200, dtype=np.uint8)
    return CountingObservation(
        rgb=image,
        left_wrist_rgb=image,
        right_wrist_rgb=image,
        joint_positions=np.tile(HOME, 2).astype(np.float32),
        instruction="stack the bowls",
        sim_time=np.array([step / 25.0], dtype=np.float32),
    )


def make_policy(monkeypatch, server):
    monkeypatch.setattr(vpp2, "xpolicylab_connection", lambda: server)
    return vpp2.make()


def test_ee_pose_has_robodojo_quaternion_sign():
    # RoboDojo's recorded link6 quaternion at joint zero (stack_blocks, frame 0).
    np.testing.assert_allclose(vpp2.robodojo_ee_pose("left", np.zeros(6))[3:], [0.707, 0, 0, 0.7072], atol=1e-3)
    # stack_blocks frame 120 (recorded [0.4999, -0.5001, 0.4999, 0.5001]) and roboclock's HOME,
    # where the rotation-matrix quaternion has the opposite sign.
    for joints in (np.array([0.1517, 2.0399, 1.4965, -1.0275, 0.0, 0.1515]), HOME[:6]):
        pose = vpp2.robodojo_ee_pose("left", joints)
        reference = wallx.link6_pose("left", joints)
        np.testing.assert_allclose(pose[:3], reference[:3])
        np.testing.assert_allclose(pose[3:], -reference[3:], atol=1e-6)
        np.testing.assert_allclose(pose[3:], [0.5, -0.5, 0.5, 0.5], atol=2e-3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        joints = HOME[:6] + rng.uniform(-1.0, 1.0, 6)
        pose = vpp2.robodojo_ee_pose("right", joints)
        # Same rotation as the matrix FK, and the sign of the continuous joint composition.
        assert abs(float(np.dot(pose[3:], wallx.link6_pose("right", joints)[3:]))) == pytest.approx(1.0, abs=1e-5)
        assert float(np.dot(pose[3:], vpp2.chain_quaternion(joints))) > 0


def test_history_steps_follow_the_server_rule():
    assert vpp2.history_steps(0) == [0] * 9
    assert vpp2.history_steps(48) == [0, 0, 0, 0, 0, 0, 0, 23, 48]
    assert vpp2.history_steps(240) == [0, 65, 90, 115, 140, 165, 190, 215, 240]


def test_needed_frames_cover_every_request_and_are_a_third():
    read = {step for request in range(0, 2400, 24) for step in vpp2.history_steps(request)}
    needed = {step for step in range(2400) if vpp2.frame_needed(step)}
    assert read <= needed
    assert len(needed) == pytest.approx(2400 / 3, abs=2)


def test_policy_records_every_step_and_requests_every_24(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    CountingObservation.renders = 0
    policy.reset(observation(0))
    assert server.calls == ["reset", "reset_client"]
    steps = 250
    for step in range(steps):
        target = policy.act(observation(step))
        assert target.shape == (16,)
        assert policy.queried == (step % 24 == 0)
        # The charged time is the server's update_obs plus get_action.
        assert policy.last_inference_latency == (pytest.approx(0.252) if step % 24 == 0 else None)
    assert server.calls.count("update_obs") == steps
    assert [now for now, _, _ in server.requests] == list(range(0, steps, 24))
    # Late requests read the anchor and eight distinct real frames.
    assert server.requests[-1][2] == [0, 65, 90, 115, 140, 165, 190, 215, 240]
    first = server.requests[0][1]
    assert first["env_idx"] == 0 and first["instruction"] == "stack the bowls"
    np.testing.assert_allclose(first["state"]["left_ee_pose"], vpp2.robodojo_ee_pose("left", HOME[:6]))
    np.testing.assert_allclose(first["state"]["right_ee_pose"], vpp2.robodojo_ee_pose("right", HOME[:6]))
    # A static chunk holds the arms where they are.
    np.testing.assert_allclose(target[:6], HOME[:6], atol=2e-3)
    np.testing.assert_allclose(target[8:14], HOME[:6], atol=2e-3)
    # The gripper sent next is the last commanded opening.
    np.testing.assert_allclose(server.requests[-1][1]["state"]["left_ee_joint_state"], [0.5])
    # A third of the steps are rendered.
    assert CountingObservation.renders == policy.rendered_frames
    assert CountingObservation.renders < 0.4 * steps


def test_two_clients_share_one_server(monkeypatch):
    server = FakeServer()
    first, second = make_policy(monkeypatch, server), make_policy(monkeypatch, server)
    first.reset(observation(0))
    for step in range(30):
        first.act(observation(step))
    second.reset(observation(0))
    for step in range(60):
        first.act(observation(30 + step))
        second.act(observation(step))
    assert first.client_id != second.client_id
    assert server.histories[first.client_id]["step"] == 89
    assert server.histories[second.client_id]["step"] == 59


def test_a_skipped_frame_the_server_reads_is_an_error(monkeypatch):
    monkeypatch.setattr(vpp2, "frame_needed", lambda step, replan=24: step % replan == 0)
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    with pytest.raises(RuntimeError, match="unrendered steps"):
        for step in range(49):
            policy.act(observation(step))


def test_policy_rejects_the_50_hz_control_rate(monkeypatch):
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    policy.act(observation(0))  # a query: the next step may include charged latency
    policy.act(observation(1))
    obs = observation(2)
    obs["sim_time"] = np.array([1 / 25 + 0.02], dtype=np.float32)
    with pytest.raises(RuntimeError, match="--policy-hz 25"):
        policy.act(obs)


def test_record_all_frames_renders_every_step(monkeypatch):
    monkeypatch.setenv("VPP2_RECORD_ALL_FRAMES", "1")
    policy = make_policy(monkeypatch, FakeServer())
    CountingObservation.renders = 0
    policy.reset(observation(0))
    for step in range(40):
        policy.act(observation(step))
    assert CountingObservation.renders == 40
