"""OpenWAM client: RoboDojo request layout, whole-chunk replay at 25 Hz, EE chunks solved to joints."""

import numpy as np
import pytest

from roboclock.envs.lab.policies import openwam, wallx, xpolicylab
from roboclock.envs.lab.robot import HOME

SIDES = ("left", "right")


class FakeServer:
    """Answers reset/infer like XPolicyLab's OpenWAM adapter: 32 RoboDojo ee actions per request.

    By default the chunk holds the reported pose; ``path`` (per-step joints) makes it answer
    with the link6 poses of that joint path instead.
    """

    def __init__(self, path=None, gripper=0.5):
        self.requests = []
        self.resets = 0
        self.path = path
        self.gripper = gripper

    def request(self, message_type, payload, **_):
        if message_type == "reset":
            self.resets += 1
            return {"ok": True}
        assert message_type == "infer"
        obs = payload["observation"]
        self.requests.append(obs)
        actions = []
        for step in range(openwam.ACTION_CHUNK):
            action = {}
            for side in SIDES:
                if self.path is None:
                    pose = np.asarray(obs["state"][f"{side}_ee_pose"], dtype=np.float32)
                else:
                    pose = wallx.link6_pose(side, self.path[min(step, len(self.path) - 1)])
                action[f"{side}_ee_pose"] = pose
                action[f"{side}_ee_joint_state"] = np.array([self.gripper], dtype=np.float32)
            actions.append(action)
        return {"actions": actions, "latency_ms": 400.0}


class CountingObservation(dict):
    renders = 0

    def __getitem__(self, key):
        if key.endswith("rgb"):
            CountingObservation.renders += key == "rgb"
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


def observation(step, joints=None):
    image = np.full((48, 64, 3), step % 255, dtype=np.uint8)
    if joints is None:
        joints = np.tile(HOME, 2).astype(np.float32)
    return CountingObservation(
        rgb=image,
        left_wrist_rgb=image,
        right_wrist_rgb=image,
        joint_positions=np.asarray(joints, dtype=np.float32),
        instruction="stack the bowls",
        sim_time=np.array([step / 25.0], dtype=np.float32),
    )


def make_policy(monkeypatch, server):
    monkeypatch.setattr(openwam, "xpolicylab_connection", lambda: server)
    return openwam.make()


def test_policy_replays_whole_chunks_with_the_robodojo_layout(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    CountingObservation.renders = 0
    policy.reset(observation(0))
    assert server.resets == 1
    steps = 100
    for step in range(steps):
        target = policy.act(observation(step))
        assert target.shape == (16,)
        assert policy.queried == (step % 32 == 0)
        assert policy.last_inference_latency == (0.4 if step % 32 == 0 else None)
    assert len(server.requests) == 4
    # Images are read (rendered) only on request frames.
    assert CountingObservation.renders == 4
    first = server.requests[0]
    assert first["env_idx"] == 0
    assert first["instruction"] == "stack the bowls"
    assert set(first["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    for camera in first["vision"].values():
        assert xpolicylab.RGB_MARKER in camera["color"][:64]
    assert set(first["state"]) == {
        f"{side}_{part}" for side in SIDES for part in ("arm_joint_state", "ee_joint_state", "ee_pose")
    }
    np.testing.assert_allclose(first["state"]["left_ee_pose"], wallx.link6_pose("left", HOME[:6]))
    np.testing.assert_allclose(first["state"]["right_ee_pose"], wallx.link6_pose("right", HOME[:6]))
    # A static chunk holds the arms where they are.
    np.testing.assert_allclose(target[:6], HOME[:6], atol=2e-3)
    np.testing.assert_allclose(target[8:14], HOME[:6], atol=2e-3)
    # The gripper state sent next is the last commanded opening.
    np.testing.assert_allclose(server.requests[-1]["state"]["left_ee_joint_state"], [0.5])
    np.testing.assert_allclose(target[[6, 7]], xpolicylab.lab_action(np.full(14, 0.5))[[6, 7]])


def test_first_request_sends_the_measured_gripper_clipped_to_the_unit_range(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    joints = np.tile(HOME, 2).astype(np.float32)
    joints[6:8] = -0.0102  # slightly past the closed end of the finger travel
    joints[14:16] = 0.044
    policy.reset(observation(0, joints))
    policy.act(observation(0, joints))
    state = server.requests[0]["state"]
    np.testing.assert_allclose(state["left_ee_joint_state"], [0.0])
    np.testing.assert_allclose(state["right_ee_joint_state"], [1.0], atol=1e-6)


def test_chunk_poses_are_solved_to_the_joints_that_reach_them(monkeypatch):
    rng = np.random.default_rng(0)
    path = [HOME[:6] + rng.normal(0.0, 0.004, 6) * (step + 1) for step in range(openwam.ACTION_CHUNK)]
    policy = make_policy(monkeypatch, FakeServer(path=path, gripper=1.0))
    policy.reset(observation(0))
    for step in range(openwam.ACTION_CHUNK):
        target = policy.act(observation(step))
        for side, joints in (("left", target[:6]), ("right", target[8:14])):
            reached = wallx.link6_pose(side, joints)
            wanted = wallx.link6_pose(side, path[step])
            np.testing.assert_allclose(reached[:3], wanted[:3], atol=1e-3)
            # Same rotation (q and -q are the same orientation).
            assert abs(float(np.dot(reached[3:], wanted[3:]))) > np.cos(np.radians(0.5) / 2)
            np.testing.assert_allclose(joints, path[step], atol=1e-2)
    np.testing.assert_allclose(target[[6, 7, 14, 15]], xpolicylab.GRIPPER_SCALE[1], atol=1e-6)


def test_replan_shortens_the_executed_chunk(monkeypatch):
    monkeypatch.setenv("OPENWAM_REPLAN", "8")
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    policy.reset(observation(0))
    queried = []
    for step in range(40):
        policy.act(observation(step))
        queried.append(policy.queried)
    assert [step for step, q in enumerate(queried) if q] == [0, 8, 16, 24, 32]


def test_policy_rejects_the_50_hz_control_rate(monkeypatch):
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    policy.act(observation(0))  # a query: the next step may include charged latency
    policy.act(observation(1))
    obs = observation(1)
    obs["sim_time"] = np.array([1 / 25 + 0.02], dtype=np.float32)
    with pytest.raises(RuntimeError, match="--policy-hz 25"):
        policy.act(obs)


def test_request_log_rows(monkeypatch, tmp_path):
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("XPOLICYLAB_LOG", str(log))
    monkeypatch.setenv("OPENWAM_FRAME_DIR", str(tmp_path / "frames"))
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    for step in range(33):
        policy.act(observation(step))
    import json

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["frame"] for row in rows] == [0, 32]
    assert rows[0]["server_ms"] == 400.0
    assert len(rows[0]["gripper_commands"]) == openwam.ACTION_CHUNK
    assert rows[0]["ik_error_m"] < 1e-3
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["head_00000.jpg", "head_00032.jpg"]
