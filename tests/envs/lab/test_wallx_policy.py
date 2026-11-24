"""Wall-X client: RoboDojo frames, EE chunks solved to joints, and the keyframe-memory requests."""

import numpy as np
import pytest

memory_client = pytest.importorskip(
    "XPolicyLab.policy.WallX.memory_client",
    reason="needs the RoboDojo checkout with XPolicyLab/policy/WallX on PYTHONPATH",
)

from roboclock.envs.lab.policies import wallx  # noqa: E402
from roboclock.envs.lab.robot import HOME  # noqa: E402

PLAN = {
    "span": 60,
    "window": 48,
    "num_candidates": 16,
    "select_stride": 48,
    "short_term": [0, 1],
    "selection": True,
}


def test_link6_pose_is_in_the_robodojo_environment_frame():
    # RoboDojo episodes start at joint zero; its recorded left/right link6 positions there.
    left = wallx.link6_pose("left", np.zeros(6))
    right = wallx.link6_pose("right", np.zeros(6))
    np.testing.assert_allclose(left[:3], [-0.299, -0.352, 0.921], atol=2e-3)
    np.testing.assert_allclose(right[:3], [0.301, -0.352, 0.921], atol=2e-3)
    assert np.linalg.norm(left[3:]) == pytest.approx(1.0, abs=1e-5)


def test_solve_chunk_recovers_the_joints_of_robodojo_poses():
    rng = np.random.default_rng(0)
    path = [HOME[:6] + rng.normal(0.0, 0.05, 6) * step for step in range(1, 6)]
    actions = [
        {
            "left_ee_pose": wallx.link6_pose("left", joints),
            "right_ee_pose": wallx.link6_pose("right", joints),
            "left_ee_joint_state": np.array([0.25], dtype=np.float32),
            "right_ee_joint_state": np.array([1.0], dtype=np.float32),
        }
        for joints in path
    ]
    vectors, worst = wallx.solve_chunk(actions, [HOME[:6], HOME[:6]])
    assert vectors.shape == (5, 14)
    # Damped least squares stops within a millimetre.
    assert worst < 1e-3
    for row, joints in zip(vectors, path):
        np.testing.assert_allclose(row[:6], joints, atol=2e-3)
        np.testing.assert_allclose(row[7:13], joints, atol=2e-3)
    np.testing.assert_allclose(vectors[:, [6, 13]], [[0.25, 1.0]] * 5)


def server_reads(plan, requests, replan):
    """Frames each request reads under the server's rule, from the first request on."""
    reads, last_select = {}, 0
    for now in range(0, requests * replan, replan):
        reads[now] = memory_client.needed_ring_frames(plan, now, last_select)
        if now - last_select >= plan["select_stride"]:
            last_select = now
    return reads


@pytest.mark.parametrize("replan", [16, 8, 12])
def test_planned_frames_cover_every_frame_the_server_reads(replan):
    reads = server_reads(PLAN, 40, replan)
    recorded, last_select = {0}, 0
    for now in sorted(reads):
        assert set(reads[now]) <= recorded, f"request at {now} reads unrecorded frames"
        if now - last_select >= PLAN["select_stride"]:
            last_select = now
        recorded |= wallx.planned_frames(PLAN, now, last_select, replan, memory_client.needed_ring_frames)
        recorded.add(now + replan)
    total = max(reads) + replan
    # Far fewer renders than one per frame.
    assert len([f for f in recorded if f < total]) < 0.5 * total


class FakeServer:
    """Answers reset/infer like the Wall-X server: memory state per the server's rules."""

    def __init__(self):
        self.requests = []

    def request(self, message_type, payload, **_):
        assert message_type == "call"
        name = payload["func_name"]
        if name == "reset":
            return {"ok": True, "result": {"version": 1, "prev_sim_frame": None, "mem_frame_ids": [], "rng_state": None}}
        assert name == "infer"
        obs = payload["obs"]
        state = obs["kfmem_client_state_v1"]
        now = obs["sim_frame"]
        sent = {entry["sim_frame"] for entry in state["ring"]}
        if state.get("ring_sparse"):
            assert set(memory_client.needed_ring_frames(PLAN, now, state["last_sel_frame"])) <= sent
        memory = [entry["sim_frame"] for entry in state["mem"]]
        last_select = state["last_sel_frame"]
        if now - last_select >= PLAN["select_stride"]:
            last_select = now
            memory = (memory + [now - 2])[-22:]  # write the newest candidate
        self.requests.append(obs)
        action = {
            f"{side}_ee_pose": obs["state"][f"{side}_ee_pose"] for side in ("left", "right")
        } | {f"{side}_ee_joint_state": np.array([0.5], dtype=np.float32) for side in ("left", "right")}
        return {
            "ok": True,
            "latency_ms": 250.0,
            "result": {
                "actions": [action] * wallx.ACTION_CHUNK,
                "kfmem_state": {
                    "version": 1,
                    "prev_sim_frame": now,
                    "mem_frame_ids": memory,
                    "mem_boxes": {},
                    "last_sel_frame": last_select,
                    "consec_skip": 0,
                    "dump_count": 0,
                    "rng_state": b"\x01" * 16,
                    "ring_plan": PLAN,
                },
            },
        }


class CountingObservation(dict):
    renders = 0

    def __getitem__(self, key):
        if key.endswith("rgb"):
            CountingObservation.renders += key == "rgb"
        return super().__getitem__(key)


def observation(step):
    image = np.full((48, 64, 3), step % 255, dtype=np.uint8)
    return CountingObservation(
        rgb=image,
        left_wrist_rgb=image,
        right_wrist_rgb=image,
        joint_positions=np.tile(HOME, 2).astype(np.float32),
        instruction="stack the bowls",
        sim_time=np.array([step / 25.0], dtype=np.float32),
    )


def make_policy(monkeypatch, server):
    monkeypatch.setattr(wallx, "xpolicylab_connection", lambda: server)
    return wallx.make()


def test_policy_requests_every_16_steps_with_the_robodojo_layout(monkeypatch):
    server = FakeServer()
    policy = make_policy(monkeypatch, server)
    CountingObservation.renders = 0
    policy.reset(observation(0))
    steps = 200
    for step in range(steps):
        target = policy.act(observation(step))
        assert target.shape == (16,)
        assert policy.queried == (step % 16 == 0)
        assert policy.last_inference_latency == (0.25 if step % 16 == 0 else None)
    assert [obs["sim_frame"] for obs in server.requests] == list(range(0, steps, 16))
    first = server.requests[0]
    assert set(first["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
    assert first["task_id"] == "roboclock"
    np.testing.assert_allclose(first["state"]["left_ee_pose"], wallx.link6_pose("left", HOME[:6]))
    # A static chunk holds the arms where they are.
    np.testing.assert_allclose(target[:6], HOME[:6], atol=2e-3)
    # The gripper sent next is the last commanded opening.
    np.testing.assert_allclose(server.requests[-1]["state"]["left_ee_joint_state"], [0.5])
    # Later requests carry only the frames the server reads, and memory frames come along.
    assert server.requests[-1]["kfmem_client_state_v1"].get("ring_sparse")
    assert server.requests[-1]["kfmem_client_state_v1"]["mem"]
    assert CountingObservation.renders < 0.5 * steps


def test_policy_rejects_the_50_hz_control_rate(monkeypatch):
    policy = make_policy(monkeypatch, FakeServer())
    policy.reset(observation(0))
    policy.act(observation(0))  # a query: the next step may include charged latency
    policy.act(observation(1))
    obs = observation(1)
    obs["sim_time"] = np.array([1 / 25 + 0.02], dtype=np.float32)
    with pytest.raises(RuntimeError, match="--policy-hz 25"):
        policy.act(obs)


def test_record_all_frames_renders_every_step(monkeypatch):
    monkeypatch.setenv("WALLX_RECORD_ALL_FRAMES", "1")
    policy = make_policy(monkeypatch, FakeServer())
    CountingObservation.renders = 0
    policy.reset(observation(0))
    for step in range(40):
        policy.act(observation(step))
    assert CountingObservation.renders == 40
