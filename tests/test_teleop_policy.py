"""Browser teleoperation policy without Isaac Sim: a fake observation stream and a real
websocket client drive the policy."""

import asyncio
import json
import socket
import threading
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

aiohttp = pytest.importorskip("aiohttp")

from roboclock.envs.lab.robot import ARM_DOF, DOF  # noqa: E402
from roboclock.policies.teleop import policy as teleop  # noqa: E402
from roboclock.policies.teleop import server as teleop_server  # noqa: E402
from roboclock.policies.x5_kinematics import X5Kinematics, transform  # noqa: E402

START = np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0])


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def observation(joints, t, arms):
    ee = []
    for i, arm in enumerate(arms):
        pose = arm.tool_pose(joints[DOF * i : DOF * i + ARM_DOF])
        x, y, z, w = Rotation.from_matrix(pose[:3, :3]).as_quat()
        ee += [*pose[:3, 3], w, x, y, z]
    return {
        "joint_positions": joints.astype(np.float32),
        "ee_pose": np.array(ee, dtype=np.float32),
        "gripper_width": np.array([2 * joints[ARM_DOF], 2 * joints[DOF + ARM_DOF]]),
        "sim_time": np.array([t]),
        "instruction": "test task",
        "rgb": np.zeros((48, 64, 3), np.uint8),
        "left_wrist_rgb": np.zeros((24, 32, 3), np.uint8),
    }


class Client:
    """A browser stand-in on its own thread: sends key states and commands, records frames."""

    def __init__(self, port):
        self.port, self.frames, self.status = port, set(), {}
        self.loop = asyncio.new_event_loop()
        self.connected = threading.Event()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._open(), self.loop).result(5)

    async def _open(self):
        self.session = aiohttp.ClientSession()
        page = await self.session.get(f"http://127.0.0.1:{self.port}/")
        assert "RoboClock teleop" in await page.text()
        self.socket = await self.session.ws_connect(f"http://127.0.0.1:{self.port}/ws")
        asyncio.ensure_future(self._receive())

    async def _receive(self):
        async for message in self.socket:
            if message.type == aiohttp.WSMsgType.BINARY:
                self.frames.add(message.data[1 : 1 + message.data[0]].decode())
            else:
                self.status = json.loads(message.data)

    def send(self, **data):
        asyncio.run_coroutine_threadsafe(self.socket.send_str(json.dumps(data)), self.loop).result(5)
        time.sleep(0.05)


@pytest.fixture()
def rig(monkeypatch):
    port = free_port()
    monkeypatch.setattr(teleop_server, "_SERVER", None)
    monkeypatch.setenv("TELEOP_PORT", str(port))
    monkeypatch.setenv("TELEOP_FPS", "1000")
    arms = [
        X5Kinematics(base=transform(np.eye(3), [0.0, 0.3, 0.75])),
        X5Kinematics(base=transform(np.eye(3), [0.0, -0.3, 0.75])),
    ]
    joints = np.zeros(2 * DOF)
    joints[:ARM_DOF] = joints[DOF : DOF + ARM_DOF] = START
    joints[ARM_DOF : DOF] = joints[DOF + ARM_DOF :] = 0.044
    policy = teleop.make()
    client = Client(port)
    return policy, client, arms, joints


def run(policy, arms, joints, t, steps, keys=None, client=None):
    for _ in range(steps):
        if keys is not None:
            client.send(keys=keys)  # heartbeat, as the page does every 0.1 s
        action = policy.act(observation(joints, t, arms))
        joints = action.astype(float)
        t += 0.02
    return joints, t


def test_keys_move_selected_arm_and_freeze(rig):
    policy, client, arms, joints = rig
    policy.reset(observation(joints, 0.0, arms))
    start = arms[0].tool_pose(joints[:ARM_DOF])[:3, 3]

    # The episode starts frozen: act blocks until Enter.
    threading.Timer(0.3, client.send, kwargs=dict(command="start")).start()
    began = time.monotonic()
    joints, t = run(policy, arms, joints, 0.0, 1)
    assert time.monotonic() - began >= 0.25
    assert {"head", "left_wrist"} <= client.frames

    # W moves the left tool point forward (+x) about speed * time, the right arm stays.
    right_before = joints[DOF:].copy()
    joints, t = run(policy, arms, joints, t, 25, keys=["KeyW"], client=client)
    moved = arms[0].tool_pose(joints[:ARM_DOF])[:3, 3] - start
    assert moved[0] == pytest.approx(0.12 * 0.5, abs=0.02)
    assert abs(moved[1]) < 0.01 and abs(moved[2]) < 0.01
    np.testing.assert_allclose(joints[DOF:], right_before, atol=1e-6)

    # Released keys hold the pose; the gripper closes on Space.
    client.send(keys=[])
    held = joints.copy()
    client.send(command="gripper")
    joints, t = run(policy, arms, joints, t, 30)
    np.testing.assert_allclose(joints[:ARM_DOF], held[:ARM_DOF], atol=1e-6)
    assert joints[ARM_DOF] == pytest.approx(0.0, abs=1e-6)
    assert client.status["gripper"]["left"] == "closed"

    # Frozen: act blocks; "." lets 0.1 s of simulation pass, then it blocks again.
    client.send(command="freeze")
    threading.Timer(0.2, client.send, kwargs=dict(command="step")).start()
    threading.Timer(0.6, client.send, kwargs=dict(command="freeze")).start()
    began = time.monotonic()
    joints, t = run(policy, arms, joints, t, 7)
    assert time.monotonic() - began >= 0.55


def test_realtime_pacing(rig, monkeypatch):
    policy, client, arms, joints = rig
    policy.reset(observation(joints, 0.0, arms))
    client.send(command="start")
    began = time.monotonic()
    run(policy, arms, joints, 0.0, 26)  # 0.5 s of simulation after the pace is set
    assert time.monotonic() - began == pytest.approx(0.5, abs=0.1)
    client.send(command="speed", value="down")  # 1.0 -> 0.5
    began = time.monotonic()
    run(policy, arms, joints, 1.0, 11)  # 0.2 s of simulation at half speed
    assert time.monotonic() - began == pytest.approx(0.4, abs=0.1)


def test_stale_keys_are_dropped(rig):
    policy, client, arms, joints = rig
    client.send(keys=["KeyW"])
    assert policy.server.held_keys() == {"KeyW"}
    time.sleep(teleop_server.KEY_TIMEOUT + 0.1)
    assert policy.server.held_keys() == frozenset()


def test_episode_result_reaches_page(rig):
    policy, client, arms, joints = rig
    policy.reset(observation(joints, 0.0, arms))
    row = {"task_id": "A1", "difficulty": "easy", "seed": 0, "task_success": True,
           "temporal_success": False, "termination_reason": "time_budget"}

    def judged():
        time.sleep(0.3)
        assert client.status["state"] == "finished"
        assert client.status["result"]["task_success"] is True
        client.send(command="restart")

    threading.Thread(target=judged).start()
    assert policy.episode_finished(row) == "restart"  # waits for the operator
    threading.Timer(0.2, client.send, kwargs=dict(command="continue")).start()
    assert policy.episode_finished(row) == "continue"


def test_restart_button_abandons_the_attempt(rig):
    from roboclock.evaluation.evaluator import RestartEpisode

    policy, client, arms, joints = rig
    policy.reset(observation(joints, 0.0, arms))
    client.send(command="restart")  # ignored before the episode starts
    client.send(command="start")
    joints, t = run(policy, arms, joints, 0.0, 3)
    client.send(command="restart")
    with pytest.raises(RestartEpisode):
        run(policy, arms, joints, t, 1)
