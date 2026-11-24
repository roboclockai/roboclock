"""move_eef agent: the angle convention, the tool, and motions played on a kinematic arm."""

import json

import numpy as np
import pytest
from test_vision_llm_primitives import Bench, answer, spaces, tool_results, turn_text

from roboclock.envs.lab.robot import ARM_DOF
from roboclock.policies.vision_llm_move_eef import make, pose
from roboclock.policies.vision_llm_move_eef import policy as agent

HZ = 50


def start(monkeypatch, bench, replies, **space):
    sent = []

    def fake_request(url, body, timeout, cookie, **_):
        sent.append(body)
        reply = replies.pop(0) if replies else answer(("give_up", {"reason": "end of test"}))
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(agent, "request_chat", fake_request)
    policy = make()
    if space:
        policy.set_spaces(*spaces(**space))
    policy.reset(bench.observation())
    return policy, sent


def run(monkeypatch, bench, replies, steps, **space):
    policy, sent = start(monkeypatch, bench, replies, **space)
    for _ in range(steps):
        bench.apply(policy.act(bench.observation()))
    return policy, sent


def move(**targets):
    return answer(("move_eef", {"targets": targets, "note": "test"}))


def state(body):
    """The named values of a request's current observation."""
    out = {}
    for line in turn_text(body).splitlines():
        name, _, value = line.partition("=")
        if name in pose.labels():
            out[name] = float(value)
    return out


# --- the dimensions ----------------------------------------------------------------------------


def test_zero_angles_are_a_straight_down_grasp_and_the_axes_turn_as_described():
    fingers = lambda *angles: pose.angles_to_rotation(*angles)[:, 2]  # noqa: E731
    jaws = lambda *angles: pose.angles_to_rotation(*angles)[:, 1]  # noqa: E731
    assert np.allclose(fingers(0, 0, 0), [0, 0, -1]) and np.allclose(jaws(0, 0, 0), [1, 0, 0])
    assert np.allclose(fingers(90, 0, 0), [0, 1, 0], atol=1e-9)  # pitch: horizontal, at world +y
    assert np.allclose(fingers(0, 90, 0), [-1, 0, 0], atol=1e-9)  # roll: horizontal, at world -x
    assert np.allclose(fingers(0, 0, 90), [0, 0, -1]) and np.allclose(jaws(0, 0, 90), [0, 1, 0], atol=1e-9)


@pytest.mark.parametrize("angles", [(0, 0, 0), (30, -20, 110), (-150, 60, -170), (90, 0, 45), (10, 90, 0)])
def test_angles_round_trip_through_a_pose(angles):
    values = {"x": 0.1, "y": -0.2, "z": 0.9, **dict(zip(pose.ANGLE_AXES, angles))}
    back = pose.pose_to_values(pose.values_to_pose(values))
    assert pose.angle_between(pose.values_to_pose(back)[:3, :3], pose.values_to_pose(values)[:3, :3]) < 1e-4
    assert all(abs(back[axis] - values[axis]) < 1e-9 for axis in pose.POSITION_AXES)


def test_the_tool_surface_is_move_eef_wait_and_give_up(monkeypatch):
    policy, sent = run(monkeypatch, Bench(), [], 2)
    names = [tool["function"]["name"] for tool in policy.tools]
    assert names == ["move_eef", "wait", "give_up"]
    assert "call wait" in sent[0]["messages"][0]["content"]
    move_eef = policy.tools[0]["function"]
    assert move_eef["parameters"]["required"] == ["targets", "note"]
    assert len(pose.labels()) == 14 and all(label in move_eef["parameters"]["properties"]["targets"]["description"] for label in pose.labels())
    assert "left_x: [" in move_eef["description"] and "10s" in move_eef["description"]


# --- one turn ----------------------------------------------------------------------------------


def test_the_observation_names_the_14_dimensions_and_carries_no_depth_request(monkeypatch):
    bench = Bench()
    _, sent = run(monkeypatch, bench, [], 2)
    body = sent[0]
    assert set(state(body)) == set(pose.labels())
    assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False
    tool = bench.tool(0)[:3, 3]
    assert np.allclose([state(body)[f"left_{axis}"] for axis in "xyz"], tool, atol=1e-4)
    assert state(body)["left_gripper"] == pytest.approx(0.08 / 0.088, abs=0.01)
    assert "TASK. test task" in body["messages"][0]["content"]


def test_a_named_dimension_moves_and_the_others_hold(monkeypatch):
    bench = Bench()
    before = bench.tool(1).copy()
    other = bench.tool(0).copy()
    z = float(before[2, 3]) - 0.05
    policy, sent = run(monkeypatch, bench, [move(right_z=z)], 200)
    after = bench.tool(1)
    assert after[2, 3] == pytest.approx(z, abs=0.002)
    assert np.allclose(after[:2, 3], before[:2, 3], atol=0.002)
    assert pose.angle_between(after[:3, :3], before[:3, :3]) < 1.0
    assert np.allclose(bench.tool(0), other)  # the unnamed arm stands still
    result = tool_results(sent[1])[0]
    assert result.startswith("Accepted: playing") and "how close the tool point landed" in result
    assert "Arrival check for the previous move_eef target: right 0 mm and 0.0 deg away." in turn_text(sent[1])


def test_the_playout_keeps_every_joint_under_the_speed_limit_and_its_time_follows_the_distance(monkeypatch):
    bench = Bench()
    z = float(bench.tool(1)[2, 3])
    policy, sent = start(monkeypatch, bench, [move(right_z=z - 0.10)])
    bench.apply(policy.act(bench.observation()))  # the query step
    steps, fastest, previous = len(policy.queue), 0.0, bench.joints.copy()
    while policy.queue:
        bench.apply(policy.act(bench.observation()))
        fastest = max(fastest, float(np.max(np.abs(bench.joints - previous)[[*range(ARM_DOF), *range(8, 8 + ARM_DOF)]])))
        previous = bench.joints.copy()
    assert fastest <= agent.JOINT_SPEED / HZ + 1e-6
    assert fastest >= 0.9 * agent.JOINT_SPEED / HZ  # and the limiting joint runs at it
    assert f"playing {steps} control steps ({steps / HZ:.1f}s)" in policy.turn["tools"][0]["content"]


def test_a_gripper_named_with_a_pose_moves_only_after_the_arm_has_arrived(monkeypatch):
    bench = Bench()
    z = float(bench.tool(1)[2, 3]) - 0.05
    policy, _ = start(monkeypatch, bench, [move(right_z=z, right_gripper=0.0)])
    bench.apply(policy.act(bench.observation()))
    closing_from = None
    for index in range(len(policy.queue)):
        bench.apply(policy.act(bench.observation()))
        if closing_from is None and bench.width[1] < 0.0799:
            closing_from = index
            assert bench.tool(1)[2, 3] == pytest.approx(z, abs=0.001)  # the arm is there already
    assert closing_from is not None and bench.width[1] == pytest.approx(0.0, abs=1e-6)
    assert "move the gripper, once the arm has arrived" in policy.turn["tools"][0]["content"]


def test_gripper_zero_commands_the_rigs_lowest_finger_target_and_one_its_widest(monkeypatch):
    bench = Bench()
    policy, _ = run(monkeypatch, bench, [move(left_gripper=0.0)], 90, finger_min=-0.03, open_width=0.06)
    assert policy.action()[ARM_DOF] == pytest.approx(-0.03)
    policy, _ = run(monkeypatch, Bench(), [move(left_gripper=1.0)], 30, finger_min=-0.03, open_width=0.06)
    assert policy.action()[ARM_DOF] == pytest.approx(0.03)


def test_both_arms_named_in_one_call_arrive_together(monkeypatch):
    bench = Bench()
    left, right = float(bench.tool(0)[2, 3]), float(bench.tool(1)[2, 3])
    policy, _ = start(monkeypatch, bench, [move(left_z=left - 0.02, right_z=right - 0.10)])
    bench.apply(policy.act(bench.observation()))
    for _ in range(len(policy.queue) - 1):
        bench.apply(policy.act(bench.observation()))
    assert bench.tool(0)[2, 3] > left - 0.02 + 1e-4 and bench.tool(1)[2, 3] > right - 0.10 + 1e-4  # neither is there yet
    bench.apply(policy.act(bench.observation()))
    assert bench.tool(0)[2, 3] == pytest.approx(left - 0.02, abs=0.002)
    assert bench.tool(1)[2, 3] == pytest.approx(right - 0.10, abs=0.002)


def test_parallel_move_eef_calls_are_one_motion_and_other_pairs_are_refused(monkeypatch):
    bench = Bench()
    left, right = float(bench.tool(0)[2, 3]), float(bench.tool(1)[2, 3])
    two = answer(("move_eef", {"targets": {"left_z": left - 0.03}, "note": "a"}), ("move_eef", {"targets": {"right_z": right - 0.03}, "note": "b"}))
    twice = answer(("move_eef", {"targets": {"left_z": left}, "note": "a"}), ("move_eef", {"targets": {"left_z": right}, "note": "b"}))
    _, sent = run(monkeypatch, bench, [two, twice], 150)
    assert all(result.startswith("Accepted") for result in tool_results(sent[1]))
    assert bench.tool(0)[2, 3] == pytest.approx(left - 0.03, abs=0.002) and bench.tool(1)[2, 3] == pytest.approx(right - 0.03, abs=0.002)
    assert all("exactly one" in result for result in tool_results(sent[2]))


# --- refusals: nothing moves -------------------------------------------------------------------


@pytest.mark.parametrize(
    "targets, said",
    [
        ({}, "non-empty object"),
        ({"left_w": 0.1}, "unknown dimension 'left_w'"),
        ({"left_z": "high"}, "must be a finite number"),
        ({"left_gripper": 1.5}, "outside [0, 1]"),
        ({"right_roll_deg": 120}, "outside [-90, 90]"),
        ({"left_x": 9.0}, "outside ["),
    ],
)
def test_a_bad_target_is_refused_with_the_reason(monkeypatch, targets, said):
    bench = Bench()
    before = bench.joints.copy()
    _, sent = run(monkeypatch, bench, [answer(("move_eef", {"targets": targets, "note": "x"}))], 20)
    assert tool_results(sent[1])[0].startswith("REFUSED") and said in tool_results(sent[1])[0]
    assert np.array_equal(bench.joints, before)


def test_an_unreachable_pose_is_refused_and_neither_arm_moves(monkeypatch):
    bench = Bench()
    before = bench.joints.copy()
    base = bench.arms[0].base[:3, 3]
    far = move(left_x=float(base[0]) + 0.55, left_y=float(base[1]) + 0.55, left_z=float(base[2]) + 0.55, right_z=float(bench.tool(1)[2, 3]) - 0.02)
    _, sent = run(monkeypatch, bench, [far], 20)
    result = tool_results(sent[1])[0]
    assert "left arm pose unreachable" in result and "Neither arm moved" in result
    assert np.array_equal(bench.joints, before)
    assert "Arrival check" not in turn_text(sent[1])


def test_a_motion_over_the_playout_cap_is_refused(monkeypatch):
    monkeypatch.setenv("VISION_LLM_EEF_MAX_DURATION", "0.2")
    bench = Bench()
    before = bench.joints.copy()
    _, sent = run(monkeypatch, bench, [move(right_z=float(bench.tool(1)[2, 3]) - 0.10)], 20)
    assert "over the 0.2s playout cap" in tool_results(sent[1])[0]
    assert np.array_equal(bench.joints, before)


def test_a_path_into_the_other_arm_is_refused(monkeypatch):
    bench = Bench()
    before = bench.joints.copy()
    middle = (bench.tool(0)[:3, 3] + bench.tool(1)[:3, 3]) / 2
    toward = (bench.tool(1)[:3, 3] - bench.tool(0)[:3, 3]) / np.linalg.norm(bench.tool(1)[:3, 3] - bench.tool(0)[:3, 3])
    left, right = middle - 0.02 * toward, middle + 0.02 * toward  # the two hands 4 cm apart
    targets = {f"{side}_{axis}": float(value) for side, point in (("left", left), ("right", right)) for axis, value in zip("xyz", point)}
    _, sent = run(monkeypatch, bench, [move(**targets)], 20)
    assert "two arms' bodies" in tool_results(sent[1])[0]
    assert np.array_equal(bench.joints, before)


# --- between turns -----------------------------------------------------------------------------


def test_the_model_is_asked_again_only_once_the_arm_is_at_rest(monkeypatch):
    bench = Bench(delay=10)  # the arm follows 0.2 s behind its command
    z = float(bench.tool(1)[2, 3]) - 0.05
    policy, sent = start(monkeypatch, bench, [move(right_z=z)])
    while len(sent) < 2:
        bench.apply(policy.act(bench.observation()))
    assert state(sent[1])["right_z"] == pytest.approx(z, abs=0.002)
    assert "right 0 mm" in turn_text(sent[1]) or "right 1 mm" in turn_text(sent[1])


def test_an_arm_held_off_is_reported_and_replans_from_where_it_is(monkeypatch):
    bench = Bench(floor=0.0)
    bench.floor = float(bench.tool(1)[2, 3]) - 0.04
    z = bench.floor - 0.06
    x = float(bench.tool(1)[0, 3])
    policy, sent = run(monkeypatch, bench, [move(right_z=z), move(right_x=x + 0.02)], 400)
    assert "Arrival check for the previous move_eef target: right 6" in turn_text(sent[1])  # about 60 mm short
    # The second call names no z: the hand keeps the height it is held at, not the one it was sent to.
    assert policy.arms[1].tool_pose(policy.command[1])[2, 3] == pytest.approx(bench.tool(1)[2, 3], abs=0.003)
    assert bench.tool(1)[0, 3] == pytest.approx(x + 0.02, abs=0.003)


def test_give_up_and_the_call_budget_stop_the_queries(monkeypatch):
    bench = Bench()
    _, sent = run(monkeypatch, bench, [answer(("give_up", {"reason": "cannot"}))], 60)
    assert len(sent) == 1
    monkeypatch.setenv("VISION_LLM_MAX_QUERIES", "3")
    refused = [answer(("move_eef", {"targets": {}, "note": "x"})) for _ in range(9)]
    policy, sent = run(monkeypatch, Bench(), refused, 200)
    assert len(sent) == 3 and policy.queried is False


def test_a_query_step_is_flagged_and_the_log_row_has_the_call(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_LLM_LOG", str(tmp_path / "log.jsonl"))
    bench = Bench()
    policy, _ = start(monkeypatch, bench, [move(right_z=float(bench.tool(1)[2, 3]) - 0.02)])
    bench.apply(policy.act(bench.observation()))
    assert policy.queried is True
    bench.apply(policy.act(bench.observation()))
    assert policy.queried is False
    row = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert row["calls"][0].startswith("move_eef(") and row["result"].startswith("Accepted") and row["parse_error"] is None


def test_a_reachable_pose_is_accepted_even_when_the_line_to_it_is_not():
    bench = Bench()
    policy = make()
    policy.reset(bench.observation())
    arm, rng, accepted = policy.arms[1], np.random.default_rng(1), 0
    for _ in range(60):
        target = arm.tool_pose(rng.uniform(arm.low, arm.high))
        try:
            path = policy.plan(1, policy.command[1].copy(), arm.tool_pose(policy.command[1]), target)
        except agent.Refused:
            continue
        accepted += 1
        assert policy.off(1, path[-1], target)[0] <= agent.REACH_TOLERANCE
    assert accepted >= 50  # the line alone takes about 43 of these


def test_wait_holds_still_for_the_seconds_named_and_costs_one_call(monkeypatch):
    bench = Bench()
    before = bench.observation()["joint_positions"].copy()
    policy, sent = run(monkeypatch, bench, [answer(("wait", {"seconds": 1.0, "note": "for the lamp"}))], 80)
    assert len(sent) == 2 and np.allclose(bench.observation()["joint_positions"], before, atol=1e-6)
    times = [float(turn_text(body).split("Time now: ")[1].split(" s")[0]) for body in sent]
    assert times[1] - times[0] == pytest.approx(1.0, abs=1.5 / HZ)
    assert "holding still for 1.00s" in tool_results(sent[1])[-1]


@pytest.mark.parametrize("seconds", [0.0, 11, "long", None])
def test_a_wait_outside_its_range_is_refused(monkeypatch, seconds):
    _, sent = run(monkeypatch, Bench(), [answer(("wait", {"seconds": seconds, "note": "x"}))], 12)
    assert tool_results(sent[1])[-1].startswith("REFUSED: wait.seconds")


def test_without_the_wait_tool_the_surface_is_robodojos(monkeypatch):
    monkeypatch.setenv("VISION_LLM_EEF_WAIT", "0")
    policy, sent = run(monkeypatch, Bench(), [answer(("wait", {"seconds": 1.0, "note": "x"}))], 12)
    assert [tool["function"]["name"] for tool in policy.tools] == ["move_eef", "give_up"]
    assert "call wait" not in sent[0]["messages"][0]["content"]
    assert "unknown tool 'wait'" in tool_results(sent[1])[-1]
