"""Kinematics, rig calibration, answer handling and motion planning of the vision LLM agent."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roboclock.envs import robodojo_rig
from roboclock.envs.lab.robot import ARM_DOF, HOME, JOINT_HIGH, JOINT_LOW
from roboclock.policies import vision_llm
from roboclock.policies.x5_kinematics import X5Kinematics, pose_matrix

LOW, HIGH = np.tile(JOINT_LOW, 2), np.tile(JOINT_HIGH, 2)
# A RoboDojo rig placed off the world origin and turned, as tasks place their rigs.
RIG_ORIGIN, RIG_YAW = np.array([0.2, -0.1, 0.9]), 30.0
RIG = robodojo_rig.rig_to_world(RIG_ORIGIN, RIG_YAW)
MOUNTS = tuple(
    RIG @ robodojo_rig.transform(offset, np.eye(3))
    for offset in (robodojo_rig.LEFT_BASE_IN_RIG_M, robodojo_rig.RIGHT_BASE_IN_RIG_M)
)
CONTROL_HZ = 50


def tool_pose_vector(position, rotation):
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return np.array([*position, w, x, y, z])


def observation(joints=None):
    joints = np.tile(HOME, 2).astype(np.float32) if joints is None else joints
    pose = []
    for index, mount in enumerate(MOUNTS):
        arm = X5Kinematics(base=mount)
        tool = arm.tool_pose(joints[8 * index : 8 * index + ARM_DOF].astype(float))
        pose.append(tool_pose_vector(tool[:3, 3], tool[:3, :3]))
    return {
        "rgb": np.full((72, 96, 3), 120, dtype=np.uint8),
        "left_wrist_rgb": np.full((36, 48, 3), 60, dtype=np.uint8),
        "right_wrist_rgb": np.full((36, 48, 3), 200, dtype=np.uint8),
        "joint_positions": joints,
        "ee_pose": np.concatenate(pose).astype(np.float32),
        "gripper_width": np.array([0.08, 0.08], dtype=np.float32),
        "instruction": "collect the water",
        "sim_time": np.array([1.5], dtype=np.float32),
    }


def waypoint(ref, value, tool="keep", gripper="open"):
    return {"ref": ref, "value": list(map(float, value)), "tool": tool, "gripper": gripper}


def reply_with(left, right, plan="move", wait=0.0):
    answer = {"plan": plan, "left": left, "right": right, "wait": wait}
    return {
        "choices": [{"message": {"content": json.dumps(answer)}}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 60},
    }


def chat_reply(left, right, plan="move"):
    """Each arm goes to one world point."""
    return reply_with([waypoint("world", left)], [waypoint("world", right)], plan)


def make_policy(monkeypatch, replies):
    requests = []

    def fake_request(url, body, timeout, cookie, **_):
        requests.append(body)
        return replies(body)

    monkeypatch.setattr(vision_llm, "request_chat", fake_request)
    policy = vision_llm.make()
    policy.reset(observation())
    return policy, requests


def holding(body):
    return reply_with([], [], plan="wait", wait=1.0)


def final_tool_point(policy, trajectory, arm=0):
    joints = trajectory[-1][8 * arm : 8 * arm + ARM_DOF].astype(float)
    return policy.arms[arm].tool_pose(joints)[:3, 3]


def test_kinematics_match_the_urdf_tool_frame_at_zero():
    tool = X5Kinematics().tool_pose(np.zeros(ARM_DOF))
    # Reach along the arm's links at zero, with the fingers (+Z tool) pointing along +X.
    np.testing.assert_allclose(tool[:3, 2], [1, 0, 0], atol=1e-3)
    assert tool[2, 3] > 0.1


def test_jacobian_matches_finite_differences():
    arm = X5Kinematics()
    joints = np.array([0.3, 1.2, 1.0, -0.4, 0.2, -0.3])
    pose, jacobian = arm.jacobian(joints)
    for index in range(ARM_DOF):
        moved = joints.copy()
        moved[index] += 1e-6
        numeric = (arm.tool_pose(moved)[:3, 3] - pose[:3, 3]) / 1e-6
        np.testing.assert_allclose(jacobian[:3, index], numeric, atol=1e-4)


def test_calibration_recovers_the_base_and_reproduces_the_observed_pose():
    joints = HOME[:ARM_DOF].astype(float)
    base = pose_matrix(MOUNTS[1][:3, 3], Rotation.from_euler("z", 0.4).as_quat()[[3, 0, 1, 2]])
    observed = X5Kinematics(base=base).tool_pose(joints)
    vector = tool_pose_vector(observed[:3, 3], observed[:3, :3])
    calibrated = X5Kinematics().calibrated(joints, vector[:3], vector[3:])
    np.testing.assert_allclose(calibrated.base, base, atol=1e-9)


def test_ik_reaches_a_nearby_pose_within_limits():
    arm = X5Kinematics(base=MOUNTS[0])
    joints = HOME[:ARM_DOF].astype(float)
    target = arm.tool_pose(joints)
    target[:3, 3] += [0.05, -0.04, -0.06]
    solved, residual = arm.solve(target, joints)
    assert residual < 1e-3
    assert np.all(solved >= arm.low) and np.all(solved <= arm.high)


def test_ik_reports_an_unreachable_target_instead_of_failing():
    arm = X5Kinematics()
    target = arm.tool_pose(HOME[:ARM_DOF].astype(float))
    target[:3, 3] += [3.0, 0.0, 0.0]
    _, residual = arm.solve(target, HOME[:ARM_DOF])
    assert residual > 2.0


GOOD = {"plan": "x", "left": [], "right": [], "wait": 0.5}
STEP = {"ref": "world", "value": [0, 0, 1], "tool": "keep", "gripper": "open"}


@pytest.mark.parametrize(
    "answer",
    [
        "not json",
        [],
        {**GOOD, "wait": None},
        {k: v for k, v in GOOD.items() if k != "wait"},
        {**GOOD, "left": {"target": [0, 0, 1]}},
        {**GOOD, "left": [{**STEP, "value": [0, 0]}]},
        {**GOOD, "left": [{**STEP, "value": [0, 0, float("nan")]}]},
        {**GOOD, "left": [{**STEP, "tool": "spin"}]},
        {**GOOD, "left": [{**STEP, "ref": "polar"}]},
        {**GOOD, "left": [STEP] * (vision_llm.MAX_WAYPOINTS + 1)},
    ],
)
def test_parse_answer_rejects_malformed_output(answer):
    text = answer if isinstance(answer, str) else json.dumps(answer)
    assert vision_llm.parse_answer(text) is None


def test_parse_answer_accepts_holding_and_waypoint_lists():
    assert vision_llm.parse_answer(json.dumps(GOOD)) is not None
    assert vision_llm.parse_answer(json.dumps({**GOOD, "right": [STEP, STEP]})) is not None


def test_head_camera_follows_from_the_calibrated_bases_and_the_rig_spec(monkeypatch):
    policy, _ = make_policy(monkeypatch, holding)
    camera = policy.camera
    expected = robodojo_rig.head_camera_world(RIG_ORIGIN, RIG_YAW)
    np.testing.assert_allclose(camera["pose"][:3, 3], expected[:3, 3], atol=1e-9)
    # Optical frame: x right, y down, z along the USD camera's -Z view axis.
    np.testing.assert_allclose(camera["pose"][:3, 2], -expected[:3, 2], atol=1e-9)
    np.testing.assert_allclose(camera["pose"][:3, 1], -expected[:3, 1], atol=1e-9)
    assert camera["mount"] == pytest.approx(RIG_ORIGIN[2])
    assert camera["separation_error_m"] < 1e-6
    np.testing.assert_allclose(
        camera["intrinsics"], robodojo_rig.intrinsics(robodojo_rig.HEAD_CAMERA)
    )
    assert "mounted at z = 0.90" in policy.system


def project(camera, point):
    """Pixel (u, v) of the 640x480 head image and optical depth of a world point."""
    pose = camera["pose"]
    local = pose[:3, :3].T @ (np.asarray(point, dtype=float) - pose[:3, 3])
    k = camera["intrinsics"]
    return (
        k[0, 0] * local[0] / local[2] + k[0, 2],
        k[1, 1] * local[1] / local[2] + k[1, 2],
        local[2],
    )


def test_a_pixel_lifts_to_the_world_point_it_shows(monkeypatch):
    policy, _ = make_policy(monkeypatch, holding)
    target = RIG_ORIGIN + RIG[:3, :3] @ np.array([0.35, 0.05, 0.02])
    u, v, distance = project(policy.camera, target)
    # A half-size depth image of a surface at that distance: intrinsics scale with it.
    depth = np.full((240, 320), distance, dtype=np.float32)
    point, problem = vision_llm.pixel_to_world({"depth": depth}, policy.camera, u, v, (640, 480))
    assert problem is None
    np.testing.assert_allclose(point, target, atol=1e-6)


def test_a_pixel_is_refused_without_depth_off_the_image_or_on_empty_space(monkeypatch):
    policy, _ = make_policy(monkeypatch, holding)
    camera = policy.camera
    assert "--depth" in vision_llm.pixel_to_world({}, camera, 10, 10, (640, 480))[1]
    depth = np.full((480, 640), 1.0, dtype=np.float32)
    assert "outside" in vision_llm.pixel_to_world({"depth": depth}, camera, 700, 10, (640, 480))[1]
    depth[:] = np.inf
    assert (
        "no surface" in vision_llm.pixel_to_world({"depth": depth}, camera, 10, 10, (640, 480))[1]
    )


def test_prompt_sends_three_views_the_measured_state_and_a_thinking_budget(monkeypatch):
    policy, requests = make_policy(monkeypatch, holding)
    policy.act(observation())
    system, user = requests[0]["messages"]
    assert "world coordinates are metres" in system["content"].lower()
    assert '"pixel"' in system["content"] and "640x480" in system["content"]
    images = [part for part in user["content"] if part["type"] == "image_url"]
    assert len(images) == 3
    assert all(part["image_url"]["url"].startswith("data:image/jpeg;base64,") for part in images)
    text = user["content"][-1]["text"]
    assert "collect the water" in text and "left tool point" in text
    assert "pixel waypoints are refused" in text
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert requests[0]["thinking_token_budget"] == 1024
    assert requests[0]["max_tokens"] > 1024


def test_thinking_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("VISION_LLM_THINKING", "0")
    policy, requests = make_policy(monkeypatch, holding)
    policy.act(observation())
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_token_budget" not in requests[0]


def test_prompt_states_the_head_camera_with_depth(monkeypatch):
    policy, requests = make_policy(monkeypatch, holding)
    policy.act({**observation(), "depth": np.ones((480, 640), dtype=np.float32)})
    text = requests[0]["messages"][1]["content"][-1]["text"]
    position = np.round(policy.camera["pose"][:3, 3], 3).tolist()
    focal = robodojo_rig.intrinsics(robodojo_rig.HEAD_CAMERA)[0, 0]
    assert f"Head camera: at {position}" in text and f"focal length {focal:.0f} px" in text


def test_an_answer_lasts_its_wait_then_queries_again(monkeypatch):
    policy, requests = make_policy(monkeypatch, holding)
    count = round(1.0 * policy.control_hz)
    obs = observation()
    assert policy.control_hz == pytest.approx(CONTROL_HZ)
    for step in range(count):
        policy.act(obs)
        assert policy.queried == (step == 0)
    assert len(requests) == 1
    policy.act(obs)
    assert len(requests) == 2 and policy.queried


def test_a_commanded_motion_stays_in_limits_and_reaches_the_target(monkeypatch):
    obs = observation()
    goal = obs["ee_pose"][:3] + np.array([0.04, 0.03, -0.05])
    policy, _ = make_policy(monkeypatch, lambda body: chat_reply(goal, obs["ee_pose"][7:10]))
    trajectory = [policy.act(obs), *policy.queue]
    assert all(np.all(step >= LOW) and np.all(step <= HIGH) for step in trajectory)
    np.testing.assert_allclose(final_tool_point(policy, trajectory), goal, atol=2e-3)
    assert policy.feedback == "reached as commanded"
    # The arm moves at the tool speed rather than over a fixed period.
    expected = np.ceil(np.linalg.norm([0.04, 0.03, 0.05]) / vision_llm.TOOL_SPEED * CONTROL_HZ)
    assert len(trajectory) == expected
    # The idle right arm does not move.
    np.testing.assert_allclose(trajectory[-1][8:14], obs["joint_positions"][8:14], atol=1e-3)


def test_one_answer_can_reach_grasp_and_lift(monkeypatch):
    obs = observation()
    start = obs["ee_pose"][:3].astype(float)
    left = [
        waypoint("world", start + [0.10, 0.0, 0.0]),
        waypoint("offset", [0, 0, 0], gripper="close"),
        waypoint("offset", [0, 0, 0.05], gripper="close"),
    ]
    policy, _ = make_policy(monkeypatch, lambda body: reply_with(left, [], wait=0.5))
    trajectory = [policy.act(obs), *policy.queue]
    np.testing.assert_allclose(
        final_tool_point(policy, trajectory), start + [0.10, 0.0, 0.05], atol=2e-3
    )
    np.testing.assert_allclose(trajectory[-1][6:8], vision_llm.GRIPPER_CLOSED, atol=1e-6)
    # The fingers open fully while reaching and close while the tool is still.
    reach = int(np.ceil(0.10 / vision_llm.TOOL_SPEED * CONTROL_HZ))
    np.testing.assert_allclose(trajectory[reach - 1][6:8], vision_llm.GRIPPER_OPEN, atol=1e-6)
    close = int(np.ceil(vision_llm.GRIPPER_TIME * CONTROL_HZ))
    lift = int(np.ceil(0.05 / vision_llm.TOOL_SPEED * CONTROL_HZ))
    assert len(trajectory) == reach + close + lift + round(0.5 * CONTROL_HZ)


def test_a_pixel_waypoint_goes_to_the_seen_surface_and_the_model_is_told(monkeypatch):
    policy, _ = make_policy(monkeypatch, holding)
    obs = observation()
    seen = obs["ee_pose"][:3].astype(float) + [0.05, 0.0, -0.15]
    u, v, distance = project(policy.camera, seen)
    obs["depth"] = np.full((480, 640), distance, dtype=np.float32)
    left = [waypoint("pixel", [u, v, 0.2])]
    policy, requests = make_policy(monkeypatch, lambda body: reply_with(left, []))
    trajectory = [policy.act(obs), *policy.queue]
    np.testing.assert_allclose(final_tool_point(policy, trajectory), seen + [0, 0, 0.2], atol=2e-3)
    assert f"is the surface at {np.round(seen, 3).tolist()}" in policy.feedback
    policy.queue = []
    policy.act(obs)
    assert (
        f"pixel ({u:.0f}, {v:.0f}) is the surface"
        in requests[1]["messages"][1]["content"][-1]["text"]
    )


def test_a_refused_pixel_skips_the_waypoints_that_depend_on_it(monkeypatch):
    obs = observation()
    left = [waypoint("pixel", [10, 10, 0]), waypoint("offset", [0, 0, 0.05])]
    policy, _ = make_policy(monkeypatch, lambda body: reply_with(left, []))
    trajectory = [policy.act(obs), *policy.queue]
    np.testing.assert_allclose(trajectory[-1][:6], obs["joint_positions"][:6], atol=1e-6)
    assert "--depth" in policy.feedback and "rest skipped" in policy.feedback


def test_a_long_waypoint_is_shortened_and_the_model_is_told(monkeypatch):
    obs = observation()
    policy, requests = make_policy(monkeypatch, holding)
    start = obs["ee_pose"][:3].astype(float)
    bounded, shortened = policy.bounded_target(start, start + [2.0, 0.0, 0.0])
    assert shortened
    assert np.linalg.norm(bounded - start) == pytest.approx(policy.max_step)
    far = reply_with([waypoint("offset", [2.0, 0.0, 0.0])], [])
    policy, requests = make_policy(monkeypatch, lambda body: far)
    policy.act(obs)
    policy.queue = []
    policy.act(obs)
    text = requests[1]["messages"][1]["content"][-1]["text"]
    assert "left waypoint 1 shortened to 0.60 m" in text


def test_invalid_answer_holds_and_costs_one_turn(monkeypatch):
    reply = {"choices": [{"message": {"content": "sorry"}}]}
    policy, requests = make_policy(monkeypatch, lambda body: reply)
    obs = observation()
    action = policy.act(obs)
    np.testing.assert_allclose(action, np.clip(obs["joint_positions"], LOW, HIGH))
    assert len(policy.queue) + 1 == round(vision_llm.ERROR_HOLD * CONTROL_HZ)
    policy.queue = []
    policy.act(obs)
    assert "invalid answer" in requests[1]["messages"][1]["content"][-1]["text"]


def test_a_still_arm_keeps_squeezing_what_it_grasped(monkeypatch):
    obs = observation()
    replies = iter(
        [
            reply_with([waypoint("offset", [0, 0, 0], gripper="close")], []),
            reply_with([], []),
        ]
    )
    policy, _ = make_policy(monkeypatch, lambda body: next(replies))
    policy.act(obs)
    policy.queue = []
    # The fingers stopped on an object 2 cm wide, but the command stays closed.
    gripping = obs["joint_positions"].copy()
    gripping[6:8] = 0.01
    action = policy.act({**obs, "joint_positions": gripping})
    np.testing.assert_allclose(action[6:8], vision_llm.GRIPPER_CLOSED, atol=1e-6)
    np.testing.assert_allclose(action[14:16], obs["joint_positions"][14:16], atol=1e-6)


def test_gripper_command_ramps_to_the_physical_range(monkeypatch):
    obs = observation()
    reply = reply_with(
        [waypoint("offset", [0, 0, 0], gripper="close")],
        [waypoint("offset", [0, 0, 0], gripper="open")],
        plan="grip",
    )
    policy, _ = make_policy(monkeypatch, lambda body: reply)
    policy.act(obs)
    final = policy.queue[-1]
    np.testing.assert_allclose(final[6:8], 0.0, atol=1e-6)
    np.testing.assert_allclose(final[14:16], 0.044, atol=1e-6)


def test_server_failure_is_loud_not_a_silent_hold(monkeypatch):
    def down(url, body, timeout, cookie, **_):
        raise RuntimeError("Vision LLM server unreachable")

    monkeypatch.setattr(vision_llm, "request_chat", down)
    policy = vision_llm.make()
    policy.reset(observation())
    with pytest.raises(RuntimeError, match="unreachable"):
        policy.act(observation())


def test_request_log_records_latency_and_tokens(monkeypatch, tmp_path):
    log = tmp_path / "llm.jsonl"
    monkeypatch.setenv("VISION_LLM_LOG", str(log))
    policy, _ = make_policy(monkeypatch, holding)
    policy.act(observation())
    row = json.loads(log.read_text().splitlines()[0])
    assert row["prompt_tokens"] == 900 and row["completion_tokens"] == 60
    assert row["parse_error"] is False and row["round_trip_ms"] >= 0.0


def test_policy_reports_its_model_and_each_querys_output_tokens(monkeypatch):
    # --inference-latency ideal charges these at the model's surveyed speed.
    monkeypatch.delenv("VISION_LLM_MODEL", raising=False)
    policy, _ = make_policy(monkeypatch, holding)
    policy.act(observation())
    assert policy.inference_model == "Qwen/Qwen3.8-27B"
    assert policy.last_output_tokens == 60


def test_log_measures_how_far_the_arms_ended_from_the_previous_target(monkeypatch, tmp_path):
    log = tmp_path / "llm.jsonl"
    monkeypatch.setenv("VISION_LLM_LOG", str(log))
    obs = observation()
    goal = obs["ee_pose"][:3] + np.array([0.0, 0.0, -0.05])
    policy, _ = make_policy(monkeypatch, lambda body: chat_reply(goal, obs["ee_pose"][7:10]))
    policy.act(obs)
    policy.queue = []
    policy.act(obs)
    first, second = [json.loads(line) for line in log.read_text().splitlines()]
    assert first["tracking_error_m"] is None
    # The arm never moved in this fake, so it is 5 cm from the goal on the left only.
    np.testing.assert_allclose(second["tracking_error_m"], [0.05, 0.0], atol=1e-6)


@pytest.fixture
def code_server():
    """Answers like a code-server /proxy/ route: JSON with the session cookie, else login."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Cookie") == "code-server-session=good":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"choices": []}')
            else:
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html>code-server login</html>")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/proxy/8000/v1"
    server.shutdown()
    server.server_close()


def test_proxied_server_receives_the_session_cookie(code_server):
    assert vision_llm.request_chat(code_server, {}, 5, "good") == {"choices": []}


@pytest.mark.parametrize("cookie", [None, "stale"])
def test_login_redirect_names_the_cookie_instead_of_a_json_error(code_server, cookie):
    with pytest.raises(RuntimeError, match="VISION_LLM_COOKIE"):
        vision_llm.request_chat(code_server, {}, 5, cookie)


def test_policy_sends_the_cookie_from_the_environment(monkeypatch):
    monkeypatch.setenv("VISION_LLM_COOKIE", "good")
    seen = []

    def recording(url, body, timeout, cookie, **_):
        seen.append(cookie)
        return holding(body)

    monkeypatch.setattr(vision_llm, "request_chat", recording)
    policy = vision_llm.make()
    policy.reset(observation())
    policy.act(observation())
    assert seen == ["good"]


# --- hosted APIs (VISION_LLM_API_KEY, VISION_LLM_DIALECT, retries) ---------------------------------


@pytest.fixture
def api_server():
    """A chat endpoint that records each request's headers and body and answers from a script:
    each entry is (status, payload); the last entry repeats."""

    state = {"script": [(200, {"choices": []})], "seen": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["seen"].append((dict(self.headers), body))
            status, payload = state["script"][min(len(state["seen"]), len(state["script"])) - 1]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if state.get("cost"):
                self.send_header("x-litellm-response-cost", state["cost"])
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()
    server.server_close()


def test_api_key_is_sent_as_a_bearer_token(api_server):
    assert vision_llm.request_chat(api_server["url"], {}, 5, api_key="sk-test") == {"choices": []}
    headers, _ = api_server["seen"][0]
    assert headers["Authorization"] == "Bearer sk-test"


def test_throttled_requests_are_retried(api_server, monkeypatch):
    monkeypatch.setattr(vision_llm, "RETRY_WAIT", 0.0)
    api_server["script"] = [(429, {"error": "slow down"}), (503, {}), (200, {"choices": ["ok"]})]
    assert vision_llm.request_chat(api_server["url"], {}, 5, retries=3) == {"choices": ["ok"]}
    assert len(api_server["seen"]) == 3


@pytest.mark.parametrize("status", [401, 422])
def test_refused_requests_are_not_retried(api_server, monkeypatch, status):
    # e.g. a wrong key, or a key whose budget is spent: retrying cannot help.
    monkeypatch.setattr(vision_llm, "RETRY_WAIT", 0.0)
    api_server["script"] = [(status, {"error": "budget_exceeded"})]
    with pytest.raises(RuntimeError, match=f"server {status}"):
        vision_llm.request_chat(api_server["url"], {}, 5, retries=3)
    assert len(api_server["seen"]) == 1


def test_a_content_filter_refusal_is_its_own_error(api_server, monkeypatch):
    # Azure OpenAI's filter: HTTP 400 with code content_filter. Not retried here (the agent asks
    # again on its next turn, with new images), and told apart from a broken request.
    monkeypatch.setattr(vision_llm.time, "sleep", lambda seconds: None)
    api_server["script"] = [(400, {"error": {"code": "content_filter", "message": "filtered"}})]
    with pytest.raises(vision_llm.ContentFiltered):
        vision_llm.request_chat(api_server["url"], {}, 5, retries=3)
    # The image filter answers with another code (sports B3 lost both seeds to it on 2026-11-11).
    api_server["script"] = [(400, {"error": {"code": "content_policy_violation", "message": "Image processing blocked due to content policy violation."}})]
    with pytest.raises(vision_llm.ContentFiltered):
        vision_llm.request_chat(api_server["url"], {}, 5, retries=3)
    api_server["script"] = [(400, {"error": {"code": "invalid_request", "message": "bad"}})]
    with pytest.raises(RuntimeError) as other:
        vision_llm.request_chat(api_server["url"], {}, 5, retries=3)
    assert not isinstance(other.value, vision_llm.ContentFiltered)


def test_a_timed_out_request_is_sent_again_only_when_asked(monkeypatch):
    attempts = []

    def stalled(request, timeout):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise TimeoutError("The read operation timed out")
        raise RuntimeError("stop")  # a third attempt was made: enough for the test

    monkeypatch.setattr(vision_llm.urllib.request, "urlopen", stalled)
    monkeypatch.setattr(vision_llm.time, "sleep", lambda seconds: None)
    # A local server that times out is overloaded: asking again would only lengthen its queue.
    with pytest.raises(RuntimeError, match="timed out"):
        vision_llm.request_chat("http://x.invalid/v1", {}, 5, retries=3)
    assert len(attempts) == 1
    # A hosted API that times out has stalled: the request goes out again.
    attempts.clear()
    with pytest.raises(RuntimeError, match="stop"):
        vision_llm.request_chat("http://x.invalid/v1", {}, 5, retries=3, retry_timeouts=True)
    assert len(attempts) == 3


def test_retries_give_up_after_the_last_one(api_server, monkeypatch):
    monkeypatch.setattr(vision_llm, "RETRY_WAIT", 0.0)
    api_server["script"] = [(503, {})]
    with pytest.raises(RuntimeError, match="server 503"):
        vision_llm.request_chat(api_server["url"], {}, 5, retries=2)
    assert len(api_server["seen"]) == 3


def test_default_request_is_the_vllm_one(monkeypatch):
    for name in ("VISION_LLM_DIALECT", "VISION_LLM_MAX_TOKENS", "VISION_LLM_THINKING"):
        monkeypatch.delenv(name, raising=False)
    policy = vision_llm.make()
    policy.reset(observation())
    body = policy.request_body(observation())
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["thinking_token_budget"] == 1024 and body["max_tokens"] == 2048
    assert body["temperature"] == 0.0 and body["response_format"]["type"] == "json_schema"


def test_openai_dialect_sends_only_standard_fields(monkeypatch):
    monkeypatch.setenv("VISION_LLM_DIALECT", "openai")
    monkeypatch.setenv("VISION_LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("VISION_LLM_MAX_TOKENS", "4096")
    policy = vision_llm.make()
    policy.reset(observation())
    body = policy.request_body(observation())
    assert "chat_template_kwargs" not in body and "thinking_token_budget" not in body
    assert body["reasoning_effort"] == "low" and body["max_tokens"] == 4096


def test_unknown_dialect_is_refused(monkeypatch):
    monkeypatch.setenv("VISION_LLM_DIALECT", "anthropic")
    with pytest.raises(ValueError, match="VISION_LLM_DIALECT"):
        vision_llm.make()


def test_policy_sends_the_api_key_and_retries_from_the_environment(monkeypatch):
    monkeypatch.setenv("VISION_LLM_API_KEY", "sk-env")
    monkeypatch.setenv("VISION_LLM_RETRIES", "5")
    seen = []

    def recording(url, body, timeout, cookie, **options):
        seen.append(options)
        return holding(body)

    monkeypatch.setattr(vision_llm, "request_chat", recording)
    policy = vision_llm.make()
    policy.reset(observation())
    policy.act(observation())
    assert seen == [{"api_key": "sk-env", "retries": 5}]


def test_the_gateways_billed_cost_comes_with_the_reply(api_server):
    api_server["cost"] = "0.0123"
    reply = vision_llm.request_chat(api_server["url"], {}, 5, api_key="sk-test")
    assert reply == {"choices": [], "response_cost_usd": pytest.approx(0.0123)}


def test_request_log_records_the_billed_cost(monkeypatch, tmp_path):
    log = tmp_path / "llm.jsonl"
    monkeypatch.setenv("VISION_LLM_LOG", str(log))

    def billed(body):
        reply = holding(body)
        reply["response_cost_usd"] = 0.02
        reply["usage"]["prompt_tokens_details"] = {"cached_tokens": 700}
        return reply

    policy, _ = make_policy(monkeypatch, billed)
    policy.act(observation())
    row = json.loads(log.read_text().splitlines()[0])
    assert row["cost_usd"] == 0.02 and row["cached_prompt_tokens"] == 700
