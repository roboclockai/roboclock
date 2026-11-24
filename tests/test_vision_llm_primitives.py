"""Primitive agent: tool definitions, call checking, the tool-calling conversation, and the skills
run against a kinematic arm."""

import json

import gymnasium as gym
import numpy as np
import pytest
from test_vision_llm_policy import MOUNTS, tool_pose_vector

from roboclock.envs.lab.robot import ARM_DOF, HOME, JOINT_HIGH, JOINT_LOW
from roboclock.policies import vision_llm
from roboclock.policies.vision_llm_primitives import (
    MAX_CALLS,
    client,
    geometry,
    make,
    perception,
    prompt,
    settings,
    skills,
    tools,
)
from roboclock.policies.vision_llm_primitives import policy as agent
from roboclock.policies.vision_llm_primitives.tools import check_call
from roboclock.policies.x5_kinematics import X5Kinematics

DT = 1.0 / 50


class Bench:
    """Arms that follow their joint commands exactly, with an optional floor the tool point
    cannot pass (contact), an object the closing fingers stop on, and a head image whose
    centre turns red at `red_at` seconds."""

    def __init__(self, floor=-np.inf, item=None, item_width=0.03, red_at=np.inf, depth=None, delay=0, rate=np.inf):
        self.joints = np.tile(HOME, 2).astype(np.float32)
        self.delay, self.pending = delay, []
        self.rate = rate  # rad/s each joint moves toward its command at most (a rig's joint speed cap)
        self.arms = [X5Kinematics(base=m) for m in MOUNTS]
        self.floor, self.item, self.item_width, self.red_at = floor, item, item_width, red_at
        self.depth = depth
        self.time = 0.0
        self.width = np.array([0.08, 0.08])

    def tool(self, i, joints=None):
        joints = self.joints if joints is None else joints
        return self.arms[i].tool_pose(joints[8 * i : 8 * i + ARM_DOF].astype(float))

    def observation(self):
        pose = [tool_pose_vector(t[:3, 3], t[:3, :3]) for t in (self.tool(0), self.tool(1))]
        rgb = np.full((48, 64, 3), 100, dtype=np.uint8)
        if self.time >= self.red_at:
            rgb[18:30, 26:38] = (230, 20, 20)
        obs = {
            "rgb": rgb,
            "left_wrist_rgb": np.zeros((24, 32, 3), dtype=np.uint8),
            "right_wrist_rgb": np.zeros((24, 32, 3), dtype=np.uint8),
            "joint_positions": self.joints.copy(),
            "ee_pose": np.concatenate(pose).astype(np.float32),
            "gripper_width": self.width.astype(np.float32),
            "instruction": "test task",
            "sim_time": np.array([self.time], dtype=np.float32),
        }
        if self.depth is not None:
            obs["depth"] = self.depth
        return obs

    def apply(self, action):
        # A delay makes the arms follow `delay` steps behind their commands, as drives that
        # settle behind a moving target do.
        self.pending.append(np.asarray(action, dtype=np.float32))
        action = self.pending.pop(0) if len(self.pending) > self.delay else self.joints.copy()
        for i in range(2):
            arm = slice(8 * i, 8 * i + ARM_DOF)
            candidate = self.joints.copy()
            candidate[arm] = self.joints[arm] + np.clip(action[arm] - self.joints[arm], -self.rate * DT, self.rate * DT)
            if self.tool(i, candidate)[2, 3] >= self.floor:
                self.joints[arm] = candidate[arm]
            fingers = float(action[8 * i + ARM_DOF])
            self.joints[8 * i + ARM_DOF : 8 * i + 8] = fingers
            width = 2 * fingers
            near = self.item is not None and np.linalg.norm(self.tool(i)[:3, 3] - self.item) < 0.02
            self.width[i] = max(width, self.item_width) if near else width
        self.time += DT


def answer(*calls, text=None):
    """A chat reply that calls tools, as an OpenAI-compatible server sends it."""
    tool_calls = [
        {"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for index, (name, args) in enumerate(calls)
    ]
    message = {"role": "assistant", "content": text, "tool_calls": tool_calls}
    return {"choices": [{"message": message, "finish_reason": "tool_calls"}], "usage": {}}


def spaces(finger_min=0.0, open_width=0.088):
    """Action and observation spaces of a rig: its lowest finger target and how wide its jaws open."""
    low = np.tile(JOINT_LOW, 2).astype(np.float32)
    low[[ARM_DOF, ARM_DOF + 1, 8 + ARM_DOF, 8 + ARM_DOF + 1]] = finger_min
    action = gym.spaces.Box(low, np.tile(JOINT_HIGH, 2).astype(np.float32), dtype=np.float32)
    observation = gym.spaces.Dict({"gripper_width": gym.spaces.Box(0.0, open_width, (2,), dtype=np.float32)})
    return action, observation


def start(monkeypatch, bench, replies, finger_min=None, open_width=0.088):
    """A policy on `bench` whose model answers with `replies` (then waits); (policy, requests sent)."""
    sent = []

    def fake_request(url, body, timeout, cookie, **_):
        sent.append(body)
        reply = replies.pop(0) if replies else answer(("wait", {"seconds": 1.0}))
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(agent, "request_chat", fake_request)
    policy = make()
    if finger_min is not None or open_width != 0.088:
        policy.set_spaces(*spaces(finger_min or 0.0, open_width))
    policy.reset(bench.observation())
    return policy, sent


def run(monkeypatch, bench, replies, steps, finger_min=None, open_width=0.088):
    """Run the policy on `bench` for `steps` control steps."""
    policy, sent = start(monkeypatch, bench, replies, finger_min, open_width)
    for _ in range(steps):
        bench.apply(policy.act(bench.observation()))
    return policy, sent


def tool_results(body):
    """What the latest finished turn's calls returned, read from a later request's messages."""
    out = []
    for message in reversed(body["messages"][:-1]):
        if message["role"] != "tool":
            break
        out.append(message["content"])
    return out[::-1]


def last_results(sent, request=1):
    """The results of the first turn's calls (as the second request carries them), in one string."""
    return " | ".join(tool_results(sent[request]))


def turn_text(body):
    """The measurements text of a request's current observation."""
    return body["messages"][-1]["content"][-1]["text"]


def grasp_point(bench, arm=1):
    tool = bench.tool(arm)
    forward = MOUNTS[arm][:3, 0]
    return tool[:3, 3] + 0.05 * forward + np.array([0.0, 0.0, -0.12])


# --- the tools as the API gets them ------------------------------------------------------------


def test_every_primitive_is_a_tool_with_described_and_bounded_arguments():
    definitions = tools.tool_definitions()
    assert [t["function"]["name"] for t in definitions] == list(tools.SPEC)
    assert set(tools.SPEC) == {
        "move", "move_by", "home", "grasp", "place", "press", "push", "pull", "turn", "together", "wait", "configure",
    }
    for definition in definitions:
        function = definition["function"]
        assert definition["type"] == "function" and len(function["description"]) > 40
        required, optional = tools.SPEC[function["name"]]
        parameters = function["parameters"]
        assert parameters["required"] == list(required)
        assert set(parameters["properties"]) == {*required, *optional}
        for key, schema in parameters["properties"].items():
            assert schema.get("description"), (function["name"], key)
            if schema["type"] == "number":
                table = settings.SETTINGS if function["name"] == "configure" else settings.CALL
                assert schema["minimum"] == table[key].low
                assert schema.get("maximum", np.inf) == table[key].high
    # No tool carries more than a handful of arguments; the rest are settings. The hand's tilt
    # and a turn's hinge are one object argument each for that reason.
    assert max(len(t["function"]["parameters"]["properties"]) for t in definitions[:-1]) <= 11
    turn = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in definitions}["turn"]
    assert set(turn["tilt"]["properties"]) == set(settings.TILT) == {"pitch", "roll", "heading"}
    assert turn["tilt"]["properties"]["pitch"]["maximum"] == pytest.approx(np.pi / 2)
    assert set(turn["hinge"]["properties"]) == {"pivot", "axis"}
    # A server whose guided decoding takes no numeric bounds gets them in the descriptions.
    plain = json.dumps(tools.tool_definitions(bounds=False))
    assert '"minimum"' not in plain and "Range 0.0005 to 0.03." in plain


def test_every_quantity_is_the_models_to_set_as_an_argument_or_a_setting():
    arguments = {key for required, optional in tools.SPEC.values() for key in (*required, *optional)}
    for key in settings.CALL:
        assert key in arguments, key
    assert set(tools.SPEC["configure"][1]) == set(settings.SETTINGS)
    for key in ("speed", "approach_speed", "turn_speed", "gripper_time", "standoff", "press_standoff", "contact",
                "tolerance", "arrival", "arm_clearance", "settle", "min_opening", "clearance", "retreat",
                "push_offset", "rim_radius", "watch_timeout", "watch_change", "watch_radius"):
        assert settings.SETTINGS[key].default is not None
    assert settings.defaults()["speed"] == vision_llm.TOOL_SPEED


def test_calls_are_checked_with_messages_for_the_model():
    with pytest.raises(ValueError, match="unknown tool 'actuate'"):
        check_call("actuate", {"arm": "left"})
    with pytest.raises(ValueError, match=r"grasp needs \['arm'\]"):
        check_call("grasp", {"object": {"u": 1, "v": 2}})
    with pytest.raises(ValueError, match="0..1000"):
        check_call("grasp", {"arm": "left", "object": {"u": 1200, "v": 2}})
    with pytest.raises(ValueError, match=r"push needs \['by'\]"):
        check_call("push", {"arm": "left", "target": {"u": 1, "v": 2}})
    with pytest.raises(ValueError, match=r"turn needs \['angle'\]"):
        check_call("turn", {"arm": "right", "target": {"u": 588, "v": 585}, "yaw": 0.0})
    with pytest.raises(ValueError, match="place.mode must be one of"):
        check_call("place", {"arm": "left", "target": {"u": 1, "v": 2}, "mode": "onto"})
    with pytest.raises(ValueError, match="pixel"):
        check_call("wait", {"watch": {"xyz": [0, 0, 1]}})
    with pytest.raises(ValueError, match="exactly one of seconds, until_t or watch"):
        check_call("wait", {})
    with pytest.raises(ValueError, match="exactly one of seconds, until_t or watch"):
        check_call("wait", {"seconds": 1, "until_t": 5})
    with pytest.raises(ValueError, match=r"unexpected arguments \['timeout', 'watch'\]"):
        check_call("move_by", {"arm": "left", "by": [0, 0, 0.1], "watch": {"u": 1, "v": 1}, "timeout": 3})
    with pytest.raises(ValueError, match="must be a JSON object"):
        check_call("wait", [1, 2])
    assert check_call("place", {"arm": "right", "target": {"xyz": [0, 0, 1]}})["mode"] == "on"
    assert check_call("move", {"arm": "left", "to": {"u": 5, "v": 6}, "when": {"u": 1, "v": 2, "timeout": 3}})["when"] == {
        "watch": {"u": 1.0, "v": 2.0}, "timeout": 3.0,
    }


def test_a_quantity_outside_its_range_is_refused_not_clamped(monkeypatch):
    grasp = {"arm": "left", "object": {"xyz": [0.3, 0.2, 0.9]}}
    for key, value in [("depth", 0.2), ("squeeze", 0.05), ("lift", -0.1), ("opening", 0.2)]:
        with pytest.raises(ValueError, match=f"grasp.{key} = .* is outside its range"):
            check_call("grasp", {**grasp, key: value})
    # The upper ends are the values the model is told: a fully open gripper is 0.088 m, not a
    # float32 hair under it (refused twice in a rerun, 2026-11-10).
    assert check_call("grasp", {**grasp, "opening": 0.088})["opening"] == 0.088
    with pytest.raises(ValueError, match="move.speed = 3 is outside"):
        check_call("move", {"arm": "left", "to": {"u": 1, "v": 2}, "speed": 3.0})
    with pytest.raises(ValueError, match="wait.timeout = 90 is outside"):
        check_call("wait", {"watch": {"u": 1, "v": 2}, "timeout": 90})
    with pytest.raises(ValueError, match="configure.contact = 0.5 is outside its range"):
        check_call("configure", {"contact": 0.5})
    with pytest.raises(ValueError, match="grasp.when.timeout"):
        check_call("grasp", {**grasp, "when": {"u": 1, "v": 2, "timeout": 900}})
    # The call is answered with the refusal, and the calls after it do not run.
    bench = Bench()
    start_pose = bench.tool(0)[:3, 3].copy()
    calls = [("grasp", {**grasp, "depth": 0.2}), ("move_by", {"arm": "left", "by": [0, 0, 0.05]})]
    policy, sent = run(monkeypatch, bench, [answer(*calls)], steps=40)
    refused, skipped = tool_results(sent[1])
    assert refused == "REFUSED: grasp.depth = 0.2 is outside its range (-0.05 to 0.15 m)"
    assert skipped == "not run: an earlier call of this turn failed"
    np.testing.assert_allclose(bench.tool(0)[:3, 3], start_pose, atol=1e-6)


def test_arguments_written_only_to_fill_the_schema_are_read_as_not_given():
    # Decoders behind some gateways write null, or 0 where 0 is no value, for fields they do not mean.
    args = check_call("grasp", {"arm": "left", "object": {"u": 1, "v": 2}, "squeeze": None, "opening": 0, "lift": 0, "when": {}})
    assert args == {"arm": "left", "object": {"u": 1.0, "v": 2.0}, "lift": 0.0}
    assert check_call("configure", {"speed": 0, "contact": None, "retreat": 0.05}) == {"retreat": 0.05}
    # A point given both ways: a zero half is dropped, and of two real ones the pixel wins.
    assert check_call("grasp", {"arm": "left", "object": {"u": 484, "v": 693, "xyz": [0.0, 0.0, 0.0]}})["object"] == {"u": 484.0, "v": 693.0}
    assert check_call("place", {"arm": "right", "target": {"u": 500, "v": 560, "xyz": [0.35, 0.0, 0.82]}})["target"] == {"u": 500.0, "v": 560.0}
    assert check_call("move", {"arm": "left", "to": {"u": 0, "v": 0, "xyz": [0.3, 0.2, 0.9]}})["to"] == {"xyz": [0.3, 0.2, 0.9]}
    assert check_call("wait", {"watch": {"u": 715, "v": 517, "xyz": [0.47, -0.05, 0.85]}})["watch"] == {"u": 715.0, "v": 517.0}
    assert check_call("press", {"arm": "right", "target": {}, "depth": 0.01}) == {"arm": "right", "depth": 0.01}
    # A lone zero pixel, a zero offset and a zero yaw are values and stay.
    assert check_call("place", {"arm": "left", "target": {"u": 0, "v": 0}})["target"] == {"u": 0.0, "v": 0.0}
    assert check_call("move_by", {"arm": "left", "by": [0, 0, 0]})["by"] == [0.0, 0.0, 0.0]
    assert check_call("move", {"arm": "left", "to": {"u": 1, "v": 2}, "yaw": 0})["yaw"] == 0.0
    # A real value in a wrong place is still refused.
    with pytest.raises(ValueError, match="unexpected arguments"):
        check_call("press", {"arm": "right", "target": {"u": 540, "v": 600}, "to": {"u": 7, "v": 9}})


# --- the request and the conversation -----------------------------------------------------------


def test_the_request_is_a_native_tool_calling_one(monkeypatch):
    policy, sent = run(monkeypatch, Bench(), [], steps=1)
    body = sent[0]
    assert body["tools"] == tools.tool_definitions() and body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is True and "response_format" not in body
    system, user = body["messages"]
    assert system["role"] == "system" and system["content"].endswith("TASK. test task")
    assert [part["type"] for part in user["content"]] == ["text", "image_url"] * 3 + ["text"]
    # The tools are described in their definitions, not in the system prompt.
    for name in ("grasp", "press_standoff", "rim_radius"):
        assert name not in system["content"]


def test_reasoning_is_stated_in_every_request(monkeypatch):
    def body(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        policy, sent = run(monkeypatch, Bench(), [], steps=1)
        for key in env:
            monkeypatch.delenv(key)
        return sent[0]

    local = body()
    assert local["chat_template_kwargs"] == {"enable_thinking": True} and local["thinking_token_budget"] == 1024
    assert local["temperature"] == 0.0 and local["max_tokens"] == client.ANSWER_TOKENS + 1024
    # A hosted reasoning model: an explicit effort, and no temperature (it takes none).
    hosted = body(VISION_LLM_DIALECT="openai")
    assert hosted["reasoning_effort"] == client.REASONING_EFFORT == "medium"
    assert "temperature" not in hosted and "chat_template_kwargs" not in hosted and "thinking_token_budget" not in hosted
    assert body(VISION_LLM_DIALECT="openai", VISION_LLM_REASONING_EFFORT="high")["reasoning_effort"] == "high"
    plain = body(VISION_LLM_DIALECT="openai", VISION_LLM_THINKING="0")
    assert plain["temperature"] == 0.0 and "reasoning_effort" not in plain


def test_each_call_is_answered_by_a_tool_message_and_earlier_turns_stay(monkeypatch):
    bench = Bench()
    replies = [
        answer(("move_by", {"arm": "right", "by": [0, 0, 0.02]}), ("wait", {"seconds": 0.2}), text="lift a little"),
        answer(("move_by", {"arm": "right", "by": [0, 0, -0.02]})),
    ]
    policy, sent = run(monkeypatch, bench, replies, steps=120)
    roles = [m["role"] for m in sent[2]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "user", "assistant", "tool", "user"]
    first = sent[2]["messages"][2]
    assert first["content"] == "lift a little"
    assert [c["function"]["name"] for c in first["tool_calls"]] == ["move_by", "wait"]
    assert [m["tool_call_id"] for m in sent[2]["messages"][3:5]] == ["call_0", "call_1"]
    assert "measured tool point" in sent[2]["messages"][3]["content"]
    # Views: the current turn has all three, the previous turn its head view, older turns none.
    images = [sum(part["type"] == "image_url" for part in m["content"]) if isinstance(m["content"], list) else 0
              for m in sent[2]["messages"] if m["role"] == "user"]
    assert images == [0, 1, 3]
    assert "no longer shown" in sent[2]["messages"][1]["content"]


def test_the_conversation_is_trimmed_in_blocks(monkeypatch):
    conversation = prompt.Conversation("system", keep=6)
    for index in range(20):
        conversation.add({"text": f"turn {index}", "head": None, "assistant": {"role": "assistant", "content": ""}, "tools": []})
        assert 3 <= len(conversation.turns) <= 6 or index < 3
    # Between two trims the leading messages do not change, which a prompt cache needs.
    before = conversation.messages("now")[:4]
    conversation.add({"text": "one more", "head": None, "assistant": {"role": "assistant", "content": ""}, "tools": []})
    if len(conversation.turns) > 3:
        assert conversation.messages("now")[:4] == before
    monkeypatch.setenv("VISION_LLM_HISTORY", "0")
    policy, sent = run(monkeypatch, Bench(), [answer(("wait", {"seconds": 0.2}))] * 3, steps=60)
    assert [m["role"] for m in sent[2]["messages"]] == ["system", "user"]  # every turn stands alone


def test_a_conversation_too_long_for_the_model_drops_old_turns_and_asks_again(monkeypatch):
    overflow = RuntimeError("Vision LLM server 400: b'This model's maximum context length is 16384 tokens'")
    replies = [answer(("wait", {"seconds": 0.1}))] * 4 + [overflow, overflow]
    policy, sent = run(monkeypatch, Bench(), replies, steps=60)
    past = [sum(m["role"] == "assistant" for m in body["messages"]) for body in sent[:7]]
    assert past == [0, 1, 2, 3, 4, 2, 1]  # the fifth turn was asked three times, with 4, 2 and 1 past turns
    assert policy.requests >= 5
    # Another error is not swallowed, and neither is an overflow with nothing left to drop.
    with pytest.raises(RuntimeError, match="401"):
        run(monkeypatch, Bench(), [RuntimeError("Vision LLM server 401: unauthorized")], steps=5)
    with pytest.raises(RuntimeError, match="context length"):
        run(monkeypatch, Bench(), [overflow], steps=5)


def test_a_reply_without_a_tool_call_is_told_and_costs_a_short_hold(monkeypatch):
    silent = {"choices": [{"message": {"role": "assistant", "content": "I will wait."}, "finish_reason": "stop"}], "usage": {}}
    policy, sent = run(monkeypatch, Bench(), [silent], steps=40)
    assert len(sent) == 2  # 0.5 s of hold, then the next turn
    assert "Your previous reply called no tool" in turn_text(sent[1])
    assert [m["role"] for m in sent[1]["messages"]] == ["system", "user", "assistant", "user"]


def test_calls_past_the_limit_and_broken_arguments_are_answered_too(monkeypatch):
    many = answer(*[("wait", {"seconds": 0.1})] * (MAX_CALLS + 1))
    policy, sent = run(monkeypatch, Bench(), [many], steps=60)
    results = tool_results(sent[1])
    assert len(results) == MAX_CALLS + 1 and results[-1] == f"not run: at most {MAX_CALLS} calls run per turn"
    broken = answer(("wait", {"seconds": 0.1}))
    broken["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"seconds": '
    policy, sent = run(monkeypatch, Bench(), [broken], steps=60)
    assert tool_results(sent[1])[0].startswith("REFUSED: the arguments are not valid JSON")


def test_a_request_the_provider_filters_costs_a_turn_not_the_episode(monkeypatch):
    bench = Bench()
    start_pose = bench.tool(1)[:3, 3].copy()
    replies = [vision_llm.ContentFiltered("400 content_filter"), answer(("move_by", {"arm": "right", "by": [0, 0, 0.05]}))]
    policy, sent = run(monkeypatch, bench, replies, steps=120)
    assert policy.requests >= 3
    assert "content filter refused the request" in turn_text(sent[1])
    np.testing.assert_allclose(bench.tool(1)[:3, 3], start_pose + [0, 0, 0.05], atol=0.002)


def test_the_turn_text_gives_the_clock_and_measurements_without_naming_a_state(monkeypatch):
    # Some tasks start with the fingers almost closed (hospital A1: 4 mm per finger).
    bench = Bench()
    bench.joints[ARM_DOF:8] = bench.joints[8 + ARM_DOF :] = 0.004
    bench.width[:] = 0.008
    policy, sent = run(monkeypatch, bench, [answer(("wait", {"seconds": 0.5}))], steps=40)
    first, second = turn_text(sent[0]), turn_text(sent[1])
    assert "Time now: 0.00 s on the task clock (this is your first turn)" in first
    assert "Time now: 0.50 s on the task clock (your previous turn was at 0.00 s)" in second
    assert "measured gripper opening 0.008 m (commanded 0.008 m)" in first
    everything = json.dumps(sent[1]["messages"]) + json.dumps(sent[1]["tools"])
    for word in ("holding (", "NOT holding", "closed on something", "closed on nothing", "HINTS"):
        assert word not in everything


def test_the_policy_takes_the_finger_limit_and_the_jaw_range_from_the_spaces(monkeypatch):
    policy, sent = run(monkeypatch, Bench(), [], steps=1)
    system = sent[0]["messages"][0]["content"]
    assert policy.skills.finger_min == 0.0 and "below 0.000 m" in system and "at most 0.088 m" in system
    # A household-like rig: fingers commandable past the closed stop, jaws that open 60.3 mm.
    policy, sent = run(monkeypatch, Bench(), [], steps=1, finger_min=-0.03, open_width=0.06028)
    system = sent[0]["messages"][0]["content"]
    assert policy.skills.finger_min == pytest.approx(-0.03) and "below -0.030 m" in system
    assert policy.skills.finger_max == pytest.approx(0.03014) and "at most 0.060 m" in system
    grasp = next(t for t in sent[0]["tools"] if t["function"]["name"] == "grasp")
    assert grasp["function"]["parameters"]["properties"]["opening"]["maximum"] == pytest.approx(0.06028)
    with pytest.raises(ValueError, match=r"grasp.opening = 0.07 is outside its range \(0.002 to 0.06028 m\)"):
        check_call("grasp", {"arm": "left", "object": {"u": 1, "v": 2}, "opening": 0.07}, 0.06028)
    # "Fully open" commands the widest opening, not the lab's 0.044 m per finger.
    assert policy.skills.gripper_target({"gripper": "open"}) == pytest.approx(0.03014)

    class Env:
        action_space, observation_space = spaces(-0.03, 0.06028)

    from roboclock.policies.base import load_policy

    loaded = load_policy("roboclock.policies.vision_llm_primitives:make", Env())
    assert loaded.finger_min == pytest.approx(-0.03) and loaded.open_width == pytest.approx(0.06028)


def test_policy_reports_its_model_and_each_querys_output_tokens(monkeypatch):
    # --inference-latency ideal charges these at the model's surveyed speed.
    monkeypatch.delenv("VISION_LLM_MODEL", raising=False)
    reply = answer(("wait", {"seconds": 1.0}))
    reply["usage"] = {"prompt_tokens": 2600, "completion_tokens": 1100}
    policy, _ = run(monkeypatch, Bench(), [reply], steps=1)
    assert policy.inference_model == "Qwen/Qwen3.8-27B"
    assert policy.last_output_tokens == 1100


def test_query_cap_stops_model_turns_and_holds(monkeypatch):
    monkeypatch.setenv("VISION_LLM_MAX_QUERIES", "3")
    bench = Bench()
    policy, sent = run(monkeypatch, bench, [answer(("wait", {"seconds": 0.0}))] * 100, steps=50)
    assert len(sent) == 3 and policy.requests == 3
    # Past the cap the arms hold the last command.
    held = policy.act(bench.observation())
    assert np.allclose(held, policy.act(bench.observation()))
    assert len(sent) == 3


def test_each_turn_lasts_at_least_min_turn(monkeypatch):
    bench = Bench()
    policy, sent = run(monkeypatch, bench, [answer(("wait", {"seconds": 0.0}))] * 100, steps=50)
    # 1 s of control at 50 Hz, turns of at least 0.1 s: at most 10 model turns.
    assert len(sent) <= round(1.0 / agent.MIN_TURN)


def test_primitive_agent_sends_the_api_settings(monkeypatch):
    monkeypatch.setenv("VISION_LLM_API_KEY", "sk-env")
    monkeypatch.setenv("VISION_LLM_DIALECT", "openai")
    seen = []

    def recording(url, body, timeout, cookie, **options):
        seen.append((body, options))
        return answer(("wait", {"seconds": 1.0}))

    monkeypatch.setattr(agent, "request_chat", recording)
    policy = make()
    bench = Bench()
    policy.reset(bench.observation())
    policy.act(bench.observation())
    assert seen[0][1] == {"api_key": "sk-env", "retries": 3, "retry_timeouts": False}


def test_the_log_has_a_row_per_request(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_LLM_LOG", str(tmp_path / "q.jsonl"))
    reply = answer(("wait", {"seconds": 0.2}), text="waiting")
    reply["usage"] = {"prompt_tokens": 9000, "completion_tokens": 300, "completion_tokens_details": {"reasoning_tokens": 120},
                      "prompt_tokens_details": {"cached_tokens": 7000}}
    run(monkeypatch, Bench(), [reply], steps=30)
    rows = [json.loads(line) for line in (tmp_path / "q.jsonl").read_text().splitlines()]
    assert rows[0]["calls"] == ['wait(seconds=0.2)'] and rows[0]["answer"] == "waiting" and rows[0]["parse_error"] is None
    assert (rows[0]["reasoning_tokens"], rows[0]["cached_prompt_tokens"], rows[0]["finish_reason"]) == (120, 7000, "tool_calls")
    assert rows[1]["shown_results"] == ["wait(seconds=0.2): took 0.2 s, now t=0.20 s"]


# --- the skills --------------------------------------------------------------------------------


def test_approach_frames_are_rotations_with_the_promised_axes():
    for yaw in (0.0, 0.7, -2.0):
        rotation = geometry.top_rotation(yaw)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        np.testing.assert_allclose(rotation[:, 2], [0, 0, -1])
        np.testing.assert_allclose(rotation[:, 1], [np.cos(yaw), np.sin(yaw), 0], atol=1e-12)
    np.testing.assert_allclose(geometry.top_rotation(-np.pi / 2), vision_llm.TOOL_DOWN, atol=1e-12)
    rotation = geometry.front_rotation([0.6, 0.8, 0.3])
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation[:, 2], [0.6, 0.8, 0.0], atol=1e-12)
    assert abs(rotation[2, 1]) < 1e-12  # the jaws close horizontally


def test_grasp_closes_on_the_object_and_lifts(monkeypatch):
    bench = Bench()
    point = grasp_point(bench)
    bench.item = point + [0, 0, -0.02]
    call = ("grasp", {"arm": "right", "object": {"xyz": point.tolist()}, "depth": 0.02, "lift": 0.08})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=500)
    text = last_results(sent)
    # Measurements only: the opening after closing and after the lift, never "holding".
    assert "closed: opening 0.030 m, squeezed 0.010 m per finger" in text
    # The bench's object does not come up with the hand, and the reading after the lift shows it:
    # the fingers close to their commanded 0.010 m once it is gone.
    assert "after the lift: measured tool point" in text and "opening 0.010 m" in text.split("after the lift")[1]
    assert "holding" not in text
    assert bench.tool(1)[2, 3] == pytest.approx(point[2] - 0.02 + 0.08, abs=0.01)
    np.testing.assert_allclose(bench.tool(1)[:3, 2], [0, 0, -1], atol=0.02)  # fingers down
    assert bench.tool(0)[:3, 3] == pytest.approx(Bench().tool(0)[:3, 3], abs=1e-6)  # left held


def test_a_grasp_on_nothing_fails_and_skips_the_rest(monkeypatch):
    bench = Bench()
    point = grasp_point(bench)
    calls = [
        ("grasp", {"arm": "right", "object": {"xyz": point.tolist()}}),
        ("place", {"arm": "right", "target": {"xyz": (point + [0, 0.1, 0]).tolist()}}),
    ]
    policy, sent = run(monkeypatch, bench, [answer(*calls)], steps=500)
    grasp, place = tool_results(sent[1])
    assert grasp.startswith("FAILED: ") and "the fingers met: opening 0.000 m is under min_opening 0.003 m" in grasp
    assert place == "not run: an earlier call of this turn failed"


def test_a_guarded_approach_stops_on_contact(monkeypatch):
    bench = Bench()
    point = grasp_point(bench)
    bench.floor = point[2] + 0.03  # something 3 cm above the clicked point
    call = ("push", {"arm": "right", "target": {"xyz": point.tolist()}, "by": [0.05, 0, 0]})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=600)
    assert "touched down early" in last_results(sent)


def test_a_watched_pixel_triggers_the_call_when_it_changes(monkeypatch):
    bench = Bench(red_at=1.0)
    target = bench.tool(1)[:3, 3] + [0.0, 0.0, -0.06]
    call = ("press", {"arm": "right", "target": {"xyz": target.tolist()}, "when": {"u": 500, "v": 500, "timeout": 5}})
    start_pose = bench.tool(1)[:3, 3].copy()
    moved_at = None
    policy, sent = start(monkeypatch, bench, [answer(call)])
    for _ in range(300):
        bench.apply(policy.act(bench.observation()))
        if moved_at is None and np.linalg.norm(bench.tool(1)[:3, 3] - start_pose) > 1e-4:
            moved_at = bench.time
    assert 1.0 < moved_at <= 1.0 + skills.WATCH_EVERY + 2 * DT
    text = last_results(sent)
    assert "triggered at t=1.0" in text and "pressed" in text


def test_a_watch_that_times_out_fails_the_call(monkeypatch):
    bench = Bench()
    call = ("press", {"arm": "right", "target": {"xyz": bench.tool(1)[:3, 3].tolist()}, "when": {"u": 500, "v": 500, "timeout": 0.5}})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=60)
    assert "FAILED: nothing changed at the watched pixel within 0.5 s" in last_results(sent)
    assert bench.tool(1)[:3, 3] == pytest.approx(Bench().tool(1)[:3, 3], abs=1e-6)


def test_a_watch_holds_as_long_as_the_model_asks(monkeypatch):
    bench = Bench()
    policy, sent = run(monkeypatch, bench, [answer(("wait", {"watch": {"u": 500, "v": 500}, "timeout": 12}))], steps=700)
    assert "no change at the watched pixel; took 12.0 s" in last_results(sent)


def test_pixel_targets_lift_through_the_depth_the_model_saw(monkeypatch):
    bench = Bench(depth=np.full((48, 64), 0.8, dtype=np.float32))
    policy, _ = run(monkeypatch, bench, [], steps=1)
    call = {"name": "move", "args": check_call("move", {"arm": "left", "to": {"u": 500, "v": 500}})}
    notes = []
    policy.skills.resolve(call, notes)
    camera = policy.skills.camera
    expected, _ = vision_llm.pixel_to_world(bench.observation(), camera, 320, 240, vision_llm.HEAD_SIZE)
    np.testing.assert_allclose(call["points"]["to"], expected)
    assert np.linalg.norm(call["points"]["to"] - camera["pose"][:3, 3]) == pytest.approx(0.8, abs=0.05)
    assert notes == [f"to pixel is the surface at {np.round(expected, 3).tolist()}"]
    # The depth is the one the model's views were taken with, not a later one.
    bench.depth = np.full((48, 64), 0.5, dtype=np.float32)
    policy.skills.resolve(call, [])
    np.testing.assert_allclose(call["points"]["to"], expected)
    policy.skills.view = {"rgb": bench.observation()["rgb"]}
    with pytest.raises(skills.Failed, match="no head depth"):
        policy.skills.resolve(call, [])


def test_a_hand_that_only_lags_is_not_taken_for_contact(monkeypatch):
    bench = Bench(delay=10)  # 0.2 s behind: 2 cm at the guarded speed, but still moving
    point = grasp_point(bench)
    call = ("push", {"arm": "right", "target": {"xyz": point.tolist()}, "by": [0.05, 0, 0]})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=600)
    text = last_results(sent)
    assert "pushed" in text and "touched down early" not in text


def test_press_without_target_pushes_from_the_hover_pose(monkeypatch):
    bench = Bench()
    hover = bench.tool(1)[:3, 3].copy()
    lowest = [hover[2]]
    policy, sent = start(monkeypatch, bench, [answer(("press", {"arm": "right", "depth": 0.01}))])
    for _ in range(200):
        bench.apply(policy.act(bench.observation()))
        lowest.append(bench.tool(1)[2, 3])
    # Fingers point down at HOME: the press goes 5 cm + 1 cm down and comes back.
    assert min(lowest) == pytest.approx(hover[2] - settings.SETTINGS["press_standoff"].default - 0.01, abs=0.003)
    assert bench.tool(1)[2, 3] == pytest.approx(hover[2], abs=0.003)
    assert "pressed" in last_results(sent)


def test_push_and_turn_without_target_act_from_the_hand(monkeypatch):
    bench = Bench()
    start_pose = bench.tool(1).copy()
    calls = [("push", {"arm": "right", "by": [0.0, 0.0, -0.05]}), ("turn", {"arm": "right", "angle": 0.5})]
    policy, sent = run(monkeypatch, bench, [answer(*calls)], steps=300)
    text = last_results(sent)
    assert "hand moved" in text and "(100% of by)" in text and "hand turned 29 deg" in text
    np.testing.assert_allclose(bench.tool(1)[:3, 3], start_pose[:3, 3] + [0, 0, -0.05], atol=0.003)


def test_pull_without_target_moves_the_hand_and_reports_the_opening(monkeypatch):
    # The agent does not decide whether something is in the hand: it moves and reports.
    bench = Bench()
    policy, sent = run(monkeypatch, bench, [answer(("pull", {"arm": "left", "by": [0.05, 0, 0]}))], steps=60)
    text = last_results(sent)
    assert "FAILED" not in text and "(100% of by)" in text and "gripper unchanged, opening 0.080 m" in text


def test_the_model_sets_the_squeeze_and_is_told_what_was_applied(monkeypatch):
    def grasp(squeeze, finger_min):
        bench = Bench()
        point = grasp_point(bench)
        bench.item = point + [0, 0, -0.02]
        call = ("grasp", {"arm": "right", "object": {"xyz": point.tolist()}, "squeeze": squeeze, "lift": 0.0})
        policy, sent = run(monkeypatch, bench, [answer(call)], steps=400, finger_min=finger_min)
        return policy.skills, last_results(sent)

    # The fingers stop 0.015 m from closed on the 0.03 m object; each is commanded `squeeze` further.
    hand, text = grasp(0.004, 0.0)
    assert hand.fingers[1] == pytest.approx(0.015 - 0.004)
    assert "squeezed 0.004 m per finger;" in text
    # A rig whose finger targets stop at the closed stop cannot squeeze past it, and says so.
    hand, text = grasp(0.02, 0.0)
    assert hand.fingers[1] == pytest.approx(0.0)
    assert "squeezed 0.015 m per finger of the 0.020 m asked (finger command limit 0.000 m)" in text
    # Household accepts targets below the stop: the full squeeze is applied.
    hand, text = grasp(0.02, -0.03)
    assert hand.fingers[1] == pytest.approx(-0.005)
    assert hand.action()[8 + ARM_DOF] == pytest.approx(-0.005)
    assert "squeezed 0.020 m per finger;" in text


def test_closing_never_commands_more_than_the_squeeze_past_the_fingers(monkeypatch):
    bench = Bench()
    point = grasp_point(bench)
    bench.item, bench.item_width = point + [0, 0, -0.02], 0.06
    grasp = ("grasp", {"arm": "right", "object": {"xyz": point.tolist()}, "squeeze": 0.006, "lift": 0.0})
    policy, sent = start(monkeypatch, bench, [answer(grasp)])
    deepest = 0.0
    for _ in range(300):
        action = policy.act(bench.observation())
        bench.apply(action)
        deepest = max(deepest, bench.width[1] / 2 - float(action[8 + ARM_DOF]))
    assert deepest == pytest.approx(0.006, abs=1e-6)  # the 0.06 m object was never squeezed harder


def test_an_unreachable_target_fails_before_the_arm_moves_and_keeps_its_notes(monkeypatch):
    bench = Bench()
    start_pose = bench.tool(1)[:3, 3].copy()
    far = (start_pose + [1.5, 0.0, 0.0]).tolist()
    policy, sent = run(monkeypatch, bench, [answer(("move", {"arm": "right", "to": {"xyz": far}}))], steps=40)
    text = last_results(sent)
    assert "FAILED" in text and "the target is out of reach by" in text and "the arm did not move" in text
    assert "now t=" in text
    np.testing.assert_allclose(bench.tool(1)[:3, 3], start_pose, atol=1e-6)


def test_configure_changes_settings_for_the_rest_of_the_episode(monkeypatch):
    def seconds(text):
        return float(text.split("took ")[1].split(" s")[0])

    bench = Bench()
    lift = ("move_by", {"arm": "right", "by": [0.0, 0.0, 0.10]})
    lower = ("move_by", {"arm": "right", "by": [0.0, 0.0, -0.10]})
    replies = [
        answer(lift),
        answer(("configure", {"speed": 0.05, "retreat": 0.05}), lower),
        answer(lift),
        answer(("move_by", {"arm": "right", "by": [0.0, 0.0, -0.10], "speed": 1.0})),
    ]
    policy, sent = run(monkeypatch, bench, replies, steps=700)
    assert seconds(tool_results(sent[1])[0]) == pytest.approx(0.4, abs=0.05)  # 0.25 m/s by default
    configured, moved = tool_results(sent[2])
    assert configured.startswith("settings now: speed=0.05, ") and "retreat=0.05" in configured and "took" not in configured
    assert seconds(moved) == pytest.approx(2.0, abs=0.05)  # the setting holds in the same turn ...
    assert seconds(tool_results(sent[3])[0]) == pytest.approx(2.0, abs=0.05)  # ... and in later ones
    assert seconds(tool_results(sent[4])[0]) == pytest.approx(0.1, abs=0.05)  # a call's own speed wins
    assert policy.skills.settings["speed"] == 0.05 and policy.skills.settings["contact"] == 0.012


def box_scene(policy, centre, size, yaw, table_z=0.86, top_z=0.92, shape=(240, 320)):
    """Head depth of a table with a box on it, and the box-top mask, by casting rays."""
    camera = policy.skills.camera
    height, width = shape
    k = np.asarray(camera["intrinsics"], dtype=float).copy()
    k[0] *= width / camera["resolution"][0]
    k[1] *= height / camera["resolution"][1]
    rows, columns = np.mgrid[0:height, 0:width]
    rays = np.stack([(columns + 0.5 - k[0, 2]) / k[0, 0], (rows + 0.5 - k[1, 2]) / k[1, 1], np.ones_like(rows, float)], -1)
    pose = camera["pose"]
    world = rays @ pose[:3, :3].T  # direction per unit of optical depth
    origin = pose[:3, 3]

    def hit(z):
        t = (z - origin[2]) / world[..., 2]
        return t, origin + world * t[..., None]

    t_top, p_top = hit(top_z)
    c, s = np.cos(yaw), np.sin(yaw)
    local = (p_top[..., :2] - centre) @ np.array([[c, -s], [s, c]])
    on_box = (np.abs(local[..., 0]) <= size[0] / 2) & (np.abs(local[..., 1]) <= size[1] / 2) & (t_top > 0)
    t_table, _ = hit(table_z)
    depth = np.where(on_box, t_top, t_table).astype(np.float32)
    return depth, on_box


def well_scene(policy, centre, half=0.012, top_z=0.92, floor_z=0.86, table_z=0.80, block=0.08, shape=(480, 640)):
    """Head depth of a block (its top at `top_z`) with a square well of half-width `half` down to
    `floor_z`, standing on a table, by marching each ray. Returns (depth, pixel of `centre` on
    the rim's plane as {u, v})."""
    camera = policy.skills.camera
    height, width = shape
    k = np.asarray(camera["intrinsics"], dtype=float).copy()
    k[0] *= width / camera["resolution"][0]
    k[1] *= height / camera["resolution"][1]
    rows, columns = np.mgrid[0:height, 0:width]
    rays = np.stack([(columns - k[0, 2]) / k[0, 0], (rows - k[1, 2]) / k[1, 1], np.ones_like(rows, float)], -1)
    pose = camera["pose"]
    world = rays @ pose[:3, :3].T
    origin = pose[:3, 3]

    def inside(points, size):
        return (np.abs(points[..., 0] - centre[0]) <= size) & (np.abs(points[..., 1] - centre[1]) <= size)

    t_top = (top_z - origin[2]) / world[..., 2]
    top = origin + world * t_top[..., None]
    depth = np.where(inside(top, block), t_top, (table_z - origin[2]) / world[..., 2])
    # Rays that enter the well go on until they leave its footprint (a wall) or reach its floor.
    entering = inside(top, half)
    t = t_top.copy()
    alive = entering.copy()
    for _ in range(400):
        t_next = t + 0.0005
        points = origin + world * t_next[..., None]
        alive &= inside(points, half) & (points[..., 2] > floor_z)
        t = np.where(alive, t_next, t)
    depth = np.where(entering, t, depth).astype(np.float32)
    aim = np.linalg.inv(pose) @ np.array([centre[0], centre[1], top_z, 1.0])
    pixel = {"u": (aim[0] / aim[2] * k[0, 0] + k[0, 2]) / width * 1000, "v": (aim[1] / aim[2] * k[1, 1] + k[1, 2]) / height * 1000}
    return depth, pixel


def test_mask_geometry_recovers_a_box_top_and_its_narrow_side(monkeypatch):
    policy, _ = run(monkeypatch, Bench(), [], steps=1)
    centre = np.array([0.45, -0.05])
    depth, mask = box_scene(policy, centre, size=(0.12, 0.05), yaw=0.4)
    found = perception.mask_geometry(mask, depth, policy.skills.camera)
    np.testing.assert_allclose(found["centre"], centre, atol=0.006)
    assert found["top"] == pytest.approx(0.92, abs=0.002)
    assert found["length"] == pytest.approx(0.12 * 0.9, abs=0.012)  # 5-95 percentile span
    assert found["width"] == pytest.approx(0.05 * 0.9, abs=0.008)
    narrow = np.array([np.cos(found["narrow_yaw"]), np.sin(found["narrow_yaw"])])
    assert abs(narrow @ [-np.sin(0.4), np.cos(0.4)]) == pytest.approx(1.0, abs=0.01)


def test_a_named_grasp_goes_to_the_segmented_centre(monkeypatch):
    bench = Bench()
    policy, _ = run(monkeypatch, bench, [], steps=1)
    centre = bench.tool(1)[:2, 3] + 0.05 * MOUNTS[1][:2, 0]
    depth, mask = box_scene(policy, centre, size=(0.10, 0.04), yaw=0.0, top_z=0.90)
    bench.depth = depth
    monkeypatch.setenv("VISION_LLM_SAM3_URL", "http://sam3.invalid")
    monkeypatch.setattr(skills, "segment_request", lambda url, image, prompts: [(np.zeros_like(mask), 0.9), (mask, 0.8)])
    # Click near the box's edge: the grasp should still go to its centre.
    rows, columns = np.nonzero(mask)
    edge = np.argmax(columns)
    pixel = {"u": columns[edge] / (mask.shape[1] - 1) * 1000, "v": rows[edge] / (mask.shape[0] - 1) * 1000, "name": "box"}
    bench.item = np.array([*centre, 0.90 - 0.01])
    call = ("grasp", {"arm": "right", "object": pixel, "lift": 0.0})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=400)
    # With a segmenter the tools offer the name.
    assert "name" in sent[0]["tools"][3]["function"]["parameters"]["properties"]["object"]["properties"]
    text = last_results(sent)
    assert "SAM3 'box' (by name, score 0.80)" in text and "closed: opening 0.030 m" in text
    np.testing.assert_allclose(bench.tool(1)[:2, 3], centre, atol=0.01)
    jaw = bench.tool(1)[:3, 1]
    assert abs(jaw[1]) == pytest.approx(1.0, abs=0.02)  # jaws close across the 4 cm side (world y)


def named_grasp(monkeypatch, name):
    bench = Bench(depth=np.full((240, 320), 0.9, dtype=np.float32))
    monkeypatch.setenv("VISION_LLM_SAM3_URL", "http://sam3.invalid")
    policy, _ = run(monkeypatch, bench, [], steps=1)
    call = {"name": "grasp", "args": check_call("grasp", {"arm": "left", "object": {"u": 500, "v": 500, "name": name}})}
    policy.skills.resolve(call, [])
    return call["objects"]["object"]


def test_a_name_that_is_not_under_the_pixel_segments_what_is_under_the_click(monkeypatch):
    far = np.zeros((480, 640), dtype=bool)
    far[10:20, 10:20] = True
    near = np.zeros((480, 640), dtype=bool)
    near[230:250, 310:330] = True
    clicked = []
    monkeypatch.setattr(skills, "segment_request", lambda url, image, prompts: [(far, 0.9)])
    monkeypatch.setattr(skills, "point_request", lambda url, image, u, v: clicked.append((u, v)) or (near, 0.85))
    found = named_grasp(monkeypatch, "reel")
    assert clicked == [(500, 500)]
    assert found["how"] == "under the click" and found["score"] == 0.85


def test_without_point_prompts_a_name_not_under_the_pixel_falls_back_to_the_click(monkeypatch):
    far = np.zeros((480, 640), dtype=bool)
    far[10:20, 10:20] = True
    monkeypatch.setattr(skills, "segment_request", lambda url, image, prompts: [(far, 0.9)])

    def no_tracker(url, image, u, v):
        raise RuntimeError("SAM3 point prompts: this checkpoint has no point-prompt tracker")

    monkeypatch.setattr(skills, "point_request", no_tracker)
    note = named_grasp(monkeypatch, "cup")
    assert "found no 'cup' at the clicked pixel" in note and "used the clicked point" in note


def test_a_pixel_looking_into_a_well_is_moved_to_where_it_crosses_the_rim(monkeypatch):
    policy, _ = run(monkeypatch, Bench(), [], steps=1)
    camera, radius = policy.skills.camera, settings.SETTINGS["rim_radius"].default
    centre = np.array([0.45, -0.05])
    depth, pixel = well_scene(policy, centre)
    observation = {"depth": depth}
    seen, problem = vision_llm.pixel_to_world(observation, camera, pixel["u"] / 1000 * 640, pixel["v"] / 1000 * 480, vision_llm.HEAD_SIZE)
    # The line of sight through the well's centre lands on its far wall or floor, past the centre...
    assert problem is None and seen[2] < 0.92 - 0.01
    assert np.linalg.norm(seen[:2] - centre) > 0.008
    # ... and crosses the rim's plane at the centre.
    rim, crossing = perception.opening_under(observation, camera, pixel, seen, radius)
    assert rim == pytest.approx(0.92, abs=0.003)
    assert np.linalg.norm(crossing[:2] - centre) < 0.003
    # A spot on the flat top, and one on the table beside the block, are not openings.
    for spot in ([centre[0] + 0.05, centre[1]], [centre[0], centre[1] + 0.095]):
        _, flat = well_scene(policy, np.array(spot))
        point, _ = vision_llm.pixel_to_world(observation, camera, flat["u"] / 1000 * 640, flat["v"] / 1000 * 480, vision_llm.HEAD_SIZE)
        assert perception.opening_under(observation, camera, flat, point, radius) is None


def test_place_into_releases_over_the_opening_the_pixel_points_at(monkeypatch):
    bench = Bench()
    policy, _ = run(monkeypatch, bench, [], steps=1)
    centre = bench.tool(1)[:2, 3] + 0.03 * MOUNTS[1][:2, 0]
    bench.depth, pixel = well_scene(policy, centre)
    into = ("place", {"arm": "right", "target": pixel, "mode": "into", "height": 0.12})
    policy, sent = run(monkeypatch, bench, [answer(into)], steps=500)
    text = last_results(sent)
    assert "inside an opening whose rim is at z=0.92" in text and "line of sight crosses the rim's plane" in text
    assert np.linalg.norm(bench.tool(1)[:2, 3] - centre) < 0.004
    # With a search too small to reach the rim, the surface point is used and the result says so.
    # The setting takes effect for a call of the same turn.
    bench = Bench()
    bench.depth = well_scene(policy, centre)[0]
    policy, sent = run(monkeypatch, bench, [answer(("configure", {"rim_radius": 5}), into)], steps=500)
    assert "no level rim stands around it within the rim_radius setting 5 (a larger one reaches further)" in last_results(sent)
    # The same pixel placed "on" is taken as the surface it lands on.
    bench = Bench()
    bench.depth = well_scene(policy, centre)[0]
    policy, sent = run(monkeypatch, bench, [answer(("place", {"arm": "right", "target": pixel, "height": 0.12}))], steps=500)
    assert "inside an opening" not in last_results(sent)
    assert np.linalg.norm(bench.tool(1)[:2, 3] - centre) > 0.008


def test_segment_distance():
    a, b = np.zeros(3), np.array([1.0, 0, 0])
    assert geometry.segment_distance(a, b, np.array([0.5, 0.3, 0]), np.array([0.5, 2.0, 0])) == pytest.approx(0.3)
    assert geometry.segment_distance(a, b, np.array([2.0, 0, 0.4]), np.array([3.0, 0, 0.4])) == pytest.approx(np.hypot(1, 0.4))
    assert geometry.segment_distance(a, b, np.array([0.2, -1, 0]), np.array([0.2, 1, 0])) == pytest.approx(0.0)
    assert geometry.segment_distance(a, a, np.array([0, 0, 2.0]), np.array([0, 0, 2.0])) == pytest.approx(2.0)


def test_a_motion_into_the_other_arm_is_refused_before_it_starts(monkeypatch):
    bench = Bench()
    left = bench.tool(0)[:3, 3].copy()
    start_pose = bench.tool(1)[:3, 3].copy()
    aside = (start_pose - left) / np.linalg.norm(start_pose - left)  # from the left hand toward the right arm
    into = ("move", {"arm": "right", "to": {"xyz": left.tolist()}})
    policy, sent = run(monkeypatch, bench, [answer(into)], steps=80)
    text = last_results(sent)
    assert "FAILED" in text and "would bring the right arm to" in text and "from the left arm" in text
    assert "(arm_clearance 0.02 m); the arm did not move" in text
    np.testing.assert_allclose(bench.tool(1)[:3, 3], start_pose, atol=1e-6)
    # The clearance is the model's to set: with none asked for, the same move runs.
    # Hand capsules 0.12 m apart are 0.01 m from touching: under the default clearance.
    bench = Bench()
    free = [("configure", {"arm_clearance": -0.05}), ("move", {"arm": "right", "to": {"xyz": (left + 0.12 * aside).tolist()}})]
    policy, sent = run(monkeypatch, bench, [answer(*free)], steps=200)
    assert "FAILED" not in last_results(sent)
    # From there, with the default clearance, moving apart is never refused; moving further in is.
    away = ("move_by", {"arm": "right", "by": (0.05 * aside).tolist()})
    closer = ("move_by", {"arm": "right", "by": (-0.04 * aside).tolist()})
    policy, sent = run(monkeypatch, bench, [answer(closer), answer(away)], steps=300)
    first, second = last_results(sent, 1), last_results(sent, 2)
    assert "FAILED" in first and "would bring the right arm to" in first
    assert "FAILED" not in second and "measured tool point" in second
    assert np.linalg.norm(bench.tool(1)[:3, 3] - left) == pytest.approx(0.17, abs=0.005)


def test_a_motion_waits_for_an_arm_that_trails_its_command(monkeypatch):
    # Arms that follow 0.2 s behind: at 1 m/s the commanded 0.1 m is over in 0.1 s, before the
    # hand has set off. The call ends when the hand has arrived, and reports it there.
    bench = Bench(delay=10)
    start_pose = bench.tool(1)[:3, 3].copy()
    call = ("move_by", {"arm": "right", "by": [0.0, 0.0, 0.10], "speed": 1.0})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=200)
    text = last_results(sent)
    assert "from the commanded" not in text and "stopped following" not in text
    assert 0.3 <= float(text.split("took ")[1].split(" s")[0]) <= 2.0
    reported = np.array(json.loads(text.split("measured tool point ")[-1].split(";")[0]))
    assert np.linalg.norm(reported - (start_pose + [0.0, 0.0, 0.10])) <= settings.SETTINGS["arrival"].default + 1e-3


def test_a_motion_asked_faster_than_the_joints_can_go_runs_at_their_limit_on_its_line(monkeypatch):
    # Joints capped at 0.7 rad/s (industry's arms). A level-hand move asked at 1 m/s and 4 rad/s
    # would command them several times faster: the trajectory waits for the arm instead, so the
    # commanded joints never lead the measured ones by more than JOINT_LEAD (plus one step), the
    # hand arrives, and the result states how long that took.
    bench = Bench(rate=0.7)
    start_pose = bench.tool(1)[:3, 3].copy()
    goal = start_pose + [0.10, 0.0, 0.05]
    calls = [
        ("configure", {"speed": 1.0, "turn_speed": 4.0}),
        ("move", {"arm": "right", "to": {"xyz": goal.tolist()}, "approach": "front"}),
    ]
    policy, sent = start(monkeypatch, bench, [answer(*calls)])
    arm, lead, turned = slice(8, 8 + ARM_DOF), 0.0, 0.0
    before = bench.joints[arm].copy()
    for _ in range(600):
        action = policy.act(bench.observation())
        lead = max(lead, float(np.max(np.abs(action[arm] - bench.joints[arm]))))
        bench.apply(action)
    turned = float(np.max(np.abs(bench.joints[arm] - before)))
    text = last_results(sent)
    assert "stopped following" not in text and "from the commanded" not in text
    assert lead <= skills.JOINT_LEAD + 0.7 * DT + 1e-6
    np.testing.assert_allclose(bench.tool(1)[:3, 3], goal, atol=settings.SETTINGS["arrival"].default + 1e-3)
    took = float(text.split("took ")[-1].split(" s")[0])
    assert turned / 0.7 - 0.1 <= took <= turned / 0.7 + 1.0  # the joints' own limit, not the 1 m/s asked


def test_an_unguarded_motion_that_is_held_off_stops_and_says_how_far_short(monkeypatch):
    # A floor 4 cm under the hand, a move 10 cm down without a guard: the command stops two joint
    # leads ahead of the arm at most (one taken for a load's standing offset, then one more)
    # instead of running on to the goal, and the result gives the shortfall.
    bench = Bench()
    here = bench.tool(1)[:3, 3].copy()
    bench.floor = here[2] - 0.04
    call = ("move_by", {"arm": "right", "by": [0.0, 0.0, -0.10]})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=200)
    text = last_results(sent)
    assert "the arm stopped following 0.06" in text and "something holds it off" in text
    arm = slice(8, 8 + ARM_DOF)
    assert np.max(np.abs(policy.skills.command[1] - bench.joints[arm])) <= 2 * skills.JOINT_LEAD + 0.02


def test_move_by_is_measured_from_where_the_hand_is_after_a_guard_stop(monkeypatch):
    # A guarded offset stops on a floor and stays pressed: its command sits `contact` inside it.
    # An offset back up must move the hand by that offset, not only take the press away; an
    # offset along the floor keeps the press.
    for by, rise in (([0.0, 0.0, 0.012], 0.012), ([0.02, 0.0, 0.0], 0.0)):
        bench = Bench()
        here = bench.tool(1)[:3, 3].copy()
        bench.floor = here[2] - 0.04
        calls = [("move_by", {"arm": "right", "by": [0.0, 0.0, -0.10], "guard": True}), ("move_by", {"arm": "right", "by": by})]
        policy, sent = run(monkeypatch, bench, [answer(*calls)], steps=300)
        text = last_results(sent)
        assert "stopped on contact" in text
        assert bench.tool(1)[2, 3] - bench.floor == pytest.approx(rise, abs=0.003)
        pressed = policy.skills.tool(1)[2, 3] < bench.floor - 0.005
        assert pressed == (rise == 0.0)


def test_place_opens_to_the_opening_it_is_given(monkeypatch):
    bench = Bench()
    here = bench.tool(1)[:3, 3].copy()
    target = here + [0.0, 0.05, -0.10]
    calls = [
        ("move_by", {"arm": "right", "by": [0.0, 0.0, 0.0], "opening": 0.02}),
        ("place", {"arm": "right", "target": {"xyz": target.tolist()}, "opening": 0.034}),
    ]
    policy, sent = run(monkeypatch, bench, [answer(*calls)], steps=400)
    assert "opened to 0.034 m and backed off" in last_results(sent)
    assert bench.width[1] == pytest.approx(0.034, abs=1e-4)
    with pytest.raises(ValueError):
        tools.check_call("place", {"arm": "right", "target": {"xyz": [0, 0, 0]}, "opening": 0.2})


def test_a_hand_held_off_after_its_command_ended_is_contact_for_a_guarded_motion(monkeypatch):
    # The same trailing arms meet a floor 1 cm above a place's release point: the command has
    # reached the release point (0.1 s later at 0.1 m/s) before the hand, 0.2 s behind, stops on
    # the floor. That is still reported as contact.
    bench = Bench(delay=10)
    here = bench.tool(1)[:3, 3].copy()
    target = here + [0.0, 0.05, -0.10]
    bench.floor = target[2] + 0.01
    calls = [("configure", {"contact": 0.005, "clearance": 0.02}), ("place", {"arm": "right", "target": {"xyz": target.tolist()}})]
    policy, sent = run(monkeypatch, bench, [answer(*calls)], steps=400)
    text = last_results(sent)
    z = float(text.split("stopped on contact at z=")[1].split(",")[0])
    assert z == pytest.approx(bench.floor, abs=0.003)  # the hand moves 2 mm a step
    assert float(text.split("stopped on contact at z=")[1].split(", ")[1].split(" m above")[0]) == pytest.approx(0.01, abs=0.003)


# --- hand tilt, hinges, guarded offsets, two arms at once ----------------------------------------
def test_tilt_spans_the_orientations_between_and_beyond_the_two_presets():
    # The presets are two points of the family.
    for heading in (0.3, -2.0):
        level = [np.cos(heading), np.sin(heading)]
        np.testing.assert_allclose(geometry.hand_rotation(heading, 0.0), geometry.front_rotation(level), atol=1e-12)
        np.testing.assert_allclose(
            geometry.hand_rotation(heading, np.pi / 2), geometry.top_rotation(heading + np.pi / 2), atol=1e-12
        )
    rotation = geometry.hand_rotation(0.4, 0.35, 1.2)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # Pitch tilts the fingers below level; roll 1.57 on a level hand makes the jaws close vertically.
    assert geometry.hand_rotation(0.0, 0.35)[2, 2] == pytest.approx(-np.sin(0.35))
    np.testing.assert_allclose(np.abs(geometry.hand_rotation(0.0, 0.0, np.pi / 2)[:, 1]), [0, 0, 1], atol=1e-12)
    # A call gives the angles in one `tilt` object; each has its range; `keep` has nothing to tilt.
    grasp = {"arm": "left", "object": {"u": 1, "v": 2}}
    assert check_call("grasp", {**grasp, "tilt": {"pitch": 0.35, "roll": None}})["tilt"] == {"pitch": 0.35}
    assert "tilt" not in check_call("grasp", {**grasp, "tilt": {}})
    with pytest.raises(ValueError, match=r"grasp.tilt.pitch = 3 is outside its range"):
        check_call("grasp", {**grasp, "tilt": {"pitch": 3}})
    with pytest.raises(ValueError, match="unexpected \\['yaw'\\]"):
        check_call("grasp", {**grasp, "tilt": {"yaw": 1}})
    with pytest.raises(ValueError, match="tilt needs approach top or front"):
        check_call("move", {"arm": "left", "to": {"u": 1, "v": 2}, "approach": "keep", "tilt": {"pitch": 0.3}})


def test_a_tilted_grasp_comes_in_along_the_tilted_fingers(monkeypatch):
    bench = Bench()
    point = bench.tool(1)[:3, 3] + 0.10 * MOUNTS[1][:3, 0] + [0.0, 0.0, -0.05]
    bench.item = point.copy()
    call = ("grasp", {"arm": "right", "object": {"xyz": point.tolist()}, "approach": "front",
                      "tilt": {"pitch": 0.35, "roll": 1.5708}, "depth": 0.0, "lift": 0.0})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=500)
    assert "FAILED" not in last_results(sent) and "off the asked orientation" not in last_results(sent)
    hand = bench.tool(1)
    assert hand[2, 2] == pytest.approx(-np.sin(0.35), abs=0.02)  # fingers 20 degrees below level
    assert abs(hand[2, 1]) == pytest.approx(np.cos(0.35), abs=0.03)  # jaws close in the vertical plane
    np.testing.assert_allclose(hand[:3, 3], point, atol=0.005)
    # A move with a tilt and no approach is measured from the level preset, not from "keep".
    bench = Bench()
    move = ("move", {"arm": "right", "to": {"xyz": point.tolist()}, "tilt": {"pitch": 0.6}})
    run(monkeypatch, bench, [answer(move)], steps=300)
    assert bench.tool(1)[2, 2] == pytest.approx(-np.sin(0.6), abs=0.02)
    # Where the joints cannot give the asked tilt, the route tilts the hand to reach the point
    # and the result says how far off it is.
    bench = Bench()
    back = ("move", {"arm": "right", "to": {"xyz": grasp_point(bench).tolist()}, "tilt": {"pitch": 0.2, "heading": 2.5}})
    policy, sent = run(monkeypatch, bench, [answer(back)], steps=300)
    assert "deg off the asked orientation" in last_results(sent)


def test_place_turns_what_it_holds_to_the_asked_yaw(monkeypatch):
    bench = Bench()
    here = bench.tool(1)
    jaws = np.arctan2(here[1, 1], here[0, 1])
    target = here[:3, 3] + [0.0, 0.05, -0.05]
    call = ("place", {"arm": "right", "target": {"xyz": target.tolist()}, "yaw": float(jaws + 0.6), "height": 0.0})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=500)
    assert "FAILED" not in last_results(sent), last_results(sent)
    after = bench.tool(1)
    turned = (np.arctan2(after[1, 1], after[0, 1]) - jaws + np.pi / 2) % np.pi - np.pi / 2
    assert turned == pytest.approx(0.6, abs=0.03)
    np.testing.assert_allclose(after[:3, 2], here[:3, 2], atol=0.03)  # the fingers still point where they did


def test_a_turn_about_a_hinge_swings_the_hand_on_an_arc(monkeypatch):
    bench = Bench()
    start = bench.tool(1)
    pivot = start[:3, 3] + [0.0, 0.06, 0.0]  # a vertical hinge 6 cm beside the hand, as a lever's
    call = ("turn", {"arm": "right", "angle": 1.0, "hinge": {"pivot": {"xyz": pivot.tolist()}}})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=400)
    text = last_results(sent)
    assert "FAILED" not in text and "swung 57 deg about the hinge, 0.060 m from its line (asked 57)" in text
    spin = geometry.about_axis([0, 0, 1], 1.0)
    end = bench.tool(1)
    np.testing.assert_allclose(end[:3, 3], pivot + spin @ (start[:3, 3] - pivot), atol=0.004)
    np.testing.assert_allclose(end[:3, :3], spin @ start[:3, :3], atol=0.03)  # the hand turned with the arc
    assert "gripper unchanged" in text
    # A hinge given by its axis alone passes through the tool point: the hand tilts in place.
    bench = Bench()
    start = bench.tool(1)
    axis = start[:3, 1].tolist()  # tip about the jaws' own axis, as to pour
    call = ("turn", {"arm": "right", "angle": -0.5, "hinge": {"axis": axis}})
    run(monkeypatch, bench, [answer(call)], steps=300)
    end = bench.tool(1)
    np.testing.assert_allclose(end[:3, 3], start[:3, 3], atol=0.004)
    np.testing.assert_allclose(end[:3, :3], geometry.about_axis(axis, -0.5) @ start[:3, :3], atol=0.03)
    # The other way the wrist runs out of range: refused before moving, with how far short.
    bench = Bench()
    policy, sent = run(monkeypatch, bench, [answer(("turn", {"arm": "right", "angle": 0.5, "hinge": {"axis": axis}}))], steps=60)
    assert "FAILED" in last_results(sent) and "short of the asked turn; the arm did not swing" in last_results(sent)
    np.testing.assert_allclose(bench.tool(1)[:3, :3], start[:3, :3], atol=1e-6)
    # An arc the arm cannot follow is refused before it moves.
    bench = Bench()
    far = ("turn", {"arm": "right", "angle": 2.5, "hinge": {"pivot": {"xyz": (start[:3, 3] + [0.0, 0.6, 0.0]).tolist()}}})
    policy, sent = run(monkeypatch, bench, [answer(far)], steps=60)
    assert "FAILED" in last_results(sent) and "the arc is out of reach" in last_results(sent)
    np.testing.assert_allclose(bench.tool(1)[:3, 3], start[:3, 3], atol=1e-6)


def test_a_guarded_offset_stops_on_contact_and_stays_pressed(monkeypatch):
    bench = Bench()
    start = bench.tool(1)[:3, 3].copy()
    bench.floor = start[2] - 0.04
    call = ("move_by", {"arm": "right", "by": [0.0, 0.0, -0.10], "guard": True})
    policy, sent = run(monkeypatch, bench, [answer(call)], steps=300)
    text = last_results(sent)
    assert "stopped on contact 0.060 m short and stays pressed against it" in text
    assert bench.tool(1)[2, 3] == pytest.approx(bench.floor, abs=0.003)
    # The command stays a little past the contact, so the hand keeps pressing.
    commanded = policy.skills.tool(1)[2, 3]
    assert bench.floor - 0.03 < commanded < bench.floor - 0.005
    # Its end point may lie beyond reach (under the table it is sent down onto): it still runs.
    bench = Bench()
    bench.floor = start[2] - 0.04
    deep = ("move_by", {"arm": "right", "by": [0.0, 0.0, -0.60], "guard": True})
    policy, sent = run(monkeypatch, bench, [answer(deep)], steps=400)
    assert "FAILED" not in last_results(sent) and "stays pressed against it" in last_results(sent)
    assert bench.tool(1)[2, 3] == pytest.approx(bench.floor, abs=0.003)
    # Without guard the same offset is an ordinary move.
    assert "guard" not in check_call("move_by", {"arm": "left", "by": [0, 0, 0.1], "guard": False})


def test_together_runs_two_calls_at_the_same_time(monkeypatch):
    bench = Bench()
    left, right = bench.tool(0)[:3, 3].copy(), bench.tool(1)[:3, 3].copy()
    lift = [("together", {}), ("move_by", {"arm": "left", "by": [0, 0, 0.10]}), ("move_by", {"arm": "right", "by": [0, 0, 0.10]})]
    policy, sent = start(monkeypatch, bench, [answer(*lift)])
    heights = []
    for _ in range(60):
        bench.apply(policy.act(bench.observation()))
        heights.append((bench.tool(0)[2, 3] - left[2], bench.tool(1)[2, 3] - right[2]))
    # Both hands rise in the same control steps: what they carry between them stays level.
    assert max(abs(a - b) for a, b in heights) < 1e-6
    assert heights[9][0] > 0.03 and heights[-1] == pytest.approx((0.10, 0.10), abs=1e-3)
    marker, first, second = tool_results(sent[1])
    assert marker == "the next two calls run at the same time"
    assert "took 0.4 s" in first and "took 0.4 s" in second  # 0.10 m at 0.25 m/s, once, not twice
    # One call failing does not stop the other, and what follows them is not run.
    bench = Bench()
    far = (bench.tool(0)[:3, 3] + [1.5, 0, 0]).tolist()
    calls = [("together", {}), ("move", {"arm": "left", "to": {"xyz": far}}),
             ("move_by", {"arm": "right", "by": [0, 0, 0.05]}), ("home", {"arm": "right"})]
    policy, sent = run(monkeypatch, bench, [answer(*calls)], steps=60)
    marker, first, second, after = tool_results(sent[1])
    assert first.startswith("FAILED") and "out of reach" in first and "measured tool point" in second
    assert after == "not run: an earlier call of this turn failed"
    assert bench.tool(1)[2, 3] == pytest.approx(right[2] + 0.05, abs=1e-3)


@pytest.mark.parametrize("calls,reason", [
    ([("together", {}), ("move_by", {"arm": "left", "by": [0, 0, 0.05]})], "needs two more calls"),
    ([("together", {}), ("move_by", {"arm": "left", "by": [0, 0, 0.05]}), ("home", {"arm": "left"})], "one call per arm"),
    ([("together", {}), ("move_by", {"arm": "left", "by": [0, 0, 0.05]}), ("wait", {"seconds": 1})], "two arm tools"),
    ([("together", {}), ("move_by", {"arm": "left", "by": [0, 0, 0.05]}), ("home", {"arm": "middle"})], "was refused"),
])
def test_together_is_refused_unless_two_arm_calls_one_per_arm_follow(monkeypatch, calls, reason):
    bench = Bench()
    rest = bench.tool(0)[:3, 3].copy()
    policy, sent = run(monkeypatch, bench, [answer(*calls)], steps=40)
    results = tool_results(sent[1])
    assert results[0].startswith("REFUSED") and reason in results[0]
    np.testing.assert_allclose(bench.tool(0)[:3, 3], rest, atol=1e-6)  # nothing ran
