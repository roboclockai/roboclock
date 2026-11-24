"""The policy the evaluator drives: one model turn at a time, played out by the skills.

`act` is called every control step. When no call is left to run it asks the model (a query
step: the only steps charged inference latency), checks each tool call, and then advances the
running skill by one step per call until the turn's calls are done.
"""

import json
import math
import os
import time

import numpy as np

from roboclock.envs.lab.robot import ARM_DOF, DOF
from roboclock.envs.rates import CONTROL_HZ
from roboclock.policies.vision_llm import ContentFiltered, endpoint_settings, request_chat, rig_camera
from roboclock.policies.vision_llm_primitives.client import (
    assistant_message,
    parse_arguments,
    read_reply,
    tool_body,
    tool_message,
)
from roboclock.policies.vision_llm_primitives.prompt import (
    HISTORY_TURNS,
    Conversation,
    observation_content,
    observation_text,
    observation_views,
    system_prompt,
)
from roboclock.policies.vision_llm_primitives.settings import OPEN_WIDTH, opening_range
from roboclock.policies.vision_llm_primitives.skills import Failed, Skills
from roboclock.policies.vision_llm_primitives.tools import (
    ARM_TOOLS,
    check_call,
    describe,
    tool_definitions,
)
from roboclock.policies.x5_kinematics import X5Kinematics

MAX_CALLS = 4  # tool calls run per turn; the model sees nothing new while they run
MIN_TURN = 0.1  # s of simulated time each model turn lasts at least
ERROR_HOLD = 0.5  # s both arms hold after a turn in which no call could run


class PrimitivePolicy:
    def __init__(self):
        endpoint_settings(self)
        # Optional SAM3 server (scripts/xpolicylab/serve_sam3.py) that refines named pixel targets.
        self.sam3_url = os.environ.get("VISION_LLM_SAM3_URL")
        # Output tokens of the latest model query, thinking included (--inference-latency ideal).
        self.last_output_tokens = None
        # Model turns per episode; past it both arms hold still until the episode ends (0 = no cap).
        # At zero inference latency a model looping on 0.1 s turns would otherwise never finish, and
        # a hosted API bills every turn.
        self.max_queries = int(os.environ.get("VISION_LLM_MAX_QUERIES", "0"))
        # Past turns sent with each request (0: every turn stands alone).
        self.history_turns = int(os.environ.get("VISION_LLM_HISTORY", str(HISTORY_TURNS)))
        self.control_hz = float(CONTROL_HZ)
        self.kinematics = X5Kinematics()
        self.finger_min, self.open_width = 0.0, OPEN_WIDTH
        self.tools = tool_definitions(segmenter=bool(self.sam3_url), open_width=self.open_width)

    def set_spaces(self, action_space, observation_space):
        """The environment's interface, told by the evaluator before `reset`.

        - The lowest finger target of the action space says how far a close may squeeze: 0 is
          the closed stop; household takes down to -0.03.
        - The widest the jaws open is the upper bound of the `gripper_width` observation (lab
          88 mm, household 60.3 mm), capped by what the action space can command.
        """
        low = np.asarray(action_space.low, dtype=float)
        high = np.asarray(action_space.high, dtype=float)
        width = None
        spaces = getattr(observation_space, "spaces", None) or {}
        if "gripper_width" in spaces:
            width = float(np.min(spaces["gripper_width"].high))
        self.finger_min = min(0.0, float(low[ARM_DOF]))
        self.open_width = opening_range(2 * float(high[ARM_DOF]), width)
        self.tools = tool_definitions(segmenter=bool(self.sam3_url), open_width=self.open_width)

    # ------------------------------------------------------------------ episode
    @property
    def inference_model(self):
        """Served model name; --inference-latency ideal charges queries at its surveyed speed."""
        return self.model

    def reset(self, observation):
        joints = np.asarray(observation["joint_positions"], dtype=float)
        pose = np.asarray(observation["ee_pose"], dtype=float)
        arms = [
            self.kinematics.calibrated(
                joints[DOF * i : DOF * i + ARM_DOF], pose[7 * i : 7 * i + 3], pose[7 * i + 3 : 7 * i + 7]
            )
            for i in range(2)
        ]
        camera = rig_camera(arms[0].base, arms[1].base)
        self.skills = Skills(arms, camera, joints, self.control_hz, self.finger_min, self.sam3_url, self.open_width / 2)
        self.instruction = str(observation["instruction"])
        self.conversation = Conversation(
            system_prompt(self.instruction, camera["mount"], self.open_width, self.finger_min, MAX_CALLS),
            self.history_turns,
        )
        self.calls, self.active, self.skipping = [], None, False
        self.turn = None  # the turn being played out: its observation, the model's calls, their results
        self.notes = []  # what the harness has to tell the model on its next turn
        self.results = []  # the running turn's calls and results, for the log
        self.requests = 0
        self.previous_turn = None  # task clock of the previous model turn
        self.turn_end = -math.inf
        self.queried = True

    def act(self, observation):
        self.queried = False
        now = float(np.ravel(observation["sim_time"])[0])
        while True:
            if self.active is None:
                if not self.calls:
                    if self.queried or now < self.turn_end:
                        break  # hold until the turn has lasted its minimum
                    if self.max_queries and self.requests >= self.max_queries:
                        break  # query cap reached: hold the current command
                    self.calls = self.plan(observation)
                    runnable = any("error" not in call for call in self.calls[:1])
                    self.turn_end = now + (MIN_TURN if runnable else ERROR_HOLD)
                    self.queried = True
                    continue
                call = self.calls.pop(0)
                if "error" in call:
                    self.finish(call, call["error"])
                    self.skipping = True
                    continue
                if self.skipping:
                    self.finish(call, "not run: an earlier call of this turn failed")
                    continue
                if call["name"] == "together":
                    pair = self.calls[:2]
                    problem = pair_problem(pair)
                    if problem:
                        self.finish(call, f"REFUSED: {problem}")
                        self.skipping = True
                        continue
                    self.finish(call, "the next two calls run at the same time")
                    del self.calls[:2]
                    self.active = (pair, self.skills.run_pair(pair, observation))
                else:
                    self.active = (call, self.skills.run(call, observation))
                fresh = True
            else:
                fresh = False
            call, skill = self.active
            try:
                skill.send(None if fresh else observation)
                break
            except StopIteration as done:
                if isinstance(call, list):  # a pair run together: each call has its own result
                    for one, (failed, text) in zip(call, done.value):
                        self.finish(one, f"FAILED: {text}" if failed else text)
                        self.skipping |= failed
                else:
                    self.finish(call, done.value or "done")
                self.active = None
            except Failed as error:
                self.finish(call, f"FAILED: {error}")
                self.active, self.skipping = None, True
        return self.skills.action()

    def finish(self, call, text):
        """Answer one tool call: its result goes into the conversation and the log."""
        self.turn["tools"].append(tool_message(call, text))
        self.results.append(f"{describe(call['name'], call.get('args', {}))}: {text}")

    # ------------------------------------------------------------------ model turn
    def plan(self, observation):
        """Ask the model; return its tool calls, each checked ({"args"}) or refused ({"error"})."""
        if self.turn is not None:
            self.conversation.add(self.turn)
            self.turn = None
        now = float(np.ravel(observation["sim_time"])[0])
        notes, self.notes = self.notes, []
        if str(observation["instruction"]) != self.instruction:
            self.instruction = str(observation["instruction"])
            notes.append(f"The task now reads: {self.instruction}")
        # Reading the images renders them (roboclock.envs.observation): build the request first.
        text = observation_text(observation, self.skills.fingers, self.previous_turn, notes)
        views = observation_views(observation)
        content = observation_content(text, views)
        messages = self.conversation.messages(content)
        self.skills.look(observation)
        shown, self.results, self.skipping = self.results, [], False
        row = dict(request=self.requests, sim_time=now, shown_results=shown, guarded=self.skills.diagnostics, messages=len(messages))
        self.skills.diagnostics = []
        self.requests += 1
        self.previous_turn = now
        start = time.perf_counter()
        try:
            while True:
                body = tool_body(self, messages, self.tools)
                try:
                    reply = request_chat(
                        self.url, body, self.timeout, self.cookie,
                        api_key=self.api_key, retries=self.retries, retry_timeouts=self.retry_timeouts,
                    )
                    break
                except RuntimeError as error:
                    # The conversation outgrew the model's context: drop the older half of the past
                    # turns and ask again, rather than lose the episode.
                    if not (too_long(error) and self.conversation.forget()):
                        raise
                    messages = self.conversation.messages(content)
                    row["messages"] = len(messages)
        except ContentFiltered:
            # The provider refused this one request (its filter also judges the images). The turn
            # is lost, not the episode: hold, and ask again with the next views.
            self.last_output_tokens = None
            self.notes.append("Your previous turn got no answer: the model provider's content filter refused the request.")
            self.log(row, round_trip_ms=(time.perf_counter() - start) * 1000.0, parse_error="content_filter", calls=[], answer="", reasoning="")
            return []
        round_trip = time.perf_counter() - start
        usage = reply.get("usage") or {}
        self.last_output_tokens = usage.get("completion_tokens")
        answer, asked, finish_reason = read_reply(reply, self.requests - 1)
        calls, problems = [], []
        for index, call in enumerate(asked):
            if index >= MAX_CALLS:
                call["error"] = f"not run: at most {MAX_CALLS} calls run per turn"
            else:
                try:
                    call["args"] = check_call(call["name"], parse_arguments(call["raw"]), self.open_width)
                except ValueError as problem:
                    call["error"] = f"REFUSED: {problem}"
                    problems.append(str(problem))
            calls.append(call)
        if not asked:
            cut = " (the reply was cut off at the output-token limit)" if finish_reason == "length" else ""
            self.notes.append(f"Your previous reply called no tool{cut}. Every reply must call at least one tool.")
            problems.append(f"no tool call (finish_reason {finish_reason})")
        self.turn = {
            "text": text,
            "head": views[0][1] if views and views[0][0] == "head" else None,
            "assistant": assistant_message(answer, asked),
            "tools": [],
        }
        message = reply["choices"][0].get("message") or {}
        self.log(
            row,
            round_trip_ms=round_trip * 1000.0,
            prompt_tokens=usage.get("prompt_tokens"),
            cached_prompt_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            cost_usd=reply.get("response_cost_usd"),
            finish_reason=finish_reason,
            parse_error="; ".join(problems) or None,
            calls=[describe(call["name"], call["args"]) if "args" in call else f"{call['name']}({call['raw'][:200]})" for call in calls],
            answer=answer[:2000],
            reasoning=(message.get("reasoning") or message.get("reasoning_content") or "")[:2000],
        )
        return calls

    def log(self, row, **fields):
        if self.log_path:
            tag = {"episode": self.episode_tag} if getattr(self, "episode_tag", None) else {}
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({**tag, **row, **fields}) + "\n")


def pair_problem(pair):
    """Why two calls cannot run at the same time, or None: `together` takes the next two calls of
    its turn, both arm tools, one per arm."""
    if len(pair) < 2:
        return "together needs two more calls after it in the same turn"
    if any("error" in call for call in pair):
        return "one of the two calls after together was refused: " + next(call["error"] for call in pair if "error" in call)
    names = [call["name"] for call in pair]
    if any(name not in ARM_TOOLS for name in names):
        return f"together runs two arm tools ({', '.join(ARM_TOOLS)}), not {names}"
    if pair[0]["args"]["arm"] == pair[1]["args"]["arm"]:
        return f'together needs one call per arm; both use "{pair[0]["args"]["arm"]}"'
    return None


def too_long(error):
    """Whether a server refused a request for exceeding the model's context length."""
    text = str(error).lower()
    return any(mark in text for mark in ("context length", "context_length", "max_model_len", "maximum context", "too many tokens"))


def make():
    return PrimitivePolicy()
