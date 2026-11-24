"""The model request of the primitive agent: a chat completion with native tool calling.

Endpoint settings and the HTTP call are those of roboclock.policies.vision_llm (VISION_LLM_URL,
_API_KEY, _MODEL, _DIALECT, _THINKING, _THINKING_BUDGET, _REASONING_EFFORT, _MAX_TOKENS,
_TIMEOUT, _RETRIES, _LOG). This module adds the body of a tool-calling request and the reading
of its reply.

Reasoning is stated in every request, not left to a provider's default:
- `vllm` dialect: `enable_thinking` and a `thinking_token_budget`, temperature 0;
- `openai` dialect with thinking: `reasoning_effort` (VISION_LLM_REASONING_EFFORT, default
  REASONING_EFFORT) and no temperature, which reasoning models do not take;
- `openai` dialect without thinking: temperature 0.
"""

import json

REASONING_EFFORT = "medium"  # the openai dialect's default when VISION_LLM_REASONING_EFFORT is unset
ANSWER_TOKENS = 1024  # output tokens for the tool calls themselves, before any thinking budget


def tool_body(policy, messages, tools):
    """Chat request for `policy`'s model and dialect that must answer with tool calls."""
    body = {
        "model": policy.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "max_tokens": ANSWER_TOKENS,
    }
    if policy.thinking:
        body["max_tokens"] += policy.thinking_budget
    if policy.dialect == "vllm":
        body["temperature"] = 0.0
        body["chat_template_kwargs"] = {"enable_thinking": policy.thinking}
        if policy.thinking:
            body["thinking_token_budget"] = policy.thinking_budget
    elif policy.thinking:
        body["reasoning_effort"] = policy.reasoning_effort or REASONING_EFFORT
    else:
        body["temperature"] = 0.0
    if policy.max_tokens:
        body["max_tokens"] = policy.max_tokens
    return body


def read_reply(reply, turn):
    """(text, tool calls, finish reason) of a chat reply. Each call is {"id", "name", "raw"},
    `raw` being its arguments as the model wrote them; a missing id is filled in."""
    choice = reply["choices"][0]
    message = choice.get("message") or {}
    calls = []
    for index, item in enumerate(message.get("tool_calls") or []):
        function = item.get("function") or {}
        raw = function.get("arguments")
        calls.append(
            {
                "id": item.get("id") or f"call_{turn}_{index}",
                "name": function.get("name") or "",
                "raw": raw if isinstance(raw, str) else json.dumps(raw if raw is not None else {}),
            }
        )
    return message.get("content") or "", calls, choice.get("finish_reason")


def parse_arguments(raw):
    """The arguments object a tool call carries, or ValueError with a message for the model."""
    try:
        arguments = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as error:
        raise ValueError(f"the arguments are not valid JSON ({error})") from None
    if not isinstance(arguments, dict):
        raise ValueError("the arguments must be a JSON object")
    return arguments


def assistant_message(text, calls):
    """The assistant turn as the conversation keeps it."""
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = [
            {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["raw"]}}
            for call in calls
        ]
    else:
        message["content"] = text or ""
    return message


def tool_message(call, text):
    return {"role": "tool", "tool_call_id": call["id"], "content": text}
