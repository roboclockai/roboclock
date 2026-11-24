"""Vision-LLM agent that acts through manipulation primitives on the two ARX X5 arms.

Each primitive is a tool, and the model calls the tools through the chat API's native tool
calling. One turn:

1. The model gets the conversation so far and the current observation: the head and wrist
   views, the task clock, each arm's measured tool point and gripper opening.
2. It answers with 1 to MAX_CALLS tool calls: ``move``, ``move_by``, ``home``, ``grasp``,
   ``place``, ``press``, ``push``, ``pull``, ``turn``, ``wait`` or ``configure``.
3. The calls run in order, closed loop, and each is answered by a tool message with what it
   measured. A failed or refused call ends the turn's calls.

A target is a head-image pixel normalised to 0..1000, lifted to a world point through the head
depth image the model's views were taken with, or a world point. A motion tool can carry a
``when`` condition: the arms hold until the colour at a head-image pixel changes, then the call
runs, so a reaction does not wait for the next model turn.

The vocabulary follows the C1 level of robotcs_icl's ``unified_prim`` (the model gives a pixel
or a point, the system moves): C2 would need an object segmenter, which roboclock does not ship.

The package:

    tools.py       the tool definitions (names, descriptions, argument schemas) and the check of a call
    settings.py    every quantity: the tools' numeric arguments and the episode settings
    prompt.py      the system prompt, the observation text, the conversation and its trimming
    client.py      the tool-calling request and the reading of its reply
    perception.py  watched colours, openings in the depth image, optional SAM3 masks
    geometry.py    approach orientations, the capsule model of an arm
    skills.py      the closed-loop skills that play a call out as joint commands
    policy.py      the policy the evaluator drives

The agent never reads simulator state beyond the observation: images, head depth, joint
positions, the tool pose, the gripper opening, the clock and the instruction. From the
environment's interface it also takes the action and observation spaces (how far the fingers
can be commanded, how wide the jaws open), which the evaluator passes to ``set_spaces``. What it tells the model is
what was measured; it draws no conclusion from it.

Model-server settings are those of roboclock.policies.vision_llm (VISION_LLM_URL, _API_KEY,
_MODEL, _DIALECT, _THINKING, _THINKING_BUDGET, _REASONING_EFFORT, _MAX_TOKENS, _TIMEOUT,
_RETRIES, _LOG). VISION_LLM_MAX_QUERIES caps the model turns per episode (default 0, no cap);
VISION_LLM_HISTORY sets how many past turns a request carries (default 6). Run it with head
depth:

  roboclock eval --policy roboclock.policies.vision_llm_primitives:make --depth
"""

from roboclock.policies.vision_llm_primitives.policy import MAX_CALLS, PrimitivePolicy, make

__all__ = ["MAX_CALLS", "PrimitivePolicy", "make"]
