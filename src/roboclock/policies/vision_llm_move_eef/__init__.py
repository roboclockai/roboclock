"""Vision-LLM agent whose only motion tool is `move_eef`: absolute world-frame tool-point targets.

This is the action surface of RoboDojo's GPT-6 Astra evaluation ("An Unexpected Robot Policy",
arXiv 2609.24170, Appendix A; code: XPolicyLab policy/GPT_6_Astra_Direct_EEF, originally
RoboProbe policy/RoboDojo_Agent_L3_Inspect_EEF), rebuilt on roboclock's observation and action
interface. It is the geometric-control counterpart of roboclock.policies.vision_llm_primitives:
no grasp, place or press primitive, no pixel targets, no depth, no reaction (`when`), and no
argument for speed or duration.

One turn:

1. The model gets the conversation so far and the current observation: the head and wrist RGB
   views, the task clock, both arms' tool-point state in the 14 dimensions the tool takes, the
   joint angles as context, and how far the previous motion landed from its target.
2. It answers with exactly one tool call: `move_eef(targets, note)`, `wait(seconds, note)`
   or `give_up(reason)`.
3. An accepted motion plays to its end and the arms come to rest before the next turn. A
   refused one moves nothing, and the model reads why.

`targets` maps any of 14 names to a value: per arm x, y, z (metres, world frame, the point
between the jaws), pitch_deg, roll_deg, yaw_deg (degrees from a straight-down grasp) and
gripper (0 closed to 1 open). Unnamed dimensions hold. Naming both arms moves them together
along time-aligned paths. A gripper value named with a pose moves the jaw after the arm has
arrived.

What follows RoboDojo: the tool's schema and wording, the angle convention (pose.py), holding
unnamed dimensions, the refusal of an unreachable pose without moving, the playout at a fixed
per-joint step limit with the duration following from the distance, the 10 s cap, the jaw
sequenced after the arm, the merging of parallel move_eef calls, the arrival check, the budget
of 100 model calls, and the absence of a "done" tool.

What differs, because roboclock is not RoboDojo:

- Planner. RoboDojo plans with CuRobo against its scene. A roboclock policy sees only the
  observation, so there is no scene to plan against: the tool point goes in a straight line
  (damped least-squares IK along it; where the joints cannot follow the line, they go directly
  to a solution of the target), and only the two arms' bodies are kept apart.
- Joint speed. RoboDojo reads each joint's velocity limit from the simulator. Here it is one
  constant, JOINT_SPEED = 0.6 rad/s, under every rig's cap (VISION_LLM_EEF_JOINT_SPEED).
- Coming to rest. roboclock's drives trail their command, so after the last waypoint the agent
  waits (at most 2 s) until the joints have stopped before it asks the model again.
- Unnamed dimensions hold the value the arm was sent to while it stands within 2 cm of it,
  and the measured value otherwise: holding the measured one always would sink the hand by
  the drives' sag on every call.
- Frame. The world frame and the arm bases are the rig's own; the system prompt states the
  bases, the facing direction and the head camera's position. Position bounds are each base
  +- 0.80 m.
- Conversation. Only the latest VISION_LLM_HISTORY turns are sent, with the previous turn's
  head view (RoboDojo keeps every image), and the clock replaces "env steps remaining".
- `wait` is not RoboDojo's: it lets task time pass (0.1 to 10 s) for one model call, where
  RoboDojo's model has to spend a call per control step. VISION_LLM_EEF_WAIT=0 removes it.
- `give_up` cannot end a roboclock episode: the arms hold still until the task is judged.

Model-server settings are those of roboclock.policies.vision_llm (VISION_LLM_URL, _API_KEY,
_MODEL, _DIALECT, _THINKING, _REASONING_EFFORT, _MAX_TOKENS, _TIMEOUT, _RETRIES, _LOG).
VISION_LLM_MAX_QUERIES caps the model calls per episode (default 100; 0 = no cap). Run it
without depth:

  roboclock eval --policy roboclock.policies.vision_llm_move_eef:make
"""

from roboclock.policies.vision_llm_move_eef.policy import MoveEefPolicy, make

__all__ = ["MoveEefPolicy", "make"]
