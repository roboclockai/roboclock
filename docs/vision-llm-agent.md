# Vision-LLM agents for the dual-ARX-X5 domains

roboclock has three agents that drive both ARX X5 arms with a vision-language model
served through an OpenAI-compatible API (Qwen3.8-27B under vLLM by default):

| Agent | Policy | What the model answers |
|---|---|---|
| Waypoints | `roboclock.policies.vision_llm:make` | Up to 4 waypoints per arm (head-image pixel, world point or offset), tool orientation, gripper, and a wait |
| Primitives | `roboclock.policies.vision_llm_primitives:make` | 1 to 4 tool calls (native tool calling): `move`, `move_by`, `home`, `grasp`, `place`, `press`, `push`, `pull`, `turn`, `wait` or `configure` |
| move_eef | `roboclock.policies.vision_llm_move_eef:make` | Exactly one tool call: `move_eef` (absolute world-frame tool-point targets, RoboDojo's interface), `wait` or `give_up`. RGB only; see [the section below](#the-move_eef-agent) |

All three work in every domain whose rig follows the RoboDojo spec
([robodojo-rig-spec.md](robodojo-rig-spec.md)): lab, hospital, household, industry
and sports.

## What the agents see

Every model turn gets:

- the head image (640x480) and both wrist images (320x240);
- the instruction and the clock;
- each arm's measured tool point and gripper opening;
- what the previous turn's motions measured.

The observation also includes head depth when the evaluation runs with `--depth`. The
agents use it only to lift a clicked pixel to a world point.

The agents read nothing else. In particular they never see object poses, task state,
event times or success signals. Two inputs are idealised compared with a real robot:

- Depth is the simulator's exact depth, with no noise.
- The tool pose is exact. On a real robot it would come from forward kinematics and the
  arm's base calibration.

At reset each agent recovers both arm bases from the observed joints and tool poses
(`X5Kinematics.calibrated`). It then derives the head camera pose from the bases and
the rig spec, not from the scene camera. In Isaac, the derived camera matched each
domain's scene camera:

| Domain | Error |
|---|---|
| lab, hospital, industry | 0 mm |
| household | 3 mm |
| sports | 4 mm (its tool offset is 0.14 m, not 0.145 m) |

## Primitives

The agent is the package `roboclock.policies.vision_llm_primitives`:

| Module | What it holds |
|---|---|
| `tools.py` | The tool definitions (name, description, argument schema) and the check of a call |
| `settings.py` | Every quantity: the tools' numeric arguments and the episode settings |
| `prompt.py` | The system prompt, the observation text, the conversation and its trimming |
| `client.py` | The tool-calling request and the reading of its reply |
| `perception.py` | Watched colours, openings in the depth image, optional SAM3 masks |
| `geometry.py` | Approach orientations, the capsule model of an arm |
| `skills.py` | The closed-loop skills that play a call out as joint commands |
| `policy.py` | The policy the evaluator drives |

The vocabulary follows the C1 level of robotcs_icl's `unified_prim`: the model gives a
pixel or a point, and the system moves. C2 (the model names a target) needs an object
segmenter. With SAM3 running (see [SAM3 grasp refinement](#sam3-grasp-refinement)), a
grasp can also name its target, and the system refines the grasp from the object's mask.

### Tools

Each primitive is a tool, and the model calls the tools through the chat API's native
tool calling (`tools`, `tool_choice: "required"`, `parallel_tool_calls`). A tool's
definition carries its description and, for each argument, a description and the range
of a number (`minimum`, `maximum`). Nothing about a tool is described anywhere else.

| Tool | Behaviour | Arguments |
|---|---|---|
| `move` | Straight-line move of the tool point to a target raised by `height` | `arm`, `to`, `height`, `approach`, `yaw`, `tilt`, `gripper`, `opening`, `speed`, `when` |
| `move_by` | Straight-line move by an offset from the measured tool point, keeping the hand's orientation. With `guard` it stops on contact and stays pressed against it; a later offset keeps that press unless it points away from the contact | `arm`, `by`, `guard`, `gripper`, `opening`, `speed`, `when` |
| `home` | Back to the pose the arm started the episode in | `arm`, `gripper`, `opening`, `speed`, `when` |
| `grasp` | Open, approach from the stand-off, go `depth` past the target under guard, close with `squeeze`, lift by `lift` | `arm`, `object`, `approach`, `yaw`, `tilt`, `depth`, `opening`, `squeeze`, `lift`, `when` |
| `place` | Rise to the clearance above the release point, move over it, lower (under guard for `on` and `into`), open to `opening` (default fully), back off. With `yaw` the hand first turns about the vertical so the jaws close along it | `arm`, `target`, `mode`, `height`, `yaw`, `opening`, `when` |
| `press` | Closed fingers from the press stand-off, `depth` past the point, `hold`, back off. Without a target, from where the hand is | `arm`, `target`, `approach`, `yaw`, `tilt`, `depth`, `hold`, `when` |
| `push` | Closed fingers come down behind the point, then slide by `by`. Without a target, the hand moves by `by` | `arm`, `by`, `target`, `approach`, `yaw`, `tilt`, `when` |
| `pull` | Close on the point as a grasp does, drag by `by`, release. Without a target, the hand moves by `by` | `arm`, `by`, `target`, `approach`, `yaw`, `tilt`, `depth`, `opening`, `squeeze`, `when` |
| `turn` | Close on the point, turn the hand about its finger axis by `angle`, release. Without a target, the hand turns where it is. With `hinge` the hand swings on an arc about a world line | `arm`, `angle`, `target`, `hinge`, `approach`, `yaw`, `tilt`, `depth`, `opening`, `squeeze`, `when` |
| `together` | The next two calls of the turn run at the same time, one per arm | none |
| `wait` | Hold both arms for `seconds`, until `until_t`, or until a watched pixel changes | `seconds`, `until_t`, `watch`, `timeout` |
| `configure` | Change settings for the rest of the episode; returns all of them | the 19 settings |

Four additions came out of the 2026-11-10 GPT 6 Astra sweep, where about 22 of 114 failed
episodes needed a motion the tools could not express:

- **`tilt`** (`{pitch, roll, heading}`, radians). `top` and `front` are two points of one
  family: fingers along `heading` in the table plane, `pitch` below level (front 0, top 1.57),
  jaws turned `roll` about the fingers from closing level. A tilt reaches the rest: a hand
  20 degrees below level, or jaws that close vertically around a flat tab. It needs approach
  `top` or `front`. Where the joints cannot give the asked orientation, the result says how many
  degrees off the hand is.
- **`hinge`** (`{pivot, axis}`) on `turn`. The hand swings on an arc about the world line
  through `pivot` along `axis` (default vertical), turning with it: a lever, a crank, a door.
  With `axis` alone the line passes through the tool point, which tilts what the hand holds.
  An arc the arm cannot follow, in position or in orientation, is refused before it moves.
- **`guard`** on `move_by`. The hand stops when it meets something and keeps pressing by the
  `contact` setting. Its end point may lie beyond reach. Two hands closing in this way hold a
  thing too wide for one gripper.
- **`together`**. The two calls after it run in the same control steps, one per arm; each has
  its own result, and one failing does not stop the other. The clearance check is made against
  where the other arm is when they start.

`tilt` and `hinge` are objects so that no tool takes more than eleven arguments.

In Isaac (scripted calls, no model): a hinge swing of 1.4 rad about the lab A2 valve's pivot
closed the valve (measured swing 79 of 80 degrees); a guarded offset stopped on the bench
0.214 m short of its end point; two `move_by` calls run together both took 0.4 s and ended in
the same step. Whether a model uses them well is not measured yet.

A target is a head-image pixel `{u, v}` normalised to 0..1000, lifted to a world point
through the head depth image the model's views were taken with, or a world point
`{xyz}`. A tool that moves an arm can carry `"when": {"u", "v", "timeout"}`: the arms
hold until the mean colour of the patch at that pixel changes, and the call runs at
once. A reaction therefore does not wait for the next model turn. The agent checks the
patch every 0.1 s.

### One turn

1. The request carries the system prompt, the past turns still in view and the current
   observation: the three views, the clock, and each arm's measured tool point and
   gripper opening.
2. The model answers with 1 to 4 tool calls.
3. Each call is checked. A call with an unknown argument, a missing one, or a number
   outside its range is answered `REFUSED: ...` and does not run.
4. The calls run in order. Each is answered by a tool message with what it measured. A
   failed or refused call ends the turn: the calls after it are answered `not run`.
5. The turn (observation, tool calls, results) joins the conversation.

A reply with no tool call costs a 0.5 s hold, and the next observation says so. So does a
request the provider's content filter refuses: the turn is lost, not the episode.

### The conversation

The messages follow the standard tool-calling order: `system`, then for each past turn
`user` (its observation), `assistant` (its tool calls) and one `tool` message per call,
then `user` (the current observation). The task's instruction closes the system prompt,
so it stays in view whatever is trimmed.

- A request carries at most `VISION_LLM_HISTORY` past turns (default 6; 0 makes every
  turn stand alone). Past that the older half is dropped at once, so the messages before
  the latest turns stay identical from request to request. A provider's prompt cache
  needs that: with GPT 6 Astra about 6.1k of 7 to 8k prompt tokens are read from cache.
- The current observation has all three views, the previous turn keeps its head view,
  and older turns keep their text only. The model can then compare two head views, for
  example to see that something stopped moving.
- If a server refuses a request for exceeding the model's context, the agent drops the
  older half of the past turns and asks again.

### The model sets every quantity

Each number a primitive uses is the model's to set. They come in two kinds
(`settings.py`).

Call quantities are arguments of the tools:

| Argument | Default | Range | Meaning |
|---|---|---|---|
| `height` | move 0, on 0, into 0.05, drop 0.10 m | -0.1 to 0.5 | The target is raised by this |
| `depth` | grasp 0.02, pull/turn 0.01, press 0.015 m | -0.05 to 0.15 | How far past the target point the approach goes |
| `opening` | 0.088 m before a grasp | 0.002 to 0.088 | Distance between the jaws |
| `squeeze` | 0.01 m | 0.0005 to 0.03 | How far past where it stopped each finger is commanded (grip force) |
| `lift` | 0.08 m | 0 to 0.4 | Rise after a grasp |
| `hold` | 0.3 s | 0 to 10 | How long a press stays down |
| `yaw`, `angle` | - | ±2π, ±π rad | Jaw direction of a top approach; how far a turn rotates |
| `speed` | the setting | 0.01 to 1 m/s | Tool speed of this call |
| `seconds`, `until_t`, `timeout` | - | up to 60 s | Wait and watch durations |

Settings hold for the rest of the episode once `configure` sets them:

| Setting | Default | Range | Meaning |
|---|---|---|---|
| `speed` | 0.25 m/s | 0.01 to 1 | Tool speed of unguarded motion |
| `approach_speed` | 0.10 m/s | 0.005 to 0.5 | Guarded approaches, lowering, drags, pushes, press strokes |
| `turn_speed` | 1.5 rad/s | 0.1 to 4 | How fast the hand reorients |
| `gripper_time` | 0.5 s | 0.1 to 5 | Time of a full open or close stroke |
| `standoff`, `press_standoff` | 0.10, 0.05 m | 0.01 to 0.3 | Where an approach, or a press stroke, starts |
| `contact` | 0.012 m | 0.002 to 0.1 | Lag of the measured hand behind its command that counts as contact |
| `tolerance` | 0.02 m | 0.001 to 0.2 | IK miss beyond which a call fails before the arm moves |
| `arrival` | 0.005 m | 0.001 to 0.05 | A motion ends once the measured tool point is this near where it was sent |
| `arm_clearance` | 0.02 m | -0.05 to 0.2 | Smallest gap a motion may leave between the two arms |
| `settle` | 0.3 s | 0.05 to 3 | Pause after closing before the opening is read |
| `min_opening` | 0.003 m | 0.0001 to 0.088 | A close that ends under this opening fails |
| `clearance` | 0.12 m | 0.01 to 0.5 | Carry height of a place above its release point |
| `retreat` | 0.08 m | 0.005 to 0.3 | Back-off after a release or a push |
| `push_offset` | 0.04 m | 0.005 to 0.2 | How far behind the point a push starts |
| `rim_radius` | 80 | 5 to 150 | `place into`: how far around the pixel (0..1000 image units) the opening's rim is looked for |
| `watch_timeout`, `watch_change`, `watch_radius` | 5 s, 40, 5 | up to 60 s, 1 to 441, 1 to 50 | Longest a watch holds, its threshold (RGB distance) and its patch half-size |

Until 2026-11-10 every one of these was an argument of every call that used it (up to 25
arguments on one call). Models then set them badly call by call: a `rim_radius` of 16 that
missed the opening, 1 m/s on every motion. As settings they keep their defaults unless
the model decides to change them.

A value outside its range is refused with a message naming the range. Nothing is clamped
silently. An argument written as `null`, or as 0 where 0 is outside its range, is read
as not given: decoders behind some gateways fill unused fields that way.

### Grip force

The fingers are position drives, so the grip force is the drive's stiffness times how
far the finger command is past the finger, up to the drive's effort limit:

| Domain | Finger stiffness | Effort limit | Lowest finger target |
|---|---|---|---|
| lab, hospital, sports | 1000 N/m | 10 N | 0 (the closed stop) |
| industry | 1500 N/m | 20 N | 0 |
| household | 800 N/m | 20 N | -0.03 m |

A close runs the fingers in until they stop, waits `settle`, then commands each finger
`squeeze` past where it stopped. While closing, the command leads the measured finger by
at most max(`squeeze`, 4 mm), so the fingers never arrive harder than the grip that was
asked for. The result states the squeeze applied.

The command cannot go below the rig's lowest finger target. Where that is 0, a part of
thickness w can be squeezed by at most w/2 per finger: about 3 N on a 6 mm handle at
1000 N/m. Household accepts targets down to -0.03 m, which its oracle uses. The limit is
the lower bound of the environment's action space. A policy factory gets no environment,
so `load_policy` passes the action and observation spaces to a policy that defines
`set_spaces`: they are part of the robot's interface, not of the task. The widest the jaws
open is the upper bound of the `gripper_width` observation: 88 mm on the lab X5, 60.3 mm on
household's (its bound said 88 mm until 2026-11-10).

### Following: the joints' speed limit

Every rig caps its joints' speed (industry at 0.70 to 0.75 rad/s) and the agent is not told
the cap. A trajectory timed from the tool speed alone can ask for several times that: the
default 0.25 m/s and 1.5 rad/s command industry's wrist pitch at up to 2.4 rad/s when the
hand goes from its start pose to level. The command then runs away from the arm, the hand
leaves its line, and on industry's arms the joint is left a radian behind and stops closing
the gap (sweep v3: B2's first move ended 0.25 m from its target, in both seeds).

So a trajectory advances only while that leaves no commanded joint more than `JOINT_LEAD`
= 0.08 rad ahead of the measured one (`skills.Pace`). A motion asked faster than the arm
can go runs at the arm's own limit, on its path, and its result states the time it took.
A position drive follows at about stiffness / damping x the lead, so 0.08 rad allows
0.8 rad/s on industry's arms, 2 rad/s on lab's and 2.7 rad/s on household's. Measured on
industry B2's first move (2.2 rad of wrist pitch): 5.5 s at a lead of 0.05, 3.6 s at 0.08,
5.2 s at 0.15; at 0.3 and 0.6 the joint creeps and never arrives. With the default speeds
and nothing configured the scripted B2 now succeeds (response 0.8 s).

A joint that carries a load settles further than that from its command and still follows
(a wrist lifting its side of a carton stood 0.1 rad off). So when the command has waited
0.3 s and no joint has moved 0.02 rad, the lead the arm stands at is taken as its standing
offset, once, and the lead is counted from it. If the arm still has not moved after another
0.3 s it is held off: a guarded motion reports a contact stop, any other stops there, with
the command at most twice the lead ahead of the arm, and its result says how far short of
its goal the hand stopped. What follows is planned from there, not from the goal it never reached.

### Arriving

The drives trail a moving command: by about 1 cm at 0.25 m/s and by 20 to 30 cm at
1 m/s, which a model may ask for. A commanded trajectory can therefore end long before
the hand is there. Every motion waits until the measured tool point is within `arrival`
of its commanded end, or the arm has not come nearer for 0.3 s (it is held off), or 20 s
have passed (industry's arms are slow: one 0.5 m move took over 4 s). Only then does the skill close, release, press or report. A guarded motion
that is held off further away than `contact` is reported as a contact stop.

Without this wait, a first smoke run at the model's chosen 1 m/s released tubes while
the hand was still descending and pressed a button from 25 cm away.

For the first 0.3 s of a guarded motion a trailing hand is not yet taken for a stall,
because it is still setting off.

### The other arm

One arm moves per call and the other holds still where it is. Paths are straight lines, so
the moving arm can run into the waiting one. Each arm is modelled as capsules around its
links (radius 4 cm) and its hand (5.5 cm, to 2 cm past the tool point), from the
kinematics alone. Before a motion starts, its path is checked at 12 poses against the
other arm's measured pose. If the gap would fall under `arm_clearance` and below what it
is at the start, the call fails without moving and says so. A path that moves the arms
apart always passes, so two hands that already work close together are not stuck.

The capsules do not include what a hand holds, and nothing avoids the scene: both stay
the model's to judge from the views.

### Placing into an opening

A pixel that looks into a well, a slot or a cell lands on its inner wall or floor, beyond
the spot it points at. Releasing over that point puts the object on the far rim (in
hospital C4 the clicks were right and the release points 13 to 21 mm off).

For `place` with mode `into` and a pixel target, the agent looks for the opening's rim in
the depth image. Walking outward from the pixel in 8 image directions, the surface climbs
the inner wall, and the rim is where it first stops climbing, at least 5 mm above the
clicked point and within `rim_radius`. When 6 directions find that at one height (within
12 mm), the release moves sideways to where the pixel's line of sight crosses the rim's
plane. The result gives the surface
point, the rim height and the release point. A spot on a flat surface, a spot beside a
tall object, `place on` and world-point targets are left as given.

On the depth images of hospital C4, sports C3 and lab C1, this finds the rack wells
(release moved 22 to 27 mm), the hopper cells (about 5 cm) and the rack slots (2 to
3 cm), and leaves table tops, object tops and the arms alone.

A guarded `on` or `into` still opens where contact stops it. The result says how far
above the release point that was.

### What the model is told

The observation of a turn has the same form in every domain:

```
Time now: 12.34 s on the task clock (your previous turn was at 8.10 s).
left arm: measured tool point [x, y, z], measured gripper opening 0.027 m (commanded 0.007 m)
right arm: ...
```

Everything the model is told is a measurement. The agent draws no conclusion: it never
says that something is held, moved or in place, and it gives no advice on how to grasp.
Tool results use the tool point at the measured joints, not the commanded one, and say
when the two differ by more than 2 cm. A grasp reports the opening after closing and
again after the lift.

A call fails when:

- a close ends with the opening under `min_opening` (the fingers met);
- a pixel has no depth, or the IK misses the pose by more than `tolerance`. The reach is
  checked before the arm moves;
- the path would bring the moving arm closer than `arm_clearance` to the other arm;
- its `when` times out.

The result then still states what was measured before the failure (the world point a
pixel landed on, a contact stop).

Skills run in closed loop on proprioception only:

- A guarded motion stops when the measured tool point trails the commanded one by more
  than `contact` and has nearly stopped advancing over the last 0.1 s.
- When a pose is out of reach with the requested orientation, the IK retries with the
  orientation almost free.

Each turn lasts at least 0.1 s of simulated time.

Only the turns that call the model count as queries. Under the block-and-replay time
model, those are the only steps charged inference latency. The steps a skill plays out
are not charged.

### Reasoning

Every request states how the model should reason (`client.py`):

| Dialect | Thinking on | Thinking off |
|---|---|---|
| `vllm` | `enable_thinking` with a `thinking_token_budget`, temperature 0 | `enable_thinking: false`, temperature 0 |
| `openai` | `reasoning_effort` (default `medium`), no temperature | temperature 0 |

Reasoning models take no temperature, so none is sent with `reasoning_effort`. The log
records each turn's reasoning tokens.

## SAM3 grasp refinement

A pixel target of `grasp`, `pull` or `turn` may carry a name:
`{"u": 520, "v": 610, "name": "red cup"}`.

When `VISION_LLM_SAM3_URL` points to a SAM3 server, this follows robotcs_icl's
`skills/sam3_anchor.py`:

1. When the model answers, the agent segments the name in the head image.
2. The instance under the clicked pixel is kept. If none is under it, the nearest one
   within 25 px is kept.
3. The mask's pixels are lifted to world points through the head depth.

The grasp then goes to the object's top centre (`top`) or its mid-height centre
(`front`), with the jaws across its narrow side. The depth comes from the object's size
unless the model gave one.

The result shown to the model reports the object's top height, its footprint and how far
the grasp moved from the click. If nothing is found under the click, the grasp uses the
clicked point and says so. No simulator state is read.

The server is `scripts/xpolicylab/serve_sam3.py`. It runs with transformers 5 (the vLLM
venv has it) and the `facebook/sam3` weights. Those weights are gated on Hugging Face:
request access, then download with your token. The SAM License applies.

```bash
hf download facebook/sam3 --local-dir $MODELS_DIR/sam3
python scripts/xpolicylab/serve_sam3.py $MODELS_DIR/sam3 --port 8232 --gpu 6
VISION_LLM_SAM3_URL=http://127.0.0.1:8232 roboclock eval ... --policy roboclock.policies.vision_llm_primitives:make --depth
```

The server's endpoints:

- `/segment` uses text prompts, with the request and reply format of robotcs_icl.
- `/point` uses point prompts. It is enabled only when the checkpoint holds all of the
  tracker's weights. The server refuses to start if any detector weight is missing, so
  no part of the model ever runs on random initialisation.

## Running

Serve the model:

```bash
VLLM_GPU_UTIL=0.85 VLLM_MAX_NUM_SEQS=16 VLLM_MAX_MODEL_LEN=12288 \
  bash scripts/xpolicylab/serve_qwen.sh $MODELS_DIR/Qwen3.8-27B 8000 2
```

Qwen3.8 mixes linear and full attention, and vLLM keeps one Mamba cache block per
sequence. At a memory fraction below about 0.9, set `VLLM_MAX_NUM_SEQS` to 16 or the
server refuses to start.

Evaluate:

```bash
VISION_LLM_URL=http://127.0.0.1:8000/v1 VISION_LLM_LOG=qwen.jsonl \
  roboclock eval --env industry --task B2 --difficulty reviewed_nominal --seeds 1 \
    --policy roboclock.policies.vision_llm_primitives:make --depth --inference-latency zero
```

Settings, shared by both agents:

| Variable | Default | Meaning |
|---|---|---|
| `VISION_LLM_URL` | `http://127.0.0.1:8000/v1` | Model server or hosted API (OpenAI-compatible) |
| `VISION_LLM_API_KEY` | unset | Bearer token of a hosted API; never logged |
| `VISION_LLM_MODEL` | `Qwen/Qwen3.8-27B` | Served model name |
| `VISION_LLM_DIALECT` | `vllm` | `vllm` sends vLLM's thinking controls; `openai` sends only standard OpenAI fields |
| `VISION_LLM_THINKING` | `1` | Set to `0` to turn thinking off |
| `VISION_LLM_THINKING_BUDGET` | `1024` | Thinking tokens per turn (`vllm`); with `openai` it only raises `max_tokens` |
| `VISION_LLM_REASONING_EFFORT` | `medium` (primitives), unset (waypoints) | `openai` with thinking: the `reasoning_effort` sent |
| `VISION_LLM_MAX_TOKENS` | `1024` + budget | Output tokens per turn, reasoning included |
| `VISION_LLM_TIMEOUT` | `300` | Request timeout in seconds |
| `VISION_LLM_RETRIES` | `3` | Retries after HTTP 408/429/5xx or a dropped connection (5, 10, 20, ... s apart) |
| `VISION_LLM_RETRY_TIMEOUTS` | `0` | Primitives only: `1` also retries a request that timed out (a hosted API's stalled connection) |
| `VISION_LLM_LOG` | unset | JSONL file with one row per request |
| `VISION_LLM_MAX_QUERIES` | `0` (no cap) | Primitives only: model turns per episode, then both arms hold |
| `VISION_LLM_HISTORY` | `6` | Primitives only: past turns a request carries (0: every turn stands alone) |
| `VISION_LLM_SAM3_URL` | unset | SAM3 server for named grasp targets (primitives only) |

### Hosted APIs

Any OpenAI-compatible chat API works, for example NVIDIA's inference API, which serves Gemini,
GPT, Claude, Qwen and other models under ids such as `gcp/google/gemini-4-argon`:

```bash
export VISION_LLM_URL=https://inference-api.example.com/v1 VISION_LLM_DIALECT=openai
export VISION_LLM_MODEL=gcp/google/gemini-4-argon VISION_LLM_API_KEY=...   # key from the key file
roboclock eval --env lab --task A5 --difficulty easy --seeds 1 \
  --policy roboclock.policies.vision_llm_primitives:make --depth --inference-latency zero
```

- With `openai` the request leaves out `chat_template_kwargs` and `thinking_token_budget` and
  sends `reasoning_effort`. The provider's reasoning is not capped at 1024 tokens the way the vLLM
  models' is; `max_tokens` caps reasoning plus answer.
- Gemini answers pixel targets on the same 0-1000 grid the prompt asks for, and its replies
  report reasoning tokens inside `completion_tokens`.
- The tools go to the API as `tools`, so the provider's own tool calling formats and parses the
  calls. The check of a call still drops what a decoder writes only to fill a schema: `null`, a 0
  where 0 is outside the range, an empty or all-zero point, the zero half of a point given both
  as a pixel and as a world point (of two real halves the pixel wins).
- A request the provider's content filter refuses (Azure OpenAI: HTTP 400 `content_filter`) costs
  the primitive agent one turn, not the episode.
- Under `--inference-latency measured` the round trip counts, network and provider queue included.
- Hosted APIs bill each turn. A primitive turn sends 7 to 9k prompt tokens with GPT 6 Astra, about
  6.1k of them (the tools and the system prompt) read from the provider's cache after the first
  turn: about $0.03 a turn. Cap the turns with `VISION_LLM_MAX_QUERIES`.
- [vision-llm-sweep.md](vision-llm-sweep.md) runs many models, local or hosted, over the whole
  task list.

The log rows include the calls, the results shown to the model, the guarded-approach
diagnostics and the model's reasoning.

With thinking on, a request takes about 40 s on one A100. Under
`--inference-latency measured`, that time is charged to the simulation. Most tasks then
end after 2 to 3 turns, so compare agents under `--inference-latency zero` as well.

`--inference-latency ideal` charges each turn its output tokens, thinking included, at
the model's fastest surveyed decode speed. The speeds are in
`src/roboclock/evaluation/model_speeds.yaml`; Qwen3.8-27B is listed at 1850 tokens/s
(Cerebras). A primitive turn of about 1115 tokens is then charged about 0.6 s. See
[architecture.md](architecture.md#inference-latency), and
[ideal-latency-eval.md](ideal-latency-eval.md) for how to configure and run it.

## The move_eef agent

`roboclock.policies.vision_llm_move_eef` is the action surface of RoboDojo's GPT-6 Astra
evaluation (arXiv 2609.24170, Appendix A; code in XPolicyLab
`policy/GPT_6_Astra_Direct_EEF`, originally RoboProbe), rebuilt on roboclock's observation
and action interface. It has no manipulation primitive: the model names where the tool
points go, and nothing else.

```bash
roboclock eval --env industry --task B2 --policy roboclock.policies.vision_llm_move_eef:make
VLM_AGENT=move_eef bash scripts/xpolicylab/sweep/start.sh nv/azure/openai/gpt-6-astra:1,2,3
```

The tool is `move_eef(targets, note)`. `targets` maps any of 14 names to a value; unnamed
dimensions hold.

| Dimension | Left, right | Meaning |
|---|---|---|
| Position | `left_x`, `left_y`, `left_z`, `right_…` | World frame, metres: the point between the jaws. Bounds: the arm's base ± 0.80 m |
| Orientation | `…_pitch_deg`, `…_roll_deg`, `…_yaw_deg` | Degrees from a straight-down grasp (fingers at world −z, jaws closing along world +x). Pitch turns about world +x (−180 to 180), roll about +y (−90 to 90), yaw about +z (−180 to 180), applied in that order |
| Gripper | `left_gripper`, `right_gripper` | 0 closed (the rig's lowest finger target: it keeps squeezing) to 1 fully open |

How a call plays out:

- Naming both arms moves them together along time-aligned paths.
- A gripper value named with a pose moves the jaw after the arm has arrived.
- The model has no argument for speed or duration. No joint is commanded faster than
  0.6 rad/s (`VISION_LLM_EEF_JOINT_SPEED`), so the time follows from the distance. A motion
  over 10 s (`VISION_LLM_EEF_MAX_DURATION`) is refused and has to be split.
- An unreachable pose, a path that would bring the two arms' bodies within 2 cm, or a bad
  value is refused, and nothing moves. With both arms named, one refusal stops both.
- The next turn starts once the arms have come to rest (at most 2 s after the last
  waypoint). Its observation states all 14 dimensions as measured, the joint angles, the
  clock, and how far the tool point landed from the previous target.
- `wait(seconds, note)` holds both arms still for 0.1 to 10 s of task time and costs one
  model call. RoboDojo has no such tool; `VISION_LLM_EEF_WAIT=0` removes it and leaves
  RoboDojo's two tools.
- The model answers with exactly one tool call per turn. Parallel `move_eef` calls that
  name different dimensions are folded into one motion.
- The budget is 100 model calls per episode (`VISION_LLM_MAX_QUERIES`; in a sweep,
  `VLM_EEF_MAX_QUERIES`). There is no "done" tool. `give_up` stops the queries; the arms
  hold until the task is judged.

Differences from RoboDojo's implementation:

| | RoboDojo | roboclock |
|---|---|---|
| Path | CuRobo plan against the scene | Straight line of the tool point (IK along it), or the joints directly to a solution of the target where they cannot follow the line; only the two arms are kept apart, because a policy sees no scene |
| Joint speed | Each joint's velocity limit, read from the simulator | One constant under every rig's cap |
| After the motion | Next turn at once | Waits until the joints have stopped: roboclock's drives trail their command |
| Unnamed dimensions | Hold the observed value | Hold the value the arm was sent to while it stands within 2 cm of it, the observed value otherwise (the drives sag about 1 cm at full reach, and the hand would sink by that on every call) |
| Frame | Fixed bases, table at z = 0.765 | The rig's own; the system prompt states the bases, the facing direction and the head camera's position |
| Images in the conversation | All of them | The latest `VISION_LLM_HISTORY` turns, with the previous turn's head view |
| Budget shown to the model | Env steps remaining | The task clock |
