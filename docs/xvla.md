# X-VLA (RoboDojo checkpoint) in roboclock

`roboclock.envs.lab.policies.xvla` drives the dual X5 arms with RoboDojo's X-VLA checkpoint
`ckpt/RoboDojo/X_VLA/RoboDojo-sim-arx_x5-ee-0/ckpt-100000` from the HF dataset
`RoboDojo-Benchmark/RoboDojo`: X-VLA-Pt (a 0.9B Florence-2 + soft-prompted flow-matching
policy) fine-tuned for 100k steps on the RoboDojo `arx_x5` demonstrations. It is served by
XPolicyLab's `policy/X_VLA` adapter. For the websocket protocol and the joint mapping shared
with pi0.5 and Wall-X, see [xpolicylab-pi05.md](xpolicylab-pi05.md) and [wallx.md](wallx.md).

## What the checkpoint expects

Read from the adapter (`model.py`, `deploy.py`, `deploy.yml`), its training data handler
(`xvla/datasets/domain_handler/simulations.py:RoboDojoHandler`, `xvla/meta.json`) and the
checkpoint's `config.json`:

- **Inputs.** One image, `cam_head` (RGB; the processor resizes to 224×224 bicubic with
  ImageNet normalisation), the instruction, and a 20-value proprio vector the server builds
  from `state`: `[xyz, rot6d, gripper] × (left, right)`.
  - `xyz` and the rotation are the absolute link6 `ee_pose` in RoboDojo's environment frame,
    the same frame as Wall-X (bases at (∓0.30, −0.45, 0.765), yawed +90°).
  - `rot6d` is the first two columns of the rotation matrix, flattened row-major
    (`[m00, m01, m10, m11, m20, m21]`), in training and in the adapter.
  - The `ee6d` action space zeroes both gripper channels of the proprio, so the gripper state
    sent does not reach the model. The client still sends RoboDojo's (the last commanded
    opening).
  - `domain_id` is 0 in `deploy.yml`, and the training loop maps the `RoboDojo_ee` dataset
    to 0 too (`DATA_DOMAIN_ID.get(name, 0)`).
- **Outputs.** 30 steps of absolute targets (`num_actions` 30, no `idx_for_delta`): per arm,
  xyz, rot6d (decoded by the server to a `wxyz` quaternion) and a sigmoid gripper opening in
  [0, 1], passed through unchanged as `<side>_ee_joint_state`.
  - The training targets are the **state** trajectory of the next 30 recorded frames. The
    handler declares 30 Hz with a 1 s window, so step k is exactly frame t + k of the 25 Hz
    data. In the demonstrations `action[t]` equals `state[t + 1]` to 0.08 mm, so step k is
    RoboDojo's 25 Hz action k.
- **Replanning.** `deploy.py` executes the whole chunk (30 steps, 1.2 s) before it asks again.
  The adapter keeps no history: `update_obs` overwrites one observation slot and
  `get_action` reads it.
- **Protocol.** The client sends the `infer` message, which runs the adapter's `update_obs`
  and `get_action` under one model lock. Nothing is dropped on this path: `get_action`
  returns the action dicts. The server keeps no per-episode state between requests, so
  episodes could share one server. The runs below still used one server per concurrent
  episode.
- **One act() per RoboDojo step.** Run `roboclock eval` with `--policy-hz 25`: each chunk entry
  spans two 50 Hz control steps. The client raises an error at any other rate. The head camera
  is read, and so rendered, only on request frames.
- **Poses to joints.** As for Wall-X: the client reports link6 from the measured joints
  (`wallx.link6_pose`) and solves the returned poses with `wallx.solve_chunk`, chained along
  the chunk from the measured joints.

## Setup (policy machine)

```bash
HF=$ROBODOJO_DIR/XPolicyLab/policy/WallX/.venv/bin/hf   # any huggingface_hub CLI
# Processor/tokenizer only; the X-VLA-Pt weights are not needed for evaluation.
HF_TOKEN=... $HF download 2toINF/X-VLA-Pt --local-dir $MODELS_DIR/X-VLA-Pt --exclude model.safetensors
HF_TOKEN=... $HF download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --include "ckpt/RoboDojo/X_VLA/RoboDojo-sim-arx_x5-ee-0/ckpt-100000/*" --local-dir $ROBODOJO_DIR/hf

A=$ROBODOJO_DIR/XPolicyLab/policy/X_VLA
mkdir -p $A/checkpoints/shared
ln -sfn $MODELS_DIR/X-VLA-Pt $A/checkpoints/shared/X-VLA-Pt          # deploy.yml processor_path
ln -sfn $ROBODOJO_DIR/hf/ckpt/RoboDojo/X_VLA/RoboDojo-sim-arx_x5-ee-0 \
        $A/checkpoints/RoboDojo-sim-arx_x5-ee-0                              # ckpt_name=sim, seed 0

unset TRITON_PTXAS_PATH
uv venv --python 3.10 $A/.venv && source $A/.venv/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
grep -v -E "^#|^$|^av==" $A/xvla/requirements.txt > /tmp/xvla_req.txt
uv pip install -r /tmp/xvla_req.txt
uv pip install -e $ROBODOJO_DIR/XPolicyLab
```

Differences from the published setup (`install.sh`, `setup_eval_policy_server.sh`):

| Item | Published | Here |
|---|---|---|
| Environment | conda, Python 3.10 | uv venv, Python 3.10 |
| torch | unpinned, CUDA 12.8 | 2.8.0, CUDA 12.8 |
| `av==14.4.0` | installed | left out: it has no wheel and its build needs FFmpeg libraries. Only training on video datasets imports it |
| Server start | `setup_eval_policy_server.sh` (conda) | `scripts/xpolicylab/serve_xvla.sh`: the same `setup_policy_server.py` call and overrides (`bench_name=RoboDojo ckpt_name=sim env_cfg_type=arx_x5 action_type=ee seed=0`) from the uv venv, bound to 0.0.0.0 |
| Wrist cameras | the env client sends all three | only `cam_head` is sent: the adapter reads no other camera |

The local adapter (XPolicyLab 5d0be149) matches upstream main in `model.py`, `deploy.py`,
`deploy.yml` and the RoboDojo data handler.

```bash
bash scripts/xpolicylab/serve_xvla.sh $ROBODOJO_DIR 6021 <gpu>
```

- Loading takes about 2 minutes.
- Each server uses about 4 GB of GPU memory (fp32).
- Requests take 0.15–0.35 s on an A100 shared with Isaac (10 denoising steps).

## Run (simulator machine)

```bash
export XPOLICYLAB_URL=ws://127.0.0.1:6021 XPOLICYLAB_LOG=qlog/lab_A5_s0.jsonl
PYTHONPATH=src:$ROBODOJO_DIR python -m roboclock eval \
  --env lab --task A5 --difficulty easy --seeds 1 --seed-start 0 \
  --policy roboclock.envs.lab.policies.xvla:make --policy-hz 25 \
  --inference-latency zero --output out/lab_A5_s0
```

`XVLA_REPLAN` (default 30) shortens the replayed part of each chunk; `XVLA_FRAME_DIR` keeps the
head JPEG of every request. `XPOLICYLAB_LOG` writes one row per request: server and round-trip
time, the state pose and the first target pose, gripper commands, IK error, clipped targets,
and the start jump and travel of each chunk.

## Integration check

An open-loop replay of two RoboDojo demonstrations through the client against the real server
(`replay.py`, in the run's scratch directory). Each demonstration frame becomes a roboclock
observation: the head JPEG, the joints, and the demonstration's gripper state. The client
packs it exactly as in an episode. Every 10th frame is a request, and the 30-step chunk is
compared with the demonstration's next 30 actions. The baseline holds the current pose. Means
over the requests:

| Demo (requests) | Arm | Position, model / hold | Rotation, model / hold | Gripper, model / hold | IK joints, model / hold |
|---|---|---|---|---|---|
| stack_blocks (24) | left | 10.3 / 107.2 mm | 1.2 / 6.6° | 0.010 / 0.182 | 0.023 / 0.181 rad |
| stack_blocks (24) | right | 6.0 / 59.6 mm | 1.0 / 10.6° | 0.008 / 0.093 | 0.015 / 0.134 rad |
| press_by_number (35) | left | 10.0 / 69.3 mm | 1.7 / 16.3° | 0.054 / 0.163 | 0.025 / 0.193 rad |
| press_by_number (35) | right | 5.8 / 40.0 mm | 1.6 / 12.6° | 0.055 / 0.173 | 0.015 / 0.125 rad |

- The model's position error is 7–10 times below hold. The "IK joints" column solves the
  returned chunk with `wallx.solve_chunk` and compares it with the demonstration's action
  joints, so it checks the whole pose-to-joint path.
- The link6 pose the client computes from the joints matches the recorded `ee_pose` within
  0.08 mm on average. The IK leaves at most 0.5 mm.
- roboclock's HOME pose, link6 at (∓0.30, −0.17, 1.01) with the gripper pointing down, lies
  inside the demonstrations' range of positions and orientations.

## Result: the 30-episode comparison set

Setup (2026-11-07): the same 30 episodes as for pi0.5 and Wall-X.

- lab A5/B1, hospital A3/B4, household A2/B2, sports A3/B2 and industry A1/B2;
- seeds 0–2, all at easy, including industry;
- `--inference-latency zero` and `--policy-hz 25`.

Two servers (ports 6021 and 6031) were used, with two episodes at a time.

| | Joint | Physical | Stages reached |
|---|---|---|---|
| X-VLA | 0/30 | 3/30 (industry B2) | industry B2 `stop_button_pressed` + `physically_complete` 3/3 (score 80); household A2 `glass_picked_up` 3/3 (score 20) |
| Wall-X (zero latency) | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 |
| pi0.5 (measured latency) | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 |

Neither stage is a real task skill:

- **industry B2.** The physical successes are premature presses. The button goes down at 2.3 s
  (`premature_press_s`), before the warning, in all three seeds. The task wants the press
  *after* the warning, so the temporal check fails (`temporal_violation_rate` 1.0).
- **household A2.** The glass is knocked over at 1.3 s in every seed (`glass_tipped`).
  `glass_picked_up` fires whenever the glass leaves the counter, so the stage counts the
  knock-over, not a grasp.

Failure modes, from the termination reasons:

| Task | Failure |
|---|---|
| household A2 | `glass_tipped` at 1.3 s (all seeds). |
| household B2 | `object_fell` within 3–4 s. |
| lab A5 | `developer_loading_timeout` at 15 s. |
| lab B1 | `overheat`: the beaker is never lifted. |
| hospital A3 / B4 | No progress before the end. |
| sports A3 | `no_grasp`. |
| sports B2 | `moved_during_hold`, `incomplete_combinations`. |
| industry A1 | No progress in the 95 s budget. |
| industry B2 | Premature press (see above). |

- **Server time.** 297 ms per request at p50 and 448 ms at p95, over 801 requests. This
  includes the wait for the shared lock.
- **IK.** The largest misses are in industry: 60 mm in A1 s0 and 45 mm in all of B2. Lab B1 s0
  reaches 34 mm. Elsewhere the misses stay under 12 mm. No target was clipped by the joint
  limits.
- **Start jump.** The first chunk jumps 0.8 rad in industry A1, 1.7 rad in industry B2 and
  1.2–1.4 rad in hospital A3 away from roboclock's HOME pose. RoboDojo episodes start at joint
  zero.
- Results and per-request logs are in the session scratchpad, `models/xvla/zero/`. The
  overview is in `final_summary.txt`.

Conclusion:

- The replay reproduces RoboDojo demonstrations at 7–10× below the hold error, so the
  integration is correct.
- In roboclock the checkpoint does not solve any task. Its only "successes" come from reaching
  the industry stop button before the warning.
