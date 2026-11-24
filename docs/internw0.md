# InternW0-Δ (RoboDojo) in roboclock

`roboclock.envs.lab.policies.internw0` drives the dual X5 arms with
[`InternRobotics/InternW0-Delta-RoboDojo`](https://huggingface.co/InternRobotics/InternW0-Delta-RoboDojo)
(`robodojo.pt`, step 9090). InternW0-Δ is a world-action model: a Wan2.2-TI2V-5B video expert
and a 30-layer action DiT in a mixture of transformers, conditioned on the UMT5 text encoding
and on RynnBrain1.1-2B (a Qwen3.5 VLM) reading the three views. It is served by XPolicyLab's
`policy/InternW0_delta` adapter, which vendors an evaluation-only runtime (`wam_runtime/`).
For the websocket protocol and the joint mapping shared with pi0.5, see
[xpolicylab-pi05.md](xpolicylab-pi05.md).

## What the checkpoint expects

- **Joint actions, 10 per request.** The model predicts 32 steps of RoboDojo `arx_x5` joint
  actions, 1/25 s apart: `[left j1-6, left gripper, right j1-6, right gripper]`, grippers in
  [0, 1]. The server returns the first 10 (`replan_steps: 10`) and clips the grippers to
  [0, 1]. 10 denoising steps, fixed noise seed 0 (the `seed` the launch script passes).
- **The session lives on the server.** The adapter keeps one session per `env_idx`
  (`_adapter_base.StatefulWAMAdapter`, `wam_runtime/.../robodojo_policy.py`). A session holds:
  - the memory: every step's frame (a 384×256 T canvas: head on top, wrists below) and
    z-scored joint state, trimmed to the first frame plus the last 33;
  - the 10 actions it returned, as pending until each is acknowledged by an observation;
  - a step counter (number of acknowledged actions).

  A request reads the current frame, the first frame of the episode (anchor) and the frame
  32 steps earlier (recent, padded before step 32), with their states. A session refuses a
  new request while any action is pending. `reset` clears **every** session of the server.
- **RoboDojo's request cadence.** `deploy.yml` sets `eval_batch: true`, so RoboDojo runs
  `deploy.py`'s `eval_one_episode_batch`:
  - `reset` at the start of the episode;
  - `update_obs_batch` with the observation of every step; after the first, each one
    acknowledges the action just executed and records the new step;
  - `get_action_batch([0])` whenever the 10 actions are used up, right after that step's
    `update_obs_batch` (steps 0, 10, 20, …).

  The client sends exactly these `call` / `reset` messages. The server holds no pending
  queue for the client: each `get_action_batch` returns the 10 actions, and the client plays
  them out itself while acknowledging each step.
- **One episode per server.** The session is keyed only by `env_idx`, and `reset` wipes all
  sessions, so two episodes on one server would clobber each other. Each concurrently running
  episode gets its own server.
- **State.** Measured arm joints; each gripper is the last commanded opening clipped to
  [0, 1] (RoboDojo's state semantics), the measured opening before the first command.
- **Images.** `cam_head` ← `rgb`, `cam_left_wrist` ← `left_wrist_rgb`, `cam_right_wrist` ←
  `right_wrist_rgb`, as XPL-RGB1 JPEGs. The server decodes them to RGB and builds the canvas
  as training does (resize to 240×320, then pack, no crop). RynnBrain gets the three raw
  views.
- **Rendered frames.** The server records every step's frame but reads only those at steps
  0, 10, 20, … (current) and 8, 18, 28, … (recent, 32 steps before a request). The client
  renders the cameras on those steps only (2 of every 10) and resends the last rendered images
  on the others. The joint state is always current. In the replay check below, this gave
  bit-identical chunks to rendering every step. `INTERNW0_RENDER_ALL=1` renders every step.
- **One act() per RoboDojo step.** Run `roboclock eval` with `--policy-hz 25`. The client
  raises an error at any other rate.

## Setup (policy machine)

```bash
P=$ROBODOJO_DIR/XPolicyLab/policy/InternW0_delta
# Large files under $MODELS_DIR, symlinked where the adapter expects them.
ln -s $MODELS_DIR/InternW0-Delta-RoboDojo $P/checkpoints
mkdir -p $P/assets/Wan-AI $P/assets/Alibaba-DAMO-Academy
ln -s $MODELS_DIR/Wan2.2-TI2V-5B $P/assets/Wan-AI/Wan2.2-TI2V-5B
ln -s $MODELS_DIR/RynnBrain1.1-2B $P/assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B

# install.sh, with uv instead of conda (python 3.11).
unset TRITON_PTXAS_PATH
cd $P && uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python modelscope==1.34.0 ftfy==6.3.1 flash-linear-attention==0.5.1
uv pip install --python .venv/bin/python -e wam_runtime -e ../..

# Weights. download_checkpoint.sh checks the SHA256 of robodojo.pt.
HF_TOKEN=... PATH=$P/.venv/bin:$PATH bash download_checkpoint.sh
PATH=$P/.venv/bin:$PATH bash download_assets.sh   # Wan2.2 from ModelScope; RynnBrain: see below
hf download Alibaba-DAMO-Academy/RynnBrain1.1-2B --revision 36069aa --local-dir $MODELS_DIR/RynnBrain1.1-2B
```

Every weight file matches `config/artifacts.lock.json` (SHA256 checked).

Serve one server per concurrently running episode:

```bash
bash scripts/xpolicylab/serve_internw0.sh $ROBODOJO_DIR 6024 <gpu>
```

- The script is `setup_eval_policy_server.sh` for the released weights
  (`eval.sh RoboDojo <task> robodojo arx_x5 joint 0 …`) without conda: the same
  `setup_policy_server.py`, `deploy.yml` and overrides, `seed=0`.
- Loading takes about 6 minutes; each server uses about 29.5 GB of GPU memory.
- The first request compiles kernels (about 57 s on a fresh inductor cache, 3 s once cached).
  Later requests take about 1.1–1.4 s on an A100 shared with Isaac; each `update_obs_batch`
  takes about 40 ms.

Differences from the published setup:

| Item | Published | Here |
|---|---|---|
| Environment | conda `internw0-delta`, conda-forge gcc | uv venv, system gcc 13.3 (the FLA/Triton helper compiles with it) |
| causal-conv1d | 1.6.2.post1 | not installed: no wheel for torch 2.10, and a source build needs CUDA 12.8 nvcc (the pod has 13.0). RynnBrain's linear-attention short convolution uses transformers' PyTorch fallback (`F.conv1d` + SiLU); the FLA gated-delta kernels are used as published. |
| RynnBrain source | ModelScope | Hugging Face, same repo id: ModelScope answers 404 for it. `model.safetensors` matches the lock's SHA256. |
| `config/eval_model.yaml` | lock SHA256 `0ac9abf…` | `2771fc3…`, the file in both the local adapter (5d0be149) and upstream main (6baccd5); the replay check shows the weights load and predict as trained. |
| Server launch | `setup_eval_policy_server.sh` (conda), `host=localhost` | `serve_internw0.sh`, same `setup_policy_server.py` + overrides, `host=0.0.0.0` |
| Simulator | RoboDojo Isaac scenes, episodes from joint zero | roboclock rig (RoboDojo's arm geometry), episodes from roboclock's task HOME; joint targets clipped to the lab limits |

The first two change bf16 rounding, not the model.

## Run (simulator machine)

```bash
export XPOLICYLAB_URL=ws://127.0.0.1:6024 XPOLICYLAB_LOG=qlog/lab_A5_s0.jsonl
PYTHONPATH=src:$ROBODOJO_DIR python -m roboclock eval \
  --env lab --task A5 --difficulty easy --seeds 1 --seed-start 0 \
  --policy roboclock.envs.lab.policies.internw0:make --policy-hz 25 \
  --inference-latency zero --output out/lab_A5_s0
```

`XPOLICYLAB_LOG` writes one row per request: server time of `get_action_batch` and
`update_obs_batch`, round trip, frames rendered so far, the state gripper and the gripper
commands, clipped targets, and the start jump and travel of each chunk.
`INTERNW0_FRAME_DIR` keeps the head JPEG of every request.

## Integration check

An open-loop replay of two RoboDojo demonstrations, each frame sent through the client
(`act()`, so with RoboDojo's exact cadence: one `update_obs_batch` per frame, a request every
10) against the real server, with the demonstration's state and gripper. Each returned chunk
is compared with the demonstration's next 10 actions; the baseline holds the current joints.
Errors are means over the chunk and over requests: arm joints in rad, link6 position in mm
(X5 forward kinematics in RoboDojo's frame).

| Episode | Requests | Arm | Model | Hold | Gripper error |
|---|---|---|---|---|---|
| stack_blocks ep 0 | 26 | left | 0.0095 rad / 4.8 mm | 0.080 rad / 45.1 mm | 0.005 |
| | | right | 0.0071 rad / 3.5 mm | 0.051 rad / 23.3 mm | 0.001 |
| press_by_number | 37 | left | 0.0111 rad / 6.2 mm | 0.076 rad / 32.7 mm | 0.005 |
| | | right | 0.0059 rad / 2.7 mm | 0.053 rad / 18.4 mm | 0.006 |

The model error is 12–17 % of hold. The client rendered 51 of 257 (stack_blocks) and 73 of 367
frames (press_by_number). With `INTERNW0_RENDER_ALL=1` the 26 stack_blocks chunks were
bit-identical.

An Isaac smoke episode (household A2, seed 0) then ran its 40 s budget with a request every
10 steps; the head frames show the arms moving in the scene.

## Result: the 30-episode comparison set (2026-11-07)

Setup:

- The 30 episodes used for pi0.5 and Wall-X: lab A5/B1, hospital A3/B4, household A2/B2,
  sports A3/B2 and industry A1/B2, seeds 0–2, all at easy.
- `--inference-latency zero` and `--policy-hz 25`.
- Two servers on one A100 (ports 6024 and 6034), one episode per server at a time.

| Task | Joint | Physical | Score (mean) | Stages reached | Failure |
|---|---|---|---|---|---|
| lab A5 | 0/3 | 0/3 | 0 | – | `developer_loading_timeout` at 15 s; the arms barely move |
| lab B1 | 0/3 | 0/3 | 0 | – | `overheat`, `late_reaction`, `transfer_deadline` |
| hospital A3 | 0/3 | 0/3 | 0 | – | `missed_window`: STOP is never pressed |
| hospital B4 | 0/3 | 0/3 | 0 | – | `late_action`: clamp and stopcock never handled |
| household A2 | 0/3 | 0/3 | 20 | `glass_picked_up` 3/3 | Holds the glass until the 40 s budget ends; never tips it |
| household B2 | 0/3 | 0/3 | 13.3 | `object_stopped_before_edge` 1/3 | `wrong_object_disturbed` / `object_fell` within 2–10 s |
| sports A3 | 0/3 | 0/3 | 0 | – | `no_grasp` |
| sports B2 | 0/3 | 0/3 | 0 | – | `moved_during_hold`, `incomplete_combinations` |
| industry A1 | 0/3 | 0/3 | 0 | – | No progress in the 95 s budget |
| industry B2 | 0/3 | 0/3 | 40 | `stop_button_pressed` 3/3 | Presses the stop button at 3.9–4.4 s, before the warning, so not as a reaction |

Total: 0/30 joint and 0/30 physical, the same as pi0.5 and Wall-X. The stages go a little
further than theirs (both reached only household A2 `glass_picked_up`, then tipped the
glass):

- InternW0 also reaches industry B2 `stop_button_pressed` in 3/3 and household B2
  `object_stopped_before_edge` in 1/3 (that episode is a temporal success).
- In household A2 it keeps the glass upright until the time runs out.

Other observations:

- **Server time.** `get_action_batch` 1.11 s at p50 and 1.24 s at p95, over 2875 requests;
  `update_obs_batch` 35 ms at p50 and 50 ms at p95. A100 shared with demo-collection Isaac.
- **Joint limits.** In household A2 and industry A1 the model drives the left arm's joint 2
  (and 3) below the lab limit (−0.05 rad), towards RoboDojo's joint-zero posture; hospital B4
  pushes the right joint 6 past ±2 rad. The targets are clipped.
- **Wall time.** About 4.8 h of episode wall time over the two slots, mostly rendering on a
  loaded pod.

Conclusion:

- The open-loop replay reproduces RoboDojo demonstrations within 3–6 mm (12–17 % of hold), so
  the integration is correct.
- In roboclock the checkpoint fails like pi0.5 and Wall-X. It goes for the salient target:
  the glass, the emergency-stop button, the rolling object. It completes no task in the new
  scenes, objects and instructions, and it does not react to timed cues.
