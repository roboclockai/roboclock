# Awomo-0.5 (RoboDojo) in roboclock

`roboclock.envs.lab.policies.awomo` drives the dual X5 arms with
[`Auwomo/Awomo-0.5-Robodojo`](https://huggingface.co/Auwomo/Awomo-0.5-Robodojo) at revision
`5a3e043`, the checkpoint that XPolicyLab's `policy/Awomo05` adapter pins (local XPolicyLab
5d0be149, adapter e8a3c23). It is a FLUX.2-klein-4B world-action model with a LoRA-tuned
Qwen3-VL-4B prompt encoder, fine-tuned on the RoboDojo demonstrations. For the websocket
protocol and the joint mapping shared with pi0.5, see [xpolicylab-pi05.md](xpolicylab-pi05.md).

Upstream XPolicyLab (from dc259fd, 2026-10-20) moved the adapter to a different checkpoint,
`Auwomo/Awomo-0.5I-Robodojo`, with its own code layout. That is the repository the model list
links. It was not evaluated here.

## What the checkpoint expects

- **Absolute joint actions.** Each query predicts 16 RoboDojo joint targets
  `[left j1-6, left gripper, right j1-6, right gripper]`, with each gripper in [0, 1]. The
  adapter returns the first 8 (`replan_steps: 8`), which is 0.32 s at 25 Hz. The state is the
  same 14 values, z-scored with the checkpoint's `dataset_stats.json`. roboclock's lab HOME
  pose lies within 1.6 standard deviations of the training mean on every joint.
- **A 25 Hz history held by the server.** The adapter keeps one buffer per `env_idx`. Each
  `update_obs` pushes one head frame, and the adapter assumes one push per RoboDojo step.
  - The action model reads 20 history slots, 1 s to 20 s back (bins of ±0.4 s, taken at
    their centres).
  - The Qwen3-VL prompt reads 20 head frames, 1 s apart.
  - `deploy.py` keeps no history of its own. It calls `update_obs` after every executed
    action and `get_action` when the 8 actions are used up.

  The client does the same. Every act() sends the full observation, and every eighth act()
  also queries. Every frame is needed by some later query: the slots are 25 frames apart and
  the queries 8 frames apart, so every residue is hit. All three cameras therefore render on
  every frame.
- **One act() per RoboDojo step.** Run `roboclock eval` with `--policy-hz 25` and
  `--inference-latency zero`.
  - Each action spans two 50 Hz control steps.
  - The client raises an error if two observations are not 1/25 s apart. A measured latency
    would leave gaps that the server's frame count cannot see.
- **The batch `call` path.** `deploy.yml` sets `eval_batch: true`, so RoboDojo runs
  `eval_one_episode_batch`. The client sends the same calls with one environment: `reset`,
  `update_obs_batch` and `get_action_batch(env_idx_list)`. The plain `infer` message would
  pair one `update_obs` with every `get_action` and drop the frames in between.
- **Colours: the server converts.**
  - The client sends RGB JPEGs (q95, tagged XPL-RGB1).
  - The server's `decode_obs_images` returns RGB.
  - The adapter flips to BGR itself (`input_color_order: bgr`, which the checkpoint README
    requires; the adapter README forbids client-side swaps).

  A unit test checks that the server's decoder returns the client's RGB unchanged.
- **The gripper state** is the last commanded opening, clipped to [0, 1]. Before the first
  command, it is the measured opening.

### Sharing one server

`reset` clears the buffers of every `env_idx`. So with `env_idx` 0 and a `reset` at each
episode start, which is the default and the same as `deploy.py`, a server can run only one
episode at a time.

GPU 6 has room for only one 22 GB server (see Setup). So the run used
`AWOMO_SHARED_SERVER=1`:

- Each episode sends its own random `env_idx`, which gets an empty buffer on the server.
- The client skips `reset`. Besides the buffers, `reset` only restarts the cadence of a
  diagnostic subtask decode, and the action model never reads that decode.
- The server serialises calls under its model lock, and `get_action_batch` names the
  `env_idx`. So episodes cannot read each other's frames.

Check: two demonstration replays ran at the same time on one server in this mode, alongside
an Isaac episode. Every chunk was identical to the single-client replays, which used `reset`
and `env_idx` 0. Finished episodes leave about 0.7 GB of host memory each on the server until
it restarts. In the run, the 6023 server's resident memory grew from 5.6 GB to 12 GB
over four episodes.

## Setup (policy machine)

```bash
# Checkpoint (evaluation-only licence) and base weights.
hf download Auwomo/Awomo-0.5-Robodojo --revision 5a3e043cba7932e55e759ccd3a7950340983bcae \
  model.pt dataset_stats.json SHA256SUMS LICENSE.md README.md --local-dir $MODELS_DIR/Awomo-0.5-Robodojo
(cd $MODELS_DIR/Awomo-0.5-Robodojo && sha256sum -c SHA256SUMS)
hf download black-forest-labs/FLUX.2-klein-base-4B flux-2-klein-base-4b.safetensors \
  --local-dir $MODELS_DIR/FLUX.2-klein-base-4B
# FLUX.2-dev is gated; this Apache-2.0 file is byte-identical to its ae.safetensors (below).
hf download Comfy-Org/Flux-2-Klein-4B split_files/vae/flux2-vae.safetensors \
  --local-dir $MODELS_DIR/Comfy-Org--Flux-2-Klein-4B
# Qwen3-VL-4B-Instruct: $MODELS_DIR/Qwen3-VL-4B-Instruct.

# The layout the adapter's README expects, as symlinks.
W=$ROBODOJO_DIR/XPolicyLab/policy/Awomo05/weights
mkdir -p $W/flux2-klein-base-4b $W/flux2-dev
for f in model.pt dataset_stats.json SHA256SUMS LICENSE.md; do ln -s $MODELS_DIR/Awomo-0.5-Robodojo/$f $W/$f; done
ln -s $MODELS_DIR/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors $W/flux2-klein-base-4b/
ln -s $MODELS_DIR/Comfy-Org--Flux-2-Klein-4B/split_files/vae/flux2-vae.safetensors $W/flux2-dev/ae.safetensors
ln -s $MODELS_DIR/Qwen3-VL-4B-Instruct $W/Qwen3-VL-4B-Instruct

# The venv: uv in place of install.sh's conda pip.
A=$ROBODOJO_DIR/XPolicyLab/policy/Awomo05
unset TRITON_PTXAS_PATH
uv venv --python 3.11 $A/.venv && source $A/.venv/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
grep -v -E "^torch==|^#|^$" $A/requirements.txt > /tmp/awomo_req.txt
uv pip install -r /tmp/awomo_req.txt torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
uv pip install -e $ROBODOJO_DIR/XPolicyLab
uv pip install numpy==1.26.4 opencv-python-headless==4.11.0.86
```

**The FLUX.2 autoencoder.** The adapter loads `ae.safetensors` from the gated
`black-forest-labs/FLUX.2-dev`, which returns 403 for this account. The Hub lists that file's
LFS SHA-256 as `868fe7b3…66ce8f3`, with 336,211,292 bytes.
`Comfy-Org/Flux-2-Klein-4B` (rev 5b3abca, Apache-2.0) ships
`split_files/vae/flux2-vae.safetensors` with the same hash. The downloaded file hashes to
that value, so it is the same file byte for byte. The adapter's `AutoEncoder` loads it with
`strict=True`.

Differences from the published setup:

| Item | Published | Here |
|---|---|---|
| Environment | conda env, `pip install -r requirements.txt` | uv venv (Python 3.11) with the same pins |
| torchvision | not listed | 0.23.0: transformers 5.5's Qwen3-VL processor needs it |
| numpy / OpenCV | `numpy<2` | 1.26.4 / 4.11; XPolicyLab's install pulls numpy 2 and OpenCV 5 |
| FLUX.2 autoencoder | `FLUX.2-dev/ae.safetensors` (gated) | byte-identical Comfy-Org copy |
| Checkpoint staging | copied to `/tmp/Awomo05` | read in place (`ckpt_local_copy=false`): same local disk |
| Camera images | RoboDojo sends raw RGB arrays | RGB JPEG q95 (XPL-RGB1), as the other roboclock clients |
| Episode keys | `env_idx` 0 and `reset` per episode | per-episode `env_idx`, no `reset` (shared server, above) |

## Serving

```bash
bash scripts/xpolicylab/serve_awomo.sh $ROBODOJO_DIR 6023 <gpu>
```

The server is XPolicyLab's `setup_policy_server.py` with the adapter's `deploy.yml` and the
README's options: `ckpt=model.pt input_color_order=bgr replan_steps=8 action_type=joint`.

- Loading takes about 1 minute.
- The server uses 22 GB of GPU memory after the first query.
- A query takes 2.7 s at p50 when the server runs alone (4.2 s at most), on an A100 shared
  with other servers. One query in four also decodes a diagnostic subtask, which adds
  about 1.3 s.
- Each `update_obs` takes 30–50 ms.

## Run (simulator machine)

```bash
export XPOLICYLAB_URL=ws://127.0.0.1:6023 XPOLICYLAB_LOG=qlog/lab_A5_s0.jsonl AWOMO_SHARED_SERVER=1
PYTHONPATH=src:$ROBODOJO_DIR python -m roboclock eval \
  --env lab --task A5 --difficulty easy --seeds 1 --seed-start 0 \
  --policy roboclock.envs.lab.policies.awomo:make --policy-hz 25 \
  --inference-latency zero --output out/lab_A5_s0
```

`XPOLICYLAB_LOG` writes one row per query with these fields:

- server time, the `update_obs` time, and the round trip;
- the gripper state and commands;
- clipped arm targets;
- the start jump and travel of each chunk.

`AWOMO_FRAME_DIR` keeps the head JPEG of every query.

Rendering every frame dominates the wall time.

- A lab frame took 0.5 s to render.
- A household frame took about 4 s on the crowded GPU 0, during the smoke run.

## Integration check

An open-loop replay sent every frame of two RoboDojo demonstrations through the client to the
real server, with the demonstration's gripper state.

- Each 8-action chunk is compared with the demonstration's next 8 actions.
- The hold baseline keeps the current joints.
- Errors are mean absolute joint error, and link6 position error from X5 forward kinematics.

| Demo (queries) | Arm | Model | Hold | Model / hold | Gripper error (hold) |
|---|---|---|---|---|---|
| stack_blocks (33) | left | 6.0 mm, 0.012 rad | 39.0 mm, 0.071 rad | 0.15 | 0.005 (0.052) |
| stack_blocks (33) | right | 4.1 mm, 0.009 rad | 19.5 mm, 0.042 rad | 0.21 | 0.002 (0.027) |
| press_by_number (47) | left | 5.4 mm, 0.010 rad | 27.9 mm, 0.063 rad | 0.19 | 0.006 (0.051) |
| press_by_number (47) | right | 2.4 mm, 0.006 rad | 15.9 mm, 0.046 rad | 0.15 | 0.009 (0.053) |

**Colour order.** A diagnostic server with `input_color_order=rgb` did about as well:

- stack_blocks: 5.2 / 3.4 mm, slightly better than BGR;
- press_by_number: 5.7 / 2.5 mm, slightly worse.

So these near-term chunks barely depend on colour, and the replay cannot confirm the README's
BGR requirement. The run uses BGR, as published.

**Smoke episode.** household B2, seed 0, ran 12 queries in 474 s. The episode ended at 3.7 s
with `object_fell`, the same failure as Wall-X.

## Result: the 30-episode comparison set (2026-11-07)

Setup:

- The 30 episodes used for pi0.5, Wall-X and the other RoboDojo models: lab A5/B1,
  hospital A3/B4, household A2/B2, sports A3/B2 and industry A1/B2, seeds 0–2, easy.
- `--inference-latency zero` and `--policy-hz 25`.
- Episodes shared servers in `AWOMO_SHARED_SERVER=1` mode. The run started with one server
  (6023, GPU 6) and two episodes at a time. Once the other models had finished, three more
  servers took over the remaining queue (6033, 6036, 6037 on GPUs 3, 5 and 2).

All 30 episodes finished normally (rc 0, no timeouts or restarts). Result: **0/30 joint and
0/30 physical**.

| Task | Joint | Physical | Stages reached | End |
|---|---|---|---|---|
| household A2 | 0/3 | 0/3 | `glass_picked_up` 1/3 (s0 at 34.3 s; score 6.7) | `time_budget`; the glass is never tipped |
| household B2 | 0/3 | 0/3 | none | `object_fell` within 3–4 s |
| lab A5 | 0/3 | 0/3 | none | `developer_loading_timeout` at 15 s |
| lab B1 | 0/3 | 0/3 | none | `time_budget`, `overheat`: the beaker is never lifted |
| hospital A3 | 0/3 | 0/3 | none | no progress before the end |
| hospital B4 | 0/3 | 0/3 | none | `time_budget` |
| sports A3 | 0/3 | 0/3 | none | `no_grasp` |
| sports B2 | 0/3 | 0/3 | none | `moved_during_hold`, `incomplete_combinations` |
| industry A1 | 0/3 | 0/3 | none | `time_budget` |
| industry B2 | 0/3 | 0/3 | none | `time_budget`: the stop button is never pressed |

The other models reached household A2 `glass_picked_up` 3/3 (pi0.5, Wall-X, OpenWAM, VPP2,
InternW0). Awomo reaches it only once, late, but it is the only one of these that also never
tips the glass.

- **Server time.** `get_action` took 3.3 s at p50 and 5.2 s at p95, over 3636 queries. This
  includes waits for the model lock, with up to two episodes per server. `update_obs` took
  33 ms at p50.
- **Wall time.** 4–82 min per episode, about 1 h per 95–120 s episode: 3000 renders at 0.5 s
  each, plus 375 queries.
- **Joint limits.** Arm targets were clipped at the lab joint limits often in hospital B4
  (29–197 per episode), household B2 (125–226) and household A2 s2 (144). Elsewhere clipping
  was rare or absent.
- Results and per-query logs are in the session scratchpad, `models/awomo/zero/`. The overview
  is in `final_summary.txt`.

Head frames of lab B1 show the right arm going to the hotplate and then returning to a rest
pose without grasping the beaker.
