# OpenWAM-α (RoboDojo sim) in roboclock

`roboclock.envs.lab.policies.openwam` drives the dual X5 arms with
[`OpenWAM/OpenWAM-Alpha-Sim-RoboDojo`](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-RoboDojo)
(revision 2c13022, `checkpoint_step_60000`). OpenWAM-α is a world-action model: a
Wan2.2-TI2V-5B video DiT and an ActionDiT, joined by joint self-attention with a mutual mask,
fine-tuned on the RoboDojo simulation demonstrations. It is served by XPolicyLab's
`policy/OpenWAM` adapter, which vendors the OpenWAM tree. For the websocket protocol and the
joint mapping shared with pi0.5, see [xpolicylab-pi05.md](xpolicylab-pi05.md). The EE
kinematics come from [wallx.md](wallx.md).

## What the checkpoint expects

- **Absolute EE actions, 32 per request.** Each request returns 32 RoboDojo `ee` actions,
  1/25 s apart. Per arm, each action is the link6 pose `[x, y, z, qw, qx, qy, qz]` in
  RoboDojo's environment frame and a gripper opening in [0, 1], where 0 is closed.
  - The training target is the *achieved* state at t+1 … t+32
    (`state/*_ee_poses`, `state/*_ee_joint_states`), not the commanded action.
  - `deploy.yml` sets `replan_steps: null`, and OpenWAM's `inference_horizon` is null, so the
    whole chunk is executed before the next request. The client does the same.
  - `OPENWAM_REPLAN` shortens the chunk.
- **One observation, no history.** The model is conditioned on three things:
  - one composite frame: the head camera on top, the left and right wrist cameras below,
    384×320;
  - the instruction, wrapped as "A video recorded from a robot's point of view executing the
    following instruction: …";
  - the current EE state.

  The adapter stores only the latest observation. Generation is seeded (seed 42), so a
  request depends only on what it carries.
- **The server does the EE encoding.** The client sends RoboDojo's own observation fields:
  - `<side>_ee_pose`: link6 in the environment frame, quaternion wxyz;
  - `<side>_ee_joint_state`: the gripper opening.

  The adapter (`model.py`, mirroring `openwam/dataloader/robodojo.py`) then:
  - moves each pose into its arm's base frame. The `dual_x5` bases sit at (∓0.30, −0.45,
    0.765), yawed +90° (`robodojo_contract.py`).
  - packs EEF20 per arm: xyz, then the first two columns of the rotation matrix (rot6d), then
    the gripper.
  - min/max normalises xyz and the gripper; rot6d is left as is.
  - scatters EEF20 into the 80-D unified action space (slots 0–9 and 34–43).
  - inverts all of this for the answer.

  So the client must report the link6 pose in RoboDojo's environment frame and the
  [0, 1] gripper, exactly as Wall-X does.
- **Client side**, in `openwam.py`:
  - each arm's link6 pose, from the measured joints (`wallx.link6_pose`);
  - the gripper state. RoboDojo's semantics are the last commanded opening, clipped to
    [0, 1]. Before the first command, the measured opening is used, also clipped, because
    the adapter rejects values outside [0, 1].
  - the three RGB views as XPL-RGB1 JPEGs (`cam_head` ← `rgb`, `cam_left_wrist` ←
    `left_wrist_rgb`, `cam_right_wrist` ← `right_wrist_rgb`). The server decodes them to RGB,
    as training's `decode_image_bit` does.
  - Returned poses are solved to joints by `wallx.solve_chunk`, step by step along the chunk
    from the measured joints.
  - Images are read, and so rendered, only on request frames.
- **One act() per RoboDojo step.** Run `roboclock eval` with `--policy-hz 25`. The client
  raises an error at any other rate.
- **Plain `infer` messages.** `infer` runs the adapter's `update_obs` and `get_action`
  under one server lock. `get_action` returns everything the client needs, so `call` is not
  needed (unlike Wall-X).

## Setup (policy machine)

```bash
hf download OpenWAM/OpenWAM-Alpha-Sim-RoboDojo --local-dir $MODELS_DIR/OpenWAM-Alpha-Sim-RoboDojo
P=$ROBODOJO_DIR/XPolicyLab/policy/OpenWAM
mkdir -p $P/checkpoints && ln -sfn $MODELS_DIR/OpenWAM-Alpha-Sim-RoboDojo $P/checkpoints/

unset TRITON_PTXAS_PATH
cd $P && uv venv --python 3.10 .venv && export VIRTUAL_ENV=$P/.venv
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install setuptools wheel
uv pip install -e ../..                     # XPolicyLab
DS_BUILD_OPS=0 uv pip install -e ./OpenWAM --no-build-isolation-package deepspeed
```

The checkpoint is self-contained, 24.8 GB. It holds:

- the Wan2.2 DiT, the Wan VAE and the UMT5-XXL text encoder, in one safetensors file;
- the UMT5 tokenizer;
- `normalization_stats.npy`.

The config's `model_path: /path/to/Wan2.2-TI2V-5B` is never read, because the config embeds
the component specs. No separate Wan or UMT5 download is needed.

```bash
bash scripts/xpolicylab/serve_openwam.sh $ROBODOJO_DIR 6025 <gpu>
```

- The script runs `setup_policy_server.py` with the adapter's `deploy.yml` and the overrides
  of `setup_eval_policy_server.sh`.
- Loading takes about 2 minutes. The server uses about 25 GB of GPU memory.
- A request takes about 1.45 s on an A100 shared with Isaac. The first one takes 4 s.
- The model keeps no per-episode state, and `infer` is atomic. So concurrent episodes can
  share one server under `--inference-latency zero`.
  - Checked: two clients sending interleaved requests, with a `reset` between them, got
    bit-identical chunks to solo requests.
  - The server's `latency_ms` then includes the wait for its model lock.

Differences from the published setup:

| Item | Published | Here |
|---|---|---|
| Environment | conda, Python 3.10 | uv venv, Python 3.10 (same torch 2.7.1 cu128) |
| deepspeed (training only) | pip default | built without ops (`DS_BUILD_OPS=0`) |
| Attention | torch SDPA (flash-attn is not in the install) | the same |
| Server launch | `setup_eval_policy_server.sh` (conda) | `serve_openwam.sh`, same `setup_policy_server.py` + overrides |
| Batch | `eval_batch: true`, RoboDojo batches its 10 envs into one `generate_batch` | one env per request (batch 1) through `infer` |
| EE controller | RoboDojo's | roboclock's X5 IK (`wallx.solve_chunk`) |
| Start pose | RoboDojo joint zero | roboclock's task HOME |

## Run (simulator machine)

```bash
export XPOLICYLAB_URL=ws://127.0.0.1:6025 XPOLICYLAB_LOG=qlog/lab_A5_s0.jsonl
PYTHONPATH=src:$ROBODOJO_DIR python -m roboclock eval \
  --env lab --task A5 --difficulty easy --seeds 1 --seed-start 0 \
  --policy roboclock.envs.lab.policies.openwam:make --policy-hz 25 \
  --inference-latency zero --output out/lab_A5_s0
```

`XPOLICYLAB_LOG` writes one row per request with these fields:

- server and round-trip time;
- the state pose and gripper;
- the first and last target pose;
- the gripper commands;
- the IK error, the clipped targets, and the start jump and travel of each chunk.

`OPENWAM_FRAME_DIR` keeps the head JPEG of every request.

## Integration check

An open-loop replay sent each frame of a RoboDojo demonstration through the client, against
the real server, with the demonstration's gripper state. Each 32-step chunk was compared with
the demonstration's next states:

| Demonstration | Arm | Model (mean, 32 steps) | Hold baseline |
|---|---|---|---|
| stack_blocks | left | 7.4 mm | 113 mm |
| stack_blocks | right | 4.9 mm | 64 mm |
| press_by_number | left | 19.9 mm | 72 mm |
| press_by_number | right | 10.6 mm | 44 mm |

- On stack_blocks, the rotation error is 0.5–0.9°, against 7–12° for hold.
- The client's state pose matches the recorded `ee_pose` within 0.12 mm.
- The replay scripts and logs are in the session scratchpad `models/openwam/replay/`.

## Result: the 30-episode comparison set (2026-11-07)

Setup: the same 30 episodes as for pi0.5 and Wall-X:

- lab A5/B1, hospital A3/B4, household A2/B2, sports A3/B2 and industry A1/B2;
- seeds 0–2, all at easy, including industry;
- `--inference-latency zero` and `--policy-hz 25`.

Two episodes shared one server.

| | Joint | Physical | Stages reached |
|---|---|---|---|
| OpenWAM-α | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 (score 20); nothing else |
| Wall-X (zero latency) | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 |
| pi0.5 (measured latency) | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 |

Failure modes, from the termination reasons:

| Task | Failure |
|---|---|
| household A2 | Picks up the glass; tips it in 2 of 3 (`glass_tipped` at 3.8 s and 35.9 s); s2 runs out of time. |
| household B2 | The object falls within 3–4 s (`object_fell`). |
| lab A5 | `developer_loading_timeout` at 15 s. |
| lab B1 | `overheat`: the beaker is never lifted. |
| hospital A3 / B4 | No progress before the end. |
| sports A3 | `no_grasp`. |
| sports B2 | `moved_during_hold`, `incomplete_combinations`. |
| industry A1 / B2 | No progress before the time budget runs out. |

- **Server time.** 1.64 s per request at p50 and 2.39 s at p95, over 895 requests, including
  the wait for the shared lock. A request alone takes about 1.45 s.
- **IK.** The largest IK miss is 47 mm, in household A2 s2; elsewhere it stays under 19 mm.
  No target was clipped by the joint limits.

Conclusion:

- The replay reproduces RoboDojo demonstrations within 5–20 mm, so the integration is correct.
- In roboclock the checkpoint reaches exactly the same stages as pi0.5 and Wall-X.
