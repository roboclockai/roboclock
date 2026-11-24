# Remote RoboDojo pi0.5 policy for the lab domain

This guide runs RoboDojo's official π0.5 checkpoint as a remote policy for the
dual-ARX-X5 lab tasks. One machine runs the policy server and another runs the
Isaac Sim evaluation. Both can also be the same machine.

```
simulator machine (Isaac Sim)                         policy machine (GPU)
roboclock eval --policy roboclock.envs.lab.policies.xpolicylab:make
  reset ─────────────────────────────────────────────▶ XPolicyLab websocket server
  infer: head/left/right RGB (JPEG) + 14-D state ───▶ Pi_05 (OpenPI, JAX)
  ◀──────────── 50 x 14 joint chunk (25 Hz) ─────────
  resample to 50 Hz, clip to lab limits, env.step
```

π0.5 is a flow-matching vision-language-action model. vLLM does not serve it.
RoboDojo serves its policies through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab):
a websocket server with msgpack frames, where one request returns one action chunk.
The client in `src/roboclock/envs/lab/policies/xpolicylab.py` speaks that protocol, so
any XPolicyLab `joint` policy trained on RoboDojo's `arx_x5` layout can drive the lab
arms the same way, not only π0.5.

## What is served

| Item | Value |
|---|---|
| Checkpoint | `RoboDojo-Benchmark/RoboDojo` (HF dataset) `ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-<seed>/59999`, seeds 0–2 |
| Download | `params/` (12.4 GB) and `assets/arx_x5_sim/norm_stats.json`; the 32 GB `train_state/` is not needed |
| OpenPI config | `pi05_base_aloha_full_sim_arx-x5_seed_0` (XPolicyLab `policy/Pi_05/deploy.yml`) |
| Inputs | `cam_high` (head), `cam_left_wrist`, `cam_right_wrist`, 14-D state, instruction |
| Output | 50 actions × 14 (`[left j1–j6, left gripper, right j1–j6, right gripper]`), absolute joints, 25 Hz |

Check the license terms of the RoboDojo dataset and of the Gemma weights before you
use the checkpoint.

## 1. Policy machine

Requirements:
- An NVIDIA GPU with at least 16 GB free.
- About 30 GB of disk: the checkpoint plus the OpenPI virtualenv.
- `git`.
- [`uv`](https://docs.astral.sh/uv/). OpenPI is uv-managed. Install uv with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Isaac Sim is not needed on the policy machine. From a roboclock checkout:

```bash
# One time: RoboDojo + XPolicyLab, OpenPI venv, official checkpoint (seed 0)
bash scripts/xpolicylab/setup_pi05_server.sh ~/RoboDojo 0

# Serve on port 6000, GPU 0, seed 0 (binds 0.0.0.0)
bash scripts/xpolicylab/serve_pi05.sh ~/RoboDojo 6000 0 0
```

The server loads the model before it opens the port. The first inference
compiles the JAX graph and takes noticeably longer than later ones.

`XLA_PYTHON_CLIENT_MEM_FRACTION` defaults to 0.5. Lower it when the GPU is shared.

To check connectivity before the real model is ready, run the stub server.
It needs no GPU and no checkpoint:

```bash
~/RoboDojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/xpolicylab/stub_policy_server.py --xpolicylab ~/RoboDojo/XPolicyLab --port 6000
```

The stub decodes and validates every observation the way Pi_05 would. It answers
with a chunk that holds the observed state. Use `--probe 0.2` to add a joint-1 ramp
of +0.2 rad per chunk on the left arm and −0.2 rad on the right arm.

## 2. Network between the machines

Pick the first option that works.

1. **Pod or LAN IP (best, ~ms).** On the policy machine, run `hostname -i`. Then
   use `XPOLICYLAB_URL=ws://<ip>:6000`. Some clusters block traffic between pods.
   The probe in step 3 tells you whether yours does.
2. **code-server proxy (works on remote node dev pods).** code-server forwards
   `https://<host>/proxy/<port>/` to `localhost:<port>`, including websockets. It
   requires the code-server login cookie:

   ```bash
   HOST=<policy machine code-server host, e.g. xxxx-name.example.com>
   curl -s -c /tmp/cs.txt --data-urlencode "password=<code-server password>" \
     "https://$HOST/login" -o /dev/null
   export XPOLICYLAB_COOKIE=$(awk '$6=="code-server-session"{print $7}' /tmp/cs.txt)
   export XPOLICYLAB_URL=wss://$HOST/proxy/6000/
   rm /tmp/cs.txt
   ```

   The cookie grants the same access as the password. Keep it out of git and out of
   shared logs.

## 3. Simulator machine

The Isaac venv needs the msgpack codec. It already ships `websockets`:

```bash
python -m pip install "msgpack>=1.0.8,<2" "msgpack-numpy>=0.4.8"
```

Probe the server without starting Isaac Sim:

```bash
cd roboclock
XPOLICYLAB_URL=ws://<policy ip>:6000 \
  python scripts/xpolicylab/probe_policy.py --calls 5
```

The probe should print one `call` line per request: a round-trip time, a chunk of
`(50, 14)`, and `finite=True`.

Then evaluate:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
XPOLICYLAB_URL=ws://<policy ip>:6000 \
XPOLICYLAB_LOG=runs/pi05_A1/inference.jsonl \
CUDA_VISIBLE_DEVICES=1 \
python -m roboclock eval --env lab --task A1 --difficulty easy \
  --seeds 10 --policy roboclock.envs.lab.policies.xpolicylab:make --output runs/pi05_A1
```

| Variable | Meaning |
|---|---|
| `XPOLICYLAB_URL` | `ws://host:port` or `wss://host/proxy/port/` (required) |
| `XPOLICYLAB_COOKIE` | code-server session cookie for proxied URLs |
| `XPOLICYLAB_REPLAN` | Policy actions executed per chunk before re-querying. The default is the whole chunk, as in RoboDojo's `deploy.py`. |
| `XPOLICYLAB_CONTROL_HZ` | Lab control rate, default the shared 50 Hz (`physics_dt` 1/250 s × `control_decimation` 5). |
| `XPOLICYLAB_LOG` | JSONL file with one row per inference. Each row has `server_ms`, `round_trip_ms`, chunk length, and the number of arm targets clipped to lab limits. |

One websocket connection per server URL is shared across the episodes of a run.
A policy server that is down or that restarts fails the run loudly. It is never
silently replaced.

## Observation and action mapping

| roboclock lab | RoboDojo / XPolicyLab |
|---|---|
| `obs["rgb"]` (640×480 RoboDojo head camera, `docs/robodojo-rig-spec.md`) | `vision/cam_head/color`, sent as an RGB JPEG (q95) tagged `XPL-RGB1` |
| `obs["left_wrist_rgb"]`, `obs["right_wrist_rgb"]` (640×480 D435 wrist cameras on `link6`; `left` = the arm on the rig's +y side) | `cam_left_wrist`, `cam_right_wrist` |
| `joint_positions[0:6]` (the rig's left arm, +y), `[8:14]` (the rig's right arm) | `left_arm_joint_state`, `right_arm_joint_state` |
| last commanded gripper, or the measured finger joint before the first command | `*_ee_joint_state` = (q + 0.01) / 0.054, the X5 `gripper_scale` [−0.01, 0.044] |
| `obs["instruction"]` | `instruction` (the π0.5 prompt) |
| chunk action `[j1–j6, g]` per arm, left then right | lab targets `[j1–j6, q, q]` with q = −0.01 + 0.054·g, clipped to `JOINT_LOW/HIGH`; RoboDojo's left arm → action values 0–7, right → 8–15 |

The lab URDF and RoboDojo's `X5A.urdf` share the joint axes and the 0–0.044 m finger
travel. Arm joints therefore pass through unchanged. RoboDojo reports the gripper
state as the last command, not the measured opening, and the client does the same.

## Timing

- **Inference time passes in simulation.** `roboclock eval` charges each
  inference to simulation time while the arms hold their previous targets (see
  `docs/architecture.md`, Inference latency). The client reports the server's
  own `latency_ms` as `last_inference_latency`, so network time through a proxy
  is not charged. If a server sends no `latency_ms`, the full round trip is
  charged. Use `--inference-latency zero` to pause the simulator instead, and
  `--control-delay <seconds>` to add a fixed deployment delay.
- **Only queries are charged.** Each inference returns 2 s of motion: 50
  actions at 25 Hz, resampled onto 100 lab control steps at 50 Hz. The arms
  therefore move at their trained speed. The client sets `queried = False` while
  it plays out a chunk, and those steps cost no latency.
- **Streaming mode (optional; block-and-replay is the default).** With
  `--time-model streaming` the evaluator takes whole
  chunks (`act_chunk`) and keeps executing the current chunk while the next query
  runs. The client queries again as soon as it is idle, so a new chunk replaces
  the old one every 0.10-0.12 s (`docs/architecture.md`, Streaming mode).
- **Measure network and model time separately.** `XPOLICYLAB_LOG` records both.
  Measured with the stub server:

  | Path | Round trip per inference |
  |---|---|
  | Same machine | ~20 ms |
  | Through a remote node code-server proxy (Cloudflare) | ~145 ms |

  Add the model's own inference time on top of these.

## Verification status (2026-10-30)

| Check | Result |
|---|---|
| `tests/envs/lab/test_xpolicylab_policy.py` | 6 passed: JPEG marker and RGB order, gripper scale, arm order, 25→30 Hz resampling, chunk queue, replanning |
| Client against XPolicyLab's own websocket server (`128e626`) running the stub model | hello, reset and infer work and return 50 × 14 chunks. A red frame (230, 20, 10) arrives as (231, 20, 11), so channels are not swapped. A hold chunk reproduces the observed joints exactly. |
| Same, through a remote node code-server proxy (`wss://…/proxy/6000/` with cookie) | Works. About 145 ms round trip per inference. |
| Isaac Sim 6.0.1, `--env lab --task A1 --difficulty easy`, stub server with `--probe 0.2` | Full 120 s episode: 60 inferences × 60 control steps. Left joint 1 went 0 → 0.198 → 0.395 and right joint 1 went 0 → −0.198 → −0.395, which confirms arm order and sign. Targets stopped at the lab limits. `episodes.jsonl` and `summary.json` were written and the process exited cleanly. |
| Real π0.5 inference | Run since 2026-11-05: five-domain evaluations and the replay check against RoboDojo demonstrations in the next section. |

If roboclock reports missing Isaac graphics libraries (vulkan, GLU, Xt), point
`LD_LIBRARY_PATH` at an extracted copy of them before launching. On the remote node
simulator pod that copy is `/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu`.

## Integration check against RoboDojo demonstrations (2026-11-05)

`scripts/xpolicylab/replay_check.py` checks the whole client path against the policy's
own training data, without a simulator. It reads RoboDojo episodes from the Hugging Face
dataset `RoboDojo-Benchmark/RoboDojo` (`data/RoboDojo/<task>/arx_x5/data/episode_*.hdf5`,
about 0.2-0.4 GB each) and sends every 25th frame (one per second) to the server three
times:

- **native:** the request RoboDojo's eval client sends, i.e. the stored JPEG bytes and
  stored state.
- **native again:** the same request a second time. The difference between the two is
  π0.5's own sampling noise.
- **roboclock:** the frame as a roboclock observation (decoded RGB images, 16 joint positions
  with the grippers as finger sliders), packed by `XPolicyLabPolicy.observation()`.

Each chunk is scored on its first 25 entries (1 s) against the demonstrated actions, next
to holding the current state.

```bash
PYTHONPATH=src:$ROBODOJO_DIR XPOLICYLAB_URL=ws://127.0.0.1:6011 \
  $ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/xpolicylab/replay_check.py stack_blocks/episode_0000000.hdf5 ... --json replay.json
```

Results with `RoboDojo-sim-arx_x5-joint-0`. Each cell is the mean absolute error of the arm
joints (rad) / gripper opening (0-1):

| Episode | Frames | native | native noise | roboclock | roboclock vs native | hold |
|---|---:|---:|---:|---:|---:|---:|
| stack_blocks 0 | 10 | 0.012 / 0.005 | 0.005 / 0.002 | 0.012 / 0.006 | 0.004 / 0.002 | 0.123 / 0.112 |
| stack_blocks 1 | 10 | 0.013 / 0.004 | 0.004 / 0.002 | 0.012 / 0.005 | 0.004 / 0.002 | 0.140 / 0.094 |
| press_by_number 0 | 15 | 0.013 / 0.121 | 0.010 / 0.058 | 0.013 / 0.094 | 0.005 / 0.031 | 0.136 / 0.161 |
| All | 35 | 0.013 / 0.055 | 0.007 / 0.026 | 0.012 / 0.043 | 0.004 / 0.015 | 0.134 / 0.128 |

- **The checkpoint and server work.** On the training distribution π0.5 follows the
  demonstrations about 10x more closely than holding still.
- **The roboclock client sends what RoboDojo sends.** Its predictions differ from the native
  ones by less than the native sampling noise (0.004 vs 0.007 rad). That covers image
  colors (the dataset JPEGs carry `XPL-RGB1` and decode to true RGB, as the client sends),
  arm order, joint values and the gripper scale.
- **So the 0/30 joint success of the 2026-11-05 five-domain evaluation is not an
  integration fault.** Only household A2 got past its first manipulation step (it picked
  up the glass). In the other nine tasks, the task object was never grasped, pressed or
  moved. The cause is the domain gap: the objects, scenes, instructions and start pose are
  all new to the policy.

The unit test `test_replay_check_packs_a_demo_frame_like_robodojo` pins the state packing.

## Known limitations

- **Arm order matches RoboDojo (since the 2026-11-04 alignment).** The lab and hospital
  interface lists the rig's left arm (+y, the arm `left_wrist_rgb` rides on) first in
  actions, `joint_positions`, `ee_pose`, `gripper_width` and `joint_order`, so the client
  maps arms straight through (`ARM_ORDER = (0, 1)`). Internally the lab keeps its prim names:
  `/World/ARXLeft` is the rig's right arm (and the single-arm worker, action values 8–15);
  `LabEnv` converts. Earlier on 2026-11-04 the client swapped the arms itself because the
  lab interface listed `/World/ARXLeft` first; result JSONs and recordings made before the
  alignment use that internal order.
- **The domain gap is large.** The scenes, objects, tasks and instructions all differ
  from RoboDojo. Treat the scores as a cross-benchmark baseline, not as a tuned result.
  The rig itself matches since 2026-11-04 ([robodojo-rig-spec.md](robodojo-rig-spec.md)):
  arm spacing, head and wrist camera poses and intrinsics, and the physics and policy
  rates. It differs in per-task rig placement and X5 drive gains.
- **The start pose differs from RoboDojo's unless you pass `--start-pose zero`.** RoboDojo
  episodes start with every arm joint at 0. By default roboclock tasks start from their own
  postures, e.g. lab `HOME` (`[0, 1.6, 1.3, -1.27, 0, 0]`). The effect depends on
  the task. Given the first stack_blocks frame with the `HOME` state, π0.5 pulls the arm
  back toward 0: the mean distance to 0 goes from 0.70 to 0.26 rad over one chunk. Given
  the first press_by_number frame, it moves toward the task from either start.
  `roboclock eval --start-pose zero` starts both arms at joint 0 in every task
  ([architecture.md, Start pose](architecture.md#start-pose)). The 2026-11-05 runs used
  the task postures.
- **Lab joint limits are tighter than the X5 URDF's.** Out-of-range targets are
  clipped, and `XPOLICYLAB_LOG` counts them. The gripper's −0.01 m squeeze command
  becomes the physical 0 m stop.

## Files

| Path | Role |
|---|---|
| `src/roboclock/envs/lab/policies/xpolicylab.py` | Websocket client and observation/action mapping (`make` factory) |
| `scripts/xpolicylab/setup_pi05_server.sh` | Policy machine: clone, install, download the checkpoint |
| `scripts/xpolicylab/serve_pi05.sh` | Policy machine: start the Pi_05 server |
| `scripts/xpolicylab/stub_policy_server.py` | Policy machine: model-free server for wiring checks |
| `scripts/xpolicylab/probe_policy.py` | Simulator machine: connectivity and contract probe without Isaac |
| `scripts/xpolicylab/replay_check.py` | Integration check: RoboDojo demonstrations through the client vs native requests |
| `tests/envs/lab/test_xpolicylab_policy.py` | Unit tests for the mapping, resampling and chunk queue |
