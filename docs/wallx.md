# Wall-X (WALL-OSS 1.0 lite, RoboDojo) in roboclock

`roboclock.envs.lab.policies.wallx` drives the dual X5 arms with
[`x-square-robot/wall-oss-1.0-lite-robodojo`](https://huggingface.co/x-square-robot/wall-oss-1.0-lite-robodojo),
a Qwen3.5-based VLA fine-tuned on the RoboDojo demonstrations. It is served by XPolicyLab's
`policy/WallX` adapter. For the websocket protocol and the joint mapping shared with pi0.5,
see [xpolicylab-pi05.md](xpolicylab-pi05.md).

## What the checkpoint expects

- **End-effector actions.** Each request returns 32 steps of RoboDojo `ee` actions: per arm,
  the link6 pose `[x, y, z, qw, qx, qy, qz]` and a gripper opening in [0, 1]. The client
  replays the first 16, as XPolicyLab's `policy/WallX/deploy.py` does.
- **Absolute poses in RoboDojo's frame.** The state is the absolute link6 pose, min/max
  normalised and binned into the prompt. RoboDojo reports it in its environment frame, where the
  `dual_x5` bases sit at (∓0.30, −0.45, 0.765), both yawed +90°. The client reports each
  arm's link6 pose in that frame, computed from the measured joints with the X5 kinematics.
  - This is possible because the roboclock rig keeps RoboDojo's geometry relative to the arms
    (`roboclock.envs.robodojo_rig`).
  - Returned poses go back through the inverse transform and are solved to joints step by
    step along the chunk.
  - Check against RoboDojo demonstrations: the client's pose matches the recorded `ee_pose`
    within 0.12 mm and 0.017°. roboclock's HOME pose falls inside the checkpoint's state range.
- **One act() per RoboDojo step.** Run `roboclock eval` with `--policy-hz 25`: each action spans
  two 50 Hz control steps. The client raises an error at any other rate.
- **Client-held keyframe memory.**
  - Every request carries `kfmem_client_state_v1`: the retained frame ids, the selection
    counters, the CUDA RNG state, and the raw head frames the server will read. The reply
    returns the updated state.
  - The client drives this state with XPolicyLab's own `policy/WallX/memory_client.py`.
    Put the RoboDojo checkout on `PYTHONPATH`.
  - The server selects every 48 frames from 16 candidates, at T−47, T−44, …, T−2. That
    schedule is deterministic, so the client renders only the frames the server will read:
    about 18 of every 48. A skipped frame the server asks for is an error, not a substitute.
  - `WALLX_RECORD_ALL_FRAMES=1` renders every frame instead.
- **Requests use the `call` message** (`func_name` `reset` / `infer`). The plain `infer`
  message goes through the adapter's `get_action`, which drops the memory state.

## Setup (policy machine)

```bash
# XPolicyLab main (2026-11-04 or later) has policy/WallX; copy it into the RoboDojo checkout
# if its XPolicyLab submodule is older.
git clone -b starrick/robodojo-serving https://github.com/StarrickLiu/wall-x.git $WALLX_DIR
git -C $WALLX_DIR checkout 39fa65b9c9e95b0316db0fc9dbe41211e3a0d4cc
hf download x-square-robot/wall-oss-1.0-lite-robodojo --local-dir $MODELS_DIR/wall-oss-1.0-lite-robodojo

V=$ROBODOJO_DIR/XPolicyLab/policy/WallX/.venv
unset TRITON_PTXAS_PATH
uv venv --python 3.11 $V && source $V/bin/activate
# cu130 matches the pod's nvcc 13.0, which compiles wall-x's CUDA operators.
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
grep -v -E "^torch==|^torchvision==|^torchaudio==|^#|^$" $WALLX_DIR/requirements.txt > /tmp/wallx_req.txt
uv pip install -r /tmp/wallx_req.txt setuptools wheel
uv pip install -e $ROBODOJO_DIR/XPolicyLab
TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=16 uv pip install --no-build-isolation -e $WALLX_DIR
```

Differences from the published setup:

| Item | Published | Here |
|---|---|---|
| torch | 2.10.0 with CUDA 12.8 | 2.10.0 with CUDA 13.0 |
| Vision tower | flash-attn 2.8.1 | PyTorch SDPA, through `scripts/xpolicylab/wallx_server.py` |
| Qwen3.5 linear-attention conv | causal-conv1d | Torch fallback |

These differences change bf16 rounding, not the model.

Serve one server per concurrently running episode. The server's `latency_ms` includes the
wait for its model lock.

```bash
bash scripts/xpolicylab/serve_wallx.sh $ROBODOJO_DIR 6010 <gpu>
```

- Loading takes about 4 minutes.
- Each server uses about 14 GB of GPU memory.
- The first request compiles kernels and takes about 55 s. Later requests take 1.2–1.3 s
  on an A100 shared with Isaac, or 1.6–2.4 s when a memory selection is due.

## Run (simulator machine)

```bash
export XPOLICYLAB_URL=ws://127.0.0.1:6010 XPOLICYLAB_LOG=qlog/lab_A5_s0.jsonl
PYTHONPATH=src:$ROBODOJO_DIR python -m roboclock eval \
  --env lab --task A5 --difficulty easy --seeds 1 --seed-start 0 \
  --policy roboclock.envs.lab.policies.wallx:make --policy-hz 25 \
  --inference-latency zero --output out/lab_A5_s0
```

`XPOLICYLAB_LOG` writes one row per request with these fields:

- server and round-trip time;
- the ring and memory frames;
- the state pose and the first target pose;
- gripper commands;
- IK error, clipped targets, and the start jump and travel of each chunk.

## Integration check

An open-loop replay of RoboDojo demonstrations sent each frame through the client against
the real server, with the demonstration's gripper state. stack_blocks episode 0, 16 requests:

| Arm | Model error (mean, next 16 actions) | Hold baseline | Gripper error |
|---|---|---|---|
| left | 8.8 mm | 65.1 mm | 0.012 |
| right | 4.9 mm | 34.2 mm | 0.006 |

The client rendered 95 of 251 frames, and no requested frame was missing. In
press_by_number, a memory task, the memory filled with frames at the button presses
(1, 49, 61, 76, …).

## Result: the 30-episode comparison set (2026-11-06)

Setup:

- The 30 episodes used for pi0.5 and Qwen: lab A5/B1, hospital A3/B4, household A2/B2,
  sports A3/B2 and industry A1/B2, seeds 0–2, easy. Industry also runs at easy, because main
  dropped `reviewed_nominal`.
- `--inference-latency zero`, which is RoboDojo's own protocol: the simulator waits for the
  policy.
- `--policy-hz 25`, with one server per episode.

| | Joint | Physical | Stages reached |
|---|---|---|---|
| Wall-X (this run) | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 (score 20); nothing else |
| pi0.5 (2026-11-05, measured latency) | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3; nothing else |

Failure modes, from the termination reasons and the head frames of each request:

| Task | Failure |
|---|---|
| household A2 | Picks up the glass, then tips it (`glass_tipped`) within 1.5–3.4 s. |
| household B2 | The object falls (`object_fell`). |
| lab B1 | The beaker is never lifted after the alarm (`overheat`). |
| lab A5 | The right arm hovers by the red button and never takes the film reel (`developer_loading_timeout`). |
| sports A3 | The arms barely move and the ball is never grasped (`no_grasp`). |
| sports B2 | No punches after GO cues (`incomplete_combinations`). |
| hospital A3 / B4 | STOP is never pressed; the clamp and stopcock are never handled. |
| industry A1 / B2 | No progress before the time budget runs out. |

Other observations:

- **Joint limits.** In lab B1 the IK missed some chunk targets by up to 74 mm. The right
  arm's joint 4 sat at the lab limit (−1.6 rad), while the X5 URDF RoboDojo uses allows
  ±10 rad.
  - In the two RoboDojo demonstrations checked, every joint stays inside the lab limits.
  - So the model drives the arm into postures the lab rig forbids only in these unfamiliar
    scenes.
- **Server time.** 1.5 s per request at p50 and 2.2 s at p95, over 1634 requests, on A100s
  shared with demo-collection Isaac.
  - Under `--inference-latency measured`, the arms would hold for about 1.5 s after every
    0.64 s chunk.
  - That run was not made: the zero-latency run already scores 0.

Conclusion:

- The open-loop replay reproduces RoboDojo demonstrations within 4–9 mm, so the integration
  is correct.
- In roboclock the checkpoint fails like pi0.5. It recognises some targets: it reaches for the
  glass and the infusion device. It does not complete any task in the new scenes, objects
  and instructions.
