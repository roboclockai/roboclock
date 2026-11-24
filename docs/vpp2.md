# VPP2 (Video Prediction Policy 2, RoboDojo) in roboclock

`roboclock.envs.lab.policies.vpp2` drives the dual X5 arms with
[`Haodong082399/VPP2`](https://huggingface.co/Haodong082399/VPP2), bundle `joint2b_s100000`:
a Wan2.1-I2V-14B video model trained jointly with a 2B action expert on the RoboDojo
demonstrations (100k steps). It is served by XPolicyLab's `policy/VPP2` adapter, which wraps the
upstream code ([roboterax/video-prediction-policy-2](https://github.com/roboterax/video-prediction-policy-2)
at `ed39864`). For the websocket protocol and the joint mapping, see
[xpolicylab-pi05.md](xpolicylab-pi05.md); for the RoboDojo end-effector frame, see
[wallx.md](wallx.md).

## What the checkpoint expects

- **Absolute EE16 actions.** Each request returns RoboDojo `ee` actions: per arm, the link6 pose
  `[x, y, z, qw, qx, qy, qz]` in RoboDojo's environment frame and a gripper opening in [0, 1],
  left arm first. The model predicts 32 steps; the server returns the first 24
  (`replan_steps`), and the client replays them before it asks again, as
  `policy/VPP2/deploy.py` does.
- **State in RoboDojo's frame, with RoboDojo's quaternion sign.**
  - The client computes each arm's link6 pose from the measured joints, as the Wall-X client
    does (`wallx.link6_pose`). It solves the returned poses to joints along the chunk
    (`wallx.solve_chunk`).
  - VPP2 z-scores the state quaternion component by component, so q and −q are different
    inputs. RoboDojo's `ee_pose` is PhysX's link pose: the composition of the base yaw and the
    joint rotations `[cos(θ/2), sin(θ/2)·axis]`, whose sign varies continuously with the joints.
  - A quaternion taken from the rotation matrix has the opposite sign on about a quarter of the
    demonstration frames, and at roboclock's HOME pose: (−0.5, 0.5, −0.5, −0.5) instead of
    RoboDojo's (0.5, −0.5, 0.5, 0.5).
  - `vpp2.robodojo_ee_pose` keeps the rotation and takes the sign of that composition. It
    matches every recorded quaternion of both replay demonstrations (644 frames per arm).
  - With the wrong sign, the open-loop check below fails on those frames. The model returns
    quaternions of norm 0.55–0.62, which are 38–109° off.
- **The observation history lives on the server.**
  - The adapter keeps one history per client id (`_vpp2_client_id`) and records one frame per
    `update_obs` call. The step of a frame is the number of frames recorded before it.
  - A request at step q is conditioned on the episode's first frame (the anchor) and on the
    frames at q, q−25, …, q−175. Steps ≤ 0 read the anchor. All nine frames are composed into
    the 240×416 T-shaped view: the head view at 480×640 beside both wrist views at 240×240.
  - As in `deploy.py`, the client calls `update_obs` on every RoboDojo step and `get_action`
    on every 24th. Requests use the `call` message, with `reset` and `reset_client` at the
    start of each episode and a fresh client id per episode.
  - Two episodes can share one server, because their histories never mix. The server's
    `latency_ms` includes the wait for its model lock.
- **Render only the frames the server reads.**
  - With replan 24 and stride 25, a request reads step s only if s + 25j is a request step for
    some j in 0..7. That holds when s mod 24 is 0 or 17–23: a third of the steps.
  - The client renders those steps. On the other steps it sends the last rendered JPEGs again,
    which keeps the server's step count right; the server never reads those images.
  - Before each request the client checks that every step the request reads was rendered.
    A missing step is an error, never a substitute.
  - `VPP2_RECORD_ALL_FRAMES=1` renders every step instead.
- **One act() per RoboDojo step.** Run `roboclock eval` with `--policy-hz 25`. The client raises
  an error at any other rate.
- **Released inference settings**: 10 Euler steps, sigma shift 1, seed 1. Request k of an
  episode uses noise seed 1 + k (per client), so the same observations give the same chunks.

## Setup (policy machine)

```bash
P=$ROBODOJO_DIR/XPolicyLab/policy/VPP2
# Weights: the eval bundle and the shared Wan2.1 encoders (download_checkpoints.sh, eval stage),
# 53.5 GB of the 252 GB repository.
hf download Haodong082399/VPP2 --revision 04824ddb3aae11f13db50984ed7d540c3b77dd9f \
  --local-dir $MODELS_DIR/VPP2 \
  --include config.json 'checkpoints/Wan2.1-I2V-14B-480P/*' 'checkpoints/Wan2.1-I2V-14B-480P/**' \
            'checkpoints/joint2b_s100000/*'
mkdir -p $P/checkpoints
ln -s $MODELS_DIR/VPP2/checkpoints/joint2b_s100000 $P/checkpoints/
ln -s $MODELS_DIR/VPP2/checkpoints/Wan2.1-I2V-14B-480P $P/checkpoints/

# install.sh with uv in place of conda
git clone https://github.com/roboterax/video-prediction-policy-2.git $P/upstream
git -C $P/upstream checkout --detach ed398648fac3a0d7fb1e1eb9179db9d71cf326de
unset TRITON_PTXAS_PATH
uv venv --python 3.10 $P/.venv
sed -E '/^(torch|torchvision)==/d' $P/upstream/environment-reference.txt > /tmp/vpp2_constraints.txt
uv pip install --python $P/.venv/bin/python torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python $P/.venv/bin/python -e "$P/upstream[eval]" -c /tmp/vpp2_constraints.txt
uv pip install --python $P/.venv/bin/python -e $ROBODOJO_DIR/XPolicyLab -c /tmp/vpp2_constraints.txt
$P/.venv/bin/python $P/launch_policy.py --dry-run   # checks bundle sizes and encoders

bash scripts/xpolicylab/serve_vpp2.sh $ROBODOJO_DIR 6022 <gpu>
```

`serve_vpp2.sh` runs the adapter's `launch_policy.py`, which validates the bundle and execs
`setup_policy_server.py` with `deploy.yml`, as `setup_eval_policy_server.sh` does.

- Loading takes about 5 minutes.
- The server uses about 48.5 GB of GPU memory.
- A request takes about 1.4 s on an A100 by itself, and about 2.4 s on an A100 shared with
  demo-collection Isaac. With two episodes on one server, each also waits for the other's
  request.

Differences from the published setup:

| Item | Published | Here |
|---|---|---|
| Environment | conda, Python 3.10, ffmpeg from conda-forge | uv venv, Python 3.10; no ffmpeg (only data conversion needs it); no modelscope |
| torch | 2.11.0 with CUDA 13.0 | the same |
| Reference GPU | one 96 GiB RTX PRO 6000 | one 80 GB A100 (the README calls smaller devices unverified) |
| Weights | `download_checkpoints.sh` into `policy/VPP2/checkpoints` | the same files, in `$MODELS_DIR/VPP2` and symlinked; HF revision 04824dd |
| Simulator client | RoboDojo env client with `deploy.py` | roboclock client, the same calls |

## Run (simulator machine)

```bash
export XPOLICYLAB_URL=ws://127.0.0.1:6022 XPOLICYLAB_LOG=qlog/lab_A5_s0.jsonl
PYTHONPATH=src:$ROBODOJO_DIR python -m roboclock eval \
  --env lab --task A5 --difficulty easy --seeds 1 --seed-start 0 \
  --policy roboclock.envs.lab.policies.vpp2:make --policy-hz 25 \
  --inference-latency zero --output out/lab_A5_s0
```

`XPOLICYLAB_LOG` writes one row per request with these fields:

- server time (`update_obs` plus `get_action`) and round-trip time;
- frames rendered so far, and the steps the request reads;
- the state pose and gripper, and the first and last target poses;
- gripper commands;
- IK error, clipped targets, and the start jump and travel of each chunk.

`VPP2_FRAME_DIR` keeps the head JPEG of every request.

## Integration check

An open-loop replay of RoboDojo demonstrations ran every frame through the client against the
real server, with one `update_obs` per frame and a request every 24. The gripper state came
from the demonstration. Each returned 24-step chunk was compared with the demonstration's next
24 actions.

| Demonstration | Arm | Position error (mean) | Hold baseline | Rotation error | Hold baseline | Gripper error |
|---|---|---|---|---|---|---|
| stack_blocks, 11 requests | left | 1.1 mm | 89.3 mm | 0.16° | 6.8° | 0.001 |
| | right | 0.9 mm | 47.7 mm | 0.25° | 8.9° | 0.001 |
| press_by_number, 15 requests | left | 1.2 mm | 55.1 mm | 0.21° | 12.5° | 0.004 |
| | right | 0.8 mm | 33.5 mm | 0.15° | 9.6° | 0.004 |

- The client's FK matches the recorded `ee_pose` within 0.125 mm.
- The client rendered 81 of 243 and 113 of 353 frames, and no request read a frame that was not
  rendered.
- Both replays ran at the same time against one server without mixing histories.
- The errors are this small most likely because the model saw these demonstrations in training.
- Before the quaternion-sign fix, stack_blocks gave 8.3 mm (left) and 2.2 mm (right) for
  position but 19.7° for left rotation, worse than the 6.8° of the hold baseline.

## Result: the 30-episode comparison set (2026-11-07)

Setup: the same 30 episodes as for pi0.5 and Wall-X.

- lab A5/B1, hospital A3/B4, household A2/B2, sports A3/B2 and industry A1/B2;
- seeds 0–2, all at easy, including industry;
- `--inference-latency zero` and `--policy-hz 25`.

Two servers were used: port 6022 on GPU 2 and port 6032 on GPU 5, each shared by up to two
episodes. Episodes keep separate histories.

| | Joint | Physical | Stages reached |
|---|---|---|---|
| VPP2 | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 (score 20); nothing else |
| Wall-X (zero latency) | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 |
| pi0.5 (measured latency) | 0/30 | 0/30 | household A2 `glass_picked_up` 3/3 |

Failure modes, from the termination reasons:

| Task | Failure |
|---|---|
| household A2 | Lifts the glass. s0 holds it until the 40 s budget; s1 and s2 tip it (`glass_tipped` at 9.8 s and 29.5 s). |
| household B2 | `wrong_object_disturbed` (s0, s2) or `object_fell` (s1) within 2–5 s. |
| lab A5 | `developer_loading_timeout` at 15 s. |
| lab B1 | `overheat`: the beaker is never lifted. |
| hospital A3 / B4 | No progress before the end. |
| sports A3 | `no_grasp`. |
| sports B2 | `moved_during_hold`, `incomplete_combinations`. |
| industry A1 / B2 | No progress before the time budget runs out. |

- **Server time.** One `update_obs` plus `get_action` takes 2.09 s at p50 and 3.82 s at p95,
  over 1189 requests. This includes the wait for the shared lock.
- **Wall time.** Episodes took 23–43 min for the long budgets.
- **IK.** The largest miss is 24 mm, in hospital B4 s2; elsewhere it stays under 13 mm. No
  target was clipped by the joint limits.
- Results and per-request logs are in the session scratchpad, `models/vpp2/zero/`. The
  overview is in `final_summary.txt`.

Conclusion:

- The replay reproduces RoboDojo demonstrations within about 1 mm, after the quaternion-sign
  fix, so the integration is correct.
- In roboclock the checkpoint reaches exactly the same stage as pi0.5 and Wall-X.
