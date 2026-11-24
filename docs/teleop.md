# Browser keyboard teleoperation

`roboclock.policies.teleop` lets a person drive the Dual ARX X5 rig with the keyboard from
a web page, for a human baseline. It is an ordinary `roboclock eval` policy, so episodes,
scoring and result rows are the same as for any model. No display, NVENC or WebRTC is needed. The
page loads from a small aiohttp server that the policy starts inside the eval process. Camera
frames go to the browser as JPEG over a websocket, and held keys come back.

## Running

```bash
cd $ROBOCLOCK_DIR
CUDA_VISIBLE_DEVICES=1 TELEOP_PORT=8765 TELEOP_LOG=results/teleop/logs \
LD_LIBRARY_PATH=/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu OMNI_KIT_ACCEPT_EULA=YES \
python -m roboclock eval --env household --task A1 --difficulty easy \
  --seeds 1 --policy roboclock.policies.teleop:make --inference-latency zero \
  --output results/teleop/household_A1
```

Open the page through code-server's proxy, with the trailing slash:
`https://<code-server host>/proxy/8765/` (on the remote node pod, `$VSCODE_PROXY_URI` with
`{{port}}` = 8765). The page uses only relative URLs, so it also works through a VS Code
port forward or at `http://<host>:8765/`. Click the page once so it has keyboard focus.

Each episode starts **frozen** and waits for Enter. The evaluator builds a new policy per
episode, but the server stays up for the whole process, so the tab stays open across a
multi-seed or multi-task plan. When an episode ends, the page shows its result (physical,
temporal and joint success, plus the score) and waits for one of two buttons:

- **Restart (same seed)**: run the same plan entry again from reset.
- **Continue**: keep this attempt and go to the next plan entry, or end the run after the last.

**Restart episode** (or Backspace; both ask first) abandons the running attempt and starts it
again. To do this, the policy raises `evaluator.RestartEpisode` from `act()`.
`episodes.jsonl` holds one row per plan entry, the attempt the operator kept, and its
`attempt` field says which attempt that was. Every redone attempt is written to
`restarted_attempts.jsonl`, so retries stay visible. An abandoned attempt has
`termination_reason: restarted`; an attempt redone after it finished keeps its reason.

## Keys

| Key | Action |
|---|---|
| Enter | start the episode / resume |
| P | freeze or unfreeze (no simulation time passes while frozen) |
| `.` | while frozen, advance 0.1 s |
| `[` `]` | slower / faster: 0.1, 0.25, 0.5, 1.0 × real time |
| 1 / 2 / 3 | control the left / right / both arms |
| W S, A D, R F | tool point forward/back, left/right, up/down |
| Q E, I K, J L | yaw (world up), pitch, roll |
| Shift (held) | fine motion (×0.25) |
| Space | open / close the selected gripper |
| H | move the selected arm back to its start joints |

"Forward" is the direction the arm bases face in the horizontal plane. "Left" is world up ×
forward. In the head camera, forward points away from the robot. Motion is defined in
simulation time: 0.12 m/s and 0.8 rad/s, capped at 0.6 rad/s per joint (the move_eef agent's
cap). Closing a gripper commands the rig's lowest finger target, so it squeezes whatever it holds.

Keys are re-sent every 0.1 s. If the server receives nothing for 0.3 s, it drops all held
keys, so a lost tab or connection stops the arms. Each arm follows its target through the
pure-Python X5 kinematics, calibrated from the first observation, as in the move_eef agent.
A target the arm cannot follow (out of reach, or limited by joint speed) is pulled back to
the arm's pose once it lags more than 2 cm, so releasing a key always stops the arm.

## Time

The policy reports zero inference latency, so the operator's thinking time is never charged.
It also paces the simulation itself:

- **Speed s ≤ 1**: one simulated second takes at least 1/s wall seconds. A simulator slower
  than that pace runs as fast as it can and does not try to catch up. The page shows the
  achieved ratio.
- **Frozen**: the clock stops. This is the freeze protocol for tasks the simulator cannot run
  in real time.

Frames are rendered every `TELEOP_SPEED / TELEOP_FPS` seconds of simulation time, which
gives `TELEOP_FPS` frames per wall second when the simulator keeps pace. The schedule uses
simulation time on purpose. With a wall-clock schedule, a step slower than 1/fps would render
on every step, and household ran at 0.13× instead of 0.42×.

Measured on this pod (L40S, GPU 1 idle, 2026-11-17), household:

| | Per control step (20 ms of simulation) |
|---|---|
| physics only (A1) | 22 ms |
| render all three rig cameras every 7th step (A1) | 38 ms |
| render on every step (A1) | 144 ms. One render (`world.render` + Replicator step) is about 115 ms |
| teleop at speed 1.0, 10 fps (B2) | achieved 0.42× real time |

To get closer to real time, lower `TELEOP_FPS`. The render cost is the domain's exact
observation path, so this policy does not change it. Domains with slower physics (lab C1,
sports A1, see the frame wall-time notes) cannot reach real time at any frame rate. Use
slow motion or freeze for those.

## Settings

| Variable | Default | |
|---|---|---|
| `TELEOP_PORT` | 8765 | server port (binds 0.0.0.0) |
| `TELEOP_FPS` | 10 | frames per wall second at pace |
| `TELEOP_SPEED` | 1.0 | starting speed, (0, 1] |
| `TELEOP_LINEAR` / `TELEOP_ANGULAR` | 0.12 m/s / 0.8 rad/s | tool-point speeds |
| `TELEOP_FINE` | 0.25 | Shift scale |
| `TELEOP_JOINT_SPEED` | 0.6 rad/s | joint speed cap |
| `TELEOP_LOG` | unset | directory for one JSONL per episode: every step's sim time, held keys, action, measured `ee_pose` and targets; every command; the result |

## Pieces

- `src/roboclock/policies/teleop/policy.py`: key mapping, IK, grippers, freeze, pacing and frame schedule
- `src/roboclock/policies/teleop/server.py`: aiohttp server thread (page, websocket, dead-man)
- `src/roboclock/policies/teleop/index.html`: the page
- `evaluation/evaluator.py`: optional `policy.episode_finished(row)` hook, called after each row is written
- `tests/test_teleop_policy.py`: runs without Isaac (fake observations, real websocket client)
