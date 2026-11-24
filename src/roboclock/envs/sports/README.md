# Sports / Games

One package for the current ten fixed Dual ARX X5 tasks. Difficulty and random
seed are runtime arguments, not separate copies of the source. There are two
entry points over the same sub-packages:

- **Benchmark evaluation** — `roboclock eval --env sports` runs every task through
  the shared evaluator with the shared 16-dim Dual-X5 action/observation
  contract and writes schema-valid `episodes.jsonl` rows (see below).
- **Recording driver** — `python -m roboclock.envs.sports` runs one task's
  original demo loop and writes its native `result.json`, traces and optional
  videos (used for `data/sports/videos`).

```bash
roboclock tasks --env sports
OMNI_KIT_ACCEPT_EULA=YES roboclock eval --env sports --policy oracle --oracle \
  --task B4 --difficulty easy --seed-start 0 --seeds 1 --output results/sports/unified/B4
roboclock summarize --normalize results/sports/unified

OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src \
  .venv-isaac6/bin/python -m roboclock.envs.sports \
  --task B1 --difficulty easy --seed 0 --video --output results/sports/b1
```

`dual_arx_x5` is the default embodiment. Use a fresh output directory for each
run. Physics/oracle-only runs that need no images use the no-render
configuration (see [RoboDojo rig, cameras and no-render runs](#robodojo-rig-cameras-and-no-render-runs)):

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src python -m roboclock eval --env sports \
  --policy oracle --oracle --task B4 --difficulty easy --seed-start 0 --seeds 1 \
  --config src/roboclock/envs/sports/configs/no_render.yaml --output results/sports/headless/B4
```

The oracles are the privileged scripted controllers of the recordings;
successful rendering alone does not establish task success.

All tasks run physics at exactly 250 Hz (the configuration refuses any other
rate, including multiples; the PhysX scene's rate is read back at every reset)
and control at 50 Hz (`roboclock.envs.robodojo_rig`, `roboclock.envs.rates`). The
oracles act at 50 Hz; `roboclock eval --policy-hz 25` runs a policy at the RoboDojo
policy rate of 25 Hz. In evaluation the cameras render when a policy reads an
observation (docs/architecture.md, "Rendering on read"). The
current videos and results in `data/sports/videos` were recorded at these rates
(25 fps video); the superseded collections under `data/sports/videos/superseded/`
record their own rates. Each stored `result.json` records its rates.

Taiko counts as event-reactive (hit each note as it is cued) since 2026-11-19. The
sub-packages and the recording driver (`python -m roboclock.envs.sports --task ...`) keep their
own task kinds: arcade A2 = taiko, arena A3 = basketball (`tasks.NATIVE_KINDS`).

| ID | Task | Source | Episode budget |
|---|---|---|---|
| A1 | Chess and clock | `chess/` | 140 s (native cap) |
| A2 | Timed basketball shot (A3 until 2026-11-19; arena kind A3) | `arena/` | shot start + clock + 4 s grace, +1 s |
| B1 | Nine-hole moles | `arcade/` | 51 s (native) |
| B2 | Reactive boxing | `arena/` | last combination end, +1 s |
| B3 | Bench-press assistance | `bench_press/`; runtime in `_shared/precision/` | 45 s timeout, +1 s |
| B4 | Stop/go puck pushing | `freeze/` | seeded horizon (48 s of green; easy 101-111 s, medium 81-87 s, hard 71-75 s over seeds 0-9; 62 s before 2026-11-17), +1 s |
| B5 | Two-stick taiko (A2 until 2026-11-19; arcade kind A2) | `arcade/` | 53 s (native; 34 s before 2026-11-17) |
| C1 | Fifteen-ball pool racking | `pool_rack/` | horizon 5 t_o = 683.15 s, every difficulty |
| C2 | Speed Cups | `arena/` | 2 s + horizon 5 t_o = 157.05 s, every difficulty |
| C3 | Ball service | `ball_service/` | 1.25 x the last scheduled demand |

Each native evaluator ends its own episode (success, failure or deadline); the
budget only bounds an episode the evaluator has not ended, matching the caps of
the recording drivers. The C tasks follow the C-class rule
([c-class-timing](../../../../docs/c-class-timing.md), score_version 2,
`c_class.py`): C1 and C2 run the same episode at every difficulty and end when all
jobs are done and settled (2 s), at the stall stop or at the horizon; the difficulty
only picks the judged time tier. C3 runs one episode per difficulty: demand k is due at
m x (6.44 + 3.57 k) s (m = 4 / 2 / 1), the hard schedule being the tightest the slowest
oracle meets (`ball_service/config.py`).

`demo.py` routes task IDs for the recording driver. `common/` holds the shared
tabletop backend, task primitives and physics/evaluation helpers (the original
single-Franka numeric tasks 31–40, `--env sports_legacy`, were removed on 2026-11-21).
`_shared/` contains reused workflow and precision scene, task and controller modules.
These are dependencies, not alternative current versions. The separate `envs/chess/`
package supplies the chess rules, task model and scene used by A1; every task runs only on
the Dual ARX X5. `envs/lab/` remains the laboratory domain.

## Unified evaluation runtime

| Module | Role |
|---|---|
| `spec.py` | `sports` registration |
| `env.py` | `SportsBenchmarkEnv`: one SimulationApp, one active task backend at a time |
| `adapters/` | per sub-package runtime: `chess.py` (A1), `arcade.py` (B5, B1), `workflow.py` (A2, B2, C2, B3, B4, C1, C3) |
| `tasks.py` | catalog, simulator-free native configurations and episode specs, result rows |
| `oracle.py` | `SportsOracle`: the recordings' privileged controllers behind the 16-dim action |
| `joints.py` | native X5A articulation to the shared joint layout; joint speed limits across control rates |
| `config.py`, `configs/benchmark.yaml` | policy camera, observation rendering, task rates; `configs/no_render.yaml` (physics-only runs) |
| `rig_cameras.py` | RoboDojo rig: head + two wrist cameras from `robodojo_rig`, rig/link6/wrist check at every reset, renderer settings, rig-camera video writer |
| `head_view.py` | key objects and cues of every task in the head view: display moves toward the head camera, projected pixel margins |
| `recording.py` | recording-driver setup: 1280x960 overview, rig cameras with `--video`, X5 finger speed limit as in evaluation |
| `results.py` | native result to schema row (recorded `result.json` and live episodes) |

**Episodes.** `reset(seed, options)` honours `task_id` and `difficulty`; it
closes the previous task's backend and builds the requested task's scene on a
fresh stage with the sub-package's own configuration (the configuration the
recording driver uses), so the evaluator can switch tasks inside one
SimulationApp. The policy cameras are the RoboDojo rig's head and wrist cameras
(below), rendered with one renderer update and one RTX subframe when a policy
reads an observation (once per physics tick at most; steps whose observation
nobody reads render nothing). The task's presentation camera is built but never rendered in
evaluation. Capturing never advances physics (checked). With `sim.no_render:
true` no camera is built and nothing is rendered; every image key is then a zero
image of the same shape.

**Step.** `step(action)` validates shape, finiteness and the shared joint
limits, writes the 16 targets into the native command, runs one control period
of physics (the native evaluator observes every tick), updates the task displays and returns
`(obs, float(task.joint_success), terminated, truncated, info)`. `terminated`
means the native evaluator ended the episode; `truncated` means the budget did.
How a command is held matches each recording driver: the workflow backend
re-applies it with gravity feed-forward on every physics tick
(`ReplacementBackend.tick`), the arcade and chess backends apply it once per
control period and the position drives hold it. `advance_latency(seconds)`
ticks physics under the previous command. `info` is `{}` unless `--oracle`.

**Observation**: `rgb` (head, 480x640x3), `left_wrist_rgb` and `right_wrist_rgb`
(480x640x3), `joint_positions` (16, measured, shared order), `ee_pose` (14: per
arm the X5 grasp-frame position and `wxyz` quaternion from `RobotRig.pose`),
`gripper_width` (2: summed finger joints), `sim_time`, `instruction`; with
`--depth` also `depth`, `left_wrist_depth`, `right_wrist_depth` (480x640, m).

**Joint mapping.** The sports scenes load the manufacturer X5A pair
(`common/assets/dual_arx_x5`) as one articulation with DOFs `left_joint1` …
`right_joint8` in PhysX order (`left_joint1, right_joint1, left_joint2, …,
left_joint7, left_joint8, right_joint7, right_joint8`). The shared action
`left/jointK`, `right/jointK` maps to native `left_jointK`, `right_jointK`: a
pure permutation with the same units, sign and zero.

| Joint | Native X5A limits | Shared action bounds (`envs/lab/robot.py`) |
|---|---|---|
| joint1 | −2.618 … 3.1416 rad | −3.14 … 2.618 |
| joint2 | 0 … 3.665 | −0.05 … 3.50 |
| joint3 | 0 … 3.1416 | −0.10 … 3.20 |
| joint4 | −1.571 … 1.571 | −1.60 … 1.55 |
| joint5 | −1.571 … 1.571 | −1.57 … 1.57 |
| joint6 | −2.094 … 2.094 | −2.0 … 2.0 |
| joint7, joint8 (fingers) | 0 … 0.044 m | 0 … 0.044 |

Targets inside the shared bounds but outside the native range are clipped by
the native drive. Native ranges outside the shared bounds (joint1 > 2.618,
joint2 > 3.50, joint4 > 1.55, |joint6| > 2.0) cannot be commanded; none of
the ten recorded oracle episodes commands them (checked over their traces).
Measured joint positions are clipped to the shared bounds so that every
observation is a valid hold action (finger noise below 1e-5 m otherwise
leaves them).

**Scoring.** Every row comes from the sub-package's own evaluator of measured
physical state (`DuelTask`, `TaikoTask`/`MalletMoles`, `arena.tasks.Task`,
`BenchPress`, `freeze.task.Task`, `pool_rack.task.Task`,
`ball_service.task.Task`), with its unchanged success definitions, converted
by `results.normalize`, the same adapter used for the recorded `result.json`
files: physical → `task_success`, temporal → `temporal_success`, native
`success` → `joint_success`; failures → `temporal_violations` /
`physical_failures`; `completed_at` → `completion_time`; the C tasks' judged
fields (`throughput`: per-tier verdicts, job times, t_Q, termination reason) override
these last; reaction latencies
(mean) → `reaction_latency`; balls/cards/served per minute →
`operations_per_minute`; arm collision counts → `collision_count`. The full
native result (without events) is kept under `native_result`; events go to
`events`.

**Oracle.** `--policy oracle --oracle` constructs `SportsOracle(env)`, which
builds the task's recorded controller (`chess_oracle.RefinedChessOracle` +
retire motion, `DualToolOracle`, `DualMoleOracle`, `arena.oracle.Oracle`,
`BenchOracle`, `freeze.oracle.Oracle`, `pool_rack.oracle.Oracle`,
`ball_service.oracle.Oracle`) against the live backend. Each still reads
privileged state and writes a native command; the oracle returns it as the
16-dim action, which the environment applies. All of them are expressible as
joint-position targets.

**Environment-side mechanisms** (part of the task, not of the action):

- A1: the BLACK Dual X5 opponent runs the recordings' chess controller and
  retire motion. A turn passes once the active arm has returned within
  0.025 rad of its reset pose at least 1.5 s after its clock press; the episode
  ends 2 s after the last return. The scripted retire motion lifts above 1.09 m
  first; until 2026-11-10 the handoff required that lift too, which the
  instruction did not state (`adapters/chess.py`). BLACK's clock deadline counts
  from the handoff, WHITE's from BLACK's press (`chess/task.py`).
- B5/B1: drum springs and mole drives; B3: the bounded human-lifter wrench
  and its release on a measured catch; C3: hopper gate drives fired by the
  task schedule.
- All tasks: X5 drive gains, force limits and gravity feed-forward
  (`RobotRig.apply`), and per-task settings applied at reset (B4 finger force
  and solver iterations, C1 drive gains, B2 glove tool offset).
- The C1 and C3 controllers may record `controller_timeout_*` failures in the
  task (as in the recordings); other policies never trigger them.

**Simulator version.** The recordings used Isaac Sim 6.2.0; the unified runtime
was validated with Isaac Sim 6.0.1. Two 6.0.1 differences are handled in the
adapters, without touching the shared runtime:

- The arcade and chess backends import `isaacsim.robot.manipulators.examples`,
  which the shared runtime enables only for 6.2; the adapters enable it.
- The converted X5A USD stores the fingers' manufacturer speed limit (URDF
  `velocity=0.15` m/s) as `physxJoint:maxJointVelocity = 8.594` (a rad-to-deg
  conversion). Isaac Sim 6.2.0 applied 0.15 m/s; 6.0.1 applies 8.594 m/s. On
  B3 the jaws then close in about 25 ms instead of 160 ms and strike the falling bar,
  the pads chatter on and off contact and the 60 ms bilateral-hold dwell
  completes 75 ms later (catch latency 0.6875 s > 0.65 s deadline). Each reset
  therefore applies the URDF finger limit in SI units through the physics view
  (`robot.enforce_urdf_finger_velocity`, on by default; recorded in the row's
  `finger_velocity_limits`). With it, B3 reproduces the recording exactly.

The recording drivers (`python -m roboclock.envs.sports`) apply the same limit
since the rig migration (`recording.prepare`, the same
`robot.enforce_urdf_finger_velocity` setting), so a recording and an evaluation
episode run the same physics; each `result.json` lists the applied limits.

## RoboDojo rig, cameras and no-render runs

All ten tasks use the RoboDojo dual ARX X5 rig (`docs/robodojo-rig-spec.md`;
every value comes from `roboclock.envs.robodojo_rig`): the manufacturer X5A pair
(`common/assets/dual_arx_x5`) with both arm bases on one crossbeam, same height,
heading and forward coordinate, **0.60 m apart** (`left_mount`/`right_mount` at
y = ±0.30 m). Only the rig as a whole is placed per task (`embodiment.position`
= base midpoint on the mounting plane, `embodiment.yaw`); no task's placement
changed. A1, A2 and C1 drive one arm; the other X5 holds its observation pose.

**Cameras** (`rig_cameras.py`), built with RoboDojo's camera calls
(`set_focal_length`, `set_horizontal_aperture(h, False)`,
`set_vertical_aperture(v, False)`):

| Key | Camera | Mount | Intrinsics |
|---|---|---|---|
| `rgb` | `cam_head`, Gemini 345Lg | rig frame (0.04, 0, 0.543) m, 60° down, no roll; visual post, no collider | 640x480, focal 10, apertures 22.212 x 14.266 (HFOV 96°) |
| `left_wrist_rgb` / `right_wrist_rgb` | `cam_left_wrist` / `cam_right_wrist`, D435 | child prim of each arm's `link6` at `wrist_camera_in_link6()` (0.08484, 0, 0.05094) m, 30° below link6 +x | 640x480, focal 13, apertures 20.955 x 15.716 |

Every reset measures both base frames and the head camera from the stage and runs
`robodojo_rig.check_rig`, compares each measured `link6` (PhysX) with forward
kinematics of RoboDojo's X5A.urdf at the measured joint angles, and each wrist
camera with `link6 @ wrist_camera_in_link6()` (1 mm / 0.1°). The row's
`robodojo_rig` field holds `robodojo_rig.describe()`, the measured check, the
intrinsics the camera API reports (`get_intrinsics_matrix()`) and the renderer
settings read back from carb. Note: the camera API reports fy = 336.5 px from the
head camera's 14.266 mm vertical aperture, but the RTX renderer keeps square
pixels (fy = fx = 288.1 px, vertical FOV 79.6°); the rendered frames match the
square-pixel projection, which `head_view` uses. Results record that matrix as
`camera_intrinsics_rendered` next to the reported ones (`cameras[view].intrinsics_matrix`);
use it for projection and depth unprojection.

**Renderer.** `rig_cameras.apply_renderer` calls Isaac Lab's
`omni.replicator.core.settings.set_render_rtx_realtime(antialiasing="DLAA")`, then
sets `/rtx/rendermode = RaytracedLighting` (on Isaac Sim 6.0.1 that replicator call
selects RTX Real-Time 2.0) and every key of `RENDER_SETTINGS`; the read-back
values (and any antialiasing fallback) are stored in the row.

**Key objects in the head view** (`head_view.py`). At the 60° pitch the head
camera sees nothing higher than about 20° below its horizon, so the cue displays
that used to hang above or beside the workspace are turned toward the head
camera and placed low (`head_view.DISPLAYS`, visual geometry only): the A1
turn/clock panel under BLACK's mount, the B5 note screen on the table's far left,
the A2 shot clock on the table's free left side, the B2 coach screen as a floor
monitor in front of the robot (`arena/coach_display.py`), the B3, B4, C1 and C3
status boards beyond the workspace, the C2 pattern screen beyond the table. The
B4 stop/go signal stands 0.17 m lower. C1's ball caddies moved inward and forward
(rows |y| = 0.44/0.54 m, columns x = 0.24-0.54 m; `pool_rack/config.py`), the only
physical layout change. Every reset projects each task's key objects
(`head_view.KEY_OBJECTS`) into the head image and records the pixel margin (at
least 10 px required); the only exception is B3's bar, whose sleeves and plates
extend past the image borders (`head_view.EXCEPTIONS` gives the numbers).

**No-render runs.** `sim.no_render: true` (`configs/no_render.yaml`, which
extends `benchmark.yaml` through its `base:` key) runs physics and the oracle
only: the adapters build no camera, make no render call and set no renderer
setting, and observations carry zero images of the same shapes (also with
`--depth`). The arm frames are still measured and checked;
`robodojo_rig.rendered` and `observation_rendered` are `false`. Use it for layout
and controller iteration; acceptance runs and anything a visual policy sees use
the rendered default.

**Recordings.** Each recording driver writes the overview (the task's
presentation camera) at 1280x960 / 25 fps and, with `--video`, installs the rig
cameras before its first render and writes `head.mp4`, `left_wrist.mp4` and
`right_wrist.mp4` (640x480, 25 fps, the overview's timestamps and frame count),
their first/last frames and `rig_views.json` (`recording.attach`). Without
`--video` no camera is installed. `--presentation-only` is accepted and has no
effect (the old ego/single-wrist views are gone).

## Assets

Assets moved with their owning modules. Mesh geometry and textures are unchanged;
twelve derived chess USDs now use portable relative texture references.

- `arcade/assets/`: chess meshes, room textures, HDR and arcade assets.
- `arena/assets/`: arena/boxing/cup resources and source manifests.
- `bench_press/assets/`: human model and grip data.
- `pool_rack/assets/`: numbered ball textures.
- `_shared/workflow/assets/`: shared equipment assets.
- `common/assets/`: original manipulation-object library.

Each asset directory retains its source/license records. Main bundles the prepared X5 robot under `common/assets/dual_arx_x5/`;
historical checkouts use the prepared local cache. See
robot assets.
