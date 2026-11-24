# Household: current Dual-X5 tasks and accepted demos

## Current design: Dual ARX X5, revision R2 (2026-10-21)

[tasks.json](../../src/roboclock/envs/household/tasks.json) lists the ten current tasks: four temporal-window, three event-reactive and three throughput. Each task has its own package `src/roboclock/envs/household/<id>_x5/` containing:

- `design.json`: the contract, covering instruction, process model with nominal and hidden parameters, timing events, completion and failure, permitted strategies, arm roles and open items;
- `task.py`: the task, process and scoring model implementing the shared evaluator contract (`make_task`, `spec`, `gen_instruction`, `metrics`), plus `update(signals)`, `advance(dt)` and `cues()` for a physics backend;
- `scene.py`: the static initial scene builder;
- `sim.py`: the task's physics overlay (colliders, rigid bodies, appliance joints) and its `Oracle` class: `measure()` (signals from measured physics), `appliance_step()` and `update_cues()` are the environment's probe; `command()` is the privileged oracle controller.

Shared code is in `x5/` (task base, scene library, the dual-arm rig `rig.py` (base layout and head camera), X5 kinematics, still renderer, and the Gym environment: `env.py`, `episode.py`, `interface.py`, `oracle.py`, `scenes.py`, `config.py`, `validation.py`) and `x5/sim/` (articulated X5 arms, servo). R2 assets are in `assets/x5/`. The Lightwheel kitchen is downloaded rather than committed: `scripts/household/fetch_lightwheel_kitchen.py` unpacks the files the X5 tasks load into `assets/household/lightwheel_kitchen/` and links them.

- [Task table and design rationale](x5-tasks.md) and the review that motivated R2
- [Task interface for the simulation backend](x5-task-interface.md): signals, cues and hidden parameters per task
- [R2 asset sources and licenses](x5-assets.md)
- [Physics episodes and demos for all ten tasks (A1–A4, B1–B3, C1–C3)](x5-simulation.md): runner, independent validator, robot model, results and limitations
- Demos: `data/household/<ID>/demo.mp4` and `demo.json` (see data/README.md); the earlier `x5_demos/<ID>/` recordings are in git at a417e24
- Initial scenes: built in code at every reset (`x5/scenes.py` runs `x5/build.py` with the task's `scene.py`); `data/household/x5_initial/` was removed in b35aedd. Built at the recordings' folder depth they are byte-identical to `initial_scene_sha256` in `tasks.json` (`tests/envs/household/test_x5_env.py` rebuilds all ten).

```bash
roboclock tasks --env household
roboclock plan --env household --task all --difficulty all --seeds 3
python -m pytest tests/envs/household
# on the GPU server (OpenUSD / Isaac Sim 6.0.1 Pythons):
python scripts/household/fetch_lightwheel_kitchen.py
# shared CLI, Isaac Sim 6.0.1 Python (see "Evaluate with the shared CLI" below)
roboclock eval --env household --policy oracle --oracle --task A1 --difficulty medium --seeds 1 --output results/household/unified/A1
# headless (RoboDojo sim.no_render: no cameras, no rendering, zero images)
roboclock eval --env household --config src/roboclock/envs/household/configs/no_render.yaml --policy oracle --oracle --task A1 --difficulty medium --seeds 1 --output OUT
# head-camera visibility of every key object (>= 10 px inside the 640x480 image)
PYTHONPATH=src python -m roboclock.envs.household.x5.visibility SCENES/*/initial_state.json
# standalone tools; static scenes go to $ROBOCLOCK_X5_INITIAL/<ID> (default data/household/x5_initial)
ROBOCLOCK_X5_INITIAL=SCENES PYTHONPATH=src /opt/household/usd_static/bin/python -m roboclock.envs.household.x5.build
PYTHONPATH=src python scripts/household/check_x5_reach.py
/opt/household/runtime_601/venv/bin/python scripts/household/run_x5_episode.py --task A1 --initial SCENES/A1/initial_scene.usda --out RUN_DIR [--record]
/opt/household/runtime_601/venv/bin/python scripts/household/validate_x5_episode.py RUN_DIR
```

**Status:**
- **Done:** the task, process and scoring models; tests driving every task with synthetic early, on-time and late signal sequences; static scenes and renders; a reach screen confirming all 50 operation points are reachable.
- **A1 and A2:** physics episodes with a privileged oracle controller, validated on seeds 0–2 by the task model and an independent trajectory check, and recorded (seed 0). A2 is the ice dispenser (half a glass of ice; a countertop unit, described as a refrigerator-door style dispenser since 2026-11-16) since 2026-11-03 (the countertop-oven task is kept in `a2_oven_x5/`); the A2 rows in the runtime comparisons below are the oven task.
- **A3, A4, B1, B2, B3, C1, C2 and C3:** one successful reference episode each (seed 0, medium), validated by the task model and the independent check, and recorded. A3, A4, B3 and C3 use R2.1 layout revisions (A4 is a two-arm pour); C1 uses R2.2. C2 was replaced on 2026-11-02 by two-arm cola pouring; since R4 (2026-11-03) the six glasses stand in one row on a bare counter and the cola is a PhysX particle fluid.
- **C1 R3 (2026-11-06):** the serving relay that replaced the microwave task is validated physics-only through the environment (easy, medium and hard, several seeds; see [x5-simulation.md](x5-simulation.md)), but not yet recorded: `data/household/x5_demos/C1/` still holds the R2.2 episode, and the R2.2 rows below describe that recording.
- **Unified runtime (2026-10-29):** `roboclock eval --env household` runs all ten tasks through the shared evaluator (`x5/env.py:HouseholdX5Env`, below). With the privileged oracle at each recording's difficulty and seed it reproduces the recorded outcomes; see [Unified runtime vs. recordings](#unified-runtime-vs-recordings).

## Evaluate with the shared CLI

```bash
# Isaac Sim 6.0.1 (Python 3.12), kitchen in place; GPU and graphics setup as for the other domains
roboclock smoke --env household --task A1
roboclock eval --env household --policy oracle --oracle --task A1 --difficulty medium \
    --seed-start 0 --seeds 1 --output results/household/unified/A1
roboclock eval --env household --policy hold --task B2 --difficulty medium --seeds 1 \
    --output results/household/unified/hold_B2
roboclock summarize --normalize results/household/unified
roboclock render --env household --task A2 --output results/household/render/A2.png
roboclock validate --env household --task B1 --difficulty medium --seeds 1 --seconds 10 [--passive]
```

**First frame of a scene.** Each task scene needs its RTX ray-tracing pipeline compiled by the GPU
driver. On a cold driver shader cache this takes minutes (8–10 min per process on the shared server
on 2026-10-29), and `reset()` waits for it (up to 30 min). Kit keeps that cache in
`<portable root>/cache/nv_shadercache`, which is limited to about 4 GB by default. When several
projects share it, entries are evicted and every process recompiles. For batches, give the runs one
portable root of their own and a larger limit, e.g.
`ROBOCLOCK_ISAAC_PORTABLE_ROOT=<dir> __GL_SHADER_DISK_CACHE_SIZE=17179869184`. Then each scene is
compiled once. Keep Kit's worker threads at 16 or more (`ROBOCLOCK_CPU_THREADS`, default 32): with 4,
the asynchronous pipeline compilation occupied them and PhysX settling stalled.

**Environment** (`x5/env.py`, registered in `spec.py`; settings in the `x5` section of
`configs/benchmark.yaml`, the other sections being the legacy schema of the earlier recorders):

- `reset(seed, options)` makes the task (`options["task_id"]`, `["difficulty"]` switch tasks inside one
  SimulationApp), builds its initial scene in code (`x5/scenes.py`) into a per-environment folder
  `results/household/.x5-scenes-*` (removed by `close()`), adds the physics overlay
  (`x5/sim/physics.py` and `<id>_x5/sim.py:build`), opens it as a new stage, sets the ready pose and
  settles passively for 3 s. The task clock starts after that, as in the recordings. Every row records
  the scene's `initial_scene.sha256` and whether it equals the recording's (`matches_recorded_demo`).
- `step(action)` applies 16 joint-position targets for one control period: physics 250 Hz, control
  50 Hz, the rates shared by every domain (`roboclock.envs.rates`). Each physics tick runs, in the recorder's order, the physics step,
  appliance mechanisms (oven rack linkage, B2 release peg, the C1 waiter, customer
  pickups), the measurement, `task.update(signals)` and `task.advance(dt)`, with process cues every
  10 ticks. Reward is `float(task.joint_success)`. `advance_latency()` keeps the previous targets.
  An arm whose 8 targets did not change is not written again. A PhysX target write also wakes a
  sleeping articulation, and the recorder wrote only what its controller commanded. Re-sending
  resting arms every step shifted trajectories by about 1e-7 rad, which was enough to lose a marginal
  C3 mug grasp at 35 s and to mis-flip a C2 slice.
- Scoring is the task model `<id>_x5/task.py`, fed only by measured signals (rigid-body poses,
  contact forces, joint and tool poses), never by controller phases. The throughput tasks follow the
  C-class rule since 2026-11-09 ([x5-tasks.md § 17](x5-tasks.md)): they end early once all is done and
  settled (C2 and C3 also on the stall stop, C2 on a failure), otherwise at `max_time`, and write
  `termination_reason`. The task clock sums 1/250 s steps, exactly as the
  recorder does (event times depend on it: C1's former 56 s pickup fired at 56.004 s). It can end a hair short of a
  whole-second horizon, so an episode runs until the clock reaches `max_time`, possibly one tick past
  `round(max_time / dt)`, as the recorder does. Otherwise a pickup scheduled at `max_time`, like C2's
  at 6:00, could go unscored.
- `info` is empty unless `--oracle`. `task.metrics()["events"]` is the same list the evaluator writes
  as `row["events"]`; first-occurrence times by name are in `event_times`.

| Space | Shape | Household content |
|---|---|---|
| action | 16 | `left/joint1..8`, then `right/joint1..8`. Joints 1–6: radians, lab limits (identical to the household X5 chain). Joints 7–8: the fingers, as per-finger opening from the closed stop in metres (lab convention; native `jawP = opening − 0.03014`, `jawN = 0.03014 − opening`). **Finger limits differ from the lab robot:** [−0.03, 0.044] instead of [0, 0.044]. The household jaw travels 30.14 mm per side; a negative target lies past the closed stop, so the 20 N jaw drive clamps a thin part (the oracle squeezes by up to 25 mm). Values above 0.03014 press against the open stop. |
| `rgb` | 480×640×3 | `cam_head` = `/World/AgentCamera`, the RoboDojo head camera (`roboclock.envs.robodojo_rig`: 0.04 m ahead of the base midpoint, 0.543 m above the mount tops, 60° down, Gemini 345Lg, rendered 96.0° × 79.6° with square pixels; `get_intrinsics_matrix()` reports 71.0° vertical, which RTX does not render), RTX `RaytracedLighting` + DLAA, rendered once per 25 Hz policy observation (steps in between return the latest render). Zeros under `no_render` |
| `left_wrist_rgb`, `right_wrist_rgb` | 480×640×3 each | `cam_left_wrist` / `cam_right_wrist`: D435 cameras rigid on each arm's `link6` (RoboDojo mount; left = the arm at x = 0.00, the rig's left). Zeros under `no_render` |
| `joint_positions` | 16 | measured, action convention, clipped to the action limits |
| `ee_pose` | 14 | per arm: tool point (link6 + 0.145 m) xyz and quaternion w, x, y, z with +Z along the approach axis (lab convention) |
| `gripper_width` | 2 | per arm: sum of the two finger openings (at most 60.3 mm) |
| `sim_time` | 1 | task clock |
| `instruction` | text | `task.gen_instruction()` |
| `depth`, `left_wrist_depth`, `right_wrist_depth` (with `--depth`) | 480×640 each | distance to the image plane |

**Oracle** (`--policy oracle --oracle`, `x5/oracle.py`): the recordings' controller
(`<id>_x5/sim.py:Oracle.command`) in policy form. Its servos are unchanged; their articulation
writes are captured into the 16-value action, which the environment applies at the same 50 Hz
control ticks. It reads privileged state: object poses, contact forces, the hidden process cues and
the simulation clock. It also names the arms its controller wrote in that tick
(`X5Episode.rewrite`), so arms re-commanded with unchanged values are written as well and the PhysX
writes are exactly the recorder's. At the recordings' 120/60 Hz rates, A4 (all 614 trace samples) and
C3 (the first 40 s) reproduced the recorded trajectories bit for bit in this environment. Differences from
`scripts/household/run_x5_episode.py`:
- the evaluator ends an episode at task termination, without the recorder's 2.5 s tail;
- fixed-horizon tasks run to `max_time`, where the recorder stopped once the oracle had parked
  (C1 at 542 s of 600 s, C3 at 240 s of 300 s). The policy holds its last targets after that;
- after a controller failure (phase timeout, lost grasp, IK step outside the X5 bounds) the policy
  holds its last targets instead of aborting the run.

### 250/50 Hz runtime vs. the 120/60 Hz runtime

Until 2026-11-02 the recordings in `data/household/x5_demos` were made at physics 120 Hz, control 60 Hz; all
ten have since been re-recorded at 250/50 Hz with 25 fps video (B2 and C2 as new tasks). This section keeps the
first 250/50 Hz comparison, which used the old recordings; its B2 and C2 rows are the replaced tasks. `roboclock eval --env household --policy oracle --oracle
--task <ID> --difficulty medium --seed-start 0 --seeds 1` at 250/50 Hz (2026-10-30, output in
`results/household/unified_250_50/<ID>`) compared with the 120/60 Hz rows of the next section:

| Task | 120/60 phys/temp/joint | 250/50 phys/temp/joint | Key events (s) | Values | Sim s |
|---|---|---|---|---|---|
| A1 | ✓/✓/✓ | ✓/✓/✓ | extraction 9.05 → 9.04; spoon_returned 35.79 → 35.53 | strength_at_extraction 0.5951 → 0.5951 | 35.79 → 35.53 |
| A2 | ✓/✓/✓ | ✓/✓/✓ | door_opened 127.83 → 127.90; retrieval 168.63 → 168.40; tray_on_trivet 177.93 → 178.31 | browning_at_retrieval 0.6027 → 0.6027 | 188.63 → 188.40 |
| A3 | ✓/✓/✓ | ✓/✓/✓ | flip 49.58 → 51.14 | first_side_doneness 0.6241 → 0.6235 | 54.71 → 54.92 |
| A4 | ✓/✓/✓ | ✓/✓/✓ | kettle_down 37.93 → 38.85 | final_dose_g 298.33 → 298.70 | 40.93 → 41.85 |
| B1 | ✓/✓/✓ | ✓/✓/✓ | effective_closure 3.37 → 3.36 | overflow_l 0 → 0 | 3.37 → 3.36 |
| B2 | ✓/✓/✓ | ✓/✓/✓ | ejected 5.24 → 5.23 | char_at_eject 0.6547 → 0.6545 | 10.24 → 10.23 |
| B3 | ✓/✓/✓ | ✓/✓/✓ | cover 15.80 → 15.68; heat_off 27.06 → 26.96; secured 37.05 → 36.96 | flame_exposure 4.319 → 4.277 | 37.05 → 36.96 |
| C1 (R2.2) | ✓/✓/✓ | ✓/✓/✓ | pickup2 300.01 → 300.00; pickup4 540.01 → 540.00 | qualified_meals 4 → 4; bowl_temp_at_pickup_c [68.45, 69.83, 75.55, 77.78] → [68.44, 69.85, 75.61, 77.87] | 600.01 → 600.00 |
| C2 | ✓/✓/✓ | ✓/✓/✓ | pickup1 210.01 → 210.00; pickup2 360.01 → 360.00 | qualified_slices 4 → 4 | 360.01 → 360.00 |
| C3 | ✓/✓/✓ | ✓/✓/✓ | setting1_ready 93.21 → 99.63; setting2_ready 155.81 → 164.35 | settings_ready_on_time 2 → 2; final_residue [0.09423, 0.09469] → [0.09476, 0.09478] | 300.01 → 300.00 |

- Jaw closing, press regulation and all integrators now run in physical units on the control period,
  so they keep their rate at any control frequency. The per-tick values they replaced assumed 60 Hz.
- A3 and C2 needed a slower lever (A3 2 s → 3 s, C2 1.5 s → 2.25 s). With the faster lever the pancake
  or square fell back: A3 did not flip (the arm ran up to 11 cm behind its goal, with joint 4 at its
  limit), and C2's second square fell back, which led to a later turner drop. Both now turn through the
  blade contact and the closed-loop topple. A3 also passes at easy seed 2 and hard seed 1 (physics-only).
- C3's settings are ready later (6.4 s and 8.5 s) but before their requests. The cause was not traced.
- C2 at other difficulties (physics-only, all four squares turned once):
  - easy seed 2 fails: pickup 2 falls at 480 s (360 s / 0.75), past `max_time` = 360 s. That is a
    property of the task spec, independent of the rates.
  - hard seed 1 fails: 3 of 4 slices qualify, the fourth is delivered at 251.6 s for a pickup at 240 s.
  Neither has a 120/60 Hz baseline.
- `hold_B2_25hz` (`--policy hold --policy-hz 25`, 2 control steps per action): measured on the old toaster task,
  the same result as `hold_B2` below.

### Unified runtime vs. recordings

These runs predate the 250/50 Hz rates (physics 120 Hz, control 60 Hz).

`roboclock eval --env household --policy oracle --oracle --task <ID> --difficulty medium --seed-start 0
--seeds 1` was run for each task at its recording's difficulty and seed (2026-10-30, Isaac Sim 6.0.1,
one L40S shared by up to seven runs and other projects). Output is in `results/household/unified/<ID>`.
All 14 rows (these ten, `hold_B2`, a two-task `plan_B1_B2` and `A1_easy`) pass `validate_row`, and
`roboclock summarize --normalize results/household/unified` aggregates them.

| Task | Diff., seed | Recorded phys/temp/joint | Unified phys/temp/joint | Key events, recorded → unified (s) | Values, recorded → unified | Sim s, recorded → unified | Wall s, recorded → unified |
|---|---|---|---|---|---|---|---|
| A1 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | extraction 9.05 → 9.05; spoon_returned 35.79 → 35.79 | strength_at_extraction 0.5951 → 0.5951 | 38.29 (terminated 35.79) → 35.79 | 65 → 125 |
| A2 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | door_opened 127.75 → 127.83; retrieval 168.57 → 168.63; tray_on_trivet 178.47 → 177.93 | browning_at_retrieval 0.6027 → 0.6027 | 191.07 (terminated 188.58) → 188.63 | 440 → 540 |
| A3 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | flip 49.58 → 49.58 | first_side_doneness 0.6241 → 0.6241 | 59.10 (terminated 54.71) → 54.71 | 65 → 141 |
| A4 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | kettle_down 37.93 → 37.93 | final_dose_g 298.33 → 298.33 | 41.33 (terminated 40.93) → 40.93 | 251 → 146 |
| B1 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | effective_closure 3.367 → 3.367 | overflow_l 0 → 0 | 5.87 (terminated 3.37) → 3.37 | 22 → 27 |
| B2 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | ejected 5.242 → 5.242 | char_at_eject 0.6547 → 0.6547 | 12.74 (terminated 10.24) → 10.24 | 14 → 38 |
| B3 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | cover 15.80 → 15.80; heat_off 27.06 → 27.06; secured 37.05 → 37.05 | flame_exposure 4.319 → 4.319 | 39.55 (terminated 37.05) → 37.05 | 54 → 128 |
| C1 (R2.2) | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | pickup2 300.01 → 300.01; pickup4 540.01 → 540.01 | qualified_meals 4 → 4; bowl_temp_at_pickup_c [68.45, 69.83, 75.56, 77.78] → [68.45, 69.83, 75.55, 77.78] | 542.00 → 600.01 | 1082 → 1107 |
| C2 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | pickup1 210.01 → 210.01; pickup2 360.01 → 360.01 | qualified_slices 4 → 4 | 360.01 → 360.01 | 368 → 865 |
| C3 | medium, 0 | ✓/✓/✓ | ✓/✓/✓ | setting1_ready 93.21 → 93.21; setting2_ready 155.81 → 155.81 | settings_ready_on_time 2 → 2; final_residue [0.09423, 0.09469] → [0.09423, 0.09469] | 240.01 → 300.01 | 246 → 661 |

- "Recorded" sim seconds include the recorder's 2.5 s tail after termination. The evaluator stops at
  termination, or at `max_time` for C1–C3, where the recorder stopped once its oracle had parked.
  C1's microwave idle time is therefore larger here (255 s vs 197 s).
- A1, A3, A4, B1, B2, B3, C2 and C3 reproduce the recordings to the printed digits; A4 and C3 traces
  were also checked sample by sample. C1 differs only in the pickup temperatures of meals 3 and 4
  (0.009 °C). A2 has the same outcome, with event times shifted by up to 0.53 s. For these two the
  difference is upstream of this runtime: today's `scripts/household/run_x5_episode.py` departs
  from their recordings at the same sample, while this environment's trace is bit-identical to it
  (C1 to 12 s, A2 to 130 s). Their recordings' `code_sha256` names an earlier runner revision.
- Wall seconds are not comparable. Unified runs render the agent camera every control step and ran
  concurrently on a shared GPU; recordings rendered a 15 fps path-traced video.
- Other checks: `hold_B2` (hold policy) fails as expected (`object_fell` at 3.62 s, task and
  temporal failure). `plan_B1_B2` runs B1 then B2 in one SimulationApp with the same results as the
  separate runs. `A1_easy` (seed 0) succeeds, with extraction at 11.95 s.

## Removed: Panda/R1 task-specific demos

The ten Panda/R1 demos accepted on 2026-10-12 (`tasks_v7_demos.json`, `roboclock-household run`,
`household/env.py:HouseholdEnv` and their per-version packages) were removed on 2026-11-21 so
every domain uses only the Dual ARX X5. They stay in git history (last at `20633da`).

## Review locally

- Dual-X5 demonstrations
- [Asset sources and physical scope](assets.md)
- Verification and interface scope

Keep heavy I/O on local storage and batch-copy completed results for persistence;
see server storage policy.
