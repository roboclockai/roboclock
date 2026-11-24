# Sports / Games

> **Renumbered 2026-11-19:** taiko (was A2) is now **B5**, an event-reactive task, and basketball
> (was A3) is now **A2**. The task table and commands below use the new IDs; dated calibration and
> result tables keep the IDs of their time. The sub-packages and the recording driver
> (`python -m roboclock.envs.sports --task ...`) keep their own kinds (arcade A2 = taiko, arena
> A3 = basketball).

Ten fixed Dual ARX X5 tasks on one common rig, with one current recording per
task in data/sports/videos. Every task uses the
RoboDojo dual ARX X5 rig ([spec](robodojo-rig-spec.md)): both X5 bases on one
crossbeam, 0.60 m apart, at the same height and heading; the policy cameras are
the RoboDojo head camera (`rgb`: rig frame (0.04, 0, 0.543) m, 60° down, Gemini
345Lg 640x480) and one D435 wrist camera per arm (`left_wrist_rgb`,
`right_wrist_rgb`, 640x480, on `link6`). Physics runs at exactly 250 Hz, control
at 50 Hz, policy observations and rendering at 25 Hz. Each recording has the
1280x960 presentation overview and the three 640x480 rig-camera videos.
Gallery ·
Validation ·
[Source and assets](../src/roboclock/envs/sports/README.md)

## Run a task

All sports implementations are under
[`src/roboclock/envs/sports`](../src/roboclock/envs/sports/README.md);
easy/medium/hard and seeds are configuration arguments. Two entry points share
the same sub-packages, scenes, controllers and scoring.

**Benchmark evaluation** — the registered `sports` environment exposes the ten
tasks below through the shared evaluator and the shared 16-dim Dual-X5
action/observation contract (`left/joint1..8`, `right/joint1..8`; laboratory
observation keys):

```bash
roboclock tasks --env sports
roboclock plan --env sports --difficulty easy --seeds 1
OMNI_KIT_ACCEPT_EULA=YES roboclock eval --env sports --policy oracle --oracle \
  --task B4 --difficulty easy --seed-start 0 --seeds 1 --output results/sports/unified/B4
OMNI_KIT_ACCEPT_EULA=YES roboclock eval --env sports --policy hold \
  --task A2 --difficulty easy --seeds 1 --output results/sports/unified/A2_hold
roboclock summarize --normalize results/sports/unified
# Physics and oracle only: no camera, no render call, zero images.
OMNI_KIT_ACCEPT_EULA=YES roboclock eval --env sports --policy oracle --oracle \
  --task B4 --difficulty easy --seed-start 0 --seeds 1 \
  --config src/roboclock/envs/sports/configs/no_render.yaml --output results/sports/headless/B4
```

One process can evaluate several tasks (`--task A2,B5,B3,A1`): each reset closes
the previous scene and builds the next one in the same SimulationApp. Rows pass
`roboclock.evaluation.schema.validate_row`; scoring comes from each task's native
evaluator of measured physical state via `roboclock.envs.sports.results:normalize`,
the adapter also used for the recorded `result.json` files. See the
[package guide](../src/roboclock/envs/sports/README.md#unified-evaluation-runtime)
for the joint mapping, observation, rates and environment-side mechanisms
(e.g. A1's scripted BLACK opponent). The original numeric single-Franka tasks
31–40 (`--env sports_legacy`) were removed on 2026-11-21; every task runs on the Dual ARX X5.

**Recording driver** — `python -m roboclock.envs.sports` runs one task's original
demo loop and writes its native `result.json`, traces and optional videos. With
`--video` it writes the 1280x960 presentation overview and the three RoboDojo
rig cameras (`head.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, 640x480, same
frames and timestamps); without `--video` no camera is installed. `ffmpeg` must
be on `PATH`. Use a Python environment with the intended Isaac Sim version
installed:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src python -m roboclock.envs.sports   --task B1 --difficulty easy --seed 0 --video --output results/sports/B1
```

Difficulty and seed are arguments, not separate source copies. Use a fresh output
directory. The default robot is `dual_arx_x5`. Controllers are privileged scripted
feedback baselines, not trained visual agents. Read physical, temporal and joint
outcomes from the generated result; a rendered video alone does not prove success.

| ID | Task | Primary category | Implementation |
|---|---|---|---|
| A1 | Legal chess exchanges and clock press | Temporal window/deadline | `chess/` |
| A2 | Basketball release during the shooting window (A3 until 2026-11-19) | Temporal window | `arena/` |
| B1 | Hit valid moles, avoid colored decoys | Event-reactive | `arcade/` |
| B2 | Boxing combinations after cues | Event-reactive | `arena/` |
| B3 | Catch and rerack after bench-press fatigue | Event-reactive | `bench_press/`, `_shared/precision/` |
| B4 | Push puck on green, stop and hold on red | Event-reactive | `freeze/` |
| B5 | Two-stick taiko: hit each drum as its note is cued (A2 until 2026-11-19) | Event-reactive | `arcade/` |
| C1 | Rack 15 pool balls into a legal triangle | Throughput | `pool_rack/` |
| C2 | Arrange cups and ring the bell | Throughput | `arena/` |
| C3 | Feed a fixed ball-service station | Throughput | `ball_service/` |

C1 uses a normal 7-foot table and never requires grasping the cue ball. B4 and C3
use fixed arms. B3 waits in standby until fatigue, then catches, raises and
reracks the bar, releasing it onto stable hook support.

### C tasks: judging since 2026-11-09 (score_version 2)

The C tasks follow the shared [C-class rule](c-class-timing.md); it replaces the
per-difficulty budgets and timing statements further down this page.

| Task | Jobs | Physical (end of episode, no time terms) | Temporal | Episode end |
|---|---|---|---|---|
| C1 | 15 balls; job time = acceptance in a slot; t_Q = rack handed over (legal, 15 accepted, hands > 0.12 m away for 2 s) | all 15 accepted, rack legal at the end, no failure | per tier: t_Q <= 4 / 2 / 1 x 136.63 s | rack handed over (all_done), stall stop, or 683.15 s |
| C2 | 2 cards; job time = valid bell press; time counts from the 2 s preparation | both cards, no early or incorrect bell, no dropped cup, no arm collision | per tier: 2nd card - 2 s <= 4 / 2 / 1 x 31.41 s | 2 s after the 2nd card (all_done), stall stop, or 2 + 157.05 s |
| C3 | 6 demands; deadline = scheduled firing time; job time = stocking time of the ball fired for it | all 6 demands served through the outlet, no drop, arm collision, occupied cell or overflow | every demand stocked by its deadline m x (6.44 + 3.57 k) s, k = 0-5, m = 4 / 2 / 1 (stream judge, one run per difficulty) | 2 s after the 6th ball passes the outlet (all_done) or 1.25 x the last demand (121.45 / 60.73 / 30.36 s) |

**Arm collision** (all three): in some control step an arm link (not the gripper) presses on
the table, a scene object or the other arm with more than 20 N (`c_class.ARM_COLLISION_N`). A
graze of a few newtons is harmless on a real arm and is only reported (`arm_contact_peak`);
whatever it moves or knocks over is judged by the task's own checks. Before 2026-11-09 the three
tasks used 3 N for 40 ms, 2 N and 8 N; in policy runs grazes stayed at or below 4 N and
collisions measured 96 N and more, while the oracle never touches anything with an arm link.

C1 and C2 run one episode per seed whatever the difficulty and are scored at all
three tiers (`temporal_success_by_tier`); the difficulty only picks the tier for
`temporal_success`. Their instructions ask to finish "as quickly as possible" and
state no seconds, and the C1/C2 screens show counts, not countdowns. C3 runs one
episode per difficulty on its own demand schedule, every demand time m x its hard time.
The hard schedule (first demand 6.44 s, then every 3.57 s) is the tightest the slowest
of 14 oracle runs meets (seeds 0-6, 50-56, measured with `configs/c3_eager.yaml`; source in
`ball_service/config.py`); it replaced 18 + U(0, 2) s then 9 / 7 / 5 s. The schedule has no
seed jitter, and the 4-cell hopper stays: at hard the oracle is just in time and never
banks a ball, at medium and easy up to 4 balls can be banked, so at least 2 must still be
stocked after service starts. A demand that finds no ball is served late by the next ball
stocked instead of being lost. The stall stop ends a batch
episode once 2 t_o have passed with no new job for `throughput.stall_window(t_o, Q)` (C1: 0.5 t_o; C2 with 2 cards: 1.5 T_easy / 2 = 3 t_o). Speed-cup stages count the
cups of both cards (12).

## Runtime, assets and evidence

The current recordings (2026-11-03) all use **Isaac Sim 6.0.1** at 250 Hz physics /
50 Hz control. Exact package versions are in
`data/sports/videos/evidence/runtime/`. The superseded 2026-10-31 collection
(`data/sports/videos/superseded/2026-10-31/`) used Isaac Sim 6.2 for A1 and
6.0.1 for the other nine at 240 Hz physics; A1 then had two incomplete 6.0.1
attempts, disclosed in the validation guide. The current A1 recording completes
under 6.0.1 once the driver enables the manipulator extensions and applies the X5
finger speed limit, as the evaluation runtime does.
Follow the repository installation instructions without overriding another
domain's dependency configuration.

Task assets and license records live beside their owning source modules.
Robot assets describes the manufacturer model and main's
packaged asset location. Isaac Sim itself is an external dependency.

All ten selected easy/seed-0 recordings pass their recorded checks. The preserved
90-case diagnostic matrix has **66/90** joint
successes; it predates the presentation update and the rig migration. Failed seeds, tighter
timing failures and four controller-fault tests remain published. The cleanup
does not alter task code, scoring, deadlines or recorded outcomes.

### Difficulty pass (2026-11-04)

Every task now passes easy, medium and hard on seeds 0 and 1 in headless
`roboclock eval`, with at least 10 % margin on its timing metric
(table). The
oracles were made faster first; difficulty settings changed only where the X5
could not fit them:

| Task | Oracle change (old → new) | Difficulty change (old → new) |
|---|---|---|
| A1 | both players' controller: 0.22 → 0.34 m/s Cartesian, pawn lowering 0.16 → 0.22 m/s, brake from 35 → 25 mm, joints 1.5 → 2.4 rad/s; worst clock latency 10.24 → 7.76 s | none (hard 10 s) |
| B1 | mallets 0.23 → 0.60 m/s, joints 1.7 → 3.5 rad/s, wrist swing 2.3 → 5 rad/s, idle over the board, swing starts within 35 mm of the aim; worst hit 1.25 → 0.98 s | hard mole time-up 0.7 → 1.10 s (medium 1.4, easy 2.8 unchanged): 0.59–0.98 s is the X5's reach-and-strike time |
| B3 | approach 0.85 → 1.6 m/s, joints 3.4 → 5 rad/s, jaws pre-closed to 26 mm on the way down; catch 0.640 → 0.576 s (seed 0), 0.412 s (seed 1) | catch deadline 0.65 / 0.40 / 0.25 → 0.80 / 0.72 / 0.64 s: from the fixed standby the drives cannot reach and hold the bar faster (9 rad/s and 2.5 m/s targets gave the identical 0.576 s) |
| B4 | lift-off from the puck on red 12 mm back / 8 mm up at 0.14 m/s → 6 / 6 mm at 0.40 m/s; worst stop 0.204 → 0.180 s | none (hard 0.2 s) |
| C1 | none (faster travel, lift, settle/release and an earlier rack hand-over knocked placed balls out of their slots in every trial) | horizon 180 / 140 / 110 → 180 / 155 / 140 s; the oracle racks all 15 in 124.1 s |
| C2 | cups 0.18 → 0.36 m/s, descent 0.10 → 0.25 m/s, lowering 0.06 → 0.18 m/s with a 0.05 m/s brake over the last 25 mm, release 0.45 → 0.25 s, joints 1.7 → 3.0 rad/s; two cards 75.0 → 31.4 s | none (hard 40 s) |

### B1 waiting pose (2026-11-15)

An oracle holds its start pose until the cue is observable. The B1 oracle waited over the
board ([0.50, ±0.12, 1.07] m) between moles, a pose the instruction does not ask for. It now
waits where the pickup ended, lifted above each mallet's rest (`arcade/dual_moles.py`; the
`mole_idle_target` setting names another pose). From there a hit lands 1.16–1.76 s after the
rise (seeds 0–2, 18 moles): easy (2.8 s) passes; medium (1.4 s) and hard (1.10 s) failed with
the time-ups of that day; the recalibration below sets them from this oracle.

### Time-limit recalibration (2026-11-16)

Every tiered A/B limit that is a free parameter measured from a cue was re-set against the
scripted privileged oracle, a **reference controller** (not an optimal time), run on seeds
0–9 at every tier with zero inference latency and zero control delay. With t the slowest
oracle time of the quantity the limit is compared against: hard = max(t / 0.8, t + 0.3 s),
rounded up at the limit's grain; easy unchanged unless the new hard reaches it; medium =
sqrt(easy × hard), rounded. The oracle's timing is the same at every tier for these tasks,
so t is the maximum over all tiers (B1: easy episodes, the only tier the old limits let it
finish; the mole schedule, a rise every 4 s, does not depend on the tier).

| Task | Compared quantity | t (s) | Old e / m / h | New e / m / h |
|---|---|---:|---|---|
| A1 | BLACK's press → WHITE's press | 7.90 | 30 / 18 / 10 s | 30 / 17 / 10 s |
| A3 | READY → release | 4.59 | 14 / 10 / 5 s | 14 / 9 / 6 s |
| B1 | mole rise → hit | 1.756 | 2.8 / 1.4 / 1.1 s | 2.8 / 2.5 / 2.2 s |
| B2 | GO cue → hit | 0.680 | 1.6 / 1.15 / 0.8 s | 1.6 / 1.25 / 1.0 s |
| B2 | hit → glove back in guard | 0.716 | 1.4 / 1.1 / 0.8 s | 1.4 / 1.2 / 1.05 s |
| B3 | bar descent → bilateral catch | 0.576 | 0.80 / 0.72 / 0.64 s | 1.10 / 0.98 / 0.88 s |
| B4 | RED → confirmed rest | 0.180 | 0.8 / 0.4 / 0.2 s | 0.8 / 0.6 / 0.5 s |

B3's new hard (0.88 s) exceeded the old easy (0.80 s), so easy became 1.10 s (hard × 1.25, the
old easy/hard ratio). A2's ±0.5 / 0.25 / 0.12 s note window is a symmetric tolerance and was
unchanged: the oracle strikes every note 0.104 s early (hard margin 0.016 s). The 2026-11-04
difficulty-pass values above are superseded. The A2, B3 and B4 values of this table are
superseded by the one-second floor below.

### One-second floor (2026-11-17)

Every temporal limit a policy must meet is now at least 1.0 s at every tier: sub-second limits
approach human reaction time, so a near-zero score could not separate a weak policy from an
unreachable constraint. Three sports limits were below 1 s; medium = sqrt(easy × hard) except
B4, whose 8 / 4 / 2 s the user set.

| Task | Limit | Old e / m / h | New e / m / h | Schedule / physics change |
|---|---|---|---|---|
| A2 | note window (±, about each note) | 0.5 / 0.25 / 0.12 s | 2.0 / 1.41 / 1.0 s | notes every 2.5 → 5 s, so consecutive windows never overlap (1 s gap at easy); last note 29.5 → 47 s, native max_time 34 → 53 s |
| B3 | bar descent → bilateral catch | 1.10 / 0.98 / 0.88 s | 2.0 / 1.41 / 1.0 s | failed rep lowered slowly (below) |
| B4 | RED → confirmed rest | 0.8 / 0.6 / 0.5 s | 8 / 4 / 2 s | each red = limit + seeded 2.2–3.2 s hold; greens 4–6 → 7–10 s; horizon 62 s → 48 s of green (101–111 / 81–87 / 71–75 s) |

**A2.** Paired (chord) notes share one timestamp and each is judged against its own window;
there is no separate simultaneity tolerance. Hits still map to the nearest open note of their
drum, and a strike outside every window is a fault.

**B3.** With the old exhausted-lifter model (support decayed to 0.97 × weight, a 0.29 m/s²
fall) the bar reached the 0.76 m danger height 1.0–1.3 s after the descent was detected, so a
2 s deadline would have been decided by the safety failure, not the clock. The exhausted
lifter now lowers the bar under control: its support never exceeds the bar's weight, and its
target sinks at 5 cm/s after a 0.25 s ramp (`bench_press/process.py:eccentric_drop`,
`fatigue_descent_speed`, `fatigue_descent_ramp`). The descent is detected about 0.26 s after the
stall ends; at the easy deadline (2 s later) the bar is at ≥ 0.83 m (lowest stall height
0.933 m), and it reaches 0.76 m 3.2–5.3 s after detection (1-D model of the same wrench).

**B4.** A red now lasts the response limit plus the seed's hold, so a stop at the very deadline
is still followed by a 2.2–3.2 s hold (the whole red of the old design). Holds and greens come
from the seed's timing stream and are the same at every tier; only the reds stretch. The
horizon ends the green in which 48 s of green have elapsed (the oracle needs 35.4–35.7 s of
green for its two goals); three reactions are still required and always fall in the first
30 s of green. Revision `fixed-freeze-v3`.

Oracle check (headless `roboclock eval --policy oracle --oracle`, `configs/no_render.yaml`, seeds
0–9 at every tier, Isaac Sim 6.0.1, 2026-11-17): all 90 episodes pass jointly.

| Task | Easy / medium / hard | Slowest oracle time | Margin at hard |
|---|---|---|---|
| A2 | 10/10, 10/10, 10/10 | every one of the 110 hits per tier 0.104 s early | 0.896 s |
| B3 | 10/10, 10/10, 10/10 | catch 0.380–0.512 s after descent; lowest bar 0.859 m | 0.488 s |
| B4 | 10/10, 10/10, 10/10 | stop 0.076–0.136 s after RED; 3–4 reactions; ends 67–79 / 55–63 / 49–55 s | 1.868 s |

The former single-Franka task configuration (`common/configs/benchmark.yaml`, e.g.
task 37's `stop_deadline` 0.7 / 0.4 / 0.2 s) and the unused mobile-freeze / lift settings of
`_shared/workflow/config.py` (`stop_response`, `lower_response`) keep their sub-second values:
no current A1–C3 task reads them.

### RoboDojo rig migration (2026-11-04)

Every task now uses the RoboDojo rig cameras and renderer and runs physics at
exactly 250 Hz. Headless `roboclock eval` at easy / medium / hard (seed 0), rendered
eval at easy (one episode per process) and the recordings all ran from the same
code; headless and rendered rows are identical event for event and the
recordings reproduce the end times. All ten passed at easy; the medium/hard
failures of that check (A1 hard, B1 hard, B3 medium/hard, B4 hard, C1 hard, C2
medium/hard) are resolved by the difficulty pass above. Cue displays were turned toward the 60°
head camera and the C1 ball caddies moved into its view (C1 easy now ends at
124.04 s instead of 139.64 s; C1 hard racks 13/15 by its 110 s horizon, was
12/15). Details: validation,
[package guide](../src/roboclock/envs/sports/README.md#robodojo-rig-cameras-and-no-render-runs).

### Earlier rig migration check (2026-11-03, 40° head camera, superseded)

Every task was run three times at easy / seed 0 from commit `7c0658d`, Isaac Sim
6.0.1, one process per episode: `roboclock eval --env sports --policy oracle
--oracle` with `configs/no_render.yaml` (physics and oracle only), the same
evaluation rendered (head camera every observation), and the recording driver
with `--video` (the published overview + `head.mp4`). All thirty runs pass
physically, temporally and jointly; no-render and rendered rows are identical
event for event, and the recordings reproduce their outcomes, end times and key
measurements (A1, C1 and B3 included). Full table, rows and the B2 screen
evidence: validation.

| Task | End (s) | Key measurement (identical in the three runs) |
|---|---:|---|
| A1 | 43.78 | clock presses 9.40, 19.38, 29.60, 39.84 s |
| A2 | 30.70 | 11/11 notes, 21.50/min |
| A3 | 10.42 | release 10.224 s, basket |
| B1 | 46.98 | 6/6 moles |
| B2 | 36.98 | 9/9 strikes, mean latency 0.671 s |
| B3 | 12.05 | catch latency 0.640 s (deadline 0.65 s) |
| B4 | 49.68 | 2 rounds, stops 0.06–0.20 s (limit 0.8 s) |
| C1 | 139.64 | 15/15 balls |
| C2 | 77.00 | 2 cards, 1.60/min |
| C3 | 66.36 | 6/6 served, 7.66/min |

Under Isaac Sim 6.0.1 the X5 fingers must be given the manufacturer speed limit
explicitly (see the [package guide](../src/roboclock/envs/sports/README.md#unified-evaluation-runtime));
both the evaluation runtime and, since the rig migration, the recording drivers
do so.
The earlier unified check (2026-10-29, presentation camera as the policy view)
is in Git history.

## Repository organization

- `src/roboclock/envs/sports/`: runnable code, task configs and assets.
- `data/sports/<task>/`: one oracle demonstration per task (`demo.mp4`, `demo.json`).
- The earlier showcase videos (`data/sports/videos/`) and validation records
  (`data/sports/validation/`) were removed on 2026-11-20; they are in git at
  a417e24.
- `scripts/sports/`: current preparation, validation and audit tools.
- `results/sports/`: ignored local run output.

Discarded designs and full raw traces remain in Git history. The current
recordings were made from commit `7c0658d` (`recording_revision` in the
manifest). For exact reproduction of the superseded 2026-10-31 recordings, use a
separate checkout at `c57ae44434c0e634bf2baeb671e1357b9d708d1f`, its documented
source snapshot and recorded runtime. Do not extract historical source archives over current main.
`--env sports` and the commands above select the current ten tasks (the numeric single-Franka
tasks 31–40 were removed on 2026-11-21).
