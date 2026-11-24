# Dual-X5 household: task interface for the simulation backend

The ten current household tasks ([tasks.json](../../src/roboclock/envs/household/tasks.json)) implement the shared evaluator's task contract. The Isaac Sim backend is `roboclock.envs.household.x5.env:HouseholdX5Env` (`roboclock eval --env household`, see [README](README.md)): `x5/episode.py` loads the scene, measures the signals below with each task's `<id>_x5/sim.py` probe and drives the visual cues.

## How a backend uses a task

```python
from roboclock.tasks.registry import make_task

task = make_task("B3", difficulty="medium", seed=0, environment="household")
instruction = task.gen_instruction()          # agent-facing text; no hidden parameters
while not task.terminated and task.time < task.spec.max_time:
    # ... step physics / controller, then measure the declared physical facts:
    task.update({"lid_seated": lid_seated, "knob_angle_deg": knob_deg, ...})
    task.advance(dt)                          # process evolves; never paused for agent thinking
    drive_appearance(task.cues())             # e.g. flame size, tea colour, scale reading
metrics = task.metrics()                      # task/manipulation/temporal/joint success + task metrics
```

- `task.spec` is a frozen dataclass (serialised by `roboclock plan`). Seed 0 is the nominal contract. Other seeds draw the hidden parameters listed below.
- `difficulty` sets `spec.time_scale`: easy 0.75, medium 1.0, hard 1.5. It multiplies process rates and divides deadlines. This is a proposed difficulty knob, kept separate from geometry. C1–C3 are the exception (`time_scale` 1; see [x5-tasks.md § 17](x5-tasks.md)): C2 and C3 run the same episode at every difficulty, and the difficulty only selects the time tier that fills `temporal_success`; C1's difficulty selects its pickup schedule, 4 / 2 / 1 × the hard one.
- `update()` rejects undeclared signal names. Signals must come from measured physical state (contacts, joint positions, poses), never from controller phase labels.
- `cues()` returns only appearance-driving values. `metrics()` and `spec` are privileged and must not be shown to an evaluated agent.
- Throughput tasks (C1–C3) follow the C-class rule ([docs/c-class-timing.md](../c-class-timing.md)): they end early once every job is done and settled (C2, C3 also on the stall stop, C2 on a failure) and write `termination_reason` (`all_done`, `stall`, `failure`, `horizon`) to their metrics.
- `tests/envs/household/test_x5_tasks.py` drives every task with synthetic signal sequences: early, on time and late. Use it as an executable example.

## Scenes and static overrides

- **Build:** the environment builds each initial scene in code at reset (`x5/scenes.py:build_initial_scene`, which runs `x5/build.py`). `ROBOCLOCK_X5_INITIAL=<dir> python -m roboclock.envs.household.x5.build` (OpenUSD Python) writes `<dir>/<ID>/initial_scene.usda` and `initial_state.json`. Built three folders below a directory directly under the repository root (as the former `data/household/x5_initial/<ID>`), the file is byte-identical to the recorded `initial_scene_sha256` in `tasks.json`.
- **What `initial_state.json` lists:**
  - every object's source, scale and bounds;
  - every native joint, which the static stage deactivates and the backend must re-enable: toaster lever, ice dispenser paddle, microwave door and knob, sink handle and drain, cooktop knobs;
  - the key operation points with their suggested arm and approach;
  - the cameras: overview `/World/Camera` and the rig's head camera `/World/AgentCamera`;
  - the robot's ready pose.
- **Static overrides to undo or replace in simulation:**
  - A2: the dispenser paddle (native Refrigerator001_Switch) is referenced as a separate part on its hinge;
  - flame, smoke, foam and water are static cards whose appearance should follow `cues()`.
- **Robot:** Dual ARX X5 on the common rig (`x5/rig.py`): bases at (0.00, −0.255) and (0.60, −0.255), 0.60 m apart, facing +y; head camera `/World/AgentCamera` at (0.30, −0.215, 1.4104), 0.04 m ahead of the base midpoint and 0.54 m above the mount tops, pitched 55° down (all ten tasks; the earlier bases were (−0.06, −0.255) / (0.66, −0.255)). Ready pose `q = [0, 0.94, 1.34, -1.40, 0, 0]`. `scripts/household/check_x5_reach.py` screens all operation points (IK only).

## Signals, cues and hidden parameters per task

### A1 — Sweeten a glass of tea with sugar from a jar and stir it

Module `roboclock.envs.household.a1_x5.task` · category `temporal_window` · `max_time` 90 s

| Signal | Meaning | Initial |
|---|---|---|
| `jar_tilt_deg` | jar axis tilt from upright | `0.0` |
| `mouth_over_glass` | the stream from the jar's lowest mouth point falls inside the glass mouth | `False` |
| `jar_upright_released` | jar released, upright and supported on the counter | `True` |
| `spoon_in_cup` | spoon head inside the glass, under the tea surface | `False` |
| `stir_turns` | net clockwise turns of the spoon head about the glass axis | `0.0` |
| `spoon_in_basket_released` | spoon released, supported, low speed in its basket | `False` |
| `glass_displacement` | glass displacement from its initial pose (m) | `0.0` |
| `dropped` | jar, spoon or glass dropped / tipped | `False` |

Cues (`task.cues()`): `sugar_g`, `undissolved_g`, `jar_g`, `flow_g_s`, `in_air_g`. Band per tier: easy 10–30 g, medium 10–25 g, hard 10–15 g. Hidden per seed: `jar_sugar_g` ∈ [60, 85], `flow_coefficient` ∈ [0.85, 1.15].

### A2 — Fill a glass about half full of ice at the refrigerator-door style ice dispenser

Module `roboclock.envs.household.a2_x5.task` · category `temporal_window` · `max_time` 40 s

| Signal | Meaning | Initial |
|---|---|---|
| `paddle_deg` | measured dispenser paddle deflection, 0 = rest (switch at 5°, release at 3°) | `0.0` |
| `ice_level` | top of the highest resting cube in the glass over the glass's inner height (0 empty, 1 at the rim) | `0.0` |
| `cubes_in_glass` | cubes inside the glass (centre within the inner wall, below the rim) | `0` |
| `cubes_outside` | dispensed cubes that came to rest outside the glass | `0` |
| `cubes_moving` | any dispensed cube still moving faster than 2 cm/s (displacement over 0.2 s) | `False` |
| `glass_tilt_deg` | glass tilt from upright | `0.0` |
| `glass_on_counter_released` | glass upright on the counter, no robot contact | `True` |
| `glass_dropped` | glass fell (below the counter or on its side) | `False` |

Cues (`task.cues()`): `cubes_dispensed` (deliveries into the chute; the simulation drops a rigid cube for each), `auger_running`, `ice_level`, `cubes_in_glass`. Target band 0.40–0.60. Hidden per seed: `spin_up_s` ∈ [0.6, 1.3], `interval_s` ∈ [1.4, 1.7] (nominal 1.5), per-cube jitter ±5 % (none at seed 0). The cube cadence runs ×0.75 / ×1 / ×1.15 at easy / medium / hard (spin-up and coast ×1.5 at hard), so the release window is ≥ 1.0 s at every tier (2026-11-17).

### A3 — Flip one pancake at the right moment

Module `roboclock.envs.household.a3_x5.task` · category `temporal_window` · `max_time` 150 s

| Signal | Meaning | Initial |
|---|---|---|
| `pancake_on_pan` | pancake underside in contact with the pan floor | `True` |
| `pancake_rotation_deg` | accumulated rotation about a horizontal axis since t = 0 | `0.0` |
| `pancake_in_pan_released` | pancake fully inside the pan, not touched, settled | `True` |
| `turner_clear` | turner withdrawn from the pan area | `True` |
| `dropped` | pancake left the pan | `False` |
| `pan_moved` | pan moved off the burner | `False` |

Cues (`task.cues()`): `bottom_doneness`, `top_bubble_state`. Hidden per seed: `tau` ∈ [50, 80], `t0` ∈ [15, 25].

### A4 — Pour 300 g of hot water and stop in time

Module `roboclock.envs.household.a4_x5.task` · category `temporal_window` · `max_time` 75 s

| Signal | Meaning | Initial |
|---|---|---|
| `kettle_tilt_deg` | kettle tilt from upright toward the spout | `0.0` |
| `spout_over_jug` | stream lands inside the jug mouth | `False` |
| `kettle_upright_released` | kettle released, upright and supported on the counter | `True` |
| `jug_on_scale` | jug still supported by the scale | `True` |
| `dropped` | kettle or jug dropped/tipped | `False` |

Cues (`task.cues()`): `scale_reading_g`, `stream_ml_s`. Hidden per seed: `kettle_ml` ∈ [700, 1100], `flow_coefficient` ∈ [0.85, 1.15].

### B1 — Shut off a running faucet before the bowl overflows

Module `roboclock.envs.household.b1_x5.task` · category `event_reactive` · `max_time` 60 s

| Signal | Meaning | Initial |
|---|---|---|
| `lever_opening` | measured faucet valve opening, 1 = fully open | `1.0` |
| `hand_on_lever` | any robot contact with the faucet lever | `False` |

Cues (`task.cues()`): `water_level_cm`, `water_radius_cm`, `water_below_rim_cm`, `stream_l_s`, `overflowing`, `spilled_l`. Hidden per seed: `headroom_l` ∈ [0.8, 1.4].

### B2 — Catch the object rolling off the counter

Module `roboclock.envs.household.b2_x5.task` · category `event_reactive` · `max_time` 12 s

| Signal | Meaning | Initial |
|---|---|---|
| `released` | the peg holding the target has retracted | `False` |
| `object_speed` | target speed (m/s) | `0.0` |
| `edge_distance` | target centre to the counter edge along its heading (m), negative past the edge | `1.0` |
| `object_fell` | target below the counter top | `False` |
| `object_held` | both jaws on the target and the target clear of the counter | `False` |
| `peak_contact_n` | largest jaw contact force on the target so far (N) | `0.0` |
| `others_moved_m` | largest displacement of the objects that were not let go (m) | `0.0` |

No process cues. Hidden per seed: `release_time` ∈ [1, 3] s and `target`, one of cup, egg, pill bottle, apple. Difficulty sets the board slope: 4.5°, 6°, 9°.

### B3 — Turn off the burner under a pot of burning oil

Module `roboclock.envs.household.b3_x5.task` · category `event_reactive` · `max_time` 60 s

| Signal | Meaning | Initial |
|---|---|---|
| `knob_angle_deg` | center burner knob angle, 0 = OFF | `90.0` |
| `hand_on_knob` | robot contact with the burner knob | `False` |
| `water_contact` | water entered the pot | `False` |
| `pot_displacement` | pot displacement from the burner (m) | `0.0` |
| `wrong_knob_turned` | another burner knob was turned | `False` |

Cues (`task.cues()`): `flame_intensity`. Hidden per seed: `growth` ∈ [0.05, 0.08] 1/s, `f0` ∈ [0.12, 0.18].

### C1 — Serve the ordered meals at the pass

Module `roboclock.envs.household.c1_x5.task` · category `throughput` (stream) · `max_time` 1.25 × the last pickup: easy 247.05 s, medium 123.525 s, hard 61.7625 s. R3 (2026-11-06) replaced the 600 s microwave relay; see [x5-tasks.md § 16](x5-tasks.md) and § 17 for the judging.

Pickups (`pickups_s`, wall seconds; `time_scale` 1): four orders at every difficulty, at m × `HARD_PICKUPS_S` with m = 4 / 2 / 1 (`throughput.MULTIPLIER`). Hard 10.53 / 23.49 / 36.45 / 49.41 s (a first pickup at 10.53 s, then every 12.96 s: the tightest such schedule the slowest oracle meets, from oracle runs on 14 seeds; § 17), medium 21.06 / 46.98 / 72.9 / 98.82 s, easy 42.12 / 93.96 / 145.8 / 197.64 s. Until 2026-11-09 the time scale divided process-second pickups and hard took three orders. At each pickup the waiter takes the bowl resting released and upright on the board; the order qualifies if it is the meal wanted at that pickup. A pickup without its meal leaves the order open: the waiter takes that meal once it has rested on the board for 1 s. Physical success: every ordered meal served by the end, late or not; temporal: each by its pickup. The instruction names the wanted meals in order and the pickup times.

| Signal | Meaning | Initial |
|---|---|---|
| `bowl_location` | per-bowl location, one of counter/held/mat/board/taken/dropped | `('mat', 'counter', 'counter', 'counter')` |
| `board_bowl` | index of the bowl resting released and upright on the board; -1 when none | `-1` |

No cues. Per seed (`task.layout`): the wanted order (a permutation of curry, rice, soup, pasta; the first wanted meal starts on the mat), which waiting meal stands in which counter slot, and ±1.2 cm / ±8° jitter of the counter slots. Seed 0 is curry, rice, soup, pasta in the nominal slots.

### C2 — Pour cola into six glasses with two arms

Module `roboclock.envs.household.c2_x5.task` · category `throughput` (batch, Q = 6) · `max_time` H = 5 t_o = 216.45 s at every difficulty; the episode ends 2 s after all six glasses are served and both bottles stand released, on a failure, or on the stall stop. t_Q = `all_served` (all six served, bottles released for 1 s); tier budgets 173.2 / 86.6 / 43.3 s. The cola is a PhysX particle fluid; every volume below is counted from its particles.

| Signal | Meaning | Initial |
|---|---|---|
| `glass_ml` | per-glass cola volume, counted from the fluid particles inside it | `(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)` |
| `bottle_ml` | per-bottle cola volume left inside it | `(0.0, 0.0)` |
| `spilled_ml` | cola on or below the counter outside the glasses | `0.0` |
| `air_ml` | cola in the air (streams and splashes) | `0.0` |
| `bottle_released_upright` | per-bottle: released, upright and standing on the counter | `(True, True)` |
| `bottle_dropped` | per-bottle: dropped or tipped over | `(False, False)` |
| `glass_upright` | per-glass: upright and in its place | `(True, True, True, True, True, True)` |

Cues (`task.cues()`): none; the rendered fluid is the cue. No hidden process parameters.

### C3 — Cutlery organization: sort the utensils into the drawer

Module `roboclock.envs.household.c3_x5.task` · category `throughput` (batch, Q = 6) · `max_time` H = 5 t_o = 127.8 s at every difficulty (t_o = 25.56 s); the episode ends 2 s after all six loose utensils (two of each kind) are sorted, or on the stall stop. t_Q = `all_sorted`, the first time all six are sorted at once; tier budgets 102.2 / 51.1 / 25.6 s.

| Signal | Meaning | Initial |
|---|---|---|
| `correct` | loose utensils resting, released, in the compartment of their kind (0-6) | `0` |
| `wrong` | loose utensils resting, released, in another compartment (0-6) | `0` |
| `placed` | names of the loose utensils currently counted as correct | `()` |
| `dropped` | loose utensils that fell off the counter (0-6) | `0` |
| `references_moved` | a reference utensil left its compartment | `False` |

Cues (`task.cues()`): none (the utensils themselves). Hidden per seed: the table layout and the compartment order (`c3_x5/scene.py:layout`).

