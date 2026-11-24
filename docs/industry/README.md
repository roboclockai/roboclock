# Industry: Dual ARX X5 workcells (A1–C4, recorded as tasks 41–50)

Ten fixed-workcell industrial tasks for the **Dual ARX X5/X5A** (two complete
original articulations; a task may use one or both hands). Task IDs follow the
benchmark's class numbering; the original numbers are accepted everywhere a task
ID is (`roboclock ... --task 45` is `B2`).

| ID | No. | Category | Task | Physics |
|---|---|---|---|---|
| A1 | 41 | temporal window | Zincate dip and rinse of an aluminium part: pick it from the loading pad, immerse it, whole-part exit 20–40 s after immersion, release in the rinse bath | 250 Hz |
| A2 | 42 | temporal window | Pump housing into the carrier on the rotary accumulation table, inside the 60° loading sector | 250 Hz |
| A3 | 43 | temporal window | Cover seated 2–14 s after the hot-melt adhesive bead completes | 250 Hz |
| B1 | 44 | event reactive | Robot as the coolant line's emergency shut-off: isolation valve closed within 5 s of leak onset | 250 Hz |
| B2 | 45 | event reactive | Stop button actuated within 3 s of the fault warning | 250 Hz |
| B3 | 46 | event reactive | Rolling bearing arrested within 3 s of zone entry, then recovered | 250 Hz |
| C1 | 47 | throughput | Two complete maintenance kits, as quickly as possible | 250 Hz |
| C2 | 48 | throughput | Three mixed-SKU cartons onto matching shelves, as quickly as possible | 250 Hz |
| C3 | 49 | throughput | Three parts weighed and sorted by weight, as quickly as possible | 250 Hz |
| C4 | 50 | throughput | Two four-color eyeshadow palettes assembled and delivered, as quickly as possible | 250 Hz |

**What each timed task is described as** (2026-11-16; text and labels only, no timing changed):

* **A1** is a zincate immersion dip on an aluminium part followed by a rinse: the
  pre-treatment that prepares aluminium for plating. It needs no current, like the scene's
  plain dip. Its duration is real: "20–120 sec." for powder and liquid zincates in S. Rudy,
  "Aluminum Finishing: the Zincating Facts", Finishing and Coating, 2024
  (<https://finishingandcoating.com/index.php/plating/1857-aluminum-finishing-the-zincating-facts>),
  and "a period of 20-50 seconds followed by a thorough cold water rinse" for the first dip of
  a double-zincate cycle in US 5,182,006 (<https://patents.google.com/patent/US5182006A/en>).
  A chromate conversion dip on zinc plate is of the same order ("5 to 30 seconds", EPI
  E-Chrome 851-L data sheet). So the 20 s minimum immersion is a real duration. The window
  widths are not taken from these sources, and the scene simulates no chemistry. The bath
  sign reads ZINCATE DIP (it read PLATING, then NICKEL STRIKE); the part's appearance is
  unchanged. As nickel plating, the first description, the step would be about 60 times
  compressed. Code and data keep the earlier name (`plating_to_rinse`, `plating_task.py`,
  `41_plating_to_rinse.mp4`), and the recorded videos show the earlier sign.
* **A2**'s turning table is a rotary accumulation table. It turns at 5°/s = 0.83 rpm. Real
  ones run at 1–3 rpm ("Fixed Speed: 1 rpm or 3 rpm", "Variable Speed: 1 through 3 rpm",
  Mini-Mover rotary table accumulators,
  <https://www.universalpowerconversion.com/12-16-inch-rta-c-595.html>; "Up to 13
  revolutions per minute" variable, Eastey TT,
  <https://eastey.com/products/case-product-handling/accumulation-table/>), so 1 rpm (6°/s)
  is the nearest catalogued speed and ours is 17 % below it. The speed is not changed: the
  oracle is tuned to 5°/s (below). Placing a part into a fixtured carrier on such a table is
  our addition; real ones are flat discs on which parts collect loose.
* **A3**'s bead is a hot-melt adhesive. Hot-melt data sheets give open times (the longest
  time between applying the adhesive and joining the parts) of about 15–40 s: "Open Time
  <15 Seconds" for a fast packaging grade (Infinity Bond E2190), "Open Time OSB 40 s" for a
  general assembly grade (3M 3764). The seating windows close 16.47 / 18.15 / 20 s (hard / medium / easy) after the bead
  finishes, which is on that scale. Data-sheet open time runs from application, ours from
  the end of the bead.
* **B1**: the robot acts as the emergency shut-off for the coolant line. The 6.28–8.4 s deadline
  is 1.05–1.4× the longest of the 0.5–6 s that data sheets print for the 90° stroke of small actuated
  quarter-turn valves (Georg Fischer EA15/EA25 "5 s", EA45 "6 s"; Omega pneumatic "approx
  1⁄2 to 1 second", electric "Cycle Time: 6 seconds"). Those are stroke times alone, while
  our deadline also contains detection and the reach, so a real actuated valve tripped by a
  leak sensor would finish sooner. As a manual operator response the deadline has no source
  (process-safety practice credits manual responses in minutes). It is not a containment
  limit either: the leak now runs at 0.14 L/s (it was 0.4 L/s), so the tray's 2.0 L limit is
  reached about 14 s after onset.
* **B2**: no re-description. For context, measured human emergency-stop button presses take
  about 0.2–0.9 s (K. Fukaya, RIIS-SRR-No.13, 1993, ch. 5: means of about 200–440 ms when
  only monitoring, up to 913 ± 214 ms with a secondary task; hand on or near the button,
  event expected,
  <https://www.jniosh.johas.go.jp/publication/doc/srr/SRR-No13-5.pdf>), so the 2–3 s
  deadline is looser than a human response. ISO 13850 sets no operator reaction time.

The recordings ran the approved "within 90 s"; the C tasks are now judged by
[C-class timing](../c-class-timing.md) (below).

October 29 revisions to Tasks 41 and 49 replaced the
two videos: Task 41 was recorded with its own path-tracing profile, Task 49 has the
outward left source and waiting lane. Since the RoboDojo-spec pass (2026-11-04) every task
renders with the spec's RaytracedLighting + DLAA (runtime profile `isaac601-robodojo`).

Instructions (`configs/tasks.json`) are given to the policy as text, never drawn
into RGB; images contain only equipment identifiers, safety marks and machine cues.
Recordings, sidecars and traces are in `data/industry/`.

## Common Dual ARX rig

Every task uses the RoboDojo-aligned rig of [`docs/robodojo-rig-spec.md`](../robodojo-rig-spec.md)
(`src/roboclock/envs/robodojo_rig.py`; `native/rig.py` only maps it onto the scenes and keeps
no copies of its values):

* both ARX bases at the same height and forward coordinate, facing world +Y, **0.60 m
  apart**;
* **head camera** (`rgb`): 0.04 m in front of the base midpoint, 0.543 m above the mount
  plane, **60° down**, Gemini 345Lg template, 640×480;
* **wrist cameras** (`left_wrist_rgb`, `right_wrist_rgb`): D435 template, 640×480, rigid
  on each arm's `link6`;
* **physics 250 Hz** in every task, RaytracedLighting + DLAA + Isaac Lab's quality
  preset;
* a recording-only third-person overview camera per task (1280×960,
  `native/overview.py`).

A task chooses where the rig stands; scenes moved where the arms could not reach or the
head camera could not see. Mounts are the ARX `base_link` origins (m). The key objects and
cues of every task are listed in `native/visibility.py` and checked by a projection test.
The [RoboDojo-spec pass](#robodojo-spec-pass-2026-11-04) moved cues and fixtures for the
spec's 60° head pitch; the table states the current placement.

| ID | Recorded left / right mount | Rig left / right mount | Layout change and reason |
|---|---|---|---|
| A1 41 | [-0.43, 0.08, 1.02] / [0, 0.08, 1.02] | [-0.60, 0.08, 1.02] / [0, 0.08, 1.02] | Right (working) arm kept; the mount brace spans the new bases. |
| A2 42 | [-0.39, 0.02, 1.06] / [0.24, -0.25, 1.06] | [-0.39, 0.02, 1.06] / [0.21, 0.02, 1.06] | Left (working) arm kept; idle right arm moved beside it. |
| A3 43 | [-0.35, 0.08, 1.04] / [0.37, 0.08, 1.04] | [-0.35, 0.08, 1.04] / [0.25, 0.08, 1.04] | Left (working) arm kept. |
| B1 44 | [-0.42, 0.05, 1.13] / [0, 0.14, 1.13] | [-0.45, 0.20, 1.13] / [0.15, 0.20, 1.13] | Right base 0.15 m right and 0.06 m toward the valve so the head camera sees the valve, the downstream flange/leak and the spill tray's far half. Centering the rig on the valve (right base x=0.30) left the closed handle out of reach. |
| B2 45 | [-0.96, -0.15, 0.92] / [-0.48, -0.15, 0.92] | [-1.08, -0.15, 0.92] / [-0.48, -0.15, 0.92] | Right (working) arm kept. The HMI box and sign with the red fault lamp moved from x=0.55 to x=-0.55, behind the stop button; at 0.55 the lamp was 55° off-axis, outside the head view. |
| B3 46 | [0.65, 0.59, 0.98] / [1.05, 0.59, 0.98] | [0.55, 0.56, 0.98] / [1.15, 0.56, 0.98] | Rig centered on the rework tray's x, 0.03 m back. The camera that followed the bearing is removed. The head camera sees the capture zone and lane, the whole rework tray (made 0.235 m deep and 0.0275 m further forward in the spec pass) and the bearing on it. Further back, the right arm cannot lift its downward brake vertically (joint6 at ±2 rad; y=0.50 failed at `brake_withdraw`). |
| C1 47 | [-0.64, 0.2, 1.1] / [0.64, 0.2, 1.1] | [-0.30, 0.2, 1.1] / [0.30, 0.2, 1.1] | Two 0.545 m kit cases do not fit between bases 0.60 m apart. Roles swap: the right arm packs kit 0 at x=+0.66 from sources at +0.22/+0.38/+0.54, the left arm kit 1 (mirror); each kit group moved 0.94 m, so each arm sees the recorded geometry of the arm it replaces. Bracket tool yaw follows the kit ({left: 180°, right: 0°}); bench widened to 2.0 m. The kit cases sit beside the bases; the head camera sees the sources and each insert's inner far corner. |
| C2 48 | [-0.38, 1.36, 0.36] / [0.62, 1.36, 0.36], yaw 0° / 180° | [-0.18, 0.70, 0.79] / [0.42, 0.70, 0.79] | Rebuilt around the centre-line clamp corridor: lift-table infeed and two-level A-01/B-01 shelf (below). |
| C3 49 | [-0.3, 0.32, 1.2] / [0.3, 0.32, 1.2] | unchanged (already the rig) | Per-arm PASS/REJECT receivers in the head view (below). |
| C4 50 | [-0.3, 0.22, 0.98] / [0.3, 0.22, 0.98] | unchanged (already the rig) | Output trays moved from behind the bases (±0.60, 0.015, floor 0.757 m) onto the bench in front of each nest (±0.50, 0.69, floor 0.897 m) in the spec pass, into the head view. |

**C2 on the rig (2026-11-03, second pass).** Both hands clamp a carton by its 0.28 m ends,
tools pointing at each other and pitched 30° down. The recorded grasp geometry, cartons,
masses, frictions, hold and placement windows and the 90 s budget are unchanged. One
oracle setting changed: the clamp-force regulation now targets 5–18 N for every carton. On the rig the clamp
pushes across each wrist; the recorded 15–25 N first-carton profile (it peaked at 45 N)
drove the left `joint5` (5 N·m cap) back by 0.15 rad. 1.6 N per hand already holds a
0.30 kg carton at μ 0.95. With both bases facing +Y
0.60 m apart, the wrist (`joint5` ±1.57 rad) cannot turn a tool back across its arm
plane, so the clamp only works within 0.03–0.09 m of the rig's centre line, 0.10–0.50 m
forward (URDF IK; `industry_tools/reach48_clamp.py`). The recorded pallet and the
full-width rack (A-01 and B-01 0.40 m apart, upper deck at 1.50 m) do not fit that
corridor, so the cell is rebuilt around it (`native/putaway_rig.py`):

* **Infeed:** a lift-table magazine 0.21 m in front of the bases holds the three cartons
  stacked in pick order. A top photo-eye raises the table at 0.04 m/s until the topmost
  carton still in the magazine reaches the presentation height (0.71 m), so every carton is
  picked at the same spot. This is a scene process, run for every policy; no product pose is written.
* **Shelf:** a two-level unit on the centre line, 0.31 m in front of the bases. A-01 is
  the upper deck at 0.775 m (first A at the back, second at the front), B-01 the lower deck
  at 0.45 m under it. The upper deck is 0.36 m wide, narrower than the 0.56 m wrist span,
  so the forearms pass beside it when placing B. SKU matching is by deck: an A on the
  lower deck or a B on the upper deck is not credited.
* The rig stands on 0.79 m stands at [-0.18, 0.70] / [0.42, 0.70]; the warehouse racks and
  inbound pallet stay as scenery. The empty return backs out of the shelf at the release
  height and waits for the magazine to present the next carton before lowering.

**C3 receivers on each arm's side (2026-11-03, second pass).** The recorded shared PASS
(y=0.41) and REJECT (y=0.12) receivers sat at or behind the head camera. Every position
the head camera sees at tray height (y ≳ 0.52) that clears the input tray lies beside one
arm and only that arm reaches it. So each arm now has its own PASS (inner, x=±0.45) and REJECT
(outer, x=±0.645) receiver at y=0.625 (0.18 × 0.21 m trays, y 0.52–0.73, all in view).
Each arm sorts into its own side with a 90° downward deposit yaw and no shared
deposit lock. A part counts in either side's receiver of its result
(`InspectionScene.contained`, `InspectionMonitor`). The inter-arm contact check, the
inspection rules and the 90 s budget are unchanged.

**C3 is a weighing task (2026-11-19).** At the user's request C3 dropped the camera
inspection: the D455 camera, its post and ring light, the three face views and the QC lamp
are gone (`native/weigh_sort_task.py` replaced `inspection_arx_task.py`; the task name is now
`weigh_and_sort`). Each part is set on a static bench scale between the arms (pan
0.12 × 0.10 m centred at (0, 0.47), top at the feed-rail height 0.928 m), released and read.
A part of 125 g or more goes to a PASS tray and a lighter part goes to a REJECT tray
(`resolved["sort_results"]`). The reading is the measured native contact normal force of the
parts on the pan divided by g (`native/weigh_scale.py`). The display is a 3-digit 7-segment
remote indicator with "MIN 125 g", on a post behind the input tray at (0, 0.965, 1.02),
tilted 50° toward the head camera. A readout on the scale itself was hidden by the wrist
whenever a gripper was above the pan. It has a scale's stability gate: every 0.2 s it shows
the 0.4 s mean once the load has stayed within 2 g, and otherwise it holds its last stable
value. Without the gate, placement and re-grasp spikes flashed on the display, and every
segment switch ghosts for a few frames under DLAA. A part is weighed once it rests alone,
released and at rest on the pan for 0.5 s with the display settled (`WeighTracker`, shared by
the scene and `WeighingMonitor`). Seed child 3 draws each part's result independently; a
REJECT draw is a short shot of 70 % mass (105 vs 150 g, both × `mass_scale`), so every seed
keeps its old PASS/REJECT split. The oracle (`WeighSortController`) moves every Cartesian
segment along a minimum-jerk profile and every joint transit along a minimum-jerk joint
interpolation. Between parts it makes one joint transit to the next nest (no fold/swing/park).
While the reading settles it lifts 5 cm and then re-grasps straight down. Both arms work at
once: only the scale is exclusive, and an arm claims it while the other arm's tool is outside
|x| ≤ 0.2 m and not weighing. Neighbouring nests (< 0.3 m apart) are not entered by both arms
at once. Seed 50 takes 64.8 s, with no inter-arm contact. The native budget is 120 s. t_o was
re-measured at 69.36 s (seeds 0–9 easy, all successful with no contacts, seed 4 slowest; it was 83.20 s).

### RoboDojo-spec pass (2026-11-04)

Industry now imports every rig value from `robodojo_rig` (merges of `rig/robodojo-spec`
924c344 and 3903057): the head camera went from the common rig's 45° pitch, 0.54 m height and
16 mm / 36 mm pinhole to the spec's 60°, 0.543 m and Gemini 345Lg template (640×480);
D435 wrist cameras were added on both `link6`; physics went to 250 Hz everywhere; the
renderer went from each task's profile (RealTimePathTracing for most, Task 41's own
path-tracing capture) to RaytracedLighting + DLAA. Changes per task:

| ID | Physics | Spec-pass layout change | Why |
|---|---|---|---|
| A1 41 | 250 Hz (unchanged) | none | Part (on the loading pad between the baths since 2026-11-19), zincate dip bath and recipe lamps are in the 60° view. The rinse bath's front-right rim is not (see below). |
| A2 42 | 250 Hz | none | Loading sector and table in view; the pump's left edge on the infeed cradle is at the image border. |
| A3 43 | 250 Hz | process beacon from a 1.63–1.70 m pole at (0.62, 0.78) to 0.97–1.04 m at (0.45, 0.82) | At 60° the old beacon was above the image. |
| B1 44 | 250 Hz | leak beacon from 1.67–1.74 m at (0.89, 0.97) to 1.01–1.08 m at (0.62, 0.75) | Same: above the image. |
| B2 45 | 250 Hz | HMI box and its fault lamp lowered onto the bench behind the stop button: lamp from 1.19 m at y 0.823 to 0.98 m at y 0.743 | The lamp was above the image. |
| B3 46 | 1000 → 250 Hz | Rework tray 0.29 → 0.235 m deep, centre y 0.65 → 0.6775 (inner floor y 0.54 → 0.595); a rim cradle in the tray (two 1.5 mm steel rails across x, inner edges ±8 mm around the release point (0.87, 0.672)); arrest/recovery lamp from a 1.43 m post at (1.50, 1.19) to 1.00 m at (1.35, 1.25); bearing and static contact offsets 2 mm, bearing depenetration ≤ 0.3 m/s | The tray's near edge and the released bearing were below the image; the lamp was above it. At 250 Hz with the default offsets the bearing jittered on the tray and the hand-off's retention window timed out (`Bearing handoff timeout withdraw`); the 2 mm offsets fix the hand-off. The released bearing stands on its rim, and on the flat floor it kept rocking and rolling (0.05–0.2 mm per control tick, 0.1–0.3 rad/s): the rendered eval credited a 0.5 s retention window at 37.49 s, which broke again before the episode end, so physical success failed. On the cradle the rim rests on both rail edges 0.5 mm above the floor (inside the 2 mm floor band) and moves < 0.02 mm over 5 s. |
| C1 47 | 500 → 250 Hz | none | Kit cases stay beside the bases (they cannot move inward, see the rig table). |
| C2 48 | 500 → 250 Hz | none | Centre-line layout kept; carton, A-01 and B-01 decks in view. |
| C3 49 | 250 Hz | QC lamp from 1.525 m to 1.20 m on the D455 post | The lamp was above the image. Per-arm receivers kept, seeds 0–8 rechecked. |
| C4 50 | 500 → 250 Hz | output trays onto the bench in front of the nests (rig table) | They were behind the bases. The loaded case now moves straight forward (`case_transfer_xy_m`). |

**Key objects in the head view.** `native/visibility.py` lists each task's key objects
and cues; `test_key_objects_project_inside_the_head_image` requires every *required*
corner ≥ 10 px inside the 640×480 rendered image (`robodojo_rig.intrinsics()`, square pixels,
below), and every *partial* object partly inside.
Checked so far on the rendered B2 (Task 45) head frames; the other tasks' `_v5` frames are pending.

| ID | Required (all corners in view) | Partial in the head view (wrist-camera coverage to be confirmed on the `_v5` recordings) |
|---|---|---|
| A1 | part on the loading pad, zincate dip bath rim, recipe lamps | rinse bath: its front-right rim corner is 0.88 m right of the camera at 0.2 m depth, past the 96° horizontal FOV; the right (working) arm's 0.5 m reach pins the rig between the baths. |
| A2 | loading sector, rotary table | pump on the infeed cradle: left edge at the image border; the recorded left-arm path is at its reach limit (a 4 cm base shift leaves 4.6 mm IK error). |
| A3 | cover source, housing, nest, dispenser head, process beacon | — |
| B1 | valve, leak outlet, spill tray rim, leak beacon | — |
| B2 | stop button, fault lamp | motor under test: context only, 0.8 m right of the camera behind its guard. |
| B3 | capture lane, zone edges and end marker, rework tray, bearing on the tray, arrest/recovery lamp | — |
| C1 | kit 0 and kit 1 source rows | kit inserts: the 0.545 m cases sit beside the bases (x 0.39–0.93), where the horizontal FOV ends 0.55–0.66 m from the centre line, and their near edge is below the image; only each insert's inner far corner is in the head view. |
| C2 | presented carton, A-01 upper deck, B-01 lower deck | — |
| C3 | input tray, bench scale pan, scale indicator, all four receivers | — |
| C4 | both pan-feed rows, both output trays | carriers with the case: outer-front corner next to the base line, past the horizontal FOV. |

#### Rendered intrinsics and renderer readback

Read back from Kit in every rendered run (`renderer` in results and sidecars):
`/rtx/rendermode` = `RaytracedLighting`, `/rtx/post/aa/op` = 4 (DLAA; available on the
A100, no fallback), `/rtx/post/dlss/execMode` = 2, and each value of the spec's quality
preset as set. Isaac Lab's `set_render_rtx_realtime(antialiasing="DLAA")` switches Kit 110
to RealTimePathTracing, so `native/rig.apply_renderer()` sets the spec's mode after it.

The head camera's apertures (22.212 × 14.266 mm, 1.557:1) do not match its 4:3 image.
`get_intrinsics_matrix()` reports fx 288.13, fy 336.46 (cx 320, cy 240) from the
apertures, but RTX draws square pixels: in the rendered Task 45 head frame the fault lamp's
centroid is at (398.0, 69.6) px, square pixels predict (398.4, 70.4), both apertures
(398.4, 42.0). So the head image has **fx = fy = 288.13** (horizontal FOV 96.0°,
vertical 79.6°, not the 71.0° the apertures imply). The D435 wrist apertures are 4:3,
so reported and rendered agree: fx = fy = 397.04, cx 320, cy 240 (77.7° × 62.3°).
The spec (3903057, merged) now states this: `robodojo_rig.intrinsics()` and
`field_of_view_deg()` describe the rendered image. Metadata carries both
`intrinsics_reported` (the API's matrix) and `intrinsics_rendered` (`robodojo_rig.intrinsics()`),
and the projection test uses the latter.

## Package layout

```
src/roboclock/envs/industry/
  spec.py, tasks.py, config.py, results.py   registration, task catalog, config, sidecar adapter
  env.py        IndustryEnv (BenchmarkEnv)
  oracle.py     IndustryOracle: the recorded controllers as a 16-joint policy
  scoring.py    per-task monitors: success from measured scene state
  configs/      benchmark.yaml, tasks.json, task<NN>/reviewed.json (recorded episodes)
  videos.py, export.py   native recording entry point and sidecar export
  native/       merged native sources (scenes, controllers, process models)
    catalog.py    per-task scene/controller and recorded loop settings
    runner.py     standalone native episode (the recording path)
    paths.py, appearance.py, dual_arx.py, arx_cell_task.py, minimal_context.py,
    <task>_task.py / *_scene.py / ..., v2/ (kinematics, geometry, runtime)
```

Everything is one importable package that runs in-process with the main
`roboclock` package. Importing it does not start Isaac Sim.

## Benchmark environment

```bash
roboclock tasks --env industry
roboclock plan  --env industry --task C3 --seeds 1 --seed-start 1
roboclock eval  --env industry --policy oracle --oracle --task B2 --seeds 1 --seed-start 0 \
  --output results/industry/B2
roboclock eval  --env industry --policy hold --task B2 --seeds 1 --output results/industry/B2-hold
roboclock summarize --normalize results/industry data/industry
```

Run with Isaac Sim 6.0.1 / Python 3.12 (the recorded version), e.g.
`OMNI_KIT_ACCEPT_EULA=YES python -m roboclock.cli eval ...`. With
`CUDA_VISIBLE_DEVICES=N` the shared `IsaacRuntime` renders on that GPU.

**Difficulty levels.** `easy`, `medium` and `hard` are defined in
`configs/difficulty.yaml`, which records where each value comes from.

- **The seed.** It draws the resolver's bounded layout, timing (event onsets) and
  physics randomization. These draws are the same at every level. Where the oracle
  showed that a draw breaks the setup, the task turns that factor off (Rand. column,
  reasons below).
- **The level (A/B).** It sets one temporal knob per task and the episode budget, then
  re-hashes the resolved spec. The spec records `difficulty` and `temporal`, and
  `time_requirement` states the level's numbers. The policy's instruction is the same
  at every level.
- **C tasks** follow [C-class timing](../c-class-timing.md): the level changes nothing in
  the episode. The resolved spec (hash included) is the same at easy, medium and hard; its
  `temporal` records the frozen oracle time t_o, the tiers' budgets T = 4 / √5 / 1.25 × t_o and
  the horizon H = 5 t_o, and its `budget_s` (the native oracle controllers give up after it)
  is H. `time_requirement` asks for the jobs "as quickly as possible", with no seconds. The
  level only picks the tier that fills `temporal_success` and `joint_success`.
- **How the values were chosen** (recalibrated 2026-11-16). Reference: the scripted
  privileged oracle (a reference level, not an optimal time), holding its start pose until
  the cue, seeds 0–9 at every level, zero inference latency and control delay. With t its
  slowest time over the seeds of the quantity a knob is compared against,
  hard = max(t / 0.8, t + 0.30 s) (rounded up to 0.01 s); easy keeps its earlier value
  unless the oracle fails it or the new hard reaches it (then a proposed value);
  medium = sqrt(easy × hard). A2's sector is physical and keeps its values. The approved
  values from `configs/tasks.json` stay in `reviewed_nominal` (`reviewed.json`).
- **Budgets.**
  - A/B: the latest time the requirement can still be met, plus 1.5x the oracle's
    remaining work and the recorded post-completion hold.
  - C: the horizon H at every level (table below the A/B one).

| Task | Knob | easy | medium | hard | Budget easy / medium / hard (s) | Rand. |
|---|---|---|---|---|---|---|
| A1 zincate dip | exit window after immersion (opens at the seed's 18–22 s) | 24 s wide | 15.6 s (approved 20 s) | 10.14 s | 140 / 130 / 125 | L T P |
| A2 rotary loading | loading sector at 5°/s (0.83 rpm), narrowed from its entry edge | **60° (12 s)** | 50° (10 s) | 40° (8 s) | 65 / 65 / 65 | L T |
| A3 cover bonding | window after the bead | 2–20 s | 2–18.15 s | 2–16.47 s (approved 2–14 s) | 65 / 60 / 60 | L T P |
| B1 leak isolation | close within | 8.4 s (was 6 s) | 7.26 s (approved 5 s) | 6.28 s | 40 / 40 / 35 | L T P |
| B2 emergency stop | actuate within | **3 s** | 2.75 s | 2.53 s | 30 / 30 / 30 | L T P |
| B3 bearing arrest | stable arrest within | **3 s** | 2.12 s | 1.5 s | 60 / 60 / 60 | L T |

Bold marks the approved value; Rand. lists the randomized factors (layout, timing,
physical).

| Task | Q (jobs) | t_o (s) | T easy / medium / hard (s) | H (s) | Rand. |
|---|---|---|---|---|---|
| C1 kitting | 2 kits (6 components) | 64.50 | 258.0 / 144.2 / 80.6 | 322.5 | L T P |
| C2 putaway | 3 cartons | 45.32 | 181.3 / 101.3 / 56.6 | 226.6 | L T P |
| C3 weighing | 3 parts | 69.36 | 277.4 / 155.1 / 86.7 | 346.8 | L T P |
| C4 palettes | 2 palettes (8 pans) | 81.46 | 325.8 / 182.2 / 101.8 | 407.3 | L T P |

Until score_version 2 the C levels were time limits (C1 120 / 100 / 80 s, C2 90 / 75 /
60 s, C3 and C4 135 / 112.5 / 90 s) and the episode lasted exactly that limit.

- **A2** keeps the recorded table speed and narrows the sector. Its oracle is tuned to
  5°/s: faster (5.5, 6°/s) its tracking lag (~15 mm) touches the carrier while
  lowering, slower (3.3, 4°/s) the housing lands off the carrier support and is never
  released. The green 60° ring and its posts are colliders (everything under
  `/World/LoadingZone`), so they stay; a narrower level covers the closed part with a
  visual-only black mask and a white post at its entry edge, and the HMI and floor
  signs show the level's width and window. The exit edge (−120°) is the same at every
  level, so is the budget.
- **B1**'s easy level was 6 s, set when the leak ran at 0.4 L/s and a longer leak would
  have overflowed the 2.6 L tray. At the current 0.14 L/s the 2.0 L limit is reached only
  about 14 s after onset, so the tray no longer bounds any level's deadline. The oracle now
  holds the reset pose until the leak cue and has the valve closed 4.92–5.02 s after onset
  (it used to wait 2.5 cm above the handle and close in 3.6 s), so hard is 6.28 s, above the
  old easy. Easy was raised to 8.4 s (the earlier easy:hard ratio 4/3), approved 2026-11-16.
- **B2**: the oracle now holds the reset pose until the warning and actuates the stop
  1.97–2.02 s after it (from the earlier pre-positioned standby it took 0.8 s).
- **A2**'s seed 3 fails at every level: the oracle enters `track` at the usual carrier angle
  but never reaches `lower` (no contact or collision is logged; since 2026-11-20 the
  controller prints its failure string as `FAILURE ...` in the eval log). Its timing draw (initial table angle 232.0°, the
  earliest carrier of seeds 0–9) is the only difference from the passing seeds.
- **B3's deadline** is on the *stable* arrest. An un-braked bearing crosses the 0.3 m
  zone in 0.74–0.79 s, so at every level the brake must already be in the lane by then;
  an active brake settles in about 1.1 s. The hold policy (seed 0) gets no credit: the
  bearing rolls into the resting right hand and settles only 6.2 s after entry.

**Spatial (layout) randomization** is on for every task since 2026-11-20. It is drawn from
the seed's layout stream, so it is the same at every level. What moves:

| Task | Layout draw (uniform) |
|---|---|
| A1 | loading pad offset ±8 mm x/y, part yaw ±8° |
| A2 | housing **and its infeed cradle pins** ±10 mm x/y (was ±4 mm on the housing alone, off) |
| A3 | cover source ±4 mm x/y, yaw ±5° |
| B1 | grasp radius 0.15–0.17 m and, new, the whole valve line (skid, piping, flanges, valve, leak nipple) ±20 mm x/y; the oracle grasps at the moved pivot |
| B2 | stop button ±8 mm x/y |
| B3 | gate release x ±5 mm, lane y ±2 mm, outfeed source x ±5 mm (was off) |
| C1 | each source part ±1 mm x/y |
| C2 | the infeed carton stack ±10 mm x/y (was ±1 mm, off) |
| C3 | each part ±1 mm x/y |
| C4 | each feeder pan and its post ±2 mm x/y (was ±0.8 mm, off) |

C2's layout flag used to permute the A, A, B carton order as well; every seed 0–5 drew
A, B, A or B, A, A and the scripted oracle (built for A, A, B) failed all of them. The
permutation is now a separate resolver option (`resolve_putaway_throughput(...,
sku_order=True)`), off in evaluation.

Randomization still turned off:

| Task | Off | Why |
|---|---|---|
| A2 | physical | seed 1's 0.284 kg housing slipped out of the grasp during the lift |
| B3 | physical | seed 0's physical draw (mass ×1.07, friction 0.233) gave an unexpected contact while the left hand closed |

**Oracle with layout on** (2026-11-20; no render, seeds 0–4 × easy / medium / hard; a C
episode is the same at every level, so C rows are one medium run judged at all three tiers;
before = the same runs with the previous flags):

| Task | Before (e / m / h) | After (e / m / h) | Failing seeds after |
|---|---|---|---|
| A2 | 4 / 4 / 4 | 3 / 3 / 3 | 2: `Lost physical grasp` in the carry (offset +8.7 / −7.1 mm); 3: fails as before |
| B1 | 5 / 5 / 5 | 5 / 5 / 5 | — |
| B3 | 5 / 5 / 5 | 3 / 3 / 3 | 2, 3: `Unexpected native robot contact`, left and right fingers (link8) touch while the left hand closes in the hand-off |
| C2 | 5 / 5 / 5 | 5 / 5 / 5 | — |
| C4 | 3 / 3 / 3 (seeds 0, 2 stall) | 5 / 5 / 5 (also at ±0.8 mm) | — |

Why A2 and B3 lose seeds. Before, their layout was off and physical off as well, so every
seed ran the same grasp (A2) and the same bearing roll and arrest (B3); only the table
angle or gate time changed. With any spatial draw the casting's grasp and the bearing's
arrest pose change, and the oracles' margins show. Neither failure follows the magnitude:
A2 at ±5 mm failed seeds 0 and 1 and passed 2 and 3 (seed 3 failing at every level before);
B3 failed with each of its three layout components held at nominal in turn. In B3 the
failing seeds arrest the bearing about 6 mm to +y of the lane centre, where the closing
left fingers meet the right brake's; a higher or lower pick point, a different bore offset,
a wider approach yaw limit, or letting the oracle continue through the finger touch did
not help. Oracle changes (scoring unchanged): A2 aborts only on contacts the scoring voids
(`scoring.voiding_contacts`; a finger touching the carrier while lowering does not void),
which recovered seed 1; B3 releases once the held bearing is within 3 mm of the tray
floor (was 1 mm; the tray's cradle rails are 1.5 mm high), which recovered seed 1. Spec
hashes of every level of A2, B1, B3, C2 and C4 change (new draws and bounds); the
`reviewed_nominal` replays (`reviewed.json`) are unchanged.

**Recalibration measurement** (2026-11-16; oracle holding its start pose until the cue,
no render, seeds 0–9 × easy / medium / hard, zero inference latency and control delay;
success counts are under the levels before the recalibration):

| Task | Oracle measurement over seeds 0–9 | Success e / m / h (old levels) | New hard | Oracle uses |
|---|---|---|---|---|
| A1 | exit 28.20–29.02 s after immersion (part starting immersed; superseded 2026-11-19, see below) | 10 / 10 / 10 | width 17.07 s (closes 35.3–39.0 s) | ≤ 80 % of the close |
| A2 | release at −110.4 to −111.0° (seed 2: −119.2°) | 9 / 9 / 9 (seed 3 fails) | unchanged 8 s (physical) | 77–79 % (seed 2: 98 %) |
| A3 | seated release 12.79–13.17 s after the bead | 10 / 10 / 10 | 2–16.47 s | 80 % |
| B1 | valve closed 4.92–5.02 s after leak onset | 10 / 8 / 0 | 6.28 s | 80 % |
| B2 | stop actuated 1.97–2.02 s after the warning | 10 / 10 / 6 | 2.53 s | 80 % |
| B3 | stable arrest 1.03–1.19 s after zone entry | 10 / 10 / 10 | 1.5 s | 79 % |

**A1 pick-and-dip start** (2026-11-19). The part no longer starts immersed: it waits dry on a
loading pad on the drip bridge between the baths, and the policy picks it up, lowers it into
the zincate bath (full immersion starts the recipe clock), lifts it out inside the window and
releases it in the rinse bath. Stages: immersed 20, leaving the liquid 35, cleared 55. Oracle,
no render, seeds 0–9, easy: 10 / 10 joint success; immersion 30.2–30.5 s after reset, exit
22.10–25.64 s after immersion (3.67–3.73 s after the window opens; it holds the immersed
part and lifts 1 s after the opening), rinse retention 80.1–83.6 s. New hard width 10.14 s
(seed 4: exit 25.64 s, opening 21.91 s), medium √(24 × 10.14) = 15.6 s; budgets
1.5 × 30.5 + (22 + width) + 1.5 × 30.7 s = 140 / 130 / 125 s. A1 rows elsewhere in this file
that give an exit of 28.2–29.0 s or a completion near 56 s describe the earlier start.

**Oracle validation of the levels** (2026-11-06, under the retired C time limits;
privileged oracle, no render, seeds 0–2, randomization as in the Rand. column). Every task
and level passed 3/3. B3 and C4 ran at hard only: their oracle run is the same at every
level, which only moves B3's deadline or C4's time limit, and both finished inside hard's.
C hard is now t_o itself, the slowest recorded oracle demo, so the oracle uses up to 100 %.

| Task | Oracle measurement over the seeds | Hard requirement | Oracle uses |
|---|---|---|---|
| A1 | exit 28.2–28.3 s after immersion (window opens 19.9–21.7 s) | closes 35.9–37.7 s | 79 % |
| A2 | release at −110.1 to −110.2°, retention 52.2–54.1 s | 40° sector (−80 to −120°), 8 s | 75 % of the window |
| A3 | seated 12.8–13.3 s after the bead | 2–14 s | 95 % (approved value) |
| B1 | valve closed 3.52–3.59 s after leak onset (earlier oracle, pre-positioned above the handle; now about 4.9–5.0 s from the reset pose) | 4.5 s | 80 % (earlier oracle) |
| B2 | stop actuated 0.78–0.79 s after the warning | 2 s | 40 % |
| B3 | stable arrest 1.07–1.15 s after zone entry | 2 s | 58 % |
| C1 | 2 kits by 64.0–64.5 s | 80 s | 81 % |
| C2 | 3 cartons by 43.9–45.0 s | 60 s | 75 % |
| C3 | 3 parts by 80.9–81.6 s | 90 s | 91 % (approved value) |
| C4 | 2 palettes by 80.6–81.4 s | 90 s | 90 % (approved value) |

The A2 oracle's phase timeout read the previous phase's age on the step it left
`wait_window` (`native/rotary_task.py`); it now times the current phase. This never
fired at the recorded 5°/s, where `wait_window` lasts under 20 s.

`reviewed_nominal` replays the recorded episodes unchanged: for the reviewed seed (0,
or 1 for C3) the episode is exactly `configs/task<NN>/reviewed.json`, hash-checked,
and an A/B budget is 90 s. A C episode runs and is judged as at the levels (horizon H),
with the medium tier's verdict in `temporal_success` / `joint_success` (`NOMINAL_TIER`);
its spec keeps the recorded 90 s `budget_s`, which every recorded oracle finishes inside.
Another seed resolves the same nominal layout, timing and physics, and changes only
seed-derived values (C3's QC results). `make_task` and the environment accept it, but
`roboclock eval --difficulty all` does not include it. A `tasks.<ID>.max_time` in the config
sets the episode length for short debugging runs (a cut C episode ends with
`termination_reason` "horizon"); it never changes the spec or the judging.

**Timing.** Physics runs at the spec's 250 Hz in every task, control at
50 Hz (`control_decimation` = 5, `roboclock.envs.rates`), cameras at 25 Hz. `env.config["sim"]["physics_dt"]` is
set to the current task's rate on reset. A/B budgets are per level (above; 90 s under
`reviewed_nominal`); A/B episodes terminate once the monitor measures completion plus the
recorded runner's post-completion time (3 s for A1/A2, 2.5 s otherwise). C episodes run to
H and end early once all Q jobs are done and `final_valid` has held for 2 s (`all_done`),
or when, from T_medium on, there was no progress (a job or a credited component, pan or face) for `throughput.stall_window(t_o, steps)` = max(0.5 t_o, 1.5 T_easy / steps) (`stall`); otherwise `horizon`.
They score the state at their end. Industry has no C failure stop: an arm collision
voids physical success but the episode goes on, so t_Q is still measured.

**Arm contacts.** Every contact of an arm body is recorded (`unexpected_contacts`), except a
finger on the product. A recorded contact voids physical success only when it is an arm
collision: an arm link other than a finger (link7, link8) pressing on anything with more than
`scoring.ARM_COLLISION_N` = 20 N of normal force, the sports rule. The fingers may touch parts
and fixtures; lighter touches of other links are reported as `arm_contact_peak_N`. A row lists
the voiding pairs under `arm_collisions` and as `arm_collision:<a>|<b>` codes in
`physical_failures`. Until 2026-11-10 any recorded contact voided the verdict, whatever its
force. The scripted oracles make no contact at all, and their own validation
(`native/runner.py`, each task's validator) still requires none.

**Actions** are `Box(tile(JOINT_LOW, 2), tile(JOINT_HIGH, 2))` from
`roboclock.envs.lab.robot`: 16 joint-position targets in the order
`[left/joint1..joint8, right/joint1..joint8]`. Joints 1–6 are the arm revolute
joints (rad), 7/8 the two finger sliders (m, 0–0.044). These are exactly the
industry `DualARX` articulations' DOFs; the environment maps them by name
(`dual.names[side].index(f"joint{i}")`) onto `DualARX.commands`, which are
re-applied as PD position targets on every physics step of the control period
(the native drives: kp 700/kd 70 for the arm, 1500/80 for the fingers, the
recorded per-task velocity caps). The bounds equal the industry robot manifest's
`arm_lower/upper_rad` and `gripper_travel_m`.

**Observations** match the laboratory layout and the RoboDojo spec:
- `rgb` (480×640×3, the head camera of [Common Dual ARX rig](#common-dual-arx-rig));
- `left_wrist_rgb` and `right_wrist_rgb` (480×640×3).

All three are rendered once per 25 Hz policy slot (RaytracedLighting + DLAA,
`sim.render_subframes` subframes) and repeated on the in-between 50 Hz step.
`sim.no_render: true` creates no cameras, makes no render calls and returns zero images
of the same shapes. The other keys are:
`joint_positions` (16, measured, clipped to the action bounds: a closed slider can
read ~1e-5 m below 0), `ee_pose` (14: per arm tool-point xyz and wxyz quaternion
of link6 rotated so +Z points to the fingertips — the lab backend convention; the
tool point is link6 + 0.145 m along its X), `gripper_width` (2, joint7+joint8),
`sim_time`, `instruction`; `depth`/`left_wrist_depth`/`right_wrist_depth` with `--depth`. The
head camera is fixed to the rig: B3's recorded camera that followed the bearing was
removed. `episode_metadata()` adds `rig` and `renderer`:
- `rig` holds `robodojo_rig.describe()`, the task's placement, the camera intrinsics
  reported by `get_intrinsics_matrix()` (`intrinsics_reported`), the intrinsics the
  rendered images actually have (`intrinsics_rendered`, see
  [Rendered intrinsics](#rendered-intrinsics-and-renderer-readback)) and the measured rig check
  (`check_rig` on the bases and head camera, wrist cameras against `wrist_camera_in_link6`);
- `renderer` is the `/rtx/rendermode` and antialiasing read back from Kit.

**Scoring** (`scoring.py`) uses the native scenes' own measurements and success
predicates, evaluated for every policy. Where the recorded controller had decided
part of the outcome, the same predicate and stability window is evaluated from
scene state instead, without controller phases:

| Task | Physical (`task_success`) | Temporal |
|---|---|---|
| A1 | part retained released on the rinse floor (0.5 s) at the end | whole-part exit 20–40 s after full immersion (the part starts dry on the loading pad; the clock starts once it is wholly below the liquid inside the zincate bath) |
| A2 | supported, released placement in the carrier at the end; grasp measured as both fingers > 10 N for 0.25 s | release angle inside −120…−60° |
| A3 | aligned released seating confirmed 0.5 s, still seated | seating 2–14 s after the bead; no earlier housing contact; lid grasped (> 8 N, 0.25 s) and cleared from its source |
| B1 | valve ≥ 89° confirmed 0.15 s after post-leak finger contact; closed, not leaking at the end | within 5 s of leak onset; no valve motion before the leak |
| B2 | 8 mm press with finger contact 0.2 s, motor braked and stopped | within 3 s of the warning; no press before it |
| B3 | released bearing retained on the rework tray (0.5 s window) at the end | arrest (inside zone, a fingertip braking, 0.15 s window; lane support is not required since 2026-11-11: a firm catch lifts the bearing off the lane) ≤ 3 s after zone entry; no hand contact before entry |
| C1 | 6 components credited (whole in insert, on floor, supported, released, 0.3 s window); all contained at the end | t_Q (2nd kit complete) ≤ T_tier |
| C2 | 3 cartons credited by `valid_product` (inside the SKU's slot, on that SKU's deck, supported, released, at rest) held > 0.5 s; all still valid at the end | t_Q (3rd carton) ≤ T_tier |
| C3 | 3 parts sorted: weighed (resting alone, released and at rest on the scale pan for 0.5 s with the display settled), then released > 0.3 s in a receiver of the part's weight result (PASS at 125 g or more, else REJECT; either arm's side); all still resting there at the end | t_Q (3rd part) ≤ T_tier |
| C4 | 8 pans seated (0.2 s window), two palettes delivered (`payload_native_valid`, 0.5 s window); both still valid at the end | t_Q (2nd palette) ≤ T_tier |

C physical success has no time term, however late the jobs were done; temporal success
has no physical term. `roboclock.evaluation.throughput.judge` scores every C row at all
three tiers (`temporal_success_by_tier`, `joint_success_by_tier`) from one run, with t_Q
the tick of the Q-th credit (`completion_time`).

Unexpected robot contacts (the scenes' contact books) void physical success.
`joint_success` requires both. Rows add `reaction_latency` (B), `completion_time`,
`correct_operations`, `operations_per_minute`, `job_times_s`, `t_q_s`, `oracle_time_s`,
`budget_s_by_tier`, `judged_tier` and `termination_reason` (C), `measured_events` and the
scene's event times. Release/withdrawal *after* completion was a controller phase in the
recordings and is not scored separately except where a scene measures it.

**Task processes** run in the environment for every policy, on control ticks:
A1's recipe lamps, A3's adhesive dispenser and cue (recorded inside the
controller), B3's machine outfeed, QC lamp and upstream gate
(recorded inside the controller), C3's bench scale and its indicator
(`scene.process_step`), and B1/B2's leak/fault models inside
`scene.observe()`. A joint-position action cannot express one task rule, which the
environment applies: C4 switches a palette
arm's finger drives to the declared case-lip gains (4000 N/m, 80 N·s/m) once that
palette's four pans are credited (the controller switched them when it started
the case transfer, i.e. slightly later). C1's controller re-applies the unchanged
1500 N/m finger gains, which is a no-op.

**Oracle** (`--policy oracle --oracle`): the recorded privileged controller
(contact forces, product poses, IK) is created with the scene at time 0 and run on
each observation; its commands are the action. Its event writes go to a private
overlay, so it cannot influence scoring.

## Native runner and recordings

`python -m roboclock.envs.industry.videos --task <41-50> [--record]` runs
`native/runner.py` in a fresh process (no `PYTHONPATH` changes) and exports the
movie sidecar. It accepts the same replay/seed/randomization flags as before:

```bash
python -m roboclock.envs.industry.videos --task 49 --dry-run
python -m roboclock.envs.industry.videos --task 49 --seed 2 \
  --randomize-layout --randomize-timing --randomize-physics --output <dir>
python -m roboclock.envs.industry.videos --task 49 --replay <dir>/resolved.json --output <dir2>
```

`ffmpeg` is needed for
`--record`. Verify assets and recorded media without Isaac Sim with
`python scripts/industry/verify_delivery.py`. Install with
`pip install -e '.[sim6,industry]'`; `scripts/industry/requirements-replay.txt`
pins the recording environment's direct dependencies.

## Merge of the recorded source profiles

Through commit `dd853c8` (unchanged to `8bddf10`) each task ran a frozen copy of a `roboclock` tree in
`src/roboclock/envs/industry/profiles/task<NN>/` as a separate process with that
tree on `PYTHONPATH`. Those ten copies are merged into `native/`:

* 44 of the 52 distinct files were byte-identical wherever present and were taken once.
* `dual_arx.py` (4 variants): superset; `mount_yaw_deg` and the link-transform cache
  are opt-in, and the slider max-velocity cap (95% of 0.04 m/s), which only 46–50 were
  recorded with, is `gripper_velocity_cap` (False for the 41–45 scenes).
* `arx_cell_task.py` (5 variants): the 41/42/48 version is a superset (point samples,
  minimum-separation events, per-mount rotations, tunable gains); the 43–45 version's
  hard-coded `impulse * 240` equals its `1/physics_dt` at 240 Hz.
* `minimal_context.py` (3 variants): the 45/49 version (machine-signal lamps) plus the
  47 two-kit labels. Its never-recorded per-product 48 slot relabelling was dropped
  (it would relabel the B-01 shelf sign as A-01 in the throughput scene).
* `bearing_arx_task.py`, `industrial_workcell.py`, `putaway_arx_task.py` (2 variants
  each): a description string, an optional sign background color and two cached
  thresholds — supersets.
* The four runner scripts (5+2 variants) became `native/runner.py` with per-task
  loop settings in `native/catalog.py` (observe cadence, post-completion time,
  final observation, read-back cache, render switch, runtime profile and renderer).
* Shadowed main modules: `simulators/isaac/{diagnostics,visual_geometry,rendering}.py`
  are the main versions (identical or supersets; native calls pass explicit capture
  subframes). The laboratory appearance is the main one except its label textures,
  kept as `native/appearance.py` ("fit-v2" shrink-to-fit labels). The main module
  also makes emissive materials glow in their requested color (the frozen copy used
  the palette default), a visual-only change. `resources.py`/`_industry_paths.py`
  became `native/paths.py`; textures/HDR resolve to byte-identical main assets.
* The October 29 revisions (`d34b46b`) landed in `profiles/task41` and `task49` while
  the merge was in progress and are carried over: Task 49's outward left source, wider
  input tray, waiting lane, slower fingers and any-impulse inter-arm failure
  (`arx_cell_task.py`, `inspection_arx_task.py`, `InspectionMonitor` in `scoring.py`),
  and Task 41's fresh-capture path (`native/plating_capture.py`, profile
  `isaac601-pt-stability`, set per task in `native/catalog.py`). `--capture-history`,
  which had one value, was dropped.
* `kitting_arx_task.py` (single-kit Task 47 variant reachable only from other tasks'
  runners) was not used by any recording and was not carried over.
* A3's dispenser and B3's outfeed/gate moved from the controllers' `step()` into
  `scene.process_step()`, which both the runner and the environment call before
  the controller on each tick (same order as recorded). `runtime_profiles.py`
  defaults to the recorded Isaac Sim 6.0.1 profile and drops its renderer GPU pin
  when `CUDA_VISIBLE_DEVICES` is set (the pin selected a GPU without a visible CUDA
  twin and RTX failed to start).

**Exact historical reproduction**: check out commit `dd853c8` (HEAD when the merge
began) in a separate directory and run its
`python -m roboclock.envs.industry.videos --task <NN>` (commit `d34b46b` for Tasks 41
and 49, whose current videos it generated); that tree contains the frozen
profiles, `profiles/manifest.json` and `profiles/source_integrity.json`
(recording/release hashes). The Industry sources are unchanged from `dd853c8` through
`8bddf10`; commits from `7a4825e` on also track the `assets/industry` binaries, which
`dd853c8` expects from the asset bundle. Recorded sidecars keep their original
`generation` paths. Do not mix old profile sources into this tree. On hosts with
several GPUs and `CUDA_VISIBLE_DEVICES`, the old runtime's `activeGpu=0` pin can stop
RTX from starting; drop that launch key (it does not affect physics).

## Verification on the RoboDojo rig spec (2026-11-04, Isaac Sim 6.0.1, A100, shared node)

Four modes per reviewed episode (seed 0; seed 1 for C3), all with the spec cameras and 250 Hz:
the headless native runner (`runner --no-render`), the headless environment eval
(`roboclock eval --env industry --policy oracle --oracle` with `sim.no_render: true`), the
rendered environment eval (same command, RaytracedLighting + DLAA, head and wrist cameras
rendered at 25 Hz) and the rendered `videos.py --record` run that made the `_v5` recordings.
Times are simulation seconds; budgets and deadlines are judged on simulation time, so the
node's load (1-min 23–111 during these runs) changed only wall time. Headless and rendered
runs gave identical event times in every task.

| ID | Native runner (headless; same in the `_v5` recording) | Environment eval, headless = rendered (physical / temporal / joint) |
|---|---|---|
| A1 | ✓; whole-part exit 28.352 s after immersion (window 20–40 s: 8.35 s / 11.65 s margin) | ✓ / ✓ / ✓; completion 56.096 s |
| A2 | ✓; supported release 52.044 s at −110.18° (sector −120…−60°: 9.82° margin) | ✓ / ✓ / ✓; completion 52.344 s |
| A3 | ✓; seated 12.804 s after the bead (window 2–14 s: 1.196 s margin), confirmed 47.016 s | ✓ / ✓ / ✓; completion 47.016 s |
| B1 | ✓; valve closed 3.600 s after leak onset (≤ 5 s: 1.40 s margin) | ✓ / ✓ / ✓; 3.600 s, completion 25.600 s |
| B2 | ✓; stop 0.796 s after the warning (≤ 3 s: 2.204 s margin) | ✓ / ✓ / ✓; 0.796 s, completion 17.328 s |
| B3 | ✓; arrest 1.152 s after zone entry (≤ 3 s: 1.848 s margin), tray retention 38.492 s (controller) | ✓ / ✓ / ✓; 1.152 s, completion 36.372 s (rim cradle; see above) |
| C1 | ✓; kits 68.752 / 68.812 s (controller), 21.19 s to spare | ✓ / ✓ / ✓; 2 kits at 63.532 s, **26.468 s** to spare |
| C2 | ✓; cartons 16.31 / 33.41 / 44.13 s (controller, 45.87 s to spare); monitor replay 15.332 / 32.332 / 43.072 s | ✓ / ✓ / ✓; 3 cartons at 43.072 s, **46.928 s** to spare, 0 contacts |
| C3 (seed 1) | ✓; controller completion 82.752 s (7.248 s to spare), 0 inter-arm contacts; monitor replay 81.252 s | ✓ / ✓ / ✓; sorted 37.712 / 59.012 / 81.252 s, **8.748 s** to spare, 0 contacts (inter-arm included) |
| C4 | ✓; deliveries 81.03 / 81.13 s (controller, 8.87 s to spare) | ✓ / ✓ / ✓; 2 palettes at 80.052 s, **9.948 s** to spare |

Physics changed from the common-rig pass only where the table in
[RoboDojo-spec pass](#robodojo-spec-pass-2026-11-04) says: 46/47/48/50 moved to 250 Hz. C2
gained 0.34 s (monitor 43.41 → 43.07 s) and C4 5.34 s (monitor 85.39 → 80.05 s, the output
trays are now in front of the nests); C1 and C3 are unchanged. C3 at every seed 0–8 (headless,
per-arm receivers): all sort 3/3 with 0 inter-arm contacts; controller completion
81.572–83.732 s (6.268–8.428 s to spare), monitor sort credit (replayed on the runner traces)
80.092–82.272 s (7.728–9.908 s to spare).

The B3 rendered eval first failed physically on the flat tray (credited at 37.49 s, the
retention window broke before the episode end); the rim cradle fixed it and the runs above
use it. Every rendered run read back RaytracedLighting / DLAA and passed the rig check
(`check_rig`; wrist cameras ≤ 3.3e-8 m from `wrist_camera_in_link6`).

**What only the wrist cameras see** (checked on the `_v5` head and wrist streams):
* A1: the part inside the rinse bath and the bath's front-right part (right wrist, from
  the transfer at ~40 s). The right wrist camera dips below the liquid surface while the
  hand grasps the immersed part in the zincate dip bath (20.8–26.2 s) and places it on the rinse
  floor (53.8–58.6 s); those frames are dark. The part must be immersed and the camera sits
  on `link6`, so this follows from the task.
* A2: the pump casting on the infeed cradle close up during the grasp (left wrist); the head
  camera has its left edge at the image border.
* C1: the kit inserts' pockets and the placed bearing, bracket and pin (both wrists during
  each placement); the head camera shows only the cases' inner far corners in its bottom corners.
* C4: the case pockets while each pan is placed (both wrists). For 1.1–1.3 s in two of the
  four placements per arm the camera is lowered into the case recess and its image is dark.
* B2: the button face during the press (right wrist); the head camera sees button and lamp too.
In the other tasks every key object is in the head view; the wrists add close-ups.

## Verification on the common rig (2026-11-03, superseded by the spec pass above)

Headless: `native/runner.py --no-render` (physics + privileged oracle, no camera).
Rendered: `roboclock eval --env industry --policy oracle --oracle --task <ID> --seeds 1
--seed-start <0, or 1 for C3>` and the `videos.py --record` runs that made the `_v3`
recordings (`_v4` for C2 and C3 after their second pass). Difficulty `reviewed_nominal` is the only registered one. Times are
simulation seconds. Budgets and deadlines are judged on simulation time, so CPU load
changes only wall time. With the earlier shared receivers, C3 seed 1 ran twice headless
(1-min load ≈25 and ≈50–77) and finished at 89.752 s both times. With the per-arm receivers,
the headless run and the rendered recording both finish at 82.752 s.

| ID | Headless runner (`--no-render`) | Rendered `roboclock eval` (physical / temporal / joint) | Rendered recording `_v3` / `_v4` |
|---|---|---|---|
| A1 | ✓; whole-part exit 28.352 s after immersion (window 20–40 s) | ✓ / ✓ / ✓; exit 28.352 s, completion 56.10 s | ✓; exit 28.352 s |
| A2 | ✓; supported release 52.04 s at −110.18° (sector −120…−60°) | ✓ / ✓ / ✓; −110.18°, completion 52.34 s | ✓; −110.18° |
| A3 | ✓; seated 12.80 s after the bead (window 2–14 s), confirmed 47.02 s | ✓ / ✓ / ✓; 12.80 s, completion 47.02 s | ✓; 12.80 s |
| B1 | ✓; valve closed 3.600 s after leak onset (≤5 s) | ✓ / ✓ / ✓; 3.600 s | ✓; 3.600 s |
| B2 | ✓; stop 0.796 s after the warning (≤3 s) | ✓ / ✓ / ✓; 0.796 s | ✓; 0.796 s |
| B3 | ✓; arrest 1.370 s after entry (≤3 s), tray retention 44.48 s | ✓ / ✓ / ✓; 1.370 s, completion 44.48 s | ✓; 1.370 s, 44.48 s |
| C1 | ✓; kits 68.71 / 68.79 s (controller), 21.2 s to spare | ✓ / ✓ / ✓; kits 63.43 / 63.51 s, **26.49 s** to spare | ✓; 68.71 / 68.79 s, 6/6 parts credited |
| C2 (rig, second pass) | ✓; cartons 16.41 / 33.55 / 44.45 s (controller, **45.55 s** to spare); monitor replay 15.43 / 32.49 / 43.41 s | ✓ / ✓ / ✓; 15.43 / 32.49 / 43.41 s, **46.59 s** to spare, 0 contacts | ✓; 16.41 / 33.55 / 44.45 s, placed {0,1,2} (`_v4`) |
| C3 (seed 1) | ✓; controller completion 82.752 s (**7.248 s** to spare), 0 inter-arm contacts; monitor replay 81.252 s (8.748 s) | ✓ / ✓ / ✓; sorted 37.71 / 59.01 / 81.25 s, **8.748 s** to spare, 0 contacts (inter-arm included) | ✓; 82.752 s, 0 inter-arm contacts, 9 inspection images (`_v4`) |
| C4 | ✓; deliveries 86.47 / 86.57 s (controller) | ✓ / ✓ / ✓; 85.35 / 85.39 s, **4.61 s** to spare | ✓; 86.47 / 86.57 s |

Rendering changed no physical or process event time. The same event appears at the same simulation time headless, in the rendered eval
and in the recording. C margins differ by source. The environment monitor credits an
operation once the released part has been at rest for its dwell window (`required_operations_s`). The native controller's completion
(`completed_s` and the like) waits for its own release and withdraw motion. For C3 the
controller completion is the stricter number: 7.248 s to spare, against the monitor's
8.748 s, which decides the benchmark's temporal success. With the shared receivers these
were 0.248 s and 1.748 s.

C3 at every seed 0–8 (headless, per-arm receivers): all sort 3/3 with 0 inter-arm contacts.
Controller completion 81.57–83.73 s (6.27–8.43 s to spare); the monitor's sort credit,
replayed on the runner traces, is 80.09–82.27 s (7.73–9.91 s to spare). The replay matches the
rendered eval exactly at seed 1. With the shared receivers, seeds 0, 2 and 6 had stopped at 39.56 s on an inter-arm
touch (the right arm leaving the central PASS receiver while the left arm moved to the
underside view), and seeds 4, 5 and 8 sorted only 2 of 3 in 90 s. The per-arm receivers remove the
shared deposit zone and its wait, which fixes both.

## Verification on the shared time base (2026-10-30, Isaac Sim 6.0.1, one L40S)

Physics moved from 240/480/960 Hz to 250/500/1000 Hz and control from 60 Hz to 50 Hz
(`roboclock.envs.rates`). The oracle was rerun on every task with
`roboclock eval --env industry --policy oracle --oracle --task <ID> --seeds 1
--seed-start <0, or 1 for C3>`. Thresholds, deadlines and scoring did not change.

| ID | Old physical / temporal / joint | New | Completion s (old → new) | Key timing (old → new) |
|---|---|---|---|---|
| A1 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 56.05 → 56.10 | whole-part exit 28.338 → 28.364 s |
| A2 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 53.50 → 52.44 | release 53.20 → 52.14 s, at −115.96° → −110.66° |
| A3 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 47.03 → 47.19 | |
| B1 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 25.64 → 25.66 | response 3.642 → 3.660 s |
| B2 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 17.33 → 17.33 | response 0.796 → 0.796 s |
| B3 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 40.35 → 40.18 | arrest response 1.402 → 1.230 s |
| C1 | ✓ / ✓ / ✓, 2 kits | ✓ / ✓ / ✓, 2 kits | 63.64 → 63.79 | kits 63.01 / 63.64 → 63.13 / 63.79 s |
| C2 | ✓ / ✓ / ✓, 3 cartons | ✓ / ✓ / ✓, 3 cartons | 75.74 → 76.65 | credits 17.97 / 44.86 / 75.74 → 18.71 / 45.67 / 76.65 s |
| C3 | ✓ / ✓ / ✓, 3 parts | ✓ / ✓ / ✓, 3 parts | 87.46 → 87.75 | sorted 38.53 / 59.08 / 87.46 → 38.61 / 59.23 / 87.75 s |
| C4 | ✓ / ✓ / ✓, 2 palettes | ✓ / ✓ / ✓, 2 palettes | 85.17 → 85.35 | deliveries 85.02 / 85.17 → 85.29 / 85.35 s |

Rates in seconds carried over directly. A2 and B3 failed at first, because two free
parts are sensitive to the physics step, so their episodes changed:

* A2: dropped onto the cradle, the casting rocked into a rest pose yawed 25° from the
  recorded one. It now starts in the recorded rest pose. The close also leaves both
  slider drives at their 20 N limit, where the casting slid along the jaws during the
  transfer and dropped. After the lift the symmetric target backs off to an 18 N
  spring, which centers it.
* B3: the bearing arrests yawed to either side of the lane, and at the new rate it
  landed on the side opposite the recording (bore at 96° instead of 76°). The left
  hand now follows the measured bore (`handoff.approach_from_bore_deg`), up to 72°
  (`approach_yaw_max_deg`), past which its elbow hits the right arm. The machine-ready
  window also counted a fixed 31 samples, which is now derived from the control rate,
  and `gate_clear_s` is only recorded after the open command, because the gate
  overshoots for a moment at reset.

`configs/task4{6,7,8}/reviewed.json` and `task50` changed with these rates, so their
hashes differed from the 240–960/60 Hz sidecars in `data/industry`. The 2026-11-03
`_v3` recordings (see the rig verification above) replay the current specs, and
`verify_delivery.py` passes again; the older recordings are kept as superseded.

## Verification at the recorded rates (2026-10-29, Isaac Sim 6.0.1, one L40S)

Before the merge, the frozen profile runners for Tasks 45 and 49 were run unchanged
(apart from the GPU pin above) and reproduced every recorded native event time.
The merged `native/runner.py` reproduced every recorded event time of Tasks 41–50
(`events` in each `*_evidence/native-result.json`, 9–154 events per task).

`roboclock eval --env industry --policy oracle --oracle --task <ID> --seeds 1
--seed-start <0, or 1 for C3>` (results under `results/industry/unified/`, git-ignored):

| ID | Recorded physical / temporal / joint | Unified | Sim s (recorded → unified) | Key timing |
|---|---|---|---|---|
| A1 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 66.68 → 59.05 | whole-part exit 28.338 s, rinse retention 56.050 s (identical) |
| A2 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 57.98 → 56.50 | grasp 16.229 s, release 53.200 s at −115.96° (identical) |
| A3 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 51.28 → 49.53 | bead 33.713 s, seated 46.533 s, confirmed 47.033 s (identical) |
| B1 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 30.70 → 28.14 | response 3.642 s (identical) |
| B2 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 19.98 → 19.83 | response 0.796 s, motor stopped 17.329 s (identical) |
| B3 | ✓ / ✓ / ✓ | ✓ / ✓ / ✓ | 43.02 → 42.85 | arrest response 1.402 s (identical); tray retention 40.353 s (controller 40.520 s) |
| C1 | ✓ / ✓ / ✓, 2 kits | ✓ / ✓ / ✓, 2 kits | 89.99 → 90.00 | kits complete 63.01 / 63.64 s (controller credits 63.97 / 64.61 s) |
| C2 | ✓ / ✓ / ✓, 3 cartons | ✓ / ✓ / ✓, 3 cartons | 89.99 → 90.00 | credits 17.97 / 44.86 / 75.74 s (controller 19.17 / 45.94 / 76.76 s) |
| C3 | ✓ / ✓ / ✓, 3 parts | ✓ / ✓ / ✓, 3 parts | 90.00 → 90.01 | all nine face acquisitions and QC reveals identical; sorted 38.53 / 59.08 / 87.46 s |
| C4 | ✓ / ✓ / ✓, 2 palettes | ✓ / ✓ / ✓, 2 palettes | 89.99 → 90.00 | deliveries 85.02 / 85.17 s (controller 86.26 / 86.27 s); case-grip gains at 54.04 / 54.06 s (controller 54.81 / 54.82 s) |

The physical/process event times are bit-identical to the recordings; A/B episodes
end earlier because the environment stops a fixed time after the *measured*
completion rather than after the controller's withdrawal; C credits come slightly
earlier because the monitors do not wait for the controller's release motion. The
recorded C3 sidecar's `manipulation_success: false` is an artifact of the native
inspection scene, which never set that flag; the unified row reports the physical
outcome. `--policy hold` on B2 ran the full 90 s and scored 0 (no press).

The reviewed episodes are the validated ones. Other seeds of `reviewed_nominal` change
only seed-derived values; for C3, seed 0 draws QC results REJECT/PASS/PASS and the
native runner's controller then stopped on an unexpected robot contact (not a
scoring change: the episode is simply harder for the diagnostic controller).

## Limitations

* The oracle is a privileged diagnostic controller; its results are not evidence of
  learned-policy performance. Only `reviewed_nominal` episodes are registered.
* Chemistry, liquid flow, adhesive and symbolic inspection results are uncalibrated
  process models; masses, frictions and gains are declared candidates.
* The shared `IsaacRuntime` cannot apply the native runtime's launch-only render
  switches (single-tick sensor rendering, synchronous hydra, 4 CPU threads); the
  oracle episodes below reproduce the recorded physics event times, but RGB can
  differ slightly from the recordings.
* Original traces 41/42 lack per-arm joint samples; traces written by the merged
  runner include them.

See task descriptions,
instructions and
asset provenance.
