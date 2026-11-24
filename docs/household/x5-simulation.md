# Dual-X5 physics episodes and demos (A1–A4, B1–B3, C1–C3)

**Updated:** 2026-11-03. **Simulator:** Isaac Sim 6.0.1, PhysX, RTX real-time path tracing.

All ten tasks (A1–A4, B1–B3, C1–C3) run as physics episodes in Isaac Sim, driven by a privileged rule-based (oracle) controller. Each episode is checked twice before and after recording. The R2 task model (`<id>_x5/task.py`) scores it online from measured signals. An independent post-hoc check re-derives the result from the recorded trajectory and the ground-truth hidden parameters.

From B1 on, each demo is one successful reference episode (seed 0, medium): one physics-only run checked by the task model and the independent check, then the same configuration recorded. The 2026-11-02 changes were re-run on the other seeds listed under the seed table.

The oracle reads privileged state: object poses, contact forces and the numeric process cue behind the rendered colour. An evaluated agent would not get this. The agent-facing action and observation interface is still undefined.

## Results

The `data/household/x5_demos` recordings in the Demo column were removed from the checkout on
2026-11-20; they are in git at a417e24. The current
demonstrations are `data/household/<ID>/demo.mp4` (medium seed 50, see data/README.md).

| Task | Seed / difficulty | Timed event | Process value at the event | Physical checks | Demo |
|---|---|---|---|---|---|
| A1 | 0 / medium | full-basket extraction at 9.08 s (online model: 9.04 s) | strength 0.597 (band 0.55–0.70) | 2.05 clockwise turns measured in the cup; infuser and spoon released and supported ≥ 1 s (spoon basket at (0.34, 0.10) on the 0.60 m rig) | `data/household/x5_demos/A1/demo.mp4` |
| A2 (countertop ice dispenser, R3.1) | 0 / medium | paddle pressed at 11.20 s; released at 18.08 s | ice level 0.51 of the glass (band 0.40–0.60), 5 cubes = 5 deliveries owed for that press; none outside | glass picked up from the counter, rim held on the paddle at 8°, set down upright and released, max tilt 1.8° | `data/household/x5_demos/A2/demo.mp4` |
| A3 | 0 / medium | pancake turned (rotation ≥ 150°) at 50.6 s | first-side doneness 0.625 (band 0.55–0.75) | levered and toppled by blade contact; final rotation 164.4°; centre 10.1 cm from the pan axis, fully inside the rim; still for 9.4 s; turner withdrawn | `data/household/x5_demos/A3/demo.mp4` |
| A4 | 0 / medium | kettle released upright at 25.3 s (budget 46 s; easy 60 s, hard 36 s since 2026-11-16) | dose 299.7 g (target 300 ± 10 g), spill 0 g | two-arm grasp (handle bar + lid knob) held through 88.0° tilt; kettle set down upright and released | `data/household/x5_demos/A4/A4.mp4` |
| B1 | 0 / medium | faucet closed at 2.40 s (the bowl would overflow at 12 s) | spill 0 L (fail at 0.5 L) | lever pushed shut by contact (final tilt 0.02°); no hand contact for 1 s after closing; not reopened | `data/household/x5_demos/B1/demo.mp4` |
| B2 | 0 / medium | egg stopped 1.6 s after it started to roll, 5.9 cm short of the edge along y | no process model; peak jaw force 11.8 N (limit 30 N) | held clear of the counter for 3.5 s; the other three objects did not move | 8.1 s, 203 frames, real time; the other three objects in `B2/objects/` |
| B3 | 0 / medium | burner off at 5.6 s, flame out at 8.5 s, secured at 11.4 s | peak flame 0.20 (spread at 0.8); exposure 1.1 | knob turned to OFF by contact in one turn (final 2.9°), hand released | `data/household/x5_demos/B3/demo.mp4` |
| C1 (R3) | 0 / medium | 4 of 4 wanted meals on the board at their pickups (20 / 32 / 44 / 56 s); on the board at 10.5 / 23.3 / 35.8 / 48.7 s | slack 9.5 / 8.7 / 8.2 / 7.3 s | no bowl dropped (max tilt 3.6°); every grasp a close-until-contact handle pinch; one arm at a time at the mat | not yet recorded (see the C1 section) |
| C2 | 0 / medium | 6 of 6 glasses served; all served and both bottles back at 44.1 s (window 90 s) | 86–91 ml per glass (line 90, band 80–105), counted from the particle fluid; spill 0.03 ml (limit 15) | both bottles pinched once and held through 3 pours each, set down upright and released; no glass moved | `data/household/x5_demos/C2/demo.mp4` |
| C3 | 0 / medium | all 9 utensils sorted by 38.6 s (horizon 45 s) | no process model; 9 correct, 0 wrong (quota 6) | each utensil pinched off the table, laid in its compartment and released; none dropped; reference utensils stayed | `data/household/x5_demos/C3/demo.mp4` |

Videos, at 25 fps with 250 Hz physics and 50 Hz control (re-recorded on the 0.60 m rig on 2026-11-03; C2 by its own session): A1 35.6 s, 890 frames; A2 28.7 s, 717 frames (real time); A3 60.2 s, 1504 frames (real time); A4 28.6 s, 715 frames (oracle speed-up, 2026-11-04); B1 6.0 s, 149 frames; B2 8.1 s, 203 frames; B3 14.0 s, 349 frames; C1 285.6 s, 7141 frames (the 542 s episode; waits with both arms still are 8× time-lapses); C2 47.2 s, 1180 frames (real time); C3 38.6 s, 966 frames. Time-lapses are labelled on the frame. All decode fully.

Robustness runs without video for A1 and A2 (task model and independent check both pass; A2 is the 2026-11-03 ice dispenser):

| Task | Seeds (medium) | Hidden parameters | Outcome |
|---|---|---|---|
| A1 | 0, 1, 2 | seed 0 nominal τ = 10 s; seeds 1–2 sample τ ∈ U[8, 12.5] s | extraction at 9.07 / 11.20 / 8.40 s, strength 0.596 / 0.594 / 0.598, all passed |
| A2 | 0, 1, 2 at easy, medium and hard | seed 0 nominal (0.9 s spin-up, 1.35 s interval); seeds 1–2 sample spin-up U[0.6, 1.3] s, interval U[1.1, 1.6] s, ±12 % per-cube jitter (the cadence before 2026-11-17; now 1.5 s nominal, U[1.4, 1.7] s, ±5 %, hard cadence ×1.15) | 9 of 9 pass (countertop dispenser, half a glass): final ice level 0.43–0.58 with 4 or 5 cubes, each equal to the deliveries owed; none outside. `roboclock eval --policy oracle --oracle` passes medium 0 and hard 1 |

On 2026-11-02, after the review changes, these also passed: A1, A2 (the oven task) and B3 at easy 0, medium 1 and hard 0 (B3 also after its 2026-11-02 simplification); B1 at easy 0, medium 1, hard 0 and hard 2; C3 at medium 1 and hard 0. C3 at easy 0 failed only because of the horizon described under Limitations (since resolved). C2 is listed in its own section.

On the 0.60 m rig (2026-11-03, headless recorder, seed 0): A1, A2 (oven, before its replacement), A3, B1, B2 and B3 also pass at easy and hard; A4 passes at easy; C3 passes at hard (7 of 9 sorted in 30 s, quota 6). C1 passes at easy, medium and hard on seeds 0 and 1 after its difficulty revision (below). B2 also passes at medium with each object (targets 0–3 by override; seeds 1, 4 and 8 without). A4 at hard failed on time only (kettle down at 38.4 s against the hard budget of 45 / 1.5 = 30 s); after the 2026-11-04 speed-up it passes, with the kettle down at 24.4–26.4 s (see the A4 section).

These are single episodes per seed of a fixed layout. They do not establish success rates, hardware transfer or learned-policy performance.

## How an episode runs

```bash
# on the GPU server, in a checkout under /opt/household with the kitchen linked
ISAAC=/opt/household/runtime_601/venv/bin/python
# 1) ground-truth validation without rendering
$ISAAC scripts/household/run_x5_episode.py --task A1 --out RUN_DIR
$ISAAC scripts/household/validate_x5_episode.py RUN_DIR
# 2) once it passes: record the same frozen configuration into a fresh directory, then re-check it
nohup $ISAAC scripts/household/run_x5_episode.py --task A1 --record --out REC_DIR > rec.log 2>&1 &
$ISAAC scripts/household/validate_x5_episode.py REC_DIR
```

- **Recording writes physics results to USD** (`enable_fabric(False)`, `/physics/updateToUsd`). With the Fabric-only path some recordings showed physics-driven prims frozen at reset while cue colours still changed. The runner aborts a recording whose frames stop following the arms.
- **Concurrent Isaac processes need separate `--cache` directories.** Sharing one Kit portable cache stalled a run.
- Every 10 s of simulation the runner writes the latest samples to `trajectory.partial.json`, so a native crash still leaves evidence. The frozen-render guard is checked from t = 2 s; the path tracer needs a few frames to show motion that starts at t = 0.
- `--spec key=value` overrides the hidden process parameters, for example `t_melt=-55` to start A2's browning early. It is for debugging only. Reported runs use the seeded spec.

**`run_x5_episode.py`:**
1. Writes `simulation_scene.usda`, a new layer over the unchanged R2 initial scene, via `x5/sim/physics.py` plus the task's `sim.build()`.
2. Sets the ready pose and settles passively for 3 s; the task clock starts after this.
3. Each physics step (250 Hz): measures the task signals, calls `task.update()` / `task.advance()`, applies appliance mechanisms, drives the visible cue (e.g. tea colour) and runs the oracle (50 Hz).
4. Writes `result.json` (task metrics, events, code hashes), `trajectory.json` (25 Hz samples) and, with `--record`, the RoboDojo-format videos `<ID>.mp4` (overview, 1280×960) and `head.mp4` / `left_wrist.mp4` / `right_wrist.mp4` (640×480), 25 fps, frame-synchronized.
5. Each video frame is a blocking zero-time Replicator capture after the physics step: a rendered `world.step()` left the camera on an older image, so the picture changed only every third frame (5 Hz in the old 15 fps videos). The capture does not change the physics; the validator rejects a video whose frames repeat.
6. While the oracle is only waiting on the process and both arms are still, the video keeps one frame in eight, and the header says "time-lapse". A moving arm always gets every frame.

**`validate_x5_episode.py`:** re-derives every predicate from the trajectory samples without using controller phases:
- **Timed event:** basket vertices above the tea for A1; the paddle press (≥ 5°) and release (≤ 3°) from the measured paddle angle for A2; the first sample with the pancake rotated ≥ 150° for A3; the start of the final released-upright hold of the kettle for A4; the lever at its top stop for B2; the lid seated on the rim (re-derived from the lid pose) for B3.
- **Process value:** recomputed from the hidden spec. For A2 the deliveries owed for the measured press are recomputed from the hidden spec and compared with the cubes counted in the glass from the measured cube and glass poses; A3 browning from the measured pan contact, the A4 dose from the measured kettle tilt (flow law) routed by the measured spout position, B2 char from the ejection time, and the B3 flame from the measured lid seating and knob angle.
- **Grasp and contact checks:** two opposing pads in contact during transport, contact support after release, low speed, containment, tilt.
- **Video:** all four videos fully decoded; sizes, 25 fps and equal frame counts checked; the overview must show a new image every frame. The result's rig check must have passed.

## Dual-arm rig (2026-11-03, `x5/rig.py`)

All ten tasks now use one rig:

- **Bases:** side by side at (0.00, −0.255) and (0.60, −0.255). They are 0.60 m apart centre to centre, at the same height (mount top z = 0.8704) and both face +y. Before this the bases were at (−0.06, −0.255) and (0.66, −0.255), 0.72 m apart. The midpoint did not move.
- **Head camera** (`/World/AgentCamera`, the policy's `rgb`): since the RoboDojo rig spec (below) it sits 0.04 m forward of the base midpoint and 0.543 m above the mount tops, at (0.30, −0.215, 1.4134), pitched 60° down, Gemini 345Lg intrinsics (640×480; rendered 96.0° × 79.6°, square pixels: RTX ignores the 14.266 mm vertical aperture, so the 71.0° vertical field of view that `get_intrinsics_matrix()` reports is not what is rendered). Before: 0.54 m, 55°, 18 mm lens on 30 × 22.5 mm. The pose is the same for every task; the per-task close-up cameras in `<id>_x5/scene.py` are gone.
- **One switch:** `rig.use(task_id)` sets the shared `BASES` dict before the scene, the physics overlay and the servos are built. `x5/build.py`, `x5/episode.py` and `run_x5_episode.py` all call it. `rig.RIG_TASKS` lists all ten tasks (C2 was moved in another session); a task outside it would keep the 0.72 m bases and its own camera.

Layout changes made because the arms moved 6 cm inward. Each was found by a headless oracle run and fixed by moving the object, not by changing the controller:

| Task | Change | Why |
|---|---|---|
| A1 | spoon basket and spoon moved from (0.46, 0.04) to (0.34, 0.10) | The spoon handle ended up 0.29 m from the right base. Every candidate handle grasp put a wrist joint at its stop, and the planner raised "no feasible grasp". At (0.34, 0.04) the fingers rolled the spoon out of the basket while sliding onto the handle. |
| A3 | cooktop footprint shrunk to 64 % (height, centre, burners and pan unchanged); turner moved from (0.354, −0.25) to neck (0.47, −0.14), turned 10° toward the left arm | At the old rest the handle grasp was 0.23 m from the left base and the arm took it turned backwards (joint 1 at −2.5 rad). The rest was also out of the head camera's view. On the freed counter strip in view, a turner along +x could not be grasped (a +x tool forward of the left base is out of reach); turned 10°, the IK margin is 0.30 rad. |
| A2 (oven, since replaced) | oven moved from x 0.30 to 0.44; trivet from (0.30, −0.285) to (0.12, −0.08), 2 cm narrower; the tray is carried out of the cavity, sideways over the open door, then down | The open drop-down door covers the counter in front of the oven, so the trivet sat beyond it and out of view. Beside the oven, the right hand's palm (±9 cm along the jaw axis) rubbed the oven's side and could not rise off the released tray (easy run); 8 cm in front of the oven line it clears. The right arm cannot pinch the door bar left of x ≈ 0.38, so the oven moved right and the trivet to its left. |
| B2 | wedge board moved 6 cm toward +x (toe at (0.40, −0.03)) | The cup lane's catch pose fell 7 cm from the left base and was out of reach. |

A4, B1, B3 and C3 passed unchanged.

**C1 (converted 2026-11-03, second round; R2.2, superseded by R3 on 2026-11-06).** With the 0.72 m layout, no placement of the staging mat and the serving board was both outside the side-hinged door's sweep and within the right arm's reach. The right arm reaches a bowl only beyond about 0.28 m from its base, so it cannot work in the front-right quadrant. The left arm, likewise, cannot work in the front-left quadrant. Only a 10 cm strip around (0.30, −0.25) is reachable by both. The layout was redesigned around those reach maps; the relay and the scoring are unchanged:

- **Microwave:** 75 % size, so the door is 0.24 m wide. It stands at (0.18, 0.17), turned 30° to face the right arm, and its door opens 90° toward the front left. The door springs 1.6 s after the release button is pressed and let go, by which time the right hand has backed 16 cm out of the swing: the button sits 4 cm from the door's free end, and a hand left in the swing had stopped the door.
- **Bowls:** 80 % size (10 cm). The 6 mm lug handles keep their size for the jaws, and the mass is scaled with the volume (0.14 kg). In the cavity the bowl sits centred, with handle A pointing straight out of the opening, so the right hand grips it at the opening without reaching in. The angled 132.5° grasp caught the opening's side and lintel.
- **Mat, cold bowls and board:** meal 2 starts on the mat at (0.30, −0.25). Meals 3 and 4 wait at (0.50, −0.09) and (0.64, 0.03), where the left wrist stays at least 0.17 m from the right base. The board is at (0.58, 0.27).
- **Right-arm routes:**
  - Mat to mouth: rise above the open door's top edge, pass its free end at (0.36, −0.10), descend to the mouth.
  - Mouth to board: straight out, along the microwave front 11 cm away (handle B points at it), around its right end.
  - Panel: approached in joint space from 25 cm above, because the Cartesian path from the park pose sagged about 9 cm at full reach and swept the cold bowls.
- **Door jamb:** the collider is cleared up to the 90° stop.

**C1 difficulty revision (2026-11-03; R2.2, superseded by R3).** Easy and hard failed for any controller, for two different reasons. The scoring, the relay and the medium schedule are unchanged.

| | easy | medium | hard |
|---|---|---|---|
| time scale (process rate) | 0.75 | 1.0 | 1.5 |
| `max_time` | 600 → **800 s** (600 / time scale) | 600 s | 600 → **400 s** |
| pickups (wall clock) | 240 / 400 / 560 / 720 s | 180 / 300 / 420 / 540 s | 120 / 200 / 280 / 360 → **100 / 200 / 300 / 400 s** |
| heating (wall clock, card 90 / 60 / 120 / 75 s) | 120 / 80 / 160 / 100 s | 90 / 60 / 120 / 75 s | 60 / 40 / 80 / 50 s |
| stand before pickup / cooling time constant | 40 s / 800 s | 30 s / 600 s | 20 s / 400 s |
| handling time per turnaround before the binding pickup | 73 s | 55 s | 50 s (was 34 s) |
| time to set the first timer before meal 1 must be done | 80 s | 60 s | 20 s |

- **Easy:** the horizon now scales with the process, like every other time in the task. The alternative, fitting the easy pickups into 600 s, would have moved easy's deadlines earlier than its slower heating allows and changed what easy means.
- **Hard:** with the pickups scaled, they come every 80 s. One microwave cycle is the heating plus a robot turnaround (release, unload, serve, load, door, timer) that does not scale with the process, so no relay could keep up. First, the oracle was made faster (below). Second, hard's pickups are now 100 s apart. Hard is still harder than medium on every axis: 29 % more meals per minute, 9 % less handling time per turnaround, a third of the time for the first timer, faster cooling and shorter standing.
- **Faster turnaround (run end to next start: about 50 s → 40 s):**
  - The right hand waits at the door button 3 s before the run ends and presses as the run stops.
  - The left arm waits outside the open door while the right one inserts the bowl; the door stands between the hands. It starts the push once the right hand is out.
  - The door is pushed at 0.11 m/s instead of 0.08 m/s.
  - While the door shuts, the right hand waits 10 cm above the panel.
  - Buttons are tapped at 0.08 m/s instead of 0.05 m/s, backing off 1.5 cm instead of 3 cm between presses.
- **Headless results** (recorder, seeds 0 and 1). Margin = pickup − (heating end + stand), per meal:
  - easy 0: 4 / 4, margins 76 / 116 / 59 / 79 s
  - medium 0: 4 / 4, margins 56 / 76 / 28 / 34 s
  - hard 0: 4 / 4, margins 16 / 36 / 15 / 25 s
  - seed 1 at each difficulty: 4 / 4 with the same timeline (hidden power 1.09, so 74–83 °C at pickup)
  - Rendered `roboclock eval --policy oracle --oracle` and the recorder (seed 0, each difficulty) pass with the same temperatures and start times as headless. The validator passes, including the image cadence. The easy and hard recordings, which are not delivered, are 8912 and 6273 frames.
  - The smallest margins are 15 % of the pickup spacing at hard, 23 % at medium and 37 % at easy. Every hot bowl is on the board 19 s or more before its pickup.

**Head camera view:** under the 60° RoboDojo head camera every key object and operation point of A1–A4, B1–B3, C2 and C3 is ≥ 10 px inside the image (`x5/visibility.py`, rendered square-pixel model, 96.0° × 79.6°) after the B3 and C2 moves above; the smallest margins are C2's outer glasses (19 px), B3's water glass (25 px) and C3's drawer (53 px). **C1 is the exception, accepted by decision (2026-11-04): the staging mat is seen by the wrist cameras only.** Under the rendered model the staging mat at (0.30, −0.25) and the bowl staged on it (`MealBowl2`) are about 13–19 px below the head image; the bowl's handle point is about +20 px, inside it. (Before 2026-11-04 these numbers came from the reported 71.0° model, which does not match the render: 55–62 px below, handle 17 px below.) `x5/visibility.py` lists both in `WRIST_ONLY` and reports their margins without failing. Under the reported model the mat had to be at y ≥ −0.165 to be in view; the mat is where both arms meet. Tried on 2026-11-04 (diff kept in `results/household/robodojo_2026-11-03/c1_attempt/`): at (0.30, −0.16) the right hand needs the bowl at yaw 90° (tool azimuth ≤ 180°), the folded left arm needs a joint-space lift after release, and the microwave door, springing open to 90°, still hits the staged bowl's body (body 0.21 m from the hinge, the door is 0.25 m long). Recorder hard 0 passed once, but `roboclock eval` failed at every difficulty when the door knocked a staged bowl. At (0.30, −0.12) the bowl stops the door. Clearing the sweep while staying in view needs the microwave hinge about 8 cm further back, which puts the panel and cavity out of the right arm's reach, so C1 keeps its layout. The staged bowl is seen only by the wrist cameras, when an arm works at the mat.

## RoboDojo rig spec (2026-11-03, `x5/robodojo.py`)

Every value comes from `roboclock.envs.robodojo_rig` ([spec](../robodojo-rig-spec.md)); the household code keeps no copies.

- **Rig frame:** origin at the base midpoint on the mount plane, (0.30, −0.255, 0.8704), yaw 90°. `x5/rig.py` derives the base layout and the head camera from it; `build.py` authors `/World/AgentCamera` with the exact spec pose (scene hashes changed for every task).
- **Cameras:** head `cam_head` → `rgb`; one D435 wrist camera per arm, a USD child of the arm's `link6` at `wrist_camera_in_link6()` (`cam_left_wrist` → `left_wrist_rgb` on the arm at x = 0.00, the rig's left; `cam_right_wrist` → `right_wrist_rgb`). All 640×480; RoboDojo's Camera API calls (focal length, both apertures with `maintain_square_pixels=False`, clipping). The wrist cameras are added to the opened simulation stage, not to the initial scene. `--depth` adds `depth`, `left_wrist_depth`, `right_wrist_depth`. **Rendered head intrinsics:** `get_intrinsics_matrix()` reports fy = 336.5 (from the 14.266 mm vertical aperture, 71.0°), but RTX renders square pixels: on a rendered B1 frame the mount plates' front edge lies at v 392, where fy = fx = 288.1 puts it, not at v 418. The head image therefore spans 96.0° × 79.6°. Results record both: `camera_intrinsics` (reported) and `camera_intrinsics_rendered` (square pixels, `robodojo_rig.intrinsics()`); use the rendered matrix for projection and depth unprojection. Results delivered before 2026-11-04 got `camera_intrinsics_rendered` added by `scripts/patch_rendered_intrinsics.py` without re-recording. The D435 wrist apertures are 4:3 like the image, so for the wrist cameras both matrices agree (fx = fy = 397.0).
- **Rates:** physics 250 Hz in every task (it already was), control 50 Hz, one render per 25 Hz policy observation (every 10th physics tick); environment steps in between return the latest render.
- **Renderer:** `rep.settings.set_render_rtx_realtime(antialiasing="DLAA")`, then `/rtx/rendermode = RaytracedLighting` (Kit 110's call selects RealTimePathTracing) and Isaac Lab's quality preset `RENDER_SETTINGS`; applied again after the stage's first frames, then read back into every result (`renderer`, with any setting that did not stick). The recorder used RealTimePathTracing before.
- **No-render runs:** `x5.no_render: true` (or `sim.no_render`), file `configs/no_render.yaml`: no camera prims or objects, no render calls; every observation carries zero images of the same shapes.
- **Rig check at every reset** (`robodojo_rig_check` in every result): both `base_link` frames from physics and the head camera from the stage go through `check_rig`; each wrist camera is measured relative to its measured `link6` (1 mm / 0.1°); the measured `link6` is compared with X5A.urdf's `link6` at the measured joints.
- **Known deviation:** the household arm is the arx5-sdk `X5.urdf`, not RoboDojo's `X5A.urdf`. Joint axes and `link6` orientation are identical, but joint1/joint2 heights and the joint5/joint6 offsets differ, so `link6` sits up to 3.0 mm away from X5A's at the same joints (2.9 mm at the ready pose). Every rig check reports it as `link6_vs_x5a.matches = false`; the wrist cameras are mounted on the household `link6`.
- **Visibility** (`x5/visibility.py`): every audited object's bounding box and every operation point projects ≥ 10 px inside the head image under the square-pixel model RTX renders (`conservative=True` also checks the reported model, apertures honoured, and keeps the smaller margin). C1's staging mat and the bowl on it are wrist-camera-only by decision (`WRIST_ONLY`). Layout changes this needed: B3's water glass (0.80, 0.02) → (0.80, 0.10); C2's bottles (−0.10, −0.04) / (0.70, −0.04) → (0.06, 0.00) / (0.54, 0.00) with pickup azimuth 75° / 105°. C1's staging mat is wrist-camera-only by decision (see "Head camera view" below). Independently of the camera, A3's pancake now starts 5.3 cm (was 4.5 cm) off the pan centre against the turn: through `roboclock eval` (float32 actions) hard 0 landed it on the wall slope at 157° and it never settled.
- **Recordings:** `head.mp4`, `left_wrist.mp4`, `right_wrist.mp4` (640×480, exactly the policy images) and the overview `<ID>.mp4` (1280×960, the three policy views as insets), all 25 fps with equal frame counts; `frames.json` lists each frame's simulation time (idle stretches keep 1 frame in 8 in all four). Stills: `initial.png`, `final.png` and `initial_cam_*.png` / `final_cam_*.png`.

## Robot model (shared, `x5/sim/physics.py`)

- **Arms:** two six-DOF X5 chains from the pinned arx5-sdk URDF, with PhysX position drives (stiffness 3000, damping 90) bounded by the SDK effort and position limits. Self-collision is disabled within each arm; arm–arm and arm–scene contacts are active.
- **Gripper:** the native link6 CAD is split into a palm and two fingers on approximate symmetric prismatic jaws: 60.3 mm opening, 30.14 mm travel per side, **20 N per side**. The 20 N figure is an assumption consistent with the arm's 1.5–2 kg rated payload, not a published X5 gripper value.
- **Finger collision:** each finger has a proximal hull and a 15 × 1.5 × 10 mm distal pad. The finger bodies are about 6 cm tall plates and the palm spans ±9 cm along the jaw axis; both constrain where the gripper fits.
- **Servo (`x5/sim/servo.py`):** bounded Cartesian steps solved by URDF IK, or a damped Jacobian, with rate-limited joint targets. `grip(squeeze=…)` sets the closing target past the closed stop, so thin parts are clamped at the force limit. `grip_each()` closes each jaw separately until it touches.

## A1: sugar for a glass of tea (R3, 2026-11-19)

The tea-infuser episode was replaced (x5-tasks.md § 18). The jar is a dynamic hollow compound (base disc, 24 wall boxes, shoulder ring, neck tube; 200 g with its sugar); the glass and the basket are fixed fixtures. Sugar is the process model in task.py, driven by the measured jar tilt and the measured landing point of the stream.

1. The left arm pinches the jar body 3.5 cm up and lifts it to 10.5 cm above the counter.
2. It carries the jar upright beside the glass and tilts it toward the glass while the mouth moves over it; the tilt rises at 45°/s to the first flow, is servoed to 2.5 g/s and tips back when the grams reach the middle of the band (privileged).
3. The jar goes back to its spot, upright, and is released; the arm returns home.
4. The right arm, which picked the spoon up meanwhile, stirs two measured clockwise turns in the glass and returns the spoon to the wire basket.

The mound, the stream and the sugar in the jar are drawn from the task's grams (`Oracle.update_cues`, by scale). The right-arm notes below carry over from the tea version.

Fixes needed for R2:
- **Spoon grasp:** chosen with a joint-limit margin; the symmetric jaws allow a grasp rolled 180°.
- **Spoon approach:** the fingers slide onto the flat handle along its axis. A vertical descent sweeps a finger through the tilted handle.
- **Spoon release:** one pad lies under the handle and the wrist is near joints 5 and 6's stops, so a fast Cartesian retreat clipped the joints, swung the hand up and flung the spoon about 100° round in the basket. The hand now slides 2.5 cm off the handle at 3 cm/s, then returns to the ready pose in joint space, and the spoon does not move.

## A2: oven cheese toast (layout R2.1)

The R2 layout had no physically valid solution, so A2 was revised; see [x5-tasks.md § 6](x5-tasks.md). The instruction, process model and scoring are unchanged. The episode runs:

1. **Door.** The right arm pinches the handle bar and swings the drop-down door to 45° with it, then releases. A damped hinge lets the door settle open at 84°. Opening the door slows browning to ×0.4.
2. **Rack.** The rack is linked to the door and auto-ejects as it opens, an appliance mechanism common on countertop ovens. This brings the tray about 10 cm out of the oven, while its back half stays in the cavity and keeps browning.
3. **Pinch.** Each arm pinches the middle of its own 2.2 cm short rim (x = 0.185 and 0.412), with the tool 68° forward-down and the jaw axis on the rim normal. Each jaw closes until it touches, then both clamp at about 15 N.
4. **Lift out.** When the browning cue reaches 0.60, both arms lift the tray out over the door. This is the timed retrieval event.
5. **Place.** The arms set the tray on the trivet, release, rise 6 cm and return to the ready pose.

Geometry choices forced by the X5 and the oven, found during development:

- **Oven shell:** box colliders around an open cavity, because a hull of the body mesh would fill the cavity. The shell is held 3 mm behind the closed door, so the door does not jam.
- **Door collider:** raised 4.3 cm, because the hinge axis is only 1 mm above the counter.
- **Handle approach:** a narrow 22 mm pre-opening, because the bar sits only 2.2 cm in front of the door panel.
- **Rim pinch:** the jaw axis must lie exactly on the rim normal; a tilted jaw axis pushed or lifted the tray. Each pad must close until contact; independent jaws otherwise shove the tray before the second pad arrives.
- **Short-rim grasp:** pinching 1 cm off the tray's centre let it droop 0.35°/s in the pads, so the pinches sit on the tray's y centre. The 68° pitch is a narrow window: steeper hits the oven front with the wrist (1.8 cm clear), shallower runs joint 6 into its stop near the trivet (±1.95 of ±2 rad).
- **Clamp force:** with both jaws at the 20 N drive limit nothing centres the tray, and it chattered along the line through the pinches; the toasts walked 3–4 cm with no score penalty. A 15 N clamp keeps the jaws' position drive active and stops this.
- **No robot pull-out:** a hooked pull of the tray pressed on or wedged against the low rim and tipped the tray in some seeds. This is why the rack auto-ejects instead.

## A3: pancake flip (layout R2.1)

The R2 layout had no physically valid solution, so A3 was revised; see [x5-tasks.md § 8](x5-tasks.md). The pan is a static triangle-mesh collider. The pancake is a rigid 45 g disc whose lower edge is bevelled, so a thin blade can wedge under it as under a real pancake's thin edge. The turner is the native mesh bent 30° at the neck, a 60 g rigid body with a thin bevelled blade collider. The task model integrates the underside browning while the pancake lies on the pan; the underside colour follows it and shows once the pancake is turned. Only the left arm works:

1. **Turner.** The arm moves in joint space above the handle, inserts to within 1.5 mm of the grasp point, closes each jaw until contact and ramps the clamp. It lifts the turner and holds it over the front of the pan.
2. **Wait.** When the browning cue reaches 0.61, the arm lowers the blade to the pan floor.
3. **Slide.** The blade slides under the pancake along a 50° approach line. The tip follows the local floor height; the blade pitches up only where the pan wall rises behind the tip.
4. **Lever.** The blade rotates to 85° in 3 s about the pancake's left edge. The pancake stands on that edge, leaning on the blade.
5. **Topple.** The blade pushes sideways, closed loop on the measured pancake angle, until it passes 120° (at most 8 cm). The pancake falls over, turned.
6. **Withdraw.** The blade lifts 8 cm and pulls back, and the arm parks the turner off the pan.

Findings that shaped the controller:

- **Tunnelling:** at the lever the pancake was pushed through the bare triangle-mesh floor. A 12 mm cylinder collider now backs the pan floor.
- **Friction:** the blade has μ 0.4 with min combine, the pancake μ 0.8. At μ 0.2 the pancake slid off the blade during the lever.
- **Grasp:** after both pads touch, the clamp ramps up by 0.6 mm of jaw target per tick; jumping straight to full clamp flicked the round handle out. A grasp 16 cm along the handle squirted out, and one nearer the blade put the hand on the rim, so the grasp is at 14 cm.
- **Lever angle:** beyond about 88° the lever is out of reach, and a lever around the far edge threw the pancake out of the pan; 85° about the side edge, then the topple push, is deterministic.

## A4: two-arm kettle pour (layout R2.1)

The R2 single-arm pour had no physically valid solution, so A4 was revised; see [x5-tasks.md § 9](x5-tasks.md). The kettle is a rigid body: a 0.8 kg shell plus the remaining water, its mass updated every 0.5 s. Its colliders are a body hull, the lid knob and three boxes around the handle that leave the handle opening free. Water is the task's scalar flow model: the stream, the jug water level and the scale's 7-segment reading follow the task cues. The episode runs:

1. **Approach.** The left arm, then the right, moves in joint space at 2.5 rad/s: it turns its base while tucked, unfolds up to 15 cm above the pre-grasp, then descends. Moving both at once made the palms meet mid-air.
2. **Grasp.** The left arm pinches the handle bar with one pad through the handle opening; the right arm pinches the lid knob from the front-right, tool 20° down. Each jaw closes until contact, then both clamp.
3. **Raise.** Both tools follow the kettle target through the recorded grasps (ramps 0.35 m/s², 1.2 rad/s²): up 10 cm at 0.10 m/s, then over to the jug at 0.12 m/s with the base 5 cm above the jug rim.
4. **Pour.** The tilt command ramps at 40°/s to 8° below the pour onset, then at 18°/s to 32° past it (at most 88°), rate-limited at 40°/s². The onset rises as the kettle empties, so the tilt follows it (about 35 ml/s). The spout tip moves down to 3 cm above the rim and 2.5 cm short of the jug axis; a slow integrator on the measured spout removes the 1 cm sag of the held kettle.
5. **Stop.** The arms tilt back at 20°/s once the privileged jug mass plus the predicted tail reaches 299 g. The tail is 0.4 × flow × tilt-back time plus the rate reversal: flow ∝ excess^1.5 and the excess falls linearly.
6. **Set down.** Once the stream has stopped and the tilt is 30° below the onset, the kettle turns the rest of the way upright (0.6 rad/s) while it is carried back to 6 cm above its rest spot; it is then set down at 0.04 m/s and released. The tools back 6 cm straight out along their axes, then rise 4 cm.

**Speed-up (2026-11-04).** The hard budget was then 45 / 1.5 = 30 s (36 s since the 2026-11-16 recalibration), and the oracle had set the kettle down at 38.4 s. Changes, old → new: approach 1.2 → 2.5 rad/s; insert 0.03 → 0.06 m/s; lift 0.06 → 0.10 m/s; carry 0.08 → 0.12 m/s; carry ramps 0.25 m/s² / 1.0 rad/s² → 0.35 / 1.2; pour 25° past the onset at 12°/s (about 24 ml/s) → 32° at 18°/s (about 35 ml/s, capped at 88°); tilt back 12 → 20°/s; tilt ramp 30 → 40°/s². The kettle no longer returns upright over the jug and is no longer carried to the 10 cm lift point before set-down. Kettle down, seed 0: 38.4 → 25.3 s, at every difficulty; seeds 1 and 2: 24.4 and 26.4 s. The smallest hard margin is 3.6 s (12 %), at seed 2. Dose 300.2 / 298.6 / 298.8 g. The task, target, tolerance, budgets and scoring are unchanged. The instruction now states the budget scored at each difficulty (easy 60 s, medium 45 s, hard 30 s); it said 45 s at every difficulty.

Findings that shaped the controller:

- **Elbow margin:** with the pour point at the jug axis and 7 cm above the rim, the left elbow reached its limit (3.18 of 3.2 rad) at 77° tilt. The arms then fought, and the right pinch let go of the knob as the tilt reversed.
- **Tilt rates:** tilting back at 60°/s, or reversing the tilt rate in one step, pulled the knob out of the right pinch and spilled. Hence the rate limits (20°/s back since the speed-up) and the rate ramp.
- **Retreat (speed-up):** the held kettle slips 3–6° in the pinches during the pour, so it lands slightly tilted. The old retreat moved back 8 cm and up 3 cm at once; that lifted the left pad into the handle's upper arm (about 1 cm clearance) and, in `roboclock eval` runs, tipped the kettle over and spilled. The retreat now backs out straight, then rises. Handing over to the carry at 15° below the onset (with a 1.5 rad/s turn) also slipped the grasps; the handover is at 30°.
- **Raise first:** at zero tilt the spout reaches only 1.5 cm past the body. Carrying the kettle straight to a spout-over-jug pose drove its body into the jug and knocked it out of both hands.

## B1: running faucet, filling bowl

A 1.6 L mixing bowl stands in the empty sink right under the spout and starts with about 0.4 L in it. At the faucet's 0.10 L/s (medium; 0.075 L/s at easy, 0.138 L/s = 8.3 L/min at hard, the US federal maximum for kitchen faucets of 2.2 gpm, since 2026-11-16) the level rises about 4 mm/s, so the overflow at 12 s (8–14 s across seeds) can be seen coming, where a full sink rose a fraction of a millimetre per second. The task was changed to this on 2026-11-02.

The scene's native single-lever handle (`FaucetLever`) is a 0.08 kg rigid body on a damped revolute hinge about y, 0–15°, balanced on its pivot so the cartridge friction holds any position. Flow is proportional to the measured tilt. The task model integrates the water in the bowl and the spill. The water surface, bubbles, stream, and the streams over the rim and puddle of an overflow follow its cues; level and surface radius come from the bowl's inner profile, measured from its mesh (`BOWL_LEVEL_CM`, `BOWL_RADIUS_CM`, `BOWL_VOLUME_L` in `b1_x5/task.py`).

1. **Approach.** At t = 0 the right arm moves in joint space to 4 cm on the +x side of the open lever, tool pitched 45° down toward the spout.
2. **Push.** The tool follows the lever's closing arc on the rod's +x face, 2 cm below its tip, leading the measured tilt by 4°, until the tilt is below 0.3°.
3. **Release and retreat.** The arm backs off before the closure counts (no hand contact for 1 s), then returns to the ready pose.

- **Bowl:** the `bowl_1` asset is a shallow bowl, stretched to 23.5 cm across and 15 cm tall. It is a fixed collider, so the water adds no load and a push does not move it.
- **Basin:** the island's cabinet panels used to fill the basin, hidden under the old water plane. `cut_island_top(opening=...)` removes them, so the sink floor is the real one.
- **Render:** a removed stream stays on the transparent water for seconds in the renderer's temporal history, so the second after the flow stops is rendered with 12 subframes per frame (zero simulation time). A faint trace remains on the water for about a second.

The validator re-integrates the spill from the recorded lever angle. Closure requires the flow to be shut with no hand contact for 1 s, after the robot has touched the lever.

- **Push point:** the lever is a 1 cm rod on a round cartridge base. Pushing at the bounding-box top skimmed over the rod, so the push point is taken from the native mesh (rod face x 9–19 mm, tip at 91.6 mm).

## B2: rolling object edge catch

Four objects rest on a wedge board, each behind a peg: cup, egg, pill bottle and apple. One peg sinks at a hidden time, 1–3 s. From then on the object moves by gravity, contact and friction alone. It gathers speed on the 27 cm board, rolls across the counter and leaves over the front edge unless it is stopped.

| Difficulty | Slope | Release to edge, arms idle (cup, egg, bottle, apple) | Speed at the edge |
|---|---|---|---|
| easy | 4.5° | 2.32, 1.68, 1.80, 1.60 s | 0.38–0.54 m/s |
| medium | 6° | 1.96, 1.56, 1.60, 1.40 s | 0.45–0.60 m/s |
| hard | 9° | 1.60, 1.32, 1.28, 1.16 s | 0.55–0.71 m/s |

Every idle run ends with the object on the floor.

1. **Detect.** The arms do not move until an object's measured speed passes 5 cm/s, about 80 ms after release. That object is the target.
2. **Intercept.** The left arm moves in joint space at 3 rad/s to the waiting line, y = −0.315, in about 0.8 s. The tool points at azimuth 60°, 55° below horizontal, so the upper jaw clears the object and the lower jaw is in its path.
3. **Track.** The hand slides along the line to where the measured velocity crosses it. The egg needs this: its heading ranged from 239° to 252° in the idle runs, against 240° for the others.
4. **Stop.** The object runs into the lower jaw or the palm.
5. **Seat.** The hand brings the pinch point to the object's centre, across the mouth and sideways first. Going straight for the centre landed the upper jaw on the apple at 35 N.
6. **Close and lift.** Each jaw closes until it touches, both squeeze to 10 N, the hand lifts 6 cm and holds. The right arm only turns out of the camera's line to the board.

Results from `roboclock eval` (seeds 0–7 medium, 0–5 hard, 1–4 easy, plus the cup at seed 8): 20 of 20 succeeded. The object stopped 4.2–7.2 cm short of the edge along its heading, with peak jaw force 11.3–14.5 N. All 12 object and difficulty pairs also pass in the recorder. The `hold` policy fails: the egg falls at 3.62 s.

- **Sizes:** the 60.3 mm jaw opening sets them. The objects are 41–52 mm across, so the apple is a small one and the cup a small tumbler.
- **Heading:** the rolling direction is diagonal because the X5 wrist cannot aim the hand straight back from beyond its own base. Only the left arm reaches the catch poses.
- **Rolling resistance:** PhysX has none, so angular damping of 0.12 /s stands in for it.
- **Egg:** three spheres on its axis, so it rolls smoothly and drifts, but it does not precess like a real egg.
- **Crushing:** a contact-force limit only. Nothing breaks.
- **Peg:** a scripted disturbance, and the only scripted motion.
- **Learned policies:** the oracle knows the object's true pose and speed with no latency. On hard the arm needs 0.8 s of about 1.2 s, which leaves roughly 0.3 s for perception and planning. An LLM in the loop cannot meet that.

## B3: pot fire, burner off

Simplified on 2026-11-02 at the user's request: no lid. The oil in the pot on the centre burner starts without flames and catches fire at a hidden random time, uniformly sampled from 2–8 s after the first observation (seed 0: 5 s). The seed reproduces the ignition time at every difficulty. The robot waits for visible flames and then turns the burner off, and the flame dies within a few seconds (an abstraction: cutting the heat does not put out real burning oil, and real guidance also covers the pan). The lid version (cover, then knob) is in git history and in `results/household/x5_demos_2026-11/B3_lid`.

The pot is a static triangle-mesh collider. The centre knob is a 30 g rigid body on a damped revolute joint about z, from 90° (on, as placed) to its OFF stop 90° clockwise. The task model grows the flame logistically while the knob is above 10° (spread at 0.8, about 48 s nominal after ignition) and decays it with a 1 s time constant once it is off; the flame cards follow that cue. Success: knob at OFF by contact, hand released, flame out for 3 s.

1. **Approach (right).** After the flame cue appears, the right arm moves in joint space above the knob, tool straight down, jaw axis at −135°.
2. **Pinch and turn.** It descends, pinches the knob, and joint 6 alone turns it 87.5° clockwise in one move. Joint 6 stops at 2 rad, so the turn ends 2.5° short of the knob's stop, inside OFF.
3. **Release and retreat.** The arm opens, rises and returns to its ready pose. The left arm stays at its ready pose throughout.

The current recording is `data/household/B3/demo.mp4` (medium, seed 50): ignition at 4.228 s,
heat off at 9.864 s, secured at 15.508 s. Its `demo.json` and HDF5 contain the sampled ignition time.
The older `data/household/x5_demos/B3` recording and catalog scene hash predate delayed ignition;
they remain historical references and do not match the rebuilt initial scene.

- **Hand clearance:** at jaw azimuths −133°, −127° and −120° the hand stalled 1.3–2.6 cm above the pinch, next to the pot's front handle, so the turn starts at −135°.
- The glass of water stays on the counter as a trap; water on the oil fire is a failure.

## C1: serving relay at the pass (R3, 2026-11-06)

R3 replaces the R2.2 microwave relay (600 process seconds) with a 60 s serving relay; see [x5-tasks.md § 16](x5-tasks.md). The bowls, handles, grasps, staging mat and serving board are R2.2's. Before the episode starts the bowls are moved to the seed's layout (`task.layout`).

1. **Left arm, counter to mat.** It takes the next wanted meal by handle B from its counter slot (joint-space approach: tuck, turn the base, unfold 10 cm above, descend), lifts it straight up to the transit height, and carries it to a waiting point at (0.14, 0.04), along an arc for the far slot. If the mat is free it goes on at once (from the near slot straight to the mat, without stopping); otherwise it waits there. It sets the meal down on the mat, releases, backs off 5 cm and frees the mat.
2. **Right arm, mat to board.** It waits beside the mat, 3 cm above its pre-grasp pose. Once the meal is staged and the mat is free, it takes it by handle A, lifts it and carries it in one straight move to beside the board, (0.48, 0.14), 16 cm from the board bowl. It frees the mat once the bowl is 15 cm from the mat's centre. It holds the meal there until the previous bowl has been taken, then sets it down (short back-off, then straight up, as in R2.2) and goes back beside the mat.
3. **Waiter.** At each pickup time the bowl resting on the board is moved off the counter.

The relay delivers a meal beside the board about every 12.8 s. Each meal is on the board about 3.2 s after the board frees up.

**Results** (physics only, through the roboclock environment and `roboclock eval`'s oracle path; the independent check of `scripts/household/validate_x5_episode.py` re-derives every pickup from the 25 Hz probe trace and agreed with the task model on the four runs it was applied to: medium 0 and 11, easy 12, hard 13):

| Difficulty | Seeds | Orders qualified | Smallest slack |
|---|---|---|---|
| easy (80 s) | 0, 12 | 4 / 4 each | 12.7 s |
| medium (60 s) | 0, 1, 3, 7, 11 | 4 / 4 each | 6.9 s |
| hard (40 s) | 0, 1, 13 | 3 / 3 each | 2.4 s |

No bowl dropped (largest tilt 4.2°) and no oracle failure. The reference recording (`data/household/x5_demos/C1/`, `scripts/household/run_x5_episode.py --record`) still holds the R2.2 episode. On the pod where R3 was built, the runner stalled in its 3 s settling loop for C3 as well, so recording R3 is left to a machine where the runner runs. Until then `test_designs_scene_builders_and_demos_exist` and `test_initial_scenes_rebuild_byte_identical_to_the_recordings` fail for C1 (and, since the six-utensil scene of 2026-11-09, for C3). The table above predates the C-class judging and its schedule (x5-tasks.md § 17; below).

**C-class schedule (2026-11-09).** Four orders at every tier, picked up at 4 / 2 / 1 × the hard schedule, a first pickup at 10.53 s and one every 12.96 s (x5-tasks.md § 17 has the derivation). Measured with `roboclock eval`'s oracle on seeds 0–6 and 50–56, first with every pickup moved to 0.5–2 s, so that the waiter took each meal 1 s after its set-down (the late rule) and the board never held the oracle back (`scripts/household/c1_oracle_free_run.py` overrides `pickups_s` and logs the oracle's phases): all 14 physical, first meal on the board at 10.524 s on every seed, later ones 12.52–12.94 s apart (3.244 s from beside the board to set down), slowest per order 10.524 / 23.464 / 36.344 / 49.204 s (seed 2; seed 54 ties the second). With an interval of 12.94 s, seed 2's second meal landed at 23.484 s at hard, 0.014 s after its pickup: taking the first meal 1 s earlier than in the free runs shifts later set-downs by up to 0.1 s. Hence 12.96 s. The runs are deterministic: seed 2 at hard twice gave identical times. Validation at the final schedule, through `roboclock eval` (`--policy oracle --oracle`):

| Tier | Seeds | Set-down times | Smallest slack | Result |
|---|---|---|---|---|
| hard (10.53 / 23.49 / 36.45 / 49.41 s, H 61.76 s) | 0–6, 50–56 | order 2: 23.064–23.484 s; order 3: 35.624–36.324 s; order 4: 48.544–49.144 s | 0.006 s (order 1, every seed; order 2, seed 2) | every order by its pickup, physical, `all_done` at 51.41 s, all 14 |
| medium (21.06 / 46.98 / 72.9 / 98.82 s, H 123.53 s) | 0, 2, 54 | 10.524 / 24.324 / 50.244 / 76.164 s on each seed | 10.54 s | every order by its pickup, physical, `all_done` at 100.82 s |
| easy (42.12 / 93.96 / 145.8 / 197.64 s, H 247.05 s) | 0, 2, 54 | 10.524 / 45.384 / 97.224 / 149.064 s on each seed | 31.60 s | every order by its pickup, physical, `all_done` at 199.64 s |

At medium and easy the oracle holds each meal beside the board until the previous pickup (up to 39 s at easy) and sets it down 3.264 s after it, so its times no longer depend on the seed. No bowl dropped and no oracle failure.

Findings that shaped the controller:

- **Region in front of each base:** no handle pose is reachable within about 0.28 m in front of an arm's own base. A right-arm waypoint at (0.44, −0.10) put the grasp point 0.15 m from the right base, and the arm stopped there. Every waypoint was screened offline with the oracle's IK (`x5/sim/servo.py` Kinematics) before it went into the simulation.
- **The mat is the shared resource:** with the mat freed only after the right arm reached the board side, and after the left arm's tuck, each bowl held the mat for about 18 s. Freeing it by the held bowl's distance (right) and right after the back-off (left), waiting beside the mat (right) and skipping the waiting point when the mat is free (left) brought the cycle to about 12.8 s.
- **Speeds:** empty-hand joint moves at 2.4 rad/s, empty-hand Cartesian moves at 0.28 m/s, descents to the place at 0.10 m/s, loaded carries at 0.14–0.18 m/s. The carry ramps (0.25 m/s², 1 rad/s²) are R2.2's.

## C2: two arms pour cola into six glasses (R4, 2026-11-03)

R4 changes R3 on request: the glasses stand in one row, the cooktop and pan are gone (bare island), and the cola is a fluid simulation. C2 is on the common rig (bases 0.60 m apart).

- **Fluid:** a PhysX position-based particle fluid on the GPU: 3 mm particles (about 27 µl each), water-like material, rendered as an isosurface. About 310 ml starts at rest in each bottle. It flows out of the neck only when the bottle is tilted far enough, falls as a stream, and settles in a glass or on the counter. The particles carry their own mass, so the arm holds a heavy bottle that gets lighter.
- **Measurement:** every scored volume is a particle count at 50 Hz: in each glass (inside its cylinder, below the rim), in each bottle, in the air, and spilled (on or below the counter outside the glasses). A glass is served once it has held 80–105 ml (line 90 ml) for 1 s with nothing in the air.
- **Hollow colliders:** bottles and glasses are rigid compounds (a base disc, a ring of 24 wall boxes and, for a bottle, a shoulder ring and a neck tube), so the fluid sits inside them.
- **Layout:** six tumblers 10 cm apart in one row at y = 0.10 m, three outside each base; each arm's bottle starts in front of its base, on the inner side: (0.06, 0.00) and (0.54, 0.00) since the RoboDojo head camera (before: (−0.10, −0.04) and (0.70, −0.04), outside the 60° view); pickup tool azimuth 75° (left) and 105° (right).

Controller (privileged, the same routine for both arms at once):

1. Pinch the bottle body 12 cm up, tool 25° down, and lift it with its base above the glass rims.
2. For each glass, inner first: carry the bottle beside it and tilt it outward along the row while the mouth moves over the glass. The bottle turns about its own axis between glasses, so each pour uses its reach-screened tool azimuth (≥ 0.36 rad joint margin). The bottle is raised wherever any part of it would come within 12 mm of another glass's rim.
3. Raise the tilt quickly until cola flows. Then servo the tilt to a measured outflow of 30 ml/s, and tip back once the glass, the cola in the air and 0.35 s of flow reach the line.
4. Stand the bottle back, open fully and go home.

Findings that shaped the controller:

- **Inner glass:** the glass next to each base is reachable only up to a 100° tilt. It is poured first, from the full bottle, which flows from about 90°; the outer glasses allow 125°.
- **Bottle body over a glass:** at 70–80° the bottle's body spans the glass poured before. A clearance check on the base alone let the body rest on that rim, the mouth rose 3.5 cm and the stream overshot; the whole bottle is now checked.
- **Aim:** the stream leaves the mouth moving outward, so a raised mouth aims further back, toward the bottle.

| Path | Seeds | Outcome |
|---|---|---|
| Recorder | medium 0 | 6 of 6 served (84–92 ml), spill 0; all served and both bottles back at 44.1 s (window 90 s) |
| Recorder | hard 0 (window 60 s) | 6 of 6 (86–92 ml), spill 0, finished at 44.4 s |
| Recorder | easy 0 (window 120 s) | 6 of 6 (85–92 ml), spill 0, finished at 43.8 s |
| `roboclock eval` | medium 0 | 6 of 6 (87–92 ml), joint success, ended at 43.9 s |
| `roboclock eval` | medium 1 (no hidden parameters: same episode) | 6 of 6, ended at 44.4 s |
| Recording | medium 0 | 6 of 6 (86–91 ml), spill 0.03 ml, finished at 44.1 s; 47.2 s video, every frame a new image |

## C3: cutlery organization (R3, 2026-11-02)

C3 replaced "clear two place settings, then wipe each mat" on 2026-11-02; the earlier episode is in `results/household/x5_demos_superseded_2026-11-01/C3` and the 2026-11-03 superseded folder. Nine loose utensils (three spoons, three forks, three knives; rigid bodies of 40–45 g with box colliders: a shared 16 × 8 mm handle, a neck and a head box per kind) lie flat on the counter. The open drawer is static: floor, outer walls and a three-compartment insert, with one reference utensil per compartment. The seed picks the table slot of each utensil and the compartment order (`scene.layout(seed)`). A utensil counts as sorted while it rests inside the compartment of its kind, below the wall tops, touched by no jaw. The score is taken at the horizon, 45 s / time_scale.

**Oracle.** Each arm sorts the utensils on its side of the drawer, the ones nearest the drawer first. For each utensil: a joint-space move above it, a top-down descent with the jaw axis across the handle, close until both pads press, lift, a Cartesian carry above its compartment (turned so the utensil lies along it), lower onto the stack, open, rise. Only one arm is over the drawer at a time.

- **Reach:** a top-down pinch at the counter reaches about 0.46 m from each base, so the far slots and compartments are approached at 6–9 cm height instead of 13 cm.
- **Carry:** a joint-space carry dipped the held utensil into the drawer walls (56 N, knocked out), so the carry is Cartesian.
- **0.60 m rig:** C3 passed unchanged on the new rig (9 of 9 sorted at medium seed 0).
- **Six utensils (2026-11-09):** under the C-class rule every loose utensil is a job, so the table now holds two of each kind in the six slots nearest the drawer (references Spoon3, Fork3, Knife3; `initial_scene_sha256` 70703ee0…). The recorded demo and the results above are the nine-utensil scene; the six-utensil oracle run and recording, and the re-measured t_o, are pending (x5-tasks.md § 17).

## Limitations

- Each task is one fixed layout with a privileged oracle. No learned or LLM policy is involved, and there are no success-rate statistics.
- Liquids, heat, browning and cheese are symbolic process models with colour cues. The toast is a rigid box. The A3 pancake is a rigid disc, and A4's water is a scalar flow with a cylinder stream; the kettle's mass follows it, but the water does not slosh. B2 smoke and the B3 flame are cards scaled by the task cues. C1 food temperature is symbolic (no colour cue); the microwave interior light marks a run. C2 squares are rigid slabs with symbolic per-side browning (face colours follow it). C3 residue is symbolic, integrated from the sponge's measured pressed sweep; the stains are scaled by it, and the sponge is a rigid body.
- `roboclock eval` and the recorder are not bit-identical. On the former C2 (French toast) they first differ at t = 0.04 s, by 1.2e-7 rad (one float32 step), and contact amplifies that to 1 mm by 9 s, so a marginal contact can end differently on the two paths. A pass on one path does not prove a pass on the other.
- Resolved 2026-11-03: easy horizons. C1 and C3 used to fail at easy for any policy, because their deadlines stretched with the time scale but `max_time` did not (C1's 4th pickup at 720 s against 600 s). C3 now sets `max_time` = 45 s / time scale and C1 `max_time` = 600 s / time scale (easy 800 s), so every pickup lies inside the episode. C2's window already scaled and ends the episode.
- The gripper is an approximate CAD linkage with an assumed 20 N clamp. Arm inertias use diagonal URDF terms.
- A2's oven is a 0.6-scaled wall oven. The door balance, hinge damping, runner travel and door–rack linkage (a drive tracking the door angle) are modelling choices, not measured appliance data.
- C1's waiter moves the bowl on the board off the counter by a pose write at each pickup time. This is the only pose write in an episode after setup (the seeded layout is applied before the simulation starts).
- B2's peg is a scripted disturbance. The B3 pot is a static collider, so its displacement is not modelled; the other four burner knobs are static, so a wrong knob cannot be turned.

## A2 countertop ice dispenser (R3.1, 2026-11-03)

- **Scene physics:** static box colliders for the dispenser body around its recess, the recess back wall and the drip tray. The paddle (native Refrigerator001_Switch, ×0.85) is a 20 g rigid body on a revolute hinge about x at its top pivot, 0–12°, with a return spring of 0.03 N·m/deg preloaded 2°. The glass is a free 200 g hollow compound (base disc and 24 wall boxes). The cubes are 8.5 g rigid crescents (convex hull, friction 0.15/0.10, CCD) waiting on a hidden shelf in the bin.
- **Dispenser mechanism** (`appliance_step`, not control): for each delivery of the task model, the next waiting cube is placed just inside the outlet, over the pressed glass's axis, moving down at 0.3 m/s; from there it falls freely. The recess lamp is on while the paddle is held in.
- **Measurement:** the paddle angle from its rigid pose; cubes in the glass frame (centre within the inner wall, below the rim); the ice level from each resting cube's oriented box (a cube still falling through the mouth would read as a full glass); cube motion from displacement over 0.2 s, because stacked cubes report 2–4 cm/s solver velocities while still. A cube in the outlet counts as outside only 0.5 s after its delivery.
- **Oracle (right arm):** tool azimuth 125°, 20° down, pads 3.5 cm above the glass base (at 100° the wrist motor hid the glass mouth from the head camera). It pulls back 2.2 cm when the resting ice reaches 0.46, sets the glass down, rises 8 cm above the rim (the joint-space park otherwise swept a jaw across the rim) and parks. A 20 cm recess was too narrow for the palm holding the glass; it is 26 cm.
- **Appearance:** opaque frosty ice; partially transparent ice was nearly invisible inside the glass with RTX fractional cutout opacity. The loose pile can settle by up to about 0.09 after the glass is moved (hard 1: 0.52 at the release, 0.43 on the counter).
- **Rig:** recorded on the 0.60 m rig at 55° head pitch, before the RoboDojo rig migration (`docs/robodojo-rig-spec.md`); that migration adds the wrist cameras and re-records A2 with the other tasks.

