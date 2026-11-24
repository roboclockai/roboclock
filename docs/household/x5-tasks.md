# Household ten-task design (revision R2)

**Updated:** 2026-10-25. **Robot:** Dual ARX X5, a stationary dual-arm mount on the kitchen island.

**Status:**
- **Done:** task design, asset collection, static initial states, and the task/process/scoring models.
- **All ten tasks (A1–A4, B1–B3, C1–C3):** Isaac Sim physics episodes with a privileged oracle controller, validated by the task model and an independent trajectory check, and recorded; see [x5-simulation.md](x5-simulation.md). A2, A3, A4, B3 and C2 (R2.1) and C1 (R2.2) have physics-driven layout revisions (below); C3 was replaced by cutlery organization on 2026-11-02 (§ 13); A2 was replaced by a countertop ice dispenser (half a glass of ice) on 2026-11-03 (§ 15); C2 also runs at a lowered browning pace (§ 12).
- **Not implemented:** the agent-facing action/observation interface.

The machine-readable contract for each task is `src/roboclock/envs/household/<id>_x5/design.json`. The runnable task, process and scoring model is `task.py` in the same folder; its backend interface is described in [x5-task-interface.md](x5-task-interface.md). Initial-state images are `data/household/x5_initial/<ID>/overview.png` (presentation view) and `agent_view.png` (task close-up).

## 1. What changed in R2 and why

The changes follow the design review. The agent/LLM interface is deliberately not defined yet. Every task's timing comes from its own physical or household process, and the world keeps evolving while the agent deliberates.

| Change | Content | Reason |
|---|---|---|
| **A2 replaced twice** | Under-counter oven tray → countertop-oven cheese toast → **fill a glass about half full of ice at a countertop ice dispenser** (R3/R3.1, 2026-11-03, on request) | The old tray sat below the counter, directly under the robot bases, and the X5 could not reach it; the toast task ran on a minutes scale. The user replaced it with an ice dispenser: release the dispenser control once enough ice has been delivered. See § 15. |
| **B2 replaced twice** | Rolling-can interception → smoking toaster → **catch the object rolling off the counter** (2026-11-02, on request) | The earlier swap dropped interception because a nudged can reaches the edge too fast for an LLM in the loop. That is still true, and is now the point: the task tests dynamic tracking and fast interception. |
| **C2 replaced twice** | Breakfast-box packing → French toast in one pan → **two arms pour cola into six glasses** (R3, 2026-11-02; R4 2026-11-03: glasses in a row, no cooktop, cola as a particle fluid) | French toast was laggy and staggered in review, and the larger squares asked for did not fit the 60 mm jaw and the turner. The user replaced the task with pouring, as much as possible in the same time. See § 14. |
| A3 re-assetted | Deep pot + bread → **frying pan + pancake**, flipped with a turner | The old "frying pan" was B3's stock pot. Bubbles on a pancake's top surface are the classic visible flip cue. |
| A4 vessels and cue | Opaque pitcher + shallow measuring scoop → **kettle + graduated glass jug on a digital scale**; target 300 g, 46 s budget (medium) | The scale reading is a readable continuous cue. The time budget creates a real speed–accuracy trade-off. |
| B1 appearance | White counter slab and opaque cyan water → native terrazzo counter; since 2026-11-02 the real black basin, with a bowl filling under the spout | Makes the water level readable and keeps materials consistent. |
| B3 decisions | **Water-glass trap** added; pot rotated so its handles point front/back; left arm takes the lid, right arm turns the center knob | Tests safety knowledge under pressure and fixes the unreachable lid position. |
| C1 demand | Four bowls with individual heating times and pickup times over a 10 min horizon | The microwave is a single bottleneck, so every delay becomes appliance idle time. |
| C3 two settings | Two place settings, a bus tray, sponge wiping and two seating times (replaced on 2026-11-02 by cutlery organization, § 13) | Creates real turnover demand instead of a single cleanup with a countdown. |
| Global | Lower ready pose; an agent close-up camera per task; reach screen; all asset paths relative | Less occlusion, cues readable from the agent view, a portable package. |

Semantics previously specified by the task owner are unchanged:
- **A1:** the 8–12 s demo window, then stirring two turns and returning the spoon after extraction.
- **B3:** turn the knob off (the lid step was dropped on 2026-11-02).
- **C3:** clear the dishes before wiping with the sponge (until the 2026-11-02 replacement).

## 2. Task table

"Nominal" values are the alpha = 1 evaluation contract. Hidden parameters are sampled per seed and never shown to the agent, who must judge progress from visible cues.

| ID / category | Summary | Goal | Workflow | Timing (process source) | Assets (source) |
|---|---|---|---|---|---|
| **A1** window (R3) | Sweeten a glass of tea: pour sugar from an open jar, stop in time, then stir | Sugar in the glass in the tier's band (easy 10–30 g, medium 10–25 g, hard 10–15 g); jar set down upright and released; two stirring turns, spoon back in its basket | Left arm picks up the jar → tilts it over the glass → stops the pour → sets the jar down; then the right arm stirs two turns → returns the spoon | Flow q = 4 g/s × c × min(1, (θ − θ0)/35°)^1.5 past the onset tilt θ0 = 30 + 80 (1 − jar sugar / 94 g)°; same pour at every tier, the band is the tier; hard band ≥ 1.09 s at the fastest flow; hidden jar sugar U[60, 85] g (θ0 38–59°), c U[0.85, 1.15]; cue: the sugar mound on the glass bottom (§ 18) | Authored glass of tea (OmniGlass), "Glass Jar" by Extrone (Sketchfab, CC BY 4.0, via Objaverse), HANDAL spoon, Lightwheel wire basket (×0.5) |
| **A2** window (R3.1) | Fill a glass about half full of ice at a refrigerator-door style ice dispenser (a countertop unit in the scene) | Ice level 40–60 % of the glass; glass set down upright on the counter | Right arm picks the glass up from the counter → holds its mouth under the ice outlet and pushes the dispenser paddle with the rim → pulls back when the ice reaches about half the glass → sets the glass down | While the paddle is held in, ice is metered out in batches of one cube, one per interval after a spin-up (a refrigerator-door dispenser delivers a batch about every 1.5 s; § 3); the auger coasts 0.12 s after the release; one press only. Nominal 0.9 s spin-up, **1.5 s** interval (spin-up and coast ×0.75 / ×1.5 rate at easy / hard; the cube cadence ×0.75 / ×1.15, i.e. one cube every 2.0 / 1.5 / 1.3 s); each cube raises the ice by about 6–17 % of the glass, so the band holds one to three cube counts and the release window (one interval minus the coast in the worst case) is ≥ 1.61 / 1.21 / 1.08 s; hidden spin-up U[0.6, 1.3] s, interval U[1.4, 1.7] s, per-cube jitter ±5 % (2026-11-17; was 1.35 s, U[1.1, 1.6] s, ±12 %, ×1.5 at hard: worst window 0.56 s) | Authored countertop ice dispenser with the native Lightwheel Refrigerator001 dispenser paddle (×0.85) on its hinge, authored tumbler and crescent ice cubes |
| **A3** window | Flip a pancake in time | Flip with the turner when the underside is golden; pancake lands fully back in the pan | Right arm takes the turner → slides it under the pancake → lifts and turns it → releases it in the pan; the left arm may hold the pan handle (R2.1: the left arm levers the pancake over with an offset turner; see § 8) | Underside doneness d, band [0.55, 0.75]: nominal window **t ≈ 32–70 s**; cue is top bubbles forming → popping → set edges; hidden τ ∈ U[50, 80] s | Lightwheel 5-burner gas cooktop, RoboCasa pan_2, RoboCasa turner, authored pancake (rigid disc with textured top), blue burner-ring card |
| **A4** window | Pour 300 g of hot water and stop in time | Scale reads 300 ± 10 g, no spill, kettle set down within the tier budget: easy 60 s, medium 46 s, hard 36 s (recalibrated 2026-11-16 on the reference oracle: slowest kettle-down 28.69 s over seeds 0–9, hard = max(t / 0.8, t + 0.3 s), medium = geometric mean; was 45 s / time_scale = 60 / 45 / 30 s) | Left arm lifts the kettle → aligns the spout over the jug → tilts to pour → watches the scale → returns upright (tail flow) → sets the kettle down (R2.1: both arms hold the kettle, by the handle and the lid knob; see § 9) | Stop interval follows from the flow: at full tilt (≈40 ml/s) the tolerance is a fraction of a second, and the scale display lags 0.3 s; **60 / 46 / 36 s budget**; under-pouring can be topped up, overshoot is final; hidden kettle volume U[0.7, 1.1] L, flow coefficient ±15 % | RoboCasa kettle_0, authored glass jug (OmniGlass, 500 ml graduations), authored digital scale (reads 0 g), recipe card |
| **B1** reactive | Bowl about to overflow: shut the faucet | Stop the water before the bowl under the faucet overflows, minimising the spill | Right arm pushes the single lever closed → releases | Inflow 0.10 L/s (6.0 L/min) into a 1.6 L bowl; 0.075 L/s at easy and 0.138 L/s at hard (8.3 L/min, the US federal maximum for kitchen faucets, 2.2 gpm; § 3). Nominal 1.2 L of room overflows after **≈12 s**, the level rising about 4 mm/s; hidden room U[0.8, 1.4] L, i.e. 8–14 s (hard 5.8–10.1 s); the spill never decreases, fail at 0.5 L | Lightwheel Sink054 (native faucet D6 joint), counter cut-out in native material, RoboCasa bowl stretched into a mixing bowl, authored water surface and bubbles |
| **B2** reactive | Catch the object rolling off the counter | Stop whichever object starts rolling before it falls, then hold it | Left arm meets it near the front edge with the gripper open, tilted so the object runs into the lower jaw, then pinches and lifts | Four objects (cup, egg, pill bottle, apple) sit on a wedge board behind pegs; one is let go at a hidden time in [1, 3] s. Slope 4.5° / 6° / 9°: **1.2–2.3 s** from release to the edge | Authored board and objects |
| **B3** reactive | Oil fire in a pot: turn the burner off | Center knob at OFF, flame out; never use water | Right arm pinches the center knob → turns it to OFF → releases (simplified 2026-11-02: no lid) | Flame f grows logistically while the burner is on and reaches the spread threshold 0.8 (failure) at **≈48 s** nominal, the time scale of an oil pan fire under wall cabinets (§ 3); with the burner off it dies within a few seconds (τ = 1 s; an abstraction, no lid is required); loss = ∫ f dt | Lightwheel cooktop (5 native knobs), Pot057 body and matching lid, authored flame silhouettes, authored glass of water |
| **C1** throughput (R3) | Serve the ordered meals at the pass | Physical: every ordered meal set down released and upright on the serving board by the end, late or not; nothing dropped. Temporal: each by its pickup (§ 17) | Left arm takes the next wanted meal from the counter → sets it on the staging mat; right arm takes it from the mat → holds it beside the board → sets it down once the waiter has taken the previous one (the board holds one bowl; see § 16) | Four orders at every tier; waiter pickups at 4 / 2 / 1 × the hard schedule **10.53 / 23.49 / 36.45 / 49.41 s** (medium 21.06 / 46.98 / 72.9 / 98.82 s; § 17); horizon 1.25 × the last pickup; per seed: wanted order, meal per counter slot, pose jitter | 4 RoboCasa bowls with authored lug handles and food surfaces (curry, rice, soup, pasta), authored staging mat and wooden serving board, order card |
| **C2** throughput (R4) | Two arms pour cola into six glasses in a row | All 6 glasses filled to the 90 ml line (band 80–105 ml, held 1 s) at the end, both bottles stood back released, spill < 15 ml; temporal: t_Q within the tier budget (§ 17) | Pick up an opened bottle → tilt it until the cola flows over a glass → keep the stream in the glass → tip back in time for the line → next glass; both arms at once | Cola is a PhysX particle fluid; flow, stream and tail come from the simulation; one run to H = 216.45 s, budgets **173.2 / 86.6 / 43.3 s** (easy / medium / hard); no hidden parameters |
| **C3** throughput (R3) | Cutlery organization: sort mixed utensils into the drawer | All 6 loose utensils resting, released, in the compartment of their kind at the end; nothing wrong, nothing dropped; temporal: t_Q within the tier budget (§ 17) | Tell spoon / fork / knife apart by the head → match it to the compartment holding the same kind → pinch the handle off the table → turn it along the compartment → lay it in; the two arms share the drawer | One run to H = 127.8 s, budgets **102.2 / 51.1 / 25.6 s** (easy / medium / hard; t_o = 25.56 s); hidden: which utensil lies where and the compartment order (per seed) | Authored drawer with three compartments, 9 authored utensils (2 + 1 reference of each kind) |

## 3. Shared conventions

- **Time origin:** t = 0 is the agent's first observation of the initial state. Physics and processes then run continuously; the world does not pause while the agent deliberates.
- **alpha:** process time-scale factor. Evaluation uses alpha = 1. Demo videos may use alpha > 1 to shorten processes, but the value must be stated in the video metadata. A1's 8–12 s window is the task owner's setting; since 2026-11-16 it is described as a gongfu-style infusion, which runs at that speed in real time (alpha = 1), no longer as a black-tea brew compressed about 20 times. In code, `difficulty` maps to `spec.time_scale` (easy 0.75, medium 1.0, hard 1.5), except for C1–C3 (time scale 1; § 17).
- **Real-process time scale (2026-11-16):** each instruction names a process that runs at the simulated speed, or the compression is stated here.
  - **A1** (since 2026-11-19 a sugar pour, § 18) has no process clock: the time is the agent's own pour. The 4 g/s maximum is slower than sugar runs from an open jar mouth; it keeps the hard band at least 1 s long (§ 18). Until 2026-11-19 A1 was a gongfu-style tea infusion (git history).
  - **A2** is described as a refrigerator-door style dispenser, which meters ice in batches while the paddle is held: one batch about every 1.5 s (Whirlpool, US patent 4,942,979). Ours delivers one cube every 2.0 / 1.5 / 1.3 s (2026-11-17, slowed from 1.8 / 1.35 / 0.9 s so that every release window is at least 1.0 s; hard caps the cube cadence at ×1.15); the mass rate (4.3 / 5.7 / 6.5 g/s) has no source. The unit in the scene stands on the counter.
  - **A3** is compressed: at the flip the first side has cooked 52–90 s at medium (69–120 s easy, 35–60 s hard), about 2 times shorter than the 2–3 min of pancake recipes (Betty Crocker; Cook's Illustrated), and 3–3.5 times shorter at hard.
  - **A4** is not compressed, and not sourced: the pour runs in real time (300 g takes at least 7.5 s at the 40 ml/s maximum, the order of the 3.2–8.7 s of human volume-targeted pours, arXiv 2011.10150), and the 60 / 46 / 36 s budget is calibrated on the reference controller (the oracle: a reference level, not an optimal time).
  - **B1** faucet flow is 4.5 / 6.0 / 8.3 L/min. Hard is held at the US federal maximum for kitchen faucets, 2.2 gpm (8.3 L/min) at 60 psi (US EPA WaterSense kitchen-faucet sheet), instead of ×1.5 = 9.0 L/min; time to overflow at hard is therefore 5.8–10.1 s (nominal 8.7 s) instead of 5.3–9.3 s (8 s). No deadline or score was retuned for it.
  - **B3** has the time scale of an oil pan fire under wall cabinets without a range hood: the cabinets are at risk after as little as 40 s of free burning and can ignite from 75 s (Budnick and Klein, NBSIR 75-788). Spread is at 64 / 48 / 32 s nominal, so hard is about 1.25 times short. The scene's burning pot stands on the island, with no cabinets above it, so the instruction says only that the oil has caught fire. The task requires the burner off and the knob released, nothing else: the flame then dies within seconds, which is an abstraction (real guidance also covers the pan with a lid; the lid step was removed on 2026-11-02).
- **Hidden parameters:** sampled per seed, never exposed to the agent. The agent must judge progress from visible cues: tea colour, cheese, pancake bubbles, scale reading, water level, smoke, flame.
- **Scoring:** physical completion, temporal/quality success and their conjunction are reported separately. An effective intervention is timed by its physical result (the water has actually stopped, the lid is actually seated), not by when a command was issued.
- **Arms (common rig, 2026-11-03):** both bases side by side at (0.00, −0.255) and (0.60, −0.255), 0.60 m apart, same height (mount top z = 0.8704) and both facing +y (the back wall). The head camera `/World/AgentCamera` is fixed to the rig: 0.04 m forward of the base midpoint and 0.54 m above the mount tops, at (0.30, −0.215, 1.4104), pitched 55° down, 18 mm lens on a 30 × 22.5 mm aperture (79.6° × 64.0° field of view); the same pose for every task (`x5/rig.py`). All ten tasks use it (`rig.RIG_TASKS`); the earlier layout had the bases at (−0.06, −0.255) / (0.66, −0.255) with a camera per task. The ready pose is q = [0, 0.94, 1.34, −1.40, 0, 0], which puts the grippers about 27 cm above the counter.
- **Frame:** meters, z up, island counter top at z = 0.8584.

## 4. Reach screen

`data/household/x5_initial/REACH_CHECK.json` records, for every task's key operation points, a numerical IK check against the URDF joint limits of tool position and approach direction. It does not check collisions, payload or trajectories. The first pass found three problems, all fixed:

- the B3 lid, originally on the rear-right grate, was out of reach for both arms; it now sits on the front-left grate for the left arm;
- the C3 sponge, originally at the back center, was out of reach for the right arm; it now sits between the two mats;
- the A1 stirring reference point was corrected for the held-spoon height.

The final result is 50/50 operation points reachable (after the R2.1/R2.2 layout revisions). The model used here limits joint6 to ±90°, so the A3 flips need combined wrist motion or a slide-and-turn motion. The official X5A model bundled on `main` allows ±2.0 rad.

## 5. Handoff for the simulation backend

- **`design.json`:** each task's instruction, core actions, process model (nominal values and hidden ranges), timing events, completion and failure rules, permitted and prohibited strategies, arm roles, demo guidance and open items.
- **`task.py`:** implements that process model and scoring. The backend calls `update(signals)` and `advance(dt)` every step, drives appearance from `cues()`, and reports `metrics()`.
- **`data/household/x5_initial/<ID>/initial_state.json`:** object bounds and sources, native joints (deactivated in the static scene and to be re-enabled in simulation), key operation points, cameras, and the robot ready pose and end-effector poses.
- **Suggested demo workflow:** first run GT feedback control without video until the physical and timing checks pass; then freeze the configuration and record; finally verify that the recorded run itself also passes.
- **Implemented:** the per-task process models and scoring (pure Python); `tests/envs/household/test_x5_tasks.py` covers early, on-time and late cases. For all ten tasks: the Isaac Sim physics overlay, signal measurement, appearance drivers, oracle controllers, episode runner, independent validator and recorded demos ([x5-simulation.md](x5-simulation.md)).
- **Not implemented:** the agent-facing action/observation interface.

## 6. A2 layout revision R2.1 (2026-10-22, superseded by § 15)

Building the A2 physics episode showed that the R2 layout had no physically valid solution for the X5 pair:

- the drop-down door is hinged at counter height, so when open it covered every counter spot in front of the 0.75-scale oven that both arms can reach; there was nowhere to set the tray down with two hands;
- one X5 gripper cannot carry a tray by one edge, because the off-centre load tips it out of a two-pad pinch;
- the X5 wrist cannot point straight down about 0.3 m in front of its base, and its palm spans ±9 cm along the jaw axis, so rim pinches need the rim about 10 cm clear of the oven face.

R2.1 keeps the instruction, process model, timing event and scoring. It changes only the layout and assets: the oven is scaled ×0.6 and pushed to the island's back edge; the trivet moves in front of the open door; the upper rack is removed; the tray rim is 2.2 cm tall; and the rack sits on full-extension runners (0.178 m) and auto-ejects with the door, a common countertop-oven linkage. The auto-eject replaced a robot pull: the X5 finger bodies are about 6 cm tall plates, so hooking the 2.2 cm rim pressed on or wedged against it and tipped the tray. Details are in `a2_oven_x5/design.json` (`layout_revision`); the oven task is kept in `a2_oven_x5/` for reference.

## 7. C1 layout revision R2.2 (2026-10-22)

Building the C1 episode showed that the R2 layout, in which each arm carries bowls across the counter, had no physically valid solution for the X5 pair:

- **Grasp:** a bowl cannot be rim-pinched inside the 16 cm cavity, so each bowl got two lug handles. With the jaw axis near vertical, the palm reaches 7.6 cm below the tool point, so the handles sweep up 20° and are pinched with the tool pitched 20° down.
- **Reach:** an X5 holds a bowl by a handle only away from its own base. Within about 0.2 m in front of either base, no handle grasp is reachable. So the right arm cannot bring a bowl from the back-right around the microwave corner, and the left arm cannot carry one across the front to the left side.
- **Cavity:** the palm's upper edge is above the cavity top, so the palm must stay in front of the mouth. The near handle must also clear the closing door. This fixes the tool azimuth at 132.5° and the bowl 2.5 cm in front of the plate centre.

R2.2 keeps the instruction, card, process model and scoring. The new layout:

- **Roles:** the relay is reversed. The left arm stages each cold bowl on a mat in front of the microwave and pushes the door shut. The right arm sets the timer on the digital panel (+15 s, start, door release), loads the staged bowl, unloads the hot bowl and serves it on a one-place board.
- **Bowls:** meal 1 starts in the idle microwave. The counter right of the microwave has room for only three bowls outside the zones the right hand sweeps at the mat and the board. All bowls use the deep bowl model, so the palm clears the counter by about 2 cm.

Details are in `c1_x5/design.json` (`layout_revision`).

## 8. A3 layout revision R2.1 (2026-10-22)

Building the A3 episode showed that the R2 layout had no physically valid solution for the X5:

- **Turner:** the native turner is flat. Lying on the counter, its handle is level with the blade, so the palm (7.8–14.5 cm behind the tool point, ±9 cm along the jaw axis) met the counter at the grasp and the pan rim while the blade slid under the pancake.
- **Flip:** a two-pad pinch cannot toss or roll a rigid pancake reliably. Fast wrist rolls threw it unpredictably, and levering about its far edge threw it out of the pan. Levering about the side edge parallel to the blade, to about 85°, and then pushing the pancake over is deterministic.
- **Reach:** that lever (blade along a 50° line, rolled to 85°) is reachable only for the left arm, and not beyond about 88°.
- **Pan:** on the 0.44-scale pan the flat floor was too small to turn a 12.4 cm pancake over inside the rim.

R2.1 keeps the instruction, process model, flip event (rotation ≥ 150°), band and scoring. The new layout:

- **Turner:** bent 30° at the neck into an ordinary offset turner. It lies blade-flat in front of the pan with the handle rising toward the left arm, which takes it; the right arm stays parked.
- **Pan and pancake:** the pan is scaled ×0.55 (rim radius 0.18 m). The pancake is poured 5.3 cm off-centre against the turn (4.5 cm until 2026-11-04, when an eval run landed it on the wall slope), so it lands as far from the wall as it started.

Details are in `a3_x5/design.json` (`layout_revision`).

## 9. A4 layout revision R2.1 (2026-10-22)

Building the A4 episode showed that the R2 single-arm pour had no physically valid solution:

- **Torque:** one X5 pinch on the handle cannot hold the full kettle (0.8 kg shell + 1.0 kg water) while tilting. The gravity torque about the pinch, about 1.5–2 N·m, far exceeds the torsional capacity of two 15 × 10 mm pads at 20 N, so the kettle rotates out.
- **Grasps:** top-down two-arm grasps and grasps with the jaw axis along the kettle were out of IK reach. Front grasps through the handle opening and on the lid knob are reachable.
- **Spout:** the spout reaches only 1.5 cm beyond the body. The base must clear the jug rim before the spout is over the jug, and the pour point sits 2.5 cm short of the jug axis, 3 cm above the rim, so the left elbow keeps about 1 rad of joint margin at 83° tilt.

R2.1 keeps the instruction, 300 ± 10 g target, 45 s budget (46 s at medium since 2026-11-16), spill limit, flow model and scoring. The new layout:

- **Roles:** the left arm pinches the handle bar with one pad through the handle opening; the right arm pinches the lid knob. The arms approach one after the other, because their palms would meet mid-air.
- **Placement:** the kettle stands at (0.30, 0.05) with the jug and scale beside it at (0.51, 0.05). The scale LCD shows the live, lagged reading on 7-segment digits instead of a fixed 0 g texture.

Details are in `a4_x5/design.json` (`layout_revision`).

## 10. B3 layout revision R2.1 (2026-10-24)

Building the B3 episode showed that the lid as placed could not be carried level by one X5 pinch:

- **Bar rolling:** the lid's top handle is a 6 mm round bar. Pinched across its section, the bar rolls between the pads, and the hanging lid tilted 6–11°. It then landed off-centre and wedged on the pot rim.
- **Flat sides:** the bar's sides are flat, and pads closed on them resist the roll. The tool must then run along the bar, and the left arm cannot reach that pose with the handle along x as placed.

R2.1 keeps the instruction, process model, timing and scoring. The only change: the lid rests on the same grate turned 60° about its axis, so its handle points toward the left arm. Details are in `b3_x5/design.json` (`layout_revision`).

## 11. C3 layout revision R2.1 (2026-10-25, superseded by § 13)

Building the C3 episode showed that the R2 layout had no physically valid solution for the X5 pair:

- **Plates:** the flat 24 cm dinner plate is 1.2 cm tall and has no wall a pinch can hold. One pinch on a rim cannot carry a plate either; the off-centre load tips it out of the pads, as with the A2 tray.
- **Reach:** on the 36 × 28 cm mats, parts of each setting lay where only one arm reaches, so the two-arm plate lift was impossible.
- **Sponge:** the 9.5 cm sponge is wider than the 6 cm jaw opening in every orientation that leaves it flat.
- **Tray:** the 34 cm tray's flat floor holds one plate, with no room left for the mugs.

R2.1 keeps the instruction, card, process model, seating times and scoring. The new layout:

- **Settings:** two 25 × 30 cm mats side by side (centres x = 0.145 and 0.455 m), inside the zone both arms reach. Each holds a deep plate (a soup plate: the bowl model flattened to 20 × 4.2 cm) and a mug behind it.
- **Plates:** both arms pinch opposite walls of one plate and lift it together; plate 2 stacks on plate 1 in the tray.
- **Mugs:** each arm pinches the rim wall of its own mug with the tool tilted 30° along the rim. Mug 1 goes into the top plate, mug 2 stands on the tray floor.
- **Tray and sponge:** the bus tray is 44 cm long and lies across the front. A 7 cm sponge lies between the mats, long side front to back. The left arm wipes mat 1 and puts the sponge back; then the right arm wipes mat 2.

Details are in `c3_x5/design.json` (`layout_revision`).

## 13. C3 replaced: cutlery organization (R3, 2026-11-02)

On request, C3 is now a throughput sorting task: spoons, forks and knives lie mixed on the table, and the robot puts each into the right compartment of an open drawer, as many as it can before time is up.

- **Scene:** an open drawer (27.2 × 26.6 cm, three 8 × 25 cm compartments, 4.5 cm walls) on the counter between the arms, centre (0.30, −0.10) m. Each compartment already holds one utensil of its kind, which is the only cue to what goes where. Nine loose utensils lie in nine table slots, five left of the drawer and four right of it.
- **Utensils:** authored meshes with one shared dark handle (11 × 1.6 × 0.8 cm), so only the head tells them apart: a spoon bowl, four fork tines or a knife blade (19.5 / 19.7 / 22.7 cm, 40–45 g). The handle is thicker than steel cutlery so the 15 × 10 mm pads can pinch it flat off the counter.
- **Hidden per seed:** which utensil lies in which slot (with ±0.6 cm, ±1 cm and ±8° jitter) and the compartment order (one of six). Seed 0: spoons, forks, knives from left to right.
- **Scoring:** a loose utensil counts while it rests (< 5 cm/s), released, with its middle inside the compartment of its kind and below the wall tops. The count at the horizon (45 s / time scale: 60, 45, 30 s) is the result; temporal success needs 6 of 9; physical success needs at least one sorted, none in a wrong compartment, none off the counter, and the reference utensils left in place. Reported: correct, wrong, correct − wrong, utensils per minute and the first time each was sorted.
- **Reach:** a top-down pinch at the counter reaches about 0.46 m from each base, so the table slots stay within x −0.04 to 0.125 m (left) and 0.475 to 0.64 m (right). Over the far compartment, and over the farthest slot, the arm can only be 6–9 cm up instead of 13 cm.

Details are in `c3_x5/design.json`.

## 14. C2 replaced: two-arm cola pouring (R3 2026-11-02, R4 2026-11-03)

The French-toast task (R2.1) was dropped on request: the user asked for pouring cola or juice instead, as much as possible in the same time.

- **R3 (2026-11-02):** cola with a scalar flow and foam model, glasses in two columns beside the cooktop.
- **R4 (2026-11-03, on request: "C2换成并排的，去掉中间的灶台，只留杯子和瓶子，可乐用流体仿真"):**
  - **Layout:** the island is a bare counter with only glasses and bottles. Six tumblers (about 170 ml, white line at 90 ml) stand side by side in one row, 10 cm apart, three outside each arm's base. C2 is on the common rig.
  - **Fluid:** the cola is a PhysX particle fluid, about 310 ml per bottle. Flow, stream, splashing and the tail after tipping back come from the simulation, and every scored volume is a particle count. The R3 foam model is gone; no hidden parameters remain, and difficulty sets only the window.
- **Scoring:** served glasses when the window closes (90 s / time scale). Success needs all six, none above the band, both bottles released upright, nothing dropped or knocked, and spill under 15 ml.

Details are in `c2_x5/design.json` (`layout_revision`).

## 15. A2 replaced: countertop ice dispenser, half a glass (R3.1, 2026-11-03)

The user replaced the countertop-oven toast with: *an ice dispenser drops ice into a cup, and the robot must release the dispenser control once enough ice has been delivered. Releasing too early leaves too little, releasing too late gives too much.* A first version (R3) used the refrigerator's own door dispenser and asked for exactly 5 cubes. On review the user asked for a countertop ice machine instead of the whole refrigerator, wrist cameras, and half a glass of ice as the goal.

- **Scene.** An authored countertop ice dispenser (36 × 33 × 52 cm) stands against the island's back edge, its front at y = 0.05 m; the island is at its usual place. Its dispensing recess (26 cm wide, 15.5 cm deep) has a drip tray, a nozzle housing with an octagonal ice outlet, and the native dispenser paddle of the Lightwheel Refrigerator001 ("Ice | Water", `Refrigerator001_Switch`, ×0.85, with its native material) on its top hinge. Hinge, outlet and recess keep the native dispenser's proportions. The recess is 26 cm wide because the X5 palm (±9 cm along the jaw axis) holding the glass hit the walls of a 20 cm recess. The outlet (1.18 m) and the recess are in the head camera's view.
- **Control.** The glass rim pushes the paddle back, as people use it; a gripper pressing the paddle directly would sit in the path of the falling ice. The paddle hangs 15° forward from its top hinge; the switch closes at 5° (opens below 3°); the stop is at 12°.
- **Process.** While the paddle is held in, the auger runs. After a spin-up it delivers one cube per interval with a small per-cube jitter; after the release it coasts 0.12 s. Dispensing is one press: a second press delivers nothing. Cadence (2026-11-17): one cube every 1.5 s nominal at medium (hidden U[1.4, 1.7] s, ±5 % per-cube jitter), 2.0 s at easy and 1.3 s at hard (the cube cadence is capped at ×1.15 at hard; spin-up and coast keep ×1.5).
- **Goal and scoring.** The ice level is the top of the highest resting cube in the glass over the glass's inner height (0 empty, 1 at the rim). The task is physically complete when the paddle was pressed and released and the glass stands released on the counter with every cube at rest for 1 s; it is temporally successful when the level is 0.40–0.60 (quality linear to 0 at 0.20 / 0.80). Each cube raises the level by about 0.06–0.17: held under the outlet with the paddle in (Isaac, seeds 0–9 at every tier), the band held one to three cube counts. With a single count the release must come after that cube and at least the coast before the next one is due, so the worst-case release window is one interval minus the coast: ≥ 1.61 / 1.21 / 1.08 s at easy / medium / hard. The earlier cadence (1.35 s nominal, U[1.1, 1.6] s, ±12 %, ×1.5 at hard) allowed 0.56 s at hard, and hard seed 2 (band = the 5th cube only) measured 0.94 s, under the 1.0 s floor for any temporal limit. A loose pile can settle by up to about 0.09 after the glass is moved.
- **Physics.** The cubes are 8.5 g rigid crescents. Each delivery places the next waiting cube just inside the outlet, moving down; from there it falls freely into the glass. The glass is a free 200 g hollow compound, 56 mm across so it fits the 60.3 mm jaw. Level and counts come from the measured cube and glass poses. A cube that comes to rest outside the glass is a failure.
- **Oracle.** Right arm: pinch the glass from behind (tool azimuth 125°, 20° down, pads 3.5 cm above the base, which keeps the wrist off the head camera's line of sight to the glass mouth), carry it under the outlet, push the paddle to about 8°, pull back 2.2 cm when the resting ice reaches 0.46, set the glass down, rise and park. The left arm is idle.
- **Wrist cameras** are part of the shared rig migration (`docs/robodojo-rig-spec.md`), implemented for all household tasks by that work, not in this section.
- The oven task is kept in `src/roboclock/envs/household/a2_oven_x5/` (not in `tasks.json`); its last delivered demo is in `results/household/x5_demos_superseded_2026-11-03/A2_oven/`. The refrigerator version (R3) demo is in `results/household/x5_demos_superseded_2026-11-03/A2_fridge/`.

## 16. C1 replaced: serving relay at the pass (R3, 2026-11-06)

The user asked for a shorter C1, about 60 s per episode, as another efficiency/throughput task: R2.2 ran 600 process seconds (800 / 600 / 400 s at easy / medium / hard), a quarter of the simulated time of the whole household suite, mostly waiting on the microwave (one microwave turnaround alone took the oracle about 40 s). Of the options offered, the user chose a serving relay without the microwave.

- **Task.** Four meals (curry, rice, soup, pasta) in the R2.2 bowls with lug handles. The first wanted meal is ready on the staging mat; the other three stand on the counter in front of the left arm. The serving board, in front of the right arm, holds one bowl; a waiter takes the bowl on it at each pickup time. An order qualifies when the meal wanted at that pickup rests released and upright on the board. The instruction names the wanted meals in order and the pickup times; the card on the counter states the rule.
- **Timing (until 2026-11-09; now § 17).** 60 process seconds; pickups at 20 / 32 / 44 / 56 s at medium, scaled by the time scale at easy (80 s horizon). Hard takes three orders at 14 / 26 / 39 s of 40 s, because four scaled pickups (8 s apart) are shorter than one relay turnaround (about 12.8 s for the oracle), which does not scale; hard keeps medium's spacing with an earlier first pickup, so the oracle has about 3 s of slack per order instead of 7–9 s.
- **Per seed.** The wanted order (a permutation; the first wanted meal starts on the mat), which waiting meal stands in which counter slot, and ±1.2 cm / ±8° jitter of the slots. Seed 0 is the nominal layout. The train and test seeds therefore differ in order and layout, not only in a hidden parameter.
- **Layout.** Neither arm reaches both the counter slots and the board (an X5 holds a handle pose only beyond about 0.28 m from its own base), which makes it a relay through the R2.2 staging mat, the one place where one handle axis suits both arms. The counter slots (0.20, 0.16), (0.03, 0.22), (−0.14, 0.18) face the left arm; R2.2's slots in front of the right base made the left arm unfold over the board, so it could stage only after each pickup. In R3 the arms stay on their own sides: the left arm waits with the next meal at (0.12, 0.12) while the right one works the mat, and the right arm leaves the mat via (0.41, −0.04) and holds the meal beside the board at (0.48, 0.14) until the pickup.
- **Unchanged from R2.2.** The bowls, handles and grasps (close-until-contact pinch, tool pitched 20° down), the carry profile, the staging mat and the serving board. The R2.2 microwave relay is in git history before this revision.

## 17. C-class judging (score_version 2, 2026-11-09)

The household C tasks follow the shared rule in [docs/c-class-timing.md](../c-class-timing.md) (`roboclock.evaluation.throughput`). Every C task is Q jobs; physical success is all Q done and still valid at the end with no failure (time plays no part); temporal success is per tier from the job times; joint is both. Rows carry `job_times_s`, `t_q_s`, `termination_reason` (`all_done`, `stall`, `failure`, `horizon`) and, for C2 and C3, the `*_by_tier` fields.

- **C2 (batch, Q = 6 glasses).** t_o = 43.29 s. One run per seed to H = 5 t_o (216.45 s) at every difficulty; the difficulty only picks the tier (T = 4 / 2 / 1 t_o). t_Q is `all_served`: the first time all six are served and both bottles stand released for 1 s (as `finish_time` was). Physical: all six in the band at the end, both bottles released upright, nothing dropped, knocked or spilled; an overfilled glass is recorded (`glassN_overfilled`) and fails the final band check. Ends 2 s after all is done, on a failure, or on the stall stop (from T_medium, no glass served for 1.0 t_o = max(0.5 t_o, 1.5 T_easy / 6), while glasses remain). The serving window is gone; the instruction asks for all six "as quickly as possible".
- **C3 (batch, Q = 6 utensils).** The table now holds two of each kind (Spoon1-2, Fork1-2, Knife1-2; references Spoon3, Fork3, Knife3) in six slots, three per arm: the three slots nearest the drawer on each side of the former nine. t_Q is the first time all six are sorted at once. Physical: all six sorted at the end, none wrong, none dropped, references in place. Ends 2 s after all six are sorted or on the stall stop (from T_medium, no new utensil sorted for 1.0 t_o = max(0.5 t_o, 1.5 T_easy / 6)); a dropped utensil does not end it (as before). t_o = 25.56 s, the slowest of 14 oracle runs on the six-utensil scene (seeds 0-6 and 50-56: 23.5-25.6 s, all physical); the C3 demo still shows the nine-utensil scene and must be re-recorded.
- **C1 (stream, Q = 4 orders).** One run per difficulty, because the schedule is the difficulty. Each order's deadline is its pickup; its time is when its meal was set down on the board, kept while the meal stays there or once the waiter took it. A pickup that finds the board without its meal leaves the order open, and the waiter takes that meal as soon as it has rested on the board for 1 s (`late_pickupN`), so a late meal counts for physical success and frees the board for the next order. **Schedule (2026-11-09):** the same four orders at every tier, picked up at m × the hard schedule with m = 4 / 2 / 1 (easy / medium / hard), so the first pickup and the intervals stretch alike; time scale 1, nothing else changes. Hard is a first pickup at 10.53 s and one every 12.96 s (10.53 / 23.49 / 36.45 / 49.41 s), the tightest first-pickup-plus-interval schedule (0.01 s steps) that the slowest oracle meets. It comes from 28 oracle runs on seeds 0–6 and 50–56: 14 with the board freed 1 s after each set-down, so that no pickup held the oracle back (on the former schedules it did: the right arm holds the next meal beside the board until the pickup, and at the former easy schedule every meal waited there), then 14 at the new hard schedule. The first meal is on the board at 10.524 s on every seed (it starts on the mat); each later one follows after a relay turnaround of 12.5–13.0 s, which depends on the counter slot the left arm fetches from. The slowest set-downs are 10.524 s for the first order and 23.484 / 36.344 / 49.204 s (seed 2) for the others. The second order binds (10.53 + 12.96 ≥ 23.484, seed 2 at hard); the third and fourth keep 0.11 and 0.21 s, which absorbs the up to 0.1 s by which moving a pickup shifts the later set-downs. Per-order maxima (10.53 / 23.49 / 36.35 / 49.21 s) would be tighter by those 0.1–0.2 s, but would leave every order a few milliseconds against its slowest run, less than those shifts. At hard the next meal arrives beside the board about 8.8 s or more after each pickup, so the board never holds the oracle back. Medium is 21.06 / 46.98 / 72.9 / 98.82 s, easy 42.12 / 93.96 / 145.8 / 197.64 s. Horizon 1.25 × the last pickup (easy 247.05 s, medium 123.525 s, hard 61.7625 s); the episode ends 2 s after the last pickup once every ordered meal is served. Until 2026-11-09 the time scale divided process-second pickups (20 / 32 / 44 / 56 s at medium), and hard took three orders at 14 / 26 / 39 s.

## 18. A1 replaced: sugar for a glass of tea (R3, 2026-11-19)

The user replaced the tea infuser with pouring sugar and set the bands per tier: easy 10–30 g, medium 10–25 g, hard 10–15 g. The only cue is what can be seen (user's choice over a kitchen scale, which A4 already uses): the sugar sinks and piles up in a white mound on the bottom of a clear glass of tea.

- **Scene.** A glass of tea (lathe, OmniGlass; the tea a tinted clear volume, 56 mm deep) stands where the mug stood, with the same fixed cavity collider, so the right arm's stir and spoon return are unchanged. An open glass jar of sugar stands at (0.06, 0.14) for the left arm: "Glass Jar" by Extrone (Sketchfab, CC BY 4.0, from the Objaverse 1.0 mirror; `scripts/household/import_sugar_jar.py`), the cap left out, scaled to 54 mm across and 74 mm tall so the 60.3 mm jaws pinch its body. The repository had no sugar container (only a YCB sugar box); Poly Haven and Gazebo Fuel had none either. Glass and tea cast no shadow (`no_shadow`): RTX Real-Time shadows refractive glass as opaque and the sugar inside rendered black.
- **Process (task.py).** Sugar runs once the measured jar tilt θ passes θ0 = 30 + 80 (1 − m_jar / 94 g)° (38–59° for the hidden 60–85 g), at q = 4 g/s × c × min(1, (θ − θ0)/35°)^1.5 (c ∈ U[0.85, 1.15]); it lands with a 0.12 s time constant, in the glass if the stream from the jar's lowest mouth point falls inside the glass mouth (sim.py), else on the counter; 2 g spilled fails. The grams in the glass are judged when the jar has stood upright and released on the counter for 1 s; quality falls to 0 at 5 g and at 10 g above the band. The pour is the same at every tier (time scale unused). At the fastest flow the hard band lasts 5 g / 4.6 g/s = 1.09 s (1 s floor of the time-limit rule); the agent can pour more slowly by tilting less.
- **Cue.** The mound is a cone at the 33° slope of settled sugar: 10 / 15 / 25 / 30 g are 5.2 / 5.9 / 7.0 / 7.5 cm across (the bottom is 7.7 cm). The instruction gives the sizes at the tier's band edges. The stream shows while sugar runs, and the jar's sugar level drops. Stirring dissolves the mound (appearance only).
- **Oracle.** The left arm pinches the jar body 3.5 cm up (tool azimuth 45°, 10° down; reach screen ≥ 0.27 rad joint margin), carries it upright beside the glass, tilts it toward +x while the mouth moves over the glass, servoes the flow to 2.5 g/s and tips back when glass + air + 0.3 s of flow reach the middle of the band. The right arm stirs once the jar is released.
- **Superseded.** The tea-infuser version (R2) is in git history; its demo is in `results/household/x5_demos_superseded_2026-11-19/A1_tea`.
