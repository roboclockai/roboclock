# Dual-X5 household asset sources and licenses (R2)

Scenes in `data/household/x5_initial/<ID>/` reference assets by relative path under `src/roboclock/envs/household/assets/x5/`. Per-object source/scale/bounds are in each `initial_state.json → objects`. The kitchen is not stored in git; run `scripts/household/fetch_lightwheel_kitchen.py`, or `scripts/household/link_x5_kitchen.sh` (link or `COPY=1`) for an existing copy. Licenses below must travel with any redistribution; several are non-commercial.

| Group | Package path | Source | License | Used in |
|---|---|---|---|---|
| Kitchen room, island, native appliances (Stovetop012, Sink054, Toaster003, Microwave017, WallStackOven004, Pot057 + lid, Kitchen_Basket) | `assets/x5/lightwheel_kitchen/Collected_KitchenRoom` (linked, not committed) | [Lightwheel Kitchen](https://github.com/LightwheelAI/Lightwheel_Kitchen), project-recorded revision `10b0f7a86135a27e8c2fba9c690f6bf5c4f06bb1` | CC BY-NC 4.0 (as recorded in the project manifest) | all |
| Dual ARX X5 | `assets/x5/robot/arx5-sdk` (URDF/STL, LICENSE), `assets/x5/robot/visual` | [real-stanford/arx5-sdk](https://github.com/real-stanford/arx5-sdk) @ `c160db7834530a3d80fec524547c74e1d4fa8035` | MIT | all |
| RoboCasa objects: pan_2, pan_3, kettle_0, baking_sheet_1, tray_0, tray_1, plate_1, plate_2, mug_0, mug_2, bowl_1, bowl_5, fork_0, spatula_0, sponge_0 | `assets/x5/objects/<name>/` (visual mesh USD + texture, `SOURCE.json`) | [robocasa/robocasa-assets](https://huggingface.co/datasets/robocasa/robocasa-assets/tree/1b92c3d02ca4354984fec961357db0bff7b32166) `aigen_objs.zip` @ `1b92c3d0`; bounded range reads | CC BY 4.0 (dataset card); generated meshes, not scans | A2–C3 |
| Sugar jar (A1, since 2026-11-19) | `assets/x5/objects/sugar_jar_0/` (body mesh USD, raw GLB, `SOURCE.json`; `scripts/household/import_sugar_jar.py`) | "Glass Jar" by Extrone, [Sketchfab 4823762b…](https://sketchfab.com/3d-models/4823762be90c437ea35d45dfa976e085), via the Objaverse 1.0 mirror; cap left out, scaled ×0.305 | CC BY 4.0 | A1 |
| Alarm clock (promo scene `alarm_x5`, 2026-11-22; not a benchmark task) | `assets/x5/objects/alarm_clock_0/` (USD split into body, bells, hammer, glass and hands; raw glTF with 1k textures; `SOURCE.json`; `scripts/household/import_alarm_clock.py`) | "Alarm Clock 01" by James Ray Cock and Yann Kervran, [Poly Haven](https://polyhaven.com/a/alarm_clock_01); native size | CC0 | promo video |
| Mug, draining bowl (historical A1) | `assets/x5/reused/mug.usd`, `bowl.usd` | YCB-Video derivatives | MIT (`assets/x5/reused/LICENSE`) | none since 2026-11-19 |
| Stirring spoon (A1) | `assets/x5/reused/spoon` | HANDAL derivative | CC BY-NC-SA 4.0 (`assets/x5/reused/spoon/LICENSE`) | A1 |
| Tea infuser, bread/toast slice | `assets/x5/reused/mesh_infuser.usda`, `toast.usda` | project-authored proxies (historical household repo) | project | B2, C2 (the infuser: A1 until 2026-11-19) |
| HDR environment | `assets/x5/textures/studio_small_09_1k.hdr` | Poly Haven, Sergej Majboroda | CC0 | all |
| Authored scene elements | defined in `<id>_x5/scene.py`; textures from `x5/textures.py` | this package (deterministic, seeded) | project | see below |

Authored proxies (disclosed; appearance only unless noted):
- **A1:** glass of tea (lathe, OmniGlass; the tea a tinted clear volume), sugar mound (cone) and falling stream (cylinder), sugar inside the jar (cylinder), all sized from the task's grams.
- **A2:** countertop ice dispenser body (recess, drip tray, nozzle housing, octagonal outlet), 56 mm glass tumbler (lathe, OmniGlass), crescent ice cubes (8.5 g rigid half-disc prisms, opaque frosty material).
- **A3:** pancake (rigid 12.4 cm disc plus textured top), blue gas flame ring.
- **A4:** 500 ml glass jug (lathe, OmniGlass, graduation decal), digital scale with "0 g" LCD, recipe card.
- **B1:** counter slabs around the sink cut-out (native terrazzo material), water surface (OmniGlass), soap foam, faucet stream.
- **B2:** wedge cutting board, pegs, cup, egg, pill bottle and apple (surfaces of revolution, `b2_x5/scene.py`).
- **B3:** flame silhouettes, oil surface, glass tumbler with water.
- **C1:** food surfaces in bowls, serving board, order card.
- **C2:** egg-soaked bread texture, order card.
- **C3:** place mats, sauce stains, water rings, seating card.
- **All tasks:** robot mount plates.

A2 (R3.1, 2026-11-03) uses only the dispenser paddle of the native Refrigerator001 (`Refrigerator001_Switch`, Lightwheel kitchen) at ×0.85, referenced as a separate part on its native top hinge, with its native material; the countertop ice dispenser around it is authored. (Earlier A2 versions: the whole refrigerator at ×0.85, and a countertop oven from WallStackOven004 ×0.6 with a RoboCasa baking sheet, kept in `a2_oven_x5/`.)

Candidate assets that were downloaded for selection but not used remain outside this package in `/opt/household/V1.0_X5/assets/robocasa/aigen_objs`. `scripts/household/collect_x5_assets.py` re-fetches the selected ones.
