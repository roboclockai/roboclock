# Household assets and physical scope

Dual-X5 robot and object assets: [x5-assets.md](x5-assets.md). The Panda, Ridgeback and
RoboLab-scan assets of the earlier Panda/R1 demos were removed on 2026-11-21.
Exact dimensions, assumed masses, friction and collision geometry are recorded in
the selected implementation's catalog and geometry/design files. These are simulation parameters,
not hardware calibration.

| Collection | Source / provenance | Bundle scope |
| --- | --- | --- |
| Kitchen, pot, toaster, orange | [Lightwheel Kitchen](https://github.com/LightwheelAI/Lightwheel_Kitchen), commit `10b0f7a86135a27e8c2fba9c690f6bf5c4f06bb1`, CC BY-NC 4.0 | The 1.4 GB room is an external local dependency. Existing persistent collection: `$WORK_DIR/household/assets/lightwheel_kitchen/Collected_KitchenRoom`; active copy: `/opt/household/assets/lightwheel_kitchen/Collected_KitchenRoom`. Read the active copy during simulation. |
| Procedural interaction fixtures | Project-authored USD geometry and textures | Tea basket, oven rails/dough, food pieces, utensil rests, articulated fridge/lids/knob, receiving console, door/pull bars, tray and open pantry. Actual collision bodies/constraints are retained. |

The kitchen dependency is deliberately not duplicated into Git. The personal server
already has both local and persistent copies. For another server, collect the pinned
upstream kitchen into the same local asset directory before running.
Reusable collection/build scripts are under `scripts/household`.

Manipulation is rigid-body physics with measured contacts. Steeping, readiness,
cooking, ignition/extinction, refrigeration deadlines and residue drying are symbolic
processes with compressed simulation time. No fluid, combustion, thermal, cutting,
flexible-food or cloth dynamics are claimed. The controllers use privileged feedback;
these movies demonstrate feasibility, not VLM/VLA policy performance.

B1 opens an initially closed **unlatched** door through gripper/pull-bar contact.
It does not model latch unlocking. The mounted tray is physically fixed to the base;
groceries remain independent dynamic bodies. After returning fully inside, measured
door closure precedes transport. Released food requires correct shelf support and
full containment; settling uses a continuous 0.5-second position span of at most
3 mm, with displacement beyond 6 mm after receipt rejected. All frame capture freezes
physics time. The approved `rt_stable` lighting/capture profile is retained.
