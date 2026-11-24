# RoboClock usage guide (detailed)

An extensible Isaac Sim benchmark for application-derived timing requirements.
The first environment, `lab`, implements sixteen atomic laboratory tasks:
**6 temporal-window, 7 event-reactive, and 3 throughput tasks**, with
**16 tasks × 3 difficulty levels × 10 seeds = 480 episodes per policy**.

Physical manipulation uses **Dual ARX X5**, two parallel-jaw grippers, rigid-body
collision/contact physics, and an RGB camera with optional depth. Liquid volume, phase interfaces, chromatographic bands, indicator color and
alarms use explicit process models that drive visible geometry and materials. All timing uses elapsed **physics simulation time**.

The implementation follows [RoboDojo's](https://github.com/RoboDojo-Benchmark/RoboDojo)
separation of scene/simulator configuration, task logic, policy integration, and
evaluation artifacts. It is an independent implementation using Isaac Sim's core
APIs; RoboDojo, Isaac Lab, and XPolicyLab are not runtime dependencies.

## Household branch: latest ten demos

This branch adds the current household task definitions, simulation implementations,
assets and one accepted showcase per task (A1–A4, B1–B3, C1–C3).
Start with the [household guide](household/README.md) or the
Dual ARX X5 demonstrations.
`roboclock tasks --env household` lists the latest definitions;
`scripts/household/run_x5_episode.py` runs and records one episode.
Recordings retain their actual source revisions and fixed-seed evaluation scope.

## Sports / games: current demonstrations

The current ten sports/game demonstrations use fixed Dual ARX X5 arms on one common
rig (arm mounts 0.60 m apart, rig head camera); each recording has a presentation
overview and a head-camera video.
Videos and original results ·
[Task guide](sports-current.md) ·
Diagnostic evaluation and known failures.
The current checkout keeps one accepted recording per task; previous recordings,
raw traces and source snapshots remain in Git history.
`roboclock eval --env sports` evaluates the same ten tasks through the shared evaluator
([Sports tasks](#sports-tasks-a1c3)).

## Assets

Scenes, models and textures are distributed as the Hugging Face dataset
[Roboclockai/roboclock-assets](https://huggingface.co/datasets/Roboclockai/roboclock-assets)
(2.2 GB). Its folders mirror this repository: scene-level assets in `assets/<domain>/`, package
assets next to their environment code. Download it and copy it over this checkout, then link
the kitchen into the household package:

```bash
hf download Roboclockai/roboclock-assets --repo-type dataset --local-dir ../roboclock-assets
rsync -a --exclude README.md --exclude .gitattributes --exclude .cache ../roboclock-assets/ ./
mkdir -p src/roboclock/envs/household/assets/x5/lightwheel_kitchen
ln -sfn ../../../../../../../assets/household/lightwheel_kitchen/Collected_KitchenRoom \
  src/roboclock/envs/household/assets/x5/lightwheel_kitchen/Collected_KitchenRoom
```

After that, assets/README.md lists where each domain's assets live.
The dataset already contains the part of the Lightwheel kitchen the Dual-X5 household tasks
use; `python scripts/household/fetch_lightwheel_kitchen.py` re-downloads it from the upstream
project instead.

## Installation

The simulation backend is pinned to **Isaac Sim 5.1.0 and Python 3.11**. Isaac Sim **6.0.1 with Python 3.12** is also supported through its bundled legacy core APIs. New recordings identify the actual simulator version. NVIDIA's
[installation instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
specify Python 3.11, NVIDIA's package index, and Linux GLIBC 2.35 or newer.
**Isaac Sim 6.0.1 (Python 3.12)** also works and is needed on machines with NVIDIA driver 595, where
Isaac Sim 5.1 crashes at startup. Use a separate venv so the 5.1 install stays intact:

```bash
python3.12 -m venv venv60 && venv60/bin/python -m pip install --upgrade pip
venv60/bin/pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
venv60/bin/pip install --no-deps -e .   # then: pip install PyYAML gymnasium Pillow if they are missing
export OMNI_KIT_ACCEPT_EULA=YES
```

`scripts/isaac_api_probe.py` checks that every Isaac Sim API this repo uses is available.

The chess recordings and the superseded sports A1 recording were made with **Isaac Sim 6.2.0 and Python 3.12**
(`pip install -e .[sim62]`); the current sports recordings use Isaac Sim 6.0.1. See the
installed environment and upgrade validation.

Use an NVIDIA RTX GPU and a compatible driver; see the
[system requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
On minimal Ubuntu images, install the graphics libraries with
`apt-get install libglu1-mesa libxt6 libvulkan1 vulkan-tools`. For containers,
enable NVIDIA's `graphics` driver capability as described in the
[NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
Headless camera rendering still requires a working NVIDIA Vulkan driver.

From this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
roboclock doctor
roboclock smoke --task A1 --steps 60
```

`requirements.txt` installs the editable package with its `sim` extra through
NVIDIA's package index. Dependency versions and the `sim`/`dev` groups are
defined once in `pyproject.toml`. The first simulator launch follows NVIDIA's
normal license acceptance and asset-download flow.
The official ARX X5A meshes, adapted URDF, local USD, and provenance are bundled in
`src/roboclock/envs/lab/assets/arx_x5`. Rebuild the USD with
`PYTHONPATH=src python scripts/lab/prepare_arx_x5.py`. Both arms are present in every
task on one common rig: the two bases stand side by side at the same height, both
facing the work area, 0.60 m apart center to center, and the head camera is fixed
0.04 m ahead of and 0.54 m above their midpoint (`robot.rig` in
`src/roboclock/envs/lab/configs/benchmark.yaml`; only the rig's position, yaw and
mount height vary by task). Raised mounting columns provide downward access to the
lab fixtures. The policy observation (`rgb`/`depth`, and XPolicyLab `cam_head`) comes
from this head camera (`sim.observation_camera: head`, the default since 2026-11-03);
set `observation_camera: scene` to get the fixed third-person camera used by earlier
policy evaluations and by the third-person recordings.

If Isaac Sim is already installed as a standalone application, install this
package in its Python environment and use its `python.sh`:

```bash
/path/to/isaac-sim/python.sh -m pip install -e .
ROBOCLOCK_PYTHON=/path/to/isaac-sim/python.sh bash scripts/roboclock.sh smoke --task A1
```

Task planning, metrics, and tests work without Isaac Sim on Python 3.11 or 3.12:

```bash
python -m pip install -e '.[dev]'
roboclock tasks
roboclock plan --task all --difficulty all --seeds 10 > plan.json
python -m pytest -q
```

Installing the lightweight package does not install the simulation runtime.
For Isaac Sim 6.0.1, create a Python 3.12 environment and install `.[sim6,dev]`
using NVIDIA's package index. The runtime enables the legacy core, manipulator,
camera and Lula extensions bundled with 6.0.1.

## Chess move and clock environment

The `chess` domain adds a complete standard-chess rules engine, labeled cylinder
pieces, an existing CC0 chessboard/clock model, and contact-validated placement
followed by a physical clock press. D starts at the opponent's measured press.
The environment assigns a legal move; each episode evaluates one robot turn.
It runs as sports A1 on the Dual ARX X5 (`--env sports`); the standalone single-Franka
`--env chess` was removed on 2026-11-21. See [chess rules, assets and limitations](chess.md).

## Sports tasks (A1–C3)

The current ten tasks use fixed Dual ARX X5 arms: A1 chess, A2 basketball;
B1 moles, B2 boxing, B3 bench-press assistance, B4 stop/go puck pushing, B5 taiko;
C1 pool racking, C2 Speed Cups and C3 ball service (renumbered 2026-11-19: taiko was A2,
basketball A3). Source is consolidated under
[`src/roboclock/envs/sports`](../src/roboclock/envs/sports/README.md); the numbered
`sports_v2`–`sports_v5` packages are now descriptive internal subpackages.
See [current tasks and validation](sports-current.md) and
latest videos.

`--env sports` runs all ten through the shared evaluator with the shared 16-dim
Dual-X5 joint-position action and laboratory observation keys; each task keeps
its own scene, recorded physics/control rates and native scoring of measured
physical state:

```bash
roboclock tasks --env sports
OMNI_KIT_ACCEPT_EULA=YES roboclock eval --env sports --policy oracle --oracle \
  --task A2 --difficulty easy --seeds 1 --output results/sports/unified/A2
roboclock summarize --normalize results/sports/unified
```

The recording driver used for the videos remains available:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src \
  .venv-isaac6/bin/python -m roboclock.envs.sports \
  --task B1 --difficulty easy --seed 0 --video --output results/sports/b1
```

The oracles (`--oracle`) are the recordings' privileged scripted controllers;
A1's opponent (BLACK) is part of the environment. The original single-Franka numeric tasks
31–40 (`--env sports_legacy`) were removed on 2026-11-21.

## Laboratory tasks

| ID | Task | Category | Physical goal / timing measurement |
|---|---|---|---|
| A1 / T1 | Water collection before overflow | Window | Start with an empty beaker, open the faucet (a 10 mL/s stream at every level), then close it with 440 / 460 / 480 mL to 500 mL collected (easy / medium / hard) before the 500 mL beaker overflows |
| A2 / T2 | Liquid phase separation | Window | Close stopcock before the upper phase contaminates the receiver |
| A3 / T3 | Chromatography fractions | Window | Exchange collection tubes at visible colored-band boundaries |
| A4 / T4 | Titration endpoint | Window | Stop delivery at the pale-pink endpoint before excess titrant |
| A5 | Photographic film development | Window | Load a film reel into developer from a dry stand, start the countdown on full immersion, keep it there for the full 12 s demonstration, then lift it out, immerse it in the stop bath and release it |
| A6 | Heat-shock ice transfer | Window | Take the capped sample from the bench stand into the warm bath, remove it 30 s after immersion, then place and release it in the ice bucket |
| B1 | Hotplate over-temperature isolation | Reactive | After the alarm, retain an upright lift at least 2 cm clear of the hotplate within 4/2.43/1.48 s; then release into the safety tray within 12/8.73/6.35 s |
| B2 | Thermal short-circuit prevention | Reactive | When the load's upward-facing status LED turns from green to red (overheat), physically open the knife switch within 4/2.74/1.88 s; timeout melts insulation and the same LED flashes red |
| B3 | DC motor stall isolation | Reactive | Hold the initial robot posture until the horizontal, guarded test wheel stops, then move to open the knife switch within 4/2.82/1.99 s |
| B4 | Tipping test-tube rescue | Reactive | Start with an upright half-filled tube; react only after near-fall tilt and retain an upright lift before excessive spill or the 4/2.87/2.06 s deadline |
| B5 | Freezer door rescue | Reactive | Grasp and latch the ajar insulated door within 3/2.47/2.03 s of the alarm |
| B6 | Vacuum bumping relief | Reactive | Press the vacuum breaker through its full stroke within 2/1.41/1 s |
| B7 | Waste carboy overflow | Reactive | When the bench console's LEVEL lamp turns from green to red (carboy IR high level), clamp the incoming waste hose within 2/1.41/1 s |
| C1 | Test-tube rack organization | Throughput | Sort mixed rear tubes into three front racks, one color per rack; score correct operations/minute and accuracy over the full time budget |
| C2 | Rack loading | Throughput | Repeated upright insertion into matching color slots |
| C3 | Calibration placement | Throughput | Match sample and marker colors, position, and directional orientation |

C1 uses the supplied pack-3 test-tube racks in the C2/C3 daylit room. See
the original C1 notes for asset provenance
and scoring. The task demonstrations use Dual ARX X5.

B1 uses the same v2 laboratory background, imported furniture, mountain panorama,
and lighting as A1. A nearly dry beaker rests on a hotplate; a scripted temperature
rise triggers a flashing red numeric panel. The robot must lift the **whole beaker**
at least 2 cm above the heating surface, with measured bilateral grasp and no heater
contact, and retain that lift for 50 ms. Merely touching or grasping the beaker on
the plate is insufficient. The strict isolation windows are 4/2.43/1.48 seconds for
easy/medium/hard. Released, settled placement in the safety tray must follow within
12/8.73/6.35 seconds of the alarm. (Lab B medium/hard limits are calibrated against the
scripted oracle, a reference controller: see `reactive_tasks` in
`src/roboclock/envs/lab/configs/benchmark.yaml`.) Early grasp/removal, returning to the hotplate, and
missed deadlines remain failures even after later recovery. Physical completion
and temporal success are reported separately.

These are benchmark deadlines, not a prediction of glass fracture. Overheating
is recorded as a latched failure; glass fragmentation and fluid spills are not
simulated. The task uses the removal response; pressing the model's decorative
power switch is not an alternative scoring action. The RGB observation exposes
the alarm visually. Recorded videos add a synchronized synthetic two-tone warning;
the environment does not expose microphone observations or simulate acoustics.

The hotplate mesh comes from MilkAndBanana's CC0
[Lab Assets](https://opengameart.org/content/lab-assets). Its source model, license,
atlas, converted USD and provenance are bundled under
`src/roboclock/envs/lab/assets/hotplate/`; `scripts/lab/prepare_hotplate.py` reproduces
the import. The beaker has an open glass visual and a conservative cylindrical
collision envelope. The B1 demonstration is in `data/lab/B1/`.

B5–B7 use a contact-driven hinged door and spring plungers. Their strict
response windows end at measured mechanical closure, followed by a 0.2-second
retention check. The robot holds its initial joint command until the observed alarm; pre-event
reaching or wrist rotation is a violation. B5 begins closed: an internal catch fault releases the spring hinge,
and measured opening triggers the alarm. No detached part is shown. The diagnostic uses privileged state and is not a trained visual policy. B6 adapts
the requested apparatus to a vacuum desiccator, B7 uses the hose-clamping response,
and B5 uses an authored insulated cabinet. B6 and B7 import meshes from the same CC0
Lab Assets pack. See the original asset decisions and task contracts.
The B2 status LED, the B3 test wheel and the B7 LEVEL lamp sit inside both the fixed
head camera and the task arm's resting wrist camera, so each cue can be seen before any motion.
After the cue, the idle right arm frames the cue and the actuator together. Each lamp state is
a pre-coloured lens, and only the active one is scaled to full size: RTX shows a transform
change on the same frame, while visibility or material edits reach the image several frames late.
data/lab holds one Dual ARX X5 oracle demonstration per task.
The thermal/pressure consequences are compressed scenario states; neither sample
thawing nor fluid splashing is a calibrated physical prediction.

Easy/medium/hard primarily change window widths, reaction bounds, required
operation rates, and placement tolerances. Given a seed, geometry, object count,
and physical randomization stay constant across difficulty levels. A1, A2 and A4
derive their live deadlines from delivered volume and measured valve opening;
A3 uses separate collection intervals for four visible fractions. See the
liquid task descriptions for apparatus, cues and timing rules. Spatial placement,
orientation, mass, friction, lighting, camera position, event times, and process
parameters use independent seeded sampling. The default throughput sample range
is 20–30; configure 10–50 in `num_samples_range` with compatible rate targets.
Configuration validation rejects rates that exceed the available samples per
fixed time budget.

## Hospital environment

`--env hospital` places every task where it happens in the left bay of the scene-02 ward
(`assets/hospital/ward_02_preview_left.usdc`, built from the supplied pack by
`scripts/hospital/prepare_preview_left_ward.py`), with textured props (CC0 sets in
`src/roboclock/envs/hospital/assets/textures/sources.json`). It has 14 tasks:

- A1–A2 process windows: drip chamber level, sequential bag switch; A3 bag-valve ventilation of
  an intubated manikin, one squeeze per 6 s breath window (since 2026-11-19; the syringe pump
  cutoff until then); A4 hang bag then start pump (B6 until 2026-11-19).
- B1–B3 emergencies; B4–B6 bedside reactions: ventilator circuit reconnection (since
  2026-11-19; the infusion-line switch until then), anaphylaxis: adrenaline auto-injector into
  the thigh, held 3 s (since 2026-11-20; hang bag then open clamp until then), massive limb
  haemorrhage: turn the tourniquet's windlass rod half a turn into its clip (since 2026-11-20;
  the pump key until then). In A4 the instruction shows from the start, the robot
  starts tucked at its base and finishes every step within the limit; in B4, B5 and B6 the robot
  also starts tucked, and the limit runs from the seeded moment the circuit comes off (B4),
  the monitor alarms (B5) or the wound starts to bleed (B6).
- C1–C4 two-arm throughput: pharmacy order bins, instrument tray sort, crash-cart vials,
  blood-tube triage.

One oracle demonstration per task is on the [task page](https://roboclockai.github.io/tasks.html).

```bash
python scripts/hospital/prepare_preview_left_ward.py
roboclock tasks --env hospital
python -m pytest tests/envs/hospital -q
```

## Running policies and evaluation

```bash
# A simulator smoke test, with an interactive window:
roboclock smoke --task A2 --gui --depth --steps 60

# All 480 episodes using the hold-position baseline:
roboclock eval --policy hold --task all --difficulty all --seeds 10

# A privileged task-specific IK diagnostic:
roboclock eval --policy oracle --oracle --task A1 --difficulty easy --seeds 1

# A custom policy, charged its measured inference time plus a fixed control
# delay of 0, 0.1 or 0.5 s per query (1530 episodes):
roboclock eval --policy my_policy:Policy --task all --difficulty all \
  --seeds 10 --control-delay 0 0.1 0.5

# Summarize an existing run:
roboclock summarize results/<run>/episodes.jsonl
```

`--output DIR` selects a result directory. Existing episode artifacts are never
overwritten. Each run writes `manifest.json` (configuration, assets, policy,
episode plan, oracle flag), `episodes.jsonl` (metrics and event log), and
`summary.json` (overall, task/difficulty/latency, and category aggregation).
Use `--task A1,B1` or `--difficulty easy,hard` to select subsets and `--seed-start`
to choose a seed range. `--config PATH` and `--asset-catalog PATH` select custom
YAML configuration.

The diagnostic oracle is a measured-pose/contact state machine using Isaac
Sim's Lula inverse kinematics solver. It issues the same joint actions as other policies and
uses privileged task state. It is **not a validated expert or a guaranteed
successful baseline**, especially for catches, narrow slots, and moving targets.
Oracle results are explicitly labeled in the artifacts.

An external policy factory receives no environment or scoring state:

```python
class Policy:
    def reset(self, observation):
        # Initialize recurrent state for this episode.
        pass

    def act(self, observation):
        # Return 16 finite joint targets: left arm, then right arm (left = the rig's
        # +y arm, the one `left_wrist_rgb` rides on).
        # Per arm: 6 arm positions (radians), then 2 finger positions (meters).
        # Finger limits are [0, 0.044]; arm limits follow the bundled X5 model.
        return observation["joint_positions"].copy()
```

Policies receive `rgb` (H×W×3 uint8), optional `depth` (H×W float32, meters),
`joint_positions`, `ee_pose` (world position + wxyz quaternion at the grasp
center), `gripper_width`, `instruction`, and elapsed `sim_time`. Every task concatenates
the left and right arms in that order: 16 joint positions, 14 pose values,
and 2 gripper widths. "Left" is the physical side: the arm on the rig's +y side (rig
frame +x forward, +y left), the one `left_wrist_rgb` rides on, as in every RoboDojo
domain (lab and hospital since 2026-11-04; `episode_metadata()["joint_order"]` lists the
labels). In lab and hospital single-arm tasks the working arm is the rig's RIGHT arm,
i.e. action values 8-15; values 0-7 drive the idle left arm. A single-arm policy must hold
the unused arm's targets.
C1–C3's diagnostic policy gives each sample to an arm that can reach both its
pick and place pose (measured X5 reach map) and balances the shared ones. A
separation guard keeps the two side-by-side arms' footprints 15 cm apart: the arm with
less delicate work backs off toward its own mount, and jobs whose swept area
overlaps the other arm's current job wait. Placement goals are computed from the
measured object-in-hand offset with integral correction, descents stay above the
object until centred, and a held object is never released in mid-air: a failed
transfer is re-targeted or carried back to its source. On the side-by-side rig
each arm works the half of the C1–C3 layout in front of it (C1's blue rack sits
between the arms and is shared); an arm whose queue is finished returns to its
start pose. The X5 link2/base_link convex-hull contact is filtered for C1–C3 (it
blocked the shoulder at 2.45 rad when the arm points sideways). Instructions
state reaction bounds and throughput requirements. Future event timestamps,
object poses, process scalars, and scoring predicates are absent. Normal
Gymnasium `info` is empty; explicit `oracle=True` enables privileged diagnostics.

`LabEnv.reset()` and `step()` use the Gymnasium signatures. An action holds joint
targets for five physics ticks by default (250 Hz physics / 50 Hz control).
`advance_latency(seconds)` continues physics with the previous motor command,
updating every process/event and checking every physics tick. Evaluation applies
the action computed from the observation taken **before** that delay. Latency is
rounded upward to a physics tick and capped by the episode budget. The evaluator
charges each model query its inference time plus `--control-delay` (block-and-replay,
see [docs/architecture.md](architecture.md#inference-latency)); use
`--inference-latency zero` to pause simulation during inference instead.
Block-and-replay is the benchmark's time model and the default. `--time-model streaming`
is an optional alternative: the robot keeps executing the previous actions while the
policy computes, with observations every policy period
([streaming mode](architecture.md#streaming-mode)).

## Scoring

Task/manipulation success measures the physical goal. Temporal success measures
timing compliance; joint success requires both. A temporal violation stays
latched while manipulation can continue, preserving the distinction between
eventual physical completion and application success. Legacy placement-window endpoints are inclusive; the
liquid overflow, contamination and over-titration boundaries are excluded. Reactive bounds are strict (`latency < max_latency`); B1 measures completed retained isolation, and its transfer
deadline is checked separately.

Placement requires release, measured destination geometry, low velocity, and a
0.2-second dwell. The timing timestamp is the first insertion or entry, before
settling. Upright cylinder insertion ignores yaw symmetry; calibration objects
use full quaternion angular distance and visible directional markings. Drops
are detected below the table; catches also reject table contacts. Grasping
requires contacts with both fingers, a closed gripper, and proximity to the
measured tool. Collision count records onset of arm contact with the table or
fixtures (finger/sample manipulation contacts are excluded).

Throughput always runs for the full time budget: `N_correct / T` cannot be
inflated by terminating early. Objects earn credit once; misplaced or dropped
objects contribute to attempted-operation accuracy. Physical success additionally
requires every sample to finish in its destination. Reports include completion
time, actions, collisions, drops, reaction/first-motion latency, temporal error,
operations per minute, accuracy, and placement errors where applicable.

## Assets and project layout

The default scene uses the bundled official ARX X5A geometry for both arms and clearly identified primitive
object proxies, with physical open trays/racks built from floors and separate
collision walls. The current A1–A4 scenes also use the user-supplied laboratory packs under
`assets/`, converted into visual USD backgrounds under `assets/lab/`.
Task apparatus uses explicit authored geometry and physical valve/rail joints.
External packs are visual references, not SimReady collision assets. To use RoboLab,
VOMP/HOPE/YCB/HANDAL, or other licensed SimReady assets, set `usd_path` and `scale`
in `src/roboclock/envs/lab/assets/catalog.yaml` or a supplied catalog. Each asset must have
one rigid root and fit the task's grasp/slot envelope. Set `radius`/`height` to
its dimensions after scaling; layout heights and probe endpoints use these
dimensions. The importer configures rigid mass, collision, and friction and
preserves authored appearance and collision approximations. Custom assets must
supply matching visible identities. Verify them with a GUI smoke test.

```text
pyproject.toml         Package metadata and dependency definitions
requirements.txt       Isaac Sim installation entry point
src/roboclock/
  envs/                Environment registry, public contract, lazy factory
    lab/               Laboratory implementation, independent of other domains
      configs/         Laboratory configuration and difficulty definitions
      assets/          Laboratory catalog, textures, provenance and licenses
      mdp/             Laboratory state, process events and geometry predicates
      tasks/           A1–A4 temporal-window / B1–B7 reactive / C1–C3 throughput
      policies/        Laboratory trajectories and diagnostic oracle
      validation.py    Laboratory geometry/contact/physics probes
      clutter.py       Seeded non-task tabletop rigid bodies
      videos.py        Offline recording of actual task episodes
  simulators/isaac/    Shared runtime, diagnostics, rendering and visual mesh tools
  assets/shared/       Assets reusable across domains (currently HDR lighting)
  physics/             Reusable contact tracking, rotation and pose math
  tasks/registry.py    Environment-scoped task dispatch
  policies/base.py     Hold baseline and policy loading
  evaluation/          Shared episode plans, latency runner, metrics, validation dispatch
  resources.py         env://, lab:// and shared:// resource resolution
  cli.py               envs / tasks / doctor / plan / render / validate / smoke / eval / summarize
scripts/lab/           Laboratory texture generation and fixture validation
scripts/roboclock.sh     Entry point supporting Isaac Sim's python.sh
tests/envs/lab/        Laboratory task and environment tests
tests/simulators/      Shared simulator tests
docs/                 Architecture, materials, and recorded validation evidence
data/<env>/<task>/    One oracle demonstration per task (preview, sidecar, linked HDF5)
skills/build-robotics-simulation/  Simulation authoring skill and references
```

Select domains with `--env`; omitted `--env` still selects `lab`:

```bash
roboclock envs
roboclock tasks --env lab
roboclock smoke --env lab --task A1
```

New domains keep scene, tasks, configuration, controllers and assets under
`envs/<name>/`, then declare an `EnvironmentSpec`. Installed domain packages can
register through the `roboclock.environments` entry-point group. Shared commands
and evaluation do not import the laboratory implementation directly. Task IDs
are scoped by environment, and result artifacts include the environment name.
Default outputs are organized under `results/<env>/`.
See [architecture and extension contracts](architecture.md) for the public
interfaces, asset namespaces, plugin registration and import-path migration.

V1 runs one environment per process and rebuilds the scene for each seeded
episode. The shared `IsaacRuntime` owns Kit; closing it uses Kit's process-exiting
fast shutdown. Close individual environments between episodes and close the
runtime only when the simulation process is finished. Additional domains can
now be added through the registry; parallel environment cloning, real robot
support and policy servers remain outside this release.

## Rendering and physics validation

```bash
# Actual RTX camera image, metadata, and optional USD stage:
roboclock render --task C2 --seed 0 --config docs/images/render_config.yaml \
  --output results/laboratory.png --usd

# 153 actual-PhysX trajectory diagnostics, including independent rack drop probes:
roboclock validate --task all --difficulty all --seeds 3 --seconds 45 \
  --rack-probes --images --output results/physics

# Full-budget policy evaluation remains a separate benchmark command:
roboclock eval --policy oracle --oracle --task all --difficulty all --seeds 10

# Current liquid-process videos (requires FFmpeg/libx264):
PYTHONPATH=src python -m roboclock.envs.lab.videos --tasks A1 A2 A3 A4 \
  --renderer PathTracing --samples 32 --subframes 1 --fps 5 \
  --resolution 1280 960 --output results/lab/liquid-videos

# Current B1, sharing the A1 v2 room:
PYTHONPATH=src python -m roboclock.envs.lab.videos --tasks B1 \
  --renderer PathTracing --samples 32 --subframes 1 --fps 5 \
  --resolution 1280 960 --output results/lab/videos-v2

# V2 equipment faults and test-tube rescue:
PYTHONPATH=src python -m roboclock.envs.lab.videos --tasks B2 B3 B4 \
  --renderer PathTracing --samples 32 --subframes 1 --fps 15 \
  --resolution 1280 960 --output results/lab/videos-v2

# Optional C1 grasp-assist diagnostic: 24 tubes, 180 s (not the benchmark profile):
PYTHONPATH=src python scripts/lab/render_throughput_videos.py --tasks C1 --expert --subframes 1 \
  --output results/lab/c1-videos

# Throughput diagnostics:
PYTHONPATH=src python -m roboclock.envs.lab.videos --tasks C1 C2 C3 \
  --output results/lab/c-videos
```

The showcase recordings that `data/lab/videos-v2/` held until 2026-11-06 (last in commit
17295d4) predate the shared 250/50 Hz time base and keep their original rates. The original B/C recordings use Isaac Sim's RTX renderer at 1280×960 and 15 fps, with
120 Hz physics and 30 Hz control. Throughput tasks retain the full 120-second
budget. Individual MP4 sidecars record seeds, actual outcomes, frame timestamps,
clutter poses and contacts, and capture-clock checks. The diagnostic controller
uses privileged state; these recordings are not guaranteed successful expert
demonstrations. Use `--renderer PathTracing --samples 32 --subframes 4` for
slower offline path tracing. Existing videos are preserved by refusing overwrite.
`roboclock eval` renders every lab task, liquid tasks A1–B4 included, with RaytracedLighting
(`liquid_visuals.renderer` in the lab benchmark.yaml; PathTracing there was the default before 2026-11-03).
The revised A1–A4 liquid-process videos use PathTracing at 1280×960 and 5 fps;
1920×1440 material previews use 512 samples per pixel. The revised scenes include
a detailed pack 3 cabinet bench and original pack 5 computer/chair meshes, an 8K marble desk and mountain windows.
One oracle demonstration per task is indexed in data/lab;
task definitions are in this README.
Add `--view first_person` to record from the rig's head camera (0.04 m ahead of and
0.54 m above the midpoint of the two bases, 55 degrees below the horizon). These recordings keep one view throughout and use the
`_first_person.mp4` suffix, so both views can share an output directory.

Use `--view first_person_wrist --active-observation` for synchronized center, left-wrist,
and right-wrist panes. The otherwise idle right arm moves to an inspection pose while
the left arm operates; cooperative tasks retain both arms for manipulation. These
recordings use the `_first_person_wrist.mp4` suffix. Each sidecar includes the rigid
wrist-camera extrinsics, per-frame poses, and measured inspection reach time.



Previous render · Calibration render

See [materials and labels](materials-and-labels.md) for PBR values, HDR
attribution, label semantics, camera conventions, and collision/appearance
boundaries. See trajectory and validation notes for the
historical results and limits of the diagnostic controller. The pre-migration 90-run
matrix passed all physics checks, with 46 physical goals and 45 joint timing
goals reached. Insertion, moving-probe, catching, and throughput control still
have recorded failures; the diagnostic policy is not an expert baseline.

`validate` uses actual rigid bodies, contact reports, and articulation motors.
It checks initial support/overlaps, slot clearance, finite PhysX states,
release events, continuous moving-fixture motion, and optional gravity seating
experiments. The fixture experiments initialize objects above the rack before
stepping physics; their results never count as benchmark manipulation scores.
Trajectory diagnostics use privileged state and proprioception with rendering
at setup/export. Their partial rollouts are not fresh-RGB policy evaluations or
full-budget throughput scores. `eval` preserves the normal sensing and full
120-second throughput budget.

On Linux, the runtime and `doctor` test Vulkan GPU enumeration. When the default
NVIDIA GLX ICD fails but NVIDIA EGL enumerates the GPU, headless execution can
select an owned temporary EGL ICD for this process. Existing `VK_ICD_FILENAMES`
and `VK_DRIVER_FILES` choices are respected; system driver files are unchanged.
This fallback was necessary for the development container's RTX 4090.

## Industry41–50 — Dual ARX X5

Industry recordings, per-video JSON and collection verification:
data/industry/videos-v2.
Generation sources, exact replay inputs and imported dependencies follow the lab
layout; see [Industry setup and replay](industry/README.md) and
current task descriptions. These are native
privileged-controller diagnostics; the generic policy-evaluation adapter is not
yet registered. Existing lab tasks and assets are preserved.

## Sports / games gallery

Watch the ten current demonstrations.
The [task guide](sports-current.md) lists commands and runtime requirements.
Validation distinguishes selected successful
videos from the 90-case diagnostic matrix and known failures.
