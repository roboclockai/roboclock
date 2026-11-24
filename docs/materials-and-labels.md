# Laboratory materials, textures, lighting and labels

The current previews are actual Isaac Sim 5.1 RTX camera frames:
rack loading and
calibration placement.
The previous scene is preserved for comparison:
rack loading before and
calibration before.

## What was taken from the RoboDojo reference

The [RoboDojo website](https://robodojo-benchmark.com/), its simulation examples,
and the [official repository](https://github.com/robodojo-benchmark/RoboDojo)
were inspected. The source checkout was commit
`726e9aabfaa642203722eb126f5eaf0f37f3e1ad`.
The concrete reference points were:

- [Default scene configuration](https://github.com/robodojo-benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/env_cfg/scene/default.yml): explicit room, table material, floor material and 4K HDR environment.
- [Table material binding](https://github.com/robodojo-benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/env/scene_manager/objects/table.py): MDL material assignment and mesh/UV handling.
- [HDR background](https://github.com/robodojo-benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/env/scene_manager/objects/background.py): a textured dome and extra subframes for material changes.
- [Camera render synchronization](https://github.com/robodojo-benchmark/RoboDojo/blob/726e9aabfaa642203722eb126f5eaf0f37f3e1ad/env/camera_manager/capture/render_sync.py): startup settings that wait for Hydra rendering before reading image buffers.

RoboClock applies those material/lighting/synchronization principles to its own
laboratory geometry. It uses NVIDIA's MDL OmniPBR and OmniGlass, with a separate
path-traced still-image profile. That profile is our configuration, not a claim
that RoboDojo uses the same settings. RoboDojo's downloaded asset packs are not
redistributed here. Background equipment and samples remain authored laboratory
proxies, rather than scanned manufacturer's digital twins.

## Specified materials and texture maps

`visuals.materials` in `src/roboclock/envs/lab/configs/benchmark.yaml` selects each texture
set, world-space UV scale, normal strength and roughness-map weight. Every set
has a diffuse map, a roughness map and an OpenGL normal map. Diffuse inputs are
marked sRGB; roughness and normal inputs are marked raw/linear. `flip_tangent_v`
is false for the supplied OpenGL normals. Texture projection uses world-space
coordinates so changing a fixture's size does not stretch its surface grain.
Labels retain explicit mesh UVs.

| Surface | MDL / texture set | Constant roughness | Metallic | Normal strength | Roughness-map weight |
|---|---|---:|---:|---:|---:|
| Dark epoxy worktop | OmniPBR / `epoxy` | 0.38 | 0.00 | 0.18 | 0.50 |
| Brushed steel | OmniPBR / `brushed_steel` | 0.30 | 0.92 | 0.20 | 0.65 |
| Aluminum mounting and service panels | OmniPBR / `brushed_steel` | 0.24 | 0.88 | 0.15 | 0.45 |
| Powder-coated equipment body | OmniPBR / `powder_coat` | 0.46 | 0.00 | 0.18 | 0.50 |
| Painted laboratory wall | OmniPBR / `painted_plaster_wall` | 0.85 | 0.00 | 0.15 | 0.60 |
| Tiled floor | OmniPBR / `grey_tiles` | 0.35 | 0.00 | 0.15 | 0.70 |
| Sample cap / calibration block | OmniPBR / seeded visible identity color | 0.32 | 0.00 | — | — |
| Rack holder | OmniPBR / matching identity color | 0.32 | 0.00 | — | — |
| Vial body | OmniGlass / clear borosilicate appearance | 0.06 frosting | — | — | — |

The glass uses IOR 1.47, a 25 mm volume-absorption scale, and a hollow mesh with
approximately 1.5 mm wall thickness. Thin-walled mode is false. Vials have
shoulders, necks, a cap skirt, crown, tamper ring and 48 grip flutes. They are
empty rigid-object proxies; liquid chemistry and flow are not represented by
those meshes. The Franka retains the supplied USD asset's authored materials.

Equipment panels, bench edges and rack skins have small physical-scale bevels
for realistic edge reflections. The analyzer has a lid gasket, rubber feet,
recessed display, button, vents and panel fasteners. Cabinets, drawers, handles,
a service strip and sockets give the laboratory background depth and scale.
The analyzer's `STANDBY` display is decorative; task state is shown by `STATE`.

## Tabletop clutter

Actual frame from the updated A1 video.

The shared scene now includes ten loose laboratory clutter objects: four spare
glass bottles, two tip boxes, two pipette trays, and two notebooks. Bottles use
the same hollow OmniGlass construction as samples, with teal OmniPBR caps.
Boxes and trays use textured polymer and paper materials; notebooks use paper
pages and a dark matte polymer cover. English labels (`BUFFER`, `RINSE`, `TIPS`,
and `TOOLS`) identify non-task equipment and reveal no process timing.

`scene.clutter` in the laboratory configuration controls enablement, count,
the independent seed stream, and placement clearances. Density is held constant
across easy/medium/hard so temporal comparisons keep their meaning. Set
`enabled: false` to reproduce the earlier uncluttered scene. Existing saved
configurations without this section retain their original uncluttered layout.

Every clutter object is a dynamic rigid body with a conservative cylinder or
box collider, mass of 0.06–0.16 kg, static friction 0.65, dynamic friction 0.55,
and zero restitution. Tray contents use one conservative solid envelope;
clutter trays are not insertion targets. Visual details remain non-colliding.
Objects can be pushed or knocked over by real contacts; they are excluded from
task operation counts. Robot contact filters include clutter, and episode
metadata stores clutter initial poses and contact onset counts for the seven
arm links. Sample contact filters also distinguish clutter from target fixtures.

Placement rejects overlap with samples, grasp clearance, target entries,
the moving slot's swept volume, the robot base, and existing bench equipment.
This adds distractors and obstacles around the workspace while preserving
valid approach and insertion regions. Runtime checks also verify support and
finite poses after initial settling. The earlier still images and historical
validation results predate this clutter revision; the task videos record the
updated scene.

## Sources and reproduction

The painted-wall and tiled-floor maps are CC0 assets from Poly Haven:
[Painted Plaster Wall](https://polyhaven.com/a/painted_plaster_wall) and
[Grey Tiles](https://polyhaven.com/a/grey_tiles).
Their 1K files are bundled. The 2K
[Studio Small 09 HDR](https://polyhaven.com/a/studio_small_09), by Sergej Majboroda,
is also CC0. Download URLs and SHA256 checksums are in
`src/roboclock/envs/lab/assets/textures/sources.json`; attribution is bundled with the maps.

Epoxy, brushed-steel and powder-coat maps are authored procedural materials,
not photographic scans. Rebuild them with:

```bash
python scripts/lab/build_surface_textures.py
```

The maps use deterministic seed 7401. The asset source manifest distinguishes
these generated maps from the photographic Poly Haven sources. All maps and
HDR files are included in the Python wheel; runtime rendering needs no texture
downloads. The Franka USD still uses Isaac Sim's asset service or a local asset root.

## Lighting, camera and renderer

The 2K HDR provides reflection detail, combined with an overhead area key and a
weaker fill at 5400 K. Default intensities are 180 / 420 / 90 for dome / key / fill;
dome strength scales with the episode's seeded lighting sample. The wall has a
slight cool tint. These are renderer intensities, not measured laboratory lux.

The still-image profile in `docs/images/render_config.yaml` specifies:

- 1536 × 1152 RGB, 24 mm focal length and 36 mm horizontal aperture.
- Camera position `[1.60, 1.35, 2.10]`, looking toward `[0.42, 0.0, 1.07]`, plus seeded jitter.
- RTX `PathTracing`, 8 samples per pixel per subframe, 512 total samples, 8 general bounces and 12 specular/transmission bounces.
- 72 explicit Replicator subframes, OptiX denoising, and no artificial depth of field obscuring contacts.

Normal policy observations use `RaytracedLighting` at 960 × 720. A zero-time
Replicator capture refreshes the RGB buffer for each observation. Kit starts
with the render-completion synchronization settings inspired by RoboDojo.
Still export uses `delta_time=0.0` and checks that `World.current_time` is
identical before and after capture. JSON sidecars record both times, renderer,
camera, material choices, task and seed. The supplied C2/C3 previews were captured
at task time zero.

The camera uses USD's local −Z viewing convention. Its pose comes from the
inverse of a look-at matrix. See NVIDIA's
[OmniPBR material reference](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniPBR.html)
and [RTX path-tracing reference](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html).

```bash
roboclock render --task C2 --seed 0 --config docs/images/render_config.yaml \
  --output results/laboratory.png
roboclock render --task C3 --seed 0 --config docs/images/render_config.yaml \
  --output results/calibration.png
```

`--usd` also exports the stage. Referenced robot assets, bundled textures/HDR
and cached label textures must remain available when opening that stage.

## Labels and physics boundaries

`S01`, `S02`, … identify samples and matching C2/C3 targets. `BIN 0/1/2` plus
colors express sorting. Calibration arrows express orientation. `ASSAY`,
`LAB 01`, `RACK` and `STATE` label equipment. Serial patterns are decorative.
Labels contain no hidden event timestamps, process rates or evaluator outcomes.
Curved vial labels and moving-rack labels follow their owning rigid roots.
Rack nameplates sit 0.5 mm off wall faces to prevent coplanar flicker.

The visual meshes have no collision API. The table, rack floors/walls and full
cylindrical vial envelopes keep their established colliders, mass and friction.
The invisible cylindrical envelope conservatively includes the vial shoulders;
the hollow glass mesh is a render asset, not a new collision model. Other
proxies remain sphere/cylinder/cuboid colliders. Imported USD samples retain
authored visual materials and collider approximations.

Rack interiors remain 50 × 50 mm with separate 4 mm floors and four walls.
A2/C2 collars are 22 mm high and A4 is 40 mm. The A4 fixture uses one kinematic
rigid root with child colliders and PhysX kinematic targets. B2 uses 0.025 table
friction while retaining seeded sample/finger friction. B2/B3 holders remain
120 mm behind samples. Bench/background equipment details are visual only;
the indicator housing, table, heater and manipulation fixtures are physical.
The existing capacity-specific source-grid/bin/indicator clearances are retained.
