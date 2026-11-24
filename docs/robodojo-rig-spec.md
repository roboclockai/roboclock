# RoboDojo-aligned rig spec (all domains)

Every roboclock domain (industry, hospital, sports, lab, household) uses one dual ARX X5
rig, one set of policy cameras, one physics rate and one renderer, copied from RoboDojo.
A policy trained on RoboDojo then sees the same embodiment geometry in every domain.

The single source of truth is `src/roboclock/envs/robodojo_rig.py`, which is simulator-free.
Its docstring cites every RoboDojo file each value comes from, and
`tests/test_robodojo_rig.py` checks the module against RoboDojo's world poses and the X5A
URDF. Domains import these constants; they keep no local copies.

## Values

### Rig and cameras

| Item | Value |
|---|---|
| Arms | Dual ARX X5. Bases side by side, same height, same forward coordinate, same yaw, 0.60 m apart center to center |
| Rig frame | Origin at the base midpoint on the mount plane, +x forward, +y left, +z up. Each task may place the rig (origin, yaw) anywhere in its scene |
| Head camera pose | (0.04, 0, 0.543) m in the rig frame, pitched 60° down, no roll |
| Head camera intrinsics | Gemini 345Lg template: 640×480, focal 10 mm, apertures 22.212 × 14.266 mm (HFOV 96.0°), clipping 0.005–10 m |
| Wrist cameras | One per arm, rigid on `link6` at (0.08484, 0, 0.05094) m. The optical axis is 30° below link6 +x, toward −z |
| Wrist camera intrinsics | D435 template: 640×480, focal 13 mm, apertures 20.955 × 15.716 mm (77.7° × 62.3°), clipping 0.0001–10 m |

The head camera's apertures (1.557:1) do not match its 4:3 image. RTX keeps square pixels,
so the rendered head image is 96.0° × 79.6° with fx = fy = 288.13 px. This was measured in
Isaac Sim 6.0.1 on industry Task 45: a lamp sat at y = 69.6 px, against 70.4 px predicted
with square pixels and 42.0 px with both apertures.

`Camera.get_intrinsics_matrix()` reports fy = 336.46 px from the vertical aperture, and that
value does not describe the rendered image. Use `robodojo_rig.intrinsics()` for projection.

The wrist camera orientation in `link6` is (w, x, y, z) = (0.61238, 0.35354, −0.35354,
−0.61238) in USD camera axes.

### Observations and rates

| Item | Value |
|---|---|
| Observation keys | `rgb` (head, `cam_head`), `left_wrist_rgb` (`cam_left_wrist`), `right_wrist_rgb` (`cam_right_wrist`). With `--depth`: `depth`, `left_wrist_depth`, `right_wrist_depth` |
| Left / right | The left wrist camera rides the arm on the rig's left (+y) |
| Physics | Exactly 250 Hz in every task. Multiples of 250 Hz are not used |
| Control / policy | Environment step 50 Hz (`rates.CONTROL_HZ`, unchanged). Policy observation and action rate 25 Hz (`rates.POLICY_HZ`), as in RoboDojo's `collect_freq` |
| Render | RoboDojo renders every observation at 25 Hz. roboclock renders an observation when a policy reads it (`robodojo_rig.POLICY_RENDER = "on_read"`, docs/architecture.md "Rendering on read"): once per model query, at its observation instant |

### Renderer and recordings

| Item | Value |
|---|---|
| Renderer | `/rtx/rendermode = RaytracedLighting`, DLAA antialiasing, plus Isaac Lab's `quality` preset (`RENDER_SETTINGS`) |
| No-render runs | Config key `sim.no_render: true` creates no cameras and makes no render calls in every domain |
| Recordings | `head.mp4`, `left_wrist.mp4` and `right_wrist.mp4` at 640×480 and 25 fps, frame-synchronized with equal frame counts. A third-person overview `<task>.mp4` at 1280×960 and 25 fps |

## Requirements for each domain

1. **Use the shared module for every value.**
   - Build the head and wrist cameras from `robodojo_rig`. Replicate RoboDojo's camera API
     calls: focal length, `set_horizontal_aperture(h, False)` and
     `set_vertical_aperture(v, False)`.
   - `describe()` carries the intrinsics of the rendered image (`rendered_intrinsics`,
     from `intrinsics()`). Where a domain also records `get_intrinsics_matrix()`, label
     it as the API-reported value.
2. **Check the rig at every reset.**
   - Measure both base frames and the head camera from the stage, and run `check_rig`.
   - Measure both wrist cameras against `wrist_camera_in_link6()`, applied to the measured
     `link6` pose (1 mm / 0.1°).
   - Confirm that the domain's X5 `link6` frame matches X5A.urdf `link6`.
3. **Expose the observations.**
   - Add `left_wrist_rgb` and `right_wrist_rgb` (640×480) to the observation space and to
     every observation.
   - The head `rgb` becomes 640×480.
   - `sim.no_render` returns zero images of the same shapes.
4. **Physics at 250 Hz.**
   - Every task runs physics at 250 Hz.
   - A task that needed 500 or 1000 Hz for contact stability must be made stable at
     250 Hz, for example with solver iterations, contact offsets or the grasp. Its oracle
     must pass again.
5. **Renderer.**
   - Apply `RENDER_MODE`, `ANTIALIASING` and `RENDER_SETTINGS`. Isaac Lab uses
     `omni.replicator.core.settings.set_render_rtx_realtime(antialiasing="DLAA")`.
   - In Isaac Sim 6.0.1 that call also switches `/rtx/rendermode` to RealTimePathTracing.
     Set `/rtx/rendermode` to `RaytracedLighting` after it.
   - Read back `/rtx/rendermode` and the antialiasing setting, and store both in the result.
   - If DLAA is not available on the GPU, record the fallback that is in effect.
6. **Visibility.**
   - Every task's key objects and cues must be inside the head image at the 60° pitch.
     Check this with a projection test (corners at least 10 px inside the image) and with a
     rendered frame.
   - Layout may change to achieve it: moving fixtures, displays and the per-task rig
     placement.
   - Report anything that only the wrist cameras can see.
7. **Recordings and results.**
   - Re-record the delivered videos in the format above.
   - Keep the previous recordings as superseded; do not delete them.
   - Put `robodojo_rig.describe()` and the measured rig check in every result JSON.

Not adopted from RoboDojo:

- RoboDojo pins the rig at one world pose for all tasks; roboclock keeps per-task placement.
- RoboDojo's X5 drive gains (stiffness 4400, damping 40, gripper 2300 / 100) are not adopted.
