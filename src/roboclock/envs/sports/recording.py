"""Setup shared by the seven sports recording drivers (``python -m roboclock.envs.sports``).

Each driver calls ``configure`` on its native configuration before building the
backend, ``prepare`` after building the scene and presentation camera (before
its first render), and ``attach`` right after creating its overview encoder:

- The overview (the task's presentation camera) is written at 1280x960, 25 fps.
  The old ego / single-wrist robot views (``embodiments.views``) are off.
- The X5 fingers get the manufacturer speed limit in SI units, exactly as in
  ``roboclock eval --env sports`` (``robot.enforce_urdf_finger_velocity`` of the
  sports configuration, ``adapters.base.enforce_finger_velocity``), so a recording
  and an evaluation episode run the same physics.
- The task's cue displays are turned toward the head camera (``head_view.apply``) and
  the key objects' head-image margins are recorded (``head_view.check``).
- The RoboDojo rig is measured and checked (``rig_cameras``). With ``--video`` the
  renderer gets the RoboDojo settings and the three rig cameras are installed
  before the first render and written as ``head.mp4``, ``left_wrist.mp4`` and
  ``right_wrist.mp4`` (640x480, 25 fps, the overview's timestamps, equal frame
  counts; ``rig_views.json``). Without a video no camera is installed.
"""

from roboclock.envs import robodojo_rig
from roboclock.envs.sports import rig_cameras

OVERVIEW_RESOLUTION = [1280, 960]


def configure(config):
    """Overview at 1280x960 and no legacy robot-view cameras; physics is untouched."""
    config["sim"]["camera_resolution"] = list(OVERVIEW_RESOLUTION)
    config["visuals"]["record_robot_views"] = False
    return config


def prepare(backend, *, video, task_id, rigs=None):
    """Finger speed limit, head-view displays, rig check and (with a video) cameras."""
    from roboclock.envs.sports import head_view
    from roboclock.envs.sports.adapters.base import enforce_finger_velocity, measure_physics_rate
    from roboclock.envs.sports.config import load_config

    displays = head_view.apply(backend, task_id)
    rigs = [backend.rig] if rigs is None else list(rigs)
    enforce = load_config().get("robot", {}).get("enforce_urdf_finger_velocity", True)
    limits = {rig.path: enforce_finger_velocity(rig) for rig in rigs} if enforce else {}
    if video:
        renderer = rig_cameras.apply_renderer()
        cams = rig_cameras.ensure(backend)
        check, cameras = cams.measured, cams.intrinsics()
    else:
        renderer = cameras = None
        check = rig_cameras.measure_and_check(backend.rig, backend.config, cameras=False)
    return dict(
        finger_velocity_limits=limits,
        physics_scene=measure_physics_rate(backend),
        robodojo_rig=rig_cameras.describe(check, cameras, renderer),
        head_view=dict(
            displays_moved=displays,
            key_objects=head_view.check(backend, task_id),
            margin_required_px=head_view.MARGIN_PX,
        ),
        overview=dict(
            resolution=list(backend.config["sim"]["camera_resolution"]),
            fps=robodojo_rig.VIDEO_FPS,
            camera="task presentation camera (third person)",
        ),
    )


def attach(encoder, backend, output):
    """Write the rig cameras beside the overview encoder (``None`` without a video)."""
    return rig_cameras.attach(encoder, backend, output)


def renderer_readback():
    """Renderer settings read back at the end of a recording."""
    return rig_cameras.renderer_state()
