"""``roboclock render --env sports``: converged stills of the RoboDojo rig cameras (head + wrists)."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def export_render(env, output, *, usd=False):
    from roboclock.simulators.isaac.rendering import capture_camera
    from roboclock.simulators.isaac.runtime import simulator_version

    adapter = env.adapter
    if adapter.no_render:
        raise ValueError("roboclock render needs the rig cameras; sim.no_render is set")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter.before_capture()
    before = float(adapter.world.current_time)
    for _ in range(3):
        adapter.world.render()
    capture_camera(subframes=32)
    if not np.isclose(before, float(adapter.world.current_time), atol=1e-12, rtol=0):
        raise RuntimeError("Snapshot capture advanced physics time")
    images = adapter.cams.images()
    Image.fromarray(images["rgb"]).save(path)
    for key in ("left_wrist_rgb", "right_wrist_rgb"):
        Image.fromarray(images[key]).save(path.with_name(f"{path.stem}_{key}{path.suffix}"))
    metadata = {
        "simulator": "isaacsim==" + simulator_version(),
        "environment": env.environment_id,
        "task": env.task.task_id,
        "difficulty": env.task.spec.difficulty,
        "seed": env.task.spec.seed,
        "sim_time": env.task.time,
        "camera": "RoboDojo rig head camera (rgb); wrist stills beside it",
        "rig_cameras": adapter.cams.describe(),
    }
    if usd:
        stage = path.with_suffix(".usda")
        adapter.backend.stage.GetRootLayer().Export(str(stage.resolve()))
        metadata["usd"] = stage.name
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def validate_physics(env, plan, output, **kwargs):
    raise ValueError(
        "Sports physics is validated per recording (data/sports/videos) and through "
        "roboclock eval --env sports --policy oracle --oracle"
    )
