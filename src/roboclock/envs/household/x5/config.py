"""Household configuration: the legacy schema plus the Dual-X5 runtime section ``x5``.

``configs/benchmark.yaml`` keeps the validated common schema read by the earlier task-specific
recorders (``roboclock.envs.household.config.load_config``). The Dual-X5 environment reads only
the ``x5`` section and the shared ``visuals.renderer``; ``env_config`` turns them into the
environment's ``config`` dict (``sim.physics_dt``, ``visuals``, ...).
"""

import math

from roboclock.envs import robodojo_rig as RD
from roboclock.envs.household.config import HOUSEHOLD_ROOT
from roboclock.envs.household.config import load_config as load_legacy_config
from roboclock.envs.rates import check_rates

DEFAULT_PATH = HOUSEHOLD_ROOT / "configs/benchmark.yaml"


def check_x5(section):
    """Validate the ``x5`` section in place; return it."""
    if not isinstance(section, dict):
        raise ValueError("Configuration section x5 must be a mapping")
    for key in (
        "physics_dt",
        "control_decimation",
        "settle_seconds",
        "camera",
        "camera_resolution",
        "cue_interval_ticks",
    ):
        if key not in section:
            raise ValueError(f"Missing configuration value x5.{key}")
    dt = section["physics_dt"]
    if not isinstance(dt, int | float) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("x5.physics_dt must be positive")
    for key in ("control_decimation", "cue_interval_ticks"):
        if not isinstance(section[key], int) or isinstance(section[key], bool) or section[key] < 1:
            raise ValueError(f"x5.{key} must be a positive integer")
    check_rates(dt, section["control_decimation"], where="x5")
    settle = section["settle_seconds"]
    if not isinstance(settle, int | float) or not math.isfinite(settle) or settle < 0:
        raise ValueError("x5.settle_seconds must be finite and nonnegative")
    resolution = section["camera_resolution"]
    if (
        not isinstance(resolution, list | tuple)
        or len(resolution) != 2
        or not all(isinstance(v, int) and v >= 16 for v in resolution)
    ):
        raise ValueError("x5.camera_resolution must be [width, height] in pixels")
    if not isinstance(section["camera"], str) or not section["camera"].startswith("/"):
        raise ValueError("x5.camera must be an absolute USD prim path")
    if section.get("scene_dir") is not None and not isinstance(section["scene_dir"], str):
        raise ValueError("x5.scene_dir must be a path or null")
    if not isinstance(section.get("no_render", False), bool):
        raise ValueError("x5.no_render must be true or false")
    if list(resolution) != list(RD.HEAD_CAMERA["resolution"]) or section["camera"] != "/World/AgentCamera":
        raise ValueError("x5 head camera must be /World/AgentCamera at the RoboDojo 640x480")
    if not math.isclose(1 / dt, RD.PHYSICS_HZ):
        raise ValueError(f"x5 physics must run at exactly {RD.PHYSICS_HZ} Hz (RoboDojo rig spec)")
    if not isinstance(section.get("physics_to_usd", False), bool):
        raise ValueError("x5.physics_to_usd must be true or false")
    return section


def load_config(path=None):
    """Spec loader: legacy validation for every household config, plus ``x5`` when present."""
    config = load_legacy_config(path or DEFAULT_PATH)
    if "x5" in config:
        check_x5(config["x5"])
    return config


def env_config(config):
    from roboclock.envs.start_pose import start_pose

    """The Dual-X5 environment's JSON-serializable ``config`` (also written to manifests)."""
    if "x5" not in config:
        raise ValueError("The Dual-X5 household environment requires the x5 configuration section")
    x5 = check_x5(dict(config["x5"]))
    return {
        "sim": {
            "physics_dt": float(x5["physics_dt"]),
            "control_decimation": int(x5["control_decimation"]),
            "settle_seconds": float(x5["settle_seconds"]),
            "cue_interval_ticks": int(x5["cue_interval_ticks"]),
            "physics_to_usd": bool(x5.get("physics_to_usd", False)),
            "camera": x5["camera"],
            "camera_resolution": [int(v) for v in x5["camera_resolution"]],
            # RoboDojo spec key sim.no_render; x5.no_render is the household section's spelling.
            "no_render": bool(x5.get("no_render", False) or config.get("sim", {}).get("no_render", False)),
        },
        "scene": {"scene_dir": x5.get("scene_dir")},
        # Shared key robot.start_pose (roboclock.envs.start_pose): "zero" starts both arms at
        # joint 0 instead of READY_Q.
        "robot": {"start_pose": start_pose(config)},
        # The runtime is started with the shared visuals.renderer (cli.py); the legacy material
        # presets in the same section do not apply to the Dual-X5 scenes.
        "visuals": {"renderer": config.get("visuals", {}).get("renderer", "RaytracedLighting")},
    }
