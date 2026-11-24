"""Unified sports runtime configuration: observation settings and task rates."""

import copy
import math
from pathlib import Path

import yaml

from roboclock.envs import rates, robodojo_rig

DEFAULT_CONFIG = Path(__file__).parent / "configs/benchmark.yaml"
TASK_IDS = ("A1", "A2", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3")


def _merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read(path):
    """One YAML file; ``base: <file>`` (relative to it) is loaded first and overridden."""
    path = Path(path)
    config = yaml.safe_load(path.read_text()) or {}
    base = config.pop("base", None)
    if base is not None:
        config = _merge(_read(path.parent / base), config)
    return config


def load_config(path=None) -> dict:
    config = _read(path or DEFAULT_CONFIG)
    sim = config["sim"]
    # Physics/oracle-only runs: no head camera, no render call, blank rgb (env.py, adapters).
    sim.setdefault("no_render", False)
    if not isinstance(sim["no_render"], bool):
        raise ValueError("sim.no_render must be true or false")
    if not (isinstance(sim["physics_dt"], (int, float)) and sim["physics_dt"] > 0):
        raise ValueError("sim.physics_dt must be positive")
    if not (isinstance(sim["control_decimation"], int) and sim["control_decimation"] >= 1):
        raise ValueError("sim.control_decimation must be a positive integer")
    resolution = sim["camera_resolution"]
    if len(resolution) != 2 or not all(isinstance(n, int) and n >= 16 for n in resolution):
        raise ValueError("sim.camera_resolution must be [width, height] in pixels")
    observation = config.setdefault("observation", {})
    observation.setdefault("render_updates", 1)
    observation.setdefault("rt_subframes", 1)
    if observation["render_updates"] < 0 or observation["rt_subframes"] < 1:
        raise ValueError("observation.render_updates >= 0 and rt_subframes >= 1 required")
    config.setdefault("visuals", {}).setdefault("renderer", "RaytracedLighting")
    tasks = config.get("tasks", {})
    if set(tasks) != set(TASK_IDS):
        raise ValueError(f"Sports configuration must list rates for {', '.join(TASK_IDS)}")
    for task_id, task in tasks.items():
        physics, control = task["physics_hz"], task["control_hz"]
        if physics <= 0 or control <= 0 or physics % control:
            raise ValueError(f"{task_id}: physics_hz must be a positive multiple of control_hz")
        rates.check_rates(1 / physics, physics // control, where=task_id)
        if physics != robodojo_rig.PHYSICS_HZ:
            raise ValueError(f"{task_id}: physics must run at exactly {robodojo_rig.PHYSICS_HZ} Hz")
    return config


def task_rates(config, task_id):
    """(physics_dt, control_decimation) of one task."""
    task = config["tasks"][task_id]
    return 1.0 / task["physics_hz"], task["physics_hz"] // task["control_hz"]


def with_task_rates(config, task_id):
    config = copy.deepcopy(config)
    dt, decimation = task_rates(config, task_id)
    config["sim"].update(physics_dt=dt, control_decimation=decimation)
    config["active_task"] = task_id
    return config


def check_rates(config, task_id, native):
    """The sub-package must run at the configured rates; refuse silent drift."""
    dt, decimation = task_rates(config, task_id)
    if not math.isclose(native["sim"]["physics_dt"], dt, rel_tol=1e-9) or (
        native["sim"]["control_decimation"] != decimation
    ):
        raise ValueError(
            f"{task_id}: native rates {native['sim']['physics_dt']}/"
            f"{native['sim']['control_decimation']} differ from configured {dt}/{decimation}"
        )
