"""Industry configuration: shared simulator settings plus per-task native rates."""

import math
from pathlib import Path

import yaml

from roboclock.envs.rates import CONTROL_HZ, check_rates


def load_config(path=None):
    path = Path(path) if path else Path(__file__).parent / "configs/benchmark.yaml"
    config = yaml.safe_load(path.read_text())
    for key in ("physics_dt", "control_dt"):
        value = config["sim"][key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"sim.{key} must be finite and positive")
    config["sim"].setdefault("render_subframes", 2)
    config["sim"].setdefault("no_render", False)
    config["sim"]["control_decimation"] = round(
        config["sim"]["control_dt"] / config["sim"]["physics_dt"]
    )
    check_rates(config["sim"]["physics_dt"], config["sim"]["control_decimation"])
    config.setdefault("visuals", {})
    config.setdefault("tasks", {})
    for task, values in config["tasks"].items():
        hz = values.get("physics_hz")
        if hz is not None:
            check_rates(1 / hz, hz // CONTROL_HZ, where=f"tasks.{task}.physics_hz")
        # An explicit episode length (short debugging runs); C horizons reach 415.5 s.
        budget = values.get("max_time")
        if budget is not None and not (math.isfinite(budget) and budget > 0):
            raise ValueError(f"tasks.{task}.max_time must be finite and positive (s)")
    return config
