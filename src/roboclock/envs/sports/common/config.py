"""Validated sports parameters with independently seeded physical sampling."""

import copy
import math
from pathlib import Path

import yaml

from roboclock.envs.rates import check_rates


def load_config(path=None):
    source = Path(path) if path else Path(__file__).parent / "configs/benchmark.yaml"
    config = yaml.safe_load(source.read_text())

    def check(value, path="config"):
        if isinstance(value, dict):
            for k, v in value.items():
                check(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                check(v, f"{path}[{i}]")
        elif isinstance(value, (int, float)) and (not math.isfinite(value)):
            raise ValueError(f"{path} must be finite")

    check(config)
    if config["sim"]["physics_dt"] <= 0 or config["sim"]["control_decimation"] < 1:
        raise ValueError("Positive simulation time steps required")
    for key in ("control_decimation", "settle_steps"):
        if not isinstance(config["sim"][key], int) or config["sim"][key] < 1:
            raise ValueError(f"{key} must be integral")
    check_rates(config["sim"]["physics_dt"], config["sim"]["control_decimation"])
    for task_id, settings in config["tasks"].items():
        if settings["max_time"] <= 0:
            raise ValueError(f"Task {task_id} requires a positive episode budget")
    if config["tasks"]["38"]["pins"] != 10 or config["tasks"]["39"]["cups"] != 5:
        raise ValueError("The supplied contracts require ten pins and five cups")
    feed = config["tasks"]["40"]
    if not 0 <= feed["warmup"] < feed["max_time"]:
        raise ValueError("Feeding warmup must be inside the episode")
    if not isinstance(feed["balls"], int) or not 1 <= feed["balls"] <= 16:
        raise ValueError("The source basket supports 1–16 balls")
    if not isinstance(feed["capacity"], int) or not 1 <= feed["capacity"] <= feed["balls"]:
        raise ValueError(
            "Hopper capacity must be a positive integer within the supplied ball count"
        )
    if config["scene"]["spatial_jitter"] < 0 or config["scene"]["spatial_jitter"] > 0.02:
        raise ValueError("Spatial jitter must be within validated 0–20 mm clearance")
    for level, limits in config["difficulties"].items():
        if any(v <= 0 for v in limits.values()):
            raise ValueError(f"Positive limits required: {level}")
        if limits["ball_deadline"] > config["tasks"]["38"]["max_time"]:
            raise ValueError("The next ball must occur inside the pin-setting episode")
        if limits["stop_deadline"] > config["tasks"]["37"]["red_duration"]:
            raise ValueError("Stop deadline cannot extend beyond the red interval")
        if (
            limits["pins_per_min"]
            > 60 * config["tasks"]["38"]["pins"] / config["tasks"]["38"]["max_time"]
        ):
            raise ValueError("Pin rate exceeds the ten-pin episode supply")
        shots = (
            math.floor(
                (config["tasks"]["40"]["max_time"] - config["tasks"]["40"]["warmup"])
                / limits["fire_interval"]
            )
            + 1
        )
        if shots > config["tasks"]["40"]["balls"]:
            raise ValueError("Insufficient supplied balls for firing schedule")
    return config


def chess_config(config):
    from roboclock.envs.chess.config import load_config as load_chess

    result = load_chess()
    result["sim"] = copy.deepcopy(config["sim"])
    result["control"]["joint_target_speed"] = config["control"]["joint_target_speed"]
    result["episode"]["max_time"] = config["tasks"]["31"]["max_time"]
    result["episode"]["history"] = config["tasks"]["31"]["history"]
    for level, values in config["difficulties"].items():
        result["difficulties"][level]["deadline"] = values["chess_deadline"]
    return result
