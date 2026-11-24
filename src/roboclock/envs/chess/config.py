"""Strict chess configuration and deterministic sampling."""

from pathlib import Path

import numpy as np
import yaml


def load_config(path=None):
    path = Path(path) if path else Path(__file__).parent / "configs/benchmark.yaml"
    config = yaml.safe_load(path.read_text())
    for section, keys in {
        "sim": ("physics_dt", "control_decimation", "settle_steps"),
        "scene": ("square_size", "board_height", "piece_radius", "piece_height", "piece_mass"),
        "physics": (
            "placement_dwell",
            "max_linear_speed",
            "max_angular_speed",
            "max_tilt",
            "contact_force",
            "button_travel",
            "button_threshold",
            "button_dwell",
        ),
        "episode": ("max_time",),
        "control": ("joint_target_speed", "transit_height"),
    }.items():
        for key in keys:
            if not np.isfinite(config[section][key]) or config[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be finite and positive")
    if config["scene"]["piece_radius"] * 2 + 0.004 >= config["scene"]["square_size"]:
        raise ValueError("Pieces must fit completely inside a square with clearance")
    if config["physics"]["button_threshold"] >= config["physics"]["button_travel"]:
        raise ValueError("Button threshold must be below its travel limit")
    a, b = config["episode"]["opponent_press_range"]
    if not 0 <= a <= b or not np.isfinite(b):
        raise ValueError("Invalid opponent press range")
    for level in config["difficulties"].values():
        d = level["deadline"]
        if not np.isfinite(d) or d <= 0 or d + b >= config["episode"]["max_time"]:
            raise ValueError("Deadline must be positive and finish inside the episode budget")
    from roboclock.envs.chess.rules import game_board

    game_board(config["episode"])
    return config
