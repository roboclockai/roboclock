"""Workflow pressure varies independently of physical layouts."""

from roboclock.envs.sports._shared.workflow.config import config as previous_config
import numpy as np


def config(task, difficulty="easy", seed=0):
    if task not in ("B3", "C1"):
        raise ValueError(task)
    c = previous_config(task, difficulty, seed)
    level = ("easy", "medium", "hard").index(difficulty)
    c["revision"] = "sports-v11"
    if task == "C1":
        c["robot_mount"] = [0.65, -0.55, 0.70]
    c["primary_temporal_category"] = "event_reactive" if task == "B3" else "throughput"
    c["workflow"].update(
        lift_heights=[1.04, 1.10, 1.00],
        lift_holds=[2.0, 3.0, 1.5],
        lift_timeout=100.0,
        hold_reminder_duration=0.65,
        flat_lift_handles=True,
        pool_mode="continuous_clearance",
        pool_horizon=60.0,
        minimum_pots_per_minute=[1.0, 2.0, 3.0][level],
        settle_duration=0.4,
    )
    rng = np.random.default_rng(seed + 117)
    c["workflow"]["initial_bar_offset"] = [float(rng.uniform(-0.002, 0.002)),
                                              float(rng.uniform(-0.003, 0.003)), 0.0]
    if task == "B3":
        from roboclock.envs.sports.bench_press.config import configure
        configure(c)
    return c
