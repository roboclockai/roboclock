"""Temporal parameters vary independently of physical geometry."""

from roboclock.envs.sports.arena.config import config as previous_config

NAMES = {"B3": "Lift", "B4": "Freeze", "C1": "Pool", "C3": "Ball Service"}


def config(task, difficulty="easy", seed=0):
    if task not in NAMES:
        raise ValueError(task)
    c = previous_config("C2", difficulty, seed)
    level = ("easy", "medium", "hard").index(difficulty)
    c["task_id"] = task
    c["visuals"]["anti_aliasing"] = 2
    c["visuals"]["render_scale"] = 2
    c["scene"]["randomness_scale"] = 1.0
    c["workflow"] = dict(
        hold_duration=2.0,
        lower_response=[1.2, 0.4, 0.12][level],
        hold_height=1.04,
        lift_rounds=3,
        lift_timeout=85.0,
        stop_response=[0.8, 0.4, 0.2][level],
        freeze_horizon=90.0,
        shot_clock=[6.0, 3.0, 2.0][level],
        pool_horizon=45.0,
        preparation=60.0,
        firing_interval=[40.0, 32.0, 24.0][level],
        service_horizon=160.0,
        capacity=4,
        carrier_capacity=2,
        support_dwell=0.25,
    )
    return c
