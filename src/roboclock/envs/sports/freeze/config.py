"""Independent seeded cue timing and reachable tabletop geometry."""

import numpy as np

from roboclock.envs.sports._shared.workflow.config import config as base_config
from roboclock.simulators.isaac.embodiments.config import configure


# RED -> confirmed rest of hands, pushers and puck (s). 2026-11-17: every policy-facing limit
# is at least 1 s (sub-second limits approach a human's stop reaction, so a near-zero score
# could not separate a weak policy from an unreachable constraint); the user set 8 / 4 / 2 s.
# Before: 0.8 / 0.6 / 0.5 s (2026-11-16 recalibration), 0.8 / 0.4 / 0.2 s. The reference
# oracle stops 0.044-0.180 s after RED (seeds 0-9, every tier).
RESPONSE_LIMIT = {"easy": 8.0, "medium": 4.0, "hard": 2.0}
# Schedule (s). A red lasts the response limit plus a seeded hold, so even a stop at the very
# deadline is followed by a 2.2-3.2 s hold (the whole red of the sub-second design). The
# first green (tool pickup) and later greens are seeded too. Draw order: first green, then
# (hold, green) per red, from the seed's timing stream, so every tier shares the holds and
# greens and only the reds stretch with the limit.
FIRST_GREEN = (8.0, 10.0)
RED_HOLD = (2.2, 3.2)
GREEN = (7.0, 10.0)
# Green time in an episode: the horizon ends the green in which this much green has elapsed.
# The reference oracle needs 35.4-35.7 s of green for its two goals (seeds 0-9); 48 s is
# 1.35x that. Three reds always fall in the first 30 s of green (the requirement below is
# three reactions), since the first three greens last at most 10 s each.
GREEN_BUDGET = 48.0


def red_schedule(rng, limit):
    """Red windows [[onset, end], ...] and the horizon (s) of one seed's timing stream."""
    windows, now, green = [], 0.0, 0.0
    g = float(rng.uniform(*FIRST_GREEN))
    while green + g < GREEN_BUDGET:
        now += g
        green += g
        end = now + limit + float(rng.uniform(*RED_HOLD))
        windows.append([now, end])
        now = end
        g = float(rng.uniform(*GREEN))
    return windows, now + GREEN_BUDGET - green


def config(difficulty="easy", seed=0):
    c = base_config("B4", difficulty, seed)
    c["workflow"]["fixed_freeze"] = True
    c["revision"] = "fixed-freeze-v3"  # v3: 8 / 4 / 2 s limits, reds = limit + hold
    c["primary_temporal_category"] = "event_reactive"
    layout, timing = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    limit = RESPONSE_LIMIT[difficulty]
    windows, horizon = red_schedule(timing, limit)
    x = 0.44 + float(layout.uniform(-0.008, 0.008))
    c["freeze"] = dict(
        horizon=horizon,
        rounds=2,
        red_windows=windows,
        response_limit=limit,
        stable_dwell=0.15,
        goal_dwell=0.35,
        grasp_dwell=0.12,
        hand_linear_limit=0.025,
        hand_angular_limit=0.25,
        body_linear_limit=0.018,
        body_angular_limit=0.30,
        hold_translation=0.012,
        hold_rotation=0.06,
        surface_z=0.84,
        puck_radius=0.028,
        puck_height=0.018,
        puck_start=[x, 0.27 + float(layout.uniform(-0.008, 0.008)), 0.85],
        lane_x=x,
        goals=[[x, -0.27, 0.85], [x, 0.27, 0.85]],
        goal_half_size=[0.065, 0.060],
        tool_home={"left": [0.44, 0.40, 0.85], "right": [0.44, -0.40, 0.85]},
        tool_grip_offset=0.105,
        tool_radius=0.035,
        tool_mass=0.10,
        joint_speed=2.4,
        push_speed=0.065,
        # Stop on red: a short, fast lift-off from the puck (2026-11-04; was 12 mm
        # back / 8 mm up at 0.14 m/s, which alone took about 0.2 s).
        red_backoff=(0.006, 0.006),
        red_backoff_speed=0.40,
        finger_force=6.0,
        grip_preload=0.0015,
    )
    c["randomization"] = dict(
        seed=int(seed),
        version="fixed-freeze-v1",
        layout="Independent +/-8 mm initial position and lane jitter",
        timing="Independent first-green, red hold and green durations; hidden future schedule",
        difficulty=(
            "Only the reaction deadline changes, and with it each red (deadline + the seed's "
            "hold); geometry, physics, holds and greens unchanged"
        ),
    )
    configure(c, "B4", "dual_arx_x5")
    c["embodiment"]["position"] = [0.08, 0, 1.0]
    c["visuals"].update(render_scale=1, record_robot_views=False)
    return c
