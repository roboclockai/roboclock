"""Seeded stock layout; timing difficulty independent of manipulation geometry."""

import math

import numpy as np

from roboclock.envs.sports._shared.workflow.config import config as base_config
from roboclock.evaluation.throughput import HORIZON_FACTOR, MULTIPLIER
from roboclock.simulators.isaac.embodiments.config import configure

DEMAND_COUNT = 6
CAPACITY = 4

# C-class stream schedule (score_version 2, docs/c-class-timing.md): demand k (k = 0..5) is
# due at m * (HARD_PREPARATION + k * HARD_INTERVAL), so a tier stretches the whole reference
# schedule and keeps its demand count. m is TIER_FACTOR below: hard 1.25 (20 % headroom over the
# reference oracle), easy capped so its firing interval stays within a ball machine's slowest
# feed setting (Lobster: one ball every 2-12 s), medium the geometric mean of the two.
#
# The reference schedule (factor 1) is the tightest such schedule the slowest oracle meets. ORACLE_ACCEPTED_S[k] is the
# latest time, over 14 oracle runs (seeds 0-6 and 50-56, c-class-unify, 2026-11-09), at which
# the oracle had its (k+1)-th ball accepted in a hopper cell, measured with
# configs/c3_eager.yaml: every demand is owed from the start, so each ball is fired as soon as
# it is accepted and the oracle never waits for a free cell (scripts/sports/c3_hard_schedule.py).
# The first demand is due when the slowest first ball is in (seed 2); the interval is the
# shortest that keeps every later demand behind the slowest oracle's ball for it,
# max_k (ORACLE_ACCEPTED_S[k] - HARD_PREPARATION) / k = 3.562 s (3rd ball, seed 6). Both are
# rounded up to 0.01 s, so every oracle run meets hard without the time tolerance.
# Frozen: re-measure deliberately, never at run time.
ORACLE_ACCEPTED_S = (6.440, 9.144, 13.564, 16.068, 20.236, 23.220)
HARD_PREPARATION = 6.44
HARD_INTERVAL = 3.57

EASY_MAX_INTERVAL = 12.0  # s, slowest feed of a ball machine
_EASY = min(MULTIPLIER["easy"], EASY_MAX_INTERVAL / HARD_INTERVAL)
TIER_FACTOR = {
    "easy": _EASY,
    "medium": math.sqrt(_EASY * MULTIPLIER["hard"]),
    "hard": MULTIPLIER["hard"],
}

SCHEDULE_KEYS = ("preparation", "firing_interval", "horizon")


def tier_schedule(difficulty):
    """(preparation, firing interval) of a tier in seconds: TIER_FACTOR x the reference schedule."""
    m = TIER_FACTOR[difficulty]
    return round(m * HARD_PREPARATION, 6), round(m * HARD_INTERVAL, 6)


def config(difficulty="easy", seed=0, schedule=None):
    """C3 configuration. `schedule` overrides preparation / firing_interval / horizon (s).

    The override is for calibration runs only (``configs/c3_eager.yaml``); it is recorded under
    ``service.schedule_override`` and the benchmark never sets it.
    """
    c = base_config("C3", difficulty, seed)
    c["revision"] = "fixed-service-v3"  # v3: recalibrated tier factors (2026-11-16); v2: m x hard, no jitter
    c["primary_temporal_category"] = "throughput"
    c["workflow"]["fixed_service"] = True
    # The seed's first child stream draws the layout. The second one drew an 18 + U(0, 2) s
    # preparation until 2026-11-09; it is no longer used, so every seed keeps its layout.
    layout = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
    preparation, interval = tier_schedule(difficulty)
    last_demand = preparation + (DEMAND_COUNT - 1) * interval
    c["service"] = dict(
        radius=0.033,
        mass=0.057,
        capacity=CAPACITY,
        support_dwell=0.20,
        grasp_dwell=0.10,
        preparation=preparation,
        firing_interval=interval,
        demand_count=DEMAND_COUNT,
        # Late balls are still served, so the episode runs to 1.25 x the last scheduled
        # firing time (until 2026-11-09 it ended 2 s after the last demand).
        horizon=HORIZON_FACTOR * last_demand,
        gate_timeout=1.4,
        joint_speed=2.4,
        sources={
            f"ball{i}": [
                x + float(layout.uniform(-0.003, 0.003)),
                sign * 0.36 + float(layout.uniform(-0.003, 0.003)),
                0.859,
            ]
            for i, (sign, x) in enumerate((s, x) for s in (1, -1) for x in (0.23, 0.35, 0.47))
        },
        cells=[[x, y, 0.859] for y in (0.12, -0.12) for x in (0.37, 0.49)],
        cell_half_width=0.049,
        lip_z=0.95,
        floor_z=0.826,
    )
    if schedule:
        unknown = set(schedule) - set(SCHEDULE_KEYS)
        if unknown:
            raise ValueError(f"C3 schedule override: unknown keys {sorted(unknown)}")
        s = c["service"]
        s.update({k: float(v) for k, v in schedule.items()})
        if "horizon" not in schedule:
            last = s["preparation"] + (DEMAND_COUNT - 1) * s["firing_interval"]
            s["horizon"] = HORIZON_FACTOR * last
        s["schedule_override"] = dict(schedule)
    c["randomization"] = dict(
        version="fixed-service-v2",
        seed=int(seed),
        layout="Independent +/-3 mm source jitter; both baskets reachable",
        timing="None: the demand schedule is fixed per difficulty (m x the hard schedule)",
    )
    configure(c, "C3", "dual_arx_x5")
    c["embodiment"]["position"] = [0.08, 0, 1.0]
    c["visuals"].update(render_scale=1, record_robot_views=False)
    return c
