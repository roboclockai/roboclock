"""Task timing is independent of physical geometry and randomization."""

import numpy as np

from roboclock.envs.sports import c_class
from roboclock.envs.sports.arcade.tasks import configuration

NAMES = {"A3": "Buzzer Shot", "B2": "Boxing", "C2": "Speed Cups"}


def config(task, difficulty="easy", seed=0):
    if task not in NAMES or difficulty not in ("easy", "medium", "hard"):
        raise ValueError("Unknown workflow task or difficulty")
    c = configuration("A2", room_style="v5")
    c["task_id"], c["difficulty"], c["seed"] = task, difficulty, seed
    c["sim"]["camera_resolution"] = [1280, 960]
    c["scene"].update(table_height=0.75, room_style="v5")
    level = ("easy", "medium", "hard").index(difficulty)
    c["workflow"] = dict(
        # Tier limits recalibrated 2026-11-16 against the reference oracle (seeds 0-9,
        # identical at every tier): hard = max(t / 0.8, t + 0.3) rounded up, medium =
        # sqrt(easy * hard). A3: READY -> release t = 4.59 s (was 14 / 10 / 5 s).
        shot_clock=[14.0, 9.0, 6.0][level],
        shot_start=2.0,
        flight_grace=4.0,
        # B2: GO cue -> hit t = 0.680 s (was 1.6 / 1.15 / 0.8 s); hit -> guard t = 0.716 s
        # (was 1.4 / 1.1 / 0.8 s). Rounded to 0.05 s.
        # Easy (2026-11-23 user rule: a documented time where one exists, else 4 x t, capped by
        # the scene's process): 4 x t would be 2.72 and 2.86 s, but a combination's next cue
        # comes 2.8-3.5 s after the last and a command ends 2.6 s after its last cue
        # (boxing.schedule), so hit plus return must fit in about 3 s: easy stays 1.6 + 1.4 s.
        cue_response=[1.6, 1.25, 1.0][level],
        guard_deadline=[1.4, 1.2, 1.05][level],
        boxing_start=3.0,
        # C2 counts cards from the end of preparation to the horizon H = 5 t_o, the same at
        # every difficulty (score_version 2, docs/c-class-timing.md); the difficulty only
        # selects the judged time tier. Until 2026-11-09: cups_budget 75 / 55 / 40 s.
        cups_horizon=c_class.horizon("C2"),
        cups_required=2,
        preparation=2.0,
        support_dwell=0.25,
    )
    if task == "C2":
        # Cup oracle motion (2026-11-04): faster transfers and vertical moves with a
        # short brake above each support (was 0.18 / 0.10 / 0.06 m/s, 0.45 s release).
        c["workflow"].update(
            cups_speed=0.36,
            cups_descend_speed=0.25,
            cups_lower_speed=0.18,
            cups_brake_height=0.025,
            cups_brake_speed=0.05,
            cups_release_time=0.25,
        )
        # Joint step limit of the shared workflow backend (read from "service").
        c["service"] = dict(joint_speed=3.0)
    if task == "A3":
        # Separate streams keep layout stable when timing sampling evolves.
        timing_seed, layout_seed = np.random.SeedSequence(seed).spawn(2)
        timing_rng = np.random.default_rng(timing_seed)
        layout_rng = np.random.default_rng(layout_seed)
        c["workflow"]["shot_start"] = float(timing_rng.uniform(2.0, 6.0))
        c["basketball"] = dict(
            pickup_xy=layout_rng.uniform([0.28, -0.27], [0.37, -0.17]).tolist(),
        )
        c["randomization"] = dict(
            version="basketball-v1",
            generator="numpy.PCG64",
            seed=int(seed),
            shot_start_range=[2.0, 6.0],
            pickup_xy_bounds=[[0.28, -0.27], [0.37, -0.17]],
            distribution="Independent uniform timing and layout streams",
            shot_start=c["workflow"]["shot_start"],
            pickup_xy=c["basketball"]["pickup_xy"].copy(),
        )
    elif seed:
        rng = np.random.default_rng(seed)
        c["workflow"]["shot_start"] += float(rng.uniform(-0.2, 0.2))
        c["workflow"]["boxing_start"] += float(rng.uniform(-0.2, 0.2))
    return c
