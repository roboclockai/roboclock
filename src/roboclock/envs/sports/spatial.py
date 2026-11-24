"""Seeded spatial layout randomization of the sports stations (roboclock eval).

Since 2026-11-20 every reset of A1, B1, B2, B3, B5 and C2 moves its station by a per-seed
offset drawn from a dedicated layout stream: ``SeedSequence([seed, LAYOUT_STREAM, task])``.
No timing or discrete draw (opening moves, schedules, fatigue, colours) shares that stream,
so those are unchanged, and the offsets are the same at every difficulty. Seed 0 is an
ordinary random instance (no seed is exempt).

``apply`` writes each offset into the native configuration the task's scene, evaluator and
oracle already read (one source of truth per task, see ``docs/sports-randomization.md``):

* A1 chess: ``scene.board_center`` and ``scene.clock_position`` shift together by
  U(+/-2 cm) in x and y (board squares, piece spawns, placement checks, clock buttons).
* B1 moles / B5 taiko: ``v2.station_offset`` [dx, dy], U(+/-2 cm) each: the mole cabinet
  (holes, walls, trim) or both drums, and both tool rests with their tools.
* B2 boxing: ``workflow.pad_offset`` [dx, dy]: the pad trainer moves U(+/-1.5 cm) in x
  (toward / away from the robot) and U(+/-2 cm) in y. Guard stays robot-fixed.
* B3 bench press: ``workflow.rack_x`` U(+/-1.5 cm), ``rack_height`` U(+/-1 cm), the press
  line ``bench_x`` U(+/-1.5 cm) and the bar's start on the hooks
  (``initial_bar_offset``, U(+/-1 cm) x, U(+/-1.5 cm) along the bar).
* C2 speed cups: ``workflow.cup_source_offsets``: each side's start row moves U(+/-8 mm) in
  x and y, each cup a further U(+/-1 mm) in x (along the row; 5 mm gaps between cups) and
  U(+/-2 mm) in y. Card targets stay fixed.

The recording driver (``python -m roboclock.envs.sports``) does not call ``apply``: it keeps
the nominal layout of the recorded demos.
"""

import numpy as np

VERSION = "sports-spatial-v1"
LAYOUT_STREAM = 20261120
TASK_KEYS = {"A1": 1, "B1": 2, "B2": 3, "B3": 4, "B5": 5, "C2": 6}

# Half-ranges (m) of the uniform draws.
STATION = {"A1": (0.02, 0.02), "B1": (0.02, 0.02), "B5": (0.02, 0.02), "B2": (0.015, 0.02)}
BENCH = dict(rack_x=0.015, rack_height=0.010, bench_x=0.015, bar_x=0.010, bar_y=0.015)
CUPS = dict(row=(0.008, 0.008), cup=(0.001, 0.002))


def layout_rng(task_id, seed):
    """The task's layout stream, independent of every timing or discrete draw."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), LAYOUT_STREAM, TASK_KEYS[task_id]])
    )


def _uniform(rng, half):
    half = np.asarray(half, float)
    return [round(float(v), 6) for v in rng.uniform(-half, half)]


def sample(task_id, seed):
    """The seed's offsets of one task (m), or None for a task without spatial randomization."""
    if task_id not in TASK_KEYS:
        return None
    rng = layout_rng(task_id, seed)
    if task_id in STATION:
        return dict(station_offset=_uniform(rng, STATION[task_id]))
    if task_id == "B3":
        b = BENCH
        return dict(
            rack_dx=_uniform(rng, [b["rack_x"]])[0],
            rack_dz=_uniform(rng, [b["rack_height"]])[0],
            bench_dx=_uniform(rng, [b["bench_x"]])[0],
            bar_offset=_uniform(rng, [b["bar_x"], b["bar_y"]]),
        )
    offsets = {}
    for side in ("left", "right"):
        row = np.asarray(_uniform(rng, CUPS["row"]))
        for i in range(3):
            offsets[f"{side}_{i}"] = [
                round(float(v), 6) for v in row + np.asarray(_uniform(rng, CUPS["cup"]))
            ]
    return dict(cup_source_offsets=offsets)


def apply(task_id, seed, c):
    """Write the seed's offsets into native config ``c`` and record them; returns the record."""
    drawn = sample(task_id, seed)
    if drawn is None:
        return None
    if task_id == "A1":
        dx, dy = drawn["station_offset"]
        scene = c["scene"]
        x, y = scene["board_center"]
        scene["board_center"] = [x + dx, y + dy]
        clock = list(scene["clock_position"])
        scene["clock_position"] = [clock[0] + dx, clock[1] + dy, *clock[2:]]
    elif task_id in ("B1", "B5"):
        c["v2"]["station_offset"] = list(drawn["station_offset"])
    elif task_id == "B2":
        c["workflow"]["pad_offset"] = list(drawn["station_offset"])
    elif task_id == "B3":
        w = c["workflow"]
        w["rack_x"] += drawn["rack_dx"]
        w["rack_height"] += drawn["rack_dz"]
        w["bench_x"] += drawn["bench_dx"]
        w["initial_bar_offset"] = [*drawn["bar_offset"], 0.0]
    else:
        offsets = drawn["cup_source_offsets"]
        c["workflow"]["cup_source_offsets"] = {k: list(v) for k, v in offsets.items()}
    record = dict(
        version=VERSION,
        generator="numpy.PCG64",
        stream=[int(seed), LAYOUT_STREAM, TASK_KEYS[task_id]],
        **drawn,
    )
    c["spatial_randomization"] = record
    return record
