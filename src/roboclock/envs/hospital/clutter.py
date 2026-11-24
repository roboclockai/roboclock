"""Seeded non-task bench clutter; sampling never touches the task's random stream."""

import numpy as np

from roboclock.envs.lab.clutter import ClutterObject

# (kind, size, mass, label); "bottle" builds a vial, anything else a labeled box.
KINDS = (
    ("box", [0.095, 0.060, 0.075], 0.10, "GLOVES"),
    ("bottle", [0.050, 0.050, 0.120], 0.14, "ALCOHOL"),
    ("box", [0.080, 0.060, 0.030], 0.06, "GAUZE"),
    ("box", [0.110, 0.080, 0.012], 0.08, "CHART"),
    ("bottle", [0.054, 0.054, 0.125], 0.16, "RINSE"),
)

# An operating room has sterile supplies on the back table, not bottles of rinse.
OR_KINDS = (
    ("box", [0.095, 0.060, 0.045], 0.08, "GAUZE"),
    ("box", [0.080, 0.055, 0.030], 0.06, "SUTURE"),
    ("box", [0.110, 0.080, 0.025], 0.08, "SPONGES"),
    ("box", [0.090, 0.060, 0.030], 0.06, "DRAPE"),
    ("box", [0.095, 0.060, 0.060], 0.09, "GLOVES"),
)


def sample_clutter(task, layout, config):
    if layout.info.get("bedside_task"):
        return []  # Prepared clinical work area; arm/device clearance is part of the setup.
    spec = config["scene"].get("clutter", {})
    if not spec.get("enabled", False) or task.task_id == "B2":  # nothing is left lying on a patient
        return []
    count = spec.get("count", 8)
    if not isinstance(count, int) or not 0 <= count <= 30:
        raise ValueError("scene.clutter.count must be an integer in [0, 30]")
    rng = np.random.default_rng(
        np.random.SeedSequence([task.spec.seed, spec.get("seed_offset", 202)])
    )
    low, high = layout.workspace_bounds
    pair = spec.get("pair_clearance", 0.012)
    margin = spec.get("source_clearance", 0.045)
    from roboclock.envs.lab.robot import mount_positions

    # Internal arm 0 (rig right) works; arm 1 (rig left) idles.
    worker, idle = (np.array(m[:2]) for m in mount_positions(config, task.task_id))
    regions = [(c, h + margin) for c, h in layout.keepouts(worker)]
    regions.append((idle, np.array([0.18, 0.18]) + margin))  # the idle arm-1 column
    result = []
    kinds = OR_KINDS if task.task_id == "A3" else KINDS
    for index in range(count):
        kind, size, mass, label = kinds[index % len(kinds)]
        size = np.array(size)
        half = size[:2] / 2
        for _ in range(2000):
            xy = rng.uniform(low[:2] + half + 0.025, high[:2] - half - 0.025)
            if any(np.all(np.abs(xy - c) < half + e) for c, e in regions):
                continue
            if any(
                np.all(np.abs(xy - o.position[:2]) < half + o.size[:2] / 2 + pair) for o in result
            ):
                continue
            position = np.array([*xy, layout.table_height + size[2] / 2 + 0.001])
            result.append(ClutterObject(f"clutter_{index:02d}", kind, position, size, mass, label))
            break
        else:
            raise ValueError(f"Cannot place {count} clutter objects for {task.task_id}")
    return result
