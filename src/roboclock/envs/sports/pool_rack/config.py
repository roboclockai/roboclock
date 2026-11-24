"""Physical dimensions, legal seeded rack permutations and the episode horizon."""

import numpy as np

from roboclock.envs.sports import c_class
from roboclock.envs.sports._shared.precision.config import config as base_config
from roboclock.simulators.isaac.embodiments.config import configure

# Ball caddies beside the rack (|y| of the two rows, x of the four columns).
CADDY_ROWS = (0.44, 0.54)
CADDY_COLUMNS = (0.24, 0.34, 0.44, 0.54)
CADDY_CENTER = (0.39, 0.49, 0.849)  # x, |y|, z of the caddy floor centre


def config(difficulty="easy", seed=0):
    c = base_config("C1", difficulty, seed)
    configure(c, "C1", "dual_arx_x5")
    rng = np.random.default_rng(seed)
    diameter = 0.05715
    slots = [
        [
            0.508 - row * np.sqrt(3) * diameter / 2,
            (column - row / 2) * diameter,
            0.780 + diameter / 2,
        ]
        for row in range(5)
        for column in range(row + 1)
    ]
    order = [None] * 15
    order[4] = 8
    order[10], order[14] = int(rng.integers(1, 8)), int(rng.integers(9, 16))
    remaining = [i for i in range(1, 16) if i not in order]
    rng.shuffle(remaining)
    for i in range(15):
        if order[i] is None:
            order[i] = remaining.pop()
    sides = [
        "left" if p[1] > 1e-6 or (abs(p[1]) < 1e-6 and i != 4) else "right"
        for i, p in enumerate(slots)
    ]
    # Caddy cells (scene.CADDY): two rows of four per side, inside the RoboDojo head
    # camera's view (until 2026-11-03: rows 0.59/0.69 m, columns 0.12-0.42 m).
    stock = {
        side: [[x, sign * y, 0.888575] for y in CADDY_ROWS for x in CADDY_COLUMNS]
        for side, sign in (("left", 1), ("right", -1))
    }
    sources = {}
    for i, number in enumerate(order):
        p = np.asarray(stock[sides[i]].pop(0))
        p[:2] += rng.uniform(-0.002, 0.002, 2)
        sources[f"ball{number:02d}"] = p.tolist()
    c["revision"] = "sports-c1-rack-v1"
    c["workflow"].update(pool_racking=True)
    c["primary_temporal_category"] = "throughput"
    c["pool"].update(
        dual_cue_exchange=False, spot_clearance=False, finger_force_n=10.0, joint_speed=2.5
    )
    c["rack"] = dict(
        table_length=2.032,
        table_width=1.016,
        bed_height=0.780,
        radius=diameter / 2,
        mass=0.170,
        slots=slots,
        order=order,
        sides=sides,
        sources=sources,
        # Episode horizon H = 5 t_o (score_version 2, docs/c-class-timing.md), the same at
        # every difficulty: the difficulty only selects the judged time tier (4 / 2 / 1 t_o).
        # Until 2026-11-09 the horizon was the per-difficulty budget 180 / 155 / 140 s.
        horizon=c_class.horizon("C1"),
        settle_duration=0.5,
        grasp_duration=0.12,
        placement_tolerance=0.004,
        neighbor_gap_tolerance=0.002,
        grasp_height=0.006,
        release_clearance=0.0004,
        open_grip=0.040,
        closed_grip=0.010,
        transfer_height=1.04,
        template_thickness=0.0004,
        template_hole_radius=0.0045,
        template_outer_radius=0.009,
        initial_pitch=180.0,
        grasp_pitch=180.0,
        home={"left": [0.25, 0.42, 1.12], "right": [0.25, -0.42, 1.12]},
    )
    c["embodiment"].update(
        position=[0.14, 0.0, 1.02],
        yaw=0.0,
        active_arms=["left", "right"],
        auxiliary_arm_role="Bimanual ball pickup and placement",
        secondary_arm_role="Coordinated access to the common rack",
    )
    c["robot_mount"] = c["embodiment"]["position"].copy()
    c["visuals"].update(render_scale=1, record_robot_views=False)
    c["visuals"]["presentation_camera"].update(
        eye=[1.80, -2.35, 2.65],
        target=[0.50, 0.02, 0.83],
        focal_length=0.041,
        light_intensity_multiplier=1.75,
    )
    c["randomization"] = dict(
        seed=seed,
        layout="Source ball XY +/-2 mm; legal randomized ball identities",
        difficulty="None in the episode; it selects the judged time tier only",
    )
    return c
