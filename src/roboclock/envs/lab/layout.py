"""Seeded scene bindings shared by the physical backend and task predicates."""

from dataclasses import dataclass, field

import numpy as np

from roboclock.envs.lab.mdp.state import Target


@dataclass(frozen=True)
class ObjectConfig:
    name: str
    asset: str
    position: np.ndarray
    orientation: np.ndarray
    color: np.ndarray
    mass: float
    friction: float


@dataclass(frozen=True)
class ApparatusFrame:
    """Per-seed rigid placement of a task's fixed apparatus relative to the robot rig.

    A nominal (authored) world point p is placed at R(yaw) (p - anchor) + anchor + offset
    (xy only, z unchanged). This is the single source of truth for B2/B3/B5/B6/B7 and C1:
    the scene builders author their prims in nominal coordinates under a parent Xform that
    carries this transform (`author_xform`), and everything consumed outside that USD
    hierarchy (layout objects, world-attached joint frames, oracle, cues, clutter keep-outs)
    maps nominal points through `point` / `vector`.
    """

    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw: float = 0.0
    anchor: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def vector(self, vector):
        """A nominal world vector (xy rotated, z kept)."""
        v = np.asarray(vector, dtype=float)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.r_[c * v[0] - s * v[1], s * v[0] + c * v[1], v[2:]]

    def point(self, point):
        """The placed world point of a nominal world point (xy or xyz)."""
        p = np.asarray(point, dtype=float)
        xy = self.vector(np.r_[p[:2] - self.anchor, 0.0])[:2] + self.anchor + self.offset
        return np.r_[xy, p[2:]]

    def inverse_point(self, point):
        p = np.asarray(point, dtype=float)
        local = np.r_[p[:2] - self.anchor - self.offset, 0.0]
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        xy = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]]) + self.anchor
        return np.r_[xy, p[2:]]

    @property
    def identity(self):
        return not np.any(self.offset) and self.yaw == 0.0

    def author_xform(self, stage, path):
        """Define `path` as an Xform carrying this placement (no ops when identity)."""
        from pxr import Gf, UsdGeom

        root = UsdGeom.Xform.Define(stage, path)
        if not self.identity:
            pivot = self.anchor + self.offset
            root.AddTranslateOp(opSuffix="apparatusPlace").Set(Gf.Vec3d(*map(float, pivot), 0.0))
            root.AddRotateZOp(opSuffix="apparatusYaw").Set(float(np.rad2deg(self.yaw)))
            root.AddTranslateOp(opSuffix="apparatusAnchor").Set(
                Gf.Vec3d(*map(float, -self.anchor), 0.0)
            )
        return root

    def info(self):
        return {
            "offset_xy_m": [float(v) for v in self.offset],
            "yaw_rad": float(self.yaw),
            "anchor_xy_m": [float(v) for v in self.anchor],
        }


IDENTITY_FRAME = ApparatusFrame()


def apparatus_frame(layout):
    """The layout's apparatus placement (identity for layouts built without one)."""
    return getattr(layout, "apparatus", None) or IDENTITY_FRAME


# Per-seed apparatus placement, drawn from the layout stream's common xy jitter (and the
# sample yaw draw) that these tasks used to discard, so no other layout draw changes.
# x range (m), y range (m), yaw half range (rad), yaw anchor (nominal world xy).
# B2/B3: y >= -0.01 m: the disconnect stand (x 0.18-0.42, y 0.04-0.20 nominal) would
# otherwise cut the corner of the task arm's mount column (x 0.04-0.16, y -0.09-0.03, rig
# origin (0.10, 0.27)) at x -2.5 cm. No yaw: the knife switch hinge (world
# Y axis), its measured opening angle (world pitch) and the oracle's pull arc assume it.
# B5: yaw about the freezer centre; B6/B7: yaw about the plunger. C1: all seven racks
# shift together (no yaw: slot targets and grasp orientations are axis aligned); x <= 0:
# the far slots sit at the X5's planned reach (0.475 m at transit height), and a +5 mm (or
# a -20 mm) shift already leaves tubes no able arm (test_expert_recovery partition check).
APPARATUS_JITTER = {
    "B2": ((-0.025, 0.025), (-0.010, 0.025), 0.0, (0.0, 0.0)),
    "B3": ((-0.025, 0.025), (-0.010, 0.025), 0.0, (0.0, 0.0)),
    "B5": ((-0.025, 0.025), (-0.025, 0.025), np.deg2rad(3.0), (0.755, 0.16)),
    "B6": ((-0.025, 0.025), (-0.025, 0.025), np.deg2rad(3.0), (0.40, 0.16)),
    "B7": ((-0.025, 0.025), (-0.025, 0.025), np.deg2rad(3.0), (0.40, 0.16)),
    "C1": ((-0.015, 0.0), (-0.015, 0.015), 0.0, (0.0, 0.0)),
}


def apparatus_placement(task_id, jitter, yaw_draw, spatial_jitter, yaw_range=0.15):
    """ApparatusFrame from the layout's unit draws (jitter in +-spatial_jitter, yaw in
    +-yaw_range); identity for other tasks or when spatial jitter is disabled."""
    if task_id not in APPARATUS_JITTER or spatial_jitter <= 0:
        return IDENTITY_FRAME
    x_range, y_range, half_yaw, anchor = APPARATUS_JITTER[task_id]
    u = np.asarray(jitter, dtype=float) / spatial_jitter  # in [-1, 1]
    offset = np.array(
        [(low + high) / 2 + (high - low) / 2 * k for (low, high), k in zip((x_range, y_range), u)]
    )
    return ApparatusFrame(offset, float(half_yaw * yaw_draw / yaw_range), np.array(anchor))


@dataclass
class LabLayout:
    objects: dict[str, ObjectConfig]
    targets: dict[str, Target]
    assignments: dict[str, str]
    heating_zone: Target
    table_height: float
    light_intensity: float
    camera_jitter: np.ndarray
    dimensions: dict[str, tuple[float, float]]
    workspace_bounds: tuple[np.ndarray, np.ndarray]
    # Per-seed rigid placement of the task's apparatus (identity unless APPARATUS_JITTER).
    apparatus: ApparatusFrame = field(default_factory=ApparatusFrame)


COLORS = ((0.9, 0.15, 0.1), (0.1, 0.3, 0.95), (0.1, 0.8, 0.2))
HEIGHTS = {
    "bottle": 0.09,
    "container": 0.075,
    "beaker": 0.085,
    "knife_switch": 0.06,
    "safety_actuator": 0.04,
    "test_tube": 0.14,
    "pellet": 0.024,
    "probe": 0.16,
    "calibration": 0.05,
}


# C2/C3 grids for the common side-by-side rig (rig origin (0, 0), yaw 0; arm
# bases at y=-0.30 and +0.30). Samples alternate between the two arms' halves;
# in each half the source row pair sits in front of the target row pair, 8
# columns centered at y=-/+0.28, so every sample's source and target share a
# column and lie within one arm's reach (<= 0.47 m, >= 0.20 m from its base).
THROUGHPUT_GRID = {
    True: {"rows_x": (0.20, 0.26), "spacing": 0.06},  # sources
    False: {"rows_x": (0.34, 0.402), "spacing": 0.065},  # targets
}
GRID_HALF_Y = 0.28
GRID_COLUMNS = 8


def throughput_grid(index, source):
    """XY of C2/C3 sample `index` (source) or of its target (supports 32 samples)."""
    half, slot = index % 2, index // 2
    column, row = slot % GRID_COLUMNS, slot // GRID_COLUMNS
    if row > 1:
        raise ValueError("C2/C3 grids hold at most 32 samples")
    grid = THROUGHPUT_GRID[source]
    y = (-GRID_HALF_Y if half == 0 else GRID_HALF_Y) + (column - (GRID_COLUMNS - 1) / 2) * grid[
        "spacing"
    ]
    return np.array([grid["rows_x"][row], y])


def check_config(config):
    """Check all laboratory layouts without loading the simulation runtime."""
    from roboclock.envs.lab.tasks.registry import TASKS, make_task

    if "rig" not in config.get("robot", {}):
        raise ValueError("Laboratory tasks require the common dual-X5 rig (robot.rig)")
    for key in TASKS:
        task = make_task(key, config=config)
        task.bind(make_layout(task, config))


def indicator_position(task):
    if getattr(task, "rack_organization", False):
        # Beyond the racks, clear of both arm columns.
        return np.array([0.70, 0.35, task.layout.table_height + 0.045])
    if getattr(task, "tube_task", False):
        # B4's rig is centred on the tube: (0.70, 0.35) lies under the rig-left arm in
        # the head image. Beyond the rack, just left of the tube's line of sight.
        return np.array([0.60, -0.06, task.layout.table_height + 0.045])
    # Larger racks occupy the original indicator location in their fifth row.
    x = 0.82 if task.category == "throughput" and task.spec.num_samples > 34 else 0.70
    return np.array([x, 0.35, task.layout.table_height + 0.045])


def make_layout(task, config: dict, catalog=None) -> LabLayout:
    if getattr(task, "bath_task", False):
        from roboclock.envs.lab.bath_scene import bath_layout

        return bath_layout(task, config)
    if getattr(task, "liquid_process", False):
        from roboclock.envs.lab.liquid_scene import liquid_layout

        return liquid_layout(task, config)
    rng = np.random.default_rng(np.random.SeedSequence([task.spec.seed, 101]))
    scene = config["scene"]
    z = scene["table_height"]
    count = (
        task.spec.num_samples
        if task.category == "throughput"
        else (3 if task.task_id == "A3" else 1)
    )
    objects, targets, assignments = {}, {}, {}
    # Common translation preserves spacing, preventing overlaps across random seeds.
    jitter = rng.uniform(-scene["spatial_jitter"], scene["spatial_jitter"], 2)
    if getattr(task, "rack_organization", False) and count > 36:
        raise ValueError("C1 has three 12-slot color racks: at most 36 tubes are supported")
    source_order = None
    frame = None  # apparatus placement, from the first sample's draws (below)
    if getattr(task, "rack_organization", False):
        from roboclock.envs.lab.rack_organization import source_slots

        source_order = source_slots(count, rng)
    for i in range(count):
        name = f"sample_{i}"
        if task.category == "throughput":
            # Sources sit 6-8 cm in front of the target slots: shift them only
            # toward the rig (x) and by at most 2 cm sideways.
            xy = (
                throughput_grid(i, source=True) + jitter * [0.3, 0.4] - [0.015, 0.0]
                if task.task_id in ("C2", "C3")
                else np.array(  # superseded V1 sorting (tests only)
                    [0.23 + (i % 7) * 0.06, (-0.46 if count > 42 else -0.40) + (i // 7) * 0.06]
                )
                + jitter
            )
        else:
            xy = np.array([0.43 + i * 0.065, -0.18]) + jitter
        height = catalog[task.asset]["height"] if catalog else HEIGHTS[task.asset]
        yaw = float(rng.uniform(-0.15, 0.15))
        if frame is None:
            frame = apparatus_placement(task.task_id, jitter, yaw, scene["spatial_jitter"])
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        pos = np.array([*xy, z + height / 2 + 0.002])
        if getattr(task, "hotplate_task", False):
            from roboclock.envs.lab.hotplate_scene import PLATE_HEIGHT

            pos[2] += PLATE_HEIGHT
        if getattr(task, "power_task", False):
            from roboclock.envs.lab.knife_switch_scene import GRIP_RADIUS, PIVOT_HEIGHT, PIVOT_XY

            # The disconnect follows the apparatus placement (translation only).
            pos = frame.point([PIVOT_XY[0] + GRIP_RADIUS, PIVOT_XY[1], z + PIVOT_HEIGHT])
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        if getattr(task, "safety_task", False):
            from roboclock.envs.lab.safety_geometry import actuator_pose

            pos, quat = actuator_pose(task.task_id, z, frame)
        if getattr(task, "tube_task", False):
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            # Loose bottom socket leaves room for the cylinder collision envelope.
            pos[2] = z + task.rack_support + height / 2 + 0.006
        if task.task_id == "B3" and not getattr(task, "power_task", False):
            pos[2] = z + 0.45
        if getattr(task, "rack_organization", False):
            from roboclock.envs.lab.rack_organization import rack_spec, slot_position

            pos = slot_position(source_order[i], z, source=True, frame=frame)
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            target_name = f"target_{(i % 3) * 12 + i // 3}"
            if not targets:
                for slot in range(36):
                    targets[f"target_{slot}"] = Target(
                        slot_position(slot, z, frame=frame),
                        np.array([0.019, 0.019, 0.075]),
                        kind="slot",
                        support_z=z + rack_spec()["support_z"],
                        rim_z=z + rack_spec()["rim_z"],
                    )
        elif getattr(task, "bin_sorting", False):
            target_name = f"bin_{i % 3}"
            targets[target_name] = Target(
                np.array([0.34 + (i % 3) * 0.18, 0.22, z + height / 2 + 0.004]),
                np.array([0.069, 0.108, 0.05]) if count > 36 else np.array([0.065, 0.095, 0.05]),
                kind="tray",
                support_z=z + 0.004,
                rim_z=z + 0.060,
            )
        elif task.category == "throughput":
            target_name = f"target_{i}"
            kind = "slot" if task.task_id == "C2" else "marker"
            targets[target_name] = Target(
                np.array(
                    [*throughput_grid(i, source=False), z + height / 2 + 0.004]
                    if task.task_id in ("C2", "C3")
                    else [0.27 + (i % 7) * 0.065, 0.08 + (i // 7) * 0.062, z + height / 2 + 0.004]
                ),
                np.array([0.025, 0.025, 0.055]),
                kind=kind,
                support_z=z + 0.004,
                rim_z=z + (0.022 if kind == "slot" else 0.004),
            )
        else:
            target_name = "destination"
            kind = {"A2": "slot", "A3": "vessel", "A4": "slot"}.get(task.task_id, "tray")
            targets[target_name] = Target(
                np.array([0.5, 0.20, z + height / 2 + 0.004]),
                np.array([0.07, 0.07, 0.055])
                if kind in ("tray", "vessel")
                else np.array([0.025, 0.025, 0.095]),
                kind=kind,
                support_z=z + 0.004,
                rim_z=z + ((0.040 if task.task_id == "A4" else 0.022) if kind == "slot" else 0.060),
            )
        color = np.array(COLORS[i % 3])
        if task.task_id in ("C2", "C3"):
            import colorsys

            color = np.array(colorsys.hsv_to_rgb(i / count, 0.75, 0.9))
        objects[name] = ObjectConfig(
            name,
            task.asset,
            pos,
            quat,
            color,
            (
                __import__("roboclock.envs.lab.safety_geometry", fromlist=["mechanism"]).mechanism(
                    task.task_id, z, frame
                )["mass"]
                if getattr(task, "safety_task", False)
                else float(rng.uniform(*scene["mass_range"]))
            ),
            float(rng.uniform(*scene["friction_range"])),
        )
        assignments[name] = target_name
    heating = Target(objects["sample_0"].position.copy(), np.array([0.055, 0.055, 0.055]))
    if getattr(task, "hotplate_task", False):
        heating = Target(
            np.array([*objects["sample_0"].position[:2], z + PLATE_HEIGHT]),
            np.array([0.08, 0.05, 0.002]),
            kind="hotplate",
            support_z=z + PLATE_HEIGHT,
            rim_z=z + PLATE_HEIGHT,
        )
    return LabLayout(
        objects,
        targets,
        assignments,
        heating,
        z,
        float(rng.uniform(*scene["light_intensity_range"])),
        rng.uniform(-0.03, 0.03, 3),
        {
            name: (
                (
                    catalog[task.asset]["radius"]
                    if catalog
                    else {
                        "bottle": 0.022,
                        "container": 0.018,
                        "beaker": 0.025,
                        "knife_switch": 0.012,
                        "safety_actuator": 0.014,
                        "test_tube": 0.014,
                        "pellet": 0.012,
                        "probe": 0.009,
                        "calibration": 0.02,
                    }[task.asset]
                ),
                (catalog[task.asset]["height"] if catalog else HEIGHTS[task.asset]),
            )
            for name in objects
        },
        (
            np.array(scene["table_center"]) - np.array(scene["table_size"]) / 2,
            np.array(scene["table_center"]) + np.array(scene["table_size"]) / 2,
        ),
        frame,
    )
