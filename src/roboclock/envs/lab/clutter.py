"""Reproducible non-task obstacles; sampling never changes the task's random stream."""

from dataclasses import dataclass

import numpy as np

from roboclock.envs.lab.layout import apparatus_frame, indicator_position
from roboclock.envs.lab.robot import mount_positions


@dataclass(frozen=True)
class ClutterObject:
    name: str
    kind: str
    position: np.ndarray
    size: np.ndarray
    mass: float
    label: str


KINDS = (
    ("bottle", [0.054, 0.054, 0.125], 0.16, "BUFFER"),
    ("tip_box", [0.085, 0.065, 0.050], 0.12, "TIPS"),
    ("pipette_tray", [0.105, 0.035, 0.024], 0.06, "TOOLS"),
    ("notebook", [0.105, 0.075, 0.010], 0.08, "NOTES"),
    ("bottle", [0.050, 0.050, 0.110], 0.14, "RINSE"),
)


def reserved_regions(task, layout, config):
    """XY envelopes including gripper approach clearance and the moving slot sweep."""
    spec = config["scene"].get("clutter", {})
    source_clearance = spec.get("source_clearance", 0.045)
    target_clearance = spec.get("target_clearance", 0.045)
    regions = [
        (np.array([0.86, -0.32]), np.array([0.088, 0.12])),
        (indicator_position(task)[:2], np.array([0.047, 0.035])),
    ]
    regions.extend(
        (np.asarray(position[:2]), np.array([0.09, 0.09]))
        for position in mount_positions(config, task.task_id)
    )
    if getattr(task, "rack_organization", False):
        from roboclock.envs.lab.rack_organization import (
            COLOR_RACKS,
            SOURCE_RACKS,
            rack_center,
            rack_extent,
        )

        for source, racks in ((False, COLOR_RACKS), (True, SOURCE_RACKS)):
            for i in range(len(racks)):
                regions.append(
                    (
                        rack_center(i, layout.table_height, source, apparatus_frame(layout))[:2],
                        rack_extent(i, source),
                    )
                )
    if task.task_id == "A5":
        from roboclock.envs.lab.bath_scene import DEST_XY, DRY_XY, SOURCE_XY

        from roboclock.envs.lab.station_frame import station_of

        # Nominal station points, placed by the episode's seeded station frame (the extents
        # carry 1-4 cm of margin, more than a 6 degree turn adds to their corners).
        station = station_of(layout)
        for xy, extent in [
            (SOURCE_XY, [0.16, 0.16]),
            (DEST_XY, [0.16, 0.16]),
            (DRY_XY, [0.10, 0.10]),
            ([0.79, 0.18], [0.16, 0.09]),
            ([0.76, -0.39], [0.08, 0.08]),
        ]:
            regions.append((station.point(xy), np.array(extent)))
    if task.task_id == "A6":
        # The tube starts on the bench stand, so the warm bath is no longer under the sample.
        from roboclock.envs.lab.bath_scene import DRY_XY, SOURCE_XY

        from roboclock.envs.lab.station_frame import station_of

        station = station_of(layout)
        for xy, extent in [(SOURCE_XY, [0.16, 0.16]), (DRY_XY, [0.08, 0.08])]:
            regions.append((station.point(xy), np.array(extent)))
    for name, obj in layout.objects.items():
        radius = layout.dimensions[name][0]
        regions.append((obj.position[:2], np.full(2, radius + source_clearance)))
    for target in layout.targets.values():
        extent = target.half_extents[:2] + target_clearance + 0.006
        if task.task_id == "A4":
            extent = extent + np.array([task.amplitude, 0.035])
        regions.append((target.position[:2], extent))
    if task.task_id == "A1":
        regions.append((layout.heating_zone.position[:2], np.array([0.07, 0.07])))
    if getattr(task, "hotplate_task", False):
        regions.append((layout.heating_zone.position[:2], np.array([0.12, 0.12])))
    if task.task_id == "B5":
        from roboclock.envs.lab.safety_geometry import freezer_footprint

        regions.append(freezer_footprint(apparatus_frame(layout)))
    if task.task_id in ("B2", "B3"):
        holder = layout.objects["sample_0"].position[:2] + [0, 0.12]
        regions.append((holder, np.array([0.05, 0.03])))
    return regions


def sample_clutter(task, layout, config):
    spec = config["scene"].get("clutter", {})
    if not spec.get("enabled", False):
        return []
    count = spec.get("count", 10)
    if not isinstance(count, int) or not 0 <= count <= 30:
        raise ValueError("scene.clutter.count must be an integer in [0, 30]")
    rng = np.random.default_rng(
        np.random.SeedSequence([task.spec.seed, spec.get("seed_offset", 202)])
    )
    low, high = layout.workspace_bounds
    pair_clearance = spec.get("pair_clearance", 0.012)
    regions = reserved_regions(task, layout, config)
    result = []
    for index in range(count):
        kind, size, mass, label = KINDS[index % len(KINDS)]
        size = np.array(size)
        half = size[:2] / 2
        for _ in range(2000):
            xy = rng.uniform(low[:2] + half + 0.025, high[:2] - half - 0.025)
            if any(np.all(np.abs(xy - center) < half + extent) for center, extent in regions):
                continue
            if any(
                np.all(np.abs(xy - other.position[:2]) < half + other.size[:2] / 2 + pair_clearance)
                for other in result
            ):
                continue
            result.append(
                ClutterObject(
                    f"clutter_{index:02d}",
                    kind,
                    np.array([*xy, layout.table_height + size[2] / 2 + 0.001]),
                    size,
                    mass,
                    label,
                )
            )
            break
        else:
            raise ValueError(
                f"Cannot place {count} clutter objects for {task.task_id}; reduce clutter density"
            )
    return result


def build_clutter(backend):
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    appearance = backend.appearance
    physical_material = PhysicsMaterial(
        "/World/Materials/ClutterFriction",
        static_friction=0.65,
        dynamic_friction=0.55,
        restitution=0.0,
    )
    bodies = {}
    for item in backend.clutter_specs:
        path = f"/World/Clutter/{item.name}"
        root = UsdGeom.Xform.Define(backend.stage, path).GetPrim()
        half_height = item.size[2] / 2
        if item.kind == "bottle":
            collider = appearance.cylinder_child(
                path,
                "Collider",
                item.size[0] / 2,
                item.size[2],
                0,
                appearance.material("polymer"),
                collision=True,
            )
            cap = appearance.material("ClutterCap", color=[0.18, 0.39, 0.44], roughness=0.4)
            appearance.vial(path, item.size[0] / 2, item.size[2], cap)
            if backend.config["visuals"]["labels"]:
                appearance.label(
                    path + "/Label",
                    item.label,
                    [0, 0, -0.004],
                    [0.047, 0.024],
                    cylindrical_radius=item.size[0] / 2 + 0.0004,
                )
        else:
            shape = UsdGeom.Cube.Define(backend.stage, path + "/Collider")
            shape.CreateSizeAttr(1.0)
            shape.AddScaleOp().Set(Gf.Vec3f(*item.size.tolist()))
            collider = shape.GetPrim()
            UsdPhysics.CollisionAPI.Apply(collider)
            material = "paper" if item.kind == "notebook" else "polymer"
            appearance.rounded(path + "/Body", [0, 0, 0], item.size, material, 0.002)
            if item.kind == "tip_box":
                appearance.rounded(
                    path + "/Lid",
                    [0, 0, half_height - 0.005],
                    [item.size[0] - 0.002, item.size[1] - 0.002, 0.010],
                    "paper",
                    0.001,
                )
            elif item.kind == "pipette_tray":
                for index, y in enumerate([-0.008, 0.008]):
                    appearance.rounded(
                        path + f"/Pipette_{index}",
                        [0, y, half_height - 0.003],
                        [0.082, 0.004, 0.004],
                        "paper",
                        0.001,
                    )
            elif item.kind == "notebook":
                cover = appearance.material(
                    "ClutterNotebookCover", color=[0.12, 0.25, 0.28], roughness=0.65
                )
                appearance.rounded(
                    path + "/Cover",
                    [0, 0, half_height - 0.001],
                    [item.size[0], item.size[1], 0.002],
                    cover,
                    0.0005,
                )
            if backend.config["visuals"]["labels"] and item.kind != "notebook":
                appearance.label(
                    path + "/Label",
                    item.label,
                    [0, item.size[1] / 2 + 0.0003, 0],
                    [item.size[0] * 0.8, item.size[2] * 0.5],
                )
        UsdGeom.Imageable(collider).MakeInvisible()
        backend._collision_settings(collider)
        UsdShade.MaterialBindingAPI.Apply(collider).Bind(
            physical_material.material, materialPurpose="physics"
        )
        UsdPhysics.RigidBodyAPI.Apply(root)
        PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateEnableCCDAttr(True)
        body = SingleRigidPrim(
            prim_path=path,
            name=item.name,
            position=item.position,
            mass=item.mass,
        )
        bodies[item.name] = backend.world.scene.add(body)
    return bodies
