"""Geometry predicates over complete object envelopes and measured fixture dimensions."""

from dataclasses import dataclass

import numpy as np

from roboclock.envs.lab.mdp.state import Target
from roboclock.physics.geometry import rotation_matrix


@dataclass(frozen=True)
class Shape:
    """Collision envelope of a manipulated object, in meters in its own frame (z up)."""

    kind: str  # "cylinder" (radius, height) or "box" (half extents)
    radius: float = 0.0
    height: float = 0.0
    half: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_radius: float = 0.0  # half the width the fingers close across
    grasp_height: float = 0.0  # longest dimension, bounds the tool-to-object distance
    # Newtons per finger, set by how far the fingers are told to close. None closes fully and
    # lets the actuator's force cap decide (works for most items; ejects light cups).
    grip_force: float | None = None

    @property
    def rest_height(self) -> float:
        """Center height above its support when resting upright or flat."""
        return self.height / 2 if self.kind == "cylinder" else self.half[2]


@dataclass(frozen=True)
class Zone:
    """Bench region that is not a physical container: storage area, sterile field, spill."""

    center: np.ndarray  # world xy
    half_extents: np.ndarray | None = None  # rectangular zone
    radius: float | None = None  # circular zone
    support_z: float = 0.754
    margin: float = 0.0  # extra clearance required to count as outside


def object_extents(shape: Shape, orientation) -> np.ndarray:
    rotation = rotation_matrix(orientation)
    if shape.kind == "box":
        return np.abs(rotation) @ np.asarray(shape.half)
    axis = rotation[:, 2]
    return shape.radius * np.sqrt(np.maximum(0, 1 - axis * axis)) + shape.height / 2 * np.abs(axis)


def bottom_z(shape: Shape, obj) -> float:
    return float(obj.position[2] - object_extents(shape, obj.orientation)[2])


def fits_xy(shape: Shape, obj, target: Target, clearance=0.0005) -> bool:
    extent = object_extents(shape, obj.orientation)
    return bool(
        np.all(
            np.abs(obj.position[:2] - target.position[:2]) + extent[:2]
            <= target.half_extents[:2] - clearance
        )
    )


def seated(shape: Shape, obj, target: Target) -> bool:
    if not fits_xy(shape, obj, target):
        return False
    # Instruments may stack inside a basket; elsewhere the object must rest on the floor.
    top = target.rim_z - 0.012 if target.kind in ("basket", "bin") else target.support_z + 0.006
    return target.support_z - 0.003 <= bottom_z(shape, obj) <= top


def entry_crossing(shape: Shape, current, previous, target: Target) -> bool:
    """Cross a container lip from above; side entry or a hovering center does not count."""
    if previous is None or not fits_xy(shape, current, target):
        return False
    now, before = bottom_z(shape, current), bottom_z(shape, previous)
    plane = target.rim_z - target.insertion_depth
    return before >= plane and target.support_z - 0.003 <= now < plane and now < before


def footprint_inside(zone: Zone, shape: Shape, obj, clearance=0.0) -> bool:
    extent = object_extents(shape, obj.orientation)[:2]
    offset = np.abs(obj.position[:2] - zone.center)
    if zone.radius is not None:
        return bool(np.hypot(*(offset + extent)) <= zone.radius - clearance)
    return bool(np.all(offset + extent <= zone.half_extents - clearance))


def footprint_outside(zone: Zone, shape: Shape, obj, margin=0.0) -> bool:
    """True when no part of the object's bounding footprint touches the zone plus margin."""
    extent = object_extents(shape, obj.orientation)[:2]
    gap = np.abs(obj.position[:2] - zone.center) - extent
    if zone.radius is not None:
        return bool(np.hypot(*np.maximum(gap, 0.0)) > zone.radius + margin)
    return bool(np.any(gap > zone.half_extents + margin))


def supported_in_zone(zone: Zone, shape: Shape, obj, tolerance=0.012) -> bool:
    """Resting on the zone's surface within its footprint, not lifted away from it."""
    return footprint_inside(zone, shape, obj) and bottom_z(shape, obj) <= zone.support_z + tolerance


def on_bench(shape: Shape, obj, table_height: float, bounds) -> bool:
    low, high = bounds
    extent = object_extents(shape, obj.orientation)
    return bool(
        np.all(obj.position[:2] - extent[:2] >= low[:2] + 0.002)
        and np.all(obj.position[:2] + extent[:2] <= high[:2] - 0.002)
        and bottom_z(shape, obj) >= table_height - 0.004
    )
