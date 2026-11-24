"""Geometry predicates use the complete object envelope and actual fixture dimensions."""

import numpy as np

from roboclock.physics.geometry import rotation_matrix


def object_extents(task, name, obj):
    radius, height = task.layout.dimensions[name]
    rotation = rotation_matrix(obj.orientation)
    if task.asset == "pellet":
        return np.full(3, radius)
    if task.asset == "calibration":
        return np.abs(rotation) @ np.array([radius, radius, height / 2])
    axis = rotation[:, 2]
    return radius * np.sqrt(np.maximum(0, 1 - axis * axis)) + height / 2 * np.abs(axis)


def bottom_tip(task, name, obj):
    return (
        obj.position - rotation_matrix(obj.orientation)[:, 2] * task.layout.dimensions[name][1] / 2
    )


def fits_xy(task, name, obj, target, offset=0.0, clearance=0.0005):
    center = target.position.copy()
    center[0] += offset
    extent = object_extents(task, name, obj)
    return bool(
        np.all(
            np.abs(obj.position[:2] - center[:2]) + extent[:2]
            <= target.half_extents[:2] - clearance
        )
    )


def seated(task, name, obj, target, offset=0.0):
    if not fits_xy(task, name, obj, target, offset):
        return False
    bottom = obj.position[2] - object_extents(task, name, obj)[2]
    return target.support_z - 0.003 <= bottom <= target.support_z + 0.006


def in_slot(task, name, obj, target, clearance=0.0005):
    """A cylinder rests in a hole: its axis passes through the hole at the floor and at the rim.

    ``seated`` asks the whole tilted envelope to fit the hole footprint, so a tall tube that
    leans against the hole wall by more than ~2 degrees fails it while still standing in its
    hole. Here the bottom face and the axis point at rim height must each lie inside the
    footprint (with the end disk's projected half-width), and the lowest point must rest on
    the floor as in ``seated``. Every pose ``seated`` accepts is accepted here.
    """
    radius, height = task.layout.dimensions[name]
    axis = rotation_matrix(obj.orientation)[:, 2]
    if axis[2] <= 0:
        return False
    half = radius * np.sqrt(np.maximum(0, 1 - axis * axis))[:2]
    bottom = obj.position - axis * height / 2
    at_rim = bottom + axis * (target.rim_z - bottom[2]) / axis[2]
    for point in (bottom, at_rim):
        if not np.all(
            np.abs(point[:2] - target.position[:2]) + half <= target.half_extents[:2] - clearance
        ):
            return False
    lowest = obj.position[2] - object_extents(task, name, obj)[2]
    return target.support_z - 0.003 <= lowest <= target.support_z + 0.006


def insertion_crossing(task, name, current, previous, target, offset=0.0):
    """Cross a rack lip from above; side entry or a hovering COM does not count."""
    if previous is None or not fits_xy(task, name, current, target, offset):
        return False
    now, before = bottom_tip(task, name, current), bottom_tip(task, name, previous)
    plane = target.rim_z - target.insertion_depth
    return before[2] >= plane and target.support_z - 0.003 <= now[2] < plane and now[2] < before[2]


def safe_workspace(task, name, obj, margin=0.002):
    low, high = task.layout.workspace_bounds
    extents = object_extents(task, name, obj)
    return bool(
        np.all(obj.position[:2] - extents[:2] >= low[:2] + margin)
        and np.all(obj.position[:2] + extents[:2] <= high[:2] - margin)
        and obj.position[2] - extents[2] >= task.layout.table_height - 0.004
    )
