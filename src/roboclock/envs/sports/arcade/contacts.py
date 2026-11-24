"""Classify measured world-space contact points on the lower striking head."""

import numpy as np

from roboclock.physics.geometry import rotation_matrix


def head_contacts(
    points,
    forces,
    position,
    orientation,
    rectangular=False,
    *,
    radius=0.024,
    lower_only=True,
    head_offset=0.125,
):
    points = np.asarray(points).reshape(-1, 3)
    forces = np.abs(np.asarray(forces).reshape(-1))
    local = (points - np.asarray(position)) @ rotation_matrix(orientation)
    relative = local - np.array([0, 0, -head_offset])
    inside = (
        np.all(np.abs(relative) <= [0.053, 0.018, 0.026], axis=1)
        if rectangular
        else np.linalg.norm(relative, axis=1) <= radius + 0.003
    )
    # A downward strike must contact the lower portion, not the shaft junction.
    mask = (
        inside
        & ((local[:, 2] < -head_offset + 0.005) if lower_only else np.ones(len(points), dtype=bool))
        & np.isfinite(forces)
    )
    return float(forces[mask].sum()), points[mask]


def mole_head_activated(head_force, held, exposed, travel):
    """The same grasp/head/exposure/travel predicate for either mallet."""
    return bool(head_force > 0.05 and held and exposed and 0.002 <= travel <= 0.035)
