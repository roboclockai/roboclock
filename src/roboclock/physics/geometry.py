"""Domain-independent rotation, quaternion distance and axial tilt helpers."""

import numpy as np


def rotation_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def orientation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest SO(3) angular distance; quaternion signs are equivalent."""
    dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return float(2 * np.arccos(np.clip(dot, 0, 1)))


def upright_error(quaternion: np.ndarray) -> float:
    """Tilt of a cylindrical container; rotation about its own axis is irrelevant."""
    _, x, y, _ = quaternion / np.linalg.norm(quaternion)
    return float(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def angular_velocity_from_poses(previous, current, dt):
    """World angular velocity from two normalized wxyz orientations.

    Quaternion sign flips represent the same orientation and yield zero motion.
    """
    a = np.asarray(previous, dtype=float)
    b = np.asarray(current, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    w = b[0] * a[0] + np.dot(b[1:], a[1:])
    v = -b[0] * a[1:] + a[0] * b[1:] - np.cross(b[1:], a[1:])
    if w < 0:
        w, v = -w, -v
    length = np.linalg.norm(v)
    return 2 * v / dt if length < 1e-10 else v * (2 * np.arctan2(length, w) / (length * dt))
