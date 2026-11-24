"""Geometry the skills use: approach orientations and the capsule model of an arm's body."""

import math

import numpy as np

LINK_RADIUS = 0.04  # m radius of the capsule around each arm link (joint origin to joint origin)
HAND_RADIUS = 0.055  # m radius of the capsule from the wrist to past the fingertips (open jaws span 0.088 m)
HAND_BEYOND = 0.02  # m the hand capsule extends past the tool point
UP = np.array([0.0, 0.0, 1.0])


def top_rotation(yaw):
    """Fingers down, jaws closing along (cos yaw, sin yaw, 0)."""
    jaw = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(jaw, z), jaw, z])


def hand_rotation(heading, pitch, roll=0.0):
    """Tool orientation for fingers pointing along `heading` in the table plane, tilted `pitch`
    below level, with the jaws turned `roll` about the fingers' axis from closing level across
    the heading. Pitch 0 is `front_rotation`; pitch pi/2 is `top_rotation(heading + pi/2)`."""
    level = np.array([math.cos(heading), math.sin(heading), 0.0])
    z = math.cos(pitch) * level - math.sin(pitch) * UP
    y = np.cross(UP, level)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        y = y * c + np.cross(z, y) * s + z * (z @ y) * (1 - c)  # Rodrigues, about the fingers
    return np.column_stack([np.cross(y, z), y, z])


def about_axis(axis, angle):
    """Rotation matrix by `angle` about the world direction `axis` (right-hand rule)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def front_rotation(direction):
    """Fingers along the horizontal `direction`, jaws closing horizontally across it."""
    z = np.array([direction[0], direction[1], 0.0])
    z /= max(np.linalg.norm(z), 1e-9)
    y = np.cross(UP, z)
    return np.column_stack([np.cross(y, z), y, z])



def segment_distance(p1, q1, p2, q2):
    """Shortest distance between the segments p1-q1 and p2-q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            s, t = min(max(-c / a, 0.0), 1.0), 0.0
        else:
            b = float(d1 @ d2)
            denominator = a * e - b * b
            s = min(max((b * f - c * e) / denominator, 0.0), 1.0) if denominator > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = min(max(-c / a, 0.0), 1.0), 0.0
            elif t > 1.0:
                s, t = min(max((b - c) / a, 0.0), 1.0), 1.0
    return float(np.linalg.norm(p1 + d1 * s - p2 - d2 * t))


def arm_capsules(arm, joints):
    """An arm's body at `joints` as capsules (start, end, radius): one per link between the joint
    origins from the base up, and one for the hand, from the wrist to past the fingertips. It is
    built from the kinematics alone, so it does not include anything the hand holds."""
    pose, origins, _ = arm._chain(np.asarray(joints, dtype=float), True)
    points = [arm.base[:3, 3], *origins]
    capsules = [(points[n], points[n + 1], LINK_RADIUS) for n in range(len(points) - 1)]
    capsules.append((origins[-1], pose[:3, 3] + pose[:3, 2] * HAND_BEYOND, HAND_RADIUS))
    return capsules


def capsules_gap(one, other):
    """Smallest surface-to-surface distance between two capsule sets (negative when they overlap)."""
    return min(segment_distance(a, b, c, d) - ra - rb for a, b, ra in one for c, d, rb in other)

