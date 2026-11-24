"""Privileged physical measurements used only by simulator-side scoring."""

from dataclasses import dataclass, field

import numpy as np

from roboclock.physics.geometry import orientation_error, upright_error  # noqa: F401


@dataclass(frozen=True)
class ObjectState:
    position: np.ndarray
    orientation: np.ndarray  # scalar-first quaternion (w, x, y, z)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grasped: bool = False  # bilateral finger contacts, closed gripper, near tool
    robot_contact: bool = False
    table_contact: bool = False
    fixture_contacts: frozenset[str] = field(default_factory=frozenset)
    relative_tool_speed: float = float("inf")
    release_speed: float = 0.0


@dataclass(frozen=True)
class PhysicalState:
    objects: dict[str, ObjectState]
    ee_position: np.ndarray
    gripper_open: bool = True
    robot_moving: bool = False
    ee_orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0]))

    controls: dict = field(default_factory=dict)  # Privileged measured fixture state.
    # Tool point of the rig's other arm (None on a single-arm backend). `ee_position` is the
    # first arm's (ROOTS[0], the rig's right arm); judging that asks "is a hand here?" must look
    # at both, or it depends on which arm the policy chose (see `tool_points`).
    secondary_ee_position: np.ndarray | None = None

    def tool_points(self):
        """The tool points of every arm, the first arm's first."""
        points = [np.asarray(self.ee_position, dtype=float)]
        if self.secondary_ee_position is not None:
            points.append(np.asarray(self.secondary_ee_position, dtype=float))
        return points


@dataclass(frozen=True)
class Target:
    position: np.ndarray
    half_extents: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    kind: str = "tray"
    support_z: float = 0.754
    rim_z: float = 0.789
    insertion_depth: float = 0.012

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(np.abs(point - self.position) <= self.half_extents))
