"""Contact-based hold verification; proximity alone never constitutes a grasp."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraspTracker:
    minimum_time: float = 0.04
    force_threshold: float = 0.06
    max_relative_speed: float = 0.20
    durations: dict = field(default_factory=dict)

    def update(
        self,
        name,
        forces,
        width,
        position,
        tool_position,
        velocity,
        tool_velocity,
        radius,
        height,
        dt,
    ):
        norms = np.linalg.norm(forces[:2], axis=-1)
        opposing = float(np.dot(forces[0], forces[1])) < -0.2 * norms[0] * norms[1]
        relative_speed = float(np.linalg.norm(velocity - tool_velocity))
        candidate = (
            bool(np.all(norms > self.force_threshold))
            and opposing
            and 0.001 < width <= 2 * radius + 0.014
            and np.linalg.norm(position - tool_position) < height * 0.5 + 0.025
            and relative_speed <= self.max_relative_speed
        )
        self.durations[name] = self.durations.get(name, 0.0) + dt if candidate else 0.0
        return bool(self.durations[name] + 1e-9 >= self.minimum_time), relative_speed
