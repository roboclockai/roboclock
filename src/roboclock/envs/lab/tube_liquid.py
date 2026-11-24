"""B4 reduced-order liquid: volume-conserving free surface and gravity outflow.

This is a one-way coupling to measured PhysX tube poses, not a particle/CFD
solver. Equal-area columns integrate the rounded cavity; a damped surface mode
represents slosh. Escaped volume is irreversible and drives ballistic drops.
"""

import numpy as np

from roboclock.physics.geometry import rotation_matrix


class TubeLiquid:
    radius = 0.0122  # 0.3 mm rendering clearance from the inner glass wall
    top = 0.070
    bottom_center = -0.056
    spill_limit_ml = 0.75  # benchmark tolerance for the first escaping drops

    def __init__(self, fill_fraction=0.5):
        angles = np.arange(64) * (2 * np.pi / 64)
        radii = self.radius * np.sqrt((np.arange(20) + 0.5) / 20)
        self.xy = (
            radii[:, None, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        ).reshape(-1, 2)
        self.base = self.bottom(self.xy)
        self.area = np.pi * self.radius**2 / len(self.base)
        self.capacity = float(np.sum(self.top - self.base) * self.area)
        self.initial_volume = self.volume = self.capacity * fill_fraction
        self.spilled_ml = 0.0
        self.slope = None
        self.slope_velocity = np.zeros(2)
        self.normal = np.array([0.0, 0.0, 1.0])
        self.level = self.top
        self.rim_clearance = 0.0
        self.minimum_clearance = float("inf")
        self.drops = []
        self.drop_remainder = 0.0
        self.puddle_ml = 0.0
        self.puddle_position = np.zeros(3)
        self.solve_level()
        self.minimum_clearance = self.rim_clearance

    def bottom(self, xy):
        return self.bottom_center - np.sqrt(np.maximum(0, self.radius**2 - np.sum(xy**2, axis=-1)))

    def heights(self, xy):
        z = (self.level - xy @ self.normal[:2]) / self.normal[2]
        return np.clip(z, self.bottom(xy), self.top)

    def solve_level(self):
        n = self.normal
        low = float(np.min(self.xy @ n[:2] + self.base * n[2]))
        high = float(np.max(self.xy @ n[:2] + self.top * n[2]))
        for _ in range(28):
            self.level = (low + high) / 2
            volume = float(np.sum(self.heights(self.xy) - self.base) * self.area)
            if volume < self.volume:
                low = self.level
            else:
                high = self.level
        self.level = (low + high) / 2
        rim = self.top * n[2] - self.radius * np.linalg.norm(n[:2])
        self.rim_clearance = float(rim - self.level)

    def step(self, position, orientation, velocity, dt, table_height=0.75):
        if dt > 1 / 120 + 1e-10:
            count = int(np.ceil(dt * 120))
            for _ in range(count):
                self.step(position, orientation, velocity, dt / count, table_height)
            return
        rotation = rotation_matrix(orientation)
        gravity_normal = rotation.T @ np.array([0.0, 0.0, 1.0])
        # The reduced model is intended for upright/partially tipped tubes.
        target = gravity_normal[:2] / max(0.10, gravity_normal[2])
        target = np.clip(target, -8, 8)
        if self.slope is None:
            self.slope = target.copy()
        # A single damped gravity-wave mode. It settles to a world-horizontal plane.
        self.slope_velocity += (
            18**2 * (target - self.slope) - 2 * 0.55 * 18 * self.slope_velocity
        ) * dt
        self.slope += self.slope_velocity * dt
        self.normal = np.r_[self.slope, 1.0]
        self.normal /= np.linalg.norm(self.normal)
        self.solve_level()
        self.minimum_clearance = min(self.minimum_clearance, self.rim_clearance)
        head = max(0.0, -self.rim_clearance)
        # Free-overfall approximation Q = Cd * width * sqrt(2g) * head^(3/2).
        width = min(2 * self.radius, 2 * np.sqrt(2 * self.radius * head))
        escaped = min(self.volume, 0.60 * width * np.sqrt(2 * 9.81) * head**1.5 * dt)
        if gravity_normal[2] < 0:  # inverted tubes cannot retain the sample
            escaped = max(escaped, min(self.volume, self.initial_volume * dt * 2))
        self.volume -= escaped
        self.spilled_ml = (self.initial_volume - self.volume) * 1e6
        self.drop_remainder += escaped * 1e6
        outward = -self.normal[:2] / max(np.linalg.norm(self.normal[:2]), 1e-9)
        local_rim = np.r_[outward * (self.radius + 0.0018), self.top]
        while self.drop_remainder >= 0.012:
            self.drop_remainder -= 0.012
            self.drops.append(
                {
                    "p": (position + rotation @ local_rim).tolist(),
                    "v": (
                        velocity + rotation @ np.r_[outward * np.sqrt(2 * 9.81 * head), 0]
                    ).tolist(),
                }
            )
        airborne = []
        for drop in self.drops:
            v = np.array(drop["v"]) + [0, 0, -9.81 * dt]
            p = np.array(drop["p"]) + v * dt
            if p[2] <= table_height + 0.001:
                self.puddle_position = (self.puddle_position * self.puddle_ml + p * 0.012) / (
                    self.puddle_ml + 0.012
                )
                self.puddle_position[2] = table_height + 0.001
                self.puddle_ml += 0.012
            else:
                airborne.append({"p": p.tolist(), "v": v.tolist()})
        self.drops = airborne
        self.solve_level()

    def state(self):
        return {
            "liquid_model": "volume_conserving_free_surface_slosh_v1",
            "liquid_initial_ml": self.initial_volume * 1e6,
            "liquid_remaining_ml": self.volume * 1e6,
            "liquid_spilled_ml": self.spilled_ml,
            "liquid_spill_limit_ml": self.spill_limit_ml,
            "liquid_rim_clearance_mm": self.rim_clearance * 1000,
            "liquid_minimum_rim_clearance_mm": self.minimum_clearance * 1000,
            "liquid_plane_normal": self.normal.tolist(),
            "liquid_plane_level": self.level,
        }
