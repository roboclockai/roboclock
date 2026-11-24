"""Metric pool-fixture layout shared by scene, planner and capture predicates."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoolLayout:
    scale: float = 1.0
    center_x: float = .70
    center_y: float = 0.
    pocket_half_width: float = .09

    @classmethod
    def from_config(cls, config):
        c = config.get('pool', {})
        return cls(c.get('table_scale', 1.), *c.get('table_center', [.70, 0.]), c.get('pocket_half_width', .09))

    def point(self, value):
        p = np.array(value, dtype=float, copy=True)
        p[:2] = (p[:2] - [.70, 0.]) * self.scale + [self.center_x, self.center_y]
        return p

    def inverse_point(self, value):
        p = np.array(value, dtype=float, copy=True)
        p[:2] = (p[:2] - [self.center_x, self.center_y]) / self.scale + [.70, 0.]
        return p

    @property
    def cue_home(self):
        if self.scale == 1.:
            return self.point([.34, -.28, .794])
        low, _ = self.bed_bounds
        return np.array([low[0] + .17, low[1] + .07, .794])

    @property
    def bed_bounds(self):
        return self.point([.23, -.38]), self.point([1.17, .38])

    def throat(self, pocket):
        return np.array([self.point([1.17 if pocket[0] > self.center_x else .23, 0])[0], pocket[1]])

    @property
    def end_segments(self):
        h = self.pocket_half_width
        return ((-.38, -.12 - h), (-.12 + h, .22 - h), (.22 + h, .38))

    def jaw_boxes(self):
        for x in (.20, 1.20):
            for lo, hi in self.end_segments:
                yield self.point([x - .0325, lo, .748]), self.point([x + .0325, hi, .808])

        for sign in (-1, 1):
            y0, y1 = sorted((sign * .356, sign * .45))
            yield self.point([.15, y0, .748]), self.point([1.25, y1, .802])

    def ball_path_clear(self, start, end, radius):
        """Conservative full-sphere corridor through the actual cushion jaws."""
        start, end = np.asarray(start)[:2], np.asarray(end)[:2]
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / .003)) + 1)
        points = start + np.linspace(0., 1., count)[:, None] * (end - start)
        for lower, upper in self.jaw_boxes():
            nearest = np.clip(points, lower[:2], upper[:2])
            if np.any(np.linalg.norm(points - nearest, axis=1) < radius + .0015):
                return False
        return True

    def capture_half_extents(self, ball_radius):
        # Inside physical pocket walls, including the complete ball envelope.
        return np.array([.094, self.pocket_half_width - .010]) * self.scale - ball_radius + .004

    def apply_fixture_transform(self, stage):
        from pxr import Gf, UsdGeom
        translation = [self.center_x - .70 * self.scale, self.center_y, 0.]
        for path in ['/World/Pool', '/World/PoolBed', *[f'/World/Pocket{i}' for i in range(4)]]:
            xf = UsdGeom.Xform.Define(stage, path)
            xf.AddTranslateOp(opSuffix='compactLayout').Set(Gf.Vec3d(*translation))
            xf.AddScaleOp(opSuffix='compactLayout').Set(Gf.Vec3f(self.scale, self.scale, 1.))
