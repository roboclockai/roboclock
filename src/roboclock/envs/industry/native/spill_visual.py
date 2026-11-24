"""Geometry-fitted illustrative spill volume; no fluid/physics collision actors."""
import hashlib
from pathlib import Path

import numpy as np

from .paths import asset_path, portable_json_loads


class TraySpillVisual:

    def __init__(self, scene, limit_l=2.0):
        from pxr import UsdGeom, UsdShade
        path = Path(asset_path('/opt/industry/arx-x5/spill-tray-v1/tray-fill-geometry.json'))
        self.geometry = portable_json_loads(path.read_text())
        tray = Path(asset_path('/opt/industry/assets/v2/task46/prepared/tray.usd'))
        assert hashlib.sha256(tray.read_bytes()).hexdigest() == self.geometry['tray_usd_sha256']
        self.geometry_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.levels = np.asarray(self.geometry['levels_z_m'])
        self.volumes = np.asarray(self.geometry['integrated_volume_l'])
        self.rings = np.asarray(self.geometry['inner_rings_xy_m'])
        self.center = np.array([0.44, 0.32])
        self.limit_l = limit_l
        self.warning_z = float(np.interp(limit_l, self.volumes, self.levels))
        ring = self.ring_at(self.warning_z)
        mark = UsdGeom.Mesh.Define(scene.stage, '/World/SpillGraphics/MaximumLine')
        upper = np.column_stack([ring, np.full(64, self.warning_z + 0.0007)])
        lower = np.column_stack([ring, np.full(64, self.warning_z - 0.0007)])
        mark.CreatePointsAttr(np.vstack([lower, upper]).tolist())
        mark.CreateFaceVertexCountsAttr([4] * 64)
        mark.CreateFaceVertexIndicesAttr([v for i in range(64) for v in [i, (i + 1) % 64, (i + 1) % 64 + 64, i + 64]])
        mark.CreateSubdivisionSchemeAttr('none')
        mark.CreateDoubleSidedAttr(True)
        material = scene.appearance.material('spill_warning', color=[0.95, 0.035, 0.008], roughness=0.4)
        UsdShade.MaterialBindingAPI.Apply(mark.GetPrim()).Bind(material.material)
        scene.sign('SpillLimit', f'MAX {limit_l:.1f} L', [0.44, 0.177, 0.758], [0.14, 0.018])
        mesh = UsdGeom.Mesh.Define(scene.stage, '/World/SpillGraphics/EstimatedWater')
        faces = []
        for i in range(64):
            j = (i + 1) % 64
            faces.extend([[0, 1 + i, 1 + j], [65, 66 + j, 66 + i], [1 + i, 66 + i, 66 + j], [1 + i, 66 + j, 1 + j]])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        self.points = mesh.CreatePointsAttr()
        water = scene.appearance.material('spill_surface', color=[0.06, 0.42, 0.58], roughness=0.2, metallic=0.08)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(water.material)
        self.update(0)

    def ring_at(self, height):
        j = int(np.clip(np.searchsorted(self.levels, height), 1, len(self.levels) - 1))
        u = np.clip((height - self.levels[j - 1]) / (self.levels[j] - self.levels[j - 1]), 0, 1)
        return (1 - u) * self.rings[j - 1] + u * self.rings[j]

    def update(self, volume_l):
        height = float(np.interp(volume_l, self.volumes, self.levels))
        if volume_l <= 0:
            self.points.Set(np.tile([0.44, 0.32, -10.0], (130, 1)).tolist())
        else:
            ring = self.ring_at(height)
            top = np.column_stack([ring, np.full(64, height)])
            bottom = np.column_stack([self.rings[0], np.full(64, self.levels[0])])
            points = np.vstack([[*self.center, height], top, [*self.center, self.levels[0]], bottom])
            self.points.Set(points.tolist())
        return height
