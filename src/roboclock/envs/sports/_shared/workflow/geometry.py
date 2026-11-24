"""Beveled high-resolution equipment visuals over simple checked colliders."""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind
from roboclock.envs.sports.arena.geometry import rod as collision_rod


def rod(b, path, start, end, radius, material="steel", collision=False):
    from pxr import Gf, UsdGeom

    from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

    shape = collision_rod(b, path, start, end, radius, material, collision)
    UsdGeom.Imageable(shape).MakeInvisible()
    p, q = np.asarray(start, float), np.asarray(end, float)
    v = q - p
    length = np.linalg.norm(v)
    bevel = min(radius * 0.16, length * 0.16, 0.004)
    profile = [
        (0, -length / 2),
        (radius - bevel, -length / 2),
        (radius, -length / 2 + bevel),
        (radius, length / 2 - bevel),
        (radius - bevel, length / 2),
        (0, length / 2),
    ]
    mesh = author_mesh(b.stage, path + "Visual", revolved(profile, 64), (p + q) / 2)
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, v / length))).GetQuat()
    mesh.AddOrientOp().Set(Gf.Quatf(rot.GetReal(), Gf.Vec3f(rot.GetImaginary())))
    bind(b, path + "Visual", b.appearance.material(material))
    return shape
