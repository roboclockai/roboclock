"""Original licensed valve on a passive quarter-turn joint.

The fixed housing uses a declared conservative hull because a fixed-root
articulation still requires a dynamic-compatible collision representation.
No drive or controller torque is installed on the scored lever joint.
"""
import hashlib
from pathlib import Path

import numpy as np

from ..paths import asset_path, portable_json_loads

ROOT = Path(asset_path('/opt/industry/assets/v2/task44'))

def build_passive_valve(world, position, yaw=0):
    from isaacsim.core.prims import SingleArticulation
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    manifest = portable_json_loads((ROOT / 'asset-manifest.json').read_text())
    stage = world.stage
    parent = UsdGeom.Xform.Define(stage, '/World/Valve')
    parent.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    parent.AddRotateZOp().Set(float(np.rad2deg(yaw)))
    for name in ['housing', 'handle']:
        source = ROOT / 'prepared-v1' / (name + '.usd')
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['prepared'][name]['usd_sha256']:
            raise RuntimeError('Prepared original valve changed')
        prim = UsdGeom.Xform.Define(stage, '/World/Valve/' + name).GetPrim()
        prim.GetReferences().AddReference(str(source))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if name == 'housing':
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(20)
            for mesh in Usd.PrimRange(prim):
                if mesh.IsA(UsdGeom.Mesh) and mesh.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('convexHull')
    anchor = UsdPhysics.FixedJoint.Define(stage, '/World/Valve/Anchor')
    anchor.CreateBody1Rel().SetTargets(['/World/Valve/housing'])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*map(float, position)))
    anchor.CreateLocalRot0Attr(Gf.Quatf(float(np.cos(yaw / 2)), Gf.Vec3f(0, 0, float(np.sin(yaw / 2)))))
    UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
    lever = UsdPhysics.RevoluteJoint.Define(stage, '/World/Valve/LeverJoint')
    lever.CreateBody0Rel().SetTargets(['/World/Valve/housing'])
    lever.CreateBody1Rel().SetTargets(['/World/Valve/handle'])
    lever.CreateAxisAttr('Z')
    lever.CreateLowerLimitAttr(0)
    lever.CreateUpperLimitAttr(90)
    lever.CreateCollisionEnabledAttr(False)
    if UsdPhysics.DriveAPI(lever.GetPrim(), 'angular'):
        raise RuntimeError('Scored valve joint must remain passive')
    valve = world.scene.add(SingleArticulation('/World/Valve', name='passive_valve'))
    return (valve, dict(source=manifest, position_m=list(position), yaw_rad=float(yaw), passive=True, drive_api=False, housing_collision='Declared conservative convex hull; source visual geometry unchanged', adjacent_body_collision=False, handle_mass_kg=0.1, limit_degrees=[0, 90]))
