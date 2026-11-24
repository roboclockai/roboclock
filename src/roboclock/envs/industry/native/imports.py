"""Reference existing fixture geometry/materials under a normalized rigid root."""

import numpy as np


def reference_fixture(backend, path, uri, center, size=None, orientation=None, collision=True):
    from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = backend.stage
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    content = UsdGeom.Xform.Define(stage, path + "/Asset").GetPrim()
    content.GetReferences().AddReference(str(uri))
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    bounds = cache.ComputeWorldBound(content).ComputeAlignedRange()
    low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    if not np.all(np.isfinite([*low, *high])) or np.any(high <= low):
        raise RuntimeError(f"Fixture geometry unavailable: {uri}")
    scale = np.ones(3) if size is None else np.array(size) / (high - low)
    # Transform the outer root; a child translation cancels the authored bounding center.
    # Preserve imported transform ops and apply a parent centering transform instead.
    wrapper = UsdGeom.Xform.Define(stage, path + "/Offset")
    # Reparent by referencing the same USD into Offset/Asset, remove first temporary reference.
    stage.RemovePrim(path + "/Asset")
    content = UsdGeom.Xform.Define(stage, path + "/Offset/Asset").GetPrim()
    content.GetReferences().AddReference(str(uri))
    wrapper.AddTranslateOp().Set(Gf.Vec3d(*(-(low + high) / 2).tolist()))
    colliders = []
    for p in Usd.PrimRange(content):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if p.HasAPI(UsdPhysics.MassAPI):
            p.RemoveAPI(UsdPhysics.MassAPI)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(collision)
            if collision:
                backend._collision_settings(p)
                colliders.append(p)
    if collision:
        if not colliders:
            raise RuntimeError(f"Fixture has no authored collision shapes: {uri}")
        UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
        body = SingleRigidPrim(
            prim_path=path,
            name=path.rsplit("/", 1)[-1],
            position=np.array(center),
            orientation=orientation,
            scale=scale,
            mass=1.0,
        )
    else:
        body = SingleXFormPrim(
            prim_path=path,
            name=path.rsplit("/", 1)[-1],
            position=np.array(center),
            orientation=orientation,
            scale=scale,
        )
    return backend.world.scene.add(body), {
        "uri": str(uri),
        "source_bounds": [low.tolist(), high.tolist()],
        "scale": scale.tolist(),
        "collision": collision,
    }
