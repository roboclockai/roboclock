"""Preserve the rail-to-tip concavity of the manufacturer's complete X5 fingers."""


def configure_finger_collisions(stage, root):
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    links = [
        p
        for p in Usd.PrimRange(root)
        if p.GetName() in ("left_link7", "left_link8", "right_link7", "right_link8")
    ]
    count = 0
    for link in links:
        # Imported collision geometry is a separate instance with a _1 suffix.
        for child in link.GetChildren():
            if child.GetName() in ("link7_1", "link8_1"):
                child.SetInstanceable(False)
                for prim in Usd.PrimRange(child):
                    if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                            "convexDecomposition"
                        )
                        api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                        api.CreateMaxConvexHullsAttr(64)
                        api.CreateHullVertexLimitAttr(64)
                        api.CreateVoxelResolutionAttr(1000000)
                        api.CreateMinThicknessAttr(0.0001)
                        api.CreateShrinkWrapAttr(True)
                        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                        collision.CreateContactOffsetAttr(0.001)
                        collision.CreateRestOffsetAttr(0)
                        count += 1
    if count != 4:
        raise ValueError(f"Expected four native finger colliders, found {count}")
    root.SetCustomDataByKey("benchmark:fingerCollision", "native-mesh-convex-decomposition-64")
