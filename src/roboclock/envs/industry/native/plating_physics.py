"""Static collision authoring for the approved scene, with explicit scope."""

import numpy as np

OPERATION_EYE = [0.90, -0.60, 1.90]
OPERATION_TARGET = [-0.04, 0.43, 0.97]


def add_mounts_and_collisions(scene, mounts):
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage, a = scene.stage, scene.appearance
    for side, (x, y, z) in mounts.items():
        a.rounded(
            f"/World/ARXMounts/{side}Column",
            [x, y, (z - 0.02) / 2],
            [0.075, 0.075, z - 0.02],
            "frame",
            radius=0.005,
        )
        a.rounded(
            f"/World/ARXMounts/{side}Foot", [x, y, 0.022], [0.22, 0.22, 0.04], "frame", radius=0.006
        )
        # Root is bolted via native joint. A3mm gap avoids redundant root
        # contact constraints; no passive mount-load validation is claimed.
        a.rounded(
            f"/World/ARXMounts/{side}Plate",
            [x, y, z - 0.012],
            [0.15, 0.14, 0.018],
            "steel",
            radius=0.004,
        )
    left, right = np.asarray(mounts["left"], float), np.asarray(mounts["right"], float)
    a.rounded(
        "/World/ARXMounts/Brace",
        [(left[0] + right[0]) / 2, left[1], 0.55],
        [abs(right[0] - left[0]), 0.055, 0.065],
        "frame",
        radius=0.004,
    )
    prefixes = (
        "/World/Building/",
        "/World/Station/",
        "/World/Extraction/",
        "/World/Electrical/",
        "/World/ARXMounts/",
    )
    records = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prefixes) or "/Liquid" in path:
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(0.0007)
        api.CreateRestOffsetAttr(0)
        records.append(path)
    if not any("/Bath0/" in p for p in records):
        raise RuntimeError("No plating bath collision mesh")
    if not any("/Bath1/" in p for p in records):
        raise RuntimeError("No rinse bath collision mesh")
    # Apply physical contact reports to robot bodies in DualARX; static mesh
    # callbacks are delivered through those reporting bodies as well.
    return dict(
        static_colliders=records,
        geometry="Original static render surfaces, unaltered open bath triangle meshes",
        collision_exclusions=[
            "liquid visual planes",
            "signs",
            "floor paint",
            "distant service equipment",
        ],
        limitations="No electrochemistry, liquid buoyancy/drag or passive mounting-load claim",
    )


def camera_pose(camera, eye, target):
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)
    )
    camera.set_world_pose(
        position=np.asarray(eye),
        orientation=gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat()),
        camera_axes="usd",
    )
