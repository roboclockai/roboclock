"""Supplied wet-bench mesh, matching open-basin colliders and a physical tap knob."""

import json

import numpy as np

KNOB_PARTS = {
    "qingmo_444f116b_a811_4b30_9e96_d8450d6f3139",
    "qingmo_8236896c_548c_4914_81f1_d1d7f09547ec",
}
BASIN_OBJECT = "qingmo_c4a513fc_3baa_4702_a992_e6c4fc60df45"
TOP_OBJECT = "qingmo_6be0ef9a_20b1_491b_b27f_66d0e7d83bd7"


def build_station(scene, asset_root, pivot):
    """`pivot` and every coordinate below are nominal: the bench, its colliders and the
    apparatus are placed by `scene.station` (the episode's seeded station frame); the knob body
    and its joint take the world pivot."""
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    from roboclock.envs.lab.liquid_scene import CLOSE_ANGLE

    a, stage, backend = scene.a, scene.stage, scene.b
    asset = asset_root / "wet_bench.usdc"
    data = json.loads(asset.with_suffix(".json").read_text())
    basin = next(p for p in data["parts"] if p["source_object"] == BASIN_OBJECT)
    center = (np.array(basin["min"]) + basin["max"]) / 2
    top = max(p["max"][2] for p in data["parts"] if p["source_object"] == TOP_OBJECT)
    scale = np.array([0.94, -1.0, 0.83333333])
    position = np.array([0.46, 0.21, scene.z]) - [
        center[0] * scale[0],
        center[1] * scale[1],
        top * scale[2],
    ]
    root = UsdGeom.Xform.Define(stage, "/World/WetBench")
    root.GetPrim().GetReferences().AddReference(str(asset))
    from roboclock.envs.lab.station_frame import place

    place(scene.station, root)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    root.AddScaleOp().Set(Gf.Vec3f(*scale))
    chrome = a.material("TapChrome", color=[0.65, 0.69, 0.72], roughness=0.13, metallic=0.96)
    originals = []
    for part in data["parts"]:
        path = "/World/WetBench" + part["prim"].removeprefix("/Lab")
        prim = stage.GetPrimAtPath(path)
        key, obj = part["material"], part["source_object"]
        material = None
        if obj == BASIN_OBJECT or "9804af0e" in key:
            material = a.material("steel")
        elif "6bacb72e" in key or "2c5a5d01" in key:
            material = chrome
        elif obj == TOP_OBJECT:
            material = a.material("worktop")
        elif any(t in key for t in ["ebd0a63a", "c86261f0"]):
            material = a.material("LabCabinetEnamel")
        elif "f92e884a" in key:
            material = a.material("rubber")
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material)
        if obj in KNOB_PARTS:
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            originals.append((part, bound))
            UsdGeom.Imageable(prim).MakeInvisible()

    # Countertop collision strips leave the imported basin genuinely open.
    transformed = np.array([basin["min"], basin["max"]]) * scale + position
    low, high = transformed.min(0), transformed.max(0)
    extent = np.array(data["bounds_m"]) * scale + position
    left, right = extent[:, 0].min(), extent[:, 0].max()
    back, front = extent[:, 1].min(), extent[:, 1].max()
    strips = [
        (
            "Left",
            [(left + low[0]) / 2, (back + front) / 2, scene.z - 0.018],
            [low[0] - left, front - back, 0.036],
        ),
        (
            "Right",
            [(right + high[0]) / 2, (back + front) / 2, scene.z - 0.018],
            [right - high[0], front - back, 0.036],
        ),
        (
            "Back",
            [(low[0] + high[0]) / 2, (back + low[1]) / 2, scene.z - 0.018],
            [high[0] - low[0], low[1] - back, 0.036],
        ),
        (
            "Front",
            [(low[0] + high[0]) / 2, (front + high[1]) / 2, scene.z - 0.018],
            [high[0] - low[0], front - high[1], 0.036],
        ),
    ]
    for name, pos, size in strips:
        collider = a.cube("/World/WetCollision/Counter" + name, pos, size, "steel", True)
        UsdGeom.Imageable(collider.prim).MakeInvisible()
        UsdGeom.Imageable(stage.GetPrimAtPath(collider.prim_path + "_Visual")).MakeInvisible()
    floor = low[2] + 0.006
    for name, pos, size in [
        (
            "Floor",
            [0.46, 0.21, floor - 0.005],
            [high[0] - low[0] - 0.01, high[1] - low[1] - 0.01, 0.01],
        ),
        (
            "Left",
            [low[0] + 0.005, 0.21, (floor + scene.z) / 2],
            [0.01, high[1] - low[1], scene.z - floor],
        ),
        (
            "Right",
            [high[0] - 0.005, 0.21, (floor + scene.z) / 2],
            [0.01, high[1] - low[1], scene.z - floor],
        ),
        (
            "Back",
            [0.46, low[1] + 0.005, (floor + scene.z) / 2],
            [high[0] - low[0], 0.01, scene.z - floor],
        ),
        (
            "Front",
            [0.46, high[1] - 0.005, (floor + scene.z) / 2],
            [high[0] - low[0], 0.01, scene.z - floor],
        ),
    ]:
        collider = a.cube("/World/WetCollision/Basin" + name, pos, size, "steel", True)
        UsdGeom.Imageable(collider.prim).MakeInvisible()
        UsdGeom.Imageable(stage.GetPrimAtPath(collider.prim_path + "_Visual")).MakeInvisible()

    # The original cold-water cap is the moving visual, attached to a passive joint.
    body_root = "/World/Objects/sample_0"
    body_prim = UsdGeom.Xform.Define(stage, body_root).GetPrim()
    source_frame = UsdGeom.Xform.Define(stage, body_root + "/SourceKnob")
    source_frame.AddTranslateOp().Set(Gf.Vec3d(*(position - np.array(pivot))))
    source_frame.AddScaleOp().Set(Gf.Vec3f(*scale))
    for i, (part, material) in enumerate(originals):
        mesh = UsdGeom.Mesh.Define(stage, str(source_frame.GetPath()) + f"/Part{i}")
        mesh.GetPrim().GetReferences().AddReference(str(asset), part["prim"])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    col = scene.cylinder(body_root + "/KnobCollider", [0, 0, 0], 0.029, 0.05, chrome, True)
    UsdGeom.Imageable(col).MakeInvisible()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr(0.06)
    friction = UsdPhysics.MaterialAPI.Apply(
        stage.DefinePrim("/World/Materials/TapGrip", "Material")
    )
    friction.CreateStaticFrictionAttr(1.4)
    friction.CreateDynamicFrictionAttr(1.2)
    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(
        UsdShade.Material(friction.GetPrim()), materialPurpose="physics"
    )
    world_pivot = scene.station.point(pivot)
    body = backend.world.scene.add(
        SingleRigidPrim(
            prim_path=body_root,
            name="sample_0",
            position=world_pivot,
            orientation=np.array([np.cos(CLOSE_ANGLE / 2), 0, 0, np.sin(CLOSE_ANGLE / 2)]),
        )
    )
    backend.objects = {"sample_0": body}
    scene.pivot = world_pivot
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/ValveJoint")
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body_root)])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world_pivot)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLowerLimitAttr(-2.0)
    joint.CreateUpperLimitAttr(float(np.degrees(CLOSE_ANGLE) + 2))
    bearing = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    bearing.CreateTypeAttr("force")
    bearing.CreateStiffnessAttr(0.0)
    bearing.CreateDampingAttr(0.02)
    bearing.CreateTargetVelocityAttr(0.0)
    bearing.CreateMaxForceAttr(0.03)
    backend.control_joint = joint

    # A removable steel draining rack bridges the original basin rim. It keeps
    # the 500 mL beaker upright under the original cold-water outlet.
    rack_z = scene.z + 0.006
    for i, y in enumerate(np.linspace(-0.005, 0.135, 9)):
        a.rounded(
            f"/World/Apparatus/DrainRack/Bar{i}",
            [(low[0] + high[0]) / 2, y, rack_z],
            [high[0] - low[0], 0.003, 0.003],
            "steel",
            0.0014,
        )
    for i, x in enumerate([low[0] - 0.003, high[0] + 0.003]):
        a.rounded(
            f"/World/Apparatus/DrainRack/Rail{i}",
            [x, 0.065, rack_z],
            [0.004, 0.16, 0.004],
            "steel",
            0.0019,
        )
    rack = a.cube(
        "/World/WetCollision/DrainRack",
        [(low[0] + high[0]) / 2, 0.065, rack_z],
        [high[0] - low[0], 0.15, 0.003],
        "steel",
        True,
    )
    UsdGeom.Imageable(rack.prim).MakeInvisible()
    UsdGeom.Imageable(stage.GetPrimAtPath(rack.prim_path + "_Visual")).MakeInvisible()
    base = np.array([0.286, 0.050, rack_z + 0.002])
    scene.vessel("/World/Apparatus/Beaker", base, 0.043, 0.100, "collection")
    height = scene.task.volume / 1e6 / (np.pi * 0.0415**2)
    scene.fluid("Water", base + [0, 0, 0.004 + height / 2], 0.0415, height, [0.97, 0.99, 1.0])
    # The source tap's outlet is directly above the basin; the stream ends at water.
    nozzle_z = 0.9010
    nozzle_xy = np.array([0.2864, 0.0178])
    surface_z = base[2] + 0.004 + height
    scene.fluid(
        "Stream",
        [*nozzle_xy, (nozzle_z + surface_z) / 2],
        0.0007,
        nozzle_z - surface_z,
        [0.97, 0.99, 1.0],
    )
    scene.water_geometry = {
        "beaker_base": base,
        "nozzle_z": nozzle_z,
        "nozzle_xy": nozzle_xy,
        # The drawn stream radius follows flow = pi r^2 v at the falling jet's
        # mean speed (about 1.2 m/s over this drop): ~1.6 mm at 10 mL/s.
        "stream_radius": 0.0007,
        "jet_speed": 1.2,
        "basin_low": low,
        "basin_high": high,
    }


def closeup_camera(backend):
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    from roboclock.envs.lab.station_frame import station_of

    # water_geometry is nominal (local to the station roots); the camera is in the world.
    base = station_of(backend.task).point(backend.liquid_scene.water_geometry["beaker_base"])
    target = base + [0.0, 0.0, 0.135]
    position = target + [0.28, 0.52, 0.36]
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    backend.camera.set_world_pose(
        position, gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat()), camera_axes="usd"
    )
    backend.camera.set_focal_length(0.038)
