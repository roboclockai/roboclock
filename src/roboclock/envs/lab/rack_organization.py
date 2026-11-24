"""Imported pack-3 tube racks with real open holes and per-slot seating contacts."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[4] / "assets/lab"


def configure_recording_logging():
    """Avoid repeated SDK contact/interpolation warnings filling video logs; retain errors."""
    import carb.logging as logging

    for source in ("omni.physx.plugin", "isaacsim.core.simulation_manager.plugin"):
        logging.acquire_logging().set_level_threshold_for_source(
            source, logging.LogSettingBehavior.OVERRIDE, logging.LEVEL_ERROR
        )


def update_expert_grasp(backend, sample, close, state, side=0):
    """Optional demo-only physical constraint, enabled only after measured grasp contact."""
    from pxr import Gf, Sdf, UsdPhysics

    from roboclock.envs.lab.policies.rack_organization import multiply
    from roboclock.physics.geometry import rotation_matrix

    path = "/World/C1ExpertGrasp" + ("Right" if side else "")
    sample_attribute = "expert_grasp_sample_right" if side else "expert_grasp_sample"
    active = backend.stage.GetPrimAtPath(path).IsValid()
    previous = getattr(backend, sample_attribute, None)
    if active and (not close or sample != previous):
        backend.stage.RemovePrim(path)
        backend.task.event_log.append(
            {
                "time": backend.task.time,
                "type": "expert_grasp_assist_released",
                "object": previous,
                "arm": "left" if side else "right",  # side 1 = rig left
            }
        )
        active = False
    if active or not close or sample is None or not state.objects[sample].grasped:
        return
    robot = backend.secondary_robot if side else backend.robot
    position, orientation = robot.end_effector.get_world_pose()
    obj = state.objects[sample]
    local_position = rotation_matrix(orientation).T @ (obj.position - position)
    local_orientation = multiply(orientation * [1, -1, -1, -1], obj.orientation)
    joint = UsdPhysics.FixedJoint.Define(backend.stage, path)
    root = "/World/ARXRight" if side else "/World/ARXLeft"
    joint.CreateBody0Rel().SetTargets([Sdf.Path(root + "/link6")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/Objects/{sample}")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, local_position)))
    joint.CreateLocalRot0Attr(
        Gf.Quatf(float(local_orientation[0]), Gf.Vec3f(*map(float, local_orientation[1:])))
    )
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateExcludeFromArticulationAttr(True)
    setattr(backend, sample_attribute, sample)
    backend.task.event_log.append(
        {
            "time": backend.task.time,
            "type": "expert_grasp_assist_enabled",
            "object": sample,
            "arm": "left" if side else "right",  # side 1 = rig left
        }
    )


@lru_cache(maxsize=1)
def rack_spec():
    return json.loads((ASSETS / "test_tube_rack.json").read_text())


# Rack layout for the common side-by-side rig (rig origin (0, 0), yaw 0: internal
# side 0 (/World/ARXLeft, the rig's RIGHT arm) base at y=-0.30, side 1 (the rig's
# LEFT arm) base at y=+0.30, both at x=0 facing +x). Every slot is within the C1
# reach (0.475 m horizontally at the 0.30 m transit height, 0.15 m minimum) of the
# arm that uses it:
# - side 0 (rig right, -y): red rack between two source racks (rows along x);
# - side 1 (rig left, +y): green rack between two source racks, mirroring side 0;
# - blue rack between the arms, turned to run along x, reachable by both.
# Source tubes are placed so that red ones sit on the -y side and green ones on
# the +y side (blue anywhere), so every tube has an arm that reaches it and its rack.
# Each entry: (center xy, turned 90 degrees about z).
COLOR_RACKS = (((0.305, -0.30), False), ((0.185, 0.0), True), ((0.305, 0.30), False))
SOURCE_RACKS = (
    ((0.195, -0.30), False),
    ((0.415, -0.30), False),
    ((0.195, 0.30), False),
    ((0.415, 0.30), False),
)
LEFT_SOURCE_SLOTS = 24  # source racks 0 and 1 (side 0, rig right, -y); racks 2 and 3 are side 1's
SOURCE_SLOTS = 12 * len(SOURCE_RACKS)


def rack_pose(index, z, source=False, frame=None):
    """(center xyz, turned) of a color rack (0-2) or source rack (0-3).

    The constants above are nominal; `frame` (the layout's apparatus placement, a common
    per-seed shift of all seven racks) places them in the world.
    """
    xy, turned = (SOURCE_RACKS if source else COLOR_RACKS)[index]
    center = np.array([xy[0], xy[1], z])
    return (center if frame is None else frame.point(center)), turned


def rack_center(index, z, source=False, frame=None):
    return rack_pose(index, z, source, frame)[0]


def rack_extent(index, source=False):
    """Clutter keep-out half extents (xy) around a rack, matching its orientation."""
    turned = rack_pose(index, 0.0, source)[1]
    return np.array([0.205, 0.085]) if turned else np.array([0.085, 0.205])


def _slot_offset(index, turned):
    x, y = rack_spec()["slots_xy"][index % 12]
    return np.array([-y, x]) if turned else np.array([x, y])


def slot_position(index, z, source=False, frame=None):
    """Slot: rack index // 12, hole index % 12 (color racks 0-35, `source` racks 0-47)."""
    spec = rack_spec()
    center, turned = rack_pose(index // 12, z, source, frame)
    center[:2] += _slot_offset(index, turned)
    center[2] += spec["support_z"] + 0.07
    return center


def source_slots(count, rng):
    """Random source holes for tubes 0..count-1 (color = index % 3: red, blue, green).

    Red tubes go to side 0's (rig right) source racks and green ones to side 1's (rig left);
    blue ones fill the remaining holes on whichever side has more left.
    """
    if count > 36:
        raise ValueError("C1 has three 12-slot color racks: at most 36 tubes")
    order = [int(s) for s in rng.permutation(SOURCE_SLOTS)]
    sides = [[s for s in order if s < LEFT_SOURCE_SLOTS], [s for s in order if s >= LEFT_SOURCE_SLOTS]]
    slots = {}
    for i in sorted(range(count), key=lambda i: {0: 0, 2: 1, 1: 2}[i % 3]):
        color = i % 3
        # Blue tubes balance the two arms' queues (tubes per side).
        placed = [sum(s < LEFT_SOURCE_SLOTS for s in slots.values()), 0]
        placed[1] = len(slots) - placed[0]
        side = 0 if color == 0 else 1 if color == 2 else int(placed[1] < placed[0])
        if not sides[side]:
            side = 1 - side
            if color != 1 or not sides[side]:
                raise ValueError("C1 supports at most 12 tubes per color")
        slots[i] = sides[side].pop(0)
    return [slots[i] for i in range(count)]


def build_racks(backend):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    z = backend.layout.table_height
    from roboclock.envs.lab.layout import COLORS, apparatus_frame

    frame = apparatus_frame(backend.layout)
    count = len(COLOR_RACKS)
    for i in range(count + len(SOURCE_RACKS)):
        source = i >= count
        root = UsdGeom.Xform.Define(backend.stage, f"/World/TubeRacks/Rack_{i}")
        root.GetPrim().GetReferences().AddReference(str(ASSETS / "test_tube_rack.usdc"))
        center, turned = rack_pose(i - count if source else i, z, source, frame)
        root.AddTranslateOp().Set(Gf.Vec3d(*center))
        if turned:
            root.AddRotateZOp().Set(90.0)
        color = (0.45, 0.48, 0.5) if source else COLORS[i]
        material = backend.appearance.material(f"RackBand_{i}", color=np.array(color))
        # Color band and name plate on the rack end that faces the rig (head camera).
        band_offset, band_size = (
            ([-0.183, 0, 0.035], [0.003, 0.12, 0.048])
            if turned
            else ([0, -0.183 if center[1] > 0 else 0.183, 0.035], [0.12, 0.003, 0.048])
        )
        backend.appearance.cube(
            f"/World/RackBands/Band_{i}", center + band_offset, band_size, material
        )
        label = backend.appearance.label(
            f"/World/RackBands/Label_{i}",
            "MIXED" if source else backend.task.rack_colors[i].upper(),
            center + np.array(band_offset) * [185 / 183, 185 / 183, 36 / 35],
            [0.11, 0.034],
            color,
        )
        if turned:
            label.AddRotateZOp().Set(90.0)
        elif center[1] > 0:
            label.AddRotateZOp().Set(180.0)
        for prim in Usd.PrimRange(root.GetPrim()):
            if prim.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    backend.appearance.material("aluminum").material
                )
                # Floor contact is provided by the individual slot pads, so its
                # contact is unambiguously assigned to the matching target.
                if source or prim.GetName() != "Part_78998":
                    UsdPhysics.CollisionAPI.Apply(prim)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
                    backend._collision_settings(prim)


def build_slot(backend, name, target):
    from isaacsim.core.api.objects import FixedCuboid
    from pxr import UsdGeom

    from roboclock.envs.lab.layout import COLORS

    color_index = int(name.split("_")[-1]) // 12
    color = COLORS[color_index]
    z = backend.layout.table_height
    support = target.support_z - z
    position = np.array([*target.position[:2], z + support / 2])
    body = backend.world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Targets/{name}/floor",
            name=name + "_floor",
            position=position,
            scale=np.array([0.038, 0.038, support]),
        )
    )
    backend._collision_settings(body.prim)
    UsdGeom.Imageable(body.prim).MakeInvisible()
    backend.target_parts[name] = [(body, position)]
    turned = rack_pose(color_index, z)[1]
    offset = [-0.023, 0.0] if turned else [0.0, 0.023]
    label = backend.appearance.label(
        f"/World/Labels/{name}",
        backend.task.rack_colors[color_index][0].upper(),
        [target.position[0] + offset[0], target.position[1] + offset[1], target.rim_z + 0.0005],
        [0.027, 0.012],
        color,
    )
    if turned:
        label.AddRotateZOp().Set(90.0)
    label.AddRotateXOp().Set(-90.0)


def tube_visual(appearance, path, obj):
    from pxr import Usd, UsdGeom, UsdShade

    collider = appearance.cylinder_child(
        path, "Body", 0.014, 0.14, 0, appearance.material("polymer"), True
    )
    UsdGeom.Imageable(collider).MakeInvisible()
    root = UsdGeom.Xform.Define(appearance.stage, path + "/ImportedTube")
    root.GetPrim().GetReferences().AddReference(str(ASSETS / "organizing_test_tube.usdc"))
    for prim in Usd.PrimRange(root.GetPrim()):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(appearance.glass().material)
    color = appearance.material("Tube_" + obj.name, color=obj.color, roughness=0.28)
    appearance.cylinder_child(path, "ColorCap", 0.0145, 0.012, 0.066, color)
    appearance.label(
        path + "/Identity",
        ("RED", "BLUE", "GREEN")[int(obj.name.split("_")[-1]) % 3],
        [0, 0, 0.015],
        [0.036, 0.023],
        obj.color,
        cylindrical_radius=0.0144,
    )
