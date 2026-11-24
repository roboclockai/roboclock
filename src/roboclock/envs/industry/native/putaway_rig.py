"""Task 48 (C2) on the common Dual ARX rig: three mixed-SKU cartons into their shelf slots.

The recorded cell clamped each carton between two facing arms (yaw 0 / 180 deg) on 0.36 m
floor stands between a pallet and a full-width rack. On the common rig both arms face
+Y, 0.60 m apart, so the unchanged two-hand end clamp (tools 0.307 m apart, pitched
30 deg down) works only within a few centimetres of the rig's centre line, at 0.10-0.50 m
forward. The layout is rebuilt around that corridor; the cartons, grasp, force regulation,
hold/placement/credit rules and the 90 s budget are unchanged:

* Infeed: a lift-table magazine holds the three cartons stacked in pick order on the centre
  line, 0.21 m in front of the bases. A top-level photo-eye raises the table at 0.04 m/s
  until the topmost carton still in the magazine reaches the fixed presentation height, so
  every carton is picked at the same place. This is a scene process (``process_step``), the
  same for every policy.
* Shelf: a two-level unit on the centre line, 0.31 m in front of the bases. A-01 is the
  upper deck (two brake-kit cartons, back then front), B-01 the lower deck under it. The
  upper deck is 0.36 m wide, narrower than the 0.56 m wrist span, so the forearms pass
  beside it when the arms place B underneath.

Every pick, carry, place and empty-return waypoint was checked against the original URDF
limits (planning/rig-handoff/industry_tools/reach48_clamp.py and putaway_rig.py notes).
"""

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from . import rig
from .arx_cell_task import ContactBook, add_mounts, collide
from .putaway_throughput import (
    PutawayThroughputController,
    ThroughputArmOperation,
    resolve_putaway_throughput,
)

RIG_CENTER = [0.12, 0.70, 0.79]
MOUNTS = rig.mounts(RIG_CENTER)
X = RIG_CENTER[0]
LOWER_DECK_Z = round(RIG_CENTER[2] - 0.34, 6)  # B-01 support surface
UPPER_DECK_Z = round(LOWER_DECK_Z + 0.325, 6)  # A-01 support surface
PRESENT_TOP_Z = round(LOWER_DECK_Z + 0.26, 6)  # photo-eye height: top of the presented carton
MAGAZINE_Y = round(RIG_CENTER[1] + 0.21, 6)
SHELF_FRONT_Y = round(RIG_CENTER[1] + 0.31, 6)
SIZES = dict(A=[0.28, 0.10, 0.14], B=[0.28, 0.18, 0.26])
LABELS = dict(A="BRAKE KITS / A", B="FILTERS / B")
SLOTS = dict(A="A-01", B="B-01")


def resolve_putaway_rig(seed=0, *, layout=False, timing=False, physical=False):
    d = resolve_putaway_throughput(seed, layout=layout, timing=timing, physical=physical)
    d.pop("sha256", None)
    for key in (
        "mounts_m", "source_center_xy_m", "source_floor_m", "source_label", "destination_center_xy_m",
        "destination_bounds_xy_m", "destination_bay", "source_support_raise_m",
        "source_support_adaptation", "empty_initial_clearance_z_m", "empty_joint_waypoints_rad",
        "empty_joint_speed_rad_s", "empty_return_inward_shift_m", "empty_return_above_z_m",
        "empty_return_raise_z_m", "mounting_assumption", "scene_adaptation", "transport_strategy",
        "placement_reference_bias_scope", "return_path", "shelf_support_floor_m",
    ):
        d.pop(key, None)
    order = [p["sku"] for p in d["products"]]
    jitter = np.asarray(d["carton_xy_jitter_m"])
    seen_a = 0
    products = []
    for index, sku in enumerate(order):
        size = SIZES[sku]
        if sku == "A":
            # First A to the back of A-01, second to the front.
            y = SHELF_FRONT_Y + (0.18 if seen_a == 0 else 0.065)
            seen_a += 1
            floor = UPPER_DECK_Z
            bounds = [[round(X - 0.17, 6), round(SHELF_FRONT_Y, 6)], [round(X + 0.17, 6), round(SHELF_FRONT_Y + 0.27, 6)]]
            # Clear the upper deck by 2.5 cm before carrying in.
            lift = UPPER_DECK_Z + 0.025 - (PRESENT_TOP_Z - size[2])
            clearance = 0.03
        else:
            y = SHELF_FRONT_Y + 0.11
            floor = LOWER_DECK_Z
            bounds = [[round(X - 0.30, 6), round(SHELF_FRONT_Y, 6)], [round(X + 0.30, 6), round(SHELF_FRONT_Y + 0.26, 6)]]
            lift = 0.02
            clearance = 0.02
        products.append(
            dict(
                id=index, sku=sku, label=LABELS[sku], slot=SLOTS[sku], size_m=size,
                source_center_m=[round(float(v), 6) for v in (X + jitter[0], MAGAZINE_Y + jitter[1], PRESENT_TOP_Z - size[2] / 2)],
                source_floor_m=round(PRESENT_TOP_Z - size[2], 6),
                destination_center_xy_m=[X, round(y, 6)],
                destination_bounds_xy_m=bounds,
                destination_floor_m=round(floor, 6),
                lift_height_m=round(lift, 6),
                minimum_source_lift_m=round(max(0.012, lift - 0.025), 6),
                source_clear_x_shift_m=0.0,
                source_clear_y_m=round(float(MAGAZINE_Y + jitter[1]), 6),
                placement_tool_bias_xy_m=[0.0, 0.0],
                transport_clearance_m=clearance,
            )
        )
    d.update(
        products=products,
        rig_center_m=RIG_CENTER,
        fixed_mounts_m=MOUNTS,
        clamp_tool_yaw_deg=dict(left=0, right=180),
        # On the rig the clamp force acts across each arm's wrist instead of along its reach;
        # the recorded 15-25 N first-carton profile drove the left joint5 (5 Nm cap) back by
        # 0.15 rad. 5-18 N still holds a 0.30 kg carton (needs ~1.6 N per hand at mu 0.95).
        grasp_force_profiles=[dict(minimum_N=5.0, target_N=8.0, upper_N=12.0, carry_target_N=12.0, carry_upper_N=18.0)
                              for _ in range(3)],
        prepose_spread_m=0.02,
        prepose_rise_m=0.05,
        transport_clearance_m=0.03,
        magazine=dict(
            center_xy_m=[X, MAGAZINE_Y],
            presentation_top_z_m=PRESENT_TOP_Z,
            footprint_half_xy_m=[0.12, 0.09],
            raise_speed_m_s=0.04,
            # 0.16 m deep so the rising plate keeps 2 cm from the shelf front (a 0.20 m plate
            # caught on the lower deck at 0.43 m and stalled the lift).
            platform_size_m=[0.30, 0.16, 0.02],
            scope="Lift-table infeed: a top photo-eye raises the stack until the topmost carton "
            "still in the magazine reaches the presentation height; no product pose writes.",
        ),
        shelf=dict(
            front_y_m=SHELF_FRONT_Y,
            lower_deck_z_m=LOWER_DECK_Z,
            upper_deck_z_m=UPPER_DECK_Z,
            lower_deck_size_m=[0.62, 0.26],
            upper_deck_size_m=[0.36, 0.27],
            scope="Two-level slot unit on the rig centre line: A-01 upper deck (back/front), "
            "B-01 lower deck. The upper deck is narrower than the wrist span.",
        ),
        empty_return_back_y_m=MAGAZINE_Y + 0.04,
        # Read by the carton operation's clear_source phase; no move off the pickup line.
        source_clear_y_m=MAGAZINE_Y,
        mounting_assumption="Common Dual ARX rig (rig.py): bases 0.60 m apart facing +Y on 0.79 m "
        "fixed stands; head camera 0.04 m forward and 0.54 m above the base midpoint.",
        scene_adaptation="Common-rig layout (2026-11-03): the recorded pallet and full-width rack cannot be "
        "served by a side-by-side rig with the unchanged end clamp (centre-line corridor only). "
        "Lift-table magazine + two-level centre-line slot unit; same cartons, sizes, masses, "
        "frictions, grasp geometry, force profiles, hold/placement windows and 90 s budget.",
        return_path="Back out of the shelf at the release height to the magazine, then down to the "
        "next carton's pre-grasp once the magazine has presented it.",
    )
    d["sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


class RigPutawayScene:
    title = "A/B CARTONS / MATCH SKU TO SHELF LOCATION"

    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .layout_review_revisions import build_layout_review
        from .remaining_environments import BOX
        from .v2.convex_partition import partition_convex
        from .v2.geometry import body_envelope_points
        from .v2.kinematics import UrdfArm

        self.base = build_layout_review(48, output, (1280, 800))
        c = self.base.cell
        self.world, self.stage = c.world, c.stage
        self.world.set_simulation_dt(physics_dt=1 / resolved["physics_hz"], rendering_dt=1 / CONTROL_HZ)
        self.resolved, self.events = resolved, {}
        self.success = self.physical_completed = self.manipulation_success = False
        self.placed = set()
        self.cue_state = "INFEED"
        products = resolved["products"]
        magazine = resolved["magazine"]
        shelf = resolved["shelf"]
        material = PhysicsMaterial("/World/CartonContact", static_friction=0.95, dynamic_friction=0.85, restitution=0)

        # Shelf unit (static colliders under /World/RigShelf).
        front, lower, upper = shelf["front_y_m"], shelf["lower_deck_z_m"], shelf["upper_deck_z_m"]
        (lw, ld), (uw, ud) = shelf["lower_deck_size_m"], shelf["upper_deck_size_m"]
        c.box("RigShelf/LowerDeck", [X, front + ld / 2, lower - 0.0125], [lw, ld, 0.025], "steel")
        c.box("RigShelf/Cabinet", [X, front + ld / 2, (lower - 0.025) / 2], [lw, ld, lower - 0.025], "frame")
        c.box("RigShelf/UpperDeck", [X, front + ud / 2, upper - 0.0125], [uw, ud, 0.025], "steel")
        for i, dx in enumerate([-0.165, 0.165]):
            c.box(f"RigShelf/Post{i}", [X + dx, front + ud - 0.015, (lower + upper - 0.025) / 2],
                  [0.03, 0.03, upper - 0.025 - lower], "frame")
        c.box("RigShelf/Back", [X, front + ld + 0.006, (lower + upper) / 2], [uw, 0.012, upper - lower], "frame")
        # Slot identifiers on the deck fronts (kept by MinimalContext as A-01 / B-01).
        c.sign("Slot0", "A-01", [X, front - 0.002, upper - 0.0125], [0.30, 0.024], background_rgb=(31, 94, 55))
        c.sign("Slot1", "B-01", [X, front - 0.002, lower - 0.0125], [0.30, 0.024], background_rgb=(30, 65, 122))

        # Lift-table magazine: fixed base + prismatic platform with a position drive.
        heights = [p["size_m"][2] for p in products]
        self.platform_top0 = magazine["presentation_top_z_m"] - sum(heights) - 0.002 * len(heights)
        mx, my = magazine["center_xy_m"]
        px, py, pz = magazine["platform_size_m"]
        c.box("Magazine/Column", [mx, my, (self.platform_top0 - 0.03) / 2], [0.10, 0.10, self.platform_top0 - 0.03], "frame")
        anchor_at = np.array([mx, my, self.platform_top0 - pz / 2])
        UsdGeom.Xform.Define(self.stage, "/World/Magazine/Lift").AddTranslateOp().Set(Gf.Vec3d(*anchor_at))
        for name, mass in [("base", 5.0), ("platform", 1.0)]:
            prim = UsdGeom.Xform.Define(self.stage, "/World/Magazine/Lift/" + name).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        plate = UsdGeom.Cube.Define(self.stage, "/World/Magazine/Lift/platform/Plate")
        plate.CreateSizeAttr(1.0)
        plate.AddScaleOp().Set(Gf.Vec3f(px, py, pz))
        UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(c.appearance.material("steel").material)
        UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(material.material, materialPurpose="physics")
        ram = UsdGeom.Cube.Define(self.stage, "/World/Magazine/Lift/platform/Ram")
        ram.CreateSizeAttr(1.0)
        ram.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.16))
        ram.AddScaleOp().Set(Gf.Vec3f(0.06, 0.06, 0.30))
        UsdShade.MaterialBindingAPI.Apply(ram.GetPrim()).Bind(c.appearance.material("steel").material)
        fixed = UsdPhysics.FixedJoint.Define(self.stage, "/World/Magazine/Lift/Anchor")
        fixed.CreateBody1Rel().SetTargets(["/World/Magazine/Lift/base"])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*anchor_at))
        UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
        joint = UsdPhysics.PrismaticJoint.Define(self.stage, "/World/Magazine/Lift/LiftJoint")
        joint.CreateBody0Rel().SetTargets(["/World/Magazine/Lift/base"])
        joint.CreateBody1Rel().SetTargets(["/World/Magazine/Lift/platform"])
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(0)
        joint.CreateUpperLimitAttr(0.40)
        joint.CreateCollisionEnabledAttr(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(20000)
        drive.CreateDampingAttr(2000)
        drive.CreateMaxForceAttr(500)
        self.lift = self.world.scene.add(SingleArticulation("/World/Magazine/Lift", name="magazine_lift"))
        self.lift_target = 0.0
        c.sign("MagazineID", "INFEED / LIFT TABLE", [mx, my - 0.105, 0.30], [0.26, 0.05])

        # Cartons: the recorded carton asset and native convex decomposition at the SKU sizes,
        # stacked in pick order (first pick on top).
        stack_z = self.platform_top0 + 0.002
        centers = {}
        for index in reversed(range(len(products))):
            h = products[index]["size_m"][2]
            centers[index] = np.array([*products[index]["source_center_m"][:2], stack_z + h / 2])
            stack_z += h + 0.002
        first_size = np.asarray(products[0]["size_m"])
        static = c.asset("Pilot/Carton", BOX, centers[0], first_size)
        path = static.prim_path
        self.world.scene.remove_object("Carton", registry_only=True)
        root = self.stage.GetPrimAtPath(path)
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = 0.30 * resolved["mass_scale"]
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        api.CreateEnableCCDAttr(True)
        api.CreateAngularDampingAttr(2.0)
        api.CreateSolverPositionIterationCountAttr(64)
        api.CreateSolverVelocityIterationCountAttr(16)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0)
        scale = np.array(UsdGeom.Xformable(root).GetOrderedXformOps()[-1].Get())
        original_hull = body_envelope_points(self.stage, root)
        pieces, self.collision_partition = partition_convex(original_hull)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(False)
        for i, (vertices, faces) in enumerate(pieces):
            mesh = UsdGeom.Mesh.Define(self.stage, path + f"/ContactPieces/Piece{i}")
            mesh.CreatePointsAttr(vertices.tolist())
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim()).CreateHullVertexLimitAttr(64)
            col = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            col.CreateContactOffsetAttr(0.0003)
            col.CreateRestOffsetAttr(0)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material.material, materialPurpose="physics")
        layer = self.stage.GetRootLayer()
        self.boxes, self.scales, self.envelopes, self.initials = [], [], [], []
        for index, product in enumerate(products):
            size = np.asarray(product["size_m"])
            if index == 0:
                body_path = path
            else:
                body_path = f"/World/Pilot/Carton{index}"
                Sdf.CopySpec(layer, path, layer, body_path)
            body = self.world.scene.add(SingleRigidPrim(body_path, name=f"pilot_carton{index}" if index else "pilot_carton", mass=mass))
            product_scale = scale * size / first_size
            if index:
                body.set_local_scale(product_scale)
            body.set_world_pose(position=centers[index], orientation=np.array([1.0, 0, 0, 0]))
            body.set_default_state(position=centers[index], orientation=np.array([1.0, 0, 0, 0]))
            self.boxes.append(body)
            self.scales.append(product_scale)
            self.envelopes.append(original_hull * product_scale)
            self.initials.append(centers[index])
            # SKU print on the carton face toward the robot (inside the free rigid root).
            face_offset = np.array([0, -size[1] / 2 - 0.0004, 0])
            name = f"ProductSKU{index}"
            c.sign(name, product["label"], centers[index] + face_offset, [0.26, min(0.075, size[2] * 0.5)],
                   background_rgb=(31, 94, 55) if product["sku"] == "A" else (30, 65, 122))
            Sdf.CopySpec(layer, "/World/Signs/" + name, layer, body_path + "/SKU")
            label = UsdGeom.Xformable(self.stage.GetPrimAtPath(body_path + "/SKU"))
            label.ClearXformOpOrder()
            label.AddTranslateOp().Set(Gf.Vec3d(*(face_offset / product_scale)))
            label.AddScaleOp().Set(Gf.Vec3f(*(1 / product_scale)))
            self.stage.RemovePrim("/World/Signs/" + name)

        self.static_colliders = collide(self.stage, ["/World/RigShelf/", "/World/Inbound/", "/World/Shelves/"], material)
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS)
        self.velocity_limit_overrides = []
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith("/World/ARX/") and prim.IsA(UsdPhysics.RevoluteJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(prim)
                original = float(api.GetMaxJointVelocityAttr().Get())
                limit = float(np.rad2deg(resolved["runtime_joint_velocity_limit_rad_s"]))
                assert limit < original
                api.CreateMaxJointVelocityAttr(limit)
                self.velocity_limit_overrides.append(dict(joint=str(prim.GetPath()), imported_deg_s=original, configured_deg_s=limit))
            elif str(prim.GetPath()).startswith("/World/ARX/") and prim.IsA(UsdPhysics.PrismaticJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(prim)
                imported = float(api.GetMaxJointVelocityAttr().Get())
                limit = resolved["runtime_finger_velocity_limit_m_s"]
                assert 0 < limit < imported <= 0.04
                api.CreateMaxJointVelocityAttr(limit)
                self.velocity_limit_overrides.append(dict(joint=str(prim.GetPath()), imported_m_s=imported, configured_m_s=limit))
        add_mounts(c, MOUNTS, foot_size=[0.14, 0.16, 0.04])
        self.static_colliders += collide(self.stage, ["/World/ARXMounts/"], material)
        for side in ["left", "right"]:
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/ARX/" + side)):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, materialPurpose="physics")
        self.contacts = ContactBook(self.world, [b.prim_path for b in self.boxes])
        self.world.reset()
        self.dual.initialize()
        for body in self.boxes:
            body.initialize()
        self.lift.initialize()
        self.lift.get_articulation_controller().set_gains(kps=np.array([20000.0]), kds=np.array([2000.0]))
        self.activate(0)
        # Reset posture: both hands at the first carton's pre-grasp (IK, original limits).
        kin = UrdfArm(self.dual.manifest["derived_urdf"], "link6", self.dual.manifest["tool"]["offset_m"],
                      np.eye(3), [f"joint{i}" for i in range(1, 7)])
        self.initial_postures = {}
        pitch = Rotation.from_euler("y", resolved["side_grasp_pitch_deg"], degrees=True).as_matrix()
        for side, sign in [("left", -1), ("right", 1)]:
            bp, bR = self.dual.body_world(side, "base_link")
            R = Rotation.from_euler("z", resolved["clamp_tool_yaw_deg"][side], degrees=True).as_matrix() @ pitch
            goal = self.initial + [sign * (resolved["side_tool_offset_x_m"] + resolved["prepose_spread_m"]), 0,
                                   resolved["side_tool_offset_z_m"] + resolved["prepose_rise_m"]]
            profile = self.dual.manifest["profile"]
            q, error = kin.solve(bR.T @ (goal - bp), bR.T @ R, dict(zip(self.dual.names[side], self.dual.commands[side])),
                                 profile["arm_lower_rad"], profile["arm_upper_rad"], restarts=8)
            if error > 0.001:
                raise RuntimeError("Reset pre-grasp unreachable: " + str((side, error)))
            initial = self.dual.commands[side].copy()
            for i, name in enumerate(kin.arm_names):
                initial[self.dual.names[side].index(name)] = q[i]
            for name in ["joint7", "joint8"]:
                initial[self.dual.names[side].index(name)] = 0.002
            robot = self.dual.robots[side]
            robot.set_joints_default_state(positions=initial, velocities=np.zeros(8))
            robot.set_joint_positions(initial)
            robot.set_joint_velocities(np.zeros(8))
            self.dual.commands[side] = initial.copy()
            self.initial_postures[side] = dict(q=initial.tolist(), tool_goal_m=goal.tolist())
        for data in [self.contacts.interval, self.contacts.pairs, self.contacts.minimum, self.contacts.active, self.contacts.unexpected]:
            data.clear()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=48)
        self.asset_manifest = dict(
            task_id=48,
            original_carton=str(BOX),
            original_carton_sha256=hashlib.sha256(BOX.read_bytes()).hexdigest(),
            approved_layout=self.base.metadata,
            static_colliders=self.static_colliders,
            collision_partition=self.collision_partition,
            velocity_limit_overrides=self.velocity_limit_overrides,
            initial_postures=self.initial_postures,
            fixed_mounts_m=MOUNTS,
            products=products,
            product_roots=[b.prim_path for b in self.boxes],
            reset_product_poses={str(i): v.tolist() for i, v in enumerate(self.initials)},
            magazine=resolved["magazine"],
            shelf=resolved["shelf"],
            mounting_assumption=resolved["mounting_assumption"],
            adaptations=resolved["scene_adaptation"],
            scope=resolved["scope"],
            physical_assumptions=resolved["assumptions"],
        )

    # ------------------------------------------------------------------ process
    def magazine_top(self):
        """Highest carton top inside the magazine footprint (the photo-eye's reading)."""
        m = self.resolved["magazine"]
        center, half = np.asarray(m["center_xy_m"]), np.asarray(m["footprint_half_xy_m"])
        top = None
        for index, body in enumerate(self.boxes):
            p, q = body.get_world_pose()
            if np.all(np.abs(np.asarray(p[:2]) - center) <= half):
                R = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
                z = float((self.envelopes[index] @ R.T + p)[:, 2].max())
                top = z if top is None else max(top, z)
        return top

    def magazine_ready(self):
        top = self.magazine_top()
        return top is not None and top >= self.resolved["magazine"]["presentation_top_z_m"] - 0.003

    def process_step(self, o):
        """Lift table: raise at constant speed while the top carton is below the photo-eye."""
        from isaacsim.core.utils.types import ArticulationAction

        m = self.resolved["magazine"]
        top = self.magazine_top()
        if top is not None and top < m["presentation_top_z_m"] - 0.0005:
            step = min(m["raise_speed_m_s"] / CONTROL_HZ, m["presentation_top_z_m"] - top)
            self.lift_target = min(0.40, self.lift_target + step)
            self.events.setdefault(f"magazine_raise_{len(self.placed)}_s", o["time_s"])
        self.lift.apply_action(ArticulationAction(joint_positions=np.array([self.lift_target])))

    # ------------------------------------------------------------------ controller hooks
    def activate(self, index):
        """Select observation/control targets only; no native actor-state writes."""
        self.active_index = index
        product = self.resolved["products"][index]
        self.lift_height = product["lift_height_m"]
        self.minimum_source_lift = product["minimum_source_lift_m"]
        self.box = self.boxes[index]
        self.envelope = self.envelopes[index]
        self.initial = np.asarray(product["source_center_m"], float).copy()
        self.source_floor = product["source_floor_m"]
        self.target_floor = product["destination_floor_m"]
        self.destination = np.r_[product["destination_center_xy_m"], self.target_floor + product["size_m"][2] / 2]
        self.destination_bounds = np.asarray(product["destination_bounds_xy_m"])
        self.cue_state = product["label"] + " -> " + product["slot"]

    @property
    def detail(self):
        return f"Correct locations: {len(self.placed)}/3 | A-01 brake kits / B-01 filters"

    def _shelf_state(self, body):
        return dict(
            shelf_support_N=self.contacts.force(body.prim_path, "/World/RigShelf/"),
            shelf_touching=any(
                v["active"] and body.prim_path in v["actors"] and any(a.startswith("/World/RigShelf/") for a in v["actors"])
                for v in self.contacts.active.values()
            ),
        )

    def other_product_support(self, body):
        others = {b.prim_path for b in self.boxes if b is not body}
        return sum(
            force for key, force in self.contacts.interval.items()
            if body.prim_path in key.split(" | ") and others.intersection(key.split(" | "))
        )

    def all_product_states(self):
        result = {}
        for index, body in enumerate(self.boxes):
            p, q = body.get_world_pose()
            R = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
            points = self.envelopes[index] @ R.T + p
            result[str(index)] = dict(
                position=p.tolist(), rotation=R.tolist(), bounds_min_m=points.min(0).tolist(),
                bounds_max_m=points.max(0).tolist(), linear_velocity=body.get_linear_velocity().tolist(),
                angular_velocity=body.get_angular_velocity().tolist(),
                hand_forces_N={side: self.contacts.force(body.prim_path, "/World/ARX/" + side) for side in ["left", "right"]},
                magazine_support_N=self.contacts.force(body.prim_path, "/World/Magazine/"),
                other_product_support_N=self.other_product_support(body),
                **self._shelf_state(body),
            )
        return result

    def observe(self):
        p, q = self.box.get_world_pose()
        R = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        points = self.envelope @ R.T + p
        return dict(
            time_s=float(self.world.current_time),
            tool_position=self.dual.tool_world("left")[0].tolist(),
            box_position=p.tolist(),
            box_rotation=R.tolist(),
            bounds_min_m=points.min(0).tolist(),
            bounds_max_m=points.max(0).tolist(),
            linear_velocity=self.box.get_linear_velocity().tolist(),
            angular_velocity=self.box.get_angular_velocity().tolist(),
            tools_world={side: dict(position=self.dual.tool_world(side)[0].tolist(),
                                    rotation=self.dual.tool_world(side)[1].tolist()) for side in ["left", "right"]},
            hand_forces_N={side: self.contacts.force(self.box.prim_path, "/World/ARX/" + side) for side in ["left", "right"]},
            pallet_support_N=self.contacts.force(self.box.prim_path, "/World/Magazine/"),
            source_support_N=self.contacts.force(self.box.prim_path, "/World/Magazine/") + self.other_product_support(self.box),
            **self._shelf_state(self.box),
            magazine_top_z_m=self.magazine_top(),
            magazine_lift_m=float(self.lift.get_joint_positions()[0]),
            cue_state=self.cue_state,
            unexpected_contacts=self.contacts.unexpected.copy(),
            active_product_index=self.active_index,
            placed_ids=sorted(self.placed),
            products=self.all_product_states(),
            bimanual_control=getattr(self, "controller_debug", None),
        )

    def valid_product(self, index):
        state = self.all_product_states()[str(index)]
        product = self.resolved["products"][index]
        bounds = np.asarray(product["destination_bounds_xy_m"])
        return bool(
            np.all(np.array(state["bounds_min_m"][:2]) >= bounds[0])
            and np.all(np.array(state["bounds_max_m"][:2]) <= bounds[1])
            and abs(state["bounds_min_m"][2] - product["destination_floor_m"]) < 0.001
            and (state["shelf_support_N"] > 2 or state["shelf_touching"])
            and max(state["hand_forces_N"].values()) < 0.01
            and np.linalg.norm(state["linear_velocity"]) < 0.012
            and np.linalg.norm(state["angular_velocity"]) < 0.1
        )

    def final_valid(self, o):
        return bool(len(self.placed) >= 3 and all(self.valid_product(i) for i in self.placed)
                    and o["time_s"] <= self.resolved["budget_s"] + 0.02)


class RigPutawayController(PutawayThroughputController):
    """The recorded throughput carton operation; empty return rebuilt for the rig layout."""

    def step_return(self, observation):
        s, op, t = self.s, self.operation, observation["time_s"]
        nxt = s.resolved["products"][s.active_index + 1]
        if getattr(self, "_return_plan_for", None) != s.active_index:
            self._return_plan_for = s.active_index
            self.next_prepose = {
                side: np.asarray(nxt["source_center_m"], float)
                + [sign * (s.resolved["side_tool_offset_x_m"] + s.resolved["prepose_spread_m"]), 0,
                   s.resolved["side_tool_offset_z_m"] + s.resolved["prepose_rise_m"]]
                for side, sign in [("left", -1), ("right", 1)]
            }
            self.return_back = {
                side: np.r_[self.next_prepose[side][0], s.resolved["empty_return_back_y_m"], p[2]]
                for side, p in self.return_origins.items()
            }
        if self.return_phase == "raise":  # first phase name set by the base controller
            self.phase = "empty_return:back_out_of_shelf"
            if all([self.empty_waypoint(side, self.return_back[side]) for side in ["left", "right"]]):
                self.return_phase, self.return_entered = "wait_infeed", t
        elif self.return_phase == "wait_infeed":
            self.phase = "empty_return:wait_infeed"
            for side in ["left", "right"]:
                op.m[side].move(self.return_back[side], 0.0, op.R[side])
            if s.magazine_ready():
                s.events.setdefault(f"infeed_presented_{s.active_index + 1}_s", t)
                self.return_phase, self.return_entered = "lower", t
        elif self.return_phase == "lower":
            self.phase = "empty_return:lower_to_next"
            done = [op.m[side].move(self.next_prepose[side], s.resolved["empty_return_lower_speed_m_s"], op.R[side])
                    for side in ["left", "right"]]
            if all(done):
                s.activate(s.active_index + 1)
                self.operation = ThroughputArmOperation(s, self.mode)
                self.return_phase = None
                s.events[f"empty_return_complete_{s.active_index}_s"] = t
                self.phase = f"sku{s.active_index}:preposition"
        if self.return_phase is not None and t - self.return_entered > 18:
            self.failure = "Empty return timeout " + self.return_phase
