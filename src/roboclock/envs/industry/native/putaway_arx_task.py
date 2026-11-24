"""Native docked bimanual carton-to-correct-shelf operation.

Approved warehouse layout retained. A filters carton starts on the shelf dock
load tray, then is physically carried into Bay B, released and retained. The
upstream mobile transfer is NOT simulated and earns no delivery/throughput credit.
"""

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs import robodojo_rig
from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ

from .arx_cell_task import ArmMotion, ContactBook, add_mounts, collide

MOUNTS = dict(left=[-0.20, 1.22, 0.36], right=[0.44, 1.22, 0.36])
# Seeded floor-plane offset of the carton source (layout stream); on the common rig (C2) it
# shifts the whole infeed stack, so every carton keeps its place in the stack.
CARTON_XY_JITTER_M = 0.010


def resolve_single_putaway(seed=0, *, layout=False, timing=False, physical=False):
    rs = [np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(3)]
    d = dict(
        task_id=48,
        embodiment="dual_arx_x5",
        seed=seed,
        carton_xy_jitter_m=rs[0].uniform(-CARTON_XY_JITTER_M, CARTON_XY_JITTER_M, 2).tolist()
        if layout
        else [0.0, 0.0],
        start_s=float(rs[1].uniform(1, 2)) if timing else 1.0,
        mass_scale=float(rs[2].uniform(0.9, 1.1)) if physical else 1.0,
        bounds=dict(
            carton_xy_jitter_m=[-CARTON_XY_JITTER_M, CARTON_XY_JITTER_M],
            start_s=[1, 2],
            mass_scale=[0.9, 1.1],
        ),
        randomization=dict(layout=layout, timing=timing, physical=physical),
        required_shelved_boxes=3,
        physics_hz=robodojo_rig.PHYSICS_HZ,
        side_grasp_pitch_deg=30.0,
        side_tool_offset_x_m=0.15361589976775225,
        side_tool_offset_z_m=0.022076215482045622,
        minimum_grasp_hand_N=5.0,
        target_transport_hand_N=5.0,
        lift_height_m=0.145,
        transport_clearance_m=0.055,
        minimum_source_lift_m=0.125,
        grasp_geometry="Original source finger STL/URDF at2mm closed command:30degree pitch gives13.31590mm inward reach and-2.07622mm contact height. Camera housing stays33mm behind the actual contact plane. No gripper mesh change.",
        operating_mode="fixed_shelf_dock_component",
        destination_bay="B",
        destination_center_xy_m=[0.12, 1.66],
        destination_bounds_xy_m=[[-0.30, 1.49], [0.54, 2.01]],
        shelf_support_floor_m=50.596 / 301.00006103515625 * 1.8,
        budget_s=90.0,
        runtime_joint_velocity_limit_rad_s=0.73,
        runtime_finger_velocity_limit_m_s=0.030,
        hold_validation=dict(
            dwell_s=0.8,
            position_span_m=0.0005,
            orientation_from_first_deg=0.3,
            scope="Continuous native pose window with both hands retaining contact and no pallet support; raw constraint-step velocities remain logged separately",
        ),
        scope="Shelf dock operation: original filters carton on a declared dock load tray, two opposing complete X5 hands, native grasp/lift/transport into correct original Bay B/lower/release/retention. Initial pallet is explicitly staged within fixed-arm reach for this component; upstream mobile transfer is not represented. No three-box90s or full mobile benchmark pass.",
        assumptions="Proposed rigid0.3kg filters carton; no cardboard deformation. Friction/mass are candidate parameters, not measured manufacturer data.",
    )
    d["sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


class PutawayARXScene:
    title = "BAY B / BIMANUAL SHELF PLACEMENT"

    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .layout_review_revisions import build_layout_review
        from .plating_physics import camera_pose
        from .remaining_environments import BOX, ROOT, TRAY
        from .v2.convex_partition import partition_convex
        from .v2.geometry import body_envelope_points

        self.base = build_layout_review(48, output, (1280, 800))
        c = self.base.cell
        mounts = resolved.get("mounts_m", MOUNTS)
        self.world, self.stage = c.world, c.stage
        self.world.set_simulation_dt(physics_dt=1 / resolved["physics_hz"], rendering_dt=1 / CONTROL_HZ)
        self.resolved, self.events = resolved, {}
        self.success = self.physical_completed = self.manipulation_success = False
        self.cue_state = "SHELF DOCK / MOBILE TRANSFER NOT SIMULATED"
        # Existing reusable load tray gives a continuous original flat floor;
        # bare irregular pallet slats rocked the released carton in v3.
        self.source_floor = resolved.get("source_floor_m", 0.21182544341264272)
        self.target_floor = resolved["shelf_support_floor_m"]
        self.lift_height = resolved["lift_height_m"]
        self.minimum_source_lift = resolved["minimum_source_lift_m"]
        size = np.asarray(resolved.get("products", [{"size_m": [0.28, 0.33, 0.32]}])[0]["size_m"])
        self.destination = np.r_[
            resolved["destination_center_xy_m"], self.target_floor + size[2] / 2
        ]
        self.destination_bounds = np.asarray(resolved["destination_bounds_xy_m"])
        source_xy = resolved.get("source_center_xy_m", [0.12, 0.835])
        self.initial = np.r_[source_xy, self.source_floor + size[2] / 2 + 0.002]
        self.initial[:2] += resolved["carton_xy_jitter_m"]
        c.asset(
            "Pilot/StagingPallet",
            ROOT / "task48/recovered/pallet.usd",
            [*source_xy, 0.10],
            [0.80, 0.60, 0.20],
        )
        c.asset("Pilot/LoadTray", TRAY, [*source_xy, 0.22], [0.70, 0.50, 0.040], material="steel")
        c.sign(
            "DockScope",
            resolved.get("source_label", "FILTERS B / LOCAL PICK DOCK"),
            [source_xy[0], source_xy[1] - 0.306, 0.12],
            [0.70, 0.07],
        )
        static = c.asset("Pilot/Carton", BOX, self.initial, size)
        path = static.prim_path
        # The imported wrapper already normalizes source units/dimensions.
        # Replace only the registry adapter; preserve all authored USD transforms.
        self.world.scene.remove_object("Carton", registry_only=True)
        root = self.stage.GetPrimAtPath(path)
        material = PhysicsMaterial(
            "/World/CartonContact", static_friction=0.95, dynamic_friction=0.85, restitution=0
        )
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = 0.30 * resolved["mass_scale"]
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        api.CreateEnableCCDAttr(True)
        api.CreateAngularDampingAttr(2.0)
        api.CreateSolverPositionIterationCountAttr(64)
        api.CreateSolverVelocityIterationCountAttr(16)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0)
        for mesh in Usd.PrimRange(root):
            if mesh.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull")
                hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh)
                hull.CreateHullVertexLimitAttr(255)
                hull.CreateMinThicknessAttr(0.0001)
                col = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                col.CreateContactOffsetAttr(0.0003)
                col.CreateRestOffsetAttr(0)
                UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
                    material.material, materialPurpose="physics"
                )
        scale = np.array(UsdGeom.Xformable(root).GetOrderedXformOps()[-1].Get())
        # body_envelope_points cancels the outer transform, including its scale.
        original_hull = body_envelope_points(self.stage, root)
        self.envelope = original_hull * scale
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
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim()).CreateHullVertexLimitAttr(
                64
            )
            col = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            col.CreateContactOffsetAttr(0.0003)
            col.CreateRestOffsetAttr(0)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                material.material, materialPurpose="physics"
            )
        self.box = self.world.scene.add(SingleRigidPrim(path, name="pilot_carton", mass=mass))
        self.box.set_default_state(position=self.initial, orientation=np.array([1.0, 0, 0, 0]))
        # SKU is attached visually inside the free rigid root, never world-pinned.
        c.sign("PilotSKU", "FILTERS / B", self.initial + [0, -0.1654, 0], [0.26, 0.06])
        layer = self.stage.GetRootLayer()
        Sdf.CopySpec(layer, "/World/Signs/PilotSKU", layer, path + "/SKU")
        label = UsdGeom.Xformable(self.stage.GetPrimAtPath(path + "/SKU"))
        # Label was authored in world metres before parenting into a scaled root.
        label.ClearXformOpOrder()
        label.AddTranslateOp().Set(Gf.Vec3d(*(np.array([0, -0.1654, 0]) / scale)))
        label.AddScaleOp().Set(Gf.Vec3f(*(1 / scale)))
        self.stage.RemovePrim("/World/Signs/PilotSKU")
        self.boxes = [self.box]
        self.scales = [scale.copy()]
        self.envelopes = [self.envelope.copy()]
        self.active_index = 0
        self.placed = set()
        self.initials = [self.initial.copy()]
        for i, product in enumerate(resolved.get("products", [])[1:], start=1):
            clone_path = f"/World/Pilot/Carton{i}"
            Sdf.CopySpec(layer, path, layer, clone_path)
            center = np.asarray(product["source_center_m"], float)
            body = self.world.scene.add(
                SingleRigidPrim(clone_path, name=f"pilot_carton{i}", mass=mass)
            )
            product_scale = scale * np.asarray(product["size_m"]) / size
            body.set_local_scale(product_scale)
            body.set_world_pose(position=center, orientation=np.array([1.0, 0, 0, 0]))
            body.set_default_state(position=center, orientation=np.array([1.0, 0, 0, 0]))
            self.boxes.append(body)
            self.scales.append(product_scale)
            self.envelopes.append(original_hull * product_scale)
            self.initials.append(center)
        self.static_colliders = collide(
            self.stage,
            [
                "/World/Pilot/StagingPallet",
                "/World/Pilot/LoadTray",
                "/World/Inbound/",
                "/World/Shelves/",
            ],
            material,
        )
        self.dual = DualARX(
            self.world, manifest, mounts=mounts, mount_yaw_deg=dict(left=0, right=180)
        )
        self.velocity_limit_overrides = []
        for joint in self.stage.Traverse():
            if str(joint.GetPath()).startswith("/World/ARX/") and joint.IsA(
                UsdPhysics.RevoluteJoint
            ):
                api = PhysxSchema.PhysxJointAPI.Apply(joint)
                original = float(api.GetMaxJointVelocityAttr().Get())
                assert abs(original - np.rad2deg(0.75)) < 0.001
                limit = float(np.rad2deg(resolved["runtime_joint_velocity_limit_rad_s"]))
                assert limit < original
                api.CreateMaxJointVelocityAttr(limit)
                self.velocity_limit_overrides.append(
                    dict(
                        joint=str(joint.GetPath()), imported_deg_s=original, configured_deg_s=limit
                    )
                )
            elif str(joint.GetPath()).startswith("/World/ARX/") and joint.IsA(
                UsdPhysics.PrismaticJoint
            ):
                api = PhysxSchema.PhysxJointAPI.Apply(joint)
                imported = float(api.GetMaxJointVelocityAttr().Get())
                limit = resolved["runtime_finger_velocity_limit_m_s"]
                assert 0 < limit < imported <= 0.04
                api.CreateMaxJointVelocityAttr(limit)
                self.velocity_limit_overrides.append(
                    dict(joint=str(joint.GetPath()), imported_m_s=imported, configured_m_s=limit)
                )
        add_mounts(c, mounts, foot_size=[0.14, 0.16, 0.04])
        self.static_colliders += collide(self.stage, ["/World/ARXMounts/"], material)
        for side in ["left", "right"]:
            for p in Usd.PrimRange(self.stage.GetPrimAtPath("/World/ARX/" + side)):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(
                        material.material, materialPurpose="physics"
                    )
        if resolved.get("products"):
            self.configure_multisku(c, scale)
        self.contacts = ContactBook(self.world, [b.prim_path for b in self.boxes])
        self.world.reset()
        self.dual.initialize()
        for body in self.boxes:
            body.initialize()
        # Collision-free declared initial posture at the fixed dock only.
        # Every product remains free; no body or robot reset after time zero.
        from .v2.kinematics import UrdfArm

        kin = UrdfArm(
            self.dual.manifest["derived_urdf"],
            "link6",
            self.dual.manifest["tool"]["offset_m"],
            np.eye(3),
            [f"joint{i}" for i in range(1, 7)],
        )
        initial_postures = {}
        for side, sign in [("left", -1), ("right", 1)]:
            bp, bR = self.dual.body_world(side, "base_link")
            initial_R = (
                self.dual.rotations[side]
                @ Rotation.from_euler(
                    "y", resolved["side_grasp_pitch_deg"], degrees=True
                ).as_matrix()
            )
            goal = self.initial + [
                sign * (resolved["side_tool_offset_x_m"] + 0.06),
                0,
                resolved["side_tool_offset_z_m"] + 0.02,
            ]
            profile = self.dual.manifest["profile"]
            q, error = kin.solve(
                bR.T @ (goal - bp),
                bR.T @ initial_R,
                dict(zip(self.dual.names[side], self.dual.commands[side])),
                profile["arm_lower_rad"],
                profile["arm_upper_rad"],
                restarts=8,
            )
            if error > 0.001:
                raise RuntimeError("Dock reset posture unreachable: " + str((side, error)))
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
            initial_postures[side] = dict(
                q=initial.tolist(),
                tool_goal_m=goal.tolist(),
                scope="Reset initial condition only; ideal fixed dock stands, no mobile motion",
            )
        for data in [
            self.contacts.interval,
            self.contacts.pairs,
            self.contacts.minimum,
            self.contacts.active,
            self.contacts.unexpected,
        ]:
            data.clear()
        self.camera = c.camera
        self.camera.initialize()
        self.camera.set_focal_length(0.028)
        self.camera.set_horizontal_aperture(0.036)
        self.camera.set_clipping_range(0.01, 30)
        camera_pose(self.camera, [-1.2, -0.3, 1.65], [0.12, 1.30, 0.55])
        self.asset_manifest = dict(
            task_id=48,
            original_carton=str(BOX),
            original_carton_sha256=hashlib.sha256(BOX.read_bytes()).hexdigest(),
            original_pallet=str(ROOT / "task48/recovered/pallet.usd"),
            adaptations="Approved warehouse scene and original280x330x320mm carton retained. Existing pallet and load tray represent a declared reachable local pick dock; six upstream cartons remain scenery. Complete original arms fixed on test pedestals. Original Bay B triangle collisions and native released support mandatory. No source return completion credit, no mobile-base motion.",
            source_load_tray=str(TRAY),
            source_load_tray_sha256=hashlib.sha256(TRAY.read_bytes()).hexdigest(),
            source_floor_z_m=self.source_floor,
            target_floor_z_m=self.target_floor,
            target_center_m=self.destination.tolist(),
            target_support_prefix="/World/Shelves/Bay1",
            approved_layout=self.base.metadata,
            static_colliders=self.static_colliders,
            collision_partition=self.collision_partition,
            velocity_limit_overrides=self.velocity_limit_overrides,
            initial_postures=initial_postures,
            mount_yaw_deg=dict(left=0, right=180),
            fixed_mounts_m=mounts,
            mounting_assumption="Complete original arms on360mm fixed floor stands,640mm spacing atY1.22m,140x160mm bolted feet clear the actual pallet and intact rack. Existing pallet reused at800x600mm in the local pick dock; original reusable load tray gives continuous support.145mm source lift and55mm shelf clearance retain the unchanged original carton; no rack cuts/disabled contacts/mobile motion.",
            scope=resolved["scope"],
            physical_assumptions=resolved["assumptions"],
        )
        if resolved.get("products"):
            self.asset_manifest.update(
                products=resolved["products"],
                product_roots=[b.prim_path for b in self.boxes],
                reset_product_poses={
                    str(i): b.get_world_pose()[0].tolist() for i, b in enumerate(self.boxes)
                },
                normalization_scales=[x.tolist() for x in self.scales],
                mounting_assumption=resolved["mounting_assumption"],
                adaptations=resolved["scene_adaptation"],
            )

    def configure_multisku(self, cell, scale):
        """Reset-only existing-asset composition; never move a product at runtime."""
        from pxr import Gf, Sdf, UsdGeom

        from .arx_cell_task import StatusDisplay

        self.title = "A/B CARTONS / MATCH SKU TO SHELF LOCATION"
        shelf = UsdGeom.Xformable(self.stage.GetPrimAtPath("/World/Shelves/Bay1"))
        op = shelf.GetOrderedXformOps()[-1]
        value = np.array(op.Get())
        value[1] *= 1.4 / 0.90
        op.Set(type(op.Get())(*value))
        for i, x, delta in [(0, -1.15, -0.20), (2, 1.39, 0.20)]:
            body = self.world.scene.get_object(f"Bay{i}")
            position = np.array([x, 1.75, 0.90])
            orientation = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
            body.set_world_pose(position=position, orientation=orientation)
            body.set_default_state(position=position, orientation=orientation)
            for path in [f"/World/Signs/Target{i}", f"/World/ShelfTargets/Edge{i}"]:
                UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).AddTranslateOp(
                    opSuffix="rackSpacing"
                ).Set(Gf.Vec3d(delta, 0, 0))
        for name in ["Bay0", "Bay1", "Bay2", "Target0", "Target1", "Target2"]:
            self.stage.RemovePrim("/World/Signs/" + name)
        for index, product in enumerate(self.resolved["products"]):
            center = np.array(product["source_center_m"])
            product_scale = self.scales[index]
            face_offset = np.array([0, -product["size_m"][1] / 2 - 0.0004, 0])
            name = f"ProductSKU{index}"
            label_color = (31, 94, 55) if product["sku"] == "A" else (30, 65, 122)
            cell.sign(
                name,
                product["label"],
                center + face_offset,
                [0.26, 0.075],
                background_rgb=label_color,
            )
            root = self.boxes[index].prim_path
            self.stage.RemovePrim(root + "/SKU")
            layer = self.stage.GetRootLayer()
            Sdf.CopySpec(layer, "/World/Signs/" + name, layer, root + "/SKU")
            label = UsdGeom.Xformable(self.stage.GetPrimAtPath(root + "/SKU"))
            label.ClearXformOpOrder()
            label.AddTranslateOp().Set(Gf.Vec3d(*(face_offset / product_scale)))
            label.AddScaleOp().Set(Gf.Vec3f(*(1 / product_scale)))
            self.stage.RemovePrim("/World/Signs/" + name)
            cell.sign(
                f"Slot{index}",
                product["label"] + " / " + product["slot"],
                [product["destination_center_xy_m"][0], 1.455, self.target_floor + 0.012],
                [0.36, 0.065],
                background_rgb=label_color,
            )
        cell.sign("PutawayCount", "CORRECT PUTAWAY / 0 OF 2", [0.12, 1.95, 1.10], [0.90, 0.10])
        self.count_display = StatusDisplay(
            cell, "PutawayCount", {str(i): f"CORRECT PUTAWAY / {i} OF 2" for i in range(3)}
        )
        self.activate(0)

    def activate(self, index):
        """Select observation/control targets only; no native actor-state writes."""
        self.active_index = index
        product = self.resolved["products"][index]
        self.lift_height = product.get("lift_height_m", self.resolved["lift_height_m"])
        self.minimum_source_lift = product.get(
            "minimum_source_lift_m", self.resolved["minimum_source_lift_m"]
        )
        self.box = self.boxes[index]
        self.envelope = self.envelopes[index]
        self.initial = self.initials[index].copy()
        self.source_floor = product["source_floor_m"]
        self.destination = np.r_[
            product["destination_center_xy_m"], self.target_floor + product["size_m"][2] / 2
        ]
        self.destination_bounds = np.asarray(product["destination_bounds_xy_m"])
        self.cue_state = product["label"] + " -> " + product["slot"]

    @property
    def detail(self):
        if self.resolved.get("products"):
            return f"Correct locations: {len(self.placed)}/2 | A-01 brake kits / B-01 filters | Mobile pending"
        return "FILTERS -> BAY B | Actual grasp, carry, release | Mobile transfer pending"

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
            tools_world={
                side: dict(
                    position=self.dual.tool_world(side)[0].tolist(),
                    rotation=self.dual.tool_world(side)[1].tolist(),
                )
                for side in ["left", "right"]
            },
            hand_forces_N={
                side: self.contacts.force(self.box.prim_path, "/World/ARX/" + side)
                for side in ["left", "right"]
            },
            pallet_support_N=self.contacts.force(self.box.prim_path, "/World/Pilot/StagingPallet"),
            source_support_N=self.contacts.force(self.box.prim_path, "/World/Pilot/LoadTray")
            + self.other_product_support(self.box),
            shelf_support_N=self.contacts.force(self.box.prim_path, "/World/Shelves/Bay1"),
            shelf_touching=any(
                v["active"]
                and self.box.prim_path in v["actors"]
                and any(a.startswith("/World/Shelves/Bay1/") for a in v["actors"])
                for v in self.contacts.active.values()
            ),
            cue_state=self.cue_state,
            unexpected_contacts=self.contacts.unexpected.copy(),
            active_product_index=self.active_index,
            placed_ids=sorted(self.placed),
            products=self.all_product_states() if self.resolved.get("products") else {},
        )

    def all_product_states(self):
        result = {}
        for index, body in enumerate(self.boxes):
            p, q = body.get_world_pose()
            R = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
            points = self.envelopes[index] @ R.T + p
            result[str(index)] = dict(
                position=p.tolist(),
                rotation=R.tolist(),
                bounds_min_m=points.min(0).tolist(),
                bounds_max_m=points.max(0).tolist(),
                linear_velocity=body.get_linear_velocity().tolist(),
                angular_velocity=body.get_angular_velocity().tolist(),
                hand_forces_N={
                    side: self.contacts.force(body.prim_path, "/World/ARX/" + side)
                    for side in ["left", "right"]
                },
                shelf_support_N=self.contacts.force(body.prim_path, "/World/Shelves/Bay1"),
                shelf_touching=any(
                    v["active"]
                    and body.prim_path in v["actors"]
                    and any(a.startswith("/World/Shelves/Bay1/") for a in v["actors"])
                    for v in self.contacts.active.values()
                ),
                load_tray_support_N=self.contacts.force(body.prim_path, "/World/Pilot/LoadTray"),
                other_product_support_N=self.other_product_support(body),
            )
        return result

    def other_product_support(self, body):
        # Rigid product identities require exact actor matching: Carton is a
        # substring of Carton1, unlike the deliberate arm/static prefix queries.
        others = {b.prim_path for b in self.boxes if b is not body}
        return sum(
            force
            for key, force in self.contacts.interval.items()
            if body.prim_path in key.split(" | ") and others.intersection(key.split(" | "))
        )

    def valid_product(self, index):
        state = self.all_product_states()[str(index)]
        bounds = np.asarray(self.resolved["products"][index]["destination_bounds_xy_m"])
        return bool(
            np.all(np.array(state["bounds_min_m"][:2]) >= bounds[0])
            and np.all(np.array(state["bounds_max_m"][:2]) <= bounds[1])
            and abs(state["bounds_min_m"][2] - self.target_floor) < 0.001
            and (state["shelf_support_N"] > 2 or state["shelf_touching"])
            and max(state["hand_forces_N"].values()) < 0.01
            and np.linalg.norm(state["linear_velocity"]) < 0.012
            and np.linalg.norm(state["angular_velocity"]) < 0.1
        )

    def final_valid(self, o):
        return False


class PutawayArmOperation:
    def __init__(self, scene, mode):
        self.s, self.mode = scene, mode
        self.m = {side: ArmMotion(scene.dual, side) for side in ["left", "right"]}
        for m in self.m.values():
            m.position_gain = 1.2
            m.position_correction_limit_m = 0.030
            m.orientation_feedback = True
        # The clamp tools point at each other along world X. The recorded scene derived
        # that from the facing mount yaws (0/180 deg); the common rig (both arms facing
        # +Y) declares the tool yaw explicitly as clamp_tool_yaw_deg.
        yaw = scene.resolved.get("clamp_tool_yaw_deg")
        self.R = {
            side: (
                Rotation.from_euler("z", yaw[side], degrees=True).as_matrix()
                if yaw is not None
                else scene.dual.rotations[side]
            )
            @ Rotation.from_euler(
                "y", scene.resolved["side_grasp_pitch_deg"], degrees=True
            ).as_matrix()
            for side in ["left", "right"]
        }
        self.phase, self.failure, self.complete = "preposition", None, False
        self.entered, self.since, self.press = float(scene.world.current_time), None, 0.0
        self.lower_bias = 0.0
        self.hold_samples = []
        self.retention_press, self.lost_since = 0.0, None
        self.minimum_grasp_hand_N = scene.resolved["minimum_grasp_hand_N"]
        self.target_transport_hand_N = scene.resolved["target_transport_hand_N"]
        self.empty_points = dict(
            left=scene.initial + [-0.38, 0, 0.03], right=scene.initial + [0.38, 0, 0.03]
        )
        self.turn_R = {side: self.m[side].R.copy() for side in ["left", "right"]}
        self.prepose = {}
        self.grip = {}
        for side, sign in [("left", -1), ("right", 1)]:
            self.grip[side] = scene.initial + [
                sign * scene.resolved["side_tool_offset_x_m"],
                0,
                scene.resolved["side_tool_offset_z_m"],
            ]
            spread = scene.resolved.get("prepose_spread_m", 0.06)
            rise = scene.resolved.get("prepose_rise_m", 0.02)
            self.prepose[side] = self.grip[side] + [sign * spread, 0, rise]
            m = self.m[side]
            bp, bR = scene.dual.body_world(side, "base_link")
            profile = scene.dual.manifest["profile"]
            q, e = m.kin.solve(
                bR.T @ (self.prepose[side] - bp),
                bR.T @ self.R[side],
                dict(zip(scene.dual.names[side], scene.dual.commands[side])),
                profile["arm_lower_rad"],
                profile["arm_upper_rad"],
                restarts=8,
            )
            if e > 0.004:
                raise RuntimeError("Unreachable bimanual prepose: " + str((side, e)))

    def transition(self, phase):
        self.phase = phase
        self.entered = float(self.s.world.current_time)
        self.s.events[self.key(phase + "_s")] = self.entered
        print("PHASE", phase, self.entered, flush=True)

    def key(self, name):
        return f"{self.s.active_index}_{name}" if self.s.resolved.get("products") else name

    def move_pair(self, goals, speed):
        if self.phase in ["lift", "hold", "clear_source", "transfer_to_shelf", "lower"]:
            goals = {
                side: np.asarray(goal)
                + [self.retention_press if side == "left" else -self.retention_press, 0, 0]
                for side, goal in goals.items()
            }
        done = [self.m[side].move(goals[side], speed, self.R[side]) for side in ["left", "right"]]
        return all(done)

    def step(self, o):
        if self.complete or self.failure or self.mode in ["absent", "passive"]:
            return
        s, t = self.s, o["time_s"]
        if t < s.resolved["start_s"]:
            return
        try:
            if o["unexpected_contacts"]:
                raise RuntimeError("Unexpected native robot contact")
            if self.phase in ["lift", "hold", "clear_source", "transfer_to_shelf"]:
                if min(o["hand_forces_N"].values()) < self.target_transport_hand_N:
                    self.retention_press = min(0.003, self.retention_press + 0.002 / CONTROL_HZ)
                midpoint = np.mean(
                    [s.dual.tool_world(side)[0] for side in ["left", "right"]], axis=0
                )
                retained = (
                    min(o["hand_forces_N"].values()) > 0.2
                    and np.linalg.norm(midpoint - np.array(o["box_position"]) - self.held_offset)
                    < 0.012
                )
                self.lost_since = (
                    None if retained else (t if self.lost_since is None else self.lost_since)
                )
                if self.lost_since is not None and t - self.lost_since > 0.2:
                    raise RuntimeError("Actual two-hand carton grasp lost during transport")
            for side in ["left", "right"]:
                s.dual.jaw_travel(side, 0.002)
            if self.phase == "empty_approach":
                # First translate with the existing downward tools above the
                # pallet. Direct joint interpolation to a horizontal prepose
                # dipped a finger into the original pallet in nominal-v1.
                done = [
                    self.m[side].move(self.empty_points[side], 0.10) for side in ["left", "right"]
                ]
                if all(done):
                    self.transition("rotate_empty")
            elif self.phase == "rotate_empty":
                done = []
                for side in ["left", "right"]:
                    actual_R = s.dual.tool_world(side)[1]
                    delta = Rotation.from_matrix(self.R[side] @ self.turn_R[side].T).as_rotvec()
                    if Rotation.from_matrix(self.turn_R[side] @ actual_R.T).magnitude() < 0.05:
                        self.turn_R[side] = (
                            Rotation.from_rotvec(
                                delta * min(1, 0.35 / CONTROL_HZ / max(np.linalg.norm(delta), 1e-12))
                            ).as_matrix()
                            @ self.turn_R[side]
                        )
                    arrived = self.m[side].move(self.empty_points[side], 0.035, self.turn_R[side])
                    done.append(
                        arrived
                        and Rotation.from_matrix(self.R[side] @ actual_R.T).magnitude() < 0.03
                        and np.linalg.norm(delta) < 0.005
                    )
                if all(done):
                    self.transition("preposition")
            elif self.phase == "preposition":
                if self.move_pair(self.prepose, 0.075):
                    self.transition("approach")
            elif self.phase == "approach":
                if self.move_pair(self.grip, 0.045):
                    self.transition("close")
            elif self.phase == "close":
                both = min(o["hand_forces_N"].values()) > self.minimum_grasp_hand_N
                if not both:
                    self.press = min(0.006, self.press + 0.001 / CONTROL_HZ)
                goals = {
                    side: self.grip[side] + [self.press if side == "left" else -self.press, 0, 0]
                    for side in ["left", "right"]
                }
                self.move_pair(goals, 0.015)
                self.since = (self.since if self.since is not None else t) if both else None
                if self.since is not None and t - self.since > 0.15:
                    self.held_goals = {
                        side: s.dual.tool_world(side)[0].copy() for side in ["left", "right"]
                    }
                    self.held_offset = np.mean(list(self.held_goals.values()), axis=0) - np.array(
                        o["box_position"]
                    )
                    self.grasp_body_position = np.array(o["box_position"])
                    self.transition("lift")
            elif self.phase == "lift":
                if self.move_pair(
                    {side: p + [0, 0, s.lift_height] for side, p in self.held_goals.items()},
                    0.035,
                ):
                    self.since = None
                    self.transition("hold")
            elif self.phase == "hold":
                self.move_pair(
                    {side: p + [0, 0, s.lift_height] for side, p in self.held_goals.items()},
                    0.02,
                )
                valid = (
                    o["bounds_min_m"][2] > s.source_floor + s.minimum_source_lift
                    and min(o["hand_forces_N"].values()) > 1.8
                    and o["source_support_N"] < 0.01
                )
                self.hold_samples = (
                    self.hold_samples
                    + [(t, np.array(o["box_position"]), np.array(o["box_rotation"]))]
                    if valid
                    else []
                )
                limits = s.resolved["hold_validation"]
                while (
                    len(self.hold_samples) > 1 and self.hold_samples[1][0] <= t - limits["dwell_s"]
                ):
                    self.hold_samples.pop(0)
                if self.hold_samples:
                    positions = np.array([a[1] for a in self.hold_samples])
                    rotations = np.array([a[2] for a in self.hold_samples])
                    span = float(np.linalg.norm(np.ptp(positions, axis=0)))
                    angle = float(
                        Rotation.from_matrix(rotations @ rotations[0].T).magnitude().max()
                    )
                    valid = bool(
                        t - self.hold_samples[0][0] >= limits["dwell_s"] - 1e-6
                        and span <= limits["position_span_m"]
                        and angle <= np.deg2rad(limits["orientation_from_first_deg"])
                    )
                else:
                    valid = False
                if valid:
                    s.events[self.key("hold_pose_validation")] = dict(
                        time_s=t,
                        samples=len(self.hold_samples),
                        native_position_span_m=span,
                        native_orientation_from_first_rad=angle,
                        raw_velocity_gate=False,
                        scope=limits["scope"],
                    )
                    self.transition(
                        "clear_source" if s.resolved.get("products") else "transfer_to_shelf"
                    )
            elif self.phase == "clear_source":
                if self.move_pair(
                    {
                        side: p
                        + [
                            s.resolved["products"][s.active_index].get("source_clear_x_shift_m", 0),
                            s.resolved["products"][s.active_index].get(
                                "source_clear_y_m", s.resolved["source_clear_y_m"]
                            )
                            - s.initial[1],
                            s.lift_height,
                        ]
                        for side, p in self.held_goals.items()
                    },
                    0.08,
                ):
                    self.transition("transfer_to_shelf")
            elif self.phase == "transfer_to_shelf":
                offset = (
                    s.destination
                    - self.grasp_body_position
                    + [0, 0, s.resolved["transport_clearance_m"]]
                )
                if self.move_pair({side: p + offset for side, p in self.held_goals.items()}, 0.055):
                    if o["source_support_N"] > 0.01 or min(o["hand_forces_N"].values()) < 1.0:
                        raise RuntimeError(
                            "Shelf transport lacks two-hand support or still contacts source"
                        )
                    self.shelf_goals = {
                        side: p + s.destination - self.grasp_body_position
                        for side, p in self.held_goals.items()
                    }
                    s.events[self.key("actual_shelf_transport_arrived_s")] = t
                    s.cue_state = (
                        s.resolved["products"][s.active_index]["label"]
                        + " / PLACE INTO "
                        + s.resolved["products"][s.active_index]["slot"]
                        if s.resolved.get("products")
                        else "FILTERS / PLACE INTO BAY B"
                    )
                    self.transition("lower")
            elif self.phase == "lower":
                if o["shelf_support_N"] < 1.5:
                    if o["bounds_min_m"][2] < s.target_floor - 0.001:
                        raise RuntimeError(
                            "Original carton envelope below tray floor before native support"
                        )
                    self.lower_bias = max(-0.006, self.lower_bias - 0.002 / CONTROL_HZ)
                self.move_pair(
                    {side: p + [0, 0, self.lower_bias] for side, p in self.shelf_goals.items()},
                    0.025,
                )
                if o["shelf_support_N"] > 1.5:
                    self.release_origins = {
                        side: s.dual.tool_world(side)[0].copy() for side in ["left", "right"]
                    }
                    self.transition("release")
            elif self.phase == "release":
                goals = {
                    side: p + [(-0.045 if side == "left" else 0.045), 0, 0]
                    for side, p in self.release_origins.items()
                }
                if self.move_pair(goals, 0.04):
                    self.since = None
                    self.transition("settle")
            elif self.phase == "settle":
                valid = (
                    (o["shelf_support_N"] > 2.0 or o["shelf_touching"])
                    and abs(o["bounds_min_m"][2] - s.target_floor) < 0.001
                    and np.all(np.array(o["bounds_min_m"][:2]) >= s.destination_bounds[0])
                    and np.all(np.array(o["bounds_max_m"][:2]) <= s.destination_bounds[1])
                    and max(o["hand_forces_N"].values()) < 0.01
                    and np.linalg.norm(o["linear_velocity"]) < 0.012
                    and np.linalg.norm(o["angular_velocity"]) < 0.1
                )
                self.since = (self.since if self.since is not None else t) if valid else None
                if self.since is not None and t - self.since > 0.5:
                    self.complete = True
                    s.manipulation_success = s.physical_completed = True
                    s.events[self.key("released_correct_shelf_retention_s")] = t
                    s.cue_state = (
                        s.resolved["products"][s.active_index]["slot"]
                        if s.resolved.get("products")
                        else "BAY B"
                    ) + " PLACED / MOBILE TRANSFER PENDING"
                    self.transition("complete")
            if t - self.entered > 18:
                raise RuntimeError("Bimanual timeout " + self.phase)
        except RuntimeError as exc:
            self.failure = str(exc)
            print("FAILURE", self.failure, flush=True)


def resolve_putaway(seed=0, *, layout=False, timing=False, physical=False):
    d = resolve_single_putaway(seed, layout=layout, timing=timing, physical=physical)
    jitter = np.asarray(d["carton_xy_jitter_m"])
    floor = 0.21182544341264272
    slots = {
        "A": dict(
            label="BRAKE KITS / A",
            slot="A-01",
            size_m=[0.28, 0.28, 0.14],
            destination_center_xy_m=[-0.08, 1.64],
            destination_bounds_xy_m=[[-0.25, 1.49], [0.09, 2.01]],
        ),
        "B": dict(
            label="FILTERS / B",
            slot="B-01",
            source_clear_x_shift_m=0.0,
            source_clear_y_m=1.36,
            lift_height_m=0.32,
            minimum_source_lift_m=0.29,
            size_m=[0.28, 0.33, 0.32],
            destination_center_xy_m=[0.32, 1.66],
            destination_bounds_xy_m=[[0.15, 1.49], [0.49, 2.01]],
        ),
    }
    order = ["A", "B"]
    if layout:
        order = (
            np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[3])
            .permutation(order)
            .tolist()
        )
    products = []
    for i, sku in enumerate(order):
        source_floor = floor + (slots[order[1]]["size_m"][2] + 0.002 if i == 0 else 0)
        products.append(
            dict(
                id=i,
                sku=sku,
                **slots[sku],
                source_floor_m=source_floor,
                source_center_m=[
                    0.12 + jitter[0],
                    0.985 + jitter[1],
                    source_floor + slots[sku]["size_m"][2] / 2 + 0.002,
                ],
            )
        )
    d.update(
        products=products,
        source_center_xy_m=[0.12, 0.985],
        source_floor_m=products[0]["source_floor_m"],
        source_label="MIXED A/B STOCK / LOCAL PICK DOCK",
        mounts_m=dict(left=[-0.46, 1.37, 0.52], right=[0.70, 1.37, 0.52]),
        destination_center_xy_m=products[0]["destination_center_xy_m"],
        destination_bounds_xy_m=products[0]["destination_bounds_xy_m"],
        lift_height_m=0.055,
        minimum_source_lift_m=0.04,
        source_clear_y_m=1.36,
        operation_required_boxes=2,
        runtime_joint_velocity_limit_rad_s=0.70,
        empty_initial_clearance_z_m=0.70,
        empty_joint_waypoints_rad=[1.57, 2.4, -1.5, 0.0, 0.0],
        empty_joint_speed_rad_s=0.60,
        operating_mode="two_sku_fixed_dock_component",
        sku_order_sampling="SeedSequence child3 permutation when layout randomization enabled; independent of timing/mass streams.",
        mounting_assumption="Complete original X5s on520mm fixed stands at1160mm spacing; same140x160mm feet. Existing pallet/tray staged under two actually stacked original cartons. No base motion.",
        scene_adaptation="Original central rack reused at1400mm width rather than900mm, with distinct A-01/B-01 locations; reserve racks shifted200mm outward. Existing carton asset reused as compact280x280x140mm A/brake-kit package and tall280x330x320mm B/filter package, with matching green/blue printed labels. Short A package leaves native wrist clearance beside tall B. No new product mesh, rack cuts or disabled contacts.",
        scope="Continuous two-SKU local pallet-to-correct-labelled-shelf operation, native grasp/support/release and visible0/2->2/2 count. Required full3boxes90s and upstream mobile delivery remain uncredited; no mobile base invented.",
    )
    d["bounds"]["sku_orders"] = [["A", "B"], ["B", "A"]]
    d.pop("sha256", None)
    d["sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


class PutawayARXController:
    """Serial real carton operations with a collision-checked empty return corridor."""

    def __init__(self, scene, mode):
        self.s, self.mode = scene, mode
        self.operation = PutawayArmOperation(scene, mode)
        self.phase, self.failure, self.complete = "sku0:preposition", None, False
        self.return_phase = None
        self.return_entered = None
        self.return_origins = None

    def step(self, observation):
        if self.complete or self.failure or self.mode in ["absent", "passive"]:
            return
        s, t = self.s, observation["time_s"]
        if t > s.resolved["budget_s"]:
            self.failure = "Operation incomplete by original90s budget"
            return
        if observation["unexpected_contacts"]:
            self.failure = "Unexpected native robot contact"
            return
        if self.return_phase is not None:
            try:
                self.step_return(observation)
            except RuntimeError as exc:
                self.failure = str(exc)
            return
        self.operation.step(observation)
        self.phase = f"sku{s.active_index}:" + self.operation.phase
        if self.operation.failure:
            self.failure = self.operation.failure
            return
        if not self.operation.complete:
            return
        s.placed.add(s.active_index)
        if hasattr(s, "count_display"):
            s.count_display.set(str(len(s.placed)))
        s.events[f"{s.active_index}_correct_slot_credit_s"] = t
        if len(s.placed) == len(s.boxes):
            if s.resolved.get("products") and not all(s.valid_product(i) for i in s.placed):
                self.failure = "A credited carton is not actually retained in its correct slot"
                s.physical_completed = s.manipulation_success = False
                return
            self.complete = True
            s.physical_completed = s.manipulation_success = True
            s.cue_state = "A/B CORRECTLY SHELVED / MOBILE PENDING"
            s.events["all_operation_boxes_placed_s"] = t
            return
        s.physical_completed = s.manipulation_success = False
        self.return_origins = {
            side: s.dual.tool_world(side)[0].copy() for side in ["left", "right"]
        }
        self.return_phase, self.return_entered = "raise", t
        s.events["empty_return_started_s"] = t

    def step_return(self, observation):
        s, op, t = self.s, self.operation, observation["time_s"]
        next_product = s.resolved["products"][s.active_index + 1]
        if self.return_phase == "raise":
            goals = {
                side: np.r_[p[:2], s.resolved["empty_initial_clearance_z_m"]]
                for side, p in self.return_origins.items()
            }
            done = [op.m[side].move(goals[side], 0.10, op.R[side]) for side in ["left", "right"]]
            self.phase = "empty_return:raise_clear"
            if all(done):
                self.plan_joint_return(next_product, t)
            elif t - self.return_entered > 18:
                self.failure = "Empty return timeout raise"
            return
        d = s.dual
        if self.return_phase.startswith("joint_"):
            tracking = all(
                float(
                    abs(
                        np.asarray(d.robots[side].get_joint_positions())[op.m[side].ids]
                        - d.commands[side][op.m[side].ids]
                    ).max()
                )
                < 0.04
                for side in ["left", "right"]
            )
            if tracking:
                self.return_progress = min(
                    1.0, self.return_progress + 1 / (CONTROL_HZ * self.return_duration)
                )
            for side in ["left", "right"]:
                goal = self.return_waypoints[side][self.return_stage]
                d.commands[side][op.m[side].ids] = (
                    self.return_start[side] * (1 - self.return_progress)
                    + goal * self.return_progress
                )
                d.jaw_travel(side, 0.002)
            arrived = self.return_progress >= 1 and all(
                float(
                    abs(
                        np.asarray(d.robots[side].get_joint_positions())[op.m[side].ids]
                        - self.return_waypoints[side][self.return_stage]
                    ).max()
                )
                < 0.02
                for side in ["left", "right"]
            )
            self.phase = self.return_phase
            if arrived:
                self.return_stage += 1
                if self.return_stage == 3:
                    for side in ["left", "right"]:
                        op.m[side].command = d.tool_world(side)[0].copy()
                        op.m[side].correction[:] = 0
                        op.m[side].orientation_correction[:] = 0
                    self.return_phase = "settle_next"
                    self.return_entered = t
                else:
                    self.start_joint_stage(t)
        elif self.return_phase == "settle_next":
            done = [
                op.m[side].move(self.next_prepose[side], 0.05, op.R[side])
                for side in ["left", "right"]
            ]
            self.phase = "empty_return:settle_next"
            if all(done):
                s.activate(s.active_index + 1)
                self.operation = PutawayArmOperation(s, self.mode)
                self.return_phase = None
                s.events["empty_return_complete_s"] = t
                self.phase = f"sku{s.active_index}:preposition"
        if self.return_phase is not None and t - self.return_entered > 18:
            self.failure = "Empty return timeout " + self.return_phase

    def plan_joint_return(self, next_product, t):
        s, op, d = self.s, self.operation, self.s.dual
        profile = d.manifest["profile"]
        self.return_waypoints, self.next_prepose = {}, {}
        for side, sign in [("left", -1), ("right", 1)]:
            p = np.asarray(next_product["source_center_m"], float) + [
                sign * (s.resolved["side_tool_offset_x_m"] + 0.06),
                0,
                s.resolved["side_tool_offset_z_m"] + 0.02,
            ]
            self.next_prepose[side] = p
            bp, bR = d.body_world(side, "base_link")
            q, error = op.m[side].kin.solve(
                bR.T @ (p - bp),
                bR.T @ op.R[side],
                dict(zip(d.names[side], d.commands[side])),
                profile["arm_lower_rad"],
                profile["arm_upper_rad"],
                restarts=4,
            )
            if error > 0.001:
                self.failure = "Bounded next-stock prepose unreachable: " + str(error)
                return
            folded = d.commands[side][op.m[side].ids].copy()
            folded[1:] = s.resolved["empty_joint_waypoints_rad"]
            turned = folded.copy()
            turned[0] = q[0]
            self.return_waypoints[side] = [folded, turned, q]
        self.return_stage = 0
        self.start_joint_stage(t)

    def start_joint_stage(self, t):
        d, op = self.s.dual, self.operation
        self.return_start = {
            side: d.commands[side][op.m[side].ids].copy() for side in ["left", "right"]
        }
        self.return_duration = max(
            0.1,
            max(
                float(
                    abs(
                        self.return_waypoints[side][self.return_stage] - self.return_start[side]
                    ).max()
                )
                / self.s.resolved["empty_joint_speed_rad_s"]
                for side in ["left", "right"]
            ),
        )
        self.return_progress = 0.0
        self.return_phase = ["joint_fold", "joint_swing", "joint_unfold"][self.return_stage]
        self.return_entered = t
        self.s.events[self.return_phase] = dict(
            time_s=t,
            planned_duration_s=self.return_duration,
            goal_rad={
                side: self.return_waypoints[side][self.return_stage].tolist()
                for side in ["left", "right"]
            },
            scope="Original bounded native joint drives; no product state writes",
        )
