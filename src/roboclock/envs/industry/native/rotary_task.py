"""Final Task42: complete ARX arms and native moving carrier probes.

The machine's inspection/unload scenery is not an implemented process. No
product pose writes or attachments are used after clean episode initialization.
"""

import hashlib
import json

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from ..scoring import voiding_contacts
from . import rig
from .arx_cell_task import SIDES, finger_side, jaws_open
from .v2.kinematics import UrdfArm

# Only the left arm works; it keeps its recorded base and the parked right arm sits 0.60 m
# beside it (recorded: right [0.24, -0.25, 1.06], staggered behind the left).
MOUNTS = rig.mounts_keeping("left", [-0.39, 0.02, 1.06])
# Seeded floor-plane offset of the housing and its infeed cradle pins (layout stream).
SOURCE_XY_OFFSET_M = 0.010


def resolve_rotary(seed=0, *, layout=False, timing=False, physical=False):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    r = dict(
        task_id=42,
        seed=seed,
        embodiment="dual_arx_x5",
        source_xy_offset_m=streams[0].uniform(-SOURCE_XY_OFFSET_M, SOURCE_XY_OFFSET_M, 2).tolist()
        if layout
        else [0.0, 0.0],
        rotor_initial_deg=240.0 + (float(streams[1].uniform(-10, 10)) if timing else 0.0),
        rotation_deg_s=-5.0,
        loading_sector_deg=[-120.0, -60.0],
        window_s=12.0,
        mass_kg=float(streams[2].uniform(0.27, 0.33)) if physical else 0.30,
        randomization=dict(layout=layout, timing=timing, physical=physical),
        bounds=dict(
            source_xy_offset_m=[-SOURCE_XY_OFFSET_M, SOURCE_XY_OFFSET_M],
            rotor_phase_offset_deg=[-10, 10],
            mass_kg=[0.27, 0.33],
        ),
        independence="Independent SeedSequence layout, temporal-phase and physical-property streams",
        physical_provenance="Declared compact cast-aluminium demonstration mass/inertia; not measured source properties",
        controller="Privileged native-state bounded joint-drive oracle; not a trained policy",
    )
    r["sha256"] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
    return r


class RotaryScene:
    def __init__(self, assets, manifest, output, resolved, *, grip_offset=(0.038, 0, 0.042)):
        import omni.physx
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .rotary_scene import build_rotary_environment

        self.base = build_rotary_environment(
            assets, output, resolution=(1280, 800),
            sector_deg=resolved["loading_sector_deg"], speed_deg_s=-resolved["rotation_deg_s"],
        )
        self.world, self.stage = self.base.world, self.base.stage
        self.physics_hz = 1.0 / float(self.world.get_physics_dt())
        self.resolved, self.events = resolved, {}
        self.grip_offset = np.asarray(grip_offset)
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS, gripper_velocity_cap=False)
        stage, a = self.stage, self.base.appearance
        # Visible fixture pins follow original-mesh underside samples. The
        # casting rocks on a flat plate; no product attachment is added.
        support_samples = [
            (-0.040, -0.035, -0.00951),
            (-0.040, 0.035, 0.01945),
            (0.040, -0.035, 0.03174),
            (0.040, 0.035, 0.01856),
        ]
        # The infeed cradle's pins move with the housing's seeded offset (layout stream), so
        # the casting rests on them exactly as at the nominal spot; the 0.15 m support plate
        # under them stays put.
        source_offset = np.asarray(resolved["source_xy_offset_m"], float)
        for station, origin, plate_top, center_z in [
            ("Feed/Cradle3", [-0.80 + source_offset[0], 0.25 + source_offset[1]], 0.858, 0.900),
            ("Carriers/C0", [0.0, 0.27], 0.875, 0.917),
        ]:
            for i, (dx, dy, underside_z) in enumerate(support_samples):
                top = center_z + underside_z
                pin = UsdGeom.Cylinder.Define(stage, f"/World/{station}/CradlePin{i}")
                pin.CreateRadiusAttr(0.003)
                pin.CreateHeightAttr(top - plate_top)
                pin.AddTranslateOp().Set(
                    Gf.Vec3d(origin[0] + dx, origin[1] + dy, (top + plate_top) / 2)
                )
                UsdShade.MaterialBindingAPI.Apply(pin.GetPrim()).Bind(a.material("steel").material)
        for side, (x, y, z) in MOUNTS.items():
            a.rounded(
                f"/World/ARXMounts/{side}Column",
                [x, y, (z - 0.02) / 2],
                [0.075, 0.075, z - 0.02],
                "frame",
                radius=0.005,
            )
            a.rounded(
                f"/World/ARXMounts/{side}Foot",
                [x, y, 0.022],
                [0.22, 0.22, 0.04],
                "frame",
                radius=0.006,
            )
            a.rounded(
                f"/World/ARXMounts/{side}Plate",
                [x, y, z - 0.012],
                [0.15, 0.14, 0.018],
                "steel",
                radius=0.004,
            )
        # Remove staged examples in the rotor; one actual free scored product
        # replaces the nearest input casting. The other conveyor items are scenery.
        for path in ["/World/Products/Loaded1", "/World/Products/Loaded2", "/World/Products/WIP3"]:
            self.world.scene.remove_object(path.rsplit("/", 1)[-1], registry_only=True)
            stage.RemovePrim(path)
        rotor_root = UsdGeom.Xform.Define(stage, "/World/Rotor").GetPrim()
        off = UsdGeom.Xform.Define(stage, "/World/Rotor/Offset")
        off.AddTranslateOp().Set(Gf.Vec3d(0, -0.55, -0.856))
        layer = stage.GetRootLayer()
        for source in ["/World/Carriers", "/World/Turntable/Disc", "/World/Turntable/Hub"] + [
            f"/World/Turntable/HubBolt{i}" for i in range(8)
        ]:
            target = "/World/Rotor/Offset/" + source.rsplit("/", 1)[-1]
            Sdf.CopySpec(layer, source, layer, target)
            stage.RemovePrim(source)
        UsdPhysics.RigidBodyAPI.Apply(rotor_root)
        mass = UsdPhysics.MassAPI.Apply(rotor_root)
        mass.CreateMassAttr(40.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2.0, 2.0, 4.0))
        PhysxSchema.PhysxContactReportAPI.Apply(rotor_root).CreateThresholdAttr(0)
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/RotaryDrive")
        joint.CreateBody1Rel().SetTargets(["/World/Rotor"])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0.55, 0.856))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0)
        drive.CreateDampingAttr(10000.0)
        drive.CreateMaxForceAttr(100.0)
        drive.CreateTargetVelocityAttr(resolved["rotation_deg_s"])
        self.rotor = self.world.scene.add(
            SingleRigidPrim(
                "/World/Rotor",
                name="rotor",
                position=np.array([0, 0.55, 0.856]),
                orientation=np.array(
                    [
                        np.cos(np.deg2rad(resolved["rotor_initial_deg"]) / 2),
                        0,
                        0,
                        np.sin(np.deg2rad(resolved["rotor_initial_deg"]) / 2),
                    ]
                ),
            )
        )
        contact_material = PhysicsMaterial(
            "/World/RotaryContactMaterial",
            static_friction=0.8,
            dynamic_friction=0.65,
            restitution=0,
        )
        self.collider_paths = []
        for prim in list(stage.Traverse()):
            path = str(prim.GetPath())
            if not prim.IsA(UsdGeom.Gprim):
                continue
            if not path.startswith(
                (
                    "/World/Building/",
                    "/World/Machine/",
                    "/World/Process/",
                    "/World/LoadingZone/",
                    "/World/ARXMounts/",
                    "/World/Feed/Support",
                    "/World/Feed/Cradle3/",
                    "/World/Rotor/",
                )
            ):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexHull" if path.startswith("/World/Rotor/") else "none"
                )
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(0.0005)
            api.CreateRestOffsetAttr(0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                contact_material.material, materialPurpose="physics"
            )
            self.collider_paths.append(path)
        self.part_path = "/World/Pump"
        root = UsdGeom.Xform.Define(stage, self.part_path).GetPrim()
        root.GetReferences().AddReference(str(assets / "pump/pump_casing.usd"))
        cache = UsdGeom.XformCache()
        vertices, counts, indices = [], [], []
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    a.material("cast_metal").material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                transform = cache.GetLocalToWorldTransform(prim)
                offset = len(vertices)
                vertices.extend(
                    tuple(transform.Transform(Gf.Vec3d(*map(float, v))))
                    for v in mesh.GetPointsAttr().Get()
                )
                counts.extend(mesh.GetFaceVertexCountsAttr().Get())
                indices.extend(offset + int(i) for i in mesh.GetFaceVertexIndicesAttr().Get())
        self.envelope = np.unique(np.asarray(vertices), axis=0)
        self.outer_vertices = self.envelope[ConvexHull(self.envelope).vertices]
        collider = UsdGeom.Mesh.Define(stage, self.part_path + "/ContactSurface")
        collider.CreatePointsAttr(vertices)
        collider.CreateFaceVertexCountsAttr(counts)
        collider.CreateFaceVertexIndicesAttr(indices)
        collider.CreateSubdivisionSchemeAttr("none")
        collider.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("sdf")
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(collider.GetPrim()).CreateSdfResolutionAttr(256)
        api = PhysxSchema.PhysxCollisionAPI.Apply(collider.GetPrim())
        api.CreateContactOffsetAttr(0.0005)
        api.CreateRestOffsetAttr(0)
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(resolved["mass_kg"])
        dimensions = np.ptp(self.envelope, axis=0)
        inertia = resolved["mass_kg"] / 12 * (np.sum(dimensions**2) - dimensions**2)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        rb.CreateSolverPositionIterationCountAttr(64)
        rb.CreateSolverVelocityIterationCountAttr(16)
        rb.CreateSleepThresholdAttr(0)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0)
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(
            contact_material.material, materialPurpose="physics"
        )
        for side in self.dual.robots:
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/ARX/" + side)):
                if prim.HasAPI(UsdPhysics.CollisionAPI) and any(
                    "/" + n in str(prim.GetPath()) for n in ("link7", "link8")
                ):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        contact_material.material, materialPurpose="physics"
                    )
        # Start in the casting's stable rest on the cradle. Dropped onto it, the casting
        # rocks into one of several rest poses, and which one depends on the physics step.
        position = np.array([-0.79957, 0.25015, 0.90077])
        position[:2] += resolved["source_xy_offset_m"]
        self.part = self.world.scene.add(
            SingleRigidPrim(
                self.part_path,
                name="free_pump",
                position=position,
                orientation=np.array([0.99836, 0.02378, -0.03378, 0.03955]),
            )
        )
        self.interval, self.unexpected, self.contacts, self.geometry = {}, {}, {}, {}
        # Arms whose fingers have touched the housing (the release is judged on those).
        self.finger_sides = set()

        def contact(headers, data):
            for h in headers:
                actors = [str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.actor0, h.actor1)]
                colliders = [
                    str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.collider0, h.collider1)
                ]
                for index in range(
                    h.contact_data_offset, h.contact_data_offset + h.num_contact_data
                ):
                    item = data[index]
                    impulse = abs(float(np.dot(np.asarray(item.impulse), np.asarray(item.normal))))
                    if impulse < 1e-8:
                        continue
                    key = " | ".join(sorted(actors))
                    self.contacts.setdefault(key, float(self.world.current_time))
                    separation = float(item.separation)
                    self.geometry[key] = min(self.geometry.get(key, 0), separation)
                    arm = next((p for p in actors if p.startswith("/World/ARX/")), None)
                    finger = arm is not None and arm.endswith(("/link7", "/link8"))
                    if self.part_path in actors:
                        kind = (
                            ("finger7" if arm.endswith("/link7") else "finger8")
                            if finger
                            else (
                                "carrier"
                                if any("/Carriers/C0/" in p for p in colliders)
                                else "source"
                                if any(
                                    "/Feed/Support3/" in p
                                    or p.endswith("/Feed/Support3")
                                    or "/Feed/Cradle3/" in p
                                    for p in colliders
                                )
                                else "other"
                            )
                        )
                        self.interval[kind] = self.interval.get(kind, 0.0) + impulse * self.physics_hz
                        if finger:
                            self.finger_sides.add(finger_side(arm))
                    # Open fingers may touch the target fixture during release;
                    # record this rather than pretending it is product support.
                    if arm and not (self.part_path in actors and finger):
                        entry = self.unexpected.setdefault(
                            key,
                            dict(
                                time_s=float(self.world.current_time),
                                colliders=colliders,
                                impulse_Ns=impulse,
                            ),
                        )
                        # scoring.voiding_contacts judges the contact by how hard it pressed.
                        entry["peak_force_N"] = max(
                            entry.get("peak_force_N", 0.0), impulse * self.physics_hz
                        )

        self.subscription = (
            omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
        )

        def initialize_posture():
            self.dual.initialize()
            # Initial standby only: park the unused complete right arm away
            # from the main view/loading area, within the original profile.
            command = self.dual.commands["right"]
            command[self.dual.names["right"].index("joint1")] = -1.8
            robot = self.dual.robots["right"]
            robot.set_joints_default_state(positions=command, velocities=np.zeros(8))
            robot.set_joint_positions(command)
            robot.set_joint_velocities(np.zeros(8))

        self.world.reset()
        initialize_posture()
        self.world.reset()
        initialize_posture()
        self.part.initialize()
        self.rotor.initialize()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=42)
        self.start_s = float(self.world.current_time)
        self.initial_part = self.part.get_world_pose()[0].copy()
        self.retained_since, self.success = None, False

    def observe(self):
        p, q = self.part.get_world_pose()
        sector = self.resolved["loading_sector_deg"]
        rp, rq = self.rotor.get_world_pose()
        R = Rotation.from_quat(np.asarray(rq)[[1, 2, 3, 0]]).as_matrix()
        center = rp + R @ np.array([0, -0.28, 0.061])
        angle = float(np.rad2deg(np.arctan2(center[1] - 0.55, center[0])))
        tool, _ = self.dual.tool_world("left")
        part_R = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
        local_points = (self.outer_vertices @ part_R.T + p - rp) @ R
        velocity = self.part.get_linear_velocity()
        omega = self.rotor.get_angular_velocity()
        inside = bool(
            np.max(np.abs(local_points[:, :2] - [0, -0.28])) < 0.079
            and np.min(local_points[:, 2]) > 0.018
        )
        forces = self.interval.copy()
        # Either arm may place the housing: the jaws that must be open are those of every arm
        # whose fingers have held it (the left arm's in the recorded oracle, as before).
        jaws = {
            side: self.dual.robots[side].get_joint_positions()[
                [self.dual.names[side].index("joint7"), self.dual.names[side].index("joint8")]
            ]
            for side in SIDES
        }
        released = bool(
            jaws_open(jaws, self.finger_sides)
            and max(forces.get("finger7", 0), forces.get("finger8", 0)) < 0.10
        )
        supported = bool(
            forces.get("carrier", 0) > 0.3
            and inside
            and np.linalg.norm(velocity - np.cross(omega, p - rp)) < 0.035
        )
        t = float(self.world.current_time)
        if (
            "bilateral_grasp_s" in self.events
            and "source_cleared_s" not in self.events
            and forces.get("source", 0) < 0.1
        ):
            world_points = self.outer_vertices @ part_R.T + p
            if world_points[:, 2].min() > 0.934:
                self.events["source_cleared_s"] = t
        if supported and released and "source_cleared_s" in self.events:
            if self.retained_since is None:
                self.retained_since, self.retention_angle = t, angle
            if t - self.retained_since > 0.30 and "supported_release_s" not in self.events:
                self.events.update(
                    supported_release_s=self.retained_since,
                    retention_confirmed_s=t,
                    release_angle_deg=self.retention_angle,
                    temporal_success=bool(sector[0] <= self.retention_angle <= sector[1]),
                )
                self.success = self.events["temporal_success"]
        else:
            self.retained_since = None
        return dict(
            time_s=t,
            elapsed_s=t - self.start_s,
            tool_position=tool.tolist(),
            part_position=p.tolist(),
            part_quaternion=np.asarray(q).tolist(),
            part_velocity=velocity.tolist(),
            carrier_center=center.tolist(),
            carrier_angle_deg=angle,
            rotor_omega_rad_s=omega.tolist(),
            rotor_rotation=R.tolist(),
            window_open=bool(sector[0] <= angle <= sector[1]),
            forces=forces,
            released=released,
            supported=supported,
            unexpected_contacts=self.unexpected.copy(),
        )


class RotaryController:
    def __init__(self, scene, mode):
        self.s, self.mode, self.phase = scene, mode, "settle"
        self.entered = float(scene.world.current_time)
        self.failure, self.complete, self.bilateral_since = None, False, None
        self.travel = 0.044
        d = scene.dual
        self.kin = UrdfArm(
            d.manifest["derived_urdf"],
            "link6",
            d.manifest["tool"]["offset_m"],
            np.eye(3),
            [f"joint{i}" for i in range(1, 7)],
        )
        self.ids = [d.names["left"].index(n) for n in self.kin.arm_names]
        self.R = d.rotation @ Rotation.from_euler("y", np.pi / 2).as_matrix()
        self.grasp_R = self.R.copy()
        self.start = d.tool_world("left")[0]
        self.target, self.correction = self.start.copy(), np.zeros(3)
        self.held_offset = None
        self.missing_grasp_since = None
        self.source_part_R = None

    def transition(self, phase, target=None):
        self.phase = phase
        self.entered = float(self.s.world.current_time)
        self.start = self.s.dual.tool_world("left")[0]
        self.target = self.start.copy() if target is None else np.asarray(target, float)
        self.correction[:] = 0
        self.s.events["phase_" + phase + "_s"] = self.entered
        print("PHASE", phase, self.entered, flush=True)

    def move(self, o, speed=0.055, tracking=False):
        d = self.s.dual
        progress = min(
            1.0,
            (o["time_s"] - self.entered)
            / max(0.5, np.linalg.norm(self.target - self.start) / speed),
        )
        desired = self.target if tracking else self.start + (self.target - self.start) * progress
        measured = np.asarray(o["tool_position"])
        self.correction = np.clip(self.correction + 0.8 * (desired - measured) / CONTROL_HZ, -0.012, 0.012)
        bp, bR = d.body_world("left", "base_link")
        profile = d.manifest["profile"]
        q, error = self.kin.solve(
            bR.T @ (desired + self.correction - bp),
            bR.T @ self.R,
            dict(zip(d.names["left"], d.commands["left"])),
            profile["arm_lower_rad"],
            profile["arm_upper_rad"],
        )
        if error > 0.004:
            raise RuntimeError(f"Unreachable {self.phase}: {error:.4f}m")
        d.commands["left"][self.ids] += np.clip(
            q - d.commands["left"][self.ids], -0.70 / CONTROL_HZ, 0.70 / CONTROL_HZ
        )
        return progress >= 1 and np.linalg.norm(measured - self.target) < 0.004

    def step(self, o):
        if self.failure:
            return
        t, p = o["time_s"], np.asarray(o["part_position"])
        age = t - self.entered
        try:
            # Abort on the contacts the scoring voids (scoring.voiding_contacts); a finger
            # touching the carrier while the housing is lowered onto it does not void.
            if voiding_contacts(o["unexpected_contacts"]):
                raise RuntimeError("Unexpected native robot contact")
            if p[2] < 0.70:
                raise RuntimeError("Product fell outside work surface")
            if self.mode == "passive":
                return
            if self.phase == "settle" and age > 1.5:
                self.source_part_R = Rotation.from_quat(
                    np.asarray(o["part_quaternion"])[[1, 2, 3, 0]]
                ).as_matrix()
                self.transition("above_source", p + self.s.grip_offset + [0, 0, 0.13])
            elif self.phase == "above_source":
                if self.move(o):
                    self.transition("descend_grasp", p + self.s.grip_offset)
            elif self.phase == "descend_grasp":
                if self.move(o, 0.025):
                    self.transition("close")
            elif self.phase == "close":
                self.move(o, 0.025)
                self.travel = max(0.001, self.travel - 0.008 / CONTROL_HZ)
                both = min(o["forces"].get("finger7", 0), o["forces"].get("finger8", 0)) > 10.0
                self.bilateral_since = (
                    (t if self.bilateral_since is None else self.bilateral_since) if both else None
                )
                if self.bilateral_since is not None and t - self.bilateral_since > 0.25:
                    self.s.events["bilateral_grasp_s"] = t
                    self.held_offset = p - np.asarray(o["tool_position"])
                    self.transition("lift", np.asarray(o["tool_position"]) + [0, 0, 0.13])
            elif self.phase == "lift":
                if self.move(o, 0.04):
                    if self.mode == "grasp":
                        self.transition("hold")
                    else:
                        self.transition("standby", [-0.10, 0.28, 1.10])
            elif self.phase == "hold":
                self.move(o)
                if age > 1.5:
                    self.complete = True
            elif self.phase == "standby":
                if self.move(o):
                    # Estimate the actual settled grasp transform from native
                    # observations before aligning to the moving fixture.
                    tool_p, tool_R = self.s.dual.tool_world("left")
                    part_R = Rotation.from_quat(
                        np.asarray(o["part_quaternion"])[[1, 2, 3, 0]]
                    ).as_matrix()
                    align = self.source_part_R @ part_R.T
                    self.grasp_R = align @ tool_R
                    self.held_offset = align @ (p - tool_p)
                    self.transition("wait_window")
            elif self.phase == "wait_window":
                self.move(o)
                if -110 < o["carrier_angle_deg"] < -90:
                    # Select a feasible interception from measured native
                    # frames, with an inset from the original joint bounds.
                    d = self.s.dual
                    bp, bR = d.body_world("left", "base_link")
                    r = np.asarray(o["rotor_rotation"])
                    target = np.asarray(o["carrier_center"]) - r @ self.held_offset + [0, 0, 0.08]
                    profile = d.manifest["profile"]
                    _, error = self.kin.solve(
                        bR.T @ (target - bp),
                        bR.T @ r @ self.grasp_R,
                        dict(zip(d.names["left"], d.commands["left"])),
                        np.asarray(profile["arm_lower_rad"]) + 0.02,
                        np.asarray(profile["arm_upper_rad"]) - 0.02,
                    )
                    if error < 0.001:
                        self.transition("track")
            elif self.phase in ("track", "lower", "open"):
                center = np.asarray(o["carrier_center"])
                carrier_R = np.asarray(o["rotor_rotation"])
                self.R = carrier_R @ self.grasp_R
                self.target = (
                    center
                    - carrier_R @ self.held_offset
                    + [0, 0, 0.08 if self.phase == "track" else -0.001]
                )
                reached = self.move(o, tracking=True)
                if self.phase == "track" and reached and age > 1.0:
                    self.transition("lower")
                elif self.phase == "lower" and o["forces"].get("carrier", 0) > 0.4:
                    self.transition("open")
                elif self.phase == "open":
                    self.travel = min(0.044, self.travel + 0.040 / CONTROL_HZ)
                    if self.s.success:
                        self.transition("withdraw", np.asarray(o["tool_position"]) + [0, 0, 0.08])
            elif self.phase == "withdraw":
                if self.move(o):
                    self.complete = True
                    self.transition("complete")
            if self.held_offset is not None and self.phase in (
                "lift",
                "hold",
                "standby",
                "wait_window",
                "track",
                "lower",
            ):
                if self.phase == "lift":
                    if min(o["forces"].get("finger7", 0), o["forces"].get("finger8", 0)) < 17:
                        self.travel = max(0.001, self.travel - 0.003 / CONTROL_HZ)
                elif self.phase != "hold":
                    # The close leaves both slider drives at their effort limit, where the
                    # casting slides freely along the jaws. An 18 N spring centers it.
                    d = self.s.dual
                    q = d.robots["left"].get_joint_positions()
                    opening = (q[d.names["left"].index("joint7")] + q[d.names["left"].index("joint8")]) / 2
                    self.travel = min(self.travel + 0.003 / CONTROL_HZ, opening - 0.012)
                # An irregular rim grasp can settle/rotate under gravity while
                # maintaining both contacts. Reject observed loss of bilateral
                # contact and gross separation; do not demand a rigid weld.
                bilateral = min(o["forces"].get("finger7", 0), o["forces"].get("finger8", 0)) > 0.3
                self.missing_grasp_since = (
                    None
                    if bilateral
                    else (t if self.missing_grasp_since is None else self.missing_grasp_since)
                )
                if np.linalg.norm(p - np.asarray(o["tool_position"])) > 0.11 or (
                    self.missing_grasp_since is not None and t - self.missing_grasp_since > 0.08
                ):
                    raise RuntimeError("Lost physical grasp")
            # Time in the current phase. `age` was taken before this step's transition, so
            # leaving a long wait_window (> 20 s at table speeds below 5 deg/s) read as a
            # timeout of the phase just entered.
            if t - self.entered > 20 and self.phase not in ("wait_window", "complete"):
                raise RuntimeError("Phase timeout: " + self.phase)
            self.s.dual.jaw_travel("left", self.travel)
        except Exception as exc:
            self.failure = str(exc)
            print("FAILURE", self.failure, flush=True)
