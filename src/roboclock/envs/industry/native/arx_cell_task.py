"""Shared native drives/contact logging for approved final industrial cells."""

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from .v2.kinematics import UrdfArm


def remove_preview(scene, path):
    """Remove the registered static wrapper as well as its prim."""
    scene.world.scene.remove_object(path.rsplit("/", 1)[-1], registry_only=True)
    scene.stage.RemovePrim(path)


def set_signal(material, color):
    """Keep emitted light and surface color in the same measured state."""
    from pxr import Gf

    material.set_color(np.asarray(color))
    material.shaders_list[0].GetInput("emissive_color").Set(Gf.Vec3f(*color))
    material.shaders_list[0].GetInput("emissive_intensity").Set(2.0)


def add_mounts(scene, mounts, base_z=0, foot_size=None):
    for side, (x, y, z) in mounts.items():
        scene.box(
            f"ARXMounts/{side}Column",
            [x, y, base_z + (z - base_z - 0.02) / 2],
            [0.075, 0.075, z - base_z - 0.02],
        )
        scene.box(
            f"ARXMounts/{side}Foot",
            [x, y, base_z + 0.022],
            foot_size
            if foot_size is not None
            else ([0.15, 0.10, 0.04] if base_z else [0.22, 0.22, 0.04]),
        )
        scene.box(f"ARXMounts/{side}Plate", [x, y, z - 0.012], [0.15, 0.14, 0.018], "steel")


# Every camera prim of a sensor asset carries Kit's viewport gizmo, a 0.2 m mesh that is never
# rendered. Given a collider it is an invisible shell around the sensor: C3's D455 has four,
# standing 5 cm in front of its ring light, and every "camera" collision of the 2026-11 sweep
# was with them, never with the visible body.
VIEWPORT_GIZMO = "OmniverseKitViewportCameraMesh"


def collide(stage, prefixes, material=None):
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    paths = []
    for p in stage.Traverse():
        if not str(p.GetPath()).startswith(tuple(prefixes)) or not p.IsA(UsdGeom.Gprim):
            continue
        if VIEWPORT_GIZMO in str(p.GetPath()):
            continue
        UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("none")
        a = PhysxSchema.PhysxCollisionAPI.Apply(p)
        a.CreateContactOffsetAttr(0.0003)
        a.CreateRestOffsetAttr(0)
        if material:
            UsdShade.MaterialBindingAPI.Apply(p).Bind(material.material, materialPurpose="physics")
        paths.append(str(p.GetPath()))
    return paths


SIDES = ("left", "right")
# A2 and A3 count a release only once each finger slider is open past this (of 0.044 m).
RELEASED_JAW_M = 0.041


def finger_side(actor):
    """The arm ("left" or "right") a contact actor is a gripper finger of, else None."""
    if actor.startswith("/World/ARX/") and actor.endswith(("/link7", "/link8")):
        return actor.split("/")[3]
    return None


def jaws_open(jaws_by_side, sides):
    """Whether both finger sliders of every arm in `sides` are open past RELEASED_JAW_M.

    `sides` are the arms whose fingers have held the product: the release is judged on
    whichever arm did the work, not on the arm the recorded oracle uses. An arm that never
    touched the product does not have to open.
    """
    return all(min(jaws_by_side[side]) > RELEASED_JAW_M for side in sides)


class ContactBook:
    def __init__(self, world, product_paths, watched_prefixes=(), *, record_points=False):
        import omni.physx
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools

        self.interval, self.pairs, self.minimum, self.active, self.unexpected = {}, {}, {}, {}, {}
        self.contact_samples = {}
        self.minimum_events = {}
        self.interarm_events = {}
        self.physics_hz = 1.0 / float(world.get_physics_dt())
        cache = {}

        def path(value):
            key = int(value)
            if key not in cache:
                cache[key] = str(PhysicsSchemaTools.intToSdfPath(value))
            return cache[key]

        def contact(headers, data):
            for h in headers:
                actors = [path(h.actor0), path(h.actor1)]
                colliders = [path(h.collider0), path(h.collider1)]
                key = " | ".join(sorted(actors))
                scored = any(p in actors for p in product_paths)
                arm = next((p for p in actors if p.startswith("/World/ARX/")), None)
                if not (
                    scored or arm or any(a.startswith(tuple(watched_prefixes)) for a in actors)
                ):
                    continue
                patch = " | ".join(sorted(colliders))
                self.active[patch] = dict(
                    actors=actors, active=h.type != ContactEventType.CONTACT_LOST
                )
                # Do not hide zero-impulse inter-arm contact behind the force gate.
                interarm = (any(a.startswith('/World/ARX/left/') for a in actors)
                            and any(a.startswith('/World/ARX/right/') for a in actors))
                if interarm and h.type != ContactEventType.CONTACT_LOST:
                    self.interarm_events.setdefault(patch, dict(time_s=float(world.current_time),
                        actors=actors, colliders=colliders, force_threshold_applied=False))
                force = 0.0
                samples = []
                for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                    d = data[i]
                    impulse = abs(float(np.dot(np.asarray(d.impulse), np.asarray(d.normal))))
                    force += impulse * self.physics_hz
                    if impulse > 1e-8:
                        separation = float(d.separation)
                        previous_minimum = self.minimum.get(key, 0)
                        self.minimum[key] = min(previous_minimum, separation)
                        if separation < previous_minimum:
                            self.minimum_events[key] = dict(
                                time_s=float(world.current_time),
                                separation_m=separation,
                                normal_force_N=impulse * self.physics_hz,
                                position_m=list(map(float, d.position)),
                                colliders=colliders,
                            )
                        if record_points and scored and arm:
                            samples.append(
                                dict(
                                    position_m=list(map(float, d.position)),
                                    normal=list(map(float, d.normal)),
                                    normal_force_N=impulse * self.physics_hz,
                                    separation_m=float(d.separation),
                                )
                            )
                if samples:
                    t = float(world.current_time)
                    previous = self.contact_samples.get(key)
                    if previous is None or previous["time_s"] != t:
                        previous = dict(time_s=t, actors=actors, points=[])
                        self.contact_samples[key] = previous
                    previous["points"].extend(samples)
                if force < 1e-6:
                    continue
                self.pairs.setdefault(key, float(world.current_time))
                self.interval[key] = self.interval.get(key, 0.0) + force
                finger = arm and arm.endswith(("/link7", "/link8"))
                if arm and not (scored and finger):
                    entry = self.unexpected.setdefault(
                        key, dict(time_s=float(world.current_time), colliders=colliders)
                    )
                    # scoring.voiding_contacts judges the contact by how hard it pressed.
                    entry["peak_force_N"] = max(entry.get("peak_force_N", 0.0), force)

        self.subscription = (
            omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
        )

    def force(self, a, b):
        return sum(v for k, v in self.interval.items() if a in k.split(" | ") and b in k)

    def finger_sides(self, product):
        """Arms whose fingers have pressed on `product` since the contact maps were cleared."""
        sides = set()
        for key in self.pairs:
            actors = key.split(" | ")
            if product in actors:
                sides.update(side for side in map(finger_side, actors) if side)
        return sides

    def touching(self, a, b):
        return any(
            v["active"] and a in v["actors"] and b in v["actors"] for v in self.active.values()
        )


class ArmMotion:
    """Bounded original-URDF IK targets; native feedback remains authoritative."""

    def __init__(self, dual, side="left"):
        self.dual, self.side = dual, side
        self.kin = UrdfArm(
            dual.manifest["derived_urdf"],
            "link6",
            dual.manifest["tool"]["offset_m"],
            np.eye(3),
            [f"joint{i}" for i in range(1, 7)],
        )
        self.ids = [dual.names[side].index(n) for n in self.kin.arm_names]
        mount_rotation = getattr(dual, "rotations", {}).get(side, dual.rotation)
        self.R = mount_rotation @ Rotation.from_euler("y", np.pi / 2).as_matrix()
        self.goal = self.command = dual.tool_world(side)[0].copy()
        self.correction = np.zeros(3)
        self.position_gain = 0.65
        self.position_correction_limit_m = 0.010
        self.orientation_feedback = False
        self.command_speed_rad_s = 0.70
        self.orientation_correction = np.zeros(3)

    def move(self, goal, speed=0.055, R=None):
        d, side = self.dual, self.side
        goal = np.asarray(goal, float)
        actual = d.tool_world(side)[0]
        delta = goal - self.command
        # Trajectory does not race ahead when the native tool has not followed.
        if np.linalg.norm(actual - self.command) < 0.018:
            self.command = self.command + delta * min(
                1, speed / CONTROL_HZ / max(np.linalg.norm(delta), 1e-12)
            )
        self.correction = np.clip(
            self.correction + self.position_gain * (self.command - actual) / CONTROL_HZ,
            -self.position_correction_limit_m,
            self.position_correction_limit_m,
        )
        bp, bR = d.body_world(side, "base_link")
        profile = d.manifest["profile"]
        desired_R = self.R if R is None else R
        command_R = desired_R
        if self.orientation_feedback:
            actual_R = d.tool_world(side)[1]
            error_R = Rotation.from_matrix(desired_R @ actual_R.T).as_rotvec()
            self.orientation_correction = np.clip(
                self.orientation_correction + 0.8 * error_R / CONTROL_HZ, -0.05, 0.05
            )
            command_R = Rotation.from_rotvec(self.orientation_correction).as_matrix() @ desired_R
        q, error = self.kin.solve(
            bR.T @ (self.command + self.correction - bp),
            bR.T @ command_R,
            dict(zip(d.names[side], d.commands[side])),
            profile["arm_lower_rad"],
            profile["arm_upper_rad"],
        )
        if error > 0.004:
            raise RuntimeError(f"Unreachable bounded arm target: {error:.5f}m")
        d.commands[side][self.ids] += np.clip(
            q - d.commands[side][self.ids],
            -self.command_speed_rad_s / CONTROL_HZ,
            self.command_speed_rad_s / CONTROL_HZ,
        )
        actual_R = d.tool_world(side)[1]
        target_R = self.R if R is None else R
        angular_error = np.linalg.norm(Rotation.from_matrix(target_R @ actual_R.T).as_rotvec())
        return (
            np.linalg.norm(actual - goal) < 0.003
            and np.linalg.norm(self.command - goal) < 1e-5
            and angular_error < 0.03
        )


class NativeReadbackAudit:
    """Read-only cross-check of individual and native bulk product getters."""

    def __init__(self, world, bodies, name):
        from isaacsim.core.prims import RigidPrim

        self.bodies = list(bodies)
        self.view = world.scene.add(
            RigidPrim(
                prim_paths_expr=[b.prim_path for b in self.bodies],
                name=name,
                reset_xform_properties=False,
                prepare_contact_sensors=False,
                disable_stablization=False,
            )
        )
        self.checks, self.previous = [], None

    def initialize(self):
        self.view.initialize()
        self.paths = list(self.view.prim_paths)

    def check(self, t):
        if self.previous is not None and t - self.previous < 1 - 1e-6:
            return
        assert self.view.is_physics_handle_valid(), "Native product view unavailable"
        positions, quats = map(np.asarray, self.view.get_world_poses())
        rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        linear = np.asarray(self.view.get_linear_velocities())
        angular = np.asarray(self.view.get_angular_velocities())
        pose_error = velocity_error = 0.0
        for b in self.bodies:
            i = self.paths.index(b.prim_path)
            p, q = b.get_world_pose()
            R = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
            pose_error = max(
                pose_error, float(abs(p - positions[i]).max()), float(abs(R - rotations[i]).max())
            )
            velocity_error = max(
                velocity_error,
                float(abs(b.get_linear_velocity() - linear[i]).max()),
                float(abs(b.get_angular_velocity() - angular[i]).max()),
            )
        assert pose_error < 1e-7 and velocity_error < 1e-7
        self.checks.append(
            dict(
                time_s=t,
                bodies=len(self.bodies),
                max_pose_error=pose_error,
                max_velocity_error=velocity_error,
            )
        )
        self.previous = t


def set_visual_target(body, position):
    """Update non-physical process graphics through the supported Fabric API.

    These views have no rigid-body or collision schema and are never scored
    products. Runtime rendering reads Fabric; USD-only writes can remain stale.
    """
    # Maintain both authoring and runtime representations: a later scene-sync
    # must not restore the original authoring pose over the Fabric target.
    positions = np.asarray(position, dtype=float)[None, :]
    body.set_world_poses(positions=positions, usd=True)
    body.set_world_poses(positions=positions, usd=False)


class StatusDisplay:
    """Non-physical screen panels, synchronized through Fabric transforms.

    These are process indicators, never scored products or physical fixtures.
    Preloaded textures avoid material recompilation during an episode.
    """

    def __init__(self, scene, name, messages):
        from isaacsim.core.prims import XFormPrim
        from pxr import UsdGeom

        original = scene.stage.GetPrimAtPath("/World/Signs/" + name)
        self.front = np.array(UsdGeom.Xformable(original).GetOrderedXformOps()[0].Get())
        points = np.asarray(UsdGeom.Mesh(original).GetPointsAttr().Get())
        extent = np.ptp(points, axis=0)
        size = [float(extent[0]), float(extent[2])]
        scene.stage.RemovePrim(original.GetPath())
        self.bodies = {}
        self.state = next(iter(messages))
        for i, (state, text) in enumerate(messages.items()):
            child = name + "State" + str(i)
            pos = self.front + [0, 0 if i == 0 else 0.06, 0]
            scene.sign(child, text, pos, size)
            prim = scene.stage.GetPrimAtPath("/World/Signs/" + child)
            self.bodies[state] = scene.world.scene.add(XFormPrim(str(prim.GetPath()), name=child))

    def set(self, state):
        if state != self.state:
            for key, body in self.bodies.items():
                set_visual_target(body, self.front + [0, 0 if key == state else 0.06, 0])
            self.state = state


class LeakVisual:
    """Animated mesh symbols for process state, with no physics/collision API.

    Fixed transform and explicit point-buffer updates avoid stale instance
    transforms on tiny process graphics. This is not a fluid simulation.
    """

    def __init__(self, scene, offset_xy=(0.0, 0.0)):
        from pxr import UsdGeom, UsdShade

        # Floor-plane shift of the leak nipple (Task44's seeded valve-line offset).
        self.offset = np.r_[np.asarray(offset_xy, float), 0.0]
        for i in range(11):
            scene.stage.RemovePrim(f"/World/Leak/Drop{i}")
        mesh = UsdGeom.Mesh.Define(scene.stage, "/World/Leak/ProcessJet")
        self.symbol = np.array(
            [
                [0.003, 0, 0],
                [-0.003, 0, 0],
                [0, 0.003, 0],
                [0, -0.003, 0],
                [0, 0, 0.003],
                [0, 0, -0.003],
            ]
        )
        faces = np.array([4, 0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 5, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3])
        mesh.CreateFaceVertexCountsAttr([3] * 8 * 16)
        mesh.CreateFaceVertexIndicesAttr(
            np.concatenate([faces + 6 * i for i in range(16)]).tolist()
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        self.points = mesh.CreatePointsAttr()
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            scene.appearance.material("water").material
        )
        self.update(False, 0)

    def update(self, leaking, age):
        if leaking:
            u = (age / 0.19 + np.arange(16) / 16) % 1
            centers = np.column_stack(
                [np.full(16, 0.44), 0.413 - 0.05 * u, 0.96216751 - 0.18 * u * u]
            ) + self.offset
            points = (centers[:, None, :] + self.symbol[None, :, :]).reshape(-1, 3)
        else:
            # Degenerate off-screen triangles are absent from the visible jet;
            # no visibility/material or physics state is used to hide a product.
            points = np.tile([0.44, 0.46, -10.0], (96, 1))
        self.points.Set(points.tolist())
