"""Retarget the sourced NVIDIA biped mesh; visual skinning never moves the bar."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def align(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis, dot = np.cross(a, b), np.clip(np.dot(a, b), -1.0, 1.0)
    if np.linalg.norm(axis) < 1e-8:
        if dot > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(dot)).as_matrix()


@lru_cache(maxsize=1)
def grip_pose():
    return json.loads((Path(__file__).parent / "assets/human/grip_pose.json").read_text())


class Athlete:
    def __init__(self, b):
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, Usd, UsdGeom, UsdShade, UsdSkel, Vt

        from roboclock.envs.sports.arcade.refinement import pbr

        self.b = b
        root = add_reference_to_stage(
            str((Path(__file__).parent / "assets/human/biped_demo_meters.usd").resolve()),
            "/World/Athlete",
        )
        root.SetTypeName("SkelRoot")
        self.base = np.eye(4)
        self.base[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.base[:3, 3] = [1.83, 0.0, 0.62]
        self.xf = UsdGeom.Xformable(root)
        self.xf.ClearXformOpOrder()
        self.transform_op = self.xf.AddTransformOp()
        self.transform_op.Set(Gf.Matrix4d(self.base.T.tolist()))
        self.sk = next(UsdSkel.Skeleton(p) for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Skeleton))
        self.joints = list(self.sk.GetJointsAttr().Get())
        self.parents = np.asarray(UsdSkel.Topology(self.joints).GetParentIndices())
        self.rest = np.array([np.array(m).T for m in self.sk.GetRestTransformsAttr().Get()])
        self.ids = {j.rsplit("/", 1)[-1]: i for i, j in enumerate(self.joints)}
        self.anim = UsdSkel.Animation.Define(b.stage, "/World/Athlete/BenchAnimation")
        self.anim.CreateJointsAttr(self.joints)
        UsdSkel.BindingAPI.Apply(self.sk.GetPrim()).CreateAnimationSourceRel().SetTargets(
            [self.anim.GetPath()]
        )
        UsdSkel.BindingAPI.Apply(root).CreateAnimationSourceRel().SetTargets([self.anim.GetPath()])
        for name, color, rough in [
            ("athlete_skin", [0.52, 0.30, 0.19], 0.62),
            ("athlete_shirt", [0.035, 0.20, 0.23], 0.85),
            ("athlete_pants", [0.045, 0.055, 0.08], 0.9),
            ("athlete_shoes", [0.15, 0.16, 0.17], 0.75),
        ]:
            pbr(
                b.appearance,
                name,
                color,
                roughness=rough,
                texture="leather_white" if name in ("athlete_shirt", "athlete_pants") else None,
                uv=(8, 8),
            )
        # All source materials are replaced below. Disable their unused shaders
        # so the original asset's absent relative texture files are not loaded.
        for prim in list(Usd.PrimRange(root)):
            if prim.IsA(UsdShade.Shader):
                prim.SetActive(False)
        self.hand_surfaces = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                b.appearance.material("athlete_skin").material
            )
            pts = np.asarray(mesh.GetPointsAttr().Get())
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            ids = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            from .hand_surface import HandSurface

            binding = UsdSkel.BindingAPI(prim)
            joint_map = np.array([self.joints.index(j) for j in binding.GetJointsAttr().Get()])
            influences = binding.GetJointIndicesPrimvar().GetElementSize()
            skin_indices = joint_map[
                np.array(binding.GetJointIndicesPrimvar().Get()).reshape(-1, influences)
            ]
            skin_weights = np.array(binding.GetJointWeightsPrimvar().Get()).reshape(-1, influences)
            faces = []
            start = 0
            for count in counts:
                face = ids[start : start + count]
                faces.extend((face[0], face[k], face[k + 1]) for k in range(1, count - 1))
                start += count
            bind = np.array([np.array(m).T for m in self.sk.GetBindTransformsAttr().Get()])
            self.hand_surfaces.append(
                (
                    mesh,
                    HandSurface(
                        pts, skin_indices, skin_weights, bind, self.joints, np.array(faces)
                    ),
                )
            )
            centers = (
                np.add.reduceat(pts[ids], np.r_[0, np.cumsum(counts)[:-1]], axis=0)
                / counts[:, None]
            )
            material = np.zeros(len(counts), int)
            material[
                (centers[:, 1] > 0.95) & (centers[:, 1] < 1.50) & (np.abs(centers[:, 0]) < 0.34)
            ] = 1
            material[(centers[:, 1] > 0.12) & (centers[:, 1] <= 0.95)] = 2
            material[centers[:, 1] <= 0.12] = 3
            for i, name in enumerate(
                ("athlete_skin", "athlete_shirt", "athlete_pants", "athlete_shoes")
            ):
                subset = UsdGeom.Subset.CreateGeomSubset(
                    mesh,
                    name,
                    UsdGeom.Tokens.face,
                    Vt.IntArray(np.where(material == i)[0].tolist()),
                    "materialBind",
                    UsdGeom.Tokens.partition,
                )
                UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
                    b.appearance.material(name).material
                )
        self.update(b.bar_home, False)

    def fk(self, local):
        out = np.zeros_like(local)
        for i, parent in enumerate(self.parents):
            out[i] = (self.base if parent < 0 else out[parent]) @ local[i]
        return out

    def point_bone(self, local, name, child, target):
        i, j = self.ids[name], self.ids[child]
        glob = self.fk(local)
        rot = (
            align(glob[j, :3, 3] - glob[i, :3, 3], np.asarray(target) - glob[i, :3, 3])
            @ glob[i, :3, :3]
        )
        parent = self.parents[i]
        local[i, :3, :3] = np.linalg.solve((self.base if parent < 0 else glob[parent])[:3, :3], rot)

    def limb(self, local, names, target, pole):
        a, mid, end = names
        glob = self.fk(local)
        p, q, r = [glob[self.ids[n], :3, 3] for n in names]
        l1, l2 = np.linalg.norm(q - p), np.linalg.norm(r - q)
        vec = np.asarray(target) - p
        length = np.clip(np.linalg.norm(vec), abs(l1 - l2) + 0.001, l1 + l2 - 0.001)
        axis = vec / np.linalg.norm(vec)
        bend = np.asarray(pole) - p
        bend -= axis * np.dot(bend, axis)
        bend /= np.linalg.norm(bend)
        d = (l1 * l1 - l2 * l2 + length * length) / (2 * length)
        elbow = p + axis * d + bend * np.sqrt(max(0, l1 * l1 - d * d))
        self.point_bone(local, a, mid, elbow)
        self.point_bone(local, mid, end, p + axis * length)

    # The source has 22 cm hands. Scale only the wrist subtree to 18 cm;
    # arm lengths and the source USD / bind poses remain unchanged.
    hand_scale = 0.82

    def hand_frame(self, side):
        """Anatomical distal/radial frame, including mirrored source wrists."""
        rest = self.fk(self.rest)

        def pos(n):
            return rest[self.ids[side + "_" + n], :3, 3]

        distal = pos("MiddleFinger2") - pos("Wrist")
        distal /= np.linalg.norm(distal)
        radial = pos("IndexFinger2") - pos("PinkyFinger2")
        radial -= distal * np.dot(radial, distal)
        radial /= np.linalg.norm(radial)
        return np.column_stack((distal, radial, np.cross(distal, radial)))

    def hand(self, local, side, sign, bar_rotation, opening, thumb_escape=0.0):
        """Pronated power grip: palm below shaft, thumb opposed to fingers.

        Finger1 is a metacarpal in this rig, NOT the proximal phalanx.
        Finger2/3/4 are MCP/PIP/DIP; all three require independent flexion.
        """
        if 0.0 < opening < 1.0:
            closed, opened = local.copy(), local.copy()
            self.hand(closed, side, sign, bar_rotation, 0.0, thumb_escape)
            self.hand(opened, side, sign, bar_rotation, 1.0, thumb_escape)
            for j, name in enumerate(self.joints):
                if side + "_Wrist" not in name:
                    continue
                scale = np.linalg.norm(closed[j, :3, :3], axis=0)
                start = closed[j, :3, :3] / scale
                end = opened[j, :3, :3] / scale
                delta = Rotation.from_matrix(start.T @ end).as_rotvec()
                # Uncurl from fingertip to knuckle; simultaneous interpolation
                # sweeps the middle phalanges through the shaft.
                alpha = opening
                if "Thumb" in name:
                    alpha = np.clip(opening / 0.4, 0.0, 1.0)
                elif "Finger" in name and not name.endswith("End"):
                    onset, duration = {
                        1: (0.6, 0.4),
                        2: (0.4, 0.6),
                        3: (0.15, 0.55),
                        4: (0.0, 0.35),
                    }[int(name[-1])]
                    alpha = np.clip((opening - onset) / duration, 0.0, 1.0)
                local[j, :3, :3] = (start @ Rotation.from_rotvec(delta * alpha).as_matrix()) * scale
            return
        distal = bar_rotation.apply([-1.0, 0.0, 0.0])
        radial = bar_rotation.apply([0.0, -sign, 0.0])
        target = np.column_stack((distal, radial, np.cross(distal, radial)))
        rest = self.fk(self.rest)
        i = self.ids[side + "_Wrist"]
        glob = self.fk(local)
        desired = target @ self.hand_frame(side).T @ rest[i, :3, :3]
        local[i, :3, :3] = np.linalg.solve(glob[self.parents[i], :3, :3], desired * self.hand_scale)
        if opening == 1.0:
            # Briefly unwrap away from the shaft, then restore the sourced
            # anatomical thumb. Never keep the escape pose at rest.
            for segment in range(1, 4):
                name = f"{side}_ThumbFinger{segment}"
                child = (
                    f"{side}_ThumbFinger{segment + 1}" if segment < 3 else f"{side}_ThumbFingerEnd"
                )
                g = self.fk(local)
                pos = g[self.ids[name], :3, 3]
                direction = g[self.ids[child], :3, 3] - pos
                direction /= np.linalg.norm(direction)
                escape = bar_rotation.apply([0.72, -sign * 0.67, -0.18])
                direction = (1 - thumb_escape) * direction + thumb_escape * escape
                self.point_bone(local, name, child, pos + direction)
            return
        # Limit palm arch changes to 12 degrees; most grip flexion belongs
        # to the phalanges, not the metacarpals.
        for finger, headings in grip_pose()["finger_headings_deg"].items():
            name, child = f"{side}_{finger}Finger1", f"{side}_{finger}Finger2"
            g = self.fk(local)
            pos = g[self.ids[name], :3, 3]
            direction = bar_rotation.inv().apply(g[self.ids[child], :3, 3] - pos)
            direction = Rotation.from_euler(
                "y", grip_pose()["metacarpal_pitch_deg"][finger], degrees=True
            ).apply(direction)
            self.point_bone(local, name, child, pos + bar_rotation.apply(direction))
            for segment, heading in enumerate(headings, 2):
                name = f"{side}_{finger}Finger{segment}"
                child = (
                    f"{side}_{finger}Finger{segment + 1}"
                    if segment < 4
                    else f"{side}_{finger}FingerEnd"
                )
                angle = np.deg2rad(heading)
                direction = [-np.cos(angle), 0.0, np.sin(angle)]
                pos = self.fk(local)[self.ids[name], :3, 3]
                self.point_bone(local, name, child, pos + bar_rotation.apply(direction))
        for segment, angles in enumerate(grip_pose()["thumb_elevation_azimuth_deg"], 1):
            elevation, azimuth = np.deg2rad(angles)
            direction = [
                -np.cos(elevation) * np.cos(azimuth),
                -sign * np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
            name = f"{side}_ThumbFinger{segment}"
            child = f"{side}_ThumbFinger{segment + 1}" if segment < 3 else f"{side}_ThumbFingerEnd"
            pos = self.fk(local)[self.ids[name], :3, 3]
            self.point_bone(local, name, child, pos + bar_rotation.apply(direction))

    def pose(self, bar, orientation, release=0.0):
        """Pure skeletal pose; usable by the mesh-clearance audit without Kit."""
        from .process import smooth

        opening = smooth(min(1.0, release / 0.30))
        retreat = smooth(max(0.0, (release - 0.52) / 0.48))
        local = self.rest.copy()
        bar_rotation = Rotation.from_quat(np.roll(orientation, -1))
        grip_rotation = bar_rotation * Rotation.from_euler("y", 90, degrees=True)
        for side, sign in (("L", 1), ("R", -1)):
            # Source L is +X and maps to world +Y; the labels were correct.
            # The old single-vector wrist alignment left palm roll ambiguous.
            offset = np.array(grip_pose()["wrist_offset"])
            offset[1] = sign * self.b.config["workflow"]["human_grip_y"]
            wrist = np.asarray(bar) + grip_rotation.apply(offset)
            withdraw = smooth(max(0.0, (opening - 0.65) / 0.35))
            arc = np.sin(np.pi * np.clip((opening - 0.12) / 0.88, 0.0, 1.0))
            arc_x, arc_z = grip_pose()["release_arc_m"]
            wrist += bar_rotation.apply(
                [-0.045 * withdraw + arc_x * arc, 0.0, -0.16 * withdraw + arc_z * arc]
            )
            wrist[1] += sign * 0.02 * smooth(max(0.0, (release - 0.25) / 0.15))
            wrist = (1 - retreat) * wrist + retreat * np.array([0.80, sign * 0.30, 0.50])
            self.limb(
                local,
                (side + "_UpArm", side + "_LoArm", side + "_Wrist"),
                wrist,
                (1 - retreat) * np.array([0.66, sign * 0.52, 0.72])
                + retreat * np.array([0.36, sign * 1.0, 0.58]),
            )
            relaxed_rotation = bar_rotation * Rotation.from_euler(
                "y", 90 + 90 * smooth(max(0.0, (release - 0.32) / 0.35)), degrees=True
            )
            # Once clear, settle beside the hips with slightly bent elbows and
            # palms inward. The final resting pose is independent of bar tilt.
            rest_rotation = Rotation.from_euler(
                "x", -sign * 75, degrees=True
            ) * Rotation.from_euler("y", 180, degrees=True)
            delta = (relaxed_rotation.inv() * rest_rotation).as_rotvec()
            relaxed_rotation = relaxed_rotation * Rotation.from_rotvec(delta * retreat)
            self.hand(
                local,
                side,
                sign,
                relaxed_rotation,
                opening,
                1.0 - smooth(max(0.0, (release - 0.30) / 0.22)),
            )
            ankle = np.array([1.46, sign * 0.18, 0.10])
            self.limb(
                local,
                (side + "_UpLeg", side + "_LoLeg", side + "_Ankle"),
                ankle,
                [1.35, sign * 0.18, 0.85],
            )
            self.point_bone(local, side + "_Ankle", side + "_Ball", ankle + [0.13, 0.0, -0.045])
        return local

    def update(self, bar, released):
        from .process import smooth

        # Exclude Kit's extra import-time Y-up conversion: base already applies it.
        self.xf.SetXformOpOrder([self.transform_op])
        orientation = (
            getattr(self.b, "last_state", {})
            .get("objects", {})
            .get("bar", {})
            .get("orientation", [1.0, 0.0, 0.0, 0.0])
        )
        release = (
            smooth((self.b.time - self.b.human_effort.release_time) / 0.8) if released else 0.0
        )
        local = self.pose(bar, orientation, release)
        self.apply_pose(local, bar, orientation)

    def apply_pose(self, local, bar, orientation):
        """Author the visual rig, also used by clearly labeled inspection renders."""
        from pxr import Gf, UsdGeom, UsdSkel, Vt

        for mesh, surface in self.hand_surfaces:
            points, self.hand_geometry = surface.apply(self.fk(local), np.asarray(bar), orientation)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.GetNormalsAttr().Set(
                Vt.Vec3fArray.FromNumpy(surface.normals(points).astype(np.float32))
            )
        matrices = Vt.Matrix4dArray([Gf.Matrix4d(m.T.tolist()) for m in local])
        translations, rotations, scales = UsdSkel.DecomposeTransforms(matrices)
        self.anim.CreateTranslationsAttr().Set(translations)
        self.anim.CreateRotationsAttr().Set(rotations)
        self.anim.CreateScalesAttr().Set(scales)
