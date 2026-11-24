"""Serial FK/IK from actual USD joint frames, including intervening fixed joints."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def frame(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(
        [*quaternion.GetImaginary(), quaternion.GetReal()]
    ).as_matrix()
    result[:3, 3] = position
    return result


class USDKinematics:
    def __init__(self, stage, joint_paths, tool_path, tcp_offset, tcp_rotation=None):
        from pxr import UsdGeom, UsdPhysics

        cache = UsdGeom.XformCache()
        self.names = []
        self.frames = []
        self.axes = []
        low, high = [], []
        previous = None
        for path in joint_paths:
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
            parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            child = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
            parent_world = np.array(cache.GetLocalToWorldTransform(parent)).T
            if previous is None:
                self.base = parent_world
                bridge = np.eye(4)
            else:
                bridge = np.linalg.inv(previous) @ parent_world
            a = bridge @ frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            b = np.linalg.inv(frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()))
            self.frames.append((a, b))
            self.axes.append(np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())])
            low.append(joint.GetLowerLimitAttr().Get())
            high.append(joint.GetUpperLimitAttr().Get())
            self.names.append(joint.GetPrim().GetName())
            previous = np.array(cache.GetLocalToWorldTransform(child)).T
        tool = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(tool_path))).T
        self.tail = np.linalg.inv(previous) @ tool
        self.offset = np.asarray(tcp_offset, float)
        self.tcp_rotation = np.eye(3) if tcp_rotation is None else np.asarray(tcp_rotation)
        self.low = np.deg2rad(low) + 1e-5
        self.high = np.deg2rad(high) - 1e-5
        self.last = (self.low + self.high) / 2

    def chain(self, joints):
        pose = self.base.copy()
        origins, axes = [], []
        for value, axis, (a, b) in zip(joints, self.axes, self.frames):
            pose = pose @ a
            origins.append(pose[:3, 3].copy())
            axes.append(pose[:3, :3] @ axis)
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            pose = pose @ rotation @ b
        pose = pose @ self.tail
        p = pose[:3, 3] + pose[:3, :3] @ self.offset
        r = pose[:3, :3] @ self.tcp_rotation
        angular = np.asarray(axes).T
        linear = np.cross(np.asarray(axes), p - np.asarray(origins)).T
        return p, r, linear, angular

    def fk(self, joints):
        p, r, _, _ = self.chain(joints)
        return p, r

    def error_and_jacobian(self, joints, position, rotation):
        p, r, linear, angular = self.chain(joints)
        rv = Rotation.from_matrix(rotation @ r.T).as_rotvec()
        theta = np.linalg.norm(rv)
        x, y, z = rv
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        coefficient = 1 / 12 if theta < 1e-5 else (1 - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
        inverse_right = np.eye(3) + 0.5 * skew + coefficient * skew @ skew
        return np.r_[4 * (p - position), 0.6 * rv], np.vstack(
            [4 * linear, -0.6 * inverse_right @ angular]
        )

    def solve(self, position, rotation, seed=None, multistart=False):
        start = self.last if seed is None else np.asarray(seed)
        cached_q, cached_result = None, None

        def evaluate(joints):
            nonlocal cached_q, cached_result
            if cached_q is None or not np.array_equal(joints, cached_q):
                cached_q = joints.copy()
                cached_result = self.error_and_jacobian(joints, position, rotation)
            return cached_result

        def residual(joints):
            return evaluate(joints)[0]

        def jacobian(joints):
            return evaluate(joints)[1]

        seeds = [start]
        if multistart:
            seeds.extend(np.random.default_rng(0).uniform(self.low, self.high, (20, len(start))))
        best = None
        for initial in seeds:
            fit = least_squares(
                residual,
                np.clip(initial, self.low, self.high),
                jac=jacobian,
                bounds=(self.low, self.high),
                max_nfev=160 if multistart else 35,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
            )
            error = float(np.linalg.norm(residual(fit.x)))
            if best is None or error < best[0]:
                best = error, fit.x
            if error < 0.0005:
                break
        self.last = best[1].copy()
        return self.last.copy(), best[0]
