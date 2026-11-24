"""Pure-Python ARX X5 kinematics from the URDF, usable outside Isaac Sim.

Custom policies receive only the observation, so an arm's base is not given to them.
`calibrated` recovers it from one observed joint state and its tool pose.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.lab.robot import ARM_DOF, ASSET_DIR, JOINT_HIGH, JOINT_LOW

JOINTS = tuple(f"joint{i}" for i in range(1, ARM_DOF + 1))
TOOL_JOINT = "tool_frame"


def transform(rotation, position):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def origin_transform(joint):
    origin = joint.find("origin")
    xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
    rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
    return transform(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)


def pose_matrix(position, quaternion_wxyz):
    w, x, y, z = quaternion_wxyz
    return transform(Rotation.from_quat([x, y, z, w]).as_matrix(), position)


def rotation_error(target, current):
    return Rotation.from_matrix(target @ current.T).as_rotvec()


class X5Kinematics:
    def __init__(self, urdf=ASSET_DIR / "arx_x5.urdf", base=None):
        joints = {joint.get("name"): joint for joint in ET.parse(urdf).getroot().iter("joint")}
        self.origins = [origin_transform(joints[name]) for name in JOINTS]
        self.axes = [
            np.array([float(value) for value in joints[name].find("axis").get("xyz").split()])
            for name in JOINTS
        ]
        self.tool = origin_transform(joints[TOOL_JOINT])
        self.base = np.eye(4) if base is None else base
        self.low, self.high = JOINT_LOW[:ARM_DOF].astype(float), JOINT_HIGH[:ARM_DOF].astype(float)

    def calibrated(self, joints, position, quaternion_wxyz):
        """A copy whose base places `joints` exactly at the observed world tool pose."""
        base = pose_matrix(position, quaternion_wxyz) @ np.linalg.inv(
            self.tool_pose(joints, base=False)
        )
        result = X5Kinematics.__new__(X5Kinematics)
        result.__dict__.update(self.__dict__, base=base)
        return result

    def tool_pose(self, joints, base=True):
        return self._chain(joints, base)[0]

    def jacobian(self, joints):
        """Tool pose and the 6xN world Jacobian [linear, angular]."""
        pose, origins, axes = self._chain(joints, True)
        linear = np.cross(axes, pose[:3, 3] - origins).T
        return pose, np.vstack([linear, axes.T])

    def _chain(self, joints, base):
        pose = self.base.copy() if base else np.eye(4)
        origins, axes = [], []
        for value, origin, axis in zip(joints, self.origins, self.axes):
            pose = pose @ origin
            origins.append(pose[:3, 3].copy())
            axes.append(pose[:3, :3] @ axis)
            pose = pose @ transform(Rotation.from_rotvec(axis * value).as_matrix(), np.zeros(3))
        return pose @ self.tool, np.asarray(origins), np.asarray(axes)

    def solve(self, target, seed, *, orientation_weight=0.3, damping=0.02, iterations=60):
        """Damped least squares towards a 4x4 world tool pose, within the lab joint limits.

        Returns the joints and the remaining position error in metres; an unreachable
        target stops at the closest pose found rather than failing.
        """
        joints = np.clip(np.asarray(seed, dtype=float), self.low, self.high)
        weights = np.array([1.0, 1.0, 1.0, *([orientation_weight] * 3)])
        best, best_error = joints.copy(), np.inf
        for _ in range(iterations):
            pose, jacobian = self.jacobian(joints)
            error = np.r_[target[:3, 3] - pose[:3, 3], rotation_error(target[:3, :3], pose[:3, :3])]
            position_error = float(np.linalg.norm(error[:3]))
            score = position_error + 0.1 * float(np.linalg.norm(error[3:]))
            if score < best_error:
                best, best_error = joints.copy(), score
            if position_error < 5e-4 and np.linalg.norm(error[3:]) < 5e-3:
                break
            weighted = jacobian * weights[:, None]
            hessian = weighted.T @ weighted + damping**2 * np.eye(len(joints))
            step = np.linalg.solve(hessian, weighted.T @ (error * weights))
            joints = np.clip(joints + np.clip(step, -0.3, 0.3), self.low, self.high)
        position = self.tool_pose(best)[:3, 3]
        return best, float(np.linalg.norm(target[:3, 3] - position))
