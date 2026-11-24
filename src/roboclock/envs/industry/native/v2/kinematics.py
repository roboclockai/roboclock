"""Manufacturer URDF kinematics for bounded arm target solving.

This predicts joint targets only. Native contacts, gravity, tool feedback and
the task evaluator remain authoritative; an IK result is not a physical pass.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class UrdfArm:
    def __init__(self, path, body, tool_translation, tool_rotation, arm_names):
        document = ET.parse(path).getroot()
        parents = {j.find("child").get("link"): j for j in document.findall("joint")}
        self.children = {j.get("name"):j.find("child").get("link") for j in document.findall("joint")}
        self.chain = []
        child = body
        while child != "base_link":
            joint = parents[child]
            origin = joint.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            axis = joint.find("axis")
            self.chain.append((joint.get("name"), joint.get("type"), transform,
                               np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else None))
            child = joint.find("parent").get("link")
        self.chain.reverse()
        self.tool = np.eye(4)
        self.tool[:3, 3], self.tool[:3, :3] = tool_translation, tool_rotation
        self.arm_names = list(arm_names)
        side=self.arm_names[0].split("_")[0]
        self.elbow_link=f"{side}_fr3v2_1_link3" if "fr3v2_1" in self.arm_names[0] else None

    def forward(self, values, with_frames=False):
        transform = np.eye(4)
        frames={}
        for name, kind, origin, axis in self.chain:
            transform = transform @ origin
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = Rotation.from_rotvec(axis * values.get(name, 0)).as_matrix()
            elif kind == "prismatic":
                motion[:3, 3] = axis * values.get(name, 0)
            transform = transform @ motion
            frames[self.children[name]]=transform.copy()
        result=transform @ self.tool
        return (result,frames) if with_frames else result

    def solve(self, target_position, target_rotation, values, lower, upper, restarts=0):
        initial = np.array([values[name] for name in self.arm_names])
        lower, upper = np.asarray(lower) + 1e-4, np.asarray(upper) - 1e-4

        def residual(q):
            pose,frames = self.forward({**values, **dict(zip(self.arm_names, q))},with_frames=True)
            error=np.r_[pose[:3, 3] - target_position,
                        .25 * Rotation.from_matrix(target_rotation @ pose[:3, :3].T).as_rotvec()]
            if self.elbow_link:
                # Use FR3's redundant DOF to keep its elbow behind the forward
                # workbench. A position-only IK solution can pierce the stock
                # tray with the forearm while placing the tool correctly.
                error=np.r_[error,.02*(frames[self.elbow_link][0,3]-.43)]
            return error

        rng = np.random.default_rng(41050)
        solutions = []
        for attempt in range(restarts+1):
            seed = np.clip(initial, lower, upper) if attempt == 0 else rng.uniform(lower, upper)
            solved = least_squares(residual, seed, bounds=(lower, upper),
                                   max_nfev=160, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            error = float(np.linalg.norm(residual(solved.x)))
            solutions.append((error, solved.x))
            if error < 1e-5:
                break
        error, q = min(solutions, key=lambda value: value[0])
        return q, float(np.linalg.norm(residual(q)[:6]))
