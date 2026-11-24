"""Single active arm integration for native single and dual-arm variants."""

import numpy as np
from scipy.spatial.transform import Rotation

from .rig import RobotRig


def spawn(backend):
    c = backend.config["embodiment"]
    backend.rig = RobotRig(backend.world, backend.stage, c["model"], c["position"])
    backend.rig.x5_finger_force = c.get("finger_force_n", 30)
    backend.robot = backend.rig.robot
    backend.active_side = c.get("active_side", "right")
    backend.finger_paths = backend.rig.fingers[backend.active_side]
    backend.hand_path = backend.rig.tools[backend.active_side].prim_path
    backend.arm_paths = backend.rig.arm_paths


def initialize(backend):
    rotation = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    targets = {
        "left": (np.array([0.08, 0.48, 1.12]), rotation, 0.04),
        "right": (np.array([0.30, -0.27, 1.06]), rotation, 0.04),
    }
    if backend.rig.model == "dual_arx_x5":
        from .views import observation_pose

        c = backend.config["embodiment"]
        targets["left"] = observation_pose(c["position"], c.get("yaw", 0))
    backend.rig.initialize(
        {s: targets[s] for s in backend.rig.sides},
        zero_arms=backend.config["embodiment"].get("start_pose") == "zero",
    )
    # Reset task objects with the now-known native robot default pose. No object
    # is relocated after settling or episode start.
    backend.world.reset()
    backend.rig.apply()


def width(backend):
    return backend.rig.width(backend.active_side)
