"""Bridge the existing chess oracle's target convention to native robot IK."""

import numpy as np
from scipy.spatial.transform import Rotation


class ChessIKSolver:
    def __init__(self, rig, side):
        self.rig, self.side = rig, side

    def compute_inverse_kinematics(self, target_position, target_orientation):
        from isaacsim.core.utils.types import ArticulationAction

        rotation = Rotation.from_quat(np.roll(target_orientation, -1)).as_matrix()
        ids = self.rig.arm_ids[self.side]
        actual = self.rig.robot.get_joint_positions()
        # The chess oracle's targets carry a 3.4 mm offset from its original arm's IK frame.
        # Undo it and solve for the X5's real grasp frame.
        target = np.asarray(target_position) - [0, 0, 0.0034]
        angles, error = self.rig.kin[self.side].solve(target, rotation, seed=actual[ids])
        return ArticulationAction(joint_positions=angles, joint_indices=np.array(ids)), error < 0.02
