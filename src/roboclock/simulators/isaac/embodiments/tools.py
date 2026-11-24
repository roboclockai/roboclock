"""Contact geometry in tool coordinates, independent of the robot wrist axis."""

import numpy as np


def glove_contact_mask(wrist_local_points, glove_rotation=None):
    points = np.asarray(wrist_local_points).reshape(-1, 3)
    if glove_rotation is not None:
        points = points @ np.asarray(glove_rotation)
    return np.sum(((points - [0, 0, 0.174]) / [0.058, 0.054, 0.073]) ** 2, axis=-1) <= 1.15
