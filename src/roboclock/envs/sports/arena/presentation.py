"""Ringside presentation camera with task cues and hands kept in frame."""

import numpy as np

BOXING_EYE = (1.85, -2.65, 1.82)
BOXING_TARGET = (0.28, 0.05, 1.03)
BOXING_FOCAL_LENGTH = 0.032


def boxing_camera(b, evaluation_end=None):
    if b.kind != "B2":
        return
    from roboclock.simulators.isaac.embodiments.presentation import apply_presentation_camera

    if apply_presentation_camera(b):
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    # Keep the camera inside the room's visual envelope; no pull-back exposing
    # the outside HDR dome or miniature cutaway walls at the beginning/end.
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*BOXING_EYE), Gf.Vec3d(*BOXING_TARGET), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
    b.camera.set_world_pose(np.asarray(BOXING_EYE), orientation, camera_axes="usd")
    b.camera.set_focal_length(BOXING_FOCAL_LENGTH)
