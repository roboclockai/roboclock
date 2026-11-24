import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from roboclock.envs import rates, robodojo_rig as rig

ROOT = Path(__file__).resolve().parents[1]
X5A_URDF = ROOT / "assets/industry/runtime/arx-x5/robot-v1/source/ARX-X5/X5A.urdf"

# RoboDojo f57e163 world values: env_cfg/robot/dual_x5.yml and env_cfg/camera/camera_config.yml.
ROBODOJO_ORIGIN, ROBODOJO_YAW = (0.0, -0.45, 0.765), 90.0
ROBODOJO_LEFT, ROBODOJO_RIGHT = (-0.3, -0.45, 0.765), (0.3, -0.45, 0.765)
ROBODOJO_HEAD = (0.0, -0.41, 1.308)


def test_bases_and_head_reproduce_robodojo_world_poses():
    left, right = rig.base_positions(ROBODOJO_ORIGIN, ROBODOJO_YAW)
    np.testing.assert_allclose(left, ROBODOJO_LEFT, atol=1e-9)
    np.testing.assert_allclose(right, ROBODOJO_RIGHT, atol=1e-9)
    head = rig.head_camera_world(ROBODOJO_ORIGIN, ROBODOJO_YAW)
    np.testing.assert_allclose(head[:3, 3], ROBODOJO_HEAD, atol=1e-9)
    # RoboDojo sets euler XYZ (30, 0, 0) deg on a USD camera with identity local pose.
    np.testing.assert_allclose(head[:3, :3], rig.euler_xyz(30, 0, 0), atol=1e-9)
    np.testing.assert_allclose(rig.quat_wxyz(head[:3, :3]), [0.96593, 0.25882, 0, 0], atol=1e-5)


def test_head_looks_forward_sixty_degrees_down_without_roll():
    pose = rig.head_camera_in_rig()
    look = -pose[:3, 2]
    assert math.degrees(math.atan2(-look[2], look[0])) == pytest.approx(60.0)
    assert look[1] == pytest.approx(0.0, abs=1e-12)
    assert pose[:3, 0] @ [0, 0, 1] == pytest.approx(0.0, abs=1e-12)  # image x axis stays level


def test_wrist_camera_matches_x5a_urdf_and_robodojo_mount():
    joints = {j.get("name"): j for j in ET.parse(X5A_URDF).getroot().iter("joint")}
    for (position, kind, angles), name in zip(rig._WRIST_CHAIN[:2], ("hand_to_camera_mount", "camera_joint")):
        origin = joints[name].find("origin")
        assert kind == "rpy"
        np.testing.assert_allclose([float(v) for v in origin.get("xyz").split()], position)
        np.testing.assert_allclose([float(v) for v in origin.get("rpy").split()], angles, atol=1e-6)
    assert joints["hand_to_camera_mount"].find("parent").get("link") == rig.WRIST_PARENT_LINK
    pose = rig.wrist_camera_in_link6()
    np.testing.assert_allclose(pose[:3, 3], [0.08484, 0.0, 0.05094], atol=1e-5)
    np.testing.assert_allclose(-pose[:3, 2], [math.cos(math.pi / 6), 0, -0.5], atol=1e-4)  # 30 deg below link6 +x
    np.testing.assert_allclose(pose[:3, 1], [0.5, 0, math.cos(math.pi / 6)], atol=1e-4)


def test_fields_of_view_and_rates():
    # Rendered (square pixels) vs aperture-only values; see RENDERER_SQUARE_PIXELS.
    assert rig.field_of_view_deg(rig.HEAD_CAMERA) == pytest.approx((96.00, 79.59), abs=0.01)
    assert rig.field_of_view_deg(rig.HEAD_CAMERA, square_pixels=False) == pytest.approx((96.00, 71.00), abs=0.01)
    assert rig.field_of_view_deg(rig.WRIST_CAMERA) == pytest.approx((77.73, 62.30), abs=0.01)
    assert rig.field_of_view_deg(rig.WRIST_CAMERA, square_pixels=False) == pytest.approx((77.73, 62.30), abs=0.01)
    head = rig.intrinsics(rig.HEAD_CAMERA)
    assert head[0, 0] == head[1, 1] == pytest.approx(288.13, abs=0.01)  # measured fx = fy
    assert rig.intrinsics(rig.HEAD_CAMERA, square_pixels=False)[1, 1] == pytest.approx(336.46, abs=0.01)
    assert rig.intrinsics(rig.WRIST_CAMERA)[0, 0] == pytest.approx(397.04, abs=0.01)
    assert rig.HEAD_CAMERA["resolution"] == rig.WRIST_CAMERA["resolution"] == (640, 480)
    assert rig.PHYSICS_HZ == rates.PHYSICS_HZ
    assert rig.POLICY_HZ == rates.POLICY_HZ
    assert rig.VIDEO_FPS == rates.VIDEO_FPS
    assert rates.CONTROL_HZ % rig.POLICY_HZ == 0


def test_check_rig_accepts_spec_and_rejects_deviations():
    rig_pose = rig.rig_to_world((1.0, 2.0, 0.9), 35.0)
    left = rig_pose @ rig.transform(rig.LEFT_BASE_IN_RIG_M, np.eye(3))
    right = rig_pose @ rig.transform(rig.RIGHT_BASE_IN_RIG_M, np.eye(3))
    head = rig.head_camera_world((1.0, 2.0, 0.9), 35.0)
    assert rig.check_rig(left, right, head)["yaw_deg"] == pytest.approx(35.0)
    with pytest.raises(ValueError, match="left base"):
        rig.check_rig(right, left, head)
    tilted = head.copy()
    tilted[:3, :3] = tilted[:3, :3] @ rig.euler_xyz(1.0, 0, 0)
    with pytest.raises(ValueError, match="turned"):
        rig.check_rig(left, right, tilted)
    wide = right.copy()
    wide[:3, 3] -= 0.01 * left[:3, 1]
    with pytest.raises(ValueError, match="apart"):
        rig.check_rig(left, wide, head)


def test_describe_records_rendered_intrinsics():
    summary = rig.describe()
    head = np.asarray(summary["head_camera"]["rendered_intrinsics"])
    assert head[0, 0] == head[1, 1] == pytest.approx(288.13, abs=0.01)
    assert summary["head_camera"]["fov_deg"] == pytest.approx((96.00, 79.59), abs=0.01)
    wrist = np.asarray(summary["wrist_camera"]["rendered_intrinsics"])
    assert wrist[0, 0] == pytest.approx(397.04, abs=0.01)
