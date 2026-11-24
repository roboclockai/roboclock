"""RoboDojo rig cameras and head-view placements of the sports tasks (no simulator).

The rendered checks (key-object margins at every reset, head/wrist stills and videos)
are in the episode metadata and the recording evidence; these tests keep the
geometry that makes the key objects and cues visible from regressing.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from roboclock.envs import robodojo_rig as spec
from roboclock.envs.sports import head_view, rig_cameras
from roboclock.envs.sports.arena import coach_display
from roboclock.envs.sports.tasks import native_config

IDS = ("A1", "A2", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3")
DERIVED_URDF = (
    Path(rig_cameras.__file__).parent / "common/assets/dual_arx_x5/dual_arx_x5.urdf"
)


@pytest.mark.parametrize("task_id", IDS)
def test_head_camera_is_the_robodojo_pose_on_the_task_rig(task_id):
    config = native_config(task_id)
    assert config["embodiment"]["model"] == "dual_arx_x5"
    origin, yaw = rig_cameras.rig_placement(config)
    head = rig_cameras.head_world(config)
    np.testing.assert_allclose(head, spec.head_camera_world(origin, yaw), atol=1e-12)
    look = -head[:3, 2]
    assert math.degrees(math.asin(-look[2])) == pytest.approx(spec.HEAD_PITCH_DOWN_DEG)
    left, right = spec.base_positions(origin, yaw)
    assert np.linalg.norm(left - right) == pytest.approx(0.60)


def test_effective_intrinsics_keep_square_pixels():
    k = head_view.effective_intrinsics()
    assert k[0, 0] == pytest.approx(640 * 10 / 22.212) == pytest.approx(k[1, 1])
    config = native_config("B4")
    head = rig_cameras.head_world(config)
    uv, depth = head_view.project(head[:3, 3] - 2.0 * head[:3, 2], config)
    np.testing.assert_allclose(uv[0], [320, 240], atol=1e-9)
    assert depth[0] == pytest.approx(2.0)


def test_every_task_lists_key_objects_and_display_moves():
    assert set(head_view.KEY_OBJECTS) == set(IDS)
    assert set(head_view.DISPLAYS) == set(IDS) - {"B1"}
    # Only B3's bar (plates beyond the image border) is a documented exception.
    assert set(head_view.EXCEPTIONS) == {"B3"} and set(head_view.EXCEPTIONS["B3"]) == {"bar"}


@pytest.mark.parametrize(
    "task_id", [t for t in IDS if head_view.DISPLAYS.get(t)]
)
def test_display_moves_face_the_head_camera(task_id):
    config = native_config(task_id)
    head = rig_cameras.head_world(config)
    for move in head_view.DISPLAYS[task_id]:
        pivot = np.zeros(3) if isinstance(move["pivot"], str) else np.asarray(move["pivot"])
        matrix = head_view.move_matrix(dict(move, pivot=pivot), config)
        scale = move.get("scale", 1.0)
        rotation = matrix[:3, :3] / scale
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(matrix[:3, 3] + matrix[:3, :3] @ pivot, move["center"], atol=1e-9)
        normal = rotation @ np.array([0.0, -1.0, 0.0])
        to_eye = head[:3, 3] - move["center"]
        assert normal @ to_eye / np.linalg.norm(to_eye) == pytest.approx(1.0)
        # Text rows run parallel to the image rows.
        assert (rotation @ np.array([1.0, 0, 0])) @ head[:3, 1] == pytest.approx(0.0, abs=1e-9)
        # The panel centre projects well inside the head image.
        assert head_view.margin([move["center"]], config) > 60


def test_b2_coach_screen_is_a_floor_monitor_inside_the_head_view():
    config = native_config("B2")
    corners = coach_display.screen_corners()
    assert head_view.margin(corners, config) > 100
    eye = rig_cameras.head_world(config)[:3, 3]
    to_eye = eye - corners.mean(0)
    assert coach_display.facing() @ to_eye / np.linalg.norm(to_eye) == pytest.approx(1.0)
    # Below and between the pads (and the gloves striking them) in the head image.
    pads = [[x, s * y, z] for x in (0.567, 0.63) for s in (-1, 1) for y in (0.157, 0.303) for z in (0.964, 1.156)]
    pad_uv, _ = head_view.project(pads, config)
    screen_uv, _ = head_view.project(corners, config)
    assert screen_uv[:, 1].min() > pad_uv[:, 1].max() + 50
    assert corners[:, 2].max() < 0.7  # far below the 0.92 m arm mounts


def test_b2_coach_screen_stays_in_the_presentation_view():
    profile = native_config("B2")["visuals"]["presentation_camera"]
    eye = np.asarray(profile["eye"], float)
    forward = np.asarray(profile["target"]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    d = coach_display.screen_corners() - eye
    half = 0.036 / (2 * profile["focal_length"])
    x = (d @ right) / (d @ forward) / half
    y = (d @ up) / (d @ forward) / (half * 3 / 4)
    assert np.abs(x).max() < 0.97 and np.abs(y).max() < 0.97


def test_b4_signal_is_inside_the_head_view():
    housing = head_view.box_corners([0.8825, -0.06, 0.90], [0.9575, 0.06, 1.19])
    assert head_view.margin(housing, native_config("B4")) >= head_view.MARGIN_PX


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_c1_object_balls_start_inside_the_head_view(difficulty):
    config = native_config("C1", difficulty, 3)
    r = config["rack"]["radius"]
    for position in config["rack"]["sources"].values():
        corners = head_view.box_corners(np.subtract(position, r), np.add(position, r))
        assert head_view.margin(corners, config) >= head_view.MARGIN_PX
    # Caddies stay clear of the rack template and inside the table's side rails.
    ys = [abs(p[1]) for p in config["rack"]["sources"].values()]
    assert min(ys) - r > 0.164 + 0.15 and max(ys) + r < 0.62


def test_x5_link6_chain_matches_robodojo_x5a_urdf():
    derived = {j.get("name"): j for j in ET.parse(DERIVED_URDF).getroot().iter("joint")}
    reference = {j.get("name"): j for j in ET.parse(rig_cameras.X5A_URDF).getroot().iter("joint")}
    for name in (*rig_cameras.X5_CHAIN, "hand_to_camera_mount", "camera_joint"):
        for side in ("left", "right"):
            ours, theirs = derived[f"{side}_{name}"], reference[name]
            for tag, attr in (("origin", "xyz"), ("origin", "rpy"), ("axis", "xyz")):
                a, b = ours.find(tag), theirs.find(tag)
                if a is None and b is None:
                    continue
                np.testing.assert_allclose(
                    [float(v) for v in a.get(attr).split()],
                    [float(v) for v in b.get(attr).split()],
                    atol=1e-9,
                    err_msg=f"{side}_{name} {tag} {attr}",
                )


def test_link6_kinematics_and_wrist_mount():
    chain = rig_cameras.x5a_chain()
    home = rig_cameras.link6_in_base(np.zeros(6), chain)
    expected = np.eye(4)
    for origin, _ in chain:
        expected = expected @ origin
    np.testing.assert_allclose(home, expected, atol=1e-12)
    # Joint 1 turns link6 about the base's vertical axis.
    turned = rig_cameras.link6_in_base([0.5, 0, 0, 0, 0, 0], chain)
    np.testing.assert_allclose(
        turned[:3, 3], spec._rz(0.5) @ home[:3, 3], atol=1e-12
    )
    error = rig_cameras.pose_error(home @ spec.wrist_camera_in_link6(), home @ spec.wrist_camera_in_link6())
    assert error == (0.0, 0.0)


def test_observation_shapes_and_blank_images():
    shapes = rig_cameras.observation_shapes(depth=True)
    assert shapes == {
        "rgb": (480, 640, 3),
        "left_wrist_rgb": (480, 640, 3),
        "right_wrist_rgb": (480, 640, 3),
        "depth": (480, 640),
        "left_wrist_depth": (480, 640),
        "right_wrist_depth": (480, 640),
    }
    blank = rig_cameras.blank_images(depth=True)
    assert blank["rgb"].dtype == np.uint8 and blank["depth"].dtype == np.float32
    assert not any(image.any() for image in blank.values())
