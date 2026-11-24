"""The A tasks' controls and cues must be readable from the rig's head camera with both arms at
rest: a policy sees nothing else. Geometry only; the rendered checks are in the scene files' notes."""

import math

import numpy as np
import pytest

from roboclock.envs import robodojo_rig
from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.rig_cameras import link6_in_base
from roboclock.envs.lab.robot import HOME, mount_positions, rig_pose
from roboclock.envs.lab.tasks.registry import make_task

TABLE = load_config()["scene"]["table_height"]


def head_camera(task_id):
    origin, yaw = rig_pose(load_config(), task_id)
    return robodojo_rig.head_camera_world(origin, math.degrees(yaw))


def head_pixel(task_id, point):
    """Pixel of a world point in the rendered 640x480 head image (square pixels)."""
    x, y, z = (np.linalg.inv(head_camera(task_id)) @ np.r_[np.asarray(point, float), 1.0])[:3]
    k = robodojo_rig.intrinsics(robodojo_rig.HEAD_CAMERA)
    return np.array([k[0, 0] * x / -z + k[0, 2], -k[1, 1] * y / -z + k[1, 2]])


def resting_tool(task_id, side=0):
    """Tool point of an arm in its start pose (side 0 is the rig's right arm, the oracle's)."""
    pose = link6_in_base(HOME[:6])
    mount = np.array(mount_positions(load_config(), task_id)[side])
    return mount + pose[:3, 3] + pose[:3, :3] @ np.array([0.145, 0.0, 0.0])


@pytest.mark.parametrize("task_id", ["A2", "A4"])
def test_valve_grip_is_not_behind_the_resting_right_hand(task_id):
    from roboclock.envs.lab.liquid_scene import LEVER_RADIUS, PIVOTS

    grip = np.array(PIVOTS[task_id]) + [LEVER_RADIUS, 0.0, 0.02]
    hub = np.array(PIVOTS[task_id]) + [0.0, 0.0, 0.012]
    hand = head_pixel(task_id, resting_tool(task_id))
    # The open gripper and its rail image about 45 px either side of the tool point, and the
    # forearm runs up and to the right of it: the valve has to image to its left.
    for point in (grip, hub):
        pixel = head_pixel(task_id, point)
        assert pixel[0] < hand[0] - 55, (task_id, pixel, hand)
        assert 0 < pixel[0] < 640 and 0 < pixel[1] < 480
    # A2's first place, (0.42, -0.25, 1.04), fails this: the grip imaged inside the hand.
    old = head_pixel("A2", np.array([0.42 + LEVER_RADIUS, -0.25, 1.06]))
    assert old[0] > head_pixel("A2", resting_tool("A2"))[0]


def test_valve_stays_within_the_proven_reach_of_the_right_arm():
    from roboclock.envs.lab.liquid_scene import CLOSE_ANGLE, LEVER_RADIUS, PIVOTS

    for task_id in ("A2", "A4"):
        base = np.array(mount_positions(load_config(), task_id)[0])
        pivot = np.array(PIVOTS[task_id])
        for angle in (0.0, CLOSE_ANGLE):
            grip = pivot + LEVER_RADIUS * np.array([math.cos(angle), math.sin(angle), 0.0])
            reach = np.linalg.norm((grip - base)[:2])
            assert 0.22 < reach < 0.42, (task_id, angle, reach)
            assert -0.12 < grip[2] - base[2] < 0.02


def test_funnel_cone_is_seen_through_rather_than_mirrored():
    from roboclock.envs.lab.liquid_scene import FUNNEL_HEIGHTS, FUNNEL_RADII

    lean = math.degrees(
        math.atan2(FUNNEL_RADII[2] - FUNNEL_RADII[1], FUNNEL_HEIGHTS[2] - FUNNEL_HEIGHTS[1])
    )
    # The head camera looks down the cone at about 35 degrees from the vertical. Glass seen more
    # than about 75 degrees off its normal mostly mirrors the room.
    camera = head_camera("A2")[:3, 3]
    funnel = np.array([0.50, 0.13, TABLE + 0.30 + 0.10])
    view = camera - funnel
    from_vertical = math.degrees(math.atan2(np.linalg.norm(view[:2]), view[2]))
    incidence = 90.0 - (from_vertical - lean)
    assert lean < 20.0 and incidence < 72.0, (lean, from_vertical, incidence)
    assert FUNNEL_HEIGHTS[0] == 0.0 and list(FUNNEL_HEIGHTS) == sorted(FUNNEL_HEIGHTS)


def test_column_does_not_stand_between_the_head_camera_and_the_tubes():
    from roboclock.envs.lab.liquid_scene import COLUMN_XY, OUTLET_XY, TUBE_SPACING

    task = make_task("A3", difficulty="easy")
    camera = head_camera("A3")[:2, 3]
    column = np.array(COLUMN_XY)
    for slid in range(len(task.colors) + 1):  # the collector at each of its stops
        for tube in range(len(task.colors) + 1):
            xy = np.array(OUTLET_XY) + [(tube - slid) * TUBE_SPACING, 0.0]
            if xy[0] < column[0]:
                continue  # in front of the column
            ray = (xy - camera) / np.linalg.norm(xy - camera)
            offset = column - camera
            distance = abs(ray[0] * offset[1] - ray[1] * offset[0])
            # Column glass 27 mm in radius, waste beaker 29 mm.
            assert distance > 0.027 + 0.029 + 0.02, (slid, tube, distance)
    # Straight above the outlet, as it stood, the column hid every tube behind it.
    assert abs(OUTLET_XY[1] - camera[1]) < 0.03


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_bed_slices_cover_the_bed_and_reach_its_bottom_at_the_scored_times(difficulty):
    from roboclock.envs.lab.chromatography_visuals import BAND_SPEED, BED_BOTTOM, BED_TOP
    from roboclock.envs.lab.liquid_scene import BAND_SLICES, bed_slice_spans

    task = make_task("A3", difficulty=difficulty)
    per_band = len(BAND_SLICES) + 1  # the white below a band, then its slices

    def spans(time):
        return bed_slice_spans(time, task.intervals, BAND_SPEED, BED_BOTTOM, BED_TOP)

    for time in np.linspace(0.0, task.intervals[-1][1] + 5.0, 97):
        shown = [span for span in spans(time) if span is not None]
        # The slices tile the bed from its bottom to its top; a slice thinner than 0.1 mm is
        # dropped, which leaves that much of the white core showing.
        assert shown[0][0] - BED_BOTTOM < 1.5e-4 and BED_TOP - shown[-1][1] < 1.5e-4
        assert all(0 <= b[0] - a[1] < 1.5e-4 for a, b in zip(shown, shown[1:]))
    for band, (start, end) in enumerate(task.intervals):
        first = band * per_band + 1
        arriving = spans(start + 0.2)
        assert arriving[first - 1] is None  # the white below it has left the bed
        assert arriving[first][0] == BED_BOTTOM  # its leading edge is at the outlet
        assert all(span is None for span in spans(end + 0.05)[first : first + len(BAND_SLICES)])
        # Before the band elutes it stands above the outlet by the time left times the speed.
        ahead = spans(start - 4.0)[first][0] - BED_BOTTOM
        assert ahead == pytest.approx(4.0 * BAND_SPEED, abs=2e-4)


def test_countdown_display_faces_the_head_camera():
    from roboclock.envs.lab.bath_scene import DISPLAY_LEAN_DEG, SIGN_TILT_DEG

    lean = math.radians(DISPLAY_LEAN_DEG)
    normal = np.array([-math.cos(lean), 0.0, math.sin(lean)])
    center = np.array([0.79, 0.15, TABLE + 0.23])
    for task_id in ("A5", "A6"):
        view = head_camera(task_id)[:3, 3] - center
        angle = math.degrees(math.acos(normal @ view / np.linalg.norm(view)))
        assert angle < 15.0, (task_id, angle)
        # 70 mm digits: at least 15 px tall.
        top = head_pixel(task_id, center + 0.035 * np.array([math.sin(lean), 0, math.cos(lean)]))
        bottom = head_pixel(task_id, center - 0.035 * np.array([math.sin(lean), 0, math.cos(lean)]))
        assert bottom[1] - top[1] > 15.0
    assert 0.0 <= SIGN_TILT_DEG <= 45.0


def test_sign_letters_fill_the_label():
    from PIL import Image

    from roboclock.envs.lab.appearance import label_texture, sign_texture

    for text, aspect in (
        ("CLOSE", 0.130 / 0.040),
        ("STOP BATH", 0.28 / 0.052),
        ("F1", 0.07 / 0.046),
    ):
        path = sign_texture(text, aspect=aspect)
        image = np.asarray(Image.open(path).convert("L"))
        assert image.shape[1] == round(256 * aspect)
        inner = image[20:-20, 20:-20] < 100  # the dark letters, inside the border
        rows = np.flatnonzero(inner.any(axis=1))
        assert (rows[-1] - rows[0]) / image.shape[0] > 0.45, text
        assert path != label_texture(text)


def test_titration_floor_steps_from_clear_to_strong_pink():
    from roboclock.envs.lab.titration_visuals import FLOOR_STEPS, floor_color, floor_step

    assert floor_step(0.0) == floor_step(17.6) == 0
    endpoint = floor_step(18.0)
    assert endpoint >= 2  # the endpoint is its own, visibly pink, step
    assert floor_step(20.0) == FLOOR_STEPS - 1
    steps = [floor_step(v) for v in np.linspace(17.0, 20.5, 200)]
    assert steps == sorted(steps)
    clear, pale, strong = floor_color(0), floor_color(endpoint), floor_color(FLOOR_STEPS - 1)
    assert abs(clear[0] - clear[1]) < 0.02  # neutral before the endpoint
    assert 0.80 < pale[1] / clear[1] < 0.92 and pale[0] / clear[0] > 0.98  # pale pink
    assert strong[1] < 0.25 * strong[0] and strong[2] > 2 * strong[1]  # strong pink
    assert clear.max() < 0.5  # the bench lighting renders 0.5 as full white


def test_instructions_name_the_controls_a_policy_can_see():
    config = load_config()
    a1 = make_task("A1", "easy", 0, config).gen_instruction()
    assert "clockwise" in a1 and "knob" in a1
    for task_id in ("A2", "A4"):
        text = make_task(task_id, "easy", 0, config).gen_instruction()
        assert "black grip" in text and "plate marked CLOSE" in text
    a3 = make_task("A3", "easy", 0, config).gen_instruction()
    assert "sliding collector" in a3 and "grip" in a3 and "F1, F2, F3 and F4" in a3
    for task_id in ("A1", "A2", "A3", "A4", "A5", "A6"):
        for level in ("easy", "medium", "hard"):
            assert len(make_task(task_id, level, 0, config).gen_instruction()) <= 512
