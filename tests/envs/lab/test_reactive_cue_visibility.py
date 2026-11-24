"""B1-B7: the alarm cues are large in the RoboDojo head image, and the instructions name them.

Simulator-free: lamp positions and sizes come from the scene modules' constants and are
projected with the rig spec (roboclock.envs.robodojo_rig), as rig_cameras renders them.
"""

import math

import numpy as np
import pytest

from roboclock.config import load_config
from roboclock.envs import robodojo_rig
from roboclock.envs.lab import hotplate_scene, reactive_scene, safety_instruments
from roboclock.envs.lab.layout import apparatus_frame, indicator_position, make_layout
from roboclock.envs.lab.robot import rig_pose
from roboclock.envs.lab.tasks.registry import make_task

TASKS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
MIN_LAMP_PX = 10.0  # a lamp a policy can act on at 640x480


def head_pixels(config, task_id, point, size):
    """(u, v) of a world point in the head image and the pixels `size` metres span there."""
    origin, yaw = rig_pose(config, task_id)
    pose = robodojo_rig.head_camera_world(origin, math.degrees(yaw))
    matrix = robodojo_rig.intrinsics(robodojo_rig.HEAD_CAMERA)
    local = pose[:3, :3].T @ (np.asarray(point, dtype=float) - pose[:3, 3])
    depth = -local[2]  # USD cameras look along -z
    u = matrix[0, 0] * local[0] / depth + matrix[0, 2]
    v = matrix[1, 1] * -local[1] / depth + matrix[1, 2]
    return (u, v), matrix[0, 0] * size / depth


SEEDS = range(8)  # B2/B3/B5-B7 apparatus are placed per seed (layout.APPARATUS_JITTER)


def bound(task_id, config, seed=0):
    task = make_task(task_id, "easy", seed, config)
    task.bind(make_layout(task, config))
    return task


def lamps(task_id, config, seed=0):
    """(name, world centre, diameter) of the task's alarm lamps."""
    task = bound(task_id, config, seed)
    z = task.layout.table_height
    frame = apparatus_frame(task.layout)
    state = (
        "STATE lamp",
        indicator_position(task) + [0, 0, 0.053],
        2 * reactive_scene.STATE_LAMP_RADIUS,
    )
    if task_id == "B1":
        centre = task.layout.heating_zone.position.copy()
        centre[:2] += hotplate_scene.SCREEN_XY
        centre[2] = z + hotplate_scene.PANEL_TOP
        return [state, ("hotplate screen", centre, min(hotplate_scene.SCREEN_SIZE))]
    if task_id == "B2":
        return [
            (
                "status LED",
                reactive_scene.cue_position("B2", z, frame),
                2 * reactive_scene.STATUS_LED_RADIUS,
            )
        ]
    if task_id in ("B3", "B4"):
        return [state]
    if task_id == "B5":
        return [
            (
                "door lamp",
                safety_instruments.door_lamp_position(z, frame),
                2 * safety_instruments.DOOR_LAMP_RADIUS,
            )
        ]
    if task_id == "B6":
        return [
            (name, safety_instruments.gauge_console_point(local, z, frame), 2 * radius)
            for name, (local, radius) in (
                ("pressure lamp", safety_instruments.PRESSURE_LAMP),
                ("beacon", safety_instruments.BEACON),
            )
        ]
    return [
        (
            "LEVEL lamp",
            safety_instruments.level_lamp_position(z, frame),
            2 * safety_instruments.LEVEL_LAMP_RADIUS,
        )
    ]


@pytest.mark.parametrize("task_id", TASKS)
def test_alarm_lamps_are_inside_the_head_image_and_at_least_ten_pixels(task_id):
    config = load_config()
    width, height = robodojo_rig.HEAD_CAMERA["resolution"]
    for seed in SEEDS:
        for name, centre, diameter in lamps(task_id, config, seed):
            (u, v), pixels = head_pixels(config, task_id, centre, diameter)
            assert 20 < u < width - 20 and 20 < v < height - 20, (task_id, seed, name, u, v)
            assert pixels >= MIN_LAMP_PX, (task_id, seed, name, pixels)


def test_gauge_console_faces_the_head_camera_and_its_beacon_is_clear_of_the_pump():
    config = load_config()
    z = config["scene"]["table_height"]
    origin, yaw = rig_pose(config, "B6")
    camera = robodojo_rig.head_camera_world(origin, math.degrees(yaw))[:3, 3]
    centre = safety_instruments.gauge_console_point((0, 0, 0), z)
    normal = safety_instruments.gauge_console_point((0, 1, 0), z) - centre
    to_camera = (camera - centre) / np.linalg.norm(camera - centre)
    # The dial used to face +y, 75 degrees off the line of sight.
    assert np.degrees(np.arccos(normal @ to_camera)) < 15
    # The beacon used to stand inside the pump motor (a cylinder of radius 0.045 m along x
    # at (0.40, -0.20), 0.17 m above the bench).
    beacon = safety_instruments.gauge_console_point(safety_instruments.BEACON[0], z)
    assert np.linalg.norm(beacon[1:] - [-0.20, z + 0.17]) > 0.045 + safety_instruments.BEACON[1]
    assert beacon[2] > z


def test_level_lamp_is_on_the_console_top():
    x, y = safety_instruments.LEVEL_MONITOR_XY
    lamp = safety_instruments.level_lamp_position(0.75)
    radius = 1.35 * safety_instruments.LEVEL_LAMP_RADIUS  # its bezel
    assert abs(lamp[0] - x) + radius < 0.105 / 2 and abs(lamp[1] - y) + radius < 0.155 / 2


@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_instructions_fit_the_observation(task_id, difficulty):
    assert len(make_task(task_id, difficulty, 0, load_config()).gen_instruction()) <= 512


@pytest.mark.parametrize("task_id", ["B3", "B4"])
def test_instruction_names_the_state_lamp(task_id):
    # The STATE lamp turns red at these tasks' events (ReactiveTask.process).
    text = make_task(task_id, "easy", 0, load_config()).gen_instruction()
    assert "STATE lamp" in text and "blue to red" in text
