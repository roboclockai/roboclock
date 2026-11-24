"""Lab uses the shared RoboDojo rig spec (simulator-free checks)."""

import math

import numpy as np
import pytest

from roboclock.config import load_config
from roboclock.envs import robodojo_rig
from roboclock.envs.lab import rig_cameras
from roboclock.envs.lab.robot import (
    ASSET_DIR,
    HEAD_CAMERA_OFFSET,
    RIG_LEFT_ARM,
    RIG_RIGHT_ARM,
    RIG_SEPARATION,
    mount_positions,
    mount_yaws,
    rig_axes,
)
from roboclock.envs.lab.tasks.registry import TASKS


def test_lab_rig_names_are_the_shared_values():
    assert RIG_SEPARATION == robodojo_rig.ARM_SEPARATION_M
    assert HEAD_CAMERA_OFFSET == tuple(robodojo_rig.HEAD_IN_RIG_M)
    config = load_config()
    rig = config["robot"]["rig"]
    assert not set(rig) & {"separation", "head_camera_offset", "head_camera_pitch_deg"}


@pytest.mark.parametrize("task_id", list(TASKS))
def test_every_task_passes_the_spec_rig_check_and_runs_at_250_hz(task_id):
    config = load_config()
    assert 1 / config["sim"]["physics_dt"] == pytest.approx(robodojo_rig.PHYSICS_HZ)
    mounts = np.asarray(mount_positions(config, task_id))
    yaw = mount_yaws(config, task_id)[0]
    forward, left = rig_axes(yaw)
    # The spec's left arm (left_wrist_rgb) is the lab arm on the rig's +y side.
    assert (mounts[RIG_LEFT_ARM] - mounts[RIG_RIGHT_ARM]) @ left == pytest.approx(0.60)
    bases = [robodojo_rig.rig_to_world(m, math.degrees(yaw)) for m in mounts]
    head = robodojo_rig.head_camera_world(mounts.mean(axis=0), math.degrees(yaw))
    summary = robodojo_rig.check_rig(bases[RIG_LEFT_ARM], bases[RIG_RIGHT_ARM], head)
    np.testing.assert_allclose(summary["midpoint"], mounts.mean(axis=0))
    with pytest.raises(ValueError, match="left base"):
        robodojo_rig.check_rig(bases[RIG_RIGHT_ARM], bases[RIG_LEFT_ARM], head)


def test_lab_x5_link6_matches_x5a_urdf():
    rng = np.random.default_rng(0)
    for _ in range(20):
        joints = rng.uniform(-1.5, 1.5, 6)
        np.testing.assert_allclose(
            rig_cameras.link6_in_base(joints, ASSET_DIR / "arx_x5.urdf"),
            rig_cameras.link6_in_base(joints),
            atol=1e-12,
        )


def test_observation_keys_and_zero_images_follow_the_spec():
    images = rig_cameras.zero_images(depth=True)
    assert set(images) == set(robodojo_rig.OBSERVATION_KEYS.values()) | set(
        robodojo_rig.DEPTH_KEYS.values()
    )
    for key, value in images.items():
        assert value.shape[:2] == (480, 640)
        assert not value.any()
    assert images["rgb"].dtype == np.uint8 and images["depth"].dtype == np.float32
    spaces = rig_cameras.observation_spaces(depth=False)
    assert set(spaces) == {"rgb", "left_wrist_rgb", "right_wrist_rgb"}
    for name, (rgb_key, depth_key, _) in rig_cameras.VIEWS.items():
        assert robodojo_rig.OBSERVATION_KEYS[name] == rgb_key
        assert robodojo_rig.DEPTH_KEYS[name] == depth_key
