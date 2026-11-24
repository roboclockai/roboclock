"""The RoboDojo dual-X5 rig of every hospital task (robot.rig placement, hospital/rig.py,
values from roboclock.envs.robodojo_rig)."""

import copy
import math

import numpy as np
import pytest

from roboclock.envs import robodojo_rig as spec
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.rig import (
    check_measured,
    check_rig,
    head_camera_pose,
    head_camera_world,
    rig_mounts,
    rig_pose,
    x5a_link6_in_base,
)
from roboclock.envs.hospital.tasks.registry import TASKS

ENV_TASKS = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3", "C4")


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_every_env_task_is_on_the_rig(config):
    assert set(ENV_TASKS) == set(TASKS)
    robot = config["robot"]
    assert not set(robot) & {"mount_positions", "task_mount_positions", "task_mount_yaws"}
    assert set(robot["rig"]) == {"default", "tasks"}  # no local copies of spec values
    assert set(robot["rig"]["tasks"]) == set(ENV_TASKS)
    check_rig(config, ENV_TASKS)


def test_rates_and_head_image_follow_the_spec(config):
    sim = config["sim"]
    assert 1 / sim["physics_dt"] == pytest.approx(spec.PHYSICS_HZ)
    assert 1 / (sim["physics_dt"] * sim["control_decimation"]) == pytest.approx(50.0)
    assert tuple(sim["camera_resolution"]) == spec.HEAD_CAMERA["resolution"]
    assert config["visuals"]["renderer"] == spec.RENDER_MODE
    assert config["visuals"]["anti_aliasing"] == 4  # DLAA
    for task in config["tasks"].values():
        assert "physics_dt" not in task and "control_decimation" not in task


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["sim"].update(physics_dt=0.002, control_decimation=10), "1/250"),
        (lambda c: c["sim"].update(camera_resolution=[960, 720]), "camera_resolution"),
    ],
)
def test_config_rejects_off_spec_rates_and_images(tmp_path, config, mutate, message):
    import yaml

    bad = copy.deepcopy(config)
    mutate(bad)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize("task_id", ENV_TASKS)
def test_bases_side_by_side_and_head_camera_at_spec(config, task_id):
    (right, left), yaw = rig_mounts(config, task_id)
    origin, _ = rig_pose(config, task_id)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    leftward = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    delta = left - right
    assert np.linalg.norm(delta) == pytest.approx(spec.ARM_SEPARATION_M)
    assert delta @ leftward == pytest.approx(spec.ARM_SEPARATION_M)  # arm 0 right, arm 1 left
    assert delta @ forward == pytest.approx(0.0, abs=1e-12) and delta[2] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(0.5 * (left + right), origin, atol=1e-12)
    head = head_camera_world(config, task_id)
    in_rig = np.linalg.inv(spec.rig_to_world(origin, math.degrees(yaw))) @ head
    np.testing.assert_allclose(in_rig[:3, 3], spec.HEAD_IN_RIG_M, atol=1e-12)
    position, look, pitch = head_camera_pose(config, task_id)
    view = look - position
    assert math.degrees(math.atan2(-view[2], np.hypot(*view[:2]))) == pytest.approx(60.0)
    assert math.degrees(pitch) == pytest.approx(spec.HEAD_PITCH_DOWN_DEG)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rig: rig.update(separation=0.60), "must not be copied"),
        (lambda rig: rig.update(head_camera_pitch_deg=60.0), "must not be copied"),
        (lambda rig: rig["tasks"].update(Z9={"origin": [0, 0, 1]}), "unknown task"),
    ],
)
def test_check_rig_rejects_off_spec_rigs(config, mutate, message):
    bad = copy.deepcopy(config)
    mutate(bad["robot"]["rig"])
    with pytest.raises(ValueError, match=message):
        check_rig(bad, ENV_TASKS)


def test_per_arm_mounts_are_not_accepted(config):
    bad = copy.deepcopy(config)
    bad["robot"]["task_mount_positions"] = {"A1": [[0, 0, 1.05], [0, 0.38, 1.05]]}
    with pytest.raises(ValueError, match="per-arm mounts"):
        check_rig(bad, ENV_TASKS)


def test_x5a_link6_forward_kinematics():
    # Zero joints: the X5A.urdf joint origins add up (joint3 flips y and z, joint6 flips back).
    pose = x5a_link6_in_base(np.zeros(6))
    np.testing.assert_allclose(pose[:3, 3], [0.02 - 0.264 + 0.245 + 0.06775 + 0.02895, -0.0005, 0.0605 + 0.04 + 0.056 + 0.0865 - 0.0865], atol=1e-4)
    np.testing.assert_allclose(pose[:3, :3], spec.urdf_rpy(3.1416, 0, 0) @ spec.urdf_rpy(-3.1416, 0, 0), atol=1e-9)
    turned = x5a_link6_in_base(np.array([math.pi / 2, 0, 0, 0, 0, 0]))
    np.testing.assert_allclose(turned[:3, 3], spec._rz(math.pi / 2) @ pose[:3, 3], atol=1e-9)


def _measured(config, task_id, joints):
    (right, left), yaw = rig_mounts(config, task_id)
    bases = [spec.rig_to_world(p, math.degrees(yaw)) for p in (right, left)]
    link6 = [b @ x5a_link6_in_base(q) for b, q in zip(bases, joints)]
    wrists = [l6 @ spec.wrist_camera_in_link6() for l6 in link6]
    return bases, head_camera_world(config, task_id), link6, wrists


def test_measured_check_accepts_the_spec_and_rejects_deviations(config):
    joints = [np.array([0.1, 1.2, 0.9, -0.4, 0.2, 0.3]), np.array([-0.2, 1.6, 1.3, -1.2, 0.0, 0.0])]
    bases, head, link6, wrists = _measured(config, "B4", joints)
    result = check_measured("B4", bases, head, link6, joints, wrists)
    assert result["bases"]["right"][1] < result["bases"]["left"][1]
    assert result["wrist_cameras"]["left"]["position_m"] < 1e-9
    assert check_measured("B4", bases, head, link6, joints, [None, None])["wrist_cameras"]["left"] is None
    moved = [w.copy() for w in wrists]
    moved[1][:3, 3] += [0.002, 0, 0]
    with pytest.raises(ValueError, match="left wrist camera"):
        check_measured("B4", bases, head, link6, joints, moved)
    with pytest.raises(ValueError, match="link6"):
        check_measured("B4", bases, head, link6, [j + 0.01 for j in joints], wrists)
    with pytest.raises(ValueError, match="left base"):
        check_measured("B4", bases[::-1], head, link6[::-1], joints[::-1], wrists[::-1])
    tilted = head.copy()
    tilted[:3, :3] = tilted[:3, :3] @ spec.euler_xyz(0.5, 0, 0)
    with pytest.raises(ValueError, match="turned"):
        check_measured("B4", bases, tilted, link6, joints, wrists)


WARD_TASKS = ("A1", "A2", "A3", "A4", "B4", "B5", "B6", "C1", "C2", "C3", "C4")


@pytest.mark.parametrize("task_id", WARD_TASKS)
@pytest.mark.parametrize("seed", (0, 3))
def test_wheeled_base_stands_clear_of_ward_furniture(config, task_id, seed):
    """The rig's wheeled base (both columns, 0.13 m shell) never stands inside the scene-02 bed,
    ultrasound cart or supply trolley (after each task's `ward_pose` moves/hides)."""
    import json

    from roboclock.envs.hospital.bedside.ward import furniture, part_bounds, ward_path
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_task
    from roboclock.envs.lab.robot import mount_positions

    task = make_task(task_id, "hard", seed, config)
    pose = make_layout(task, config).info["ward_pose"]
    parts = json.loads(ward_path(config, pose.get("usd")).with_suffix(".json").read_text())["parts"]
    mounts = np.asarray(mount_positions(config, task_id))[:, :2]
    lo, hi = mounts.min(0) - 0.13, mounts.max(0) + 0.13
    hidden = set(pose.get("hide", ()))
    for key in ("bed", "ultrasound_cart", "supply_trolley"):
        names = set(furniture(key)) - hidden
        if not names:
            continue
        other_lo, other_hi = part_bounds(pose, parts, names)
        assert np.any(hi <= other_lo) or np.any(lo >= other_hi), f"{task_id}: base inside the {key}"


# --- what the policy sees: env B2/B3 key objects inside the head camera image --------------------


def _box_corners(center, size):
    c, h = np.asarray(center, float), np.asarray(size, float) / 2
    return [c + h * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def _dial_corners(config, task_id, indicator):
    """Housing box plus the round face (r 75 mm) turned to the head camera (face_dials)."""
    from roboclock.envs.hospital.backend import INDICATOR_SIZE
    from roboclock.envs.hospital.rig import head_camera_pose

    center = indicator.position + [0.0, 0.0, 0.005]
    face = center + [0.0, 0.0, INDICATOR_SIZE[2] / 2 + 0.078]
    head, _, _ = head_camera_pose(config, task_id)
    to_head = (head - face)[:2] / np.linalg.norm((head - face)[:2])
    across = np.array([-to_head[1], to_head[0], 0.0])
    ring = [face + 0.075 * (np.cos(a) * across + np.sin(a) * np.array([0, 0, 1])) for a in np.linspace(0, 2 * np.pi, 12)]
    return _box_corners(center, INDICATOR_SIZE) + ring


def _key_points(config, task_id, seed):
    """Every corner of what B2/B3 act on or react to (objects, keys, marks, the cue dials)."""
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_task

    layout = make_layout(make_task(task_id, "hard", seed, config), config)
    fixtures = {f.name: f for f in layout.fixtures}
    points = {}
    for name, obj in layout.objects.items():
        shape = layout.shapes[name]
        half = np.asarray(getattr(shape, "half", None) if getattr(shape, "half", None) is not None else [shape.radius] * 3)
        points[name] = _box_corners(obj.position, 2 * np.asarray(half)[:3])
    for name in ("aed", "shock_button", "alarm_button", "fire_dish"):
        if name in fixtures:
            points[name] = _box_corners(fixtures[name].position, fixtures[name].size)
    for name, target in layout.targets.items():  # B3: the sharps bin, floor to rim
        points[name] = _box_corners(target.position, 2 * target.half_extents)
    for name, zone in layout.zones.items():
        if name.startswith("mark_"):
            points[name] = [[*(zone.center + [dx, dy]), zone.support_z] for dx in (-0.054, 0.054) for dy in (-0.04, 0.04)]
    for indicator in layout.indicators:
        if indicator.style == "dial":
            points[f"dial:{indicator.name}"] = _dial_corners(config, task_id, indicator)
        elif indicator.style == "beacon":  # base disc (r 45 mm) up to the dome (r 50 mm, 65 mm up)
            points[f"beacon:{indicator.name}"] = _box_corners(indicator.position + [0, 0, 0.05], [0.10, 0.10, 0.17])
    return points


@pytest.mark.parametrize("task_id", ("B2", "B3"))
@pytest.mark.parametrize("seed", range(5))
def test_b2_b3_key_objects_are_inside_the_head_view(config, task_id, seed):
    from roboclock.envs.hospital.rig import head_pixels

    width, height = config["sim"]["camera_resolution"]
    for name, corners in _key_points(config, task_id, seed).items():
        pixels, depth = head_pixels(config, task_id, corners)
        assert np.all(depth > 0.1), (task_id, name)
        margin = min(pixels[:, 0].min(), width - pixels[:, 0].max(), pixels[:, 1].min(), height - pixels[:, 1].max())
        assert margin > 10, (task_id, seed, name, margin)
    expected = {"B2": {"aed", "shock_button", "pad_0", "pad_1", "tab_0", "mark_0", "mark_1", "dial:indicator", "dial:shock",
                       "beacon:alert"},
                # B3, the hazardous drug spill: the bottle on the rack, the pad, where the glass
                # comes to rest, the bin's rim and both dials.
                "B3": {"bottle", "pad", "shard_0", "shard_1", "hd_bin", "dial:contain", "dial:indicator"}}
    assert expected[task_id] <= set(_key_points(config, task_id, seed))


@pytest.mark.parametrize("task_id", ("B2", "B3"))
def test_b2_b3_targets_stay_in_top_down_reach_of_arm_0(config, task_id):
    from roboclock.envs.lab.robot import mount_positions

    arm0 = np.asarray(mount_positions(config, task_id)[0])[:2]
    for seed in range(5):
        for name, corners in _key_points(config, task_id, seed).items():
            # B3's bottle stands on the rack and is never handled; only its pieces are.
            if name.startswith(("dial:", "beacon:")) or name in ("aed", "fire_dish", "bottle"):
                continue
            center = np.mean(np.asarray(corners)[:, :2], axis=0)
            assert 0.2 < np.hypot(*(center - arm0)) < 0.5, (task_id, seed, name)
