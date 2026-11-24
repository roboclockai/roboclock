"""Ground-truth placement, reach-aware selection and no-drop recovery of the planners."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import snapshot

from roboclock.config import load_config
from roboclock.envs.lab.mdp.state import ObjectState
from roboclock.envs.lab.policies.rack_organization import (
    RackOrganizationTrajectory,
    upright_correction,
)
from roboclock.envs.lab.policies.trajectories import (
    LabTrajectory,
    multiply,
    quaternion_yaw,
    tool_orientation,
    tool_yaw,
    wrap_angle,
)
from roboclock.physics.geometry import upright_error

CONTROL = load_config()["control"]


def planner_for(task_id, **kwargs):
    cls = RackOrganizationTrajectory if task_id == "C1" else LabTrajectory
    return cls(CONTROL, 1 / 30, **kwargs)


def held(task, name, ee, yaw=0.0, offset=(0.0, 0.0, 0.0), tool_yaw_value=0.0):
    obj = ObjectState(
        np.asarray(ee) - np.asarray(offset),
        np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
        grasped=True,
    )
    state = snapshot(task, **{name: obj})
    return replace(
        state, ee_position=np.asarray(ee, float), ee_orientation=tool_orientation(tool_yaw_value)
    )


@pytest.mark.parametrize("yaw", [-2.9, -1.0, 0.0, 0.4, 2.2])
def test_tool_yaw_inverts_ik_goal_convention(yaw):
    assert wrap_angle(tool_yaw(tool_orientation(yaw)) - yaw) == pytest.approx(0.0, abs=1e-9)


def test_upright_correction_keeps_tool_yaw():
    tool = tool_orientation(0.7)
    in_hand = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])  # tube tilted in the pads
    tube = multiply(tool, in_hand)
    goal = upright_correction(tube, tool)
    assert upright_error(multiply(goal, in_hand)) < 1e-6
    assert tool_yaw(goal) == pytest.approx(0.7, abs=1e-6)


def test_c3_placement_yaw_uses_measured_in_hand_offset(task_factory):
    task = task_factory("C3")
    planner = planner_for("C3")
    planner.sample, planner.phase = "sample_0", "insert"
    target = task.layout.targets["target_0"]
    ee = target.position + [0.0, 0.0, 0.10]
    # Tool at yaw 0.2 while the block reads 0.3: the block is 0.1 rad ahead of the tool.
    state = held(task, "sample_0", ee, yaw=0.3, offset=(0, 0, 0.006), tool_yaw_value=0.2)
    command = planner.command(task, state)
    assert command.yaw == pytest.approx(quaternion_yaw(target.orientation) - 0.1, abs=1e-6)
    assert command.close


def test_c3_marker_release_sets_block_down_instead_of_dropping(task_factory):
    task = task_factory("C3")
    planner = planner_for("C3")
    planner.sample, planner.phase = "sample_0", "insert"
    target = task.layout.targets["target_0"]
    touch = target.rim_z + 0.025 + 0.006 + 0.001
    # Hovering 3.6 cm above the marker with the block in hand, 6 mm below the tool.
    ee = target.position + [0.0, 0.0, 0.036]
    state = held(task, "sample_0", ee, offset=(0, 0, 0.006))
    task.time = 1.0
    for _ in range(200):  # Same instant: the rate-limited goal converges without stalling.
        command = planner.command(task, state)
    assert planner.phase == "insert" and planner.marker_stage == "center"
    assert command.position[2] == pytest.approx(touch + 0.02, abs=1e-6)
    # Centred and settled at the hover: lower vertically onto the marker.
    state = held(task, "sample_0", [*ee[:2], touch + 0.02], offset=(0, 0, 0.006))
    for _ in range(200):
        command = planner.command(task, state)
    assert planner.marker_stage == "lower"
    assert command.position[2] == pytest.approx(touch, abs=1e-6)
    assert command.close  # not released before the block is down


def test_throughput_timeout_while_holding_lifts_instead_of_opening(task_factory):
    task = task_factory("C3")
    planner = planner_for("C3")
    planner.sample, planner.phase = "sample_0", "transfer"
    planner.grasp_offset = np.array([0, 0, 0.006])
    planner.phase_time = 0.0
    task.time = CONTROL["phase_timeout"] + 0.1
    ee = np.array([0.4, -0.2, 1.0])
    state = held(task, "sample_0", ee, offset=(0, 0, 0.006))
    command = planner.command(task, state)
    assert command.close
    assert planner.phase == "lift"
    assert planner.retries["sample_0"] == 1


def test_throughput_gives_up_by_returning_to_source_not_dropping(task_factory):
    task = task_factory("C3")
    planner = planner_for("C3")
    planner.sample, planner.phase = "sample_0", "transfer"
    planner.grasp_offset = np.array([0, 0, 0.006])
    planner.retries["sample_0"] = 2
    task.time = CONTROL["phase_timeout"] + 0.1
    ee = np.array([0.4, -0.2, 1.0])
    state = held(task, "sample_0", ee, offset=(0, 0, 0.006))
    command = planner.command(task, state)
    assert command.close and planner.returning and planner.phase == "lift"
    # Lift completes above the hold point, then the planner carries the block home.
    task.time += 0.1
    planner.phase_time = task.time
    state = replace(state, ee_position=planner.hold.copy())
    planner.command(task, state)
    assert planner.phase == "return"
    task.time += 0.1
    before = planner.last_goal.copy()
    command = planner.command(task, state)
    source = task.layout.objects["sample_0"].position
    assert command.close
    direction = source[:2] + planner.grasp_offset[:2] - before[:2]
    step = command.position[:2] - before[:2]
    assert np.dot(direction, step) > 0
    assert command.position[2] >= before[2] - 1e-9  # Carried at height, not lowered early.


def test_lost_grasp_during_transfer_reselects_instead_of_chasing(task_factory):
    task = task_factory("C3")
    planner = planner_for("C3")
    planner.sample, planner.phase = "sample_0", "insert"
    planner.grasp_offset = np.array([0, 0, 0.006])
    # The block lies on the table far below the tool: no grasp, large separation.
    state = snapshot(task)
    state = replace(state, ee_position=np.array([0.4, 0.1, 0.95]))
    task.time = 1.0
    command = planner.command(task, state)
    assert planner.phase == "select" or planner.phase == "approach"
    assert not command.close


def test_stalled_approach_skips_unreachable_sample(task_factory):
    task = task_factory("C3")
    planner = planner_for("C3")
    planner.sample, planner.phase = "sample_0", "approach"
    state = snapshot(task)
    for step in range(80):
        task.time = step / 30
        planner.command(task, state)  # Tool never moves while the goal is 0.3 m away.
    assert planner.retries["sample_0"] >= 3
    assert planner.sample != "sample_0" or planner.phase == "select"


def test_reach_filter_excludes_far_samples(task_factory):
    from roboclock.envs.lab.policies.trajectories import reachable

    task = task_factory("C3")
    base = np.array([0.10, -0.09, 1.05])
    planner = planner_for("C3", base=base)
    planner.config = {**CONTROL, "reach_margin": 0.10}
    state = snapshot(task)
    planner.command(task, state)
    assert planner.sample is not None
    assert planner.can_handle(task, planner.sample, state)
    far = [n for n in task.layout.objects if not planner.can_handle(task, n, state)]
    assert far and planner.sample not in far
    for name in far:
        assert not reachable(base, 0.10, *planner.task_points(task, name, state))


def test_measured_reach_map_limits_horizontal_reach():
    from roboclock.envs.lab.policies.trajectories import reach_limit, reachable

    base = np.zeros(3)
    assert reach_limit(-0.369) == pytest.approx(0.475)
    assert reach_limit(-0.20) > reach_limit(-0.40)
    assert reachable(base, 0.0, [0.45, 0.0, -0.2])
    assert not reachable(base, 0.0, [0.55, 0.0, -0.2])
    assert not reachable(base, 0.0, [0.05, 0.0, -0.2])  # folded into the column


def test_c1_timeout_while_holding_retargets_without_opening(task_factory):
    task = task_factory("C1")
    planner = planner_for("C1", base=[0.12, 0.0, 1.10])
    planner.sample, planner.phase = "sample_0", "transfer"
    task.layout.assignments["sample_0"] = "target_11"
    task.time = 6.5
    ee = np.array([0.45, 0.0, 1.0])
    state = held(task, "sample_0", ee, offset=(0, 0, 0.06))
    command = planner.command(task, state)
    assert command.close
    assert planner.phase == "lift"
    assert planner.target_failures == {"target_11": 1}
    assert task.layout.assignments["sample_0"] != "target_11"


def test_c1_repeated_failures_carry_tube_back_to_its_source_slot(task_factory):
    task = task_factory("C1")
    planner = planner_for("C1", base=[0.12, 0.0, 1.10])
    planner.sample, planner.phase = "sample_0", "transfer"
    planner.retries["sample_0"] = 2
    task.time = 6.5
    ee = np.array([0.45, 0.0, 1.0])
    state = held(task, "sample_0", ee, offset=(0, 0, 0.06))
    command = planner.command(task, state)
    assert command.close and planner.returning and planner.phase == "lift"
    task.time += 0.1
    planner.phase_time = task.time
    planner.command(task, replace(state, ee_position=planner.hold.copy()))
    assert planner.phase == "return"
    source = task.layout.objects["sample_0"].position
    assert np.allclose(planner.source_target(task, "sample_0").position, source)


def test_c1_lost_tube_is_not_chased_with_in_hand_offset(task_factory):
    task = task_factory("C1")
    planner = planner_for("C1")
    planner.sample, planner.phase = "sample_0", "insert"
    state = snapshot(task)  # Tube at its source, tool far away, not grasped.
    task.time = 1.0
    command = planner.command(task, state)
    assert not command.close
    assert planner.phase in ("select", "approach")


def test_c1_candidate_slots_prefer_near_unfailed_reachable_slots(task_factory):
    task = task_factory("C1")
    planner = planner_for("C1", base=[0.12, 0.0, 1.10])
    state = snapshot(task)
    candidates = planner.candidate_targets(task, "sample_0", state)
    assert candidates
    shoulder = np.array([0.12, 0.0, 1.10 + 0.1005])
    distances = [
        np.linalg.norm(task.layout.targets[key].position[:2] - shoulder[:2]) for key in candidates
    ]
    assert distances == sorted(distances)
    planner.target_failures[candidates[0]] = 1
    assert planner.candidate_targets(task, "sample_0", state)[0] == candidates[1]
    planner.config = {**CONTROL, "reach_margin": 0.5}
    assert planner.candidate_targets(task, "sample_0", state) == []


def test_bimanual_partition_gives_every_sample_to_an_able_arm(task_factory):
    from roboclock.envs.lab.policies.bimanual import partition_samples
    from roboclock.envs.lab.robot import mount_positions

    config = load_config()
    for task_id in ("C1", "C3"):
        task = task_factory(task_id)
        mounts = mount_positions(config, task_id)
        planners = [planner_for(task_id, base=m) for m in mounts]
        left, right = partition_samples(task, planners)
        assert not left.intersection(right)
        assert left.union(right) == set(task.layout.objects)
        assert abs(len(left) - len(right)) <= 4
        for side, queue in enumerate((left, right)):
            for name in queue:
                assert planners[side].can_handle(task, name), (task_id, side, name)


def test_segment_distance():
    from roboclock.envs.lab.policies.bimanual import segment_distance

    assert segment_distance([0, 0], [1, 0], [0, 1], [1, 1]) == pytest.approx(1.0)
    assert segment_distance([0, 0], [1, 1], [0, 1], [1, 0]) == 0.0
    assert segment_distance([0, 0], [0.4, 0], [0.6, 0], [1, 0]) == pytest.approx(0.2)


def test_separation_guard_parks_the_less_busy_arm(task_factory):
    from roboclock.envs.lab.policies.bimanual import BimanualThroughput
    from roboclock.envs.lab.policies.trajectories import CartesianCommand

    task = task_factory("C3")
    config = load_config()
    state = snapshot(task)
    robots = [SimpleNamespace(get_joint_positions=lambda: np.zeros(8)) for _ in range(2)]
    backend = SimpleNamespace(
        robot=robots[0],
        secondary_robot=robots[1],
        state=lambda: state,
        grasp_owners={},
        _tool_pose=lambda robot: (state.ee_position, state.ee_orientation),
    )
    policy = BimanualThroughput(SimpleNamespace(task=task, config=config, backend=backend), 1 / 30)
    left_mount, right_mount = policy.mounts
    tools = [np.array([0.40, -0.10, 0.97]), np.array([0.50, -0.10, 0.97])]
    policy.planners[0].phase = "insert"  # left is placing: it keeps going
    policy.planners[1].phase = "approach"
    commands = [
        CartesianCommand(tools[0] + [0.0, 0.0, -0.002], True),
        CartesianCommand(tools[1] + [-0.01, 0.0, 0.0], False),
    ]
    out = policy.separate(tools, commands)
    assert policy.yielding == [False, True]
    np.testing.assert_allclose(out[0].position, commands[0].position)
    # The right arm never moves toward the left arm; while low it first rises.
    assert out[1].position[0] >= tools[1][0] - 1e-9
    assert out[1].position[2] > tools[1][2]
    # Far apart: nobody waits.
    tools = [np.array([0.25, -0.10, 0.97]), np.array([0.65, -0.10, 0.97])]
    commands = [CartesianCommand(t.copy(), False) for t in tools]
    policy.separate(tools, commands)
    assert policy.yielding == [False, False]


def test_bimanual_ik_failure_restores_goal(task_factory):
    from roboclock.envs.lab.policies.bimanual import BimanualThroughput

    task = task_factory("C3")
    config = load_config()
    state = snapshot(task)
    robots = [SimpleNamespace(get_joint_positions=lambda: np.zeros(8)) for _ in range(2)]
    backend = SimpleNamespace(
        robot=robots[0],
        secondary_robot=robots[1],
        state=lambda: state,
        grasp_owners={},
        _tool_pose=lambda robot: (state.ee_position, state.ee_orientation),
    )
    policy = BimanualThroughput(SimpleNamespace(task=task, config=config, backend=backend), 1 / 30)
    policy.commands()
    before = [p.last_goal.copy() for p in policy.planners]
    task.time += 1 / 30
    policy.commands()
    assert any(np.linalg.norm(p.last_goal - b) > 0 for p, b in zip(policy.planners, before))
    policy.ik_failed(0)
    np.testing.assert_allclose(policy.planners[0].last_goal, before[0])


@pytest.mark.parametrize("task_id", ["A1", "A4", "B5", "C1", "C3"])
def test_head_camera_is_rigid_on_the_common_rig(task_id):
    from roboclock.envs import robodojo_rig
    from roboclock.envs.lab.robot import head_camera_position, mount_positions, mount_yaws, rig_axes

    config = load_config()
    mounts = np.asarray(mount_positions(config, task_id))
    forward, left = rig_axes(mount_yaws(config, task_id)[0])
    assert mounts[0][2] == pytest.approx(mounts[1][2])
    assert (mounts[1] - mounts[0]) @ forward == pytest.approx(0.0, abs=1e-12)
    assert (mounts[1] - mounts[0]) @ left == pytest.approx(robodojo_rig.ARM_SEPARATION_M)
    head = robodojo_rig.head_camera_world(mounts.mean(axis=0), np.degrees(mount_yaws(config, task_id)[0]))
    np.testing.assert_allclose(head_camera_position(config, task_id), head[:3, 3], atol=1e-12)
    np.testing.assert_allclose(head[:3, 3], mounts.mean(axis=0) + forward * 0.04 + [0, 0, 0.543])
    look = -head[:3, 2]
    assert np.degrees(np.arcsin(-look[2])) == pytest.approx(60.0)
    np.testing.assert_allclose(look[:2] / np.linalg.norm(look[:2]), forward[:2], atol=1e-12)


def test_rig_validation_rejects_staggered_or_legacy_mounts():
    import copy

    from roboclock.envs.lab.config import check_rig

    config = load_config()
    legacy = copy.deepcopy(config)
    legacy["robot"]["task_mount_positions"] = {"C1": [[0.1, 0, 1.1], [0.7, 0, 1.1]]}
    with pytest.raises(ValueError, match="replaced by robot.rig"):
        check_rig(legacy)
    for key, value in (
        ("separation", 0.6),
        ("head_camera_offset", [0.04, 0.0, 0.543]),
        ("head_camera_pitch_deg", 60.0),
    ):
        copied = copy.deepcopy(config)
        copied["robot"]["rig"][key] = value  # even equal values are local copies
        with pytest.raises(ValueError, match=f"{key} is fixed by roboclock.envs.robodojo_rig"):
            check_rig(copied)
    turned = copy.deepcopy(config)
    turned["robot"]["rig"]["tasks"]["A1"]["look_at"] = [0.3, 0.0]
    with pytest.raises(ValueError, match="only origin and yaw"):
        check_rig(turned)
    fast = copy.deepcopy(config)
    fast["sim"].update(physics_dt=0.002, control_decimation=10)
    with pytest.raises(ValueError, match="exactly 1/250"):
        check_rig(fast)
    large = copy.deepcopy(config)
    large["sim"]["camera_resolution"] = [960, 720]
    with pytest.raises(ValueError, match="camera_resolution"):
        check_rig(large)
    scene = copy.deepcopy(config)
    scene["sim"]["observation_camera"] = "scene"
    with pytest.raises(ValueError, match="head camera"):
        check_rig(scene)


def test_return_release_requires_centered_and_lowered(task_factory):
    task = task_factory("C3")
    planner = planner_for("C3")
    planner.sample, planner.phase = "sample_0", "return"
    planner.returning = True
    planner.grasp_offset = np.array([0, 0, 0.006])
    source = task.layout.objects["sample_0"].position
    # Hovering 15 cm up, 7 mm off laterally: within the old 8 mm release ball.
    ee = source + [0.007, 0.0, 0.156]
    state = held(task, "sample_0", ee, offset=(0, 0, 0.006))
    task.time = 1.0
    command = planner.command(task, state)
    assert planner.phase == "return"
    assert command.close
    # Centered and down at the surface: release.
    ee = source + [0.0, 0.0, 0.009]
    state = held(task, "sample_0", ee, offset=(0, 0, 0.006))
    planner.command(task, state)
    assert planner.phase == "release"


def test_abandoned_sample_is_handed_to_the_other_arm(task_factory):
    from roboclock.envs.lab.policies.bimanual import BimanualThroughput

    task = task_factory("C3")
    config = load_config()
    state = snapshot(task)
    robots = [SimpleNamespace(get_joint_positions=lambda: np.zeros(8)) for _ in range(2)]
    backend = SimpleNamespace(
        robot=robots[0],
        secondary_robot=robots[1],
        state=lambda: state,
        grasp_owners={},
        _tool_pose=lambda robot: (state.ee_position, state.ee_orientation),
    )
    policy = BimanualThroughput(SimpleNamespace(task=task, config=config, backend=backend), 1 / 30)
    shared = [n for n in policy.names[0] if policy.planners[1].can_handle(task, n)]
    unreachable = [n for n in policy.names[0] if not policy.planners[1].can_handle(task, n)]
    policy.planners[0].sample = None
    if shared:
        name = min(shared)
        policy.planners[0].retries[name] = 3
        policy.commands()
        assert name in policy.names[1] and name not in policy.names[0]
        policy.planners[1].retries[name] = 3
        policy.commands()
        assert name in policy.names[1]  # Handed over once; never bounces back.
    if unreachable:  # Never handed to an arm that cannot reach it.
        policy.planners[0].retries[unreachable[0]] = 3
        policy.commands()
        assert unreachable[0] in policy.names[0]


def test_park_point_never_moves_toward_the_other_arm(task_factory):
    from roboclock.envs.lab.policies.bimanual import BimanualThroughput, segment_distance

    task = task_factory("C1")
    config = load_config()
    state = snapshot(task)
    robots = [SimpleNamespace(get_joint_positions=lambda: np.zeros(8)) for _ in range(2)]
    backend = SimpleNamespace(
        robot=robots[0],
        secondary_robot=robots[1],
        state=lambda: state,
        grasp_owners={},
        _tool_pose=lambda robot: (state.ee_position, state.ee_orientation),
    )
    policy = BimanualThroughput(SimpleNamespace(task=task, config=config, backend=backend), 1 / 30)
    right_mount = policy.mounts[1]
    tool = right_mount + [-0.15, 0.0, -0.08]  # retracted right tool
    # The left arm reaches toward the right mount along the mount line.
    other = (policy.mounts[0], right_mount + [-0.19, 0.09, -0.05])
    park = policy.park_point(1, tool, other)
    assert np.linalg.norm(park[:2] - right_mount[:2]) <= 0.15 + 1e-9
    before = segment_distance(right_mount, tool, *other)
    assert segment_distance(right_mount, park, *other) > before


def test_segment_distance_3d():
    from roboclock.envs.lab.policies.bimanual import segment_distance_3d

    assert segment_distance_3d([0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]) == pytest.approx(
        np.sqrt(2)
    )
    # Crossing in xy but 0.2 m apart in height: arms passing over each other.
    assert segment_distance_3d([0, 0, 0], [1, 1, 0], [0, 1, 0.2], [1, 0, 0.2]) == pytest.approx(0.2)
    assert segment_distance_3d([0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_c3_release_opens_only_slightly_past_the_block(task_factory):
    task = task_factory("C3")
    planner = planner_for("C3")
    planner.sample, planner.phase = "sample_0", "release"
    target = task.layout.targets["target_0"]
    ee = target.position + [0.0, 0.0, 0.006]
    planner.hold = ee.copy()
    task.time = 0.1
    planner.phase_time = 0.0
    state = held(task, "sample_0", ee, offset=(0, 0, 0.006))
    command = planner.command(task, state)
    radius = task.layout.dimensions["sample_0"][0]
    assert not command.close
    assert command.gripper == pytest.approx(radius + 0.004)
    np.testing.assert_allclose(command.position, ee, atol=0.01)  # still holding position


def test_rig_tasks_start_from_the_common_home_pose():
    from roboclock.envs.lab.robot import HOME, home_joints

    config = load_config()
    # Side-by-side arms face the same way: no per-task turned start pose is needed.
    for task_id in ("A1", "C1", "C2", "C3"):
        np.testing.assert_allclose(home_joints(config, task_id), np.tile(HOME, (2, 1)))


def test_throughput_goal_stays_on_a_leash_to_the_tool(task_factory):
    from roboclock.envs.lab.policies.trajectories import GOAL_LEASH

    task = task_factory("C3")
    planner = planner_for("C3")
    state = snapshot(task)
    task.time = 0.0
    for step in range(60):  # the tool never moves: the goal must not run away
        task.time = step / 30
        command = planner.command(task, state)
    assert np.linalg.norm(command.position - state.ee_position) <= GOAL_LEASH + 1e-9


def test_finished_samples_do_not_crowd_out_cross_over_jobs(task_factory):
    from roboclock.envs.lab.policies.bimanual import ArmTask, BimanualThroughput

    task = task_factory("C1")
    config = load_config()
    state = snapshot(task)
    backend = SimpleNamespace(
        robot=None,
        secondary_robot=None,
        state=lambda: state,
        grasp_owners={},
        _tool_pose=lambda robot: (state.ee_position, state.ee_orientation),
    )
    policy = BimanualThroughput(SimpleNamespace(task=task, config=config, backend=backend), 1 / 30)
    names = set(policy.names[0])
    view = ArmTask(task, names, set())
    jobs = {n: policy.job_footprint(0, view, n, state) for n in names}
    crossing = {n for n, job in jobs.items() if job is not None and policy.crossing(0, job[1:])}
    local = {n for n, job in jobs.items() if job is not None and not policy.crossing(0, job[1:])}
    if not crossing or not local:
        pytest.skip("seed has no mix of local and cross-over jobs for the left arm")
    task.correct.update(local)  # every local job is already done
    chosen = policy.compatible_jobs(0, view, names, state.ee_position, state.ee_position, state)
    assert chosen == crossing
