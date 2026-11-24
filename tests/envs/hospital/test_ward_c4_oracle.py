"""C4 scripted oracle (bedside/ward_c4._drive): planner decisions on hand-made states, no simulator.

These check which tube and well the planner picks, when it lowers, releases or lifts a tube out
again, and the geometry of standing up a tube that has pivoted between the fingers. They do not
test PhysX contact, grasp stability or reachability (that is the oracle run itself).
"""

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import ward_c4
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState
from roboclock.physics.geometry import rotation_matrix

DT = 0.02


def bound(seed=0):
    config = load_config()
    task = make_task("C4", "medium", seed, config)
    task.bind(make_layout(task, config))
    return task


def tilted(angle, about=(1.0, 0.0, 0.0)):
    axis = np.asarray(about, dtype=float) / np.linalg.norm(about)
    return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])


def upright_at(bottom, angle=0.0, **kwargs):
    """A tube whose bottom center is at `bottom`, tipped by `angle` about x."""
    q = tilted(angle)
    axis = rotation_matrix(q)[:, 2]
    return ObjectState(np.asarray(bottom) + axis * ward_c4.TUBE_SPEC["height"] / 2, q, **kwargs)


def in_well(task, well, angle=0.0, **kwargs):
    target = task.layout.targets[well]
    return upright_at([*target.position[:2], target.support_z + 0.001], angle, **kwargs)


def state_of(task, ee, **changes):
    objects = {
        name: ObjectState(obj.position.copy(), obj.orientation.copy())
        for name, obj in task.layout.objects.items()
    }
    objects.update(changes)
    down = ward_c4.down_quat(0.0)
    return PhysicalState(objects, np.asarray(ee, dtype=float), ee_orientation=down)


def planner_at(ee):
    planner = HospitalTrajectory(load_config()["control"], DT)
    planner.last_goal = np.asarray(ee, dtype=float).copy()
    planner.partner, planner.held = None, ()
    return planner


def drive(planner, task, state, side):
    return ward_c4._drive(
        planner, task, state, state.ee_position, side, ward_c4.down_quat(planner.yaw)
    )


def side_tubes(task, side):
    info = task.layout.info["ward_c4"]
    return [t for t, s in info["tube_side"].items() if s == side]


def own_wells(task, side, key=None):
    info = task.layout.info["ward_c4"]
    return [
        w
        for w, s in info["well_side"].items()
        if s == side and (key is None or info["well_class"][w] == key)
    ]


def carrying(task, side, phase="lower"):
    """A planner of `side` that has its first tube in the hand, at `phase`."""
    tube = side_tubes(task, side)[0]
    key = task.tube_class[tube]
    well = own_wells(task, side, key)[0]
    ee = task.layout.objects[tube].position + [0.0, 0.0, 0.2]
    planner = planner_at(ee)
    drive(planner, task, state_of(task, ee), side)  # initialise the C4 fields
    planner.c4_tube, planner.c4_well = tube, well
    planner.c4_wells = {tube: well}
    planner.c4_offset = np.array([0.0, 0.0, ward_c4.GRASP_DZ])
    planner.c4_grip = ward_c4.GRASP_DZ + ward_c4.TUBE_SPEC["height"] / 2
    planner.c4_phase, planner.c4_time = phase, task.time
    planner.held = (tube,)
    return planner, tube, well


def held_state(task, tube, obj):
    """The hand closed on `obj`, GRASP_DZ above its center along its axis."""
    axis = rotation_matrix(obj.orientation)[:, 2]
    ee = obj.position + axis * ward_c4.GRASP_DZ
    return state_of(task, ee, **{tube: obj})


# --- geometry ----------------------------------------------------------------------------


@pytest.mark.parametrize("yaw", [0.0, 0.7])
def test_tilt_correction_stands_a_pivoted_tube_upright(yaw):
    down = ward_c4.down_quat(yaw)
    up = rotation_matrix(tilted(0.15, (0.3, 1.0, 0.0)))[:, 2]
    correction, angle = ward_c4.tilt_correction(down, up, yaw)
    assert angle == pytest.approx(0.15)
    assert rotation_matrix(correction) @ up == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    # Measured again with the tool already turned, the same correction comes out (no drift).
    from roboclock.envs.lab.policies.trajectories import multiply

    turned = multiply(correction, down)
    in_tool = rotation_matrix(down).T @ up
    again, slip = ward_c4.tilt_correction(turned, rotation_matrix(turned) @ in_tool, yaw)
    assert again == pytest.approx(correction, abs=1e-9) and slip == pytest.approx(0.15)


def test_tilt_correction_is_identity_for_an_upright_tube():
    correction, angle = ward_c4.tilt_correction(ward_c4.down_quat(0.0), [0.0, 0.0, 1.0])
    assert angle == pytest.approx(0.0) and correction == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_aim_point_is_on_the_axis_half_a_well_above_the_bottom():
    obj = upright_at([0.4, 0.0, 0.75], angle=0.2)
    bottom = ward_c4.tube_bottom(obj)
    assert bottom == pytest.approx([0.4, 0.0, 0.75])
    aim = ward_c4.aim_point(obj)
    assert np.linalg.norm(aim - bottom) == pytest.approx(ward_c4.AIM_DEPTH)
    assert ward_c4.tube_tilt(obj) == pytest.approx(0.2)


def test_a_tube_lying_over_a_well_occupies_it_and_a_carried_one_does_not():
    task = bound()
    well = own_wells(task, "left")[0]
    target = task.layout.targets[well]
    tube = side_tubes(task, "left")[0]
    lying = ObjectState(
        np.array([target.position[0] + 0.03, target.position[1], target.rim_z + 0.0115]),
        tilted(np.pi / 2, (0.0, 1.0, 0.0)),  # axis along x, center beside the opening
    )
    assert ward_c4._occupied(task, state_of(task, [0.4, 0, 1], **{tube: lying}), well)
    assert not ward_c4._occupied(
        task, state_of(task, [0.4, 0, 1], **{tube: lying}), well, ignore=tube
    )
    high = upright_at([*target.position[:2], target.rim_z + 0.11])
    assert not ward_c4._occupied(task, state_of(task, [0.4, 0, 1], **{tube: high}), well)
    assert ward_c4._occupied(task, state_of(task, [0.4, 0, 1], **{tube: in_well(task, well)}), well)


# --- selection ---------------------------------------------------------------------------


def test_select_takes_a_tube_out_of_a_wrong_well_and_skips_credited_tubes():
    task = bound()
    side = "right"
    tubes = side_tubes(task, side)
    wrong_tube = tubes[0]
    key = task.tube_class[wrong_tube]
    wrong_well = next(w for w in own_wells(task, side) if task.well_class[w] != key)
    placed = {wrong_tube: in_well(task, wrong_well)}
    task.placed[wrong_tube] = wrong_well  # first resting well was wrong: on the record
    ee = placed[wrong_tube].position + [0.0, 0.0, 0.15]
    planner = planner_at(ee)
    drive(planner, task, state_of(task, ee, **placed), side)
    assert planner.c4_tube == wrong_tube and planner.c4_phase == "approach"
    assert task.well_class[planner.c4_well] == key and planner.c4_well != wrong_well

    # A credited tube is never picked again, even when it is the nearest.
    right_well = own_wells(task, side, key)[0]
    placed = {wrong_tube: in_well(task, right_well)}
    task.correct.add(wrong_tube)
    ee = placed[wrong_tube].position + [0.0, 0.0, 0.15]
    planner = planner_at(ee)
    drive(planner, task, state_of(task, ee, **placed), side)
    assert planner.c4_tube not in (None, wrong_tube)


def test_select_skips_a_tube_it_just_released():
    task = bound()
    planner, tube, well = carrying(task, "left", phase="release")
    planner.c4_release = np.array([0.4, 0.0, 0.9])
    obj = in_well(task, well, angle=0.25)  # tipped, not credited
    state = state_of(task, [0.4, 0.0, 0.9], **{tube: obj})
    task.time = planner.c4_time + 0.31
    drive(planner, task, state, "left")
    assert planner.c4_phase == "retreat"
    planner.c4_phase = "select"
    drive(planner, task, state, "left")
    assert planner.c4_tube != tube


# --- carrying and placing ----------------------------------------------------------------


def test_the_command_never_runs_more_than_max_lead_ahead_of_the_hand():
    task = bound()
    planner, tube, well = carrying(task, "left", phase="transfer")
    obj = upright_at(task.layout.objects[tube].position + [0.0, 0.0, 0.2])
    state = held_state(task, tube, obj)
    planner.last_goal = state.ee_position + [0.3, 0.0, 0.0]  # a target that ran away
    command = drive(planner, task, state, "left")
    # At most MAX_LEAD plus one control step of the carry speed.
    lead = np.linalg.norm(command.position - state.ee_position)
    assert lead <= ward_c4.MAX_LEAD + 0.55 * DT + 1e-9
    assert command.close


def test_a_tube_pivoting_in_the_fingers_is_kept_and_stood_up_not_dropped():
    task = bound()
    planner, tube, well = carrying(task, "left", phase="transfer")
    target = task.layout.targets[well]
    ee = np.array([*target.position[:2], task.layout.table_height + ward_c4.TRAVEL_Z])
    # Tube tipped 0.2 rad about the grip: its center is ~6 mm off where it was grasped.
    q = tilted(0.2)
    axis = rotation_matrix(q)[:, 2]
    obj = ObjectState(ee - axis * ward_c4.GRASP_DZ, q, grasped=False)  # grasp flag flickers
    command = drive(planner, task, state_of(task, ee, **{tube: obj}), "left")
    assert planner.c4_phase == "transfer" and command.close
    assert planner.tool_tilt is not None  # the tool turns to stand the tube up
    for _ in range(40):
        drive(planner, task, state_of(task, ee, **{tube: obj}), "left")
    turn = rotation_matrix(planner.tool_tilt)
    assert turn @ axis == pytest.approx([0.0, 0.0, 1.0], abs=0.01)
    # A tube that has really left the hand ends the job (the hand is then empty anyway).
    gone = ObjectState(ee - [0.0, 0.0, 0.2], np.array([1.0, 0.0, 0.0, 0.0]))
    drive(planner, task, state_of(task, ee, **{tube: gone}), "left")
    assert planner.c4_phase == "abort"


def test_transfer_lowers_only_once_the_tube_is_over_the_well_and_upright():
    task = bound()
    planner, tube, well = carrying(task, "left", phase="transfer")
    target = task.layout.targets[well]
    travel = task.layout.table_height + ward_c4.TRAVEL_Z
    bottom_z = travel - ward_c4.GRASP_DZ - ward_c4.TUBE_SPEC["height"] / 2
    off = upright_at([target.position[0] + 0.01, target.position[1], bottom_z])
    drive(planner, task, held_state(task, tube, off), "left")
    assert planner.c4_phase == "transfer"
    over = upright_at([*target.position[:2], bottom_z])
    drive(planner, task, held_state(task, tube, over), "left")
    assert planner.c4_phase == "lower"


def test_a_tube_caught_on_the_rim_is_lifted_out_again_not_released():
    task = bound()
    planner, tube, well = carrying(task, "left", phase="lower")
    target = task.layout.targets[well]
    # Bottom on the rim, tipped: it has not reached the well floor.
    stuck = upright_at([target.position[0], target.position[1] + 0.015, target.rim_z], angle=0.3)
    command = drive(planner, task, held_state(task, tube, stuck), "left")
    assert planner.c4_phase == "transfer" and command.close and planner.c4_lowers == 1
    assert any(e["type"] == "c4_retry" for e in task.event_log)


def test_a_tube_slid_up_between_the_fingers_is_lifted_out_again():
    task = bound()
    planner, tube, well = carrying(task, "left", phase="lower")
    target = task.layout.targets[well]
    obj = upright_at([*target.position[:2], target.rim_z + 0.02])
    state = held_state(task, tube, obj)
    state = state_of(task, state.ee_position - [0.0, 0.0, 0.01], **{tube: obj})  # 1 cm lower
    command = drive(planner, task, state, "left")
    assert planner.c4_phase == "transfer" and command.close


def test_an_upright_tube_on_the_well_floor_is_released():
    task = bound()
    planner, tube, well = carrying(task, "left", phase="lower")
    obj = in_well(task, well, angle=0.05)
    command = drive(planner, task, held_state(task, tube, obj), "left")
    assert planner.c4_phase == "release" and command.close
    task.time += 0.05
    command = drive(planner, task, held_state(task, tube, obj), "left")
    assert not command.close  # fingers open at the release point


def test_a_stuck_lower_times_out_into_a_retry_not_a_release():
    task = bound()
    planner, tube, well = carrying(task, "left", phase="lower")
    target = task.layout.targets[well]
    obj = upright_at([*target.position[:2], target.rim_z + 0.03])
    task.time = planner.c4_time + ward_c4.PHASE_TIMEOUT + 0.1
    command = drive(planner, task, held_state(task, tube, obj), "left")
    assert planner.c4_phase == "transfer" and command.close


# --- joint stepping (policies/oracle.py) -------------------------------------------------


def test_synchronized_joint_step_keeps_the_joint_space_direction():
    from roboclock.envs.hospital.policies.oracle import joint_step

    joints = np.zeros(8, dtype=np.float32)
    targets = {1: 0.30, 2: 1.20, 3: -0.40}
    limits = dict.fromkeys(targets, 0.058)
    synced = joint_step(joints, targets, limits, sync=True)
    delta = np.array([synced[i] for i in targets])
    wanted = np.array(list(targets.values()))
    assert np.max(np.abs(delta)) == pytest.approx(0.058, rel=1e-5)  # the slowest joint at its bound
    assert delta / np.linalg.norm(delta) == pytest.approx(wanted / np.linalg.norm(wanted), rel=1e-5)
    # Per-joint clipping (every other task) moves each joint at its own bound and bends the path.
    clipped = joint_step(joints, targets, limits, sync=False)
    assert [clipped[i] for i in targets] == pytest.approx([0.058, 0.058, -0.058], rel=1e-5)
    # Close to the target both reach it in one step.
    near = {1: 0.01, 2: -0.02}
    for sync in (True, False):
        step = joint_step(joints, near, dict.fromkeys(near, 0.058), sync=sync)
        assert [step[1], step[2]] == pytest.approx([0.01, -0.02], rel=1e-5)
    assert joints == pytest.approx(np.zeros(8))  # the input is not modified
