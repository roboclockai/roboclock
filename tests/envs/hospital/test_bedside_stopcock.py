"""B4 line switch: two-step predicates (clamp, then stopcock), plain-USD mechanisms and placement."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.config import EpisodeSpec, load_config
from roboclock.envs.hospital.tasks.bedside.stopcock import (
    CLAMP,
    CLAMP_TRAVEL,
    HANDLE,
    ORDER_TEXT,
    StopcockSwitch,
)
from roboclock.envs.lab.mdp.state import PhysicalState

LIMITS = (("easy", 6.3), ("medium", 5.11), ("hard", 4.15))


def task(limit=4.0):
    spec = EpisodeSpec("B4", "easy", 3, 10.0, 1, 0.1, event_time=0.5, response_limit=limit)
    result = StopcockSwitch(spec)
    result.bind(SimpleNamespace(objects={}, targets={}, table_height=0.75))
    return result


def frame(
    angle=0.0, clamp=0.0, *, contact=True, grasped=True, speed=0.0,
    clamp_contact=False, clamp_speed=0.0,
):
    return PhysicalState(
        {},
        np.array([0.46, 0.0, 0.85]),
        controls={
            "stopcock_angle_rad": np.deg2rad(angle),
            "stopcock_speed_rad_s": speed,
            "stopcock_axis_error_rad": 0.0,
            "stopcock_pivot_error_m": 0.0,
            "stopcock_robot_contact": contact,
            "stopcock_grasped": grasped,
            "clamp_position_m": clamp,
            "clamp_speed_m_s": clamp_speed,
            "clamp_axis_error_m": 0.0,
            "clamp_contact": clamp_contact,
            "clamp_grasped": False,
        },
    )


def tick(t, at, angle=0.0, clamp=0.0, **kwargs):
    t.time = at
    t.process(0.02)
    t.observe(frame(angle, clamp, **kwargs), 0.02)


def close_clamp(t, start=0.6):
    """Push the roller to CLOSED with contact, then release it and hold for the retention."""
    if t.event_time is None:
        tick(t, 0.5, contact=False, grasped=False)  # the order is issued
    tick(t, start, 0, 0.012, contact=False, grasped=False, clamp_contact=True, clamp_speed=0.2)
    tick(t, start + 0.1, 0, CLAMP_TRAVEL, contact=False, grasped=False, clamp_contact=True)
    for k in range(4):
        tick(t, start + 0.2 + 0.02 * k, 0, CLAMP_TRAVEL, contact=False, grasped=False)
    return start + 0.24  # the third released 20 ms sample completes the 60 ms


def turn(t, start, final=90, clamp=CLAMP_TRAVEL):
    tick(t, start, 0, clamp)
    tick(t, start + 0.3, 45, clamp, speed=1.0)
    for k in range(5):
        tick(t, start + 0.6 + 0.02 * k, final, clamp)
    return start + 0.68


def test_clamp_then_stopcock_succeeds_and_completion_includes_retention():
    t = task()
    closed = close_clamp(t)
    assert t.clamp_time == pytest.approx(closed) and CLAMP in t.correct
    assert not t.goal_achieved
    done = turn(t, 1.2)
    assert t.physical_success and t.temporal_success and t.joint_success
    # The clock stops when the 80 ms detent retention completes, not at the detent entry.
    assert t.first_target_entry == pytest.approx(1.8)
    assert t.action_time == pytest.approx(done) == pytest.approx(1.88)
    assert t.correct == {CLAMP, HANDLE}
    m = t.metrics()
    assert m["switched_after_order_s"] == pytest.approx(1.38)
    assert m["clamp_closed_after_order_s"] == pytest.approx(closed - 0.5)
    assert m["fluid_simulated"] is False and not m["wrong_order"]
    assert m["step_order"] == [CLAMP, HANDLE] and m["order_text"] == ORDER_TEXT


def test_turning_the_stopcock_before_the_clamp_is_a_wrong_order_failure():
    t = task()
    tick(t, 0.6, 0, 0.0)
    tick(t, 0.8, 30, 0.0, speed=1.0)
    assert t.wrong_order and "wrong_order" in t.physical_failures
    close_clamp(t, 1.0)
    turn(t, 1.4)
    assert not t.physical_success and not t.joint_success


def test_stopcock_in_detent_does_not_count_until_the_clamp_is_retained():
    t = task()
    # Clamp pushed to CLOSED but still touched by the hand: not released, so not complete.
    tick(t, 0.6, 0, CLAMP_TRAVEL, contact=False, grasped=False, clamp_contact=True)
    for k in range(10):
        tick(t, 0.7 + 0.02 * k, 0, CLAMP_TRAVEL, contact=False, grasped=False, clamp_contact=True)
    assert t.clamp_time is None
    # A turn now is out of order even though the roller sits at its stop.
    tick(t, 1.0, 10, CLAMP_TRAVEL, speed=1.0)
    assert t.wrong_order


def test_clamp_reopened_before_the_switch_blocks_step_two():
    t = task()
    close_clamp(t)
    done = turn(t, 1.2, clamp=0.02)  # the roller slid back open during the turn
    assert t.clamp_time is not None and not t.goal_achieved and t.action_time is None
    for k in range(5):  # closed again: the retained detent now completes step 2
        tick(t, done + 0.02 * (k + 1), 90, CLAMP_TRAVEL)
    assert t.physical_success


def test_unreleased_or_contactless_clamp_motion_never_completes_step_one():
    t = task()
    tick(t, 0.6, 0, CLAMP_TRAVEL, contact=False, grasped=False)  # no contact ever witnessed
    for k in range(6):
        tick(t, 0.7 + 0.02 * k, 0, CLAMP_TRAVEL, contact=False, grasped=False)
    assert t.clamp_time is None
    t = task()
    tick(t, 0.6, 0, CLAMP_TRAVEL, contact=False, grasped=False, clamp_contact=True)
    for k in range(6):  # released but still sliding faster than 12 mm/s
        tick(t, 0.7 + 0.02 * k, 0, CLAMP_TRAVEL, contact=False, grasped=False, clamp_speed=0.05)
    assert t.clamp_time is None


def test_device_motion_before_the_order_is_an_early_action():
    t = task()
    tick(t, 0.2, 0, 0.004, contact=False, grasped=False, clamp_contact=True)
    assert "early_action" in t.violations
    close_clamp(t, 0.6)
    turn(t, 1.2)
    assert t.physical_success and not t.temporal_success
    t = task()
    tick(t, 0.3, 10, 0.0, speed=1.0)
    assert "early_action" in t.violations and t.wrong_order


def test_late_completion_keeps_physical_success_separate():
    t = task(limit=1.5)
    close_clamp(t)
    turn(t, 1.4)  # retained at 2.08 s, the deadline is 2.0 s
    assert t.physical_success and not t.temporal_success
    assert "late_action" in t.violations


def test_retention_completing_exactly_at_the_deadline_is_within_it():
    t = task(limit=1.38)
    close_clamp(t)
    turn(t, 1.2)  # retained at 1.88 s = 0.5 + 1.38
    assert t.joint_success


def test_unretained_detent_entry_does_not_reserve_the_deadline():
    t = task()
    close_clamp(t)
    tick(t, 1.2, 0, CLAMP_TRAVEL)
    tick(t, 1.5, 90, CLAMP_TRAVEL)
    tick(t, 1.52, 75, CLAMP_TRAVEL)
    for k in range(5):
        tick(t, 1.6 + 0.02 * k, 90, CLAMP_TRAVEL)
    assert t.first_target_entry == pytest.approx(1.5)
    assert t.action_time == pytest.approx(1.68)


@pytest.mark.parametrize("contact,grasped", [(False, False), (True, False), (False, True)])
def test_ungrasped_or_noncontact_rotation_never_succeeds(contact, grasped):
    t = task()
    close_clamp(t)
    tick(t, 1.2, 0, CLAMP_TRAVEL, contact=contact, grasped=grasped)
    for k in range(6):
        tick(t, 1.5 + 0.02 * k, 90, CLAMP_TRAVEL, contact=contact, grasped=grasped)
    assert not t.physical_success


def test_fast_pass_through_and_wrong_direction_do_not_complete():
    t = task()
    close_clamp(t)
    tick(t, 1.2, 0, CLAMP_TRAVEL)
    tick(t, 1.4, 90, CLAMP_TRAVEL, speed=3.0)
    for k in range(6):
        tick(t, 1.5 + k * 0.02, -90, CLAMP_TRAVEL)
    assert not t.physical_success and t.action_time is None


def test_missing_measurements_never_complete():
    t = task()
    for _ in range(10):
        t.observe(replace(frame(), controls={}), 0.02)
    assert not t.physical_success and t.clamp_time is None


@pytest.mark.parametrize(
    "measurement,value",
    [
        ("stopcock_axis_error_rad", 0.2),
        ("stopcock_axis_error_rad", float("nan")),
        ("stopcock_pivot_error_m", 0.01),
    ],
)
def test_detached_or_tilted_handle_cannot_score(measurement, value):
    t = task()
    close_clamp(t)
    tick(t, 1.2, 0, CLAMP_TRAVEL)
    bad = frame(90, CLAMP_TRAVEL)
    bad.controls[measurement] = value
    for k in range(8):
        t.time = 1.5 + k * 0.02
        t.observe(bad, 0.02)
    assert not t.physical_success


def test_roller_out_of_track_is_a_physical_failure():
    t = task()
    bad = frame(0, CLAMP_TRAVEL)
    bad.controls["clamp_axis_error_m"] = 0.01
    t.time = 0.7
    t.observe(bad, 0.02)
    assert "clamp_out_of_track" in t.physical_failures


# --- layout, mechanisms and placement ------------------------------------------------------------


def _builder(seed=0, level="hard"):
    from roboclock.envs.hospital.bedside.stopcock import build_layout
    from roboclock.envs.hospital.layout import _Builder, load_catalog

    config = load_config()
    t = StopcockSwitch(replace(task().spec, seed=seed, difficulty=level))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    return build_layout(_Builder(t, config, load_catalog(), rng)), config


def _author(b, config):
    from pxr import Usd, UsdGeom, UsdPhysics

    from roboclock.envs.hospital.bedside.stopcock import ROLLER_ROOT, ROOT, author
    from tests.envs.hospital.test_hospital_scene_usd import StubAppearance

    stage = Usd.Stage.CreateInMemory()
    for path in (ROOT, ROLLER_ROOT):
        root = UsdGeom.Xform.Define(stage, path)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdGeom.Cube.Define(stage, ROOT + "/GenericCollider")
    box = UsdGeom.Cube.Define(stage, ROLLER_ROOT + "/Collider")
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    UsdGeom.Cube.Define(stage, ROLLER_ROOT + "/Visual")
    look = StubAppearance(stage, None, config)
    backend = SimpleNamespace(stage=stage, appearance=look, layout=SimpleNamespace(info=b.info))
    author(backend)
    return stage


def test_passive_revolute_and_prismatic_joints_are_authored_without_motors():
    pytest.importorskip("pxr")
    from pxr import UsdGeom, UsdPhysics

    from roboclock.envs.hospital.bedside.stopcock import (
        JOINT_PATH,
        NAME,
        ROLLER_JOINT_PATH,
        ROLLER_NAME,
        ROLLER_ROOT,
        ROOT,
    )

    b, config = _builder(7)
    assert set(b.objects) == {NAME, ROLLER_NAME} and not b.targets and not b.held
    stage = _author(b, config)
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(JOINT_PATH))
    assert joint.GetBody1Rel().GetTargets() == [stage.GetPrimAtPath(ROOT).GetPath()]
    assert joint.GetAxisAttr().Get() == "Z"
    assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == (0.0, 90.0)
    drive = UsdPhysics.DriveAPI(joint.GetPrim(), "angular")
    assert drive.GetStiffnessAttr().Get() == 0.0 and drive.GetTargetVelocityAttr().Get() == 0.0
    assert not stage.GetPrimAtPath(ROOT + "/GenericCollider").IsValid()
    assert stage.GetPrimAtPath(ROOT + "/Hub").HasAPI(UsdPhysics.CollisionAPI)
    slide = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(ROLLER_JOINT_PATH))
    assert slide.GetBody1Rel().GetTargets() == [stage.GetPrimAtPath(ROLLER_ROOT).GetPath()]
    assert slide.GetAxisAttr().Get() == "X"
    assert slide.GetUpperLimitAttr().Get() == pytest.approx(CLAMP_TRAVEL)
    # The joint frame's x axis is world +y (the line direction): a 90-degree yaw.
    rot = slide.GetLocalRot0Attr().Get()
    assert rot.GetReal() == pytest.approx(np.sqrt(0.5)) and rot.GetImaginary()[2] == pytest.approx(np.sqrt(0.5))
    np.testing.assert_allclose(slide.GetLocalPos0Attr().Get(), b.info["clamp"]["origin"], atol=1e-6)
    linear = UsdPhysics.DriveAPI(slide.GetPrim(), "linear")
    assert linear.GetStiffnessAttr().Get() == 0.0 and linear.GetDampingAttr().Get() > 0.0
    # The backend's box collider is kept, its generic visual hidden.
    assert stage.GetPrimAtPath(ROLLER_ROOT + "/Collider").IsValid()
    visual = UsdGeom.Imageable(stage.GetPrimAtPath(ROLLER_ROOT + "/Visual"))
    assert visual.ComputeVisibility() == UsdGeom.Tokens.invisible
    # The roller body is yawed 90 degrees so its local x (the B5 slide axis) runs along +y.
    q = b.objects[ROLLER_NAME].orientation
    np.testing.assert_allclose(q, [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], atol=1e-9)


def test_dressing_authors_unique_visual_prims_and_only_device_colliders():
    pytest.importorskip("pxr")
    from pxr import UsdPhysics

    from roboclock.envs.hospital.bedside.stopcock import ROOT

    b, config = _builder(1)
    stage = _author(b, config)
    paths = [str(p.GetPath()) for p in stage.Traverse()]
    assert len(paths) == len(set(paths))
    for required in (
        "/World/Bedside/Stopcock/PatientLine",
        "/World/Bedside/Stopcock/MaintenanceLine",
        "/World/Bedside/Stopcock/RequestedLine",
        "/World/Bedside/Stopcock/Strap0",
        "/World/Bedside/OrderTerminal",
        "/World/Bedside/OrderScreen/Screen",
        "/World/Bedside/ClampPlates/CLOSED/Plate",
        "/World/Bedside/StopcockTags/Tag_0/Label",
        "/World/Bedside/RobotBase",
        "/World/Bedside/IVStand",
        ROOT + "/HubArrow",
    ):
        assert stage.GetPrimAtPath(required).IsValid(), required
    colliders = {
        str(p.GetPath()).rsplit("/", 1)[-1]
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.CollisionAPI)
    }
    assert colliders == {"Hub", "WingXCollider", "WingYCollider", "Housing", "Collider"}


def test_measurement_rejects_bad_quaternions_and_reads_roller_travel_along_the_line():
    from roboclock.envs.hospital.bedside.stopcock import NAME, ROLLER_NAME, measure
    from roboclock.envs.lab.mdp.state import ObjectState

    pivot = np.array([0.46, 0.045, 0.84])
    origin = np.array([0.46, -0.061, 0.804])
    backend = SimpleNamespace(
        objects={NAME: SimpleNamespace(get_angular_velocity=lambda: np.zeros(3))},
        layout=SimpleNamespace(
            info={
                "stopcock": {"pivot": pivot.tolist()},
                "clamp": {"origin": origin.tolist(), "axis": [0.0, 1.0, 0.0]},
            }
        ),
    )
    roller = ObjectState(origin + [0.001, 0.030, 0.0], np.array([1.0, 0, 0, 0]), robot_contact=True)
    for quat in ([0.5, 0.5, 0.5, 0.5], [float("nan"), 0, 0, 0], [2.0, 0, 0, 0]):
        state = PhysicalState({NAME: ObjectState(pivot, np.asarray(quat)), ROLLER_NAME: roller}, pivot)
        assert measure(backend, state)["stopcock_axis_error_rad"] > 0.05
    valid = ObjectState(pivot, np.array([2**-0.5, 0.0, 0.0, 2**-0.5]))
    result = measure(backend, PhysicalState({NAME: valid, ROLLER_NAME: roller}, pivot))
    assert result["stopcock_angle_rad"] == pytest.approx(np.pi / 2)
    assert result["stopcock_pivot_error_m"] == pytest.approx(0.0)
    assert result["clamp_position_m"] == pytest.approx(0.030)
    assert result["clamp_axis_error_m"] == pytest.approx(0.001)
    assert result["clamp_contact"] and not result["clamp_grasped"]


def test_layout_matrix_is_the_same_at_every_level():
    """The line switch left the suite on 2026-11-19 (B4 is now the ventilator reconnection,
    test_bedside_ventilator.py), so its layouts are built here without the registry."""
    from roboclock.envs.hospital.bedside.stopcock import NAME, ROLLER_NAME

    for seed in range(10):
        baseline = None
        for level, _ in LIMITS:
            b, _config = _builder(seed, level)
            positions = np.array([b.objects[n].position for n in (NAME, ROLLER_NAME)])
            if baseline is None:
                baseline = positions
            np.testing.assert_array_equal(positions, baseline)


def test_devices_are_far_out_but_inside_the_top_down_reach_for_every_seed():
    from roboclock.envs.hospital.bedside.stopcock import NAME, ROLLER_NAME, _waypoints
    from roboclock.envs.lab.robot import mount_positions

    for seed in range(12):
        b, config = _builder(seed)
        mount = np.asarray(mount_positions(config, "B4")[0])
        pivot = b.objects[NAME].position
        roller = b.objects[ROLLER_NAME].position
        for point in (pivot, roller):
            # Visibly farther than last round's 0.29 m, reaching over the rail.
            assert 0.43 <= np.hypot(*(point - mount)[:2]) <= 0.50
        start, end, grip = _waypoints(b.info)
        # The top-down X5 reaches about 0.53 m horizontally at these heights (IK-checked).
        for point in (start, end, grip):
            assert np.hypot(*(point - mount)[:2]) <= 0.505
        rail_out = b.info["stopcock_scene"]["rail"]["out_x"]
        assert pivot[0] - rail_out > 0.1 and roller[0] - rail_out > 0.1
        # Both devices sit on one holder, a few centimetres apart along the line.
        assert abs(pivot[0] - roller[0]) < 1e-9 and 0.06 < pivot[1] - roller[1] < 0.12


def test_robot_starts_tucked_at_its_base_not_at_the_devices():
    from roboclock.envs.hospital.bedside.stopcock import HAND_YAW, NAME, PUSH_WIDTH
    from roboclock.envs.lab.robot import mount_positions

    b, config = _builder(2)
    mount = np.asarray(mount_positions(config, "B4")[0])
    initial = np.asarray(b.info["initial_ee_position"])
    assert np.hypot(*(initial - mount)[:2]) < 0.30 and abs(initial[2] - mount[2]) < 0.1
    assert np.linalg.norm(initial - b.objects[NAME].position) > 0.25
    assert initial[0] < b.info["stopcock_scene"]["rail"]["out_x"] - 0.05  # behind the bed rail
    assert b.info["initial_ee_yaw"] == HAND_YAW and b.info["initial_gripper_width"] == PUSH_WIDTH


def _to_ward(point, pose):
    """World (robot frame) -> ward file frame for the 180-degree ward pose."""
    assert pose["yaw"] == 180.0
    return np.asarray(pose["translate"][:2]) - np.asarray(point[:2])


def test_holder_and_order_terminal_hang_on_the_rail_flat_top_for_every_seed():
    from roboclock.envs.hospital.bedside import stopcock as module

    for seed in range(8):
        b, _ = _builder(seed)
        info = b.info
        pose = info["ward_pose"]
        assert info["hide_bench"] and pose["usd"].endswith("ward_02_preview_left.usdc")
        assert pose["translate"][2] == 0.0
        pivot = _to_ward(b.objects[module.NAME].position, pose)
        np.testing.assert_allclose(
            pivot, [module.RAIL_WARD_X - (module.LINE_X - module.RAIL_OUT), module.GROUP_WARD_Y - module.PIVOT_XY[1]],
            atol=1e-9,
        )
        scene = info["stopcock_scene"]
        assert _to_ward([scene["rail"]["out_x"], 0.0], pose)[0] == pytest.approx(module.RAIL_WARD_X)
        lo, hi = module.RAIL_FLAT_WARD_Y
        for sy in scene["straps_y"]:
            assert lo + 0.02 < _to_ward([0.0, sy], pose)[1] < hi - 0.02
        dial = _to_ward(scene["dial"], pose)
        assert lo - 1e-9 <= dial[1] - 0.03 and dial[1] + 0.03 <= hi + 1e-9  # the 60 mm clip
        assert module.RAIL_WARD_X - module.RAIL_THICK <= dial[0] <= module.RAIL_WARD_X


def test_work_surface_is_the_holder_shelf_and_the_terminal_has_a_matching_collider():
    b, config = _builder(3)
    surface = b.info["work_surface"]
    center, size = np.asarray(surface["center"]), np.asarray(surface["size"])
    shelf = next(f for f in b.fixtures if f.name == "stopcock_rail_shelf")
    assert not shelf.collision
    np.testing.assert_allclose(center, shelf.position, atol=1e-9)
    np.testing.assert_allclose(size, shelf.size, atol=1e-9)
    for name in ("stopcock_mount", "clamp_tray"):
        fixture = next(f for f in b.fixtures if f.name == name)
        assert fixture.position[2] - fixture.size[2] / 2 == pytest.approx(center[2] + size[2] / 2)
    for obj in b.objects.values():
        assert np.all(np.abs(obj.position[:2] - center[:2]) + 0.028 < size[:2] / 2)
    rail = b.info["stopcock_scene"]["rail"]
    assert center[2] - size[2] / 2 > rail["mattress_top"] + 0.05
    terminal = next(f for f in b.fixtures if f.name == "order_terminal")
    assert terminal.collision and not terminal.visible
    assert terminal.position[2] - terminal.size[2] / 2 == pytest.approx(rail["top"] + 0.008)
    # The backend's 0.10 m dial housing sits inside the terminal, flush with its top.
    housing_top = b.indicators[0].position[2] + 0.005 + 0.05
    assert housing_top == pytest.approx(terminal.position[2] + terminal.size[2] / 2)


def test_mobile_base_stands_clear_of_the_bed_and_ultrasound_cart():
    import json
    from pathlib import Path

    from roboclock.envs.hospital.bedside.ward import ULTRASOUND_CART, part_bounds
    from roboclock.envs.lab.robot import mount_positions

    parts_file = Path(__file__).resolve().parents[3] / "assets/hospital/ward_02_preview_left.json"
    if not parts_file.exists():
        pytest.skip("ward conversion not present")
    parts = json.loads(parts_file.read_text())["parts"]
    for seed in range(8):
        b, config = _builder(seed)
        pose = b.info["ward_pose"]
        bed = part_bounds(pose, parts, {"KR_007", "KR_009", "KR_010"})
        cart = part_bounds(pose, parts, set(ULTRASOUND_CART))  # pushed clear of the rig (WARD_MOVE)
        mounts = np.asarray(mount_positions(config, "B4"))[:, :2]
        # Both rig columns on one wheeled base: base shell + bumper, plus the idle arm's home pose
        # (its hand folds about 0.3 m in front of its column).
        lo, hi = mounts.min(0) - 0.12, mounts.max(0) + np.array([0.30, 0.12])
        for other_lo, other_hi in (bed, cart):
            assert np.any(hi <= other_lo) or np.any(lo >= other_hi)


# --- oracle controller -----------------------------------------------------------------------


def _controller(phase, angle=0.0, clamp=0.0, *, grasped=False, contact=False, at=None, clamp_time=None):
    from roboclock.envs.hospital.bedside.stopcock import NAME, ROLLER_NAME, _waypoints
    from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory
    from roboclock.envs.lab.mdp.state import ObjectState

    b, config = _builder(0)
    t = task()
    t.layout = SimpleNamespace(info=b.info)
    t.event_time, t.time, t.clamp_time = 0.5, 1.0, clamp_time
    start, end, grip = _waypoints(b.info)
    pivot = b.objects[NAME].position
    objects = {
        NAME: ObjectState(pivot, np.array([1.0, 0, 0, 0]), grasped=grasped, robot_contact=contact),
        ROLLER_NAME: ObjectState(b.objects[ROLLER_NAME].position, np.array([1.0, 0, 0, 0])),
    }
    ee = {"start": start, "end": end, "grip": grip}.get(at, grip) if at else grip
    state = PhysicalState(
        objects, np.asarray(ee, dtype=float),
        controls={"stopcock_angle_rad": np.deg2rad(angle), "clamp_position_m": clamp},
    )
    p = HospitalTrajectory(config["control"], 1 / 30)
    p.phase = phase
    p.last_goal = state.ee_position.copy()
    return p, t, state, (start, end, grip)


def test_controller_waits_tucked_before_the_order():
    from roboclock.envs.hospital.bedside.stopcock import HAND_YAW, PUSH_WIDTH, command

    p, t, state, _ = _controller("select")
    t.event_time = None
    tucked = np.asarray(t.layout.info["initial_ee_position"])
    state = replace(state, ee_position=tucked)
    p.last_goal = tucked.copy()
    for _ in range(5):
        result = command(p, t, state)
        assert p.phase == "wait_order" and not result.close and result.yaw == HAND_YAW
        np.testing.assert_allclose(result.position, tucked)
    assert p.open_width == PUSH_WIDTH


def test_controller_pushes_the_clamp_then_never_turns_before_its_closure_is_confirmed():
    from roboclock.envs.hospital.bedside.stopcock import HAND_YAW, command

    p, t, state, (start, end, grip) = _controller("push_clamp", at="start")
    result = command(p, t, state)
    assert p.phase == "push_clamp" and not result.close
    assert result.position[1] > start[1]  # toward CLOSED (+y, downstream)
    # Gripping the hub before the clamp is confirmed is allowed; turning is not.
    p, t, state, _ = _controller("pinch_stopcock", clamp=CLAMP_TRAVEL, grasped=True, contact=True)
    result = command(p, t, state)
    assert p.phase == "pinch_stopcock" and result.close and result.yaw == pytest.approx(HAND_YAW)
    t.clamp_time = 1.0
    result = command(p, t, state)
    assert p.phase == "turn_stopcock" and result.yaw > HAND_YAW


def test_controller_turns_past_the_detent_only_by_the_overshoot_and_releases_after_retention():
    from roboclock.envs.hospital.bedside.stopcock import HAND_YAW, TURN_OVERSHOOT, command
    from roboclock.envs.hospital.tasks.bedside.stopcock import TARGET_ANGLE

    p, t, state, (_, _, grip) = _controller(
        "turn_stopcock", angle=88.5, clamp=CLAMP_TRAVEL, grasped=True, contact=True, clamp_time=1.0
    )
    p.turn = np.deg2rad(89.0)
    for _ in range(20):
        result = command(p, t, state)
        assert result.close and p.phase == "turn_stopcock"
        assert result.yaw <= HAND_YAW + TARGET_ANGLE + TURN_OVERSHOOT + 1e-12
        np.testing.assert_allclose(result.position, grip)
    t.goal_achieved = True
    released = command(p, t, state)
    assert not released.close and p.phase == "retain_detent"
    assert released.position[2] > state.ee_position[2]
