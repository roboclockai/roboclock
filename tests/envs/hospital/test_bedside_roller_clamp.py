"""Measured B5 behavior (hang the bag, then open the roller), independent of the oracle's commands."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import roller_clamp as module
from roboclock.envs.hospital.bedside.roller_clamp import (
    BAG_NAME,
    HOOK_CONTACT,
    ROLLER_NAME,
    author,
    build_layout,
    hang_geometry,
    measure,
)
from roboclock.envs.hospital.config import EpisodeSpec, load_config
from roboclock.envs.hospital.layout import _Builder, load_catalog
from roboclock.envs.hospital.tasks.bedside.roller_clamp import (
    HANG_DWELL,
    INSTRUCTION_TEXT,
    RETENTION_DWELL,
    TRAVEL,
    RollerClamp,
    episode_mode,
)
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState

UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def b5_task():
    def factory(seed=0, difficulty="easy", limit=1.5):
        spec = EpisodeSpec(
            "B5", difficulty, seed, 12.0, 1, 0.15, event_time=2.0, response_limit=limit
        )
        task = RollerClamp(spec)
        builder = _Builder(task, load_config(), load_catalog(), np.random.default_rng(seed))
        build_layout(builder)
        task.bind(
            SimpleNamespace(
                objects=builder.objects,
                shapes=builder.shapes,
                targets={},
                table_height=builder.z,
                info=builder.info,
                fixtures=builder.fixtures,
                indicators=builder.indicators,
            )
        )
        return task

    return factory


def bag_state(task, where="cradle", *, grasped=False, contact=False, hook=True, speed=0.0):
    """The bag in its cradle, carried, or hanging on the peg (eyelet's top bar on the peg)."""
    hang = np.asarray(task.layout.info["roller_clamp"]["hook"]["hang_position"])
    position = {
        "cradle": task.layout.objects[BAG_NAME].position,
        "carried": hang + [0.0, -0.12, 0.03],
        "hook": hang,
    }[where]
    contacts = frozenset({HOOK_CONTACT}) if where == "hook" and hook else frozenset()
    return ObjectState(
        np.asarray(position, dtype=float).copy(),
        UPRIGHT,
        velocity=np.array([0.0, 0.0, speed]),
        grasped=grasped,
        robot_contact=contact or grasped,
        fixture_contacts=contacts,
    )


def observe(
    task,
    time,
    roller=0.0,
    bag="cradle",
    *,
    roller_contact=False,
    roller_grasped=False,
    roller_speed=0.0,
    axis_error=0.0,
    **bag_kwargs,
):
    origin = task.layout.info["roller_clamp"]["origin"]
    objects = {
        ROLLER_NAME: ObjectState(
            origin + [roller, axis_error, 0.0],
            UPRIGHT,
            velocity=np.array([roller_speed, 0.0, 0.0]),
            robot_contact=roller_contact,
            grasped=roller_grasped,
        ),
        BAG_NAME: bag if isinstance(bag, ObjectState) else bag_state(task, bag, **bag_kwargs),
    }
    state = PhysicalState(objects, np.array([0.4, 0.0, 1.1]))
    state.controls.update(measure(SimpleNamespace(layout=task.layout), state))
    task.time = time
    task.process(0.01)
    task.observe(state, 0.01)
    return state


def hang(task, start=2.1):
    """Carry the bag over, then leave it released on the hook past the hang dwell."""
    observe(task, task.spec.event_time)
    observe(task, start - 0.05, bag="carried", grasped=True)
    t = start
    while t < start + HANG_DWELL + 0.02:
        observe(task, t, bag="hook")
        t = round(t + 0.01, 4)
    return t


def open_roller(task, at, bag="hook"):
    """Push the roller to OPEN (witnessed contact), then release it there for the retention."""
    observe(task, at - RETENTION_DWELL - 0.01, TRAVEL / 2, bag, roller_contact=True)
    observe(task, at - RETENTION_DWELL, TRAVEL, bag, roller_contact=True)
    for tick in range(6):  # released at the stop: 6 x 10 ms of retention, complete at `at`
        observe(task, round(at - RETENTION_DWELL + 0.01 + tick * 0.01, 6), TRAVEL, bag)


def test_instruction_names_both_steps_in_order(b5_task):
    task = b5_task()
    text = task.gen_instruction()
    assert INSTRUCTION_TEXT == "HANG SALINE, THEN OPEN CLAMP" and text.startswith(INSTRUCTION_TEXT)
    assert text.index("hang it") < text.index("fully OPEN")
    assert task.name == "hang_bag_open_clamp" and task.mode == "open"
    # The historical single-slide direction helper is unchanged for tasks that reuse the roller.
    assert episode_mode(0) == "open" and episode_mode(1) == "close"


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_hang_then_open_succeeds_with_released_retention(b5_task, seed):
    task = b5_task(seed)
    t = hang(task)
    assert task.hang_time == pytest.approx(2.1) and BAG_NAME in task.correct
    assert not task.goal_achieved
    open_roller(task, at=t + 0.3)
    assert task.physical_success and task.temporal_success
    assert task.action_time == pytest.approx(t + 0.3)
    assert task.correct == {BAG_NAME, ROLLER_NAME}
    metrics = task.metrics()
    assert metrics["hang_latency"] == pytest.approx(0.1)
    assert metrics["deadline_event"] == "bag_hung_then_released_open_roller_retained"
    assert metrics["steps"] == ["hang_bag", "open_roller_clamp"]


def test_opening_the_clamp_before_the_bag_hangs_is_a_wrong_order_failure(b5_task):
    task = b5_task()
    observe(task, 2.0)
    observe(task, 2.1, TRAVEL / 2, roller_contact=True)
    assert "wrong_order" in task.physical_failures
    t = hang(task, 2.3)
    open_roller(task, at=t + 0.2)
    assert not task.physical_success
    assert task.metrics()["wrong_order"]


def test_hanging_needs_release_hook_support_stillness_and_dwell(b5_task):
    task = b5_task()
    observe(task, 2.0)
    for tick in range(30):  # still held on the hook
        observe(task, 2.1 + tick * 0.01, bag="hook", grasped=True)
    assert task.hang_time is None
    for tick in range(30):  # released but the peg is not measured supporting it
        observe(task, 2.5 + tick * 0.01, bag="hook", hook=False)
    assert task.hang_time is None
    for tick in range(30):  # swinging
        observe(task, 2.9 + tick * 0.01, bag="hook", speed=0.2)
    assert task.hang_time is None
    for tick in range(int(HANG_DWELL / 0.01) - 3):  # released and supported, but not long enough
        observe(task, 3.3 + tick * 0.01, bag="hook")
    observe(task, 3.6, bag="carried", grasped=True)
    assert task.hang_time is None
    # A wrong-order push is still refused afterwards: the hang never held.
    observe(task, 3.7, TRAVEL / 2, "carried", roller_contact=True)
    assert "wrong_order" in task.physical_failures


def test_roller_completion_requires_the_bag_still_hanging(b5_task):
    task = b5_task()
    t = hang(task)
    fallen = bag_state(task, "hook", hook=False)
    open_roller(task, at=t + 0.3, bag=fallen)
    assert BAG_NAME not in task.correct
    assert not task.goal_achieved and task.action_time is None
    for tick in range(5):
        observe(task, t + 0.31 + tick * 0.01, TRAVEL, "hook")
    assert task.goal_achieved and task.action_time == pytest.approx(t + 0.31)


@pytest.mark.parametrize("completion,valid", [(3.49, True), (3.5, True), (3.51, False)])
def test_inclusive_completion_boundary(b5_task, completion, valid):
    task = b5_task(limit=1.5)
    hang(task)
    open_roller(task, at=completion)
    assert task.physical_success
    assert task.temporal_success is valid
    assert ("late_response" in task.violations) is (not valid)


def test_handling_the_bag_before_the_instruction_is_latched(b5_task):
    task = b5_task()
    observe(task, 1.5, bag="carried", grasped=True)
    observe(task, 1.6)
    t = hang(task)
    open_roller(task, at=t + 0.3)
    assert task.physical_success and not task.temporal_success
    assert "early_action" in task.violations


def test_touching_the_roller_before_the_instruction_is_latched(b5_task):
    task = b5_task()
    observe(task, 1.5, 0.0, roller_contact=True)
    t = hang(task)
    open_roller(task, at=t + 0.3)
    assert "early_action" in task.violations and not task.temporal_success


def test_missed_deadline_remains_failure_after_late_completion(b5_task):
    task = b5_task(limit=1.0)
    observe(task, 2.0)
    observe(task, 3.05)
    t = hang(task, 3.1)
    open_roller(task, at=t + 0.2)
    assert task.physical_success and not task.temporal_success
    assert "late_response" in task.violations


def test_held_moving_and_off_axis_endpoints_do_not_satisfy_retention(b5_task):
    task = b5_task()
    t = hang(task)
    observe(task, t, TRAVEL / 2, "hook", roller_contact=True)
    for tick in range(15):
        observe(
            task, t + 0.01 + tick * 0.01, TRAVEL, "hook", roller_contact=True, roller_grasped=True
        )
    for tick in range(15):
        observe(task, t + 0.2 + tick * 0.01, TRAVEL, "hook", roller_speed=0.05)
    assert not task.physical_success
    for tick in range(5):
        observe(task, t + 0.4 + tick * 0.01, TRAVEL, "hook")
    assert not task.physical_success
    observe(task, t + 0.45, TRAVEL, "hook")
    assert task.physical_success
    observe(task, t + 0.5, TRAVEL / 2, "hook")
    assert not task.goal_achieved
    for tick in range(10):
        observe(task, t + 0.6 + tick * 0.01, TRAVEL, "hook", axis_error=0.006)
    assert "roller_out_of_track" in task.physical_failures


def test_pose_without_robot_contact_is_not_a_valid_slide(b5_task):
    task = b5_task()
    t = hang(task)
    for tick in range(20):
        observe(task, t + tick * 0.01, TRAVEL, "hook")
    assert not task.physical_success and task.action_time is None


def test_measure_uses_observed_bodies_not_commands(b5_task):
    task = b5_task()
    state = observe(task, 2.0, TRAVEL / 2, "hook")
    state.controls["roller_open_fraction"] = 1.0
    state.controls["bag_hanging"] = False
    result = measure(SimpleNamespace(layout=task.layout), state)
    assert result["roller_position_m"] == pytest.approx(TRAVEL / 2)
    assert result["roller_open_fraction"] == pytest.approx(0.5)
    assert result["bag_hanging"] and result["bag_hook_contact"]


def test_hang_geometry_needs_the_eyelet_on_the_peg_and_upright(b5_task):
    task = b5_task()
    hook = task.layout.info["roller_clamp"]["hook"]
    on = bag_state(task, "hook")
    assert hang_geometry(task.layout, on)["hanging"]
    # Beyond the peg's free end the eyelet cannot be on it.
    off = ObjectState(on.position - [0.0, 0.12, 0.0], UPRIGHT, fixture_contacts=on.fixture_contacts)
    assert on.position[1] - 0.12 < hook["tip_y"] and not hang_geometry(task.layout, off)["hanging"]
    # Resting on the peg 1 cm too high (a body corner, not the eyelet's top bar).
    high = ObjectState(
        on.position + [0.0, 0.0, 0.01], UPRIGHT, fixture_contacts=on.fixture_contacts
    )
    assert not hang_geometry(task.layout, high)["hanging"]
    tilt = 0.3
    tilted = ObjectState(
        on.position,
        np.array([np.cos(tilt / 2), 0.0, np.sin(tilt / 2), 0.0]),
        fixture_contacts=on.fixture_contacts,
    )
    assert not hang_geometry(task.layout, tilted)["hanging"]


def test_geometry_is_constant_across_difficulty(b5_task):
    easy, hard = b5_task(7, "easy"), b5_task(7, "hard")
    for name in (ROLLER_NAME, BAG_NAME):
        assert np.array_equal(
            easy.layout.objects[name].position, hard.layout.objects[name].position
        )
    assert set(easy.layout.objects) == {ROLLER_NAME, BAG_NAME}


def test_usd_authors_the_roller_slide_bag_tab_line_and_ward_dressing(b5_task):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from roboclock.envs.hospital.appearance import HospitalAppearance

    class Appearance(HospitalAppearance):
        def material(self, name, **kwargs):
            return SimpleNamespace(material=UsdShade.Material.Define(self.stage, f"/Looks/{name}"))

        def glass(self):
            return self.material("glass")

    task = b5_task()
    stage = Usd.Stage.CreateInMemory()
    for name in (ROLLER_NAME, BAG_NAME):
        root = UsdGeom.Xform.Define(stage, f"/World/Objects/{name}")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        body = UsdGeom.Cube.Define(stage, f"/World/Objects/{name}/Body")
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    layout = task.layout
    layout.indicators = list(layout.indicators)
    backend = SimpleNamespace(
        stage=stage,
        layout=layout,
        appearance=Appearance(stage, None, load_config()),
        config=load_config(),
    )
    joint = author(backend)
    assert joint.GetAxisAttr().Get() == "X"
    assert joint.GetLowerLimitAttr().Get() == 0.0
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(TRAVEL)
    assert [str(path) for path in joint.GetBody1Rel().GetTargets()] == ["/World/Objects/roller_0"]
    drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/Joints/iv_roller_slide"), "linear")
    assert drive.GetStiffnessAttr().Get() == 0.0
    assert drive.GetTargetVelocityAttr().Get() == 0.0
    assert drive.GetDampingAttr().Get() > 0.0
    # The grasp tab is a collider on the free bag; the drip set under it is visual only.
    tab = stage.GetPrimAtPath("/World/Objects/bag_0/TrainerGripTab")
    assert tab.IsValid() and tab.HasAPI(UsdPhysics.CollisionAPI)
    assert not stage.GetPrimAtPath("/World/Objects/bag_0/DripChamber").HasAPI(
        UsdPhysics.CollisionAPI
    )
    for path in (
        "/World/Objects/roller_0/KnurledWheel",
        "/World/Objects/roller_0/Rib_23",
        "/World/Objects/bag_0/OutletPort",
        "/World/Bedside/RollerLine/Slack_00",
        "/World/Bedside/RollerLine/Tube_4",
        "/World/Bedside/RollerLine/Plate_CLOSED",
        "/World/Bedside/RollerLine/Plate_OPEN",
        "/World/Bedside/HookArm",
        "/World/Bedside/CabinetTop",
        "/World/Bedside/RobotBase",
        "/World/Bedside/InstructionDisplay/Screen/Text",
    ):
        assert stage.GetPrimAtPath(path).IsValid(), path
    assert (
        stage.GetPrimAtPath("/World/Bedside/IVStand").IsValid()
        or stage.GetPrimAtPath("/World/Bedside/IVPole").IsValid()
    )
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            assert np.isfinite(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation())).all()
    # The slack span follows the measured bag: its first segment leaves the drip chamber's outlet.
    first = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Bedside/RollerLine/Slack_00"))
    before = np.asarray(first.GetLocalTransformation().ExtractTranslation())
    state = observe(task, 2.0, 0.0, "hook")
    backend._a2_camera_for = None
    measure(backend, state)
    after = np.asarray(first.GetLocalTransformation().ExtractTranslation())
    assert np.linalg.norm(after - before) > 0.05


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ward_placement_reach_and_supports(b5_task, seed):
    from roboclock.envs.lab.robot import mount_positions

    task = b5_task(seed)
    info = task.layout.info
    rc, shift = info["roller_clamp"], np.asarray(info["roller_clamp"]["shift"])
    pose = info["ward_pose"]
    assert pose["yaw"] == pytest.approx(180.0) and info["hide_bench"]
    assert np.allclose(pose["translate"][:2], np.asarray(module.WARD_TRANSLATE[:2]) + shift)
    fixtures = {f.name: f for f in task.layout.fixtures}
    # The organizer's outer jaw touches the rail's outer face and its top plate rests on the rail.
    outer, top = fixtures["roller_clamp_outer"], fixtures["roller_clamp_top"]
    assert outer.position[0] + outer.size[0] / 2 == pytest.approx(module.RAIL_X[0] + shift[0])
    assert top.position[2] - top.size[2] / 2 == pytest.approx(module.RAIL_TOP)
    arm, tray = fixtures["roller_holder_arm"], fixtures["roller_holder_base"]
    assert arm.position[0] - arm.size[0] / 2 == pytest.approx(tray.position[0] + tray.size[0] / 2)
    # The hook peg runs from the pole along -y; the hanging bag's eyelet rests on its top.
    peg, pole = fixtures["iv_hook_peg"], fixtures["iv_stand_pole"]
    assert peg.position[1] + peg.size[1] / 2 == pytest.approx(pole.position[1])
    hang = np.asarray(rc["hook"]["hang_position"])
    bag = rc["bag"]
    eye_bar_bottom = (
        hang[2] + bag["body_size"][2] / 2 + bag["eyelet"]["height"] - bag["eyelet"]["bar"]
    )
    assert eye_bar_bottom == pytest.approx(peg.position[2] + peg.size[2] / 2)
    # The work surface is the cabinet top; the cradle shoulders stand on it.
    surface = info["work_surface"]
    assert surface["center"][2] + surface["size"][2] / 2 == pytest.approx(task.layout.table_height)
    assert fixtures["bag_cradle_left"].position[2] - fixtures["bag_cradle_left"].size[
        2
    ] / 2 == pytest.approx(task.layout.table_height)
    # Reach: the hook and roller hand poses sit near the top-down X5's limit (~0.53 m), with margin;
    # the pole stands farther out. The hand starts tucked at its own column.
    mount = np.asarray(mount_positions(load_config(), "B5")[0])
    hook_hand = hang + bag["grasp_offset"]
    roller_hand = np.asarray(rc["origin"]) + [TRAVEL - module.PUSH_OFFSET, 0.0, module.PUSH_HEIGHT]
    assert 0.46 <= np.hypot(*(hook_hand[:2] - mount[:2])) <= 0.51
    assert 0.44 <= np.hypot(*(roller_hand[:2] - mount[:2])) <= 0.48
    assert -0.10 <= hook_hand[2] - mount[2] <= 0.0 and -0.37 <= roller_hand[2] - mount[2] <= -0.30
    assert np.hypot(*(np.asarray(rc["hook"]["pole_xy"]) - mount[:2])) >= 0.54
    rest = np.asarray(info["initial_ee_position"])
    assert np.linalg.norm(rest[:2] - mount[:2]) < 0.2 and abs(rest[2] - mount[2]) < 0.1
    # The response dial and the instruction console stand on the cabinet top, clear of the bag.
    dial = task.layout.indicators[0].position
    assert np.linalg.norm(dial[:2] - task.layout.objects[BAG_NAME].position[:2]) > 0.15


def test_controller_waits_tucked_until_the_instruction(b5_task):
    from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory

    task = b5_task()
    planner = HospitalTrajectory(load_config()["control"], 4 / 120)
    rest = np.asarray(task.layout.info["initial_ee_position"])
    state = observe(task, 1.0)
    state = PhysicalState(state.objects, rest.copy(), controls=state.controls)
    command = module.command(planner, task, state)
    assert np.allclose(command.position, rest) and not command.close
    assert planner.b5_phase == "wait"
    observe(task, 2.0)
    command = module.command(planner, task, state)
    assert planner.b5_phase == "approach"
    assert command.position[1] < rest[1]  # heading for the bag on the cabinet


def test_layout_stays_on_the_ward_surface_for_every_level(b5_task):
    """"Hang the bag, open the clamp" left the suite on 2026-11-20 (B5 is now the anaphylaxis
    auto-injector, test_bedside_autoinjector.py), so its layouts are built here without the
    registry; old recordings still resolve to its class."""
    from roboclock.envs.hospital.layout import _surface_bounds
    from roboclock.envs.hospital.tasks.registry import RETIRED_BEDSIDE_TASKS, TASKS

    config = load_config()
    for level in ("easy", "medium", "hard"):
        for seed in range(5):
            layout = b5_task(seed, level).layout
            low, high = _surface_bounds(layout.info.get("work_surface"), config["scene"])
            for name in (ROLLER_NAME, BAG_NAME):
                assert np.all(low[:2] <= layout.objects[name].position[:2])
                assert np.all(layout.objects[name].position[:2] <= high[:2])
    assert RETIRED_BEDSIDE_TASKS["hang_bag_open_clamp"] is RollerClamp
    assert TASKS["B5"] is not RollerClamp
