"""A4 start-infusion: spike + hang, then pump START, in order; these are not PhysX episodes."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import iv_bag
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import _Builder, load_catalog, make_layout
from roboclock.envs.hospital.tasks.bedside.iv_bag import IVBagSpikingHanging
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState

DT = 0.01


@pytest.fixture
def task():
    spec = SimpleNamespace(
        event_time=1.0,
        response_limit=3.0,
        orientation_tolerance=0.3,
        seed=7,
        difficulty="hard",
        num_items=1,
    )
    task = IVBagSpikingHanging(spec)
    b = _Builder(task, load_config(), load_catalog(), np.random.default_rng(7))
    iv_bag.build_layout(b)
    task.bind(
        SimpleNamespace(
            objects=b.objects,
            shapes=b.shapes,
            targets={},
            info=b.info,
            table_height=b.z,
            fixtures=b.fixtures,
        )
    )
    return task


def key_state(task, travel=0.0, contact=False, dx=0.0):
    origin = np.array(task.layout.info["iv_bag"]["start_key"]["origin"])
    return ObjectState(
        origin + [dx, 0.0, -travel], np.array([1.0, 0.0, 0.0, 0.0]), robot_contact=contact
    )


def state_at(
    task,
    dz=0.0,
    *,
    grasped=True,
    contact=False,
    xy=(0.0, 0.0),
    yaw=0.0,
    robot_contact=False,
    velocity=None,
    key=None,
):
    pos = np.array(task.layout.info["iv_bag"]["hang_position"]) + [*xy, dz]
    obj = ObjectState(
        pos,
        np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
        grasped=grasped,
        robot_contact=robot_contact,
        fixture_contacts=frozenset([iv_bag.HOOK_CONTACT]) if contact else frozenset(),
        velocity=np.zeros(3) if velocity is None else np.asarray(velocity),
    )
    objects = {iv_bag.BAG: obj, iv_bag.START_KEY: key if key is not None else key_state(task)}
    return PhysicalState(objects, pos + [0.0, 0.0, 0.121])


def observe(task, state, when, dt=DT):
    task.time = when
    task.process(dt)
    task.observe(state, dt)


def insert(task, start=1.1):
    if task.event_time is None and start >= task.spec.event_time:
        task.time = task.spec.event_time
        task.process(DT)
    observe(task, state_at(task, dz=0.030), start)
    observe(task, state_at(task, dz=0.020), start + 0.02)
    observe(task, state_at(task, dz=0.003), start + 0.08)
    assert task.insertion_time is not None


def hang(task, at, **kwargs):
    hanging = state_at(task, grasped=False, contact=True, **kwargs)
    for i in range(21):
        observe(task, hanging, at + i * DT)


HOLD_SAMPLES = 4  # 40 ms of 10 ms samples: the press completes at the fourth held sample


def press(task, at, steps=5, travel=0.005, contact=True):
    """Hold START down from `at` for `steps` samples; returns when a full hold would end."""
    for i in range(steps):
        when = at + i * DT
        observe(
            task,
            state_at(task, grasped=False, contact=True, key=key_state(task, travel, contact)),
            when,
        )
    return at + (HOLD_SAMPLES - 1) * DT


# --- step 1: spike and hang (unchanged mechanism) --------------------------------------------------


def test_requires_real_insertion_entry_before_hanging(task):
    observe(task, state_at(task), 1.5)
    hang(task, 1.6)
    assert not task.physical_success
    assert "hanging_before_insertion" in task.physical_failures


def test_hanging_alone_is_step_one_not_completion(task):
    insert(task)
    hang(task, 2.0)
    assert task.hanging_time == pytest.approx(2.0)
    assert task.correct >= {"spike", "hang"}
    assert task.device_state["bag_hung"]
    assert not task.goal_achieved and not task.physical_success
    assert task.start_time is None and not task.temporal_success


def test_missing_hook_contact_or_unreleased_grip_never_completes(task):
    insert(task)
    for contact, grasped in ((False, False), (True, True)):
        for i in range(30):
            observe(task, state_at(task, contact=contact, grasped=grasped), 2.0 + i * DT)
        assert task.hanging_time is None
    press(task, 2.5)
    assert task.start_time is None and not task.physical_success


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xy": (0.020, 0.0)},
        {"xy": (0.0, 0.16)},
        {"yaw": np.pi / 2},
        {"robot_contact": True},
        {"velocity": [0.0, 0.0, 0.2]},
        {"dz": 0.020},
    ],
)
def test_incidental_peg_contact_is_not_valid_hanging(task, kwargs):
    inserted = state_at(task, grasped=False, contact=True, **kwargs)
    result = iv_bag.geometry(task.layout, inserted.objects[iv_bag.BAG])
    assert not (result["hanging"] and result["inserted"])


def test_wrong_side_entry_is_latched(task):
    observe(task, state_at(task, dz=0.030, xy=(0.012, 0.0)), 1.1)
    observe(task, state_at(task, dz=0.020, xy=(0.012, 0.0)), 1.2)
    insert(task, 1.3)
    hang(task, 2.0)
    press(task, 2.5)
    assert "misaligned_spike_entry" in task.physical_failures
    assert not task.physical_success


def test_tilted_spike_full_envelope_must_fit_not_just_tip(task):
    state = state_at(task)
    obj = state.objects[iv_bag.BAG]
    angle = 0.4
    obj = replace(obj, orientation=np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]))
    assert not iv_bag.geometry(task.layout, obj)["inserted"]


def test_removing_spike_after_insertion_fails_even_if_reinserted(task):
    insert(task)
    observe(task, state_at(task, dz=0.030), 1.4)
    assert "spike_disconnected" in task.physical_failures


# --- step 2: START, in order -------------------------------------------------------------------


def test_start_after_verified_hang_completes_the_order(task):
    insert(task)
    hang(task, 2.0)
    end = press(task, 2.4)
    assert task.start_time == pytest.approx(end)
    assert task.correct == {"spike", "hang", iv_bag.START_KEY}
    assert task.goal_achieved and task.physical_success and task.temporal_success
    assert task.device_state == {"bag_hung": True, "pump_running": True}
    assert task.action_time == pytest.approx(task.start_time)  # the limit ends at the START press


def test_start_before_the_hang_is_a_wrong_order_failure(task):
    insert(task)
    press(task, 1.5)
    assert "start_before_hang" in task.physical_failures
    hang(task, 2.0)
    press(task, 2.4)
    assert task.start_time is None
    assert not task.physical_success


def test_start_during_the_hang_verification_is_wrong_order(task):
    insert(task)
    hanging = state_at(task, grasped=False, contact=True)
    for i in range(10):  # 0.1 s of the 0.2 s support verification
        observe(task, hanging, 2.0 + i * DT)
    press(task, 2.1)
    assert "start_before_hang" in task.physical_failures
    assert not task.physical_success


def test_start_needs_contact_travel_and_the_full_hold(task):
    insert(task)
    hang(task, 2.0)
    press(task, 2.3, steps=5, travel=0.0035)  # short of the 4 mm actuation travel
    press(task, 2.4, steps=5, travel=0.005, contact=False)  # moved without robot contact
    press(task, 2.5, steps=3)  # 30 ms of the 40 ms hold
    observe(task, state_at(task, grasped=False, contact=True), 2.54)
    assert task.start_time is None and not task.physical_failures
    press(task, 2.6, steps=5)
    assert task.start_time is not None and task.physical_success


def test_start_does_not_count_once_the_bag_left_the_hook(task):
    insert(task)
    hang(task, 2.0)
    key = key_state(task, 0.005, True)
    for i in range(6):
        observe(task, state_at(task, grasped=False, contact=False, key=key), 2.3 + i * DT)
    assert task.start_time is None


@pytest.mark.parametrize("offset,expected", [(-0.001, True), (0.0, True), (0.001, False)])
def test_inclusive_deadline_is_the_end_of_the_start_press(task, offset, expected):
    insert(task)
    hang(task, 2.0)
    deadline = task.event_time + task.spec.response_limit
    press(task, deadline + offset - (HOLD_SAMPLES - 1) * DT)
    assert task.physical_success
    assert task.start_time == pytest.approx(deadline + offset)
    assert task.temporal_success is expected


def test_hang_after_the_deadline_is_late_even_before_start(task):
    insert(task)
    observe(task, state_at(task), task.event_time + task.spec.response_limit + 0.01)
    assert "late_completion" in task.violations


def test_touching_the_resting_bag_before_the_instruction_is_early(task):
    source = task.layout.objects[iv_bag.BAG]
    state = PhysicalState(
        {
            iv_bag.BAG: ObjectState(source.position, source.orientation, robot_contact=True),
            iv_bag.START_KEY: key_state(task),
        },
        np.zeros(3),
    )
    observe(task, state, 0.5)
    assert "early_action" in task.violations


def test_key_press_before_the_instruction_is_early(task):
    source = task.layout.objects[iv_bag.BAG]
    state = PhysicalState(
        {
            iv_bag.BAG: ObjectState(source.position, source.orientation),
            iv_bag.START_KEY: key_state(task, 0.003, True),
        },
        np.zeros(3),
    )
    observe(task, state, 0.5)
    assert "early_action" in task.violations


def test_early_transfer_violation_stays_latched_after_valid_completion(task):
    observe(task, state_at(task, dz=0.030), 0.5)
    insert(task)
    hang(task, 2.0)
    press(task, 2.4)
    assert task.physical_success
    assert not task.temporal_success
    assert "early_action" in task.violations


@pytest.mark.parametrize("late, expected", [(0.0, True), (0.001, False)])
def test_deadline_starts_at_actual_cue_tick_for_non_grid_event(task, late, expected):
    task.spec.event_time = 1.001
    task.time = 1.0
    task.process(1 / 120)
    assert task.event_time is None
    task.time = 1.0 + 1 / 120
    task.process(1 / 120)
    assert task.event_time == pytest.approx(1.0 + 1 / 120)
    insert(task)
    hang(task, 2.0)
    press(task, task.event_time + task.spec.response_limit + late - (HOLD_SAMPLES - 1) * DT)
    assert task.physical_success
    assert task.temporal_success is expected


# --- scene, configuration and controller -------------------------------------------------------


def test_usd_port_tab_damping_and_start_key_joint(task):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Objects/bag_0").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdGeom.Mesh.Define(stage, "/World/Objects/bag_0/Model/Pouch")

    class Look:
        def material(self, name, **kwargs):
            return SimpleNamespace(
                material=UsdShade.Material.Define(stage, f"/World/Materials/{name}")
            )

    backend = SimpleNamespace(
        stage=stage, appearance=Look(), layout=task.layout, config={"visuals": {"labels": False}}
    )
    iv_bag.author(backend)
    walls = [p for p in root.GetChildren() if p.GetName().startswith("TrainerPort")]
    assert len(walls) == 4
    assert all(p.HasAPI(UsdPhysics.CollisionAPI) for p in walls)
    assert all(not p.HasAPI(UsdPhysics.RigidBodyAPI) for p in walls)
    tab = stage.GetPrimAtPath("/World/Objects/bag_0/TrainerGripTab")
    assert tab.HasAPI(UsdPhysics.CollisionAPI) and not tab.HasAPI(UsdPhysics.RigidBodyAPI)
    try:  # the Isaac interpreter's USD carries the PhysX schema; plain usd-core does not
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    if PhysxSchema is not None:
        damping = PhysxSchema.PhysxRigidBodyAPI(root).GetAngularDampingAttr().Get()
        assert damping == pytest.approx(iv_bag.BAG_ANGULAR_DAMPING)
    material, _ = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath("/World/Objects/bag_0/Model/Pouch")
    ).ComputeBoundMaterial()
    assert material.GetPath() == "/World/Materials/iv_bag"
    # The bag has no joint; START is a spring-returned prismatic key with a 6 mm stroke.
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert [str(p.GetPath()) for p in joints] == [f"/World/Joints/{iv_bag.START_KEY}"]
    key = UsdPhysics.PrismaticJoint(joints[0])
    assert [str(t) for t in key.GetBody1Rel().GetTargets()] == [f"/World/Objects/{iv_bag.START_KEY}"]
    assert key.GetLowerLimitAttr().Get() == pytest.approx(-iv_bag.START_STROKE)
    assert key.GetUpperLimitAttr().Get() == pytest.approx(0.0)


def test_config_issues_the_instruction_at_start_with_ordered_limits():
    config = load_config()
    block = config["tasks"]["A4"]
    assert block["instruction_time"] == pytest.approx(0.0)
    limits = [
        make_task("A4", level, 0, config).spec.response_limit for level in ("easy", "medium", "hard")
    ]
    assert limits[0] > limits[1] > limits[2]
    hard = make_task("A4", "hard", 3, config)
    assert hard.spec.event_time == pytest.approx(0.0)
    assert hard.spec.max_time == pytest.approx(max(12.0, hard.spec.response_limit + 4.0))
    assert make_task("hang_and_start_infusion").task_id == "A4"
    assert "START INFUSION" in hard.gen_instruction()


def test_layout_cart_hook_and_pump_follow_the_mount():
    from roboclock.envs.lab.robot import mount_positions

    config = load_config()
    mount = np.array(mount_positions(config, "A4")[0])
    for seed in range(5):
        task = make_task("A4", "hard", seed, config)
        layout = make_layout(task, config)
        info = layout.info["iv_bag"]
        assert layout.table_height == pytest.approx(mount[2] - iv_bag.CART_BELOW_MOUNT)
        assert info["hook_center"][2] == pytest.approx(mount[2] + iv_bag.HOOK_ABOVE_MOUNT)
        assert layout.info["hide_bench"]
        assert set(iv_bag.ULTRASOUND_CART) <= set(layout.info["ward_pose"]["hide"])
        # The cart top is the only support: the cradle and the START key's footprint lie over it.
        surface = layout.info["work_surface"]
        low = np.asarray(surface["center"]) - np.asarray(surface["size"]) / 2
        high = np.asarray(surface["center"]) + np.asarray(surface["size"]) / 2
        assert high[2] == pytest.approx(layout.table_height)
        for fixture in layout.fixtures:
            if fixture.name.startswith("iv_cradle"):
                assert fixture.position[2] - fixture.size[2] / 2 == pytest.approx(high[2])
        key = np.asarray(info["start_key"]["origin"])
        assert np.all(key[:2] > low[:2]) and np.all(key[:2] < high[:2])
        # The pump hangs on the pole (side clamp), below the bag, above the cart.
        pole = np.asarray(info["pole_xy"])
        pump = np.asarray(info["pump"]["origin"])
        assert layout.table_height < pump[2] < info["pump"]["top"] < info["spike_base"][2]
        assert np.linalg.norm(pump[:2] + [iv_bag.PUMP_SIZE[1] / 2, 0.0] - pole) < 0.03
        # The X5 wrist sits radially beyond a pressed key: keep the key's bearing off the bag's.
        bag = np.asarray(info["hang_position"][:2]) - mount[:2]
        bearing = np.degrees(
            np.arctan2(key[1] - mount[1], key[0] - mount[0]) - np.arctan2(bag[1], bag[0])
        )
        assert bearing > 30.0
        # The pole stands just beyond the cart's bed-side edge, not through it.
        assert pole[0] - iv_bag.POLE_RADIUS > high[0]


def test_robot_starts_tucked_near_its_own_mount():
    from roboclock.envs.lab.robot import mount_positions

    config = load_config()
    task = make_task("A4", "hard", 0, config)
    layout = make_layout(task, config)
    mount = np.array(mount_positions(config, "A4")[0])
    initial = np.asarray(layout.info["initial_ee_position"])
    assert np.linalg.norm(initial[:2] - mount[:2]) < 0.25
    assert abs(initial[2] - mount[2]) < 0.05
    grasp = layout.objects[iv_bag.BAG].position + layout.info["iv_bag"]["grasp_offset"]
    assert np.linalg.norm(initial - grasp) > 0.2  # not prepared at the cradle


def test_controller_rests_until_the_instruction_then_reaches(task):
    class Planner:
        last_goal = None
        iv_bag_phase = "done"
        _iv_bag_task = object()

        def step_yaw(self, yaw):
            self.yaw = yaw

        def finish(self, state, goal, close, speed):
            self.last_goal = np.asarray(goal)
            return goal, close, speed

    planner = Planner()
    source = task.layout.objects[iv_bag.BAG]
    initial = np.asarray(task.layout.info["initial_ee_position"])
    state = PhysicalState(
        {
            iv_bag.BAG: ObjectState(source.position, source.orientation),
            iv_bag.START_KEY: key_state(task),
        },
        initial,
    )
    task.time = 0.5
    goal, closed, _ = iv_bag.command(planner, task, state)
    assert planner.iv_bag_phase == "wait" and planner._iv_bag_task is task
    np.testing.assert_allclose(goal, initial)
    assert not closed
    task.time = task.spec.event_time
    task.process(DT)
    iv_bag.command(planner, task, state)
    assert planner.iv_bag_phase == "reach"
