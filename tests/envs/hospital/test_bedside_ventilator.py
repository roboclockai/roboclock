"""B4 ventilator circuit reconnection: geometry, scoring and registration (not PhysX episodes)."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import ventilator
from roboclock.envs.hospital.bedside.ventilator import ELBOW, POST_HEIGHT, SEAT_DEPTH
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.bedside.ventilator import VentilatorReconnect
from roboclock.envs.hospital.tasks.registry import TASKS, make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState

DT = 0.004
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
ON_TUBE = frozenset(["fixture:trach_flange"])


@pytest.fixture
def task():
    task = make_task("B4", "hard", 3)
    task.bind(make_layout(task, load_config()))
    return task


def info(task):
    return task.layout.info["ventilator"]


def elbow(task, where="seated", dz=0.0, dxy=(0.0, 0.0), **kwargs):
    center = np.array(info(task)[f"{where}_center"]) + [*dxy, dz]
    kwargs.setdefault("fixture_contacts", ON_TUBE if where == "seated" and dz < 0.002 else frozenset())
    return ObjectState(center, kwargs.pop("orientation", UPRIGHT).copy(), **kwargs)


def step(task, obj, duration=DT, free=True):
    """Advance the task clock and score one measured state per physics tick."""
    for _ in range(max(1, round(duration / DT))):
        task.advance(DT)
        measured = ventilator.geometry(task.layout, obj)
        measured["circuit_free"] = free and task.event_time is not None
        state = PhysicalState({ELBOW: obj}, np.array([0.4, 0.0, 1.0]), controls={"ventilator": measured})
        task.observe(state, DT)


def disconnect(task):
    """Run to the seeded event, then the scripted pop-off: the elbow ends on the towel."""
    step(task, elbow(task), task.spec.event_time + DT, free=False)
    assert task.event_time == pytest.approx(task.spec.event_time, abs=2 * DT)
    step(task, elbow(task, dz=0.03), 0.2, free=False)
    step(task, elbow(task, "park"), 0.3)


def reseat(task, hold=0.3):
    step(task, elbow(task, "park", grasped=True, robot_contact=True), 0.2)
    step(task, elbow(task, dz=POST_HEIGHT + 0.01, grasped=True, robot_contact=True), 0.2)
    step(task, elbow(task, dz=0.002, grasped=True, robot_contact=True), 0.1)
    step(task, elbow(task, grasped=True, robot_contact=True), 0.1)
    step(task, elbow(task), hold)


# --- registration and episode sampling -----------------------------------------------------------


def test_b4_is_the_ventilator_reconnection():
    assert TASKS["B4"] is VentilatorReconnect
    assert VentilatorReconnect.name == "ventilator_reconnect"
    assert VentilatorReconnect.category == "event_reactive"
    assert "line_switch" not in {cls.name for cls in TASKS.values()}


def test_disconnection_time_is_seeded_and_the_same_at_every_level():
    lo, hi = load_config()["episode"]["event_time_range"]
    times = set()
    for seed in range(10):
        events = {make_task("B4", level, seed).spec.event_time for level in ("easy", "medium", "hard")}
        assert len(events) == 1
        times |= events
    assert len(times) == 10 and all(lo <= t <= hi for t in times)


def test_limits_shrink_with_difficulty_and_never_fall_below_one_second():
    limits = [make_task("B4", level, 0).spec.response_limit for level in ("easy", "medium", "hard")]
    assert limits[0] > limits[1] > limits[2] >= 1.0
    assert limits[1] == pytest.approx(np.sqrt(limits[0] * limits[2]), abs=0.01)


def test_instruction_names_the_limit_and_fits_the_prompt_budget(task):
    text = task.gen_instruction()
    assert f"{task.spec.response_limit:.1f} seconds" in text
    assert len(text) <= 512


def test_old_stopcock_rows_are_retired_not_mistaken_for_the_new_b4():
    from roboclock.envs.hospital.results import normalize
    from roboclock.envs.hospital.stages import STAGES

    old = normalize({"task_id": "B4", "task_name": "line_switch", "task_success": True})
    assert old["retired"] and old["task_id"] == "B4-line_switch" and old["benchmark_id"] is None
    new = normalize({"task_id": "B4", "task_name": "ventilator_reconnect", "task_success": True})
    assert new["benchmark_id"] == "B4" and not new.get("retired")
    assert [s.name for s in STAGES["B4"]({"task_name": "line_switch"})] == [
        "maintenance_clamp_closed",
        "stopcock_at_target_detent",
    ]
    assert [s.name for s in STAGES["B4"]({"task_name": "ventilator_reconnect"})] == [
        "elbow_grasped",
        "connector_seated",
    ]


# --- layout ----------------------------------------------------------------------------------


def test_layout_seats_the_elbow_on_the_connector(task):
    layout = task.layout
    start = ObjectState(
        layout.objects[ELBOW].position.copy(), UPRIGHT.copy(), fixture_contacts=ON_TUBE
    )
    measured = ventilator.geometry(layout, start)
    assert measured["seated"] and measured["retained"]
    assert measured["depth"] == pytest.approx(POST_HEIGHT, abs=0.001)


def test_towel_is_beside_the_connector_and_higher(task):
    park, seated = np.array(info(task)["park_center"]), np.array(info(task)["seated_center"])
    assert 0.12 < np.linalg.norm(park[:2] - seated[:2]) < 0.18
    assert park[2] > seated[2] + 0.05
    assert not ventilator.geometry(task.layout, elbow(task, "park"))["seated"]


def test_everything_the_policy_needs_is_in_the_head_image(task):
    from roboclock.envs.hospital.rig import head_pixels

    points = [
        info(task)["stoma"],
        info(task)["park_center"],
        info(task)["display"],
        task.layout.indicators[0].position,
    ]
    pixels, depth = head_pixels(load_config(), "B4", points, conservative=True)
    assert np.all(depth > 0)
    assert np.all((pixels[:, 0] > 10) & (pixels[:, 0] < 630))
    assert np.all((pixels[:, 1] > 10) & (pixels[:, 1] < 470))


def test_start_and_work_poses_are_within_the_arm_reach(task):
    for key in ("seated_center", "park_center"):
        point = np.array(info(task)[key])
        assert np.linalg.norm(point[:2]) < 0.50  # arm 0's mount is at the origin
    assert np.linalg.norm(np.array(task.layout.info["initial_ee_position"])[:2]) < 0.25


# --- measurement -------------------------------------------------------------------------------


def test_depth_and_alignment_are_measured_in_the_elbow_frame(task):
    assert ventilator.geometry(task.layout, elbow(task, dz=0.010))["depth"] == pytest.approx(
        POST_HEIGHT - 0.010, abs=1e-6
    )
    shallow = ventilator.geometry(task.layout, elbow(task, dz=POST_HEIGHT - SEAT_DEPTH + 0.002))
    assert shallow["aligned"] and not shallow["seated"]
    aside = ventilator.geometry(task.layout, elbow(task, dxy=(0.006, 0.0)))
    assert not aside["aligned"] and not aside["seated"]
    tilt = 0.3
    tipped = elbow(task, orientation=np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0.0, 0.0]))
    assert not ventilator.geometry(task.layout, tipped)["aligned"]


def test_retained_needs_release_rest_and_tube_contact(task):
    def check(**kwargs):
        return ventilator.geometry(task.layout, elbow(task, **kwargs))["retained"]

    assert check()
    assert not check(grasped=True)
    assert not check(robot_contact=True)
    assert not check(velocity=np.array([0.0, 0.0, 0.1]))
    assert not check(fixture_contacts=frozenset())


def test_pop_off_path_clears_the_connector_before_moving_sideways(task):
    start, park = np.array(info(task)["seated_center"]), np.array(info(task)["park_center"])
    previous = start
    for s in np.linspace(0.0, 1.0, 101):
        point = ventilator.eject_reference(start, park, s)
        sideways = np.linalg.norm(point[:2] - start[:2])
        if sideways > 0.001:
            assert point[2] - start[2] > POST_HEIGHT + 0.005
        if sideways > 0.06:  # over the towel
            assert point[2] >= park[2] - 1e-9
        assert np.linalg.norm(point - previous) < 0.012
        previous = point
    assert np.allclose(ventilator.eject_reference(start, park, 1.0), park)


def test_breathing_waveform_is_periodic_and_bounded():
    values = [ventilator.breath(t) for t in np.arange(0.0, 8.0, 0.02)]
    assert 0.0 <= min(values) and max(values) == pytest.approx(1.0, abs=0.01)
    assert ventilator.breath(0.4) == pytest.approx(ventilator.breath(0.4 + ventilator.BREATH_PERIOD))


# --- scoring -----------------------------------------------------------------------------------


def test_reconnection_in_time_is_a_joint_success(task):
    disconnect(task)
    assert task.device_state["alarm_active"] and not task.goal_achieved
    reseat(task)
    m = task.metrics()
    assert task.physical_success and task.temporal_success and task.joint_success
    assert m["grasp_time"] < m["socket_entry_time"] <= m["seat_time"] < m["reconnect_time"] + 0.3
    assert 0 < m["response_latency"] < task.spec.response_limit
    assert task.device_state == {"circuit_connected": True, "alarm_active": False, "ventilating": True}


def test_reconnection_is_timed_at_the_first_released_instant_not_after_the_dwell(task):
    disconnect(task)
    step(task, elbow(task, "park", grasped=True, robot_contact=True), 0.2)
    step(task, elbow(task, dz=POST_HEIGHT + 0.01, grasped=True, robot_contact=True), 0.1)
    step(task, elbow(task, grasped=True, robot_contact=True), 0.1)
    released_at = task.time + DT
    step(task, elbow(task), 0.1)
    assert task.reconnect_time is None  # the 0.2 s dwell is not over
    step(task, elbow(task), 0.2)
    assert task.reconnect_time == pytest.approx(released_at, abs=1e-9)


def test_holding_the_elbow_on_the_connector_is_not_a_reconnection(task):
    disconnect(task)
    step(task, elbow(task, dz=POST_HEIGHT + 0.01, grasped=True, robot_contact=True), 0.1)
    step(task, elbow(task, grasped=True, robot_contact=True), 1.0)
    assert task.seat_time is not None and task.reconnect_time is None
    assert not task.physical_success


def test_late_reconnection_is_physical_but_not_temporal(task):
    disconnect(task)
    step(task, elbow(task, "park"), task.spec.response_limit)
    reseat(task)
    assert task.physical_success and not task.temporal_success
    assert "late_response" in task.violations
    assert task.metrics()["response_latency"] > task.spec.response_limit


def test_never_reconnected_is_late_and_not_physical(task):
    disconnect(task)
    step(task, elbow(task, "park"), task.spec.response_limit + 0.5)
    assert "late_response" in task.violations
    assert not task.physical_success and not task.temporal_success


def test_touching_the_circuit_before_it_comes_off_is_an_early_action(task):
    step(task, elbow(task), 0.5, free=False)
    step(task, elbow(task, robot_contact=True), 0.1, free=False)
    assert "early_action" in task.violations
    disconnect(task)
    reseat(task)
    assert task.physical_success and not task.temporal_success


def test_pulling_the_circuit_off_early_is_an_early_action(task):
    step(task, elbow(task, dz=0.02, grasped=True, robot_contact=True), 0.1, free=False)
    assert "early_action" in task.violations


def test_the_pop_off_itself_scores_nothing(task):
    step(task, elbow(task), task.spec.event_time + DT, free=False)
    step(task, elbow(task, dz=0.01), 0.1, free=False)  # still inside the socket, mid-event
    step(task, elbow(task, "park"), 0.3)
    assert task.entry_time is None and task.seat_time is None and not task.violations


def test_circuit_coming_off_again_after_reconnection_fails_physically(task):
    disconnect(task)
    reseat(task)
    assert task.physical_success
    step(task, elbow(task, dz=POST_HEIGHT + 0.02, grasped=True, robot_contact=True), 0.1)
    assert "circuit_came_off_again" in task.physical_failures and not task.physical_success


def test_episode_ends_a_second_after_the_reconnection(task):
    disconnect(task)
    reseat(task, hold=0.3)
    assert not task.terminated
    step(task, elbow(task), task.hold_after)
    assert task.terminated


def test_dropped_elbow_ends_the_episode_as_a_failure(task):
    disconnect(task)
    fallen = ObjectState(np.array([0.2, -0.3, 0.05]), UPRIGHT.copy())
    step(task, fallen, 0.1)
    assert task.terminated and not task.physical_success


def test_author_adds_the_socket_walls_to_the_elbow_body(task):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, f"/World/Objects/{ELBOW}")
    backend = SimpleNamespace(stage=stage, appearance=None, layout=task.layout)
    ventilator.author(backend)
    walls = [
        p for p in stage.GetPrimAtPath(f"/World/Objects/{ELBOW}").GetChildren()
        if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert len(walls) == 4
    del pxr
