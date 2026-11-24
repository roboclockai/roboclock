"""Pump-panel key press (`pump_panel_response`): physical key semantics and trigger boundaries.

The pump panel was B6 until 2026-11-20, when the haemorrhage tourniquet took the ID
(test_bedside_tourniquet.py). Its code stays for old recordings and for B1, which stands in
its bay; these tests build it from its last config block and rig origin, without the registry.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside.pump_panel import author, measure
from roboclock.envs.hospital.config import load_config as load_suite_config
from roboclock.envs.hospital.config import sample_episode
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.bedside.pump_panel import MODES, PumpPanel
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState


def load_config():
    """The suite config with the retired pump panel's own B6 block and rig origin."""
    config = load_suite_config()
    config["tasks"]["B6"] = {
        "bedside_profile": "bedside",
        "instruction_time": 0.0,
        "film_iso": 112,
        "num_items_range": [1, 1],
        "easy": {"response_limit": [3.5, 3.5]},
        "medium": {"response_limit": [2.86, 2.86]},
        "hard": {"response_limit": [2.33, 2.33]},
    }
    config["robot"]["rig"]["tasks"]["B6"] = {"origin": [0.15, 0.30, 1.05]}
    return config


def make_task(task_id, difficulty="easy", seed=0, config=None):
    assert task_id == "B6"
    return PumpPanel(sample_episode("B6", difficulty, seed, config or load_config()))


def task_for(mode):
    t = make_task("B6", seed=MODES.index(mode))
    t.spec = replace(t.spec, event_time=2.0, response_limit=1.0)
    t.bind(make_layout(t, load_config()))
    return t


def tick(t, time, key=None, depth=0.005, contact=True, dt=0.01):
    objects = {}
    for name, obj in t.layout.objects.items():
        delta = np.array([0.0, 0.0, -depth if name == f"pump_{key}" else 0.0])
        objects[name] = ObjectState(
            obj.position + delta, obj.orientation, robot_contact=contact and name == f"pump_{key}"
        )
    state = PhysicalState(objects, np.array([0.48, 0.0, 1.05]))
    state.controls.update(measure(SimpleNamespace(layout=t.layout), state))
    t.time = time
    t.process(dt)
    t.observe(state, dt)


@pytest.mark.parametrize("mode", MODES)
def test_requested_key_triggers_device_state(mode):
    t = task_for(mode)
    tick(t, 2.0)
    for i in range(40):
        tick(t, 2.01 + i * 0.01, mode)
    assert t.joint_success
    if mode == "mute":
        assert t.device_state["alarm_silenced"] and t.device_state["alarm_active"]
        assert not t.device_state["pump_running"]
    elif mode == "stop":
        assert not t.device_state["pump_running"]
        assert t.device_state["alarm_active"]
    else:
        assert t.device_state["bolus_active"]


@pytest.mark.parametrize("at,valid", [(3.0, True), (3.01, False)])
def test_inclusive_deadline(at, valid):
    t = task_for("stop")
    tick(t, 2.0)
    for i in range(4):
        tick(t, at - 0.03 + i * 0.01, "stop")
    assert t.physical_success
    assert t.temporal_success is valid


def test_bolus_requires_continuous_hold():
    t = task_for("bolus")
    tick(t, 2.0)
    for i in range(20):
        tick(t, 2.01 + i * 0.01, "bolus")
    tick(t, 2.21)
    for i in range(20):
        tick(t, 2.22 + i * 0.01, "bolus")
    assert not t.physical_success


@pytest.mark.parametrize("depth,contact", [(0.001, True), (0.005, False), (0.02, True)])
def test_partial_travel_no_contact_or_outside_joint_cannot_trigger(depth, contact):
    t = task_for("stop")
    tick(t, 2.0)
    for i in range(10):
        tick(t, 2.01 + 0.01 * i, "stop", depth, contact)
    assert not t.physical_success


def test_wrong_key_and_early_press_remain_failures():
    t = task_for("mute")
    tick(t, 1.0, "mute")
    tick(t, 2.0, "stop")
    for i in range(10):
        tick(t, 2.01 + 0.01 * i, "mute")
    assert "early_action" in t.violations
    assert "wrong_button" in t.physical_failures
    assert not t.joint_success


def test_pump_has_three_passive_spring_keys_and_reuses_local_mesh():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from roboclock.envs.hospital.appearance import HospitalAppearance

    class Appearance(HospitalAppearance):
        def material(self, name, **kwargs):
            return SimpleNamespace(material=UsdShade.Material.Define(self.stage, "/Looks/" + name))

        def label(self, path, *args, **kwargs):
            return UsdGeom.Xform.Define(self.stage, path)

    t = task_for("stop")
    stage = Usd.Stage.CreateInMemory()
    for key in MODES:
        UsdGeom.Xform.Define(stage, "/World/Objects/pump_" + key)
    author(
        SimpleNamespace(
            stage=stage, layout=t.layout, appearance=Appearance(stage, None, load_config())
        )
    )
    for key in MODES:
        p = stage.GetPrimAtPath("/World/Joints/pump_" + key)
        j = UsdPhysics.PrismaticJoint(p)
        assert j.GetAxisAttr().Get() == "Z"
        assert j.GetLowerLimitAttr().Get() == pytest.approx(-0.006)
        drive = UsdPhysics.DriveAPI(p, "linear")
        assert drive.GetTargetPositionAttr().Get() == 0.0
        assert drive.GetStiffnessAttr().Get() > 0.0
    assert stage.GetPrimAtPath("/World/Bedside/PumpHousing").HasAuthoredReferences()


def _ward_xy(layout, point):
    """A robot-frame point in the ward file's frame (the ward pose is yaw 90 plus a translation)."""
    pose = layout.info["ward_pose"]
    assert pose["yaw"] == pytest.approx(90.0)
    dx, dy = np.asarray(point[:2]) - np.asarray(pose["translate"][:2])
    return np.array([dy, -dx])


@pytest.mark.parametrize("seed", range(6))
def test_pump_stands_at_the_bedside_pendant_in_the_ward(seed):
    from roboclock.envs.hospital.bedside import pump_panel

    t = make_task("B6", "hard", seed)
    t.bind(make_layout(t, load_config()))
    layout, info = t.layout, t.layout.info
    assert info["hide_bench"] and set(pump_panel.TROLLEY) <= set(info["ward_pose"]["hide"])
    # The seed shift moves the ward with the pump: the pump keeps its place at the pendant.
    pump = _ward_xy(layout, info["pump_panel"]["pump_origin"])
    nominal = (pump_panel.ROBOT_X - 0.015, pump_panel.ROBOT_Y - pump_panel.KEY_X)
    assert pump == pytest.approx(nominal, abs=1e-9)
    # In front of the pendant's glass shelves (ward y > -1.625), clear of the bed (ward x < 0.80).
    assert _ward_xy(layout, info["pump_panel"]["pole_xy"])[1] > -1.625 + 0.019
    assert pump[0] + 0.12 < 0.80
    # The support collider is the pump housing: keys sit inside its footprint, above its top.
    surface = info["work_surface"]
    low, high = layout.workspace_bounds
    top = surface["center"][2] + surface["size"][2] / 2
    for key in MODES:
        obj = layout.objects[f"pump_{key}"]
        assert np.all(low[:2] < obj.position[:2]) and np.all(obj.position[:2] < high[:2])
        assert obj.position[2] - 0.006 - 0.006 > top  # full stroke never reaches the housing
    # The response dial's housing stands on the pendant's middle glass shelf.
    dial = layout.indicators[0]
    assert _ward_xy(layout, dial.position) == pytest.approx(pump_panel.DIAL_WARD, abs=1e-9)
    assert dial.position[2] - 0.045 == pytest.approx(pump_panel.SHELF_TOP + pump_panel.WARD_Z)
    assert info["pump_panel"]["camera"]["focal_mm"] > 0


def test_instruction_is_on_screen_from_the_first_frame_and_the_clock_starts_with_it():
    from roboclock.envs.hospital.bedside import pump_panel
    from roboclock.envs.hospital.mdp.events import BLUE, RED
    from roboclock.envs.hospital.tasks.bedside.pump_panel import INSTRUCTION_TEXT

    config = load_config()
    # Shown from frame 0, so it is issued at 0: a press made on reading it cannot be early.
    assert config["tasks"]["B6"]["instruction_time"] == pytest.approx(0.0)
    for seed, mode in enumerate(MODES):
        t = make_task("B6", "hard", seed, config)
        t.bind(make_layout(t, config))
        assert t.spec.event_time == pytest.approx(0.0)
        # The order card at the dial and the pump display carry the instruction from frame 0.
        assert t.layout.indicators[0].label == INSTRUCTION_TEXT[mode]
        assert INSTRUCTION_TEXT[mode] in t.gen_instruction()
        words = pump_panel.SCREEN_TEXT["cue"][mode][0].replace("\n", " ")
        assert mode.upper() in words and ("PRESS" in words or "HOLD" in words)
    assert set(pump_panel.SCREEN_TEXT) == {"cue", "done"}  # no idle screen hides the instruction
    t = task_for("stop")
    commands = t.advance(0.01)  # before the instruction: a blue dial, no countdown yet
    assert len(commands) == 1 and commands[0].value == BLUE
    t.time = 2.0
    commands = t.process(0.01)
    assert commands[0].value == RED
    assert commands[1].kind == "timer" and commands[1].value[0] == pytest.approx(1.0)
    t.time = 2.5
    assert t.process(0.01)[1].value[0] == pytest.approx(0.5)


@pytest.mark.parametrize("level,limit", [("easy", 3.5), ("medium", 2.86), ("hard", 2.33)])
def test_limits_cover_the_reach_from_rest(level, limit):
    t = make_task("B6", level, 0, load_config())
    assert t.spec.response_limit == pytest.approx(limit)
    assert t.spec.max_time == pytest.approx(max(12.0, limit + 4.0))


@pytest.mark.parametrize("seed", range(6))
def test_robot_starts_at_rest_folded_at_its_column_and_every_key_is_in_reach(seed):
    from roboclock.envs.hospital.bedside import pump_panel
    from roboclock.envs.lab.robot import mount_positions

    config = load_config()
    t = make_task("B6", "hard", seed, config)
    t.bind(make_layout(t, config))
    mount = np.asarray(mount_positions(config, "B6")[0])
    rest = np.asarray(t.layout.info["initial_ee_position"])
    assert rest == pytest.approx(mount + pump_panel.TUCK)
    assert np.linalg.norm(rest[:2] - mount[:2]) < 0.2  # near its own column
    for key, center in t.layout.info["pump_panel"]["centers"].items():
        center = np.asarray(center)
        assert np.linalg.norm(center - rest) > 0.25  # the reach is a visible motion
        # The top-down X5 presses comfortably within 0.45 m of its mount (IK fails beyond ~0.5 m).
        assert np.linalg.norm(center[:2] - mount[:2]) < 0.45


def test_oracle_holds_rest_until_the_instruction_then_hovers_and_presses():
    from roboclock.envs.hospital.bedside import pump_panel
    from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory

    t = task_for("mute")
    rest = np.asarray(t.layout.info["initial_ee_position"])
    planner = HospitalTrajectory(load_config()["control"], 1 / 30)
    planner.last_goal = rest.copy()
    state = SimpleNamespace(ee_position=rest.copy())
    t.time = 1.0
    command = pump_panel.command(planner, t, state)
    assert command.position == pytest.approx(rest) and command.close
    t.time = t.event_time = 2.0
    center = np.asarray(t.layout.info["pump_panel"]["centers"]["mute"])
    hover = center + [0, 0, pump_panel.PRESS_OFFSET + pump_panel.HOVER]
    for i in range(120):
        t.time = 2.0 + i / 30
        command = pump_panel.command(planner, t, state)
    assert command.position == pytest.approx(hover)  # waits above the key for the hand
    state.ee_position = hover.copy()
    command = pump_panel.command(planner, t, state)
    assert command.position[2] < hover[2] and command.position[:2] == pytest.approx(hover[:2])


def test_public_keypad_helpers_build_measure_and_author_a_keypad():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    from roboclock.envs.hospital.bedside import pump_panel

    t = task_for("stop")
    centers = t.layout.info["pump_panel"]["centers"]
    assert list(centers) == list(MODES)
    assert centers["mute"][0] - centers["stop"][0] == pytest.approx(pump_panel.KEY_PITCH)
    stage = Usd.Stage.CreateInMemory()
    for key in MODES:
        UsdGeom.Xform.Define(stage, "/World/Objects/pump_" + key)
    pump_panel.author_keys(stage, centers, labels=False)
    for key in MODES:
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/pump_" + key))
        assert joint.GetLowerLimitAttr().Get() == pytest.approx(-0.006)
    objects = {
        name: ObjectState(obj.position, obj.orientation) for name, obj in t.layout.objects.items()
    }
    buttons = pump_panel.measure_keys(PhysicalState(objects, np.zeros(3)), centers)
    assert set(buttons) == set(MODES)
    assert all(b["depression_m"] == pytest.approx(0.0) and b["aligned"] for b in buttons.values())


def test_partial_press_after_the_request_latches_key_moved_time():
    """The stage score's partial travel: the requested key pressed in under contact."""
    t = task_for("stop")
    tick(t, 1.0, "stop", 0.003)  # before the request: an early action, not progress
    tick(t, 2.0)
    assert t.metrics()["key_moved_time"] is None
    tick(t, 2.1, "stop", 0.001)  # under 1.5 mm
    tick(t, 2.2, "stop", 0.003, contact=False)
    assert t.metrics()["key_moved_time"] is None
    tick(t, 2.3, "stop", 0.003)
    tick(t, 2.4)
    assert t.metrics()["key_moved_time"] == pytest.approx(2.3)
    assert not t.physical_success
