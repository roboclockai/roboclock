"""A3 bag-valve ventilation: sampling, breath scoring, bag measurement, layout, visibility and
oracle logic (no simulator). Also the retirement of the syringe pump cutoff that held A3 before."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.hospital.bedside import window_bvm as wb
from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.registry import RETIRED_TASKS, TASKS, make_task
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState

DT = 1 / 250
LEVELS = ("easy", "medium", "hard")


def episode(level="hard", seed=0):
    config = load_config()
    task = make_task("A3", level, seed, config)
    task.bind(make_layout(task, config))
    backend = SimpleNamespace(layout=task.layout, task=task, config=config)
    return task, backend


def reading(backend, depth=0.0, contact=False, offset=(0.0, 0.0)):
    """Measure the bag's pad pushed `depth` meters down (robot contact as given)."""
    origin = np.asarray(backend.layout.info["window_bvm"]["pad_center"])
    pad = ObjectState(
        origin + [offset[0], offset[1], -depth], np.array([1.0, 0.0, 0.0, 0.0]), robot_contact=contact
    )
    state = PhysicalState({"bvm_bag": pad}, origin + [0.0, 0.0, 0.05])
    state.controls.update(wb.measure(backend, state))
    return state


def run(level="hard", seed=0, squeezes=(), depth=0.032, hold=0.3):
    """Advance the task; the bag is held `depth` down for `hold` s from each time in `squeezes`."""
    task, backend = episode(level, seed)
    while not task.terminated:
        task.advance(DT)
        down = any(start - 1e-9 <= task.time <= start + hold for start in squeezes)
        task.observe(reading(backend, depth if down else 0.0, contact=down), DT)
    return task


# --- episode sampling ---------------------------------------------------------------------------


def test_a3_is_the_ventilation_task_and_the_syringe_task_is_retired():
    assert TASKS["A3"].name == "bvm_ventilation" and TASKS["A3"].category == "temporal_window"
    assert TASKS["A3"].bedside_module == "window_bvm"
    assert "syringe_pump_cutoff" in RETIRED_TASKS
    assert all(cls.name != "syringe_pump_cutoff" for cls in TASKS.values())


@pytest.mark.parametrize("seed", range(10))
def test_four_breaths_six_seconds_apart_on_beats_shared_by_every_level(seed):
    config = load_config()
    tasks = [make_task("A3", level, seed, config) for level in LEVELS]
    assert len({task.beats for task in tasks}) <= 1 or np.allclose(
        [task.beats for task in tasks], tasks[0].beats
    )
    widths = [task.windows[0][1] - task.windows[0][0] for task in tasks]
    assert widths == pytest.approx([3.0, 2.0, 1.0])
    for task in tasks:
        assert len(task.windows) == 4 and task.period == pytest.approx(6.0)
        assert np.allclose(np.diff(task.beats), 6.0)
        assert 7.0 <= task.beats[0] <= 9.0
        assert all(end - start == pytest.approx(widths[tasks.index(task)]) for start, end in task.windows)
        assert task.spec.max_time == pytest.approx(task.windows[-1][1] + 2.0)
        # 10 breaths per minute; consecutive windows never touch.
        assert all(b[0] > a[1] for a, b in zip(task.windows, task.windows[1:]))


def test_rhythm_config_is_validated():
    import copy

    from roboclock.envs.hospital import config as cfg

    good = load_config()
    path_config = copy.deepcopy(good)
    path_config["tasks"]["A3"]["hard"]["window_width"] = [6.0, 6.0]
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(path_config, stream)
    with pytest.raises(ValueError, match="shorter than beat_period"):
        cfg.load_config(stream.name)


def test_instruction_states_the_rate_the_count_and_the_band_width():
    for level, width in zip(LEVELS, ("3", "2", "1")):
        text = make_task("A3", level, 0, load_config()).gen_instruction()
        assert "10 breaths per minute" in text and "every 6 seconds" in text
        assert "Deliver 4 breaths" in text and f"{width} seconds wide" in text
        assert len(text) <= 512


# --- the cycle the monitor shows ------------------------------------------------------------------


def test_cycle_starts_half_a_period_before_the_first_beat_and_has_the_beat_in_its_middle():
    task, _ = episode()
    assert task.cycle(task.beats[0] - 3.5) == (-1, 0.0)
    for k, beat in enumerate(task.beats):
        index, seconds = task.cycle(beat)
        assert index == k and seconds == pytest.approx(3.0)
    lo, hi = task.band()
    assert (lo, hi) == pytest.approx((2.5, 3.5))  # hard: 1 s around the 3 s mark
    for k, (start, end) in enumerate(task.windows):
        assert task.cycle(start)[1] == pytest.approx(lo) and task.cycle(end)[1] == pytest.approx(hi)


def test_display_shows_squeeze_only_while_an_unserved_window_is_open():
    task, _ = episode()
    task.time = task.beats[0] - 2.0
    assert wb.display_state(task)[1:] == ("wait", 0)
    task.time = task.beats[0]
    assert wb.display_state(task)[1:] == ("due", 0)
    task._breath(task.time)
    assert wb.display_state(task)[1:] == ("wait", 1)
    assert wb.display_state(task, breaths=9)[2] == 4  # the counter stops at the number asked


# --- breath scoring -------------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_a_squeeze_on_every_beat_succeeds(level):
    probe, _ = episode(level)
    task = run(level, squeezes=[beat - 0.01 for beat in probe.beats])
    assert task.physical_success and task.temporal_success and task.joint_success
    m = task.metrics()
    assert m["rhythm_outcome"] == "in_rhythm" and m["breaths_in_window"] == 4
    assert np.allclose(m["breath_intervals"], 6.0, atol=2 * DT)
    assert max(abs(e) for e in m["beat_errors"]) < 0.02
    assert task.time == pytest.approx(task.breath_times[-1] + task.hold_after, abs=2 * DT)


def test_breaths_anywhere_inside_the_windows_count_and_edges_are_inclusive():
    probe, _ = episode("easy")
    task = run("easy", squeezes=[start + 0.02 for start, _ in probe.windows])
    assert task.joint_success
    task = run("easy", squeezes=[end - 0.05 for _, end in probe.windows])
    assert task.joint_success


def test_an_early_squeeze_is_hyperventilation_but_still_a_physical_breath():
    probe, _ = episode()
    times = [beat for beat in probe.beats]
    times[1] = probe.windows[1][0] - 0.6  # before its window opens
    task = run(squeezes=times)
    assert task.physical_success and not task.temporal_success
    assert "early_breath" in task.violations and "missed_window" in task.violations
    assert task.metrics()["rhythm_outcome"] == "hyperventilation"
    assert task.temporal_error == pytest.approx(0.6, abs=0.05)


def test_an_extra_squeeze_between_or_inside_served_windows_fails_the_timing():
    probe, _ = episode("easy")
    between = run("easy", squeezes=[*probe.beats, probe.beats[0] + 3.0])
    assert between.physical_success and not between.temporal_success
    assert between.violations == ["early_breath"] and len(between.breath_times) == 5
    inside = run("easy", squeezes=[*probe.beats, probe.beats[0] + 0.9], hold=0.2)
    assert not inside.temporal_success and inside.violations == ["extra_breath"]


def test_a_late_or_missed_breath_is_hypoventilation():
    probe, _ = episode()
    late = run(squeezes=[probe.beats[0], probe.windows[1][1] + 0.4, probe.beats[2], probe.beats[3]])
    assert late.physical_success and not late.temporal_success
    assert "missed_window" in late.violations
    missed = run(squeezes=probe.beats[:3])
    assert not missed.physical_success and not missed.temporal_success
    assert missed.metrics()["rhythm_outcome"] == "hypoventilation"
    assert missed.metrics()["breaths_delivered"] == 3


def test_four_fast_squeezes_are_physically_complete_but_out_of_rhythm():
    task = run(squeezes=[1.0, 2.0, 3.0, 4.0])
    assert task.physical_success and not task.temporal_success
    assert "early_breath" in task.violations and len(task.served) == 0


def test_no_squeeze_misses_every_window_and_runs_to_the_end():
    task = run()
    assert not task.physical_success and not task.temporal_success
    assert task.metrics()["rhythm_outcome"] == "no_breath"
    assert task.violations == ["missed_window"]
    assert task.time == pytest.approx(task.windows[-1][1] + task.hold_after, abs=2 * DT)


def test_stage_fields_follow_the_breaths():
    from roboclock.envs.hospital.stages import STAGES

    probe, _ = episode()
    row = run(squeezes=probe.beats[:2]).metrics()
    stages = STAGES["A3"](row)
    assert [s.name for s in stages] == [
        "bag_compressed", "first_breath_delivered", "half_of_breaths_delivered"
    ]
    assert all(s.reached(row) for s in stages)
    row = run(squeezes=probe.beats[:1]).metrics()
    assert [s.reached(row) for s in STAGES["A3"](row)] == [True, True, False]
    old = {"task_name": "syringe_pump_cutoff", "device_moved_time": 1.0}
    assert [s.name for s in STAGES["A3"](old)] == ["stop_key_moved"]


def test_old_syringe_rows_are_kept_as_retired_rows():
    from roboclock.envs.hospital.results import normalize

    old = normalize({"task_id": "A3", "task_name": "syringe_pump_cutoff", "task_success": True})
    assert old["retired"] and old["task_id"] == "A3-syringe_pump_cutoff"
    assert old["recorded_task_id"] == "A3" and old["benchmark_id"] is None
    new = normalize({"task_id": "A3", "task_name": "bvm_ventilation", "task_success": True})
    assert new["benchmark_id"] == "A3" and not new.get("retired")


# --- bag measurement ------------------------------------------------------------------------------


def test_a_breath_needs_contact_alignment_and_the_delivery_travel():
    for depth, contact, aligned in (
        (wb.DELIVER - 0.002, True, True),
        (wb.DELIVER + 0.004, False, True),
        (wb.DELIVER + 0.004, True, False),
    ):
        monitor = wb.BagMonitor()
        for k in range(1, 60):
            monitor.update(k * DT, DT, depth, contact, aligned)
        assert not monitor.breath_times
    monitor = wb.BagMonitor()
    monitor.update(DT, DT, wb.DELIVER, True, True)
    assert monitor.breath_times == [pytest.approx(DT)] and monitor.moved


def test_holding_the_bag_down_is_one_breath_and_it_must_refill_before_the_next():
    monitor = wb.BagMonitor()
    t = 0.0
    for _ in range(200):
        t += DT
        monitor.update(t, DT, 0.035, True, True)
    assert len(monitor.breath_times) == 1 and not monitor.armed
    for depth in (0.020, 0.035):  # eased to 20 mm (still above RELEASE) and pushed again
        for _ in range(50):
            t += DT
            monitor.update(t, DT, depth, True, True)
    assert len(monitor.breath_times) == 1
    for depth in (0.005, 0.035):  # released past RELEASE, then squeezed
        for _ in range(50):
            t += DT
            monitor.update(t, DT, depth, True, True)
    assert len(monitor.breath_times) == 2


def test_moved_needs_contact_and_the_chest_follows_the_compression():
    monitor = wb.BagMonitor()
    monitor.update(DT, DT, 0.002, True, True)
    monitor.update(2 * DT, DT, 0.02, False, True)
    assert not monitor.moved
    t = 2 * DT
    for _ in range(250):
        t += DT
        monitor.update(t, DT, wb.DELIVER, True, True)
    assert monitor.moved and monitor.lung > 0.95
    for _ in range(500):
        t += DT
        monitor.update(t, DT, 0.0, False, True)
    assert monitor.lung < 0.05


def test_repeated_reading_at_the_same_time_does_not_advance_the_monitor():
    monitor = wb.BagMonitor()
    for _ in range(10):
        monitor.update(DT, DT, 0.01, True, True)
    assert monitor.lung == pytest.approx(min(0.01 / wb.DELIVER, 1.0) * DT / wb.LUNG_FILL)


def test_measure_reports_controls_and_keeps_one_monitor_per_task():
    task, backend = episode()
    task.time = 1.0
    state = reading(backend, 0.03, True)
    assert state.controls["device_moved"] and len(state.controls["breath_times"]) == 1
    assert state.controls["bag"]["compression_m"] == pytest.approx(0.03)
    other, _ = episode(seed=1)
    backend.task = other
    other.time = 0.5
    assert reading(backend).controls["breath_times"] == ()  # a new episode starts with a full bag


# --- deformation the cameras see -------------------------------------------------------------------


def test_squeezed_bag_follows_the_pad_and_stays_on_the_mat():
    top, span = 0.157, (-0.13, 0.13)
    points = np.array([[0.0, 0.0, top], [0.0, 0.0, 0.0], [0.0, 0.05, top / 2], [0.13, 0.0, top / 2], [0.3, 0.0, 0.1]])
    weights = wb.bag_weights(points, span, top)
    out = wb.squeezed(points, weights, 0.03, top)
    assert out[0, 2] == pytest.approx(top - 0.03)  # the top wall under the pad
    assert out[1, 2] == pytest.approx(0.0)  # the bottom stays put
    assert out[2, 1] > points[2, 1]  # the sides bulge
    assert np.allclose(out[3:], points[3:])  # the necks and the valve do not move


def test_chest_rises_only_on_the_front_of_the_chest():
    top = 0.216
    points = np.array([[0.0, -0.03, top], [0.0, -0.03, 0.02], [0.0, -0.36, top], [0.0, 0.40, top]])
    bump = wb.chest_weights(points, top)
    out = wb.inflated(points, bump, wb.CHEST["rise"])
    assert out[0, 2] - points[0, 2] == pytest.approx(wb.CHEST["rise"], rel=0.01)
    assert out[1, 2] == pytest.approx(points[1, 2])  # the back lies on the mattress
    assert out[2, 2] - points[2, 2] < 0.002 and out[3, 2] - points[3, 2] < 0.002  # head, lower trunk


def test_timer_scale_image_is_built_with_the_track_on_its_card(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setenv("ROBOCLOCK_ASSET_CACHE", str(tmp_path))
    image = Image.open(wb._gauge_texture((2.5, 3.5), 6.0))
    (cy, cz), (w, h) = wb.GAUGE_CARD["center"], wb.GAUGE_CARD["size"]
    assert image.size[0] == 1024 and image.size[1] / 1024 == pytest.approx(h / w, rel=0.01)
    assert cy - w / 2 < wb.GAUGE["y0"] < wb.GAUGE["y1"] < cy + w / 2
    assert abs(cy) + w / 2 <= wb.SCREEN[0] / 2 and abs(cz) + h / 2 <= wb.SCREEN[1] / 2
    green = np.asarray(image)[..., :3]
    assert ((green[..., 1] > 180) & (green[..., 0] < 90)).any()  # the due band is drawn


# --- layout ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_layout_puts_the_bag_in_reach_on_the_trolley_and_the_hand_at_rest(seed):
    from roboclock.envs.hospital.bedside import window_syringe as ws
    from roboclock.envs.lab.robot import mount_positions

    task, _ = episode(seed=seed)
    info = task.layout.info
    bvm = info["window_bvm"]
    pad = np.asarray(bvm["pad_center"])
    base = np.asarray(mount_positions(load_config(), "A3")[0])
    assert 0.30 <= np.hypot(*(pad[:2] - base[:2])) <= 0.48  # inside the top-down reach
    assert pad[2] + wb.PAD_SIZE[2] / 2 == pytest.approx(bvm["bag_top"], abs=1e-6)
    surface = info["work_surface"]
    top = surface["center"][2] + surface["size"][2] / 2
    assert pad[2] - wb.PAD_SIZE[2] / 2 - wb.STROKE > top + 0.05  # the pad never reaches the tray
    assert np.allclose(info["initial_ee_position"], base + wb.TUCK)  # tucked, not over the bag
    assert info["hide_bench"] and info["bedside_task"] == "window_bvm"
    # The ward moves with the seed shift: the bag stays on the trolley, the manikin on the mattress.
    shift = np.asarray(info["ward_pose"]["translate"][:2]) - [-ws.ROBOT_WARD[0], -ws.ROBOT_WARD[1]]
    bag = np.asarray(bvm["bag_xy"]) - shift
    assert ws.TROLLEY_X[0] < bag[0] < ws.TROLLEY_X[1] and ws.TROLLEY_Y[0] < bag[1] < ws.TROLLEY_Y[1]
    body = np.asarray(bvm["manikin_xy"]) - shift
    assert 0.866 + 0.25 <= body[0] <= 1.74 - 0.25  # on the mattress (robot x 0.866-1.736)
    assert -0.70 + 0.425 <= body[1] <= 0.53 - 0.40  # on its flat section
    assert bvm["mouth"][2] > wb.MATTRESS_TOP + 0.15
    # Press geometry: past the delivery travel, short of the end stop.
    travel = wb.hover_target(bvm)[2] - wb.press_target(bvm)[2] - wb.HOVER_CLEAR
    assert wb.DELIVER < travel < wb.STROKE
    assert wb.RELEASE < wb.DELIVER


def test_bag_manikin_monitor_and_dial_are_inside_the_head_camera_image():
    from roboclock.envs.hospital.rig import head_camera_pose, head_pixels

    config = load_config()
    for seed in range(5):
        task, _ = episode(seed=seed)
        info = task.layout.info["window_bvm"]
        yaw = math.radians(info["monitor_yaw"])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        screen = np.array([*info["monitor_xy"], 0.0]) + rotation @ np.asarray(info["screen_center"])
        corners = [
            screen + rotation @ np.array([0.0, sy * wb.SCREEN[0] / 2, sz * wb.SCREEN[1] / 2])
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        mx, my = info["manikin_xy"]
        chest = [mx + wb.CHEST["center"][0], my + wb.CHEST["center"][1], info["manikin_top"]]
        dial = next(i for i in task.layout.indicators if i.name == "indicator").position
        points = [info["pad_center"], info["patient_port"], info["mouth"], chest, dial, *corners]
        pixels, depth = head_pixels(config, "A3", points, conservative=True)
        assert np.all(depth > 0.2)
        assert np.all((pixels[:, 0] > 8) & (pixels[:, 0] < 632)), pixels
        assert np.all((pixels[:, 1] > 8) & (pixels[:, 1] < 472)), pixels
        # The screen faces the head camera.
        head = head_camera_pose(config, "A3")[0]
        toward = head[:2] - np.asarray(info["monitor_xy"])
        assert rotation[:2, 0] @ toward / np.linalg.norm(toward) > 0.999


# --- privileged oracle ----------------------------------------------------------------------------


def test_oracle_reaches_the_hover_squeezes_before_each_beat_and_lets_go():
    from roboclock.envs.hospital.policies.trajectories import HospitalTrajectory

    task, backend = episode()
    info = task.layout.info["window_bvm"]
    planner = HospitalTrajectory(load_config()["control"], 5 * DT)
    rest = np.asarray(task.layout.info["initial_ee_position"])
    state = reading(backend)
    state = PhysicalState(state.objects, rest.copy(), controls=state.controls)
    planner.last_goal = rest.copy()
    task.time = 0.1
    for _ in range(600):
        command = wb.command(planner, task, state)
    assert np.allclose(command.position, wb.hover_target(info)) and command.close
    task.time = task.beats[0] - wb.PRESS_LEAD + 0.01
    for _ in range(100):
        command = wb.command(planner, task, state)
    assert np.allclose(command.position, wb.press_target(info))
    assert [e["breath"] for e in task.event_log if e["type"] == "oracle_press_start"] == [0]
    task.time = task.beats[0]
    task._breath(task.time)  # delivered: hold briefly, then back up to the hover
    task.time += wb.HOLD + 0.01
    for _ in range(100):
        command = wb.command(planner, task, state)
    assert np.allclose(command.position, wb.hover_target(info))
    task.time = task.beats[1] - wb.PRESS_LEAD + 0.01
    for _ in range(100):
        command = wb.command(planner, task, state)
    assert np.allclose(command.position, wb.press_target(info))
    assert [e["breath"] for e in task.event_log if e["type"] == "oracle_press_start"] == [0, 1]


def test_author_builds_a_sprung_damped_pad_on_a_prismatic_joint():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    task, backend = episode()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Objects/bvm_bag")
    shell = UsdGeom.Cube.Define(stage, "/World/Objects/bvm_bag/Shell")
    backend.stage = stage
    wb.author(backend)
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath("/World/Joints/bvm_bag"))
    assert joint.GetAxisAttr().Get() == "Z"
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(-wb.STROKE)
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(0.0)
    drive = UsdPhysics.DriveAPI(joint.GetPrim(), "linear")
    assert drive.GetStiffnessAttr().Get() > 0 and drive.GetDampingAttr().Get() > 0
    assert drive.GetTargetPositionAttr().Get() > 0  # preloaded against the upper stop
    origin = np.asarray(task.layout.info["window_bvm"]["pad_center"])
    assert np.allclose(np.asarray(joint.GetLocalPos0Attr().Get()), origin, atol=1e-6)
    # The catalog box is hidden: the scanned bag is the pad's visual.
    assert UsdGeom.Imageable(shell).ComputeVisibility() == UsdGeom.Tokens.invisible


def test_models_are_installed_with_cc_by_credits():
    from roboclock.envs.hospital.props import model_info

    for name in ("bvm_resuscitator", "bvm_manikin", "bvm_tube", "patient_monitor"):
        info = model_info(name)
        assert info is not None, name
        assert info["license"] == "CC-BY-4.0" and info["source"].startswith("https://sketchfab.com/")
        assert info["author"] and len(info["source_glb_sha256"]) == 64
