"""Simulator-free checks of the merged Industry package and its BenchmarkEnv wiring."""

import importlib
import json
import pkgutil
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from roboclock.envs.rates import CONTROL_HZ
from roboclock.envs.registry import get_environment, import_symbol
from roboclock.evaluation import throughput
from roboclock.evaluation.schema import collect_results, validate_row
from roboclock.evaluation.score import score_fields

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "src/roboclock/envs/industry"
NUMBERS = range(41, 51)


def test_native_package_imports_without_simulator_or_path_shadowing():
    script = (
        "import importlib, pkgutil, sys\n"
        "import roboclock.envs.industry.native as native\n"
        "import roboclock.envs.industry.native.v2 as v2\n"
        "for m in pkgutil.iter_modules(native.__path__):\n"
        "    importlib.import_module('roboclock.envs.industry.native.' + m.name)\n"
        "for m in pkgutil.iter_modules(v2.__path__):\n"
        "    importlib.import_module('roboclock.envs.industry.native.v2.' + m.name)\n"
        "import roboclock.envs.industry.env, roboclock.envs.industry.oracle\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('isaacsim', 'omni', 'pxr')]\n"
        "assert not loaded, loaded\n"
        "import roboclock.resources, roboclock.envs.lab.appearance\n"
        "assert roboclock.resources.__file__.endswith('src/roboclock/resources.py')\n"
        "assert 'roboclock._industry_paths' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert not (PACKAGE / "profiles").exists(), "recorded profiles were merged into native/"


def test_native_sources_use_main_package_modules():
    from roboclock.envs.industry.native.appearance import IndustryAppearance
    from roboclock.envs.lab.appearance import LabAppearance

    assert issubclass(IndustryAppearance, LabAppearance)
    text = "\n".join(p.read_text() for p in (PACKAGE / "native").rglob("*.py"))
    assert "_industry_paths" not in text and "from roboclock.envs.industry.native" not in text


def test_catalog_matches_configured_rates_and_reviewed_specs():
    from roboclock.envs.industry.config import load_config
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.tasks import LEGACY_IDS

    config = load_config()
    assert sorted(TASKS) == list(NUMBERS)
    for number, native in TASKS.items():
        task_id = LEGACY_IDS[str(number)]
        assert config["tasks"][task_id]["physics_hz"] == native.physics_hz
        assert native.control_stride * CONTROL_HZ == native.physics_hz
        reviewed = native.reviewed()
        assert reviewed["task_id"] == number
        assert reviewed.get("physics_hz", native.physics_hz) == native.physics_hz
        # Throughput tasks always ran their budget; A/B stopped after completion.
        assert (native.post_complete_s is None) == (number >= 47)
        assert native.final_observe == (number >= 47)


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("sim", "control_dt", 1 / 60, "50 Hz"),
        ("tasks", "B3", {"physics_hz": 960}, "exactly 250 Hz"),
    ],
)
def test_config_rejects_rates_off_the_shared_time_base(tmp_path, section, key, value, message):
    from roboclock.envs.industry.config import load_config

    config = yaml.safe_load((PACKAGE / "configs/benchmark.yaml").read_text())
    config[section][key] = value
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize("number", NUMBERS)
def test_reviewed_seed_replays_recorded_episode(number):
    from roboclock.envs.industry.native.catalog import verify
    from roboclock.envs.industry.tasks import LEGACY_IDS, make_task

    reviewed = json.loads((PACKAGE / f"configs/task{number}/reviewed.json").read_text())
    task_id = LEGACY_IDS[str(number)]
    # A/B replays keep the recorded 90 s; C episodes run to their horizon at every difficulty.
    length = throughput.horizon(throughput.oracle_time("industry", task_id)) if number >= 47 else 90.0
    for alias in (number, str(number), task_id):
        task = make_task(alias, "reviewed_nominal", reviewed["seed"])
        assert task.task_id == task_id
        assert task.spec.resolved["sha256"] == reviewed["sha256"]
        assert task.spec.max_time == length
    other = make_task(number, "reviewed_nominal", reviewed["seed"] + 7)
    verify(other.spec.resolved)
    assert other.spec.resolved["seed"] == reviewed["seed"] + 7
    assert not any(other.spec.resolved["randomization"].values())


def test_unknown_difficulty_and_task_are_rejected():
    from roboclock.envs.industry.tasks import make_task

    with pytest.raises(ValueError):
        make_task("A1", "extreme", 0)
    with pytest.raises(ValueError):
        make_task("D1", "reviewed_nominal", 0)


def test_spec_declares_oracle_and_three_levels():
    spec = get_environment("industry")
    assert spec.difficulties == ("easy", "medium", "hard")
    assert import_symbol(spec.oracle_factory).__name__ == "IndustryOracle"
    assert spec.asset_root == REPO / "assets/industry"


def test_pose_window_matches_native_controller_windows():
    from roboclock.envs.industry.scoring import PoseWindow

    window = PoseWindow(0.15, 0.0002, 0.2)
    identity = np.eye(3)
    stable = None
    for i in range(12):
        stable = window.add(i / CONTROL_HZ, [0.0, 0.0, 1e-5 * i], identity)
    assert stable is not None and stable["position_span_m"] <= 0.0002
    assert window.add(12 / CONTROL_HZ, [0.0, 0.0, 0.01], identity) is None
    window.clear()
    assert window.add(0.0, [0, 0, 0], identity) is None


class _Scene:
    def __init__(self, completed, success):
        self.events = {"warning_onset_s": 16.0, "button_confirmed_s": 16.8, "response_s": 0.8}
        if completed:
            self.events["motor_stopped_s"] = 17.3
        self.physical_completed, self.success = completed, success
        self.contacts = SimpleNamespace(unexpected={})

    def final_valid(self, o):
        return self.physical_completed and abs(o["rotor_speed_rad_s"]) < 0.05


@pytest.mark.parametrize(("completed", "success"), [(True, True), (True, False), (False, False)])
def test_monitor_rows_pass_the_result_schema(completed, success):
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.scoring import MotorMonitor
    from roboclock.envs.industry.tasks import make_task

    task = make_task("B2", "reviewed_nominal", 0)
    monitor = MotorMonitor(_Scene(completed, success), task.spec.resolved, TASKS[45])
    o = {"time_s": 18.0, "rotor_speed_rad_s": 0.0}
    monitor.physics(o)
    task.record(monitor.outcome(o))
    row = {**task.metrics(), "environment": "industry"}
    json.dumps(row, allow_nan=False)
    assert validate_row(row) == []
    assert row["joint_success"] == (completed and success)
    assert row["reaction_latency"] == pytest.approx(0.8)
    assert monitor.completed_at == (17.3 if completed else None)


def test_recorded_sidecars_normalize_to_class_ids():
    rows, _ = collect_results([REPO / "data/industry"])
    by_task = {row["task_id"]: row for row in rows}
    assert set(by_task) == {"A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "C4"}
    assert by_task["C3"]["seed"] == 1
    assert all(validate_row(row) == [] for row in rows)


def _video_args(task, output):
    return SimpleNamespace(
        task=task,
        assets=REPO / "assets/industry",
        output=output,
        mode="nominal",
        seconds=90,
        visual_context="minimal",
        replay=None,
        seed=None,
        randomize_layout=False,
        randomize_timing=False,
        randomize_physics=False,
        record=False,
    )


def test_videos_dry_run_uses_merged_runner_without_pythonpath(tmp_path):
    from roboclock.envs.industry import videos

    command, env, _, _ = videos.make_command(_video_args(42, tmp_path / "out"))
    assert command[1:3] == ["-m", "roboclock.envs.industry.native.runner"]
    assert "--assets" in command and command[command.index("--task") + 1] == "42"
    assert "PYTHONPATH" not in env or "profiles" not in env["PYTHONPATH"]


@pytest.mark.parametrize(
    "number, profile, renderer",
    [(41, "isaac601-robodojo", "RaytracedLighting"), (49, "isaac601-robodojo", "RaytracedLighting")],
)
def test_videos_select_each_tasks_recording_profile(tmp_path, number, profile, renderer):
    from roboclock.envs.industry import videos
    from roboclock.envs.industry.native.v2.runtime_profiles import PROFILES

    _, env, _, _ = videos.make_command(_video_args(number, tmp_path / "out"))
    assert env["ROBOCLOCK_V2_RUNTIME_PROFILE"] == profile
    assert env["ROBOCLOCK_V2_RENDERER"] == renderer
    assert renderer in PROFILES[profile].allowed_renderers


def test_task49_resolver_reproduces_the_reviewed_clearance_layout():
    from roboclock.envs.industry.native.catalog import TASKS

    native = TASKS[49]
    reviewed = native.reviewed()
    assert reviewed["source_positions_xy_m"][0] == [-0.28, 0.73]
    assert reviewed["runtime_finger_velocity_limit_m_s"] == 0.02
    assert native.load()[2](reviewed["seed"])["sha256"] == reviewed["sha256"]


def test_weighing_score_fails_on_a_zero_impulse_arm_contact():
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.scoring import WeighingMonitor

    native = TASKS[49]
    contacts = SimpleNamespace(unexpected={}, interarm_events={})
    monitor = WeighingMonitor(SimpleNamespace(contacts=contacts), native.reviewed(), native)
    assert monitor.unexpected() == {}
    contacts.interarm_events["left/link6 | right/link6"] = dict(time_s=24.0)
    assert list(monitor.unexpected()) == ["left/link6 | right/link6"]


def test_every_native_module_is_used_by_some_task():
    from roboclock.envs.industry.native.catalog import TASKS

    modules = {m.name for m in pkgutil.iter_modules([str(PACKAGE / "native")])}
    for native in TASKS.values():
        assert native.module in modules
        importlib.import_module(f"roboclock.envs.industry.native.{native.module}")


def test_oracle_keeps_controller_events_and_float64_commands_private():
    from roboclock.envs.industry.oracle import IndustryOracle

    names = [f"joint{i}" for i in (1, 2, 3, 4, 5, 6, 8, 7)]  # native order may differ
    dual = SimpleNamespace(
        names={"left": names, "right": names},
        commands={side: np.linspace(0.1, 0.8, 8) + (0.01 if side == "right" else 0.0) for side in ("left", "right")},
    )
    scene = SimpleNamespace(dual=dual, events={"warning_onset_s": 16.0})

    class Controller:
        def step(self, o):
            assert scene.events["warning_onset_s"] == 16.0  # reads scene events
            scene.events["phase_press_s"] = o["time_s"]  # controller-only event
            dual.commands["left"][0] += 1e-9  # float64 increment below float32 resolution

    env = SimpleNamespace(
        native_controller=Controller(),
        scene=scene,
        native_observation={"time_s": 16.0125},
        joint_index={side: [names.index(f"joint{i}") for i in range(1, 9)] for side in ("left", "right")},
    )
    oracle = IndustryOracle(env)
    oracle.reset({})
    start = oracle.commands["left"][0]
    action = oracle.act({})
    assert action.dtype == np.float32 and action.shape == (16,)
    assert scene.events == {"warning_onset_s": 16.0}
    assert oracle.overlay == {"phase_press_s": 16.0125}
    # joint7/joint8 are reordered by name; the right arm follows the left.
    assert action[6] == np.float32(dual.commands["left"][names.index("joint7")])
    # The environment writes the float32 action back; the controller keeps float64.
    dual.commands["left"][:] = np.float32(dual.commands["left"])
    oracle.act({})
    assert oracle.commands["left"][0] == pytest.approx(start + 2e-9, rel=0, abs=1e-15)


# Every task is on the common Dual ARX rig (native/rig.py); module holding its mounts.
RIG_MODULES = {
    41: "plating_task",
    42: "rotary_task",
    43: "bonding_task",
    44: "isolation_task",
    45: "motor_arx_task",
    46: "bearing_arx_task",
    47: "kitting_batch",
    48: "putaway_rig",
    49: "weigh_sort_task",
    50: "palette_arx_task",
}


@pytest.mark.parametrize("number", sorted(RIG_MODULES))
def test_rig_tasks_mount_both_arms_side_by_side_with_the_head_camera(number):
    from roboclock.envs import robodojo_rig as spec
    from roboclock.envs.industry.native import rig

    mounts = importlib.import_module(f"roboclock.envs.industry.native.{RIG_MODULES[number]}").MOUNTS
    rig.check(mounts)
    left, right = (np.asarray(mounts[side], float) for side in ("left", "right"))
    assert right[0] - left[0] == pytest.approx(spec.ARM_SEPARATION_M) and left[1:] == pytest.approx(right[1:])
    # Spec head camera: base midpoint + (0.04 forward, 0.543 up), 60 deg down, arms facing +Y.
    head = rig.head_pose(mounts)
    assert head[:3, 3] == pytest.approx((left + right) / 2 + [0, 0.04, 0.543])
    look = -head[:3, 2]
    assert np.degrees(np.arctan2(-look[2], look[1])) == pytest.approx(spec.HEAD_PITCH_DOWN_DEG)
    left_pose, right_pose = (np.eye(4) for _ in range(2))
    for pose, p in ((left_pose, left), (right_pose, right)):
        pose[:3, :3] = spec.rig_to_world((0, 0, 0), rig.DEFAULT_YAW_DEG)[:3, :3]
        pose[:3, 3] = p
    spec.check_rig(left_pose, right_pose, head)


def test_rig_rejects_other_spacings_heights_and_depths():
    from roboclock.envs.industry.native import rig

    for left, right in [
        ([-0.3, 0.2, 1.1], [0.35, 0.2, 1.1]),
        ([-0.3, 0.2, 1.1], [0.3, 0.2, 1.0]),
        ([-0.3, 0.2, 1.1], [0.3, 0.25, 1.1]),
    ]:
        with pytest.raises(AssertionError):
            rig.check(dict(left=left, right=right))
    kept = rig.mounts_keeping("right", [0.15, 0.2, 1.13])
    assert kept["right"] == pytest.approx([0.15, 0.2, 1.13])
    assert kept["left"] == pytest.approx([-0.45, 0.2, 1.13])


def test_every_task_uses_the_robodojo_rate_renderer_and_cameras():
    from roboclock.envs import robodojo_rig as spec
    from roboclock.envs.industry.config import load_config
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.native.overview import OVERVIEW

    config = load_config()
    assert "camera_resolution" not in config["sim"] and config["sim"]["no_render"] is False
    for number, native in TASKS.items():
        assert native.physics_hz == spec.PHYSICS_HZ
        assert native.reviewed().get("physics_hz", spec.PHYSICS_HZ) == spec.PHYSICS_HZ
        assert native.renderer == spec.RENDER_MODE and native.runtime_profile == "isaac601-robodojo"
        assert number in OVERVIEW


def test_rendered_intrinsics_record_square_pixels_for_the_head_camera():
    """Metadata records the spec's rendered intrinsics (RTX square pixels: head fx = fy,
    79.6 deg vertical FOV) next to the API's reported ones; projection uses the same."""
    from roboclock.envs import robodojo_rig as spec
    from roboclock.envs.industry.native import rig

    rendered = rig.rendered_intrinsics()
    head = np.asarray(rendered["rgb"])
    assert head == pytest.approx(spec.intrinsics(spec.HEAD_CAMERA), abs=1e-4)
    assert head[0, 0] == pytest.approx(288.13, abs=0.01) and head[1, 1] == pytest.approx(head[0, 0])
    for key in ("left_wrist_rgb", "right_wrist_rgb"):
        assert np.asarray(rendered[key]) == pytest.approx(spec.intrinsics(spec.WRIST_CAMERA), abs=1e-4)
    # The old 4.5 cm task 45 fault lamp's measured centroid in the rendered head frame: (398.0, 69.6) px
    # (that point now lies inside the larger fault panel; the projection check is unchanged).
    from roboclock.envs.industry.native.motor_arx_task import MOUNTS

    (u, v), = rig.project_head(MOUNTS, [[-0.55, 0.743, 0.98]])
    assert abs(u - 398.0) < 1.5 and abs(v - 69.6) < 1.5


def test_task47_rig_layout_swaps_roles_and_keeps_each_arms_recorded_geometry():
    from roboclock.envs.industry.native import kitting_batch as kb
    from roboclock.envs.industry.native.catalog import TASKS

    reviewed = TASKS[47].reviewed()
    assert reviewed["fixed_mounts_m"] == kb.MOUNTS and reviewed["kit_centers_x_m"] == kb.KIT_X
    assert reviewed["kit_sides"] == ["right", "left"]
    # Recorded: left arm at x=-0.64 packed kit 0 at -0.28 from sources -0.72/-0.56/-0.40,
    # right arm at +0.64 packed kit 1 (mirror). Each kit now keeps that geometry
    # relative to the arm that packs it.
    recorded_base = {0: -0.64, 1: 0.64}
    recorded_kit = {0: -0.28, 1: 0.28}
    recorded_sources = {0: [-0.72, -0.56, -0.40], 1: [0.72, 0.56, 0.40]}
    for kit, side in enumerate(reviewed["kit_sides"]):
        base = kb.MOUNTS[side][0]
        assert kb.KIT_X[kit] - base == pytest.approx(recorded_kit[kit] - recorded_base[kit])
        assert np.asarray(reviewed["source_x_m"][kit]) - base == pytest.approx(
            np.asarray(recorded_sources[kit]) - recorded_base[kit]
        )
    # 0.545 m cases at +/-0.66 clear each other and the 0.15 m base plates.
    assert kb.KIT_X[0] - 0.2727 > kb.MOUNTS["right"][0] + 0.075
    assert kb.KIT_X[1] + 0.2727 < kb.MOUNTS["left"][0] - 0.075
    assert reviewed["bracket_tool_yaw_deg"] == {"left": 180, "right": 0}


def test_task48_rig_layout_keeps_cartons_and_clamp_on_the_centre_line():
    from roboclock.envs.industry.native import putaway_rig as pr
    from roboclock.envs.industry.native.catalog import TASKS

    reviewed = TASKS[48].reviewed()
    assert TASKS[48].module == "putaway_rig" and reviewed["fixed_mounts_m"] == pr.MOUNTS
    assert reviewed["clamp_tool_yaw_deg"] == {"left": 0, "right": 180}
    centre = (np.asarray(pr.MOUNTS["left"]) + np.asarray(pr.MOUNTS["right"])) / 2
    for product in reviewed["products"]:
        # Unchanged carton sizes; pickup and slot both on the rig centre line, 0.10-0.50 m ahead.
        assert product["size_m"] in ([0.28, 0.10, 0.14], [0.28, 0.18, 0.26])
        for x, y in (product["source_center_m"][:2], product["destination_center_xy_m"]):
            assert x == pytest.approx(centre[0])
            assert 0.10 <= y - centre[1] <= 0.50
    floors = {p["sku"]: p["destination_floor_m"] for p in reviewed["products"]}
    assert floors["A"] > floors["B"]  # A-01 upper deck, B-01 lower deck
    assert reviewed["budget_s"] == 90.0


def test_task49_receivers_sit_beside_each_arm_in_the_head_view():
    from roboclock.envs.industry.native import rig
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.native.weigh_sort_task import MOUNTS

    reviewed = TASKS[49].reviewed()
    receivers = reviewed["receivers_by_arm_xy_m"]
    assert set(receivers) == {"left", "right"} and reviewed["budget_s"] == 120.0
    width, depth, _ = reviewed["receiver_size_m"]
    for side, trays in receivers.items():
        assert set(trays) == {"PASS", "REJECT"}
        for x, y in trays.values():
            assert np.sign(x) == (-1 if side == "left" else 1)
            corners = [[x + dx * width / 2, y + dy * depth / 2, 0.9225] for dx in (-1, 1) for dy in (-1, 1)]
            assert rig.inside_head_image(MOUNTS, corners)


@pytest.mark.parametrize("number", range(41, 51))
def test_key_objects_project_inside_the_head_image(number):
    """robodojo_rig visibility: every required key object/cue corner >= 10 px inside the
    640x480 rendered head image (robodojo_rig.intrinsics); known partial ones stay partly visible."""
    from roboclock.envs.industry.native import rig
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.native.visibility import KEY_OBJECTS, corners

    module = importlib.import_module(f"roboclock.envs.industry.native.{TASKS[number].module}")
    mounts = getattr(module, "MOUNTS", None) or importlib.import_module(
        f"roboclock.envs.industry.native.{RIG_MODULES[number]}"
    ).MOUNTS
    for name, box, mode in KEY_OBJECTS[number]["required"]:
        assert rig.inside_head_image(mounts, corners(box, mode)), (number, name)
    for name, box, mode, reason in KEY_OBJECTS[number]["partial"]:
        points = corners(box, mode)
        inside = sum(rig.inside_head_image(mounts, [p]) for p in points)
        assert 0 < inside < len(points), (number, name, inside)
        assert reason


LEVEL_TASKS = ("A1", "A2", "A3", "B1", "B2", "B3")
C_TASKS = ("C1", "C2", "C3", "C4")
# Keys a level may change; everything else in the resolved spec is drawn from the seed.
TEMPORAL_KEYS = {
    "process_max_s", "loading_sector_deg", "window_s", "window_relative_s",
    "response_deadline_s", "difficulty", "temporal", "sha256",
}


@pytest.mark.parametrize("task_id", LEVEL_TASKS)
def test_levels_change_only_the_temporal_requirement(task_id):
    from roboclock.envs.industry.native.catalog import verify
    from roboclock.envs.industry.tasks import LEVEL_TABLE, LEVELS, make_task

    flags = {**LEVEL_TABLE["randomization"], **LEVEL_TABLE["tasks"][task_id].get("randomization", {})}
    specs = {level: make_task(task_id, level, 5) for level in LEVELS}
    for task in specs.values():
        verify(task.spec.resolved)
        assert task.spec.resolved["randomization"] == flags
    easy, medium, hard = (specs[level].spec.resolved for level in LEVELS)
    for key in set(easy) - TEMPORAL_KEYS:
        assert easy[key] == medium[key] == hard[key], key
    # Easy is never tighter than medium, nor medium than hard.
    budgets = [specs[level].spec.max_time for level in LEVELS]
    assert budgets == sorted(budgets, reverse=True)
    assert len({task.spec.time_requirement for task in specs.values()}) == 3
    assert len({task.spec.resolved["sha256"] for task in specs.values()}) == 3


def _seconds(text):
    return re.findall(r"(\d+(?:\.\d+)?)\s*s\b", text)


@pytest.mark.parametrize("task_id", C_TASKS)
def test_throughput_levels_share_one_episode(task_id):
    """docs/c-class-timing.md: the difficulty changes nothing in a C episode; it only picks
    the tier that fills temporal_success / joint_success."""
    from roboclock.envs.industry.native.catalog import verify
    from roboclock.envs.industry.tasks import LEVEL_TABLE, LEVELS, make_task

    flags = {**LEVEL_TABLE["randomization"], **LEVEL_TABLE["tasks"][task_id].get("randomization", {})}
    t_o = throughput.oracle_time("industry", task_id)
    horizon = throughput.horizon(t_o)
    specs = {level: make_task(task_id, level, 5).spec for level in LEVELS}
    for spec in specs.values():
        verify(spec.resolved)
        assert spec.resolved["randomization"] == flags
        assert "difficulty" not in spec.resolved
        # The native controllers' give-up time is the whole episode.
        assert spec.resolved["budget_s"] == spec.max_time == pytest.approx(5 * t_o) == horizon
        assert spec.resolved["temporal"] == {
            "oracle_time_s": t_o, "budget_s_by_tier": throughput.budgets(t_o), "horizon_s": horizon,
        }
        # No level values for C tasks; no seconds in the requirement or the instruction (the
        # only number is C3's 0.50 s scale rest, a task rule).
        assert set(LEVEL_TABLE["tasks"][task_id]) <= {"randomization"}
        assert "as quickly as possible" in spec.time_requirement and not _seconds(spec.time_requirement)
        assert set(_seconds(spec.instruction)) <= {"0.50"}
    easy, medium, hard = (specs[level] for level in LEVELS)
    assert easy.resolved == medium.resolved == hard.resolved
    assert easy.max_time == medium.max_time == hard.max_time
    assert easy.time_requirement == medium.time_requirement == hard.time_requirement
    assert easy.instruction == medium.instruction == hard.instruction


def test_level_values_reach_the_resolved_spec():
    from roboclock.envs.industry.tasks import make_task

    a1 = make_task("A1", "hard", 2).spec.resolved
    assert a1["process_max_s"] == pytest.approx(a1["process_min_s"] + 10.14)
    a2 = make_task("A2", "hard", 2).spec
    assert a2.resolved["rotation_deg_s"] == -5.0 and a2.resolved["window_s"] == 8.0
    assert a2.resolved["loading_sector_deg"] == [-120.0, -80.0]
    assert a2.time_requirement == (
        "Complete supported placement while the carrier is inside the 40° loading sector at "
        "5°/s, an 8 s window."
    )
    easy = make_task("A2", "easy", 2).spec
    assert easy.resolved["loading_sector_deg"] == [-120.0, -60.0]
    assert easy.time_requirement == make_task("A2", "reviewed_nominal", 2).spec.time_requirement
    assert make_task("A3", "medium", 2).spec.resolved["window_relative_s"] == [2.0, 25.67]
    assert make_task("B2", "hard", 2).spec.resolved["response_deadline_s"] == 2.53
    c3 = make_task("C3", "medium", 2).spec
    assert c3.resolved["temporal"]["oracle_time_s"] == 69.36
    assert c3.resolved["temporal"]["budget_s_by_tier"]["medium"] == pytest.approx(throughput.MULTIPLIER["medium"] * 69.36)
    assert c3.max_time == pytest.approx(throughput.horizon(69.36))
    assert c3.time_requirement == (
        "Weigh each of the 3 parts and sort every part correctly, as quickly as possible."
    )


def test_reviewed_nominal_is_unchanged_and_config_max_time_sets_the_episode_length():
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.tasks import NOMINAL_TIER, make_task

    reviewed = make_task("C4", "reviewed_nominal", 0)
    assert reviewed.spec.resolved == TASKS[50].reviewed()  # recorded spec, 90 s controller limit
    assert reviewed.spec.max_time == pytest.approx(throughput.horizon(81.46)) and reviewed.tier == NOMINAL_TIER
    assert make_task("B2", "reviewed_nominal", 0).spec.max_time == 90.0
    # tasks.<ID>.max_time is only an episode length (short debugging runs): the spec, the
    # tiers' budgets and the stall stop stay the frozen oracle time's.
    hard = make_task("C4", "hard", 0)
    short = make_task("C4", "hard", 0, {"tasks": {"C4": {"max_time": 10.0}}})
    assert short.spec.max_time == 10.0 and short.spec.resolved == hard.spec.resolved
    assert short.metrics()["budget_s_by_tier"] == throughput.budgets(81.46)


def test_config_accepts_an_episode_length_past_the_old_150_s_cap(tmp_path):
    from roboclock.envs.industry.config import load_config

    config = yaml.safe_load((PACKAGE / "configs/benchmark.yaml").read_text())
    config["tasks"]["C3"]["max_time"] = 415.45
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["tasks"]["C3"]["max_time"] == 415.45
    config["tasks"]["C3"]["max_time"] = 0
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="max_time"):
        load_config(path)


# ---------------------------------------------------------------- C-class judging
class _Shelf:
    """Fake C2 scene: carton i is valid on its shelf while `valid[i]`."""

    def __init__(self):
        self.valid = [False, False, False]
        self.events = {}
        self.contacts = SimpleNamespace(unexpected={})

    def valid_product(self, index):
        return self.valid[index]


def _putaway(tick=0.02):
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.scoring import PutawayMonitor

    scene = _Shelf()
    monitor = PutawayMonitor(scene, TASKS[48].reviewed(), TASKS[48])
    clock = {"t": 0.0}

    def run_until(t_end, keep=False, **valid):
        """Set cartons' validity (c<i>=bool), then tick at 50 Hz up to t_end or, unless
        `keep`, the monitor's early end. A carton made valid now is credited 0.54 s later
        (valid on the next tick, then held > 0.5 s)."""
        for index, value in valid.items():
            scene.valid[int(index[1:])] = value
        while clock["t"] < t_end - 1e-9 and (keep or not monitor.finished(clock["t"])):
            clock["t"] = round(clock["t"] + tick, 6)
            monitor.tick({"time_s": clock["t"]})
        return clock["t"]

    return monitor, scene, run_until


def _row(monitor, difficulty, time):
    from roboclock.envs.industry.tasks import make_task

    task = make_task("C2", difficulty, 0)
    task.time = time
    task.record(monitor.outcome({"time_s": time}))
    row = {**task.metrics(), "environment": "industry"}
    json.dumps(row, allow_nan=False)
    assert validate_row(row) == []
    return row


def test_throughput_monitors_read_job_times_from_their_credit_events():
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.scoring import MONITORS

    events = {
        47: {"kit1_complete_s": 50.0, "kit0_bearing_credited_s": {"time_s": 10.0}},
        48: {"0_correct_slot_credit_s": 15.3, "2_correct_slot_credit_s": 43.1},
        49: {"1_sorted_s": 30.0, "1_weighed": {"time_s": 20.0}},
        50: {"delivery0_s": {"time_s": 79.8, "window": {}}, "pan1_0_credited_s": {"time_s": 9.0}},
    }
    expected = {47: [None, 50.0], 48: [15.3, None, 43.1], 49: [None, 30.0, None], 50: [79.8, None]}
    scene = SimpleNamespace(contacts=SimpleNamespace(unexpected={}, interarm_events={}), events={})
    for number, monitor_class in ((n, MONITORS[n]) for n in expected):
        monitor = monitor_class(scene, TASKS[number].reviewed(), TASKS[number])
        assert monitor.task_id == f"C{number - 46}"
        assert monitor.oracle_time == throughput.oracle_time("industry", monitor.task_id)
        monitor.events.update(events[number])
        assert monitor.job_times() == expected[number]
        assert len(monitor.job_times()) == monitor.required_operations
        assert monitor.last_job_time() == max(t for t in expected[number] if t is not None)
    # C2's scene holds exactly Q = 3 cartons.
    assert len(TASKS[48].reviewed()["products"]) == MONITORS[48].required_operations


def test_throughput_final_validity_has_no_time_term():
    """Physical success needs every job done and still valid at the end, however late."""
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.scoring import MONITORS

    late = {"time_s": 1000.0}
    kitting = MONITORS[47](SimpleNamespace(contained=lambda name, j: True), TASKS[47].reviewed(), TASKS[47])
    kitting.credited = {(j, name) for j in (0, 1) for name in kitting.names}
    kitting.mark("required_operations_s", 999.0)
    floor = kitting.resolved["kit_floor_m"]
    released = {
        "whole_contained": True, "bounds_min_m": [0.0, 0.0, floor], "kit_support_N": 0.2,
        "kit_touching": True, "finger_forces_N": {"left": [0.0, 0.0], "right": [0.0, 0.0]},
    }
    parts = {f"{name}{j}": released for name in kitting.names for j in (0, 1)}
    assert kitting.final_valid({**late, "parts": parts})
    held = {**released, "finger_forces_N": {"left": [0.5, 0.5], "right": [0.0, 0.0]}}
    assert not kitting.final_valid({**late, "parts": {**parts, "bracket0": held}})
    kitting.scene.contained = lambda name, j: (name, j) != ("pin", 1)
    assert not kitting.final_valid({**late, "parts": parts})

    still = SimpleNamespace(get_linear_velocity=lambda: np.zeros(3), get_angular_velocity=lambda: np.zeros(3))
    inspection_scene = SimpleNamespace(
        contained=lambda i: True, receiver_touching=lambda i: True, parts=[still] * 3,
        contacts=SimpleNamespace(unexpected={}, interarm_events={}),
    )
    inspection = MONITORS[49](inspection_scene, TASKS[49].reviewed(), TASKS[49])
    inspection.sorted = {0, 1, 2}
    inspection.weigh.weighed = {i: 150.0 for i in range(3)}
    inspection.mark("required_operations_s", 999.0)
    parts = {str(i): {"support_N": 1.0, "finger_forces_by_arm": {"left": [0, 0], "right": [0, 0]}} for i in range(3)}
    assert inspection.final_valid({**late, "parts": parts})
    inspection.sorted = {0, 1}
    assert not inspection.final_valid({**late, "parts": parts})

    palette_scene = SimpleNamespace(
        payload_native_valid=lambda o, c, pans=None: True, contained=lambda c, color, body: True,
        pan={(color, k): None for color in range(4) for k in (0, 2)},
    )
    palette = MONITORS[50](palette_scene, TASKS[50].reviewed(), TASKS[50])
    palette.delivered = {0, 1}
    palette.pan_of = {(c, color): 1 for c in (0, 1) for color in range(4)}
    palette.mark("delivery0_s", 500.0, window={})
    palette.mark("delivery1_s", 900.0, window={})
    assert palette.final_valid(late)
    palette.delivered = {0}
    assert not palette.final_valid(late)


def test_throughput_physical_needs_all_jobs_valid_at_the_end_and_no_time_term():
    monitor, scene, run_until = _putaway()
    run_until(200.0, keep=True, c0=True, c1=True)  # past the stall stop; keep measuring
    assert monitor.termination == "stall"
    end = run_until(400.0, keep=True, c2=True)
    row = _row(monitor, "easy", end)
    assert row["jobs_done"] == 3 and row["t_q_s"] == pytest.approx(monitor.completed_at)
    assert row["task_success"] and row["manipulation_success"]  # late, but physically done
    assert row["temporal_success_by_tier"] == {"easy": False, "medium": False, "hard": False}
    assert not row["joint_success"]
    # A carton that left its shelf by the end voids physical success.
    scene.valid[1] = False
    row = _row(monitor, "easy", end)
    assert not row["task_success"]
    assert "done work no longer valid at the end of the episode" in row["temporal_violations"]
    # So does an arm collision: a link other than a finger pressing with over 20 N.
    scene.valid[1] = True
    hand = "/World/ARX/left/Geometry/base_link/link1/link2/link3/link4/link5/link6"
    scene.contacts.unexpected[f"{hand} | /World/RigShelf/UpperDeck"] = {"time_s": 12.0, "peak_force_N": 2.0}
    assert _row(monitor, "easy", end)["task_success"]  # a graze is only reported
    scene.contacts.unexpected[f"{hand} | /World/RigShelf/UpperDeck"]["peak_force_N"] = 60.0
    assert not _row(monitor, "easy", end)["task_success"]


def test_throughput_temporal_tiers_come_from_t_q():
    t_o = throughput.oracle_time("industry", "C2")  # 45.32: T = 181.28 / 101.34 / 56.65 s
    monitor, scene, run_until = _putaway()
    run_until(20.0, c0=True)
    run_until(60.0, c1=True)
    end = run_until(70.0, c2=True)
    t_q = monitor.completed_at
    assert monitor.job_times() == pytest.approx([0.54, 20.54, 60.54]) and t_q == pytest.approx(60.54)
    assert throughput.budgets(t_o)["hard"] < t_q < throughput.budgets(t_o)["medium"]
    rows = {tier: _row(monitor, tier, end) for tier in throughput.TIERS}
    for tier, row in rows.items():
        assert row["t_q_s"] == row["completion_time"] == pytest.approx(t_q)
        assert row["task_success"]
        assert row["temporal_success_by_tier"] == {"easy": True, "medium": True, "hard": False}
        assert row["joint_success_by_tier"] == row["temporal_success_by_tier"]
        assert row["temporal_success"] == row["joint_success"] == (tier != "hard")
        assert row["judged_tier"] == tier and row["oracle_time_s"] == t_o
    assert rows["hard"]["temporal_violations"] == [
        f"all 3 jobs done at {t_q:.2f} s, after the hard budget {throughput.budgets(t_o)['hard']:.2f} s"
    ]
    scores = score_fields("industry", rows["hard"])
    assert scores["score_by_tier"] == {"easy": 100, "medium": 100, "hard": 80}
    # reviewed_nominal is no tier; it reports NOMINAL_TIER's verdict.
    nominal = _row(monitor, "reviewed_nominal", end)
    assert nominal["judged_tier"] == "medium" and nominal["joint_success"]


def test_throughput_ends_once_all_jobs_stay_valid_for_the_settle_time():
    monitor, scene, run_until = _putaway()
    run_until(1.0, c0=True, c1=True, c2=True)
    t_q = monitor.completed_at
    assert t_q == pytest.approx(0.54) and monitor.termination is None
    # A carton knocked off its shelf restarts the settle time.
    run_until(t_q + 1.5)
    run_until(t_q + 1.6, c1=False)
    assert monitor.termination is None
    end = run_until(60.0, c1=True)  # valid again from the next tick on
    assert monitor.termination == "all_done" and monitor.finished(end)
    assert end == pytest.approx(t_q + 1.6 + 0.02 + throughput.SETTLE_S)
    row = _row(monitor, "hard", end)
    assert row["termination_reason"] == "all_done" and row["joint_success"]
    # The environment's horizon never overwrites an earlier reason.
    monitor.end("horizon")
    assert monitor.termination == "all_done"


@pytest.mark.parametrize(
    "credits, stop",
    [
        ({}, throughput.MULTIPLIER["medium"] * 45.32),  # nothing done: armed at T_medium, fires at once
        ({80.0: "c0"}, 80.54 + throughput.stall_window(45.32, 3)),  # window after the last job
    ],
)
def test_throughput_stall_stop(credits, stop):
    monitor, scene, run_until = _putaway()
    for t, carton in credits.items():
        run_until(t)
        run_until(t + 1.0, **{carton: True})
    end = run_until(400.0)
    assert monitor.termination == "stall"
    assert stop - 1e-9 <= end <= stop + 0.02 + 1e-9  # first control tick at or after the stop
    row = _row(monitor, "easy", end)
    assert row["termination_reason"] == "stall" and not row["task_success"]
    assert row["jobs_done"] == len(credits) and row["t_q_s"] is None
    assert not any(row["temporal_success_by_tier"].values())


def test_throughput_stall_counts_credited_sub_steps():
    """A kit placed component by component is progress: whole kits alone would stop a policy
    still working at T_medium (C1: one kit takes the oracle ~32 s)."""
    from roboclock.envs.industry import scoring

    class Kits(scoring.ThroughputMonitor):
        progress_patterns = scoring.KittingMonitor.progress_patterns

        def __init__(self):
            self.events = {}

        def job_times(self):
            return [None, None]

    monitor, t_o = Kits(), throughput.oracle_time("industry", "C1")
    monitor.events["kit0_pin_credited_s"] = throughput.MULTIPLIER["medium"] * t_o - 1.0
    assert monitor.last_job_time() is None
    assert monitor.last_progress_time() == pytest.approx(throughput.MULTIPLIER["medium"] * t_o - 1.0)
    assert throughput.stalled(throughput.MULTIPLIER["medium"] * t_o, monitor.last_job_time(), t_o)
    assert not throughput.stalled(throughput.MULTIPLIER["medium"] * t_o, monitor.last_progress_time(), t_o)
    for cls in (scoring.KittingMonitor, scoring.PutawayMonitor, scoring.WeighingMonitor,
                scoring.PaletteMonitor):
        assert cls.progress_patterns, cls.__name__


def test_throughput_horizon_reason_and_running_rows():
    monitor, scene, run_until = _putaway()
    end = run_until(30.0, c0=True)
    assert _row(monitor, "medium", end)["termination_reason"] is None  # still running
    monitor.end("horizon")  # the environment reached spec.max_time
    row = _row(monitor, "medium", end)
    assert row["termination_reason"] == "horizon" and row["jobs_done"] == 1
    assert row["job_times_s"] == [pytest.approx(0.54), None, None]


class _World:
    def __init__(self, hz):
        self.dt, self.current_time = 1.0 / hz, 0.0

    def step(self, render=False):
        self.current_time = round(self.current_time + self.dt, 9)


@pytest.mark.parametrize(
    "valid, max_time, reason, end",
    [
        # Ticks at 0.004 + 0.02 k s: credited at 0.524 s, settled 2 s later.
        (True, None, "all_done", 0.524 + throughput.SETTLE_S),
        (False, None, "stall", throughput.MULTIPLIER["medium"] * 45.32),  # nothing done by T_medium
        (False, 5.0, "horizon", 5.0),  # config max_time cuts the episode
    ],
)
def test_environment_ends_throughput_episodes_early_and_names_the_reason(valid, max_time, reason, end):
    """IndustryEnv's stepping loop (no simulator): C2 with a fake shelf scene."""
    from roboclock.envs.industry.env import IndustryEnv
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.scoring import PutawayMonitor
    from roboclock.envs.industry.tasks import make_task

    config = {"tasks": {"C2": {"max_time": max_time}}} if max_time else None
    env = object.__new__(IndustryEnv)
    env.task = make_task("C2", "medium", 0, config)
    native = TASKS[48]
    scene = _Shelf()
    scene.valid = [valid] * 3
    scene.world = _World(native.physics_hz)
    scene.dual = SimpleNamespace(apply=lambda: None)
    scene.observe = lambda: {"time_s": scene.world.current_time}
    env.scene = scene
    env.native = SimpleNamespace(
        control_stride=native.control_stride, observe_every_step=native.observe_every_step,
        final_observe=native.final_observe, interval=lambda s: SimpleNamespace(clear=lambda: None),
    )
    env.monitor = PutawayMonitor(scene, env.task.spec.resolved, native)
    env.physics_steps = 0
    env.max_physics_steps = round(env.task.spec.max_time * native.physics_hz)
    assert env.task.spec.max_time == (max_time or throughput.horizon(45.32))
    env._physics_step()
    env._advance(0)
    while not env.done:
        env._advance(env.native.control_stride)
    row = env.task.metrics()
    assert row["termination_reason"] == reason
    assert env.task.terminated == (reason != "horizon")
    assert env.task.time == pytest.approx(end, abs=0.021)
    assert row["task_success"] == row["joint_success"] == valid
    if valid:
        assert row["t_q_s"] == pytest.approx(0.524) and env.task.time == pytest.approx(end)


def test_kitting_final_check_needs_released_parts():
    """Review 2026-11-09: a credited part still pinched in the insert is not valid at the end."""
    from roboclock.envs.industry import scoring

    part = {
        "whole_contained": True, "bounds_min_m": [0.0, 0.0, 0.1005], "kit_support_N": 0.2,
        "kit_touching": True, "finger_forces_N": {"left": [0.0, 0.0], "right": [0.0, 0.0]},
    }
    assert scoring.KittingMonitor.placed(part, 0.1)
    held = {**part, "finger_forces_N": {"left": [0.4, 0.3], "right": [0.0, 0.0]}}
    assert not scoring.KittingMonitor.placed(held, 0.1)
    lifted = {**part, "bounds_min_m": [0.0, 0.0, 0.11]}
    assert not scoring.KittingMonitor.placed(lifted, 0.1)


def test_the_instruction_a_policy_gets_states_the_time_requirement():
    """The observation's instruction is all a policy is told: it must carry the level's
    window or deadline (as in the other domains) and the contact rule every task is judged
    by, and fit the observation space."""
    import string

    from roboclock.envs.industry.tasks import TASKS, judged_rules, make_task

    charset = set(string.printable + "–°×≤≥")  # IndustryEnv's `instruction` space
    for task_id in TASKS:
        texts = set()
        for level in ("easy", "medium", "hard", "reviewed_nominal"):
            for seed in (0, 2, 5):  # seed-drawn numbers (A1's recipe window) change the length
                task = make_task(task_id, level, seed)
                text = task.gen_instruction()
                assert text == (
                    f"{task.spec.instruction} {task.spec.time_requirement} "
                    f"{judged_rules(task_id, task.spec.resolved)}"
                )
                assert len(text) <= 1024 and set(text) <= charset
                if seed == 2:
                    texts.add(text)
        if TASKS[task_id].category != "throughput":
            assert len(texts) > 1  # an A or B level changes the numbers the policy reads
    assert "within 2.53 s of the warning" in make_task("B2", "hard", 2).gen_instruction()


def test_any_pan_of_a_colour_fills_its_pocket():
    """C4: the six feed pans of a colour are identical and the instruction names none, so the
    judge credits whichever is seated in the colour's pocket (it credited only the oracle's
    pan, index 0 or 2, until the 2026-11 audit), and each pan fills one pocket only."""
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.scoring import MONITORS

    scene = SimpleNamespace(
        pose=lambda body: (np.zeros(3), np.eye(3)), case={0: None, 1: None},
        payload_native_valid=lambda o, c, pans=None: False,
    )
    monitor = MONITORS[50](scene, TASKS[50].reviewed(), TASKS[50])
    seat = dict(contained=True, support_N=0.05, touching=True)

    def observe(t, seated, held=()):
        pans = {
            f"{color}_{j}": dict(
                position=[0.1 * color, 0.01 * j, 0.0], rotation=np.eye(3).tolist(),
                finger_forces_N=[1.0, 0.0] if (color, j) in held else [0.0, 0.0],
                cases={str(seated[color, j]): dict(seat)} if (color, j) in seated else {},
            )
            for color in range(4) for j in range(6)
        }
        return {"time_s": t, "pans": pans, "cases": {"0": {}, "1": {}}}

    # Rose pan 4 in the left case, champagne pans 3 and 0 both there, copper pan 5 still held.
    seated = {(1, 4): 0, (0, 3): 0, (0, 0): 0, (2, 5): 0}
    for step in range(30):
        monitor.tick(observe(1.0 + step / 50, seated, held={(2, 5)}))
    assert monitor.credited == {(0, 0), (0, 1)}
    assert monitor.pan_of == {(0, 0): 0, (0, 1): 4}  # the oracle's pan first where two qualify
    # The same rose pan cannot also fill the right case's pocket; another one can.
    for step in range(30):
        monitor.tick(observe(2.0 + step / 50, {(1, 4): 1}))
    assert (1, 1) not in monitor.credited
    for step in range(30):
        monitor.tick(observe(3.0 + step / 50, {(1, 1): 1}))
    assert monitor.pan_of[1, 1] == 1
    assert monitor.result(observe(4.0, {}))["pan_of"] == {"0_0": 0, "0_1": 4, "1_1": 1}


def test_a_firm_catch_inside_the_zone_is_an_arrest():
    """B3: a bearing stopped and held by the fingers inside the zone is arrested whether or not
    the lane still carries it (a two-finger catch lifts its weight off the lane)."""
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.scoring import MONITORS

    scene = SimpleNamespace(entry_at=8.7, events={}, tray_valid=lambda o: False, first_hand_at=8.9)
    monitor = MONITORS[46](scene, TASKS[46].reviewed(), TASKS[46])
    zone = monitor.resolved["zone_x_m"]
    for step in range(20):
        monitor.tick({
            "time_s": 9.0 + step / 50, "bearing_position": [sum(zone) / 2, 0.9, 0.95],
            "bearing_rotation": np.eye(3).tolist(), "lane_support_N": 0.0,
            "finger_forces_N": {"right": [16.0, 16.0], "left": [0.0, 0.0]},
            "bounds_min_m": [zone[0] + 0.05, 0.85, 0.9], "bounds_max_m": [zone[1] - 0.05, 0.95, 1.0],
        })
    assert monitor.events["response_s"] == pytest.approx(0.3 + monitor.arrest.dwell, abs=0.03)


def test_the_instruction_states_the_rules_a_policy_cannot_see():
    """Sweep v3 audit (2026-11-11): B1's closing direction and the C tasks' stall stop were
    judged but stated nowhere. C3 states its weight limit and scale rest (2026-11-19)."""
    from roboclock.envs.industry.tasks import make_task
    from roboclock.evaluation import throughput

    assert "closes counterclockwise seen from above" in make_task("B1", "easy", 0).gen_instruction()
    c3 = make_task("C3", "easy", 0).gen_instruction()
    assert "125 g or more is PASS" in c3 and "rested there alone for 0.50 s" in c3
    for task_id, steps in (("C1", 6), ("C2", 3), ("C3", 6), ("C4", 10)):
        t_o = throughput.oracle_time("industry", task_id)
        medium_limit = throughput.MULTIPLIER["medium"] * t_o
        rule = f"From {medium_limit:.0f} s on, the episode ends once {throughput.stall_window(t_o, steps):.0f} s pass"
        for level in ("easy", "hard", "reviewed_nominal"):
            assert rule in make_task(task_id, level, 1).gen_instruction()
    assert "the episode ends once" not in make_task("A1", "easy", 0).gen_instruction()


def test_the_instruction_states_what_the_arms_may_touch():
    """Sweep v2 (2026-11-10): 18 of 20 episodes had an arm contact that voided them and no
    instruction said what an arm may touch. The rule is now a force limit on the arm's links,
    fingers excepted, and every instruction states it."""
    from roboclock.envs.industry import scoring
    from roboclock.envs.industry.tasks import TASKS, make_task

    rule = f"Arm links other than fingers: no contact over {scoring.ARM_COLLISION_N:g} N."
    assert rule == "Arm links other than fingers: no contact over 20 N."
    assert make_task("B2", "easy", 0).gen_instruction().endswith(rule)
    for task_id in TASKS:
        assert make_task(task_id, "medium", 3).gen_instruction().count(rule) == 1


def test_an_arm_contact_voids_by_its_force_and_never_for_a_finger():
    from roboclock.envs.industry import scoring

    arm = "/World/ARX/right/Geometry/base_link/link1/link2/link3/link4/link5/link6"
    left = arm.replace("/right/", "/left/")
    limit = scoring.ARM_COLLISION_N
    # A finger on the bench, a fixture or a part: manipulation, whatever the force.
    for other in ("/World/TestBenchTop", "/World/Carriers/Carrier1", "/World/Feed/Kit0_pin/Offset/Asset/Mesh_0"):
        assert not scoring.contact_voids(f"{arm}/link7 | {other}", {"peak_force_N": 150.0})
    # Fingers of the two arms on each other (a handover).
    assert not scoring.contact_voids(f"{left}/link8 | {arm}/link7", {"peak_force_N": 60.0})
    # Any other link: a graze passes, an impact voids. The palm on the handle it turns (B1),
    # a link on its own mount plate, on the product, on the other arm, on the arm itself.
    for other in ("/World/Valve/handle", "/World/ARXMounts/rightPlate", "/World/Products/WorkingCover",
                  f"{left}/link8", arm.rsplit("/", 2)[0]):
        pair = " | ".join(sorted([arm, other]))
        assert not scoring.contact_voids(pair, {"peak_force_N": 0.3})
        assert not scoring.contact_voids(pair, {"peak_force_N": limit})
        assert scoring.contact_voids(pair, {"peak_force_N": limit + 0.1})
    # A2 and A1 keep their actors in the record; the key may name colliders.
    assert scoring.contact_voids("a | b", {"actors": [arm, "/World/Bath0/Tank"], "peak_force_N": 45.0})
    assert not scoring.contact_voids("a | b", {"actors": [f"{arm}/link8", "/World/Bath0/Tank"], "peak_force_N": 45.0})
    # The zero-impulse arm-to-arm log has no force and does not void by itself.
    assert not scoring.contact_voids("x | y", {"actors": [left, arm], "force_threshold_applied": False})
    # A record from before forces were kept: the old strict verdict.
    assert scoring.contact_voids(f"{arm} | /World/TestBenchTop", {"time_s": 1.0})
    assert not scoring.contact_voids(f"{arm}/link7 | /World/TestBenchTop", {"time_s": 1.0})
    records = {
        f"{arm} | /World/TestBenchTop": {"peak_force_N": 4.0},
        f"{arm}/link7 | /World/TestBenchTop": {"peak_force_N": 30.0},
        f"/World/Camera/D455 | {arm.rsplit('/', 2)[0]}": {"peak_force_N": 96.0},
    }
    assert list(scoring.voiding_contacts(records)) == [f"/World/Camera/D455 | {arm.rsplit('/', 2)[0]}"]
    assert scoring.arm_contact_peak(records) == 96.0
    assert scoring.arm_contact_peak({f"{arm}/link7 | /World/TestBenchTop": {"peak_force_N": 30.0}}) == 0.0


def test_rows_say_why_the_physical_verdict_is_false():
    """Industry rows had `unexpected_contacts` but no `physical_failures`, the field the
    cross-domain summary reads; a run whose only fault was a contact showed no reason."""
    from roboclock.envs.industry import scoring
    from roboclock.envs.industry.native.catalog import TASKS
    from roboclock.envs.industry.tasks import make_task

    arm = "/World/ARX/right/Geometry/base_link/link1/link2/link3/link4/link5/link6"
    assert scoring.physical_failures(True, {}) == []
    assert scoring.physical_failures(False, {}) == ["task_not_completed"]
    assert scoring.physical_failures(
        True, {f"{arm}/link7 | /World/TestBenchTop": {}, f"{arm} | /World/Valve/handle": {}}
    ) == ["arm_collision:right/link6|Valve/handle", "arm_collision:right/link7|TestBenchTop"]
    # C3's arm-to-arm keys name colliders below the links.
    left = arm.replace("/right/", "/left/")
    assert scoring.physical_failures(False, {f"{left}/collisions/mesh_0 | {arm}/link8/collisions": {}}) == [
        "task_not_completed", "arm_collision:left/link6|right/link8",
    ]

    # B2 seed 0 of the sweep: the motor stopped in time, the fingers touched the bench top.
    task = make_task("B2", "easy", 0)
    scene = _Scene(True, True)
    monitor = scoring.MotorMonitor(scene, task.spec.resolved, TASKS[45])
    o = {"time_s": 18.0, "rotor_speed_rad_s": 0.0}
    monitor.physics(o)
    task.record(monitor.outcome(o))
    assert task.metrics()["task_success"] and task.metrics()["physical_failures"] == []
    # Its fingers on the bench top no longer void it, however they pressed.
    contacts = [f"{arm}/link7 | /World/TestBenchTop", f"{arm}/link8 | /World/TestBenchTop"]
    for key in contacts:
        scene.contacts.unexpected[key] = {"time_s": 16.9, "peak_force_N": 12.0}
    task.record(monitor.outcome(o))
    row = {**task.metrics(), "environment": "industry"}
    assert validate_row(row) == []
    assert row["task_success"] and row["joint_success"]
    assert row["unexpected_contacts"] == contacts and row["collision_count"] == 2  # still listed
    assert row["arm_collisions"] == [] and row["physical_failures"] == []
    assert row["arm_contact_peak_N"] == 0.0
    # The forearm hitting the bench does: a graze is reported, an impact voids.
    graze = f"{arm.rsplit('/', 2)[0]} | /World/TestBenchTop"
    scene.contacts.unexpected[graze] = {"time_s": 17.0, "peak_force_N": 3.5}
    task.record(monitor.outcome(o))
    assert task.metrics()["task_success"] and task.metrics()["arm_contact_peak_N"] == 3.5
    scene.contacts.unexpected[graze]["peak_force_N"] = 96.0
    task.record(monitor.outcome(o))
    row = task.metrics()
    assert not row["task_success"] and row["temporal_success"]
    assert row["arm_collisions"] == [graze]
    assert row["physical_failures"] == ["arm_collision:right/link4|TestBenchTop"]
    # The scene's own predicate failing is reported too.
    task.record(scoring.MotorMonitor(_Scene(False, False), task.spec.resolved, TASKS[45]).outcome(o))
    assert task.metrics()["physical_failures"] == ["task_not_completed"]
    assert make_task("B2", "easy", 0).metrics()["physical_failures"] == []  # nothing recorded yet


# ---------------------------------------------------------------- release read on both arms
_ARM = "/World/ARX/{side}/Geometry/base_link/link1/link2/link3/link4/link5/link6"
_LINKS = {f"link{i}": f"Geometry/base_link/link1/link2/link3/link4/link5/link6/link{i}" for i in (7, 8)}


def _fake_dual(jaws):
    """Dual-arm stand-in: `jaws[side]` are that arm's two finger sliders (m)."""
    names = [f"joint{i}" for i in range(1, 9)]
    return SimpleNamespace(
        robots={
            side: SimpleNamespace(get_joint_positions=lambda side=side: np.r_[np.zeros(6), jaws[side]])
            for side in ("left", "right")
        },
        names={"left": names, "right": names},
        manifest={"relative_body_paths": _LINKS},
        tool_world=lambda side: (np.zeros(3), np.eye(3)),
    )


def test_release_helpers_judge_the_arm_that_held_the_product():
    from roboclock.envs.industry.native.arx_cell_task import ContactBook, finger_side, jaws_open

    assert finger_side(_ARM.format(side="right") + "/link8") == "right"
    assert finger_side(_ARM.format(side="left")) is None  # link6, the palm
    assert finger_side("/World/Pump") is None
    jaws = {"left": [0.0, 0.0], "right": [0.044, 0.042]}
    assert jaws_open(jaws, set())  # nothing held yet
    assert jaws_open(jaws, {"right"}) and not jaws_open(jaws, {"left"})
    assert not jaws_open(jaws, {"left", "right"})
    assert not jaws_open({"left": [0.044, 0.041], "right": [0.0, 0.0]}, {"left"})  # past 41 mm, each

    book = object.__new__(ContactBook)  # the constructor subscribes to PhysX
    book.pairs = {
        f"{_ARM.format(side='right')}/link7 | /World/Products/WorkingCover": 3.0,
        f"{_ARM.format(side='left')} | /World/Products/WorkingCover": 4.0,  # palm: not a finger
        f"{_ARM.format(side='left')}/link8 | /World/Input/LidPallet2": 5.0,  # not the product
    }
    assert book.finger_sides("/World/Products/WorkingCover") == {"right"}


@pytest.mark.parametrize(
    ("held_by", "jaws", "released"),
    [
        ({"left"}, {"left": [0.044, 0.044], "right": [0.0, 0.0]}, True),    # the recorded oracle
        ({"left"}, {"left": [0.030, 0.030], "right": [0.044, 0.044]}, False),
        ({"right"}, {"left": [0.0, 0.0], "right": [0.044, 0.044]}, True),   # was False: left jaws read
        ({"right"}, {"left": [0.044, 0.044], "right": [0.030, 0.030]}, False),  # was True
    ],
)
def test_rotary_release_is_judged_on_the_arm_that_placed_the_housing(held_by, jaws, released):
    """A2 read the left arm's jaws whichever arm placed the housing (brief: judging must not
    depend on the arm). The scene's own observe() runs here on stand-in bodies."""
    from roboclock.envs.industry.native.rotary_task import RotaryScene

    rotor_p = np.array([0.0, 0.55, 0.9])
    part_p = rotor_p + [0.0, -0.28, 0.05]
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    scene = SimpleNamespace(
        part=SimpleNamespace(get_world_pose=lambda: (part_p, identity), get_linear_velocity=lambda: np.zeros(3)),
        rotor=SimpleNamespace(get_world_pose=lambda: (rotor_p, identity), get_angular_velocity=lambda: np.zeros(3)),
        resolved={"loading_sector_deg": [-120.0, -60.0]},
        dual=_fake_dual(jaws),
        outer_vertices=np.array([[0.01, 0.01, 0.0], [-0.01, -0.01, 0.02]]),
        interval={"carrier": 1.0},
        finger_sides=set(held_by),
        world=SimpleNamespace(current_time=30.0),
        events={"bilateral_grasp_s": 20.0, "source_cleared_s": 22.0},
        retained_since=None,
        start_s=0.0,
        unexpected={},
        success=False,
    )
    o = RotaryScene.observe(scene)
    assert o["supported"] and o["released"] is released
    scene.world.current_time = 30.4
    RotaryScene.observe(scene)
    assert ("supported_release_s" in scene.events) is released
    # A finger still pressing on the housing is not a release, whichever arm it is.
    scene.interval["finger8"] = 0.5
    assert not RotaryScene.observe(scene)["released"]


@pytest.mark.parametrize(
    ("held_by", "jaws", "touching", "released"),
    [
        ("left", {"left": [0.044, 0.044], "right": [0.0, 0.0]}, None, True),   # the recorded oracle
        ("left", {"left": [0.030, 0.030], "right": [0.044, 0.044]}, None, False),
        ("right", {"left": [0.0, 0.0], "right": [0.044, 0.044]}, None, True),  # was False
        ("right", {"left": [0.044, 0.044], "right": [0.030, 0.030]}, None, False),  # was True
        ("right", {"left": [0.044, 0.044], "right": [0.044, 0.044]}, "right", False),  # was True
        ("left", {"left": [0.044, 0.044], "right": [0.044, 0.044]}, "left", False),
    ],
)
def test_bonding_release_is_judged_on_the_arm_that_seated_the_cover(held_by, jaws, touching, released):
    """A3 read the left arm's jaws and finger touches only."""
    from roboclock.envs.industry.native import bonding_task
    from roboclock.envs.industry.native.arx_cell_task import ContactBook

    lid, housing = bonding_task.LID, bonding_task.HOUSING
    finger = _ARM.format(side=held_by) + "/link7"
    book = object.__new__(ContactBook)
    book.pairs = {" | ".join(sorted([finger, lid])): 12.0}
    book.interval = {f"{housing}/Mesh | {lid}": 2.0}
    book.active = {"seat": dict(actors=[lid, housing + "/Mesh"], active=True)}
    if touching:
        actor = f"/World/ARX/{touching}/" + _LINKS["link8"]
        book.active["finger"] = dict(actors=[actor, lid], active=True)
    book.unexpected = {}
    still = SimpleNamespace(
        get_world_pose=lambda: (np.array([0.1, 0.2, 0.9]), np.array([1.0, 0.0, 0.0, 0.0])),
        get_linear_velocity=lambda: np.zeros(3),
        get_angular_velocity=lambda: np.zeros(3),
    )
    scene = SimpleNamespace(
        part=still,
        lid_mean=np.zeros(3),
        seat=np.array([0.1, 0.2, 0.9]),
        envelope=np.array([[0.05, 0.08, 0.0], [-0.05, -0.08, 0.01]]),
        contacts=book,
        dual=_fake_dual(jaws),
        world=SimpleNamespace(current_time=40.0),
        events={},
        resolved={"window_relative_s": [2.0, 14.0]},
        retained_since=None,
        physical_completed=False,
        success=False,
        dispenser=SimpleNamespace(get_joint_positions=lambda: np.zeros(2)),
        axes=[0, 1],
    )
    o = bonding_task.BondingScene.observe(scene)
    assert o["aligned"] and o["stable"] and o["support_contact_active"]
    assert o["released"] is released and o["physical_seating"] is released
    assert o["jaws_m"] == list(jaws["left"])  # the oracle's own reading stays the left arm's


def test_grasp_and_release_instructions_give_the_judged_numbers():
    """A2 and A3 credit the grasp by finger force and the release by jaw opening."""
    from roboclock.envs.industry import scoring
    from roboclock.envs.industry.native.arx_cell_task import RELEASED_JAW_M
    from roboclock.envs.industry.tasks import TASKS, make_task

    assert RELEASED_JAW_M == 0.041 and scoring.GraspEvents.DWELL_S == 0.25
    forces = {"A2": scoring.RotaryMonitor.grasp_force_n, "A3": scoring.BondingMonitor.grasp_force_n}
    assert forces == {"A2": 10.0, "A3": 8.0}
    for task_id in TASKS:
        text = make_task(task_id, "easy", 0).gen_instruction()
        if task_id in forces:
            assert text.endswith(
                f"The grasp counts with over {forces[task_id]:g} N on both fingers for 0.25 s, "
                "the release once each finger is open past 41 mm."
            )
        else:
            assert "The grasp counts" not in text and "open past" not in text
    # The monitors judge with the stated force: just above it for longer than the dwell.
    grasp = scoring.GraspEvents(forces["A2"])
    assert not grasp.update(0.0, [10.5, 10.5]) and not grasp.update(0.2, [10.5, 10.5])
    assert grasp.update(0.3, [10.5, 10.5])
    assert not scoring.GraspEvents(forces["A2"]).update(0.0, [10.5, 9.5])


# ---------------------------------------------------------------- B oracles wait for the cue
class _HeldArm:
    """Stands in for ``ArmMotion``: records tool-space moves, solves IK to a fixed posture."""

    REACH = np.array([0.0, 1.8, 1.0, -0.8, 0.0, 1.5])

    def __init__(self, dual, side="left"):
        self.dual, self.side = dual, side
        self.ids = list(range(6))
        self.kin = SimpleNamespace(
            arm_names=[f"joint{i}" for i in range(1, 7)],
            solve=lambda *args, **kwargs: (self.REACH.copy(), 0.0),
        )
        self.command_speed_rad_s = 0.70
        self.position_correction_limit_m = 0.010
        self.command, self.correction = np.zeros(3), np.zeros(3)
        self.R = np.eye(3)
        self.moves = dual.moves

    def move(self, goal, speed=0.055, R=None):
        self.moves.append(self.side)
        return False


def _held_dual(jaws):
    """Both arms at the default reset posture with the task's reset jaw openings."""
    names = [f"joint{i}" for i in range(1, 9)]
    reset = {
        side: np.array([sign * 0.8, 1.3789, 1.242, -1.4339, 0.0, sign * 0.8, jaws[side], jaws[side]])
        for side, sign in (("left", 1.0), ("right", -1.0))
    }
    dual = SimpleNamespace(
        names={side: names for side in reset},
        commands={side: q.copy() for side, q in reset.items()},
        manifest={"profile": {"arm_lower_rad": [-3.0] * 6, "arm_upper_rad": [3.0] * 6}},
        moves=[],
        tool_world=lambda side: (np.array([0.3, 0.35, 1.27]), np.eye(3)),
        body_world=lambda side, name: (np.zeros(3), np.eye(3)),
    )

    def jaw_travel(side, travel):
        dual.commands[side][6:] = travel

    dual.jaw_travel = jaw_travel
    return dual, reset


def _b1_oracle(monkeypatch, mode):
    from roboclock.envs.industry.native import isolation_task

    monkeypatch.setattr(isolation_task, "ArmMotion", _HeldArm)
    dual, reset = _held_dual({"left": 0.04, "right": 0.027})
    scene = SimpleNamespace(
        dual=dual, events={}, physical_completed=False, world=SimpleNamespace(current_time=0.0),
        grasp=lambda angle: (np.array([0.16, 0.5, 1.082]), np.eye(3)),
    )
    observation = dict(angle_rad=0.0, unexpected_contacts={}, closed=False, finger_forces_N=[0.0, 0.0], jaws_m=[0.027, 0.027])
    return isolation_task.IsolationController(scene, mode), scene, reset, observation, "leak_onset_s", 5.5


def _b2_oracle(monkeypatch, mode):
    from roboclock.envs.industry.native import motor_arx_task

    monkeypatch.setattr(motor_arx_task, "ArmMotion", _HeldArm)
    dual, reset = _held_dual({"left": 0.04, "right": 0.0})
    scene = SimpleNamespace(
        dual=dual, events={}, physical_completed=False, world=SimpleNamespace(current_time=0.0),
        asset_manifest={"button": {"cap_front_offset_m": 0.02}},
        button_position=np.array([-0.48, 0.3, 1.04]), axis=np.array([0.0, 1.0, 0.0]),
        finger_points={name: np.array([[0.05, 0.0, 0.0], [0.0, 0.01, 0.0]]) for name in ("link7", "link8")},
        final_valid=lambda o: True,
    )
    observation = dict(unexpected_contacts={}, finger_front_m=0.05)
    return motor_arx_task.MotorARXController(scene, mode), scene, reset, observation, "warning_onset_s", 3.5


def _b3_oracle(monkeypatch, mode):
    from roboclock.envs.industry.native import bearing_handoff

    monkeypatch.setattr(bearing_handoff, "ArmMotion", _HeldArm)
    dual, reset = _held_dual({"left": 0.04, "right": 0.02})
    scene = SimpleNamespace(
        dual=dual, events={}, entry_at=None, first_hand_at=None, world=SimpleNamespace(current_time=0.0),
        resolved={"lane_y_m": 0.91, "zone_x_m": [0.9, 1.2], "capture_minimum_command_m": 0.0002},
        surface_z=lambda x: 0.93,
    )
    observation = dict(
        unexpected_contacts={}, bounds_min_m=[0.5, 0.87, 0.9], bounds_max_m=[0.57, 0.95, 1.0],
        finger_forces_N={"left": [0.0, 0.0], "right": [0.0, 0.0]}, lane_support_N=3.0,
        bearing_position=[0.53, 0.91, 0.95], bearing_rotation=np.eye(3).tolist(),
    )
    return bearing_handoff.BearingARXController(scene, mode), scene, reset, observation, "zone_entry_s", 3.3


@pytest.mark.parametrize("build", [_b1_oracle, _b2_oracle, _b3_oracle], ids=["B1", "B2", "B3"])
@pytest.mark.parametrize("mode", ["nominal", "late"])
def test_reactive_oracles_hold_the_reset_pose_until_the_cue(monkeypatch, build, mode):
    """Before the leak / warning / zone entry is in the scene, a B oracle writes no command:
    the arms stay at the pose the environment reset them to (a policy's start pose). It moves
    from the first control tick that observes the cue; the ``late`` violator waits past the
    deadline first."""
    controller, scene, reset, observation, cue, late_delay = build(monkeypatch, mode)
    dual = scene.dual

    def step(t):
        scene.world.current_time = t
        controller.step({**observation, "time_s": t})

    def held():
        return not dual.moves and all(np.array_equal(dual.commands[side], reset[side]) for side in reset)

    for t in np.arange(0.0, 14.0, 0.02):
        step(t)
    assert controller.phase == "standby" and controller.failure is None and held()
    # The cue: the scene's event, as its observe() records it.
    onset = 14.0
    scene.events[cue] = onset
    if cue == "zone_entry_s":
        scene.entry_at = onset
    if mode == "late":
        step(onset + min(late_delay, 3.0) - 0.1)
        assert controller.phase == "standby" and held()
        onset += late_delay + 0.02
    step(onset)
    if (mode, cue) == ("late", "zone_entry_s"):
        # B3's late violator never brakes: the controller's own 3 s check ends it first.
        assert controller.failure == "3s arrest deadline expired" and held()
        return
    assert controller.phase != "standby" and controller.failure is None
    assert not held()
    if cue != "zone_entry_s":
        # One control tick of the reach: every joint moves at most ArmMotion's command rate.
        moved = np.abs(dual.commands["right"][:6] - reset["right"][:6])
        assert 0 < moved.max() <= 0.70 / CONTROL_HZ + 1e-12
        assert np.array_equal(dual.commands["left"], reset["left"])
    else:
        assert dual.moves == ["right"]  # the brake hand starts servoing only now
