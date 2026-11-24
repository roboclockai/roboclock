"""Unified sports runtime (``--env sports``): catalog, joint layout, scoring and dispatch.

These checks run without Isaac Sim. A stand-in adapter replaces the per-task
Isaac backends so the environment's reset/step/latency logic and the shared
evaluator can be exercised end to end; recorded native results provide the
scoring inputs.
"""

import json
import string
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from roboclock.cli import main
from roboclock.envs import robodojo_rig
from roboclock.envs.lab.robot import JOINT_HIGH, JOINT_LOW
from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ, check_rates
from roboclock.envs.registry import get_environment, import_symbol
from roboclock.envs.sports import adapters, head_view, rig_cameras
from roboclock.envs.sports._shared.workflow.backend import at_control_rate
from roboclock.envs.sports.adapters.base import Adapter
from roboclock.envs.sports.config import DEFAULT_CONFIG, load_config
from roboclock.envs.sports.env import SportsBenchmarkEnv
from roboclock.envs.sports.joints import JOINT_ORDER, NATIVE_NAMES, JointMap, joint_step_speed
from roboclock.envs.sports.results import normalize
from roboclock.envs.sports.tasks import CATALOG, TASKS, make_task, native_config
from roboclock.evaluation.evaluator import episode_plan, evaluate
from roboclock.evaluation.schema import CATEGORIES, validate_row

REPO = Path(__file__).resolve().parents[3]
RECORDINGS = REPO / "data/sports/videos"
IDS = ("A1", "A2", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3")
# The showcase recordings keep the IDs they were made with: taiko A2, basketball A3 (renumbered 2026-11-19).
RECORDED_IDS = {"A2": "B5", "A3": "A2"}
RECORDED = {
    RECORDED_IDS.get(path.stem.split("_")[0], path.stem.split("_")[0]): path
    for path in RECORDINGS.glob("[ABC][0-9]_*.json")
}
# PhysX order of the manufacturer X5A pair (recorded native_joint_names).
NATIVE_ORDER = [f"{side}_joint{i}" for i in range(1, 7) for side in ("left", "right")] + [
    "left_joint7",
    "left_joint8",
    "right_joint7",
    "right_joint8",
]
CHARSET = set(string.printable + "–°")
NO_RENDER = DEFAULT_CONFIG.parent / "no_render.yaml"


def test_sports_registers_the_current_ten_tasks():
    spec = get_environment("sports")
    assert tuple(import_symbol(spec.task_registry)) == IDS
    assert spec.result_adapter == "roboclock.envs.sports.results:normalize"
    assert import_symbol(spec.factory) is SportsBenchmarkEnv
    assert SportsBenchmarkEnv.environment_id == "sports"
    for task_id, cls in TASKS.items():
        assert cls.category in CATEGORIES
        assert (
            cls.category
            == {"A": "temporal_window", "B": "event_reactive", "C": "throughput"}[task_id[0]]
        )


def test_single_franka_environments_are_not_registered():
    # sports_legacy (single-Franka tasks 31-40) and the standalone Franka chess env were removed:
    # every domain uses only the Dual ARX X5.
    from roboclock.envs.registry import environment_names

    assert not {"sports_legacy", "chess"} & set(environment_names())


def test_runtime_modules_import_without_the_simulator():
    script = (
        "import sys\n"
        "import roboclock.envs.sports.env, roboclock.envs.sports.oracle\n"
        "import roboclock.envs.sports.adapters.base, roboclock.envs.sports.adapters.workflow\n"
        "import roboclock.envs.sports.adapters.arcade, roboclock.envs.sports.adapters.chess\n"
        "from roboclock.envs.sports.tasks import make_task\n"
        "[make_task(t, 'hard', 3) for t in ('A1','A2','B1','B2','B3','B4','B5','C1','C2','C3')]\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('isaacsim', 'omni', 'pxr')]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("task_id", IDS)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_episode_specs_use_the_task_rates(task_id, difficulty):
    config = load_config()
    task = make_task(task_id, difficulty, 1, config)
    rates = config["tasks"][task_id]
    assert task.spec.task_id == task_id and task.spec.seed == 1
    assert task.spec.max_time > 0
    assert round(1 / task.spec.physics_dt) == rates["physics_hz"]
    assert round(1 / (task.spec.physics_dt * task.spec.control_decimation)) == rates["control_hz"]
    native = task.native_config
    assert native["sim"]["physics_dt"] == pytest.approx(task.spec.physics_dt)
    assert native["sim"]["control_decimation"] == task.spec.control_decimation
    assert native["embodiment"]["model"] == "dual_arx_x5"
    assert native["sim"]["camera_resolution"] == config["sim"]["camera_resolution"]
    assert set(task.gen_instruction()) <= CHARSET and len(task.gen_instruction()) <= 512


def test_every_task_runs_on_the_shared_time_base(tmp_path):
    config = load_config()
    assert set(config["tasks"]) == set(IDS)
    for task_id in IDS:
        assert config["tasks"][task_id] == {"physics_hz": PHYSICS_HZ, "control_hz": CONTROL_HZ}
        native = native_config(task_id)
        check_rates(native["sim"]["physics_dt"], native["sim"]["control_decimation"])
    path = tmp_path / "rates.yaml"
    config["tasks"]["A1"]["physics_hz"] = 240
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="A1: physics"):
        load_config(path)


def test_joint_speed_limits_keep_the_measured_arm_speed():
    # Drive probe: limit 2.4 at 30 Hz and 3.40 at 50 Hz both reach 1.49 rad/s.
    assert joint_step_speed(2.4, 30, 1 / 50) == pytest.approx(3.40, abs=0.01)
    assert joint_step_speed(5.2, 60, 1 / 50) == pytest.approx(4.52, abs=0.01)
    assert joint_step_speed(2.4, 50, 1 / 50) == pytest.approx(2.4)
    config = native_config("B4")
    assert at_control_rate(config)["freeze"]["joint_speed"] == pytest.approx(3.40, abs=0.01)
    assert config["freeze"]["joint_speed"] == 2.4
    assert at_control_rate(native_config("B2"))["service"]["joint_speed"] == pytest.approx(
        2.41, abs=0.01
    )
    assert at_control_rate(native_config("C1"))["pool"]["joint_speed"] == 2.5


def test_seed_changes_randomized_episodes_only_through_native_configs():
    assert make_task("A2", "easy", 0).spec.timing != make_task("A2", "easy", 1).spec.timing
    # C-class (score_version 2): batch tasks run the same episode at every difficulty; the
    # stream task C3 keeps its per-difficulty schedule.
    assert make_task("C1", "easy", 0).spec.max_time == make_task("C1", "hard", 0).spec.max_time
    assert make_task("C3", "easy", 0).spec.max_time > make_task("C3", "hard", 0).spec.max_time


def test_joint_map_is_a_named_permutation():
    joints = JointMap(NATIVE_ORDER)
    native = np.arange(16, dtype=np.float32) / 100
    unified = joints.to_unified(native)
    for k, name in enumerate(NATIVE_NAMES):
        assert unified[k] == native[NATIVE_ORDER.index(name)]
    back = joints.to_native(unified, np.zeros(16, dtype=np.float32))
    assert back.dtype == np.float32 and np.array_equal(back, native)
    assert list(joints.describe()) == list(JOINT_ORDER)
    assert joints.describe()["right/joint7"] == "right_joint7"
    with pytest.raises(ValueError):
        JointMap(NATIVE_ORDER[:-1])


def test_observed_joints_are_valid_hold_actions():
    joints = JointMap(NATIVE_ORDER)
    native = np.zeros(16, dtype=np.float32)
    native[NATIVE_ORDER.index("left_joint7")] = -2e-6
    native[NATIVE_ORDER.index("right_joint8")] = 0.04400004
    observed = joints.observe(native)
    low, high = np.tile(JOINT_LOW, 2), np.tile(JOINT_HIGH, 2)
    assert np.all(observed >= low) and np.all(observed <= high)


@pytest.mark.parametrize("task_id", sorted(RECORDED))
def test_recorded_native_results_score_as_valid_rows(task_id):
    """The unified scorer reproduces the recorded outcome from each native result."""
    document = json.loads(RECORDED[task_id].read_text())
    task = make_task(task_id, document.get("difficulty", "easy"), int(document.get("seed", 0)))
    task.bind(Replay(document))
    row = task.metrics()
    assert validate_row({**row, "environment": "sports"}) == []
    reference = normalize(document)
    assert row["task_id"] == task_id and row["category"] == TASKS[task_id].category
    for key in ("task_success", "temporal_success", "joint_success"):
        assert row[key] == reference[key], key
    json.dumps(row, allow_nan=False)


def test_arcade_rows_use_current_task_ids():
    document = json.loads(RECORDED["B5"].read_text())
    row = normalize(document)
    assert row["task_id"] == "B5" and row["legacy_task_id"] == "32"
    assert normalize({**document, "configuration": "unparsed"})["task_id"] == "B5"


class Replay:
    """Adapter stand-in that only reports a stored native result."""

    ticks = 0
    time = 1.0
    done = True

    def __init__(self, document):
        self.document = document

    def result(self):
        return self.document

    def events(self):
        return self.document.get("events", [])

    def joint_success(self):
        return bool(normalize(self.document)["joint_success"])


class FakeRig:
    def __init__(self):
        self.robot = self
        self.dof_names = NATIVE_ORDER
        self.command = np.zeros(16, dtype=np.float32)
        self.command[
            [NATIVE_ORDER.index(f"{s}_joint{i}") for s in ("left", "right") for i in (7, 8)]
        ] = 0.04

    def get_joint_positions(self):
        return self.command.copy()

    def pose(self, side):
        return np.zeros(3), np.array([1.0, 0, 0, 0])

    def width(self, side):
        return 0.08


class FakeAdapter(Adapter):
    """Advances a clock instead of PhysX; scores with the recorded native result."""

    built = []
    closed = []

    def build(self):
        self.rig_ = FakeRig()
        self.clock = 0.0
        self.applied = []
        self.after_build()
        FakeAdapter.built.append(self.spec.task_id)

    @property
    def rig(self):
        return self.rig_

    def rigs(self):
        return []  # no PhysX articulation to receive the finger speed limit

    @property
    def time(self):
        return self.clock

    @property
    def done(self):
        return self.clock >= 0.5

    def apply(self, native):
        self.applied.append(np.array(native, copy=True))
        self.rig_.command = np.array(native, copy=True)

    def tick(self):
        self.clock += self.physics_dt
        self.ticks += 1

    def capture_images(self, depth=False):
        return rig_cameras.blank_images(depth)

    def result(self):
        return json.loads(RECORDED[self.spec.task_id].read_text())

    def events(self):
        return [{"time": self.clock, "event": "tick"}]

    def state(self):
        return {}

    def close(self):
        FakeAdapter.closed.append(self.spec.task_id)


RIG_CHECK = {"check_rig": {"midpoint": [0.08, 0.0, 1.0], "yaw_deg": 0.0}, "separation_m": 0.6}


class StubCams:
    """Stand-in for the Isaac rig cameras (``rig_cameras.RigCameras``)."""

    measured = RIG_CHECK

    def __init__(self, backend):
        self.cameras = {view: None for view in rig_cameras.VIEWS}

    def intrinsics(self):
        return {view: {"resolution": [640, 480]} for view in rig_cameras.VIEWS}


def _simulator_free(monkeypatch, *, cams):
    """Replace the stage/renderer touch points of ``Adapter.after_build``."""
    from roboclock.envs.sports.adapters import base

    monkeypatch.setattr(rig_cameras, "ensure", cams)
    monkeypatch.setattr(rig_cameras, "apply_renderer", lambda: {"render_mode": "RaytracedLighting"})
    monkeypatch.setattr(rig_cameras, "renderer_state", lambda: {"render_mode": "RaytracedLighting"})
    monkeypatch.setattr(
        rig_cameras, "measure_and_check", lambda rig, config, cameras=True: dict(RIG_CHECK)
    )
    monkeypatch.setattr(rig_cameras, "disable_updates", lambda camera: False)
    monkeypatch.setattr(head_view, "apply", lambda b, task_id: [])
    monkeypatch.setattr(
        head_view, "check", lambda b, task_id, config=None: {"stub": {"margin_px": 50.0, "inside": True}}
    )
    monkeypatch.setattr(
        base, "measure_physics_rate", lambda backend: {"physics_hz": 250.0, "physics_dt": 0.004}
    )


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(adapters, "adapter_class", lambda task_id: FakeAdapter)
    _simulator_free(monkeypatch, cams=StubCams)
    FakeAdapter.built.clear()
    FakeAdapter.closed.clear()
    env = SportsBenchmarkEnv("A2", "easy", runtime=object())
    yield env
    env.close()


class BlankAdapter(FakeAdapter):
    """Uses the base observation path, so ``sim.no_render`` decides what is captured."""

    capture_images = Adapter.capture_images


@pytest.fixture
def no_render_env(monkeypatch):
    def ensure(*args, **kwargs):
        raise AssertionError("sim.no_render must not build cameras")

    monkeypatch.setattr(adapters, "adapter_class", lambda task_id: BlankAdapter)
    _simulator_free(monkeypatch, cams=ensure)
    monkeypatch.setattr(
        rig_cameras, "apply_renderer", lambda: (_ for _ in ()).throw(AssertionError("render"))
    )
    env = SportsBenchmarkEnv("B4", "easy", runtime=object(), config_path=NO_RENDER)
    yield env
    env.close()


def test_no_render_config_only_switches_rendering_off():
    rendered, blank = load_config(), load_config(NO_RENDER)
    assert rendered["sim"]["no_render"] is False and blank["sim"]["no_render"] is True
    blank["sim"]["no_render"] = False
    assert blank == rendered


def test_rig_cameras_are_the_policy_views(fake_env):
    obs, _ = fake_env.reset(seed=0, options={"task_id": "B2"})
    assert isinstance(fake_env.adapter.cams, StubCams)
    for key in ("rgb", "left_wrist_rgb", "right_wrist_rgb"):
        assert obs[key].shape == (480, 640, 3) and obs[key].dtype == np.uint8
    assert fake_env.observation_space.contains(obs)
    metadata = fake_env.episode_metadata()
    assert metadata["observation_rendered"] and metadata["render"] == "on_read"
    assert metadata["robodojo_rig"]["robodojo_rig"] == json.loads(json.dumps(robodojo_rig.describe()))
    assert metadata["robodojo_rig"]["rig_check"] == RIG_CHECK
    assert set(metadata["robodojo_rig"]["cameras"]) == set(rig_cameras.VIEWS)
    assert metadata["robodojo_rig"]["renderer"]["render_mode"] == "RaytracedLighting"
    assert metadata["head_view"]["margin_required_px"] == 10


def test_no_render_skips_the_cameras_and_all_rendering(no_render_env):
    obs, _ = no_render_env.reset(seed=0)
    assert no_render_env.observation_space.contains(obs)
    for key in ("rgb", "left_wrist_rgb", "right_wrist_rgb"):
        assert obs[key].shape == (480, 640, 3) and not obs[key].any()
    with pytest.raises(RuntimeError, match="no_render"):
        no_render_env.adapter.camera
    action = obs["joint_positions"].copy()
    while not no_render_env.done:
        obs, *_ = no_render_env.step(action)
    assert not obs["rgb"].any() and not obs["right_wrist_rgb"].any()
    metadata = no_render_env.episode_metadata()
    assert metadata["observation_rendered"] is False and metadata["renders"] == 0
    rig = metadata["robodojo_rig"]
    assert rig["rendered"] is False and rig["cameras"] is None and rig["renderer"] is None
    assert rig["rig_check"]["separation_m"] == 0.6
    assert rig["robodojo_rig"]["head_pitch_down_deg"] == 60.0


def test_no_render_depth_returns_zero_depth_images(monkeypatch):
    monkeypatch.setattr(adapters, "adapter_class", lambda task_id: BlankAdapter)
    _simulator_free(monkeypatch, cams=None)
    env = SportsBenchmarkEnv("B4", runtime=object(), config_path=NO_RENDER, depth=True)
    obs, _ = env.reset(seed=0)
    for key in ("depth", "left_wrist_depth", "right_wrist_depth"):
        assert obs[key].shape == (480, 640) and obs[key].dtype == np.float32 and not obs[key].any()
    assert env.observation_space.contains(obs)
    env.close()


def test_capture_renders_the_current_tick_once(monkeypatch):
    """Images render when an observation is read: once per physics tick, on any tick."""
    import sys
    import types

    calls = []
    rendering = types.ModuleType("roboclock.simulators.isaac.rendering")
    rendering.capture_camera = lambda subframes=1: calls.append("capture")
    monkeypatch.setitem(sys.modules, "roboclock.simulators.isaac.rendering", rendering)
    adapter = Adapter.__new__(Adapter)
    adapter.no_render, adapter.ticks, adapter._images, adapter._image_tick = False, 0, None, None
    adapter.settings = {"observation": {"render_updates": 1, "rt_subframes": 1}}
    world = types.SimpleNamespace(current_time=0.0, render=lambda: calls.append("render"))
    adapter.backend = types.SimpleNamespace(world=world)
    adapter.cams = types.SimpleNamespace(images=lambda depth=False: rig_cameras.blank_images(depth))
    adapter.capture_images()
    adapter.capture_images()
    assert calls == ["render", "capture"]
    adapter.ticks = 7  # off the 25 Hz grid
    assert adapter.capture_images()["rgb"].shape == (480, 640, 3)
    assert calls == ["render", "capture"] * 2 and adapter.renders == 2


def test_spaces_follow_the_shared_dual_x5_contract(fake_env):
    assert fake_env.action_space.shape == (16,)
    assert np.array_equal(fake_env.action_space.low, np.tile(JOINT_LOW, 2))
    assert np.array_equal(fake_env.action_space.high, np.tile(JOINT_HIGH, 2))
    shapes = {k: v.shape for k, v in fake_env.observation_space.spaces.items()}
    assert shapes["joint_positions"] == (16,) and shapes["ee_pose"] == (14,)
    assert shapes["gripper_width"] == (2,) and shapes["sim_time"] == (1,)
    with pytest.raises(RuntimeError):
        fake_env.step(np.zeros(16))
    json.dumps(fake_env.config, allow_nan=False)
    assert fake_env.config["sim"]["physics_dt"] == pytest.approx(1 / PHYSICS_HZ)


def test_reset_switches_tasks_and_rates_inside_one_runtime(fake_env):
    obs, info = fake_env.reset(seed=0, options={"task_id": "B5", "difficulty": "medium"})
    assert info == {}
    assert fake_env.observation_space.contains(obs)
    assert fake_env.task.task_id == "B5" and fake_env.task.spec.difficulty == "medium"
    assert fake_env.config["sim"]["physics_dt"] == pytest.approx(1 / PHYSICS_HZ)
    fake_env.reset(seed=1, options={"task_id": "C1"})
    assert FakeAdapter.built == ["B5", "C1"] and FakeAdapter.closed == ["B5"]
    assert fake_env.adapter.decimation == PHYSICS_HZ // CONTROL_HZ
    assert fake_env.episode_metadata()["joint_mapping"]["left/joint1"] == "left_joint1"


def test_step_validates_maps_and_reports_termination(fake_env):
    obs, _ = fake_env.reset(seed=0, options={"task_id": "B2"})
    with pytest.raises(ValueError):
        fake_env.step(np.zeros(15))
    with pytest.raises(ValueError):
        fake_env.step(np.full(16, np.nan))
    too_far = obs["joint_positions"].copy()
    too_far[0] = 3.0
    with pytest.raises(ValueError, match="limits"):
        fake_env.step(too_far)
    action = obs["joint_positions"].copy()
    action[JOINT_ORDER.index("right/joint2")] = 1.25
    obs, reward, terminated, truncated, _ = fake_env.step(action)
    native = fake_env.adapter.applied[-1]
    assert native[NATIVE_ORDER.index("right_joint2")] == np.float32(1.25)
    assert fake_env.adapter.ticks == 5 and obs["sim_time"][0] == pytest.approx(5 / PHYSICS_HZ)
    fake_env.advance_latency(0.1)
    assert fake_env.adapter.ticks == 5 + 25
    while not fake_env.done:
        obs, reward, terminated, truncated, _ = fake_env.step(action)
    assert terminated and not truncated and reward == 1.0


def test_evaluator_writes_schema_rows_across_tasks(fake_env, tmp_path):
    plan = episode_plan(["A2", "C3", "B4"], ["easy"], [0], [0.0, 0.05], environment="sports")
    summary = evaluate(fake_env, plan, "hold", tmp_path / "run")
    rows = [json.loads(line) for line in (tmp_path / "run/episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and summary["overall"]["episodes"] == 6
    for row in rows:
        assert validate_row(row) == []
        assert row["environment"] == "sports" and row["robot"] == "dual_arx_x5"
        assert row["joint_order"] == list(JOINT_ORDER)
    assert main(["summarize", "--normalize", str(tmp_path / "run")]) == 0


def test_oracle_maps_native_controller_commands(fake_env):
    from roboclock.envs.sports.oracle import SportsOracle

    class Controller:
        def act(self):
            native = np.zeros(16, dtype=np.float32)
            native[NATIVE_ORDER.index("left_joint2")] = 9.0  # beyond the action bound
            return native

    fake_env.reset(seed=0, options={"task_id": "A2"})
    fake_env.adapter.make_controller = Controller
    oracle = SportsOracle(fake_env)
    oracle.reset(None)
    action = oracle.act(None)
    assert action.dtype == np.float32 and action.shape == (16,)
    assert action[JOINT_ORDER.index("left/joint2")] == np.float32(JOINT_HIGH[1])
    assert oracle.clipped_steps == 1


def test_plan_lists_ten_tasks(capsys):
    assert main(["plan", "--env", "sports", "--difficulty", "easy", "--seeds", "1"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [item["task_id"] for item in plan["plan"]] == list(IDS)
    assert {item["task_id"] for item in plan["plan"]} == {row[0] for row in CATALOG}


def test_failure_codes_split_into_physical_and_temporal():
    document = {
        "task": "A2",
        "difficulty": "easy",
        "seed": 0,
        "physical_success": False,
        "temporal_success": False,
        "success": False,
        "failures": ["no_grasp"],
    }
    row = normalize(document)
    assert row["physical_failures"] == ["no_grasp"]
    assert row["temporal_violations"] == ["temporal_requirement_not_met"]
    chess = normalize(
        {**document, "task": "A1", "failures": ["1:deadline_missed", "3:arm_collision"]}
    )
    assert chess["temporal_violations"] == ["1:deadline_missed"]
    assert chess["physical_failures"] == ["3:arm_collision"]
    # Supplemental-matrix status records point to their result file instead.
    assert normalize({**document, "physical_success": None, "runtime_completed": True}) is None
