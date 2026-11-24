"""Dual-X5 household environment contract, checked without Isaac Sim.

The Isaac-side episode (``x5/episode.py``) is replaced by a stub that advances the real task
model, so the gym plumbing (spaces, validation, timing, termination, rows) is tested here;
``roboclock eval --env household`` exercises the simulator (see docs/household/README.md).
"""

import dataclasses
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from roboclock.envs.household.tasks.registry import CATALOG, make_task
from roboclock.envs.household.x5 import interface as X
from roboclock.envs.lab.robot import JOINT_HIGH, JOINT_LOW
from roboclock.envs.registry import get_environment, import_symbol
from roboclock.evaluation.schema import validate_row

REPO = Path(__file__).resolve().parents[3]
DT = 1 / 250


# ------------------------------------------------------------------ registration
def test_spec_points_at_the_dual_x5_runtime():
    from roboclock.envs.base import BenchmarkEnv
    from roboclock.envs.household.x5.env import HouseholdX5Env
    from roboclock.envs.household.x5.oracle import HouseholdX5Oracle

    spec = get_environment("household")
    assert import_symbol(spec.factory) is HouseholdX5Env
    assert issubclass(HouseholdX5Env, BenchmarkEnv) and HouseholdX5Env.environment_id == "household"
    assert import_symbol(spec.oracle_factory) is HouseholdX5Oracle
    assert spec.validation_module == "roboclock.envs.household.x5.validation"


def test_factories_import_without_the_simulator():
    script = (
        "import sys\n"
        "from roboclock.envs.registry import get_environment, import_symbol\n"
        "spec = get_environment('household')\n"
        "import_symbol(spec.factory); import_symbol(spec.oracle_factory)\n"
        "import roboclock.envs.household.x5.validation, roboclock.envs.household.x5.scenes\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('isaacsim', 'omni', 'pxr')]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- rig
def test_common_rig_geometry():
    # The RoboDojo rig (roboclock.envs.robodojo_rig): bases side by side, 0.60 m apart, same height
    # and heading; head camera 0.04 m ahead of the base midpoint, 0.543 m above the mount tops,
    # pitched 60 deg down. All ten tasks use it.
    from roboclock.envs import robodojo_rig as RD
    from roboclock.envs.household.x5 import kinematics, rig

    left, right = rig.COMMON["Left"], rig.COMMON["Right"]
    assert left[1] == right[1] and right[0] - left[0] == pytest.approx(0.60, abs=1e-12)
    head = rig.head_camera()
    mid = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    assert head["eye"] == pytest.approx([mid[0], mid[1] + 0.04, rig.MOUNT_TOP + 0.543], abs=1e-6)
    look = np.subtract(head["look"], head["eye"])
    assert math.degrees(math.atan2(-look[2], look[1])) == pytest.approx(60.0)
    assert look[0] == pytest.approx(0.0, abs=1e-12)
    assert (head["focal"], head["h_ap"], head["v_ap"]) == (10.0, 22.212, 14.266)
    # The authored pose passes the shared check with the household base frames (+x = forward).
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # yaw 90 deg
    bases = {}
    for side, (x, y) in rig.COMMON.items():
        bases[side] = pose.copy()
        bases[side][:3, 3] = [x, y, rig.MOUNT_TOP]
    frame = RD.check_rig(bases["Left"], bases["Right"], np.array(head["matrix"]))
    assert frame["yaw_deg"] == pytest.approx(90.0)
    try:
        for task in ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C3"):
            assert rig.use(task) == rig.COMMON and kinematics.BASES is rig.BASES
        # Tasks outside RIG_TASKS (and no task) keep the earlier 0.72 m X5 layout.
        assert rig.use(None) == {"Left": (-0.06, -0.255), "Right": (0.66, -0.255)}
        T = kinematics.X5().base("Left")
        assert np.allclose(T[:3, 3], [-0.06, -0.255, rig.MOUNT_TOP])
    finally:
        rig.use(None)  # back to the default (legacy) layout


def test_robodojo_observations_and_no_render_config():
    from roboclock.config import load_config
    from roboclock.envs.household.x5.config import check_x5, env_config
    from roboclock.envs.household.x5.robodojo import blank_observations

    base = load_config(environment="household")
    assert env_config(base)["sim"]["no_render"] is False
    no_render = load_config(
        REPO / "src/roboclock/envs/household/configs/no_render.yaml", environment="household"
    )
    assert env_config(no_render)["sim"]["no_render"] is True
    assert {**no_render["x5"], "no_render": False} == base["x5"]
    for bad in (
        {"camera_resolution": [320, 240]},
        {"physics_dt": 1 / 500, "control_decimation": 10},
        {"no_render": "yes"},
    ):
        with pytest.raises(ValueError):
            check_x5({**base["x5"], **bad})
    obs = blank_observations(depth=True)
    for key in ("rgb", "left_wrist_rgb", "right_wrist_rgb"):
        assert obs[key].shape == (480, 640, 3) and obs[key].dtype == np.uint8 and not obs[key].any()
    for key in ("depth", "left_wrist_depth", "right_wrist_depth"):
        assert obs[key].shape == (480, 640) and obs[key].dtype == np.float32


def test_wrist_cameras_ride_link6_at_the_robodojo_mount():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    from roboclock.envs import robodojo_rig as RD
    from roboclock.envs.household.x5 import robodojo as RDH

    stage = Usd.Stage.CreateInMemory()
    link6 = {}
    for k, side in enumerate(RDH.SIDES):
        pose = RD.transform([0.1 * k, 0.2, 1.1], RD.euler_xyz(10, 20 + k, 30))
        link6[side] = pose
        xf = UsdGeom.Xform.Define(stage, f"/World/Arms/{side}/link6")
        xf.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
    paths = RDH.add_wrist_cameras(stage, "/World/Arms")
    for side, path in paths.items():
        world = RDH._world(stage.GetPrimAtPath(path))
        err = RDH.pose_error(world, link6[side] @ RD.wrist_camera_in_link6())
        assert err[0] < 1e-9 and err[1] < 1e-6


def test_household_link6_against_x5a():
    # The household arm is the arx5-sdk X5.urdf; RoboDojo's is X5A.urdf. Same joint axes and link6
    # orientation, link offsets differ by up to 3 mm (joint1/2 heights, joint5/6 offsets): reported
    # in every rig check as link6_vs_x5a. The household FK used by the scenes is the same URDF.
    from roboclock.envs.household.x5 import kinematics, rig
    from roboclock.envs.household.x5 import robodojo as RDH

    rng = np.random.default_rng(0)
    worst = 0.0
    try:
        rig.use("A1")
        robot = kinematics.X5()
        for q in [np.array([0.0, 0.94, 1.34, -1.40, 0.0, 0.0])] + [
            rng.uniform(-1.5, 1.5, 6) for _ in range(50)
        ]:
            household = RDH.link6_in_base(q, RDH.HOUSEHOLD_URDF)
            _, frames = robot.fk(q, "Left")
            link6 = np.linalg.inv(robot.base("Left")) @ frames[5]
            assert np.allclose(household, link6, atol=1e-9)
            dp, da = RDH.pose_error(household, RDH.link6_in_base(q, RDH.X5A_URDF))
            assert da < 0.01
            worst = max(worst, dp)
    finally:
        rig.use(None)
    assert 0.002 < worst < 0.0031


def test_head_camera_sees_the_a2_dispenser_outlet():
    # Projection test used for every task (x5/visibility.py); A2's outlet (z 1.178, y 0.12) at 60 deg.
    from roboclock.envs.household.x5 import visibility

    assert visibility.margin([[x, 0.12, 1.178] for x in (0.265, 0.30, 0.335)]) > 60
    # Rendered square pixels (79.6 deg vertical): 0.30 m behind the base line is just inside (9 px),
    # the reported 71.0 deg model (conservative) puts it below the image.
    assert 0 < visibility.margin([[0.30, -0.30, 0.86]]) < 10
    assert visibility.margin([[0.30, -0.30, 0.86]], conservative=True) < 0
    assert visibility.margin([[0.30, -0.35, 0.86]]) < 0  # further back: below the rendered image


# --------------------------------------------------------------------- interface
def test_action_layout_and_limits():
    assert X.JOINT_ORDER == tuple(f"{s}/joint{i}" for s in ("left", "right") for i in range(1, 9))
    assert X.ACTION_LOW.shape == X.ACTION_HIGH.shape == (16,)
    for k in (0, 8):  # arm joints: the lab limits (identical to the household X5 chain)
        assert np.array_equal(X.ACTION_LOW[k : k + 6], JOINT_LOW[:6])
        assert np.array_equal(X.ACTION_HIGH[k : k + 6], JOINT_HIGH[:6])
        # Fingers: lab opening convention; below 0 = target past the closed stop (squeeze).
        assert X.ACTION_LOW[k + 6] == np.float32(-0.03) and X.ACTION_HIGH[k + 6] == JOINT_HIGH[6]


@pytest.mark.parametrize("jaw", [0.0, -X.JAW_TRAVEL, -(X.JAW_TRAVEL + 0.025), -0.0123])
def test_finger_convention_round_trip(jaw):
    native = np.array([*X.READY_Q, jaw, -jaw])  # jawP <= 0, jawN >= 0 (symmetric close)
    action = X.native_to_action(native)
    assert action[6] == pytest.approx(action[7]) and action[6] == pytest.approx(X.JAW_TRAVEL + jaw)
    assert np.allclose(X.action_to_native(action), native, rtol=0, atol=1e-15)
    assert np.all(action.astype(np.float32) >= X.ACTION_LOW[:8])
    assert np.all(action.astype(np.float32) <= X.ACTION_HIGH[:8])


def test_ready_action_is_open_ready_pose():
    ready = X.ready_action()
    assert ready.shape == (16,)
    assert np.allclose(ready[:6], X.READY_Q) and np.allclose(ready[6:8], X.JAW_TRAVEL)
    assert np.array_equal(X.action_to_native(ready[:8])[6:], [0.0, 0.0])


# ------------------------------------------------------------------------ config
def test_config_sections():
    from roboclock.config import load_config
    from roboclock.envs.household.config import load_config as legacy
    from roboclock.envs.household.x5.config import check_x5, env_config

    config = env_config(load_config(environment="household"))
    assert config["sim"]["physics_dt"] == pytest.approx(1 / 250)
    assert config["sim"]["control_decimation"] == 5 and config["sim"]["settle_seconds"] == 3.0
    assert config["visuals"]["renderer"]
    json.dumps(config)
    assert {"sim", "scene", "difficulty", "episode", "visuals", "physics", "control"} <= set(
        legacy()
    )
    for bad in (
        {"control_decimation": 0},
        {"camera_resolution": [640]},
        {"physics_dt": -1.0},
        {"physics_dt": 1 / 120, "control_decimation": 2},
        {"control_decimation": 10},
        {"camera": "Camera"},
    ):
        with pytest.raises(ValueError):
            check_x5({**load_config(environment="household")["x5"], **bad})
    with pytest.raises(ValueError):
        env_config({"visuals": {}})


# ---------------------------------------------------------------- task clock/rows
def test_task_clock_sums_physics_steps_like_the_recorder():
    # Plain float summation, as in run_x5_episode.py: a C1 pickup at 56 s (the schedule before
    # 2026-11-09) fires one tick after 56 s, because 14000 steps of 1/250 s end a hair short of it.
    task = make_task("C1", "medium", 0, pickups_s=(20.0, 32.0, 44.0, 56.0))
    while "pickup4" not in task.events:
        task.advance(DT)
    assert task.steps == 14001 and task.events["pickup4"] == pytest.approx(56.004)
    task = make_task("C2", "medium", 0)
    while not task.terminated:
        task.advance(DT)
    # nothing served: the stall stop ends the episode at T_medium = sqrt(5) t_o (C-class rule), one tick at or after it
    assert task.end_reason == "stall" and task.events["episode_end"] == pytest.approx(5**0.5 * 43.21, abs=DT)


def test_c2_glasses_are_scored_under_the_environment_stopping_rule():
    task = make_task("C2", "medium", 0)

    def signals(t):
        ml = tuple(min(92.0, max(0.0, (t - 2.0 - 6.0 * (j % 3)) * 20.0)) for j in range(6))
        done = t >= 25.0
        return dict(glass_ml=ml, air_ml=0.0, bottle_ml=(60.0, 60.0), bottle_released_upright=(done, done))

    while task.time < task.spec.max_time and not task.terminated:  # the environment's stopping rule
        task.update(signals(task.time))
        task.advance(DT)
    m = task.metrics()
    assert m["served_glasses"] == 6 and m["joint_success"], m
    assert "all_served" in task.events and task.time < 30.0 and m["termination_reason"] == "all_done"


def test_metrics_events_match_the_evaluator_and_serialize():
    task = make_task("A1", "medium", 0)
    for k in range(int(40 / DT)):
        t = task.time
        task.update(
            dict(
                jar_tilt_deg=np.float64(90.0 if 2 <= t < 6.5 else 0.0),
                mouth_over_glass=np.bool_(True),
                jar_upright_released=np.bool_(not 1 <= t < 7),
                spoon_in_cup=14 <= t < 24,
                stir_turns=np.float64(0.0 if t < 14 else min(2.1, (t - 14) / 4)),
                spoon_in_basket_released=t >= 26,
            )
        )
        task.advance(DT)
        if task.terminated:
            break
    row = task.metrics()
    assert row["events"] == task.event_log and [e["type"] for e in row["events"]][:1] == [
        "pour_started"
    ]
    assert set(row["event_times"]) == {e["type"] for e in task.event_log}
    row.update(environment="household")
    assert validate_row(row) == []
    assert type(row["task_success"]) is bool
    json.dumps(row, allow_nan=False)


# --------------------------------------------------------------- env (stubbed sim)
class StubEpisode:
    """Stands in for X5Episode: advances the real task model with constant signals."""

    def __init__(self, task, config, folder, initial_scene, *, depth=False):
        self.task, self.dt, self.depth = task, config["sim"]["physics_dt"], depth
        self.control_dt = self.dt * config["sim"]["control_decimation"]
        self.cue_interval, self.settle_ticks = config["sim"]["cue_interval_ticks"], 360
        self.camera_path, self.commanded, self.applied = (
            config["sim"]["camera"],
            X.ready_action(),
            [],
        )
        self.ticks, self.events, self.setup = 0, [], {}
        self.module = types.SimpleNamespace(__name__="stub")
        self.probe = types.SimpleNamespace(row={})
        self.resolution = config["sim"]["camera_resolution"]

    def tick(self):
        self.ticks += 1
        self.task.advance(self.dt)

    def images(self):
        from roboclock.envs.household.x5.robodojo import blank_observations

        self.renders = getattr(self, "renders", 0) + 1
        return blank_observations(self.depth)

    def apply(self, action):
        self.applied.append(np.array(action))
        self.commanded = np.array(action, dtype=np.float64)

    def observation(self, instruction, sim_time, images=True):
        return {
            **(self.images() if images else {}),
            "joint_positions": self.commanded.astype(np.float32),
            "ee_pose": np.zeros(14, np.float32),
            "gripper_width": np.full(2, 0.06, np.float32),
            "sim_time": np.array([sim_time], np.float32),
            "instruction": instruction,
        }

    def close(self):
        pass


@pytest.fixture
def env(monkeypatch, tmp_path):
    from roboclock.envs.household.x5 import episode
    from roboclock.envs.household.x5.env import HouseholdX5Env

    monkeypatch.setattr(episode, "X5Episode", StubEpisode)
    env = HouseholdX5Env("B2", "medium", runtime=object())
    monkeypatch.setattr(
        env, "_initial_scene", lambda task_id: tmp_path / task_id / "initial_scene.usda"
    )
    env._workdir = tmp_path
    yield env
    env.close()


def test_reset_switches_tasks_and_steps_one_control_period(env):
    obs, info = env.reset(seed=3, options={"task_id": "A1", "difficulty": "hard"})
    assert info == {}
    assert {"rgb", "left_wrist_rgb", "right_wrist_rgb"} <= set(obs) <= set(env.observation_space.spaces)
    assert (env.task.task_id, env.task.spec.difficulty, env.task.spec.seed) == ("A1", "hard", 3)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(obs["joint_positions"])
    assert (
        env.task.time == pytest.approx(5 * DT) and env.x5.ticks == 5 and env.number_of_actions == 1
    )
    assert reward == 0.0 and not terminated and not truncated and info == {}
    env.advance_latency(0.05)
    assert env.x5.ticks == 5 + math.ceil(0.05 / DT - 1e-12) and len(env.x5.applied) == 1
    obs, *_ = env.reset(seed=0, options={"task_id": "B2", "difficulty": "medium"})
    assert env.task.task_id == "B2" and env.task.time == 0.0 and env.number_of_actions == 0


def test_every_instruction_fits_the_observation_space(env):
    # The instructions state the judged rules (2026-11-10) and grew: each must still be a valid observation.
    from roboclock.envs.household.tasks.registry import TASKS

    space = env.observation_space["instruction"]
    assert space.max_length == 512
    for task_id in TASKS:
        for difficulty in ("easy", "medium", "hard"):
            for seed in range(8):
                text = make_task(task_id, difficulty, seed).gen_instruction()
                assert space.contains(text), (task_id, difficulty, seed, len(text))
    obs, _ = env.reset(seed=2, options={"task_id": "C1", "difficulty": "easy"})  # the longest one
    assert env.observation_space.contains(obs) and obs["instruction"] == env.task.gen_instruction()


def test_step_validates_actions(env):
    obs, _ = env.reset(seed=0)
    for bad in (np.zeros(15), np.full(16, np.nan)):
        with pytest.raises(ValueError):
            env.step(bad)
    action = X.ready_action()
    action[6] = -0.031  # past the allowed squeeze
    with pytest.raises(ValueError):
        env.step(action)
    action = X.ready_action()
    action[14] = 0.044  # the float64 bound itself is accepted at float32 precision
    env.step(action)
    assert env.x5.applied[-1][14] == 0.044


def test_episode_truncates_at_max_time(env):
    obs, _ = env.reset(seed=0, options={"task_id": "C1"})  # nothing served: runs to its horizon (no stall stop)
    steps = 0
    while not env.done:
        obs, reward, terminated, truncated, _ = env.step(obs["joint_positions"])
        steps += 1
    assert truncated and not terminated
    # The episode ends on the first tick where the summed clock reaches max_time.
    assert env.task.spec.max_time <= env.task.time < env.task.spec.max_time + DT
    assert env.x5.ticks == env.task.steps and env.task.time - DT < env.task.spec.max_time
    assert steps == math.ceil(env.x5.ticks / 5)
    assert env.task.metrics()["termination_reason"] == "horizon"
    with pytest.raises(RuntimeError):
        env.step(obs["joint_positions"])


def test_c_batch_episode_ends_on_the_stall_stop(env):
    obs, _ = env.reset(seed=0, options={"task_id": "C3", "difficulty": "hard"})
    assert env.task.spec.max_time == pytest.approx(5 * 25.97)  # H at every difficulty
    while not env.done:
        obs, reward, terminated, truncated, _ = env.step(obs["joint_positions"])
    # nothing sorted: the stall stop fires at T_medium = sqrt(5) t_o, before the horizon
    assert terminated and not truncated and env.task.time == pytest.approx(5**0.5 * 25.97, abs=DT)
    row = env.task.metrics()
    assert row["termination_reason"] == "stall" and not row["task_success"] and row["t_q_s"] is None
    assert row["joint_success_by_tier"] == {"easy": False, "medium": False, "hard": False}


def test_oracle_mode_info_and_rows(env, monkeypatch):
    env.oracle = True
    obs, info = env.reset(seed=0, options={"task_id": "C3"})
    assert set(info) == {"metrics", "physical_state", "episode_spec", "probe_events"}
    assert dataclasses.is_dataclass(info["episode_spec"])


# ------------------------------------------------------------------- oracle capture
def test_oracle_servo_writes_become_the_action():
    from roboclock.envs.household.x5.oracle import _CapturingArticulation

    names = [
        "joint1",
        "joint2",
        "jawP_joint",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "jawN_joint",
    ]  # any DOF order
    robot = types.SimpleNamespace(
        dof_names=names, _articulation_view=None, get_joint_positions=lambda: np.zeros(8)
    )
    arm = _CapturingArticulation(robot, X.action_to_native(X.ready_action()[:8]))
    arm_dofs = np.array([names.index(f"joint{i}") for i in range(1, 7)])
    jaws = np.array([names.index("jawP_joint"), names.index("jawN_joint")])

    def action(positions, indices):
        return types.SimpleNamespace(
            joint_positions=np.asarray(positions),
            joint_indices=indices,
            joint_velocities=None,
            joint_efforts=None,
        )

    arm.apply_action(action([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], arm_dofs))
    squeeze = X.JAW_TRAVEL + 0.025  # Servo.grip(True, squeeze=.025): targets 25 mm past the stop
    arm.apply_action(action([-squeeze, squeeze], jaws))
    out = X.native_to_action(arm.targets)
    assert np.allclose(out[:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(out[6:], [-0.025, -0.025]) and np.all(
        out.astype(np.float32) >= X.ACTION_LOW[:8]
    )
    arm.apply_action(action([0.0, 0.0], jaws))  # open: arm targets persist
    assert np.allclose(
        X.native_to_action(arm.targets), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, X.JAW_TRAVEL, X.JAW_TRAVEL]
    )


# ---------------------------------------------------------- initial scene provenance
def _kitchen_ready():
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    from roboclock.envs.household.x5.scenes import HOUSEHOLD

    return (
        HOUSEHOLD / "assets/x5/lightwheel_kitchen/Collected_KitchenRoom/KitchenRoom_A1.usdc"
    ).exists()


@pytest.mark.skipif(
    not _kitchen_ready(), reason="needs OpenUSD (pxr) and the linked Lightwheel kitchen"
)
def test_initial_scenes_rebuild_byte_identical_to_the_recordings():
    import shutil

    from roboclock.envs.household.x5.scenes import (
        build_initial_scene,
        make_workdir,
        recorded_initial_sha,
        sha256,
    )

    workdir = make_workdir()  # <repo>/results/household/.x5-scenes-*: the recordings' scene depth
    try:
        for row in CATALOG["tasks"]:
            scene = build_initial_scene(row["id"], workdir)
            assert (
                sha256(scene) == row["initial_scene_sha256"] == recorded_initial_sha(row["id"])
            ), row["id"]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_unchanged_arm_targets_are_not_rewritten():
    # A PhysX target write wakes a sleeping articulation; the recorder wrote only what its
    # controller commanded, so the environment skips an arm whose 8 targets did not change.
    from roboclock.envs.household.x5.episode import X5Episode

    calls = []

    class Robot:
        def __init__(self, name):
            self.name = name

        def apply_action(self, action):
            calls.append((self.name, np.asarray(action.joint_positions).copy()))

    ep = X5Episode.__new__(X5Episode)
    ep.robots = {"Left": Robot("Left"), "Right": Robot("Right")}
    ep.dof_index = {side: np.arange(8) for side in X.SIDES}
    ep._action = lambda joint_positions, joint_indices: types.SimpleNamespace(
        joint_positions=joint_positions, joint_indices=joint_indices
    )
    ep.commanded = X.ready_action()
    ep.rewrite = set()
    ep._native = {
        s: X.action_to_native(ep.commanded[8 * k : 8 * k + 8]) for k, s in enumerate(X.SIDES)
    }
    ep.apply(X.ready_action())
    assert calls == []  # nothing commanded yet: no writes
    action = X.ready_action()
    action[0] = 0.1
    ep.apply(action)
    ep.apply(action)
    assert [name for name, _ in calls] == ["Left"] and calls[0][1][0] == 0.1
    assert np.array_equal(ep.commanded, action)
    ep.rewrite = {"Right"}  # the privileged oracle's controller wrote the right arm (same values)
    ep.apply(action)
    assert [name for name, _ in calls] == ["Left", "Right"] and ep.rewrite == set()
