"""Exercise HospitalEnv, the evaluator and the CLI plan without Isaac Sim."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from hospital_helpers import snapshot

from roboclock.cli import main
from roboclock.envs import create_env
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.lab.robot import HOME
from roboclock.evaluation.evaluator import episode_plan, evaluate


class FakeBackend:
    """Clock/API test double; provides no manipulation dynamics."""

    def __init__(self, runtime, config, asset_catalog):
        self.config, self.catalog, self.collision_count = config, {}, 0
        self.commands, self.actions, self.ticks = [], [], 0
        self.renders, self.frame_time = [], None

    def reset(self, task):
        self.task = task
        self.layout = make_layout(task, self.config)
        task.bind(self.layout)

    def render_frame(self, subframes=1):
        self.renders.append(self.task.time)
        self.frame_time = self.task.time

    def images(self, depth=False):
        # As HospitalBackend.images: render unless this instant was already rendered.
        if self.frame_time is None or self.frame_time != self.task.time:
            self.render_frame()
        width, height = self.config["sim"]["camera_resolution"]
        return {
            "rgb": np.zeros((height, width, 3), dtype=np.uint8),
            "left_wrist_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "right_wrist_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        }

    def observation(self, depth=False, images=True):
        return {
            **(self.images(depth) if images else {}),
            "joint_positions": np.tile(HOME, 2).astype(np.float32),
            "ee_pose": np.tile([0.4, 0, 1.0, 1, 0, 0, 0], 2).astype(np.float32),
            "gripper_width": np.array([0.08, 0.08], dtype=np.float32),
            "sim_time": np.array([self.task.time], dtype=np.float32),
            "instruction": self.task.gen_instruction(),
        }

    def state(self):
        return snapshot(self.task)

    def command(self, command):
        self.commands.append(command)

    def tick(self, render=False):
        self.ticks += 1

    def apply_action(self, joints):
        self.actions.append(joints.copy())

    def close(self):
        pass


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr("roboclock.envs.hospital.backend.HospitalBackend", FakeBackend)
    instance = create_env("hospital", "B1", "easy", runtime=object())
    instance.config["sim"]["physics_dt"] = 0.01
    instance.config["episode"]["event_time_range"] = [0.01, 0.01]
    instance.config["tasks"]["B1"]["easy"]["response_limit"] = [0.02, 0.02]
    instance.reset(seed=3)
    yield instance
    instance.close()


def test_environment_identity_and_observation_contract(env):
    assert env.environment_id == "hospital"
    obs = env.get_obs()
    assert env.observation_space.contains(obs)
    assert env._info() == {}
    assert "alarm" in obs["instruction"]


def test_policy_cameras_are_robodojo_640x480_and_render_only_when_read(env):
    from roboclock.envs.observation import StaleObservationError

    spaces = env.observation_space.spaces
    for key in ("rgb", "left_wrist_rgb", "right_wrist_rgb"):
        assert spaces[key].shape == (480, 640, 3)
    env.config["sim"]["physics_dt"] = 0.004  # the rig's 250 Hz: one step = 5 ticks = 0.02 s
    env.backend.renders.clear()
    action = np.tile(HOME, 2).astype(np.float32)
    for _ in range(5):
        obs = env.step(action)[0]
    # Steps whose observation nobody reads render nothing, whatever the time grid.
    assert env.backend.renders == []
    assert obs["joint_positions"].shape == (16,) and env.backend.renders == []
    obs["rgb"], obs["left_wrist_rgb"]
    assert env.backend.renders == pytest.approx([0.1])  # one render, at this instant
    env.advance_latency(0.012)  # off the 25 Hz grid
    late = env.get_obs()
    late["right_wrist_rgb"]
    assert env.backend.renders == pytest.approx([0.1, 0.112])
    stale = env.step(action)[0]
    env.advance_latency(0.004)
    with pytest.raises(StaleObservationError):
        stale["rgb"]


def test_latency_advances_events_and_deadlines(env):
    env.advance_latency(0.05)
    assert env.task.time == pytest.approx(0.05)
    assert "late_response" in env.task.violations  # B1: the alarm's limit passed with no response
    assert env.task.action_count == 0
    assert any(c.kind == "indicator" for c in env.backend.commands)


def test_evaluation_artifacts_carry_environment_identity(env, tmp_path):
    env.config["episode"]["max_time"] = 0.1
    summary = evaluate(
        env, episode_plan(["B2"], ["easy"], [3], environment="hospital"), "hold", tmp_path
    )
    row = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert row["environment"] == "hospital" and row["domain"] == "hospital"
    assert row["initial_state"]["pad_0"]["asset"] == "aed_pad"
    assert set(row["zones"]) == {"mark_0", "mark_1"}
    assert "hospital" in summary["by_environment"]


def test_cli_plan_and_tasks(capsys):
    assert main(["tasks", "--env", "hospital"]) == 0
    listing = capsys.readouterr().out
    assert "blood_tube_triage" in listing and "sterilizer_loading" not in listing
    from roboclock.envs.hospital.tasks.registry import TASKS

    for task_id in ("A4", "B4", "B5", "B6"):
        assert TASKS[task_id].name in listing
    assert (
        main(["plan", "--env", "hospital", "--task", "C3", "--difficulty", "hard", "--seeds", "2"])
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["episodes"] == 2 and plan["plan"][0]["spec"]["cycle_time"] is None
    from roboclock.evaluation import throughput

    t_o = throughput.oracle_time("hospital", "C3")
    assert plan["plan"][0]["spec"]["max_time"] == pytest.approx(5 * t_o)  # H = 5 t_o


def test_discovery_does_not_import_the_hospital_backend():
    import subprocess

    code = (
        "import sys\n"
        "from roboclock.envs import environment_names\n"
        "assert 'hospital' in environment_names()\n"
        "assert 'roboclock.envs.hospital.backend' not in sys.modules\n"
        "assert 'roboclock.envs.hospital.env' not in sys.modules\n"
        "assert 'isaacsim' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("task_id,limit", [("A4", 8.33), ("B4", 3.93), ("B5", 8.63), ("B6", 5.44)])
def test_cli_plans_one_bedside_operation_with_its_response_deadline(capsys, task_id, limit):
    assert (
        main(
            ["plan", "--env", "hospital", "--task", task_id, "--difficulty", "hard", "--seeds", "2"]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["episodes"] == 2
    for row in plan["plan"]:
        assert row["task_id"] == task_id
        assert row["spec"]["num_items"] == 1
        assert row["spec"]["response_limit"] == pytest.approx(limit)
        if task_id in ("B4", "B5", "B6"):  # the cue comes at a seeded time, not at frame 0
            assert 4.0 <= row["spec"]["event_time"] <= 10.0
        else:  # the instruction is in the first observation, so it is issued at frame 0
            assert row["spec"]["event_time"] == 0
        assert not row["spec"]["windows"] and not row["spec"]["expiries"]


@pytest.mark.parametrize(
    "task_id,object_name,operations,delta",
    [
        ("B4", "vent_elbow", {"elbow", "connector"}, (0.0, 0.0, 0.0)),
        ("B5", "epi_pen", {"epi_pen", "site", "hold"}, (0.0, 0.0, 0.0)),
        ("A4", "bag_0", {"spike", "hang"}, (0.0, 0.0, -0.022)),
        ("B6", "windlass_rod", {"windlass", "clip"}, (0.0, 0.0, 0.0)),
    ],
)
def test_bedside_metadata_maps_physical_operations_to_objects(
    env, task_id, object_name, operations, delta
):
    env.reset(seed=0, options={"task_id": task_id})
    task = env.task
    state = snapshot(task)
    obj = state.objects[object_name]
    state.objects[object_name] = replace(obj, position=obj.position + delta)
    state.controls["measured_travel"] = float(np.linalg.norm(delta))
    task.previous = state
    task.correct = operations.copy()
    task.attempted = operations.copy()
    task.goal_achieved = True
    task.event_time, task.action_time = 1.0, 1.2
    if task_id == "A4":  # step 1 hung, step 2 START pressed (scored at the end of the press)
        task.hanging_time = task.start_time = task.action_time
    if task_id == "B4":  # the released elbow retained on the connector
        task.reconnect_time = task.action_time
    if task_id == "B5":  # the injector fired on the pad and held for its 3 s
        task.delivered_time = task.action_time
    if task_id == "B6":  # the released rod retained in the clip
        task.secure_time = task.action_time
    # The historical 4 cm motion threshold and per-object success keys miss these actions.
    task.moved.clear()
    metadata = env.episode_metadata()
    final = metadata["final_state"][object_name]
    assert final["moved"] and final["correct"]
    assert set(final["operations"]) == operations
    device = metadata["bedside_state"]
    assert device["physical_success"] and device["temporal_success"]
    assert set(device["completed_operations"]) == operations
    assert device["measured_controls"]["measured_travel"] == pytest.approx(np.linalg.norm(delta))
    if task_id == "B6":
        assert device["symbolic_device_state"] == task.device_state
    json.dumps(metadata, allow_nan=False)
    # Exported evidence remains a snapshot when the environment changes afterwards.
    state.controls["measured_travel"] = 9.0
    assert device["measured_controls"]["measured_travel"] != 9.0


def test_bedside_metadata_does_not_mark_partial_insertion_as_completed(env):
    env.reset(seed=0, options={"task_id": "A4"})
    env.task.previous = snapshot(env.task)
    env.task.correct = {"spike"}
    env.task.attempted = {"spike"}
    result = env.episode_metadata()
    assert result["final_state"]["bag_0"]["moved"]
    assert not result["final_state"]["bag_0"]["correct"]
    assert result["bedside_state"]["completed_operations"] == ["spike"]
    assert not result["bedside_state"]["physical_success"]


def windlass_state(task, angle, contact):
    """A measured B6 windlass state: the rod turned `angle` from its start (bedside/tourniquet.py)."""
    state = snapshot(task)
    state.controls["tourniquet"] = {
        "angle_rad": angle, "speed_rad_s": 0.0, "contact": contact, "grasped": contact,
        "clip_engaged": True,
    }
    return state


@pytest.mark.parametrize("render", (False, True))
@pytest.mark.parametrize("response_limit,successful", [(0.1, True), (0.02, False)])
def test_bedside_terminal_light_matches_joint_result_without_extra_ticks(
    env, render, response_limit, successful
):
    from roboclock.envs.hospital.mdp.events import GREEN, RED
    env.reset(seed=0, options={"task_id": "B6"})
    task = env.task
    task.spec = replace(task.spec, event_time=0.01, response_limit=response_limit)
    task.secure_dwell = task.hold_after = 0.0  # scored and ended at the first released instant
    # The windlass is held at the clip for three ticks and let go on the fourth (0.04 s).
    clip = task.turn_required  # the episode's turn from the rod's start to its clip
    held, released = windlass_state(task, clip, True), windlass_state(task, clip, False)
    env.backend.state = lambda: released if task.time > 0.035 else held
    renders = []
    env.backend.world = SimpleNamespace(
        render=lambda: renders.append((task.time, env.backend.ticks))
    )
    env._ticks(10, render=render)
    assert env.done and task.physical_success
    assert task.joint_success is successful
    assert task.time == pytest.approx(0.04)
    assert env.backend.ticks == 4  # the terminal refresh performs no fifth physics tick
    final_command = env.backend.commands[-1]
    assert final_command.name == "indicator"
    assert final_command.value == (GREEN if successful else RED)
    ring_command = env.backend.commands[-2]
    assert ring_command.kind == "timer"
    assert ring_command.value == (1.0, "green" if successful else "red")
    # Physics ticks never render; reading the final observation renders the outcome once,
    # without another physics tick.
    assert not renders
    env.backend.renders.clear()
    env.get_obs()["rgb"]
    assert env.backend.renders == pytest.approx([0.04]) and env.backend.ticks == 4


def test_bedside_physical_failure_ends_red_without_an_extra_tick(env):
    from roboclock.envs.hospital.mdp.events import RED
    from roboclock.envs.hospital.tasks.bedside.tourniquet import OVERTURN

    env.reset(seed=0, options={"task_id": "B6"})
    task = env.task
    task.spec = replace(task.spec, event_time=0.01, response_limit=0.1, max_time=0.05)
    # The windlass forced past its clip: over-tightened on the first tick after the bleeding starts.
    state = windlass_state(task, task.turn_required + OVERTURN + 0.05, True)
    env.backend.state = lambda: state
    env._ticks(10, render=False)
    assert env.done and not task.physical_success
    assert "over_tightened" in task.physical_failures
    assert env.backend.commands[-1].value == RED
    assert task.time == pytest.approx(0.01) and env.backend.ticks == 1
