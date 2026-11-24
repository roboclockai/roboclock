import json

import numpy as np
import pytest
from helpers import snapshot

from roboclock.envs.lab.env import LabEnv
from roboclock.envs.lab.layout import make_layout
from roboclock.envs.lab.robot import HOME
from roboclock.evaluation.evaluator import episode_plan, evaluate
from roboclock.evaluation.metrics import aggregate


class InstrumentedBackend:
    """Test double for clock/API tests; deliberately provides no manipulation dynamics."""

    def __init__(self, runtime, config, asset_catalog):
        self.config = config
        self.catalog = {}
        self.collision_count = 0
        self.ticks = 0
        self.actions = []
        self.commands = []

    def reset(self, task):
        self.task = task
        self.layout = make_layout(task, self.config)
        task.bind(self.layout)
        self.ticks = 0

    def images(self, depth=False):
        width, height = self.config["sim"]["camera_resolution"]
        self.renders = getattr(self, "renders", 0) + 1
        images = {"rgb": np.zeros((height, width, 3), dtype=np.uint8)}
        if depth:
            images["depth"] = np.zeros((height, width), dtype=np.float32)
        return images

    def observation(self, depth=False, images=True):
        result = {
            **(self.images(depth) if images else {}),
            "joint_positions": np.tile(HOME, 2).astype(np.float32),
            "ee_pose": np.tile([0.4, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 2).astype(np.float32),
            "gripper_width": np.array([0.08, 0.08], dtype=np.float32),
            "sim_time": np.array([self.task.time], dtype=np.float32),
            "instruction": self.task.gen_instruction(),
        }
        return result

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
    monkeypatch.setattr("roboclock.envs.lab.backend.IsaacBackend", InstrumentedBackend)
    instance = LabEnv("B1", runtime=object())
    instance.config["sim"]["physics_dt"] = 0.01
    instance.config["sim"]["control_decimation"] = 4
    instance.config["episode"]["max_time"] = 0.1
    instance.config["episode"]["event_time_range"] = [0.01, 0.01]
    instance.config["difficulty"]["easy"]["reaction_latency"] = [0.02, 0.02]
    instance.config["reactive_tasks"]["B1"]["easy"]["reaction_latency"] = [0.02, 0.02]
    instance.reset(seed=3)
    yield instance
    instance.close()


def test_latency_advances_physics_events_and_deadlines_without_new_action(env):
    env.advance_latency(0.031)
    assert env.task.time == pytest.approx(0.04)
    assert env.backend.ticks == 4
    assert env.task.event_time == pytest.approx(0.01)
    assert "late_reaction" in env.task.violations
    assert not env.backend.actions
    assert env.task.action_count == 0


def test_action_applies_for_control_decimation_and_budget_is_exact(env):
    obs = env.get_obs()
    assert env.observation_space.contains(obs)
    assert env._info() == {}
    env.step(obs["joint_positions"])
    assert env.task.time == pytest.approx(0.04)
    env.advance_latency(0.5)
    assert env.done and env.task.time == pytest.approx(0.1)
    assert env.backend.ticks == 10
    with pytest.raises(RuntimeError, match="ended"):
        env.step(obs["joint_positions"])


@pytest.mark.parametrize("action", [np.zeros(7), np.full(16, np.nan), np.full(16, 100.0)])
def test_invalid_actions_are_rejected(env, action):
    with pytest.raises(ValueError):
        env.step(action)
    assert env.task.steps == 0


def test_privileged_state_requires_explicit_oracle(env):
    assert set(env.get_obs()) == {
        "rgb",
        "joint_positions",
        "ee_pose",
        "gripper_width",
        "sim_time",
        "instruction",
    }
    env.oracle = True
    assert "physical_state" in env._info()
    assert "episode_spec" in env._info()
    # Even oracle observations retain the same public observation contract.
    assert "physical_state" not in env.get_obs()


def test_evaluation_artifacts_and_latency_overshoot(env, tmp_path):
    summary = evaluate(env, episode_plan(["B1"], ["easy"], [3], [0.5]), "hold", tmp_path)
    row = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert row["elapsed_time"] == pytest.approx(0.1)
    assert row["latency_sim_time"] == pytest.approx(0.1)
    assert row["number_of_actions"] == 0
    assert summary["overall"]["joint_success_rate"] == 0
    assert row["deadline_missed"]
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "summary.json").exists()
    with pytest.raises(FileExistsError):
        evaluate(env, episode_plan(["B1"], ["easy"], [3]), "hold", tmp_path)


def test_aggregate_separates_success_dimensions():
    rows = [
        {
            "task_id": "A1",
            "difficulty": "easy",
            "category": "temporal_window",
            "task_success": True,
            "manipulation_success": True,
            "temporal_success": False,
            "joint_success": False,
            "temporal_violations": ["late_action"],
            "completion_time": 3.0,
        }
    ]
    result = aggregate(rows)["overall"]
    assert result["task_success_rate"] == 1.0
    assert result["temporal_success_rate"] == 0.0
    assert result["joint_success_rate"] == 0.0
