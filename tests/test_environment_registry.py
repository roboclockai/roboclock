"""Exercise a second domain through the public API, without a laboratory backend."""

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.cli import main
from roboclock.config import load_config
from roboclock.envs import (
    EnvironmentSpec,
    create_env,
    get_environment,
    register_environment,
    registry,
)
from roboclock.envs.base import BenchmarkEnv
from roboclock.evaluation.evaluator import (
    control_steps_per_action,
    episode_plan,
    evaluate,
    execute_action,
)
from roboclock.evaluation.metrics import aggregate
from roboclock.policies.base import load_policy
from roboclock.resources import resolve_asset, resolve_asset_reference
from roboclock.tasks.registry import make_task, task_registry


@dataclass(frozen=True)
class ExampleEpisode:
    task_id: str
    difficulty: str
    seed: int
    max_time: float = 0.2


class ExampleTask:
    category = "temporal_window"
    discipline = "example"
    name = "example_task"

    def __init__(self, spec):
        self.spec = spec
        self.time = 0.0
        self.terminated = False
        self.event_log = []

    def gen_instruction(self):
        return "Example domain instruction"

    def metrics(self):
        return {
            "task_id": self.spec.task_id,
            "difficulty": self.spec.difficulty,
            "seed": self.spec.seed,
            "category": self.category,
            "task_success": True,
            "manipulation_success": True,
            "temporal_success": True,
            "joint_success": True,
            "temporal_violations": [],
        }


EXAMPLE_TASKS = {"A1": ExampleTask}  # Deliberately reuses the laboratory task ID.


def example_task(task_id, difficulty="standard", seed=0, config=None):
    return EXAMPLE_TASKS[task_id](ExampleEpisode(task_id, difficulty, seed))


def example_config(path):
    return json.loads(path.read_text())


class ExampleEnv(BenchmarkEnv):
    environment_id = "example_domain"

    def __init__(self, task_id, difficulty, **kwargs):
        self.task_id, self.difficulty = task_id, difficulty
        self.config = load_config(environment=self.environment_id)
        self.oracle = kwargs.get("oracle", False)
        self.done = False
        self.actions = []
        self.kwargs = kwargs

    def reset(self, *, seed=None, options=None):
        options = options or {}
        self.task = example_task(
            options.get("task_id", self.task_id),
            options.get("difficulty", self.difficulty),
            seed,
        )
        self.done = False
        return {"joint_positions": np.zeros(2)}, {}

    def advance_latency(self, seconds):
        self.task.time = min(self.task.time + seconds, self.task.spec.max_time)
        self.done = self.task.time >= self.task.spec.max_time

    def step(self, action):
        self.actions.append(action)
        self.advance_latency(0.1)
        return {"joint_positions": np.zeros(2)}, 1.0, False, self.done, {}

    def benchmark_metadata(self):
        return {"environment": self.environment_id, "simulator": "test-double", "assets": {}}

    def episode_metadata(self):
        return {"scene_id": "example_scene", "robot": "example_two_joint_robot"}


@pytest.fixture
def example(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_ENVIRONMENTS", registry._ENVIRONMENTS.copy())
    monkeypatch.setattr(registry, "_PLUGINS_LOADED", True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"sim": {"physics_dt": 0.1}, "visuals": {}}))
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "model.usda").write_text("#usda 1.0\n")
    spec = EnvironmentSpec(
        name="example_domain",
        description="Test domain, not a physics simulation",
        factory=f"{__name__}:ExampleEnv",
        task_registry=f"{__name__}:EXAMPLE_TASKS",
        task_factory=f"{__name__}:example_task",
        config_loader=f"{__name__}:example_config",
        default_config=config,
        asset_root=assets,
        default_task="A1",
        render_task="A1",
        difficulties=("standard",),
    )
    register_environment(spec)
    return spec


def test_domain_config_tasks_and_factory_are_independent(example):
    assert get_environment(example.name) == example
    assert task_registry(example.name)["A1"] is ExampleTask
    assert task_registry("lab")["A1"] is not ExampleTask
    assert "scene" not in load_config(environment=example.name)
    task = make_task("A1", "standard", 7, environment=example.name)
    assert task.spec.seed == 7
    runtime = object()
    env = create_env(example.name, runtime=runtime)
    assert env.task_id == "A1" and env.difficulty == "standard"
    assert env.kwargs["runtime"] is runtime


def test_cli_selects_domain_defaults_and_difficulty(example, capsys):
    assert main(["envs"]) == 0
    assert example.name in capsys.readouterr().out
    assert main(["tasks", "--env", example.name]) == 0
    assert "example_task" in capsys.readouterr().out
    assert main(["plan", "--env", example.name, "--seeds", "2"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["episodes"] == 2
    assert {row["difficulty"] for row in plan["plan"]} == {"standard"}
    assert {row["environment"] for row in plan["plan"]} == {example.name}


def test_generic_evaluator_uses_domain_metadata_and_latency(example, tmp_path):
    env = create_env(example.name)
    output = tmp_path / "run"
    plan = episode_plan(seeds=[3], delays=[0.2], environment=example.name)
    result = evaluate(env, plan, "hold", output)
    manifest = json.loads((output / "manifest.json").read_text())
    row = json.loads((output / "episodes.jsonl").read_text())
    assert manifest["simulator"] == "test-double"
    assert row["environment"] == example.name
    assert row["scene_id"] == "example_scene"
    assert row["robot"] == "example_two_joint_robot"
    assert row["latency_sim_time"] == pytest.approx(0.2)
    assert row["inference_latency_mode"] == "zero"  # hold is a diagnostic
    assert not env.actions
    assert result["by_environment"][example.name]["episodes"] == 1


def test_policy_rate_ramps_targets_across_control_steps():
    class Env:
        config = {"sim": {"physics_dt": 0.004, "control_decimation": 5}}
        done = False
        targets = []

        def step(self, action):
            self.targets.append(action)
            return {}, 0.0, False, False, {}

    env = Env()
    assert control_steps_per_action(env, "module:factory", None) == 1
    assert control_steps_per_action(env, "module:factory", 50) == 1
    assert control_steps_per_action(env, "module:factory", 25) == 2
    with pytest.raises(ValueError, match="must divide"):
        control_steps_per_action(env, "module:factory", 30)
    with pytest.raises(ValueError, match="control rate"):
        control_steps_per_action(env, "oracle", 25)
    action = np.array([1.0, 3.0])
    execute_action(env, {}, action, np.array([0.0, 1.0]), 2)
    assert env.targets[0].tolist() == [0.5, 2.0]
    assert env.targets[1] is action


def test_mixed_domain_plan_is_rejected_before_writing(example, tmp_path):
    env = create_env(example.name)
    output = tmp_path / "invalid_run"
    with pytest.raises(ValueError, match="different environment"):
        evaluate(env, episode_plan(["A1"], ["easy"], [0]), "hold", output)
    assert not output.exists()


def test_aggregation_keeps_reused_task_ids_separate(example):
    row = example_task("A1").metrics()
    result = aggregate(
        [
            {**row, "environment": "lab"},
            {**row, "environment": example.name},
        ]
    )
    assert len(result["by_task_difficulty_latency"]) == 2
    assert set(result["by_environment"]) == {"lab", example.name}


def test_domain_and_shared_asset_namespaces(example, tmp_path):
    assert resolve_asset("env://model.usda", environment=example.name).exists()
    assert resolve_asset(f"{example.name}://model.usda") == example.asset_root / "model.usda"
    assert resolve_asset("shared://hdr/studio_small_09_2k.hdr").exists()
    assert resolve_asset("assets/hdr/studio_small_09_2k.hdr").exists()
    assert resolve_asset("assets/textures/epoxy_diff_1k.png").exists()
    assert resolve_asset_reference("model.usda", base_dir=tmp_path) == str(tmp_path / "model.usda")
    url = "omniverse://server/Assets/robot.usd"
    assert resolve_asset_reference(url) == url
    with pytest.raises(ValueError, match="namespace"):
        resolve_asset("env://../model.usda", environment=example.name)


def test_duplicate_invalid_and_unknown_environment_names(example):
    with pytest.raises(ValueError, match="already registered"):
        register_environment(example)
    with pytest.raises(ValueError, match="lowercase"):
        register_environment(replace(example, name="Bad/Name"))
    with pytest.raises(ValueError, match="Unknown environment"):
        create_env("missing_domain")
    env = create_env(example.name, oracle=True)
    with pytest.raises(ValueError, match="no diagnostic oracle"):
        load_policy("oracle", env)


def test_installed_plugin_discovery(example, monkeypatch):
    plugin = replace(example, name="installed_domain")
    monkeypatch.setattr(registry, "_PLUGINS_LOADED", False)
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda **kwargs: [SimpleNamespace(name=plugin.name, load=lambda: plugin)],
    )
    assert get_environment(plugin.name) == plugin
    assert get_environment(plugin.name) == plugin  # Discovery registers each plugin once.


def test_discovery_does_not_import_laboratory_or_simulator_implementations():
    code = """
import sys
from roboclock.envs import environment_names
assert 'lab' in environment_names()
assert 'roboclock.envs.lab.env' not in sys.modules
assert 'roboclock.envs.lab.backend' not in sys.modules
assert 'roboclock.envs.lab.tasks.registry' not in sys.modules
assert 'isaacsim' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
