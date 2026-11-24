"""Lazy registration of the ten current Dual ARX X5 sports tasks (A1–C3)."""

from pathlib import Path

from roboclock.envs.spec import EnvironmentSpec

ROOT = Path(__file__).parent
SPEC = EnvironmentSpec(
    name="sports",
    description="Sports / games: ten Dual ARX X5 tasks (A1–C3), window/reactive/throughput",
    factory="roboclock.envs.sports.env:SportsBenchmarkEnv",
    task_registry="roboclock.envs.sports.tasks:TASKS",
    task_factory="roboclock.envs.sports.tasks:make_task",
    config_loader="roboclock.envs.sports.config:load_config",
    default_config=ROOT / "configs/benchmark.yaml",
    asset_root=ROOT / "common/assets",
    default_task="A2",
    render_task="A2",
    oracle_factory="roboclock.envs.sports.oracle:SportsOracle",
    validation_module="roboclock.envs.sports.validation",
    result_adapter="roboclock.envs.sports.results:normalize",
    stages="roboclock.envs.sports.stages:STAGES",
)
