"""Dual-X5 household registration: task/process/scoring models, Isaac Sim environment, oracle."""

from pathlib import Path

from roboclock.envs.spec import EnvironmentSpec

ROOT = Path(__file__).parent
SPEC = EnvironmentSpec(
    name="household",
    description="Dual-X5 household kitchen: 4 window, 3 reactive, 3 throughput tasks (A1-C3)",
    factory="roboclock.envs.household.x5.env:HouseholdX5Env",
    task_registry="roboclock.envs.household.tasks.registry:TASKS",
    task_factory="roboclock.envs.household.tasks.registry:make_task",
    config_loader="roboclock.envs.household.x5.config:load_config",
    default_config=ROOT / "configs/benchmark.yaml",
    asset_root=ROOT / "assets",
    default_task="A1",
    render_task="A2",
    oracle_factory="roboclock.envs.household.x5.oracle:HouseholdX5Oracle",
    validation_module="roboclock.envs.household.x5.validation",
    stages="roboclock.envs.household.stages:STAGES",
)
