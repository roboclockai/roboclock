"""Laboratory registration metadata: importing this does not load the scene."""

from pathlib import Path

from roboclock.envs.spec import EnvironmentSpec

LAB_ROOT = Path(__file__).parent
SPEC = EnvironmentSpec(
    name="lab",
    description="Laboratory temporal manipulation benchmark",
    factory="roboclock.envs.lab.env:LabEnv",
    task_registry="roboclock.envs.lab.tasks.registry:TASKS",
    task_factory="roboclock.envs.lab.tasks.registry:make_task",
    config_loader="roboclock.envs.lab.config:load_config",
    default_config=LAB_ROOT / "configs/benchmark.yaml",
    asset_root=LAB_ROOT / "assets",
    default_task="A1",
    render_task="A2",
    oracle_factory="roboclock.envs.lab.policies.oracle:OraclePickPlace",
    validation_module="roboclock.envs.lab.validation",
    check_config="roboclock.envs.lab.layout:check_config",
    stages="roboclock.envs.lab.stages:STAGES",
)
