"""Hospital registration metadata: importing this does not load the scene."""

from pathlib import Path

from roboclock.envs.spec import EnvironmentSpec

HOSPITAL_ROOT = Path(__file__).parent
SPEC = EnvironmentSpec(
    name="hospital",
    description="Hospital: scene-02 ward process-window (A1-A4), reaction (B1-B6) and two-arm throughput (C1-C4) tasks",
    factory="roboclock.envs.hospital.env:HospitalEnv",
    task_registry="roboclock.envs.hospital.tasks.registry:TASKS",
    task_factory="roboclock.envs.hospital.tasks.registry:make_task",
    config_loader="roboclock.envs.hospital.config:load_config",
    default_config=HOSPITAL_ROOT / "configs/benchmark.yaml",
    asset_root=HOSPITAL_ROOT / "assets",
    default_task="B4",
    render_task="B4",
    oracle_factory="roboclock.envs.hospital.policies.oracle:OraclePickPlace",
    validation_module="roboclock.envs.hospital.validation",
    check_config="roboclock.envs.hospital.layout:check_config",
    result_adapter="roboclock.envs.hospital.results:normalize",
    stages="roboclock.envs.hospital.stages:STAGES",
)
