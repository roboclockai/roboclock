"""Industry registration metadata: importing this does not load a scene."""

from pathlib import Path

from roboclock.envs.spec import EnvironmentSpec

ROOT = Path(__file__).parent
SPEC = EnvironmentSpec(
    name="industry",
    description="Industry: 3 window, 3 reactive, 4 throughput Dual-X5 workcell tasks (41–50)",
    factory="roboclock.envs.industry.env:IndustryEnv",
    task_registry="roboclock.envs.industry.tasks:TASKS",
    task_factory="roboclock.envs.industry.tasks:make_task",
    config_loader="roboclock.envs.industry.config:load_config",
    default_config=ROOT / "configs/benchmark.yaml",
    # Imported/prepared Industry assets live at the repository level (see its README).
    asset_root=ROOT.parents[3] / "assets/industry",
    default_task="B2",
    render_task="B2",
    # reviewed_nominal (the recorded episodes) is also accepted by make_task and the env.
    difficulties=("easy", "medium", "hard"),
    oracle_factory="roboclock.envs.industry.oracle:IndustryOracle",
    result_adapter="roboclock.envs.industry.results:normalize",
    stages="roboclock.envs.industry.stages:STAGES",
)
