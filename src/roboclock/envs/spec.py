"""Declarative environment metadata; no simulator or domain imports."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentSpec:
    """One domain's implementation, configuration, assets and optional diagnostics."""

    name: str
    description: str
    factory: str
    task_registry: str
    task_factory: str
    config_loader: str
    default_config: Path
    asset_root: Path
    default_task: str
    render_task: str
    difficulties: tuple[str, ...] = ("easy", "medium", "hard")
    runtime_factory: str = "roboclock.simulators.isaac.runtime:IsaacRuntime"
    oracle_factory: str | None = None
    validation_module: str | None = None
    check_config: str | None = None
    # module:function turning a domain-native recorded result into a schema row.
    result_adapter: str | None = None
    # module:TABLE of per-task stage scores (roboclock.evaluation.score): task id -> stages,
    # or task id -> function(result row) -> stages for count-based tasks.
    stages: str | None = None
