"""Cross-domain contract: every benchmark domain registers, plans and scores alike.

Simulator-free checks always run. ``ROBOCLOCK_ISAAC_CONTRACT=1`` additionally runs
one oracle episode per domain in Isaac Sim through ``roboclock eval``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from roboclock.cli import main
from roboclock.envs.base import BenchmarkEnv
from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ, check_rates
from roboclock.envs.registry import environment_names, get_environment, import_symbol
from roboclock.evaluation.metrics import aggregate
from roboclock.evaluation.schema import (
    CATEGORIES,
    canonical_category,
    collect_results,
    normalize_result,
    require_valid,
    validate_row,
)
from roboclock.tasks.registry import task_registry

REPO = Path(__file__).resolve().parents[1]
DOMAINS = ("lab", "household", "hospital", "sports", "industry")
# Recorded episodes kept in git for each domain (one oracle demo per task in data/<env>/<task>/
# once the domain is promoted by scripts/collect/promote_demos.py).
RECORDED = {
    "lab": "data/lab",
    "household": "data/household/x5_demos",
    "hospital": "data/hospital",
    "sports": "data/sports/videos",
    "industry": "data/industry",
}
ISAAC = os.environ.get("ROBOCLOCK_ISAAC_CONTRACT") == "1"


def test_all_five_domains_are_registered():
    assert set(DOMAINS) <= set(environment_names())


@pytest.mark.parametrize("domain", DOMAINS)
def test_tasks_are_numbered_by_temporal_class(domain):
    tasks = task_registry(domain)
    assert tasks
    for task_id, cls in tasks.items():
        assert cls.category in CATEGORIES, (task_id, cls.category)
        # A = temporal window, B = event reactive, C = throughput in every domain.
        assert canonical_category(None, task_id) == cls.category, task_id


@pytest.mark.parametrize("domain", DOMAINS)
def test_plan_samples_every_task_and_difficulty(domain, capsys):
    spec = get_environment(domain)
    assert main(["plan", "--env", domain, "--seeds", "2"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["episodes"] == len(task_registry(domain)) * len(spec.difficulties) * 2
    for item in plan["plan"]:
        assert item["environment"] == domain
        assert item["spec"]["max_time"] > 0


@pytest.mark.parametrize("domain", DOMAINS)
def test_runtime_factories_import_without_the_simulator(domain):
    spec = get_environment(domain)
    script = (
        "import sys\n"
        "from roboclock.envs.registry import get_environment, import_symbol\n"
        f"spec = get_environment({domain!r})\n"
        "env = import_symbol(spec.factory)\n"
        "oracle = import_symbol(spec.oracle_factory)\n"
        "print(env.__module__, env.__name__, env.environment_id)\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('isaacsim', 'omni', 'pxr')]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert spec.oracle_factory is not None
    env = import_symbol(spec.factory)
    assert isinstance(env, type) and issubclass(env, BenchmarkEnv)
    assert env.environment_id == domain


@pytest.mark.parametrize("domain", DOMAINS)
def test_recorded_results_normalize_to_valid_rows(domain):
    rows, _ = collect_results([REPO / RECORDED[domain]])
    assert rows, f"no recorded results for {domain}"
    require_valid(rows)
    assert {row["environment"] for row in rows} == {domain}
    # Recordings of retired task versions stay summarizable under their own IDs.
    current = {row["task_id"] for row in rows if not row.get("retired")}
    assert current <= set(task_registry(domain))
    assert not {row["task_id"] for row in rows if row.get("retired")} & set(task_registry(domain))


def _task_rates(domain):
    """(physics_dt, control_decimation) of every task."""
    config = import_symbol(get_environment(domain).config_loader)()
    if domain == "household":
        return [(config["x5"]["physics_dt"], config["x5"]["control_decimation"])]
    if domain == "sports":
        from roboclock.envs.sports.config import task_rates

        return [task_rates(config, task) for task in config["tasks"]]
    if domain == "industry":
        control_dt = config["sim"]["control_dt"]
        return [
            (1 / task["physics_hz"], round(task["physics_hz"] * control_dt))
            for task in config["tasks"].values()
        ]
    return [(config["sim"]["physics_dt"], config["sim"]["control_decimation"])]


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_task_runs_on_the_shared_time_base(domain):
    for physics_dt, decimation in _task_rates(domain):
        check_rates(physics_dt, decimation, where=domain)


def test_time_base_requires_exactly_250_hz_physics():
    check_rates(1 / PHYSICS_HZ, PHYSICS_HZ // CONTROL_HZ)
    with pytest.raises(ValueError, match="exactly"):
        check_rates(1 / 240, 4)
    with pytest.raises(ValueError, match="exactly"):
        check_rates(1 / (2 * PHYSICS_HZ), 2 * PHYSICS_HZ // CONTROL_HZ)
    with pytest.raises(ValueError, match="control"):
        check_rates(1 / PHYSICS_HZ, 10)


def test_five_domains_summarize_into_one_table(capsys):
    assert main(["summarize", "--normalize", *(str(REPO / p) for p in RECORDED.values())]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary["by_environment"]) == set(DOMAINS)
    assert set(summary["by_category"]) == set(CATEGORIES)
    assert summary["overall"]["episodes"] == sum(
        v["episodes"] for v in summary["by_environment"].values()
    )


def _row(**changes):
    row = {
        "environment": "lab",
        "task_id": "A1",
        "difficulty": "easy",
        "seed": 0,
        "category": "temporal_window",
        "task_success": True,
        "manipulation_success": True,
        "temporal_success": False,
        "joint_success": False,
        "temporal_violations": ["late"],
    }
    return {**row, **changes}


def test_schema_rejects_rows_aggregate_cannot_score():
    assert validate_row(_row()) == []
    assert "missing seed" in validate_row({k: v for k, v in _row().items() if k != "seed"})
    assert validate_row(_row(category="reactive"))
    assert validate_row(_row(seed=True))
    assert validate_row(_row(joint_success=True))
    assert validate_row(_row(completion_time="12 s"))
    with pytest.raises(ValueError, match="summarize --normalize"):
        aggregate([_row(category="reactive")])


def test_category_spellings_and_class_letters_are_canonicalized():
    assert canonical_category("reactive") == "event_reactive"
    assert canonical_category("B") == "event_reactive"
    assert canonical_category(None, "C4") == "throughput"
    with pytest.raises(ValueError):
        canonical_category("timing", "41")


def test_domain_native_layouts_are_converted():
    sports = normalize_result(
        {
            "task": "B4",
            "difficulty": "hard",
            "seed": 0,
            "success": False,
            "physical_success": True,
            "temporal_success": False,
            "failures": ["late_stop", "lost_robot_grasp"],
            "reactions": [{"latency": 0.2}, {"latency": 0.4}],
        },
        environment="sports",
    )
    assert sports["category"] == "event_reactive"
    assert sports["temporal_violations"] == ["late_stop"]
    assert sports["physical_failures"] == ["lost_robot_grasp"]
    assert sports["reaction_latency"] == pytest.approx(0.3)
    assert validate_row(sports) == []

    metrics = {
        "task_id": 44,
        "category": "reactive",
        "difficulty": "reviewed_nominal",
        "seed": 0,
        "task_success": True,
        "manipulation_success": True,
        "temporal_success": True,
        "joint_success": True,
        "temporal_violations": [],
    }
    industry = normalize_result({"environment": "industry", "metrics": metrics})
    assert (industry["task_id"], industry["legacy_task_id"]) == ("B1", 44)
    assert industry["category"] == "event_reactive"
    # Raw native evidence without episode identity is not an episode result.
    assert normalize_result({"task_id": 44, "task_success": True}, environment="industry") is None


@pytest.mark.skipif(not ISAAC, reason="set ROBOCLOCK_ISAAC_CONTRACT=1 to run Isaac Sim episodes")
@pytest.mark.parametrize("domain", DOMAINS)
def test_oracle_episode_writes_schema_rows(domain, tmp_path):
    spec = get_environment(domain)
    output = tmp_path / domain
    command = [
        sys.executable,
        "-m",
        "roboclock.cli",
        "eval",
        "--env",
        domain,
        "--policy",
        "oracle",
        "--oracle",
        "--task",
        spec.default_task,
        "--difficulty",
        spec.difficulties[0],
        "--seeds",
        "1",
        "--output",
        str(output),
    ]
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    require_valid(rows)
    assert rows[0]["environment"] == domain
