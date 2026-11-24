"""Acceptance checks for the timing-only interventions (roboclock.evaluation.ablations).

They run on CPU: no scene is built, so the whole matrix is checked before any Isaac time.
"""

import pytest
import yaml

from roboclock.envs.registry import get_environment, import_symbol
from roboclock.evaluation import ablations as ab

SEEDS = (0, 1, 2)


def tier_config(condition, tier, tmp_path):
    """The loaded domain configuration for one tier, or None where the tier needs no patch."""
    patch = condition.tier_patch(tier)
    if not patch:
        return None
    path = ab.write_config(
        condition.environment,
        patch,
        tmp_path / f"{condition.environment}_{condition.task_id}_{tier.label}.yaml",
    )
    return import_symbol(get_environment(condition.environment).config_loader)(path)


@pytest.mark.parametrize("condition", ab.conditions(), ids=lambda c: c.key)
def test_tiers_keep_the_instance(condition, tmp_path):
    """Every tier of a task at one seed resolves the same non-temporal episode."""
    configs = {tier.label: tier_config(condition, tier, tmp_path) for tier in condition.tiers}
    for seed in SEEDS:
        digests = {
            tier.label: ab.instance_hash(
                condition.environment, condition.task_id, tier.difficulty, seed, configs[tier.label]
            )
            for tier in condition.tiers
        }
        assert len(set(digests.values())) == 1, f"{condition.key} seed {seed}: {digests}"


@pytest.mark.parametrize("condition", ab.conditions(), ids=lambda c: c.key)
def test_the_instruction_tracks_the_tier(condition, tmp_path):
    """What the model is told about the requirement differs between the tiers.

    A tier that moved a deadline without moving the instruction would measure a policy's luck,
    not its response to the stated requirement.
    """
    configs = {tier.label: tier_config(condition, tier, tmp_path) for tier in condition.tiers}
    statements = {
        tier.label: ab.statement(
            condition.environment, condition.task_id, tier.difficulty, 0, configs[tier.label]
        )
        for tier in condition.tiers
    }
    if condition.scoring == "by_tier":
        # One rollout judged at three budgets: the instruction states no seconds by design
        # (docs/c-class-timing.md), so it must NOT differ.
        assert len(set(statements.values())) == 1, f"{condition.key}: {statements}"
    else:
        assert len(set(statements.values())) == len(condition.tiers), f"{condition.key}"


@pytest.mark.parametrize("condition", ab.conditions(), ids=lambda c: c.key)
def test_the_episode_length_is_one_across_tiers(condition, tmp_path):
    """A tighter tier must not also end the episode earlier (except sports C3, whose horizon
    follows its demand schedule: that tier is the schedule intervention itself)."""
    if condition.key == "sports/C3":
        pytest.skip("the tier is the demand schedule, so the horizon moves with it")
    lengths = set()
    for tier in condition.tiers:
        config = tier_config(condition, tier, tmp_path)
        spec = ab.resolved_spec(
            condition.environment, condition.task_id, tier.difficulty, 0, config
        )
        lengths.add(round(float(spec.get("max_time") or spec.get("budget_s")), 6))
    assert len(lengths) == 1, f"{condition.key}: episode lengths {lengths}"


def test_a_patch_outside_the_whitelist_is_refused():
    with pytest.raises(ValueError, match="not an ablation parameter"):
        ab.apply_patch({"difficulty": {}}, {"difficulty.medium.position_tolerance": 0.5}, "lab")


def test_a_tier_config_is_a_copy_of_the_default(tmp_path):
    """The generated configuration differs from the domain default only in the patch."""
    condition = ab.EXPERIMENT1["lab/B2"]
    tier = condition.tiers[-1]
    path = ab.write_config(condition.environment, condition.tier_patch(tier), tmp_path / "c.yaml")
    written = yaml.safe_load(path.read_text())
    default = ab.default_config("lab")
    assert written["reactive_tasks"]["B2"]["medium"]["reaction_latency"] == [1.88, 1.88]
    written["reactive_tasks"]["B2"]["medium"]["reaction_latency"] = default["reactive_tasks"][
        "B2"
    ]["medium"]["reaction_latency"]
    assert written == default


def test_the_matrix_is_the_planned_size():
    """6 tasks; 5 of them one run per tier, lab C2 one run judged at three tiers."""
    assert len(ab.conditions()) == 6
    assert len(ab.runs()) == len(ab.BUDGETS_S) * (5 * 3 + 1)


def test_every_budget_still_charges_the_interface_delay():
    assert ab.INTERFACE_DELAY_S > 0
    assert ab.BUDGETS_S[0] == 0.0
