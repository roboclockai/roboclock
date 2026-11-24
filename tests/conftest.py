"""Shared test configuration.

Some tests check the recorded evaluation results and demonstrations under ``data/``. Those
recordings are not part of this repository, so the tests are skipped unless
``ROBOCLOCK_RECORDINGS=1`` is set in a checkout that has them.
"""
import os

import pytest

NEEDS_RECORDINGS = (
    "tests/test_domain_contract.py::test_recorded_results_normalize_to_valid_rows",
    "tests/test_domain_contract.py::test_five_domains_summarize_into_one_table",
    "tests/envs/household/test_x5_env.py::test_initial_scenes_rebuild_byte_identical_to_the_recordings",
    "tests/envs/household/test_x5_tasks.py::test_designs_scene_builders_and_demos_exist",
    "tests/envs/industry/test_industry_benchmark.py::test_recorded_sidecars_normalize_to_class_ids",
    "tests/envs/sports/test_unified_env.py::test_arcade_rows_use_current_task_ids",
    "tests/envs/sports/test_unified_env.py::test_no_render_skips_the_cameras_and_all_rendering",
    "tests/envs/sports/test_unified_env.py::test_step_validates_maps_and_reports_termination",
    "tests/envs/sports/test_unified_env.py::test_evaluator_writes_schema_rows_across_tasks",
)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ROBOCLOCK_RECORDINGS") == "1":
        return
    skip = pytest.mark.skip(reason="needs the recorded results under data/ (set ROBOCLOCK_RECORDINGS=1)")
    for item in items:
        if item.nodeid.split("[")[0] in NEEDS_RECORDINGS:
            item.add_marker(skip)
