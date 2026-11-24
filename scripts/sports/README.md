# Sports tools

Current task execution is provided by `python -m roboclock.envs.sports`.
Use a compatible Isaac Sim environment for physics/rendering commands. Write new
runs under ignored `results/`; the accepted collection is `data/sports/videos/`.

| Purpose | Tools |
|---|---|
| Full task/level/seed evaluation | `validate_supplemental.py`, `validate_fixed_embodiments.py` |
| Real controller-fault tests | `validate_fault_injection.py` |
| Summaries and trace audits | `report_supplemental.py`, `replay_supplemental.py`, `audit_freeze.py`, `audit_matrix_resets.py` |
| Video decoding and frame review | `review_collection.py` |
| Seeded configuration audit | `audit_a1_a3_randomization.py` |
| C3 hard demand schedule from eager oracle runs (`configs/c3_eager.yaml`) | `c3_hard_schedule.py` |
| Historical X5 cache preparation | `prepare_arx_x5_official.py` (uses `prepare_arx_x5.py`) |
| Geometry and interaction diagnostics | `audit_arx_official.py`, `audit_c1_rack.py`, `audit_bench.py`, `audit_bench_hands.py`, `probe_x5_basketball.py` |

Diagnostic scripts may expect a particular local asset cache or prior runner
output; inspect their arguments before running. They are not extra task versions.
Most version-specific publishers and copied source archives are retired to Git
history. The main branch retains `export_arx_official.py`,
`export_arx_polished.py`, `export_b4_fixed.py`, `export_c3_fixed.py` and their
`export_arx_videos.py` dependency. These legacy exporters require historical
inputs and are not the current collection entrypoint. Their main-branch versions
are preserved during the merge. Publish one current video/result pair per task;
retain evaluation failures and source/runtime provenance.
