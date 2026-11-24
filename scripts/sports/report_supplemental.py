"""Summarize current-task matrix without equating process exit with task success."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

TASKS = ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1", "C2", "C3"]
LEVELS = ["easy", "medium", "hard"]


def temporal_audit(task, r, c):
    """Independent event arithmetic only; not a replacement physical evaluator."""
    checks = {}
    if task == "A1":
        turns = r["turns"]
        checks["turn_deadlines_match"] = all(
            x["temporal_success"]
            == (
                x["opponent_press_time"] is None
                or x["press_time"] - x["opponent_press_time"] <= x["deadline"] + 1e-9
            )
            for x in turns
        )
        checks["latencies_match"] = all(
            abs(x["latency"] - (x["press_time"] - (x["opponent_press_time"] or 0))) < 1e-8
            for x in turns
        )
        checks["physical_turn_has_prior_placement"] = all(
            not x["physical_success"]
            or x["placement_time"] is not None
            and x["placement_time"] < x["press_time"]
            for x in turns
        )
    elif task == "A3":
        release = r["release_time"]
        w = c["workflow"]
        expected = (
            release is not None and w["shot_start"] <= release < w["shot_start"] + w["shot_clock"]
        )
        checks["release_window_matches"] = expected == r["temporal_success"]
    elif task == "B2":
        checks["each_strike_deadline_matches"] = all(
            h["on_time"] == (h["time"] - h["cue_time"] <= c["workflow"]["cue_response"])
            for h in r["hits"]
        )
        checks["strike_latency_matches"] = all(
            abs(h["latency"] - (h["time"] - h["cue_time"])) < 1e-8 for h in r["hits"]
        )
    elif task == "B3":
        fatigue, catch = r["exhaustion_time"], r["catch_time"]
        expected = (
            fatigue is not None and catch is not None and catch - fatigue <= r["catch_deadline"]
        )
        checks["catch_deadline_matches"] = expected == r["temporal_success"]
        checks["catch_latency_matches"] = (
            catch is None or abs(r["reaction_latency"] - (catch - fatigue)) < 1e-8
        )
    elif task == "B4":
        checks["strict_stop_deadlines_match"] = all(
            x["within_deadline"]
            == (x["latency"] is not None and x["latency"] < c["freeze"]["response_limit"])
            for x in r["reactions"]
        )
        checks["reaction_timestamps_match"] = all(
            x["latency"] is None or abs(x["latency"] - (x["stop_time"] - x["event_time"])) < 1e-8
            for x in r["reactions"]
        )
    elif task == "C2":
        events = [e for e in r["events"] if e.get("kind") == "card_complete"]
        w = c["workflow"]
        checks["unique_card_count_matches"] = (
            len({e["card"] for e in events}) == len(events) == r["cards"]
        )
        checks["cards_inside_budget"] = all(
            w["preparation"]
            <= e["time"]
            <= w["preparation"] + w.get("cups_budget", w.get("cups_horizon"))
            for e in events
        )
        checks["full_budget_rate_matches"] = (
            abs(
                r["cards_per_minute"]
                - len(events) * 60 / w.get("cups_budget", w.get("cups_horizon"))
            )
            < 1e-8
        )
    elif task == "C3":
        accepted = [e for e in r["events"] if e["event"] == "accepted"]
        fired = [e for e in r["events"] if e["event"] == "outlet_passage"]
        checks["accepted_once"] = (
            len({e["ball"] for e in accepted}) == len(accepted) == r["accepted"]
        )
        checks["fired_once"] = len({e["ball"] for e in fired}) == len(fired) == r["served"]
        checks["every_exit_after_acceptance"] = all(
            any(a["ball"] == e["ball"] and a["time"] < e["time"] for a in accepted) for e in fired
        )
    return dict(
        scope="Independent arithmetic on recorded events; does not prove contact validity",
        checks=checks,
        passed=all(checks.values()) if checks else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for status in sorted(root.glob("*.status.json")):
        d = json.loads(status.read_text())
        if not d["runtime_completed"]:
            rows.append(
                dict(
                    task=d["task"],
                    difficulty=d["difficulty"],
                    seed=d["seed"],
                    runtime_completed=False,
                    success=False,
                    full_episode=False,
                    failures=["runtime_error"],
                )
            )
            continue
        path = root / d["result"]
        r = json.loads(path.read_text())
        m = r.get("metrics", r)
        cfg = r.get("configuration") or json.loads((path.parent / "config.json").read_text())
        physical = m.get("physical_success", m.get("manipulation_success"))
        reasons = list(
            dict.fromkeys(
                m.get("failures", m.get("physical_failures", [])) + m.get("temporal_violations", [])
            )
        )
        if r.get("arm_collision_steps", 0) > 0 and "arm_collision" not in reasons:
            reasons.append("arm_collision")
        row = dict(
            task=d["task"],
            difficulty=d["difficulty"],
            seed=d["seed"],
            runtime_completed=True,
            full_episode=r.get("full_episode", False),
            success=m.get("success", m.get("joint_success", False)),
            physical_success=physical,
            temporal_success=m.get("temporal_success"),
            failures=reasons,
            wall_seconds=d["wall_seconds"],
            result=d["result"],
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            sim_time=r.get("evaluation_end", r.get("sim_time", m.get("elapsed_time"))),
        )
        row["event_audit"] = temporal_audit(d["task"], r, cfg)
        rows.append(row)
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (root / "matrix.csv").open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "difficulty",
                "seed",
                "runtime_completed",
                "full_episode",
                "success",
                "physical_success",
                "temporal_success",
                "failures",
                "sim_time",
                "wall_seconds",
                "result",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {k: "; ".join(r[k]) if k == "failures" else r.get(k) for k in writer.fieldnames}
            )
    lines = [
        "# Supplemental sports validation",
        "",
        "Scope: current Dual ARX X5, privileged scripted baseline, full task budgets. No deadline, collision geometry or success predicate is relaxed.",
        "",
        f"Completed process records: {len(rows)} / {manifest['expected_episodes']}. Task successes: {sum(bool(r['success']) for r in rows)}.",
        "",
        "| Task | Easy | Medium | Hard |",
        "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        cells = []
        for level in LEVELS:
            group = [r for r in rows if r["task"] == task and r["difficulty"] == level]
            cells.append(f"{sum(bool(r['success']) for r in group)}/{len(group)}")
        lines.append("| " + task + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Counts above are task successes / completed records, not statistical confidence estimates.",
        "",
        "[Interpretation and remaining issues](FINDINGS.md) · [Outcome chart](outcomes.png) · [Reproduction](../../../../docs/sports-supplemental-validation.md)",
        "",
        "## Failures retained",
        "",
    ]
    for task in TASKS:
        failed = [r for r in rows if r["task"] == task and not r["success"]]
        if failed:
            reasons = Counter(reason for r in failed for reason in r["failures"])
            lines.append(
                f"- {task}: {len(failed)} failed cases; "
                + ", ".join(f"{k} ({v})" for k, v in reasons.items())
            )
    lines += [
        "",
        "See `matrix.csv` and per-case `result.json`, configuration and compressed measured traces. Rendered stills are retained where the task entrypoint exports them; no new presentation videos were produced.",
        "",
        "## Scope limits",
        "",
        "- Three seeds per difficulty are a diagnostic matrix, not exhaustive robustness validation.",
        "- Physics outcomes and temporal outcomes are reported separately. Current script failures are not automatically environment or evaluator defects.",
        "- Predicate unit tests use constructed states; fault-injection episodes use actual PhysX and only alter controller decisions.",
        "- Event audits recompute timestamps/count arithmetic, not independent collision geometry.",
        "- Initial launch attempts stopped at the non-interactive license gate without running episodes; prior user license acceptance was then applied through OMNI_KIT_ACCEPT_EULA=YES.",
        "- Task workers can run concurrently; per-output task locks prevent duplicate writes. Future-case shards publish only before the primary starts that case. A C3 hard seed-2 overlap is retained separately as an extra replication, excluded from the 90-case matrix. Runner revisions are preserved.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    print(
        f"{len(rows)}/{manifest['expected_episodes']} records; {sum(bool(r['success']) for r in rows)} successes"
    )


if __name__ == "__main__":
    main()
