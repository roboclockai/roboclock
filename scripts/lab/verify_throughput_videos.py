"""Verify throughput recordings, fixed-budget rates and retained C2/C3 source metadata."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from verify_liquid_videos import verify


def verify_c1(directory, require_success=False):
    result = verify(directory, ("C1",))
    data = json.loads((directory / "C1_rack_organization.json").read_text())
    metrics = data["metrics"]
    elapsed = metrics["elapsed_time"]
    correct, attempted = metrics["correct_operations"], metrics["attempted_operations"]
    checks = {
        "complete_requested_episode": data["full_budget_episode"]
        and abs(elapsed - data["requested_simulation_seconds"]) < 1e-6,
        "matching_reference_format": data["fps"] == 15 and data["resolution"] == [1280, 960],
        "throughput_uses_full_elapsed_budget": abs(
            metrics["operations_per_minute"] - 60 * correct / elapsed
        )
        < 1e-9,
        "accuracy_uses_all_attempts": abs(metrics["accuracy"] - correct / max(1, attempted)) < 1e-9,
        "rack_organization_task": metrics["task_name"] == "rack_organization",
        "three_color_sorting": metrics.get("sorting_rule") == "one_color_per_rack_any_free_slot",
        "three_color_racks": metrics.get("rack_colors") == ["red", "blue", "green"],
        "color_counts_sum_to_total": sum(metrics.get("correct_by_color", {}).values()) == correct,
        "review_frames_present": set(data["previews"]) == {"initial", "midpoint", "final"},
    }
    if require_success:
        final = data["object_final_state"]
        checks.update(
            joint_success=bool(metrics["joint_success"]),
            all_tubes_sorted=correct == len(data["initial_state_and_targets"]["initial_state"]),
            perfect_accuracy=metrics["accuracy"] == 1.0,
            no_physical_failures=not metrics["physical_failures"],
            no_grasp_assist_at_end=not data["expert_control"]["active_grasp_constraint_at_end"],
            unique_final_slots=len({obj["assigned_target"] for obj in final.values()}) == correct,
            final_colors_match=all(
                int(name.split("_")[-1]) % 3 == int(obj["assigned_target"].split("_")[-1]) // 12
                for name, obj in final.items()
            ),
            all_released_and_in_contact=all(
                not obj["grasped"] and obj["assigned_target"] in obj["fixture_contacts"]
                for obj in final.values()
            ),
        )
    result["throughput_checks"] = checks
    result["correct_operations"] = correct
    result["attempted_operations"] = attempted
    result["operations_per_minute"] = metrics["operations_per_minute"]
    result["accuracy"] = metrics["accuracy"]
    result["all_throughput_checks_passed"] = result["all_integrity_checks_passed"] and all(
        checks.values()
    )
    (directory / "verification_C1.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["all_throughput_checks_passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=["C1", "C2", "C3"], default=["C2", "C3"])
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Require C1 to be a successful expert episode",
    )
    args = parser.parse_args()
    if "C1" in args.tasks:
        verify_c1(args.directory, args.require_success)
    existing = [task for task in args.tasks if task != "C1"]
    if not existing:
        return
    result = verify(args.directory, existing)
    root = Path(__file__).resolve().parents[2]
    source = root / "docs/validation/throughput_source"
    provenance = json.loads((args.directory / "source_manifest_C2_C3.json").read_text())
    source_unchanged = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in provenance["source_sha256"].items()
    )
    original_config = yaml.safe_load((source / "render_config.yaml").read_text())
    config_path = args.directory / "render_config_C2_C3.yaml"
    if not config_path.exists():
        config_path = args.directory / "render_config.yaml"
    current_config = yaml.safe_load(config_path.read_text())
    configuration_unchanged = all(
        current_config[key] == original_config[key]
        for key in ("physics", "control", "episode", "difficulty")
    ) and all(
        current_config["sim"][key] == original_config["sim"][key]
        for key in ("physics_dt", "control_decimation", "settle_steps", "placement_dwell")
    )
    for row in result["videos"]:
        current = json.loads((args.directory / Path(row["file"]).with_suffix(".json")).read_text())
        original = json.loads((source / Path(row["file"]).with_suffix(".json")).read_text())
        row["retained_source_metadata_unchanged"] = source_unchanged
        row["task_instruction_unchanged"] = current["instruction"] == original["instruction"]
        row["initial_layout_unchanged"] = all(
            current["initial_state_and_targets"][key] == original["initial_state_and_targets"][key]
            for key in ("initial_state", "targets", "clutter_initial_state")
        )
        row["complete_120_second_episode"] = (
            current["full_budget_episode"] and abs(current["simulation_seconds"] - 120.0) < 1e-6
        )
        row["original_format_retained"] = (
            current["fps"] == original["fps"] == 15
            and current["resolution"] == original["resolution"] == [1280, 960]
            and current["total_frames"] == original["total_frames"] == 1816
        )
        row["source_correct_operations"] = original["metrics"]["correct_operations"]
        row["current_correct_operations"] = current["metrics"]["correct_operations"]
    result["physics_and_control_configuration_unchanged"] = configuration_unchanged
    result["source_video_verification"] = (
        "Original videos were removed during cleanup; source sidecars and configuration "
        "are checked against their recorded hashes. Current videos are fully decoded."
    )
    result["all_appearance_revision_checks_passed"] = configuration_unchanged and all(
        all(
            row[key]
            for key in (
                "retained_source_metadata_unchanged",
                "task_instruction_unchanged",
                "initial_layout_unchanged",
                "complete_120_second_episode",
                "original_format_retained",
            )
        )
        for row in result["videos"]
    )
    output = args.directory / "verification_C2_C3.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["all_appearance_revision_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
