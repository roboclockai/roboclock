"""Render the C1/C2/C3 throughput tasks in the shared daylit laboratory setting."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.videos import main as record_videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/lab/videos-v2"))
    parser.add_argument("--tasks", nargs="+", choices=["C1", "C2", "C3"], default=["C2", "C3"])
    parser.add_argument(
        "--expert",
        action="store_true",
        help="C1 grasp-assist diagnostic: 24 tubes, 180 seconds, privileged control",
    )
    parser.add_argument("--subframes", type=int, default=4)
    args = parser.parse_args()
    if args.expert and args.tasks != ["C1"]:
        parser.error("--expert is a C1-only demonstration profile")
    for task in args.tasks:
        name = {"C1": "rack_organization", "C2": "rack_loading", "C3": "calibration_placement"}[
            task
        ]
        destination = args.output / f"{task}_{name}.mp4"
        if destination.exists() or destination.with_suffix(".partial.mp4").exists():
            parser.error(
                f"Existing recording for {task} in {args.output}; use a fresh output directory"
            )
    root = Path(__file__).resolve().parents[2]
    source = root / "docs/validation/throughput_source"
    config = load_config(source / "render_config.yaml")
    # Retain the historical room profile while using the current embodiment
    # and task-specific mounts, including the opposed C2/C3 workspaces.
    config["robot"] = load_config()["robot"]
    if args.expert:
        config["episode"].update(num_samples_range=[24, 24], max_time=180.0)
        config["control"]["c1_expert_grasp_assist"] = True
    config["scene"].update(
        throughput_background_v2=True,
        throughput_camera_position=[1.45, 2.3, 2.05],
        throughput_camera_look_at=[0.25, -0.45, 1.15],
        throughput_camera_focal_length=29.0,
        liquid_background_bench="closed_laboratory_bench.usdc",
        liquid_background_bench_position=[1.25, -2.0, 0.0],
    )
    config["liquid_visuals"] = {
        **load_config().get("liquid_visuals", {}),
        "renderer": "RaytracedLighting",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(args.tasks)
    config_path = args.output / f"source_config_{suffix}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {}
    for task in args.tasks:
        name = {"C1": "rack_organization", "C2": "rack_loading", "C3": "calibration_placement"}[
            task
        ]
        if task == "C1":
            continue
        sidecar = source / f"{task}_{name}.json"
        data = json.loads(sidecar.read_text())
        metadata[task] = data
    records = {
        "scope": "New C1 rack organization and/or existing C2/C3 background revisions",
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for task, data in metadata.items()
            for p in [source / Path(data["file"]).with_suffix(".json")]
        },
        "removed_source_video_sha256": {
            data["file"]: data["sha256"] for data in metadata.values()
        },
        "source_config_sha256": hashlib.sha256(
            (source / "render_config.yaml").read_bytes()
        ).hexdigest(),
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unchanged_physics_and_control": False,
        "embodiment_revision": "Dual ARX X5 with current benchmark mounts and controller",
        "profile": "C1 grasp-assist diagnostic: 24 tubes / 180 s" if args.expert else "benchmark",
        "new_tasks": [t for t in args.tasks if t == "C1"],
    }
    (args.output / f"source_manifest_{suffix}.json").write_text(json.dumps(records, indent=2))
    cli = [
        "--tasks",
        *args.tasks,
        "--config",
        str(config_path),
        "--output",
        str(args.output),
        "--renderer",
        "RaytracedLighting",
        "--samples",
        "32",
        "--subframes",
        str(args.subframes),
        "--fps",
        "25",
        "--resolution",
        "1280",
        "960",
        "--seconds",
        str(config["episode"]["max_time"]),
    ]
    for task, data in metadata.items():
        cli += ["--task-seed", f"{task}={data['metrics']['seed']}"]
    record_videos(cli)


if __name__ == "__main__":
    main()
