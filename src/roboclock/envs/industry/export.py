"""Export native diagnostic metadata without inventing policy or sensor measurements."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def completed_operations(task, result):
    final = result.get("final", {})
    if task == 47:
        credited = set(map(tuple, final.get("credited", [])))
        return sum(
            all((kit, part) in credited for part in ["bearing", "bracket", "pin"])
            for kit in [0, 1]
        )
    if task == 48:
        return len(set(final.get("placed_ids", [])))
    if task == 49:
        return len(final.get("sorted", []))
    if task == 50:
        return len(final.get("delivered", []))
    return int(
        bool(
            result.get("manipulation_success")
            or result.get("task_success")
            or result.get("physical_completed")
            or result.get("physical_transfer_completed")
            or result.get("physical_placement_completed")
        )
    )


def export_movies(output, task, instruction, result, resolved, seconds=None):
    """Native success is recorded; independent physical/RGB review stays pending.

    `seconds` is the native runner's requested length (videos.py --seconds). Its throughput
    runs (47-50) always use all of it; that is a recording length, not a benchmark budget
    (C episodes are judged per docs/c-class-timing.md).
    """
    output = Path(output)
    # robodojo_rig recordings: the overview is the reviewed movie; head/left_wrist/right_wrist
    # are its frame-synchronized camera streams (listed in its sidecar).
    movies = [output / "overview.mp4"] if (output / "overview.mp4").exists() else sorted(output.glob("*.mp4"))
    if not movies:
        return []
    tracefile = output / "trace.jsonl"
    if tracefile.exists():
        trace = [json.loads(line) for line in tracefile.open() if line.strip()]
    else:
        value = json.loads((output / "trace.json").read_text())
        trace = value if isinstance(value, list) else value.get("trace", value.get("rows", []))
    sampled = trace[:: max(1, len(trace) // 60)]
    for row in sampled:
        if "time_s" in row:
            row["time"] = row["time_s"]
        if "robot_joints" in row:
            row["robot_joint_positions"] = [
                row["robot_joints"][side]["q"][row["robot_joints"][side]["names"].index(f"joint{i}")]
                for side in ["left", "right"]
                for i in range(1, 9)
            ]
    elapsed = trace[-1].get("time_s", trace[-1].get("time")) if trace else None
    correct = completed_operations(task, result)
    success = bool(result.get("task_success"))
    previews = {name: f"{name}.png" for name in ["initial", "final"] if (output / f"{name}.png").exists()}
    if not previews and (output / "final_scene.png").exists():
        previews = {"final": "final_scene.png"}
    files = []
    for movie in movies:
        stream = None
        if shutil.which("ffprobe"):
            info = json.loads(subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(movie)]
            ))
            stream = info["streams"][0]
        metadata = {
            "file": movie.name,
            "environment": "industry",
            "robot": "dual_arx_x5",
            "simulator": "isaacsim==6.0.1.0",
            "policy": "privileged_native_contact_ik_diagnostic",
            "purpose": "Fresh native diagnostic; independent original-mesh/contact and encoded-RGB review pending.",
            "instruction": instruction["instruction"],
            "time_requirement": instruction["time_requirement"],
            "metrics": {
                "task_id": task,
                "domain": "industry",
                "temporal_category": instruction["temporal_category"],
                "seed": resolved["seed"],
                "task_success": success,
                "joint_success": success,
                "temporal_success": True if success else result.get("temporal_success"),
                "correct_operations": correct,
                "elapsed_time": elapsed,
                "operations_per_minute": correct * 60 / elapsed if elapsed else None,
                "physical_failures": [result["failure"]] if result.get("failure") else [],
                "episode_spec": resolved,
            },
            "episode_spec": resolved,
            "events": result.get("events", {}),
            "previews": previews,
            "process_trace": sampled,
            "native_trace": tracefile.name if tracefile.exists() else "trace.json",
            "native_result": "result.json",
            "camera_cuts": [],
            "maximum_capture_physics_delta": 0.0
            if result.get("capture_advanced_physics") is False
            else None,
            "simulation_seconds": elapsed,
            "video_seconds": float(stream["duration"]) if stream else None,
            "resolution": [stream["width"], stream["height"]] if stream else None,
            "total_frames": int(stream["nb_frames"]) if stream and stream.get("nb_frames") else None,
            "full_budget_episode": bool(
                task >= 47 and elapsed is not None and seconds is not None and elapsed >= seconds - 0.02
            ),
            "sha256": hashlib.sha256(movie.read_bytes()).hexdigest(),
            "verification": {"independent_physics_review": "pending", "encoded_rgb_review": "pending"},
            "limitations": [
                "Privileged diagnostic controller; generic policy adapter and full random-seed/difficulty matrix are not validated.",
                "Pre-merge traces 41/42 did not contain per-arm native joint samples; traces from the merged runner do.",
            ],
        }
        sidecar = movie.with_suffix(".json")
        sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
        files.append(sidecar.name)
    return files
