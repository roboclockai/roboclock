"""CPU-only integrity checks for the Industry release, independent of Isaac Sim."""

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path


def verify(root, media=True):
    errors = []
    assets = root / "assets/industry"
    manifest = json.loads((assets / "manifest.json").read_text())
    for item in manifest["files"]:
        p = assets / item["file"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"Missing/changed asset: {item['file']}")
    for item in manifest.get("additional_notices", []):
        p = assets / item["file"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"Missing/changed notice: {item['file']}")
    package = root / "src/roboclock/envs/industry"
    # The per-task recorded source profiles were merged into native/ after commit
    # dd853c8; their release hashes are verifiable from that commit.
    sources = sorted((package / "native").rglob("*.py"))
    for p in sources:
        ast.parse(p.read_text())
    for t in range(41, 51):
        p = package / f"configs/task{t}/reviewed.json"
        spec = json.loads(p.read_text())
        digest = spec.pop("sha256")
        if hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() != digest:
            errors.append(f"Replay checksum mismatch: {t}")
    if media:
        folder = root / "data/industry/videos-v2/dual-arx-x5"
        videos = json.loads((folder / "collection_manifest.json").read_text())["videos"]
        if len(videos) != 10:
            errors.append("Expected ten reviewed videos")
        for item in videos:
            movie = folder / item["file"]
            sidecar = json.loads(movie.with_suffix(".json").read_text())
            task = int(item["task"])
            spec = json.loads((package / f"configs/task{task}/reviewed.json").read_text())
            if sidecar["episode_spec"] != spec:
                errors.append(f"Video/replay configuration mismatch: {task}")
            if sidecar["sha256"] != item["sha256"]:
                errors.append(f"Sidecar/collection video checksum mismatch: {task}")
            if hashlib.sha256(movie.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f"Video checksum mismatch: {item['task']}")
            trace = sidecar["native_trace"]
            tracefile = folder / trace["file"]
            if hashlib.sha256(tracefile.read_bytes()).hexdigest() != trace["sha256"]:
                errors.append(f"Trace checksum mismatch: {item['task']}")
            with gzip.open(tracefile, "rt") as stream:
                count = sum(1 for line in stream if line.strip())
            if count != trace["rows"]:
                errors.append(f"Trace row mismatch: {item['task']}")
            if not all((folder / f).is_file() for f in sidecar["previews"].values()):
                errors.append(f"Missing preview: {item['task']}")
            # robodojo_rig recordings: head / wrist streams frame-synchronized with the overview.
            for stream, info in sidecar.get("camera_streams", {}).items():
                path = folder / info["file"]
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                    errors.append(f"Camera stream changed or missing: {item['task']} {stream}")
                if info["frames"] != sidecar.get("total_frames"):
                    errors.append(f"Camera stream not frame-synchronized: {item['task']} {stream}")
            if task == 49:
                captures = sidecar.get("inspection_rgb_evidence", {})
                if captures.get("count") != 9 or not captures.get("manifest"):
                    errors.append("Expected nine matched Task49 inspection-camera images")
                else:
                    index = folder / captures["manifest"]
                    images = json.loads(index.read_text())["images"]
                    if len(images) != 9:
                        errors.append("Task49 inspection image count mismatch")
                    for entry in images:
                        file = index.parent / entry["file"]
                        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
                            errors.append(f"Task49 inspection image changed: {entry['file']}")
            if task >= 47:
                expected = {47: 2, 48: 3, 49: 3, 50: 2}[task]
                metrics = sidecar["metrics"]
                if (
                    metrics["correct_operations"] < expected
                    or not metrics["full_throughput_target_tested"]
                    or not metrics["task_success"]
                    or sidecar["simulation_seconds"] < 89.98
                ):
                    errors.append(f"Full90s throughput target not qualified: {task}")
                native = json.loads(
                    (folder / f"{movie.stem}_evidence/native-result.json").read_text()
                )
                if not native["task_success"] or native.get("failure"):
                    errors.append(f"Native throughput episode did not succeed: {task}")
                if task == 47 and set(map(tuple, native["final"]["credited"])) != {
                    (kit, part) for kit in [0, 1] for part in ["bearing", "bracket", "pin"]
                }:
                    errors.append("Task47 does not contain six unique released components")
                if task == 48 and set(native["final"]["placed_ids"]) != {0, 1, 2}:
                    errors.append("Task48 does not contain three unique shelf placements")
    return {
        "passed": not errors,
        "errors": errors,
        "asset_files": len(manifest["files"]),
        "source_files": len(sources),
        "media_checked": media,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--skip-media", action="store_true")
    args = parser.parse_args()
    result = verify(args.root, media=not args.skip_media)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
