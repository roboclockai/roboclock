"""Full-budget current Dual X5 matrix, with one Isaac process per task.

No video is encoded. Existing demo entrypoints retain physics-rate traces and
initial/final renders. A successful invocation is distinct from task success.
"""

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

TASKS = ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1", "C2", "C3"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def worker(args):
    # A lock permits independent task workers without duplicate episodes.
    lock = (args.output / (args.worker + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    expected = [
        args.output / f"{args.worker}-{level}-seed{seed}.status.json"
        for level, seed in itertools.product(args.difficulties, args.seeds)
    ]
    if all(p.exists() for p in expected):
        return
    from roboclock.simulators.isaac import runtime as runtime_module

    runtime = runtime_module.IsaacRuntime()
    runtime_module.IsaacRuntime = lambda: contextlib.nullcontext(runtime)
    from roboclock.envs.sports.demo import main as demo

    records = []
    try:
        for level, seed in itertools.product(args.difficulties, args.seeds):
            case = f"{args.worker}-{level}-seed{seed}"
            published = args.output / case
            status = args.output / (case + ".status.json")
            if status.exists():
                records.append(json.loads(status.read_text()))
                continue
            scratch = Path(tempfile.mkdtemp(prefix="roboclock-matrix-"))
            output = scratch / case
            command = [
                "--task",
                args.worker,
                "--difficulty",
                level,
                "--seed",
                str(seed),
                "--embodiment",
                "dual_arx_x5",
                "--presentation-only",
                "--output",
                str(output),
            ]
            started = time.time()
            print("MATRIX_START", case, flush=True)
            row = dict(
                task=args.worker,
                difficulty=level,
                seed=seed,
                arguments=command,
                published_output=str(published),
            )
            try:
                sys.argv = ["sports-matrix", *command]
                demo()
                result = json.loads((output / "result.json").read_text())
                metrics = result.get("metrics", result)
                row.update(
                    runtime_completed=True,
                    full_episode=result.get("full_episode", False),
                    success=metrics.get("success", metrics.get("joint_success", False)),
                    physical_success=metrics.get("physical_success"),
                    temporal_success=metrics.get("temporal_success"),
                    failures=metrics.get("failures", metrics.get("physical_failures", [])),
                    temporal_violations=metrics.get("temporal_violations", []),
                    result=str((published / "result.json").relative_to(args.output)),
                )
            except Exception:
                row.update(runtime_completed=False, traceback=traceback.format_exc())
                traceback.print_exc()
            row["wall_seconds"] = time.time() - started
            # Preserve exact exported stage while avoiding repeated large text files.
            stage = output / "scene.usda"
            if stage.exists():
                with (
                    stage.open("rb") as src,
                    gzip.open(str(stage) + ".gz", "wb", compresslevel=3) as dst,
                ):
                    shutil.copyfileobj(src, dst)
                stage.unlink()
            if output.exists():
                shutil.copytree(output, published)
            shutil.rmtree(scratch)
            write(status, row)
            records.append(row)
            print("MATRIX_RESULT", json.dumps(row), flush=True)
        write(args.output / (args.worker + "-index.json"), records)
    finally:
        runtime.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    p.add_argument(
        "--difficulties",
        nargs="+",
        choices=["easy", "medium", "hard"],
        default=["easy", "medium", "hard"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", choices=TASKS)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    with zipfile.ZipFile(args.output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for base in ("src", "configs", "scripts/sports", "tests/envs"):
            for path in sorted(Path(base).rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix not in (".pyc", ".mp4")
                ):
                    hashes[str(path)] = digest(path)
                    archive.write(path)
    assets = {
        str(f): digest(f)
        for f in sorted(Path("src/roboclock/assets").rglob("*"))
        if f.is_file() and "__pycache__" not in f.parts
    }
    write(
        args.output / "manifest.json",
        dict(
            scope="Full-budget PhysX diagnostic matrix; privileged scripted baseline, not a learned visual policy",
            tasks=args.tasks,
            difficulties=args.difficulties,
            seeds=args.seeds,
            expected_episodes=len(args.tasks) * len(args.difficulties) * len(args.seeds),
            embodiment="dual_arx_x5",
            source_hashes=hashes,
            asset_hashes=assets,
            source_zip_sha256=digest(args.output / "source.zip"),
            python=sys.version,
            command=sys.argv,
            video=False,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    for task in args.tasks:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            task,
            "--output",
            str(args.output),
            "--difficulties",
            *args.difficulties,
            "--seeds",
            *map(str, args.seeds),
        ]
        with (args.output / (task + ".log")).open("w") as log:
            code = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
            ).returncode
        print("TASK_PROCESS_DONE", task, code, flush=True)
        write(args.output / (task + "-process.json"), dict(command=command, returncode=code))
        rows = [json.loads(f.read_text()) for f in sorted(args.output.glob("*.status.json"))]
        write(args.output / "index.json", rows)
    rows = json.loads((args.output / "index.json").read_text())
    print(
        "MATRIX_FINISHED",
        len(rows),
        "episodes;",
        sum(bool(r.get("success")) for r in rows),
        "successes",
        flush=True,
    )
    if len(rows) != len(args.tasks) * len(args.difficulties) * len(args.seeds) or any(
        not r["runtime_completed"] for r in rows
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
