"""Run fixed-robot integration/evaluation cases in isolated Isaac processes."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from roboclock.simulators.isaac.embodiments.config import ASSIGNMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=list(ASSIGNMENTS), default=list(ASSIGNMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--difficulties", nargs="+", choices=["easy", "medium", "hard"], default=["easy"]
    )
    parser.add_argument(
        "--duration", type=float, help="Diagnostic cap; omission runs each full task budget"
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for task in args.tasks:
        for difficulty in args.difficulties:
            for seed in args.seeds:
                name = f"{task}-{difficulty}-seed{seed}"
                destination = args.output / name
                command = [
                    sys.executable,
                    "-m",
                    "roboclock.envs.sports.demo",
                    "--task",
                    task,
                    "--difficulty",
                    difficulty,
                    "--seed",
                    str(seed),
                    "--output",
                    str(destination),
                ]
                if args.duration is not None:
                    command += ["--duration", str(args.duration)]
                if args.video:
                    command += ["--video"]
                with (args.output / (name + ".log")).open("w") as log:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                result_path = destination / "result.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                metrics = result.get("metrics", result)
                case = dict(
                    task=task,
                    difficulty=difficulty,
                    seed=seed,
                    model=ASSIGNMENTS[task],
                    command=command,
                    exit_code=code,
                    result=str(result_path),
                    full_episode=result.get("full_episode", False),
                    success=metrics.get("success", metrics.get("joint_success", False)),
                    failures=metrics.get("failures", metrics.get("physical_failures", [])),
                    temporal_violations=metrics.get("temporal_violations", []),
                )
                cases.append(case)
                (args.output / "index.json").write_text(json.dumps(cases, indent=2) + "\n")
                print(json.dumps(case), flush=True)
    if any(case["exit_code"] != 0 for case in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
