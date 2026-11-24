"""Regenerate the reviewed Industry native-physics diagnostics on an RTX host.

Runs ``roboclock.envs.industry.native.runner`` (the merged native sources) in a fresh
Isaac Sim process, then exports the movie sidecar. Exact pre-merge reproduction
of the recordings is available from git commit dd853c8, and from d34b46b for Tasks
41 and 49 (see docs/industry/README.md).

Example: python -m roboclock.envs.industry.videos --task 49 --record \
    --output /opt/industry/results/industry/task49/replay
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from roboclock.envs.industry.native.catalog import TASKS
from roboclock.envs.industry.native.paths import SCENE_ASSETS

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parents[3]


def resolved_path(path):
    return Path(path).resolve()


def make_command(args):
    task = args.task
    assets = resolved_path(args.assets)
    if not (assets / "manifest.json").is_file():
        raise ValueError(f"Missing asset bundle: {assets}. See docs/industry/README.md.")
    output = resolved_path(args.output)
    if output.exists():
        raise ValueError(f"Choose a new output directory; refusing to overwrite {output}")
    # Native frontends create output themselves. Cache and scratch stay next to it.
    local_root = output.parent
    env = os.environ.copy()
    env.update(
        ROBOCLOCK_INDUSTRY_ASSET_ROOT=str(assets),
        ROBOCLOCK_INDUSTRY_LOCAL_ROOT=str(local_root),
        ROBOCLOCK_V2_RUNTIME_PROFILE=TASKS[task].runtime_profile,
        ROBOCLOCK_V2_RENDERER=TASKS[task].renderer,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        TMPDIR=str(local_root / ".industry-runtime" / "tmp"),
        XDG_CACHE_HOME=str(local_root / ".industry-runtime" / "cache"),
        ROBOCLOCK_ASSET_CACHE=str(local_root / ".industry-runtime" / "cache" / "roboclock"),
    )
    command = [
        sys.executable,
        "-m",
        "roboclock.envs.industry.native.runner",
        "--task",
        str(task),
        "--manifest",
        str(assets / "runtime/arx-x5/robot-v1/prepared-v2/manifest.json"),
        "--output",
        str(output),
        "--mode",
        args.mode,
        "--seconds",
        str(args.seconds),
        "--visual-context",
        args.visual_context,
    ]
    if task in SCENE_ASSETS:
        command += ["--assets", str(assets / "runtime" / SCENE_ASSETS[task])]
    randomized = args.seed is not None or any(
        [args.randomize_layout, args.randomize_timing, args.randomize_physics]
    )
    if args.replay and randomized:
        raise ValueError("Choose replay OR fresh seeded randomization, not both.")
    if args.replay or not randomized:
        replay = args.replay or PACKAGE / f"configs/task{task}/reviewed.json"
        command += ["--replay", str(Path(replay).resolve())]
    else:
        command += ["--seed", str(args.seed if args.seed is not None else 0)]
        for flag in ["randomize_layout", "randomize_timing", "randomize_physics"]:
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
    if args.record:
        command.append("--record")
    return command, env, output, local_root


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=int, choices=range(41, 51), required=True)
    p.add_argument("--assets", type=Path, default=REPOSITORY / "assets/industry")
    p.add_argument("--output", type=Path)
    p.add_argument("--seconds", type=float, default=90)
    p.add_argument(
        "--mode",
        default="nominal",
        choices=["nominal", "passive", "early", "late", "absent", "grasp", "seat", "delivery"],
    )
    p.add_argument("--record", action="store_true")
    p.add_argument("--visual-context", choices=["minimal", "debug"], default="minimal")
    p.add_argument("--replay", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--randomize-layout", action="store_true")
    p.add_argument("--randomize-timing", action="store_true")
    p.add_argument("--randomize-physics", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved invocation without importing Isaac Sim.",
    )
    args = p.parse_args(argv)
    if not 0 < args.seconds <= 120:
        p.error("seconds must be greater than0 and at most120")
    if args.record and args.mode != "nominal":
        p.error("Record the nominal workflow; negative modes are diagnostics only.")
    if args.output is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = REPOSITORY / f"results/industry/task{args.task}/{stamp}"
    try:
        command, env, output, local_root = make_command(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "environment": {
                        k: env[k]
                        for k in env
                        if k.startswith("ROBOCLOCK_") or k in ["TMPDIR", "XDG_CACHE_HOME"]
                    },
                },
                indent=2,
            )
        )
        return 0
    if not shutil.which("ffmpeg") and args.record:
        p.error("ffmpeg is required to record MP4.")
    for key in ["TMPDIR", "XDG_CACHE_HOME", "ROBOCLOCK_ASSET_CACHE"]:
        resolved_path(env[key]).mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, env=env, cwd=REPOSITORY)
    if process.returncode:
        return process.returncode
    result = json.loads((output / "result.json").read_text())
    resolved = json.loads((output / "resolved.json").read_text())
    instruction = next(
        r
        for r in json.loads((PACKAGE / "configs/tasks.json").read_text())["task_descriptions"]
        if r["task_id"] == args.task
    )
    summary = {
        "environment": "industry",
        "robot": "dual_arx_x5",
        "task_id": args.task,
        "instruction": instruction["instruction"],
        "time_requirement": instruction["time_requirement"],
        "seed": resolved["seed"],
        "episode_spec": resolved,
        "native_result": "result.json",
        "policy": "privileged_native_contact_ik_diagnostic",
        # The same scenes are evaluated as roboclock's "industry" BenchmarkEnv.
        "benchmark_environment": "roboclock.envs.industry.env:IndustryEnv",
        "mobile_base_required": False,
        "task_success": bool(result.get("task_success")),
        "scope": "Native fixed-workcell diagnostic; consult result.json for the actual physical/temporal outcome. No full seed/difficulty matrix claim.",
        "review_trim_start_s": {44: 19.5, 45: 13.5}.get(args.task, 0),
    }
    from roboclock.envs.industry.export import export_movies

    summary["video_metadata"] = export_movies(
        output, args.task, instruction, result, resolved, seconds=args.seconds
    )
    (output / "episode.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "task_success": summary["task_success"],
                "native_failure": result.get("failure"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
