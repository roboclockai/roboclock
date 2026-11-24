"""RoboDojo-style tasks/doctor/smoke/eval/summarize entry points."""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from roboclock.config import load_config
from roboclock.envs.registry import create_env, environment_names, get_environment, import_symbol
from roboclock.envs.start_pose import START_POSES
from roboclock.evaluation.evaluator import (
    INFERENCE_LATENCY_MODES,
    TIME_MODELS,
    episode_plan,
    evaluate,
)
from roboclock.evaluation.metrics import aggregate
from roboclock.tasks.registry import make_task, task_registry


def _selection(value, choices):
    if value == "all":
        return list(choices)
    selected = value.split(",")
    unknown = set(selected) - set(choices)
    if unknown:
        raise ValueError(f"Unknown selection: {', '.join(sorted(unknown))}")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description="Isaac Sim temporal robotics benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("envs", help="List registered environments without launching the simulator")
    for name, help_text in (
        ("tasks", "List tasks in an environment without launching the simulator"),
        ("doctor", "Check runtime prerequisites and environment configuration"),
    ):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("--env", default="lab", help="Registered environment name")
    summary_parser = commands.add_parser(
        "summarize", help="Aggregate episodes.jsonl files, or any recorded results with --normalize"
    )
    summary_parser.add_argument("files", nargs="+", type=Path)
    summary_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Accept recording sidecars, domain-native result.json files and directories",
    )
    summary_parser.add_argument(
        "--env", help="Environment of every input (default: from the result or its path)"
    )
    summary_parser.add_argument("--rows", type=Path, help="Also write the normalized rows (JSONL)")
    for name in ("plan", "eval", "smoke", "render", "validate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--env", default="lab", help="Registered environment name")
        cmd.add_argument("--task")
        cmd.add_argument("--difficulty")
        cmd.add_argument("--seeds", type=int, default=10)
        cmd.add_argument("--seed-start", "--seed", type=int, default=0)
        cmd.add_argument("--config", type=Path)
        cmd.add_argument(
            "--control-delay",
            "--latency",
            dest="control_delay",
            type=float,
            nargs="+",
            default=[0.0],
            help="Fixed delay (s) charged per model query on top of inference latency",
        )
        cmd.add_argument(
            "--inference-budget",
            type=float,
            nargs="+",
            help="Inference budget L (s) per model query, for a latency ablation. The delay "
            "charged is L + --interface-delay, and the episode records L, the interface "
            "delay and their sum separately. Not combined with --control-delay",
        )
        cmd.add_argument(
            "--interface-delay",
            type=float,
            default=0.0,
            help="The control interface's own delay (s), charged on every model query. With "
            "--inference-budget, L = 0 still costs this much",
        )
        if name != "plan":
            cmd.add_argument(
                "--start-pose",
                choices=START_POSES,
                help="task: each task's own arm posture (the config default); zero: both arms "
                "start with joints 1-6 at 0 rad, RoboDojo's start pose (robot.start_pose)",
            )
            cmd.add_argument("--gui", action="store_true")
            cmd.add_argument("--depth", action="store_true")
            cmd.add_argument("--asset-catalog", type=Path)
        if name == "eval":
            cmd.add_argument("--policy", default="hold")
            cmd.add_argument(
                "--policy-hz",
                type=float,
                help="Policy rate; must divide the 50 Hz control rate (default: 50, also 25)",
            )
            cmd.add_argument("--oracle", action="store_true")
            cmd.add_argument(
                "--inference-latency",
                choices=INFERENCE_LATENCY_MODES,
                default="measured",
                help="measured: each query's inference time passes in simulation (default); "
                "zero: simulation pauses during inference; ideal: each query is charged its "
                "output tokens at the model's fastest surveyed decode speed (--model-speeds). "
                "hold and oracle always use zero",
            )
            cmd.add_argument(
                "--model-speeds",
                type=Path,
                help="YAML of surveyed decode speeds per served model for --inference-latency "
                "ideal (default: roboclock/evaluation/model_speeds.yaml)",
            )
            cmd.add_argument(
                "--time-model",
                choices=tuple(TIME_MODELS),
                default="block",
                help="block: the robot holds its targets while the policy computes, then acts "
                "(default); streaming: observations every policy period, the robot keeps "
                "executing the previous actions while the policy computes",
            )
            cmd.add_argument("--output", type=Path)
        if name == "smoke":
            cmd.add_argument("--steps", type=int, default=30)
        if name == "render":
            cmd.add_argument("--output", type=Path)
            cmd.add_argument("--usd", action="store_true")
        if name == "validate":
            cmd.add_argument("--output", type=Path)
            cmd.add_argument("--seconds", type=float, default=24.0)
            cmd.add_argument("--passive", action="store_true")
            cmd.add_argument("--images", action="store_true")
            cmd.add_argument("--rack-probes", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "envs":
            for name in environment_names():
                spec = get_environment(name)
                print(f"{name:16} {spec.description}")
            return 0
        if args.command == "tasks":
            for key, cls in task_registry(args.env).items():
                print(f"{key:2}  {cls.category:16}  {cls.discipline:10}  {cls.name}")
            return 0
        if args.command == "doctor":
            from roboclock.simulators.isaac.diagnostics import graphics_status, select_headless_driver
            from roboclock.simulators.isaac.version import installed_version, required_python

            spec = get_environment(args.env)
            config = load_config(environment=args.env)
            if spec.check_config:
                import_symbol(spec.check_config)(config)
            gpu = (
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if shutil.which("nvidia-smi")
                else None
            )
            available = importlib.util.find_spec("isaacsim") is not None
            sim_version = installed_version()
            expected_python = required_python(sim_version)
            driver = select_headless_driver() if sys.platform.startswith("linux") else None
            try:
                graphics = graphics_status() if sys.platform.startswith("linux") else None
                if graphics is not None:
                    graphics["driver_selection"] = (
                        "process-local NVIDIA EGL fallback" if driver else "environment/default"
                    )
            finally:
                if driver:
                    os.environ.pop("VK_ICD_FILENAMES", None)
                    os.unlink(driver)
            ok = (
                sys.version_info[:2] == expected_python
                and available
                and gpu is not None
                and gpu.returncode == 0
                and (
                    graphics is None
                    or (
                        all(graphics["libraries"].values())
                        and graphics["vulkan_gpu_enumeration"] is not False
                    )
                )
            )
            print(
                json.dumps(
                    {
                        "python": platform.python_version(),
                        "environment": args.env,
                        "required_python": ".".join(map(str, expected_python))
                        if expected_python
                        else None,
                        "isaacsim_version": sim_version,
                        "isaacsim_installed": available,
                        "gpu": gpu.stdout.strip() if gpu else "nvidia-smi unavailable",
                        "graphics": graphics,
                        "config_and_tasks": "valid",
                        "ready": ok,
                    },
                    indent=2,
                )
            )
            return 0 if ok else 1
        if args.command == "summarize":
            if args.normalize:
                from roboclock.evaluation.schema import collect_results, require_valid

                rows, skipped = collect_results(args.files, environment=args.env)
                require_valid(rows)
                if skipped:
                    print(f"roboclock: skipped {len(skipped)} non-result JSON files", file=sys.stderr)
                if args.rows:
                    args.rows.write_text(
                        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
                    )
            else:
                if args.env or args.rows:
                    raise ValueError("--env and --rows require --normalize")
                rows = []
                for file in args.files:
                    rows.extend(
                        json.loads(line) for line in file.read_text().splitlines() if line.strip()
                    )
            print(json.dumps(aggregate(rows), indent=2, allow_nan=False))
            return 0
        import math

        spec = get_environment(args.env)
        if args.command in ("render", "validate"):
            args.output = args.output or Path("results") / args.env / (
                "render.png" if args.command == "render" else "physics"
            )
        args.task = args.task or (
            spec.render_task
            if args.command == "render"
            else spec.default_task
            if args.command == "smoke"
            else "all"
        )
        args.difficulty = args.difficulty or (
            spec.difficulties[0] if args.command in ("smoke", "render") else "all"
        )
        if args.seeds < 1 or args.seed_start < 0:
            raise ValueError("--seeds must be positive and --seed-start nonnegative")
        if any(not math.isfinite(x) or x < 0 for x in args.control_delay):
            raise ValueError("--control-delay values must be finite and nonnegative")
        if not math.isfinite(args.interface_delay) or args.interface_delay < 0:
            raise ValueError("--interface-delay must be finite and nonnegative")
        budgets = getattr(args, "inference_budget", None)
        if budgets is not None:
            if args.control_delay != [0.0]:
                raise ValueError("--inference-budget and --control-delay are alternatives")
            if any(not math.isfinite(x) or x < 0 for x in budgets):
                raise ValueError("--inference-budget values must be finite and nonnegative")
            # D = L + delta. The episode records all three (roboclock.evaluation.ablations).
            args.control_delay = [budget + args.interface_delay for budget in budgets]
        tasks = _selection(args.task, task_registry(args.env))
        difficulties = _selection(args.difficulty, spec.difficulties)
        if args.command in ("smoke", "render") and (len(tasks) != 1 or len(difficulties) != 1):
            raise ValueError("smoke/render require exactly one task and difficulty")
        plan = episode_plan(
            tasks,
            difficulties,
            range(args.seed_start, args.seed_start + args.seeds),
            args.control_delay,
            environment=args.env,
            interface_delay=args.interface_delay,
            inference_budgets=budgets,
        )
        if args.command == "plan":
            config = load_config(args.config, environment=args.env)
            sampled = []
            for item in plan:
                task = make_task(
                    item["task_id"], item["difficulty"], item["seed"], config, environment=args.env
                )
                sampled.append({**item, "spec": asdict(task.spec)})
            print(json.dumps({"episodes": len(plan), "plan": sampled}, indent=2))
            return 0
        with import_symbol(spec.runtime_factory)(
            headless=not args.gui,
            renderer=load_config(args.config, environment=args.env)["visuals"].get(
                "renderer", "RaytracedLighting"
            ),
        ) as runtime:
            env = create_env(
                args.env,
                tasks[0],
                difficulties[0],
                runtime=runtime,
                config_path=args.config,
                asset_catalog=args.asset_catalog,
                depth=args.depth,
                oracle=getattr(args, "oracle", False),
                **({"start_pose": args.start_pose} if args.start_pose else {}),
            )
            try:
                if args.command == "smoke":
                    if args.steps < 1:
                        raise ValueError("--steps must be positive")
                    obs, _ = env.reset(seed=args.seed_start)
                    for _ in range(args.steps):
                        if env.done:
                            break
                        obs, *_ = env.step(obs["joint_positions"])
                    if not env.observation_space.contains(obs):
                        raise RuntimeError("Simulator observation does not match observation_space")
                    print(
                        json.dumps(
                            {
                                "smoke": "passed",
                                "environment": env.environment_id,
                                "rgb_shape": list(obs["rgb"].shape),
                                "physics_steps": env.task.steps,
                                "sim_time": env.task.time,
                            }
                        )
                    )
                elif args.command == "render":
                    from roboclock.evaluation.validation import export_render

                    env.reset(seed=args.seed_start)
                    print(export_render(env, args.output, usd=args.usd).resolve())
                elif args.command == "validate":
                    from roboclock.evaluation.validation import validate_physics

                    if not math.isfinite(args.seconds) or args.seconds <= 0:
                        raise ValueError("--seconds must be finite and positive")
                    result = validate_physics(
                        env,
                        plan,
                        args.output,
                        seconds=args.seconds,
                        controller=not args.passive,
                        images=args.images,
                        rack_probes=args.rack_probes,
                    )
                    print(
                        json.dumps(
                            {k: v for k, v in result.items() if k not in ("config", "plan")},
                            indent=2,
                        )
                    )
                    if result["physics_checks_passed"] != result["episodes"]:
                        runtime.exit_code = 1
                        return 1
                else:
                    run_id = (
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        + "_"
                        + uuid.uuid4().hex[:8]
                    )
                    output = args.output or Path("results") / args.env / run_id
                    result = evaluate(
                        env,
                        plan,
                        args.policy,
                        output,
                        inference_latency=args.inference_latency,
                        policy_hz=args.policy_hz,
                        time_model=args.time_model,
                        model_speeds=args.model_speeds,
                    )
                    print(json.dumps(result["overall"], indent=2))
                    print(f"Results: {output.resolve()}")
            finally:
                env.close()
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"roboclock: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
