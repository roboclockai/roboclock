#!/usr/bin/env python3
"""Run experiment 1 of the RoboClock ablations: temporal difficulty x inference budget.

The matrix is ``roboclock.evaluation.ablations.runs()`` crossed with the models and the seeds:
six tasks, three temporal tiers each (lab C2 judged at three tiers from one rollout), four
inference budgets L. Every cell is a fresh closed-loop episode.

    scripts/ablations/run_matrix.py plan                 # the matrix and its manifest, no GPU
    scripts/ablations/run_matrix.py check                # paired instances + instructions
    scripts/ablations/run_matrix.py run --models nv/... --gpus 0,1,2,3

``run`` is resumable: a cell whose output already holds an episode is skipped, so an
interrupted matrix is finished by running the same command again.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from roboclock.envs.registry import get_environment, import_symbol  # noqa: E402
from roboclock.evaluation import ablations as ab  # noqa: E402

EPISODE = Path(__file__).resolve().parent / "episode.sh"


def model_dir_name(model):
    return model.replace("/", "_")


class Cell:
    """One episode of the matrix."""

    def __init__(self, model, condition, tier, budget, seed, root):
        self.model = model
        self.condition = condition
        self.tier = tier  # None for a by_tier condition: one run judged at every tier
        self.budget = budget
        self.seed = seed
        self.root = root

    @property
    def tier_label(self):
        return self.tier.label if self.tier else "all"

    @property
    def difficulty(self):
        # A by_tier condition runs at one level; the tiers only pick the scoring budget, and
        # the episode is identical at each (lab throughput_episode).
        return self.tier.difficulty if self.tier else self.condition.tiers[0].difficulty

    @property
    def name(self):
        return (
            f"{self.condition.environment}_{self.condition.task_id}_{self.tier_label}"
            f"_L{self.budget:g}_s{self.seed}"
        )

    @property
    def out_dir(self):
        return self.root / model_dir_name(self.model) / "out" / self.name

    def done(self):
        episodes = self.out_dir / "episodes.jsonl"
        return episodes.is_file() and episodes.stat().st_size > 0

    def row(self):
        return {
            "model": self.model,
            "environment": self.condition.environment,
            "task_id": self.condition.task_id,
            "tier": self.tier_label,
            "difficulty": self.difficulty,
            "scoring": self.condition.scoring,
            "inference_budget_s": self.budget,
            "seed": self.seed,
            "instance_id": ab.instance_id(
                self.condition.environment, self.condition.task_id, self.seed
            ),
            "name": self.name,
        }


def build_configs(root, keys=None):
    """Write one domain configuration per tier that needs a patch; return {(key, tier): path}."""
    out = {}
    config_dir = root / "configs"
    for condition in ab.conditions(keys):
        for tier in condition.tiers:
            patch = condition.tier_patch(tier)
            if not patch:
                continue
            path = ab.write_config(
                condition.environment,
                patch,
                config_dir / f"{condition.environment}_{condition.task_id}_{tier.label}.yaml",
            )
            out[(condition.key, tier.label)] = path
    return out


def loaded(environment, path):
    loader = import_symbol(get_environment(environment).config_loader)
    return loader(path) if path else None


def cells(models, seeds, root, keys=None, budgets=ab.BUDGETS_S):
    for model in models:
        for condition, tier, budget in ab.runs(keys, budgets):
            for seed in seeds:
                yield Cell(model, condition, tier, budget, seed, root)


def episode_length(condition, tier, difficulty, config):
    spec = ab.resolved_spec(condition.environment, condition.task_id, difficulty, 0, config)
    return float(spec.get("max_time") or spec.get("budget_s") or 0.0)


def write_manifest(root, models, seeds, configs, keys=None, budgets=ab.BUDGETS_S):
    """The run's manifest: every cell, its instance hash and what the model is told.

    Written before any episode runs, so the paired-instance and instruction checks are on
    record independently of what the episodes then do.
    """
    rows = []
    for condition in ab.conditions(keys):
        for tier in condition.tiers:
            path = configs.get((condition.key, tier.label))
            config = loaded(condition.environment, path)
            for seed in seeds:
                rows.append(
                    {
                        "condition": condition.key,
                        "tier": tier.label,
                        "difficulty": tier.difficulty,
                        "seed": seed,
                        "instance_id": ab.instance_id(
                            condition.environment, condition.task_id, seed
                        ),
                        "physical_spec_hash": ab.instance_hash(
                            condition.environment,
                            condition.task_id,
                            tier.difficulty,
                            seed,
                            config,
                        ),
                        "statement": ab.statement(
                            condition.environment,
                            condition.task_id,
                            tier.difficulty,
                            seed,
                            config,
                        ),
                        "config": str(path) if path else None,
                        "patch": condition.tier_patch(tier),
                        "episode_length_s": episode_length(
                            condition, tier, tier.difficulty, config
                        ),
                    }
                )
    manifest = {
        "experiment": "roboclock-ablation-1-temporal-difficulty-x-latency",
        "source_commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "models": list(models),
        "seeds": list(seeds),
        "inference_budgets_s": list(budgets),
        "interface_delay_s": float(os.environ.get("ABL_INTERFACE_DELAY", ab.INTERFACE_DELAY_S)),
        "inference_latency_mode": "zero",
        "time_model": "block",
        "conditions": [
            {
                "key": c.key,
                "scoring": c.scoring,
                "note": c.note,
                "tiers": [{"label": t.label, "difficulty": t.difficulty, "patch": t.patch} for t in c.tiers],
            }
            for c in ab.conditions(keys)
        ],
        "instances": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def check(root, seeds, keys=None):
    """The CPU-only acceptance checks, on the seeds this run uses."""
    configs = build_configs(root, keys)
    problems = []
    for condition in ab.conditions(keys):
        per_tier = {
            tier.label: loaded(condition.environment, configs.get((condition.key, tier.label)))
            for tier in condition.tiers
        }
        for seed in seeds:
            digests = {
                tier.label: ab.instance_hash(
                    condition.environment,
                    condition.task_id,
                    tier.difficulty,
                    seed,
                    per_tier[tier.label],
                )
                for tier in condition.tiers
            }
            if len(set(digests.values())) != 1:
                problems.append(f"{condition.key} seed {seed}: instances differ {digests}")
        statements = {
            tier.label: ab.statement(
                condition.environment,
                condition.task_id,
                tier.difficulty,
                seeds[0],
                per_tier[tier.label],
            )
            for tier in condition.tiers
        }
        distinct = len(set(statements.values()))
        want = 1 if condition.scoring == "by_tier" else len(condition.tiers)
        if distinct != want:
            problems.append(
                f"{condition.key}: {distinct} distinct instructions over its tiers, want {want}"
            )
        if condition.key != "sports/C3":
            lengths = {
                round(episode_length(condition, t, t.difficulty, per_tier[t.label]), 6)
                for t in condition.tiers
            }
            if len(lengths) != 1:
                problems.append(f"{condition.key}: episode lengths differ {lengths}")
    for line in problems:
        print("FAIL", line)
    if not problems:
        print(f"OK  {len(ab.conditions(keys))} conditions, seeds {list(seeds)}: "
              "paired instances, tier-tracking instructions, one episode length per task")
    return 1 if problems else 0


def run_cell(cell, gpu, slot, configs, env):
    path = configs.get((cell.condition.key, cell.tier_label))
    if path is None and cell.tier is not None:
        path = configs.get((cell.condition.key, cell.tier.label))
    if path is None and cell.condition.patch:
        # A by_tier condition still carries the condition-wide patch, if it has one.
        path = configs.get((cell.condition.key, cell.condition.tiers[0].label))
    config = loaded(cell.condition.environment, path)
    length = episode_length(cell.condition, cell.tier, cell.difficulty, config)
    args = [
        "bash",
        str(EPISODE),
        cell.model,
        str(gpu),
        cell.condition.environment,
        cell.condition.task_id,
        cell.difficulty,
        str(cell.seed),
        f"{cell.budget:g}",
        f"{length:g}",
        cell.name,
        str(cell.out_dir),
        str(path) if path else "",
    ]
    child = dict(env, SWEEP_SLOT=str(slot))
    return subprocess.run(args, env=child).returncode


def worker(queue, gpu, slot, configs, env, results, lock):
    while True:
        cell = queue.get()
        if cell is None:
            queue.task_done()
            return
        start = time.time()
        try:
            rc = run_cell(cell, gpu, slot, configs, env)
        except Exception as error:  # noqa: BLE001 - one cell must not stop the matrix
            rc, error_text = 1, str(error)
        else:
            error_text = None
        with lock:
            results.append(
                {**cell.row(), "rc": rc, "wall_s": round(time.time() - start, 1),
                 "error": error_text}
            )
            state = "ok" if rc == 0 else f"rc={rc}"
            print(f"[{time.strftime('%H:%M:%S')}] {cell.model} {cell.name} {state} "
                  f"{round(time.time() - start)}s ({len(results)} done)", flush=True)
        queue.task_done()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "check", "run"))
    parser.add_argument("--root", type=Path, default=Path(os.path.expandvars("$WORK_DIR/roboclock_ablation1")))
    parser.add_argument("--models", default="", help="comma-separated, e.g. nv/<id>,qwen38")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--conditions", default="", help="comma-separated 'env/TASK' subset")
    parser.add_argument("--budgets", default="", help="comma-separated L values (s)")
    parser.add_argument("--interface-delay", type=float, default=ab.INTERFACE_DELAY_S)
    parser.add_argument("--max-cells", type=int, help="run at most this many cells (a pilot)")
    args = parser.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    keys = [k for k in args.conditions.split(",") if k] or None
    budgets = tuple(float(x) for x in args.budgets.split(",") if x) or ab.BUDGETS_S
    models = [m for m in args.models.split(",") if m]
    root = args.root

    if args.command == "check":
        return check(root, seeds, keys)

    configs = build_configs(root, keys)
    if args.command == "plan":
        manifest = write_manifest(root, models or ["<model>"], seeds, configs, keys, budgets)
        per_model = len(list(ab.runs(keys, budgets))) * len(seeds)
        print(json.dumps({
            "root": str(root),
            "conditions": len(manifest["conditions"]),
            "budgets_s": list(budgets),
            "seeds": seeds,
            "episodes_per_model": per_model,
            "episodes_total": per_model * max(len(models), 1),
            "manifest": str(root / "manifest.json"),
        }, indent=2))
        for cell in cells(models or ["<model>"], seeds, root, keys, budgets):
            print(" ", cell.name)
        return 0

    if not models:
        parser.error("run needs --models")
    write_manifest(root, models, seeds, configs, keys, budgets)
    for sub in ("logs", "qlog"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        ABL_ROOT=str(root),
        ABL_LOGS=str(root / "logs"),
        ABL_QLOG=str(root / "qlog"),
        ABL_INTERFACE_DELAY=str(args.interface_delay),
        SWEEP_ROOT=str(root),
    )
    pending = [c for c in cells(models, seeds, root, keys, budgets) if not c.done()]
    if args.max_cells:
        pending = pending[: args.max_cells]
    print(f"{len(pending)} cells to run into {root}", flush=True)
    queue = Queue()
    for cell in pending:
        queue.put(cell)
    gpus = [g for g in args.gpus.split(",") if g]
    results, lock, threads = [], threading.Lock(), []
    slot = 0
    for gpu in gpus:
        for _ in range(args.slots_per_gpu):
            thread = threading.Thread(
                target=worker, args=(queue, gpu, slot, configs, env, results, lock), daemon=True
            )
            thread.start()
            threads.append(thread)
            slot += 1
    for _ in threads:
        queue.put(None)
    queue.join()
    (root / "cells.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8"
    )
    failed = [r for r in results if r["rc"] != 0]
    print(f"done: {len(results) - len(failed)} ok, {len(failed)} failed -> {root/'cells.jsonl'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
