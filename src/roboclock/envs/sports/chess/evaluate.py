"""Seeded four-ply physical chess evaluation with retained per-case evidence."""

import argparse
import itertools
import json
from pathlib import Path

from roboclock.envs.sports.chess.demo import run
from roboclock.envs.sports.arcade.demo import serial


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--difficulties",
        nargs="+",
        choices=["easy", "medium", "hard"],
        default=["easy", "medium", "hard"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = list(itertools.product(args.difficulties, args.seeds))
    (args.output / "manifest.json").write_text(json.dumps(dict(cases=cases), indent=2))
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    results = []
    with IsaacRuntime() as runtime:
        for difficulty, seed in cases:
            result = run(
                runtime,
                output=args.output / f"A1_{difficulty}_{seed}",
                difficulty=difficulty,
                seed=seed,
                video=args.video,
            )
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(serial(results), indent=2))


if __name__ == "__main__":
    main()
