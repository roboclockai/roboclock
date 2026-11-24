"""Unified entry point for the fixed-base sports tasks."""

import argparse
import importlib
import sys

from roboclock.simulators.isaac.embodiments.config import ASSIGNMENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(ASSIGNMENTS), required=True)
    parser.add_argument(
        "--embodiment",
        choices=["dual_arx_x5"],
    )
    known, remaining = parser.parse_known_args()
    model = known.embodiment or ASSIGNMENTS[known.task]
    if known.task == "B4":
        module = "roboclock.envs.sports.freeze.demo"
        task_args = []
    elif known.task == "C1":
        module = "roboclock.envs.sports.pool_rack.demo"
        task_args = []
    elif known.task == "C3":
        module = "roboclock.envs.sports.ball_service.demo"
        task_args = []
    elif known.task == "A1":
        module = "roboclock.envs.sports.chess.demo"
        task_args = []
    elif known.task in ("A2", "B1"):
        module = "roboclock.envs.sports.arcade.demo"
        task_args = ["--task", known.task]
    elif known.task in ("A3", "B2", "C2"):
        module = "roboclock.envs.sports.arena.demo"
        task_args = ["--task", known.task]
    else:
        module = "roboclock.envs.sports._shared.precision.demo"
        task_args = ["--task", known.task]
    sys.argv = [module, *task_args, "--embodiment", model, *remaining]
    importlib.import_module(module).main()


if __name__ == "__main__":
    main()
