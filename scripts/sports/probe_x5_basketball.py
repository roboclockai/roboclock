"""Run an unrendered X5 shot probe with actual contact/flight scoring."""

import argparse
import json
from pathlib import Path

from roboclock.envs.sports.arena.config import config
from roboclock.envs.sports.arena.provenance import source_hashes
from roboclock.simulators.isaac.embodiments.config import configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c = configure(config("A3", args.difficulty, args.seed), "A3")
    if args.overrides:
        c["basketball"].update(json.loads(args.overrides.read_text()))
    (args.output / "config.json").write_text(json.dumps(c, indent=2))
    hashes = source_hashes()
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        from roboclock.envs.sports.arcade.demo import serial
        from roboclock.envs.sports.arena.oracle import Oracle
        from roboclock.envs.sports.arena.tasks import Task
        from roboclock.simulators.isaac.embodiments.backend import ReplacementBackend

        b = ReplacementBackend(runtime, c)
        b.reset()
        task = Task(b)
        oracle = Oracle(b, task)
        with (args.output / "trajectory.jsonl").open("w") as f:
            while not task.done and b.time < 30:
                oracle.act()
                for _ in range(c["sim"]["control_decimation"]):
                    state = b.tick()
                    task.update(state)
                    if task.done:
                        break
                f.write(
                    json.dumps(
                        serial(
                            dict(
                                time=b.time,
                                phase=oracle.phase,
                                ball=state["objects"]["ball"],
                                hand=state["hands"]["main"],
                                joints=state["joints"],
                                command=b.command.copy(),
                            )
                        )
                    )
                    + "\n"
                )
        result = task.result()
        result.update(
            full_episode=task.done,
            configuration=c,
            source_hashes=hashes,
            validation_scope="Unrendered PhysX episode; not visual validation",
        )
        (args.output / "result.json").write_text(json.dumps(serial(result), indent=2))
        print("SHOT_RESULT", json.dumps(serial(result)), flush=True)


if __name__ == "__main__":
    main()
