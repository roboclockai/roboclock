"""Independently replay B4 predicates from full physics-rate measured traces."""

import argparse
import gzip
import json
import lzma
from pathlib import Path
from types import SimpleNamespace

from roboclock.envs.sports.freeze.task import Task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "result.json").read_text())
    config = json.loads((args.run / "config.json").read_text())
    trace = next(
        (
            args.run / name
            for name in ("trace.jsonl.gz", "trace.jsonl.xz")
            if (args.run / name).is_file()
        )
    )
    b = SimpleNamespace(
        config=config,
        time=0.0,
        filter_paths=[
            "left_finger_0",
            "left_finger_1",
            "right_finger_0",
            "right_finger_1",
            "/World/Rink",
            "/World/Table",
            "/World/Objects/puck",
            "/World/Objects/left_pusher",
            "/World/Objects/right_pusher",
        ],
    )
    task = Task(b)
    samples = 0
    maximum = {}
    with (gzip.open if trace.suffix == ".gz" else lzma.open)(trace, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("phase") == "post_roll":
                break
            b.time = row["time"]
            task.update(row["state"])
            samples += 1
            red = task.red()
            if red is not None and task.reactions[-1]["stop_time"] is not None:
                for body, speed in task.current_speeds.items():
                    maxima = maximum.setdefault(body, dict(linear=0.0, angular=0.0))
                    for key in maxima:
                        maxima[key] = max(maxima[key], speed[key])
    replay = task.result()
    for key in (
        "success",
        "physical_success",
        "temporal_success",
        "failures",
        "events",
        "reactions",
        "rounds_completed",
    ):
        assert replay[key] == result[key], key
    output = dict(
        scope="Independent replay of unchanged contact/reaction/hold predicates; full recorded physics trace.",
        matching_evaluation=True,
        samples=samples,
        success=replay["success"],
        physical=replay["physical_success"],
        temporal=replay["temporal_success"],
        reactions=replay["reactions"],
        failures=replay["failures"],
        peak_post_confirmation_speeds=maximum,
    )
    target = args.output or args.run / "trace_audit.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{args.run}: replay matches; success={replay['success']}; samples={samples}")


if __name__ == "__main__":
    main()
