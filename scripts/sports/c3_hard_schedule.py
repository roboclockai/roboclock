"""Derive the sports C3 hard demand schedule from eager oracle runs.

Run the oracle with every demand owed from the start (one episode per process):

    python -m roboclock eval --env sports --task C3 --difficulty hard --seeds 1 --seed-start S \
        --policy oracle --oracle --output OUT/sS \
        --config src/roboclock/envs/sports/configs/c3_eager.yaml

then ``python scripts/sports/c3_hard_schedule.py OUT/*/episodes.jsonl``. Each ball is fired as
soon as it is accepted, so the oracle never waits for a free hopper cell and job k's time is
when its (k+1)-th ball was accepted. The hard schedule is the tightest one the slowest run
meets: preparation = latest first ball, interval = max_k (latest k-th ball - preparation) / k,
both rounded up to 0.01 s. Copy the printed values into ball_service/config.py by hand: they
are frozen and never derived at run time.
"""

import argparse
import json
import math
from pathlib import Path


def up(x, step=0.01):
    return math.ceil(round(x / step, 6)) * step


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("episodes", nargs="+", type=Path, help="episodes.jsonl of eager runs")
    args = p.parse_args()
    runs = []
    for path in args.episodes:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            override = row["native_configuration"]["service"].get("schedule_override")
            if not override or override.get("preparation", 1) > 0:
                raise SystemExit(f"{path}: not an eager run (configs/c3_eager.yaml)")
            times = row["job_times_s"]
            if None in times or not row["task_success"]:
                raise SystemExit(f"{path}: seed {row['seed']} did not stock every ball")
            runs.append((row["seed"], times))
            print(f"seed {row['seed']:3d}: " + " ".join(f"{t:7.3f}" for t in times))
    count = len(runs[0][1])
    # Physics ticks are 4 ms; rounding to 1 ms drops the clock's float noise (6.4400003 s).
    latest = [round(max(times[k] for _, times in runs), 3) for k in range(count)]
    slowest = [max(runs, key=lambda r: r[1][k])[0] for k in range(count)]
    print("latest:   " + " ".join(f"{t:7.3f}" for t in latest))
    print("seed:     " + " ".join(f"{s:7d}" for s in slowest))
    preparation = up(latest[0])
    need = [(latest[k] - preparation) / k for k in range(1, count)]
    interval = up(max(need))
    binding = 1 + need.index(max(need))
    print(f"ORACLE_ACCEPTED_S = ({', '.join(f'{t:.3f}' for t in latest)})")
    print(f"HARD_PREPARATION = {preparation:.2f}  # seed {slowest[0]}")
    print(f"HARD_INTERVAL = {interval:.2f}  # ball {binding + 1}, seed {slowest[binding]}")


if __name__ == "__main__":
    main()
