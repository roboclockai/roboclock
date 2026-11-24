"""Audit sampled task definitions only; this does not run or certify physics.

Covers the A1 openings, the basketball (A2, native kind A3) onset and pickup, and since
2026-11-20 the seeded station offsets of A1, B1, B2, B3, B5 and C2 (``sports/spatial.py``):
replayable, the same at every difficulty, within their bounds and distinct across seeds.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import chess
import numpy as np

from roboclock.envs.sports import spatial
from roboclock.envs.sports.arena.config import config as shot_config
from roboclock.envs.sports.chess.config import config as chess_config
from roboclock.envs.sports.tasks import make_task


def _flat(record):
    """Every offset of a spatial record as one flat list (m)."""
    values = []
    for key, value in sorted(record.items()):
        if key in ("version", "generator", "stream"):
            continue
        if isinstance(value, dict):
            values += [v for name in sorted(value) for v in value[name]]
        else:
            values += list(value) if isinstance(value, list) else [value]
    return values


def audit_spatial(count):
    """Offsets of the spatially randomized tasks: replay, tier independence, bounds, spread."""
    bounds = {
        **{task: list(half) for task, half in spatial.STATION.items()},
        # In _flat's (sorted-key) order: bar_offset x/y, bench_dx, rack_dx, rack_dz.
        "B3": [spatial.BENCH[k] for k in ("bar_x", "bar_y", "bench_x", "rack_x", "rack_height")],
        "C2": np.add(spatial.CUPS["row"], spatial.CUPS["cup"]).tolist() * 6,
    }
    report = {}
    for task_id in spatial.TASK_KEYS:
        samples = []
        for seed in range(count):
            layouts = [make_task(task_id, d, seed).spec.layout for d in ("easy", "medium", "hard")]
            assert layouts[0] == layouts[1] == layouts[2], (task_id, seed)
            assert layouts[0] == make_task(task_id, "easy", seed).spec.layout
            samples.append(_flat(layouts[0]))
        samples = np.asarray(samples)
        assert np.all(np.abs(samples).max(axis=0) > 0), task_id
        assert np.all(np.abs(samples) <= np.asarray(bounds[task_id]) + 1e-9), task_id
        assert len({tuple(s) for s in samples.tolist()}) == count, task_id
        report[task_id] = dict(
            bounds=bounds[task_id],
            min=samples.min(axis=0).round(4).tolist(),
            max=samples.max(axis=0).round(4).tolist(),
            distinct=count,
        )
    return report


def audit(count):
    scenarios, starts, positions = [], [], []
    moves = Counter()
    for seed in range(count):
        chess_cfg = chess_config(seed=seed)
        shot_cfg = shot_config("A3", seed=seed)
        assert chess_cfg == chess_config(seed=seed)
        assert shot_cfg == shot_config("A3", seed=seed)
        history = chess_cfg["episode"]["history"]
        sequence = chess_cfg["duel"]["moves"]
        board = chess.Board()
        for uci in history + sequence:
            board.push_uci(uci)
            assert board.is_valid() and not board.is_game_over()
        for difficulty in ("medium", "hard"):
            other = chess_config(difficulty, seed)
            assert other["duel"] == chess_cfg["duel"]
            assert other["episode"] == chess_cfg["episode"]
            other = shot_config("A3", difficulty, seed)
            assert other["randomization"] == shot_cfg["randomization"]
            assert other["basketball"] == shot_cfg["basketball"]
        scenarios.append(tuple(history + sequence))
        moves.update(sequence)
        starts.append(shot_cfg["workflow"]["shot_start"])
        positions.append(shot_cfg["basketball"]["pickup_xy"])
    payload = dict(openings=scenarios, shot_start=starts, pickup_xy=positions)
    station = audit_spatial(count)
    return dict(
        scope="Configuration audit only; no Isaac Sim episodes",
        seeds=list(range(count)),
        difficulties=["easy", "medium", "hard"],
        replay_equal=True,
        legal_openings=True,
        difficulty_preserves_instance=True,
        chess=dict(distinct_openings=len(set(scenarios)), move_counts=dict(moves)),
        basketball=dict(
            distinct_ready_times=len(set(starts)),
            ready_range=[min(starts), max(starts)],
            pickup_min=np.min(positions, axis=0).tolist(),
            pickup_max=np.max(positions, axis=0).tolist(),
        ),
        spatial=dict(version=spatial.VERSION, tasks=station),
        sample_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        samples=payload,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.count < 1:
        p.error("count must be positive")
    report = audit(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
