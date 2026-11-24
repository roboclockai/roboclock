"""A2 sequential bag switch: real-physics oracle runs over levels x seeds in one Isaac session.

python scripts/hospital/a2w_oracle_batch.py --output results/hospital/A2/w_oracle \
    --summary results/hospital/A2/oracle_summary.json --late-frame results/hospital/A2/late_air.png

Per episode: <level>_<seed>.json (metrics, events, episode metadata) and the task camera's initial
and final frames. The summary reports physical success (the measured detent) and within-window
success separately, the action time relative to the window (seconds and as a fraction of the
width), the frozen quantity and its outcome, and the measured actuation duration (oracle pinch
command to detent entry). --late-frame also runs one deliberately late episode (hard, seed 0, no
action) and saves its last frame, showing air in bag 1's line. Rendering happens only at the start
and the end of an episode, so it never advances the physics clock.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def run_episode(env, level, seed, out, idle=False, frames=True):
    from PIL import Image

    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    env.difficulty = level
    started = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    tag = f"{level}_{seed}" + ("_late" if idle else "")
    if frames:
        Image.fromarray(obs["rgb"]).save(out / f"{tag}_initial.png")
    policy = OraclePickPlace(env)
    policy.reset(obs)
    hold = env.backend.joint_positions().copy()
    task = env.task
    while not env.done:
        action = hold if idle else policy.act_internal({"joint_positions": env.backend.joint_positions()})
        env.backend.apply_action(action)
        task.action_count += 1
        env._ticks(env.config["sim"]["control_decimation"], render=False)
    final = None
    if frames:
        env.backend.capture_camera(subframes=4)
        final = env.render()
        Image.fromarray(final).save(out / f"{tag}_final.png")
    m = task.metrics()
    (out / f"{tag}.json").write_text(
        json.dumps(
            {
                "metrics": m,
                "events": task.event_log,
                "episode": env.episode_metadata(),
                "wall_seconds": time.perf_counter() - started,
            },
            default=plain,
            indent=1,
        )
    )
    start = next((e["time"] for e in task.event_log if e["type"] == "oracle_switch_start"), None)
    width = task.window_close - task.window_open
    act = task.action_time
    row = {
        "level": level,
        "seed": seed,
        "window": [round(task.window_open, 3), round(task.window_close, 3)],
        "window_width_s": round(width, 3),
        "physical_success": bool(m["task_success"]),
        "within_window": bool(m["temporal_success"]),
        "joint_success": bool(m["joint_success"]),
        "action_time": None if act is None else round(act, 3),
        "action_after_open_s": None if act is None else round(act - task.window_open, 3),
        "action_before_close_s": None if act is None else round(task.window_close - act, 3),
        "action_fraction_of_window": None if act is None else round((act - task.window_open) / width, 3),
        "actuation_duration_s": None if act is None or start is None else round(act - start, 3),
        "device_moved_time": None if task.device_moved_time is None else round(task.device_moved_time, 3),
        "final_quantity": round(m["final_quantity"], 2),
        "quantity_outcome": m["quantity_outcome"],
        "violations": m.get("temporal_violations"),
        "collisions": m.get("collision_count"),
        "drops": m.get("drops"),
    }
    print("A2", json.dumps(row, default=plain), flush=True)
    return row, final


def level_counts(rows):
    acts = [r["action_fraction_of_window"] for r in rows if r["action_fraction_of_window"] is not None]
    durations = [r["actuation_duration_s"] for r in rows if r["actuation_duration_s"] is not None]
    return {
        "episodes": len(rows),
        "physical_success": sum(r["physical_success"] for r in rows),
        "within_window": sum(r["within_window"] for r in rows),
        "in_band_quantity": sum(r["quantity_outcome"] == "in_band" for r in rows),
        "window_width_s": rows[0]["window_width_s"] if rows else None,
        "action_fraction_range": [min(acts), max(acts)] if acts else None,
        "action_after_open_range_s": [
            min(r["action_after_open_s"] for r in rows if r["action_after_open_s"] is not None),
            max(r["action_after_open_s"] for r in rows if r["action_after_open_s"] is not None),
        ] if acts else None,
        "actuation_duration_range_s": [min(durations), max(durations)] if durations else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=Path, default=Path("results/hospital/A2/w_oracle"))
    p.add_argument("--summary", type=Path, default=None, help="also write the summary here")
    p.add_argument("--late-frame", type=Path, default=None)
    p.add_argument("--no-frames", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.bedside import window_bags
    from roboclock.envs.hospital.env import HospitalEnv

    rows, late = [], None
    try:
        env = HospitalEnv(task_id="A2", difficulty=a.levels[0], oracle=True, runtime=runtime)
        for level in a.levels:
            for seed in a.seeds:
                rows.append(run_episode(env, level, seed, a.output, frames=not a.no_frames)[0])
        if a.late_frame is not None:
            late, frame = run_episode(env, "hard", 0, a.output, idle=True, frames=True)
            a.late_frame.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(a.late_frame)
        env.close()
    finally:
        summary = {
            "task": "A2 sequential_bag_switch",
            "simulator": "isaacsim==" + simulator_version(),
            "policy": "privileged oracle (bedside/window_bags.py::command): prepared straddle, "
            "pinch at window middle minus ACTUATION_DURATION, one quarter turn",
            "planned_actuation_duration_s": window_bags.ACTUATION_DURATION,
            "levels": {lv: level_counts([r for r in rows if r["level"] == lv]) for lv in a.levels},
            "episodes": rows,
            "late_no_action_check": late,
        }
        text = json.dumps(summary, indent=1, default=plain)
        (a.output / "summary.json").write_text(text)
        if a.summary is not None:
            a.summary.parent.mkdir(parents=True, exist_ok=True)
            a.summary.write_text(text)
        print(json.dumps(summary["levels"], indent=1), flush=True)
        runtime.close()


if __name__ == "__main__":
    main()
