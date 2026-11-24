"""B1 oxygen desaturation: actual-physics oracle check over levels x seeds, one simulator session.

Rendering happens only at reset, so the physics clock is never advanced by rendering. Records, per
episode: the response (alarm to the float settled in the band), the reach (alarm to first knob
contact turn), and any hand motion or knob turn before the alarm. Writes one JSON per episode and a
summary that reports physical success and within-deadline success separately.

python scripts/hospital/b1_oracle_batch.py --output results/hospital/B1/oracle
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


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "min": round(min(values), 3),
        "mean": round(float(np.mean(values)), 3),
        "max": round(max(values), 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--output", type=Path, default=Path("results/hospital/B1/oracle"))
    p.add_argument("--limit", type=float, help="Override response_limit (s) for every level")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    runtime = IsaacRuntime()
    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    rows = []
    env = None
    try:
        for level in a.levels:
            env = HospitalEnv(task_id="B1", difficulty=level, oracle=True, runtime=runtime)
            if a.limit is not None:
                env.config["tasks"]["B1"][level]["response_limit"] = [a.limit, a.limit]
            for seed in a.seeds:
                started = time.time()
                obs, _ = env.reset(seed=seed)
                task = env.task
                policy = OraclePickPlace(env)
                policy.reset(obs)
                start_ee = np.asarray(env.backend.state().ee_position, dtype=float)
                early_motion, early_turn = 0.0, 0.0
                while not env.done and task.time < task.spec.max_time:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                    if task.event_time is None and task.previous is not None:
                        early_motion = max(
                            early_motion, float(np.linalg.norm(task.previous.ee_position - start_ee))
                        )
                        early_turn = max(early_turn, abs(task.knob_angle))
                m = task.metrics()
                event = m["event_time"]
                row = {
                    "level": level,
                    "seed": seed,
                    "physical_success": bool(m["task_success"]),
                    "within_deadline": bool(m["temporal_success"]),
                    "joint_success": bool(m["joint_success"]),
                    "ordered_flow_l_min": m["ordered_flow_l_min"],
                    "final_flow_l_min": round(m["float_flow_l_min"], 3),
                    "response_limit_s": m["response_limit_s"],
                    "event_time_s": event,
                    "response_s": None if m["settled_time"] is None else m["settled_time"] - event,
                    "band_entry_s": None if m["band_entry_time"] is None else m["band_entry_time"] - event,
                    "first_turn_s": None if m["knob_moved_time"] is None else m["knob_moved_time"] - event,
                    "early_hand_motion_m": round(early_motion, 4),
                    "early_knob_turn_rad": round(early_turn, 4),
                    "min_spo2": round(m["min_spo2"], 2),
                    "final_spo2": round(m["final_spo2"], 2),
                    "violations": m["temporal_violations"],
                    "physical_failures": m["physical_failures"],
                    "wall_s": round(time.time() - started, 1),
                }
                rows.append(row)
                (a.output / f"{level}_{seed}.json").write_text(
                    json.dumps({"row": row, "metrics": m}, indent=2, default=plain)
                )
                print(json.dumps(row, default=plain), flush=True)
            env.close()
            env = None
        summary = {"simulator": simulator_version(), "episodes": len(rows), "levels": {}}
        for level in a.levels:
            sub = [r for r in rows if r["level"] == level]
            summary["levels"][level] = {
                "episodes": len(sub),
                "physical_success": sum(r["physical_success"] for r in sub),
                "within_deadline": sum(r["within_deadline"] for r in sub),
                "joint_success": sum(r["joint_success"] for r in sub),
                "response_limit_s": stats([r["response_limit_s"] for r in sub]),
                "response_s": stats([r["response_s"] for r in sub]),
                "band_entry_s": stats([r["band_entry_s"] for r in sub]),
                "first_turn_s": stats([r["first_turn_s"] for r in sub]),
                "max_early_hand_motion_m": max(r["early_hand_motion_m"] for r in sub),
                "max_early_knob_turn_rad": max(r["early_knob_turn_rad"] for r in sub),
            }
        summary["slowest_response_s"] = max(
            (r["response_s"] for r in rows if r["response_s"] is not None), default=None
        )
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2, default=plain))
        print(json.dumps(summary, indent=2, default=plain), flush=True)
    except Exception:  # runtime.close() exits the process before a traceback would print
        import traceback

        traceback.print_exc()
        print("FAILED", flush=True)
    finally:
        if env is not None:
            env.close()
        runtime.close()


if __name__ == "__main__":
    main()
