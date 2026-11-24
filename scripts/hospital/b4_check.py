"""B4 line switch: actual-physics oracle check over levels x seeds in one simulator session.

Per episode it writes result.json (summary row, metrics, event log, planner phase timeline) and,
with --render, initial.png / final.png; summary.json holds physical and within-deadline counts
separately. Renders happen only before the first action and after the last, so the physics clock
is never advanced by rendering. --set NAME=VALUE overrides an oracle constant of
roboclock.envs.hospital.bedside.stopcock for speed probes.

python scripts/hospital/b4_check.py --levels easy medium hard --seeds 0 1 2 3 4 \
    --output results/hospital/B4/check
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


def rel(task, timestamp):
    if timestamp is None or task.event_time is None:
        return None
    return round(timestamp - task.event_time, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=Path, default=Path("results/hospital/B4/check"))
    p.add_argument("--after", type=float, default=1.0, help="Seconds simulated after the deadline")
    p.add_argument("--render", action="store_true", help="Save initial.png and final.png")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.envs.hospital.bedside import stopcock as module

    for item in a.set:
        key, _, value = item.partition("=")
        if not hasattr(module, key):
            raise SystemExit(f"Unknown B4 constant {key}")
        old = getattr(module, key)
        setattr(module, key, type(old)(eval(value)))
        print("set", key, "=", getattr(module, key), flush=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    rows = []
    try:
        env = HospitalEnv(task_id="B4", difficulty=a.levels[0], oracle=True, runtime=runtime)
        try:
            for level in a.levels:
                env.difficulty = level
                for seed in a.seeds:
                    out = a.output / f"B4_{level}_{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    obs, _ = env.reset(seed=seed)
                    if a.render:
                        Image.fromarray(obs["rgb"]).save(out / "initial.png")
                    task = env.task
                    policy = OraclePickPlace(env)
                    policy.reset(obs)
                    limit = task.spec.event_time + task.spec.response_limit + a.after
                    phases, last_phase = [], None
                    pre_contact = pre_motion = False
                    peak_handle = peak_roller = 0.0
                    start_joints = env.backend.joint_positions().copy()
                    max_pre_joint_motion = 0.0
                    while task.time < limit and not env.done:
                        action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                        env.backend.apply_action(action)
                        task.action_count += 1
                        env._ticks(env.config["sim"]["control_decimation"], render=False)
                        state = env.backend.state()
                        c = state.controls
                        phase = policy.planner.phase
                        if phase != last_phase:
                            phases.append([rel(task, task.time) if task.event_time else -1, phase])
                            last_phase = phase
                        if task.event_time is None:
                            pre_contact |= bool(
                                c.get("stopcock_robot_contact") or c.get("clamp_contact")
                            )
                            pre_motion |= abs(c.get("stopcock_angle_rad", 0.0)) > 1e-3 or abs(
                                c.get("clamp_position_m", 0.0)
                            ) > 1e-4
                            moved = np.abs(env.backend.joint_positions()[:6] - start_joints[:6])
                            max_pre_joint_motion = max(max_pre_joint_motion, float(moved.max()))
                        peak_handle = max(peak_handle, abs(float(c.get("stopcock_speed_rad_s", 0))))
                        peak_roller = max(peak_roller, abs(float(c.get("clamp_speed_m_s", 0))))
                    if a.render:
                        env.backend.capture_camera(subframes=4)
                        Image.fromarray(env.render()).save(out / "final.png")
                    m = task.metrics()
                    row = {
                        "level": level,
                        "seed": seed,
                        "physical_success": bool(task.physical_success),
                        "temporal_success": bool(task.temporal_success),
                        "joint_success": bool(task.joint_success),
                        "response_limit_s": task.spec.response_limit,
                        "response_s": rel(task, task.action_time),
                        "clamp_closed_s": rel(task, task.clamp_time),
                        "detent_entry_s": rel(task, task.first_target_entry),
                        "event_time": task.event_time,
                        "final_angle_deg": m["stopcock_angle_deg"],
                        "final_clamp_travel_m": m["clamp_travel_m"],
                        "wrong_order": m["wrong_order"],
                        "violations": sorted(task.violations),
                        "physical_failures": list(task.physical_failures),
                        "collision_count": m.get("collision_count"),
                        "pre_order_contact": pre_contact,
                        "pre_order_device_motion": pre_motion,
                        "pre_order_max_joint_motion_rad": max_pre_joint_motion,
                        "peak_handle_speed_rad_s": peak_handle,
                        "peak_roller_speed_m_s": peak_roller,
                        "phases": phases,
                        "wall_seconds": round(time.perf_counter() - started, 1),
                    }
                    rows.append(row)
                    (out / "result.json").write_text(
                        json.dumps(
                            {
                                "simulator": simulator_version(),
                                "summary": row,
                                "metrics": m,
                                "events": task.event_log,
                                "episode": env.episode_metadata(),
                            },
                            default=plain,
                            indent=2,
                        )
                    )
                    short = {k: v for k, v in row.items() if k != "phases"}
                    print("B4", json.dumps(short, default=plain), flush=True)
                    print("  phases", json.dumps(phases), flush=True)
        finally:
            env.close()
        if rows:
            counts = {}
            for level in a.levels:
                sub = [r for r in rows if r["level"] == level]
                counts[level] = {
                    key: f"{sum(r[key] for r in sub)}/{len(sub)}"
                    for key in ("physical_success", "temporal_success", "joint_success")
                }
                responses = [r["response_s"] for r in sub if r["response_s"] is not None]
                counts[level]["response_s_max"] = max(responses) if responses else None
                counts[level]["response_limit_s"] = sub[0]["response_limit_s"]
            summary = {
                "task": "B4 line_switch",
                "simulator": simulator_version(),
                "policy": "privileged_lula_ik_trajectory_diagnostic (OraclePickPlace)",
                "levels": a.levels,
                "seeds": a.seeds,
                "overrides": a.set,
                "physical_success_total": f"{sum(r['physical_success'] for r in rows)}/{len(rows)}",
                "within_deadline_total": f"{sum(r['joint_success'] for r in rows)}/{len(rows)}",
                "by_level": counts,
                "episodes": rows,
            }
            (a.output / "summary.json").write_text(json.dumps(summary, default=plain, indent=2))
            print("SUMMARY", json.dumps(summary["by_level"]), summary["physical_success_total"],
                  summary["within_deadline_total"], flush=True)
    except BaseException as exc:
        runtime.exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
