"""A1 drip chamber level: actual-physics oracle check over levels x seeds in one simulator session.

For every episode it records physical success, within-window success, the action time relative to
the window (seconds after it opens and as a fraction of its width), the frozen chamber level and
its outcome, plus the measured slide timing (oracle slide start -> first roller motion -> closed
end). --late runs one episode with the hand held at the ready pose (no action) and saves its last
frame, to show the overfill. Renders happen only before the first action and after the last.

python scripts/hospital/a1_drip_oracle.py --levels easy medium hard --seeds 0 1 2 3 4 \
    --output results/hospital/A1/drip/oracle --summary results/hospital/A1/oracle_summary.json
python scripts/hospital/a1_drip_oracle.py --late --levels hard --seeds 0 \
    --output results/hospital/A1/drip/late --late-frame results/hospital/A1/late_overfill.png
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


def event_time(task, kind):
    return next((e["time"] for e in task.event_log if e["type"] == kind), None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=Path, default=Path("results/hospital/A1/drip/oracle"))
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--late", action="store_true", help="hold the ready pose: no action")
    p.add_argument("--late-frame", type=Path, default=None)
    p.add_argument("--frames", action="store_true", help="save initial/final frames per episode")
    p.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="add this run's episodes to an existing summary",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.bedside import window_drip
    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    if a.late:
        window_drip.slide_start = lambda *args, **kwargs: float("inf")
    rows = []
    try:
        env = HospitalEnv(task_id="A1", difficulty=a.levels[0], oracle=True, runtime=runtime)
        try:
            for level in a.levels:
                env.difficulty = level
                for seed in a.seeds:
                    out = a.output / f"A1_{level}_{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    obs, _ = env.reset(seed=seed)
                    if a.frames:
                        Image.fromarray(obs["rgb"]).save(out / "initial.png")
                    task = env.task
                    policy = OraclePickPlace(env)
                    policy.reset(obs)
                    first_motion = None
                    peak_speed = 0.0
                    while not env.done:
                        action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                        env.backend.apply_action(action)
                        task.action_count += 1
                        env._ticks(env.config["sim"]["control_decimation"], render=False)
                        controls = env.backend.state().controls
                        peak_speed = max(peak_speed, float(controls.get("roller_speed_m_s", 0.0)))
                        if (
                            first_motion is None
                            and abs(controls.get("roller_position_m", 0.0)) > 1e-4
                        ):
                            first_motion = task.time
                    if a.frames or a.late:
                        env.backend.capture_camera(subframes=16)
                        frame = env.render()
                        Image.fromarray(frame).save(out / "final.png")
                        if a.late_frame is not None:
                            a.late_frame.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(frame).save(a.late_frame)
                    m = task.metrics()
                    t_o, t_c = task.window_open, task.window_close
                    act = task.action_time
                    slide = event_time(task, "oracle_slide_start")
                    controls = task.previous.controls if task.previous is not None else {}
                    row = {
                        "level": level,
                        "seed": seed,
                        "physical_success": bool(task.physical_success),
                        "temporal_success": bool(task.temporal_success),
                        "joint_success": bool(task.joint_success),
                        "window_open_s": t_o,
                        "window_close_s": t_c,
                        "window_width_s": t_c - t_o,
                        "action_time_s": act,
                        "action_after_open_s": None if act is None else act - t_o,
                        "action_window_fraction": None
                        if act is None
                        else (act - t_o) / (t_c - t_o),
                        "action_minus_middle_s": None if act is None else act - 0.5 * (t_o + t_c),
                        "device_moved_time_s": task.device_moved_time,
                        "oracle_slide_start_s": slide,
                        "slide_start_to_closed_s": None
                        if act is None or slide is None
                        else act - slide,
                        "roller_motion_to_closed_s": None
                        if act is None or first_motion is None
                        else act - first_motion,
                        "initial_level_pct": m["initial_quantity"],
                        "final_level_pct": m["final_quantity"],
                        "quantity_outcome": m["quantity_outcome"],
                        "violations": sorted(task.violations),
                        "physical_failures": list(task.physical_failures),
                        "final_roller_position_m": controls.get("roller_position_m"),
                        "peak_roller_speed_m_s": peak_speed,
                        "collision_count": m.get("collision_count"),
                        "episode_end_s": task.time,
                        "wall_seconds": time.perf_counter() - started,
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
                    print("A1", json.dumps(row, default=plain), flush=True)
        finally:
            env.close()
        counts = {}
        for level in a.levels:
            sub = [r for r in rows if r["level"] == level]
            counts[level] = {
                key: f"{sum(r[key] for r in sub)}/{len(sub)}"
                for key in ("physical_success", "temporal_success", "joint_success")
            }
            fractions = [
                r["action_window_fraction"] for r in sub if r["action_window_fraction"] is not None
            ]
            counts[level]["action_window_fraction_range"] = (
                [min(fractions), max(fractions)] if fractions else None
            )
            levels = [r["final_level_pct"] for r in sub]
            counts[level]["final_level_pct_range"] = [min(levels), max(levels)]
        durations = [
            r["roller_motion_to_closed_s"]
            for r in rows
            if r["roller_motion_to_closed_s"] is not None
        ]
        leads = [
            r["slide_start_to_closed_s"] for r in rows if r["slide_start_to_closed_s"] is not None
        ]
        summary = {
            "task": "A1 drip_chamber_level",
            "simulator": simulator_version(),
            "policy": "privileged_lula_ik_trajectory_diagnostic (OraclePickPlace + bedside.window_drip.command)",
            "mode": "no_action (hand held at the ready pose)" if a.late else "oracle",
            "band_pct": [40.0, 55.0],
            "levels": a.levels,
            "seeds": a.seeds,
            "physical_success_total": f"{sum(r['physical_success'] for r in rows)}/{len(rows)}",
            "temporal_success_total": f"{sum(r['temporal_success'] for r in rows)}/{len(rows)}",
            "joint_success_total": f"{sum(r['joint_success'] for r in rows)}/{len(rows)}",
            "slide_duration_s": {
                "roller_motion_to_closed": [
                    min(durations),
                    float(np.mean(durations)),
                    max(durations),
                ]
                if durations
                else None,
                "oracle_slide_start_to_closed": [min(leads), float(np.mean(leads)), max(leads)]
                if leads
                else None,
                "note": "min, mean, max over episodes; the oracle starts the slide PUSH_LEAD s before the window middle",
                "push_lead_s": window_drip.PUSH_LEAD,
            },
            "by_level": counts,
            "episodes": rows,
        }
        (a.output / "summary.json").write_text(json.dumps(summary, default=plain, indent=2))
        if a.merge_into is not None:
            merged = json.loads(a.merge_into.read_text())
            key = "late_check" if a.late else "extra_check"
            merged[key] = {
                "mode": summary["mode"],
                "frame": None if a.late_frame is None else str(a.late_frame),
                "episodes": rows,
            }
            a.merge_into.write_text(json.dumps(merged, default=plain, indent=2))
        if a.summary is not None:
            a.summary.parent.mkdir(parents=True, exist_ok=True)
            a.summary.write_text(json.dumps(summary, default=plain, indent=2))
        print(
            "SUMMARY",
            json.dumps(summary["by_level"]),
            summary["joint_success_total"],
            json.dumps(summary["slide_duration_s"]),
            flush=True,
        )
    except BaseException as exc:
        runtime.exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        import traceback

        traceback.print_exc()
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
