"""B6 pump panel response: actual-physics oracle check over levels x seeds, one simulator session.

Rendering happens only at reset (and at chosen times with --frames), so the physics clock is never
advanced by rendering. Records, per episode: the response (instruction to the scored end of the
press), the reach (instruction to first key contact), first motion, and any hand motion or key
travel before the instruction. Writes one JSON per episode and a summary that reports physical
success and within-deadline success separately.

python scripts/hospital/b6_oracle.py --output results/hospital/B6/oracle \
    --summary results/hospital/B4/oracle_summary.json
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
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output", type=Path, default=Path("results/hospital/B6/oracle"))
    p.add_argument(
        "--summary", type=Path, help="Also write the summary here (e.g. results/hospital/B4)"
    )
    p.add_argument(
        "--frames", nargs="*", type=float, help="Also save frames at these task times (s)"
    )
    p.add_argument("--limit", type=float, help="Override response_limit (s) for every level")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.bedside import pump_panel
    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace
    from roboclock.envs.lab.robot import mount_positions

    rows = []
    try:
        env = HospitalEnv(task_id="B6", difficulty=a.levels[0], oracle=True, runtime=runtime)
        if a.limit:
            for level in ("easy", "medium", "hard"):
                env.config["tasks"]["B6"][level]["response_limit"] = [a.limit, a.limit]
        for level in a.levels:
            for seed in a.seeds:
                env.difficulty = level
                started = time.perf_counter()
                obs, _ = env.reset(seed=seed)
                tag = f"B7_{level}_{seed}"
                frames = sorted(a.frames or [])
                if a.frames is not None:
                    Image.fromarray(obs["rgb"]).save(a.output / f"{tag}_t0.00.png")
                policy = OraclePickPlace(env)
                policy.reset(obs)
                task = env.task
                info = task.layout.info
                start_ee = env.backend.state().ee_position.copy()
                center = np.asarray(info["pump_panel"]["centers"][task.mode])
                pre_motion, pre_travel = 0.0, 0.0
                first_contact = first_move = None
                path = 0.0
                previous = start_ee
                while task.time < task.spec.max_time - 1e-9 and not env.done:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                    state = env.backend.state()
                    ee = state.ee_position
                    keys = state.controls["pump_panel"]
                    if task.event_time is None:
                        pre_motion = max(pre_motion, float(np.linalg.norm(ee - start_ee)))
                        pre_travel = max(pre_travel, max(k["depression_m"] for k in keys.values()))
                    else:
                        path += float(np.linalg.norm(ee - previous))
                        if first_move is None and np.linalg.norm(ee - start_ee) > 0.005:
                            first_move = task.time - task.event_time
                        if first_contact is None and keys[task.mode]["contact"]:
                            first_contact = task.time - task.event_time
                    previous = ee.copy()
                    while frames and task.time >= frames[0] - 1e-9:
                        env.backend.capture_camera(subframes=8)
                        Image.fromarray(env.render()).save(
                            a.output / f"{tag}_t{frames.pop(0):.2f}.png"
                        )
                if a.frames is not None:
                    env.backend.capture_camera(subframes=8)
                    Image.fromarray(env.render()).save(a.output / f"{tag}_final.png")
                m = task.metrics()
                trigger = getattr(task, "trigger_time", None)
                mount = np.asarray(mount_positions(env.config, "B6")[0])
                shoulder = mount + [0, 0, 0.1005]
                row = {
                    "level": level,
                    "seed": seed,
                    "requested_key": task.mode,
                    "instruction_time": round(task.spec.event_time, 3),
                    "response_limit_s": task.spec.response_limit,
                    "response_s": None
                    if trigger is None or task.event_time is None
                    else round(trigger - task.event_time, 3),
                    "first_motion_after_instruction_s": None
                    if first_move is None
                    else round(first_move, 3),
                    "first_key_contact_after_instruction_s": None
                    if first_contact is None
                    else round(first_contact, 3),
                    "hand_path_m": round(path, 3),
                    "rest_to_key_m": round(float(np.linalg.norm(center - start_ee)), 3),
                    "key_from_shoulder_m": round(float(np.linalg.norm(center - shoulder)), 3),
                    "hand_motion_before_instruction_m": round(pre_motion, 4),
                    "key_travel_before_instruction_m": round(pre_travel, 4),
                    "physical": m["manipulation_success"],
                    "within_deadline": m["temporal_success"],
                    "joint": m["joint_success"],
                    "collisions": m["collision_count"],
                    "violations": m["temporal_violations"],
                    "physical_failures": m["physical_failures"],
                    "device_state": m["device_state"],
                }
                if row["response_s"] is not None:
                    row["fraction_of_limit"] = round(
                        row["response_s"] / task.spec.response_limit, 3
                    )
                rows.append(row)
                result = {
                    "row": row,
                    "metrics": m,
                    "events": task.event_log,
                    "episode": env.episode_metadata(),
                    "wall_seconds": time.perf_counter() - started,
                }
                (a.output / f"{tag}.json").write_text(json.dumps(result, default=plain, indent=1))
                print("B6", json.dumps(row, default=plain), flush=True)
        env.close()
        levels = {}
        for level in a.levels:
            group = [r for r in rows if r["level"] == level]
            limit = group[0]["response_limit_s"] if group else None
            times = [r["response_s"] for r in group]
            levels[level] = {
                "response_limit_s": limit,
                "episodes": len(group),
                "physical_success": sum(r["physical"] for r in group),
                "within_deadline": sum(r["within_deadline"] for r in group),
                "joint_success": sum(r["joint"] for r in group),
                "collisions": sum(r["collisions"] for r in group),
                # Scored at the end of the required hold (40 ms STOP/MUTE, 350 ms BOLUS).
                "response_s": stats(times),
                "max_fraction_of_limit": round(max(t for t in times if t is not None) / limit, 3)
                if any(t is not None for t in times)
                else None,
            }
        by_key = {
            key: {
                "response_s": stats([r["response_s"] for r in rows if r["requested_key"] == key]),
                "first_key_contact_s": stats(
                    [
                        r["first_key_contact_after_instruction_s"]
                        for r in rows
                        if r["requested_key"] == key
                    ]
                ),
                "rest_to_key_m": stats(
                    [r["rest_to_key_m"] for r in rows if r["requested_key"] == key]
                ),
            }
            for key in ("stop", "mute", "bolus")
        }
        layout_info = env.task.layout.info if env.task is not None else {}
        summary = {
            "task": "B6 pump_panel_response",
            "env": "hospital",
            "simulator": "isaacsim==" + simulator_version(),
            "policy": "privileged scripted oracle (bedside/pump_panel.py: command), joint actions only",
            "setup": layout_info.get("pump_panel", {}).get("placement", ""),
            "instruction": "pump display + order card at the dial, legible from frame 0; clock starts at "
            "instruction_time",
            "robot_start": "at rest, pressing hand folded near its column (bedside/pump_panel.py TUCK)",
            "ward_pose": {
                "usd": pump_panel.WARD_USD,
                "yaw_deg": pump_panel.WARD_YAW,
                "robot_origin_in_ward_m": [pump_panel.ROBOT_X, pump_panel.ROBOT_Y],
                "translate_nominal": [pump_panel.ROBOT_Y, -pump_panel.ROBOT_X, pump_panel.WARD_Z],
                "note": "the seed's bedside jitter shifts the ward together with the pump",
                "hidden": "bay instrument trolley (robot parking spot)",
            },
            "seeds": a.seeds,
            "levels": levels,
            "by_requested_key": by_key,
            "physical_passes": f"{sum(r['physical'] for r in rows)}/{len(rows)}",
            "within_deadline_passes": f"{sum(r['within_deadline'] for r in rows)}/{len(rows)}",
            "joint_passes": f"{sum(r['joint'] for r in rows)}/{len(rows)}",
            "response_time_note": "response_s = scored trigger (end of the required 40 ms / 350 ms hold) "
            "minus the instruction time",
            "episodes": rows,
        }
        for path in [a.output / "summary.json"] + ([a.summary] if a.summary else []):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, default=plain, indent=1))
        print(
            "B6 physical",
            summary["physical_passes"],
            "within deadline",
            summary["within_deadline_passes"],
            flush=True,
        )
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
