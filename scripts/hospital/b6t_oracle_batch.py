"""B6 haemorrhage tourniquet: real-physics oracle runs over levels x seeds in one Isaac session.

python scripts/hospital/b6t_oracle_batch.py --output results/hospital/B6/oracle \
    --summary results/hospital/B6/oracle_summary.json

Headless by default (sim.no_render; zero inference latency and control delay: the oracle is
stepped directly). Writes one <level>_<seed>.json per episode (metrics, events, planner phases)
and a summary with physical and within-deadline pass counts reported separately. With --render
it also saves the head, wrist and (with --task-camera) third-person frames at reset, at the
episode's end and at any --frames-at times after the start of the bleeding; rendered runs take one
episode per process (Isaac Sim hangs on a second rendered reset).
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


def level_counts(rows):
    """Physical success and within-deadline success are reported separately."""
    latencies = [r["response_latency"] for r in rows if r["response_latency"] is not None]
    limit = rows[0]["response_limit"] if rows else None
    return {
        "episodes": len(rows),
        "physical": sum(bool(r["physical"]) for r in rows),
        "within_deadline": sum(bool(r["temporal"]) for r in rows),
        "joint": sum(bool(r["joint"]) for r in rows),
        "response_limit_s": limit,
        "latency_range_s": [min(latencies), max(latencies)] if latencies else None,
        "max_fraction_of_limit": round(max(latencies) / limit, 3) if latencies else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--output", type=Path, default=Path("results/hospital/B6/oracle"))
    p.add_argument("--summary", type=Path, default=None, help="also write the summary here")
    p.add_argument("--render", action="store_true", help="render and save camera frames")
    p.add_argument("--task-camera", action="store_true", help="also the third-person view")
    p.add_argument("--frames-at", nargs="*", type=float, default=[], help="frame times after the event (s)")
    p.add_argument("--iso", type=float, default=None, help="override tasks.B6.film_iso")
    p.add_argument(
        "--placement", nargs=4, type=float, default=None,
        metavar=("BAND_SHIFT_M", "ROD_START_DEG", "CLIP_DEG", "LEG_YAW_DEG"),
        help="force the seeded placement to these values (range extremes)",
    )
    p.add_argument("--trace", action="store_true", help="save per-control-step hand/rod samples")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.envs.hospital.config import load_config

    config = load_config()
    config["sim"]["no_render"] = not a.render
    if a.task_camera:
        config["sim"]["task_camera"] = True
        config["sim"]["task_camera_resolution"] = [1280, 960]
    if a.iso is not None:
        config["tasks"]["B6"]["film_iso"] = a.iso
    if a.placement is not None:
        keys = ("band_shift_m", "rod_start_deg", "clip_deg", "leg_yaw_deg")
        config["tasks"]["B6"]["placement"] = {k: [v, v] for k, v in zip(keys, a.placement)}
    import yaml

    config_path = a.output / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    runtime = IsaacRuntime()
    from PIL import Image

    from roboclock.envs.hospital.bedside.tourniquet import NAME as ROD
    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    def save(env, tag):
        if not a.render:
            return
        frames = env.backend.render_frame(subframes=4)
        for key, name in (("rgb", "head"), ("right_wrist_rgb", "right_wrist"), ("left_wrist_rgb", "left_wrist")):
            Image.fromarray(frames[key]).save(a.output / f"{tag}_{name}.png")
        camera = getattr(env.backend, "task_camera", None)
        if camera is not None:
            rgba = camera.get_rgba()
            if rgba is not None and rgba.size:
                Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8)).save(a.output / f"{tag}_task.png")

    rows = []
    try:
        env = HospitalEnv(
            task_id="B6", difficulty=a.levels[0], oracle=True, runtime=runtime, config_path=config_path
        )
        for level in a.levels:
            for seed in a.seeds:
                env.difficulty = level
                started = time.perf_counter()
                obs, _ = env.reset(seed=seed)
                tag = f"{level}_{seed}"
                save(env, f"{tag}_initial")
                if a.render:
                    head = np.asarray(obs["rgb"], dtype=float)
                    luma = head @ [0.299, 0.587, 0.114]
                    print("head mean luma", round(float(luma.mean()), 1), "clipped %",
                          round(100 * float((head.min(axis=-1) >= 254).mean()), 2), flush=True)
                policy = OraclePickPlace(env)
                policy.reset(obs)
                pending = sorted(a.frames_at)
                trace = []
                while not env.done and env.task.time < env.task.spec.max_time:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    env.task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                    task = env.task
                    if a.trace:
                        state = env.backend.state()
                        rod = state.objects[ROD]
                        trace.append(
                            {
                                "t": round(task.time, 4),
                                "phase": getattr(policy.planner, "phase", None),
                                "ee": state.ee_position,
                                "goal": policy.planner.last_goal,
                                "yaw": getattr(policy.planner, "yaw", None),
                                "joints": env.backend.joint_positions()[:8],
                                "grasped": rod.grasped,
                                "contact": rod.robot_contact,
                                "measured": state.controls.get("tourniquet"),
                                "blood_ml": task.blood_lost,
                            }
                        )
                    while pending and task.event_time is not None and task.time - task.event_time >= pending[0]:
                        save(env, f"{tag}_t{pending.pop(0):05.2f}")
                save(env, f"{tag}_final")
                m = env.task.metrics()
                phases = [e for e in env.task.event_log if e["type"].startswith("oracle_")]
                (a.output / f"{tag}.json").write_text(
                    json.dumps(
                        {"metrics": m, "events": env.task.event_log, "phases": phases, "trace": trace},
                        indent=2,
                        default=plain,
                    )
                )
                row = {
                    "level": level,
                    "seed": seed,
                    "physical": m["task_success"],
                    "temporal": m["temporal_success"],
                    "joint": m["joint_success"],
                    "event_time": m["bleed_start_time"],
                    "response_limit": m["response_limit"],
                    "response_latency": m["response_latency"],
                    "turn_latency": m["turn_latency"],
                    "occlusion_latency": m["occlusion_latency"],
                    "blood_lost_ml": m["blood_lost_at_secure_ml"],
                    "placement": m["placement"],
                    "turn_required_deg": round(float(np.degrees(m["turn_required_rad"])), 2),
                    "violations": m.get("temporal_violations"),
                    "physical_failures": m.get("physical_failures"),
                    "collisions": m.get("collision_count"),
                    "sim_time": round(env.task.time, 3),
                    "wall_s": round(time.perf_counter() - started, 1),
                }
                rows.append(row)
                print(json.dumps(row, default=plain), flush=True)
        summary = {
            "task": "B6 haemorrhage_tourniquet",
            "simulator": "isaacsim==" + simulator_version(),
            "rendered": bool(a.render),
            "inference_latency": "zero",
            "levels": {lv: level_counts([r for r in rows if r["level"] == lv]) for lv in a.levels},
            "rows": rows,
        }
        text = json.dumps(summary, indent=2, default=plain)
        (a.output / "summary.json").write_text(text)
        if a.summary:
            a.summary.parent.mkdir(parents=True, exist_ok=True)
            a.summary.write_text(text)
        print(json.dumps(summary["levels"], indent=2, default=plain))
    finally:
        env.close()
        runtime.close()


if __name__ == "__main__":
    main()
