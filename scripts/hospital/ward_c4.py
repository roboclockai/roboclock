"""C4 blood-tube triage: render a still, run the scripted oracle over levels/seeds, record a video.

Iterations go to results/hospital/C4 (default).

python scripts/hospital/ward_c4.py still --seed 0 --level hard
python scripts/hospital/ward_c4.py oracle --levels easy medium hard --seeds 0 1 2
python scripts/hospital/ward_c4.py video --level hard --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path("results/hospital/C4")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def run_oracle(env, seed, still=None):
    from PIL import Image

    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    obs, _ = env.reset(seed=seed)
    if still is not None:
        Image.fromarray(obs["rgb"]).save(still)
    policy = OraclePickPlace(env)
    policy.reset(obs)
    task = env.task
    seen, collisions = set(), 0
    while not env.done and task.time < task.spec.max_time - 1e-9:
        action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
        env.backend.apply_action(action)
        task.action_count += 1
        env._ticks(env.config["sim"]["control_decimation"], render=False)
        planner = policy.planner
        if task.collision_count != collisions:
            collisions = task.collision_count
            task.event_log.append(
                {
                    "time": round(task.time, 3),
                    "type": "c4_collision",
                    "count": collisions,
                    "left": [getattr(planner, "c4_phase", None), getattr(planner, "c4_tube", None)],
                    "right": [
                        getattr(policy.secondary_planner, "c4_phase", None),
                        getattr(policy.secondary_planner, "c4_tube", None),
                    ],
                }
            )
        for name in sorted(task.moved - seen):
            seen.add(name)
            obj = env.backend.state().objects[name]
            task.event_log.append(
                {
                    "time": round(task.time, 3),
                    "type": "c4_first_moved",
                    "object": name,
                    "active": getattr(planner, "c4_tube", None),
                    "phase": getattr(planner, "c4_phase", None),
                    "ee": [round(float(v), 3) for v in env.backend.state().ee_position],
                    "position": [round(float(v), 3) for v in obj.position],
                }
            )
    return task.metrics()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["still", "oracle", "video"])
    p.add_argument("--level", default="hard")
    p.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--resolution", nargs=2, type=int, default=None)
    p.add_argument("--fps", type=int, default=25)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    import yaml

    from roboclock.envs.hospital.config import load_config
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    config = load_config()
    if a.resolution:
        config["sim"]["camera_resolution"] = a.resolution
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / "render_config.yaml"  # never written into the data/ deliverables
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with IsaacRuntime(renderer=config["visuals"]["renderer"]) as runtime:
        from PIL import Image

        from roboclock.envs.hospital.env import HospitalEnv

        env = HospitalEnv(
            task_id="C4", difficulty=a.level, oracle=True, runtime=runtime, config_path=config_path
        )
        try:
            if a.mode == "still":
                obs, _ = env.reset(seed=a.seed)
                path = a.output / (
                    "still.png"
                    if (a.level, a.seed) == ("hard", 0)
                    else f"still_{a.level}_{a.seed}.png"
                )
                Image.fromarray(env.render()).save(path)
                print("STILL", path, flush=True)
            elif a.mode == "video":
                from roboclock.envs.hospital.videos import record_episode

                meta = record_episode(
                    env,
                    a.output / f"C4_blood_tube_triage_{a.level}_{a.seed}.mp4",
                    seed=a.seed,
                    fps=a.fps,
                    seconds=1e9,
                    subframes=4,
                )
                print(json.dumps(meta["metrics"], default=plain), flush=True)
            else:
                rows = []
                for level in a.levels:
                    for seed in a.seeds:
                        env.difficulty = level
                        metrics = run_oracle(
                            env, seed, still=a.output / f"initial_{level}_{seed}.png"
                        )
                        row = {
                            "level": level,
                            "seed": seed,
                            "horizon": metrics["horizon_s"],
                            "correct": metrics["correct_operations"],
                            "attempted": metrics["attempted_operations"],
                            "accuracy": metrics["accuracy"],
                            "t_q": metrics["t_q_s"],
                            "end": metrics["termination_reason"],
                            "physical": metrics["task_success"],
                            "temporal_by_tier": metrics["temporal_success_by_tier"],
                            "joint_by_tier": metrics["joint_success_by_tier"],
                        }
                        rows.append(row)
                        print("ROW", json.dumps(row, default=plain), flush=True)
                        (a.output / f"oracle_{level}_{seed}.json").write_text(
                            json.dumps(
                                {"metrics": metrics, "events": env.task.event_log},
                                default=plain,
                                indent=1,
                            )
                        )
                (a.output / "oracle_summary.json").write_text(
                    json.dumps(rows, indent=1, default=plain)
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
