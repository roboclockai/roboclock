"""B4 oracle timeline probe: per control step after the order, with optional constant overrides.

python scripts/hospital/b4_speed.py --level hard --seeds 0 1 --set TURN_LEAD=0.3 \
    --output results/hospital/B4/speed/lead.json

Prints, per episode, the phase timeline and a compact trace (time after the order, phase, clamp
travel, handle angle, grasp flags, commanded and measured yaw, tool error); never renders.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="hard")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--every", type=int, default=1, help="Print every n-th control step")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON with every timeline")
    a = p.parse_args()
    from roboclock.envs.hospital.bedside import stopcock as module

    for item in a.set:
        key, _, value = item.partition("=")
        if not hasattr(module, key):
            raise SystemExit(f"Unknown B4 constant {key}")
        old = getattr(module, key)
        setattr(module, key, type(old)(eval(value)))
        print("set", key, "=", getattr(module, key), flush=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    runtime = IsaacRuntime()
    from scipy.spatial.transform import Rotation

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    rows = []
    try:
        env = HospitalEnv(task_id="B4", difficulty=a.level, oracle=True, runtime=runtime)
        try:
            for seed in a.seeds:
                obs, _ = env.reset(seed=seed)
                task = env.task
                policy = OraclePickPlace(env)
                policy.reset(obs)
                limit = task.spec.event_time + task.spec.response_limit + 1.0
                trace, step = [], 0
                while task.time < limit and not env.done:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                    if task.event_time is None:
                        continue
                    state = env.backend.state()
                    c = state.controls
                    q = env.backend.joint_positions()
                    position, orientation = env.backend._tool_pose(env.backend.robot)
                    w, x, y, z = orientation
                    yaw = Rotation.from_quat([x, y, z, w]).as_euler("ZYX")[0]
                    goal = policy.planner.last_goal
                    row = [
                        round(task.time - task.event_time, 3),
                        policy.planner.phase,
                        round(1000 * float(c.get("clamp_position_m", 0.0)), 1),
                        round(float(np.degrees(c.get("stopcock_angle_rad", 0.0))), 1),
                        int(bool(c.get("stopcock_grasped"))),
                        int(bool(c.get("stopcock_robot_contact"))),
                        int(bool(c.get("clamp_contact"))),
                        round(float(np.degrees(policy.planner.yaw)), 1),
                        round(float(np.degrees(yaw)), 1),
                        [round(1000 * float(v), 1) for v in np.asarray(position) - goal],
                        [round(float(v), 3) for v in q[:8]],
                    ]
                    trace.append(row)
                    if step % a.every == 0:
                        print(json.dumps(row), flush=True)
                    step += 1
                first = task.first_target_entry
                summary = {
                    "seed": seed,
                    "clamp_closed": None if task.clamp_time is None else round(task.clamp_time - task.event_time, 3),
                    "first_entry": None if first is None else round(first - task.event_time, 3),
                    "response": None if task.action_time is None else round(task.action_time - task.event_time, 3),
                    "physical": bool(task.physical_success),
                    "temporal": bool(task.temporal_success),
                    "violations": sorted(task.violations),
                    "failures": list(task.physical_failures),
                    "final_angle": round(float(np.degrees(task.angle)), 2),
                    "limit": task.spec.response_limit,
                }
                rows.append({**summary, "trace": trace})
                print("B4", json.dumps(summary), flush=True)
        finally:
            env.close()
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(rows, indent=1))
    except BaseException as exc:
        runtime.exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
