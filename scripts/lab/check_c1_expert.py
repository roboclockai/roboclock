"""Run a throughput task's GT planner without frame-by-frame rendering before recording.

Writes a per-step planner trace (sample, phase, IK failures, object/tool poses) so
stalls, re-targets and returns can be inspected offline. Defaults to C1.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="C1", choices=["C1", "C2", "C3"])
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument(
        "--dense-until", type=float, default=0.0, help="log every control step until this time"
    )
    a = p.parse_args()
    output = a.output or Path(f"results/lab/{a.task.lower()}-expert.json")
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime() as runtime:
        from roboclock.envs.lab.env import LabEnv
        from roboclock.envs.lab.policies.oracle import OraclePickPlace
        from roboclock.envs.lab.validation import _scene_checks

        env = LabEnv(runtime=runtime, oracle=True, config_path=a.config)
        try:
            from roboclock.envs.lab.rack_organization import configure_recording_logging

            configure_recording_logging()
            obs, _ = env.reset(seed=a.seed, options={"task_id": a.task})
            errors = _scene_checks(env)
            if errors:
                raise RuntimeError(errors)
            policy = OraclePickPlace(env)
            policy.reset(obs)
            planners = policy.bimanual.planners if policy.bimanual else [policy.planner]
            trace = []
            previous = None
            collisions_seen = 0
            contacts_log = []

            def filter_names(view):
                for attr in ("_contact_filter_prim_paths_expr", "contact_filter_prim_paths_expr"):
                    value = getattr(view, attr, None)
                    if value:
                        return value[0]
                return []

            while not env.done and env.task.time < min(a.seconds, env.task.spec.max_time) - 1e-8:
                env.backend.apply_action(
                    policy.act_internal({"joint_positions": env.backend.joint_positions()})
                )
                env.task.action_count += 1
                env._ticks(env.config["sim"]["control_decimation"], render=False)
                if env.backend.collision_count > collisions_seen:
                    collisions_seen = env.backend.collision_count
                    import numpy as np

                    dt = env.config["sim"]["physics_dt"]
                    hits = {}
                    for label, view in (
                        ("left", env.backend.arm_contacts),
                        ("right", env.backend.secondary_contacts),
                        ("inter_arm", env.backend.inter_arm_contacts),
                    ):
                        if view is None:
                            continue
                        forces = np.linalg.norm(view.get_contact_force_matrix(dt=dt), axis=-1)
                        names = filter_names(view)
                        for link, column in zip(*np.nonzero(forces > 0.5)):
                            name = names[column] if column < len(names) else column
                            hits[f"{label}:link{link}:{name}"] = round(
                                float(forces[link, column]), 2
                            )
                    contacts_log.append({"time": env.task.time, "hits": hits})
                    print(
                        json.dumps({"collision": collisions_seen, **contacts_log[-1]}), flush=True
                    )
                key = tuple(
                    (planner.sample, planner.phase, getattr(planner, "returning", False))
                    for planner in planners
                ) + (len(env.task.correct), len(env.task.attempted), tuple(policy.ik_failures))
                if key != previous or env.task.time < a.dense_until:
                    previous = key
                    state = env.backend.state()
                    row = {
                        "time": env.task.time,
                        "correct": len(env.task.correct),
                        "attempted": len(env.task.attempted),
                        "ik_failures": list(policy.ik_failures),
                        "yielding": list(policy.bimanual.yielding) if policy.bimanual else [],
                        "commanded": [
                            [round(float(v), 4) for v in c]
                            for c in getattr(policy, "last_cartesian", [])
                        ],
                        "collisions": env.backend.collision_count,
                        "arms": [],
                    }
                    for side, planner in enumerate(planners):
                        arm = {
                            "sample": planner.sample,
                            "phase": planner.phase,
                            "returning": bool(getattr(planner, "returning", False)),
                            "retries": dict(planner.retries),
                        }
                        if planner.sample is not None:
                            obj = state.objects[planner.sample]
                            arm["position"] = obj.position.tolist()
                            arm["orientation"] = obj.orientation.tolist()
                            arm["grasped"] = obj.grasped
                            arm["target"] = env.task.layout.assignments[planner.sample]
                            arm["target_position"] = env.task.layout.targets[
                                arm["target"]
                            ].position.tolist()
                        robot = env.backend.secondary_robot if side else env.backend.robot
                        arm["ee_position"] = env.backend._tool_pose(robot)[0].tolist()
                        goal = getattr(planner, "last_goal", None)
                        arm["goal"] = None if goal is None else [float(v) for v in goal]
                        arm["joints"] = [float(v) for v in robot.get_joint_positions()]
                        row["arms"].append(arm)
                    trace.append(row)
                    print(json.dumps(row), flush=True)
            explain = []
            if policy.bimanual is not None:
                from roboclock.envs.lab.policies.bimanual import ArmTask

                bim = policy.bimanual
                state = env.backend.state()
                tools = [
                    env.backend._tool_pose(r)[0]
                    for r in (env.backend.robot, env.backend.secondary_robot)
                ]
                for side, planner in enumerate(bim.planners):
                    names = bim.names[side]
                    view = ArmTask(env.task, names, set())
                    compatible = bim.compatible_jobs(
                        side, view, set(names), tools[side], tools[1 - side], state
                    )
                    for name in sorted(names):
                        if name in env.task.correct:
                            continue
                        row = {
                            "arm": side,
                            "sample": name,
                            "reason": getattr(bim, "compat_reasons", {}).get(name),
                            "other_busy": bim.other_busy(side),
                            "other": [bim.planners[1 - side].sample, bim.planners[1 - side].phase],
                            "phase": planner.phase,
                            "retries": planner.retries.get(name, 0),
                            "compatible": name in compatible,
                        }
                        if hasattr(planner, "candidate_targets"):
                            source = state.objects[name].position
                            row["source_reachable"] = bool(
                                planner.reachable(
                                    *planner.with_transit(env.task, planner.tool_point(source))
                                )
                            )
                            row["slots"] = planner.candidate_targets(view, name, state)
                            row["free"] = env.task.free_targets(name, state)
                            row["upright_error"] = float(
                                __import__(
                                    "roboclock.physics.geometry", fromlist=["upright_error"]
                                ).upright_error(state.objects[name].orientation)
                            )
                        explain.append(row)
                        print(json.dumps({"explain": row}), flush=True)
            result = {
                "task": a.task,
                "explain": explain,
                "seed": a.seed,
                "metrics": env.task.metrics(),
                "events": env.task.event_log,
                "trace": trace,
                "initial_scene_errors": errors,
                "cooperative": policy.bimanual is not None,
                "collisions": contacts_log,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2))
            print(json.dumps(result["metrics"]), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
