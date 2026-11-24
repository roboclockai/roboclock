"""Measure an A1 closed/open/collect/closed cycle using actual PhysX contacts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/lab/water-full-cycle/physics"))
    parser.add_argument("--idle", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime(renderer="RaytracedLighting") as runtime:
        from roboclock.envs.lab.env import LabEnv
        from roboclock.envs.lab.policies.oracle import OraclePickPlace

        env = LabEnv(runtime=runtime, oracle=True)
        env.task_id = "A1"
        env.config["sim"].update(overview_camera=True, overview_resolution=[640, 480])
        try:
            env.reset(seed=0)
            policy = OraclePickPlace(env)
            policy.reset(env.get_obs())
            initial = dict(env.backend.state().controls)
            idle_joints = env.backend.joint_positions().copy()
            trace = []
            while not env.done and env.task.time < (5 if args.idle else env.task.spec.max_time):
                action = (
                    idle_joints
                    if args.idle
                    else policy.act_internal({"joint_positions": env.backend.joint_positions()})
                )
                env.backend.apply_action(action)
                env._ticks(4, render=False)
                state = env.backend.state()
                trace.append(
                    dict(
                        time=env.task.time,
                        phase=policy.planner.phase,
                        controls=state.controls,
                        grasped=state.objects["sample_0"].grasped,
                        ee_position=state.ee_position.tolist(),
                        visual=env.task.visual_state(),
                    )
                )
            result = dict(
                initial=initial, metrics=env.task.metrics(), events=env.task.event_log, trace=trace
            )
            (args.output / "A1.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result["metrics"]), flush=True)
            assert initial["opening"] < 0.04
            if args.idle:
                assert env.task.opened_at is None and env.task.volume == 0
            else:
                assert env.task.joint_success
                assert env.task.opened_at < env.task.fully_open_at < env.task.stopped_at
                assert trace[-1]["phase"] == "retreat"
        finally:
            env.close()


if __name__ == "__main__":
    main()
