"""Reproduce active or idle A5/A6 PhysX diagnostics without offline video rendering."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=["A5", "A6"], default=["A5", "A6"])
    parser.add_argument(
        "--idle", action="store_true", help="Leave the robot at rest to exercise failure branches"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hold-after-loading", action="store_true", help="A5: insert reel, then stop responding"
    )
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument(
        "--previews", action="store_true", help="Capture initial, warning and final states"
    )
    parser.add_argument(
        "--renderer", choices=["PathTracing", "RaytracedLighting"], default="RaytracedLighting"
    )
    parser.add_argument("--output", type=Path, default=Path("results/lab/bath-physics"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime(renderer=args.renderer) as runtime:
        from roboclock.envs.lab.env import LabEnv
        from roboclock.envs.lab.policies.oracle import OraclePickPlace
        from roboclock.envs.lab.validation import _scene_checks

        env = LabEnv(runtime=runtime, oracle=True)
        env.config["sim"].update(overview_camera=True, overview_resolution=[1280, 960] if args.previews else [640, 480])
        env.config["liquid_visuals"]["renderer"] = args.renderer
        env.config["liquid_visuals"].update(path_tracing_spp=32, path_tracing_samples=32)
        try:
            for task_id in args.tasks:
                env.task_id = task_id
                env.reset(seed=args.seed)
                errors = _scene_checks(env)
                if errors:
                    raise RuntimeError(
                        {
                            "errors": errors,
                            "initial": env.task.layout.objects["sample_0"].position.tolist(),
                            "measured": env.backend.state().objects["sample_0"].position.tolist(),
                            "controls": env.backend.state().controls,
                        }
                    )
                policy = None if args.idle else OraclePickPlace(env)
                initial = {
                    "ee_position": env.backend.state().ee_position.tolist(),
                    "joint_positions": env.backend.joint_positions().tolist(),
                    "controls": dict(env.backend.state().controls),
                }
                idle_joints = env.backend.joint_positions().copy()
                loaded_hold = None
                trace = []
                previews = {}

                def capture(name):
                    if not args.previews:
                        return
                    import numpy as np
                    from PIL import Image

                    from roboclock.simulators.isaac.rendering import capture_camera

                    capture_camera(1)
                    path = args.output / f"{task_id}_{name}.png"
                    Image.fromarray(env.backend.camera.get_rgba()[..., :3].astype(np.uint8)).save(
                        path
                    )
                    previews[name] = path.name

                capture("initial")
                captured_event = False
                captured_loading = False
                while not env.done and env.task.time < args.seconds:
                    if (
                        args.hold_after_loading
                        and loaded_hold is None
                        and env.task.controls.get("source_supported")
                        and env.task.source_entry_time is not None
                    ):
                        loaded_hold = env.backend.joint_positions().copy()
                    if loaded_hold is not None:
                        env.backend.apply_action(loaded_hold)
                    elif policy is not None:
                        action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                        env.backend.apply_action(action)
                        env.task.action_count += 1
                    else:
                        env.backend.apply_action(idle_joints)
                    env._ticks(4, render=False)
                    if (
                        task_id == "A5"
                        and env.task.source_entry_time is not None
                        and not captured_loading
                    ):
                        capture("developer_entry")
                        captured_loading = True
                    if env.task.event_time is not None and not captured_event:
                        capture("warning")
                        captured_event = True
                    state = env.backend.state()
                    obj = state.objects["sample_0"]
                    trace.append(
                        {
                            "time": env.task.time,
                            "position": obj.position.tolist(),
                            "orientation": obj.orientation.tolist(),
                            "grasped": obj.grasped,
                            "robot_contact": obj.robot_contact,
                            "controls": state.controls,
                            "ee_position": state.ee_position.tolist(),
                            "ee_orientation": state.ee_orientation.tolist(),
                            "joint_positions": env.backend.joint_positions().tolist(),
                            "phase": None if policy is None else policy.planner.phase,
                            "visual": env.task.visual_state(),
                        }
                    )
                capture("final")
                result = {
                    "mode": "loaded_hold"
                    if args.hold_after_loading
                    else "idle"
                    if args.idle
                    else "oracle",
                    "initial": initial,
                    "metrics": env.task.metrics(),
                    "events": env.task.event_log,
                    "trace": trace,
                    "previews": previews,
                }
                (args.output / (task_id + ".json")).write_text(json.dumps(result, indent=2) + "\n")
                print(task_id, json.dumps(result["metrics"]), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
