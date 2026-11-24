"""Run contact-driven Dual ARX diagnostics before replacing benchmark recordings."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["A1"])
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--contact-debug", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/lab/dual-arx"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    with IsaacRuntime() as runtime:
        from PIL import Image

        from roboclock.envs.lab.env import LabEnv
        from roboclock.envs.lab.policies.oracle import OraclePickPlace
        from roboclock.simulators.isaac.rendering import capture_camera

        env = LabEnv(runtime=runtime, oracle=True)
        env.config["sim"].update(overview_camera=True, overview_resolution=[640, 480])
        env.config["liquid_visuals"]["renderer"] = "RaytracedLighting"
        env.config["visuals"]["renderer"] = "RaytracedLighting"
        try:
            for task_id in args.tasks:
                if task_id == "C1":
                    from roboclock.envs.lab.rack_organization import configure_recording_logging

                    configure_recording_logging()
                env.task_id = task_id
                obs, _ = env.reset(seed=0)
                policy = OraclePickPlace(env)
                policy.reset(obs)
                contact_view = None
                if args.contact_debug:
                    import numpy as np
                    from isaacsim.core.prims import RigidPrim

                    from roboclock.envs.lab.robot import ROOTS

                    links = [
                        root + "/" + name
                        for root in ROOTS
                        for name in ("base_link", *(f"link{i}" for i in range(1, 9)))
                    ]
                    filters = (
                        links
                        + list(env.backend.surface_paths.values())
                        + [
                            "/World/Objects/sample_0",
                            "/World/ARXMount0",
                            "/World/ARXMount1",
                        ]
                    )
                    filters = [p for p in filters if env.backend.stage.GetPrimAtPath(p).IsValid()]
                    contact_view = RigidPrim(
                        links,
                        name="debug_contacts",
                        track_contact_forces=True,
                        reset_xform_properties=False,
                        contact_filter_prim_paths_expr=[filters for _ in links],
                        max_contact_count=16384,
                    )
                    contact_view.initialize()
                Image.fromarray(obs["rgb"]).save(args.output / f"{task_id}_initial.png")
                trace = []
                while not env.done and env.task.time < args.seconds:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    if not env.action_space.contains(action.astype("float32")):
                        raise RuntimeError(f"{task_id}: invalid action")
                    env.backend.apply_action(action)
                    env.task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                    if env.task.steps % 120 == 0 or env.done:
                        state = env.backend.state()
                        trace.append(
                            {
                                "time": env.task.time,
                                "phase": getattr(policy.planner, "phase", None),
                                "arm_phases": [
                                    {"phase": p.phase, "sample": p.sample}
                                    for p in policy.bimanual.planners
                                ]
                                if policy.bimanual
                                else [],
                                "controls": state.controls,
                                "secondary_ee": env.backend._tool_pose(env.backend.secondary_robot)[
                                    0
                                ].tolist(),
                                "measured_ee": env.backend._tool_pose()[0].tolist(),
                                "joints": env.backend.joint_positions().tolist(),
                                "objects": {
                                    n: {"grasped": o.grasped, "position": o.position.tolist()}
                                    for n, o in state.objects.items()
                                },
                            }
                        )
                        print(
                            task_id,
                            round(env.task.time, 2),
                            trace[-1]["phase"],
                            trace[-1]["measured_ee"],
                            flush=True,
                        )
                        if contact_view is not None:
                            forces = np.linalg.norm(
                                contact_view.get_contact_force_matrix(
                                    dt=env.config["sim"]["physics_dt"]
                                ),
                                axis=-1,
                            )
                            trace[-1]["contacts"] = [
                                [links[i], filters[j], float(forces[i, j])]
                                for i, j in np.argwhere(forces > 0.05)
                            ]
                capture_camera(4)
                Image.fromarray(env.backend.camera.get_rgba()[..., :3]).save(
                    args.output / f"{task_id}_final.png"
                )
                result = {
                    "simulator": simulator_version(),
                    "ik_failures": policy.ik_failures,
                    "episode": env.episode_metadata(),
                    "metrics": env.task.metrics(),
                    "events": env.task.event_log,
                    "trace": trace,
                }
                (args.output / f"{task_id}.json").write_text(json.dumps(result, indent=2) + "\n")
                print("RESULT", task_id, json.dumps(result["metrics"]), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
