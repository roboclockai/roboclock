"""Audit A4 (hang and start infusion) clearances or run one bounded, real-physics diagnostic episode.

Geometry only (no simulator):
  python scripts/hospital/check_bedside_iv_bag.py --mode geometry

Isaac Sim interpreter with its graphics environment:
  python scripts/hospital/check_bedside_iv_bag.py --mode oracle --level hard --seed 0

The oracle mode saves actual simulator images, measured pose/contact samples, the
exported stage and outcomes. A geometry audit is never reported as a physics test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=serializable) + "\n", encoding="utf-8")


def geometry_audit(level, seed):
    from roboclock.envs.hospital.bedside.iv_bag import BAG, INSERT_DEPTH, PORT_LENGTH, geometry
    from roboclock.envs.hospital.config import load_config
    from roboclock.envs.hospital.layout import make_layout
    from roboclock.envs.hospital.tasks.registry import make_task
    from roboclock.envs.lab.mdp.state import ObjectState
    from roboclock.envs.lab.robot import mount_positions

    config = load_config()
    task = make_task("A4", level, seed, config)
    task.bind(make_layout(task, config))
    info = task.layout.info["iv_bag"]
    target = np.asarray(info["hang_position"])
    orientation = task.layout.objects[BAG].orientation
    arrival = geometry(task.layout, ObjectState(target + [0.0, 0.0, 0.026], orientation))
    placed = geometry(task.layout, ObjectState(target, orientation))
    tip = np.asarray(info["spike_tip"])
    body_bottom = target[2] - info["body_size"][2] / 2
    assert arrival["depth"] < 0.0, "Spike collides with sleeve before vertical insertion"
    assert placed["inserted"], "Final hanging pose does not retain the inserted spike"
    assert body_bottom - tip[2] >= 0.006, "Spike hits the solid pouch body"
    assert placed["depth"] >= INSERT_DEPTH
    assert not placed["hanging"], "A geometric target must not fabricate hook contact"
    # Bedside placement: the cradle stands on the cart top, the only (invisible) support collider.
    surface = task.layout.info["work_surface"]
    low = np.asarray(surface["center"]) - np.asarray(surface["size"]) / 2
    high = np.asarray(surface["center"]) + np.asarray(surface["size"]) / 2
    for fixture in task.layout.fixtures:
        if fixture.name.startswith("iv_cradle"):
            half = fixture.size / 2
            assert np.all(fixture.position[:2] - half[:2] >= low[:2]), fixture.name
            assert np.all(fixture.position[:2] + half[:2] <= high[:2]), fixture.name
            assert abs(fixture.position[2] - half[2] - high[2]) < 1e-9, fixture.name
    # The START key lies over the cart top; its bearing from the left arm is well off the bag's,
    # because the top-down X5's wrist sits radially beyond a pressed key.
    key = np.asarray(task.layout.info["iv_bag"]["start_key"]["origin"])
    assert np.all(key[:2] > low[:2]) and np.all(key[:2] < high[:2]), "START key off the cart"
    mount = np.asarray(mount_positions(config, "A4")[0])
    bearing = np.degrees(
        np.arctan2(*(key[:2] - mount[:2])[::-1]) - np.arctan2(*(target[:2] - mount[:2])[::-1])
    )
    assert bearing > 30.0, "START key on the bag's bearing"
    return {
        "validation_kind": "analytical_layout_audit_only",
        "physics_executed": False,
        "task": "A4",
        "difficulty": level,
        "seed": seed,
        "response_limit_s": task.spec.response_limit,
        "sleeve_length_m": PORT_LENGTH,
        "insertion_requirement_m": INSERT_DEPTH,
        "spike_depth_at_hanging_pose_m": placed["depth"],
        "solid_body_clearance_m": body_bottom - tip[2],
        "transfer_spike_clearance_m": -arrival["depth"],
        "support_verification_s": task.hang_dwell,
        "initial_ee_position": task.layout.info["initial_ee_position"],
        "start_key": task.layout.info["iv_bag"]["start_key"],
        "start_key_bearing_off_bag_deg": bearing,
        "hook_height_m": task.layout.info["iv_bag"]["hook_center"][2],
        "pump": task.layout.info["iv_bag"]["pump"],
        "bench_height_m": task.layout.table_height,
        "work_surface": task.layout.info["work_surface"],
        "ward_pose": task.layout.info["ward_pose"],
        "source_asset": info["visual_asset"],
        "assumptions": info["assumptions"],
    }


def run_oracle(args, output):
    # HospitalEnv starts SimulationApp before importing the Isaac backend.
    from roboclock.envs.hospital.env import HospitalEnv

    # SimulationApp forwards these supported Kit overrides after its configured defaults.
    script_argv = sys.argv
    sys.argv = [script_argv[0]]
    if args.gpu is not None:
        sys.argv += [
            f"--/renderer/activeGpu={args.gpu}",
            f"--/physics/cudaDevice={args.gpu}",
            "--/renderer/multiGpu/enabled=false",
            "--/renderer/multiGpu/maxGpuCount=1",
        ]
    try:
        env = HospitalEnv(task_id="A4", difficulty=args.level, oracle=True, headless=True)
    finally:
        sys.argv = script_argv
    result = {
        "validation_kind": "bounded_actual_simulator_oracle_diagnostic",
        "requested_simulation_seconds": args.seconds,
        "seed": args.seed,
        "selected_gpu": args.gpu,
    }
    traces = []
    try:
        from PIL import Image

        from roboclock.envs.hospital.bedside.iv_bag import BAG, START_KEY, start_key_state
        from roboclock.envs.hospital.policies.oracle import OraclePickPlace

        observation, _ = env.reset(seed=args.seed)
        Image.fromarray(observation["rgb"]).save(output / "initial.png")
        policy = OraclePickPlace(env)
        policy.reset(observation)
        while env.task.time < args.seconds and not env.done:
            action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
            env.backend.apply_action(action)
            env.task.action_count += 1
            env._ticks(env.config["sim"]["control_decimation"], render=False)
            state = env.backend.state()
            obj = state.objects[BAG]
            traces.append(
                {
                    "time": env.task.time,
                    "phase": getattr(policy.planner, "iv_bag_phase", "uninitialized"),
                    "ee_position": state.ee_position,
                    "commanded_ee": policy.planner.last_goal,
                    "bag_position": obj.position,
                    "bag_orientation": obj.orientation,
                    "bag_velocity": obj.velocity,
                    "grasped": obj.grasped,
                    "robot_contact": obj.robot_contact,
                    "fixture_contacts": obj.fixture_contacts,
                    "measurements": state.controls.get("iv_bag", {}),
                    "start_key": start_key_state(env.task.layout, state.objects[START_KEY]),
                }
            )
        env.backend.capture_camera(subframes=4)
        Image.fromarray(env.render()).save(output / "final.png")
        env.backend.stage.Export(str(output / "scene.usda"))
        result.update(
            {
                "physics_executed": True,
                "metrics": env.task.metrics(),
                "events": env.task.event_log,
                "metadata": env.episode_metadata(),
                "benchmark": env.benchmark_metadata(),
                "diagnostic_stopped_at_limit": not env.done,
                "elapsed_simulation_seconds": env.task.time,
            }
        )
        write_json(output / "measurements.json", traces)
        write_json(output / "result.json", result)
        print(
            json.dumps({"output": str(output), "metrics": result["metrics"]}, default=serializable),
            flush=True,
        )
    except Exception as exc:
        result.update({"error": str(exc), "exception_type": type(exc).__name__})
        write_json(output / "measurements.json", traces)
        write_json(output / "result.json", result)
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("geometry", "oracle"), default="geometry")
    parser.add_argument("--level", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument(
        "--gpu", type=int, help="Explicit physical GPU index; disables multi-GPU rendering"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/hospital/A4/diagnostic")
    )
    args = parser.parse_args()
    if not 0 < args.seconds <= 30:
        parser.error("--seconds must be between 0 and 30")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite diagnostic evidence: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    audit = geometry_audit(args.level, args.seed)
    write_json(args.output / "geometry.json", audit)
    if args.mode == "geometry":
        print(json.dumps(audit, indent=2, default=serializable))
    else:
        run_oracle(args, args.output.resolve())


if __name__ == "__main__":
    main()
