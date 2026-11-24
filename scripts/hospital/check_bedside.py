"""Actual-physics A1-A4 diagnostic; render only at initial/final state to keep clocks separate."""

import argparse
import json
import sys
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=["A1", "A2", "A3", "A4"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, help="Optional seeds for one simulator session")
    p.add_argument("--level", default="easy")
    p.add_argument("--seconds", type=float, default=22)
    p.add_argument("--gpu", type=int, help="Physical GPU index; disables multi-GPU rendering")
    p.add_argument("--output", type=Path, default=Path("results/hospital/bedside-v2/diagnostic"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    script_argv = sys.argv
    sys.argv = [script_argv[0]]
    if a.gpu is not None:
        sys.argv += [
            f"--/renderer/activeGpu={a.gpu}",
            f"--/physics/cudaDevice={a.gpu}",
            "--/renderer/multiGpu/enabled=false",
            "--/renderer/multiGpu/maxGpuCount=1",
        ]
    try:
        runtime = IsaacRuntime()
    finally:
        sys.argv = script_argv
    from PIL import Image

    from roboclock.envs.hospital.env import HospitalEnv
    from roboclock.envs.hospital.policies.oracle import OraclePickPlace

    try:
        for tid, seed in [(t, s) for s in (a.seeds or [a.seed]) for t in a.tasks]:
            output = a.output / f"{tid}_{a.level}_{seed}"
            output.mkdir(exist_ok=True)
            if any(output.iterdir()):
                raise FileExistsError(f"Refusing to overwrite diagnostic evidence: {output}")
            env = HospitalEnv(task_id=tid, difficulty=a.level, oracle=True, runtime=runtime)
            try:
                obs, _ = env.reset(seed=seed)
                Image.fromarray(obs["rgb"]).save(output / "initial.png")
                policy = OraclePickPlace(env)
                policy.reset(obs)
                traces = []
                while env.task.time < a.seconds and not env.done:
                    action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                    env.backend.apply_action(action)
                    env.task.action_count += 1
                    env._ticks(env.config["sim"]["control_decimation"], render=False)
                    state = env.backend.state()
                    traces.append(
                        {
                            "time": env.task.time,
                            "controls": state.controls,
                            "ee": state.ee_position,
                            "phase": getattr(policy.planner, "iv_bag_phase", policy.planner.phase),
                            "finger": env.backend.joint_positions()[6:8],
                            "objects": {
                                n: {
                                    "position": o.position,
                                    "orientation": o.orientation,
                                    "grasped": o.grasped,
                                    "contact": o.robot_contact,
                                    "fixtures": o.fixture_contacts,
                                }
                                for n, o in state.objects.items()
                            },
                        }
                    )
                # Physics-only stepping batches USD material changes. Flush RTX
                # subframes at zero delta-time so the final cue is visible too.
                env.backend.capture_camera(subframes=4)
                Image.fromarray(env.render()).save(output / "final.png")
                result = {
                    "simulator": simulator_version(),
                    "physics_executed": True,
                    "selected_gpu": a.gpu,
                    "configuration": env.config,
                    "asset_provenance": json.loads(
                        (Path(__file__).resolve().parents[2] / "docs/hospital-bedside-assets.json")
                        .read_text()
                    ),
                    "indicator_states": {
                        name: dial.color for name, dial in env.backend.timer_bars.items()
                    },
                    "scope": "one seed; privileged diagnostic; not validated expert policy",
                    "metrics": env.task.metrics(),
                    "events": env.task.event_log,
                    "episode": env.episode_metadata(),
                }
                (output / "measurements.json").write_text(
                    json.dumps(traces, default=plain, indent=2)
                )
                (output / "result.json").write_text(json.dumps(result, default=plain, indent=2))
                print(tid, json.dumps(result["metrics"], default=plain), flush=True)
            except Exception as exc:
                (output / "error.json").write_text(json.dumps({"error": str(exc)}))
                raise
            finally:
                env.close()
    except BaseException as exc:
        runtime.exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
