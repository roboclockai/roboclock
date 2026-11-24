"""``roboclock render`` / ``roboclock validate`` for the Dual-X5 household environment."""

import json
from pathlib import Path

import numpy as np


def export_render(env, output, *, usd=False):
    """Stills of the policy cameras (head, both wrists) and the overview at the current state.

    No physics step. Needs rendered observations (not ``sim.no_render``).
    """
    from isaacsim.sensors.camera import Camera
    from PIL import Image

    from roboclock.envs.household.x5 import robodojo as RDH
    from roboclock.simulators.isaac.runtime import simulator_version

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    episode = env.x5
    if episode.no_render:
        raise ValueError("roboclock render needs the policy cameras; sim.no_render is set")
    before = float(episode.world.current_time)
    shots = {"agent": (episode.cameras["head"], path)}
    for side in RDH.SIDES:
        shots[RDH.VIEWS[side]] = (episode.cameras[side], path.with_name(f"{path.stem}_{RDH.VIEWS[side]}{path.suffix}"))
    overview = None
    if episode.stage.GetPrimAtPath("/World/Camera").IsValid():
        overview = Camera("/World/Camera", resolution=(1280, 960))
        overview.initialize()
        shots["overview"] = (overview, path.with_name(path.stem + "_overview" + path.suffix))
    written = {}
    try:
        for name, (camera, target) in shots.items():
            rgba = None
            for _ in range(60):  # world.render() renders without stepping physics
                episode.world.render()
                rgba = camera.get_rgba()
                if (
                    rgba is not None
                    and np.asarray(rgba).ndim == 3
                    and np.asarray(rgba)[..., :3].std() > 1
                ):
                    break
            if rgba is None or np.asarray(rgba).ndim != 3:
                raise RuntimeError(f"No {name} camera frame")
            Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(target)
            written[name] = target.name
    finally:
        if overview is not None:
            overview.destroy()
    after = float(episode.world.current_time)
    if abs(after - before) > 1e-9:
        raise RuntimeError("Snapshot capture advanced physics time")
    metadata = {
        "simulator": "isaacsim==" + simulator_version(),
        "environment": env.environment_id,
        "renderer": env.config["visuals"]["renderer"],
        "task": env.task.task_id,
        "difficulty": env.task.spec.difficulty,
        "seed": env.task.spec.seed,
        "sim_time": env.task.time,
        "physics_time_before_capture": before,
        "physics_time_after_capture": after,
        "resolution": env.config["sim"]["camera_resolution"],
        "renderer_readback": episode.renderer,
        "robodojo_rig_check": episode.rig,
        "images": written,
        "agent_camera": episode.camera_path,
    }
    if usd:
        episode.stage.Export(str(path.with_suffix(".usda")))
        metadata["usd"] = path.with_suffix(".usda").name
        metadata["usd_note"] = (
            "Flattened from the episode stage; kitchen and X5 assets must stay reachable."
        )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def validate_physics(
    env, plan, output, *, seconds=24.0, controller=True, images=False, rack_probes=False
):
    """Run each planned episode for up to ``seconds``: oracle-controlled, or passive (hold).

    Passive runs check that the settled scene stays physically valid (no drops or tipped
    objects reported by the task model). Controlled runs use the privileged oracle controller;
    their check is that it reaches no controller failure and the task model reports no
    physical failure.
    """
    from roboclock.policies.base import HoldPolicy

    if rack_probes:
        raise ValueError("Laboratory rack probes do not apply to the household scenes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(plan):
        obs, _ = env.reset(seed=item["seed"], options=item)
        if controller:
            from roboclock.envs.household.x5.oracle import HouseholdX5Oracle

            policy = HouseholdX5Oracle(env)
        else:
            policy = HoldPolicy()
        policy.reset(obs)
        stem = f"{index:04d}_{item['task_id']}_{item['difficulty']}_{item['seed']}"
        if images:
            export_render(env, output / f"{stem}_initial.png")
        while not env.done and env.task.time < seconds - 1e-9:
            obs, *_ = env.step(policy.act(obs))
        if images:
            export_render(env, output / f"{stem}_final.png")
        metrics = env.task.metrics()
        failures = list(getattr(policy, "failures", [])) + list(
            getattr(env.x5.probe, "failures", []) if controller else []
        )
        record = {
            **item,
            "seconds": env.task.time,
            "controller": "oracle" if controller else "hold",
            "physical_failures": metrics["physical_failures"],
            "controller_failures": failures,
            "task_success": metrics["task_success"],
            "temporal_success": metrics["temporal_success"],
            "joint_success": metrics["joint_success"],
            "physics_check_passed": not metrics["physical_failures"] and not failures,
            "metrics": metrics,
        }
        (output / f"{stem}.json").write_text(json.dumps(record, indent=2, allow_nan=False))
        records.append(record)
    return {
        "environment": env.environment_id,
        "episodes": len(records),
        "physics_checks_passed": sum(r["physics_check_passed"] for r in records),
        "config": env.config,
        "plan": plan,
        "results": [{k: v for k, v in r.items() if k != "metrics"} for r in records],
    }
