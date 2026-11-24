"""Reproducible actual-PhysX scene and trajectory diagnostics.

Short rollouts validate dynamics; they are not full-budget benchmark episodes.
The privileged controller uses proprioception/contact feedback. RGB snapshots
are rendered at setup and export, not supplied as fresh frames in these runs.
"""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from roboclock.config import PACKAGE_ROOT
from roboclock.envs.lab.appearance import PALETTE
from roboclock.envs.lab.mdp.geometry import object_extents
from roboclock.simulators.isaac.runtime import simulator_version


def export_render(env, output, *, usd=False):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    from roboclock.simulators.isaac.rendering import capture_camera

    # Explicit subframes converge materials and path tracing with zero physics delta.
    before = env.backend.world.current_time
    capture_camera(env.config["visuals"].get("render_updates", 72))
    after = env.backend.world.current_time
    if not np.isclose(before, after, atol=1e-12, rtol=0):
        raise RuntimeError("Snapshot capture advanced physics time")
    # Lab policy view: the rig head camera (no overview camera outside recordings).
    rig = getattr(env.backend, "rig_cameras", None)
    camera = env.backend.camera or (rig.cameras.get("cam_head") if rig is not None else None)
    if camera is None:
        raise RuntimeError("No camera to render (sim.no_render)")
    Image.fromarray(camera.get_rgba()[..., :3]).save(path)
    metadata = {
        "simulator": "isaacsim==" + simulator_version(),
        "renderer": env.backend.appearance.config["visuals"].get("renderer", "RaytracedLighting"),
        "environment": env.environment_id,
        "texture_sources": "lab://textures/sources.json",
        "material_models": ["OmniPBR", "OmniGlass", "UsdPreviewSurface labels"],
        "task": env.task.task_id,
        "difficulty": env.task.spec.difficulty,
        "seed": env.task.spec.seed,
        "sim_time": env.task.time,
        "physics_time_before_capture": before,
        "physics_time_after_capture": after,
        "resolution": [int(v) for v in camera.get_resolution()],
        "scene": env.config["scene"],
        "visuals": env.backend.appearance.config["visuals"],
        "material_palette": PALETTE,
        "labels": "Visible sample IDs and matching target IDs; no hidden event times.",
        "proxy_assets": [
            name for name, spec in env.backend.catalog.items() if not spec["usd_path"]
        ],
    }
    if usd:
        env.backend.stage.Export(str(path.with_suffix(".usda")))
        metadata["usd"] = path.with_suffix(".usda").name
        metadata["usd_note"] = (
            "Referenced ARX X5/HDR/label textures must remain accessible on replay."
        )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _scene_checks(env):
    task, state = env.task, env.backend.state()
    if getattr(task, "liquid_process", False):
        errors = []
        for name, obj in state.objects.items():
            if not np.all(np.isfinite([*obj.position, *obj.orientation, *obj.velocity])):
                errors.append(f"{name}: nonfinite physical fixture state")
        return errors
    errors = []
    for name, obj in state.objects.items():
        if not np.all(np.isfinite([*obj.position, *obj.orientation, *obj.velocity])):
            errors.append(f"{name}: nonfinite rigid state")
        bottom = obj.position[2] - object_extents(task, name, obj)[2]
        support_z = (
            task.layout.heating_zone.support_z
            if getattr(task, "hotplate_task", False)
            else task.layout.table_height
        )
        if getattr(task, "rack_organization", False):
            from roboclock.envs.lab.rack_organization import rack_spec

            support_z += rack_spec()["support_z"]
        if getattr(task, "bath_task", False):
            from roboclock.envs.lab.bath_scene import FLOOR

            support_z += FLOOR
        if getattr(task, "tube_task", False):
            support_z += task.rack_support
        if getattr(task, "safety_task", False):
            initial = task.layout.objects[name].position
            if np.linalg.norm(obj.position - initial) > 0.004:
                errors.append(f"{name}: actuator moved during setup")
        elif getattr(task, "power_task", False):
            initial = task.layout.objects[name].position
            if np.linalg.norm(obj.position - initial) > 0.003:
                errors.append(f"{name}: power switch is not at its initial joint stop")
        elif task.task_id != "B3" and abs(bottom - support_z) > 0.008:
            errors.append(f"{name}: initial bottom is not supported by bench ({bottom:.5f})")
        if getattr(task, "hotplate_task", False) and "heater" not in obj.fixture_contacts:
            errors.append(f"{name}: initial beaker lacks measured hotplate contact")
    names = list(state.objects)
    for i, name in enumerate(names):
        a = state.objects[name]
        extent_a = object_extents(task, name, a)
        for other in names[i + 1 :]:
            b = state.objects[other]
            extent_b = object_extents(task, other, b)
            if np.all(np.abs(a.position - b.position) < extent_a + extent_b - 0.001):
                errors.append(f"initial object overlap: {name}/{other}")
    for name, target in task.layout.targets.items():
        if target.kind == "slot":
            for key in task.layout.objects:
                if task.layout.assignments[key] == name:
                    if np.any(target.half_extents[:2] <= task.layout.dimensions[key][0] + 0.001):
                        errors.append(f"{name}: slot has insufficient clearance")
    for item in getattr(env.backend, "clutter_specs", []):
        position, orientation = env.backend.clutter[item.name].get_world_pose()
        if not np.all(np.isfinite([*position, *orientation])):
            errors.append(f"{item.name}: nonfinite rigid state")
        if abs(position[2] - item.size[2] / 2 - task.layout.table_height) > 0.004:
            errors.append(f"{item.name}: initial clutter is not supported by bench")
        if np.linalg.norm(position[:2] - item.position[:2]) > 0.008:
            errors.append(f"{item.name}: clutter shifted during initial settling")
    return errors


def validate_physics(
    env, plan, output, *, seconds=24.0, controller=True, images=False, rack_probes=False
):
    from roboclock.envs.lab.policies.oracle import OraclePickPlace

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        gpu = "unavailable"
    code_hashes = {
        str(p.relative_to(PACKAGE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(PACKAGE_ROOT.rglob("*.py"))
    }
    manifest = {
        "environment": env.environment_id,
        "simulator": "isaacsim==" + simulator_version(),
        "python": platform.python_version(),
        "gpu": gpu,
        "physics_dt": env.config["sim"]["physics_dt"],
        "config": env.config,
        "assets": env.backend.catalog,
        "plan": plan,
        "seconds": seconds,
        "rack_probes": rack_probes,
        "code_sha256": code_hashes,
        "purpose": "physics and controller diagnostics; partial throughput rollouts",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    dt = env.config["sim"]["physics_dt"]
    decimation = env.config["sim"]["control_decimation"]
    for index, episode in enumerate(plan):
        observation, _ = env.reset(seed=episode["seed"], options=episode)
        probe = probe_rack(env) if rack_probes else None
        if probe is not None:
            observation, _ = env.reset(seed=episode["seed"], options=episode)
        errors = _scene_checks(env)
        if probe is not None and not probe["pass"]:
            errors.append("gravity rack seating experiment failed")
        task = env.task
        policy = OraclePickPlace(env) if controller else None
        if policy:
            policy.reset(observation)
        release_frame = None
        max_speed, max_relative_speed = 0.0, 0.0
        max_force = 0.0
        held_frames = 0
        last_offset = task.fixture_offset
        max_fixture_step = 0.0
        limit = min(seconds, task.spec.max_time)
        while task.time < limit - 1e-9 and not task.terminated:
            if policy:
                action = policy.act_internal({"joint_positions": env.backend.joint_positions()})
                env.backend.apply_action(action)
                task.action_count += 1
            for _ in range(decimation):
                if task.time >= limit - 1e-9 or task.terminated:
                    break
                commands = task.advance(dt)
                for command in commands:
                    env.backend.command(command)
                env.backend.tick(render=False)
                state = env.backend.state()
                task.observe(state, dt)
                task.collision_count = env.backend.collision_count
                obj = state.objects["sample_0"]
                speed = float(np.linalg.norm(obj.velocity))
                max_speed = max(max_speed, speed)
                if np.isfinite(obj.relative_tool_speed):
                    max_relative_speed = max(max_relative_speed, obj.relative_tool_speed)
                held_frames += sum(int(item.grasped) for item in state.objects.values())
                force = env.backend.contacts.get_contact_force_matrix(dt=dt)
                max_force = max(max_force, float(np.max(np.linalg.norm(force, axis=-1))))
                if not np.all(np.isfinite(force)) or not np.all(np.isfinite(obj.position)):
                    errors.append("nonfinite PhysX state/contact")
                    break
                if any(c.kind == "release" for c in commands):
                    release_frame = {
                        "velocity": obj.velocity.tolist(),
                        "position": obj.position.tolist(),
                        "speed": speed,
                        "applied_horizontal_speed": obj.release_speed,
                    }
                if task.task_id == "A4":
                    max_fixture_step = max(max_fixture_step, abs(task.fixture_offset - last_offset))
                    last_offset = task.fixture_offset
        if (
            task.task_id in ("B2", "B3")
            and not getattr(task, "power_task", False)
            and limit > task.spec.event_time
            and release_frame is None
        ):
            errors.append("release event was not exercised")
        if task.task_id == "A4" and not getattr(task, "liquid_process", False):
            expected = task.amplitude * task.omega * dt + 0.000001
            if max_fixture_step > expected:
                errors.append("moving fixture exceeded continuous kinematic speed bound")
        row = {
            **task.metrics(),
            "validation_errors": errors,
            "rack_probe": probe,
            "finite_and_geometry_valid": not errors,
            "rollout_seconds_requested": seconds,
            "full_budget_episode": task.time >= task.spec.max_time - 1e-9,
            "controller": "privileged_lula_ik" if controller else "passive",
            "release_frame": release_frame,
            "max_sample_speed": max_speed,
            "max_relative_tool_speed": max_relative_speed,
            "max_pair_contact_force": max_force,
            "confirmed_grasp_frames": held_frames,
            "max_fixture_step": max_fixture_step,
            "controller_phase": policy.planner.phase if policy else None,
            "events": task.event_log,
            "insertion_times": task.insertion_times,
            "final_objects": {
                name: {
                    "position": obj.position.tolist(),
                    "orientation": obj.orientation.tolist(),
                    "velocity": obj.velocity.tolist(),
                    "grasped": obj.grasped,
                    "contacts": sorted(obj.fixture_contacts),
                }
                for name, obj in env.backend.state().objects.items()
            },
        }
        rows.append(row)
        if images and episode["difficulty"] == "easy" and episode["seed"] == plan[0]["seed"]:
            export_render(env, output / f"{task.task_id}_final.png")
        (output / "physics.json").write_text(
            json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            f"[{index + 1}/{len(plan)}] {task.task_id} {task.spec.difficulty} seed={task.spec.seed} "
            f"checks={'PASS' if not errors else errors} physical={task.physical_success} "
            f"temporal={task.temporal_success} phase={row['controller_phase']}",
            flush=True,
        )
    summary = {
        "episodes": len(rows),
        "physics_checks_passed": sum(r["finite_and_geometry_valid"] for r in rows),
        "physical_goals_reached": sum(r["task_success"] for r in rows),
        "joint_goals_reached": sum(r["joint_success"] for r in rows),
        "short_diagnostic_rollouts": True,
        "seconds": seconds,
        "simulator": "isaacsim==" + simulator_version(),
        "gpu": gpu,
        "config": env.config,
        "plan": plan,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def probe_rack(env):
    """Separate fixture experiment: initialize above the mouth and let PhysX seat it.

    Pose initialization occurs before this experiment's first physics tick. No
    benchmark score is taken from these samples and no controller participates.
    """
    from roboclock.envs.lab.mdp.geometry import seated

    task = env.task
    if getattr(task, "liquid_process", False) or task.task_id not in ("A2", "A4", "C2"):
        return None
    names = list(env.backend.objects)
    start_positions = {}
    for name in names:
        target = task.layout.targets[task.layout.assignments[name]]
        position = target.position.copy()
        position[0] += task.fixture_offset if task.task_id == "A4" else 0.0
        position[2] = target.rim_z + task.layout.dimensions[name][1] / 2 + 0.018
        body = env.backend.objects[name]
        body.set_world_pose(position, np.array([1.0, 0.0, 0.0, 0.0]))
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))
        start_positions[name] = position.tolist()
    for _ in range(90):
        env.backend.tick(render=False)
    final = env.backend.state()
    outcomes = {}
    for name, obj in final.objects.items():
        target = task.layout.targets[task.layout.assignments[name]]
        fit = seated(task, name, obj, target, task.fixture_offset if task.task_id == "A4" else 0.0)
        contact = task.layout.assignments[name] in obj.fixture_contacts
        outcomes[name] = {
            "seated": bool(fit),
            "floor_contact": contact,
            "speed": float(np.linalg.norm(obj.velocity)),
            "position": obj.position.tolist(),
            "start_position": start_positions[name],
        }
    return {
        "experiment": "gravity_drop_from_above_rack",
        "simulated_seconds": 0.75,
        "pass": all(
            v["seated"] and v["floor_contact"] and v["speed"] < 0.08 for v in outcomes.values()
        ),
        "samples": outcomes,
        "benchmark_scores_used": False,
    }
