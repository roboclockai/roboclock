"""Projection test of what a hospital task shows in the RoboDojo head camera (requirement 6).

Resets one episode headless (sim.no_render: no camera, no render call), takes the world-aligned
bounding box of every prim the task authors (devices, objects, targets, dials, labels, equipment;
not the ward room, lights, robot or rig) and projects its 8 corners into the 640x480 head image with
`hospital.rig.head_pixels` (the square-pixel intrinsics RTX renders, 96.0 x 79.6 deg).
A prim passes when every corner lies at least --margin px inside the image. With --steps N it also
runs the oracle for N control steps and repeats the test on the moved objects (sampled every 25).

    PYTHONPATH=src python scripts/hospital/head_view_probe.py B4 hard 0 results/hospital/head_view/B4.json

One episode per process (Isaac Sim 6.0.1 hangs on a second reset).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

SKIP_ROOTS = {
    "ARXLeft", "ARXRight", "ARXMount0", "ARXMount1", "BedsideWard", "Lighting", "Lights", "Rig",
    "HeadCamera", "Camera", "defaultGroundPlane", "Table", "TableVisual", "Materials", "Joints",
    "Looks", "Clutter", "Ward", "Environment", "Physics", "PhysicsScene", "Room", "Walls", "Floor",
    "MobileBase", "Hdr", "DomeLight",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task")
    p.add_argument("difficulty")
    p.add_argument("seed", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--margin", type=float, default=10.0)
    p.add_argument("--depth", type=int, default=2, help="prim depth below /World to test")
    p.add_argument("--steps", type=int, default=0, help="oracle control steps after reset")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    from roboclock.envs.hospital.config import load_config

    config = load_config()
    config["sim"]["no_render"] = True
    config_path = a.out.with_suffix(".config.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime(renderer=config["visuals"]["renderer"]) as runtime:
        from pxr import Usd, UsdGeom

        from roboclock.envs.hospital.env import HospitalEnv
        from roboclock.envs.hospital.rig import head_pixels

        env = HospitalEnv(runtime=runtime, task_id=a.task, difficulty=a.difficulty, config_path=config_path, oracle=True)
        try:
            obs, _ = env.reset(seed=a.seed)
            stage = env.backend.stage
            width, height = config["sim"]["camera_resolution"]

            def prims():
                world = stage.GetPrimAtPath("/World")
                frontier = [(c, 1) for c in world.GetChildren() if c.GetName() not in SKIP_ROOTS]
                # Untyped groups at /World level (e.g. /World/Bedside) count as roots as well.
                while frontier:
                    prim, depth = frontier.pop(0)
                    # Modules author parts under untyped grouping prims: descend through them too.
                    children = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Imageable) or not c.GetTypeName()]
                    if depth >= a.depth or not children:
                        yield prim
                    else:
                        frontier += [(c, depth + 1) for c in children]

            def test(time_label):
                cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
                rows = {}
                for prim in prims():
                    if prim.IsA(UsdGeom.Imageable):
                        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
                            continue
                    elif prim.GetTypeName():
                        continue  # materials, shaders, joints
                    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    if box.IsEmpty():
                        continue
                    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
                    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                    pixels, depth = head_pixels(config, a.task, corners)
                    if np.any(depth <= 0.05):
                        margin = -1e9
                    else:
                        margin = float(min(pixels[:, 0].min(), width - pixels[:, 0].max(), pixels[:, 1].min(), height - pixels[:, 1].max()))
                    rows[str(prim.GetPath())] = {
                        "margin_px": round(margin, 1),
                        "inside": margin >= a.margin,
                        "size_m": (hi - lo).round(3).tolist(),
                        "center": ((hi + lo) / 2).round(3).tolist(),
                        "pixels_min": pixels.min(0).round(1).tolist() if margin > -1e9 else None,
                        "pixels_max": pixels.max(0).round(1).tolist() if margin > -1e9 else None,
                    }
                return {"time": time_label, "prims": rows}

            snapshots = [test(0.0)]
            if a.steps:
                from roboclock.envs.hospital.policies.oracle import OraclePickPlace

                policy = OraclePickPlace(env)
                policy.reset(obs)
                for step in range(a.steps):
                    if env.done:
                        break
                    obs = env.step(policy.act(obs))[0]
                    if (step + 1) % 25 == 0:
                        snapshots.append(test(round(env.task.time, 3)))
            result = {
                "task": a.task, "difficulty": a.difficulty, "seed": a.seed, "margin_px": a.margin,
                "rig_check": env.backend.rig_check, "snapshots": snapshots,
                "outside_at_reset": sorted(k for k, v in snapshots[0]["prims"].items() if not v["inside"]),
            }
            a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
            print("OUTSIDE", a.task, result["outside_at_reset"], flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
