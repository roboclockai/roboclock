"""Projection test: every lab task's key objects and cues inside the RoboDojo head image.

For each task, difficulty and seed the scene is reset (``sim.no_render`` unless
``--render``), the geometry of every key prim is taken from the stage and its
outline points (mesh/curve vertices, or the corners of each implicit shape's oriented
box) are projected into the 640x480 head camera (robodojo_rig pose and
intrinsics). A key prim passes when all corners lie in front of the camera and at least
``--margin`` px inside the image. The projection uses the square-pixel intrinsics the
renderer produces (``robodojo_rig.intrinsics``: fy = fx, 96.0 x 79.6 deg); the fy Isaac
reports from the 14.266 mm vertical aperture (71.0 deg) is not the rendered image.
``--conservative`` projects with that aperture model instead (stricter vertically).

With ``--render`` the head frame is also rendered and saved with the projected boxes,
so the projection can be compared with the actual image.

    python scripts/lab/check_head_visibility.py --tasks A1 B2 --output /tmp/visibility
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Key prim groups per task: each visible child is a task object, fixture, target or cue.
# Clutter (distractors) and room decoration are not key objects.
LIQUID = ("/World/Objects", "/World/Apparatus", "/World/Process")
STATE = ("/World/Indicator",)  # generic STATE lamp of the laboratory_v2 tasks
KEY_GROUPS = {
    "A1": LIQUID,
    "A2": LIQUID,
    "A3": LIQUID,
    "A4": LIQUID,
    "A5": ("/World/Objects", "/World/FaultRig", *STATE),
    "A6": ("/World/Objects", "/World/FaultRig", *STATE),
    "B1": ("/World/Objects", "/World/Hotplate", "/World/HeatingZone", "/World/Labels", *STATE),
    "B2": ("/World/Objects", "/World/FaultRig", *STATE),
    "B3": ("/World/Objects", "/World/FaultRig", "/World/MotorWheel", *STATE),
    "B4": ("/World/Objects", "/World/FaultRig", *STATE),
    "B5": ("/World/Objects", "/World/FaultRig"),
    "B6": ("/World/Objects", "/World/FaultRig"),
    "B7": ("/World/Objects", "/World/FaultRig"),
    "C1": ("/World/Objects", "/World/TubeRacks", "/World/RackBands", "/World/Labels", *STATE),
    "C2": ("/World/Objects", "/World/Labels", *STATE),
    "C3": ("/World/Objects", "/World/Labels", *STATE),
}


def corners(low, high):
    return np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])


def world_points(prim, cache, limit=400):
    """Outline points of a key prim in the world frame.

    Mesh and curve geometry contributes its own vertices (a hose is not its bounding
    box); implicit shapes (cubes, cylinders, ...) the corners of their oriented box.
    """
    from pxr import Usd, UsdGeom

    points = []
    for item in Usd.PrimRange(prim):
        if not item.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(item).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if item.IsA(UsdGeom.PointBased):
            local = np.asarray(UsdGeom.PointBased(item).GetPointsAttr().Get() or [], dtype=float)
            if not len(local):
                continue
            if len(local) > limit:
                local = local[np.linspace(0, len(local) - 1, limit).astype(int)]
            matrix = np.array(
                UsdGeom.Xformable(item).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                dtype=float,
            )
            points.append((np.c_[local, np.ones(len(local))] @ matrix)[:, :3])
        else:
            bound = cache.ComputeWorldBound(item)
            box = bound.GetRange()
            if box.IsEmpty():
                continue
            local = corners(np.array(box.GetMin()), np.array(box.GetMax()))
            matrix = np.array(bound.GetMatrix(), dtype=float)
            points.append((np.c_[local, np.ones(8)] @ matrix)[:, :3])
    return np.concatenate(points) if points else np.zeros((0, 3))


def project(points, head_pose, intrinsics):
    """Pixel coordinates and camera depth of world points (USD camera axes, -Z forward)."""
    world_to_camera = np.linalg.inv(head_pose)
    local = (world_to_camera @ np.c_[points, np.ones(len(points))].T)[:3].T
    depth = -local[:, 2]
    fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + fx * local[:, 0] / depth
        v = cy - fy * local[:, 1] / depth
    return np.c_[u, v], depth


def margin_px(pixels, depth, size):
    if np.any(depth <= 0.01):
        return -math.inf
    width, height = size
    return float(
        min(pixels[:, 0].min(), width - pixels[:, 0].max(), pixels[:, 1].min(), height - pixels[:, 1].max())
    )


# Structural supports of the A2-A4 glassware (retort stand rod, clamps, cross arms). The
# hanging vessels, their liquids and scales are key; the tops of these rods are not.
SUPPORTS = ("StandRod", "Clamp", "ColumnSupport", "BuretteArm")


def key_prims(stage, task_id):
    from pxr import UsdGeom

    prims = []
    for root in KEY_GROUPS[task_id]:
        prim = stage.GetPrimAtPath(root)
        if not prim.IsValid():
            continue
        children = [p for p in prim.GetChildren() if p.IsA(UsdGeom.Imageable)] or [prim]
        prims.extend(
            p
            for p in children
            if UsdGeom.Imageable(p).ComputeVisibility() != UsdGeom.Tokens.invisible
            and not p.GetName().startswith(SUPPORTS)
        )
    return prims


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--render", action="store_true", help="Also save annotated head frames")
    parser.add_argument("--keep-points", action="store_true", help="Store projected points")
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="Project with the reported fy (vertical aperture honoured) instead of the rendered square pixels",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime(headless=True) as runtime:
        from pxr import Usd, UsdGeom

        from roboclock.envs import robodojo_rig
        from roboclock.envs.lab.env import LabEnv
        from roboclock.envs.lab.robot import rig_pose

        env = LabEnv(task_id=args.tasks[0], runtime=runtime, oracle=True)
        env.config["sim"]["no_render"] = not args.render
        intrinsics = robodojo_rig.intrinsics(robodojo_rig.HEAD_CAMERA, square_pixels=not args.conservative)
        size = robodojo_rig.HEAD_CAMERA["resolution"]
        report = {}
        failures = 0
        try:
            for task_id in args.tasks:
                for difficulty in args.difficulties:
                    for seed in args.seeds:
                        obs, _ = env.reset(
                            seed=seed, options={"task_id": task_id, "difficulty": difficulty}
                        )
                        origin, yaw = rig_pose(env.config, task_id)
                        head = robodojo_rig.head_camera_world(origin, math.degrees(yaw))
                        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
                        rows = []
                        for prim in key_prims(env.backend.stage, task_id):
                            points = world_points(prim, cache)
                            if not len(points):
                                continue
                            pixels, depth = project(points, head, intrinsics)
                            hull = np.r_[points.min(axis=0), points.max(axis=0)]
                            rows.append(
                                {
                                    "prim": str(prim.GetPath()),
                                    "margin_px": margin_px(pixels, depth, size),
                                    "world_range": hull.round(4).tolist(),
                                    "pixel_range": [
                                        *pixels.min(axis=0).round(1).tolist(),
                                        *pixels.max(axis=0).round(1).tolist(),
                                    ],
                                }
                            )
                            if args.keep_points:  # for offline rig placement searches
                                rows[-1]["points"] = (
                                    points[:: max(1, len(points) // 200)].round(4).tolist()
                                )
                        bad = [row for row in rows if row["margin_px"] < args.margin]
                        failures += bool(bad)
                        key = f"{task_id}/{difficulty}/{seed}"
                        report[key] = {
                            "passed": not bad,
                            "minimum_margin_px": min((r["margin_px"] for r in rows), default=None),
                            "outside": [(r["prim"], round(r["margin_px"], 1)) for r in bad],
                            "prims": rows,
                        }
                        print(
                            f"VISIBILITY {key} {'PASS' if not bad else 'FAIL'} "
                            f"min={report[key]['minimum_margin_px']:.1f}px "
                            f"outside={report[key]['outside']}",
                            flush=True,
                        )
                        if args.render and difficulty == args.difficulties[0] and seed == args.seeds[0]:
                            from PIL import Image, ImageDraw

                            image = Image.fromarray(obs["rgb"])
                            draw = ImageDraw.Draw(image)
                            for row in rows:
                                if not np.all(np.isfinite(row["pixel_range"])):
                                    continue
                                color = (0, 255, 0) if row["margin_px"] >= args.margin else (255, 0, 0)
                                draw.rectangle(row["pixel_range"], outline=color)
                            image.save(args.output / f"{task_id}_head_projection.png")
                            for name in ("left_wrist_rgb", "right_wrist_rgb"):
                                Image.fromarray(obs[name]).save(args.output / f"{task_id}_{name}.png")
        finally:
            (args.output / "visibility.json").write_text(json.dumps(report, indent=1))
            env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
