"""Audit the actual biped skin over a recorded bar trajectory (no physics rerun).

Run with Isaac Sim's pxr library path or from an initialized Isaac Python shell.
The human is still a visual rig: this audit verifies geometry, not human forces.
"""

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.sports.bench_press.character import Athlete
from roboclock.envs.sports.bench_press.hand_surface import HandSurface
from roboclock.envs.sports.bench_press.process import smooth


def load_rig():
    from pxr import Usd, UsdGeom, UsdSkel

    stage = Usd.Stage.Open("src/roboclock/envs/sports/bench_press/assets/human/biped_demo_meters.usd")
    sk = UsdSkel.Skeleton(stage.GetPrimAtPath("/Biped_Demo/Root"))
    a = Athlete.__new__(Athlete)
    a.base = np.eye(4)
    a.base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    a.base[:3, 3] = [1.83, 0, 0.62]
    a.joints = list(sk.GetJointsAttr().Get())
    a.parents = np.array(UsdSkel.Topology(a.joints).GetParentIndices())
    a.rest = np.array([np.array(m).T for m in sk.GetRestTransformsAttr().Get()])
    a.ids = {j.rsplit("/", 1)[-1]: i for i, j in enumerate(a.joints)}
    a.b = SimpleNamespace(config={"workflow": {"human_grip_y": 0.23}})
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Biped_Demo/Body_Mesh"))
    binding = UsdSkel.BindingAPI(mesh)
    points = np.array(mesh.GetPointsAttr().Get())
    jmap = np.array([a.joints.index(j) for j in binding.GetJointsAttr().Get()])
    n = binding.GetJointIndicesPrimvar().GetElementSize()
    indices = jmap[np.array(binding.GetJointIndicesPrimvar().Get()).reshape(-1, n)]
    weights = np.array(binding.GetJointWeightsPrimvar().Get()).reshape(-1, n)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    ids = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    faces, start = [], 0
    for count in counts:
        face = ids[start : start + count]
        faces.extend((face[0], face[k], face[k + 1]) for k in range(1, count - 1))
        start += count
    bind = np.array([np.array(m).T for m in sk.GetBindTransformsAttr().Get()])
    forearm_joint = np.array(["_LoArm" in j for j in a.joints])
    a.forearm_mask = (forearm_joint[indices] * weights).sum(axis=1) > 0.75
    a.forearm_indices = indices[a.forearm_mask]
    a.forearm_weights = weights[a.forearm_mask]
    return a, HandSurface(points, indices, weights, bind, a.joints, np.array(faces))


def audit(run=None):
    a, surface = load_rig()
    cases = []
    # Include the full opening path and the operational height/orientation envelope.
    for height in (0.86, 1.08, 1.12):
        for tilt in (-3.0, 0.0, 3.0):
            orientation = np.roll(Rotation.from_euler("x", tilt, degrees=True).as_quat(), 1)
            for release in np.linspace(0, 1, 41):
                cases.append(("pose_sweep", [0.32, 0, height], orientation, release))
    if run:
        result = json.loads((run / "result.json").read_text())
        catch = result["catch_time"]
        with gzip.open(run / "trace.jsonl.gz", "rt") as f:
            for row in map(json.loads, f):
                o = row["state"]["objects"]["bar"]
                release = (
                    smooth((row["time"] - catch) / 0.8) if catch and row["time"] > catch else 0.0
                )
                cases.append(("recorded", o["position"], o["orientation"], release))
    records = []
    for kind, bar, orientation, release in cases:
        local = a.pose(bar, orientation, release)
        global_bones = a.fk(local)
        points, metric = surface.apply(global_bones, np.array(bar), orientation)
        matrices = (
            (global_bones @ surface.inverse_bind)[surface.indices]
            * surface.weights[..., None, None]
        ).sum(axis=1)
        v = np.einsum(
            "nij,nj->ni", matrices, np.c_[points[surface.selected], np.ones(len(surface.selected))]
        )[:, :3]
        # Conservative AABB distance to the existing torso safety proxy.
        q = np.abs(v - np.array([0.55, 0, 0.645])) - np.array([0.215, 0.155, 0.085])
        sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)
        metric["torso_proxy_clearance_m"] = float(sdf.min())
        arm_matrices = (
            (global_bones @ surface.inverse_bind)[a.forearm_indices]
            * a.forearm_weights[..., None, None]
        ).sum(axis=1)
        arm_vertices = np.einsum(
            "nij,nj->ni", arm_matrices, np.c_[points[a.forearm_mask], np.ones(a.forearm_mask.sum())]
        )[:, :3]
        bench_clearance = []
        for center, half in [
            ([0.65, 0, 0.485], [0.625, 0.145, 0.0375]),
            ([0.10, 0, 0.49], [0.145, 0.14, 0.0425]),
            ([0.65, 0, 0.40], [0.625, 0.045, 0.05]),
        ]:
            q = np.abs(arm_vertices - center) - half
            bench_clearance.append(
                float(
                    (
                        np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)
                    ).min()
                )
            )
        metric["forearm_vertex_bench_clearance_m"] = min(bench_clearance)
        if release >= 0.999:
            forward = []
            rest_error = []
            for side in ("L", "R"):
                wrist = global_bones[a.ids[side + "_Wrist"], :3, 3]
                thumb = global_bones[a.ids[side + "_ThumbFingerEnd"], :3, 3] - wrist
                distal = global_bones[a.ids[side + "_MiddleFinger2"], :3, 3] - wrist
                forward.append(
                    float(np.dot(thumb, distal) / (np.linalg.norm(thumb) * np.linalg.norm(distal)))
                )
                rest_error.extend(
                    np.max(
                        np.abs(
                            local[a.ids[f"{side}_ThumbFinger{k}"], :3, :3]
                            - a.rest[a.ids[f"{side}_ThumbFinger{k}"], :3, :3]
                        )
                    )
                    for k in (1, 2, 3)
                )
            metric["rest_thumb_forward_cosine"] = min(forward)
            metric["rest_thumb_source_rotation_error"] = float(max(rest_error))
            metric["rest_wrist_height_m"] = float(
                max(global_bones[a.ids[side + "_Wrist"], 2, 3] for side in ("L", "R"))
            )
            metric["rest_hand_max_height_m"] = float(v[:, 2].max())
        metric["left_right_order_correct"] = bool(
            global_bones[a.ids["L_Wrist"], 1, 3] > 0 and global_bones[a.ids["R_Wrist"], 1, 3] < 0
        )
        records.append(
            {
                "kind": kind,
                "release": float(release),
                "bar": list(bar),
                "orientation": list(orientation),
                **metric,
            }
        )
    report = {
        "scope": "Visual skinned hand triangles against shaft, hand vertices against torso proxy, forearm vertices against cushion/headpad/spine boxes, resting heights and side mapping. Not human contact dynamics or a full self-collision proof.",
        "pose_sweep_samples": sum(r["kind"] == "pose_sweep" for r in records),
        "recorded_samples": sum(r["kind"] == "recorded" for r in records),
        "minimum_shaft_clearance_m": min(r["minimum_shaft_clearance_m"] for r in records),
        "maximum_skin_corrective_m": max(r["maximum_skin_corrective_m"] for r in records),
        "maximum_requested_corrective_m": max(r["maximum_requested_corrective_m"] for r in records),
        "corrective_limit_m": surface.corrective_limit_m,
        "minimum_torso_proxy_clearance_m": min(r["torso_proxy_clearance_m"] for r in records),
        "all_left_right_order_correct": all(r["left_right_order_correct"] for r in records),
        "checked_hand_triangles_per_pose": records[0]["checked_hand_triangles"],
        "hand_scale": a.hand_scale,
        "minimum_rest_thumb_forward_cosine": min(
            r.get("rest_thumb_forward_cosine", 1.0) for r in records
        ),
        "maximum_rest_thumb_source_rotation_error": max(
            r.get("rest_thumb_source_rotation_error", 0.0) for r in records
        ),
        "minimum_forearm_vertex_bench_clearance_m": min(
            r["forearm_vertex_bench_clearance_m"] for r in records
        ),
        "maximum_rest_wrist_height_m": max(r.get("rest_wrist_height_m", 0) for r in records),
        "maximum_rest_hand_height_m": max(r.get("rest_hand_max_height_m", 0) for r in records),
        "failed_cases": [
            r
            for r in records
            if r["minimum_shaft_clearance_m"] < 0
            or r["maximum_requested_corrective_m"] > 0.006
            or r["torso_proxy_clearance_m"] < 0
            or r["forearm_vertex_bench_clearance_m"] < 0
        ],
    }
    report["passed"] = (
        report["minimum_shaft_clearance_m"] >= 0
        and report["minimum_torso_proxy_clearance_m"] >= 0
        and report["maximum_requested_corrective_m"] <= 0.006
        and report["all_left_right_order_correct"]
        and report["minimum_forearm_vertex_bench_clearance_m"] >= 0
        and report["maximum_rest_wrist_height_m"] < 0.55
        and report["maximum_rest_hand_height_m"] < 0.62
        and report["minimum_rest_thumb_forward_cosine"] > 0.5
        and report["maximum_rest_thumb_source_rotation_error"] < 1e-6
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
