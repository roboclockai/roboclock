"""Extract the supplied wet bench and open its overlapping sink cover faces."""

import json
import sys
from pathlib import Path

import numpy as np
from import_background_assets import convert
from pxr import Usd, UsdGeom, Vt

TOP = "qingmo_6be0ef9a_20b1_491b_b27f_66d0e7d83bd7"
CABINET = "qingmo_76fe2773_53ae_4a77_ab6f_e4f2cb5f9ea1"


def half_plane(poly, axis, threshold, direction):
    """Clip position/normal/UV records, interpolating at the cutting edge."""
    result = []
    for first, second in zip(poly, np.roll(poly, -1, axis=0)):
        d1 = direction * (first[axis] - threshold)
        d2 = direction * (second[axis] - threshold)
        if d1 >= -1e-9:
            result.append(first)
        if (d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9):
            result.append(first + (second - first) * d1 / (d1 - d2))
    return np.array(result)


def refine(target):
    record = json.loads(target.with_suffix(".json").read_text())
    stage = Usd.Stage.Open(str(target))
    rim = next(
        p for p in record["parts"] if p["source_object"] == TOP and "9804af0e" in p["material"]
    )
    low, high = np.array(rim["min"]), np.array(rim["max"])
    low[:2] += 0.006
    high[:2] -= 0.006
    for part in record["parts"]:
        if part["source_object"] not in {TOP, CABINET}:
            continue
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(part["prim"]))
        points = np.asarray(mesh.GetPointsAttr().Get())
        ids = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        ns = np.asarray(mesh.GetNormalsAttr().Get())
        uv_prim = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
        uv = np.asarray(uv_prim.Get()) if uv_prim else np.zeros((len(ids), 2))
        output, offset, cut, corrected = [], 0, 0, 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = np.hstack(
                [
                    points[ids[offset : offset + count]],
                    ns[offset : offset + count],
                    uv[offset : offset + count],
                ]
            )
            offset += count
            for k in range(1, count - 1):
                triangle = face[[0, k, k + 1]].copy()
                normal = np.cross(
                    triangle[1, :3] - triangle[0, :3], triangle[2, :3] - triangle[0, :3]
                )
                area = np.linalg.norm(normal)
                if area < 1e-12:
                    continue
                normal /= area
                if area > 0.0001 and max(abs(normal)) > 0.9999:
                    triangle[:, 3:6] = normal
                    corrected += 1
                if abs(normal[2]) > 0.99 and triangle[:, 2].min() >= low[2] - 0.002:
                    inside = triangle
                    for axis, threshold, direction in [
                        (0, low[0], 1),
                        (0, high[0], -1),
                        (1, low[1], 1),
                        (1, high[1], -1),
                    ]:
                        if len(inside) < 3:
                            break
                        outside = half_plane(inside, axis, threshold, -direction)
                        if len(outside) >= 3:
                            output.append(outside)
                        inside = half_plane(inside, axis, threshold, direction)
                    if len(inside) >= 3:
                        cut += 1
                else:
                    output.append(triangle)
        combined = np.concatenate(output).astype(np.float32)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(combined[:, :3]))
        mesh.GetFaceVertexCountsAttr().Set([len(poly) for poly in output])
        mesh.GetFaceVertexIndicesAttr().Set(list(range(len(combined))))
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(combined[:, 3:6]))
        if uv_prim:
            uv_prim.Set(Vt.Vec2fArray.FromNumpy(combined[:, 6:8]))
        part.update(
            faces=len(output), opening_cut_triangles=cut, planar_normal_corrections=corrected
        )
    stage.GetRootLayer().Save()
    record["geometry_repairs"] = (
        "Clipped overlapping horizontal sink covers to the original rim opening; interpolated UVs; repaired large planar normals."
    )
    target.with_suffix(".json").write_text(json.dumps(record, indent=2))


def main():
    selection = json.loads(Path("scripts/lab/wet_bench_selection.json").read_text())
    target = Path("assets/lab/wet_bench.usdc")
    if "--refine-only" not in sys.argv:
        convert(
            Path(selection["source"]),
            target,
            0.001,
            set(selection["source_objects"]),
            [[5800, 4950, -40], [9600, 6200, 1800]],
        )
    refine(target)


if __name__ == "__main__":
    main()
