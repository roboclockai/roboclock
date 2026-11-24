"""Extract the supplied pack-3 rack and tube from the existing imported bench."""

import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def main():
    directory = Path("assets/lab")
    source = directory / "closed_laboratory_bench.usdc"
    record = json.loads(source.with_suffix(".json").read_text())
    stage = Usd.Stage.Open(str(source))
    rack_ids = {78866, 78954, 78998, 79042, 79130}
    tube_ids = {77898, 77942, 77986, 78030, 78074, 78118, 78162, 78690, 78734, 78778, 78822, 78910}
    parts = {int(p["source_object"][7:-1]): p for p in record["parts"]}
    selected = [parts[i] for i in sorted(rack_ids)]
    low = np.min([p["min"] for p in selected], axis=0)
    high = np.max([p["max"] for p in selected], axis=0)
    center = (low + high) / 2
    center[2] = low[2]
    scale = np.array([1.2, 1.25, 0.5])

    # Rotate the long axis onto Y to fit up to five 12-hole racks on the bench.
    def transform(points):
        p = (points - center) * scale
        return np.stack([-p[..., 1], p[..., 0], p[..., 2]], axis=-1)

    for name, ids in [("test_tube_rack", rack_ids), ("organizing_test_tube", {77898})]:
        out = Usd.Stage.CreateNew(str(directory / (name + ".usdc")))
        root = UsdGeom.Xform.Define(out, "/Asset")
        out.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageUpAxis(out, "Z")
        UsdGeom.SetStageMetersPerUnit(out, 1)
        for i in sorted(ids):
            p = parts[i]
            path = "/Asset/Part_" + str(i)
            Sdf.CopySpec(stage.GetRootLayer(), p["prim"], out.GetRootLayer(), path)
            mesh = UsdGeom.Mesh(out.GetPrimAtPath(path))
            points = np.asarray(mesh.GetPointsAttr().Get())
            if name == "test_tube_rack":
                points = transform(points)
            else:
                lo, hi = points.min(0), points.max(0)
                points = (points - (lo + hi) / 2) * (np.array([0.028, 0.028, 0.14]) / (hi - lo))
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            # Let the renderer compute normals after nonuniform scale.
            mesh.GetPrim().RemoveProperty("normals")
            mesh.GetPrim().RemoveProperty("material:binding")
            mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
        out.GetRootLayer().Save()
    slots = []
    for i in sorted(tube_ids):
        p = parts[i]
        slots.append(transform((np.array(p["min"]) + p["max"]) / 2)[:2].tolist())
    slots.sort(key=lambda p: (round(p[1], 2), p[0]))
    result = {
        "source": record["source"],
        "source_sha256": record["sha256"],
        "derived_from": str(source),
        "derived_from_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rack_source_objects": [parts[i]["source_object"] for i in sorted(rack_ids)],
        "tube_source_object": parts[77898]["source_object"],
        "scale_before_z_rotation": scale.tolist(),
        "rotation_z_degrees": 90,
        "slots_xy": slots,
        "support_z": float((parts[78998]["max"][2] - low[2]) * 0.5),
        "rim_z": float((high[2] - low[2]) * 0.5),
        "license": "user-supplied; redistribution rights not established",
        "collision": "Static rack triangle meshes; per-slot floor contacts; conservative dynamic tube cylinder",
    }
    (directory / "test_tube_rack.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
