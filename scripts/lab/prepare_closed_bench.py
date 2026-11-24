"""Extract the detailed closed laboratory cabinet island from supplied pack 3."""

import json
import sys
from pathlib import Path

import numpy as np
from import_background_assets import convert
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def main():
    selection = json.loads(Path("scripts/lab/closed_bench_selection.json").read_text())
    target = Path("assets/lab/closed_laboratory_bench.usdc")
    if "--refine-only" not in sys.argv:
        convert(
            Path(selection["source"]),
            target,
            0.001,
            set(selection["source_objects"]),
            [[-1510, -5500, 2800], [1510, -3500, 4000]],
        )
    record = json.loads(target.with_suffix(".json").read_text())
    stage = Usd.Stage.Open(str(target))
    retained = []
    for part in record["parts"]:
        number = int(part["source_object"].removeprefix("qingmo_").removesuffix("P"))
        if 60200 <= number <= 64100 or number in {75081, 75118, 75155}:
            stage.RemovePrim(part["prim"])
        else:
            retained.append(part)
    record["parts"] = retained
    for part in record["parts"]:
        key = part["material"]
        role = None
        if key.endswith(("6465P", "6397P")):
            role, color, roughness, metallic = "graphite_worktop", [0.07, 0.08, 0.085], 0.27, 0.0
        elif key.endswith(("6492P", "6541P")):
            role, color, roughness, metallic = "cabinet_enamel", [0.80, 0.82, 0.82], 0.29, 0.0
        elif key.endswith("6516P"):
            role, color, roughness, metallic = (
                "brushed_steel_handle",
                [0.52, 0.57, 0.61],
                0.23,
                0.92,
            )
        elif key.endswith("6565P"):
            role, color, roughness, metallic = "recessed_plinth", [0.075, 0.08, 0.085], 0.5, 0.0
        if role is None and record["materials"][key]["opacity"] < 0.3:
            role, color, roughness, metallic = "borosilicate_glass", [0.94, 0.98, 1.0], 0.06, 0.0
        if role is None:
            continue
        mesh = stage.GetPrimAtPath(part["prim"])
        mesh.SetCustomDataByKey("surface_role", role)
        mat, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        shader = UsdShade.Shader(stage.GetPrimAtPath(str(mat.GetPath()) + "/Shader"))
        shader.GetInput("diffuseColor").DisconnectSource()
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        if role in {"graphite_worktop", "cabinet_enamel", "recessed_plinth"}:
            geometry = UsdGeom.Mesh(mesh)
            points = np.asarray(geometry.GetPointsAttr().Get())
            ids = geometry.GetFaceVertexIndicesAttr().Get()
            normals = np.asarray(geometry.GetNormalsAttr().Get()).copy()
            offset, corrected = 0, 0
            for count in geometry.GetFaceVertexCountsAttr().Get():
                face = points[ids[offset : offset + count]]
                normal = np.cross(face[1] - face[0], face[2] - face[0])
                area = np.linalg.norm(normal)
                if area > 0.0001:
                    normal /= area
                    if np.max(np.abs(normal)) > 0.9999:
                        normals[offset : offset + count] = normal
                        corrected += 1
                offset += count
            geometry.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            part["planar_normal_corrections"] = corrected
        part["surface_role"] = role
    stage.GetRootLayer().Save()
    record["material_conversion"] = (
        "Original geometry/UVs; repaired planar normals; runtime MDL surface bindings for enamel, graphite resin and steel handles."
    )
    target.with_suffix(".json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
