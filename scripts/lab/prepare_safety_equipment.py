"""Bake CC0 Lab Assets desiccator and waste bottle to USD.

The original GLBs, license, atlas and checked source archive digest are retained.
Geometry is visual-only. safety_scene.py authors separate physical mechanisms.
"""

import argparse
import hashlib
import json
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[2] / "src/roboclock/envs/lab/assets/safety_equipment"
URL = "https://opengameart.org/sites/default/files/lab_assets.zip"
DIGEST = "5202bc6759e19c43f513b848e5cf2e28136f71a0eb9355f50b0c0ac4873dd515"


def build(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected archive digest; review the source before import")
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        (ROOT / "LICENSE.txt").write_bytes(bundle.read("Lab Assets/License.txt"))
        for model in ["machine_desiccator", "bottle_plastic_bottle_large"]:
            raw = bundle.read("Lab Assets/GLB/All/" + model + ".glb")
            (ROOT / (model + ".glb")).write_bytes(raw)
            length = struct.unpack_from("<I", raw, 12)[0]
            doc = json.loads(raw[20 : 20 + length])
            binary = raw[28 + length :]

            def accessor(index):
                a = doc["accessors"][index]
                view = doc["bufferViews"][a["bufferView"]]
                width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
                dtype = {5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
                assert "byteStride" not in view and "sparse" not in a
                return (
                    np.frombuffer(
                        binary,
                        dtype=dtype,
                        count=a["count"] * width,
                        offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
                    )
                    .reshape(-1, width)
                    .copy()
                )

            view = doc["bufferViews"][doc["images"][0]["bufferView"]]
            (ROOT / "atlas.png").write_bytes(
                binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
            )
            for piece in ["desiccator" if model == "machine_desiccator" else "carboy"]:
                stage = Usd.Stage.CreateNew(str(ROOT / (piece + ".usdc")))
                root = UsdGeom.Xform.Define(stage, "/Asset")
                stage.SetDefaultPrim(root.GetPrim())
                UsdGeom.SetStageUpAxis(stage, "Z")
                UsdGeom.SetStageMetersPerUnit(stage, 1.0)
                mats = []
                for i, original in enumerate(doc["materials"]):
                    path = f"/Asset/Materials/M{i}"
                    mat = UsdShade.Material.Define(stage, path)
                    shader = UsdShade.Shader.Define(stage, path + "/Surface")
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.32)
                    tex = UsdShade.Shader.Define(stage, path + "/Atlas")
                    tex.CreateIdAttr("UsdUVTexture")
                    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                        Sdf.AssetPath("atlas.png")
                    )
                    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                    reader = UsdShade.Shader.Define(stage, path + "/UV")
                    reader.CreateIdAttr("UsdPrimvarReader_float2")
                    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                        reader.ConnectableAPI(), "result"
                    )
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                        tex.ConnectableAPI(), "rgb"
                    )
                    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                    mats.append(mat)
                all_points = []
                for i, node in enumerate(doc["nodes"]):
                    assert not any(k in node for k in ["rotation", "scale", "matrix"])
                    for j, primitive in enumerate(doc["meshes"][node["mesh"]]["primitives"]):
                        attrs = primitive["attributes"]
                        points = accessor(attrs["POSITION"]) + node.get("translation", [0, 0, 0])
                        points = points[:, [0, 2, 1]] * [1, -1, 1]
                        if piece == "desiccator":
                            scale = np.array([0.26 / 44.301033] * 3)
                            points[:, 2] += 7.66289186
                        else:
                            scale = np.array([0.24 / 15.0100965] * 2 + [0.40 / 43.6290498])
                            points[:, 2] += 19.8276329
                        points *= scale
                        normals = accessor(attrs["NORMAL"])[:, [0, 2, 1]] * [1, -1, 1] / scale
                        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
                        uv = accessor(attrs["TEXCOORD_0"])
                        uv[:, 1] = 1 - uv[:, 1]
                        indices = accessor(primitive["indices"]).flatten().astype(np.int32)
                        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/Part_{i}_{j}")
                        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
                        mesh.SetNormalsInterpolation("vertex")
                        mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
                        mesh.CreateFaceVertexIndicesAttr(indices.tolist())
                        mesh.CreateSubdivisionSchemeAttr("none")
                        mesh.CreateDoubleSidedAttr(True)
                        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                            "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
                        ).Set(Vt.Vec2fArray.FromNumpy(uv))
                        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                            mats[primitive["material"]]
                        )
                        mesh.GetPrim().SetCustomDataByKey("source_node", node["name"])
                        mesh.GetPrim().SetCustomDataByKey("source_material", primitive["material"])
                        all_points.append(points)
                stage.GetRootLayer().Save()
                pts = np.concatenate(all_points)
                records.append(
                    dict(
                        output=piece + ".usdc",
                        source=model + ".glb",
                        source_sha256=hashlib.sha256(raw).hexdigest(),
                        bounds_m=[pts.min(0).tolist(), pts.max(0).tolist()],
                    )
                )
    (ROOT / "provenance.json").write_text(
        json.dumps(
            dict(
                title="Lab Assets",
                author="MilkAndBanana",
                license="CC0-1.0",
                source_page="https://opengameart.org/content/lab-assets",
                download_url=URL,
                archive_sha256=DIGEST,
                representation="Original meshes and atlas; scaled to meters",
                runtime_material_changes="Desiccator glass uses OmniGlass; original atlas retained elsewhere",
                assets=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/tmp/roboclock-lab-assets.zip"))
    args = parser.parse_args()
    if not args.archive.exists():
        urllib.request.urlretrieve(URL, args.archive)
    build(args.archive)
