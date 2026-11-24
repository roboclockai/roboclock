"""Fetch the CC0 Lab Assets hotplate and bake its meshes to a portable USD.

Run with numpy, Pillow and usd-core installed. The source model and original
license are retained; runtime never needs a network request or a glTF importer.
"""

import hashlib
import io
import json
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

URL = "https://opengameart.org/sites/default/files/lab_assets.zip"
ARCHIVE_SHA256 = "5202bc6759e19c43f513b848e5cf2e28136f71a0eb9355f50b0c0ac4873dd515"
OUTPUT = Path(__file__).resolve().parents[2] / "src/roboclock/envs/lab/assets/hotplate"


def build(archive):
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Lab Assets archive changed; review provenance before importing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        source = bundle.read("Lab Assets/GLB/Machines/machine_hot_plate.glb")
        (OUTPUT / "LICENSE.txt").write_bytes(bundle.read("Lab Assets/License.txt"))
    (OUTPUT / "machine_hot_plate.glb").write_bytes(source)
    length = struct.unpack_from("<I", source, 12)[0]
    document = json.loads(source[20 : 20 + length])
    binary = source[28 + length :]

    def accessor(index):
        spec = document["accessors"][index]
        view = document["bufferViews"][spec["bufferView"]]
        if "byteStride" in view or "sparse" in spec:
            raise ValueError("Unexpected packed asset layout")
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[spec["type"]]
        dtype = {5123: "<u2", 5125: "<u4", 5126: "<f4"}[spec["componentType"]]
        return (
            np.frombuffer(
                binary,
                dtype=dtype,
                count=spec["count"] * width,
                offset=view.get("byteOffset", 0) + spec.get("byteOffset", 0),
            )
            .reshape(-1, width)
            .copy()
        )

    view = document["bufferViews"][document["images"][0]["bufferView"]]
    (OUTPUT / "atlas.png").write_bytes(
        binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    )
    stage = Usd.Stage.CreateNew(str(OUTPUT / "hotplate.usdc"))
    root = UsdGeom.Xform.Define(stage, "/Hotplate")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    materials = []
    for index, original in enumerate(document["materials"]):
        path = f"/Hotplate/Materials/M{index}"
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.38)
        tex = UsdShade.Shader.Define(stage, path + "/Atlas")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("atlas.png"))
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
        if "emissiveFactor" in original:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb"
            )
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials.append(mat)
    # Author-supplied dimensions are arbitrary units. Fit a 180 mm wide unit,
    # with an 80 mm high top, centered on the original heating surface.
    scale = np.array([0.18 / 39.21305465698242] * 2 + [0.08 / 18.24003505706787])
    all_points = []
    for node_index, node in enumerate(document["nodes"]):
        if any(key in node for key in ("rotation", "scale", "matrix")):
            raise ValueError("Unexpected node transform")
        for part, primitive in enumerate(document["meshes"][node["mesh"]]["primitives"]):
            attrs = primitive["attributes"]
            points = accessor(attrs["POSITION"]) + node.get("translation", [0, 0, 0])
            points = points[:, [0, 2, 1]] * [1, -1, 1]
            points[:, 2] += 3.7448482513427734
            points *= scale
            normals = accessor(attrs["NORMAL"])[:, [0, 2, 1]] * [1, -1, 1] / scale
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            uv = accessor(attrs["TEXCOORD_0"])
            uv[:, 1] = 1 - uv[:, 1]
            indices = accessor(primitive["indices"]).flatten().astype(np.int32)
            mesh = UsdGeom.Mesh.Define(stage, f"/Hotplate/Part_{node_index}_{part}")
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
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[primitive["material"]])
            mesh.GetPrim().SetCustomDataByKey("source_node", node["name"])
            all_points.append(points)
    stage.GetRootLayer().Save()
    points = np.concatenate(all_points)
    (OUTPUT / "provenance.json").write_text(
        json.dumps(
            {
                "title": "Lab Assets / machine_hot_plate",
                "author": "MilkAndBanana",
                "license": "CC0-1.0",
                "source_page": "https://opengameart.org/content/lab-assets",
                "author_page": "https://milkandbanana.itch.io/lab-assets",
                "download_url": URL,
                "archive_sha256": ARCHIVE_SHA256,
                "source_glb_sha256": hashlib.sha256(source).hexdigest(),
                "changes": "Baked glTF Y-up to USD Z-up; width 0.18 m; heating top 0.08 m. Original atlas and meshes retained. Runtime adds collision proxies, probe, and dynamic display.",
                "bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
            },
            indent=2,
        )
        + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    build(
        args.archive.read_bytes()
        if args.archive
        else urllib.request.urlopen(URL, timeout=60).read()
    )
