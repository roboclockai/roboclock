"""Download and bake Sam Feng's CC-BY-4.0 Lever Power Switch as two USD parts.

The original GLB is mirrored by AI2 Objaverse. Rest-pose skinning is baked;
Isaac Sim supplies the physical revolute joint instead of the source animation.
"""

import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[2] / "src/roboclock/envs/lab/assets/knife_switch"
UID = "374608f75c114b8a9bf41dbc02c51c50"
URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-013/" + UID + ".glb"
SHA256 = "2c76feef64ca9e725b490da70854aa6427ba5890b7fa82e88d6f240bf433d72f"
SCALE = 0.22 / 290.18618
GRIP = 195.0 * SCALE


def build():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "source.glb"
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != SHA256:
        raise ValueError("Source GLB checksum differs; review before conversion")
    length = struct.unpack_from("<I", blob, 12)[0]
    doc = json.loads(blob[20 : 20 + length])
    binary = blob[28 + length :]

    def accessor(index):
        spec = doc["accessors"][index]
        view = doc["bufferViews"][spec["bufferView"]]
        if "sparse" in spec:
            raise ValueError("Unexpected source layout")
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[spec["type"]]
        dtype = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[spec["componentType"]]
        if (
            view.get("byteStride", width * np.dtype(dtype).itemsize)
            != width * np.dtype(dtype).itemsize
        ):
            raise ValueError("Unexpected interleaved attributes")
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

    def transform(node):
        if "matrix" in node:
            return np.array(node["matrix"]).reshape(4, 4).T
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        m = np.eye(4)
        m[:3, :3] = r @ np.diag(node.get("scale", [1] * 3))
        m[:3, 3] = node.get("translation", [0] * 3)
        return m

    world = {}

    def walk(index, parent):
        world[index] = parent @ transform(doc["nodes"][index])
        for child in doc["nodes"][index].get("children", []):
            walk(child, world[index])

    for index in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        walk(index, np.eye(4))
    skin = doc["skins"][0]
    inverse = accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    matrices = np.array([world[j] @ inverse[i] for i, j in enumerate(skin["joints"])])
    for i, image in enumerate(doc["images"]):
        view = doc["bufferViews"][image["bufferView"]]
        suffix = ".png" if image["mimeType"] == "image/png" else ".jpg"
        (ROOT / ("texture_" + str(i) + suffix)).write_bytes(
            binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        )
    for index, part in enumerate(["Handle", "Panel"]):
        stage = Usd.Stage.CreateNew(str(ROOT / (part.lower() + ".usdc")))
        root = UsdGeom.Xform.Define(stage, "/Switch")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        mat = UsdShade.Material.Define(stage, "/Switch/Material")
        shader = UsdShade.Shader.Define(stage, "/Switch/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.65)
        reader = UsdShade.Shader.Define(stage, "/Switch/Material/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        for input_name, texture_index, channel in [
            ("diffuseColor", 0, "rgb"),
            ("roughness", 1, "g"),
            ("metallic", 1, "b"),
        ]:
            tex = UsdShade.Shader.Define(stage, "/Switch/Material/" + input_name)
            tex.CreateIdAttr("UsdUVTexture")
            image_index = doc["textures"][texture_index]["source"]
            image = doc["images"][image_index]
            suffix = ".png" if image["mimeType"] == "image/png" else ".jpg"
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath("texture_" + str(image_index) + suffix)
            )
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                "sRGB" if input_name == "diffuseColor" else "raw"
            )
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result"
            )
            shader.CreateInput(
                input_name,
                Sdf.ValueTypeNames.Color3f if channel == "rgb" else Sdf.ValueTypeNames.Float,
            ).ConnectToSource(tex.ConnectableAPI(), channel)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        primitive = doc["meshes"][index]["primitives"][0]
        attrs = primitive["attributes"]
        source = accessor(attrs["POSITION"])
        joints = accessor(attrs["JOINTS_0"])
        weights = accessor(attrs["WEIGHTS_0"])
        points = np.zeros_like(source)
        normals = np.zeros_like(source)
        for j in range(4):
            m = matrices[joints[:, j]]
            points += (
                np.einsum("nij,nj->ni", m, np.c_[source, np.ones(len(source))])[:, :3]
                * weights[:, j, None]
            )
            normals += (
                np.einsum("nij,nj->ni", m[:, :3, :3], accessor(attrs["NORMAL"]))
                * weights[:, j, None]
            )
        # Lay the panel on a tabletop; hinge at the origin, lever opens about -Y.
        points = (
            points[:, [1, 0, 2]] * [1, -1, 1] - [195 if part == "Handle" else 0, 0, 33.49648]
        ) * SCALE
        normals = normals[:, [1, 0, 2]] * [1, -1, 1]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        uv = accessor(attrs["TEXCOORD_0"])
        uv[:, 1] = 1 - uv[:, 1]
        indices = accessor(primitive["indices"]).ravel().astype(np.int32)
        mesh = UsdGeom.Mesh.Define(stage, "/Switch/Mesh")
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
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        stage.GetRootLayer().Save()
    info = {
        "title": "Lever Power Switch",
        "author": "Sam Feng",
        "license": "CC-BY-4.0",
        "source_page": "https://sketchfab.com/3d-models/lever-power-switch-" + UID,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "download_url": URL,
        "source_glb_sha256": SHA256,
        "grip_radius_m": GRIP,
        "changes": "Baked rest-pose skinning, separated panel/handle, oriented for tabletop and scaled panel length to 0.22 m. Original mesh and base-color/metallic/roughness maps retained. Runtime adds a passive PhysX hinge, collision proxies, labelled terminals and circuit wiring.",
    }
    (ROOT / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    (ROOT / "LICENSE.txt").write_text(
        "Lever Power Switch by Sam Feng\n"
        + info["source_page"]
        + "\nCreative Commons Attribution 4.0 International (CC BY 4.0)\nhttps://creativecommons.org/licenses/by/4.0/\n\n"
        + info["changes"]
        + "\nSource copy distributed by AI2 Objaverse: https://huggingface.co/datasets/allenai/objaverse\n"
    )
    print(ROOT, "grip radius", GRIP)


if __name__ == "__main__":
    build()
