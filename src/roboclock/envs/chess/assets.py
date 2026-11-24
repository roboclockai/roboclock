"""Import the bundled, uncompressed CC0 GLBs with normalized accessor support.

Only the subset used by these inspected files is supported; unsupported features
raise rather than silently producing incorrectly scaled physics geometry.
"""

import json
import struct
from pathlib import Path

import numpy as np

from roboclock.physics.geometry import rotation_matrix

ROOT = Path(__file__).parent / "assets"


def read_glb(path):
    raw = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", raw)
    if magic != 0x46546C67 or version != 2 or length != len(raw):
        raise ValueError("Expected binary glTF 2")
    n, kind = struct.unpack_from("<II", raw, 12)
    doc = json.loads(raw[20 : 20 + n])
    size, kind = struct.unpack_from("<II", raw, 20 + n)
    binary = raw[28 + n : 28 + n + size]

    def accessor(index):
        a = doc["accessors"][index]
        if "sparse" in a:
            raise ValueError("Sparse accessor unsupported")
        view = doc["bufferViews"][a["bufferView"]]
        dtype = np.dtype(
            {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[
                a["componentType"]
            ]
        )
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        array = np.ndarray(
            (a["count"], width),
            dtype,
            binary,
            offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
            strides=(view.get("byteStride", width * dtype.itemsize), dtype.itemsize),
        ).copy()
        if a.get("normalized"):
            array = array.astype(float) / np.iinfo(dtype).max
            array = np.maximum(array, -1)
        return array

    result = []

    def visit(index, parent):
        node = doc["nodes"][index]
        local = np.eye(4)
        if "matrix" in node:
            local = np.array(node["matrix"]).reshape(4, 4).T
        else:
            q = node.get("rotation", [0, 0, 0, 1])
            local[:3, :3] = rotation_matrix([q[3], *q[:3]]) @ np.diag(node.get("scale", [1, 1, 1]))
            local[:3, 3] = node.get("translation", [0, 0, 0])
        transform = parent @ local
        if "mesh" in node:
            for primitive in doc["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Expected triangle mesh")
                v = accessor(primitive["attributes"]["POSITION"])
                v = v @ transform[:3, :3].T + transform[:3, 3]
                # Y-up glTF -> Z-up USD, right handed.
                v = v[:, [0, 2, 1]] * [1, -1, 1]
                f = (
                    accessor(primitive["indices"]).reshape(-1, 3)
                    if "indices" in primitive
                    else np.arange(len(v)).reshape(-1, 3)
                )
                material = doc["materials"][primitive["material"]]["pbrMetallicRoughness"]
                result.append((node["name"], v, f, material))
        for child in node.get("children", []):
            visit(child, transform)

    for root in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return result


def add_model(stage, path, filename, position, scale=1.0, skip_buttons=False):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    scaling = [scale] * 3 if np.isscalar(scale) else scale
    root.AddScaleOp().Set(Gf.Vec3f(*map(float, scaling)))
    for i, (name, vertices, faces, material) in enumerate(read_glb(ROOT / "models" / filename)):
        if skip_buttons and "button" in name:
            continue
        mesh = UsdGeom.Mesh.Define(stage, f"{path}/Mesh{i}")
        mesh.CreatePointsAttr(vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mat = UsdShade.Material.Define(stage, f"{path}/Material{i}")
        shader = UsdShade.Shader.Define(stage, f"{path}/Material{i}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*material["baseColorFactor"][:3])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            material.get("roughnessFactor", 0.5)
        )
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            material.get("metallicFactor", 0)
        )
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return root
