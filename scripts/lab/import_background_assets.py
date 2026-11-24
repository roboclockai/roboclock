"""Convert supplied Y-up Wavefront laboratory packs to local Z-up USD backgrounds.

Run with a Python containing usd-core, or Isaac Sim's USD Python environment.
These are visual background references, never task collision or fluid geometry.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def convert(
    source,
    output,
    scale,
    object_names=None,
    bounds_zup_source_units=None,
    origin_zup_source_units=None,
    source_encoding="utf-8",
):
    vertices, uvs, normals, groups = [], [], [], {}
    name, material = "mesh", "default"
    with source.open(encoding=source_encoding, errors="replace") as stream:
        for line in stream:
            words = line.split()
            if not words:
                continue
            key, values = words[0], words[1:]
            if key == "v":
                x, y, z = map(float, values[:3])
                vertices.append((x, -z, y))
            elif key == "vt":
                uvs.append(tuple(map(float, values[:2])))
            elif key == "vn":
                x, y, z = map(float, values[:3])
                normals.append((x, -z, y))
            elif key == "o":
                name = "_".join(values)
            elif key == "usemtl":
                material = values[0]
            elif key == "f" and (object_names is None or name in object_names):
                group = groups.setdefault((name, material), [])
                group.append([tuple(int(x) if x else 0 for x in v.split("/")) for v in values])
    points = np.asarray(vertices, dtype=np.float32)
    if bounds_zup_source_units is not None:
        region_low, region_high = map(np.asarray, bounds_zup_source_units)
        retained = {}
        for key, faces in groups.items():
            ids = sorted({v[0] - 1 for face in faces for v in face})
            sample = points[ids]
            if np.all(sample.min(0) >= region_low) and np.all(sample.max(0) <= region_high):
                retained[key] = faces
        groups = retained
    # Center the background; source model units are documented explicitly below.
    used_ids = sorted({v[0] - 1 for faces in groups.values() for face in faces for v in face})
    used_points = points[used_ids]
    low, high = used_points.min(0), used_points.max(0)
    center = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
    if origin_zup_source_units is not None:
        center = np.asarray(origin_zup_source_units, dtype=float)
    points = (points - center) * scale
    materials = {}
    current = None
    for line in source.with_suffix(".mtl").open(errors="replace"):
        words = line.strip().split()
        if not words:
            continue
        if words[0] == "newmtl":
            current = materials.setdefault(words[1], {})
        elif current is not None:
            current[words[0]] = words[1:]
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Lab")
    stage.SetDefaultPrim(root.GetPrim())
    bound = {}
    material_records = {}
    used_materials = {mat for _, mat in groups}
    for index, (mat, data) in enumerate(materials.items()):
        if mat not in used_materials:
            continue
        p = f"/Lab/Materials/M{index:03d}"
        m = UsdShade.Material.Define(stage, p)
        shader = UsdShade.Shader.Define(stage, p + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        color = list(map(float, data.get("Kd", ["0.65"] * 3)))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        metal = any(t in mat.lower() for t in ("metal", "steel", "chrome"))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.75 if metal else 0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28 if metal else 0.48)
        opacity = float(data.get("d", ["1"])[0])
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(max(0.2, opacity))
        texture = None
        if "map_Kd" in data:
            filename = " ".join(data["map_Kd"]).replace("\\", "/").split("/")[-1]
            candidates = [
                p
                for p in (source.parent / "../textures" / filename, source.parent / filename)
                if p.is_file()
            ]
            candidates += list((source.parent / "maps").rglob(filename))
            if candidates:
                # Copy the map next to the output so the USD stays portable across machines.
                texture = Path("textures") / output.stem / candidates[0].name
                (output.parent / texture).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidates[0], output.parent / texture)
                tex = UsdShade.Shader.Define(stage, p + "/Texture")
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(f"./{texture.as_posix()}")
                )
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                reader = UsdShade.Shader.Define(stage, p + "/UV")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    reader.ConnectableAPI(), "result"
                )
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    tex.ConnectableAPI(), "rgb"
                )
        m.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        bound[mat] = m
        material_records[mat] = {"texture": str(texture) if texture else None, "opacity": opacity}
    inventory = []
    for index, ((name, mat), faces) in enumerate(groups.items()):
        mesh = UsdGeom.Mesh.Define(stage, f"/Lab/Geometry/Part_{index:04d}")
        # Remap position indices; preserve face-varying UVs and authored normals.
        ids = sorted({v[0] - 1 for f in faces for v in f})
        remap = {old: i for i, old in enumerate(ids)}
        local = points[ids].astype(np.float32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(local))
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([remap[v[0] - 1] for f in faces for v in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        if uvs and all(len(v) > 1 and v[1] > 0 for f in faces for v in f):
            uv = np.asarray([uvs[v[1] - 1] for f in faces for v in f], dtype=np.float32)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            ).Set(Vt.Vec2fArray.FromNumpy(uv))
        if normals and all(len(v) > 2 and v[2] > 0 for f in faces for v in f):
            ns = np.asarray([normals[v[2] - 1] for f in faces for v in f], dtype=np.float32)
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(ns))
            mesh.SetNormalsInterpolation("faceVarying")
        if mat in bound:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound[mat])
        mesh.GetPrim().SetCustomDataByKey("source_object", name)
        inventory.append(
            {
                "prim": str(mesh.GetPath()),
                "source_object": name,
                "material": mat,
                "faces": len(faces),
                "min": local.min(0).tolist(),
                "max": local.max(0).tolist(),
            }
        )
    stage.GetRootLayer().Save()
    record = {
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_up": "Y",
        "source_encoding": source_encoding,
        "bounds_filter_zup_source_units": bounds_zup_source_units,
        "scale_to_meters": scale,
        "center_offset": center.tolist(),
        "bounds_m": [((low - center) * scale).tolist(), ((high - center) * scale).tolist()],
        "selected_source_objects": sorted(object_names) if object_names else None,
        "representation": "visual-only imported mesh; no colliders",
        "license": "user-supplied; redistribution rights not established",
        "materials": material_records,
        "parts": inventory,
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(output, len(inventory), record["bounds_m"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("assets/lab"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--scale", type=float, default=0.001)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    selection = set(json.loads(a.selection.read_text())["source_objects"]) if a.selection else None
    convert(a.source, a.output / (a.name + ".usdc"), a.scale, selection)
