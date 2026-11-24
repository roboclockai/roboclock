"""Extract the supplied pack 5 computer, upholstered chair and marble slab.

Keep original mesh topology, normals and UVs. Replace legacy V-Ray material
constants with explicit renderer-compatible surfaces; never interpret a legacy
reflection map as a roughness map.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from import_background_assets import convert
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

SOURCE = Path("assets/5/5/5.obj")
OUTPUT = Path("assets/lab")
SELECTIONS = {
    "office_computer": [
        "qingmo_010bd9fc_f603_47f5_b3ce_8a98c2256cfb",
        "qingmo_c057b0ec_d18e_46d9_8fc4_e5008e4461d8",
        "qingmo_012a30bb_0c47_4f20_bc4b_8a3ffa163ab4",
        "qingmo_f30ab84a_db97_4838_a919_be86544b2aa6",
        "qingmo_62db0ba2_62e6_4f35_809f_84a6d2aa7c90",
        "qingmo_44700bf0_0943_4c68_9d97_bea1739c0e68",
        "qingmo_ac84957a_4f76_4024_95d9_3a60b368c38c",
    ],
    "office_chair": [
        "qingmo_5f027e74_1260_45b7_811a_55938b555c36",
        "qingmo_19f225ca_f561_4d9e_9f4e_0372e52e69f8",
        "qingmo_655a788f_3bcd_44e7_975f_df463896437f",
    ],
    "marble_slab": ["Good_027"],
}
SELECTIONS["laboratory_bench"] = json.loads(
    Path("scripts/lab/lab_bench_selection.json").read_text()
)["source_objects"]


def upholstery_maps():
    """Convert the supplied height texture to a subtle tangent-space grain normal."""
    source = SOURCE.parent / "maps/Lime._qingmo_ccc4ada7-05bb-42e2-ba55-0dc8079f5f62.jpg"
    folder = OUTPUT / "upholstery"
    folder.mkdir(exist_ok=True)
    height = np.asarray(Image.open(source).convert("L"), dtype=float) / 255.0
    dy, dx = np.gradient(height)
    normal = np.stack([-dx * 1.8, dy * 1.8, np.ones_like(height)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    target = folder / "grain_normal.png"
    Image.fromarray(np.uint8(np.clip(normal * 0.5 + 0.5, 0, 1) * 255)).save(target)
    roughness = np.uint8(np.clip(0.43 + (height - height.mean()) * 0.16, 0, 1) * 255)
    Image.fromarray(roughness).save(folder / "grain_roughness.png")
    (folder / "source.json").write_text(
        json.dumps(
            {
                "height_source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "normal": "Derived OpenGL tangent-space normal, raw; retained source resolution.",
                "roughness": "Authored subtle variation from source height, not a measured roughness scan.",
            },
            indent=2,
        )
    )


def refine(name):
    path = OUTPUT / f"{name}.usdc"
    record = json.loads(path.with_suffix(".json").read_text())
    stage = Usd.Stage.Open(str(path))
    for part in record["parts"]:
        mesh = stage.GetPrimAtPath(part["prim"])
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        shader = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/Shader"))
        role, roughness, metallic = "painted polymer", 0.32, 0.0
        key = part["material"]
        if name == "laboratory_bench":
            if "ī" in key:
                role, roughness = "graphite laboratory worktop", 0.28
                shader.GetInput("diffuseColor").DisconnectSource()
                shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.045, 0.052, 0.055))
            elif any(k in key for k in ["Good_2695", "Good_2717", "Good_2860"]):
                role, roughness, metallic = "satin steel hardware", 0.23, 0.88
            elif any(k in key for k in ["Good_2718", "Good_2816", "Good_2857"]):
                role, roughness = "satin white cabinetry", 0.32
        elif name == "marble_slab":
            role, roughness = "honed marble", 0.24
        elif name == "office_chair":
            if any(k in key for k in ["bc95a740", "3c89decf"]):
                role, roughness = "upholstery", 0.43
            elif any(k in key for k in ["dc2160b8", "f308339a"]):
                role, roughness, metallic = "satin chrome frame", 0.20, 0.92
            else:
                role, roughness = "caster polymer", 0.58
        elif name == "office_computer":
            if "ֻ" in key or "954659bf" in key:
                role, roughness = "reflective display glass", 0.085
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.012))
                shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
                shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.055)
            elif any(k in key for k in ["c687d97d", "828f8011"]):
                role, roughness, metallic = "satin aluminum", 0.25, 0.88
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
        if role == "upholstery":
            uv = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/GrainUV")
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            for label, filename, output, value_type in [
                ("Normal", "grain_normal.png", "rgb", Sdf.ValueTypeNames.Normal3f),
                ("Roughness", "grain_roughness.png", "r", Sdf.ValueTypeNames.Float),
            ]:
                tex = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Grain" + label)
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(str((OUTPUT / "upholstery" / filename).resolve()))
                )
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    uv.ConnectableAPI(), "result"
                )
                if label == "Normal":
                    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
                    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
                shader.CreateInput(label.lower(), value_type).ConnectToSource(
                    tex.ConnectableAPI(), output
                )
        if name == "marble_slab":
            directory = OUTPUT / "marble8k"
            if not (directory / "Marble012_8K-JPG_Color.jpg").exists():
                raise FileNotFoundError("Download the documented Marble012 8K maps first")
            slab = UsdGeom.Mesh(mesh)
            points = np.asarray(slab.GetPointsAttr().Get())
            counts = slab.GetFaceVertexCountsAttr().Get()
            ids = slab.GetFaceVertexIndicesAttr().Get()
            normals, offset = [], 0
            for count in counts:
                face = points[ids[offset : offset + count]]
                n = np.cross(face[1] - face[0], face[2] - face[0])
                n /= np.linalg.norm(n)
                normals.extend([n] * count)
                offset += count
            slab.GetNormalsAttr().Set(
                Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32))
            )
            part["normal_override"] = (
                "Flat face normals for the planar stone slab; source mesh topology unchanged"
            )
            reader = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/UV"))
            for label, filename, channel, value_type in [
                ("diffuseColor", "Color", "rgb", Sdf.ValueTypeNames.Color3f),
                ("roughness", "Roughness", "r", Sdf.ValueTypeNames.Float),
                ("normal", "NormalGL", "rgb", Sdf.ValueTypeNames.Normal3f),
            ]:
                tex = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Marble" + filename)
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(
                        str((directory / ("Marble012_8K-JPG_" + filename + ".jpg")).resolve())
                    )
                )
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                    "sRGB" if label == "diffuseColor" else "raw"
                )
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    reader.ConnectableAPI(), "result"
                )
                if label == "normal":
                    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
                    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
                shader.CreateInput(label, value_type).ConnectToSource(tex.ConnectableAPI(), channel)
            part["pbr_override"] = (
                "ambientCG Marble012 8K Color/Roughness/NormalGL; source UV retained"
            )
        mesh.SetCustomDataByKey("surface_role", role)
        part["surface_role"] = role
        part["roughness"] = roughness
        part["metallic"] = metallic
    stage.GetRootLayer().Save()
    record["material_conversion"] = (
        "Original diffuse maps/UVs; explicit role-specific PBR constants. Legacy reflection maps are not roughness."
    )
    path.with_suffix(".json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    upholstery_maps()
    for name, selection in SELECTIONS.items():
        convert(SOURCE, OUTPUT / f"{name}.usdc", 0.001, set(selection))
        refine(name)
    Path("scripts/lab/office_selection.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE),
                "selections": SELECTIONS,
                "description": "Pack 5 original mesh selections, separate normalized roots.",
            },
            indent=2,
        )
    )
