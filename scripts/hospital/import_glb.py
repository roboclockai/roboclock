"""Convert downloaded Sketchfab GLB models into simplified, real-world-scale USD assets for the hospital scenes.

Each model is rotated from glTF's Y-up to Z-up, scaled so that its stated dimension matches reality, set down on z = 0 and
centered on x = y = 0, simplified when it is heavy, and written with its textures (resized to <= 1024 px).
A three-view preview PNG is written next to the report so the result can be checked without the simulator.

Usage: python scripts/hospital/import_glb.py SOURCE_DIR OUTPUT_DIR [name ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

MAX_TEXTURE = 1024

# name: (file, axis, real size in meters along that axis, triangle budget, license, author, source)
# axis is the glTF Y-up axis the size refers to: "h" height, "x" width, "z" depth, "max" the longest side.
ASSETS = {
    "autoclave": (
        "autoclave.glb",
        "max",
        0.60,
        40000,
        "CC-BY-4.0",
        "RALQ",
        "autoclave-9f82b48a2e664d0393dbae90034957dc",
    ),
    "blood_bag": (
        "blood_bag_v2.glb",
        "max",
        0.30,
        12000,
        "CC-BY-4.0",
        "BarnusModels",
        "blood-bag-v2-2b5b87480e8e4eb8a94a55e73d0b05d7",
    ),
    "blood_bank_fridge": (
        "blood_bank_refrigerator.glb",
        "h",
        0.85,
        12000,
        "CC-BY-NC-4.0",
        "Kirk Nicholson",
        "blood-bank-refrigerator-7d2f1f6e62424b2aa61ad427f0ca4564",
    ),
    "training_dummy": (
        "body_dummy_prepped_for_surgical_training.glb",
        "max",
        1.70,
        33000,
        "CC-BY-NC-4.0",
        "Digital Surgery",
        "body-dummy-prepped-for-surgical-training-d9bc38002292411d8437447f331dcf2f",
    ),
    "pill": (
        "cc0_-_pill.glb",
        "max",
        0.016,
        3000,
        "CC-BY-4.0",
        "plaggy",
        "cc0-pill-4892b8f61b0c4b27a65b0dfd489d6af5",
    ),
    "crash_cart": (
        "emergency_cart.glb",
        "h",
        1.00,
        50000,
        "CC-BY-4.0",
        "tutminchai",
        "emergency-cart-7b6c0aa0f213416bbd65efbaa2d26391",
    ),
    "vial": (
        "empty_vial.glb",
        "h",
        0.065,
        4000,
        "CC-BY-4.0",
        "Feras Sultan",
        "empty-vial-2d396314c57743158fc33f8ec0dbd465",
    ),
    "fire_blanket": (
        "firefighter_vision_fire_blanket.glb",
        "h",
        0.30,
        50000,
        "CC-BY-NC-4.0",
        "Firefighter Vision",
        "firefighter-vision-fire-blanket-99ad92babf7b40489aaa7a33f2821482",
    ),
    "hospital_bed": (
        "hospital_bed.glb",
        "max",
        2.20,
        30000,
        "CC-BY-4.0",
        "Ansh_Singla",
        "hospital-bed-0e974ea6eb9a4c069c56f152ae5162a4",
    ),
    "iv_stand": (
        "iv_stand.glb",
        "h",
        1.80,
        6000,
        "CC-BY-4.0",
        "teeart",
        "iv-stand-dd3115aafb0e4a628f261d0de7a787b9",
    ),
    "medical_cart": (
        "medical_cart.glb",
        "max",
        0.90,
        18000,
        "CC-BY-4.0",
        "NicolasVB",
        "medical-cart-2a0a574f9e374702af8c415fce296f45",
    ),
    "supplies": (
        "medical_supplies_collection.glb",
        "max",
        0.80,
        90000,
        "CC-BY-4.0",
        "Miguel Adao",
        "medical-supplies-collection-eae9fc470e674c458791cf00b892340f",
    ),
    "syringe_a": (
        "medical_syringe.glb",
        "max",
        0.15,
        7000,
        "CC-BY-4.0",
        "Alexander Troianovskyi",
        "medical-syringe-22e711221a094a48a6ffe826d3fd6185",
    ),
    "wall_cupboard": (
        "medicine_wall_cupboard_hospital.glb",
        "h",
        0.66,
        1000,
        "CC-BY-4.0",
        "BillyJJackson",
        "medicine-wall-cupboard-hospital-3251d73995f54398b118d1aa0bb34489",
    ),
    "instrument_table": (
        "surgical__instrument_table_collection.glb",
        "max",
        1.40,
        36000,
        "CC-BY-NC-4.0",
        "Digital Surgery",
        "surgical-instrument-table-collection-a1fcfeab1ad646638089655e8b6f0e2b",
    ),
    "instruments": (
        "surgical_instruments.glb",
        "max",
        0.45,
        60000,
        "CC-BY-4.0",
        "Wenschel",
        "surgical-instruments-bd0ad0180f5148f1b3b331b8ebe6e494",
    ),
    "syringe_b": (
        "syringe.glb",
        "max",
        0.15,
        2000,
        "CC-BY-4.0",
        "stfuaahil",
        "syringe-324b3ed0026940808db353c8c6d7450e",
    ),
    "aed": (
        "uk_automated_external_defibrillator.glb",
        "max",
        0.30,
        4000,
        "CC-BY-4.0",
        "Warcub",
        "uk-automated-external-defibrillator-997cdb42217f4ef78b65e18e7e506e78",
    ),
}


LAYING = {"syringe_b"}  # lie along x when placed


def decimate(mesh, budget):
    """Quadric simplification that keeps UVs by nearest-vertex transfer."""
    faces = len(mesh.faces)
    if faces <= budget:
        return mesh
    import fast_simplification
    from scipy.spatial import cKDTree

    points, tri = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_reduction=1.0 - budget / faces,
    )
    out = trimesh.Trimesh(points, tri, process=False)
    uv = getattr(mesh.visual, "uv", None)
    if uv is not None:
        _, nearest = cKDTree(mesh.vertices).query(points)
        out.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(uv)[nearest], material=mesh.visual.material
        )
    elif mesh.visual.kind == "vertex":
        _, nearest = cKDTree(mesh.vertices).query(points)
        out.visual = trimesh.visual.ColorVisuals(vertex_colors=mesh.visual.vertex_colors[nearest])
    else:
        out.visual = mesh.visual
    return out


def load(path, axis, size, budget, lay=False):
    scene = trimesh.load(str(path), force="scene")
    meshes = [m for m in scene.dump(concatenate=False) if isinstance(m, trimesh.Trimesh)]
    allv = np.vstack([m.vertices for m in meshes])
    extent = allv.max(0) - allv.min(0)
    reference = {"h": extent[1], "x": extent[0], "z": extent[2], "max": extent.max()}[axis]
    scale = size / reference
    total = sum(len(m.faces) for m in meshes)
    out = []
    for m in meshes:
        m = decimate(m, max(200, int(budget * len(m.faces) / total))) if total > budget else m
        out.append(m)
    # Y-up -> Z-up: (x, y, z) -> (x, -z, y), then scale, center on xy and rest on z = 0.
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    if lay:  # an upright model (syringe) is laid down with its length along x
        rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float) @ rot
    lo = (allv @ rot.T * scale).min(0)
    hi = (allv @ rot.T * scale).max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    for m in out:
        m.vertices = (np.asarray(m.vertices) @ rot.T) * scale + shift
    return out


def texture_path(image, folder, key):
    image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
    if max(image.size) > MAX_TEXTURE:
        image.thumbnail((MAX_TEXTURE, MAX_TEXTURE))
    name = f"{key}.png" if image.mode == "RGBA" else f"{key}.jpg"
    if image.mode == "RGBA":
        image.save(folder / name)
    else:
        image.save(folder / name, quality=88)
    return name


def write_usd(name, meshes, folder, info):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.usd"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Model")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("credit", info)
    cache = {}
    for i, mesh in enumerate(meshes):
        prim = UsdGeom.Mesh.Define(stage, f"/Model/part_{i:03d}")
        prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
        prim.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32))
        )
        prim.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).ravel())
        )
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        prim.CreateSubdivisionSchemeAttr("none")
        material = getattr(mesh.visual, "material", None)
        key = id(material)
        if key not in cache:
            cache[key] = author_material(
                stage, f"/Model/Looks/m{len(cache):02d}", material, folder, f"tex{len(cache):02d}"
            )
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(cache[key])
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None:
            st = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            # trimesh already turns glTF's top-left texture origin into USD's bottom-left one: no second flip.
            st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
        elif mesh.visual.kind == "vertex":
            colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
            ).Set(Vt.Vec3fArray.FromNumpy(colors))
    stage.GetRootLayer().Save()
    return path


def author_material(stage, path, material, folder, key):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    color = (0.7, 0.7, 0.7)
    rough, metal = 0.5, 0.0
    texture = None
    if material is not None:
        factor = getattr(material, "baseColorFactor", None)
        if factor is not None:
            color = tuple(float(c) / 255.0 for c in factor[:3])
        texture = getattr(material, "baseColorTexture", None)
        rough = (
            float(material.roughnessFactor)
            if getattr(material, "roughnessFactor", None) is not None
            else rough
        )
        metal = (
            float(material.metallicFactor)
            if getattr(material, "metallicFactor", None) is not None
            else metal
        )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if texture is not None:
        name = texture_path(texture, folder, key)
        reader = UsdShade.Shader.Define(stage, path + "/St")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{name}")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def preview(meshes, path, size=420):
    """Three flat-shaded orthographic views (front, side, top), nearest triangles drawn last."""
    allv = np.vstack([m.vertices for m in meshes])
    lo, hi = allv.min(0), allv.max(0)
    span = float((hi - lo).max())
    views = {"front": (0, 2, 1), "side": (1, 2, 0), "top": (0, 1, 2)}
    canvas = Image.new("RGB", (size * 3, size), (235, 235, 235))
    for k, (_, (u, v, depth)) in enumerate(views.items()):
        image = Image.new("RGB", (size, size), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        tri = []
        for m in meshes:
            base = np.array([0.6, 0.62, 0.66])
            color = getattr(m.visual, "main_color", None)
            if color is not None:
                base = np.asarray(color[:3], dtype=float) / 255.0
            p = m.vertices[m.faces]
            normal = m.face_normals
            tri.extend(
                zip(p[:, :, [u, v]], p[:, :, depth].mean(1), normal[:, depth], [base] * len(p))
            )
        tri.sort(key=lambda t: t[1])
        for xy, _, facing, base in tri[:: max(1, len(tri) // 25000)]:
            shade = 0.45 + 0.55 * abs(facing)
            x = (xy[:, 0] - lo[u]) / span * (size - 20) + 10
            y = size - ((xy[:, 1] - lo[v]) / span * (size - 20) + 10)
            draw.polygon(
                [tuple(a) for a in zip(x, y)], fill=tuple(int(255 * c * shade) for c in base)
            )
        canvas.paste(image, (k * size, 0))
    canvas.save(path)


# Individual tools cut out of the instrument set: name: (indices of the glTF geometries that make up one tool, role).
INSTRUMENT_TOOLS = {
    "tool_clamp": (range(53, 60), "open hemostat with finger rings"),
    "tool_forceps": (range(28, 35), "small hemostat / forceps with finger rings"),
    "tool_scissors": (range(14, 19), "straight scissors"),
    "tool_scalpel": ([8, 9, 10], "scalpel"),
    "tool_retractor": ([26], "skin hook"),
}
INSTRUMENT_SCALE = 0.45 / 6.222  # the whole set is 0.45 m across in the converted asset


def import_tools(source, output):
    scene = trimesh.load(str(source / "surgical_instruments.glb"), force="scene")
    meshes = [m for m in scene.dump(concatenate=False) if isinstance(m, trimesh.Trimesh)]
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    report = {}
    for name, (indices, role) in INSTRUMENT_TOOLS.items():
        parts = []
        for i in indices:
            m = meshes[i].copy()
            m.vertices = np.asarray(m.vertices) @ rot.T * INSTRUMENT_SCALE
            parts.append(m)
        parts = [decimate(m, 6000) if len(m.faces) > 20000 else m for m in parts]
        allv = np.vstack([m.vertices for m in parts])
        if np.ptp(allv[:, 1]) > np.ptp(allv[:, 0]):  # lay the length along x
            turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
            for m in parts:
                m.vertices = np.asarray(m.vertices) @ turn.T
            allv = np.vstack([m.vertices for m in parts])
        lo, hi = allv.min(0), allv.max(0)
        shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        for m in parts:
            m.vertices = np.asarray(m.vertices) + shift
        info = f"{name}: Wenschel, CC-BY-4.0, https://sketchfab.com/3d-models/surgical-instruments-bd0ad0180f5148f1b3b331b8ebe6e494"
        usd = write_usd(name, parts, output / name, info)
        preview(parts, output / f"{name}_preview.png")
        size = allv.max(0) - allv.min(0)
        report[name] = {
            "usd": str(usd.relative_to(output)),
            "triangles": int(sum(len(m.faces) for m in parts)),
            "size_xyz_m": [round(float(v), 3) for v in size],
            "license": "CC-BY-4.0",
            "author": "Wenschel",
            "source": "https://sketchfab.com/3d-models/surgical-instruments-bd0ad0180f5148f1b3b331b8ebe6e494",
        }
        print(name, report[name]["triangles"], report[name]["size_xyz_m"], role, flush=True)
    return report


# --- variants cut from the same downloads -------------------------------------------------------


def scene_parts(path):
    """Every geometry of a GLB with its node transform applied, in glTF's Y-up frame: [(material name, mesh)]."""
    scene = trimesh.load(str(path), force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(transform)
        parts.append((name, mesh))
    return parts


def finish(parts, size_along_max, name, output, info, lay_down=False):
    """Z-up, scale so the longest side is `size_along_max` meters, center on xy, rest on z = 0, write USD."""
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    meshes = [m for _, m in parts]
    allv = np.vstack([m.vertices for m in meshes]) @ rot.T
    scale = size_along_max / float((allv.max(0) - allv.min(0)).max())
    lo, hi = (allv * scale).min(0), (allv * scale).max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    for m in meshes:
        m.vertices = np.asarray(m.vertices) @ rot.T * scale + shift
    usd = write_usd(name, meshes, output / name, info)
    preview(meshes, output / f"{name}_preview.png")
    allv = np.vstack([m.vertices for m in meshes])
    return usd, meshes, allv


def import_flat_bed(source, output):
    """The hospital bed with its backrest laid flat and its pillow removed, so a patient can lie on it."""
    parts = scene_parts(source / "hospital_bed.glb")
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    allv = np.vstack([m.vertices for _, m in parts]) @ rot.T
    scale = 2.2 / float((allv.max(0) - allv.min(0)).max())
    center = (allv.min(0) + allv.max(0)) / 2
    kept = []
    angle = np.radians(22.5)  # measured: the mattress rises 0.27 m over 0.66 m from its hinge
    pivot = np.array(
        [0.0, 0.0, 0.775]
    )  # hinge, in meters from the bed's center (x), mattress top (z)
    c, s = np.cos(-angle), np.sin(-angle)
    low = allv.min(0)[2] * scale
    for name, mesh in parts:
        if name.endswith("Material.008_0"):  # the black pillow
            continue
        v = (np.asarray(mesh.vertices) @ rot.T - center) * scale
        v[:, 2] += (-low) + (center[2] * scale)  # rest on z = 0 like finish() will
        moving = (v[:, 0] > pivot[0]) & (v[:, 0] < 1.04) & (v[:, 2] > 0.48)
        x, z = v[moving, 0] - pivot[0], v[moving, 2] - pivot[2]
        v[moving, 0] = pivot[0] + c * x - s * z
        v[moving, 2] = pivot[2] + s * x + c * z
        mesh = mesh.copy()
        mesh.vertices = (v / scale + center * 0) @ np.linalg.inv(
            rot.T
        )  # back to the glTF frame for finish()
        kept.append((name, mesh))
    return kept


def import_fridge_open(source, output):
    """The refrigerator without its doors, door handles and drawer fronts: an open cabinet a real drawer slides out of."""
    parts = scene_parts(source / "blood_bank_refrigerator.glb")
    drop = set(range(36, 68))  # drawer fronts + handles + door glass, door frames, door handles
    return [p for i, p in enumerate(parts) if i not in drop]


def lower_arms(
    parts,
    angle_deg=68.0,
    shoulder_x=0.42,
    shoulder_up=2.97,
    blend=(0.40, 0.65),
    forward_deg=0.0,
    shoulder_depth=-0.07,
):
    """Bend a T-posed figure's arms down about its shoulders (glTF Y-up frame: x across, y up).

    Only the shirt and body (arms) move, weighted so the torso stays put and the sleeve blends into the shoulder.
    """
    posed = []
    for name, mesh in parts:
        if "Shirt" in name or "Body1" in name:
            mesh = mesh.copy()
            v = np.asarray(mesh.vertices).copy()
            side = np.sign(v[:, 0])
            reach = np.abs(v[:, 0])
            w = np.clip((reach - blend[0]) / (blend[1] - blend[0]), 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            theta = np.radians(angle_deg) * w
            dx = reach - shoulder_x
            dy = v[:, 1] - shoulder_up
            moving = reach > blend[0]
            nx = shoulder_x + dx * np.cos(theta) + dy * np.sin(theta)
            ny = shoulder_up - dx * np.sin(theta) + dy * np.cos(theta)
            v[moving, 0] = (side * nx)[moving]
            v[moving, 1] = ny[moving]
            if forward_deg:  # then swing the arms forward (toward +z, the way the figure faces) about the shoulders
                f = np.radians(forward_deg) * w
                dy2 = v[:, 1] - shoulder_up
                dz2 = v[:, 2] - shoulder_depth
                v[moving, 1] = (shoulder_up + dy2 * np.cos(f) + dz2 * np.sin(f))[moving]
                v[moving, 2] = (shoulder_depth - dy2 * np.sin(f) + dz2 * np.cos(f))[moving]
            mesh.vertices = v
        posed.append((name, mesh))
    return posed


def import_extras(source, output):
    """The sterilizer, extinguisher, IV pump and a doctor with relaxed arms, from the files added later."""
    report = {}

    def record(name, usd, meshes, allv, lic, author, slug):
        size = allv.max(0) - allv.min(0)
        report[name] = {
            "usd": str(usd.relative_to(output)),
            "triangles": int(sum(len(m.faces) for m in meshes)),
            "size_xyz_m": [round(float(v), 3) for v in size],
            "license": lic,
            "author": author,
            "source": f"https://sketchfab.com/3d-models/{slug}",
        }
        print(name, report[name], flush=True)

    def thin(parts, budget):
        total = sum(len(m.faces) for _, m in parts)
        if total <= budget:
            return parts
        return [(n, decimate(m, max(60, int(budget * len(m.faces) / total)))) for n, m in parts]

    jobs = (
        (
            "fire_extinguisher",
            "fire_extinguisher.glb",
            None,
            0.55,
            3000,
            "CC-BY-4.0",
            "Loic",
            "fire-extinguisher",
        ),
        (
            "gas_sterilizer",
            "ethylene_oxide_gas__gas_sterilizer.glb",
            None,
            1.05,
            60000,
            "CC-BY-4.0",
            "Star Medical Oxygen Equipment",
            "ethylene-oxide-gas-gas-sterilizer",
        ),
        (
            "infusion_pump",
            "iv_pump_blood_vials_and_pill_packet.glb",
            slice(2, 5),
            0.24,
            2000,
            "CC-BY-4.0",
            "Ethan Cragun",
            "iv-pump-blood-vials-and-pill-packet",
        ),
        (
            "doctor",
            "doctor_administering_cpr.glb",
            "pose",
            1.75,
            60000,
            "CC-BY-4.0",
            "rwolzinger",
            "doctor-administering-cpr",
        ),
        (
            "doctor_operating",
            "doctor_administering_cpr.glb",
            "operate",
            1.75,
            60000,
            "CC-BY-4.0",
            "rwolzinger",
            "doctor-administering-cpr",
        ),
    )
    for name, file, select, size, budget, lic, author, slug in jobs:
        parts = scene_parts(source / file)
        if select == "pose":
            parts = lower_arms(parts)
        elif select == "operate":  # arms swung forward and down over an operating table
            parts = lower_arms(parts, angle_deg=84.0, forward_deg=40.0)
        elif select is not None:
            parts = parts[select]
        parts = thin(parts, budget)
        info = f"{name}: {author}, {lic}, https://sketchfab.com/3d-models/{slug}"
        usd, meshes, allv = finish(parts, size, name, output, info)
        record(name, usd, meshes, allv, lic, author, slug)
    return report


def import_variants(source, output):
    report = {}
    info_bed = "hospital_bed_flat: Ansh_Singla, CC-BY-4.0, https://sketchfab.com/3d-models/hospital-bed-0e974ea6eb9a4c069c56f152ae5162a4 (backrest flattened)"
    usd, meshes, allv = finish(
        import_flat_bed(source, output), 2.2, "hospital_bed_flat", output, info_bed
    )
    # Mattress top: the most common up-facing surface height between 0.55 and 1.0 m.
    zs, areas = [], []
    for m in meshes:
        up = m.face_normals[:, 2] > 0.9
        zs.append(m.triangles_center[up][:, 2])
        areas.append(m.area_faces[up])
    zs, areas = np.concatenate(zs), np.concatenate(areas)
    bins = np.linspace(0.55, 1.0, 46)
    hist, _ = np.histogram(zs, bins=bins, weights=areas)
    top = float(bins[int(np.argmax(hist))] + (bins[1] - bins[0]) / 2)
    size = allv.max(0) - allv.min(0)
    report["hospital_bed_flat"] = {
        "usd": str(usd.relative_to(output)),
        "triangles": int(sum(len(m.faces) for m in meshes)),
        "size_xyz_m": [round(float(v), 3) for v in size],
        "top_z": round(top, 3),
        "license": "CC-BY-4.0",
        "author": "Ansh_Singla",
        "source": "https://sketchfab.com/3d-models/hospital-bed-0e974ea6eb9a4c069c56f152ae5162a4",
    }
    print("hospital_bed_flat", report["hospital_bed_flat"], flush=True)
    info_fridge = "blood_bank_fridge_open: Kirk Nicholson, CC-BY-NC-4.0, https://sketchfab.com/3d-models/blood-bank-refrigerator-7d2f1f6e62424b2aa61ad427f0ca4564 (doors and drawer fronts removed)"
    usd, meshes, allv = finish(
        import_fridge_open(source, output), 0.85, "blood_bank_fridge_open", output, info_fridge
    )
    size = allv.max(0) - allv.min(0)
    report["blood_bank_fridge_open"] = {
        "usd": str(usd.relative_to(output)),
        "triangles": int(sum(len(m.faces) for m in meshes)),
        "size_xyz_m": [round(float(v), 3) for v in size],
        "license": "CC-BY-NC-4.0",
        "author": "Kirk Nicholson",
        "source": "https://sketchfab.com/3d-models/blood-bank-refrigerator-7d2f1f6e62424b2aa61ad427f0ca4564",
    }
    print("blood_bank_fridge_open", report["blood_bank_fridge_open"], flush=True)
    return report


def main():
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or list(ASSETS)
    report = {}
    if "extras" in names:
        names.remove("extras")
        report.update(import_extras(source, output))
    if "variants" in names:
        names.remove("variants")
        report.update(import_variants(source, output))
    if "tools" in names:
        names.remove("tools")
        report.update(import_tools(source, output))
    for name in names:
        file, axis, size, budget, lic, author, slug = ASSETS[name]
        meshes = load(source / file, axis, size, budget, lay=name in LAYING)
        info = f"{name}: {author}, {lic}, https://sketchfab.com/3d-models/{slug}"
        usd = write_usd(name, meshes, output / name, info)
        preview(meshes, output / f"{name}_preview.png")
        allv = np.vstack([m.vertices for m in meshes])
        report[name] = {
            "usd": str(usd.relative_to(output)),
            "triangles": int(sum(len(m.faces) for m in meshes)),
            "size_xyz_m": [round(float(s), 3) for s in allv.max(0) - allv.min(0)],
            "license": lic,
            "author": author,
            "source": f"https://sketchfab.com/3d-models/{slug}",
        }
        print(name, report[name]["triangles"], report[name]["size_xyz_m"], lic, flush=True)
    credits = output / "credits.json"
    merged = json.loads(credits.read_text()) if credits.exists() else {}
    merged.update(report)
    credits.write_text(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
