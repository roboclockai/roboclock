"""Import the downloaded models of hospital A3 (bag-valve-mask ventilation), all CC BY 4.0.

    python scripts/hospital/import_bvm_assets.py [--cache DIR]

Four Sketchfab models, fetched as GLB from the Objaverse 1.0 mirror on Hugging Face (no token
needed), are converted to real-scale, z-up USD files under
src/roboclock/envs/hospital/assets/models/<name>/ and entered in models/credits.json:

  bvm_resuscitator  "AMBU" by centronina3d: a photogrammetry scan of a self-inflating resuscitator.
                    The oxygen reservoir bag is cut off, the bag axis laid along +x (patient valve
                    at +x, its port up), the bag scaled to 127 mm across. Written as separate
                    meshes: `bag_*` (the compressible bag, deformed at run time) and `valve_*`.
  bvm_manikin       "Training Manikin" by pixelated_stars: a head-and-torso rescue manikin, laid
                    supine (head toward -y, face up), 0.85 m long. Its weathered pool texture is
                    replaced by plain manikin skin; the mesh is subdivided once so the chest can
                    rise smoothly at run time.
  bvm_tube          "Tubo Endotracheale" by centronina3d: a scanned tracheal tube with its 15 mm
                    connector, scaled so the connector is 15 mm across, connector at the origin.
  patient_monitor   "medical monitor" by golem.troll228: a vital-signs monitor on a roll stand,
                    scaled to 1.25 m, its screen facing +x.

The raw GLBs are kept next to the USD files (source.glb) with their SHA-256 in credits.json.
"""

import argparse
import hashlib
import importlib.util
import json
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "src/roboclock/envs/hospital/assets/models"
MIRROR = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs"

_spec = importlib.util.spec_from_file_location("import_glb", REPO / "scripts/hospital/import_glb.py")
glb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glb)

# name: (Sketchfab uid, Objaverse shard, title, author)
SOURCES = {
    "bvm_resuscitator": ("8c2e3053dc8f47acace6c5d8373af49b", None, "AMBU", "centronina3d"),
    "bvm_manikin": ("0472bfdb2bac4a648b013040ba0d0fab", None, "Training Manikin", "pixelated_stars"),
    "bvm_tube": ("bd45550906c54ed28ad7cdb0c5b0c574", None, "Tubo Endotracheale", "centronina3d"),
    "patient_monitor": ("84aa2c97829b4557bc077e8006d97e58", None, "medical monitor", "golem.troll228"),
}
SHARDS = {  # Objaverse object-paths.json.gz entries of the four uids
    "8c2e3053dc8f47acace6c5d8373af49b": "000-158",
    "0472bfdb2bac4a648b013040ba0d0fab": "000-118",
    "bd45550906c54ed28ad7cdb0c5b0c574": "000-122",
    "84aa2c97829b4557bc077e8006d97e58": "000-138",
}

BAG_DIAMETER = 0.127  # adult self-inflating bag (about 1.5 L)
TEXTURE_GAMMA = 0.62  # brightening applied to the resuscitator scan's texture
MANIKIN_LENGTH = 0.85
MONITOR_HEIGHT = 1.25
CONNECTOR = 0.015  # ISO 5356-1 15 mm tracheal tube connector
SKIN = (0.80, 0.62, 0.50)


def fetch(name, cache):
    uid = SOURCES[name][0]
    raw = MODELS / name / "source.glb"
    raw.parent.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{uid}.glb" if cache else None
    if not raw.exists():
        if cached is not None and cached.exists():
            raw.write_bytes(cached.read_bytes())
        else:
            url = f"{MIRROR}/{SHARDS[uid]}/{uid}.glb"
            raw.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    return raw


def parts(raw):
    scene = trimesh.load(str(raw), force="scene")
    return [m for m in scene.dump(concatenate=False) if isinstance(m, trimesh.Trimesh)]


def submesh(mesh, keep):
    """The faces of `mesh` whose three vertices are all in `keep`, with its texture coordinates."""
    faces = mesh.faces[np.all(keep[mesh.faces], axis=1)]
    used = np.unique(faces)
    remap = -np.ones(len(mesh.vertices), dtype=int)
    remap[used] = np.arange(len(used))
    out = trimesh.Trimesh(mesh.vertices[used], remap[faces], process=False)
    out.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(mesh.visual.uv)[used], material=mesh.visual.material
    )
    return out


def write(name, meshes, names, info):
    """glb.write_usd, then the meshes renamed (`part_000` -> names[0] ...)."""
    from pxr import Sdf, Usd

    folder = MODELS / name
    path = glb.write_usd(name, meshes, folder, info)
    stage = Usd.Stage.Open(str(path))
    edit = Sdf.BatchNamespaceEdit()
    for i, new in enumerate(names):
        edit.Add(f"/Model/part_{i:03d}", f"/Model/{new}")
    stage.GetRootLayer().Apply(edit)
    stage.GetRootLayer().Save()
    glb.preview(meshes, folder / "preview.png")
    allv = np.vstack([m.vertices for m in meshes])
    return {
        "usd": f"{name}/{name}.usd",
        "triangles": int(sum(len(m.faces) for m in meshes)),
        "size_xyz_m": [round(float(v), 3) for v in allv.max(0) - allv.min(0)],
        **info,
    }


def credit(name, raw, conversion, **extra):
    uid, _, title, author = SOURCES[name]
    return {
        "license": "CC-BY-4.0",
        "author": author,
        "title": title,
        "source": f"https://sketchfab.com/3d-models/{uid}",
        "downloaded_from": f"{MIRROR}/{SHARDS[uid]}/{uid}.glb (Objaverse 1.0)",
        "source_glb_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "conversion": conversion,
        **extra,
    }


# --- resuscitator -------------------------------------------------------------------------------

# In the scan's principal frame (unit = scan unit, axis = the long axis, reservoir at negative):
RESERVOIR_CUT = -0.22  # everything below is the oxygen reservoir bag and its neck
BAG_SPAN = (-0.12, 0.74)  # the compressible bag between its two necks
PORT_SPAN = (1.0, 1.2)  # the patient valve, where the patient port leaves the axis


def import_bvm(raw):
    meshes = parts(raw)
    allv = np.vstack([m.vertices for m in meshes])
    center = allv.mean(0)
    _, vectors = np.linalg.eigh(np.cov((allv - center).T))
    axis = vectors[:, -1]
    p0 = (allv - center) @ axis
    if np.ptp((allv - center)[p0 < -1.0] @ vectors[:, 1]) < 0.3:  # reservoir (the wide flat end) at -axis
        axis = -axis
    along = (allv - center) @ axis
    radial = (allv - center) - np.outer(along, axis)
    mid = (along > 0.2) & (along < 0.55)
    radial_center = radial[mid].mean(0)  # the bag's axis line
    radial = radial - radial_center
    port = radial[(along > PORT_SPAN[0]) & (along < PORT_SPAN[1])]
    up = port[np.argmax(np.linalg.norm(port, axis=1))]
    up = up - axis * (up @ axis)
    up /= np.linalg.norm(up)
    side = np.cross(up, axis)
    frame = np.stack([axis, side, up])  # rows: model x (axis), y, z (patient port up)
    scale = BAG_DIAMETER / (2 * np.percentile(np.linalg.norm(radial[mid], axis=1), 50))
    out, names = [], []
    lo_z = np.inf
    for mesh in meshes:  # the scan's albedo is dark (navy bag): lift it so the bag reads in the ward
        material = mesh.visual.material
        if material.baseColorTexture is not None:
            lut = [round(255 * (i / 255) ** TEXTURE_GAMMA) for i in range(256)]
            material.baseColorTexture = material.baseColorTexture.convert("RGB").point(lut * 3)
    for k, mesh in enumerate(meshes):
        local = ((np.asarray(mesh.vertices) - center) @ frame.T) - frame @ radial_center
        mesh = mesh.copy()
        mesh.vertices = local
        keep = local[:, 0] >= RESERVOIR_CUT
        bag = keep & (local[:, 0] >= BAG_SPAN[0]) & (local[:, 0] <= BAG_SPAN[1])
        # A face straddling a neck goes to the valve side, so the bag mesh ends inside the necks.
        face_bag = np.all(bag[mesh.faces], axis=1)
        face_keep = np.all(keep[mesh.faces], axis=1)
        for label, faces in (("bag", face_bag), ("valve", face_keep & ~face_bag)):
            if not faces.any():
                continue
            sub = trimesh.Trimesh(mesh.vertices, mesh.faces[faces], process=False)
            sub.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=mesh.visual.material)
            sub.remove_unreferenced_vertices()
            # Not decimated: the scan's texture atlas is a mosaic of tiny islands, and merging
            # vertices across them smears the texture.
            sub.vertices = np.asarray(sub.vertices) * scale
            lo_z = min(lo_z, float(sub.vertices[:, 2].min())) if label == "bag" else lo_z
            out.append(sub)
            names.append(f"{label}_{k}")
    bag_x = np.array(BAG_SPAN) * scale
    origin_x = bag_x.mean()
    for mesh in out:  # origin: under the middle of the bag, on the surface the bag rests on
        mesh.vertices = np.asarray(mesh.vertices) - [origin_x, 0.0, lo_z]
    bagv = np.vstack([m.vertices for m, n in zip(out, names) if n.startswith("bag")])
    valvev = np.vstack([m.vertices for m, n in zip(out, names) if n.startswith("valve")])
    patient = valvev[valvev[:, 0] > bag_x[1] - origin_x]
    top = patient[patient[:, 2] > patient[:, 2].max() - 0.004]
    extra = {
        "bag_span_x_m": [round(float(v - origin_x), 4) for v in bag_x],
        "bag_top_z_m": round(float(np.percentile(bagv[np.abs(bagv[:, 0]) < 0.03, 2], 99)), 4),
        "bag_axis_z_m": round(float(-lo_z), 4),
        "patient_port_m": [round(float(v), 4) for v in top.mean(0)],
        "lowest_z_m": round(float(min(bagv[:, 2].min(), valvev[:, 2].min())), 4),
    }
    info = credit(
        "bvm_resuscitator",
        raw,
        "photogrammetry scan; oxygen reservoir bag cut off; bag axis along +x, patient port up, "
        f"origin under the bag's middle; scaled x{scale:.4f} (bag {BAG_DIAMETER * 1000:.0f} mm across); "
        f"texture brightened (gamma {TEXTURE_GAMMA}); bag and valve bodies split into separate meshes",
        **extra,
    )
    return write("bvm_resuscitator", out, names, info)


# --- manikin ------------------------------------------------------------------------------------


def import_manikin(raw):
    (mesh,) = parts(raw)
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(0), v.max(0)
    scale = MANIKIN_LENGTH / (hi[1] - lo[1])
    # glTF frame: y up (head at +y), face toward +z. Supine: head toward -y, face up (+z).
    local = np.stack([-(v[:, 0] - (lo[0] + hi[0]) / 2), -(v[:, 1] - lo[1]), v[:, 2] - lo[2]], axis=1) * scale
    local[:, 1] += MANIKIN_LENGTH / 2  # origin under the middle of the body, on the mattress
    out = trimesh.Trimesh(local, mesh.faces, process=False)
    for _ in range(2):
        out = out.subdivide()
    out = trimesh.Trimesh(out.vertices, out.faces, process=True)  # merged vertices: smooth normals
    trimesh.smoothing.filter_taubin(out, iterations=6)
    out.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[round(255 * c) for c in SKIN] + [255], roughnessFactor=0.55, metallicFactor=0.0
        )
    )
    out.visual.uv = None
    v = np.asarray(out.vertices)
    head = v[v[:, 1] < v[:, 1].min() + 0.26 * MANIKIN_LENGTH]
    face = head[head[:, 2] > head[:, 2].max() - 0.03]
    extra = {
        "head_end": "-y",
        "top_z_m": round(float(v[:, 2].max()), 4),
        "face_top_m": [round(float(c), 4) for c in face.mean(0)[:2]] + [round(float(head[:, 2].max()), 4)],
    }
    info = credit(
        "bvm_manikin",
        raw,
        f"laid supine (head toward -y, face up), scaled x{scale:.5f} to {MANIKIN_LENGTH} m; subdivided twice and "
        "smoothed; the weathered pool texture replaced by plain manikin skin",
        **extra,
    )
    return write("bvm_manikin", [out], ["body"], info)


# --- tracheal tube ------------------------------------------------------------------------------


def import_tube(raw):
    meshes = parts(raw)
    allv = np.vstack([m.vertices for m in meshes])
    # The scan lies flat in its xz plane. The connector is the thick block: the densest wide slab.
    flat = allv[:, [0, 2]]
    grid = np.round(flat / 0.03).astype(int)
    cells, counts = np.unique(grid, axis=0, return_counts=True)
    block = cells[np.argmax(counts)] * 0.03
    near = np.linalg.norm(flat - block, axis=1) < 0.09
    center = np.array([flat[near, 0].mean(), allv[near, 1].mean(), flat[near, 1].mean()])
    width = np.percentile(np.linalg.norm(flat[near] - center[[0, 2]], axis=1), 60) * 2
    scale = 0.8 * CONNECTOR / (width * 0.55)
    out = []
    for mesh in meshes:
        v = (np.asarray(mesh.vertices) - center) * scale
        # scan plane (x, z) -> the model's vertical xz plane stays; y is the thin direction.
        mesh.vertices = v
        out.append(mesh)
    info = credit(
        "bvm_tube",
        raw,
        f"photogrammetry scan; origin at the 15 mm connector; scaled x{scale:.4f}; lies flat in its xz plane",
    )
    return write("bvm_tube", out, [f"tube_{i}" for i in range(len(out))], info)


# --- monitor ------------------------------------------------------------------------------------


def import_monitor(raw):
    (mesh,) = parts(raw)
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(0), v.max(0)
    scale = MONITOR_HEIGHT / (hi[1] - lo[1])
    head = v[v[:, 1] > 79.0]
    hlo, hhi = head.min(0), head.max(0)
    # glTF: y up, the screen faces +x (a quad at x = 13.5). Model: z up, screen toward +x, the
    # stand's column at the origin.
    local = np.stack([v[:, 0], -v[:, 2], v[:, 1] - lo[1]], axis=1) * scale
    out = trimesh.Trimesh(local, mesh.faces, process=False)
    out.visual = mesh.visual
    front = float(hhi[0] * scale)
    extra = {
        # the monitor head's front: centre of the face, and its (width along y, height along z)
        "screen_center_m": [
            round(front, 4),
            round(float(-(hlo[2] + hhi[2]) / 2 * scale), 4),
            round(float(((hlo[1] + hhi[1]) / 2 - lo[1]) * scale), 4),
        ],
        "head_size_m": [round(float((hhi[2] - hlo[2]) * scale), 4), round(float((hhi[1] - hlo[1]) * scale), 4)],
    }
    info = credit(
        "patient_monitor",
        raw,
        f"scaled x{scale:.5f} to {MONITOR_HEIGHT} m; z up, screen toward +x, column at the origin",
        **extra,
    )
    return write("patient_monitor", [out], ["monitor"], info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=None, help="folder of already downloaded <uid>.glb files")
    args = parser.parse_args()
    report = {}
    for name, convert in (
        ("bvm_resuscitator", import_bvm),
        ("bvm_manikin", import_manikin),
        ("bvm_tube", import_tube),
        ("patient_monitor", import_monitor),
    ):
        report[name] = convert(fetch(name, args.cache))
        print(name, json.dumps(report[name], indent=1))
    credits = MODELS / "credits.json"
    merged = json.loads(credits.read_text())
    merged.update(report)  # re-read just before writing: other imports edit this file too
    credits.write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()
