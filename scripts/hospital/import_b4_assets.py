"""Import the downloaded models of hospital B4 (ventilator circuit reconnection).

    python scripts/hospital/import_b4_assets.py [CACHE_DIR]

Three Sketchfab models (all CC BY 4.0), fetched from the Objaverse 1.0 mirror on Hugging Face
(no account needed) and converted with the helpers of scripts/hospital/import_glb.py into
src/roboclock/envs/hospital/assets/models/<name>/<name>.usd (z up, metres, centred on xy,
resting on z = 0; no decimation), with their entries merged into models/credits.json:

  icu_ventilator      "Ventilator" by piyustf: ICU ventilator on its trolley with two circuit
                      support arms, humidifier and hoses. Untextured; one off-white shell.
  patient_manikin     "Male base mesh" by bhagath2005, laid on its back (head toward +y, face
                      up), 1.75 m tall, its arms lowered to its sides. Untextured; the scene binds a manikin-skin material.
  tracheostomy_tube   "Tubo endotracheale" by frizzigiorgia: a cuffed airway tube with its
                      15 mm connector and pilot balloon, the bedside spare. 0.11 m long.

The reconnection mechanism itself (15 mm connector, catheter-mount elbow, hose) is authored in
bedside/ventilator.py: it has to be a measured mechanism, not a mesh.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hospital"))
from import_glb import decimate, preview, scene_parts, write_usd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "src/roboclock/envs/hospital/assets/models"
MIRROR = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs"
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
# Standing (z up, facing -y) -> supine: head toward +y, face toward +z.
SUPINE = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

# name: (Objaverse folder, Sketchfab uid, title, author, size along the longest side (m),
#        triangle budget, geometry names to drop, extra rotation, flat colour)
ASSETS = {
    "icu_ventilator": (
        "000-109", "85d9d3a0ae9e42f8af8e485766ea9e1e", "Ventilator", "piyustf",
        1.45, 80000, (), None, (0.90, 0.91, 0.90),
    ),
    "patient_manikin": (
        "000-019", "427fc72a505140abbd56451d692a3734", "Male base mesh", "bhagath2005",
        1.75, 50000, ("Sphere__0",), SUPINE, (0.80, 0.62, 0.52),
    ),
    "tracheostomy_tube": (
        "000-077", "ac4f56a34a8f45a2a6dd6356802774dc", "Tubo endotracheale", "frizzigiorgia",
        0.11, 5000, (), None, (0.93, 0.95, 0.96),
    ),
}


def fetch(folder, uid, cache):
    path = cache / f"{uid}.glb"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{MIRROR}/{folder}/{uid}.glb", timeout=180) as reply:
            path.write_bytes(reply.read())
    return path


SHOULDER = (0.19, 0.47)  # |x|, y of the shoulder joint in the supine manikin's frame (m)


def lower_arms(v, degrees=15.0):
    """The base mesh holds its arms 23 degrees out (0.91 m across the hands): bring them to about
    8 degrees, beside the hips, so the patient fits a 0.75 m mattress."""
    v = v.copy()
    side, reach = np.sign(v[:, 0]), np.abs(v[:, 0])
    w = np.clip((reach - 0.15) / 0.09, 0.0, 1.0)
    theta = np.radians(degrees) * w * w * (3 - 2 * w)
    dx, dy = reach - SHOULDER[0], v[:, 1] - SHOULDER[1]
    moving = (reach > 0.15) & (v[:, 1] < SHOULDER[1] + 0.06)
    v[moving, 0] = (side * (SHOULDER[0] + dx * np.cos(theta) + dy * np.sin(theta)))[moving]
    v[moving, 1] = (SHOULDER[1] - dx * np.sin(theta) + dy * np.cos(theta))[moving]
    return v


def convert(name, cache):
    folder, uid, title, author, size, budget, drop, turn, color = ASSETS[name]
    raw = fetch(folder, uid, cache)
    parts = [(n, m) for n, m in scene_parts(raw) if n not in drop]
    rot = Y_UP_TO_Z_UP if turn is None else turn @ Y_UP_TO_Z_UP
    allv = np.vstack([m.vertices for _, m in parts]) @ rot.T
    scale = size / float((allv.max(0) - allv.min(0)).max())
    lo, hi = (allv * scale).min(0), (allv * scale).max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    total = sum(len(m.faces) for _, m in parts)
    meshes = []
    for _, mesh in parts:
        if total > budget:
            mesh = decimate(mesh, max(200, int(budget * len(mesh.faces) / total)))
        vertices = np.asarray(mesh.vertices) @ rot.T * scale + shift
        if name == "patient_manikin":
            vertices = lower_arms(vertices)
        mesh = trimesh.Trimesh(vertices, np.asarray(mesh.faces), process=False)
        # One flat colour: the sources carry no textures (a default grey shading group).
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[*(round(255 * c) for c in color), 255],
                roughnessFactor=0.55,
                metallicFactor=0.0,
            )
        )
        meshes.append(mesh)
    source = f"https://sketchfab.com/3d-models/{uid}"
    usd = write_usd(name, meshes, MODELS / name, f"{name}: {author}, CC-BY-4.0, {source}")
    preview(meshes, cache / f"{name}_preview.png")
    allv = np.vstack([m.vertices for m in meshes])
    return {
        "usd": str(usd.relative_to(MODELS)),
        "triangles": int(sum(len(m.faces) for m in meshes)),
        "size_xyz_m": [round(float(v), 3) for v in allv.max(0) - allv.min(0)],
        "license": "CC-BY-4.0",
        "author": author,
        "title": title,
        "source": source,
        "download": f"{MIRROR}/{folder}/{uid}.glb (Objaverse 1.0 mirror)",
        "raw_glb_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "used_by": "hospital B4 ventilator_reconnect",
    }


def main():
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache/roboclock/b4_glb"
    report = {name: convert(name, cache) for name in ASSETS}
    credits = MODELS / "credits.json"
    merged = json.loads(credits.read_text())  # read just before writing: other imports add entries
    merged.update(report)
    credits.write_text(json.dumps(merged, indent=2) + "\n")
    for name, entry in report.items():
        print(name, entry["triangles"], entry["size_xyz_m"])


if __name__ == "__main__":
    main()
