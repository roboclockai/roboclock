"""Import the downloaded models of hospital B6 (massive limb haemorrhage, tourniquet).

    python scripts/hospital/import_b6_assets.py [CACHE_DIR]

Four Sketchfab models (all CC BY 4.0), fetched from the Objaverse 1.0 mirror on Hugging Face (no
account needed) and converted with the helpers of scripts/hospital/import_glb.py into
src/roboclock/envs/hospital/assets/models/<name>/<name>.usd (z up, metres, centred on xy,
resting on z = 0, source textures kept, no decimation), with their entries merged into
models/credits.json. They dress the bleeding-control kit laid out beside the injured leg:

  trauma_kit_pouch   "Tactical FIRST AID KIT" by ruskoschey: an olive MOLLE first-aid pouch, laid
                     on its back. 0.16 m long.
  bandage_roll       "Bandage Roll" by Tactical_Beard: a rolled elastic bandage. 0.075 m long.
  gauze_pack         "Gauze and plaster - Garza e cerotto" by centronina3d: photogrammetry scan of
                     a gauze roll and a roll of plaster. 0.15 m across.
  trauma_forceps     "Medical Scissors 1" by 2301253: ring-handled locking forceps. 0.16 m long.

The scene also reuses B4's patient_manikin and hospital_bed_flat. No windlass tourniquet with a
free licence was found (Sketchfab search through its public API, restricted to the uids of the
Objaverse mirror: "tourniquet", "combat tourniquet", "CAT tourniquet", "windlass", "IFAK", ...),
and the tourniquet has to be a measured mechanism anyway: bedside/tourniquet.py authors it.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hospital"))
from import_glb import preview, scene_parts, write_usd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "src/roboclock/envs/hospital/assets/models"
MIRROR = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs"
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
ON_ITS_BACK = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)  # standing -> lying along +y

# name: (Objaverse folder, Sketchfab uid, slug, title, author, longest side (m), extra rotation)
ASSETS = {
    "trauma_kit_pouch": (
        "000-000", "011c33c121284bc88bb765a85511dae1", "tactical-first-aid-kit",
        "Tactical FIRST AID KIT", "ruskoschey", 0.16, ON_ITS_BACK,
    ),
    "bandage_roll": (
        "000-138", "dd33fad87aec468daeb0398ea4a6389f", "bandage-roll",
        "Bandage Roll", "Tactical_Beard", 0.075, None,
    ),
    "gauze_pack": (
        "000-122", "1c833a1aefd2493eb5bf268b92df5c05", "gauze-and-plaster-garza-e-cerotto",
        "Gauze and plaster - Garza e cerotto", "centronina3d", 0.15, None,
    ),
    "trauma_forceps": (
        "000-099", "38885408402e4a368053de6961fe59a1", "medical-scissors-1",
        "Medical Scissors 1", "2301253", 0.16, None,
    ),
}


def fetch(folder, uid, cache):
    path = cache / f"{uid}.glb"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{MIRROR}/{folder}/{uid}.glb", timeout=180) as reply:
            path.write_bytes(reply.read())
    return path


def convert(name, cache):
    folder, uid, slug, title, author, size, turn = ASSETS[name]
    raw = fetch(folder, uid, cache)
    parts = scene_parts(raw)
    rot = Y_UP_TO_Z_UP if turn is None else turn @ Y_UP_TO_Z_UP
    allv = np.vstack([m.vertices for _, m in parts]) @ rot.T
    scale = size / float((allv.max(0) - allv.min(0)).max())
    lo, hi = (allv * scale).min(0), (allv * scale).max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    meshes = []
    for _, mesh in parts:
        mesh.vertices = np.asarray(mesh.vertices) @ rot.T * scale + shift
        meshes.append(mesh)
    source = f"https://sketchfab.com/3d-models/{slug}-{uid}"
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
        "conversion": "scaled, z up, source textures kept, no decimation; visual only",
        "used_by": "hospital B6 haemorrhage_tourniquet",
    }


def main():
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache/roboclock/b6_glb"
    report = {name: convert(name, cache) for name in ASSETS}
    credits = MODELS / "credits.json"
    merged = json.loads(credits.read_text())  # read just before writing: other imports add entries
    merged.update(report)
    credits.write_text(json.dumps(merged, indent=2) + "\n")
    for name, entry in report.items():
        print(name, entry["triangles"], entry["size_xyz_m"])


if __name__ == "__main__":
    main()
