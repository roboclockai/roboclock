"""Import the downloaded models of hospital B5 (anaphylaxis: adrenaline auto-injector).

    python scripts/hospital/import_b5_assets.py [CACHE_DIR]

Three Sketchfab models (all CC BY 4.0), fetched from the Objaverse 1.0 mirror on Hugging Face
(no account needed) and written to src/roboclock/envs/hospital/assets/models/<name>/<name>.usd
(z up, metres), with their entries merged into models/credits.json:

  epi_autoinjector      "Epipen" by andyshermanart: the labelled barrel, scaled to a 0.15 m pen and
                        turned tip down. Origin at the centre of the barrel (it is the visual of a
                        rigid body whose collider is the barrel's box), not on z = 0.
  epi_autoinjector_tip  the same model's orange needle-end shroud, as its own file with the origin
                        at the centre of the shroud's collider: in the scene it is a second rigid
                        body on a sprung prismatic joint (bedside/autoinjector.py).
  epi_safety_cap        the same model's blue safety release, lying beside the holder (the trainer
                        is armed). Resting on z = 0.
  overbed_table         "Over Bed-table" by Dwayne.Barnett: C-frame table on castors. Top long axis
                        along x, column at +x, centred on the top's xy, floor at z = 0.
  iv_stand              the infusion stand with its bag from "Crutch and IV Drip" by Matt_LeMoine
                        (the crutch is left out), 1.9 m tall, centred on xy, resting on z = 0.

The injector's mechanism (barrel collider, sprung tip, measured travel) is authored in
bedside/autoinjector.py: it has to be a measured mechanism, not a mesh.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hospital"))
from import_glb import preview, scene_parts, write_usd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "src/roboclock/envs/hospital/assets/models"
MIRROR = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs"
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)

SOURCES = {
    "epipen": ("000-020", "9011dc7a598f4eefbbe378abb8c0ddb8", "Epipen", "andyshermanart"),
    "table": ("000-017", "c93acd5558564d4ab2c2c8b5a3cbc73e", "Over Bed-table", "Dwayne.Barnett"),
    "drip": ("000-085", "5cc65c6aed374220b67f7d60e679153e", "Crutch and IV Drip", "Matt_LeMoine"),
}
PEN_LENGTH = 0.150  # m, barrel plus the tip's 10 mm (the model is 0.213 m with its blue cap)
TIP_TRAVEL = 0.010  # the orange shroud stands this far out of the barrel


def fetch(key, cache):
    folder, uid, _, _ = SOURCES[key]
    path = cache / f"{uid}.glb"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{MIRROR}/{folder}/{uid}.glb", timeout=180) as reply:
            path.write_bytes(reply.read())
    return path


def entry(key, raw, usd, meshes, conversion, **extra):
    folder, uid, title, author = SOURCES[key]
    allv = np.vstack([m.vertices for m in meshes])
    return {
        "usd": str(usd.relative_to(MODELS)),
        "triangles": int(sum(len(m.faces) for m in meshes)),
        "size_xyz_m": [round(float(v), 4) for v in allv.max(0) - allv.min(0)],
        "license": "CC-BY-4.0",
        "author": author,
        "title": title,
        "source": f"https://sketchfab.com/3d-models/{uid}",
        "download": f"{MIRROR}/{folder}/{uid}.glb (Objaverse 1.0 mirror)",
        "raw_glb_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "conversion": conversion,
        "used_by": "hospital B5 anaphylaxis_autoinjector",
        **extra,
    }


def moved(mesh, transform):
    """A copy of `mesh` with `transform(vertices)` applied, keeping its material and UVs."""
    out = mesh.copy()
    out.vertices = transform(np.asarray(mesh.vertices, dtype=float))
    return out


def epipen(cache, report):
    """The model's long axis is its own z with the orange tip at +z and the blue cap at -z."""
    raw = fetch("epipen", cache)
    parts = dict(scene_parts(raw))
    barrel, tip, hole, cap = (parts[f"Object_{k}"] for k in (2, 3, 1, 0))
    lo, hi = barrel.bounds[0][2], barrel.bounds[1][2]
    tip_end = tip.bounds[1][2]
    scale = (PEN_LENGTH - TIP_TRAVEL) / (hi - lo)  # the barrel alone is 0.14 m
    centre = (lo + hi) / 2

    def body_frame(v):  # tip down: a half turn about x, origin at the barrel's centre
        v = (v - [0.0, 0.0, centre]) * scale
        return v * [1.0, -1.0, -1.0]

    barrel_m = moved(barrel, body_frame)
    usd = write_usd(
        "epi_autoinjector", [barrel_m], MODELS / "epi_autoinjector",
        "epi_autoinjector: andyshermanart, CC-BY-4.0, " + f"https://sketchfab.com/3d-models/{SOURCES['epipen'][1]}",
    )
    preview([barrel_m], cache / "epi_autoinjector_preview.png")
    half = float(barrel_m.bounds[1][2])
    report["epi_autoinjector"] = entry(
        "epipen", raw, usd, [barrel_m],
        "labelled barrel only, scaled x%.3f to 0.14 m, turned tip down (-z); origin at the barrel centre" % scale,
        half_length_m=round(half, 4),
    )
    # The shroud: keep the part that shows outside the barrel plus what slides inside it; its
    # collider is a box TIP_LENGTH long whose lower face is the shroud's end.
    tip_length = 0.030
    shown = float((tip_end - hi) * scale)  # how far the model's tip stands out of the barrel
    tip_end_z = -half - shown  # in the barrel frame

    def tip_frame(v):
        v = body_frame(v)
        return v - [0.0, 0.0, tip_end_z + tip_length / 2]

    tip_m, hole_m = moved(tip, tip_frame), moved(hole, tip_frame)
    usd = write_usd(
        "epi_autoinjector_tip", [tip_m, hole_m], MODELS / "epi_autoinjector_tip",
        "epi_autoinjector_tip: andyshermanart, CC-BY-4.0, " + f"https://sketchfab.com/3d-models/{SOURCES['epipen'][1]}",
    )
    report["epi_autoinjector_tip"] = entry(
        "epipen", raw, usd, [tip_m, hole_m],
        "orange needle-end shroud of the same model, same scale, tip down; origin 15 mm above its end",
        shown_m=round(shown, 4),
        end_below_origin_m=round(tip_length / 2, 4),
    )
    lo_c = cap.bounds[0]

    def cap_frame(v):  # standing on its open end beside the holder
        v = (v - [0.0, 0.0, lo_c[2]]) * scale
        return v

    cap_m = moved(cap, cap_frame)
    usd = write_usd(
        "epi_safety_cap", [cap_m], MODELS / "epi_safety_cap",
        "epi_safety_cap: andyshermanart, CC-BY-4.0, " + f"https://sketchfab.com/3d-models/{SOURCES['epipen'][1]}",
    )
    report["epi_safety_cap"] = entry(
        "epipen", raw, usd, [cap_m], "blue safety release of the same model, same scale, resting on z = 0"
    )


def table(cache, report):
    raw = fetch("table", cache)
    parts = scene_parts(raw)
    top = next(m for n, m in parts if n.startswith("table top"))
    allv = np.vstack([m.vertices for _, m in parts]) @ Y_UP_TO_Z_UP.T
    floor = float(allv[:, 2].min())
    topv = np.asarray(top.vertices) @ Y_UP_TO_Z_UP.T
    cx, cy = (topv.min(0)[:2] + topv.max(0)[:2]) / 2
    # Z-up the top's long axis is y with the column at -y: turn it so the long axis is x, column at +x.
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def frame(v):
        v = v @ Y_UP_TO_Z_UP.T - [cx, cy, floor]
        return v @ turn.T

    meshes = [moved(m, frame) for _, m in parts]
    for mesh in meshes:  # flat materials: the source's wood top becomes a laminate, the frame chrome
        del mesh
    usd = write_usd(
        "overbed_table", meshes, MODELS / "overbed_table",
        "overbed_table: Dwayne.Barnett, CC-BY-4.0, " + f"https://sketchfab.com/3d-models/{SOURCES['table'][1]}",
    )
    preview(meshes, cache / "overbed_table_preview.png")
    tv = frame(np.asarray(top.vertices))
    report["overbed_table"] = entry(
        "table", raw, usd, meshes,
        "z up, unscaled; top's long axis along x with the column at +x, centred on the top, floor at z = 0",
        top_z=round(float(tv[:, 2].max()), 4),
        top_thickness_m=round(float(tv[:, 2].max() - tv[:, 2].min()), 4),
        top_size_m=[round(float(v), 4) for v in (tv.max(0) - tv.min(0))[:2]],
    )


def drip(cache, report):
    raw = fetch("drip", cache)
    parts = [(n, m) for n, m in scene_parts(raw) if n.startswith("IV Drip")]
    allv = np.vstack([m.vertices for _, m in parts]) @ Y_UP_TO_Z_UP.T
    scale = 1.90 / float(allv[:, 2].max() - allv[:, 2].min())
    lo, hi = allv.min(0), allv.max(0)
    centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]

    def frame(v):
        return (v @ Y_UP_TO_Z_UP.T - centre) * scale

    meshes = [moved(m, frame) for _, m in parts]
    usd = write_usd(
        "iv_stand", meshes, MODELS / "iv_stand",
        "iv_stand: Matt_LeMoine, CC-BY-4.0, " + f"https://sketchfab.com/3d-models/{SOURCES['drip'][1]}",
    )
    preview(meshes, cache / "iv_stand_preview.png")
    bag = frame(np.asarray(parts[0][1].vertices))
    report["iv_stand"] = entry(
        "drip", raw, usd, meshes,
        "the IV stand and its bag only (crutch left out), scaled to 1.90 m, centred on xy, resting on z = 0",
        bag_bottom_m=[round(float(v), 4) for v in [bag[:, 0].mean(), bag[:, 1].mean(), bag[:, 2].min()]],
    )


def main():
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache/roboclock/b5_glb"
    report = {}
    epipen(cache, report)
    table(cache, report)
    drip(cache, report)
    credits = MODELS / "credits.json"
    merged = json.loads(credits.read_text())  # read just before writing: other imports add entries
    merged.update(report)
    credits.write_text(json.dumps(merged, indent=2) + "\n")
    for name, item in report.items():
        print(name, item["triangles"], item["size_xyz_m"], {k: v for k, v in item.items() if k.endswith("_m") or k == "top_z"})


if __name__ == "__main__":
    main()
