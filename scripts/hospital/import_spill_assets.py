"""Import the downloaded models of hospital B3 (hazardous drug spill) into assets/models.

    python scripts/hospital/import_spill_assets.py [--cache DIR]

All four sources are Sketchfab models under CC BY 4.0, downloaded without an account from the
Objaverse 1.0 mirror on Hugging Face (allenai/objaverse, glbs/<shard>/<uid>.glb). The raw GLBs are
cached in --cache (default: a temporary folder) and are not committed; this script rebuilds the
converted USD files from them.

  hd_bottle       "Medicine Glass Bottle" by Tomaso: the amber glass body and its contents only
                  (the spray pump, dip tube and printed labels are left out; the scene adds a
                  crimp cap and a purple CYTOTOXIC band).
  hd_shard_neck,  "CT4012-2020 Broken Bottle" by Helena Lima: the neck-and-shoulder piece, the base
  hd_shard_base,  piece and the loose splinters between them, each cut out by its position along
  hd_splinters    the bottle axis, scaled to the hd_bottle's diameter and shortened along the axis.
  cytotoxic_bin,  "Biohazard Waste Trashbox" by Andersta Studios: the open yellow tub and its lid,
  cytotoxic_bin_lid  squashed to a 5 L sharps tub.
  counter_rack    "Modular Shelving Unit (Metal)" by Sebastian Webster: cut off above its second
                  shelf and scaled to a two-tier counter-top rack.

Every model is written z up, in metres, centred on x = y = 0 and resting on z = 0 (the convention
of scripts/hospital/import_glb.py), and entered in assets/models/credits.json.
"""

import argparse
import gzip
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hospital"))
import import_glb as ig  # noqa: E402

MODELS = ROOT / "src/roboclock/envs/hospital/assets/models"
MIRROR = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
SOURCES = {
    "bottle": (
        "d3f308276ed9417a889dd8c7c2c345a3",
        "Tomaso",
        "https://sketchfab.com/3d-models/medicine-glass-bottle-d3f308276ed9417a889dd8c7c2c345a3",
    ),
    "broken": (
        "b6fcf8e9855a4c93a4f42d1ed5af13f2",
        "Helena Lima",
        "https://sketchfab.com/3d-models/ct4012-2020-broken-bottle-b6fcf8e9855a4c93a4f42d1ed5af13f2",
    ),
    "bin": (
        "a43d106afda14420ad6c5aaac118c677",
        "Andersta Studios",
        "https://sketchfab.com/3d-models/biohazard-waste-trashbox-a43d106afda14420ad6c5aaac118c677",
    ),
    "rack": (
        "6a1f71934d434585aa598b004f056ca9",
        "Sebastian Webster",
        "https://sketchfab.com/3d-models/modular-shelving-unit-metal-6a1f71934d434585aa598b004f056ca9",
    ),
}
LICENSE = "CC-BY-4.0"
BOTTLE_RADIUS = 0.019  # the glass body, metres
SHARD_AXIS = 0.70  # the broken bottle is slender: its pieces are shortened along the axis
BIN_SIZE = (0.19, 0.14, 0.15)  # outer size of the open tub (x, y, z)
RACK_SCALE = 0.45
PREVIEWS = Path(tempfile.gettempdir())  # three-view check images (set to --cache in main)
Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # glTF (x, y, z) -> (x, -z, y)


def fetch(cache):
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / "object-paths.json.gz"
    if not index.exists():
        index.write_bytes(urllib.request.urlopen(MIRROR + "object-paths.json.gz", timeout=120).read())
    paths = json.load(gzip.open(index))
    files = {}
    for key, (uid, _, _) in SOURCES.items():
        files[key] = cache / f"{uid}.glb"
        if not files[key].exists():
            files[key].write_bytes(urllib.request.urlopen(MIRROR + paths[uid], timeout=120).read())
    return files


def meshes_of(path):
    scene = trimesh.load(str(path), force="scene")
    return [m for m in scene.dump(concatenate=False) if isinstance(m, trimesh.Trimesh)]


def settle(parts):
    """Centre on x = y = 0 and rest on z = 0; returns the size."""
    points = np.vstack([m.vertices for m in parts])
    lo, hi = points.min(0), points.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    for m in parts:
        m.vertices = np.asarray(m.vertices) + shift
    return hi - lo


def write(name, parts, key, note, report):
    _, author, source = SOURCES[key]
    size = settle(parts)
    ig.write_usd(name, parts, MODELS / name, f"{name}: {author}, {LICENSE}, {source}")
    report[name] = {
        "usd": f"{name}/{name}.usd",
        "triangles": int(sum(len(m.faces) for m in parts)),
        "size_xyz_m": [round(float(v), 3) for v in size],
        "license": LICENSE,
        "author": author,
        "source": source,
        "via": "Objaverse 1.0 mirror (huggingface.co/datasets/allenai/objaverse)",
        "conversion": note,
    }
    return size


def plain(mesh, color, alpha=255):
    """Drop the source material for a flat colour (the scene binds its own glass and liquid)."""
    out = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    out.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[*color, alpha], roughnessFactor=0.15, metallicFactor=0.0
        )
    )
    return out


def bottle(path, report):
    parts = meshes_of(path)
    # By bounds (see the module docstring): the 864-face amber body and the 114-face contents.
    glass = next(m for m in parts if len(m.faces) == 864)
    liquid = next(m for m in parts if len(m.faces) == 114)
    radius = float(np.ptp(glass.vertices[:, 0]) / 2)
    scale = BOTTLE_RADIUS / radius
    out = []
    for mesh, color in ((glass, (150, 70, 20)), (liquid, (150, 20, 30))):
        m = plain(mesh, color)
        m.vertices = np.asarray(m.vertices) @ Y_UP.T * scale
        out.append(m)
    # Rest on the glass base; the contents sit just inside.
    lo = out[0].vertices.min(0)
    hi = out[0].vertices.max(0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    for m in out:
        m.vertices = np.asarray(m.vertices) + shift
    out[1].vertices[:, :2] *= 0.93
    out[1].vertices[:, 2] = np.clip(out[1].vertices[:, 2], 0.002, None)
    ig.write_usd("hd_bottle", out, MODELS / "hd_bottle", f"hd_bottle: Tomaso, {LICENSE}, {SOURCES['bottle'][2]}")
    size = np.vstack([m.vertices for m in out]).max(0) - np.vstack([m.vertices for m in out]).min(0)
    _, author, source = SOURCES["bottle"]
    report["hd_bottle"] = {
        "usd": "hd_bottle/hd_bottle.usd",
        "triangles": int(sum(len(m.faces) for m in out)),
        "size_xyz_m": [round(float(v), 3) for v in size],
        "license": LICENSE,
        "author": author,
        "source": source,
        "via": "Objaverse 1.0 mirror (huggingface.co/datasets/allenai/objaverse)",
        "conversion": "glass body and contents only (pump, dip tube and labels left out), "
        f"uniform scale to a {2 * BOTTLE_RADIUS * 1000:.0f} mm body; part_000 glass, part_001 contents",
    }
    ig.preview(out, PREVIEWS / "hd_bottle_preview.png")


def broken(path, report):
    (mesh,) = meshes_of(path)
    pieces = mesh.split(only_watertight=False)
    # The bottle lies along x: neck piece up to x = 14, base piece from x = 53, splinters between.
    groups = {"hd_shard_neck": [], "hd_shard_base": [], "hd_splinters": []}
    for piece in pieces:
        lo, hi = piece.bounds[:, 0]
        aside = piece.bounds[1, 2] > 5.0  # a loose shard lying beside the bottle
        key = (
            "hd_shard_neck"
            if hi < 14.0 and not aside
            else "hd_shard_base"
            if lo > 53.0
            else "hd_splinters"
        )
        groups[key].append(piece)
    body = np.vstack([p.vertices for p in groups["hd_shard_neck"]])
    diameter = float(np.ptp(body[:, 2]))
    scale = 2 * BOTTLE_RADIUS / diameter
    stretch = np.diag([SHARD_AXIS, 1.0, 1.0])
    for name, parts in groups.items():
        out = []
        for p in parts:
            m = plain(p, (150, 70, 20))
            m.vertices = np.asarray(m.vertices) @ Y_UP.T @ stretch * scale
            out.append(m)
        merged = [trimesh.util.concatenate(out)]
        merged[0].visual = out[0].visual
        if name == "hd_splinters":
            # Scatter: keep the loose pieces' own layout, flattened onto the counter.
            merged[0].vertices[:, 2] -= merged[0].vertices[:, 2].min()
            merged[0].vertices[:, 2] *= 0.35
        write(
            name,
            merged,
            "broken",
            f"cut from the single mesh by position along the bottle axis; uniform scale x{scale:.5f} "
            f"to the hd_bottle diameter, x{SHARD_AXIS} along the axis; flat amber colour",
            report,
        )
        ig.preview(merged, PREVIEWS / f"{name}_preview.png")


def waste_bin(path, report):
    parts = meshes_of(path)
    tub = max(parts, key=lambda m: np.ptp(m.vertices[:, 1]))
    lid = min(parts, key=lambda m: np.ptp(m.vertices[:, 1]))
    native = np.ptp(tub.vertices @ Y_UP.T, axis=0)
    scale = np.array(BIN_SIZE) / native
    for name, mesh in (("cytotoxic_bin", tub), ("cytotoxic_bin_lid", lid)):
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices) @ Y_UP.T * scale
        write(
            name,
            [mesh],
            "bin",
            f"{'tub' if name == 'cytotoxic_bin' else 'lid'} only, scaled x{np.round(scale, 3).tolist()} "
            "(x, y, z) to a 5 L sharps tub; source texture kept",
            report,
        )
        ig.preview([mesh], PREVIEWS / f"{name}_preview.png")


def rack(path, report):
    parts = meshes_of(path)
    cut = 0.80  # native metres: just above the second shelf
    out = []
    for mesh in parts:
        lo, hi = mesh.bounds[:, 1]
        if lo > cut:
            continue  # the third shelf and its brackets
        mesh = mesh.copy()
        if hi > cut:  # the poles: end them at the second shelf
            mesh.vertices[:, 1] = np.minimum(mesh.vertices[:, 1], cut)
        mesh.vertices = np.asarray(mesh.vertices) @ Y_UP.T * RACK_SCALE
        out.append(mesh)
    write(
        "counter_rack",
        out,
        "rack",
        f"cut off above the second shelf, uniform scale x{RACK_SCALE} (a two-tier counter-top rack)",
        report,
    )
    # Shelf tops, for the layout: the two shelf slabs are the parts whose native height is ~25 mm.
    tops = sorted(float(m.vertices[:, 2].max()) for m in out if np.ptp(m.vertices[:, 2]) < 0.02)
    report["counter_rack"]["shelf_tops_m"] = [round(t, 4) for t in tops]
    ig.preview(out, PREVIEWS / "counter_rack_preview.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "hospital_spill_glb")
    args = parser.parse_args()
    global PREVIEWS
    PREVIEWS = args.cache
    files = fetch(args.cache)
    report = {}
    bottle(files["bottle"], report)
    broken(files["broken"], report)
    waste_bin(files["bin"], report)
    rack(files["rack"], report)
    credits = MODELS / "credits.json"
    merged = json.loads(credits.read_text())  # re-read: other importers add their own entries
    merged.update(report)
    credits.write_text(json.dumps(merged, indent=2) + "\n")
    for name, entry in report.items():
        print(name, entry["triangles"], entry["size_xyz_m"], entry.get("shelf_tops_m", ""))


if __name__ == "__main__":
    main()
