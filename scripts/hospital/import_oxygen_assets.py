"""Import the downloaded models of hospital B1 (oxygen desaturation), all CC BY 4.0.

    python scripts/hospital/import_oxygen_assets.py [--cache DIR]

Three Sketchfab models, fetched as GLB from the Objaverse 1.0 mirror on Hugging Face (no token
needed), are converted to real-scale, z-up USD files under
src/roboclock/envs/hospital/assets/models/<name>/ and entered in models/credits.json:

  oxygen_flowmeter  "Med Air Oxygen Outlet" by Dwayne.Barnett: a wall oxygen outlet plate with a
                    plugged-in Thorpe-tube flowmeter (glass tube, ball float, needle valve, barbed
                    outlet), at the model's own scale (tube 109 mm tall); the model's hose is
                    left out (the scene routes its own tubing to the patient). Model frame:
                    +x out of the wall (the plate's back face at x = 0), z up, origin on the
                    outlet axis. Written as three files so the mechanism can be articulated:
                    oxygen_flowmeter.usd (plate, valve body, tube; the glass meshes keep their
                    names so the scene can bind a clear material),
                    oxygen_flowmeter_ball.usd (the float, origin at its centre) and
                    oxygen_flowmeter_knob.usd (the control knob with its stem, turned to point out
                    of the wall along +x, origin at its centre, enlarged x1.5 to 24 mm so the X5
                    fingers can pinch it).
  patient_supine    "Ali Sleeping ECG" by GLTFconverter: a clothed adult lying supine (head toward
                    -y, face up), 1.80 m long.
  oxygen_mask       "Oxygen mask" by shadowdraak: only the oronasal cup and its tube stubs (the
                    goggles are left out), scaled to 0.11 m across, opening down (-z).

  patient_monitor_head  the display head of `patient_monitor` ("medical monitor" by
                    golem.troll228, imported by scripts/hospital/import_bvm_assets.py, which must
                    run first): cut off its roll stand, enlarged x1.5 to a 33 x 23 cm bedside
                    monitor, screen toward +x, origin at the screen's centre.

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
    "oxygen_flowmeter": ("3eb022a187a24cd78441113e77005bc6", "000-129", "Med Air Oxygen Outlet", "Dwayne.Barnett"),
    "patient_supine": ("f4439c6aa9c74cc48937aac932cf5c2c", "000-004", "Ali Sleeping ECG", "GLTFconverter"),
    "oxygen_mask": ("23ac0d5b96ca47c6a3fac637cb91119a", "000-022", "Oxygen mask", "shadowdraak"),
}
USED_BY = "hospital B1 oxygen_desaturation"
KNOB_SCALE = 1.5  # the 16 mm knob enlarged to 24 mm across
PATIENT_LENGTH = 1.80
MASK_WIDTH = 0.11


def fetch(name, cache):
    uid, shard = SOURCES[name][:2]
    raw = MODELS / name / "source.glb"
    raw.parent.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{uid}.glb" if cache else None
    if not raw.exists():
        if cached is not None and cached.exists():
            raw.write_bytes(cached.read_bytes())
        else:
            url = f"{MIRROR}/{shard}/{uid}.glb"
            raw.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    return raw


def nodes(raw):
    """{node name: mesh in the scene frame} of a GLB."""
    scene = trimesh.load(str(raw), force="scene")
    out = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        out[node] = mesh
    return out


def write(name, file, meshes, names, info):
    """glb.write_usd into MODELS/name/file.usd with the meshes named `names`."""
    from pxr import Sdf, Usd

    folder = MODELS / name
    path = glb.write_usd(file, meshes, folder, json.dumps(info))
    stage = Usd.Stage.Open(str(path))
    edit = Sdf.BatchNamespaceEdit()
    for i, new in enumerate(names):
        if new != f"part_{i:03d}":
            edit.Add(f"/Model/part_{i:03d}", f"/Model/{new}")
    stage.GetRootLayer().Apply(edit)
    stage.GetRootLayer().Save()
    return path


def credit(name, raw, conversion, meshes, **extra):
    uid, shard, title, author = SOURCES[name]
    allv = np.vstack([m.vertices for m in meshes])
    return {
        "usd": f"{name}/{name}.usd",
        "triangles": int(sum(len(m.faces) for m in meshes)),
        "size_xyz_m": [round(float(v), 3) for v in allv.max(0) - allv.min(0)],
        "license": "CC-BY-4.0",
        "author": author,
        "title": title,
        "source": f"https://sketchfab.com/3d-models/{uid}",
        "downloaded_from": f"{MIRROR}/{shard}/{uid}.glb (Objaverse 1.0)",
        "source_glb_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "conversion": conversion,
        "used_by": USED_BY,
        **extra,
    }


def safe(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


# --- flowmeter -----------------------------------------------------------------------------------

BALL = "Sphere_PlasticNob-black_0"
KNOB = ("nob_PlasticNob-black_0", "nob02_Silver_0")
HOSE = ("BezierCurve_PlasticHose_0",)
GLASS = ("cone01_Glass_0", "valve03_Glass_0", "innerTube_Glass_0")
# glTF (x right, y up, z out of the wall) -> model (x out of the wall, y, z up).
TO_MODEL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def import_flowmeter(raw):
    parts = nodes(raw)
    back = min(float(m.vertices[:, 2].min()) for n, m in parts.items() if n not in HOSE)
    for mesh in parts.values():  # the plate's back face on x = 0
        mesh.vertices = (np.asarray(mesh.vertices) - [0.0, 0.0, back]) @ TO_MODEL.T
    info = {"credit": "Med Air Oxygen Outlet by Dwayne.Barnett, CC-BY-4.0"}
    shell = {n: m for n, m in parts.items() if n != BALL and n not in KNOB and n not in HOSE}
    write("oxygen_flowmeter", "oxygen_flowmeter", list(shell.values()), [safe(n) for n in shell], info)
    ball = parts[BALL]
    ball_center = ball.vertices.mean(0)
    ball_radius = float(np.ptp(ball.vertices[:, 2]) / 2)
    ball.vertices = np.asarray(ball.vertices) - ball_center
    write("oxygen_flowmeter", "oxygen_flowmeter_ball", [ball], ["ball"], info)
    # The knob: its axis runs along the wall (model y, pointing +y) in the source; turn it to point
    # out of the wall (+x), centre it and enlarge it.
    knob = [parts[n] for n in KNOB]
    cap = knob[0].vertices
    center = (cap.max(0) + cap.min(0)) / 2
    turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +y -> +x
    for mesh in knob:
        mesh.vertices = ((np.asarray(mesh.vertices) - center) @ turn.T) * KNOB_SCALE
    write("oxygen_flowmeter", "oxygen_flowmeter_knob", knob, ["cap", "stem"], info)
    cap = knob[0].vertices
    tube = parts["innerTube_Glass_0"].vertices
    body = parts["valve03_Glass_0"].vertices
    barb = parts["bottomCone_YellowPlasitc_0"].vertices
    extra = {
        "parts": {
            "ball": "oxygen_flowmeter/oxygen_flowmeter_ball.usd",
            "knob": "oxygen_flowmeter/oxygen_flowmeter_knob.usd",
        },
        "glass_meshes": [safe(n) for n in GLASS],
        "tube_axis_xy_m": [round(float(v), 4) for v in (tube[:, :2].max(0) + tube[:, :2].min(0)) / 2],
        "tube_z_m": [round(float(tube[:, 2].min()), 4), round(float(tube[:, 2].max()), 4)],
        "tube_inner_radius_m": round(float(np.ptp(tube[:, 1]) / 2), 4),
        "tube_outer_radius_m": round(float(np.ptp(parts["cone01_Glass_0"].vertices[:, 1]) / 2), 4),
        "ball_radius_m": round(ball_radius, 4),
        "ball_rest_m": [round(float(v), 4) for v in ball_center],
        "valve_front_x_m": round(float(body[:, 0].max()), 4),
        "valve_z_m": [round(float(body[:, 2].min()), 4), round(float(body[:, 2].max()), 4)],
        "knob_radius_m": round(float(np.ptp(cap[:, 2]) / 2), 4),
        "knob_x_m": [round(float(cap[:, 0].min()), 4), round(float(cap[:, 0].max()), 4)],
        "knob_stem_x_m": round(float(knob[1].vertices[:, 0].min()), 4),
        "barb_tip_m": [round(float(v), 4) for v in barb[np.argmin(barb[:, 2])]],
        "plate_yz_m": [
            [round(float(v), 4) for v in parts["Cube_metalYellow_0"].vertices[:, 1:].min(0)],
            [round(float(v), 4) for v in parts["Cube_metalYellow_0"].vertices[:, 1:].max(0)],
        ],
    }
    glb.preview(list(shell.values()) + [ball] + knob, MODELS / "oxygen_flowmeter" / "preview.png")
    return credit(
        "oxygen_flowmeter",
        raw,
        "unscaled; +x out of the wall (plate back at x = 0), z up, origin on the outlet axis; hose "
        "left out; split into shell, ball float and control knob (knob turned from the side to "
        "the front, enlarged x1.5 to 24 mm)",
        list(shell.values()),
        **extra,
    )


# --- patient and mask ----------------------------------------------------------------------------


def import_patient(raw):
    meshes = glb.load(raw, "z", PATIENT_LENGTH, 30000)
    # Leave out the untextured cornea shells and the eyelash cards (alpha cards: the converter
    # writes opaque materials, so they would cover the eyes).
    meshes = [
        m for m in meshes
        if getattr(m.visual.material, "baseColorTexture", None) is not None
        and not (np.ptp(m.vertices[:, 2]) < 0.03 and len(m.vertices) < 600)
    ]
    info = {"credit": "Ali Sleeping ECG by GLTFconverter, CC-BY-4.0"}
    write("patient_supine", "patient_supine", meshes, [f"part_{i:03d}" for i in range(len(meshes))], info)
    glb.preview(meshes, MODELS / "patient_supine" / "preview.png")
    allv = np.vstack([m.vertices for m in meshes])
    head = allv[allv[:, 1] < allv[:, 1].min() + 0.22]
    face = head[np.argmax(head[:, 2])]
    return credit(
        "patient_supine",
        raw,
        f"scaled to {PATIENT_LENGTH} m; supine, head toward -y, face up, centred on xy, resting on z = 0; "
        "decimated; cornea shells and eyelash cards left out",
        meshes,
        nose_tip_m=[round(float(v), 4) for v in face],
    )


def import_mask(raw):
    parts = nodes(raw)
    keep = [parts[n] for n in ("OxygenPart_Mask_0", "OxygenTubes_Mask_0")]
    cup = keep[0].vertices
    scale = MASK_WIDTH / float(np.ptp(cup[:, 0]))
    center = (cup.max(0) + cup.min(0)) / 2
    # glTF y up with the cup opening toward -z (the face side): turn the opening down (-z model).
    rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for mesh in keep:
        mesh.vertices = ((np.asarray(mesh.vertices) - center) @ rot.T) * scale
    info = {"credit": "Oxygen mask by shadowdraak, CC-BY-4.0"}
    write("oxygen_mask", "oxygen_mask", keep, ["cup", "tubes"], info)
    glb.preview(keep, MODELS / "oxygen_mask" / "preview.png")
    return credit(
        "oxygen_mask",
        raw,
        f"oronasal cup and tube stubs only (goggles left out); scaled x{scale:.4f} to {MASK_WIDTH} m "
        "across; centred on the cup, source axes kept (y up, cup opening toward -z)",
        keep,
    )


# --- monitor head --------------------------------------------------------------------------------

MONITOR_HEAD_Z = 1.09  # the head starts above this height on the 1.25 m roll-stand model
MONITOR_HEAD_SCALE = 1.5


def import_monitor_head(_raw=None):
    from PIL import Image
    from pxr import Usd, UsdGeom

    source = MODELS / "patient_monitor"
    stage = Usd.Stage.Open(str(source / "patient_monitor.usd"))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    uv = np.asarray(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get(), dtype=float)
    keep = np.all(points[faces][:, :, 2] > MONITOR_HEAD_Z, axis=1)
    faces = faces[keep]
    used = np.unique(faces)
    remap = -np.ones(len(points), dtype=int)
    remap[used] = np.arange(len(used))
    head = points[used]
    center = np.array([head[:, 0].max(), (head[:, 1].max() + head[:, 1].min()) / 2,
                       (head[:, 2].max() + head[:, 2].min()) / 2])
    out = trimesh.Trimesh((head - center) * MONITOR_HEAD_SCALE, remap[faces], process=False)
    texture = Image.open(source / "tex00.jpg")
    out.visual = trimesh.visual.TextureVisuals(
        uv=uv[used], material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture)
    )
    info = {"credit": "medical monitor by golem.troll228, CC-BY-4.0 (display head only)"}
    write("patient_monitor_head", "patient_monitor_head", [out], ["head"], info)
    glb.preview([out], MODELS / "patient_monitor_head" / "preview.png")
    v = np.asarray(out.vertices)
    credits = json.loads((MODELS / "credits.json").read_text())["patient_monitor"]
    return {
        "usd": "patient_monitor_head/patient_monitor_head.usd",
        "triangles": int(len(out.faces)),
        "size_xyz_m": [round(float(x), 3) for x in v.max(0) - v.min(0)],
        **{k: credits[k] for k in ("license", "author", "title", "source", "downloaded_from", "source_glb_sha256")},
        "conversion": f"derived from models/patient_monitor: the display head only (roll stand cut off above {MONITOR_HEAD_Z} m), "
                      f"enlarged x{MONITOR_HEAD_SCALE}; screen toward +x, origin at the screen centre",
        "used_by": USED_BY,
        "bounds_m": [[round(float(x), 4) for x in v.min(0)], [round(float(x), 4) for x in v.max(0)]],
    }


IMPORTERS = {
    "patient_monitor_head": import_monitor_head,
    "oxygen_flowmeter": import_flowmeter,
    "patient_supine": import_patient,
    "oxygen_mask": import_mask,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, help="Folder holding <uid>.glb files already downloaded")
    p.add_argument("names", nargs="*", default=list(IMPORTERS))
    a = p.parse_args()
    report = {}
    for name in a.names:
        raw = fetch(name, a.cache) if name in SOURCES else None
        report[name] = IMPORTERS[name](raw)
        print(name, json.dumps(report[name])[:600], flush=True)
    credits = MODELS / "credits.json"
    merged = json.loads(credits.read_text())  # re-read: other importers write this file too
    merged.update(report)
    credits.write_text(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
