"""Import the household A1 sugar jar: "Glass Jar" by Extrone (Sketchfab, CC BY 4.0), via the Objaverse 1.0 mirror.

    python scripts/household/import_sugar_jar.py

Writes src/roboclock/envs/household/assets/x5/objects/sugar_jar_0/: the raw GLB, sugar_jar.usda (the jar body only,
the red screw cap left out: A1's jar stands open) and SOURCE.json. The body is scaled by SCALE into metres, z up,
origin at the centre of its base: 54 mm across, 74 mm tall, a 30 mm mouth, the size of the C2 bottle the X5 jaws
(60.3 mm open) already pinch. No material is bound; the scene binds glass.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, Vt

UID = '4823762be90c437ea35d45dfa976e085'
URL = f'https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-029/{UID}.glb'
SCALE = .054/(2*.08847)  # native body 177 mm across -> 54 mm
OUT = Path(__file__).resolve().parents[2]/'src/roboclock/envs/household/assets/x5/objects/sugar_jar_0'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT/'glass_jar.glb'
    if not raw.exists():
        raw.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    scene = trimesh.load(raw)
    body = scene.geometry['Cylinder.002_Bottle_0']  # geometry frame is z up, base at z = 0
    v = np.asarray(body.vertices)*SCALE
    n = np.asarray(body.vertex_normals)
    stage = Usd.Stage.CreateNew(str(OUT/'sugar_jar.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    root = UsdGeom.Xform.Define(stage, '/SugarJar')
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, '/SugarJar/Jar')
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in v]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in n]))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr([3]*len(body.faces))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(body.faces.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))])
    stage.GetRootLayer().Save()
    lo, hi = v.min(0), v.max(0)
    (OUT/'SOURCE.json').write_text(json.dumps(dict(
        source=f'https://sketchfab.com/3d-models/{UID} ("Glass Jar" by Extrone), downloaded from the Objaverse 1.0 mirror {URL}',
        license='CC BY 4.0 (Sketchfab model license; attribution: "Glass Jar" by Extrone)',
        conversion=f'body mesh only (Cylinder.002_Bottle_0; the red cap Cylinder.002_Cap_0 is left out), scaled x{SCALE:.5f} '
                   'into metres, z up, origin at the base centre; no material (the scene binds glass); '
                   'scripts/household/import_sugar_jar.py',
        size_m=(hi-lo).round(4).tolist(),
        raw_glb_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()), indent=2)+'\n')
    print('wrote', OUT, (hi-lo).round(4))


if __name__ == '__main__':
    main()
