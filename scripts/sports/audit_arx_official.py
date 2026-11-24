"""Check manufacturer geometry provenance and the derived articulation contract."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh
import numpy as np
from scipy.spatial import cKDTree
from pxr import Usd, UsdGeom

root = Path("src/roboclock/envs/sports/common/assets/dual_arx_x5")
manifest = json.loads((root / "manifest.json").read_text())
source = ET.parse(root / "manufacturer/urdf/X5A.urdf").getroot()
derived = ET.parse(root / "dual_arx_x5.urdf").getroot()
files = {}
for name, sha in manifest["source"]["file_sha256"].items():
    actual = hashlib.sha256((root / "manufacturer" / name).read_bytes()).hexdigest()
    assert actual == sha, name
    files[name] = actual
limit_adaptations = {}
for side in ("left", "right"):
    for link in source.findall("link"):
        name = link.get("name")
        target = derived.find(f"link[@name='{side}_{name}']")
        for part in ("origin", "mass", "inertia"):
            assert target.find(f"inertial/{part}").attrib == link.find(f"inertial/{part}").attrib
        for kind in ("visual", "collision"):
            mesh = target.find(f"{kind}/geometry/mesh")
            assert [float(v) for v in mesh.get("scale", "1 1 1").split()] == [1, 1, 1]
            expected = root / "manufacturer/meshes" / (name + ".STL")
            assert Path(mesh.get("filename")).resolve() == expected.resolve()
            assert Path(target.find(f"{kind}/geometry/mesh").get("filename")).name == name + ".STL"
            assert target.find(f"{kind}/origin").attrib == link.find(f"{kind}/origin").attrib
    for i in range(1, 9):
        a = source.find(f"joint[@name='joint{i}']")
        b = derived.find(f"joint[@name='{side}_joint{i}']")
        assert a.get("type") == b.get("type")
        for part in ("parent", "child"):
            assert b.find(part).get("link") == side + "_" + a.find(part).get("link")
        differences = {k: {"manufacturer": v, "simulation": b.find("limit").get(k)}
                       for k, v in a.find("limit").attrib.items()
                       if float(v) != float(b.find("limit").get(k))}
        if differences:
            limit_adaptations[b.get("name")] = differences
        for part in ("origin", "axis"):
            assert a.find(part).attrib == b.find(part).attrib
    assert (
        len(
            [
                j
                for j in derived.findall("joint")
                if j.get("name").startswith(side + "_") and j.get("type") != "fixed"
            ]
        )
        == 8
    )
# Verify the actual converted USD, including instance proxies for both arms.
# A preserved URDF filename alone cannot establish that the importer kept the mesh.
stage = Usd.Stage.Open(str(root / manifest["usd"]))
assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
native = {link.get("name"): trimesh.load(root / "manufacturer/meshes" / (link.get("name") + ".STL"))
          for link in source.findall("link")}
usd_meshes = []
for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Mesh) or prim.GetName() not in native:
        continue
    mesh = UsdGeom.Mesh(prim)
    original = native[prim.GetName()]
    points = np.asarray(mesh.GetPointsAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    assert np.all(counts == 3) and len(counts) == len(original.faces), prim.GetPath()
    distances, matched = cKDTree(original.vertices).query(points)
    reverse = cKDTree(points).query(original.vertices)[0]
    max_error = float(max(distances.max(), reverse.max()))
    assert max_error < 1e-7, (prim.GetPath(), max_error)
    # Compare triangles after remapping importer vertex indices to source vertices.
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    def canonical(triangles):
        rows = np.sort(triangles, axis=1)
        return rows[np.lexsort(rows.T[::-1])]
    np.testing.assert_array_equal(canonical(matched[faces]), canonical(original.faces))
    matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))[:3, :3]
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(matrix) > 0.999999, prim.GetPath()
    usd_meshes.append(dict(path=str(prim.GetPath()), triangles=len(counts),
                           vertex_error_m=max_error, scale_preserved=True))
assert len(usd_meshes) == 36, len(usd_meshes)  # 9 links x 2 arms x visual/collision

# Inner tip planes must agree with RobotRig's measured aperture convention.
planes = []
for i in (7, 8):
    mesh = trimesh.load(root / f"manufacturer/meshes/link{i}.STL")
    tip = mesh.vertices[mesh.vertices[:, 0] > 0.04]
    y = float(source.find(f"joint[@name='joint{i}']/origin").get("xyz").split()[1])
    planes.append(y + (tip[:, 1].min() if i == 7 else tip[:, 1].max()))
assert abs(planes[0] - planes[1]) < 1e-5, planes
report = dict(
    manufacturer_commit=manifest["source"]["commit"],
    file_sha256=files,
    checks={
        "all_nine_visual_and_collision_meshes_native": True,
        "joint_frames_native": True,
        "urdf_masses_and_inertias_native": True,
        "usd_visual_collision_meshes_verified": len(usd_meshes),
        "usd_triangle_connectivity_native": True,
        "usd_no_scaling_or_mirroring": True,
        "usd_max_vertex_error_m": max(m["vertex_error_m"] for m in usd_meshes),
        "dof_count": 16,
        "zero_aperture_error_m": abs(planes[0] - planes[1]),
    },
    usd_meshes=usd_meshes,
    joint_limit_adaptations=limit_adaptations,
    limitations=[
        "Joint limits and drive parameters are simulation adaptations, not measured hardware calibration",
        "Manufacturer CAD omits external wiring, markings and scanned material textures",
        "Does not validate dynamic grasps or controller performance; see actual episode records"
    ],
)
Path("results/arx-official").mkdir(parents=True, exist_ok=True)
Path("results/arx-official/asset_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["checks"], indent=2))
