"""Import manufacturer X5A CAD, with the benchmark's explicit dual-arm fixture."""

import copy
import hashlib
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from prepare_arx_x5 import LIMITS
from prepare_arx_x5 import prepare_urdf as prepare_legacy_urdf

ROOT = Path("src/roboclock/envs/sports/common/assets/dual_arx_x5")
COMMIT = "74386441845f12af4da203843b6400951d6c5530"
URL = f"https://raw.githubusercontent.com/ARXroboticsX/ARX_Model/{COMMIT}/X5/X5A/"
CATALOG = Path("src/roboclock/simulators/isaac/embodiments/assets.json")
# The lab environment bundles the same manufacturer commit; reuse it when the bytes match.
LAB_SOURCE = Path("src/roboclock/envs/lab/assets/arx_x5")


def prepare_urdf():
    item = json.loads(CATALOG.read_text())["dual_arx_x5"]
    for name, sha in item["file_sha256"].items():
        dest = ROOT / "manufacturer" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        local = LAB_SOURCE / ("source.urdf" if name == "urdf/X5A.urdf" else name)
        if not dest.exists() and local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == sha:
            shutil.copy2(local, dest)
        if not dest.exists():
            tmp = dest.with_suffix(".partial")
            urllib.request.urlretrieve(URL + name, tmp)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Manufacturer checksum mismatch: {name}")
            tmp.replace(dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != sha:
            raise ValueError(f"Cached manufacturer checksum mismatch: {name}")
    source = ROOT / "manufacturer/urdf/X5A.urdf"
    native = ET.parse(source).getroot()
    for mesh in native.findall(".//mesh"):
        mesh.set(
            "filename",
            str((ROOT / "manufacturer/meshes" / Path(mesh.get("filename")).name).resolve()),
        )
    # Keep the existing wrist-sensor observation contract as an explicit accessory.
    camera_source = Path("results/arx-migration/assets/embodiments/ARX-X5/X5A.urdf")
    if not camera_source.exists():
        prepare_legacy_urdf()
    for element in ET.parse(camera_source).getroot():
        if "camera" not in element.get("name", ""):
            continue
        element = copy.deepcopy(element)
        for mesh in element.findall(".//mesh"):
            mesh.set("filename", str((camera_source.parent / mesh.get("filename")).resolve()))
        native.append(element)
    robot = ET.Element("robot", name="dual_arx_x5")
    base = ET.SubElement(robot, "link", name="base_link")
    inertial = ET.SubElement(base, "inertial")
    ET.SubElement(inertial, "mass", value="8")
    ET.SubElement(inertial, "inertia", ixx="0.3", iyy="0.3", izz="0.3", ixy="0", ixz="0", iyz="0")
    for side, y in [("left", 0.30), ("right", -0.30)]:
        for element in native:
            element = copy.deepcopy(element)
            element.set("name", side + "_" + element.get("name"))
            for node in element.findall("parent") + element.findall("child"):
                node.set("link", side + "_" + node.get("link"))
            if element.tag == "joint" and element.get("type") == "revolute":
                lo, hi, effort, speed = LIMITS[int(element.get("name")[-1]) - 1]
                element.find("limit").attrib.update(
                    lower=str(lo), upper=str(hi), effort=str(effort), velocity=str(speed)
                )
            if element.tag == "joint" and element.get("type") == "prismatic":
                element.find("limit").attrib.update(effort="30", velocity=".15")
            robot.append(element)
        j = ET.SubElement(robot, "joint", name=side + "_mount", type="fixed")
        ET.SubElement(j, "parent", link="base_link")
        ET.SubElement(j, "child", link=side + "_base_link")
        ET.SubElement(j, "origin", xyz=f"0 {y} 0", rpy="0 0 0")
    out = ROOT / "dual_arx_x5.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(out, encoding="utf-8", xml_declaration=True)
    return source, out, item


def main():
    source, urdf, item = prepare_urdf()
    from roboclock.simulators.isaac.runtime import IsaacRuntime

    with IsaacRuntime():
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.asset.importer.urdf")
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        cfg = URDFImporterConfig(
            urdf_path=str(urdf.resolve()),
            usd_path=str((ROOT / "usd").resolve()),
            fix_base=True,
            merge_fixed_joints=False,
            allow_self_collision=True,
            collision_type="Convex Decomposition",
            joint_drive_type="force",
            joint_target_type="position",
            override_joint_stiffness=1500,
            override_joint_damping=50,
        )
        usd = URDFImporter(cfg).import_urdf()
        manifest = dict(
            model="dual_arx_x5",
            source=item,
            # Relative to this manifest, so the converted asset can move between machines.
            usd=Path(usd).resolve().relative_to(ROOT.resolve()).as_posix(),
            source_urdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            derived_urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
            mount_separation_m=0.6,
            arm_limits=LIMITS,
            gripper_zero_aperture_m=0.0,
            limitations=[
                "Two official X5A arms assembled on an authored fixed fixture; not an official complete dual-arm SKU",
                "Surface finishes reconstructed from manufacturer product photo; CAD STLs contain no textures",
                "RoboTwin wrist-camera accessories retained; visual-only camera housings",
                "Independent synchronized finger drives; bounded simulation actuator parameters",
                "Manufacturer repository has no standalone asset license; binaries cached locally only",
            ],
        )
        (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("ARX_OFFICIAL_ASSET", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
