"""Import pinned RoboTwin X5 meshes as a fixed Dual ARX X5 articulation."""

import copy
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path("results/arx-migration/assets")
ARCHIVE_SHA = "85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789"
URL = "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/embodiments.zip"
# Conservative X5-2025 manufacturer limits, replacing RoboTwin's unbounded placeholders.
LIMITS = [
    (-2.617994, 3.1416, 27, 5.5),
    (0, 3.665, 27, 5.5),
    (0, 3.141593, 27, 5.5),
    (-1.571, 1.571, 7, 20),
    (-1.571, 1.571, 7, 20),
    (-2.094395, 2.094, 7, 20),
]


def prepare_urdf():
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "embodiments.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != ARCHIVE_SHA:
            raise ValueError("RoboTwin archive checksum mismatch")
        temporary.replace(archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith("embodiments/ARX-X5/"):
                if ".." in Path(name).parts:
                    raise ValueError("Unsafe archive member")
                z.extract(name, ROOT)
    source = ROOT / "embodiments/ARX-X5/X5A.urdf"
    native = ET.parse(source).getroot()
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
            for mesh in element.findall(".//mesh"):
                mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
            if element.tag == "joint" and element.get("type") == "revolute":
                i = int(element.get("name")[-1]) - 1
                lo, hi, effort, speed = LIMITS[i]
                element.find("limit").attrib.update(
                    lower=str(lo), upper=str(hi), effort=str(effort), velocity=str(speed)
                )
            if element.tag == "joint" and element.get("type") == "prismatic":
                # Equal opposing bounded finger drives; measured contacts determine grasps.
                element.find("limit").attrib.update(effort="30", velocity="0.15")
            # The camera is visual-only upstream. Preserve its mesh without inventing collision.
            robot.append(element)
        joint = ET.SubElement(robot, "joint", name=side + "_mount", type="fixed")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=side + "_base_link")
        ET.SubElement(joint, "origin", xyz=f"0 {y} 0", rpy="0 0 0")
    out = ROOT / "dual_arx_x5.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(out, encoding="utf-8", xml_declaration=True)
    return source, out


def main():
    source, urdf = prepare_urdf()
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
            asset_url=URL,
            archive_sha256=ARCHIVE_SHA,
            source_urdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            derived_urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
            usd=usd,
            mount_separation_m=0.6,
            arm_limits=LIMITS,
            limitations=[
                "RoboTwin X5A geometry; conservative manufacturer X5 limits",
                "Finger drives are synchronized independent prismatic drives",
                "Camera housings are visual-only in upstream asset",
                "Asset pack has no separate asset license; cached locally, not vendored",
            ],
        )
        (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("ARX_ASSET", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
