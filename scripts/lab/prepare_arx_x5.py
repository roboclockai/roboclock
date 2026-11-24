"""Convert the bundled official X5A URDF/meshes into a fixed-base Isaac asset."""

import json
import xml.etree.ElementTree as ET

import yaml

from roboclock.envs.lab.robot import (
    ASSET_DIR,
    EFFORTS,
    HOME,
    JOINT_HIGH,
    JOINT_LOW,
    VELOCITIES,
)


def prepare_description():
    root = ET.parse(ASSET_DIR / "source.urdf").getroot()
    root.set("name", "arx_x5")
    for mesh in root.iter("mesh"):
        mesh.set("filename", mesh.get("filename").replace("package://X5A/", "./"))
    for i, joint in enumerate(root.findall("joint")):
        for key, values in (
            ("lower", JOINT_LOW),
            ("upper", JOINT_HIGH),
            ("effort", EFFORTS),
            ("velocity", VELOCITIES),
        ):
            joint.find("limit").set(key, str(float(values[i])))
    ET.SubElement(root, "link", name="grasp_center")
    joint = ET.SubElement(root, "joint", name="tool_frame", type="fixed")
    ET.SubElement(joint, "parent", link="link6")
    ET.SubElement(joint, "child", link="grasp_center")
    ET.SubElement(joint, "origin", xyz="0.145 0 0", rpy="0 1.5707963267948966 0")
    ET.indent(root)
    ET.ElementTree(root).write(ASSET_DIR / "arx_x5.urdf", encoding="utf-8", xml_declaration=True)
    description = {
        "api_version": 1.0,
        "cspace": [f"joint{i}" for i in range(1, 7)],
        "root_link": "base_link",
        "default_q": HOME[:6].tolist(),
        "acceleration_limits": [15.0] * 6,
        "jerk_limits": [1000.0] * 6,
        "cspace_to_urdf_rules": [
            {"name": f"joint{i}", "rule": "fixed", "value": 0.04} for i in (7, 8)
        ],
        "collision_spheres": [],
    }
    (ASSET_DIR / "robot_description.yaml").write_text(yaml.safe_dump(description))


def normalize_links(stage):
    """Keep stable link paths across Isaac importer versions, preserving world poses."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    root = str(stage.GetDefaultPrim().GetPath())
    cache = UsdGeom.XformCache()
    bodies = [
        (str(p.GetPath()), p.GetName(), cache.GetLocalToWorldTransform(p))
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    for old, name, matrix in sorted(bodies, key=lambda item: item[0].count("/"), reverse=True):
        target = root + "/" + name
        if old != target:
            editor = Usd.NamespaceEditor(stage)
            editor.MovePrimAtPath(old, target)
            if not editor.ApplyEdits():
                raise RuntimeError(f"Cannot normalize robot link {old}")
        transform = UsdGeom.Xformable(stage.GetPrimAtPath(target))
        transform.ClearXformOpOrder()
        transform.AddTransformOp().Set(matrix)
        transform.SetResetXformStack(False)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    prim = stage.GetPrimAtPath(root)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
    api.CreateEnabledSelfCollisionsAttr(True)
    api.CreateSolverPositionIterationCountAttr(16)
    api.CreateSolverVelocityIterationCountAttr(4)
    # The two sliding rack housings overlap in the visual-mesh convex approximation.
    # Exclude this internal pair; both fingers still collide with grasped objects.
    UsdPhysics.FilteredPairsAPI.Apply(
        stage.GetPrimAtPath(root + "/link7")
    ).CreateFilteredPairsRel().SetTargets([root + "/link8"])


def main():
    prepare_description()
    from roboclock.simulators.isaac.runtime import IsaacRuntime, simulator_version

    with IsaacRuntime():
        import omni.kit.commands
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Usd

        enable_extension("isaacsim.asset.importer.urdf")
        destination = ASSET_DIR / "arx_x5.usd"
        if simulator_version().startswith("6."):
            from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

            config = URDFImporterConfig(
                urdf_path=str(ASSET_DIR / "arx_x5.urdf"),
                usd_path=str(ASSET_DIR / "converted"),
                fix_base=True,
                merge_fixed_joints=True,
                allow_self_collision=True,
                collision_from_visuals=True,
                collision_type="Convex Decomposition",
                joint_drive_type="force",
                joint_target_type="position",
                override_joint_stiffness=1500.0,
                override_joint_damping=60.0,
                run_asset_transformer=False,
            )
            result = URDFImporter(config).import_urdf()
            stage = Usd.Stage.Open(result)
            normalize_links(stage)
            stage.Flatten().Export(str(destination))
            print(
                json.dumps(
                    {
                        "asset": str(destination),
                        "prims": [
                            str(p.GetPath())
                            for p in stage.Traverse()
                            if p.GetTypeName()
                            in ("Xform", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
                        ],
                    }
                ),
                flush=True,
            )
            return
        ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not ok:
            raise RuntimeError("URDF importer unavailable")
        config.fix_base = True
        config.merge_fixed_joints = True
        config.self_collision = True
        config.import_inertia_tensor = True
        config.convex_decomp = True
        config.default_drive_strength = 1500.0
        config.default_position_drive_damping = 60.0
        ok, result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(ASSET_DIR / "arx_x5.urdf"),
            import_config=config,
            dest_path=str(destination),
        )
        if not ok or not destination.exists():
            raise RuntimeError(f"ARX import failed: {result}")
        stage = Usd.Stage.Open(str(destination))
        normalize_links(stage)
        stage.GetRootLayer().Save()
        print(
            json.dumps({"asset": str(destination), "root": str(stage.GetDefaultPrim().GetPath())})
        )


if __name__ == "__main__":
    main()
