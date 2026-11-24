"""Report which Isaac Sim Python APIs this repo uses can be imported (run on the Isaac Sim machine).

Usage: OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_api_probe.py
"""

import importlib

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

# (module, attribute) for every Isaac Sim / USD API the simulator code imports.
CHECKS = [
    ("isaacsim.core.api", "World"),
    ("isaacsim.core.api.objects", "FixedCuboid"),
    ("isaacsim.core.api.objects", "VisualCuboid"),
    ("isaacsim.core.api.materials", "OmniPBR"),
    ("isaacsim.core.api.materials", "PhysicsMaterial"),
    ("isaacsim.core.api.materials", "PreviewSurface"),
    ("isaacsim.core.api.materials.visual_material", "VisualMaterial"),
    ("isaacsim.core.prims", "RigidPrim"),
    ("isaacsim.core.prims", "SingleRigidPrim"),
    ("isaacsim.core.utils.stage", "create_new_stage"),
    ("isaacsim.core.utils.stage", "get_current_stage"),
    ("isaacsim.core.utils.rotations", "euler_angles_to_quat"),
    ("isaacsim.sensors.camera", "Camera"),
    ("omni.replicator.core", None),
    ("omni.physics.tensors", None),
    ("omni.kit.app", None),
    ("pxr", "PhysxSchema"),
    ("pxr", "UsdPhysics"),
    ("pxr", "UsdLux"),
]

# Isaac Sim 6 moved the manipulator examples to a deprecated extension that is not enabled by
# default (the runtime still enables it to keep the recorded extension set). Try switching it on, adding the deprecated extension folder to the search path if needed.
import os  # noqa: E402

import isaacsim  # noqa: E402
import omni.kit.app  # noqa: E402

manager = omni.kit.app.get_app().get_extension_manager()
EXAMPLES = "isaacsim.robot.manipulators.examples"


def try_enable():
    try:
        return manager.set_extension_enabled_immediate(EXAMPLES, True)
    except Exception as error:
        return f"{type(error).__name__}: {error}"


print(f"ENABLE   {EXAMPLES}: {try_enable()}")
if not any("manipulators.examples" in str(m) for m in __import__("sys").modules):
    deprecated = os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated")
    try:
        import omni.ext

        manager.add_path(deprecated, omni.ext.ExtensionPathType.COLLECTION)
        print(f"ADDPATH  {deprecated}")
    except Exception as error:
        print(f"ADDPATH  failed: {type(error).__name__}: {error}")
    print(f"ENABLE   {EXAMPLES} (after path): {try_enable()}")

failures = 0
for module, attribute in CHECKS:
    try:
        loaded = importlib.import_module(module)
        if attribute is not None:
            getattr(loaded, attribute)
        print(f"OK       {module}" + (f".{attribute}" if attribute else ""))
    except Exception as error:  # report every failure, not just the first
        failures += 1
        print(
            f"MISSING  {module}"
            + (f".{attribute}" if attribute else "")
            + f"  -> {type(error).__name__}: {error}"
        )

print(f"PROBE DONE: {failures} missing of {len(CHECKS)}")
app.close()
