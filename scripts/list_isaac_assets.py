"""List NVIDIA's Isaac asset library for hospital-relevant content (run on the Isaac Sim machine).

Prints ROOT (where assets are served from), FOLDER lines for the top levels of the environment and prop
folders, and MATCH lines for any path containing a hospital or emergency keyword.

Usage: OMNI_KIT_ACCEPT_EULA=YES python scripts/list_isaac_assets.py
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.client  # noqa: E402

KEYWORDS = (
    "hospital",
    "medical",
    "surgical",
    "surgery",
    "operating",
    "bed",
    "stretcher",
    "gurney",
    "cart",
    "tray",
    "aed",
    "defib",
    "extinguisher",
    "fire",
    "monitor",
    "iv_",
    "infusion",
    "syringe",
    "mannequin",
    "manikin",
    "patient",
    "clinic",
    "lab",
    "pharmacy",
    "cabinet",
    "shelf",
)


def assets_root():
    try:
        from isaacsim.storage.native import get_assets_root_path

        return get_assets_root_path()
    except Exception as error:  # the API location differs between versions
        print(f"ROOT lookup failed: {type(error).__name__}: {error}")
        try:
            import carb.settings

            return carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
        except Exception:
            return None


def list_folder(url):
    result, entries = omni.client.list(url)
    if result != omni.client.Result.OK:
        return []
    return [
        (entry.relative_path, bool(entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN))
        for entry in entries
    ]


root = assets_root()
print(f"ROOT {root}")
if root:
    queue = [
        (f"{root}/{top}", 0)
        for top in ("Isaac/Environments", "Isaac/Props", "Isaac/People", "NVIDIA/Assets")
    ]
    seen = 0
    while queue and seen < 4000:
        url, depth = queue.pop(0)
        for name, is_folder in list_folder(url):
            child = f"{url}/{name.rstrip('/')}"
            seen += 1
            if depth < 1 and is_folder:
                print(f"FOLDER {child}")
            if any(word in name.lower() for word in KEYWORDS):
                print(f"MATCH  {child}")
            if is_folder and depth < 3:
                queue.append((child, depth + 1))
print("DONE")
app.close()
