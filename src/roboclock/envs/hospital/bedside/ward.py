"""Shared scene-02 ward; imported meshes retain source UVs/materials.

A task may place the ward per task through layout.info["ward_pose"]:
{"usd": <path>, "translate": [x, y, z], "yaw": degrees, "hide": [source object names]}. The yaw
turns the ward about the world origin (the robot side) before the translation, so tasks can face
real furniture. "hide" makes the named source objects (see the usdc's .json "parts") invisible,
for scenery that blocks the camera or crosses the arm's path. "move": {source object: [dx, dy, dz]}
shifts whole pieces of furniture (bed, cart, cabinet) by a world-frame offset in meters, for
furniture that would stand where the common dual-arm rig (hospital/rig.py) has to go.
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DEFAULT_TRANSLATE = (0.0, -0.85, 0.0)
FLOOR_SLAB = "Rectangle008"  # shared floor of both scene-02 conversions
# Scene-02's ultrasound cart (body, drawers, probe holders and its screen), parked on the bed's
# open side at the head end (ward x 1.87-2.55, y -2.60..-1.70).
ULTRASOUND_CART = [
    f"Obj24of_591669_{k}_{suffix}"
    for k, suffix in (
        (4, "386"), (5, "498"), (6, "853"), (7, "182"), (8, "786"), (9, "616"),
        (10, "520"), (11, "051"), (12, "779"), (13, "830"), (14, "113"),
    )
]


def ward_path(config, usd=None):
    path = Path(usd or config["scene"].get("bedside_ward_usd", "assets/hospital/ward_02_left.usdc"))
    return path if path.is_absolute() else REPO / path


# Movable furniture of the preview's left bay, as ward-frame xy boxes (m) holding every part of the
# piece; `furniture(key)` lists their source objects for a ward_pose "move".
FURNITURE = {
    "bed": ((0.74, -2.80), (1.82, -0.60), 1.10),  # frame, mattress, rails, head- and footboard
    "supply_trolley": ((0.24, -1.58), (0.72, -0.89), 1.00),  # two-tier trolley at the bed's side
}


def furniture(key, usd=None):
    """Source objects of one piece of ward furniture (ULTRASOUND_CART or a FURNITURE box)."""
    if key == "ultrasound_cart":
        return list(ULTRASOUND_CART)
    (x0, y0), (x1, y1), top = FURNITURE[key]
    path = ward_path({"scene": {}}, usd or "assets/hospital/ward_02_preview_left.usdc")
    names = {}
    for part in json.loads(path.with_suffix(".json").read_text())["parts"]:
        lo, hi = names.get(part["source_object"], (np.full(3, np.inf), np.full(3, -np.inf)))
        names[part["source_object"]] = (np.minimum(lo, part["min"]), np.maximum(hi, part["max"]))
    return sorted(
        name
        for name, (lo, hi) in names.items()
        if lo[0] >= x0 and hi[0] <= x1 and lo[1] >= y0 and hi[1] <= y1 and hi[2] <= top
    )


def move_offset(pose, source_object):
    """World-frame offset of a moved source object (zero when it is not moved)."""
    return np.asarray((pose or {}).get("move", {}).get(source_object, (0.0, 0.0, 0.0)), dtype=float)


def part_bounds(pose, parts, names):
    """World-frame xy bounds (lo, hi) of the named source objects under a ward pose, moves included."""
    yaw = np.radians(float((pose or {}).get("yaw", 0.0)))
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    shift = np.asarray((pose or {}).get("translate", DEFAULT_TRANSLATE), dtype=float)[:2]
    corners = []
    for part in parts:
        if part["source_object"] not in names:
            continue
        lo, hi = part["min"], part["max"]
        offset = move_offset(pose, part["source_object"])[:2]
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                corners.append(rot @ np.array([x, y]) + shift + offset)
    corners = np.asarray(corners)
    return corners.min(axis=0), corners.max(axis=0)


def build_ward(look, layout):
    from pxr import Gf, UsdGeom, UsdLux

    pose = (getattr(layout, "info", None) or {}).get("ward_pose") or {}
    path = ward_path(look.config, pose.get("usd"))
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing supplied ward conversion: {path}. "
            "Run python scripts/hospital/prepare_bedside_ward.py "
            "(or prepare_preview_left_ward.py for the preview's left bay)"
        )
    root = UsdGeom.Xform.Define(look.stage, "/World/BedsideWard")
    root.GetPrim().GetReferences().AddReference(str(path))
    root.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pose.get("translate", DEFAULT_TRANSLATE)]))
    root.AddRotateZOp().Set(float(pose.get("yaw", 0.0)))
    root.GetPrim().SetCustomDataByKey("reference", "assets/scenes/02/preview.jpg")
    hide = set(pose.get("hide", ()))
    meta = path.with_suffix(".json")
    parts = json.loads(meta.read_text())["parts"] if meta.is_file() else []
    # The ward's own floor slab has an untextured default material that renders near-black and
    # z-fights with the ground plane at z = 0. Hide it: the terrazzo ground plane is the visible
    # floor, level with the physics ground and the robot base for every ward pose.
    hide.add(FLOOR_SLAB)
    moves = pose.get("move", {})
    yaw = np.radians(float(pose.get("yaw", 0.0)))
    for part in parts:
        local = part["prim"].split("/", 2)[2]  # drop the referenced default prim (/Lab)
        if part["source_object"] in hide:
            prim = look.stage.GetPrimAtPath(f"/World/BedsideWard/{local}")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        elif part["source_object"] in moves:
            prim = look.stage.GetPrimAtPath(f"/World/BedsideWard/{local}")
            if prim.IsValid():
                # The offset is world-frame; the part lives in the ward frame (turned by yaw).
                dx, dy, dz = move_offset(pose, part["source_object"])
                c, s = np.cos(-yaw), np.sin(-yaw)
                xf = UsdGeom.Xformable(prim)
                order = xf.GetOrderedXformOps()
                op = xf.AddTranslateOp(opSuffix="rigMove")
                op.Set(Gf.Vec3d(c * dx - s * dy, s * dx + c * dy, dz))
                xf.SetXformOpOrder([op, *order])  # outermost: applied in the ward frame
    # The imported room has no light sources of its own: a broad window fill plus a grid of
    # ceiling panels over the work area keep the interior legible. Scaled by bedside_ward_light.
    scale = float(look.config["scene"].get("bedside_ward_light", 1.0))
    light = UsdLux.RectLight.Define(look.stage, "/World/Lighting/WardWindow")
    light.CreateWidthAttr(2.5)
    light.CreateHeightAttr(1.6)
    light.CreateIntensityAttr(250.0 * scale)
    light.CreateColorAttr(Gf.Vec3f(0.88, 0.94, 1.0))
    xf = UsdGeom.Xformable(light)
    xf.AddTranslateOp().Set(Gf.Vec3d(1.0, -4.15, 1.9))
    xf.AddRotateXOp().Set(90.0)
    if scale > 1.0:
        # A 3 x 3 grid of ceiling panels over the robot's work area, like a ward's recessed lights.
        grid = [(x, y) for x in (-1.0, 0.5, 2.0) for y in (-1.5, 0.0, 1.5)]
        for i, (x, y) in enumerate(grid):
            panel = UsdLux.RectLight.Define(look.stage, f"/World/Lighting/WardCeiling_{i}")
            panel.CreateWidthAttr(1.2)
            panel.CreateHeightAttr(0.6)
            panel.CreateIntensityAttr(60.0 * scale)
            panel.CreateEnableColorTemperatureAttr(True)
            panel.CreateColorTemperatureAttr(5000.0)
            UsdGeom.Xformable(panel).AddTranslateOp().Set(Gf.Vec3d(x, y, 2.55))
        hdr = look.stage.GetPrimAtPath("/World/Lighting/HDR")
        if hdr.IsValid():
            dome = UsdLux.DomeLight(hdr)
            dome.GetIntensityAttr().Set(float(dome.GetIntensityAttr().Get()) * scale)
