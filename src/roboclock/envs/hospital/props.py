"""Real furniture and equipment from NVIDIA's Isaac hospital asset pack (CC BY 4.0), as visual props.

The USD files are referenced from NVIDIA's asset server; they carry their own meshes and materials and
have no physics here. Sizes and origins were measured by `scripts/preview_props.py`, so a prop can be
placed by where its footprint center should stand, at its true size.

Attribution: NVIDIA Isaac Sim hospital environment assets, Creative Commons Attribution 4.0.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
    "Environments/Hospital/Props"
)


@dataclass(frozen=True)
class PropSpec:
    size: tuple  # x, y, z in meters
    min: tuple  # the bounding box minimum in the prop's own frame (its origin is not always its center)


# name -> measured size and bounding-box minimum, in meters
CATALOG = {
    "SM_Gurney_01a": PropSpec((2.179, 0.772, 1.022), (-1.08, -0.386, 0.003)),
    "SM_Gurney_01b": PropSpec((2.179, 0.772, 1.041), (-1.08, -0.386, 0.003)),
    "SM_HospitalBed_01b": PropSpec((1.107, 2.554, 1.302), (-0.553, -1.213, 0.0)),
    "SM_HospitalBed_02d": PropSpec((1.092, 2.214, 0.891), (-0.546, -1.107, 0.0)),
    "SM_BedSideTable_01b": PropSpec((0.53, 0.5, 0.9), (-0.28, -0.25, 0.0)),
    "SM_Cart_01a": PropSpec((1.054, 0.626, 1.59), (-0.542, -0.313, 0.0)),
    "SM_GasCart_01b": PropSpec((0.305, 0.317, 1.093), (-0.153, -0.134, -0.003)),
    "SM_GasCart_01c": PropSpec((0.176, 0.176, 0.923), (-0.088, -0.071, -0.001)),
    "SM_SupplyCart_01e": PropSpec((1.102, 0.527, 1.113), (-0.462, -0.262, 0.0)),
    "SM_SupplyCart_02a": PropSpec((0.842, 0.456, 0.782), (-0.421, -0.228, 0.0)),
    "SM_SupplyCart_03a": PropSpec((0.596, 1.113, 1.034), (-0.298, -0.556, 0.0)),
    "SM_MedicalCabinet_01a": PropSpec((0.872, 0.655, 1.968), (-0.02, -0.024, 0.0)),
    "SM_MedShelf_01d": PropSpec((0.91, 0.805, 2.012), (-0.457, -0.405, 0.0)),
    "SM_SupplyCabinet_01c": PropSpec((0.733, 1.474, 2.2), (-0.367, -0.737, 0.0)),
    "SM_MedicalBag_01a": PropSpec((0.816, 0.552, 0.421), (-0.414, -0.295, 0.001)),
    "SM_FirstAidKit_01a": PropSpec((0.364, 0.118, 0.336), (-0.186, -0.059, 0.001)),
    "SM_PillBottleSet_01a": PropSpec((0.125, 0.216, 0.103), (-0.063, -0.108, 0.0)),
    "SM_UtilitiesTray_01e": PropSpec((0.532, 0.306, 0.086), (-0.263, -0.157, -0.001)),
    "SM_Container_01a": PropSpec((0.108, 0.108, 0.201), (-0.054, -0.054, -0.002)),
    "SM_Computer_02a": PropSpec((0.9, 0.623, 0.618), (-0.413, -0.451, 0.0)),
    "Pharmacy_Low": PropSpec((1.111, 0.715, 1.063), (-1.111, -0.459, 0.0)),
    "SM_ShelfSet_01a": PropSpec((1.723, 0.56, 2.636), (-0.861, -0.308, 0.001)),
    "SM_Shelf_06f": PropSpec((0.964, 0.878, 1.747), (-0.526, -0.431, 0.0)),
    "SM_Chart_Rack_01c": PropSpec((1.248, 0.431, 1.637), (-0.624, -0.216, 0.0)),
    "SM_Washbasin": PropSpec((0.55, 1.0, 0.44), (0.0, -0.5, 0.0)),
    "SM_WheelChair_01a": PropSpec((1.105, 0.825, 1.159), (-0.553, -0.413, 0.0)),
    "SM_SideTable_02a": PropSpec((0.918, 0.421, 0.812), (-0.459, -0.211, 0.0)),
    "SM_Scale_01a": PropSpec((0.462, 0.61, 1.551), (-0.225, -0.305, 0.0)),
    "SM_SoapDispenser_01a": PropSpec((0.131, 0.126, 0.257), (0.0, -0.063, -0.129)),
    "SM_TrashCan": PropSpec((0.3, 0.3, 0.715), (-0.15, -0.15, 0.0)),
    "SM_MedicalBox_01g": PropSpec((0.527, 0.337, 0.653), (-0.527, -0.332, -0.004)),
}


def placement(name, center_xy, yaw=0.0, floor_z=0.0, scale=1.0):
    """Translation and rotation that put the prop's footprint center at `center_xy`, resting on `floor_z`."""
    spec = CATALOG[name]
    own_center = np.array(
        [spec.min[0] + spec.size[0] / 2, spec.min[1] + spec.size[1] / 2], dtype=float
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    xy = np.asarray(center_xy, dtype=float) - rotation @ (scale * own_center)
    return np.array([xy[0], xy[1], floor_z - scale * spec.min[2]]), yaw


def add_prop(
    stage, name, path, center_xy, yaw=0.0, floor_z=0.0, scale=1.0, base=BASE, suffix=".usd"
):
    """Reference a prop into the stage, centered at `center_xy`, rotated by `yaw` about the vertical.

    The placement lives on an outer node and the asset is referenced on a child, so the asset's own
    root transform (many carry a unit scale) never collides with ours.
    """
    from pxr import Gf, UsdGeom

    translation, yaw = placement(name, center_xy, yaw, floor_z, scale)
    outer = UsdGeom.Xform.Define(stage, path)
    outer.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    outer.AddRotateZOp().Set(float(np.degrees(yaw)))
    outer.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    asset = UsdGeom.Xform.Define(stage, f"{path}/Asset")
    asset.GetPrim().GetReferences().AddReference(f"{base}/{name}{suffix}")
    return outer


MODELS = Path(__file__).parent / "assets" / "models"


def model_info(name):
    """The credits entry of a converted model (size, license, optional top_z), or None when it is not installed."""
    credits = MODELS / "credits.json"
    if not (MODELS / name / f"{name}.usd").exists() or not credits.exists():
        return None
    return json.loads(credits.read_text()).get(name)


def model_size(name):
    """Real size (x, y, z meters) of a converted model, or None when it is not installed."""
    credits = MODELS / "credits.json"
    if not (MODELS / name / f"{name}.usd").exists() or not credits.exists():
        return None
    entry = json.loads(credits.read_text()).get(name)
    return tuple(entry["size_xyz_m"]) if entry else None


def add_model(stage, name, path, translate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), yaw=0.0):
    """Reference a converted GLB model (z up, centered on xy, resting on z = 0; see scripts/hospital/import_glb.py)."""
    from pxr import Gf, UsdGeom

    node = UsdGeom.Xform.Define(stage, path)
    node.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translate]))
    node.AddRotateZOp().Set(float(np.degrees(yaw)))
    node.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))
    node.GetPrim().GetReferences().AddReference(str(MODELS / name / f"{name}.usd"))
    return node
