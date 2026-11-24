"""Load NVIDIA hospital props one at a time: print their real size and render a picture of each.

Writes results/props/<name>.png, results/props/sheet.png (a contact sheet) and prints a PROP line per
asset with its bounding box in meters. Run on the Isaac Sim machine:

    OMNI_KIT_ACCEPT_EULA=YES python scripts/preview_props.py [name ...]
"""

import sys
from pathlib import Path

from roboclock.simulators.isaac.runtime import IsaacRuntime

# Start Isaac Sim exactly as the benchmark does (same renderer flags), since that is known to return frames.
runtime = IsaacRuntime(headless=True)

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.rotations import gf_quat_to_np_array  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

from roboclock.envs.hospital.config import load_config  # noqa: E402
from roboclock.simulators.isaac.rendering import configure_renderer  # noqa: E402

configure_renderer(load_config())

ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Hospital/Props"
DEFAULT = (
    "SM_Gurney_01a SM_Gurney_01b SM_HospitalBed_01b SM_HospitalBed_02d SM_BedSideTable_01b SM_Cart_01a "
    "SM_GasCart_01b SM_GasCart_01c SM_SupplyCart_01e SM_SupplyCart_02a SM_SupplyCart_03a "
    "SM_MedicalCabinet_01a SM_MedShelf_01d SM_SupplyCabinet_01c SM_MedicalBag_01a SM_FirstAidKit_01a "
    "SM_PillBottleSet_01a SM_PillBottle_01t SM_UtilitiesTray_01e SM_Container_01a SM_Computer_02a "
    "SM_Xraylightbox_01a Pharmacy_Low SM_ShelfSet_01a SM_Shelf_06f SM_Chart_Rack_01c SM_Washbasin "
    "SM_WheelChair_01a SM_SideTable_02a SM_Scale_01a SM_ReceptionDesk_01a SM_SoapDispenser_01a "
    "SM_Toweldispenser_01a SM_TrashCan SM_MedicalBox_01g SM_Intercom_01a"
).split()
names = sys.argv[1:] or DEFAULT
out = Path("results/props")
out.mkdir(parents=True, exist_ok=True)

create_new_stage()
world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.Xform.Define(stage, "/World")
dome = UsdLux.DomeLight.Define(stage, "/World/Light")
dome.CreateIntensityAttr(900.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(2500.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 35))
camera = Camera(prim_path="/World/PreviewCam", resolution=(960, 720))
world.reset()
camera.initialize()  # after the reset, as in the benchmark's own scene setup
camera.add_distance_to_image_plane_to_frame()
for _ in range(20):
    world.render()
print(f"STAGE meters_per_unit={UsdGeom.GetStageMetersPerUnit(stage)}")

images = []
for name in names:
    path = "/World/Prop"
    try:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
        add_reference_to_stage(f"{ROOT}/{name}.usd", path)
        for _ in range(6):
            world.render()
        prim = stage.GetPrimAtPath(path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
        size, center = hi - lo, (hi + lo) / 2
        print(f"PROP {name} size_xyz={np.round(size, 3).tolist()} min={np.round(lo, 3).tolist()}")
        reach = float(np.linalg.norm(size)) * 1.25 + 0.2
        eye = center + np.array([0.9, -1.0, 0.55]) / np.linalg.norm([0.9, -1.0, 0.55]) * reach
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*center.tolist()), Gf.Vec3d(0, 0, 1)
        )
        camera.set_world_pose(
            eye, gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat()), camera_axes="usd"
        )
        camera.set_focal_length(0.024)
        camera.set_clipping_range(0.02, 80.0)
        rgba = None
        for _ in range(80):  # textures stream in from the asset server; wait for a real frame
            world.render()
            rgba = camera.get_rgba()
        if rgba is not None and rgba.size and rgba[..., :3].max() == 0:
            rgba = None  # a black frame means the renderer was not ready
        if rgba is not None and rgba.size:
            image = Image.fromarray(rgba[..., :3])
            image.save(out / f"{name}.png")
            images.append((name, image.resize((320, 240))))
            print(f"IMAGE {name} saved")
        else:
            print(f"IMAGE {name} EMPTY (the camera returned no frame)")
    except Exception as error:
        print(f"PROP {name} FAILED {type(error).__name__}: {error}")

if images:
    cols = 4
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 260), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (name, image) in enumerate(images):
        x, y = (i % cols) * 320, (i // cols) * 260
        sheet.paste(image, (x, y + 20))
        draw.text((x + 4, y + 4), name, fill=(255, 255, 0))
    sheet.save(out / "sheet.png")
print("PREVIEW DONE")
runtime.close()
