"""B3 hazardous drug spill: layout, scene dressing, the break and the puddle, and the oracle's jobs.

Everything the spill task adds to the shared hospital code lives here; layout.py, appearance.py,
backend.py and policies/jobs.py each call one function of this module for the task named
``hazardous_drug_spill`` (the archived fire-blanket task keeps the ID B3 in the legacy profile).

Scene, in the robot frame before the seed shift (x away from the rig, y to its left): a two-tier
counter rack at the back of the bench with the cytotoxic drug bottle on the front of its lower
shelf; in front of it the open counter where the bottle lands; the spill kit tray with the folded absorbent pad on the arm's side; the yellow cytotoxic
sharps bin near the rig; the CONTAIN and CLEAN-UP dials beside the rack.

Downloaded models (assets/models/credits.json, scripts/hospital/import_spill_assets.py): the
bottle, both glass pieces and the splinters, the sharps bin with its lid, the rack. Built here: the
folded towel pad and its spread-out sheet, the puddle, the kit tray and the signs.
"""

import numpy as np

SCENE = {
    "zone_xy": (0.596, -0.02),  # where the bottle lands (measured)
    "zone_half": 0.11,  # kept clear of clutter around the landing point
    "rack_xy": (0.79, -0.02),  # rack centre; it is 0.426 m wide (along y) and 0.217 m deep
    "bottle_xy": (0.707, -0.02),  # at the front edge of the lower shelf
    "kit_xy": (0.40, -0.285),
    "bin_xy": (0.36, 0.22),
    "contain_dial_xy": (0.80, 0.30),
    "dial_xy": (0.80, 0.46),
}
RACK_SIZE = (0.217, 0.426)  # depth (x), width (y) as placed
SHELF_TOPS = (0.149, 0.3655)  # above the counter (credits.json: counter_rack.shelf_tops_m)
SHELF_THICKNESS = 0.011
BIN_HALF = (0.083, 0.058)  # inner half extents of the sharps bin
BIN_WALL = 0.146
PAD_SHEET = 0.20  # side of the spread-out absorbent towel
SHEET_THICKNESS = 0.020  # a towel, not a tissue
TOWEL = (0.80, 0.88, 0.95)  # pale blue terry
# Where the two glass pieces come to rest, relative to the break: clear of the pad's sheet.
# Both stay 0.36-0.45 m from arm 0's base: nearer than ~0.33 m its top-down grasp at counter
# height does not solve, beyond 0.5 m it does not reach.
SHARD_OFFSETS = {"shard_0": (-0.05, -0.150), "shard_1": (-0.10, 0.125)}
TIP_VELOCITY = (-0.55, 0.0, 0.0)  # m/s: the bottle slides off the shelf toward the robot
TIP_SPIN = (0.0, -4.0, 0.0)  # rad/s: and topples forward as it goes
AMBER = (0.59, 0.27, 0.08)
LIQUID = (0.72, 0.05, 0.08)  # doxorubicin-red
PURPLE = (0.42, 0.16, 0.55)  # the cytotoxic hazard colour


def is_spill(task) -> bool:
    return getattr(task, "name", "") == "hazardous_drug_spill"


# --- layout -------------------------------------------------------------------------------------


def build_layout(b, config):
    """Declare the spill scene on the layout builder (called from layout._build)."""
    z = b.z
    # Half the usual depth jitter: the break must stay within arm 0's top-down reach (0.5 m).
    b.shift = b.shift * np.array([0.5, 0.4])
    zx, zy = SCENE["zone_xy"]
    rx, ry = SCENE["rack_xy"]
    for k, top in enumerate(SHELF_TOPS):
        b.fixture(
            f"hd_shelf_{k}",
            (rx, ry),
            (RACK_SIZE[0], RACK_SIZE[1], SHELF_THICKNESS),
            "steel",
            z=z + top - SHELF_THICKNESS,
            visible=False,
        )
    b.zone("spill", (zx, zy), half=(SCENE["zone_half"],) * 2)
    b.item("bottle", "hd_bottle", SCENE["bottle_xy"], z + SHELF_TOPS[0], color=AMBER, yaw=0.0)
    b.held.add("bottle")  # stands on the shelf until it tips
    kx, ky = SCENE["kit_xy"]
    b.fixture("spill_kit", (kx, ky), (0.19, 0.115, 0.004), "cabinet_white", visible=False)
    b.item("pad", "spill_pad", (kx, ky), z + 0.004, color=(0.86, 0.90, 0.94), yaw=0.0)
    for k, (name, asset) in enumerate((("shard_0", "hd_shard_neck"), ("shard_1", "hd_shard_base"))):
        dx, dy = SHARD_OFFSETS[name]
        yaw = float(b.rng.uniform(-0.25, 0.25))
        b.item(name, asset, (zx + dx, zy + dy), z, color=AMBER, yaw=yaw)
        b.held.add(name)  # hidden, without collision, until the bottle breaks
    b.target("hd_bin", "bin", SCENE["bin_xy"], BIN_HALF, BIN_WALL)
    b.indicator("contain", SCENE["contain_dial_xy"], "CONTAIN")
    b.indicator("indicator", SCENE["dial_xy"], "CLEAN-UP")
    b.info["face_dials"] = True
    b.info["spill"] = {
        "center": [*map(float, b.xy(zx, zy))],
        "cover_radius": float(config["tasks"]["B3"]["cover_radius"]),
        "tip_velocity": list(TIP_VELOCITY),
        "tip_spin": list(TIP_SPIN),
        "rack": [*map(float, b.xy(rx, ry))],
        "kit": [*map(float, b.xy(kx, ky))],
        "bin": [*map(float, b.xy(*SCENE["bin_xy"]))],
        "shard_offsets": {name: list(offset) for name, offset in SHARD_OFFSETS.items()},
    }


# --- static dressing (before the objects exist) ----------------------------------------------------


def towel(look, root, size, center_z, ribs, plies=1):
    """Terry towelling under `root` (its frame): `plies` thick soft-edged layers filling `size`
    around z = center_z, rows of pile ribs on the top face and a woven hem stripe at both ends."""
    sx, sy, sz = size
    cloth = look.material("hd_towel", color=TOWEL, roughness=1.0)
    pile = look.material("hd_towel_pile", color=tuple(0.93 * c for c in TOWEL), roughness=1.0)
    hem = look.material("hd_towel_hem", color=(0.38, 0.56, 0.80), roughness=0.9)
    ply = sz / plies
    radius = min(0.45 * ply, 0.009)
    for k in range(plies):
        inset = 0.006 * (k % 2)  # the folds do not line up exactly
        look.rounded(
            f"{root}/Ply_{k}",
            [0, 0, center_z - sz / 2 + ply * (k + 0.5)],
            [sx - inset, sy - inset, ply * 0.97],
            cloth,
            radius,
        )
    top = center_z + sz / 2
    span = sx - 0.036  # between the hems
    for k in range(ribs):
        x = -span / 2 + span * (k + 0.5) / ribs
        look.rounded(
            f"{root}/Rib_{k}", [x, 0, top], [span / ribs * 0.55, sy - 0.012, 0.003], pile, 0.0014
        )
    for k, x in enumerate((-sx / 2 + 0.011, sx / 2 - 0.011)):
        look.rounded(f"{root}/Hem_{k}", [x, 0, top], [0.010, sy - 0.008, 0.0016], hem, 0.0007)


def dress(look, layout):
    """Rack, kit tray, signs, and the hidden puddle / splinters / spread pad groups."""
    from pxr import Gf, UsdGeom

    from roboclock.envs.hospital.props import add_model, add_prop, model_size

    spill = layout.info["spill"]
    z = float(layout.table_height)
    cx, cy = spill["center"]
    rx, ry = spill["rack"]
    stage = look.stage
    UsdGeom.Xform.Define(stage, "/World/Spill")
    # The rack: the model is 0.426 m wide along its own x; turned so that width runs along y.
    if model_size("counter_rack") is not None:
        add_model(stage, "counter_rack", "/World/Spill/Rack", (rx, ry, z), yaw=np.pi / 2)
    else:
        for k, top in enumerate(SHELF_TOPS):
            look.cube(
                f"/World/Spill/Shelf_{k}",
                [rx, ry, z + top - SHELF_THICKNESS / 2],
                [RACK_SIZE[0], RACK_SIZE[1], SHELF_THICKNESS],
                "stainless",
            )
    # Other stock on the rack: capped vials (the repo's vial model) and NVIDIA's pill-bottle set.
    if model_size("vial") is not None:
        for k, dy in enumerate((-0.15, -0.105, 0.10, 0.15)):
            add_model(
                stage,
                "vial",
                f"/World/Spill/Stock_{k}",
                (rx - 0.01, ry + dy, z + SHELF_TOPS[0]),
            )
        for k, dy in enumerate((-0.12, -0.06, 0.0)):
            add_model(
                stage,
                "vial",
                f"/World/Spill/StockTop_{k}",
                (rx, ry + dy, z + SHELF_TOPS[1]),
            )
    add_prop(
        stage,
        "SM_PillBottleSet_01a",
        "/World/Spill/PillBottles",
        (rx + 0.01, ry + 0.115),
        yaw=np.pi / 2,
        floor_z=z + SHELF_TOPS[1],
    )
    sign = look.label(
        "/World/Spill/RackSign",
        "CYTOTOXIC|DRUGS",
        [rx - RACK_SIZE[0] / 2 - 0.004, ry + 0.12, z + SHELF_TOPS[0] - 0.035],
        [0.13, 0.05],
        PURPLE,
    )
    if sign is not None:
        sign.AddRotateZOp().Set(-90.0)  # read from the robot's side
    # Spill kit: an open purple tray with the folded pad standing in it, and its card.
    kx, ky = spill["kit"]
    purple = look.material("hd_purple", color=PURPLE, roughness=0.5)
    look.rounded("/World/Spill/KitTray", [kx, ky, z + 0.002], [0.19, 0.115, 0.004], purple, 0.001)
    for k, (dx, dy, sx, sy) in enumerate(
        ((0.0, -0.0545, 0.19, 0.006), (0.0, 0.0545, 0.19, 0.006), (-0.092, 0.0, 0.006, 0.103), (0.092, 0.0, 0.006, 0.103))
    ):
        look.rounded(
            f"/World/Spill/KitRim_{k}", [kx + dx, ky + dy, z + 0.008], [sx, sy, 0.012], purple, 0.002
        )
    look.label(
        "/World/Spill/KitSign", "SPILL KIT", [kx, ky - 0.0585, z + 0.012], [0.10, 0.018], PURPLE
    )
    # The bin's lid lies beside it; the bin itself replaces the target's generic walls in `author`.
    bx, by = spill["bin"]
    if model_size("cytotoxic_bin_lid") is not None:
        add_model(stage, "cytotoxic_bin_lid", "/World/Spill/BinLid", (bx - 0.005, by + 0.165, z))
    # Hidden until the break: the puddle (an irregular pool: one disc and three lobes, scaled
    # together), the splinters lying in it, and the spread-out absorbent sheet.
    liquid = look.material("hd_liquid", color=LIQUID, roughness=0.08)
    puddle = UsdGeom.Xform.Define(stage, "/World/Spill/Puddle")
    puddle.AddTranslateOp().Set(Gf.Vec3d(cx, cy, z + 0.0012))
    puddle.AddScaleOp().Set(Gf.Vec3f(0.035, 0.035, 1.0))
    for k, (dx, dy, r) in enumerate(((0.0, 0.0, 0.92), (0.50, 0.22, 0.50), (-0.38, 0.45, 0.52), (-0.10, -0.55, 0.45))):
        look.cylinder(f"/World/Spill/Puddle/Lobe_{k}", [dx, dy, 0.0], r, 0.0016, liquid)
    UsdGeom.Xform.Define(stage, "/World/Spill/Splinters").AddTranslateOp().Set(
        Gf.Vec3d(cx, cy, z + 0.002)
    )
    if model_size("hd_splinters") is not None:
        add_model(stage, "hd_splinters", "/World/Spill/Splinters/Model")
    sheet = UsdGeom.Xform.Define(stage, "/World/Spill/Sheet")
    sheet.AddTranslateOp().Set(Gf.Vec3d(cx, cy, z))
    # The spread towel: a thick terry slab with soft edges, pile ribs and a woven hem at two ends.
    towel(look, "/World/Spill/Sheet", (PAD_SHEET, PAD_SHEET, SHEET_THICKNESS), SHEET_THICKNESS / 2, ribs=22)
    stain_material = look.material("hd_pad_stain", color=(0.86, 0.42, 0.46), roughness=0.9)
    stain = look.cylinder(
        "/World/Spill/Sheet/Stain", [0, 0, SHEET_THICKNESS + 0.0016], 1.0, 0.0006, stain_material
    )
    stain.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 1.0))
    for name in ("Puddle", "Splinters", "Sheet"):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/Spill/{name}")).MakeInvisible()


# --- objects (after the backend built them) ---------------------------------------------------------


def _swap_visual(backend, name, model, drop):
    """Replace an object's generic box / cup visual by a converted model resting on its base."""
    from roboclock.envs.hospital.props import add_model, model_size

    path = f"/World/Objects/{name}"
    if model_size(model) is None:
        return False
    for child in drop:
        if backend.stage.GetPrimAtPath(f"{path}/{child}").IsValid():
            backend.stage.RemovePrim(f"{path}/{child}")
    shape = backend.layout.shapes[name]
    add_model(backend.stage, model, f"{path}/Model", (0.0, 0.0, -shape.rest_height))
    return True


def author(backend):
    """Give the spill objects their real looks, hide the pieces, and dress the bin."""
    from pxr import Usd, UsdGeom

    from roboclock.envs.hospital.props import add_model, model_size

    look, layout, stage = backend.appearance, backend.layout, backend.stage
    z = float(layout.table_height)
    # Bottle: amber glass body with its contents, a crimp cap and the purple cytotoxic band.
    if _swap_visual(backend, "bottle", "hd_bottle", ("Wall", "Rim")):
        path = "/World/Objects/bottle"
        shape = layout.shapes["bottle"]
        top = model_size("hd_bottle")[2] - shape.rest_height
        look.cylinder_child(path, "Cap", 0.0125, 0.012, top + 0.004, look.material("aluminum"))
        look.cylinder_child(
            path, "Band", 0.0194, 0.030, -0.008, look.material("hd_purple", color=PURPLE, roughness=0.5)
        )
        look.cylinder_child(
            path, "BandStripe", 0.0196, 0.006, -0.008, look.material("hd_band_yellow", color=(0.95, 0.80, 0.10), roughness=0.5)
        )
    for name, model in (("shard_0", "hd_shard_neck"), ("shard_1", "hd_shard_base")):
        _swap_visual(backend, name, model, ("Shell", "Band"))
        backend.objects[name].set_visibility(False)
        backend._set_collision(name, False)
    # Folded absorbent towel pad: three thick terry plies in place of the generic box.
    path = "/World/Objects/pad"
    for child in ("Shell", "Band"):
        if stage.GetPrimAtPath(f"{path}/{child}").IsValid():
            stage.RemovePrim(f"{path}/{child}")
    sx, sy, sz = (2 * h for h in layout.shapes["pad"].half)
    towel(look, path, (sx, sy, sz), 0.0, ribs=12, plies=3)
    # Sharps bin: the yellow tub in place of the target's steel walls (the colliders stay).
    bin_root = stage.GetPrimAtPath("/World/Targets/hd_bin")
    if model_size("cytotoxic_bin") is not None and bin_root.IsValid():
        for prim in Usd.PrimRange(bin_root):
            if prim.GetName().endswith("_Visual"):
                UsdGeom.Imageable(prim).MakeInvisible()
        bx, by = layout.info["spill"]["bin"]
        add_model(stage, "cytotoxic_bin", "/World/Spill/Bin", (bx, by, z))
        sign = look.label(
            "/World/Spill/BinSign", "CYTOTOXIC", [bx - 0.101, by, z + 0.045], [0.10, 0.022], PURPLE
        )
        if sign is not None:
            sign.AddRotateZOp().Set(-90.0)
    backend._spill = {
        name: stage.GetPrimAtPath(f"/World/Spill/{name}") for name in ("Puddle", "Splinters", "Sheet")
    }


# --- runtime ---------------------------------------------------------------------------------------


def _set_op(prim, kind, value):
    from pxr import UsdGeom

    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == kind:
            op.Set(value)
            return


def command(backend, command):
    """`spill` scene commands: break (value: xy of the break), puddle (radius), contain (radius)."""
    from pxr import Gf, UsdGeom

    prims = backend._spill
    layout = backend.layout
    z = float(layout.table_height)
    translate, scale = UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeScale
    if command.name == "break":
        cx, cy = (float(v) for v in command.value)
        backend._spill_center = (cx, cy)
        # The intact bottle is gone: out of sight, parked on the floor under the bench.
        bottle = backend.objects["bottle"]
        bottle.set_visibility(False)
        bottle.set_world_pose(position=np.array([cx, cy, 0.06]))
        bottle.set_linear_velocity(np.zeros(3))
        bottle.set_angular_velocity(np.zeros(3))
        for name, (dx, dy) in layout.info["spill"]["shard_offsets"].items():
            staged = layout.objects[name]
            backend.supports[name].CreateJointEnabledAttr(False)
            body = backend.objects[name]
            body.set_world_pose(
                position=np.array([cx + dx, cy + dy, staged.position[2]]),
                orientation=staged.orientation,
            )
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))
            backend._set_collision(name, True)
            body.set_visibility(True)
        _set_op(prims["Puddle"], translate, Gf.Vec3d(cx, cy, z + 0.0012))
        _set_op(prims["Splinters"], translate, Gf.Vec3d(cx, cy, z + 0.002))
        for name in ("Puddle", "Splinters"):
            UsdGeom.Imageable(prims[name]).MakeVisible()
    elif command.name == "puddle":
        radius = float(command.value)
        _set_op(prims["Puddle"], scale, Gf.Vec3f(radius, radius, 1.0))
    elif command.name == "contain":
        # The folded pad opens out over the liquid and soaks it up: the spread sheet is drawn in
        # its place (the folded pad's collider stays where it was put down).
        cx, cy = backend._spill_center
        radius = 0.8 * min(float(command.value), PAD_SHEET / 2)
        _set_op(prims["Sheet"], translate, Gf.Vec3d(cx, cy, z))
        stain = backend.stage.GetPrimAtPath("/World/Spill/Sheet/Stain")
        _set_op(stain, scale, Gf.Vec3f(radius, radius, 1.0))
        UsdGeom.Imageable(prims["Sheet"]).MakeVisible()
        UsdGeom.Imageable(prims["Splinters"]).MakeInvisible()
        backend.objects["pad"].set_visibility(False)
    else:
        raise ValueError(f"Unknown spill command: {command.name}")


# --- oracle jobs (policies/jobs.py) ------------------------------------------------------------------


def choose(jobs, task, state, remaining):
    """Pad first; the glass only once the liquid is contained."""
    if task.event_time is None:
        return None
    if not task.contained:
        return task.pad if task.pad in remaining else None
    for name in task.shards:
        if name in remaining:
            return name
    return None


def describe(jobs, task, state, name):
    from roboclock.envs.hospital.policies.jobs import Job

    layout = task.layout
    if name == task.pad:
        z = layout.table_height + layout.shapes[name].rest_height + 0.010
        job = Job(
            jobs.pick_point(task, name, state),
            np.array([*task.spill_center, z]),
            0.0,
            fast=True,
            rush=True,
        )
    else:
        job = jobs.container_job(task, state, name, task.bin, fast=True, rush=True)
    job.hover = jobs.pick_point(task, name, state) + [0.0, 0.0, 0.10]
    return job
