"""C3 crash-cart vial restock in the scene-02 ward (left bay): apparatus, dressing and a two-arm oracle.

After a Code Blue the crash cart, pulled up at the foot of the bed, is short of emergency drugs. A
mobile dual-arm robot faces the cart's drawer side. On the cart's worktop, the open medication drawer
insert holds three color-labeled drug lanes (EPI, ATRO, AMIO), each with four generous foam pockets.
A supply tray of color-capped vials (CC-BY vial model) stands on the worktop on each side of the
insert; each arm restocks from its own tray into its own half of the insert. A vial counts once it
rests upright, released, on the floor of a pocket in its own drug's lane (scored in
`tasks/ward/crash_cart_vials.py`).

Simplifications: the vials stand upright instead of lying jumbled in a bin; the drawer insert is a
static block on the worktop (no slide joint); pockets are open boxes with clearance, not press-fits.
The oracle only produces joint targets through the shared planners; it never moves a vial.
"""

import numpy as np

from roboclock.envs.hospital.layout import Fixture

VIAL_ASSET = "c3_vial"
# The CC-BY vial model (assets/models/vial/vial.usd): 31.8 mm diameter, 65 mm tall, glass body,
# aluminum crimp and flip-off cap. Injected into the asset catalog at layout time (see _vial_spec).
VIAL_RADIUS, VIAL_HEIGHT = 0.016, 0.065
# (short code, full name, training color). Colors are distinct training colors, loosely after ISO
# 26825 user-applied label classes; they are not a clinical standard.
DRUGS = (
    ("EPI", "EPINEPHRINE", (0.85, 0.16, 0.20)),
    ("ATRO", "ATROPINE", (0.13, 0.66, 0.30)),
    ("AMIO", "AMIODARONE", (0.16, 0.40, 0.90)),
)
PER_LANE = 4  # pockets per drug lane: two for each arm
POCKET_HALF = 0.026  # inner half width: +-10 mm of slack around a 32 mm vial
POCKET_WALL = 0.028  # foam wall height above the pocket floor
WALL = 0.006
POCKET_PITCH = 2 * POCKET_HALF + 2 * WALL + 0.004
# Robot: the common rig (hospital/rig.py, robot.rig.tasks.C3): one mobile base with two
# columns 0.60 m apart, both facing the cart (+x); arm 0 at -y, arm 1 at +y. Each arm restocks
# from the tray on its own side into the pockets on its own side of the insert.
# Crash cart: existing CC-BY model (0.626 x 0.418 m, worktop recess 0.598 m inside a 0.662 m rim).
# It is modelled small; scaled 1.35x it has a real crash cart's 0.85 x 0.56 m footprint and 0.81 m
# worktop. Turned -90 degrees: its drawers face the robot (-x), its tall bin rail stands at the far
# side (+x), and its side handle is at +y.
CART_SCALE = 1.35
CART_YAW = -np.pi / 2
CART_XY = (0.43, 0.04)
CART_TOP = round(0.662 * CART_SCALE, 4)  # raised rim
CART_RECESS = round(0.598 * CART_SCALE, 4)  # the worktop inside the rim
# Usable worktop inside the rim, relative to the cart center (world axes after the -90 degree turn).
CART_TOP_DRESSING = tuple(f"part_{k:03d}" for k in (7, 8, 9, 16, 17, 18, 19, 66, 67, 68, 69, 70))
WORKTOP_X = (-0.244, 0.086)
WORKTOP_Y = (-0.39, 0.31)
INSERT_Z = CART_RECESS + 0.028  # top of the insert base the pockets stand on
LANE_X = (0.284, 0.352, 0.420)  # EPI, ATRO, AMIO lanes, nearest the robot first
POCKET_Y = (0.102, 0.034, -0.034, -0.102)  # the +y arm's outer, inner; the -y arm's inner, outer
SUPPLY_PAD = 0.012
SUPPLY_X = (0.28, 0.355, 0.43)
SUPPLY_Y = (0.235, 0.31)  # the +y arm's tray; the -y arm's tray mirrors it
# Displays past the cart's far -y corner, both turned to the rig's head camera and side by side in
# its image above the -y arm's home pose: the shift dial on a post with a small shelf, the count
# board on its own post further out. RoboDojo rig (2026-11-03): on one post at (0.32, -0.56), top
# 1.05 m, both stood beyond the right edge of the 60-degree head view; nearer the robot the -y arm
# hides them. Both stand out of reach (the -y hand works no further out than y -0.31).
POST_XY = (0.82, -0.38)
POST_TOP = 0.86
BOARD_POST_XY = (0.90, -0.76)
BOARD_POST_TOP = 0.78
# Ward placement: the cart stands in the open floor at the foot of the bed in the preview's left bay.
# The ward turns 180 degrees about the robot side, then shifts (see bedside/ward.py). The curtain
# bundle by the bed's foot (source object 对象32326195) would stand right behind the robot: hidden.
WARD_POSE = {
    "usd": "assets/hospital/ward_02_preview_left.usdc",
    "translate": [1.73, -0.04, 0.0],
    "yaw": 180.0,
    "hide": ["对象32326195"],
}
# High over the cart's far corner: the tall rear bin rail no longer hides the insert's far lanes.
CAMERA = {"position": [0.62, 0.38, 2.20], "look_at": [0.22, -0.06, 0.86], "focal_mm": 25.0}
COUNT_PREFIX = "c3_count_"
VIAL_LABEL_SIZE = (0.046, 0.032)  # around the 54 mm glass body, below the crimp
LANE_PLATE = (0.042, 0.062)  # lane label plate: across the 46 mm end strip (y) x along the lane pitch (x)
# Crash-cart model surfaces by the source color of their flat converted material: (test, key).
# Red ABS shell and drawers, aluminum drawer pulls, a beige worktop, yellow side bins, grey bins and
# panels, a dark powder-coated frame and rubber casters/bumpers.
CART_PARTS = {"part_036": "label", "part_037": "label", "part_051": "pole"}


def pocket_name(drug, slot):
    return f"pocket_{DRUGS[drug][0].lower()}_{slot}"


def _vial_spec():
    return {
        "proxy": "cylinder",
        "usd_path": None,
        "scale": [1.0, 1.0, 1.0],
        "radius": VIAL_RADIUS,
        "height": VIAL_HEIGHT,
        "grasp_radius": VIAL_RADIUS,
        "grip_force": 8.0,
        "mass_range": [0.020, 0.030],
        "min_friction": 0.8,
    }


def rig_sides(config):
    """(mount xy, sign) per arm: sign = +1 for the arm on the rig's +y side, -1 for the -y arm."""
    from roboclock.envs.lab.robot import mount_positions

    mounts = [tuple(float(v) for v in m[:2]) for m in mount_positions(config, "C3")]
    center = 0.5 * (mounts[0][1] + mounts[1][1])
    return [(m, 1.0 if m[1] > center else -1.0) for m in mounts]


def _diagonal(mount, point):
    """The +-45 degree finger yaw _yaw_for uses for this bearing from the mount."""
    return float(np.pi / 4 if np.arctan2(point[1] - mount[1], point[0] - mount[0]) >= 0 else -np.pi / 4)


def build_layout(builder):
    b = builder
    sides = rig_sides(b.config)
    # The vial collider (matching the CC-BY model) is not a shared catalog asset: register it here.
    # The backend passes its own catalog dict to the layout, so it sees the same entry.
    b.catalog.setdefault(VIAL_ASSET, _vial_spec())
    n = b.task.spec.num_items
    lanes, cx, cy = len(DRUGS), *CART_XY
    if n != 2 * lanes * (PER_LANE // 2):
        raise ValueError(f"C3 needs exactly {2 * lanes * (PER_LANE // 2)} vials")
    per_arm = n // 2
    item_drug, item_arm = {}, {}
    for side, (_, sign) in enumerate(sides):
        # Each tray holds two of each drug in a seeded arrangement.
        kinds = b.rng.permutation(np.arange(per_arm) % lanes)
        slots = [(x, sign * y) for y in SUPPLY_Y for x in SUPPLY_X]
        b.fixture(
            f"c3_supply_tray_{side}",
            (sum(SUPPLY_X) / 3, sign * sum(SUPPLY_Y) / 2),
            (0.225, 0.15, SUPPLY_PAD),
            "polymer",
            z=CART_RECESS,
        )
        for k in range(per_arm):
            name = f"vial_{side * per_arm + k}"
            code, _, color = DRUGS[int(kinds[k])]
            b.item(name, VIAL_ASSET, slots[k], CART_RECESS + SUPPLY_PAD, color=color, label=code, yaw=0.0)
            item_drug[name], item_arm[name] = code, side
    # Crash cart body: an invisible collider below the worktop, drawn by the converted model.
    b.fixture("c3_cart_body", (cx, cy), (0.56, 0.84, CART_RECESS - 0.1), "cabinet_white", z=0.0, visible=False)
    # The drawer insert on the worktop: a low base block carrying the foam pockets, lane labels at
    # both ends.
    insert_x = (LANE_X[0] - POCKET_PITCH / 2 - 0.008, LANE_X[-1] + POCKET_PITCH / 2 + 0.008)
    insert_y = POCKET_Y[0] + POCKET_PITCH / 2 + 0.044
    b.fixture(
        "c3_insert_base",
        (sum(insert_x) / 2, 0.0),
        (insert_x[1] - insert_x[0], 2 * insert_y, INSERT_Z - CART_RECESS),
        "stainless",
        z=CART_RECESS,
    )
    pocket_drug, pocket_arm, lanes_map = {}, {}, []
    for d in range(lanes):
        names = []
        for s, y in enumerate(POCKET_Y):
            name = pocket_name(d, s)
            b.target(name, "tray", (LANE_X[d], y), (POCKET_HALF, POCKET_HALF), POCKET_WALL, z0=INSERT_Z)
            pocket_drug[name] = DRUGS[d][0]
            pocket_arm[name] = next(k for k, (_, sign) in enumerate(sides) if sign * y > 0)
            names.append(name)
        lanes_map.append(names)
    # The robot's mobile base: a column from the floor under each arm mount and a base plate.
    for k, ((sx, sy), _) in enumerate(sides):
        b.fixtures.append(
            Fixture(f"c3_robot_stand_{k}", np.array([sx, sy, b.z / 2]), np.array([0.12, 0.12, b.z]), "rubber")
        )
    mx = sides[0][0][0]
    my = 0.5 * (sides[0][0][1] + sides[1][0][1])
    span = abs(sides[1][0][1] - sides[0][0][1])
    b.fixtures.append(
        Fixture("c3_robot_base", np.array([mx - 0.06, my, 0.04]), np.array([0.34, span + 0.20, 0.08]), "rubber",
                collision=False)
    )
    # Display post: shift dial on top, count board beside it.
    b.fixture("c3_post", POST_XY, (0.05, 0.05, POST_TOP - 0.02), "stainless", z=0.0)
    b.fixture("c3_post_shelf", POST_XY, (0.12, 0.12, 0.02), "stainless", z=POST_TOP - 0.02)
    b.fixture("c3_board_post", BOARD_POST_XY, (0.04, 0.04, BOARD_POST_TOP), "stainless", z=0.0)
    b.indicator("indicator", POST_XY, "SHIFT", height=POST_TOP - b.z + 0.045)
    worktop_center = (cx + sum(WORKTOP_X) / 2, cy + sum(WORKTOP_Y) / 2)
    b.info.update(
        {
            "bedside_task": "ward_c3",
            "ward_pose": {**WARD_POSE, "hide": list(WARD_POSE["hide"])},
            "hide_bench": True,
            "work_surface": {
                "center": [*map(float, b.xy(*worktop_center)), CART_RECESS - 0.05],
                "size": [WORKTOP_X[1] - WORKTOP_X[0], WORKTOP_Y[1] - WORKTOP_Y[0], 0.10],
            },
            "initial_ee_position": [*b.xy(sum(SUPPLY_X) / 3, sides[0][1] * sum(SUPPLY_Y) / 2), float(CART_RECESS + 0.20)],
            # Fingers along the diagonal _yaw_for picks for each arm's bearing to its own tray (the
            # trays sit just inboard of the 0.60 m rig's mounts), so no wrist turn precedes the work.
            "initial_ee_yaw": _diagonal(sides[0][0], (sum(SUPPLY_X) / 3, sides[0][1] * sum(SUPPLY_Y) / 2)),
            "initial_secondary_ee_yaw": _diagonal(sides[1][0], (sum(SUPPLY_X) / 3, sides[1][1] * sum(SUPPLY_Y) / 2)),
            "initial_gripper_width": 0.04,
            "c3": {
                "item_drug": item_drug,
                "item_arm": item_arm,
                "pocket_drug": pocket_drug,
                "pocket_arm": pocket_arm,
                "drugs": [list(d[:2]) + [list(d[2])] for d in DRUGS],
                "lanes": lanes_map,
                "mounts": [list(m) for m, _ in sides],
                "parks": [
                    [*map(float, b.xy(sum(SUPPLY_X) / 3, sign * sum(SUPPLY_Y) / 2)), float(CART_RECESS + 0.20)]
                    for _, sign in sides
                ],
                "insert_center_y": float(b.xy(0.0, 0.0)[1]),
                "lane_x": [float(b.xy(x, 0)[0]) for x in LANE_X],
                "label_y": float(b.xy(0, POCKET_Y[0] + POCKET_PITCH / 2 + 0.022)[1] - b.xy(0, 0)[1]),
                "insert_back_x": float(b.xy(insert_x[1], 0)[0]),
                "cart_xy": [float(v) for v in b.xy(cx, cy)],
                "post_xy": [float(v) for v in b.xy(*POST_XY)],
                "board_xy": [float(v) for v in b.xy(*BOARD_POST_XY)],
                "camera": CAMERA,
                "assets": {
                    "crash_cart": "src/roboclock/envs/hospital/assets/models/crash_cart/crash_cart.usd",
                    "vial": "src/roboclock/envs/hospital/assets/models/vial/vial.usd",
                },
            },
        }
    )


# --- scene authoring ------------------------------------------------------------------------


def _hide(stage, path):
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()


def _dress_vial(backend, name, obj, labels):
    """The CC-BY vial model on the collider, with the cap and crimp in the drug color and a label."""
    from pxr import Usd, UsdGeom, UsdShade

    from roboclock.envs.hospital.props import add_model, model_size

    stage, look = backend.stage, backend.appearance
    path = f"/World/Objects/{name}"
    if model_size("vial") is None:
        return  # keep the backend's plain cylinder visual
    for child in ("Wall", "Rim", "Identity", "Identity_material"):
        _hide(stage, f"{path}/{child}")
    node = add_model(stage, "vial", f"{path}/Model", (0.0, 0.0, -VIAL_HEIGHT / 2))
    color = tuple(float(c) for c in obj.color)
    # The model's own materials are flat colors: a molded flip-off cap and an anodized aluminum
    # crimp in the drug color (textured), and the glass body (glass stays flat).
    cap = look.textured(f"c3_cap_{obj.label}_tex", "plastic_matte", color=color, roughness=0.35)
    crimp = look.textured(
        f"c3_crimp_{obj.label}", "aluminum_brushed", color=tuple(0.35 + 0.65 * c for c in color), repeats=30.0
    )
    glass = look.material("c3_vial_glass", color=(0.86, 0.90, 0.92), roughness=0.08, metallic=0.0)
    for prim in Usd.PrimRange(node.GetPrim()):
        if prim.IsA(UsdGeom.Mesh):
            # part_000 is the flip-off cap, part_001 the crimp seal, part_002 the glass body.
            name = prim.GetName()
            material = glass if name.endswith("part_002") else crimp if name.endswith("part_001") else cap
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, UsdShade.Tokens.strongerThanDescendants)
    if labels:
        label = look.label(
            f"{path}/DrugLabel",
            obj.label,
            [0.0, 0.0, -0.007],
            list(VIAL_LABEL_SIZE),
            color,
            cylindrical_radius=VIAL_RADIUS + 0.0006,
        )
        if label is not None:
            # A paper label with the drug-color bands and the code in bold, as large as fits.
            _set_label_image(stage, f"{path}/DrugLabel", _plate_texture(
                obj.label, (0.08, 0.10, 0.14), (0.97, 0.97, 0.95), (512, 350), band=color, stretch=1.15
            ))
            label.AddRotateZOp().Set(-55.0)  # turned toward the task camera


def author(backend):
    """Dress the cart and vials, color the foam pockets and prepare the live count readout."""
    from pxr import Gf, UsdGeom

    from roboclock.envs.hospital.props import add_model, model_size

    stage, look, layout = backend.stage, backend.appearance, backend.layout
    info = layout.info["c3"]
    labels = backend.config["visuals"]["labels"]
    cx, cy = info["cart_xy"]
    _hide(stage, "/World/Fixtures/c3_cart_body_Visual")
    _robot_base(backend)
    if model_size("crash_cart") is not None:
        add_model(stage, "crash_cart", "/World/C3/CrashCart", (cx, cy, 0.0), (CART_SCALE,) * 3, yaw=CART_YAW)
        # The model's own worktop dressing (a pad kit and two sheets) would sit under the insert
        # and trays: removed, the drawer insert and supply trays take their place.
        for part in CART_TOP_DRESSING:
            _hide(stage, f"/World/C3/CrashCart/{part}")
        _texture_cart(stage, look, "/World/C3/CrashCart")
    else:  # missing model: show the collider as a plain cabinet
        look.rounded("/World/C3/CartProxy", [cx, cy, CART_RECESS / 2], [0.56, 0.84, CART_RECESS], "cabinet_white", 0.01)
    for name, obj in layout.objects.items():
        _dress_vial(backend, name, obj, labels)
    # The drawer insert is one charcoal foam block (base and pocket walls) with drug-color foam
    # pocket floors; the supply trays are molded plastic trays.
    foam = look.textured("c3_foam_tex", "foam", color=(0.17, 0.18, 0.20))
    _bind(stage, "/World/Fixtures/c3_insert_base_Visual", foam)
    tray = look.textured("c3_supply_tray", "plastic_matte", color=(0.82, 0.88, 0.94))
    for side in (0, 1):
        _bind(stage, f"/World/Fixtures/c3_supply_tray_{side}_Visual", tray)
    y0 = info["insert_center_y"]
    for d, (code, full, color) in enumerate(info["drugs"]):
        insert = look.textured(f"c3_pocket_{code}", "foam", color=tuple(0.8 * c for c in color))
        plate = look.textured(f"c3_plate_{code}", "plastic_matte", color=tuple(color), roughness=0.4)
        for name in info["lanes"][d]:
            for part in ("floor", "left", "right", "front", "back"):
                _bind(stage, f"/World/Targets/{name}/{part}_Visual", insert if part == "floor" else foam)
        # A color-coded plate at each end of the lane (the lane's edge on the foam block), with
        # the drug code in bold white, as large as the plate allows.
        for end, sign in (("L", 1.0), ("R", -1.0)):
            center = [info["lane_x"][d], y0 + sign * info["label_y"], INSERT_Z + 0.001]
            look.rounded(
                f"/World/C3/LanePlate_{code}_{end}", center, (LANE_PLATE[1], LANE_PLATE[0], 0.002), plate, 0.0008
            )
            if labels:
                label = look.label(
                    f"/World/C3/LaneLabel_{code}_{end}", code, [center[0], center[1], INSERT_Z + 0.0021],
                    list(LANE_PLATE), tuple(color),
                )
                # Letters stretched along the lane: the oblique task camera foreshortens them.
                _set_label_image(stage, f"/World/C3/LaneLabel_{code}_{end}", _plate_texture(
                    code, (1.0, 1.0, 1.0), tuple(color), (300, 440), stretch=1.6, border=True
                ))
                label.AddRotateZOp().Set(-90.0)  # text up toward the robot: read from the camera side
                label.AddRotateXOp().Set(90.0)
    if labels:
        tag = look.label(
            "/World/C3/InsertTag",
            "CODE CART MEDS",
            [info["insert_back_x"] + 0.001, y0, (INSERT_Z + CART_RECESS) / 2],
            [0.16, 0.024],
            (0.75, 0.10, 0.12),
        )
        tag.AddRotateZOp().Set(-90.0)  # on the insert's far face, read from +x
    # Live count readout on the display post: one prepared label per count, only one shown.
    total = len(layout.objects)
    need = backend.task.spec.num_items  # every vial (score_version 2)
    x, y = info["board_xy"]
    from roboclock.envs.hospital.rig import head_camera_pose

    camera = head_camera_pose(backend.config, backend.task.task_id)[0]
    board = UsdGeom.Xform.Define(stage, "/World/C3/CountBoard")
    board.AddTranslateOp().Set(Gf.Vec3d(x, y, BOARD_POST_TOP))
    board.AddRotateZOp().Set(float(np.degrees(np.arctan2(camera[1] - y, camera[0] - x))) - 90.0)
    root = "/World/C3/CountBoard"
    # A white board in a brushed aluminum frame on an aluminum post and shelf.
    trim = look.textured("c3_stand_aluminum", "aluminum_brushed")
    for name in ("c3_post", "c3_post_shelf", "c3_board_post"):
        _bind(stage, f"/World/Fixtures/{name}_Visual", trim)
    look.rounded(f"{root}/Panel", [0, 0, 0.095], [0.20, 0.012, 0.17], "cabinet_white", 0.003)
    for name, center, size in (
        ("Top", [0, 0, 0.181], [0.212, 0.016, 0.008]), ("Bottom", [0, 0, 0.009], [0.212, 0.016, 0.008]),
        ("Left", [-0.103, 0, 0.095], [0.008, 0.016, 0.18]), ("Right", [0.103, 0, 0.095], [0.008, 0.016, 0.18]),
    ):
        look.rounded(f"{root}/Frame{name}", center, size, trim, 0.002)
    if labels:
        look.label(f"{root}/Need", f"NEED {need}/{total}", [0, 0.0075, 0.14], [0.18, 0.07], (0.75, 0.10, 0.12))
    for k in range(total + 1):
        path = f"{root}/Count_{k}"
        UsdGeom.Xform.Define(stage, path)
        if labels:
            look.label(
                f"{path}/Text",
                f"IN {k}/{total}",
                [0, 0.0075, 0.06],
                [0.18, 0.07],
                (0.13, 0.66, 0.30) if k >= need else (0.20, 0.30, 0.45),
            )
        backend.zone_prims[f"{COUNT_PREFIX}{k}"] = stage.GetPrimAtPath(path)
        if k:
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()


def _bind(stage, path, material):
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material)


def cart_surface(rgb):
    """The textured surface for a crash-cart part, from its flat source color (r, g, b)."""
    r, g, b = rgb
    if r > 0.6 and g < 0.25:
        return "red"
    if r > 0.8 and g > 0.6 and b < 0.2:
        return "yellow"
    if r > 0.8 and g > 0.7 and 0.35 < b < 0.65:
        return "beige"
    if g > r + 0.1 and g > b + 0.1:
        return "green"
    if b > r + 0.1:
        return "blue"
    level = (r + g + b) / 3
    if level < 0.05:
        return "rubber"
    if level < 0.40:
        return "frame"
    if abs(level - 0.7647) < 0.01:
        return "pull"
    return "grey"


def _texture_cart(stage, look, root):
    """The converted crash-cart model has flat colors and no texture maps: bind CC0 textured
    materials to its parts by their source color (CART_PARTS overrides single parts)."""
    from pxr import Usd, UsdGeom, UsdShade

    surfaces = {
        "red": look.textured("c3_cart_red", "plastic_white", color=(1.0, 0.17, 0.11)),
        "yellow": look.textured("c3_cart_yellow", "plastic_matte", color=(0.95, 0.74, 0.10), roughness=0.45),
        "beige": look.textured("c3_cart_top", "plastic_matte", color=(0.90, 0.85, 0.66)),
        "green": look.textured("c3_cart_green", "plastic_matte", color=(0.30, 0.62, 0.24)),
        "blue": look.textured("c3_cart_blue", "plastic_matte", color=(0.30, 0.55, 0.80)),
        "grey": look.textured("c3_cart_grey", "plastic_matte", color=(0.84, 0.86, 0.87)),
        "pull": look.textured("c3_cart_pull", "aluminum_brushed"),
        "frame": look.textured("c3_cart_frame", "powder_coat_dark"),
        "rubber": look.material("rubber"),
        "pole": look.material("stainless"),
        "label": look.material("label_paper"),
    }
    node = stage.GetPrimAtPath(root)
    if not node.IsValid():
        return
    for prim in Usd.PrimRange(node):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        key = CART_PARTS.get(prim.GetName())
        if key is None:
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            rgb = _diffuse(material)
            if rgb is None:
                continue
            key = cart_surface(rgb)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(surfaces[key].material)


def _diffuse(material):
    from pxr import Usd, UsdShade

    if not material:
        return None
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        value = shader.GetInput("diffuseColor").Get() if shader and shader.GetInput("diffuseColor") else None
        if value is not None:
            return tuple(float(v) for v in value)
    return None


def _robot_base(backend):
    """The two arm columns stand on one mobile base: light grey shells like the mount columns, a
    dark powder-coated chassis with a rubber bumper strip, and rubber casters (visual only)."""
    stage, look = backend.stage, backend.appearance
    shell = look.textured("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90))
    chassis = look.textured("c3_robot_chassis", "powder_coat_dark")
    for k in range(2):
        _bind(stage, f"/World/Fixtures/c3_robot_stand_{k}_Visual", shell)
    base = next((f for f in backend.layout.fixtures if f.name == "c3_robot_base"), None)
    if base is None:
        return
    _hide(stage, "/World/Fixtures/c3_robot_base")
    (x, y, _), (sx, sy, _) = base.position, base.size
    look.rounded("/World/C3/RobotChassis", (x, y, 0.07), (sx, sy, 0.07), chassis, 0.012)
    look.rounded("/World/C3/RobotBumper", (x, y, 0.06), (sx + 0.012, sy + 0.012, 0.022), "rubber", 0.008)
    look.rounded("/World/C3/RobotCover", (x, y, 0.11), (sx - 0.02, sy - 0.02, 0.016), shell, 0.008)
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        cx, cy = x + dx * (sx / 2 - 0.05), y + dy * (sy / 2 - 0.05)
        look.cylinder(f"/World/C3/RobotCaster_{k}", (cx, cy, 0.026), 0.026, 0.024, "rubber", axis="X")


def _set_label_image(stage, path, image):
    """Show `image` on a label card made by look.label (its texture shader's file input)."""
    from pxr import Sdf, UsdShade

    texture = UsdShade.Shader(stage.GetPrimAtPath(path + "_material/Texture"))
    if texture:
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(image)))


def _bold_font(size):
    import importlib.util
    from pathlib import Path

    from PIL import ImageFont

    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.submodule_search_locations:
        candidates.append(Path(spec.submodule_search_locations[0]) / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _plate_texture(text, fg, bg, size, *, band=None, stretch=1.0, border=False):
    """A cached label image: `text` in bold as large as fits (letters stretched `stretch` x tall), on
    `bg`, with optional drug-color bands at top and bottom or a thin white border."""
    import hashlib
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw

    cache = Path(os.environ.get("ROBOCLOCK_ASSET_CACHE", Path.home() / ".cache/roboclock")) / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(repr(("c3", text, fg, bg, size, band, stretch, border)).encode()).hexdigest()[:20]
    path = cache / f"c3_{key}.png"
    if path.exists():
        return path
    width, height = size

    def rgb(color):
        return tuple(round(255 * float(v)) for v in color)

    image = Image.new("RGB", size, rgb(bg))
    draw = ImageDraw.Draw(image)
    inner = (0, 0, width, height)
    if band is not None:
        bar = round(0.17 * height)
        draw.rectangle((0, 0, width, bar), fill=rgb(band))
        draw.rectangle((0, height - bar, width, height), fill=rgb(band))
        inner = (0, bar, width, height - bar)
    if border:
        edge = max(4, round(0.03 * min(size)))
        draw.rectangle((edge, edge, width - edge, height - edge), outline=(255, 255, 255), width=max(3, edge // 2))
        inner = (edge + 2, edge + 2, width - edge - 2, height - edge - 2)
    # Draw large, crop to the ink, then scale into the free area (at most `stretch` x taller).
    font = _bold_font(200)
    left, top, right, bottom = font.getbbox(text)
    glyphs = Image.new("L", (right - left + 20, bottom - top + 20), 0)
    ImageDraw.Draw(glyphs).text((10 - left, 10 - top), text, fill=255, font=font)
    glyphs = glyphs.crop(glyphs.getbbox())
    room_w, room_h = 0.86 * (inner[2] - inner[0]), 0.80 * (inner[3] - inner[1])
    scale = min(room_w / glyphs.width, room_h / (glyphs.height * stretch))
    w, h = max(1, round(glyphs.width * scale)), max(1, round(glyphs.height * scale * stretch))
    glyphs = glyphs.resize((w, h), Image.LANCZOS)
    x0 = round((inner[0] + inner[2] - w) / 2)
    y0 = round((inner[1] + inner[3] - h) / 2)
    image.paste(Image.new("RGB", (w, h), rgb(fg)), (x0, y0), glyphs)
    image.save(path)
    return path


def _camera(backend):
    """Frame the robot, both hands, the trays and the insert (instead of the shared camera)."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["c3"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1))
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    if "focal_mm" in view:
        camera.set_focal_length(view["focal_mm"] / 1000)
    for _ in range(3):
        backend.world.render()


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_c3_camera_for", None) is not backend.task:
        backend._c3_camera_for = backend.task
        _camera(backend)
    return {}


# --- scripted two-arm oracle ------------------------------------------------------------------

CARRY_CLEAR = 0.035  # vial bottom above the walls, stocked vials and supply vials while carried
GRASP_LIFT = -0.006  # tool point slightly below center: the finger pads straddle the center of mass
CARRY_SPEED = 0.25
OPEN_WIDTH = 0.028
CLOSE_WIDTH = 0.008  # 8 mm inside the 16 mm radius: about 8 N per finger, below the 10 N effort cap,
# so the two position-controlled fingers act as springs that keep the vial centered
RELEASE_WIDTH = 0.026
# Pockets within this of the insert center line are worked by one arm at a time. On the 0.60 m rig
# both forearms angle further inward, and an arm at an inner pocket and the other at an outer pocket
# of the same lane touched forearms (link4, seed 2): every pocket is in the zone.
ZONE_HALF = 0.14
INNER_HALF = 0.06  # inner pockets: the two nearest the center line (planning order only)


def _plan_order(task, side):
    """This arm's vials in tray order; each takes a free pocket of its drug lane on this arm's half.
    Arm 0 fills its inner pockets first and arm 1 its outer ones, so the two hands
    are rarely at the insert's center line together (a lock covers the rest)."""
    info = task.layout.info["c3"]
    mine = [p for p, arm in info["pocket_arm"].items() if arm == side]
    y0 = info["insert_center_y"]

    def inner_first(pocket):
        inner = abs(task.layout.targets[pocket].position[1] - y0) < INNER_HALF
        return (0 if inner else 1) if side == 0 else (1 if inner else 0)

    free = {}
    for pocket in sorted(mine, key=inner_first):
        free.setdefault(info["pocket_drug"][pocket], []).append(pocket)
    names = sorted((n for n, arm in info["item_arm"].items() if arm == side), key=lambda n: int(n.split("_")[-1]))
    return [(name, free[info["item_drug"][name]].pop(0)) for name in names]


def _yaw_for(planner, task, point):
    """Close the fingers along a diagonal, on the side of the arm's bearing to the target: this keeps
    the wrist joint well inside its +-2 rad range (wrist ~ base bearing - yaw)."""
    mount = task.layout.info["c3"]["mounts"][planner.side]
    bearing = np.arctan2(point[1] - mount[1], point[0] - mount[0])
    return np.pi / 4 if bearing >= 0 else -np.pi / 4


def _in_zone(task, pocket):
    info = task.layout.info["c3"]
    return abs(task.layout.targets[pocket].position[1] - info["insert_center_y"]) < ZONE_HALF


def _arm(planner, task, state):
    layout = task.layout
    now = task.time
    side = getattr(planner, "side", 0)
    ee = planner.ee_position if getattr(planner, "ee_position", None) is not None else state.ee_position
    ee = np.asarray(ee, dtype=float)
    if getattr(planner, "_c3_task", None) is not task:
        planner._c3_task = task
        planner.c3_plan = _plan_order(task, side)
        planner.c3_index = 0
        planner.c3_phase, planner.c3_time = "approach", now
        planner.c3_offset = np.array([0.0, 0.0, GRASP_LIFT])
        planner.c3_bias = np.zeros(2)
        planner.c3_retries = 0
        planner.c3_zone = False
    partner = getattr(planner, "partner", None)
    partner_zone = bool(getattr(partner, "c3_zone", False)) and getattr(partner, "_c3_task", None) is task
    planner.gentle = True
    planner.open_width = OPEN_WIDTH
    planner.close_width = CLOSE_WIDTH
    park = np.asarray(layout.info["c3"]["parks"][side], dtype=float)

    def turn(yaw_goal):
        planner.yaw += float(np.clip(yaw_goal - planner.yaw, -2.5 * planner.dt, 2.5 * planner.dt))
        return abs(planner.yaw - yaw_goal) < 0.03

    if planner.c3_index >= len(planner.c3_plan):
        # Done: rise clear of the insert, then park over this arm's empty tray.
        planner.c3_zone = False
        turn(_yaw_for(planner, task, park))
        goal = park if ee[2] > park[2] - 0.02 else np.array([*ee[:2], park[2]])
        return planner.finish(state, goal, False, 0.2)
    name, pocket = planner.c3_plan[planner.c3_index]
    obstacle_top = INSERT_Z + POCKET_WALL
    for other in layout.objects:
        body = state.objects[other]
        # Resting vials (trays and stocked pockets); not this one, nor one the other hand carries.
        if other != name and not body.grasped and body.position[2] < INSERT_Z + 0.10:
            obstacle_top = max(obstacle_top, body.position[2] + layout.shapes[other].height / 2)
    obj = state.objects[name]
    target = layout.targets[pocket]
    shape = layout.shapes[name]
    pick = obj.position + np.array([0.0, 0.0, GRASP_LIFT])
    carry_z = obstacle_top + CARRY_CLEAR + shape.height / 2 + GRASP_LIFT
    phase = planner.c3_phase
    close, speed = False, None
    picking = phase in ("approach", "descend", "grasp")
    yaw_ready = turn(_yaw_for(planner, task, pick if picking else target.position))

    def go(value):
        planner.c3_phase, planner.c3_time = value, now
        task.event_log.append(
            {"time": round(now, 3), "type": "c3_planner", "arm": side, "object": name, "phase": value}
        )

    def next_vial():
        planner.c3_index += 1
        planner.c3_retries = 0
        planner.c3_zone = False
        go("approach")

    if name in task.dropped:
        next_vial()
        return planner.finish(state, np.array([*ee[:2], carry_z]), False)
    if phase == "approach":
        # Travel high over the trays and the insert, then hover above the vial before descending.
        hover = pick[2] + 0.07
        far = np.linalg.norm(ee[:2] - pick[:2]) > 0.03
        goal = np.array([*pick[:2], max(hover, carry_z) if far else hover])
        if not far and np.linalg.norm(ee[:2] - goal[:2]) < 0.006 and abs(ee[2] - goal[2]) < 0.02 and yaw_ready:
            go("descend")
    elif phase == "descend":
        goal, speed = pick.copy(), 0.22
        if np.linalg.norm(ee[:2] - pick[:2]) > 0.015:
            go("approach")
        elif np.linalg.norm(ee - pick) < 0.005 or (now - planner.c3_time > 0.6 and np.linalg.norm(ee - pick) < 0.010):
            go("grasp")
    elif phase == "grasp":
        goal, close = pick.copy(), True
        if obj.grasped and now - planner.c3_time > 0.15:
            planner.c3_offset = ee - obj.position
            planner.c3_bias = np.zeros(2)
            go("lift")
        elif now - planner.c3_time > 1.6:
            planner.c3_retries += 1
            task.event_log.append({"time": now, "type": "c3_grasp_failed", "arm": side, "object": name})
            if planner.c3_retries >= 3:
                next_vial()
            else:
                go("approach")
    elif phase == "lift":
        goal, close, speed = np.array([*planner.last_goal[:2], carry_z]), True, CARRY_SPEED
        if ee[2] > carry_z - 0.02:
            go("transfer")
    else:  # transfer, lower, release, retreat
        if phase in ("transfer", "lower") and obj.grasped:
            # Track where the vial sits in the hand (it shifts slightly as the wrist turns), and
            # near the pocket integrate its measured miss into a bounded correction.
            planner.c3_offset[:2] += 0.3 * (ee[:2] - obj.position[:2] - planner.c3_offset[:2])
            miss = target.position[:2] - obj.position[:2]
            if np.linalg.norm(miss) < 0.03:
                planner.c3_bias = np.clip(planner.c3_bias + 0.08 * miss, -0.02, 0.02)
        offset = planner.c3_offset + np.array([*planner.c3_bias, 0.0])
        centered = np.linalg.norm(obj.position[:2] - target.position[:2]) < 0.004
        above = np.array([*target.position[:2], carry_z - offset[2]]) + offset
        seat = np.array([*target.position[:2], target.support_z + shape.height / 2 + 0.003]) + offset
        if phase == "transfer":
            goal, close, speed = above, True, CARRY_SPEED
            near = np.linalg.norm(ee[:2] - above[:2]) < 0.15
            if _in_zone(task, pocket) and near and not planner.c3_zone:
                if partner_zone:
                    goal = planner.last_goal.copy()  # the other hand is at the center line: wait
                else:
                    planner.c3_zone = True
            if not obj.grasped and np.linalg.norm(obj.position - ee) > 0.06:
                planner.c3_zone = False
                go("approach")  # slipped out on the way: pick it up again where it fell
            elif centered and abs(ee[2] - above[2]) < 0.015 and yaw_ready and (planner.c3_zone or not _in_zone(task, pocket)):
                go("lower")
        elif phase == "lower":
            goal, close, speed = seat, True, 0.12
            seated_low = obj.position[2] - shape.height / 2 < target.support_z + 0.005
            if seated_low or np.linalg.norm(ee - seat) < 0.004 or (
                now - planner.c3_time > 2.0 and abs(ee[2] - seat[2]) < 0.010
            ):
                planner.c3_hold = ee.copy()
                go("release")
        elif phase == "release":
            planner.open_width = RELEASE_WIDTH
            goal, close = planner.c3_hold.copy(), False
            if now - planner.c3_time > 0.30:
                go("retreat")
        else:
            planner.open_width = RELEASE_WIDTH
            goal = np.array([*planner.c3_hold[:2], carry_z])
            if ee[2] > carry_z - 0.012:
                next_vial()
    if now - planner.c3_time > 8.0 and phase != "release" and not (phase == "transfer" and partner_zone):
        task.event_log.append({"time": now, "type": "c3_planner_timeout", "arm": side, "object": name, "phase": phase})
        if obj.grasped:
            planner.c3_hold = ee.copy()
            go("release")
        else:
            planner.c3_retries += 1
            next_vial() if planner.c3_retries >= 3 else go("approach")
    if phase in ("release", "retreat"):
        planner.gentle = False  # open at once
    return planner.finish(state, np.asarray(goal, dtype=float), close, speed)


def command(planner, task, state):
    """Left arm: privileged diagnostic pick-and-place; every waypoint becomes bounded joint commands."""
    return _arm(planner, task, state)


def command_secondary(planner, task, state):
    """Right arm: the same routine on its own tray and its own half of the insert."""
    return _arm(planner, task, state)
