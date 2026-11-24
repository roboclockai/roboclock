"""C4 blood-tube triage station in the scene-02 left ward: arrival tray, three analyzer racks.

Stoppered blood tubes arrive upright in a pneumatic-tube arrival tray. Each analyzer rack is a
row of open, funnel-mouthed wells (real static colliders) in the rack's stopper color:
purple = HEMATOLOGY (EDTA), blue = COAG (citrate), red = CHEMISTRY (serum). A tube counts once it
rests, released and upright, on the floor of a well. Plain rigid pick-and-place: no press fit, no
liquid, no hidden attachment. The tube and rack are clean primitive proxies (no suitably licensed
vacutainer/rack model was found; the lab's 12-slot tube rack is too large for three racks in reach).

Surfaces use the CC0 PBR sets of hospital/assets/textures (`look.textured`): each arrival tray
is a white plastic tray shell around a charcoal open-cell foam insert; racks, funnel lips and
title-card holders are glossy molded plastic tinted with their class color; tubes have a frosted
plastic wall, a paper label wrap and glossy colored plastic stoppers; the readout display has a
powder-coated bezel on a brushed aluminum mast; the mobile base has matte plastic shells, a
powder-coated skirt, aluminum trim and rubber casters. Screens, blood and label images stay flat.

Hooks used by the shared backend/planner: build_layout, author, measure, command.
"""

import os

import numpy as np

from roboclock.physics.geometry import rotation_matrix

_TRACE = bool(os.environ.get("C4_TRACE"))  # diagnostic: log both hands every 0.1 s
TUBE = "blood_tube"
TUBE_SPEC = {  # primitive proxy; the collider covers the stopper
    "proxy": "cylinder",
    "usd_path": None,
    "scale": [1.0, 1.0, 1.0],
    "radius": 0.0115,
    "height": 0.100,
    "grasp_radius": 0.0115,
    "grip_force": 12.0,
    "mass_range": [0.018, 0.026],
    "min_friction": 0.9,
}
SHORT = {"purple": "HEME", "blue": "COAG", "red": "CHEM"}  # rack label blocks
# (key, rack label, tube additive label, stopper color)
CLASSES = (
    ("purple", "HEMATOLOGY", "EDTA", (0.52, 0.26, 0.74)),
    ("blue", "COAG", "CITRATE", (0.22, 0.55, 0.95)),
    ("red", "CHEMISTRY", "SERUM", (0.86, 0.11, 0.11)),
)
WELL_HALF = 0.018  # 36 mm square opening for a 23 mm tube: leans at most ~13 degrees
WELL_WALL = 0.055
# Two arms sort at once. The work is split by the plane y = SPLIT_Y: the arm whose mount stands on
# the +y side ("left" half) owns the left arrival tray and the left half of every rack, the other arm
# the "right" (-y) half. On the common rig (hospital/rig.py) arm 0 stands at -y, so it works the
# right half and arm 1 the left half (`arm_half`). Each rack is a
# row of wells along y, with its label block in the middle gap between the two halves.
SPLIT_Y = -0.12
WELL_DY = (0.09, 0.15)  # well centers from the split plane, per side (two wells per rack half)
RACK_X = (0.40, 0.46, 0.52)  # HEMATOLOGY, COAG, CHEMISTRY rows, front to back
ARRIVAL_X = (0.33, 0.39, 0.45)
ARRIVAL_DY = (0.24, 0.30)  # tray rows from the split plane, per side
ARRIVAL_BASE = 0.006
ARRIVAL_BAR = 0.030
SIDES = ("left", "right")
SIGN = {"left": 1.0, "right": -1.0}
UPRIGHT_TOLERANCE = 0.30  # rad; any tube resting on a well floor stands within this
TRAVEL_Z = 0.25  # tool height while moving: the carried tube clears racked tubes by ~6 cm
GRASP_DZ = 0.030  # grasp this far above the tube center, below the stopper
INNER_LOCK_DX = 0.09  # both hands may not work inner wells closer than this in x at once
# Real ward counter (scene-02 preview's left wall): the counter front lies at robot x 0.25, the
# backsplash at 0.79 and the top at 0.75. The station sits between the analyzer-like device box
# (robot y 0.39..0.72) and the counter's far end; the stethoscope clutter there is hidden.
STETHOSCOPE = (
    "objArc02 objBox01 objChamferBox01 objChamferBox02 objChamferCyl01 objChamferCyl02 "
    "objChamferCyl03 objChamferCyl04 objCircle02 objCurve01 objCurve02 objCylinder01 "
    "objCylinder02 objCylinder03 objCylinder04 objLine01 objLine02 objLine03 objLine04 objLine05 "
    "objLine08 objRectangle01 objTube01 objTube02 objрезьба objстекло"
).split()
WARD_POSE = {
    "usd": "assets/hospital/ward_02_preview_left.usdc",
    "translate": [-2.63, 0.5, -0.07],
    "yaw": 0.0,
    "hide": STETHOSCOPE,
}
WORK_SURFACE = {"center": [0.53, -0.30, 0.70], "size": [0.56, 1.40, 0.1]}
# Everything stays in front of the open shelf (robot x >= 0.6, z >= 1.11) so no arm sweeps
# through it. The readout is a small station display standing at the back of the counter behind
# the center rack, facing the rig's head camera (RoboDojo rig, 2026-11-03: past the right tray at
# (0.48, -0.85) it stood beyond the right edge of the 60-degree head view, and full size here its
# top would hide under the wall unit).
BOARD_XY = (0.70, -0.12)
# Built at full size in local coordinates (local z 0.15 is the counter top, the panel spans
# 0.18..0.54) and scaled down: the wall unit over the counter hides anything above ~1.0 m here.
BOARD_SCALE = 0.62
BOARD_Z = 0.75 - 0.15 * BOARD_SCALE  # group origin
DIAL_XY = (0.68, -0.50)  # SHIFT dial on the counter, behind the right tray
CAMERA = {"position": [-0.20, -0.16, 1.75], "look_at": [0.46, -0.28, 0.80]}


def facing_yaw(xy, camera=None):
    """Degrees about z that turn a +y-facing card at xy toward a camera (default: the C4 task camera)."""
    camera = CAMERA["position"] if camera is None else camera
    dx, dy = camera[0] - xy[0], camera[1] - xy[1]
    return float(np.degrees(np.arctan2(dy, dx)) - 90.0)


def well_name(rack, side, k):
    return f"rack_{CLASSES[rack][0]}_{side[0]}{k}"


def _tray(b, side, n_side):
    """One open-cell foam arrival tray per arm, holding that arm's tubes upright."""
    z, sign = b.z, SIGN[side]
    ys = sorted(SPLIT_Y + sign * dy for dy in ARRIVAL_DY[: -(-n_side // len(ARRIVAL_X))])
    x0, x1 = ARRIVAL_X[0] - 0.03, ARRIVAL_X[-1] + 0.03
    y0, y1 = ys[0] - 0.03, ys[-1] + 0.03
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    b.fixture(
        f"c4_tray_{side}_base",
        (cx, cy),
        (x1 - x0 + 0.012, y1 - y0 + 0.012, ARRIVAL_BASE),
        "polymer",
    )
    top = z + ARRIVAL_BASE
    for i, x in enumerate([x0, *[(a + c) / 2 for a, c in zip(ARRIVAL_X, ARRIVAL_X[1:])], x1]):
        b.fixture(
            f"c4_tray_{side}_bar_x{i}",
            (x, cy),
            (0.006, y1 - y0 + 0.006, ARRIVAL_BAR),
            "cabinet_white",
            z=top,
        )
    for j, y in enumerate([y0, *[(a + c) / 2 for a, c in zip(ys, ys[1:])], y1]):
        b.fixture(
            f"c4_tray_{side}_bar_y{j}",
            (cx, y),
            (x1 - x0 + 0.006, 0.006, ARRIVAL_BAR),
            "cabinet_white",
            z=top,
        )
    return [(x, y) for y in ys for x in ARRIVAL_X], top


def arm_half(config):
    """Half ("left" = +y of the split, "right" = -y) each arm works, from its rig mount."""
    from roboclock.envs.lab.robot import mount_positions

    mounts = mount_positions(config, "C4")
    halves = ["left" if float(m[1]) > SPLIT_Y else "right" for m in mounts]
    if sorted(halves) != ["left", "right"]:
        raise ValueError(f"C4: the two arm mounts must straddle the split plane y = {SPLIT_Y}")
    return halves


def build_layout(builder):
    b = builder
    b.catalog.setdefault(TUBE, dict(TUBE_SPEC))
    n = b.task.spec.num_items
    per_side = len(ARRIVAL_X) * len(ARRIVAL_DY)
    if n % 2 or not 2 <= n <= 2 * min(per_side, 3 * len(WELL_DY)):
        raise ValueError(f"C4 needs an even tube count of 2..{2 * per_side}, got {n}")
    tube_class, tube_side, index = {}, {}, 0
    for side in SIDES:
        cells, top = _tray(b, side, n // 2)
        order = b.rng.permutation(len(cells))[: n // 2]
        classes = b.rng.permutation(np.arange(n // 2) % 3)  # balanced: a rack half never overflows
        for k in range(n // 2):
            name = f"tube_{index}"
            index += 1
            color = CLASSES[classes[k]][3]
            b.item(name, TUBE, cells[order[k]], top, color=color, label="", yaw=0.0)
            tube_class[name] = CLASSES[classes[k]][0]
            tube_side[name] = side
    well_class, well_side, wells = {}, {}, {}
    for r, (key, _, _, _) in enumerate(CLASSES):
        for side in SIDES:
            for k, dy in enumerate(WELL_DY):
                name = well_name(r, side, k)
                y = SPLIT_Y + SIGN[side] * dy
                b.target(name, "well", (RACK_X[r], y), (WELL_HALF, WELL_HALF), WELL_WALL)
                well_class[name], well_side[name] = key, side
                wells.setdefault(key, []).append(name)
    b.indicator("indicator", DIAL_XY, "SHIFT")
    b.info.update(
        {
            "bedside_task": "ward_c4",
            "ward_pose": {**WARD_POSE, "hide": list(STETHOSCOPE)},
            "hide_bench": True,
            "work_surface": {k: list(v) for k, v in WORK_SURFACE.items()},
            "ward_c4": {
                "tube_class": tube_class,
                "tube_side": tube_side,
                "well_class": well_class,
                "well_side": well_side,
                "wells": wells,
                "split_y": float(SPLIT_Y + b.shift[1]),
                "arm_half": arm_half(b.config),
                "classes": [list(c[:3]) + [list(c[3])] for c in CLASSES],
                "arrival": {name: b.objects[name].position.tolist() for name in tube_class},
                "board": [*BOARD_XY, BOARD_Z],
                "camera": {k: list(v) for k, v in CAMERA.items()},
                "assumptions": [
                    "tubes stand upright in two open-cell arrival trays, one per arm",
                    "generous funnel-mouthed wells, no press fit",
                    "primitive tube and rack proxies",
                ],
            },
        }
    )


# --- appearance ---------------------------------------------------------------------------


class ShiftBar:
    """A long countdown bar on the readout board; the task drives it with timer commands."""

    COLORS = {
        "green": (0.10, 0.85, 0.30),
        "amber": (1.00, 0.60, 0.05),
        "red": (0.95, 0.10, 0.10),
        "blue": (0.10, 0.35, 1.00),
    }

    def __init__(self, stage, look, path, center, length, height):
        from pxr import Gf, UsdGeom

        self.center, self.length, self.height = np.array(center, dtype=float), length, height
        self.materials = {
            k: look.material(f"c4_bar_{k}", color=c, roughness=0.4, emissive=True)
            for k, c in self.COLORS.items()
        }
        self.cube = UsdGeom.Cube.Define(stage, path)
        self.cube.CreateSizeAttr(1.0)
        self.translate = self.cube.AddTranslateOp()
        self.scale = self.cube.AddScaleOp()
        self.color = None
        self.translate.Set(Gf.Vec3d(*self.center))
        self.set(1.0, "green")

    def set(self, fraction, color):
        from pxr import Gf, UsdShade

        fraction = min(max(float(fraction), 0.0), 1.0)
        length = max(fraction * self.length, 1e-4)
        left = self.center[0] - self.length / 2
        self.translate.Set(Gf.Vec3d(left + length / 2, self.center[1], self.center[2]))
        self.scale.Set(Gf.Vec3f(length, 0.004, self.height))
        if color != self.color:
            self.color = color
            UsdShade.MaterialBindingAPI.Apply(self.cube.GetPrim()).Bind(
                self.materials[color].material
            )

    def state(self, color):
        pass


def rate_steps(rate_max=30.0):
    return [round(0.5 * k, 1) for k in range(int(rate_max / 0.5) + 1)]


def _bright(color, scale=1.0):
    """A tint for the mid-grey plastic_white maps: the hue of `color`, its largest channel at 1."""
    color = np.asarray(color, dtype=float)
    return [float(min(1.0, scale * c)) for c in color / max(float(color.max()), 1e-6)]


def _bind(stage, path, material):
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material)


def _dress_trays(backend):
    """Arrival trays: the outer frame bars are the white plastic tray shell, the base plate and
    the inner cell walls the charcoal open-cell foam insert (visual rebinding only)."""
    look, stage = backend.appearance, backend.stage
    shell = look.textured("c4_tray_shell", "plastic_matte", color=(0.97, 0.98, 0.99), roughness=0.4)
    foam = look.textured("c4_tray_foam", "foam", color=(0.11, 0.12, 0.13), repeats=10.0)
    bars = {}
    for fixture in backend.layout.fixtures:
        if not fixture.name.startswith("c4_tray_"):
            continue
        side, part = fixture.name[len("c4_tray_"):].split("_", 1)
        if part == "base":
            _bind(stage, f"/World/Fixtures/{fixture.name}_Visual", foam)
        else:
            axis, index = part.split("_")[1][0], int(part.split("_")[1][1:])
            bars.setdefault((side, axis), []).append((index, fixture.name))
    for group in bars.values():
        last = max(i for i, _ in group)
        for index, name in group:
            _bind(stage, f"/World/Fixtures/{name}_Visual", shell if index in (0, last) else foam)


def author(backend):
    from pxr import UsdGeom, UsdShade

    stage, look, layout = backend.stage, backend.appearance, backend.layout
    info = layout.info["ward_c4"]
    z = layout.table_height
    labels = backend.config["visuals"]["labels"]
    colors = {c[0]: c[3] for c in info["classes"]}
    additive = {c[0]: c[2] for c in info["classes"]}
    spec = TUBE_SPEC
    r, h = spec["radius"], spec["height"]

    # Tubes: frosted plastic wall, blood column (flat), paper label wrap with the printed label,
    # glossy colored plastic stopper.
    blood = look.material("c4_blood", color=(0.38, 0.02, 0.04), roughness=0.25)
    frosted = look.textured(
        "c4_tube_wall", "plastic_matte", color=(0.74, 0.80, 0.86), roughness=0.22, repeats=40.0
    )
    wrap = look.textured("c4_label_wrap", "paper", repeats=30.0)
    for name, key in info["tube_class"].items():
        path = f"/World/Objects/{name}"
        for child in ("Wall", "Rim", "Identity"):
            prim = stage.GetPrimAtPath(f"{path}/{child}")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        cap = look.textured(
            f"c4_cap_{key}", "plastic_white", color=_bright(colors[key]), roughness=0.2, repeats=40
        )
        body_h = h - 0.020
        look.cylinder_child(path, "TubeWall", r - 0.0012, body_h, -h / 2 + body_h / 2, frosted)
        look.cylinder_child(path, "TubeBottom", r - 0.0025, 0.004, -h / 2 + 0.002, frosted)
        look.cylinder_child(path, "BloodColumn", r - 0.0008, 0.050, -h / 2 + 0.030, blood)
        look.cylinder_child(path, "LabelWrap", r - 0.0005, 0.036, -0.004, wrap)
        look.cylinder_child(path, "Stopper", r, 0.018, h / 2 - 0.009, cap)
        look.cylinder_child(path, "StopperTop", r * 0.62, 0.003, h / 2 - 0.0005, cap)
        if labels:
            look.label(
                f"{path}/TubeLabel",
                additive[key],
                [0, 0, -0.004],
                [0.030, 0.034],
                colors[key],
                cylindrical_radius=r - 0.0002,
            )
    _dress_trays(backend)

    # Racks: each well's walls in the rack color, a funnel collar, base plate and title card,
    # all glossy molded plastic in the class color (the funnel lips a lighter shade).
    for key, wells in info["wells"].items():
        tint = _bright(colors[key])
        plastic = look.textured(
            f"c4_rack_{key}", "plastic_white", color=tint, roughness=0.35, repeats=20
        )
        collar = look.textured(
            f"c4_collar_{key}",
            "plastic_white",
            color=[0.45 + 0.55 * c for c in tint],
            roughness=0.3,
            repeats=20.0,
        )
        for well in wells:
            for part in ("floor", "left", "right", "front", "back"):
                prim = stage.GetPrimAtPath(f"/World/Targets/{well}/{part}_Visual")
                if prim.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        (look.material("cabinet_white") if part == "floor" else plastic).material
                    )
            t = layout.targets[well]
            cx, cy, rim = float(t.position[0]), float(t.position[1]), float(t.rim_z)
            # Funnel mouth: four outward-sloped lips above the rim (visual only, no collider).
            for k, (dx, dy, ry, rx) in enumerate(
                ((1, 0, 35, 0), (-1, 0, -35, 0), (0, 1, 0, -35), (0, -1, 0, 35))
            ):
                lip = look.group(
                    f"/World/C4/Funnel_{well}_{k}",
                    [cx + dx * (WELL_HALF + 0.006), cy + dy * (WELL_HALF + 0.006), rim + 0.004],
                    rx=rx,
                    ry=ry,
                )
                size = (
                    [0.003, 2 * WELL_HALF + 0.016, 0.010]
                    if dx
                    else [2 * WELL_HALF + 0.016, 0.003, 0.010]
                )
                look.rounded(f"{lip}/Lip", [0, 0, 0], size, collar, 0.0008)
        xs = [float(layout.targets[w].position[0]) for w in wells]
        ys = [float(layout.targets[w].position[1]) for w in wells]
        rack_cx, rack_cy = float(np.mean(xs)), float(np.mean(ys))
        length = max(ys) - min(ys) + 2 * WELL_HALF + 0.03
        look.rounded(
            f"/World/C4/RackBase_{key}",
            [rack_cx, rack_cy, z + 0.002],
            [2 * WELL_HALF + 0.03, length, 0.004],
            plastic,
            0.002,
        )
        # Label block in the middle gap between the two arms' halves, turned toward the camera.
        at = (rack_cx, float(info["split_y"]))
        card = look.group(f"/World/C4/RackCard_{key}", [*at, z], rz=facing_yaw(at))
        look.rounded(f"{card}/Stand", [0, -0.004, 0.038], [0.11, 0.008, 0.076], plastic, 0.002)
        if labels:
            look.label(f"{card}/Title", SHORT[key], [0, 0.0005, 0.046], [0.104, 0.052], colors[key])

    # One arrival label per tray, flat on the counter in front of it, reading from the camera.
    for side in SIDES:
        tray = [layout.objects[n].position for n, s in info["tube_side"].items() if s == side]
        at = (float(min(p[0] for p in tray)) - 0.058, float(np.mean([p[1] for p in tray])))
        if labels:
            card = look.group(f"/World/C4/ArrivalCard_{side}", [*at, z + 0.0015], rz=90.0)
            flat = look.group(f"{card}/Flat", [0.0, 0.0, 0.0], rx=90.0)
            look.label(
                f"{flat}/Title",
                "STAT ARRIVALS",
                [0, 0, 0],
                [0.15, 0.036],
                (0.75, 0.10, 0.10),
            )

    _mobile_base(backend)

    # Readout board: shift countdown bar, live sorted count and live rate.
    task = backend.task
    bx, by, bz = info["board"]
    # Station display standing on the counter, turned toward the camera.
    from roboclock.envs.hospital.rig import head_camera_pose

    head = head_camera_pose(backend.config, backend.task.task_id)[0]
    board = look.group("/World/C4/Board", [bx, by, bz], rz=facing_yaw((bx, by), head))
    from pxr import Gf

    UsdGeom.Xformable(stage.GetPrimAtPath(board)).AddScaleOp().Set(Gf.Vec3f(*[BOARD_SCALE] * 3))
    look.rounded(f"{board}/Panel", [0, -0.012, 0.36], [0.32, 0.012, 0.36], "screen_dark", 0.004)
    # Powder-coated bezel around the screen, a VESA plate and a brushed aluminum mast on the foot.
    coat = look.textured("c4_display_bezel", "powder_coat_dark", repeats=10.0)
    look.rounded(f"{board}/Bezel", [0, -0.0135, 0.36], [0.336, 0.012, 0.376], coat, 0.005)
    look.rounded(f"{board}/Mount", [0, -0.027, 0.38], [0.10, 0.014, 0.10], coat, 0.003)
    mast = look.textured("c4_display_mast", "aluminum_brushed", repeats=12.0)
    look.cylinder(f"{board}/Mast", [0, -0.045, 0.295], 0.012, 0.25, mast)
    look.rounded(f"{board}/Foot", [0, -0.03, 0.16], [0.26, 0.08, 0.02], "stainless", 0.004)
    look.rounded(f"{board}/Strut", [0, -0.02, 0.17], [0.20, 0.012, 0.03], "stainless", 0.002)
    backend.timer_bars["c4_shift_bar"] = ShiftBar(
        stage, look, f"{board}/ShiftBar", [0, -0.004, 0.515], 0.29, 0.026
    )
    look.rounded(
        f"{board}/BarTrack",
        [0, -0.0055, 0.515],
        [0.296, 0.002, 0.034],
        look.material("c4_track", color=(0.04, 0.04, 0.05), roughness=0.6),
        0.001,
    )
    if labels:
        look.label(
            f"{board}/Header", "TUBE TRIAGE", [0, -0.0055, 0.465], [0.29, 0.05], (0.10, 0.30, 0.55)
        )
        # score_version 2: the goal is every tube, as fast as possible (no rate, no seconds).
        look.label(
            f"{board}/Goal",
            f"GOAL: ALL {task.spec.num_items} TUBES",
            [0, -0.0055, 0.29],
            [0.29, 0.06],
            (0.85, 0.45, 0.05),
        )
    # The live readouts are always built: the task toggles them with show commands.
    n = len(info["tube_class"])
    for k in range(n + 1):
        prim = look.label(
            f"{board}/Count_{k}",
            f"SORTED {k}/{n}",
            [0.078, -0.0055, 0.375],
            [0.145, 0.075],
            (0.10, 0.60, 0.25),
        ).GetPrim()
        backend.zone_prims[f"c4_count_{k}"] = prim
        if k:
            UsdGeom.Imageable(prim).MakeInvisible()
    for k, value in enumerate(rate_steps()):
        prim = look.label(
            f"{board}/Rate_{k}",
            f"{value:.1f}/min",
            [-0.078, -0.0055, 0.375],
            [0.145, 0.075],
            (0.10, 0.60, 0.25),
        ).GetPrim()
        backend.zone_prims[f"c4_rate_{k}"] = prim
        if k:
            UsdGeom.Imageable(prim).MakeInvisible()


# --- measurement --------------------------------------------------------------------------


def tube_axis(obj):
    return rotation_matrix(obj.orientation)[:, 2]


def in_well(shape, obj, target, *, lenient=False):
    """Bottom of the tube on the well floor, inside the opening and upright (measured envelope)."""
    axis = tube_axis(obj)
    tilt = float(np.arccos(np.clip(axis[2], -1.0, 1.0)))
    bottom = obj.position - axis * shape.height / 2
    lowest = bottom[2] - shape.radius * np.sin(tilt)
    margin = -0.004 if lenient else 0.0
    inside = bool(
        np.all(np.abs(bottom[:2] - target.position[:2]) <= target.half_extents[:2] - margin - 0.002)
    )
    on_floor = (
        target.support_z - 0.004 <= lowest <= target.support_z + (0.020 if lenient else 0.008)
    )
    return inside and on_floor and tilt <= UPRIGHT_TOLERANCE


def _mobile_base(backend):
    """The X5 columns stand on one wheeled base on the ward floor (visual; the robot is fixed)."""
    from roboclock.envs.lab.robot import mount_positions

    look, z = backend.appearance, backend.layout.table_height
    mounts = [np.asarray(m[:2], dtype=float) for m in mount_positions(backend.config, "C4")]
    lo, hi = np.min(mounts, axis=0) - 0.13, np.max(mounts, axis=0) + 0.13
    center, size = (lo + hi) / 2, hi - lo
    # One mobile robot: matte light grey plastic shells (the columns match the textured arm
    # mounts above them), a powder-coated skirt and column plinths, brushed aluminum deck trim,
    # collars and hubs, rubber casters.
    shell = look.textured("c4_base_shell", "plastic_matte", color=(0.84, 0.86, 0.88), roughness=0.5)
    coat = look.textured("robot_mount_coat", "plastic_matte", color=(0.86, 0.88, 0.90))
    dark = look.textured("c4_base_skirt", "powder_coat_dark", repeats=8.0)
    trim = look.textured("c4_base_trim", "aluminum_brushed", repeats=12.0)
    look.rounded("/World/C4/MobileBase", [*center, 0.075], [*size, 0.11], shell, 0.02)
    look.rounded("/World/C4/MobileBumper", [*center, 0.030], [*(size + 0.012), 0.03], dark, 0.01)
    look.rounded("/World/C4/MobileDeck", [*center, 0.1305], [*(size - 0.03), 0.003], trim, 0.001)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        wheel = [center[0] + sx * (size[0] / 2 - 0.05), center[1] + sy * (size[1] / 2 - 0.05), 0.03]
        look.cylinder(f"/World/C4/Wheel_{i}", wheel, 0.03, 0.025, "rubber", axis="Y")
        hub = [wheel[0], wheel[1] + sy * 0.0135, wheel[2]]
        look.cylinder(f"/World/C4/Hub_{i}", hub, 0.014, 0.003, trim, axis="Y")
    for i, m in enumerate(mounts):
        look.rounded(
            f"/World/C4/Column_{i}", [*m, (0.13 + z) / 2], [0.12, 0.12, z - 0.13], coat, 0.006
        )
        look.rounded(f"/World/C4/Plinth_{i}", [*m, 0.15], [0.15, 0.15, 0.04], dark, 0.006)
        # A collar where the physics mount column (above the counter height) continues it.
        look.rounded(f"/World/C4/Collar_{i}", [*m, z], [0.136, 0.136, 0.03], trim, 0.006)


def _camera(backend):
    """Room-side view of the robot, tray, racks and wall display; set once per reset."""
    camera = getattr(backend, "task_camera", None)  # the third-person recording view
    if camera is None:
        return
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    view = backend.layout.info["ward_c4"]["camera"]
    position = np.asarray(view["position"]) + backend.layout.camera_jitter
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*position.tolist()), Gf.Vec3d(*view["look_at"]), Gf.Vec3d(0, 0, 1)
    )
    orientation = gf_quat_to_np_array(matrix.GetInverse().ExtractRotationQuat())
    camera.set_world_pose(position, orientation, camera_axes="usd")
    for _ in range(3):
        backend.world.render()


def measure(backend, state):
    camera = getattr(backend, "camera", None)
    if camera is not None and getattr(backend, "_c4_camera_for", None) is not backend.task:
        backend._c4_camera_for = backend.task
        _camera(backend)
    layout = backend.layout
    occupancy = {}
    for name, obj in state.objects.items():
        for well, target in layout.targets.items():
            if in_well(layout.shapes[name], obj, target, lenient=True):
                occupancy[well] = name
    return {"ward_c4": {"occupancy": occupancy}}


# --- privileged diagnostic oracle ---------------------------------------------------------

MAX_LEAD = 0.05  # m: the commanded tool point never runs further ahead of the measured one
AIM_DEPTH = WELL_WALL / 2  # the tube axis is centered in the opening at half the well depth
ALIGN_TOL = 0.0035  # m: the aim point is over the well center before the tube goes down
SETTLED_TILT = 0.06  # rad: a carried tube hangs at most this far off vertical before it goes down
SLIP_TILT = 0.035  # rad: a tube this far off the tool axis has pivoted between the fingers
BLOCKED_TILT = 0.15  # rad: a tube going down tips this far only when it sits on the rim
LOST_DRIFT = 0.05  # m: the tube has left the hand (not a pivot or slide between the fingers)
PHASE_TIMEOUT = 6.0  # s


def down_quat(yaw=0.0):
    """Tool orientation pointing straight down at `yaw` (Euler (0, pi, yaw); scalar-first)."""
    return np.array([0.0, -np.sin(yaw / 2), np.cos(yaw / 2), 0.0])


def tilt_correction(tool_orientation, tube_up, yaw=0.0):
    """(quaternion, angle): the world-frame turn of a pointing-down tool at `yaw` that stands a tube
    held at a tilt between its fingers upright, and the tilt between tube and tool axis.

    The tube's axis is measured in the tool frame, so the result does not depend on how far the tool
    is already turned: applied to the pointing-down orientation, it brings the axis to vertical.
    """
    held = rotation_matrix(tool_orientation).T @ np.asarray(tube_up, dtype=float)
    up = rotation_matrix(down_quat(yaw)) @ held  # tube axis with the tool exactly pointing down
    up = up / np.linalg.norm(up)
    angle = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
    axis = np.cross(up, [0.0, 0.0, 1.0])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]), angle
    axis = axis / norm
    return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))]), angle


def aim_point(obj):
    """Point on the tube axis AIM_DEPTH above its bottom: centered in the well opening, it keeps a
    tube that leans a little clear of both the rim and the floor edge on the way down."""
    return obj.position - tube_axis(obj) * (TUBE_SPEC["height"] / 2 - AIM_DEPTH)


def tube_bottom(obj):
    return obj.position - tube_axis(obj) * TUBE_SPEC["height"] / 2


def tube_tilt(obj):
    return float(np.arccos(np.clip(tube_axis(obj)[2], -1.0, 1.0)))


def _occupied(task, state, well, ignore=None):
    """Something rests in or across the well's opening: any part of another tube's axis below the
    height of a carried tube, within the opening (a tube lying over the rack counts too)."""
    target = task.layout.targets[well]
    reach = target.half_extents[:2] + TUBE_SPEC["radius"]
    for name, obj in state.objects.items():
        if name == ignore:
            continue
        axis = tube_axis(obj)
        for s in np.linspace(-0.5, 0.5, 5) * TUBE_SPEC["height"]:
            point = obj.position + axis * s
            if point[2] < target.rim_z + 0.06 and np.all(
                np.abs(point[:2] - target.position[:2]) < reach
            ):
                return True
    return False


def _inner(task, well):
    return (
        abs(task.layout.targets[well].position[1] - task.layout.info["ward_c4"]["split_y"]) < 0.12
    )


def _drive(planner, task, state, ee, side, tool_orientation=None):
    """Scripted pick-and-place for one arm: every waypoint becomes bounded joint commands via IK.

    Each arm only takes tubes from its own tray and fills its own half of each rack. The halves meet
    at the split plane, so the two inner wells are guarded by a lock: a hand does not descend into an
    inner well while the other hand works an inner well within INNER_LOCK_DX in x.

    The tube is held 2 cm under its stopper and hangs 8 cm below the fingers, so the carry must keep
    the tool pointing straight down: the commanded tool point stays within MAX_LEAD of the measured
    one and all joints step together (``sync_joints``). A tube that has pivoted between the fingers
    anyway is stood up by turning the tool (``tool_tilt``) before it goes into the well; it goes down
    with its axis centered in the opening, and one that catches the rim is lifted out and lowered
    again instead of being let go there. The hand never opens over the racks with the tube held.
    """
    info = task.layout.info["ward_c4"]
    now, z = task.time, task.layout.table_height
    ee = np.asarray(ee, dtype=float)
    if getattr(planner, "_c4_task", None) is not task:
        planner._c4_task = task
        planner.c4_phase, planner.c4_time = "select", now
        planner.c4_tube = planner.c4_well = None
        planner.c4_tries, planner.c4_wells, planner.c4_released = {}, {}, {}
        planner.c4_inner = None
        planner.c4_lowers = 0
        planner.c4_aligned = None
        planner.tool_tilt = None
    planner.sync_joints = True
    phase = planner.c4_phase
    planner.step_yaw(0.0)
    r = TUBE_SPEC["radius"]
    planner.open_width = r + 0.012
    planner.close_width = max(0.0, r - TUBE_SPEC["grip_force"] / 1000)
    planner.gentle = True
    travel = z + TRAVEL_Z
    close, speed = False, 0.55
    goal = np.array([*ee[:2], max(ee[2], travel)])
    partner = getattr(planner, "partner", None)

    def go(value):
        planner.c4_phase, planner.c4_time = value, now
        task.event_log.append(
            {
                "time": round(now, 3),
                "type": "c4_phase",
                "arm": side,
                "phase": value,
                "tube": planner.c4_tube,
                "well": planner.c4_well,
            }
        )

    def finish(goal, close, speed):
        # Lead-limited: a far goal is approached along the arm's own pace, not a target running
        # away from it (the per-step joint bound would then bend the path and tip the tool).
        lead = planner.last_goal - ee
        distance = float(np.linalg.norm(lead))
        if distance > MAX_LEAD:
            planner.last_goal = ee + lead * (MAX_LEAD / distance)
        return planner.finish(state, np.asarray(goal, dtype=float), close, speed)

    def near(point, tol):
        return float(np.linalg.norm(ee - point)) < tol

    def holding(name):
        held = getattr(planner, "held", None)
        return name in held if held is not None else state.objects[name].grasped

    name = planner.c4_tube
    obj = state.objects.get(name) if name else None

    def lost():
        """The tube has left the hand (a pivot, slide or grasp-flag flicker between the fingers is
        not a loss: the hand then keeps it and carries on)."""
        drift = float(np.linalg.norm(ee - obj.position - planner.c4_offset))
        if drift > LOST_DRIFT:
            other = getattr(partner, "ee_position", None)
            task.event_log.append(
                {
                    "time": round(now, 3),
                    "type": "c4_lost",
                    "arm": side,
                    "drift": round(drift, 4),
                    "ee": [round(float(v), 3) for v in ee],
                    "tube": [round(float(v), 3) for v in obj.position],
                    "held": holding(name),
                    "partner_ee": None if other is None else [round(float(v), 3) for v in other],
                    "partner_phase": getattr(partner, "c4_phase", None),
                }
            )
            return True
        return False

    def blocked(well):
        """The partner hand works an inner well close in x: wait above, at travel height."""
        other = getattr(partner, "c4_inner", None) if partner is not None else None
        if other is None or not _inner(task, well):
            return False
        dx = task.layout.targets[well].position[0] - task.layout.targets[other].position[0]
        return abs(dx) < INNER_LOCK_DX

    def stand_up():
        """Turn the tool so a tube that has pivoted between the fingers hangs vertical again."""
        if tool_orientation is None:
            return
        correction, slip = tilt_correction(tool_orientation, tube_axis(obj), planner.yaw)
        if planner.tool_tilt is None and slip <= SLIP_TILT:
            return
        if planner.tool_tilt is None:
            planner.c4_tilt = np.zeros(3)
            task.event_log.append(
                {
                    "time": round(now, 3),
                    "type": "c4_stand_up",
                    "arm": side,
                    "tube": name,
                    "slip": round(slip, 3),
                }
            )
        # Low-pass the turn (as a rotation vector) so the tool does not chase a swinging tube.
        w = float(np.clip(correction[0], -1.0, 1.0))
        vector = correction[1:] * (2 * np.arccos(w) / max(np.sqrt(1 - w * w), 1e-9))
        planner.c4_tilt = planner.c4_tilt + 0.2 * (vector - planner.c4_tilt)
        angle = float(np.linalg.norm(planner.c4_tilt))
        axis = planner.c4_tilt / angle if angle > 1e-9 else np.zeros(3)
        planner.tool_tilt = np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])

    def retry(reason):
        """Lift the tube out and line it up again (it is never let go over the rack)."""
        planner.c4_lowers += 1
        planner.c4_inner = None
        task.event_log.append(
            {
                "time": round(now, 3),
                "type": "c4_retry",
                "arm": side,
                "tube": name,
                "reason": reason,
                "tilt": round(tube_tilt(obj), 3),
            }
        )
        go("transfer")

    if phase == "select":
        planner.tool_tilt = None
        reserved = set(planner.c4_wells.values())
        options = []
        for tube, key in info["tube_class"].items():
            o = state.objects[tube]
            if (
                info["tube_side"][tube] != side
                or tube in task.correct  # a tube resting in a wrong well is taken out again
                or tube in task.dropped
                or o.grasped
                or planner.c4_tries.get(tube, 0) >= 3
                or now - planner.c4_released.get(tube, -9.0) < 1.0  # still settling
                or o.position[2] < z + 0.03
                or tube_tilt(o) > 0.3
            ):
                continue
            free = [
                w
                for w in info["wells"][key]
                if info["well_side"][w] == side
                and w not in reserved
                and not _occupied(task, state, w, ignore=tube)
            ]
            if free:
                options.append((float(np.linalg.norm(o.position[:2] - ee[:2])), tube, free[0]))
        if options:
            _, planner.c4_tube, planner.c4_well = min(options)
            planner.c4_wells[planner.c4_tube] = planner.c4_well
            planner.c4_lowers = 0
            go("approach")
        else:
            # Nothing left for this hand: park above its own tray instead of drifting with the arm.
            tray = [info["arrival"][t] for t, s in info["tube_side"].items() if s == side]
            goal = np.array([*np.mean(tray, axis=0)[:2], travel + 0.05])
        return finish(goal, False, 0.4)
    target = task.layout.targets[planner.c4_well]
    pick = obj.position + np.array([0.0, 0.0, GRASP_DZ])
    if phase == "approach":
        goal = np.array([*pick[:2], travel])
        if np.linalg.norm(ee[:2] - pick[:2]) < 0.006 and abs(ee[2] - travel) < 0.03:
            go("descend")
    elif phase == "descend":
        goal, speed = pick, 0.30
        if np.linalg.norm(ee[:2] - pick[:2]) > 0.02:
            go("approach")
        elif near(pick, 0.006):
            go("grasp")
    elif phase == "grasp":
        goal, close = pick, True
        if holding(name) and now - planner.c4_time > 0.15:
            planner.c4_pick_xy = pick[:2].copy()
            planner.c4_offset = ee - obj.position
            go("lift")
        elif now - planner.c4_time > 1.5:
            planner.c4_tries[name] = planner.c4_tries.get(name, 0) + 1
            go("abort")
    elif phase == "lift":
        goal, close, speed = np.array([*planner.c4_pick_xy, travel]), True, 0.45
        if lost():
            go("abort")
        elif ee[2] > travel - 0.02:
            planner.c4_aligned = None
            go("transfer")
    elif phase == "transfer":
        stand_up()
        aim = aim_point(obj)
        wait = blocked(planner.c4_well)
        # While the partner works a nearby inner well, hover a little toward our own side.
        aside = np.array([0.0, SIGN[side] * 0.08]) if wait else np.zeros(2)
        goal = np.array([*(target.position[:2] + ee[:2] - aim[:2] + aside), travel])
        close, speed = True, 0.40
        over = (
            np.linalg.norm(aim[:2] - target.position[:2]) < ALIGN_TOL
            and abs(ee[2] - travel) < 0.03
            and not wait
        )
        if not over:
            planner.c4_aligned = None
        elif planner.c4_aligned is None:
            planner.c4_aligned = now
        # Go down once the tube hangs upright (or after a second over the well trying to get there).
        settled = over and (tube_tilt(obj) < SETTLED_TILT or now - planner.c4_aligned > 1.0)
        if lost():
            go("abort")
        elif settled:
            if _inner(task, planner.c4_well):
                planner.c4_inner = planner.c4_well
            planner.c4_grip = float(ee[2] - tube_bottom(obj)[2])
            go("lower")
    elif phase == "lower":
        stand_up()
        aim, bottom = aim_point(obj), tube_bottom(obj)
        floor = target.support_z + 0.004
        goal = ee + np.array([*(target.position[:2] - aim[:2]), floor - bottom[2]])
        close, speed = True, 0.25
        above_floor = bottom[2] > target.support_z + 0.012
        slid = ee[2] - bottom[2] < planner.c4_grip - 0.006  # pushed up between the fingers
        if above_floor and (slid or tube_tilt(obj) > BLOCKED_TILT):
            retry("slid" if slid else "tipped")
        elif bottom[2] < target.support_z + 0.008 and tube_tilt(obj) < 0.2:
            planner.c4_release = ee.copy()
            go("release")
    elif phase == "release":
        goal, close = planner.c4_release.copy(), False
        if now - planner.c4_time > 0.30:
            planner.c4_released[name] = now
            go("retreat")
    elif phase in ("retreat", "abort"):
        goal = np.array([*ee[:2], travel])
        if ee[2] > travel - 0.03:
            planner.c4_inner = None
            planner.c4_wells.pop(name, None)
            if phase == "abort":
                planner.c4_tries[name] = planner.c4_tries.get(name, 0) + 1
            planner.c4_tube = None
            go("select")
    waiting = planner.c4_phase == "transfer" and blocked(planner.c4_well)
    if (
        now - planner.c4_time > PHASE_TIMEOUT
        and planner.c4_phase not in ("retreat", "abort", "select")
        and not waiting
    ):
        stuck = planner.c4_phase
        task.event_log.append(
            {
                "time": round(now, 3),
                "type": "c4_timeout",
                "arm": side,
                "phase": stuck,
                "ee": [round(float(v), 3) for v in ee],
                "goal": [round(float(v), 3) for v in np.asarray(goal, dtype=float)],
                "tube": None if obj is None else [round(float(v), 3) for v in obj.position],
            }
        )
        # Holding the tube, the hand carries on (a tube is never let go in the air or on a rim).
        if stuck == "lift":
            go("transfer")
        elif stuck == "transfer":
            planner.c4_grip = float(ee[2] - tube_bottom(obj)[2])
            go("lower")
        elif stuck == "lower":
            retry("timeout")
        else:
            go("abort")
    command = finish(goal, close, speed)
    if _TRACE and int(round(now / 0.1)) != getattr(planner, "_c4_traced", None):
        planner._c4_traced = int(round(now / 0.1))
        task.event_log.append(
            {
                "time": round(now, 3),
                "type": "c4_trace",
                "arm": side,
                "phase": planner.c4_phase,
                "ee": [round(float(v), 3) for v in ee],
                "cmd": [round(float(v), 3) for v in command.position],
                "yaw": round(float(planner.yaw), 3),
                "tilt": None if obj is None else round(tube_tilt(obj), 3),
            }
        )
    return command


def command(planner, task, state):
    """Arm 0 (ARXLeft, the rig's -y arm): the half of the split its mount stands on."""
    half = task.layout.info["ward_c4"].get("arm_half", SIDES)[0]
    orientation = getattr(state, "ee_orientation", None)
    return _drive(planner, task, state, state.ee_position, half, orientation)


def command_secondary(planner, task, state):
    """Arm 1 (ARXRight, the rig's +y arm), driven by its own planner and IK."""
    ee = planner.ee_position if planner.ee_position is not None else planner.last_goal
    half = task.layout.info["ward_c4"].get("arm_half", SIDES)[1]
    return _drive(planner, task, state, ee, half, getattr(planner, "ee_orientation", None))
