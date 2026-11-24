"""Seeded hospital scenes shared by the physical backend and the task predicates.

Everything the backend must build (objects, containers, supports, indicators, zones) is declared
here, so the Isaac scene code contains no per-task branching. Layout sampling uses its own random
stream and never depends on difficulty, so a seed keeps its geometry across levels.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from roboclock.envs.hospital.mdp.geometry import Shape, Zone, object_extents
from roboclock.envs.lab.mdp.state import Target

CATALOG_PATH = Path(__file__).parent / "assets/catalog.yaml"
# Medications a pharmacy request can ask for: (drug, dose).
MEDICATIONS = [
    ("METOPROLOL", "25 mg"),
    ("LISINOPRIL", "10 mg"),
    ("ATORVASTATIN", "40 mg"),
    ("METFORMIN", "500 mg"),
    ("FUROSEMIDE", "20 mg"),
    ("WARFARIN", "5 mg"),
    ("ASPIRIN", "81 mg"),
    ("AMLODIPINE", "5 mg"),
]

# What a surgeon says when each instrument is wanted; several phrasings, drawn per episode.
SITUATIONS = {
    "scalpel": ["MAKE INCISION", "OPEN THE SKIN", "DIVIDE TISSUE"],
    "forceps": ["HOLD TISSUE", "GRASP THE EDGE", "LIFT THE FLAP"],
    "scissors": ["CUT THE SUTURE", "TRIM TISSUE", "CUT THE THREAD"],
    "clamp": ["BLEEDING VESSEL", "PINCH OFF THE ARTERY", "STOP THE BLEED"],
    "retractor": ["EXPOSE THE WOUND", "SPREAD THE INCISION", "HOLD IT OPEN"],
}

PATIENT_COLORS = ((0.9, 0.2, 0.15), (0.15, 0.4, 0.95), (0.15, 0.75, 0.3), (0.95, 0.7, 0.1))
WALL = 0.006  # container wall thickness
FLOOR = 0.004  # container floor thickness


@dataclass(frozen=True)
class ObjectConfig:
    name: str
    asset: str
    position: np.ndarray
    orientation: np.ndarray
    color: np.ndarray
    mass: float
    friction: float
    label: str = ""


@dataclass(frozen=True)
class Fixture:
    """A static slab: raised pad, shelf, drape or post. Collides unless visual-only."""

    name: str
    position: np.ndarray
    size: np.ndarray
    material: str = "steel"
    label: str = ""
    collision: bool = True
    visible: bool = True  # False: a collision shape only, drawn by the scene dressing instead


@dataclass(frozen=True)
class Indicator:
    name: str
    position: np.ndarray
    label: str
    style: str = "dial"  # "dial" (clock face) or "beacon" (a flashing emergency lamp)


@dataclass(frozen=True)
class DrawerPart:
    """One rigid piece of a sliding drawer: center relative to the drawer origin."""

    name: str
    offset: np.ndarray
    size: np.ndarray
    material: str
    collision: bool = True


@dataclass(frozen=True)
class Drawer:
    """A drawer on a prismatic joint along +y, closed at `position`."""

    name: str
    position: np.ndarray
    travel: float
    mass: float
    parts: tuple


@dataclass(frozen=True)
class Tag:
    """A text card standing in the scene: slot names, step lists. Visual only."""

    text: str
    position: np.ndarray
    size: tuple[float, float]


@dataclass
class HospitalLayout:
    objects: dict[str, ObjectConfig]
    targets: dict[str, Target]
    assignments: dict[str, str]  # object -> destination, where the task defines one
    zones: dict[str, Zone]
    fixtures: list[Fixture]
    indicators: list[Indicator]
    held: frozenset[str]  # objects fixed in place until a release event
    shapes: dict[str, Shape]
    dimensions: dict[str, tuple[float, float]]  # (grasp radius, longest dimension) per object
    labels: dict[str, str]  # visible text for containers
    table_height: float
    light_intensity: float
    camera_jitter: np.ndarray
    workspace_bounds: tuple[np.ndarray, np.ndarray]
    item_types: dict[str, str] = field(default_factory=dict)  # C1/C2 item -> type key
    type_labels: dict[str, str] = field(default_factory=dict)
    orders: dict[str, dict[str, int]] = field(default_factory=dict)  # C1 bin -> type -> count
    bin_types: dict[str, str] = field(default_factory=dict)  # C2 bin -> accepted type
    alarm_station: int = 0  # B1: the first alarm's station
    alarm_stations: list = field(default_factory=list)  # B1: station of each alarm, in order
    task_id: str = ""  # selects the per-task background dressing
    tags: list = field(default_factory=list)  # Tag cards
    drawers: list = field(default_factory=list)  # sliding Drawer bodies
    info: dict = field(default_factory=dict)  # task geometry the controller needs (handle, hook)
    steps: list = field(default_factory=list)  # A3: instrument asset required by each step

    def keepouts(self, robot_xy):
        """XY envelopes (center, half extents) clutter must avoid."""
        regions = [(np.asarray(robot_xy, dtype=float), np.array([0.18, 0.18]))]
        for name, obj in self.objects.items():
            regions.append(
                (obj.position[:2], object_extents(self.shapes[name], obj.orientation)[:2])
            )
        for target in self.targets.values():
            regions.append((target.position[:2], target.half_extents[:2] + WALL))
        for fixture in self.fixtures:
            regions.append((fixture.position[:2], fixture.size[:2] / 2))
        for drawer in self.drawers:
            half = np.array([0.20, 0.17 + drawer.travel / 2])
            regions.append((drawer.position[:2] + np.array([0.0, drawer.travel / 2]), half))
        for indicator in self.indicators:
            regions.append((indicator.position[:2], np.array([0.04, 0.045])))
        for zone in self.zones.values():
            # Cups moved out of a spill need room beyond the zone itself.
            half = zone.half_extents if zone.radius is None else np.full(2, zone.radius + 0.12)
            regions.append((zone.center, half))
        from roboclock.envs.hospital.appearance import DECOR_KEEPOUTS

        for center, half in DECOR_KEEPOUTS:
            regions.append((np.array(center), np.array(half)))
        return regions


def load_catalog(path=None) -> dict:
    with open(path or CATALOG_PATH, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def make_shape(spec: dict) -> Shape:
    if spec["proxy"] in ("box", "pouch"):
        half = tuple(v / 2 for v in spec["size"])
        return Shape(
            "box",
            half=half,
            grasp_radius=spec["grasp_radius"],
            grasp_height=spec.get("grasp_height", max(spec["size"])),
            grip_force=spec.get("grip_force"),
        )
    return Shape(
        "cylinder",
        radius=spec["radius"],
        height=spec["height"],
        grasp_radius=spec["grasp_radius"],
        grasp_height=spec["height"],
        grip_force=spec.get("grip_force"),
    )


def container(name, kind, xy, half, wall_height, z0):
    """Open container whose floor top is at z0 + FLOOR; walls are built by the backend."""
    support = z0 + FLOOR
    return Target(
        np.array([xy[0], xy[1], support + wall_height / 2]),
        np.array([half[0], half[1], wall_height / 2]),
        kind=kind,
        support_z=support,
        rim_z=support + wall_height,
    )


class _Builder:
    """Collects layout parts and applies the seed-dependent common shift."""

    def __init__(self, task, config, catalog, rng):
        self.task, self.config, self.catalog, self.rng = task, config, catalog, rng
        scene = config["scene"]
        self.z = scene["table_height"]
        jitter = (
            scene.get("bedside_spatial_jitter", scene["spatial_jitter"])
            if getattr(task, "bedside_module", None)
            else scene["spatial_jitter"]
        )
        self.shift = rng.uniform(-jitter, jitter, 2)
        self.objects, self.targets, self.assignments, self.zones = {}, {}, {}, {}
        self.fixtures, self.indicators, self.labels = [], [], {}
        self.held, self.shapes = set(), {}
        self.alarm_station, self.alarm_stations = 0, [0]
        self.order_bins, self.types, self.kinds = {}, (), []
        self.tags, self.steps, self.drawers, self.info = [], [], [], {}

    def xy(self, x, y):
        return np.array([x, y]) + self.shift

    def fixture(
        self, name, xy, size, material="steel", label="", collision=True, z=None, visible=True
    ):
        z0 = self.z if z is None else z
        self.fixtures.append(
            Fixture(
                name,
                np.array([*self.xy(*xy), z0 + size[2] / 2]),
                np.array(size, dtype=float),
                material,
                label,
                collision,
                visible,
            )
        )

    def target(self, name, kind, xy, half, wall, z0=None, label=""):
        z0 = self.z if z0 is None else z0
        self.targets[name] = container(name, kind, self.xy(*xy), half, wall, z0)
        if label:
            self.labels[name] = label

    def indicator(self, name, xy, label, height=0.045, style="dial"):
        self.indicators.append(
            Indicator(name, np.array([*self.xy(*xy), self.z + height]), label, style)
        )

    def tag(self, text, xy, z, size):
        self.tags.append(Tag(text, np.array([*self.xy(*xy), z]), size))

    def zone(self, name, xy, half=None, radius=None, support=None, margin=0.0):
        self.zones[name] = Zone(
            self.xy(*xy),
            None if half is None else np.array(half, dtype=float),
            radius,
            self.z if support is None else support,
            margin,
        )

    def item(self, name, asset, xy, support_z, *, color, label="", yaw=None, lift=0.0):
        spec = self.catalog[asset]
        shape = make_shape(spec)
        self.shapes[name] = shape
        yaw = float(self.rng.uniform(-0.15, 0.15)) if yaw is None else yaw
        quaternion = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        center = np.array(
            [*self.xy(*xy), support_z + shape.rest_height + 0.002 + lift], dtype=float
        )
        lo, hi = spec["mass_range"]
        self.objects[name] = ObjectConfig(
            name,
            asset,
            center,
            quaternion,
            np.array(color, dtype=float),
            float(self.rng.uniform(lo, hi)),
            max(
                float(self.rng.uniform(*self.config["scene"]["friction_range"])),
                spec.get("min_friction", 0.0),
            ),
            label,
        )


# Item types for the pharmacy shelf (C1) and the supply cart (C2): asset, color, shelf label.
PHARMACY_TYPES = (
    ("med_box", (0.85, 0.2, 0.15), "ASA"),
    ("med_bottle", (0.15, 0.35, 0.9), "SAL"),
    ("med_box", (0.15, 0.7, 0.3), "VIT"),
)
CART_TYPES = (
    ("gauze_box", (0.9, 0.9, 0.85), "GAUZE"),
    ("wipe_can", (0.2, 0.55, 0.85), "WIPES"),
    ("saline_bottle", (0.7, 0.85, 0.95), "SALINE"),
)
SUPPLY_TYPES = (
    ("supply_mask", (0.2, 0.55, 0.95), "MASK"),
    ("supply_kit", (0.2, 0.75, 0.35), "IV KIT"),
    ("supply_tubing", (0.95, 0.55, 0.15), "TUBING"),
)
INSTRUMENT_COLOR = (0.72, 0.75, 0.78)
SYRINGE_COLOR = (0.9, 0.93, 0.95)


def _build(task, config, catalog, rng) -> _Builder:
    b = _Builder(task, config, catalog, rng)
    from roboclock.envs.hospital.bedside import module_for

    bedside = module_for(task)
    if bedside is not None:
        bedside.build_layout(b)
        return b
    z, spec, tid = b.z, task.spec, task.task_id
    n = spec.num_items
    if tid == "A1":
        # Six places on the pad in two rows. Patient vials and decoys (no bed) are shuffled over them,
        # so the right vial has to be found by its colored label before it can go to its tray.
        pad = 0.010
        slots = [(x, -0.46 + 0.085 * k) for x in (0.26, 0.38) for k in range(3)]
        # What each bed asks for, shown on the wall screen when its request arrives.
        b.info["requests"] = [
            MEDICATIONS[int(i)] for i in rng.choice(len(MEDICATIONS), size=n, replace=False)
        ]
        order = rng.permutation(len(slots))
        b.fixture("source_pad", (0.32, -0.375), (0.24, 0.30, pad), "polymer")
        for i in range(n):
            x, y = slots[order[i]]
            b.item(
                f"cup_{i}",
                "pill_vial",
                (x, y),
                z + pad,
                color=PATIENT_COLORS[i],
                label=f"BED {i + 1}",
            )
            b.target(
                f"tray_{i}",
                "tray",
                (0.58, -0.12 + 0.17 * i),
                (0.065, 0.065),
                0.045,
                label=f"{'STAT ' if spec.stat == i else ''}BED {i + 1}",
            )
            b.assignments[f"cup_{i}"] = f"tray_{i}"
            b.indicator(
                f"ind_tray_{i}",
                (0.75, -0.12 + 0.17 * i),
                f"{'STAT ' if spec.stat == i else ''}BED {i + 1}",
            )
        for j in range(min(2, len(slots) - n)):
            x, y = slots[order[n + j]]
            b.item(
                f"decoy_{j}", "pill_vial", (x, y), z + pad, color=(0.55, 0.55, 0.55), label="BED 0"
            )
    elif tid == "A2":
        # Blood bank: a refrigerator whose drawer the robot pulls open by its handle, a bag standing in
        # a cradle inside it, and a bench-top IV stand with a hook the bag's eyelet must be hung on.
        y_f, cx = -0.33, 0.40  # fridge front plane and center line
        from roboclock.envs.hospital.props import model_size

        has_fridge_model = model_size("blood_bank_fridge_open") is not None  # it draws the cabinet
        for name, xy, size in (
            ("fridge_left", (cx - 0.165, y_f - 0.15), (0.03, 0.30, 0.33)),
            ("fridge_right", (cx + 0.165, y_f - 0.15), (0.03, 0.30, 0.33)),
            ("fridge_back", (cx, y_f - 0.315), (0.36, 0.03, 0.33)),
            ("fridge_top", (cx, y_f - 0.15), (0.36, 0.30, 0.03)),
        ):
            b.fixture(
                name,
                xy,
                size,
                "cabinet_white",
                z=z + (0.30 if name == "fridge_top" else 0.0),
                visible=not has_fridge_model,
            )
        b.fixture(
            "fridge_floor",
            (cx, y_f - 0.15),
            (0.30, 0.30, 0.02),
            "cabinet_white",
            visible=not has_fridge_model,
        )
        b.fixture(
            "fridge_fascia", (cx, y_f + 0.005), (0.36, 0.02, 0.07), "cabinet_white", z=z + 0.26
        )
        b.tag("BLOOD BANK", (cx - 0.06, y_f + 0.0185), z + 0.295, (0.19, 0.040))
        origin = np.array([cx, y_f - 0.13, z + 0.025])  # drawer origin: under the bag, on its floor
        parts = (
            DrawerPart(
                "floor", np.array([0, 0, 0.006]), np.array([0.27, 0.26, 0.012]), "stainless"
            ),
            DrawerPart(
                "front", np.array([0, 0.13, 0.105]), np.array([0.30, 0.02, 0.21]), "cabinet_white"
            ),
            DrawerPart(
                "side_l", np.array([-0.1355, 0, 0.037]), np.array([0.012, 0.26, 0.05]), "stainless"
            ),
            DrawerPart(
                "side_r", np.array([0.1355, 0, 0.037]), np.array([0.012, 0.26, 0.05]), "stainless"
            ),
            DrawerPart(
                "back", np.array([0, -0.124, 0.037]), np.array([0.27, 0.012, 0.05]), "stainless"
            ),
            DrawerPart(
                "cradle_f",
                np.array([0, 0.043, 0.026]),
                np.array([0.106, 0.006, 0.028]),
                "sterile_blue",
            ),
            DrawerPart(
                "cradle_b",
                np.array([0, -0.043, 0.026]),
                np.array([0.106, 0.006, 0.028]),
                "sterile_blue",
            ),
            DrawerPart(
                "post_l",
                np.array([-0.07, 0.205, 0.145]),
                np.array([0.012, 0.13, 0.012]),
                "stainless",
            ),
            DrawerPart(
                "post_r",
                np.array([0.07, 0.205, 0.145]),
                np.array([0.012, 0.13, 0.012]),
                "stainless",
            ),
            DrawerPart(
                "handle", np.array([0, 0.27, 0.145]), np.array([0.16, 0.018, 0.018]), "stainless"
            ),
        )
        shifted = np.array([*b.xy(origin[0], origin[1]), origin[2]])
        b.drawers.append(Drawer("blood_drawer", shifted, 0.30, 1.2, parts))
        b.info["handle"] = shifted + np.array([0.0, 0.27, 0.145])
        b.info["drawer_travel"] = 0.30
        # The bag is held by the top bar of its hanging loop: a palm coming down on the body would
        # land on the loop. It is lifted clear of the drawer front before it is moved out.
        bag, eye = b.catalog["blood_bag"], b.catalog["blood_bag"]["eyelet"]
        grasp_dz = bag["size"][2] / 2 + eye["height"] - eye["bar"] / 2
        b.info["bag_grasp_dz"] = float(grasp_dz)
        b.info["bag_lift_ee_z"] = float(origin[2] + 0.21 + 0.03 + bag["size"][2] / 2 + grasp_dz)
        b.item(
            "bag_0",
            "blood_bag",
            (cx, y_f - 0.13),
            z + 0.037,
            color=(0.60, 0.04, 0.08),
            label="O NEG",
            yaw=0.0,
        )
        # IV stand with a horizontal peg; the bag's eyelet slides on from the free end.
        hx, hy, peg_z = 0.56, 0.42, z + 0.40
        b.fixture("hook_base", (hx, hy), (0.16, 0.16, 0.012), "stainless")
        b.fixture("hook_pole", (hx, hy), (0.024, 0.024, 0.41), "stainless", z=z + 0.012)
        b.fixture("hook_peg", (hx, hy - 0.07), (0.012, 0.14, 0.012), "stainless", z=peg_z - 0.006)
        peg = b.xy(hx, hy - 0.07)
        b.info["hook"] = {
            "x": float(peg[0]),
            "peg_z": float(peg_z),
            "tip_y": float(peg[1] - 0.07),
            "base_y": float(peg[1] + 0.07),
        }
        b.tag("IV STAND", (hx, hy - 0.085), z + 0.015, (0.09, 0.03))
        b.indicator("indicator", (0.80, 0.10), "TIMER")
        b.info["request"] = {
            "bed": int(rng.integers(1, 9)),
            "unit": ["O NEG", "O POS", "A POS", "B POS"][int(rng.integers(0, 4))],
        }
    elif tid == "A3":
        # Back table with five labeled instruments in shuffled places; three of them are called, in
        # order, by the procedure steps. Each step has its own light and time window.
        pad = 0.012
        tools = ("scalpel", "forceps", "scissors", "clamp", "retractor")
        shuffled = [tools[i] for i in rng.permutation(len(tools))]
        step_tools = [tools[i] for i in rng.permutation(len(tools))[:3]]
        b.steps = step_tools
        # The surgeon calls out what is needed, not the instrument: the robot has to decide which one that is.
        b.info["situations"] = [
            SITUATIONS[tool][int(rng.integers(0, len(SITUATIONS[tool])))] for tool in step_tools
        ]
        b.fixture("source_pad", (0.34, 0.26), (0.22, 0.34, pad), "stainless")
        extras = 0
        for slot, tool in enumerate(shuffled):
            y = 0.136 + 0.062 * slot
            if tool in step_tools:
                name = f"instr_{step_tools.index(tool)}"
            else:
                name = f"extra_{extras}"
                extras += 1
            b.item(name, tool, (0.34, y), z + pad, color=INSTRUMENT_COLOR, yaw=0.0)
            b.tag(tool.upper(), (0.205, y), z + 0.02, (0.075, 0.024))
        b.target("tray", "tray", (0.46, -0.32), (0.10, 0.12), 0.040, label="MAYO TRAY")
        for k in range(3):
            b.assignments[f"instr_{k}"] = "tray"
            b.indicator(f"ind_step_{k}", (0.77, 0.04 + 0.12 * k), f"STEP {k + 1}")
        b.indicator(
            "alert", (-0.30, -1.50), "", height=1.20, style="beacon"
        )  # flashes while a call is open
    elif tid == "A4":
        # Prepared doses for several beds, each with its own beyond-use time (BUD) on its label. The
        # dose that expires first is not the one nearest the robot, so the order has to be worked out
        # from the labels. Syringe k is the k-th to expire; its bed and its place on the pad are shuffled.
        pad = 0.012
        expiries = spec.expiries or (spec.expiry_time,)
        count = len(expiries)
        lanes = rng.permutation(count)
        beds = rng.permutation(count)
        mid = -0.46 + 0.085 * (count - 1) / 2
        b.fixture(
            "source_pad", (0.32, mid), (0.22, 0.085 * count + 0.01, pad), "polymer", label="STERILE"
        )
        for k, expiry in enumerate(expiries):
            bed, y = int(beds[k]), -0.46 + 0.085 * int(lanes[k])
            b.item(f"syringe_{k}", "syringe", (0.32, y), z + pad, color=SYRINGE_COLOR, yaw=0.0)
            b.tag(
                f"BED {bed + 1}  {int(expiry // 60)}:{int(expiry % 60):02d}",
                (0.185, y),
                z + 0.02,
                (0.11, 0.028),
            )
            b.assignments[f"syringe_{k}"] = f"tray_{bed}"
        for bed in range(count):
            y = -0.15 + 0.15 * bed
            b.target(
                f"tray_{bed}", "tray", (0.60, y), (0.085, 0.055), 0.040, label=f"BED {bed + 1}"
            )
            b.indicator(f"ind_{bed}", (0.80, y), f"BED {bed + 1}")
        if config["tasks"]["A4"].get("dose_prep", False):
            # The filling dock: an empty syringe is set down here and fills over a few seconds before it can go to its bed.
            dock = (0.34, 0.27)
            b.fixture("dose_dock", dock, (0.17, 0.11, 0.012), "stainless", label="DOSE DRAW")
            b.zone("dock", dock, radius=0.055, support=z + 0.012)
            b.indicator("dock", (0.17, 0.27), "FILLING")
            b.info["dose_prep"] = {"dock": [*b.xy(*dock)], "support_z": float(z + 0.012)}
    elif tid == "B1":
        # ICU bedside: three infusion pumps in a row, each on a critical short half-life infusion
        # (appearance.PUMP_DRUGS). Each has a SILENCE button on its front, toward the robot, and a
        # dial on the bench beside it; the alarming pump's light and dial run down.
        station = int(rng.integers(0, 3))
        pumps = []
        for j in range(3):
            y = -0.14 + 0.20 * j
            b.fixture(f"pump_{j}", (0.62, y), (0.15, 0.15, 0.16), "cabinet_white", visible=False)
            b.fixture(f"pump_button_{j}", (0.50, y), (0.045, 0.045, 0.016), "shock_orange")
            b.indicator(f"ind_bin_{j}", (0.77, y), f"PUMP {j + 1}")
            pumps.append({"button": [*map(float, b.xy(0.50, y)), float(z + 0.016)]})
        b.info["pumps"] = pumps
        # A spare IV kit on the bench: the simulator needs at least one free body, and it is scenery for this task.
        b.item(
            "spare_kit",
            "supply_kit",
            (0.30, -0.40),
            z,
            color=(0.2, 0.75, 0.35),
            label="IV KIT",
            yaw=0.0,
        )
        b.alarm_station = station
        others = [s for s in range(3) if s != station]
        rng.shuffle(others)
        b.alarm_stations = ([station] + others)[: max(1, len(spec.alarms))]
    elif tid == "B2":
        # Cardiac arrest at the bedside. The patient lies along the bench with the chest in reach. The AED
        # stands on an over-bed table across the patient's near hip: pull its green tab to free the two
        # pads, put pad 1 and pad 2 on their marks on the chest, wait for the unit to analyze, then press
        # the orange SHOCK button. The response dials stand on the crash cart at the bed's far corner.
        # Common rig (2026-11-03): the head camera sits between the arms, so nothing the policy needs may
        # stand at the bench edge under it (where the AED used to be); the table puts the unit in front
        # of the camera and within arm 0's reach (B2_SCENE).
        b.shift = b.shift * np.array([0.5, 0.4])
        mattress = 0.06
        chest_top = z + mattress + 0.20
        b.fixture("chest", (0.62, 0.0), (0.40, 0.50, 0.20), "skin", z=z + mattress, visible=False)
        # The head is toward the back wall (-y): pad 1 goes on the upper right chest, pad 2 on the lower left side.
        marks = B2_SCENE["marks"]
        for k, mark in enumerate(marks):
            b.zone(f"mark_{k}", mark, radius=0.04, support=chest_top)
        table_top = B2_SCENE["table_top"]
        (tx, ty), (tsx, tsy) = B2_SCENE["table_xy"], B2_SCENE["table_size"]
        b.fixture("overbed_table", (tx, ty), (tsx, tsy, 0.025), "cabinet_white", z=table_top - 0.025, visible=False)
        aed_top = table_top + 0.075
        ax, ay = B2_SCENE["aed_xy"]
        side = B2_SCENE["aed_side"]  # +1: the lid stands on the unit's +y side, away from the chest

        def on_aed(dx, dy):  # the unit's legacy offsets (lid at -y), mirrored by `side`
            return (ax + dx, ay - side * dy)

        b.fixture("aed", (ax, ay), (0.20, 0.24, 0.075), "aed_blue", z=table_top)
        button = on_aed(0.065, 0.09)
        b.fixture("shock_button", button, (0.036, 0.036, 0.008), "shock_orange", z=aed_top)
        for k, dy in enumerate((-0.08, 0.025)):
            b.item(
                f"pad_{k}",
                "aed_pad",
                on_aed(-0.02, dy),
                aed_top,
                color=(0.95, 0.95, 0.92),
                label=str(k + 1),
                yaw=0.0,
            )
            b.held.add(f"pad_{k}")  # welded in the cartridge until the tab is pulled
        b.item("tab_0", "aed_tab", on_aed(-0.085, -0.03), aed_top, color=(0.10, 0.70, 0.30), yaw=0.0)
        # Response and shock-ready dials on pole stands beyond the bed's far side, side by side across
        # the head camera's line of sight and turned to it (face_dials).
        for name, xy, label in (("indicator", B2_SCENE["dial_xy"][0], "RESPONSE"),
                                ("shock", B2_SCENE["dial_xy"][1], "SHOCK READY")):
            b.indicator(name, xy, label, height=B2_SCENE["dial_base"] - z)
        b.info["dial_stands"] = True
        # Code-blue lamp on its own floor stand beside the dials (RoboDojo rig, 2026-11-03: the old
        # lamp over the bed head stood above and behind the 60-degree head camera's view).
        b.indicator("alert", B2_SCENE["beacon_xy"], "", height=B2_SCENE["beacon_base"] - z, style="beacon")
        b.info["aed"] = {
            "button": [*b.xy(*button), aed_top + 0.008],
            "chest_top": float(chest_top),
            "tab_pull": [-0.055, 0.0],  # off the unit toward the robot, onto the table top
            "support_z": float(table_top),  # where the pulled tab is set down
            "table": {"center": [*map(float, b.xy(tx, ty))], "size": [tsx, tsy], "top": float(table_top)},
            "side": side,
        }
        b.info["face_dials"] = True
        # Carry pads and the tab over the AED (top 1.135 m) and the chest (1.01 m), not at bench + 0.30.
        b.info["clearance_z"] = float(aed_top + 0.06)
        # Before the arrest the hand waits beside its own base (to the robot's right), out of the head
        # camera's lines of sight to the AED, the chest marks and the dials.
        from roboclock.envs.lab.robot import mount_positions

        arm0 = np.asarray(mount_positions(config, tid)[0], dtype=float)
        b.info["aed"]["ready"] = [float(arm0[0] + 0.13), float(arm0[1] - 0.20), float(arm0[2] + 0.05)]
    elif tid == "B3" and getattr(task, "name", "") == "hazardous_drug_spill":
        from roboclock.envs.hospital import spill_scene

        spill_scene.build_layout(b, config)
    elif tid == "B3":
        # A dish on the bench burns; the fire-blanket pouch hangs on a plate at the back with two black tabs.
        # The arm only reaches the second tab for a narrow band of depths, so the sideways jitter is reduced.
        # Common rig (2026-11-03): the fire alarm and both dials moved off the bench edge under the head
        # camera to the bench's +y side, in front of the camera, the dials turned to it (face_dials).
        b.shift = b.shift * np.array([1.0, 0.4])
        dish_top = z + 0.045
        b.fixture("fire_dish", (0.62, 0.0), (0.17, 0.17, 0.045), "steel", visible=False)
        b.zone("fire", (0.62, 0.0), radius=0.085, support=dish_top, margin=0.0)
        pouch = B3_SCENE["pouch_xy"]
        b.fixture("pouch_plate", pouch, (0.30, 0.02, 0.34), "cabinet_white", visible=False)
        for k, x in enumerate((0.34, 0.54)):
            b.item(f"tab_{k}", "blanket_tab", (x, -0.355), z, color=(0.06, 0.06, 0.07), yaw=0.0)
        b.item("blanket", "fire_blanket", (0.44, -0.285), z, color=(0.80, 0.78, 0.70), yaw=0.0)
        b.held.add("blanket")  # locked in the pouch until both tabs are pulled
        alarm = B3_SCENE["alarm_xy"]
        b.fixture("alarm_button", alarm, (0.05, 0.05, 0.016), "fire_red")
        b.indicator("alarm_ok", B3_SCENE["alarm_dial_xy"], "ALARM")
        # Fire-alarm strobe on the bench behind the dish, inside the 60-degree head view (it hung
        # above and behind the camera before the RoboDojo rig).
        b.indicator("alert", B3_SCENE["beacon_xy"], "", height=0.015, style="beacon")
        b.indicator("indicator", B3_SCENE["fire_dial_xy"], "FIRE")
        b.info["fire"] = {
            "center": [*b.xy(0.62, 0.0)],
            "dish_top": float(dish_top),
            "cover_radius": float(config["tasks"]["B3"]["cover_radius"]),
            "tab_pull": [0.0, 0.08],  # where each tab goes, relative to where it started
            "pouch": [*b.xy(*pouch)],
            "alarm": [*b.xy(*alarm), z + 0.016],
            "alarm_station": "+x",  # the red station box stands behind the button, away from the robot
            "holder": [*b.xy(0.44, -0.285)],
        }
        b.info["face_dials"] = True
    elif tid in ("C1", "C2"):
        types = PHARMACY_TYPES if tid == "C1" else CART_TYPES
        shelf = 0.012
        rows = -(-n // 6)
        b.fixture(
            "shelf",
            (0.415, -0.45 + 0.095 * (rows - 1) / 2),
            (0.46, 0.095 * rows + 0.03, shelf),
            "aluminum",
            label="MEDICATION SHELF" if tid == "C1" else "STOCK SHELF",
        )
        kinds = rng.permutation(np.arange(n) % 3)  # balanced so a bin never overflows
        for i in range(n):
            asset, color, text = types[kinds[i]]
            b.item(
                f"item_{i}",
                asset,
                (0.24 + 0.07 * (i % 6), -0.45 + 0.095 * (i // 6)),
                z + shelf,
                color=color,
                label=text,
            )
        base = 0.0 if tid == "C1" else 0.02
        if tid == "C2":
            b.fixture("cart", (0.50, 0.20), (0.78, 0.31, base), "steel", label="SUPPLY CART")
        for j in range(3):
            half = (0.095, 0.115) if tid == "C1" else (0.105, 0.135)
            pitch, first = (0.21, 0.29) if tid == "C1" else (0.225, 0.275)
            b.target(f"bin_{j}", "bin", (first + pitch * j, 0.20), half, 0.055, z0=z + base)
        b.indicator("indicator", (0.50, 0.45), "TIME LEFT")
        if tid == "C1":
            bins = rng.permutation(np.arange(n) % 3)
            for i in range(n):
                b.order_bins[f"item_{i}"] = f"bin_{bins[i]}"
        else:
            for j in range(3):
                b.labels[f"bin_{j}"] = types[j][2]
        b.types, b.kinds = types, kinds
    elif tid == "C3":
        shelf = 0.010
        rows = -(-n // 2)
        b.fixture(
            "used_tray",
            (0.39, -0.46 + 0.055 * (rows - 1) / 2),
            (0.46, 0.055 * rows + 0.03, shelf),
            "polymer",
            label="USED",
        )
        for i in range(n):
            b.item(
                f"item_{i}",
                "instrument",
                (0.30 + 0.18 * (i % 2), -0.46 + 0.055 * (i // 2)),
                z + shelf,
                color=INSTRUMENT_COLOR,
                yaw=0.0,
            )
            b.assignments[f"item_{i}"] = "basket"
        b.target("basket", "basket", (0.62, 0.17), (0.14, 0.12), 0.090, label="STERILIZER")
        start = (0.40, 0.31)  # within the arm's reach (0.8 m was not)
        b.fixture("start_button", start, (0.05, 0.05, 0.016), "monitor_green", label="START")
        b.indicator("indicator", (0.86, 0.10), "CYCLE")
        b.info["sterilizer"] = {"start": [*b.xy(*start), z + 0.016]}
    else:
        raise ValueError(f"Unknown task: {tid}")
    return b


def make_layout(task, config: dict, catalog=None) -> HospitalLayout:
    rng = np.random.default_rng(np.random.SeedSequence([task.spec.seed, 101]))
    catalog = catalog or load_catalog()
    b = _build(task, config, catalog, rng)
    layout = HospitalLayout(
        b.objects,
        b.targets,
        b.assignments,
        b.zones,
        b.fixtures,
        b.indicators,
        frozenset(b.held),
        b.shapes,
        {name: (s.grasp_radius, s.grasp_height) for name, s in b.shapes.items()},
        b.labels,
        b.z,
        float(rng.uniform(*config["scene"]["light_intensity_range"])),
        rng.uniform(-0.03, 0.03, 3),
        _surface_bounds(b.info.get("work_surface"), config["scene"]),
        alarm_station=b.alarm_station,
        alarm_stations=b.alarm_stations or [b.alarm_station],
        task_id=task.task_id,
        tags=b.tags,
        drawers=b.drawers,
        info=b.info,
        steps=b.steps,
    )
    if task.task_id in ("C1", "C2"):
        layout.item_types = {
            f"item_{i}": f"{b.types[k][0]}:{b.types[k][2]}" for i, k in enumerate(b.kinds)
        }
        layout.type_labels = {value: value.split(":")[1] for value in layout.item_types.values()}
        if task.task_id == "C1":
            for name, bin_name in b.order_bins.items():
                orders = layout.orders.setdefault(bin_name, {})
                orders[layout.item_types[name]] = orders.get(layout.item_types[name], 0) + 1
            for bin_name in layout.targets:
                layout.orders.setdefault(bin_name, {})
                kinds = ", ".join(
                    f"{count}x{key.split(':')[1]}" for key, count in layout.orders[bin_name].items()
                )
                layout.labels[bin_name] = f"RX {int(bin_name[-1]) + 1}: {kinds}"
        else:
            layout.bin_types = {f"bin_{j}": f"{b.types[j][0]}:{b.types[j][2]}" for j in range(3)}
    validate_layout(task, layout, config)
    return layout


def _aabb(center, half):
    return np.asarray(center[:2]) - half[:2], np.asarray(center[:2]) + half[:2]


def _overlap(a, b, gap=0.0):
    return bool(np.all(a[0] < b[1] + gap) and np.all(b[0] < a[1] + gap))


# Env B2/B3 on the common rig (robot frame, before the seed shift). The head camera sits 0.04 m ahead
# of the arms' midpoint and sees the bench only from ~0.05 m ahead of itself, so the AED, the fire
# alarm and the cue dials moved in front of it (and within arm 0's top-down reach of ~0.5 m).
B2_SCENE = {
    "marks": ((0.52, -0.10), (0.63, 0.10)),
    "table_top": 1.06,  # over-bed table top: clears the sheet's hip fold (1.02 m)
    "table_xy": (0.385, 0.40),
    "table_size": (0.33, 0.30),
    "aed_xy": (0.40, 0.40),
    "aed_side": 1.0,
    "dial_xy": ((1.10, 0.05), (1.10, 0.25)),  # beyond the bed's far edge (x 1.03)
    "dial_base": 1.05,  # housing bottom on its pole stand; the dial face centers at ~1.18 m
    "beacon_xy": (1.12, 0.47),  # code-blue lamp on a pole stand left of the dials (head view)
    "beacon_base": 1.02,
}
B3_SCENE = {
    "pouch_xy": (0.44, -0.50),  # 25 mm further back: the hand brushed the plate pulling tab 0
    "alarm_xy": (0.36, 0.28),
    "alarm_dial_xy": (0.80, 0.30),
    "fire_dial_xy": (0.80, 0.46),
    "beacon_xy": (0.90, -0.10),  # fire-alarm strobe on the bench behind the dish (head view)
}

# Horizontal distance from an X5 mount: its tool reaches ~0.71 m at 0.1 m above the table.
REACH = (0.20, 0.72)


def validate_layout(task, layout: HospitalLayout, config: dict):
    """Reject scenes with initial overlaps or unreachable items before any simulation.

    On the common rig (`robot.rig`) reach is measured from the nearer of the two X5 mounts:
    single-arm tasks place everything around arm 0, two-arm tasks give each arm the items on its
    own side. Archived profiles without a rig keep the arm-0-only check they were tuned with.
    """
    from roboclock.envs.lab.robot import mount_positions

    low, high = layout.workspace_bounds
    mounts = mount_positions(config, task.task_id)
    if "rig" not in config.get("robot", {}):
        mounts = mounts[:1]
    bases = np.array([m[:2] for m in mounts], dtype=float)

    def nearest(xy):
        return float(np.min(np.hypot(*(np.asarray(xy[:2]) - bases).T)))

    boxes = {}
    for name, obj in layout.objects.items():
        extent = object_extents(layout.shapes[name], obj.orientation)
        boxes[name] = _aabb(obj.position, extent)
        reach = nearest(obj.position)
        if not REACH[0] <= reach <= REACH[1]:
            raise ValueError(f"{task.task_id}: {name} is {reach:.2f} m from the robot base")
        if np.any(obj.position[:2] - extent[:2] < low[:2]) or np.any(
            obj.position[:2] + extent[:2] > high[:2]
        ):
            raise ValueError(f"{task.task_id}: {name} lies outside the table")
    names = list(boxes)
    # Bodies of one jointed device (B5: the auto-injector's barrel and its sliding tip) overlap.
    attached = {frozenset(pair) for pair in layout.info.get("attached", ())}
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            if frozenset((first, second)) in attached:
                continue
            if _overlap(boxes[first], boxes[second], gap=0.003):
                raise ValueError(f"{task.task_id}: {first} overlaps {second}")
    walls = {name: _aabb(t.position, t.half_extents + WALL) for name, t in layout.targets.items()}
    for name, target in layout.targets.items():
        reach = nearest(target.position)
        if reach > REACH[1]:
            raise ValueError(f"{task.task_id}: {name} is {reach:.2f} m from the robot base")
        for other, box in walls.items():
            if other < name and _overlap(walls[name], box):
                raise ValueError(f"{task.task_id}: {name} overlaps {other}")
        for item, box in boxes.items():
            inside = bool(np.all(box[0] >= walls[name][0]) and np.all(box[1] <= walls[name][1]))
            if _overlap(walls[name], box) and not inside:
                raise ValueError(f"{task.task_id}: {item} overlaps container {name}")
    for indicator in layout.indicators:
        region = _aabb(indicator.position, np.array([0.04, 0.045]))
        for other, box in {**walls, **boxes}.items():
            if _overlap(region, box):
                raise ValueError(f"{task.task_id}: indicator {indicator.name} overlaps {other}")
    for name in layout.assignments:
        if layout.assignments[name] not in layout.targets:
            raise ValueError(f"{task.task_id}: {name} targets an unknown container")


def check_config(config):
    """Check every hospital layout without loading the simulation runtime."""
    from roboclock.envs.hospital.tasks.registry import TASKS, make_task

    for key in TASKS:
        for difficulty in ("easy", "hard"):
            for seed in range(3):
                task = make_task(key, difficulty, seed, config)
                task.bind(make_layout(task, config))


def _surface_bounds(surface, scene):
    """Bounds of the support surface: a ward task's own work_surface, else the generic bench."""
    surface = surface or {"center": scene["table_center"], "size": scene["table_size"]}
    center, size = np.array(surface["center"]), np.array(surface["size"])
    return center - size / 2, center + size / 2
