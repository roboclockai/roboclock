"""Reviewed redesign proposals47–49: actual static USD/RGB, no task pass.

Only scene layouts change here. Controllers remain held pending review.
"""

from types import SimpleNamespace

import numpy as np

from .industrial_workcell import IndustrialCell
from .remaining_environments import BOX, RACK, ROOT, TABLE, TRAY

# Task 49 receivers per arm (x, y of the tray centers) and tray size (m). Inner PASS
# beside the input tray, outer REJECT; y 0.52-0.73 is inside the rig head view.
RECEIVERS_49 = dict(
    left=dict(PASS=[-0.45, 0.625], REJECT=[-0.645, 0.625]),
    right=dict(PASS=[0.45, 0.625], REJECT=[0.645, 0.625]),
)
RECEIVER_SIZE_49 = [0.18, 0.21, 0.035]


def build_layout_review(task, output, resolution=(1600, 1000)):
    c = IndustrialCell(output, resolution)

    def bench(name, xy, size):
        c.asset(name, TABLE, [*xy, 0.43], [*size, 0.86])
        c.box(name + "Top", [*xy, 0.875], [*size, 0.025], "steel")

    def tray(name, xy, size, label, material="steel"):
        c.asset(name, TRAY, [*xy, 0.8875 + size[2] / 2], size, material=material)
        c.sign(
            name.replace("/", "_") + "Label",
            label,
            [xy[0], xy[1] - size[1] / 2 - 0.009, 0.884],
            [size[0], 0.047],
        )

    def marker(side, xyz):
        x, y, z = xyz
        c.box("PlannedMount/" + side, [x, y, 0.889], [0.15, 0.14, 0.003], "yellow")
        c.sign(
            "Mount_" + side,
            side.upper() + " ARM / PROPOSED BASE",
            [x, y - 0.078, 0.91],
            [0.25, 0.045],
        )

    common = dict(
        task_id=task,
        temporal_category="C",
        scope="Actual static GPU RGB layout proposal. No robot, active process, native contacts, reach-path or throughput success. Yellow footprints mark proposed base positions only.",
        approval="Awaiting layout review; do not launch superseded controllers",
        randomization="Static seed0 layout only. Independent bounded seeded layout, physical and temporal streams retained for future episodes.",
    )
    if task == 47:
        # 2.0 m wide (was 1.75 m) so the rig layout's kit cases at x=+/-0.66 rest on it.
        bench("KittingBench", [0, 0.55], [2.0, 1.10])
        bench("LidRest", [0.92, 1.38], [0.65, 0.50])
        c.asset("ServiceInventory", RACK, [-1.85, 2.10, 1.05], [1.1, 0.55, 2.1])
        for i, (name, file, h) in enumerate(
            [
                ("bearing", ROOT / "task46/prepared/bearing.usd", 0.07),
                ("bracket", ROOT / "task47/prepared/bracket.usda", 0.05),
                ("pin", ROOT / "task47/prepared/pin.usda", 0.016),
            ]
        ):
            x = -0.56 + i * 0.54
            tray("Feed/" + name, [x, 0.68], [0.45, 0.25, 0.035], name.upper())
            for j in range(3):
                c.asset(
                    f"Parts/{name}{j}",
                    file,
                    [x + (j - 1) * 0.11, 0.68, 0.89785 + h / 2],
                    material="steel",
                )
        folder = ROOT / "task47/kit-case-v1"
        body_h = 0.2190991
        c.asset(
            "Kit/Case", folder / "open_case.usda", [0, 0.22, 0.8875 + body_h / 2], material="frame"
        )
        c.asset(
            "Kit/Lid",
            folder / "lift_off_lid.usda",
            [0.92, 1.38, 0.8875 + 0.0346818 / 2],
            material="frame",
        )
        from pxr import Usd, UsdGeom, UsdShade

        # Original MDLs refer to textures absent from the pinned case package.
        # Explicitly replace active bindings with declared factory PBR paint,
        # black handles and brushed metal clasps; no geometry/UV edits.
        for root in ["/World/Kit/Case", "/World/Kit/Lid"]:
            for prim in Usd.PrimRange(c.stage.GetPrimAtPath(root)):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                name = prim.GetName()
                material = "black" if "Holder" in name else ("steel" if "Lock" in name else "frame")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    c.appearance.material(material).material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                )
        # Existing removable imported tray on a declared ordinary internal riser;
        # retain original case body/latches/handles and all source mesh geometry.
        c.box("Kit/InternalRiser", [0, 0.22, 0.9623263], [0.37, 0.24, 0.1496526], "black")
        tray("Kit/Insert", [0, 0.22], [0.38, 0.25, 0.035], "BEARING | BRACKET | PIN", "black")
        # Raise the existing insert, preserving its complete original mesh.
        insert = c.world.scene.get_object("Insert")
        insert.set_world_pose(position=np.array([0, 0.22, 1.0546527]))
        # Keep the raised insert at the same pose on the native episode reset.
        # A preview-only pose update otherwise restores the lower tray default.
        insert.set_default_state(position=np.array([0, 0.22, 1.0546527]))
        c.stage.RemovePrim("/World/Signs/Kit_InsertLabel")
        c.sign("KitID", "MAINTENANCE KIT / 3 COMPONENTS", [0, 0.035, 1.005], [0.48, 0.082])
        for x, label in [(-0.11, "BEARING"), (0, "BRACKET"), (0.11, "PIN")]:
            c.sign("Slot_" + label, label, [x, 0.094, 1.072], [0.105, 0.026])
        mounts = dict(left=[-0.43, 0.20, 0.98], right=[0.43, 0.20, 0.98])
        for side, p in mounts.items():
            marker(side, p)
        c.sign(
            "Order", "ASSEMBLE KIT -> CHECK 3 COMPONENTS -> DELIVERY", [0, 1.08, 1.27], [1.6, 0.12]
        )
        c.sign("LidNote", "ORIGINAL LIFT-OFF COVER", [0.92, 1.09, 0.96], [0.61, 0.07])
        bench("DeliveryStation", [2.15, 1.85], [0.85, 0.65])
        c.sign("Delivery", "COMPLETE KITS / LINE-SIDE DELIVERY", [2.15, 1.50, 1.04], [0.95, 0.10])
        c.sign("Cell", "CELL47 / MAINTENANCE KIT ASSEMBLY", [0, 3.72, 2.65], [4, 0.22])
        views = [
            ("01_dedicated_kit_case", [1.30, -1.24, 2.02], [0.03, 0.50, 1.04]),
            ("02_open_case_and_feed", [-0.72, -0.35, 1.75], [-0.04, 0.40, 1.03]),
            ("03_kitting_and_delivery", [3.3, -3.1, 2.8], [0.45, 1.0, 0.98]),
        ]
        common.update(
            scene="Distinct original metal maintenance case, raised removable dark insert, separate lift-off cover, three steel feed trays and line-side delivery destination",
            revision="Replace identical kit/feed trays with downloaded RoboLab VOMP case_a01; existing separate cover parked on side bench; no fabricated product box.",
            case_asset_status="Source download/LFS hashes and CPU USD composition verified. Actual static GPU import assessed by this run; native case/insert support and manipulation unvalidated.",
            case_attribution="RoboLab VOMP case_a01, pinned b6cfcf6ccd2113c4cd1216d35b33024a7d5691da; retained upstream code/object notices; asset-specific redistribution license unverified. Local review only.",
            proposed_mounts_m=mounts,
            mobile_requirement="Retained; mobile base undecided. Layout does not establish two kit deliveries/90s.",
        )
    elif task == 48:
        # Retain the approved warehouse setting and make all three actual
        # destination openings and the pallet-to-shelf route explicit.
        pallet = ROOT / "task48/recovered/pallet.usd"
        c.asset("Inbound/Pallet", pallet, [-0.85, 0.05, 0.10], [1.1, 0.8, 0.2])
        # Six physically stacked original cartons; first layer accessible from
        # the front, top row is removed before the cartons beneath it.
        for i in range(6):
            x = -1.16 + (i % 3) * 0.31
            z = 0.371825 + (i // 3) * 0.32
            c.asset(f"Inbound/Box{i}", BOX, [x, 0.05, z], [0.28, 0.33, 0.32])
            c.sign(
                f"SKU{i}",
                ("A / BRAKE KITS", "B / FILTERS", "C / SENSORS")[i % 3],
                [x, -0.121, z],
                [0.26, 0.065],
            )
        floors = []
        for i, x in enumerate([-0.95, 0.12, 1.19]):
            c.asset(f"Shelves/Bay{i}", RACK, [x, 1.75, 0.90], [0.55, 0.90, 1.80], yaw=np.pi / 2)
            # The factory source broad horizontal deck is authored at50.596cm,
            # not its tiny taller corner fittings; record transformed source floor.
            floor = 50.596 / 301.00006103515625 * 1.8
            floors.append(dict(bay=chr(65 + i), center_x_m=x, front_y_m=1.475, support_z_m=floor))
            color = ["green", "yellow", "water"][i]
            c.box(f"ShelfTargets/Edge{i}", [x, 1.466, floor - 0.018], [0.77, 0.018, 0.055], color)
            c.sign(
                f"Bay{i}",
                f"BAY {chr(65 + i)} / " + ["BRAKE KITS", "FILTERS", "SENSORS"][i],
                [x, 1.46, 0.94],
                [0.82, 0.12],
            )
            c.sign(
                f"Target{i}", "PLACE INSIDE -> RELEASE -> NEXT BOX", [x, 1.46, 0.53], [0.82, 0.07]
            )
        # The aisle is actual free floor between source and destination. No
        # mobile robot is invented while the benchmark base is unresolved.
        for i, y in enumerate([0.58, 1.15]):
            c.box(f"Route/Edge{i}", [0.12, y, 0.004], [3.3, 0.025, 0.004], "yellow")
        for i, x in enumerate([-0.75, 0, 0.75]):
            c.box(f"Route/Progress{i}", [x, 0.86, 0.005], [0.35, 0.065, 0.005], "green")
        c.box("Order/Panel", [1.54, 0.30, 1.12], [0.62, 0.10, 0.46], "frame")
        c.pipe("Order/Post", [1.54, 0.30, 0], [1.54, 0.30, 0.93], 0.035, "frame")
        c.sign("Order", "PUTAWAY / 0 OF 3 CORRECT BOXES", [1.54, 0.242, 1.23], [0.59, 0.10])
        for i, label in enumerate(
            ["A  BRAKE KITS -> BAY A", "B  FILTERS -> BAY B", "C  SENSORS -> BAY C"]
        ):
            c.sign(f"OrderRow{i}", label, [1.54, 0.242, 1.105 - i * 0.085], [0.58, 0.065])
        c.sign("Cell", "CELL48 / PALLET -> SKU MATCH -> SHELF", [0, 3.72, 2.65], [4.3, 0.22])
        views = [
            ("01_complete_putaway_layout", [2.70, -2.65, 2.25], [0.0, 0.90, 0.77]),
            ("02_pallet_aisle_shelf", [-2.40, -1.60, 2.0], [-0.05, 1.0, 0.80]),
            ("03_destination_openings", [2.4, -0.25, 1.55], [0.10, 1.60, 0.57]),
        ]
        common.update(
            scene="Inbound pallet of six labeled automotive-parts cartons, clear transport aisle, three original shelf bays with actual lower-deck targets and order board",
            revision="Replace lift-only presentation with explicit multi-box pallet-to-correct-shelf workflow. Original rack yaw90 exposes its unbraced long-side openings toward the aisle; original end braces remain intact. No raise-and-return-to-source completion credit.",
            shelf_floor_candidates=floors,
            mobile_requirement="RETAINED / BASE UNDECIDED. Full cross-aisle shelving requires the eventual mobile base. Operation portions can be tested at declared fixed dock poses; no robot/base teleportation or full mobile benchmark credit.",
            timing="Three correctly released/retained shelf placements in90s. Mobile budget starts at episode start; operation-only tests are labeled separately.",
            next_physics="Validate actual shelf triangle support, pallet stack stability, original carton geometry, grasp/transport/release and no wrong-bay counting after layout review.",
        )
    elif task == 49:
        bench("InspectionBench", [0, 0.58], [1.85, 1.20])
        c.asset("MachinedStock", RACK, [-1.8, 2.0, 1.05], [1.1, 0.55, 2.1])
        tray("Input/Tray", [0, 0.73], [0.48, 0.24, 0.035], "MACHINED PARTS / INPUT")
        for i, x in enumerate([-0.12, 0, 0.12]):
            c.asset(
                f"Input/Part{i}",
                ROOT / "task49/prepared-v3/port_block.usda",
                [x, 0.73, 0.91885],
                material="steel",
            )
        # The D455 inspection camera, its post and ring light were removed on 2026-11-19: C3 is
        # a weighing task now (weigh_sort_task.py, weigh_scale.py).
        mounts = dict(left=[-0.30, 0.32, 1.20], right=[0.30, 0.32, 1.20])
        for side, p in mounts.items():
            marker(side, p)
        # Per-arm PASS/REJECT receivers in the rig head camera's view (common rig,
        # 2026-11-03): each arm sorts into the pair on its own side. They replace the
        # shared receivers at (0, 0.41) / (0, 0.12), which sat at or behind the head camera.
        for side, trays in RECEIVERS_49.items():
            for label, xy in trays.items():
                name = f"{label}_{side}"
                width, depth, height = RECEIVER_SIZE_49
                c.asset("Output/" + name, TRAY, [*xy, 0.8875 + height / 2], RECEIVER_SIZE_49, material="steel")
                color = "green" if label == "PASS" else "red"
                c.box("Output/" + name + "Back", [xy[0], xy[1] + depth / 2 + 0.008, 0.95], [width, 0.012, 0.085], color)
                c.sign(name + "Back", label, [xy[0], xy[1] + depth / 2, 0.956], [width - 0.01, 0.065])
        receivers = RECEIVERS_49
        c.sign(
            "Recipe",
            "WEIGH -> PASS / REJECT",
            [0, 1.15, 1.63],
            [1.6, 0.11],
        )
        c.sign(
            "Cell", "CELL49 / WEIGHING + RESULT SORTING", [0, 3.72, 2.65], [4.4, 0.22]
        )
        views = [
            ("01_shared_receivers", [1.12, -1.05, 1.95], [0, 0.50, 1.05]),
            ("02_both_arm_access", [-0.9, -0.75, 1.85], [0, 0.42, 1.05]),
            ("03_camera_and_trays", [0.78, 0.0, 1.63], [0, 0.61, 1.12]),
        ]
        common.update(
            scene="Original D455 three-side inspection, centered input tray, a PASS and a REJECT receiver beside each arm in the head camera's view",
            revision="Per-arm receiver pairs (common rig): each arm sorts into its own side; either side's tray of the right result is correct. Seeded per-part QC independent of arm allocation.",
            proposed_mounts_m=mounts,
            receivers_by_arm_xy_m=receivers,
            reach_status="Original bounded URDF endpoint solves pass for each arm's own receivers with a 90 deg downward deposit tool yaw.",
            planned_qc="Independently seeded per-part symbolic simulated QC; controller result available only after all three real camera view acquisitions. Not a learned vision classifier.",
            planned_interlock="No shared deposit zone: each arm deposits on its own side; input/camera interlocks remain.",
            required_coverage="At least left PASS, left REJECT, right PASS and right REJECT across two independently seeded episodes; three inspected parts and nine views per90s episode.",
        )
    else:
        raise ValueError(task)
    return SimpleNamespace(**vars(c), cell=c, views=views, metadata=common)
