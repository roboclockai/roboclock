"""Authored functional equipment details; no evaluator text substitutes for cues.

NVIDIA sphere/cylinder assets remain the reusable rigid props. Cabinet panels,
annuli and indicator hardware are procedural metric geometry, not scanned assets.
"""

import numpy as np


def annulus(backend, path, center, inner, outer, height, color, *, square=False, collision=False):
    """Watertight annular panel; a square exterior creates a tiled nine-hole deck."""
    from isaacsim.core.api.materials import PreviewSurface
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    points = []
    n = 64
    for z in (-height / 2, height / 2):
        for radius in (inner, outer):
            for angle in np.arange(n) * 2 * np.pi / n:
                x, y = np.cos(angle), np.sin(angle)
                r = radius / max(abs(x), abs(y)) if square and radius == outer else radius
                points.append(
                    Gf.Vec3f(
                        float(center[0] + r * x), float(center[1] + r * y), float(center[2] + z)
                    )
                )
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend(
            [
                [i, j, n + j, n + i],
                [2 * n + i, 3 * n + i, 3 * n + j, 2 * n + j],
                [i, 2 * n + i, 2 * n + j, j],
                [n + i, n + j, 3 * n + j, 3 * n + i],
            ]
        )
    mesh = UsdGeom.Mesh.Define(backend.stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        backend.surface_paths.append(path)
    material = PreviewSurface(
        path + "Material", color=np.array(color), roughness=0.48, metallic=0.12
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material.material)
    return material


def build_equipment(b):
    from isaacsim.core.api.objects import FixedCuboid

    t, h = b.task, b.config["scene"]["table_height"]
    b.cue_materials = {}
    b.note_labels = {}
    b.progress_materials = {}
    b._equipment_text = {}
    if t.task_id == "34":
        station = b.config.get("v2", {}).get("station_offset", (0.0, 0.0))
        for name, spec in b.layout.controls.items():
            x, y = spec.position[:2]
            annulus(
                b,
                f"/World/Deck/{name}",
                [x, y, h + 0.0925],
                0.034,
                0.06,
                0.015,
                [0.20, 0.32, 0.29],
                square=True,
                collision=True,
            )
            annulus(
                b,
                f"/World/Rims/{name}",
                [x, y, h + 0.101],
                0.034,
                0.040,
                0.003,
                [0.045, 0.06, 0.065],
            )
            b.label(
                f"/World/HoleNumbers/{name}",
                str(int(name[-1]) + 1),
                [x + 0.049, y, h + 0.101],
                [0.017, 0.022],
                False,
            )
        b.label(
            "/World/MoleLegend",
            "GREEN: HIT     RED: SKIP",
            # Moves with the seeded station (``arcade.tasks.station_offset``).
            [0.52 + station[0], -0.19 + station[1], h + 0.101],
            [0.32, 0.018],
            False,
        )
    if t.task_id == "32":
        for i, (name, spec) in enumerate(b.layout.controls.items()):
            x, y, z = spec.position
            b.cue_materials[name] = annulus(
                b,
                f"/World/PadLED/{name}",
                [x, y, z - 0.005],
                0.044,
                0.050,
                0.004,
                [0.04, 0.06, 0.08],
            )
            b.collider(
                f"/World/PadScreens/{name}",
                "cube",
                [x + 0.088, y, h + 0.034],
                [0.065, 0.112, 0.004],
                [0.018, 0.025, 0.034],
                False,
            )
            b.note_labels[name] = ([x + 0.080, y, h + 0.037], [0.040, 0.033])
            for k in range(8):
                _, mat = b.collider(
                    f"/World/HoldLED/{name}_{k}",
                    "cube",
                    [x + 0.108, y + (k - 3.5) * 0.012, h + 0.038],
                    [0.006, 0.009, 0.002],
                    [0.03, 0.08, 0.1],
                    False,
                )
                b.progress_materials[name, k] = mat
        b.label(
            "/World/RhythmBrand",
            "PULSE / 3   RHYTHM CONTROLLER",
            [0.49, -0.214, h + 0.034],
            [0.19, 0.020],
            False,
        )
    if t.task_id in {"33", "35", "37", "38", "40"}:
        # Discrete lamps mounted in hardware; no generic WAIT/VALID/DECOY banner.
        names = {
            "33": ["HOLD", "DOWN"],
            "35": ["WAIT", "READY"],
            "37": ["STOP", "GO"],
            "38": ["SET", "BALL"],
            "40": ["LOAD", "FIRE"],
        }[t.task_id]
        x, y = (0.78, 0.15) if t.task_id == "40" else (0.52, 0.30)
        body = b.world.scene.add(
            FixedCuboid(
                "/World/IndicatorHousing",
                name="indicator_housing",
                position=np.array([x, y, h + 0.021]),
                scale=np.array([0.18, 0.09, 0.042]),
                color=np.array([0.035, 0.045, 0.055]),
            )
        )
        b.surface_paths.append(body.prim_path)
        for i, name in enumerate(names):
            px = x + (i - 0.5) * 0.085
            annulus(
                b,
                f"/World/LampBezels/{name}",
                [px, y, h + 0.045],
                0.022,
                0.026,
                0.006,
                [0.14, 0.17, 0.19],
            )
            _, mat = b.collider(
                f"/World/Lamps/{name}",
                "cylinder",
                [px, y, h + 0.046],
                [0.021, 0.006],
                [0.035, 0.045, 0.05],
                False,
            )
            b.cue_materials[name] = mat
            b.label(
                f"/World/LampLabels/{name}", name, [px, y - 0.032, h + 0.043], [0.066, 0.016], False
            )
    if t.task_id == "38":
        for name, p in b.layout.targets.items():
            annulus(
                b,
                f"/World/PinSpots/{name}",
                [p[0], p[1], h + 0.0013],
                0.015,
                0.020,
                0.001,
                [0.92, 0.85, 0.62],
            )
    if t.task_id == "37":
        p = b.layout.targets["puck"]
        annulus(
            b, "/World/GoalRing", [p[0], p[1], h + 0.0012], 0.032, 0.038, 0.001, [0.92, 0.94, 0.90]
        )
        b.label(
            "/World/FinishLabel", "GOAL", [p[0] + 0.049, p[1], h + 0.0013], [0.023, 0.07], False
        )
    if t.task_id == "40":
        b.collider(
            "/World/HopperGaugePanel",
            "cube",
            [0.671, 0.15, h + 0.127],
            [0.025, 0.14, 0.008],
            [0.04, 0.065, 0.08],
            False,
        )
        for i in range(t.settings["capacity"]):
            _, mat = b.collider(
                f"/World/HopperLevel/Level{i}",
                "cube",
                [0.67, 0.10 + i * 0.034, h + 0.132],
                [0.012, 0.026, 0.006],
                [0.05, 0.1, 0.1],
                False,
            )
            b.cue_materials[f"level_{i}"] = mat
        b.label(
            "/World/HopperLabel",
            f"HOPPER / CAPACITY {t.settings['capacity']}",
            [0.56, 0.251, h + 0.131],
            [0.19, 0.019],
            False,
        )
        b.label(
            "/World/BasketLabel", "BALL SUPPLY", [0.4625, -0.253, h + 0.009], [0.18, 0.018], False
        )


def update_equipment(b):
    t = b.task
    if t.task_id == "32":
        index = t.current_index
        e = None if index is None else t.events[index]
        for name, mat in b.cue_materials.items():
            active = e is not None and e["control"] == name
            done = active and index in t.correct
            active = active and not done
            active = active and (
                t.time <= e["time"] + t.limits["note_window"] or index in t.hold_progress
            )
            holding = active and e["hold"] > 0.1
            color = (
                [0.05, 0.65, 1.0]
                if holding
                else [0.95, 0.65, 0.04]
                if active
                else [0.04, 0.09, 0.12]
            )
            mat.set_color(np.array(color))
            label = "HOLD" if holding else "TAP" if active else "OK" if done else "-"
            if b._equipment_text.get(name) != label:
                p, size = b.note_labels[name]
                b.label(f"/World/PadInstructions/{name}", label, p, size, False)
                b._equipment_text[name] = label
            progress = (
                1.0
                if done
                else min(1.0, t.hold_progress.get(index, 0.0) / e["hold"])
                if active
                else 0.0
            )
            for k in range(8):
                b.progress_materials[name, k].set_color(
                    np.array([0.1, 0.9, 0.45] if (k + 1) / 8 <= progress else [0.035, 0.065, 0.08])
                )
    elif t.task_id == "34":
        i = t.current_index
        for name, mat in b.control_materials.items():
            e = None if i is None else t.events[i]
            active = e is not None and e["control"] == name
            mat.set_color(
                np.array(
                    [0.08, 0.65, 0.22]
                    if active and e["valid"]
                    else [0.85, 0.045, 0.07]
                    if active
                    else [0.14, 0.22, 0.19]
                )
            )
    elif b.cue_materials:
        phase = t.phase
        active = {
            "33": "DOWN" if t.down_time is not None else "HOLD",
            "35": "READY" if phase == "ready" else "WAIT",
            "37": "STOP" if phase == "red" else "GO",
            "38": "BALL" if t.time >= t.limits["ball_deadline"] else "SET",
            "40": "LOAD" if t.time < t.settings.get("warmup", 0) else "FIRE",
        }.get(t.task_id)
        for name, mat in b.cue_materials.items():
            lit = name == active
            if name.startswith("level_"):
                lit = int(name[6:]) < t.occupancy
            if name == "FIRE":
                warm = t.settings["warmup"]
                lit = t.time >= warm and (t.time - warm) % t.limits["fire_interval"] < 0.35
            color = [0.9, 0.06, 0.025] if name in {"STOP", "WAIT", "BALL"} else [0.10, 0.8, 0.24]
            mat.set_color(np.array(color if lit else [0.035, 0.045, 0.05]))
