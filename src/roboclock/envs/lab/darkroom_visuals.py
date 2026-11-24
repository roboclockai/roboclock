"""Detailed authored darkroom fixtures; see assets/darkroom/sources.json.

Visible shells retain BathScene's conservative collision floors/walls. The
deep vessels support a vertical dip transfer; they are not shallow print trays.
"""

import struct
from pathlib import Path

import numpy as np


def mesh(scene, name, points, faces, material):
    from pxr import Gf, UsdGeom, UsdShade

    shape = UsdGeom.Mesh.Define(
        scene.stage, name if name.startswith("/") else "/World/FaultRig/" + name
    )
    shape.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    shape.CreateFaceVertexCountsAttr([len(f) for f in faces])
    shape.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    shape.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material.material)
    return shape


def rounded_ring(half, radius, height, center, count=12):
    points = []
    for sx, sy, start in [(1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)]:
        for angle in np.linspace(start, start + 90, count, endpoint=False):
            a = np.deg2rad(angle)
            points.append(
                [
                    center[0] + sx * (half - radius) + radius * np.cos(a),
                    center[1] + sy * (half - radius) + radius * np.sin(a),
                    height,
                ]
            )
    return points


def rings_mesh(scene, name, rings, material, close_profile=False, caps=False):
    n = len(rings[0])
    points = [p for ring in rings for p in ring]
    faces = []
    for j in range(len(rings) if close_profile else len(rings) - 1):
        nxt = (j + 1) % len(rings)
        for i in range(n):
            k = (i + 1) % n
            faces.append([j * n + i, j * n + k, nxt * n + k, nxt * n + i])
    if caps:
        faces.extend([list(reversed(range(n))), list(range((len(rings) - 1) * n, len(points)))])
    return mesh(scene, name, points, faces, material)


def lathe(scene, name, center, profile, material):
    rings = [
        [
            [center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), center[2] + height]
            for a in np.linspace(0, 2 * np.pi, 96, endpoint=False)
        ]
        for radius, height in profile
    ]
    return rings_mesh(scene, name, rings, material, caps=True)


def curve(scene, name, points, width, material):
    from pxr import Gf, UsdGeom, UsdShade

    shape = UsdGeom.BasisCurves.Define(
        scene.stage, name if name.startswith("/") else "/World/FaultRig/" + name
    )
    shape.CreateTypeAttr("linear")
    shape.CreateCurveVertexCountsAttr([len(points)])
    shape.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    shape.CreateWidthsAttr([width])
    shape.SetWidthsInterpolation("constant")
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material.material)


def loaded_reel(a, path):
    """A rigid loaded 35 mm reel, with a narrow spindle accessible to both fingers."""
    from types import SimpleNamespace

    from pxr import UsdGeom

    scene = SimpleNamespace(stage=a.stage)
    nylon = a.material("LoadedReelNylon", color=[0.78, 0.80, 0.74], roughness=0.3)
    black = a.material("ReelSpindleABS", color=[0.022, 0.025, 0.028], roughness=0.34)
    film = a.material("WoundNegativeFilm", color=[0.12, 0.055, 0.023], roughness=0.23)
    for name, radius, height, center in [
        ("ReelEnvelope", 0.044, 0.047, -0.0365),
        ("GripEnvelope", 0.0127, 0.12, 0),
    ]:
        prim = a.cylinder_child(path, name, radius, height, center, black, True)
        UsdGeom.Imageable(prim).MakeInvisible()
    # Reuse the licensed center column as a real part of the manipulated assembly.
    raw = (Path(__file__).parent / "assets/darkroom/1-reel-tube.stl").read_bytes()
    vertices = []
    for i in range(struct.unpack_from("<I", raw, 80)[0]):
        triangle = struct.unpack_from("<12fH", raw, 84 + 50 * i)
        vertices.extend([np.array(triangle[k : k + 3]) * 0.001 + [0, 0, -0.06] for k in [3, 6, 9]])
    mesh(
        scene,
        path + "/LicensedCenterColumn",
        vertices,
        [list(range(i, i + 3)) for i in range(0, len(vertices), 3)],
        black,
    )
    a.cylinder_child(path, "GrippingExtension", 0.0127, 0.077, 0.0215, black)
    a.cylinder_child(
        path,
        "GripBand",
        0.01272,
        0.010,
        0.045,
        a.material("ReelGripBlue", color=[0.025, 0.15, 0.28], roughness=0.4),
    )
    # Continuous wound film is visible between the two open spiral flanges.
    angles = np.linspace(0, 8 * np.pi, 720)
    radii = np.linspace(0.018, 0.041, len(angles))
    points = []
    for r, t in zip(radii, angles):
        points.extend([[r * np.cos(t), r * np.sin(t), height] for height in [-0.054, -0.019]])
    ribbon = mesh(
        scene,
        path + "/WoundFilm",
        points,
        [[i, i + 2, i + 3, i + 1] for i in range(0, len(points) - 2, 2)],
        film,
    )
    ribbon.CreateDoubleSidedAttr(True)
    for level in [-0.058, -0.015]:
        label = "Lower" if level < -0.03 else "Upper"
        curve(
            scene,
            path + "/" + label + "Spiral",
            [[r * np.cos(t), r * np.sin(t), level] for r, t in zip(radii, angles)],
            0.0022,
            nylon,
        )
        circle = np.linspace(0, 2 * np.pi, 100)
        curve(
            scene,
            path + "/" + label + "Rim",
            [[0.0425 * np.cos(t), 0.0425 * np.sin(t), level] for t in circle],
            0.003,
            nylon,
        )
        for k, angle in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False)):
            curve(
                scene,
                path + f"/{label}Spoke{k}",
                [[r * np.cos(angle), r * np.sin(angle), level] for r in [0.014, 0.043]],
                0.003,
                nylon,
            )


def processing_vessel(scene, name, xy):
    """A molded polypropylene deep bath, with a thin rolled rim and real cavity."""
    from pxr import UsdGeom

    a, z = scene.a, scene.z
    x, y = xy
    # Preserve the five existing colliders and their contact-reporting paths.
    for suffix in ["_floor", "L", "R", "F", "B"]:
        UsdGeom.Imageable(
            scene.stage.GetPrimAtPath("/World/FaultRig/" + name + suffix)
        ).MakeInvisible()
    tint = [0.72, 0.74, 0.69] if name == "source" else [0.30, 0.34, 0.37]
    plastic = a.material(name + "MoldedPP", color=tint, roughness=0.31)
    rim = a.material(name + "PolishedRim", color=tint, roughness=0.23)
    # Traverse the outer wall upward then the rolled edge and inner wall downward.
    profile = [
        (0.107, 0.018, 0.022),
        (0.111, 0.024, 0.024),
        (0.119, 0.136, 0.026),
        (0.121, 0.139, 0.026),
        (0.123, 0.141, 0.027),
        (0.123, 0.144, 0.027),
        (0.121, 0.147, 0.026),
        (0.109, 0.147, 0.021),
        (0.105, 0.144, 0.020),
        (0.104, 0.140, 0.019),
        (0.097, 0.040, 0.016),
        (0.094, 0.035, 0.015),
    ]
    rings_mesh(
        scene,
        name + "MoldedShell",
        [rounded_ring(half, radius, z + height, xy) for half, height, radius in profile],
        plastic,
        close_profile=True,
    )
    # Interior base meets the actual support plane. Visible ribs stay out of the
    # carrier's center guide pocket and do not claim extra physical contacts.
    a.rounded(
        "/World/FaultRig/" + name + "InnerBase",
        [x, y, z + 0.0335],
        [0.192, 0.192, 0.003],
        plastic,
        0.001,
    )
    for i, offset in enumerate([-0.079, -0.064, 0.064, 0.079]):
        a.rounded(
            f"/World/FaultRig/{name}BaseRib{i}",
            [x + offset, y, z + 0.037],
            [0.004, 0.16, 0.004],
            rim,
            0.0015,
        )
    # A recessed grip flange on each side, clear of the vertical grasp corridor.
    for side in [-1, 1]:
        for k, yy in enumerate([y - 0.036, y + 0.036]):
            a.rounded(
                f"/World/FaultRig/{name}Handle{side + 1}_{k}",
                [x + side * 0.125, yy, z + 0.133],
                [0.018, 0.009, 0.012],
                plastic,
                0.003,
            )
        a.rounded(
            f"/World/FaultRig/{name}HandleGrip{side + 1}",
            [x + side * 0.135, y, z + 0.133],
            [0.010, 0.080, 0.012],
            plastic,
            0.004,
        )
    # Rounded pouring lip, at the rear corner rather than in the robot's path.
    lip_points = [
        [x + 0.080, y - 0.119, z + 0.144],
        [x + 0.092, y - 0.127, z + 0.145],
        [x + 0.110, y - 0.123, z + 0.145],
        [x + 0.119, y - 0.107, z + 0.144],
    ]
    curve(scene, name + "PouringLip", lip_points, 0.005, rim)
    for i, h in enumerate([0.065, 0.085, 0.105, 0.125]):
        a.rounded(
            f"/World/FaultRig/{name}LevelTick{i}",
            [x - 0.072, y + 0.119, z + h],
            [0.014, 0.0015, 0.0014],
            "paper",
            0.0004,
        )
    scene.panel_line(
        "/World/FaultRig/" + name + "VesselLabel",
        "STOP BATH" if name == "source" else "FIXER",
        [x + 0.012, y + 0.120, z + 0.082],
        [0.135, 0.020],
    )
    # Refractive liquid with a raised meniscus, not an opaque colored slab.
    color = [0.91, 0.84, 0.62] if name == "source" else [0.94, 0.97, 0.98]
    fluid = a.glass(name + "PhotographicSolution", color=color, ior=1.333, roughness=0.045)
    top = rounded_ring(0.103, 0.019, scene.level, xy, count=20)
    bottom = rounded_ring(0.095, 0.016, z + 0.036, xy, count=20)
    rings_mesh(scene, name + "SolutionVolume", [bottom, top], fluid, caps=True)
    curve(scene, name + "Meniscus", top + [top[0]], 0.0016, fluid)
    # Fine concentric surface relief catches the softbox reflection without foam.
    for j, r in enumerate([0.024, 0.039]):
        points = [
            [x + r * np.cos(t), y + r * np.sin(t), scene.level + 0.0002]
            for t in np.linspace(0, 2 * np.pi, 80)
        ]
        curve(scene, name + f"SurfaceRing{j}", points, 0.00045, fluid)


def developing_tank(scene):
    """Authored daylight tank with a layered seal and gripping collar."""
    a, z = scene.a, scene.z
    black = a.material("TankABS", color=[0.018, 0.021, 0.024], roughness=0.25)
    rubber = a.material("TankSeal", color=[0.011, 0.012, 0.013], roughness=0.54)
    red = a.material("TankCapRed", color=[0.38, 0.027, 0.018], roughness=0.29)
    center = [0.76, -0.39, z]
    lathe(
        scene,
        "DaylightTankBody",
        center,
        [
            (0.049, 0.003),
            (0.053, 0.007),
            (0.054, 0.014),
            (0.056, 0.128),
            (0.061, 0.133),
            (0.061, 0.142),
            (0.058, 0.147),
        ],
        black,
    )
    lathe(scene, "TankSealRing", center, [(0.061, 0.140), (0.062, 0.143), (0.062, 0.146)], rubber)
    lathe(
        scene,
        "TankLidCollar",
        center,
        [(0.061, 0.146), (0.064, 0.149), (0.064, 0.161), (0.060, 0.165), (0.056, 0.168)],
        black,
    )
    lathe(
        scene,
        "TankFlexibleCap",
        center,
        [
            (0.055, 0.166),
            (0.058, 0.169),
            (0.058, 0.174),
            (0.053, 0.178),
            (0.047, 0.179),
            (0.046, 0.177),
        ],
        red,
    )
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, 48, endpoint=False)):
        scene.cylinder(
            f"TankLidKnurl{i}",
            [center[0] + 0.063 * np.cos(angle), center[1] + 0.063 * np.sin(angle), z + 0.155],
            0.0012,
            0.011,
            black,
        )
    scene.panel_line(
        "/World/FaultRig/TankBadge", "DAYLIGHT", [0.76, -0.333, z + 0.095], [0.065, 0.015]
    )
    scene.panel_line(
        "/World/FaultRig/TankCapacity", "2 REEL", [0.76, -0.334, z + 0.075], [0.045, 0.011]
    )
