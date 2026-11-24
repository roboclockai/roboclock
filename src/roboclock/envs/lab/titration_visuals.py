"""Refractive A4 liquids, calibrated burette and volume-driven drop delivery.

These are geometric/optical process cues, not a CFD or acid/base chemistry solver.
"""

import numpy as np

BURETTE_CAPACITY_ML = 50.0
BURETTE_LENGTH = 0.37
BURETTE_RADIUS = np.sqrt(BURETTE_CAPACITY_ML / 1e6 / (np.pi * BURETTE_LENGTH))
BURETTE_BASE = 0.35
FLASK_BASE = 0.066
INITIAL_SOLUTION_ML = 120.0
DROP_ML = 0.04
SEGMENTS = 64
CLEAR = np.array([0.999, 0.999, 1.0])


def review_camera(backend, *, closeup=False):
    """Show the full apparatus first, then the liquid and falling drops."""
    from isaacsim.core.utils.rotations import gf_quat_to_np_array
    from pxr import Gf

    position, target = (
        ([0.68, 0.64, 1.04], [0.50, 0.15, 0.90])
        if closeup
        else ([0.91, 1.60, 1.55], [0.50, 0.15, 1.15])
    )
    from roboclock.envs.lab.station_frame import station_of

    # Authored against the nominal station; follow the episode's station frame.
    station = station_of(backend.task)
    position, target = station.point(position), station.point(target)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    backend.camera.set_world_pose(
        np.array(position),
        gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat()),
        camera_axes="usd",
    )
    backend.camera.set_focal_length(0.035)


DYE = np.array([0.015, 1.0, 0.24])  # relative absorption of the pink form in R, G, B
MAX_ABSORPTION = 0.065 + 0.58  # at 2 mL of excess titrant


def indicator_absorption(delivered_ml):
    """0 before the endpoint, 0.065 at 18 mL, rising to MAX_ABSORPTION on overshoot."""
    onset = np.clip((delivered_ml - 17.6) / 0.4, 0, 1)
    onset = onset * onset * (3 - 2 * onset)
    excess = np.clip((delivered_ml - 18.0) / 2.0, 0, 1)
    return 0.065 * onset + 0.58 * excess**1.35


def indicator_tint(delivered_ml):
    """Clear before the endpoint; faint pink at 18 mL, saturated on overshoot."""
    return np.exp(-indicator_absorption(delivered_ml) * DYE)


# The policy cameras look down into the flask, where the solution's colour shows against the
# flask's floor: light has crossed the liquid on the way in, on the way out and in between by
# scattering, about three passes of `indicator_tint`. The floor is a stack of FLOOR_STEPS discs
# of fixed colours, one shown at a time (see build_titration).
FLOOR_PASSES = 3.0
FLOOR_STEPS = 24
# A pale grey rather than white: the bench lighting renders an albedo of 0.5 as full white, and
# a pink tint on a saturated white does not show.
FLOOR_WHITE = np.array([0.40, 0.41, 0.41])
FLOOR_HIDDEN = 1e-3  # scale of the discs not shown


def floor_step(delivered_ml):
    """Index of the floor disc shown at this delivered volume (0 = before the endpoint)."""
    fraction = indicator_absorption(delivered_ml) / MAX_ABSORPTION
    return int(round(float(fraction) * (FLOOR_STEPS - 1)))


def floor_color(step):
    """Colour of floor disc `step`: pale grey, then pale pink (step 2, the 18 mL endpoint: green
    down by a sixth), then strong pink at the last step (2 mL of excess titrant)."""
    absorption = MAX_ABSORPTION * step / (FLOOR_STEPS - 1)
    return FLOOR_WHITE * np.exp(-FLOOR_PASSES * absorption * DYE)


def burette_profile(delivered_ml):
    height = BURETTE_LENGTH * np.clip(1 - delivered_ml / BURETTE_CAPACITY_ML, 0, 1)
    r = BURETTE_RADIUS
    # Graduations read the bottom of the concave meniscus, not its raised rim.
    return [(0, 0), (r, 0), (r, height + 0.00055), (0.75 * r, height + 0.00012), (0, height)]


def flask_profile(total_ml):
    # Invert the integral of the actual flask's inner cross section.
    z = np.linspace(0.003, 0.108, 600)
    r = np.interp(z, [0.003, 0.009, 0.022, 0.108], [0.0408, 0.0483, 0.0483, 0.0133])
    area = np.pi * r**2
    volume = np.r_[0, np.cumsum((area[1:] + area[:-1]) * 0.5 * np.diff(z))] * 1e6
    height = float(np.interp(total_ml, volume, z))
    levels = np.linspace(0.003, height, 14)
    radii = np.interp(levels, z, r)
    top = radii[-1]
    # A shallow stir vortex and a wetting meniscus give a visible free surface.
    return (
        [(0, 0.003)]
        + list(zip(radii, levels))
        + [
            (top, height + 0.0003),
            (top * 0.92, height),
            (top * 0.65, height - 0.00015),
            (top * 0.25, height - 0.00055),
            (0, height - 0.0006),
        ]
    )


def set_profile(mesh, profile):
    from pxr import Vt

    from roboclock.simulators.isaac.visual_geometry import smooth_revolved_normals

    angle = np.linspace(0, 2 * np.pi, SEGMENTS + 1)
    profile = np.asarray(profile)
    points = np.zeros((len(profile), SEGMENTS + 1, 3), dtype=np.float32)
    points[:, :, 0] = profile[:, 0, None] * np.cos(angle)
    points[:, :, 1] = profile[:, 0, None] * np.sin(angle)
    points[:, :, 2] = profile[:, 1, None]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.reshape(-1, 3)))
    smooth_revolved_normals(mesh, profile, SEGMENTS)


def build_titration(s, pivot):
    from pxr import Gf, UsdGeom, UsdShade

    from roboclock.envs.lab.liquid_scene import no_shadow
    from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

    s.stand(0.68, -0.08, 0.78)
    glass = s.a.glass()
    # Real hollow bore and tapered glass discharge tip, with no beaker label.
    s.lathe(
        "/World/Apparatus/Burette",
        [0.50, 0.15, s.z],
        [
            (0.0014, 0.245),
            (0.0028, 0.245),
            (0.0028, 0.31),
            (0.0082, 0.345),
            (0.0082, 0.758),
            (0.009, 0.76),
            (0.007, 0.76),
            (0.007, 0.35),
            (0.0014, 0.31),
        ],
        glass,
    )
    titrant = s.a.glass("A4Titrant", color=CLEAR, ior=1.333, roughness=0.012)
    titrant.shaders_list[0].GetInput("depth").Set(0.05)
    s.titrant_mesh = author_mesh(
        s.stage,
        "/World/Process/Titrant",
        revolved(burette_profile(0), SEGMENTS),
        [0.50, 0.15, s.z + BURETTE_BASE],
    )
    UsdShade.MaterialBindingAPI.Apply(s.titrant_mesh.GetPrim()).Bind(titrant.material)
    no_shadow(s.titrant_mesh.GetPrim())
    # A reading card behind the bore and black, calibrated mL graduations.
    s.a.cube(
        "/World/Apparatus/BuretteReadingCard",
        [0.514, 0.135, s.z + 0.535],
        [0.042, 0.001, 0.40],
        "paper",
    )
    for ml in range(51):
        level = s.z + BURETTE_BASE + BURETTE_LENGTH * (1 - ml / 50)
        major = ml % 5 == 0
        s.a.cube(
            f"/World/Apparatus/BuretteTick{ml}",
            [0.50, 0.1586, level],
            [0.011 if major else 0.005, 0.00025, 0.00035],
            "rubber",
        )
        if major:
            s.a.label(
                f"/World/Apparatus/BuretteNumber{ml}",
                str(ml),
                [0.516, 0.159, level],
                [0.014, 0.006],
                printed=True,
            )
    s.a.label(
        "/World/Apparatus/BuretteUnits",
        "mL",
        [0.516, 0.159, s.z + 0.739],
        [0.015, 0.008],
        printed=True,
    )
    for i, level in enumerate([0.39, 0.67]):
        s.a.cube(
            f"/World/Apparatus/BuretteArm{i}",
            [0.59, -0.08, s.z + level],
            [0.18, 0.012, 0.012],
            "steel",
        )
        s.a.cube(
            f"/World/Apparatus/BuretteClamp{i}",
            [0.50, 0.028, s.z + level],
            [0.012, 0.216, 0.012],
            "steel",
        )
    s.a.cube(
        "/World/Apparatus/Stirrer", [0.50, 0.15, s.z + 0.029], [0.20, 0.21, 0.058], "ceramic", True
    )
    white = s.a.material("WhiteStirPlate", color=[0.95, 0.95, 0.93], roughness=0.28)
    s.lathe(
        "/World/Apparatus/StirPlate",
        [0.50, 0.15, s.z + 0.060],
        [(0, 0), (0.071, 0), (0.073, 0.001), (0.073, 0.005), (0.071, 0.006), (0, 0.006)],
        white,
    )
    # A 250 mL flask with an explicit inner floor and conical wall.
    s.lathe(
        "/World/Apparatus/Flask",
        [0.50, 0.15, s.z + FLASK_BASE],
        [
            (0, 0),
            (0.042, 0),
            (0.05, 0.008),
            (0.05, 0.022),
            (0.016, 0.108),
            (0.016, 0.14),
            (0.0135, 0.14),
            (0.0135, 0.108),
            (0.0485, 0.022),
            (0.0485, 0.009),
            (0.041, 0.003),
            (0, 0.003),
        ],
        glass,
    )
    solution = s.a.glass("A4IndicatorSolution", color=CLEAR, ior=1.333, roughness=0.016)
    solution.shaders_list[0].GetInput("depth").Set(0.04)
    s.titration_solution_material = solution
    s.titration_solution_mesh = author_mesh(
        s.stage,
        "/World/Process/Solution",
        revolved(flask_profile(INITIAL_SOLUTION_ML), SEGMENTS),
        [0.50, 0.15, s.z + FLASK_BASE],
    )
    UsdShade.MaterialBindingAPI.Apply(s.titration_solution_mesh.GetPrim()).Bind(solution.material)
    no_shadow(s.titration_solution_mesh.GetPrim())
    # The indicator's colour, shown where a camera above the flask sees it: on the flask's floor,
    # under the stir bar. Through the refractive flask and solution alone the white stir plate
    # rendered black, and the endpoint could not be seen (patch [11, 17, 19] to [21, 27, 29]).
    # One disc per colour step, each with its own fixed material; update_titration shows one by
    # scale. A material whose colour is rewritten reaches the RTX image many frames late, which
    # a policy that renders one frame per query never catches up with; a scale shows in the next
    # rendered frame (each 1 s step across the endpoint came out distinct).
    s.titration_floor = []
    for step in range(FLOOR_STEPS):
        disc = s.cylinder(
            f"/World/Process/SolutionFloor{step}",
            [0.50, 0.15, s.z + FLASK_BASE + 0.0037],
            0.0405,
            0.001,
            s.a.material(
                f"A4IndicatorFloor{step}", color=floor_color(step), roughness=0.6, metallic=0.0
            ),
        )
        no_shadow(disc.GetPrim())
        scale = disc.AddScaleOp()
        scale.Set(Gf.Vec3f(1.0) if step == 0 else Gf.Vec3f(FLOOR_HIDDEN))
        s.titration_floor.append(scale)
    s.titration_floor_step = 0
    s.a.rounded(
        "/World/Apparatus/StirBar",
        [0.50, 0.15, s.z + FLASK_BASE + 0.007],
        [0.026, 0.006, 0.005],
        "paper",
        0.0024,
    )
    # Drops stay aligned with the open flask neck and terminate at the liquid surface.
    UsdGeom.Xform.Define(s.stage, "/World/Process/Stream")
    s.titration_drops = []
    for i in range(9):
        drop = UsdGeom.Sphere.Define(s.stage, f"/World/Process/Stream/Drop{i}")
        drop.CreateRadiusAttr(float((3 * DROP_ML / 1e6 / (4 * np.pi)) ** (1 / 3)))
        translate = drop.AddTranslateOp()
        scale = drop.AddScaleOp()
        scale.Set(Gf.Vec3f(0.90, 0.90, 1.08))
        UsdShade.MaterialBindingAPI.Apply(drop.GetPrim()).Bind(titrant.material)
        no_shadow(drop.GetPrim())
        s.titration_drops.append((drop, translate, scale))
    # The pipe stops at the stirrer's edge: run on to y = 0.10 it crossed the flask in the head
    # camera's image and covered a quarter of the solution.
    s.valve(pivot, pipe_end=0.045)
    s.a.label(
        "/World/Apparatus/StirLabel", "STIR  350 RPM", [0.50, 0.259, s.z + 0.032], [0.155, 0.024]
    )


def update_titration(s, data):
    from pxr import Gf, UsdGeom

    delivered = data["volume_ml"]
    s.stir_rotation.Set(float((s.task.time * 350 * 6) % 360))
    profile = flask_profile(INITIAL_SOLUTION_ML + delivered)
    set_profile(s.titration_solution_mesh, profile)
    set_profile(s.titrant_mesh, burette_profile(delivered))
    s.fluid_color(s.titration_solution_material, indicator_tint(delivered))
    step = floor_step(delivered)
    if step != s.titration_floor_step:
        s.titration_floor[s.titration_floor_step].Set(Gf.Vec3f(FLOOR_HIDDEN))
        s.titration_floor[step].Set(Gf.Vec3f(1.0))
        s.titration_floor_step = step
    flow = data["flow_ml_s"]
    phase = (delivered / DROP_ML) % 1
    nozzle = s.z + 0.245
    surface = s.z + FLASK_BASE + profile[-1][1]
    for i, (drop, translate, scale) in enumerate(s.titration_drops):
        if i == 0:
            z = nozzle - 0.0012 * phase
            size = max(0.15, phase ** (1 / 3))
            scale.Set(Gf.Vec3f(size * 0.8, size * 0.8, size))
        else:
            age = (phase + i - 1) * DROP_ML / max(flow, 1e-9)
            z = nozzle - 0.5 * 9.81 * age**2
        translate.Set(Gf.Vec3d(0.50, 0.15, z if z > surface else surface))
        imageable = UsdGeom.Imageable(drop)
        imageable.MakeVisible() if flow > 0.02 and z > surface else imageable.MakeInvisible()
