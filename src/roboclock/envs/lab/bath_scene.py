"""Authored film-processing trays and warm-bath/ice fixtures, with real contact floors."""

import numpy as np

from roboclock.envs.lab.safety_scene import SafetyEquipmentScene
from roboclock.physics.geometry import rotation_matrix

# The countdown display leans back this far from vertical, facing the rig.
DISPLAY_LEAN_DEG = 45.0
DISPLAY_GLOW = 500.0  # emissive intensity of a lit segment: full mint, not far above white
# Where the countdown plates not on the face are kept: under the worktop, hidden from every
# camera yet inside the head image's frame (scripts/lab/check_head_visibility.py projects every
# prim of the rig).
PLATE_PARK = np.array([0.0, 0.0, -0.45])
# Bath name plates: width along y, depth up the slope (m), and their tilt from horizontal.
SIGN_SIZE = (0.28, 0.052)
SIGN_TILT_DEG = 20.0
# Nominal station frame. Each episode places the whole station (dry stand, baths, countdown
# display, A5's tank) by its seeded placement (layout.apparatus, see station_frame): world
# positions come from bath_points / frame.point; the prims are authored nominal under
# /World/FaultRig.
SOURCE_XY = np.array([0.48, -0.23])
DEST_XY = np.array([0.48, 0.25])
DRY_XY = np.array([0.26, -0.23])
FILM_TOP = -0.013
FLOOR = 0.035


def bath_points(station):
    """World xy of the (dry stand, source bath, destination bath) centres."""
    return tuple(station.point(xy) for xy in (DRY_XY, SOURCE_XY, DEST_XY))


def bath_layout(task, config):
    from roboclock.envs.lab.layout import LabLayout, ObjectConfig
    from roboclock.envs.lab.mdp.state import Target
    from roboclock.envs.lab.station_frame import draw_station_frame, quat

    z = config["scene"]["table_height"]
    h, r = (0.12, 0.044) if task.task_id == "A5" else (0.105, 0.0165)
    station = draw_station_frame(task)
    dry_xy, _, dest_xy = bath_points(station)
    # Both samples start dry on the bench stand in front of the first bath.
    source = np.array([*dry_xy, z + FLOOR + h / 2 + 0.001])
    destination = np.array([*dest_xy, z + FLOOR + h / 2 + 0.001])
    target = Target(
        destination, np.array([0.09, 0.09, 0.10]), kind="tray", support_z=z + FLOOR, rim_z=z + 0.14
    )
    return LabLayout(
        {
            "sample_0": ObjectConfig(
                "sample_0",
                task.asset,
                source,
                quat(station),
                np.array([0.1, 0.3, 0.8]),
                0.035 if task.task_id == "A5" else 0.012,
                1.0,
            )
        },
        {"destination": target},
        {"sample_0": "destination"},
        target,
        z,
        160.0,
        np.zeros(3),
        {"sample_0": (r, h)},
        (np.array([-0.18, -0.63, z]), np.array([1.08, 0.63, z + 0.7])),
        apparatus=station,
    )


def sample_visual(a, path, task_id):
    from pxr import Gf, UsdGeom, UsdShade

    film = task_id == "A5"
    if film:
        from roboclock.envs.lab.darkroom_visuals import loaded_reel

        loaded_reel(a, path)
    else:
        collider = a.cylinder_child(path, "Envelope", 0.0165, 0.105, 0, a.material("polymer"), True)
        UsdGeom.Imageable(collider).MakeInvisible()
        # Enlarge the whole single-tube visual and its conservative collider together.
        path += "/EnlargedTube"
        tube_root = UsdGeom.Xform.Define(a.stage, path)
        tube_root.AddScaleOp().Set(Gf.Vec3f(1.5, 1.5, 1.5))
        blue = a.material("SampleCapOrange", color=[0.95, 0.23, 0.025], roughness=0.22)
        a.cylinder_child(
            path,
            "Tube",
            0.009,
            0.040,
            0.005,
            a.material("MicrotubePolymer", color=[0.68, 0.78, 0.82], roughness=0.18),
        )
        tip = UsdGeom.Cone.Define(a.stage, path + "/TaperedTip")
        tip.CreateRadiusAttr(0.009)
        tip.CreateHeightAttr(0.020)
        tip.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
        tip.AddRotateXOp().Set(180.0)
        UsdShade.MaterialBindingAPI.Apply(tip.GetPrim()).Bind(
            a.material("MicrotubePolymer", color=[0.68, 0.78, 0.82], roughness=0.18).material
        )
        a.cylinder_child(path, "Cap", 0.011, 0.015, 0.0275, blue)
        a.cylinder_child(
            path,
            "Sample",
            0.007,
            0.018,
            -0.005,
            a.material("SampleCue", color=[0.72, 0.66, 0.28], roughness=0.2),
        )
        a.label(path + "/SampleID", "S1", [0, 0.0092, 0.005], [0.015, 0.015])


class BathScene(SafetyEquipmentScene):
    def __init__(self, b):
        from roboclock.envs.lab.station_frame import root, station_of

        self.b, self.a, self.stage, self.task = b, b.appearance, b.stage, b.task
        self.z = b.layout.table_height
        # Everything below /World/FaultRig is authored nominal, placed by the station frame.
        self.station = station_of(b.layout)
        root(self.station, self.stage, "/World/FaultRig")
        _, self.source_xy, self.dest_xy = bath_points(self.station)
        self.surfaces = {}
        self.label_groups = {}
        self.white = self.a.material("BathEnamel", color=[0.82, 0.88, 0.90], roughness=0.24)
        self.blue = self.a.material("BathBlue", color=[0.025, 0.12, 0.23], roughness=0.28)
        self.red = self.a.material("BathRed", color=[0.8, 0.015, 0.01], emissive=True)
        self.green = self.a.material("BathGreen", color=[0.01, 0.8, 0.12], emissive=True)
        self.yellow = self.a.material("BathAmber", color=[0.9, 0.4, 0.01], emissive=True)
        film = self.task.task_id == "A5"
        self.level = self.z + (0.13 if film else 0.075)
        # A6's baths carry no name (2026-11-19): the water and the crushed ice tell them apart.
        self.bath("source", SOURCE_XY, "STOP BATH" if film else None, film, False)
        self.bath("destination", DEST_XY, "FIXER" if film else None, film, not film)
        self.countdown_display()
        # The sample starts on a dry stand on the bench: A5's reel on a plate, A6's tube in a
        # smaller one with the baths' loose guides.
        side = 0.12 if film else 0.08
        self.cube(
            "dry_floor",
            [*DRY_XY, self.z + FLOOR - 0.0075],
            [side, side, 0.015],
            self.white,
            True,
        )
        self.a.rounded(
            "/World/FaultRig/DryStandBase",
            [*DRY_XY, self.z + 0.010],
            [side + 0.01, side + 0.01, 0.020],
            self.blue,
            0.005,
        )
        if not film:
            self.guides("dry", DRY_XY)
        self.cube("TimerFoot", [0.79, 0.14, self.z + 0.009], [0.14, 0.10, 0.018], self.blue)
        self.cube("TimerStand", [0.79, 0.15, self.z + 0.105], [0.026, 0.04, 0.21], "steel")
        if film:
            from roboclock.envs.lab.darkroom_visuals import developing_tank

            developing_tank(self)
        self.update(self.task.visual_state())

    def countdown_display(self):
        """Both bath panels contain only simulation-clock countdown digits.

        The display faces the rig (-x) and leans back 45 degrees, square to the head camera,
        which sees its 70 mm digits about 20 px tall. It used to face +y, edge-on to that
        camera: the countdown the tasks are timed by could not be read.

        A segment is three plates: mint, red (a failed run) and black. One stands on the face
        and the other two are parked under the worktop; the display changes by moving plates.
        A plate that arrives from somewhere else is drawn fresh by the temporal antialiasing,
        whereas a segment switched off in place (by visibility, or by scale) kept its glow in
        the next rendered frames: a policy that renders one frame per query read 18.8 for 10.1
        and 08.8 for 06.1."""
        from pxr import Gf, Sdf, UsdGeom

        materials = {None: self.a.material("rubber")}
        for red, (name, color) in enumerate(
            (("CountdownMint", [0.2, 1.0, 0.65]), ("CountdownRed", [1.0, 0.12, 0.06]))
        ):
            materials[bool(red)] = self.a.material(name, color=color, emissive=True)
            materials[bool(red)].shaders_list[0].CreateInput(
                "emissive_intensity", Sdf.ValueTypeNames.Float
            ).Set(DISPLAY_GLOW)
        self.green = materials[False]
        lean = np.radians(DISPLAY_LEAN_DEG)
        center = np.array([0.79, 0.15, self.z + 0.23])
        # Display frame (width with the first digit at +, face normal, up) in the world.
        frame = np.array(
            [[0, -np.cos(lean), np.sin(lean)], [1, 0, 0], [0, np.sin(lean), np.cos(lean)]]
        )

        def mount(shape, local):
            """Move a part built at the origin to `local` in the display frame; returns its
            translate op and that position."""
            ops = UsdGeom.Xformable(shape).GetOrderedXformOps()
            home = center + frame @ np.asarray(local, dtype=float)
            ops[0].Set(Gf.Vec3d(*map(float, home)))
            shape.AddRotateZOp().Set(90.0)
            shape.AddRotateXOp().Set(float(DISPLAY_LEAN_DEG))
            return ops[0], home

        def plates(name, local, make):
            """The three plates of one segment, all parked: {None, False, True: (op, home)}."""
            result = {}
            for state, tag in ((None, "Off"), (False, ""), (True, "Red")):
                op, home = mount(make(name + tag, materials[state]), local)
                op.Set(Gf.Vec3d(*map(float, home + PLATE_PARK)))
                result[state] = (op, home)
            return result

        mount(
            self.cube("DisplayHousing", [0, 0, 0], [0.25, 0.016, 0.13], self.blue), [0, -0.008, 0]
        )
        mount(self.cube("CountdownFace", [0, 0, 0], [0.23, 0.001, 0.11], "rubber"), [0, 0.001, 0])
        self.digits = []
        for digit in range(3):
            segments = []
            for segment, (dx, dz, w, h) in enumerate(
                [
                    (0, 0.032, 0.028, 0.006),
                    (0.016, 0.016, 0.006, 0.028),
                    (0.016, -0.016, 0.006, 0.028),
                    (0, -0.032, 0.028, 0.006),
                    (-0.016, -0.016, 0.006, 0.028),
                    (-0.016, 0.016, 0.006, 0.028),
                    (0, 0, 0.028, 0.006),
                ]
            ):
                offset = [0.065, 0.010, -0.065][digit]
                segments.append(
                    plates(
                        f"Digit{digit}_{segment}",
                        [offset - dx, 0.003, dz],
                        lambda name, material, w=w, h=h: self.cube(
                            name, [0, 0, 0], [w, 0.001, h], material
                        ),
                    )
                )
            self.digits.append(segments)
        self.decimal = plates(
            "TimerDecimal",
            [-0.027, 0.004, -0.032],
            lambda name, material: self.cylinder(name, [0, 0, 0], 0.004, 0.001, material, "Y"),
        )
        self.segments_shown = {}

    def bath_sign(self, name, title, x_face, y, z_top):
        """Name plate on a bath's near rim (the wall toward the rig, outer face at `x_face`,
        top at `z_top`), tilted toward the head camera, which reads its letters about 9 px
        tall. The names on the baths' +y walls are edge-on or hidden from that camera."""
        from roboclock.envs.lab.liquid_scene import lay_flat

        width, depth = SIGN_SIZE
        tilt = np.radians(SIGN_TILT_DEG)
        center = np.array(
            [x_face - 0.002 - depth / 2 * np.cos(tilt), y, z_top - depth / 2 * np.sin(tilt)]
        )
        normal = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])
        lay_flat(
            self.cube(
                name + "SignPlate",
                center - 0.002 * normal,
                [width + 0.008, 0.003, depth + 0.008],
                self.blue,
            ),
            SIGN_TILT_DEG,
        )
        return lay_flat(
            self.a.label(
                "/World/FaultRig/" + name + "Sign", title, center, [width, depth], sign=True
            ),
            SIGN_TILT_DEG,
        )

    def water(self, x, y):
        """A closed refractive water volume with gentle surface ripples, visual only."""
        from pxr import Gf, UsdGeom, UsdShade

        n = 32
        points = []
        for j in range(n + 1):
            for i in range(n + 1):
                u, v = -0.102 + i * 0.204 / n, -0.102 + j * 0.204 / n
                ripple = 0.0009 * np.sin(95 * u + 20 * v) * np.cos(75 * v)
                meniscus = 0.0015 * (max(abs(u), abs(v)) / 0.102) ** 18
                points.append(Gf.Vec3f(x + u, y + v, self.level + ripple + meniscus))
        faces = []
        for j in range(n):
            for i in range(n):
                k = j * (n + 1) + i
                faces.append([k, k + 1, k + n + 2, k + n + 1])
        perimeter = (
            list(range(n + 1))
            + [j * (n + 1) + n for j in range(1, n + 1)]
            + [n * (n + 1) + i for i in range(n - 1, -1, -1)]
            + [j * (n + 1) for j in range(n - 1, 0, -1)]
        )
        base = len(points)
        for k in perimeter:
            points.append(Gf.Vec3f(points[k][0], points[k][1], self.z + FLOOR + 0.001))
        for i, k in enumerate(perimeter):
            nxt = (i + 1) % len(perimeter)
            faces.append([k, base + i, base + nxt, perimeter[nxt]])
        faces.append(list(reversed(range(base, len(points)))))
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/FaultRig/WarmWater")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([k for f in faces for k in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        material = self.a.glass("WarmWater", color=[0.85, 0.98, 1.0], ior=1.333, roughness=0.035)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material.material)
        # A darker basin floor gives the refracted ripples and waterline contrast.
        self.cube(
            "WaterBasinLiner",
            [x, y, self.z + FLOOR + 0.0005],
            [0.205, 0.205, 0.001],
            self.a.material("BasinLiner", color=[0.12, 0.48, 0.60], roughness=0.3),
        )

    def bath(self, name, xy, title, film, ice):
        x, y = xy
        z = self.z
        # Separate floor and side colliders leave the bath interior physically open.
        self.cube(
            name + "_floor", [x, y, z + FLOOR - 0.0075], [0.21, 0.21, 0.015], self.white, True
        )
        height = 0.14 if film else 0.10
        for suffix, offset, size in [
            ("L", [-0.112, 0], [0.015, 0.24, height]),
            ("R", [0.112, 0], [0.015, 0.24, height]),
            ("F", [0, 0.112], [0.21, 0.015, height]),
            ("B", [0, -0.112], [0.21, 0.015, height]),
        ]:
            self.cube(
                name + suffix,
                [x + offset[0], y + offset[1], z + height / 2],
                size,
                self.blue if ice else self.white,
                True,
            )
        liquid = self.a.material(
            name + "Fluid",
            color=[0.3, 0.18, 0.055] if film and name == "source" else [0.12, 0.34, 0.42],
            roughness=0.10,
        )
        if film:
            from roboclock.envs.lab.darkroom_visuals import processing_vessel

            processing_vessel(self, name, xy)
        elif not ice:
            self.water(x, y)
        else:
            self.cube(name + "Surface", [x, y, self.level], [0.205, 0.205, 0.002], liquid)
        # A5's instruction names its baths (STOP BATH, FIXER): each carries its name on its near
        # rim, toward the rig. A bench label in front of the first tank (2026-11-10) was 5 px
        # tall and half covered by the dry stand.
        if title:
            self.bath_sign(name, title, x - 0.123, y, z + 0.147)
        if ice:
            mat = self.a.material("CrushedIce", color=[0.70, 0.88, 0.96], roughness=0.10)
            for i in range(16):
                angle = i * 2.399
                radius = 0.045 + (i % 3) * 0.018
                self.a.rounded(
                    f"/World/FaultRig/Ice{i}",
                    [
                        x + radius * np.cos(angle),
                        y + radius * np.sin(angle),
                        self.level + 0.005 + (i % 2) * 0.008,
                    ],
                    [0.025, 0.023, 0.022],
                    mat,
                    0.006,
                )
        self.guides(name, xy)

    def guides(self, name, xy):
        """Loose square guides support the standing carrier/tube without attaching it."""
        x, y = xy
        z = self.z
        r, h = self.task.layout.dimensions["sample_0"]
        for i, (dx, dy, sx, sy) in enumerate(
            [
                (r + 0.018, 0, 0.006, 0.05),
                (-r - 0.018, 0, 0.006, 0.05),
                (0, r + 0.018, 0.05, 0.006),
                (0, -r - 0.018, 0.05, 0.006),
            ]
        ):
            self.cube(
                name + "Guide" + str(i),
                [x + dx, y + dy, z + FLOOR + 0.010],
                [sx, sy, 0.020],
                self.blue,
                True,
            )

    def measure(self, obj, dt):
        axis = rotation_matrix(obj.orientation)[:, 2]
        h = self.task.layout.dimensions["sample_0"][1]
        bottom = obj.position - axis * h / 2
        film = self.task.task_id == "A5"
        immersed_point = obj.position + axis * FILM_TOP if film else bottom
        top_z = immersed_point[2] + (0.044 * np.sqrt(max(0.0, 1 - axis[2] ** 2)) if film else 0)
        return dict(
            source_immersed=bool(
                np.linalg.norm(bottom[:2] - self.source_xy) < 0.095 and bottom[2] < self.level
            ),
            source_fully_immersed=bool(
                np.linalg.norm(immersed_point[:2] - self.source_xy) < 0.055 and top_z < self.level
            ),
            source_supported="source_floor" in obj.fixture_contacts,
            film_top_z=float(top_z),
            destination_immersed=bool(
                np.linalg.norm(immersed_point[:2] - self.dest_xy) < (0.055 if film else 0.095)
                and top_z < self.level
            ),
            sample_bottom_z=float(bottom[2]),
            sample_position=obj.position.tolist(),
            sample_orientation=obj.orientation.tolist(),
            sample_grasped=obj.grasped,
            destination_supported="destination_floor" in obj.fixture_contacts,
            gripper_width=float(sum(self.b.robot.get_joint_positions()[6:])),
            liquid_surface_z=self.level,
        )

    def update(self, data):
        from pxr import Gf

        mode = {
            "loading": "normal",
            "timing": "normal",
            "transfer": "alarm",
            "complete": "safe",
            "failed": "failure",
        }[data["phase"]]
        for key, labels in self.label_groups.items():
            for item in labels:
                item.MakeVisible() if key == mode else item.MakeInvisible()
        # Three digits show seconds and tenths; the timer is tied to simulation time.
        masks = [
            "abcdef",
            "bc",
            "abdeg",
            "abcdg",
            "bcfg",
            "acdfg",
            "acdefg",
            "abc",
            "abcdefg",
            "abcdfg",
        ]
        display = f"{int(np.ceil(max(0.0, data['remaining']) * 10 - 1e-8)):03d}"
        red = mode == "failure"
        shown = {
            (digit, letter): (red if letter in masks[int(char)] else None)
            for digit, char in enumerate(display)
            for letter in "abcdefg"
        }
        shown["decimal"] = red
        for key, state in shown.items():
            if self.segments_shown.get(key, "unset") == state:
                continue  # write on change only
            plates = (
                self.decimal if key == "decimal" else self.digits[key[0]]["abcdefg".index(key[1])]
            )
            for plate, (op, home) in plates.items():
                op.Set(Gf.Vec3d(*map(float, home if plate == state else home + PLATE_PARK)))
            self.segments_shown[key] = state
