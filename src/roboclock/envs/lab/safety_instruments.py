"""Functional, equipment-mounted B5–B7 instruments; no instructional billboards.

Authored geometry and dial artwork. Readings follow the existing scenario model.
Instruments are visual, non-contact accessories outside the established
manipulation surfaces.
"""

import hashlib
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent / "assets/safety_equipment/instruments"
ROOT = "/World/FaultRig"
LAMP_STATES = ("off", "red", "green", "amber")
# Lens colour and emission, as on B2's status lens: a saturated colour at 640x480.
LAMP_COLORS = {
    "red": ([1.0, 0.01, 0.004], 80.0),
    "green": ([0.01, 0.85, 0.07], 60.0),
    "amber": ([1.0, 0.42, 0.005], 80.0),
}
PARKED_Z = -0.5  # m: the prims of inactive lamp states and segments wait below the floor
# B6's gauge console: dial centre, and how far its face leans back from vertical toward the
# head camera (a face on the +y side of the chamber was seen edge-on).
GAUGE_XY = (0.64, 0.28)
GAUGE_HEIGHT = 0.105
GAUGE_ELEVATION = 50.0
# Lens centres in the console frame (+x world +y, +y the dial's normal, +z its top) and radii.
PRESSURE_LAMP = ((-0.135, -0.027, 0.040), 0.020)
BEACON = ((-0.135, -0.027, -0.050), 0.022)
LEVEL_LAMP_RADIUS = 0.018  # 36 mm lens: about 15 px in the head image (24 mm was 10 x 8 px)
DOOR_LAMP_RADIUS = 0.014
MASKS = ("abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")
# B7's level-alarm console sits under the task arm's resting wrist camera.
LEVEL_MONITOR_XY = (0.38, -0.10)


def _placed(point, frame):
    """Nominal world point (as authored under /World/FaultRig) placed by the seed's frame."""
    return point if frame is None else frame.point(point)


def level_lamp_position(table_z, frame=None):
    """World point of B7's upward-facing LEVEL lamp (nominal without the apparatus frame)."""
    point = np.array([LEVEL_MONITOR_XY[0] + 0.010, LEVEL_MONITOR_XY[1], table_z + 0.067])
    return _placed(point, frame)


DOOR_LAMP_DESIGN = (0.465 - 0.116, -0.078 + 0.010, 0.508)  # freezer design frame, above table


def door_lamp_position(table_z, frame=None):
    """World point of B5's door-alarm lamp on the freezer's control fascia."""
    from roboclock.envs.lab.safety_geometry import freezer_point

    x, y, dz = DOOR_LAMP_DESIGN
    return _placed(freezer_point([x, y, table_z + dz]), frame)


def gauge_console_point(local, table_z, frame=None):
    """World point of a point given in the frame of B6's gauge console."""
    x, y, z = local
    lean = np.deg2rad(GAUGE_ELEVATION)
    point = np.array(
        [
            GAUGE_XY[0] - y * np.cos(lean) + z * np.sin(lean),
            GAUGE_XY[1] + x,
            table_z + GAUGE_HEIGHT + y * np.sin(lean) + z * np.cos(lean),
        ]
    )
    return _placed(point, frame)


def dial_angle(pressure_kpa):
    """0–110 kPa ABS, clockwise 270-degree sweep; atmosphere is near full scale."""
    return 225.0 - 270.0 * float(np.clip(pressure_kpa, 0, 110)) / 110.0


class SafetyInstruments:
    def __init__(self, scene):
        from pxr import Sdf

        self.s, self.a, self.stage = scene, scene.a, scene.stage
        self.z = scene.z
        self.readout = []
        self.decimals = []
        self.lamps = {}
        self.lamp_shown = {}
        self.placed = {}
        self.last_display = None
        self.last_pressure = None
        self.alarm_since = None
        self.ink = self.a.material("InstrumentInk", color=[0.008, 0.012, 0.014], roughness=0.5)
        self.face = self.a.material("InstrumentFace", color=[0.018, 0.028, 0.035], roughness=0.35)
        self.lit = self.a.material("InstrumentDigits", color=[0.12, 0.85, 0.55], emissive=True)
        self.lit.shaders_list[0].GetInput("emissive_intensity").Set(1200.0)
        self.alarm_digits = self.a.material(
            "InstrumentAlarmDigits", color=[0.95, 0.04, 0.015], emissive=True
        )
        self.alarm_digits.shaders_list[0].GetInput("emissive_intensity").Set(1200.0)
        self.off = self.a.material("InstrumentUnlitLED", color=[0.08, 0.055, 0.04], roughness=0.25)
        for key, (color, intensity) in LAMP_COLORS.items():
            mat = self.a.material("InstrumentLED_" + key, color=color, roughness=0.2, emissive=True)
            mat.shaders_list[0].CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(
                intensity
            )
            setattr(self, key, mat)
        {"B5": self.freezer, "B6": self.vacuum, "B7": self.waste}[scene.task.task_id]()

    def box(self, name, pos, size, mat, bevel=0.003, root=ROOT):
        return self.a.rounded(root + "/Instrument" + name, pos, size, mat, bevel)

    def text(
        self, name, text, pos, size, *, background=(18, 25, 31), ink=(190, 220, 210), root=ROOT
    ):
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Sdf, UsdShade

        ASSETS.mkdir(parents=True, exist_ok=True)
        aspect = size[1] / size[0]
        key = hashlib.sha256(repr((text, aspect, background, ink)).encode()).hexdigest()[:20]
        texture = ASSETS / (key + ".png")
        if not texture.exists():
            w, h = 1024, max(80, round(1024 * aspect))
            im = Image.new("RGB", (w, h), background)
            draw = ImageDraw.Draw(im)
            font_size = int(h * 0.80)
            while True:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
                )
                if draw.textlength(text, font=font) < w * 0.94:
                    break
                font_size -= 1
            draw.text((w / 2, h / 2), text, font=font, fill=ink, anchor="mm")
            im.save(texture)
        path = root + "/Instrument" + name
        mesh = self.a.label(path, text, pos, size)
        tex = UsdShade.Shader(self.stage.GetPrimAtPath(path + "_material/Texture"))
        tex.GetInput("file").Set(Sdf.AssetPath(str(texture)))
        return mesh

    def variant(self, shape, park=None):
        """(translate op, shown position, parked position) of one pre-coloured prim.

        Lamps and readout segments change by moving such prims, never by a material,
        visibility or scale edit: those reach the policy cameras late. Measured in Isaac Sim
        6.0.1 with the rig's renderer (DLSS ray reconstruction), reading the head image every
        0.1 s: B7's LEVEL lamp, switched by scaling a red lens up over the green one, had moved
        7 % toward red in the first frame after the alarm and 1.5 % per frame after that; with
        the red lens arriving from under the floor it is fully red in that first frame.
        """
        from pxr import Gf, UsdGeom

        translate = UsdGeom.Xformable(shape.GetPrim()).GetOrderedXformOps()[0]
        home = Gf.Vec3d(translate.Get())
        parked = Gf.Vec3d(home[0], home[1], PARKED_Z) if park is None else Gf.Vec3d(*park(home))
        return translate, home, parked

    def show(self, variant, shown):
        """Put a prim in its place or park it; written only when that changes."""
        translate, home, parked = variant
        if self.placed.get(translate.GetAttr().GetPath()) != shown:
            self.placed[translate.GetAttr().GetPath()] = shown
            translate.Set(home if shown else parked)

    def digits(self, center, count, height, *, decimal=False, root=ROOT):
        """Seven-segment readout: per segment one prim for each colour and one unlit (`variant`)."""
        from pxr import Gf, UsdGeom, UsdShade

        def segment(name, pos, size, mat):
            cube = UsdGeom.Cube.Define(self.stage, root + "/" + name)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
            cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat.material)
            return self.variant(cube)

        x, y, z = center
        spacing = height * 0.68
        for digit in range(count):
            segments = []
            cx = x + ((count - 1) / 2 - digit) * spacing
            for index, (dx, dz, w, h) in enumerate(
                [
                    (0, 0.45, 0.36, 0.075),
                    (0.22, 0.225, 0.075, 0.35),
                    (0.22, -0.225, 0.075, 0.35),
                    (0, -0.45, 0.36, 0.075),
                    (-0.22, -0.225, 0.075, 0.35),
                    (-0.22, 0.225, 0.075, 0.35),
                    (0, 0, 0.36, 0.075),
                ]
            ):
                segments.append(
                    {
                        key: segment(
                            f"InstrumentDigit{digit}_{index}_{key}",
                            [cx - dx * height, y, z + dz * height],
                            [w * height, 0.0007, h * height],
                            mat,
                        )
                        for key, mat in (
                            ("lit", self.lit),
                            ("alarm", self.alarm_digits),
                            ("unlit", self.face),
                        )
                    }
                )
            self.readout.append(segments)
        if decimal:
            dot_x = x - (count / 2 - 1) * spacing
            self.decimals.append(
                {
                    key: self.variant(
                        self.s.cylinder(
                            "InstrumentDecimal_" + key,
                            [dot_x, y, z - height * 0.45],
                            height * 0.045,
                            0.0008,
                            mat,
                            "Y",
                            root=root,
                        )
                    )
                    for key, mat in (("lit", self.lit), ("alarm", self.alarm_digits))
                }
            )

    def led(self, name, pos, radius=0.006, axis="Y", *, bulge=0.65, root=ROOT, park=None):
        """Dome lens on a steel bezel, one pre-coloured sphere per lamp state (as B2's lens).

        `pos` is in the frame of `root`; `bulge` is how far the dome stands out of its bezel,
        as a fraction of its radius (1 = a hemisphere, for a lamp seen from the side).
        """
        from pxr import Gf, UsdGeom, UsdShade

        back = np.array([0, 0.002, 0] if axis == "Y" else [0, 0, 0.002])
        self.s.cylinder(
            "InstrumentBezel" + name,
            np.asarray(pos) - back,
            radius * 1.35,
            0.004,
            "steel",
            axis,
            root=root,
        )
        self.lamps[name] = {}
        for key in LAMP_STATES:
            lens = UsdGeom.Sphere.Define(self.stage, f"{root}/InstrumentLED{name}_{key}")
            lens.CreateRadiusAttr(radius)
            lens.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
            lens.AddScaleOp().Set(Gf.Vec3f(1, bulge, 1) if axis == "Y" else Gf.Vec3f(1, 1, bulge))
            UsdShade.MaterialBindingAPI.Apply(lens.GetPrim()).Bind(getattr(self, key).material)
            self.lamps[name][key] = self.variant(lens, park)

    def flat_text(self, name, text, pos, size):
        """Console-top text that reads upright from the head and wrist cameras."""
        mesh = self.text(name, text, pos, size)
        mesh.AddRotateZOp().Set(90.0)
        mesh.AddRotateXOp().Set(90.0)
        return mesh

    def screws(self, x, y, z, w, h, root=ROOT):
        for i, (dx, dz) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            self.s.cylinder(
                "InstrumentScrew" + str(i),
                [x + dx * w / 2, y, z + dz * h / 2],
                0.002,
                0.0015,
                "steel",
                "Y",
                root=root,
            )

    def freezer(self):
        from roboclock.envs.lab.safety_scene import FREEZER_ROOT

        # Design-frame coordinates (fascia facing +y) under the freezer's frame, which turns
        # the fascia toward the rig (safety_geometry.FREEZER_YAW).
        root = FREEZER_ROOT
        x, y, z = 0.465, -0.078, self.z + 0.50
        # Fixed control fascia above the insulated door; no stand or remote sign.
        self.box(
            "FreezerHeader", [x, -0.24, z - 0.003], [0.35, 0.30, 0.095], self.s.white, 0.006, root
        )
        self.box("FreezerBezel", [x, y, z], [0.276, 0.014, 0.071], self.face, 0.004, root)
        self.digits([x + 0.020, y + 0.008, z + 0.003], 4, 0.043, decimal=True, root=root)
        unit = [x - 0.078, y + 0.009, z + 0.005]
        self.text("TemperatureUnit", "°C", unit, [0.035, 0.026], root=root)
        probe = [x + 0.015, y + 0.009, z - 0.026]
        self.text("ProbeLabel", "CHAMBER", probe, [0.095, 0.010], root=root)
        # A protruding pilot lamp (a flat 10 mm LED was 5 x 2 px when the fascia was seen
        # obliquely); the dome is about 14 x 10 px.
        lamp_x, lamp_y, lamp_dz = DOOR_LAMP_DESIGN
        self.led(
            "door", [lamp_x, lamp_y, self.z + lamp_dz], DOOR_LAMP_RADIUS, bulge=1.0, root=root
        )
        legend = [x - 0.112, y + 0.010, z - 0.022]
        self.text("DoorLegend", "DOOR", legend, [0.030, 0.009], root=root)
        self.screws(x, y + 0.011, z, 0.26, 0.055, root=root)
        for i in range(2):
            self.s.cylinder(
                "InstrumentKey" + str(i),
                [x + 0.104 - i * 0.027, y + 0.010, z - 0.020],
                0.008,
                0.004,
                "rubber",
                "Y",
                root=root,
            )
        plate = [x, -0.075, self.z + 0.438]
        self.text("FreezerPlate", "ULT  /  -80 °C", plate, [0.12, 0.013], root=root)

    def vacuum(self):
        """Gauge console on the bench beyond the VENT plunger, leaning back to face the head
        camera: the dial with its needle, the red pressure lamp and the amber beacon."""
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        r = 0.078
        root = ROOT + "/GaugeConsole"
        # Console frame: +y is the dial's normal, +z its top, +x world +y (image left).
        console = UsdGeom.Xform.Define(self.stage, root)
        console.AddTranslateOp().Set(Gf.Vec3d(*GAUGE_XY, self.z + GAUGE_HEIGHT))
        console.AddRotateZOp().Set(90.0)
        console.AddRotateXOp().Set(GAUGE_ELEVATION)

        def part(name, pos, radius, height, mat, axis="Y"):
            return self.s.cylinder(name, pos, radius, height, mat, axis, root=root)

        brass = self.a.material(
            "GaugeBrass", color=[0.45, 0.29, 0.075], metallic=0.8, roughness=0.23
        )
        self.box(
            "GaugePanel", [-0.055, -0.037, -0.005], [0.31, 0.012, 0.20], self.s.white, root=root
        )
        self.box(
            "GaugeStand",
            [GAUGE_XY[0] + 0.07, GAUGE_XY[1] - 0.055, self.z + 0.04],
            [0.06, 0.24, 0.08],
            self.s.white,
        )
        part("InstrumentGaugeStem", [0, -0.012, -0.088], 0.012, 0.022, brass, "Z")
        part("InstrumentGaugeCase", [0, -0.012, 0], r, 0.038, "steel")
        part("InstrumentGaugeGasket", [0, 0.009, 0], r * 0.94, 0.007, "rubber")
        part("InstrumentGaugeFace", [0, 0.014, 0], r * 0.90, 0.004, self.s.white)
        ASSETS.mkdir(parents=True, exist_ok=True)
        texture = ASSETS / "absolute_pressure_dial.png"
        if not texture.exists():
            im = Image.new("RGB", (1024, 1024), (237, 235, 221))
            draw = ImageDraw.Draw(im)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 52)
            for value in range(0, 111, 2):
                theta = np.deg2rad(dial_angle(value))
                major = value % 10 == 0

                def at(rad):
                    return (512 + rad * np.cos(theta), 512 - rad * np.sin(theta))

                draw.line(
                    [at(450), at(405 if major else 428)], fill=(28, 32, 34), width=6 if major else 3
                )
                if major:
                    draw.text(at(350), str(value), font=font, fill=(28, 32, 34), anchor="mm")
            draw.text((512, 340), "kPa ABS", font=font, fill=(28, 32, 34), anchor="mm")
            draw.text((512, 740), "VACUUM", font=font, fill=(28, 32, 34), anchor="mm")
            im.save(texture)
        # Reuse the label material, but apply it to a circular dial (no square corners).
        label = self.text("DialMaterial", "", [0, 0, -10], [0.1, 0.1])
        path = str(label.GetPath())
        UsdShade.Shader(self.stage.GetPrimAtPath(path + "_material/Texture")).GetInput("file").Set(
            Sdf.AssetPath(str(texture))
        )
        material = UsdShade.Material(self.stage.GetPrimAtPath(path + "_material"))
        UsdGeom.Imageable(label).MakeInvisible()
        points, uv = [Gf.Vec3f(0, 0.017, 0)], [(0.5, 0.5)]
        for i in range(96):
            angle = i * 2 * np.pi / 96
            u, v = np.cos(angle), np.sin(angle)
            points.append(Gf.Vec3f(-r * 0.895 * u, 0.017, r * 0.895 * v))
            uv.append(((u + 1) / 2, (v + 1) / 2))
        mesh = UsdGeom.Mesh.Define(self.stage, root + "/InstrumentDial")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * 96)
        mesh.CreateFaceVertexIndicesAttr(
            [j for i in range(96) for j in (0, i + 1, (i + 1) % 96 + 1)]
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(uv)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        needle = UsdGeom.Xform.Define(self.stage, root + "/InstrumentNeedle")
        needle.AddTranslateOp().Set(Gf.Vec3d(0, 0.019, 0))
        self.needle_rotation = needle.AddRotateYOp()
        pointer = UsdGeom.Mesh.Define(self.stage, str(needle.GetPath()) + "/Pointer")
        # A broad pointer: 8 mm at the hub is about 3 px in the head image.
        pointer.CreatePointsAttr([(0.016, 0, -0.008), (-0.064, 0, 0), (0.016, 0, 0.008)])
        pointer.CreateFaceVertexCountsAttr([3])
        pointer.CreateFaceVertexIndicesAttr([0, 1, 2])
        pointer.CreateDoubleSidedAttr(True)
        pointer.CreateSubdivisionSchemeAttr("none")
        UsdShade.MaterialBindingAPI.Apply(pointer.GetPrim()).Bind(
            self.a.material("GaugePointer", color=[0.65, 0.015, 0.008], roughness=0.25).material
        )
        part("InstrumentNeedleHub", [0, 0.020, 0], 0.009, 0.004, "steel")

        # Lamps on the panel beside the dial; their idle prims wait far behind the panel.
        def behind(home):
            return home[0], home[1] - 1.6, home[2]

        self.led("pressure", *PRESSURE_LAMP, root=root, park=behind)
        self.text("PressureLegend", "PRESSURE", [-0.135, -0.0305, -0.003], [0.07, 0.016], root=root)
        self.led("beacon", *BEACON, bulge=1.0, root=root, park=behind)
        self.text("BeaconLegend", "ALARM", [-0.135, -0.0305, -0.092], [0.05, 0.014], root=root)
        self.s.wire(
            "InstrumentPressureLine",
            [
                [GAUGE_XY[0] + 0.07, GAUGE_XY[1] - 0.17, self.z + 0.03],
                [0.70, 0.04, self.z + 0.10],
                [0.68, -0.10, self.z + 0.35],
            ],
            [0.07, 0.075, 0.08],
        )

    def waste(self):
        # Lamps face up so the head and resting wrist cameras see the level change.
        x, y = LEVEL_MONITOR_XY
        top = self.z + 0.0655
        self.box("LevelMonitor", [x, y, self.z + 0.0355], [0.12, 0.17, 0.055], self.s.white, 0.009)
        self.box("LevelMonitorFace", [x, y, top - 0.0015], [0.105, 0.155, 0.003], self.face, 0.001)
        self.flat_text(
            "LevelMonitorTitle", "LEVEL ALARM", [x + 0.044, y, top + 0.0005], [0.12, 0.016]
        )
        lamp = level_lamp_position(self.z)
        self.led("power", [lamp[0], y + 0.05, lamp[2]], 0.007, "Z")
        self.led("level", lamp, LEVEL_LAMP_RADIUS, "Z")
        self.led("flow", [lamp[0], y - 0.05, lamp[2]], 0.007, "Z")
        for label, offset, size in [
            ("POWER", 0.05, [0.04, 0.012]),
            ("LEVEL", 0, [0.05, 0.016]),
            ("PUMP", -0.05, [0.04, 0.012]),
        ]:
            self.flat_text(label, label, [x - 0.030, y + offset, top + 0.0005], size)
        for i in range(5):
            self.box(
                "BuzzerSlot" + str(i),
                [x - 0.038 - i * 0.003, y + 0.05, top + 0.0003],
                [0.0012, 0.027, 0.001],
                self.ink,
                0.0003,
            )
        self.s.cylinder(
            "InstrumentMuteButton", [x - 0.045, y - 0.05, top + 0.002], 0.007, 0.005, "rubber"
        )
        for i, dx in enumerate([-0.04, 0.04]):
            self.box(
                "LevelFoot" + str(i), [x + dx, y, self.z + 0.004], [0.03, 0.14, 0.008], self.ink
            )
        # The IR level sensor on the floor carboy reports to this console.
        self.s.wire(
            "InstrumentSensorCable",
            [
                [x + 0.06, y + 0.05, self.z + 0.03],
                [x + 0.09, y + 0.07, self.z + 0.005],
                [0.70, 0.10, self.z + 0.005],
                [0.84, 0.42, self.z + 0.02],
                [0.97, 0.58, self.z + 0.02],
                [0.97, 0.77, 0.47],
                [0.935, 0.88, 0.44],
            ],
            [0.018, 0.020, 0.022],
        )
        self.s.cylinder("InstrumentSensorHousing", [0.935, 0.88, 0.425], 0.020, 0.028, "steel")
        self.box("IRProbe", [0.925, 0.88, 0.405], [0.025, 0.030, 0.035], self.ink)

    def update(self, data, mode):
        # 3 Hz flashing that starts lit at the alarm (a phase tied to the clock left the
        # lamps dark for up to 0.17 s after it).
        if mode != "alarm":
            self.alarm_since = None
        elif self.alarm_since is None:
            self.alarm_since = data["time"]
        blink = mode == "alarm" and int((data["time"] - self.alarm_since) * 6 + 1e-6) % 2 == 0
        task_id = self.s.task.task_id
        if task_id == "B5":
            text = f"{int(round(data['temperature_c'] * 10)):04d}"
        else:
            text = ""
        # Keep readings legible during alarms; only the warning lamps flash.
        display_key = (text, mode)
        if self.last_display != display_key:
            colour = "alarm" if mode in ("alarm", "failure") else "lit"
            for char, segments in zip(text, self.readout):
                mask = "g" if char == "-" else MASKS[int(char)]
                for letter, variants in zip("abcdefg", segments):
                    # An unlit segment shows its bezel-coloured prim: one that only left
                    # would stay in the policy cameras' image for many frames.
                    segment = colour if letter in mask else "unlit"
                    for key, variant in variants.items():
                        self.show(variant, key == segment)
            for variants in self.decimals:
                for key, variant in variants.items():
                    self.show(variant, key == colour)
            self.last_display = display_key
        if task_id == "B6" and data["pressure_kpa"] != self.last_pressure:
            self.last_pressure = data["pressure_kpa"]
            self.needle_rotation.Set(dial_angle(data["pressure_kpa"]))
        full = data["alarm"] or data["failed"] or data["secured"]
        lamp_materials = {
            "door": self.red if mode == "failure" or (mode == "alarm" and blink) else self.off,
            "pressure": self.red if mode == "failure" or (mode == "alarm" and blink) else self.off,
            "beacon": self.amber if mode == "alarm" and blink else self.off,
            "power": self.green,
            # Steady green while below the IR trigger, steady red once full. Closing the
            # inflow does not empty the container, so the red indication is retained.
            "level": self.red if full else self.green,
            "flow": self.green if data.get("flow_ml_s", 0) > 0 else self.off,
        }
        for name, variants in self.lamps.items():
            # Written on change only: see SafetyEquipmentScene.update.
            if self.lamp_shown.get(name) is lamp_materials[name]:
                continue
            self.lamp_shown[name] = lamp_materials[name]
            for key, variant in variants.items():
                self.show(variant, getattr(self, key) is lamp_materials[name])
