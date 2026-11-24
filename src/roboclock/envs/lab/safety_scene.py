"""Authored, articulated B5–B7 equipment in the shared laboratory room.

CC0 equipment meshes use authored PhysX hinges, sliders and contact surfaces;
the passive catches latch only after a measured stroke with robot contact.
"""

import numpy as np

from roboclock.envs.lab.reactive_scene import ReactiveEquipmentScene
from roboclock.envs.lab.safety_geometry import (
    DESIGN_XY,
    FREEZER_XY,
    FREEZER_YAW,
    body_hinge_offset,
    hinge_angle,
    joint_frames,
    mechanism,
)

FREEZER_ROOT = "/World/FaultRig/Freezer"


def freezer_frame(stage):
    """Xform whose children are authored in the freezer's design frame (safety_geometry)."""
    from pxr import Gf, UsdGeom

    root = UsdGeom.Xform.Define(stage, FREEZER_ROOT)
    root.AddTranslateOp(opSuffix="place").Set(Gf.Vec3d(*map(float, FREEZER_XY), 0.0))
    root.AddRotateZOp().Set(float(np.rad2deg(FREEZER_YAW)))
    root.AddTranslateOp(opSuffix="design").Set(Gf.Vec3d(*map(float, -DESIGN_XY), 0.0))
    return FREEZER_ROOT


def imported(appearance, path, name, position, *, glass=False):
    from pxr import Gf, Usd, UsdGeom, UsdShade

    from roboclock.resources import resolve_asset

    root = UsdGeom.Xform.Define(appearance.stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    root.GetPrim().GetReferences().AddReference(
        str(resolve_asset("lab://safety_equipment/" + name + ".usdc"))
    )
    if glass:
        for prim in Usd.PrimRange(root.GetPrim()):
            if prim.IsA(UsdGeom.Mesh) and prim.GetCustomDataByKey("source_material") == 0:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(appearance.glass().material)
    return root


def actuator_geometry(appearance, path, task_id):
    from pxr import Gf, UsdGeom, UsdPhysics

    white = appearance.material("SafetyEnamel", color=[0.83, 0.88, 0.90], roughness=0.27)
    blue = appearance.material("SafetyBlue", color=[0.015, 0.13, 0.29], roughness=0.26)
    red = appearance.material("SafetyButton", color=[0.75, 0.035, 0.015], roughness=0.25)

    def box(name, pos, size, mat):
        collider = UsdGeom.Cube.Define(appearance.stage, path + "/" + name + "Collider")
        collider.CreateSizeAttr(1.0)
        collider.AddTranslateOp().Set(Gf.Vec3d(*pos))
        collider.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdGeom.Imageable(collider).MakeInvisible()
        appearance.rounded(path + "/" + name, pos, size, mat, 0.003)

    if task_id == "B5":
        # Body frame: +y out of the door (toward the rig once yawed), hinge at +x.
        box("InsulatedDoor", [0.09, -0.055, -0.23], [0.29, 0.035, 0.40], white)
        box("Grip", [0, 0, 0], [0.075, 0.022, 0.024], blue)
        for i, x in enumerate([-0.027, 0.027]):
            box("GripMount" + str(i), [x, -0.022, -0.026], [0.012, 0.056, 0.054], blue)
        appearance.label(path + "/Label", "ULT", [0.09, -0.036, -0.17], [0.15, 0.055])
    else:
        box("Plunger", [0, 0, -0.025], [0.026, 0.026, 0.07], appearance.material("steel"))
        box("PushPad", [0, 0, 0.015], [0.085, 0.065, 0.024], red if task_id == "B6" else blue)
        # Bold, and upright in the head image (up = +x): the old label's letters were 5 mm
        # (2 px) and read mirrored from above.
        label = appearance.label(
            path + "/Press",
            "VENT" if task_id == "B6" else "STOP",
            [0, 0, 0.0275],
            [0.06, 0.04],
            sign=True,
        )
        label.AddRotateZOp().Set(90.0)
        label.AddRotateXOp().Set(90.0)


class SafetyEquipmentScene(ReactiveEquipmentScene):
    def __init__(self, backend):
        from pxr import UsdGeom

        self.b, self.a, self.stage, self.task = (
            backend,
            backend.appearance,
            backend.stage,
            backend.task,
        )
        self.z = backend.layout.table_height
        self.surfaces = {}
        self.secured = False
        self.contact_dwell = 0.0
        self.grasp_seen = False
        self.fault_released = False
        self.opening_started = False
        self.opening_reached = False
        from roboclock.envs.lab.layout import apparatus_frame

        # Prims below are authored in nominal coordinates under /World/FaultRig, which carries
        # the seed's apparatus placement; the joint (world-attached) uses the placed geometry.
        self.frame = apparatus_frame(backend.layout)
        self.frame.author_xform(self.stage, "/World/FaultRig")
        self.spec = mechanism(self.task.task_id, self.z, self.frame)
        self.white = self.a.material("SafetyEnamel", color=[0.83, 0.88, 0.90], roughness=0.27)
        self.blue = self.a.material("SafetyBlue", color=[0.015, 0.13, 0.29], roughness=0.26)
        self.red = self.a.material("AlarmRed", color=[0.9, 0.008, 0.003], emissive=True)
        self.yellow = self.a.material("AlarmAmber", color=[0.95, 0.45, 0.005], emissive=True)
        self.green = self.a.material("SafeGreen", color=[0.01, 0.75, 0.12], emissive=True)
        self.label_groups = {}
        self.written_mode = None
        self.build_joint()
        {"B5": self.freezer, "B6": self.vacuum, "B7": self.waste}[self.task.task_id]()
        self.failure_visuals = []
        if self.task.task_id in ("B6", "B7"):
            liquid = self.a.material("HazardSpill", color=[0.85, 0.13, 0.015], roughness=0.10)
            # Visual consequence only: no fluid solver or collision-backed puddle.
            for i in range(14):
                if self.task.task_id == "B6":
                    dx = 0.075 * np.sin(i * 4.3)
                    pos = [
                        0.68 + dx,
                        -0.10 + np.sqrt(0.093**2 - dx**2),
                        self.z + 0.22 + 0.035 * np.cos(i * 2.7),
                    ]
                    shape = self.cylinder(
                        "BumpingDrop" + str(i), pos, 0.006 + i % 3 * 0.002, 0.001, liquid, "Y"
                    )
                else:
                    pos = [0.84 + 0.16 * np.sin(i * 4.3), 0.81 + 0.12 * np.cos(i * 2.7), 0.003]
                    shape = self.cylinder("FloorSpill" + str(i), pos, 0.085, 0.002, liquid)
                imageable = UsdGeom.Imageable(shape.GetPrim())
                imageable.MakeInvisible()
                self.failure_visuals.append(imageable)
        from roboclock.envs.lab.safety_instruments import SafetyInstruments

        self.instruments = SafetyInstruments(self)
        self.update(self.task.visual_state())

    def build_joint(self):
        from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

        hinged = self.task.task_id == "B5"
        cls = UsdPhysics.RevoluteJoint if hinged else UsdPhysics.PrismaticJoint
        self.joint = cls.Define(self.stage, "/World/FaultRig/ActuatorJoint")
        self.joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Objects/sample_0")])
        self.joint.CreateAxisAttr(self.spec["axis"])
        self.joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, self.spec["pivot"])))
        self.joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, body_hinge_offset(self.spec))))
        rot0, rot1 = joint_frames(self.spec)
        self.joint.CreateLocalRot0Attr(Gf.Quatf(*map(float, rot0)))
        self.joint.CreateLocalRot1Attr(Gf.Quatf(*map(float, rot1)))
        self.joint.CreateLowerLimitAttr(-1.0 if hinged else -0.024)
        self.joint.CreateUpperLimitAttr(40.0 if hinged else 0.0)
        self.drive = UsdPhysics.DriveAPI.Apply(
            self.joint.GetPrim(), "angular" if hinged else "linear"
        )
        self.drive.CreateTypeAttr("force")
        self.drive.CreateStiffnessAttr(20.0 if hinged else 80.0)
        self.drive.CreateDampingAttr(2.0 if hinged else 2.0)
        self.drive.CreateTargetPositionAttr(
            float(np.rad2deg(self.spec["angle"])) if hinged else 0.0
        )
        self.drive.CreateMaxForceAttr(2.0 if hinged else 10.0)
        # Counterbalanced hinges and spring plungers: no hidden position updates.
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            self.b.objects["sample_0"].prim
        ).CreateDisableGravityAttr(True)

    def release_fault(self):
        """An internal catch fault lets the counterbalanced door spring open."""
        if not self.fault_released:
            self.fault_released = True
            self.opening_started = True
            self.drive.GetTargetPositionAttr().Set(float(np.rad2deg(self.spec["open_angle"])))
            self.drive.GetStiffnessAttr().Set(0.25)
            self.drive.GetDampingAttr().Set(0.08)

    def feet(self, x, y, width, depth, root="/World/FaultRig"):
        for i, (dx, dy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            self.cylinder(
                "Foot" + str(i),
                [x + dx * width / 2, y + dy * depth / 2, self.z + 0.008],
                0.017,
                0.016,
                "rubber",
                root=root,
            )

    def displays(self, center, title, labels, width=0.23):
        from pxr import UsdGeom

        x, y, z = center
        self.cube("DisplayHousing", [x, y - 0.008, z], [width + 0.025, 0.016, 0.13], self.blue)
        # Separate lines keep long failure messages readable with the standard label atlas.
        for mode, lines in labels.items():
            self.label_groups[mode] = []
            for i, line in enumerate(lines):
                mesh = self.panel_line(
                    "/World/FaultRig/Display_" + mode + str(i),
                    line,
                    [x, y + 0.001, z + 0.025 - i * 0.035],
                    [width, 0.031],
                    [0.7, 0.025, 0.01] if mode == "failure" else [0.02, 0.25, 0.4],
                )
                self.label_groups[mode].append(UsdGeom.Imageable(mesh.GetPrim()))
        self.panel_line("/World/FaultRig/Title", title, [x, y + 0.001, z + 0.052], [width, 0.017])
        self.cylinder("RedAlarm", [x + width / 2 + 0.025, y, z], 0.011, 0.009, self.red, "Y")
        self.cylinder("GreenStatus", [x - width / 2 - 0.025, y, z], 0.011, 0.009, self.green, "Y")
        self.cylinder("BeaconBase", [x + width / 2, y - 0.025, z + 0.085], 0.017, 0.01, "rubber")
        self.cylinder(
            "AmberBeacon", [x + width / 2, y - 0.025, z + 0.110], 0.014, 0.04, self.yellow
        )
        self.digits = []
        for digit in range(4):
            segments = []
            for segment, (dx, dz, w, h) in enumerate(
                [
                    (0, 0.010, 0.009, 0.002),
                    (0.005, 0.005, 0.002, 0.009),
                    (0.005, -0.005, 0.002, 0.009),
                    (0, -0.010, 0.009, 0.002),
                    (-0.005, -0.005, 0.002, 0.009),
                    (-0.005, 0.005, 0.002, 0.009),
                    (0, 0, 0.009, 0.002),
                ]
            ):
                mesh = self.cube(
                    f"Digit{digit}_{segment}",
                    [x + (1.5 - digit) * 0.018 - dx, y + 0.002, z - 0.044 + dz],
                    [w, 0.001, h],
                    self.green,
                )
                segments.append(mesh.GetPrim())
            self.digits.append(segments)

    def panel_line(self, path, text, center, size, color=(0.02, 0.25, 0.4)):
        """Fit actual equipment text without cropped text or decorative serial bars."""
        import hashlib
        from pathlib import Path

        from PIL import Image, ImageDraw, ImageFont
        from pxr import Sdf, UsdShade

        mesh = self.a.label(path, text, center, size, color)
        directory = Path.home() / ".cache/roboclock/safety_displays"
        directory.mkdir(parents=True, exist_ok=True)
        texture = directory / (hashlib.sha256((text + str(color)).encode()).hexdigest() + ".png")
        if not texture.exists():
            im = Image.new("RGB", (1024, 160), (12, 22, 30))
            draw = ImageDraw.Draw(im)
            font_size = 110
            while True:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
                )
                if draw.textlength(text, font=font) <= 980:
                    break
                font_size -= 2
            ink = (255, 65, 40) if color[0] > 0.5 else (130, 240, 225)
            draw.text((512, 80), text, font=font, fill=ink, anchor="mm")
            im.save(texture)
        tex = UsdShade.Shader(self.stage.GetPrimAtPath(path + "_material/Texture"))
        tex.GetInput("file").Set(Sdf.AssetPath(str(texture)))
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(path + "_material/Shader"))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        return mesh

    def frame_cube(self, root, name, pos, size, material, collision=False):
        """A rounded box under a transformed root; a static box collider when `collision`."""
        from pxr import Gf, UsdGeom, UsdPhysics

        self.a.rounded(root + "/" + name, pos, size, material)
        if collision:
            collider = UsdGeom.Cube.Define(self.stage, root + "/" + name + "Collider")
            collider.CreateSizeAttr(1.0)
            collider.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
            collider.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            UsdGeom.Imageable(collider).MakeInvisible()
            self.surfaces[name] = str(collider.GetPath())

    def freezer(self):
        # Design-frame coordinates (door toward +y); freezer_frame turns it to face the rig.
        root = freezer_frame(self.stage)
        x, y = DESIGN_XY
        self.feet(x, y, 0.28, 0.25, root=root)
        for name, pos, size in [
            ("CabinetBack", [x, y - 0.14, self.z + 0.235], [0.35, 0.035, 0.44]),
            ("CabinetLeft", [0.302, y, self.z + 0.235], [0.025, 0.30, 0.44]),
            ("CabinetRight", [0.628, y, self.z + 0.235], [0.025, 0.30, 0.44]),
            ("CabinetRoof", [x, y, self.z + 0.443], [0.35, 0.30, 0.024]),
            ("CabinetFloor", [x, y, self.z + 0.029], [0.35, 0.30, 0.024]),
        ]:
            self.frame_cube(root, name, pos, size, self.white, True)
        for k in range(2):
            shelf_z = self.z + 0.09 + k * 0.15
            self.frame_cube(
                root, "ColdShelf" + str(k), [x, y + 0.015, shelf_z], [0.29, 0.25, 0.009], "steel"
            )
            for j in range(3):
                self.frame_cube(
                    root,
                    f"SampleBox{k}_{j}",
                    [x - 0.09 + j * 0.09, y + 0.045, shelf_z + 0.034],
                    [0.075, 0.15, 0.06],
                    self.blue,
                )
        for name, pos, size in [
            ("GasketTop", [0.465, -0.085, self.z + 0.42], [0.29, 0.017, 0.010]),
            ("GasketBottom", [0.465, -0.085, self.z + 0.03], [0.29, 0.017, 0.010]),
            ("GasketLeft", [0.323, -0.085, self.z + 0.225], [0.010, 0.017, 0.39]),
            ("GasketRight", [0.607, -0.085, self.z + 0.225], [0.010, 0.017, 0.39]),
        ]:
            self.frame_cube(root, name, pos, size, self.a.material("rubber"))
        for i in range(8):
            self.frame_cube(
                root,
                "CompressorVent" + str(i),
                [0.644, y - 0.06 + i * 0.018, self.z + 0.10],
                [0.002, 0.006, 0.10],
                self.a.material("rubber"),
            )

    def plunger_base(self, title):
        # Leave a guide bore for the stem throughout its full physical stroke.
        for name, pos, size in [
            ("BaseLeft", [0.345, 0.16, self.z + 0.045], [0.05, 0.13, 0.09]),
            ("BaseRight", [0.455, 0.16, self.z + 0.045], [0.05, 0.13, 0.09]),
            ("BaseFront", [0.40, 0.21, self.z + 0.045], [0.06, 0.03, 0.09]),
            ("BaseBack", [0.40, 0.11, self.z + 0.045], [0.06, 0.03, 0.09]),
            ("BaseFloor", [0.40, 0.16, self.z + 0.006], [0.16, 0.13, 0.012]),
        ]:
            self.cube(name, pos, size, self.white, True)
        for dx in [-0.055, 0.055]:
            self.cylinder(
                "Bolt" + str(dx).replace("-", "m").replace(".", "p"),
                [0.40 + dx, 0.16, self.z + 0.095],
                0.005,
                0.01,
                "steel",
            )
        self.a.label(
            "/World/FaultRig/ActuatorLabel", title, [0.40, 0.226, self.z + 0.05], [0.145, 0.032]
        )

    def vacuum(self):
        from pxr import UsdShade

        from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

        self.plunger_base("VACUUM BREAKER")
        x, y = 0.68, -0.10
        self.cube("VacuumBase", [x, y, self.z + 0.045], [0.30, 0.30, 0.09], self.white, True)
        self.feet(x, y, 0.24, 0.24)
        chamber = author_mesh(
            self.stage,
            "/World/FaultRig/GlassChamber",
            revolved([(0.127, 0.09), (0.127, 0.34), (0.12, 0.34), (0.12, 0.095), (0.127, 0.09)]),
            [x, y, self.z],
        )
        UsdShade.MaterialBindingAPI.Apply(chamber.GetPrim()).Bind(self.a.glass().material)
        from pxr import UsdGeom

        UsdGeom.Imageable(chamber).MakeInvisible()
        imported(
            self.a,
            "/World/FaultRig/ImportedDesiccator",
            "desiccator",
            [x, y, self.z + 0.09],
            glass=True,
        )
        self.cylinder("ChamberLid", [x, y, self.z + 0.345], 0.133, 0.015, "steel", collision=True)
        self.cylinder("ChamberKnob", [x, y, self.z + 0.37], 0.025, 0.04, "rubber")
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/FaultRig/ChamberLid")).MakeInvisible()
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/FaultRig/ChamberKnob")).MakeInvisible()
        self.cylinder("SampleCup", [x, y, self.z + 0.145], 0.047, 0.10, self.a.glass())
        self.cylinder(
            "SampleLiquid",
            [x, y, self.z + 0.12],
            0.044,
            0.045,
            self.a.material("VacuumLiquid", color=[0.85, 0.32, 0.015], roughness=0.14),
        )
        # Inside the RoboDojo head camera image (was at 0.31, -0.26, beyond its right edge).
        px, py = 0.40, -0.20
        self.cube("VacuumPump", [px, py, self.z + 0.07], [0.20, 0.14, 0.14], self.blue, True)
        self.cylinder("PumpMotor", [px, py, self.z + 0.17], 0.045, 0.15, "aluminum", "X")
        self.wire(
            "VacuumHose",
            [
                [x, y, self.z + 0.35],
                [0.52, -0.10, self.z + 0.40],
                [0.40, 0.16, self.z + 0.09],
                [px, py, self.z + 0.10],
            ],
            [0.08, 0.09, 0.10],
        )

    def waste(self):
        from pxr import UsdShade

        from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

        self.plunger_base("PINCH TO STOP")
        # The clamp is reachable on the bench; the carboy is in an external floor tray.
        x, y = 0.88, 0.88
        self.cube("SpillTray", [x, y, 0.025], [0.38, 0.34, 0.05], self.blue, True)
        body = author_mesh(
            self.stage,
            "/World/FaultRig/Carboy",
            revolved(
                [
                    (0, 0.055),
                    (0.11, 0.055),
                    (0.12, 0.08),
                    (0.12, 0.34),
                    (0.09, 0.39),
                    (0.04, 0.42),
                    (0.04, 0.445),
                    (0.035, 0.445),
                    (0.035, 0.42),
                    (0.085, 0.385),
                    (0.113, 0.335),
                    (0.113, 0.065),
                    (0, 0.065),
                ]
            ),
            [x, y, 0],
        )
        UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(self.a.glass().material)
        from pxr import UsdGeom

        UsdGeom.Imageable(body).MakeInvisible()
        imported(self.a, "/World/FaultRig/ImportedCarboy", "carboy", [x, y, 0.055])
        self.cylinder(
            "WasteLiquid",
            [x, y, 0.205],
            0.112,
            0.28,
            self.a.material("WasteAmber", color=[0.82, 0.20, 0.01], roughness=0.12),
        )
        self.cylinder("CarboyCap", [x, y, 0.45], 0.046, 0.025, "rubber")
        funnel = author_mesh(
            self.stage,
            "/World/FaultRig/WasteFunnel",
            revolved(
                [
                    (0.012, 0.45),
                    (0.012, 0.49),
                    (0.085, 0.56),
                    (0.085, 0.565),
                    (0.080, 0.565),
                    (0.008, 0.49),
                    (0.008, 0.45),
                ]
            ),
            [x, y, 0],
        )
        UsdShade.MaterialBindingAPI.Apply(funnel.GetPrim()).Bind(
            self.a.material("polymer").material
        )
        self.a.label(
            "/World/FaultRig/WasteLabel", "HAZARDOUS WASTE", [x, y + 0.12, 0.23], [0.19, 0.075]
        )
        self.cube("Pump", [0.60, -0.20, self.z + 0.065], [0.20, 0.15, 0.13], self.white, True)
        self.cylinder("PumpHead", [0.60, -0.12, self.z + 0.07], 0.045, 0.018, self.blue, "Y")
        self.wire(
            "WasteHose",
            [
                [0.60, -0.11, self.z + 0.07],
                [0.40, 0.04, self.z + 0.09],
                [0.40, 0.16, self.z + 0.09],
                [0.40, 0.28, self.z + 0.09],
                [0.72, 0.52, self.z + 0.09],
                [x, 0.70, self.z + 0.09],
                [x, y, 0.70],
                [x, y, 0.57],
            ],
            [0.72, 0.40, 0.12],
        )
        self.ir_sensor = self.cylinder("IRSensor", [x + 0.055, y, 0.44], 0.016, 0.035, self.green)

    def measure(self, obj, dt):
        w, x, y, z = obj.orientation
        hinged = self.task.task_id == "B5"
        angle = (
            float(
                hinge_angle(self.spec, np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            )
            if self.task.task_id == "B5"
            else float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        )
        travel = float(self.spec["pivot"][2] - obj.position[2]) if not hinged else 0.0
        if hinged:
            self.opening_reached |= bool(self.opening_started and angle >= np.deg2rad(5))
        self.grasp_seen |= obj.grasped
        reached = angle < np.deg2rad(2.0) if hinged else travel >= 0.016
        contact_valid = (
            obj.robot_contact
            and (self.grasp_seen or self.task.task_id != "B5")
            and (not hinged or self.opening_reached)
        )
        self.contact_dwell = self.contact_dwell + dt if reached and contact_valid else 0.0
        if not self.secured and self.contact_dwell >= 0.05 - 1e-9:
            self.secured = True
            # Model the mechanical catch at the measured stop; never move the body.
            self.drive.GetTargetPositionAttr().Set(0.0 if hinged else -0.022)
            self.drive.GetStiffnessAttr().Set(50.0 if hinged else 2000.0)
            self.drive.GetDampingAttr().Set(2.0 if hinged else 20.0)
        if self.secured and ((hinged and angle > np.deg2rad(5)) or (not hinged and travel < 0.012)):
            self.secured = False
        controls = {
            "secured": self.secured,
            "angle_rad": angle if hinged else 0.0,
            "travel_m": travel,
            "robot_contact": obj.robot_contact,
            "contact_dwell": self.contact_dwell,
            "grasp_seen": self.grasp_seen,
            "fault_released": self.fault_released,
            "opening_reached": self.opening_reached,
        }
        return controls

    def update(self, data):
        from pxr import UsdShade

        mode = (
            "failure"
            if data["failed"]
            else "safe"
            if data["secured"]
            else "alarm"
            if data["alarm"]
            else "normal"
        )
        for name, items in self.label_groups.items():
            for item in items:
                item.MakeVisible() if name == mode else item.MakeInvisible()
        self.instruments.update(data, mode)
        # Write on change only, and steady rather than blinking at 8 Hz: these materials
        # were edited and the sensor re-bound on every tick, which kept the RTX scene in
        # flux (measured on B7's LEVEL lamp while it was still switched by scale: about 10 %
        # toward red in the first frame after the alarm with the per-tick writes, 60 % without).
        if mode == self.written_mode:
            return
        self.written_mode = mode
        for material, intensity in [
            (self.red, 1000.0 if mode in ("alarm", "failure") else 0.0),
            (self.yellow, 800.0 if mode == "alarm" else 0.0),
            (self.green, 500.0 if mode in ("safe", "normal") else 0.0),
        ]:
            material.shaders_list[0].GetInput("emissive_intensity").Set(intensity)
        if data["failed"]:
            for item in self.failure_visuals:
                item.MakeVisible()
        if self.task.task_id == "B7":
            UsdShade.MaterialBindingAPI.Apply(self.ir_sensor.GetPrim()).Bind(
                (self.green if mode == "normal" else self.red).material
            )
