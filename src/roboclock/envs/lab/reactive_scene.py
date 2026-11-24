"""Repo-native USD lab apparatus for B2–B4 in the shared v2 room.

Only fault injection is scripted. Switch travel, motor rotation and tube
motion are simulated. The disconnect includes the bundled CC-BY Sam Feng asset.
"""

import numpy as np

# B2's load and B3's motor share one compact bench spot between the disconnect
# and the supply: inside the head view and the task arm's resting wrist view.
DEVICE_XY = (0.51, -0.05)
STATUS_LED_XY = (DEVICE_XY[0] + 0.008, DEVICE_XY[1] + 0.044)
STATUS_LED_RADIUS = 0.014
WHEEL_HEIGHT = 0.088
WHEEL_RADIUS = 0.075
# Moderate emission keeps the hue: at 400 the small red lens bloomed to orange.
STATUS_COLORS = {
    "green": ([0.01, 0.85, 0.07], 60.0),
    "red": ([1.0, 0.01, 0.004], 80.0),
    "off": ([0.06, 0.07, 0.06], None),
}


# The bench STATE lamp: blue while waiting, red from the task's event, green once resolved.
STATE_LAMP_RADIUS = 0.024
STATE_COLORS = {
    "blue": ([0.02, 0.10, 1.0], 80.0),
    "red": ([1.0, 0.01, 0.004], 80.0),
    "green": ([0.01, 0.85, 0.07], 60.0),
}


class StateLamp:
    """Upward-facing lens of the bench STATE indicator, for the head camera.

    backend.py builds the indicator as a coloured plate on the +y face of its housing, for
    the former review camera on that side; the RoboDojo head camera looks along +x from
    above and sees that face edge-on (no colour at all). This lens on the housing's top
    shows the same state. It reads the colour the task last commanded for the indicator,
    and changes by moving pre-coloured lenses (see safety_instruments.variant: material
    edits reach the policy cameras late).
    """

    def __init__(self, backend):
        from pxr import Gf, UsdGeom, UsdShade

        from roboclock.envs.lab.mdp.events import BLUE, GREEN, RED

        self.material = backend.indicator_material
        self.reference = {"blue": BLUE, "green": GREEN, "red": RED}
        self.shown = None
        a, stage = backend.appearance, backend.stage
        top = backend.indicator_position(backend.task) + np.array([0.0, 0.0, 0.045])
        a.rounded(
            "/World/Equipment/StateLampCap", top + [0, 0, 0.002], [0.072, 0.064, 0.004], "ceramic"
        )
        bezel = UsdGeom.Cylinder.Define(stage, "/World/Equipment/StateLampBezel")
        bezel.CreateRadiusAttr(STATE_LAMP_RADIUS * 1.25)
        bezel.CreateHeightAttr(0.004)
        bezel.AddTranslateOp().Set(Gf.Vec3d(*map(float, top + [0, 0, 0.006])))
        UsdShade.MaterialBindingAPI.Apply(bezel.GetPrim()).Bind(a.material("steel").material)
        self.lenses = {}
        for key, (color, intensity) in STATE_COLORS.items():
            material = a.material("StateLamp_" + key, color=color, roughness=0.2, emissive=True)
            material.shaders_list[0].GetInput("emissive_intensity").Set(intensity)
            lens = UsdGeom.Sphere.Define(stage, "/World/Equipment/StateLamp_" + key)
            lens.CreateRadiusAttr(STATE_LAMP_RADIUS)
            home = Gf.Vec3d(*map(float, top + [0, 0, 0.008]))
            translate = lens.AddTranslateOp()
            lens.AddScaleOp().Set(Gf.Vec3f(1, 1, 0.7))
            UsdShade.MaterialBindingAPI.Apply(lens.GetPrim()).Bind(material.material)
            self.lenses[key] = (translate, home, Gf.Vec3d(home[0], home[1], -0.5))
        if backend.config["visuals"]["labels"]:
            # Flat on the bench in front of the housing, upright in the head image.
            sign = a.label(
                "/World/Equipment/StateLampSign",
                "STATE",
                [top[0] - 0.066, top[1], backend.layout.table_height + 0.001],
                [0.12, 0.045],
                sign=True,
            )
            sign.AddRotateZOp().Set(90.0)
            sign.AddRotateXOp().Set(90.0)
        self.update()

    def update(self):
        color = np.asarray(self.material.shaders_list[0].GetInput("diffuseColor").Get())
        state = min(self.reference, key=lambda key: np.linalg.norm(color - self.reference[key]))
        if state != self.shown:
            self.shown = state
            for key, (translate, home, parked) in self.lenses.items():
                translate.Set(home if key == state else parked)


def cue_position(task_id, table_z, frame=None):
    """World point of the visible fault cue: B2's status LED or B3's test wheel.

    Nominal (as authored under /World/FaultRig) unless the seed's apparatus `frame` is given.
    """
    if task_id == "B2":
        point = np.array([*STATUS_LED_XY, table_z + 0.099])
    else:
        point = np.array([*DEVICE_XY, table_z + WHEEL_HEIGHT])
    return point if frame is None else frame.point(point)


class ReactiveEquipmentScene:
    def __init__(self, backend):
        self.b = backend
        self.a, self.stage, self.task = backend.appearance, backend.stage, backend.task
        self.z = backend.layout.table_height
        self.surfaces = {}
        self.power_on = True
        self.press_dwell = 0.0
        self.red = self.a.material("EmergencyRed", color=[0.72, 0.025, 0.015], roughness=0.24)
        self.yellow = self.a.material("SafetyYellow", color=[0.92, 0.60, 0.015])
        self.green = self.a.material("PowerGreen", color=[0.02, 0.8, 0.12], emissive=True)
        self.pilot_blue = self.a.material("PowerPilotBlue", color=[0.12, 0.38, 1.0], emissive=True)
        # Not in B2: there the STATE indicator stays blue through the fault, and the load's
        # status lens is the only green/red cue.
        self.state_lamp = None if self.task.task_id == "B2" else StateLamp(backend)
        source = backend.layout.objects["sample_0"].position
        if getattr(self.task, "tube_task", False):
            from roboclock.envs.lab.tube_liquid_visuals import TubeLiquidVisuals

            tube = backend.layout.objects["sample_0"]
            self.task.liquid.step(tube.position, tube.orientation, np.zeros(3), 0.0, self.z)
            self.tube_liquid = TubeLiquidVisuals(backend)
            self.tube_liquid.update()
            self.rack(source)
            self.tube_support(source)
            return
        from roboclock.envs.lab.knife_switch_scene import build_disconnect
        from roboclock.envs.lab.layout import apparatus_frame

        # B2/B3 apparatus prims are authored in nominal coordinates under /World/FaultRig,
        # which carries the seed's placement; world-attached joints and the motor wheel
        # body use self.frame.point(...). The layout's handle is already placed.
        self.frame = apparatus_frame(backend.layout)
        self.frame.author_xform(self.stage, "/World/FaultRig")
        source = self.frame.inverse_point(source)
        build_disconnect(self, source)
        self.supply(source)
        if self.task.task_id == "B2":
            self.thermal_load()
        else:
            self.motor()
        self.circuit_wiring()
        self.update(self.task.visual_state())

    def cube(self, name, pos, size, material="polymer", collision=False):
        path = "/World/FaultRig/" + name
        shape = self.a.cube(path, pos, size, material, collision=collision)
        if collision:
            self.surfaces[name] = path
        return shape

    def cylinder(
        self,
        name,
        pos,
        radius,
        height,
        material="steel",
        axis="Z",
        collision=False,
        root="/World/FaultRig",
    ):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        shape = UsdGeom.Cylinder.Define(self.stage, root + "/" + name)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        shape.CreateAxisAttr(axis)
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        mat = self.a.material(material) if isinstance(material, str) else material
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat.material)
        if collision:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            if root == "/World/FaultRig":
                self.surfaces[name] = str(shape.GetPath())
        return shape

    def wire(self, name, points, color):
        from pxr import Gf, UsdGeom, UsdShade

        curve = UsdGeom.BasisCurves.Define(self.stage, "/World/FaultRig/" + name)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        curve.CreateWidthsAttr([0.005])
        curve.SetWidthsInterpolation("constant")
        UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(
            self.a.material(name, color=color, roughness=0.45).material
        )

    def supply(self, source):
        x, y = 0.40, -0.26
        self.cube("Supply", [x, y, self.z + 0.075], [0.22, 0.16, 0.15], "polymer", True)
        self.cube("SupplyPanel", [x, y + 0.081, self.z + 0.075], [0.205, 0.003, 0.13], "ceramic")
        self.a.label(
            "/World/FaultRig/SupplyLabel", "DC 12 V", [x, y + 0.084, self.z + 0.12], [0.14, 0.032]
        )
        self.a.label(
            "/World/FaultRig/SupplyType",
            "REGULATED SUPPLY",
            [x, y + 0.084, self.z + 0.035],
            [0.19, 0.025],
        )
        # B2's only green/red cue is the load's status lens; a green supply pilot
        # beside it read as "still normal" while the robot was already reacting.
        pilot = self.pilot_blue if self.task.task_id == "B2" else self.green
        self.led = self.cylinder(
            "SupplyLED", [x - 0.065, y + 0.086, self.z + 0.075], 0.008, 0.006, pilot, "Y"
        )
        self.supply_terminals = {}
        for i, polarity in enumerate(["+", "-"]):
            point = np.array([x + 0.035 + i * 0.03, y + 0.092, self.z + 0.075])
            self.supply_terminals[polarity] = point
            self.cylinder(
                "Terminal" + str(i),
                point - [0, 0.004, 0],
                0.009,
                0.012,
                self.red if i == 0 else "rubber",
                "Y",
            )
            self.a.label(
                "/World/FaultRig/Polarity" + str(i),
                polarity,
                point + [0, 0.002, 0.022],
                [0.016, 0.014],
            )

    def circuit_wiring(self):
        """Four continuous cables connect labelled supply, disconnect and load posts."""
        z = self.z + 0.005
        s, d, load = self.supply_terminals, self.switch_terminals, self.load_terminals
        routes = {
            # Both supply cables leave along the corridor in front of the supply,
            # clear of the fault device and the arm mounting column.
            "SupplyPositive": (
                [s["+"], [0.435, -0.14, z], [0.205, -0.14, z], [0.205, 0.07, z], d["IN+"]],
                [0.65, 0.018, 0.012],
            ),
            "SupplyReturn": (
                [s["-"], [0.465, -0.152, z], [0.175, -0.152, z], [0.175, 0.17, z], d["IN-"]],
                [0.018, 0.023, 0.03],
            ),
        }
        # Device leads: the return lead stays outside the positive one, so they never cross.
        routes["LoadPositive"] = (
            [d["OUT+"], *self.load_route["+"], load["+"]],
            [0.65, 0.018, 0.012],
        )
        routes["LoadReturn"] = (
            [d["OUT-"], *self.load_route["-"], load["-"]],
            [0.018, 0.023, 0.03],
        )
        self.wiring = []
        for name, (points, color) in routes.items():
            # Rounded bends remain on the tabletop and never end in empty space.
            smooth = np.asarray(points, dtype=float)
            for _ in range(3):
                pairs = np.stack(
                    [
                        0.75 * smooth[:-1] + 0.25 * smooth[1:],
                        0.25 * smooth[:-1] + 0.75 * smooth[1:],
                    ],
                    axis=1,
                ).reshape(-1, 3)
                smooth = np.vstack([smooth[0], pairs, smooth[-1]])
            self.wire(name, smooth.tolist(), color)
            self.wiring.append(
                {
                    "name": name,
                    "start": list(map(float, self.frame.point(points[0]))),
                    "end": list(map(float, self.frame.point(points[-1]))),
                }
            )

    def flat_label(self, name, text, pos, size):
        """Bench-top label that reads upright from the head and wrist cameras."""
        label = self.a.label("/World/FaultRig/" + name, text, pos, size)
        label.AddRotateZOp().Set(90.0)
        label.AddRotateXOp().Set(90.0)
        return label

    def front_label(self, name, text, pos, size):
        """Label on a face that looks back toward the robot (-X)."""
        label = self.a.label("/World/FaultRig/" + name, text, pos, size)
        label.AddRotateZOp().Set(90.0)
        return label

    def thermal_load(self):
        from pxr import Gf, UsdGeom, UsdShade

        cx, cy = DEVICE_XY
        z = self.z
        self.cube("LoadBase", [cx, cy, z + 0.02], [0.15, 0.15, 0.04], "ceramic", True)
        body_x = cx + 0.03
        self.cube("DCLoadBody", [body_x, cy, z + 0.065], [0.08, 0.12, 0.05], "polymer", True)
        # The front-left corner of the top stays clear for the status lens.
        for i in range(5):
            self.cube(
                "LoadHeatSinkFin" + str(i),
                [body_x, cy - 0.045 + i * 0.018, z + 0.096],
                [0.07, 0.004, 0.012],
                "aluminum",
            )
        self.front_label(
            "LampLabel", "HEAT / INSULATION TEST", [body_x - 0.0405, cy, z + 0.078], [0.10, 0.012]
        )
        self.insulation_mat = self.a.material(
            "SocketInsulation", color=[0.18, 0.18, 0.17], roughness=0.55
        )
        copper = self.a.material("ThermalCopper", color=[0.55, 0.21, 0.07], metallic=0.8)
        self.load_terminals = {}
        self.insulation_scales = []
        # Input posts face the disconnect; insulated leads run back into the load body.
        for i, (polarity, x, end_y) in enumerate(
            [("+", cx - 0.055, -0.04), ("-", cx - 0.025, -0.025)]
        ):
            post = np.array([x, cy + 0.062, z + 0.046])
            self.cylinder(
                "LampTerminal" + str(i), post, 0.007, 0.012, self.red if i == 0 else "rubber"
            )
            self.load_terminals[polarity] = post + [0, 0, 0.006]
            core_end = cy + end_y
            self.cylinder(
                "SocketCore" + str(i),
                [x, (cy + 0.055 + core_end) / 2, z + 0.049],
                0.003,
                cy + 0.055 - core_end,
                copper,
                "Y",
            )
            sleeve = self.cylinder(
                "SocketInsulation" + str(i),
                [x, cy + 0.015, z + 0.049],
                0.009,
                0.06,
                self.insulation_mat,
                "Y",
            )
            self.insulation_scales.append(sleeve.AddScaleOp())
            self.wire(
                "SocketFeed" + str(i),
                [[x, core_end, z + 0.049], [body_x - 0.045, core_end, z + 0.058]],
                [0.55, 0.21, 0.07],
            )
        self.load_route = {
            "+": [[0.42, 0.045, self.z + 0.005], [0.445, 0.03, self.z + 0.044]],
            "-": [
                [0.44, 0.16, self.z + 0.005],
                [0.478, 0.09, self.z + 0.005],
                [0.486, 0.03, self.z + 0.044],
            ],
        }
        bridge = self.cube(
            "FailedInsulationBridge",
            [cx - 0.04, cy + 0.015, z + 0.049],
            [0.03, 0.008, 0.006],
            copper,
        )
        self.short_bridge = UsdGeom.Imageable(bridge.GetPrim())
        self.short_bridge.MakeInvisible()
        # One round, upward-facing panel lens changes colour; no separate alarm lamps.
        # Each colour is its own pre-coloured lens; only the active one has full size.
        # RTX shows a transform change on the same frame, whereas visibility and
        # material edits reach the image several frames late (measured, Isaac Sim 6.0).
        led = cue_position("B2", z)
        self.cylinder(
            "StatusLEDBezel",
            [led[0], led[1], z + 0.092],
            STATUS_LED_RADIUS * 1.35,
            0.004,
            "steel",
        )
        self.status_lenses = {}
        for status, (color, intensity) in STATUS_COLORS.items():
            material = self.a.material(
                "ThermalStatusLED_" + status, color=color, roughness=0.2, emissive=bool(intensity)
            )
            if intensity:
                material.shaders_list[0].GetInput("emissive_intensity").Set(intensity)
            lens = UsdGeom.Sphere.Define(self.stage, "/World/FaultRig/StatusIndicator_" + status)
            lens.CreateRadiusAttr(STATUS_LED_RADIUS)
            lens.AddTranslateOp().Set(Gf.Vec3d(led[0], led[1], z + 0.094))
            scale = lens.AddScaleOp()
            scale.Set(Gf.Vec3f(1, 1, 0.65))
            UsdShade.MaterialBindingAPI.Apply(lens.GetPrim()).Bind(material.material)
            self.status_lenses[status] = scale
        self.flat_label(
            "StatusLabel", "STATUS", [led[0] + 0.027, led[1], z + 0.0905], [0.03, 0.009]
        )

    def motor(self):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        cx, cy = DEVICE_XY
        # The wheel spins about a vertical shaft so every robot camera sees its face.
        self.pivot = np.array([cx, cy, self.z + WHEEL_HEIGHT])
        x, y, z = self.pivot
        self.cube("MotorFoot", [x, y, self.z + 0.012], [0.15, 0.15, 0.024], "ceramic", True)
        self.cylinder("MotorHousing", [x, y, self.z + 0.0495], 0.042, 0.051, "aluminum", "Z", True)
        self.cylinder("MotorBearingCap", [x, y, self.z + 0.076], 0.02, 0.003, "steel")
        self.cylinder("Shaft", [x, y, self.z + 0.08], 0.008, 0.012, "steel")
        self.front_label(
            "MotorLabel", "DC MOTOR / TEST WHEEL", [x - 0.0758, y, self.z + 0.012], [0.14, 0.015]
        )
        self.load_terminals = {}
        for i, polarity in enumerate(["+", "-"]):
            ty = y + (-0.014 if polarity == "+" else 0.014)
            self.cylinder(
                "MotorTerminal" + str(i),
                [x - 0.047, ty, self.z + 0.05],
                0.006,
                0.01,
                self.red if i == 0 else "rubber",
                "X",
            )
            self.load_terminals[polarity] = np.array([x - 0.052, ty, self.z + 0.05])
        # Leads drop off the disconnect, run in front of the foot and climb to the posts.
        tz = self.z + 0.005
        self.load_route = {
            "+": [
                [0.41, 0.03, tz],
                [0.418, y - 0.014, tz],
                [0.43, y - 0.014, tz],
                [0.445, y - 0.014, self.z + 0.03],
            ],
            "-": [
                [0.44, 0.15, tz],
                [0.43, 0.06, tz],
                [0.427, y + 0.014, tz],
                [0.433, y + 0.014, tz],
                [0.445, y + 0.014, self.z + 0.03],
            ],
        }
        root = "/World/MotorWheel"
        prim = UsdGeom.Xform.Define(self.stage, root).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.12)
        self.cylinder("Hub", [0, 0, 0], 0.022, 0.019, "steel", "Z", True, root)
        for i in range(4):
            angle = i * np.pi / 2
            spoke = UsdGeom.Cube.Define(self.stage, root + "/Spoke" + str(i))
            spoke.CreateSizeAttr(1.0)
            spoke.AddTranslateOp().Set(
                Gf.Vec3d(float(0.035 * np.cos(angle)), float(0.035 * np.sin(angle)), 0)
            )
            spoke.AddRotateZOp().Set(float(i * 90))
            spoke.AddScaleOp().Set(Gf.Vec3f(0.07, 0.016, 0.014))
            UsdPhysics.CollisionAPI.Apply(spoke.GetPrim())
            material = self.yellow if i == 0 else self.a.material("aluminum")
            UsdShade.MaterialBindingAPI.Apply(spoke.GetPrim()).Bind(material.material)
        from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

        rim = author_mesh(
            self.stage,
            root + "/Rim",
            revolved(
                [
                    (WHEEL_RADIUS - 0.009, -0.008),
                    (WHEEL_RADIUS, -0.008),
                    (WHEEL_RADIUS, 0.008),
                    (WHEEL_RADIUS - 0.009, 0.008),
                    (WHEEL_RADIUS - 0.009, -0.008),
                ]
            ),
        )
        UsdShade.MaterialBindingAPI.Apply(rim.GetPrim()).Bind(self.a.material("aluminum").material)
        # The wheel body is outside /World/FaultRig: place it with the apparatus frame.
        placed_pivot = self.frame.point(self.pivot)
        self.wheel = self.b.world.scene.add(
            SingleRigidPrim(root, name="motor_wheel", position=placed_pivot, mass=0.12)
        )
        joint = UsdPhysics.RevoluteJoint.Define(self.stage, "/World/FaultRig/MotorShaftJoint")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(root)])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, placed_pivot)))
        self.motor_drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        self.motor_drive.CreateTypeAttr("force")
        self.motor_drive.CreateStiffnessAttr(0)
        self.motor_drive.CreateDampingAttr(0.03)
        self.motor_drive.CreateTargetVelocityAttr(240.0)
        self.motor_drive.CreateMaxForceAttr(0.025)
        # An open-topped guard ring surrounds the rim; no hand enters it.
        guard = author_mesh(
            self.stage,
            "/World/FaultRig/WheelGuard",
            revolved(
                [(0.085, -0.010), (0.091, -0.010), (0.091, 0.010), (0.085, 0.010), (0.085, -0.010)]
            ),
            self.pivot,
        )
        UsdShade.MaterialBindingAPI.Apply(guard.GetPrim()).Bind(self.yellow.material)
        for i in range(4):
            angle = np.pi / 4 + i * np.pi / 2
            self.cube(
                "GuardPost" + str(i),
                [x + 0.088 * np.cos(angle), y + 0.088 * np.sin(angle), self.z + 0.051],
                [0.010, 0.010, 0.054],
                self.yellow,
                True,
            )

    def rack(self, source):
        x, y = source[:2]
        z = self.z
        self.cube("rack_floor", [x, y, z + 0.0175], [0.19, 0.13, 0.035], "polymer", True)
        for side in [-1, 1]:
            self.cube(
                "rack_end" + str(side).replace("-", "m"),
                [x + side * 0.09, y, z + (0.025 if side == 1 else 0.058)],
                [0.012, 0.13, 0.014 if side == 1 else 0.08],
                "polymer",
                True,
            )
            self.cube(
                "rack_rail" + str(side).replace("-", "m"),
                [x, y + side * 0.041, z + 0.075],
                [0.17, 0.012, 0.01],
                "aluminum",
                True,
            )
        self.a.label(
            "/World/FaultRig/RackLabel", "LOOSE TUBE", [x, y + 0.066, z + 0.020], [0.15, 0.026]
        )
        # A second, retained tube makes the rack's purpose visible.
        self.cylinder("ReferenceTube", [x - 0.06, y, z + 0.095], 0.012, 0.12, self.a.glass())
        self.cylinder(
            "ReferenceFill",
            [x - 0.06, y, z + 0.06],
            0.010,
            0.045,
            self.a.material("ReferenceBlue", color=[0.02, 0.3, 0.65]),
        )

    def tube_support(self, source):
        from pxr import Gf, Sdf, UsdPhysics

        # A failing, damped bottom socket tips slowly under a torque-limited drive.
        # It releases on measured grasp; the robot then holds a free rigid body.
        pivot = np.array(source) + [0, 0, -0.056]
        self.cylinder("LooseSocket", pivot + [0, 0, -0.007], 0.021, 0.014, "polymer")
        self.tube_joint = UsdPhysics.RevoluteJoint.Define(
            self.stage, "/World/FaultRig/TubeSocketJoint"
        )
        self.tube_joint.CreateAxisAttr("Y")
        self.tube_joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Objects/sample_0")])
        self.tube_joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
        self.tube_joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, -0.056))
        self.tube_joint.CreateLowerLimitAttr(0.0)
        self.tube_joint.CreateUpperLimitAttr(90.0)
        self.tube_drive = UsdPhysics.DriveAPI.Apply(self.tube_joint.GetPrim(), "angular")
        self.tube_drive.CreateTypeAttr("force")
        self.tube_drive.CreateStiffnessAttr(2.0)
        self.tube_drive.CreateDampingAttr(0.6)
        self.tube_drive.CreateTargetPositionAttr(0.0)
        self.tube_drive.CreateTargetVelocityAttr(0.0)
        self.tube_drive.CreateMaxForceAttr(0.15)
        self.tube_released = False

    def disturb_tube(self):
        self.tube_drive.GetStiffnessAttr().Set(0.0)
        self.tube_drive.GetTargetVelocityAttr().Set(10.0)

    def measure(self, obj, dt):
        if getattr(self.task, "tube_task", False):
            from roboclock.envs.lab.mdp.state import upright_error

            if not self.tube_released and (
                obj.grasped or upright_error(obj.orientation) >= np.deg2rad(78)
            ):
                self.tube_joint.CreateJointEnabledAttr(False)
                self.tube_released = True
            return {"tube_support_released": self.tube_released}
        if not getattr(self.task, "power_task", False):
            return {}
        from roboclock.envs.lab.knife_switch_scene import OPEN_ANGLE

        w, x, y, z = obj.orientation
        angle = float(-np.arctan2(2 * (w * y - x * z), 1 - 2 * (x * x + y * y)))
        opened = angle >= OPEN_ANGLE and obj.robot_contact
        self.press_dwell = self.press_dwell + dt if opened else 0.0
        if self.press_dwell + 1e-9 >= 0.025:
            self.power_on = False
        # Electrical state follows measured separation; a reclosed switch restores power.
        if angle < np.deg2rad(15):
            self.power_on = True
        data = {
            "switch_angle_rad": angle,
            "switch_robot_contact": obj.robot_contact,
            "power_on": self.power_on,
            "wiring": self.wiring,
        }
        if self.task.task_id == "B3":
            data["wheel_speed_rad_s"] = float(self.wheel.get_angular_velocity()[2])
        return data

    def update(self, data):
        from pxr import Gf

        if self.state_lamp is not None:
            self.state_lamp.update()
        if getattr(self.task, "tube_task", False):
            self.tube_liquid.update()
            return
        on = data["power_on"]
        for pilot in (self.green, self.pilot_blue):
            pilot.shaders_list[0].GetInput("emissive_intensity").Set(10.0 if on else 0.0)
        if self.task.task_id == "B2":
            status = data["status_indicator"]
            if status == "flashing_red":
                status = "red" if int(data["time"] * 4) % 2 == 0 else "off"
            for name, scale in self.status_lenses.items():
                scale.Set(Gf.Vec3f(1, 1, 0.65) if name == status else Gf.Vec3f(1e-4, 1e-4, 1e-4))
            heat = max(0, min(1, (data["insulation_temperature_c"] - 35) / 125))
            self.insulation_mat.set_color(
                np.array([0.18 - 0.1 * heat, 0.18 - 0.13 * heat, 0.17 - 0.14 * heat])
            )
            for scale in self.insulation_scales:
                damage = data["insulation_damage"]
                scale.Set(Gf.Vec3f(1 - 0.65 * damage, 1 - 0.6 * damage, 1 - 0.65 * damage))
            if data["short_circuit"]:
                self.short_bridge.MakeVisible()
        else:
            # An internal shaft fault abruptly brakes the rotor without adding geometry.
            # PhysX integrates the braking torque; electrical power stays connected
            # until the robot physically opens the knife switch.
            fault = data["fault"]
            self.motor_drive.GetTargetVelocityAttr().Set(0.0 if fault else 240.0)
            self.motor_drive.GetDampingAttr().Set(1.0 if fault else 0.03)
            self.motor_drive.GetMaxForceAttr().Set(1.0 if fault else 0.025 if on else 0.0)


def test_tube_visual(appearance, path, radius, height):
    """Open glass test tube; the measured-pose liquid is authored separately."""
    from pxr import UsdGeom, UsdShade

    from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

    collider = appearance.cylinder_child(
        path, "Body", radius, height, 0.0, appearance.material("polymer"), True
    )
    UsdGeom.Imageable(collider).MakeInvisible()
    bottom, top = -height / 2, height / 2
    profile = [(0, bottom)]
    for angle in np.linspace(0, np.pi / 2, 12)[1:]:
        profile.append((radius * np.sin(angle), bottom + radius * (1 - np.cos(angle))))
    inner = radius - 0.0015
    profile += [(radius, top), (inner, top), (inner, bottom + radius)]
    for angle in np.linspace(np.pi / 2, 0, 12)[1:]:
        profile.append((inner * np.sin(angle), bottom + radius - inner * np.cos(angle)))
    glass = author_mesh(appearance.stage, path + "/Glass", revolved(profile))
    UsdShade.MaterialBindingAPI.Apply(glass.GetPrim()).Bind(appearance.glass().material)
