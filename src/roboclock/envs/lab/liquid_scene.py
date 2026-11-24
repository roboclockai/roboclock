"""Imported laboratory setting and physical valve/fraction-collector workstations."""

from pathlib import Path

import numpy as np

from roboclock.envs.lab.layout import LabLayout, ObjectConfig
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState, Target

ASSET_ROOT = Path(__file__).resolve().parents[4] / "assets/lab"
PIVOTS = {
    "A1": (0.28589073, 0.04089259, 0.97043899),
    # A2: moved from (0.42, -0.25, 1.04), where the resting right hand hid the lever, the grip
    # and the label from the head camera (the grip imaged at (520, 232) px, inside the hand's
    # footprint). Here it stands 0.24 m ahead of the same arm's base and images 40 px clear of
    # the hand. Farther out, at (0.42, -0.05), the arm's forearm struck the stand's upper clamp on
    # the way down to the grip (12 N on link4), so it is kept 0.08 m nearer.
    "A2": (0.34, -0.02, 1.04),
    "A4": (0.42, -0.25, 1.06),
}
LEVER_RADIUS = 0.06
CLOSE_ANGLE = 1.35
TUBE_SPACING = 0.075
# A3: the column stands 0.12 m to the +y side of its outlet and a delivery tube carries the
# eluent across. Straight above the outlet (as it stood until 2026-11-10) the column lay between
# the head camera (y = 0.20) and the row of tubes (y = 0.21) and hid every fresh tube.
COLUMN_XY = (0.48, 0.33)
OUTLET_XY = (0.48, 0.21)
# A2: inner outline of the separatory funnel (height above its outlet, radius), a pear with a
# long lower cone. The head camera looks down at about 55 degrees: it sees through a cone wall
# that leans 13 degrees (65 degrees from the wall's normal), so the phase boundary shows all the
# way down to the outlet. The earlier squat cone leaned 29 degrees: seen at 84 degrees the glass
# only mirrored the room and the boundary was hidden once it left the widest part.
FUNNEL_HEIGHTS = (0.0, 0.025, 0.215, 0.245, 0.27, 0.285)
FUNNEL_RADII = (0.004, 0.006, 0.050, 0.047, 0.030, 0.015)
# A3: the packed bed is a stack of opaque slices that slides down the column: white silica, then
# each dye band as five slices (start, end as fractions of the band, dye concentration), a
# soft-edged band as chromatography_visuals.band_concentrations draws it.
BAND_SLICES = (
    (0.0, 0.12, 0.34),
    (0.12, 0.30, 0.73),
    (0.30, 0.70, 1.0),
    (0.70, 0.88, 0.73),
    (0.88, 1.0, 0.34),
)
# The bench lighting renders an albedo of 0.5 as full white: the silica is kept below that.
BED_ALBEDO = 0.42


def bed_slice_spans(time, intervals, speed, bottom, top):
    """Height span (low, high) on the bed, in m above the worktop, of every slice at `time`,
    from the lowest up: white, the first band's BAND_SLICES, white, the next band's, ..., white.
    None once a slice has left the bed. A band's leading edge reaches the bed's bottom at its
    interval's start and its trailing edge at the interval's end (the scored times)."""
    spans, previous = [], bottom
    for start, end in intervals:
        fractions = [first for first, _, _ in BAND_SLICES] + [1.0]
        edges = [
            round(
                float(np.clip(bottom + (start + f * (end - start) - time) * speed, bottom, top)), 4
            )
            for f in fractions
        ]
        spans.append((previous, edges[0]))
        spans.extend(zip(edges[:-1], edges[1:]))
        previous = edges[-1]
    spans.append((previous, top))
    return [span if span[1] - span[0] > 1e-4 else None for span in spans]


def nominal_pivot(task_id, z):
    """The valve pivot (A1, A2, A4) or the collector's grip (A3) in the nominal station frame."""
    return np.array(PIVOTS.get(task_id, (0.48, 0.04, z + 0.15)), dtype=float)


def liquid_layout(task, config):
    from roboclock.envs.lab.station_frame import draw_station_frame

    z = config["scene"]["table_height"]
    # The whole station (stand, vessels, valve or collector) moves rigidly with the episode's
    # seeded station frame; everything below is in world coordinates.
    station = draw_station_frame(task)
    pivot = station.point(nominal_pivot(task.task_id, z))
    grip = (
        pivot + np.array([LEVER_RADIUS, 0, 0]) if task.task_id not in {"A1", "A3"} else pivot.copy()
    )
    obj = ObjectConfig(
        "sample_0",
        "container",
        grip,
        np.array([1.0, 0, 0, 0]),
        np.array([0.1, 0.5, 0.8]),
        0.10,
        0.8,
    )
    destination = Target(
        grip.copy(), np.array([0.04, 0.04, 0.04]), kind="process_control", support_z=z
    )
    return LabLayout(
        {"sample_0": obj},
        {"destination": destination},
        {"sample_0": "destination"},
        destination,
        z,
        2200.0,
        np.zeros(3),
        {"sample_0": (0.012, 0.04)},
        (np.array([-0.2, -0.6, z]), np.array([1.1, 0.65, z + 0.9])),
        apparatus=station,
    )


def no_shadow(prim):
    """Stop a glass or liquid prim casting shadows. RTX Real-Time shadows refractive glass as if
    it were opaque, so whatever stands inside a vessel is unlit: the phases in the separatory
    funnel and the flask on its white plate rendered black in the policy cameras."""
    from pxr import Sdf, UsdGeom

    UsdGeom.PrimvarsAPI(prim).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)


def lay_flat(label, tilt=0.0):
    """Turn an `appearance.label` quad (made facing +y) face up, its text upright in the head
    camera (letters' tops toward +x, reading toward -y); `tilt` degrees raise the far edge."""
    label.AddRotateZOp().Set(90.0)
    label.AddRotateXOp().Set(90.0 - float(tilt))
    return label


class LiquidScene:
    def __init__(self, backend, *, setting_only=False, background_only=False):
        self.b = backend
        self.a = backend.appearance
        self.stage = backend.stage
        self.task = backend.task
        self.z = self.task.layout.table_height
        self.transforms = {}
        self.liquids = {}
        self.liquid_bases = {}
        if background_only:
            self.build_background()
            return
        self.build_setting()
        if setting_only:
            return
        from roboclock.envs.lab.station_frame import root, station_of

        # The station is authored in nominal coordinates under roots placed in the episode's
        # seeded station frame (A1's wet bench and its colliders included); rigid bodies and
        # joints take world poses from self.station.point.
        self.station = station_of(self.task)
        roots = ["/World/Apparatus", "/World/Process"]
        if self.task.task_id == "A1":
            roots.append("/World/WetCollision")
        for path in roots:
            root(self.station, self.stage, path)
        if self.task.task_id == "A1":
            self.water_station()
        elif self.task.task_id == "A2":
            self.separation_station()
        elif self.task.task_id == "A3":
            self.column_station()
        else:
            self.titration_station()
        # A visible stir bar rotates independently of the rigid flask.
        if self.task.task_id == "A4":
            from pxr import UsdGeom

            self.stir_rotation = UsdGeom.Xformable(
                self.stage.GetPrimAtPath("/World/Apparatus/StirBar")
            ).AddRotateZOp()
        self.update(self.task.visual_state())

    def cylinder(self, path, pos, radius, height, material, collision=False):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        shape = UsdGeom.Cylinder.Define(self.stage, path)
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(height))
        shape.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
            (self.a.material(material) if isinstance(material, str) else material).material
        )
        if collision:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            self.b._collision_settings(shape.GetPrim())
        return shape

    def lathe(self, path, pos, profile, material, shadow=None):
        """A revolved mesh. Glass casts no shadow (see `no_shadow`); `shadow=False` says the same
        for an opaque liquid that stands above something the cameras must see lit."""
        from pxr import UsdShade

        from roboclock.simulators.isaac.visual_geometry import (
            author_mesh,
            revolved,
            smooth_revolved_normals,
        )

        mesh = author_mesh(self.stage, path, revolved(profile), pos)
        if self.task.task_id in {"A3", "A4"}:
            smooth_revolved_normals(mesh, profile)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material.material)
        if shadow is None:
            shadow = not material.shaders_list[0].GetInput("glass_color")
        if not shadow:
            no_shadow(mesh.GetPrim())
        return mesh

    def vessel(self, path, pos, r=0.044, h=0.12, kind="beaker"):
        if kind == "flask":
            profile = [
                (0.003, 0),
                (r * 0.82, 0),
                (r, 0.008),
                (r, 0.025),
                (r * 0.32, h * 0.72),
                (r * 0.30, h),
                (r * 0.24, h),
                (r * 0.26, h * 0.72),
                (r - 0.0015, 0.025),
                (r - 0.0015, 0.01),
                (0.003, 0.003),
            ]
        elif kind == "tube":
            profile = [
                (0.0, 0.0),
                (r * 0.7, 0.002),
                (r, 0.009),
                (r, h),
                (r - 0.001, h),
                (r - 0.001, 0.01),
                (0.001, 0.003),
            ]
        elif kind == "funnel":
            # Outer wall up, 2 mm outside FUNNEL_RADII, then the neck and the inner wall down.
            inner = list(zip(FUNNEL_RADII, FUNNEL_HEIGHTS))
            profile = (
                [(radius + 0.002, height) for radius, height in inner[:-1]]
                + [(0.018, 0.285), (0.018, 0.31), (0.015, 0.31)]
                + inner[::-1]
            )
        else:
            profile = [
                (0.001, 0),
                (r * 0.95, 0),
                (r, 0.004),
                (r, h),
                (r - 0.0015, h),
                (r - 0.0015, 0.004),
                (0.001, 0.004),
            ]
        glass = self.a.glass()
        if kind == "funnel":
            # The phases are read through this glass at 65 degrees, where it mirrors a tenth of
            # a room far brighter than white: a white veil over the lower phase. Dim the mirror.
            from pxr import Gf, Sdf

            glass = self.a.glass("FunnelGlass", color=[0.99, 0.995, 1.0])
            shader = glass.shaders_list[0]
            shader.GetInput("depth").Set(0.0015)
            shader.CreateInput("reflection_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2))
        self.lathe(path, pos, profile, glass)
        if kind == "collection":
            for amount in range(20, 501, 20):
                level = pos[2] + 0.004 + amount / 1e6 / (np.pi * 0.0415**2)
                major = amount % 100 == 0
                self.a.rounded(
                    path + f"_Tick{amount}",
                    [pos[0] - 0.012, pos[1] + r + 0.0003, level],
                    [0.020 if major else 0.008, 0.0003, 0.0005 if major else 0.0003],
                    "rubber",
                    0.0001,
                )
                if major:
                    self.a.label(
                        path + f"_Volume{amount}",
                        str(amount),
                        [pos[0] + 0.013, pos[1] + r + 0.0005, level + 0.001],
                        [0.020, 0.008],
                        color=(0.024, 0.029, 0.035),
                        printed=True,
                    )
            self.a.label(
                path + "_Units",
                "mL",
                [pos[0] - 0.018, pos[1] + r + 0.0005, pos[2] + 0.084],
                [0.015, 0.007],
                color=(0.024, 0.029, 0.035),
                printed=True,
            )
        elif kind == "beaker":
            # Graduations occupy only a narrow side strip, leaving the liquid visible.
            for j in range(1, 6):
                self.a.rounded(
                    path + f"_Tick{j}",
                    [pos[0] + r * 0.55, pos[1] + r * 0.85, pos[2] + h * j / 6],
                    [0.017, 0.0007, 0.0007],
                    "paper",
                    0.0002,
                )
            self.a.label(
                path + "_Capacity",
                "250 mL" if self.task.task_id == "A3" else "500 mL",
                [pos[0], pos[1] + r + 0.001, pos[2] + h * 0.80],
                [r * 1.1, 0.013],
            )

    def fluid(self, name, pos, r, h, color):
        from pxr import Gf, UsdGeom

        if self.task.task_id == "A1" and name in {"Water", "Stream"}:
            material = self.a.glass("Water", color=[0.97, 0.99, 1.0], ior=1.333, roughness=0.015)
        elif self.task.task_id in {"A3", "A4"}:
            material = self.a.glass("Eluent_" + name, color=color, ior=1.333, roughness=0.018)
            material.shaders_list[0].GetInput("depth").Set(0.015)
        else:
            # A2's opaque phases are matt: with a mirror finish (roughness 0.16, full specular)
            # the side of each phase, seen through the cone, showed the bright room instead of
            # the phase's colour.
            from pxr import Sdf

            material = self.a.material("Fluid_" + name, color=color, roughness=0.5, metallic=0.0)
            material.shaders_list[0].CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(
                0.15
            )
        if self.task.task_id == "A3" and name.startswith("TubeFill"):
            # Wetting raises the rim slightly above the center of the surface.
            shape = self.lathe(
                "/World/Process/" + name,
                pos,
                [(0, -0.5), (r, -0.5), (r, 0.5), (r * 0.88, 0.49), (r * 0.65, 0.486), (0, 0.486)],
                material,
            )
        else:
            shape = self.cylinder("/World/Process/" + name, pos, r, 1.0, material)
        no_shadow(shape.GetPrim())
        scale = shape.AddScaleOp()
        scale.Set(Gf.Vec3f(1, 1, max(h, 0.0002)))
        trans = UsdGeom.Xformable(shape).GetOrderedXformOps()[0]
        self.liquids[name] = (trans, scale, material)
        self.liquid_bases[name] = np.array(pos) - [0, 0, h / 2]
        return shape

    def level(self, name, height, color=None):
        from pxr import Gf

        trans, scale, material = self.liquids[name]
        base = self.liquid_bases[name]
        trans.Set(Gf.Vec3d(*map(float, base + [0, 0, max(height, 0.0002) / 2])))
        scale.Set(Gf.Vec3f(1, 1, max(height, 0.0002)))
        if color is not None:
            self.fluid_color(material, color)

    def fluid_color(self, material, color):
        from pxr import Gf

        tint = material.shaders_list[0].GetInput("glass_color")
        if tint:
            tint.Set(Gf.Vec3f(*map(float, color)))
        else:
            material.set_color(np.asarray(color))

    def build_setting(self):
        from pxr import UsdGeom

        # The background has exactly two worktables: one imported lab bench
        # and one office desk. Keep the aisle open, matching the supplied reference.
        # Local bench with cabinet seams, drawers, toe kick and a resilient worktop.
        if self.task.task_id != "A1":
            self.a.cube("/World/Bench/Carcass", [0.45, 0, 0.37], [1.28, 1.25, 0.70], "polymer")
            self.a.cube("/World/Bench/ToeKick", [0.45, 0.02, 0.055], [1.20, 1.16, 0.10], "rubber")
            for i, x in enumerate([0.04, 0.45, 0.86]):
                for j in range(3):
                    self.a.cube(
                        f"/World/Bench/Drawer{i}_{j}",
                        [x, 0.632, 0.22 + j * 0.18],
                        [0.39, 0.018, 0.166],
                        "paper",
                    )
                    self.a.cube(
                        f"/World/Bench/Handle{i}_{j}",
                        [x, 0.65, 0.27 + j * 0.18],
                        [0.14, 0.025, 0.009],
                        "steel",
                    )
        # Collidable bottle/box clutter occupies the rear and side strips only.
        self.b.clutter = {}
        self.b.clutter_specs = []
        self.b.clutter_robot_contact_count = 0
        positions = (
            [(-0.95, -0.18), (-1.10, 0.02), (-1.25, 0.18)]
            if self.task.task_id == "A1"
            else [(0.90, -0.30), (0.94, -0.12), (0.84, -0.44)]
        )
        for i, (x, y) in enumerate(positions):
            root = f"/World/Accessories/Reagent{i}"
            self.vessel(root, [x, y, self.z], 0.025, 0.105)
            self.cylinder(root + "_Cap", [x, y, self.z + 0.11], 0.024, 0.016, "polymer")
            self.a.label(
                root + "_Label",
                ["BUFFER", "WASH", "SOLVENT", "ELUENT", "RINSE"][i],
                [x, y + 0.026, self.z + 0.054],
                [0.039, 0.025],
            )
            self.cylinder(root + "_Collider", [x, y, self.z + 0.06], 0.025, 0.12, "polymer", True)
            # Hide only the conservative collision envelope.
            UsdGeom.Imageable(self.stage.GetPrimAtPath(root + "_Collider")).MakeInvisible()
        self.a.cube(
            "/World/Accessories/TipBox",
            [
                -0.85 if self.task.task_id == "A1" else 0.15,
                0.20 if self.task.task_id == "A1" else -0.39,
                self.z + 0.025,
            ],
            [0.15, 0.10, 0.05],
            "paper",
            True,
        )
        self.a.cube(
            "/World/Accessories/TipLid",
            [
                -0.85 if self.task.task_id == "A1" else 0.15,
                0.20 if self.task.task_id == "A1" else -0.39,
                self.z + 0.053,
            ],
            [0.155, 0.105, 0.008],
            "polymer",
        )
        self.build_background()

    def build_background(self):
        """Add the shared visual room without changing task fixtures or clutter."""
        from pxr import Gf, UsdGeom

        # Imported analytical bench adds test-tube racks and instrument enclosures.
        source = ASSET_ROOT / self.b.config["scene"].get(
            "liquid_background_bench", "closed_laboratory_bench.usdc"
        )
        root = UsdGeom.Xform.Define(self.stage, "/World/Imported/AnalyticalBench")
        root.GetPrim().GetReferences().AddReference(str(source))
        root.AddTranslateOp().Set(
            Gf.Vec3d(
                *self.b.config["scene"].get("liquid_background_bench_position", [1.25, -2.0, 0.0])
            )
        )
        root.AddRotateZOp().Set(0.0)
        # Bind the original cabinet geometry to subtle textured physical surfaces.
        from pxr import Usd, UsdShade

        specifications = self.b.config["visuals"].setdefault("materials", {})
        specifications["LabCabinetEnamel"] = {
            **specifications["polymer"],
            "normal_strength": 0.04,
            "roughness_map_weight": 0.15,
            "tint": [2.7, 2.4, 2.3],
        }
        specifications["LabGraphiteWorktop"] = {
            **specifications["worktop"],
            "normal_strength": 0.10,
            "roughness_map_weight": 0.35,
            "tint": [0.65, 0.65, 0.65],
        }
        surfaces = {
            "graphite_worktop": self.a.material(
                "LabGraphiteWorktop", color=[0.07, 0.08, 0.085], roughness=0.28
            ),
            "cabinet_enamel": self.a.material(
                "LabCabinetEnamel", color=[0.80, 0.82, 0.82], roughness=0.29
            ),
            "brushed_steel_handle": self.a.material("steel"),
            "recessed_plinth": self.a.material("rubber"),
            "borosilicate_glass": self.a.glass(),
        }
        for prim in Usd.PrimRange(root.GetPrim()):
            role = prim.GetCustomDataByKey("surface_role")
            if role in surfaces:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(surfaces[role].material)
        if source.name == "closed_laboratory_bench.usdc":
            # The source doors stop below the slab; close the exposed assembly gap
            # with a narrow enamel rail, retaining the original panel geometry.
            for side in [-1, 1]:
                self.a.rounded(
                    str(root.GetPath()) + f"/AssemblyRail{side + 1}",
                    [0.0, side * 0.703, 0.773],
                    [2.998, 0.012, 0.024],
                    surfaces["cabinet_enamel"],
                    0.001,
                )
        from roboclock.envs.lab.room_details import add_room_details

        add_room_details(self, ASSET_ROOT)

    def stand(self, x, y, height=0.62):
        self.a.cube(
            "/World/Apparatus/StandBase", [x, y, self.z + 0.012], [0.22, 0.17, 0.024], "steel", True
        )
        self.cylinder(
            "/World/Apparatus/StandRod", [x, y, self.z + height / 2], 0.008, height, "steel", True
        )
        for i, z in enumerate([self.z + 0.31, self.z + 0.49]):
            self.a.cube(
                f"/World/Apparatus/Clamp{i}", [x - 0.06, y, z], [0.13, 0.016, 0.017], "steel", True
            )
            self.cylinder(
                f"/World/Apparatus/ClampScrew{i}", [x, y + 0.021, z], 0.014, 0.016, "rubber"
            )

    def valve(self, pivot, pipe_end=0.10):
        """The quarter-turn lever valve on its pedestal. Its pipe runs toward +y to `pipe_end`.
        `pivot` is nominal: the fixture is authored under the station roots, the lever body and
        its joint at the world pivot."""
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        # A pedestal and pipe connect the manipulable quarter-turn handle to
        # the fixture while preserving finger clearance around the grip.
        support_top = pivot[2] - 0.04
        self.cylinder(
            "/World/Apparatus/ValvePedestal",
            [pivot[0], pivot[1], (self.z + support_top) / 2],
            0.009,
            support_top - self.z,
            "steel",
            True,
        )
        self.a.cube(
            "/World/Apparatus/ValveHousing",
            [pivot[0], pivot[1], pivot[2] - 0.030],
            [0.045, 0.045, 0.025],
            "steel",
        )
        self.a.cube(
            "/World/Apparatus/ValvePipe",
            [pivot[0], (pivot[1] + pipe_end) / 2, pivot[2] - 0.045],
            [0.015, pipe_end - pivot[1], 0.015],
            "steel",
        )
        root = "/World/Objects/sample_0"
        prim = UsdGeom.Xform.Define(self.stage, root).GetPrim()
        self.cylinder(root + "/Hub", [0, 0, 0], 0.019, 0.023, "steel", False)
        self.a.cube(
            root + "/Lever", [LEVER_RADIUS / 2, 0, 0], [LEVER_RADIUS, 0.014, 0.014], "steel"
        )
        col = UsdGeom.Cube.Define(self.stage, root + "/LeverCollider")
        col.CreateSizeAttr(1.0)
        col.AddTranslateOp().Set(Gf.Vec3d(LEVER_RADIUS / 2, 0, 0))
        col.AddScaleOp().Set(Gf.Vec3f(LEVER_RADIUS, 0.014, 0.014))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdGeom.Imageable(col).MakeInvisible()
        self.cylinder(root + "/Grip", [LEVER_RADIUS, 0, 0], 0.011, 0.04, "rubber", True)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.06)
        mat = UsdPhysics.MaterialAPI.Apply(
            self.stage.DefinePrim("/World/Materials/GripFriction", "Material")
        )
        mat.CreateStaticFrictionAttr(1.4)
        mat.CreateDynamicFrictionAttr(1.2)
        for child in UsdGeom.Xformable(prim).GetPrim().GetChildren():
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(child).Bind(
                    UsdShade.Material(mat.GetPrim()), materialPurpose="physics"
                )
        world = self.station.point(pivot)
        body = self.b.world.scene.add(
            SingleRigidPrim(prim_path=root, name="sample_0", position=world)
        )
        self.b.objects = {"sample_0": body}
        self.pivot = world
        joint = UsdPhysics.RevoluteJoint.Define(self.stage, "/World/ValveJoint")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(root)])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, world)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLowerLimitAttr(-2.0)
        joint.CreateUpperLimitAttr(float(np.degrees(CLOSE_ANGLE) + 2))
        # A passive viscous valve bearing resists incidental motion; no pose motor.
        bearing = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        bearing.CreateTypeAttr("force")
        bearing.CreateStiffnessAttr(0.0)
        bearing.CreateDampingAttr(0.06)
        bearing.CreateTargetVelocityAttr(0.0)
        bearing.CreateMaxForceAttr(0.08)
        self.b.control_joint = joint
        # CLOSE marks the end of the lever's travel: the lever starts along +x and swings toward
        # +y, over this plate, which lies flat on the pipe and reads upright in the head camera
        # (letters about 8 px tall). The label used to stand on edge to that camera, 2 px tall.
        plate = [pivot[0], pivot[1] + 0.0925, pivot[2] - 0.036]
        self.a.cube("/World/Apparatus/ValveLabelPlate", plate, [0.044, 0.134, 0.003], "steel")
        lay_flat(
            self.a.label(
                "/World/Apparatus/ValveLabel",
                "CLOSE",
                [plate[0], plate[1], plate[2] + 0.0017],
                [0.130, 0.040],
                sign=True,
            )
        )

    def water_station(self):
        from roboclock.envs.lab.wet_bench import build_station

        build_station(self, ASSET_ROOT, PIVOTS["A1"])

    def separation_station(self):
        self.stand(0.68, -0.08)
        self.vessel("/World/Apparatus/Funnel", [0.50, 0.13, self.z + 0.30], 0.052, 0.31, "funnel")
        self.cylinder("/World/Apparatus/Stem", [0.50, 0.13, self.z + 0.27], 0.006, 0.10, "steel")
        self.vessel("/World/Apparatus/Receiver", [0.50, 0.13, self.z + 0.008], 0.05, 0.14)
        # Albedos chosen for the lit funnel (the glass no longer shadows its contents): the
        # earlier (0.025, 0.48, 0.78) and (0.95, 0.62, 0.10) wash out to pale cyan and cream.
        blue, amber = [0.012, 0.20, 0.62], [0.40, 0.17, 0.015]
        self.fluid("Lower", [0.50, 0.13, self.z + 0.43], 0.048, 0.12, blue)
        self.fluid("Upper", [0.50, 0.13, self.z + 0.535], 0.048, 0.09, amber)
        self.fluid("Received", [0.50, 0.13, self.z + 0.013], 0.046, 0.001, blue)
        # Amber that ran into the beaker after the lower phase was gone (the failure), floating
        # on the blue. While empty it is a 0.2 mm film at the beaker floor, under the opaque blue
        # (no visibility switch, which RTX shows several frames late).
        self.fluid("ReceivedUpper", [0.50, 0.13, self.z + 0.013], 0.046, 0.001, amber)
        self.fluid("Stream", [0.50, 0.13, self.z + 0.205], 0.002, 0.12, [0.04, 0.5, 0.8])
        self.stream_colors = {"lower": [0.04, 0.5, 0.8], "upper": amber}
        self.stream_phase = "lower"
        self.valve(PIVOTS["A2"], pipe_end=0.14)
        self.a.label(
            "/World/Apparatus/PhaseLabel",
            "LOWER PHASE",
            [0.5, 0.172, self.z + 0.46],
            [0.105, 0.022],
        )

    def column_station(self):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        self.stand(0.70, -0.10, 0.69)
        from roboclock.envs.lab.chromatography_visuals import (
            BED_BOTTOM,
            BED_RADIUS,
            BED_TOP,
            CLEAR_SOLVENT,
        )

        # A continuous hollow glass wall, a tapered outlet and a solvent head.
        column_glass = self.a.glass("ColumnGlass", color=[0.99, 0.995, 1.0], roughness=0.025)
        cx, cy = COLUMN_XY
        ox, oy = OUTLET_XY
        self.lathe(
            "/World/Apparatus/Column",
            [cx, cy, self.z],
            [
                (0.003, 0.20),
                (0.0045, 0.20),
                (0.0045, 0.248),
                (0.025, 0.268),
                (0.025, 0.708),
                (0.027, 0.710),
                (0.027, 0.714),
                (0.0235, 0.714),
                (0.0235, 0.271),
                (0.003, 0.25),
            ],
            column_glass,
        )
        self.fluid("SolventHead", [cx, cy, self.z + 0.684], 0.0230, 0.036, CLEAR_SOLVENT)
        # Delivery tube from the column tip to the drip point above the collector's row.
        self.a.cube(
            "/World/Apparatus/DeliveryTube",
            [ox, (cy + oy) / 2, self.z + 0.212],
            [0.008, cy - oy + 0.008, 0.008],
            "steel",
        )
        self.cylinder(
            "/World/Apparatus/DeliveryTip", [ox, oy, self.z + 0.207], 0.004, 0.014, "steel"
        )
        self.cylinder(
            "/World/Apparatus/Frit",
            [cx, cy, self.z + 0.272],
            BED_RADIUS,
            0.004,
            self.a.material("ColumnFrit", color=[0.91, 0.91, 0.86], roughness=0.65),
        )
        # The packed bed is opaque, as wet silica is: a stack of white and dyed slices that slides
        # down the column as one (move_bed), over a fixed white core. It used to be one refractive
        # body tinted by a texture whose offset moved the bands. RTX takes many rendered frames
        # to show a rewritten material input, so a policy that renders one frame per query saw
        # the bands where they had been queries earlier (a band 30 s gone still tinted the glass),
        # and against the dark worktop the tint hardly showed. A moving surface is redrawn where
        # it now is; with the white between the bands moving too, no pixel of the bed keeps a
        # band that has passed.
        bed = self.a.material("ColumnBed", color=[BED_ALBEDO] * 3, roughness=0.85, metallic=0.0)
        core = BED_RADIUS - 0.0005
        self.lathe(
            "/World/Process/Packing",
            [cx, cy, self.z],
            [(0, BED_BOTTOM), (core, BED_BOTTOM), (core, BED_TOP), (0, BED_TOP)],
            bed,
            shadow=False,
        )
        materials = [bed]
        for i, color in enumerate(self.task.colors):
            dye = 0.55 * (np.asarray(color) * 0.90 + 0.025)
            for _, _, strength in BAND_SLICES:
                materials.append(
                    self.a.material(
                        f"ColumnBand{i}_{round(strength * 100)}",
                        color=BED_ALBEDO * (1 - strength) + dye * strength,
                        roughness=0.85,
                        metallic=0.0,
                    )
                )
            materials.append(bed)
        self.bed_slices = []
        for k, material in enumerate(materials):
            # Lathes as finely faceted as the core, so no facet dips under it.
            shape = self.lathe(
                f"/World/Process/BedSlice{k}",
                [cx, cy, self.z + BED_TOP],
                [(0, -0.5), (BED_RADIUS, -0.5), (BED_RADIUS, 0.5), (0, 0.5)],
                material,
                shadow=False,
            )
            scale = shape.AddScaleOp()
            self.bed_slices.append((UsdGeom.Xformable(shape).GetOrderedXformOps()[0], scale))
        self.bed_spans = {}
        self.move_bed(0.0)
        # Clamp arms from the stand rod (0.70, -0.10) pass behind the collector and reach the
        # glass from its far side, so neither the bands nor the tubes are masked.
        for i, height in enumerate([0.32, 0.62]):
            self.a.cube(
                f"/World/Apparatus/ColumnSupport{i}",
                [0.70, (cy - 0.10) / 2, self.z + height],
                [0.012, cy + 0.10 + 0.012, 0.012],
                "steel",
            )
            self.a.cube(
                f"/World/Apparatus/ColumnArm{i}",
                [(cx + 0.025 + 0.70) / 2, cy, self.z + height],
                [0.70 - cx - 0.025, 0.012, 0.012],
                "steel",
            )
        # Four fraction tubes and a waste beaker on one constrained collector.
        root = "/World/Objects/sample_0"
        prim = UsdGeom.Xform.Define(self.stage, root).GetPrim()
        self.a.cube(root + "/Base", [0.15, 0.10, -0.10], [0.44, 0.22, 0.02], "polymer")
        # A dark rack bar on two posts holds the fresh tubes: clear glass tubes alone are a few
        # faint pixels against the pale collector plate.
        rack = self.a.material("CollectorRack", color=[0.02, 0.10, 0.22], roughness=0.4)
        count = len(self.task.colors)
        self.a.cube(
            root + "/Rack",
            [(count + 1) * TUBE_SPACING / 2, 0.17, -0.039],
            [count * TUBE_SPACING + 0.02, 0.034, 0.012],
            rack,
        )
        for k, x in enumerate([0.5 * TUBE_SPACING, (count + 0.5) * TUBE_SPACING]):
            self.a.cube(root + f"/RackPost{k}", [x, 0.17, -0.0675], [0.010, 0.030, 0.045], rack)
        col = UsdGeom.Cube.Define(self.stage, root + "/BaseCollider")
        col.CreateSizeAttr(1.0)
        col.AddTranslateOp().Set(Gf.Vec3d(0.15, 0.10, -0.10))
        col.AddScaleOp().Set(Gf.Vec3f(0.44, 0.22, 0.02))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdGeom.Imageable(col).MakeInvisible()
        for i in range(len(self.task.colors) + 1):
            r = 0.009 if i else 0.029
            self.vessel(
                root + f"/Tube{i}",
                [i * TUBE_SPACING, 0.17, -0.09],
                r,
                0.12,
                "tube" if i else "beaker",
            )
            self.fluid(
                f"TubeFill{i}",
                [0.48 + i * TUBE_SPACING, 0.21, self.z + 0.069],
                r - 0.002,
                0.001,
                CLEAR_SOLVENT,
            )
            # Signs on the plate beside each vessel, tilted toward the head camera (the old
            # labels stood on the tubes' +y side, edge-on to it and 20 mm tall).
            width, depth = (0.12, 0.040) if i == 0 else (0.07, 0.046)
            lay_flat(
                self.a.label(
                    root + f"/ID{i}",
                    ("WASTE" if i == 0 else f"F{i}"),
                    [i * TUBE_SPACING, 0.078 if i == 0 else 0.105, -0.088 + depth / 4],
                    [width, depth],
                    sign=True,
                ),
                tilt=30.0,
            )
        self.cylinder(root + "/GripStem", [0, 0, -0.056], 0.005, 0.068, "steel")
        self.cylinder(root + "/Grip", [0, 0, 0], 0.012, 0.045, "rubber", True)
        # A yellow collar under the black grip marks the collector's one handle.
        self.cylinder(
            root + "/GripCollar",
            [0, 0, -0.026],
            0.019,
            0.004,
            self.a.material("CollectorGripCollar", color=[0.85, 0.55, 0.02], roughness=0.5),
        )
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.18)
        position = self.station.point(nominal_pivot("A3", self.z))
        body = self.b.world.scene.add(
            SingleRigidPrim(prim_path=root, name="sample_0", position=position)
        )
        self.b.objects = {"sample_0": body}
        self.pivot = position
        joint = UsdPhysics.PrismaticJoint.Define(self.stage, "/World/CollectorJoint")
        joint.CreateAxisAttr("X")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(root)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, position)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLowerLimitAttr(-len(self.task.colors) * TUBE_SPACING - 0.01)
        joint.CreateUpperLimitAttr(0.008)
        self.b.control_joint = joint
        self.a.cube(
            "/World/Apparatus/Rail",
            [0.48, 0.14, self.z + 0.0225],
            [0.68, 0.045, 0.035],
            "steel",
            True,
        )
        self.fluid("Stream", [ox, oy, self.z + 0.19], 0.0009, 0.02, CLEAR_SOLVENT)

    def move_bed(self, time):
        """Place every slice of the bed where it is at `time` (write on change)."""
        from pxr import Gf

        from roboclock.envs.lab.chromatography_visuals import BAND_SPEED, BED_BOTTOM, BED_TOP

        spans = bed_slice_spans(time, self.task.intervals, BAND_SPEED, BED_BOTTOM, BED_TOP)
        for k, span in enumerate(spans):
            if self.bed_spans.get(k, "unset") == span:
                continue
            translate, scale = self.bed_slices[k]
            if span is None:  # off the bed: parked inside the bench, under the worktop
                translate.Set(Gf.Vec3d(*COLUMN_XY, self.z - 0.30))
                scale.Set(Gf.Vec3f(1, 1, 1e-3))
            else:
                translate.Set(Gf.Vec3d(*COLUMN_XY, self.z + (span[0] + span[1]) / 2))
                scale.Set(Gf.Vec3f(1, 1, span[1] - span[0]))
            self.bed_spans[k] = span

    def titration_station(self):
        from roboclock.envs.lab.titration_visuals import build_titration

        build_titration(self, PIVOTS["A4"])

    def update(self, data):
        from pxr import Gf, UsdGeom

        task = self.task
        if task.task_id == "A1":
            height = min(data["volume_ml"], 500) / 1e6 / (np.pi * 0.0415**2)
            self.level("Water", height)
            water = self.water_geometry
            surface = water["beaker_base"][2] + 0.004 + height
            trans, scale, _ = self.liquids["Stream"]
            trans.Set(Gf.Vec3d(*water["nozzle_xy"], (water["nozzle_z"] + surface) / 2))
            radius = np.sqrt(data["flow_ml_s"] / 1e6 / (np.pi * water["jet_speed"]))
            width = max(radius, 0.0003) / water["stream_radius"]
            scale.Set(Gf.Vec3f(width, width, max(0.0002, water["nozzle_z"] - surface)))
        elif task.task_id == "A2":
            remaining = data["lower_ml"]
            # Integrate the vessel cross section, then invert volume to height.
            heights = np.linspace(0, FUNNEL_HEIGHTS[-1], 160)
            # The liquid stands half a millimetre inside the glass, except at the two ends.
            inset = np.array([0, 0, 0.0005, 0.0005, 0.0005, 0])
            radius = np.interp(heights, FUNNEL_HEIGHTS, np.array(FUNNEL_RADII) - inset)
            cumulative = np.cumsum(np.pi * radius**2) * (heights[1] - heights[0])
            cumulative = cumulative / cumulative[-1] * 220.0
            boundary = float(np.interp(remaining, cumulative, heights))
            surface = float(np.interp(remaining + data["upper_ml"], cumulative, heights))
            for name, bottom, top in [("Lower", 0.0, boundary), ("Upper", boundary, surface)]:
                samples = np.linspace(bottom, max(top, bottom + 0.0001), 24)
                radii = np.interp(samples, heights, radius)
                profile = [(0.0, bottom)] + list(zip(radii, samples)) + [(0.0, top)]
                self.stage.RemovePrim("/World/Process/" + name)
                self.lathe(
                    "/World/Process/" + name,
                    [0.50, 0.13, self.z + 0.30],
                    profile,
                    self.liquids[name][2],
                    shadow=False,  # or the receiving beaker below stands in the phases' shadow
                )
            lower_in = data.get("received_lower_ml", data["volume_ml"])
            upper_in = data.get("received_upper_ml", 0.0)
            self.level("Received", lower_in / 220 * 0.105)
            base = self.liquid_bases["Received"]
            # Empty: 0.1 mm lower than the blue film, so the blue top face covers it.
            lift = lower_in / 220 * 0.105 if upper_in > 0 else -0.0001
            self.liquid_bases["ReceivedUpper"] = base + [0, 0, lift]
            self.level("ReceivedUpper", upper_in / 220 * 0.105)
            # The stream carries whichever phase is at the stopcock (write on change).
            phase = "upper" if remaining <= 1e-7 else "lower"
            if phase != self.stream_phase:
                self.fluid_color(self.liquids["Stream"][2], self.stream_colors[phase])
                self.stream_phase = phase
        elif task.task_id == "A3":
            t = data["time"]
            position = (
                self.b.objects["sample_0"].get_world_pose()[0] if self.b.objects else self.pivot
            )
            from roboclock.envs.lab.chromatography_visuals import (
                CLEAR_SOLVENT,
                solution_tint,
            )

            self.move_bed(t)
            for i in range(len(self.task.colors) + 1):
                name = f"TubeFill{i}"
                # The fills live under /World/Process (station frame): nominal coordinates.
                self.liquid_bases[name] = self.station.inverse_point(position) + [
                    i * TUBE_SPACING,
                    0.17,
                    -0.0865,
                ]
                volume = data["tube_volumes"][i]
                color = solution_tint(data["collection"][i], volume, task.colors)
                height = min(0.105, volume / 1e6 / (np.pi * (0.007 if i else 0.027) ** 2))
                # An empty tube's fill is a 0.2 mm film (level): no visibility switch, which RTX
                # renders several frames late.
                self.level(name, height, color)
            active = next((i for i, (s, e) in enumerate(task.intervals) if s <= t < e), None)
            color = (
                0.22 + 0.78 * np.array(task.colors[active]) if active is not None else CLEAR_SOLVENT
            )
            self.fluid_color(self.liquids["Stream"][2], color)
            # End the jet at the actual liquid surface; the old jet intersected
            # the metal tip and extended through the collection liquid.
            receiver = data["receiver"]
            surface = position[2] - 0.0865
            if receiver >= 0:
                surface += min(
                    0.105,
                    data["tube_volumes"][receiver]
                    / 1e6
                    / (np.pi * (0.007 if receiver else 0.027) ** 2),
                )
            nozzle = self.z + 0.20
            trans, scale, _ = self.liquids["Stream"]
            trans.Set(Gf.Vec3d(0.48, 0.21, float((nozzle + surface) / 2)))
            scale.Set(Gf.Vec3f(1, 1, float(max(0.0002, nozzle - surface))))
        else:
            from roboclock.envs.lab.titration_visuals import update_titration

            update_titration(self, data)
        stream = UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/Process/Stream"))
        if data["flow_ml_s"] < 0.02:
            stream.MakeInvisible()
        else:
            stream.MakeVisible()


def reset_liquid(b, task):
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.stage import create_new_stage, get_current_stage
    from pxr import Gf, UsdGeom, UsdPhysics

    from roboclock.envs.lab.appearance import LabAppearance
    from roboclock.simulators.isaac.rendering import configure_renderer

    b.close()
    World.clear_instance()
    create_new_stage()
    b.task = task
    b.layout = liquid_layout(task, b.config)
    task.bind(b.layout)
    b.world = World(
        stage_units_in_meters=1.0,
        physics_dt=b.config["sim"]["physics_dt"],
        rendering_dt=b.config["sim"]["physics_dt"],
    )
    b.world.get_physics_context().enable_ccd(True)
    b.world.scene.add_default_ground_plane()
    b.stage = get_current_stage()
    appearance_config = dict(b.config)
    appearance_config["visuals"] = dict(b.config["visuals"], **b.config.get("liquid_visuals", {}))
    appearance_config["visuals"]["hdr_path"] = str(ASSET_ROOT / "mountain/alps_field_2k.hdr")
    configure_renderer(appearance_config)
    b.appearance = LabAppearance(b.stage, b.world, appearance_config)
    # Reuse the repository's MDL glass, with optical depth suited to thin lab walls.
    glass = b.appearance.glass()
    glass.shaders_list[0].GetInput("depth").Set(0.0015)
    glass.shaders_list[0].GetInput("glass_color").Set(Gf.Vec3f(0.99, 0.995, 1.0))
    b.objects = {}
    b.target_parts = {}
    b.supports = {}
    b.kinematic_targets = {}
    b.kinematic_sim = None
    from roboclock.envs.lab.robot import create_robots

    create_robots(b)
    if task.task_id != "A1":
        table = b.world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([0.45, 0, 0.72]),
                scale=np.array([1.3, 1.3, 0.06]),
            )
        )
        UsdGeom.Imageable(table.prim).MakeInvisible()
        b.appearance.rounded(
            "/World/TableVisual", [0.45, 0, 0.72], [1.3, 1.3, 0.06], "worktop", 0.006
        )
    b.appearance.lights(b.layout)
    b.liquid_scene = LiquidScene(b)
    b.contacts = b.world.scene.add(
        RigidPrim(
            prim_paths_expr=["/World/Objects/sample_0"],
            name="sample_contacts",
            track_contact_forces=True,
            reset_xform_properties=False,
            # Both arms' fingers and hand: columns 0-2 the first arm (ROOTS[0]), 3-5 the other.
            # No instruction says which arm works the valve or the collector, so contact and
            # grip by either arm count (measure_liquid).
            contact_filter_prim_paths_expr=[
                [f"{root}/link{i}" for root in ("/World/ARXLeft", "/World/ARXRight") for i in (7, 8, 6)]
            ],
            max_contact_count=4096,
        )
    )
    # Count arm contact onsets against foreground fixtures, excluding fingers.

    fixtures = [
        str(p.GetPath())
        for p in b.stage.Traverse()
        if p.HasAPI(UsdPhysics.CollisionAPI)
        and not str(p.GetPath()).startswith(("/World/ARXLeft", "/World/ARXRight", "/World/Objects"))
    ]
    b.arm_contacts = b.world.scene.add(
        RigidPrim(
            prim_paths_expr=[f"/World/ARXLeft/link{i}" for i in range(1, 7)],
            name="arm_contacts",
            track_contact_forces=True,
            reset_xform_properties=False,
            contact_filter_prim_paths_expr=[fixtures for _ in range(6)],
            max_contact_count=4096,
        )
    )
    from roboclock.envs.lab.dual_arm import build_collision_monitor

    b.secondary_contacts = build_collision_monitor(b, fixtures)
    scene = b.config["scene"]
    position = np.array(scene.get("liquid_camera_position", [1.65, 2.15, 1.70]))
    look = scene.get("liquid_camera_look_at", [0.40, -0.12, 1.05])
    if task.task_id == "A1":
        position = np.array([1.80, 2.30, 2.15])
        look = [-0.25, -0.30, 0.88]
    b.camera = None
    position, orientation = b.create_overview_camera(position, look)
    b.create_rig_cameras()
    b.world.reset()
    from roboclock.envs.lab.robot import initialize_robots

    initialize_robots(b)
    b.initialize_overview_camera(
        position, orientation, scene.get("liquid_camera_focal_length", 32.0), 30.0
    )
    for _ in range(round(b.config["sim"]["settle_seconds"] / b.config["sim"]["physics_dt"])):
        b.world.step(render=False)
    b.finish_camera_setup()
    b.last_collision = False
    b.collision_count = 0
    b._previous_tool = None
    b._previous_grip = None
    b._grip_contact_dwell = 0.0
    b._cached_state = measure_liquid(b, 0.0)
    task.previous = b._cached_state
    if task.task_id != "A3":
        task.opening = b._cached_state.controls["opening"]
    return b.observation()


def liquid_grips(contact, width, secondary_width):
    """(first arm grips, other arm grips) the valve or collector grip: both of an arm's fingers
    touch it with that gripper closed under 0.06 m. `contact` lists, per arm, fingers then hand
    (columns 0-2 the first arm, 3-5 the other; a three-column list is a single-arm rig)."""
    first = bool(contact[0] and contact[1] and width < 0.06)
    other = bool(len(contact) >= 5 and contact[3] and contact[4] and secondary_width < 0.06)
    return first, other


def measure_liquid(b, dt):
    body = b.objects["sample_0"]
    position, quat = body.get_world_pose()
    velocity = body.get_linear_velocity()
    ee, orientation = b._tool_pose()
    ee2, _ = b._tool_pose(b.secondary_robot)
    q = b.robot.get_joint_positions()
    q2 = b.secondary_robot.get_joint_positions()
    forces = b.contacts.get_contact_force_matrix(dt=b.config["sim"]["physics_dt"])
    contact = np.linalg.norm(forces[0], axis=-1) > 0.03
    grips = liquid_grips(contact, q[6] + q[7], q2[6] + q2[7])
    if b.task.task_id == "A3":
        grip = position.copy()
        offset = (b.liquid_scene.pivot[0] - position[0]) / TUBE_SPACING
        index = int(round(offset))
        receiver = index if 0 <= index <= len(b.task.colors) and abs(offset - index) < 0.18 else -1
        controls = {
            "receiver": receiver,
            "rack_x": float(position[0]),
            "robot_contact": bool(contact.any()),
        }
    else:
        w, x, y, z = quat
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        radius = 0.0 if b.task.task_id == "A1" else LEVER_RADIUS
        grip = position + np.array([radius * np.cos(yaw), radius * np.sin(yaw), 0])
        controls = {
            "opening": float(np.clip(1 - yaw / CLOSE_ANGLE, 0, 1)),
            "valve_angle": yaw,
            "robot_contact": bool(contact.any()),
        }
    bilateral = any(grips)
    b._grip_contact_dwell = b._grip_contact_dwell + dt if bilateral else 0.0
    # The hand the grip moves with: the other arm's when only that one holds it.
    holder = 1 if grips[1] and not grips[0] else 0
    tool = ee2 if holder else ee
    relative_speed = float("inf")
    if (
        dt > 0
        and b._previous_tool is not None
        and b._previous_grip is not None
        and getattr(b, "_previous_holder", 0) == holder
    ):
        relative_speed = float(
            np.linalg.norm((grip - b._previous_grip - tool + b._previous_tool) / dt)
        )
    b._previous_tool, b._previous_grip, b._previous_holder = tool.copy(), grip.copy(), holder
    magnitudes = np.linalg.norm(forces[0], axis=-1)
    controls["finger_forces_N"] = magnitudes[:3].tolist()
    controls["secondary_finger_forces_N"] = magnitudes[3:].tolist()
    controls["gripping_arm"] = holder if bilateral else None
    obj = ObjectState(
        grip,
        quat,
        velocity,
        b._grip_contact_dwell >= 0.04,
        bool(contact.any()),
        False,
        relative_tool_speed=relative_speed,
    )
    return PhysicalState(
        {"sample_0": obj},
        ee,
        bool(q[6] + q[7] > 0.065),
        bool(np.linalg.norm(b.robot.get_joint_velocities()[:6]) > 0.03),
        orientation,
        controls,
        ee2.copy(),
    )
