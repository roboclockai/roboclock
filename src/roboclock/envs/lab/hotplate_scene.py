"""B1 apparatus assembled from the bundled CC0 hotplate and local glass geometry."""

import numpy as np

from roboclock.resources import resolve_asset

PLATE_HEIGHT = 0.080
# Controller panel beside the plate (plate frame: origin on the bench under its centre).
SCREEN_XY = (0.02, 0.1375)
SCREEN_SIZE = (0.085, 0.130)
PANEL_TOP = 0.028
DIGIT_HEIGHT = 0.060
PARKED_Z = -0.5  # world height where the prims of inactive display states wait


def beaker_visual(appearance, path, radius, height):
    """Open glass wall with a shallow residue layer; no cap or liquid dynamics."""
    from pxr import UsdShade

    from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

    bottom, top = -height / 2, height / 2
    profile = [
        (0.0, bottom),
        (radius - 0.002, bottom),
        (radius, bottom + 0.002),
        (radius, top - 0.001),
        (radius - 0.001, top),
        (radius - 0.0015, top - 0.001),
        (radius - 0.0015, bottom + 0.003),
        (0.0, bottom + 0.003),
    ]
    mesh = author_mesh(appearance.stage, path + "/Glass", revolved(profile))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(appearance.glass().material)
    residue = appearance.material("BeakerResidue", color=[0.42, 0.19, 0.025], roughness=0.5)
    appearance.cylinder_child(path, "Residue", radius - 0.0016, 0.001, bottom + 0.0035, residue)
    for i, text in enumerate(("30 --", "60 --", "90 --")):
        appearance.label(
            path + f"/Graduation_{i}",
            text,
            [0, 0, bottom + 0.021 + i * 0.022],
            [0.031, 0.008],
            color=(0.08, 0.14, 0.17),
            cylindrical_radius=radius + 0.0002,
            printed=True,
        )


class HotplateScene:
    def __init__(self, backend):
        from isaacsim.core.api.objects import FixedCuboid
        from pxr import Gf, UsdGeom

        self.backend = backend
        a = backend.appearance
        center = backend.layout.heating_zone.position.copy()
        center[2] = backend.layout.table_height
        self.center = center
        root = UsdGeom.Xform.Define(backend.stage, "/World/Hotplate")
        root.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))
        visual = backend.stage.DefinePrim("/World/Hotplate/Imported", "Xform")
        visual.GetReferences().AddReference(str(resolve_asset("lab://hotplate/hotplate.usdc")))
        # Replace the original static illuminated digits with the live panel below.
        UsdGeom.Imageable(
            backend.stage.GetPrimAtPath("/World/Hotplate/Imported/Part_3_1")
        ).MakeInvisible()
        self.surfaces = {}
        for name, path, pos, size in (
            ("heater", "/World/HeatingZone", [0, 0, 0.076], [0.16, 0.10, 0.008]),
            ("heater_body", "/World/HotplateBody", [0, 0, 0.036], [0.18, 0.12, 0.072]),
        ):
            body = backend.world.scene.add(
                FixedCuboid(
                    prim_path=path,
                    name=name,
                    position=center + pos,
                    scale=np.array(size),
                )
            )
            backend._collision_settings(body.prim)
            UsdGeom.Imageable(body.prim).MakeInvisible()
            self.surfaces[name] = path
        # Surface temperature probe, outside the grasping envelope.
        a.cube("/World/Hotplate/Probe", [0.087, -0.025, 0.105], [0.004, 0.004, 0.07], "steel")
        a.cube("/World/Hotplate/ProbeFoot", [0.086, -0.025, 0.078], [0.012, 0.012, 0.004], "steel")
        self.placed = {}
        self.display(a)
        from roboclock.envs.lab.reactive_scene import StateLamp

        self.state_lamp = StateLamp(backend)
        self.last_display = None
        self.update(backend.task.visual_state())

    def display(self, a):
        """Controller panel on the bench beside the plate, its display facing up.

        The display used to be on the plate's +y side face (78 x 32 mm, digits 20 mm): in the
        head camera, which looks along +x from above, that face is seen at a grazing angle,
        about 5 x 24 px. This one is about 27 x 42 px: the whole screen flashes red in the
        alarm and the digits are 19 px tall, upright in the head image (up = +x).

        Everything on it changes by moving pre-coloured prims between their place and a
        parking spot under the floor (see safety_instruments.variant): binding another
        material or toggling visibility reached the policy cameras several frames late.
        """
        from pxr import Gf, UsdGeom, UsdShade

        stage = self.backend.stage
        cx, cy = SCREEN_XY
        a.rounded(
            "/World/Hotplate/DisplayPanel",
            [cx, cy, PANEL_TOP / 2],
            [SCREEN_SIZE[0] + 0.02, SCREEN_SIZE[1] + 0.015, PANEL_TOP],
            "rubber",
            0.004,
        )
        self.green = a.material("HotplateGreen", color=[0.08, 0.95, 0.30], emissive=True)
        self.red = a.material("HotplateRed", color=[1.0, 0.01, 0.004], emissive=True)
        for material, intensity in ((self.green, 60.0), (self.red, 80.0)):
            material.shaders_list[0].GetInput("emissive_intensity").Set(intensity)
        dark = a.material("HotplateScreenDark", color=[0.01, 0.014, 0.018], roughness=0.25)
        materials = {"green": self.green, "red": self.red, "dark": dark}
        parked_z = PARKED_Z - self.center[2]

        def plate(name, pos, size, material):
            cube = UsdGeom.Cube.Define(stage, "/World/Hotplate/" + name)
            cube.CreateSizeAttr(1.0)
            translate = cube.AddTranslateOp()
            cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material.material)
            home = Gf.Vec3d(*map(float, pos))
            return translate, home, Gf.Vec3d(home[0], home[1], parked_z)

        self.screen = {
            key: plate(
                "Screen_" + key, [cx, cy, PANEL_TOP + 0.001], [*SCREEN_SIZE, 0.002], materials[key]
            )
            for key in ("dark", "red")
        }
        # Seven segments per digit as (right, up, width, height) in digit heights; a digit's
        # top is toward +x and the number reads toward -y (left to right in the head image).
        height = DIGIT_HEIGHT
        self.digits = []
        for digit in range(3):
            segments = []
            for index, (right, up, w, h) in enumerate(
                (
                    (0, 0.44, 0.34, 0.12),
                    (0.23, 0.22, 0.12, 0.34),
                    (0.23, -0.22, 0.12, 0.34),
                    (0, -0.44, 0.34, 0.12),
                    (-0.23, -0.22, 0.12, 0.34),
                    (-0.23, 0.22, 0.12, 0.34),
                    (0, 0, 0.34, 0.12),
                )
            ):
                segments.append(
                    {
                        key: plate(
                            f"Digit_{digit}_{index}_{key}",
                            [
                                cx + up * height,
                                cy + ((1 - digit) * 0.68 - right) * height,
                                PANEL_TOP + 0.0025,
                            ],
                            [h * height, w * height, 0.001],
                            materials[key],
                        )
                        for key in ("green", "red", "dark")
                    }
                )
            self.digits.append(segments)

    def update(self, values):
        self.state_lamp.update()
        text = f"{round(values['temperature_c']):03d}"
        # 2 Hz flashing from the alarm on. Fault remains red after a late recovery.
        lit = not values["alarm"] or int(values["alarm_elapsed"] * 4) % 2 == 0
        red = values["alarm"] or values["failed"]
        key = (text, lit, red)
        if key == self.last_display:
            return
        self.last_display = key
        # Normal: green digits on a dark screen. Alarm: the screen alternates between red
        # with dark digits and dark with red digits, so the reading stays legible.
        screen = "red" if red and lit else "dark"
        colour = "green" if not red else "dark" if lit else "red"
        # An unlit segment shows the prim of the screen's own colour. A segment that only
        # left would stay in the policy cameras' image for many frames (measured: "099" was
        # still shown 1 s later, at 117 C); a prim that arrives replaces it at once.
        masks = (
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
        )
        self.place(self.screen, screen)
        for char, segments in zip(text, self.digits):
            for letter, variants in zip("abcdefg", segments):
                self.place(variants, colour if letter in masks[int(char)] else screen)

    def place(self, variants, shown):
        """Bring one of a part's pre-coloured prims to its place and park the others."""
        for name, (translate, home, parked) in variants.items():
            path = translate.GetAttr().GetPath()
            if self.placed.get(path) != (name == shown):
                self.placed[path] = name == shown
                translate.Set(home if name == shown else parked)
