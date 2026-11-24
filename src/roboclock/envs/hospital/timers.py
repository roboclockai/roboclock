"""A countdown bar beside a deadline light: it shrinks as the deadline approaches.

Only USD calls are used, so this builds in a plain USD stage as well as in Isaac Sim. The bar drains
toward -y; its color goes green, then amber, then red as the time runs out.
"""

BAR_LENGTH, BAR_WIDTH, BAR_HEIGHT = 0.12, 0.014, 0.008
OFFSET = 0.068  # from the light's center along +x, clear of its housing
COLORS = {
    "green": (0.10, 0.85, 0.30),
    "amber": (1.00, 0.60, 0.05),
    "red": (0.95, 0.10, 0.10),
    "blue": (0.10, 0.35, 1.00),
}


class TimerBar:
    def __init__(self, stage, appearance, name, light_position, table_z):
        from pxr import Gf, UsdGeom

        self.x = float(light_position[0]) + OFFSET
        self.y0 = float(light_position[1]) - BAR_LENGTH / 2
        self.z = float(table_z) + BAR_HEIGHT / 2 + 0.0005
        self.materials = {
            key: appearance.material(f"timer_{key}", color=color, roughness=0.4, emissive=True)
            for key, color in COLORS.items()
        }
        self.color = None
        self.track = UsdGeom.Cube.Define(stage, f"/World/Timers/{name}_Track")
        self.track.CreateSizeAttr(1.0)
        self.track.AddTranslateOp().Set(Gf.Vec3d(self.x, self.y0 + BAR_LENGTH / 2, self.z - 0.0008))
        self.track.AddScaleOp().Set(
            Gf.Vec3f(BAR_WIDTH + 0.006, BAR_LENGTH + 0.006, BAR_HEIGHT * 0.6)
        )
        self._bind(self.track, appearance.material("timer_track", color=(0.08, 0.09, 0.10)))
        self.bar = UsdGeom.Cube.Define(stage, f"/World/Timers/{name}_Bar")
        self.bar.CreateSizeAttr(1.0)
        self.translate = self.bar.AddTranslateOp()
        self.scale = self.bar.AddScaleOp()
        for prim in (self.track.GetPrim(), self.bar.GetPrim()):
            UsdGeom.Imageable(prim).MakeInvisible()

    @staticmethod
    def _bind(cube, material):
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material.material)

    def set(self, fraction, color):
        """Show the bar with `fraction` of its length left, in the named color."""
        from pxr import Gf, UsdGeom

        fraction = min(max(float(fraction), 0.0), 1.0)
        length = max(fraction * BAR_LENGTH, 1e-4)
        self.translate.Set(Gf.Vec3d(self.x, self.y0 + length / 2, self.z))
        self.scale.Set(Gf.Vec3f(BAR_WIDTH, length, BAR_HEIGHT))
        if color != self.color:
            self.color = color
            self._bind(self.bar, self.materials[color])
        for prim in (self.track.GetPrim(), self.bar.GetPrim()):
            UsdGeom.Imageable(prim).MakeVisible()


class Dial:
    """An analog countdown dial that replaces the bottle-shaped status light.

    A round clock face standing on its housing and facing +x: a ring of segments that empties as time runs out, a hand
    that sweeps once around, and a lit hub in the middle that shows the state (blue waiting, green open, red late).
    `scale` multiplies every dimension (the default dial is 9.4 cm across). Only USD calls are used, so it builds in a
    plain USD stage as well as in Isaac Sim.
    """

    SEGMENTS = 24

    def __init__(self, stage, appearance, name, center, hub_material=None, scale=1.0):
        import numpy as np
        from pxr import Gf, UsdGeom, UsdShade

        self.stage, self.appearance = stage, appearance
        self.k = k = float(scale)
        self.center = np.asarray(center, dtype=float)
        self.base = f"/World/Dials/{name}"
        self.timed = False
        self.color = None
        self.fraction = 1.0
        self.ring, self.hand_length = 0.036 * k, 0.030 * k
        cx, cy, cz = self.center
        UsdGeom.Xform.Define(stage, "/World/Dials")
        appearance.cylinder(
            f"{self.base}_Bezel", [cx, cy, cz], 0.047 * k, 0.012 * k, "stainless", axis="X"
        )
        appearance.cylinder(
            f"{self.base}_Face",
            [cx + 0.0045 * k, cy, cz],
            0.042 * k,
            0.004 * k,
            "clock_face",
            axis="X",
        )
        tick = appearance.material("dial_tick", color=(0.05, 0.05, 0.06))
        for hour in range(12):
            angle = np.radians(30 * hour)
            length = (0.0075 if hour % 3 == 0 else 0.0045) * k
            self._cube(
                f"{self.base}_Tick_{hour}",
                (cx + 0.0068 * k, cy + 0.0405 * k * np.sin(angle), cz + 0.0405 * k * np.cos(angle)),
                (0.001 * k, 0.0022 * k, length),
                -np.degrees(angle),
                tick,
            )
        self.materials = {
            key: appearance.material(f"timer_{key}", color=color, roughness=0.4, emissive=True)
            for key, color in COLORS.items()
        }
        self.segments = []
        for n in range(self.SEGMENTS):
            angle = 2 * np.pi * (n + 0.5) / self.SEGMENTS
            self.segments.append(
                self._cube(
                    f"{self.base}_Seg_{n}",
                    (
                        cx + 0.0075 * k,
                        cy + self.ring * np.sin(angle),
                        cz + self.ring * np.cos(angle),
                    ),
                    (0.002 * k, 0.0085 * k, 0.0065 * k),
                    -np.degrees(angle),
                    self.materials["blue"],
                )
            )
        self.hand = UsdGeom.Cube.Define(stage, f"{self.base}_Hand")
        self.hand.CreateSizeAttr(1.0)
        self._hand_t = self.hand.AddTranslateOp()
        self._hand_r = self.hand.AddRotateXOp()
        self._hand_s = self.hand.AddScaleOp()
        self._hand_s.Set(Gf.Vec3f(0.0025 * k, 0.0030 * k, self.hand_length))
        UsdShade.MaterialBindingAPI.Apply(self.hand.GetPrim()).Bind(
            appearance.material("dial_hand", color=(0.04, 0.04, 0.05)).material
        )
        self._point(1.0)
        hub = hub_material if hub_material is not None else self.materials["blue"]
        appearance.sphere(f"{self.base}_Hub", [cx + 0.0085 * k, cy, cz], 0.0085 * k, hub)

    def _cube(self, path, position, scale, rotate_x, material):
        from pxr import Gf, UsdGeom, UsdShade

        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
        cube.AddRotateXOp().Set(float(rotate_x))
        cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material.material)
        return cube.GetPrim()

    def _point(self, fraction):
        """Hand at 12 o'clock when full, sweeping clockwise once around as the time runs out."""
        import numpy as np
        from pxr import Gf

        angle = (1.0 - fraction) * 2 * np.pi
        cx, cy, cz = self.center
        reach = self.hand_length / 2
        self._hand_t.Set(
            Gf.Vec3d(cx + 0.0085 * self.k, cy + reach * np.sin(angle), cz + reach * np.cos(angle))
        )
        self._hand_r.Set(float(-np.degrees(angle)))

    def _paint(self, color):
        from pxr import UsdShade

        if color != self.color:
            self.color = color
            for prim in self.segments:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.materials[color].material)

    def _show(self, count):
        from pxr import UsdGeom

        for n, prim in enumerate(self.segments):
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeVisible() if n < count else imageable.MakeInvisible()

    def set(self, fraction, color):
        """Show `fraction` of the time left: that share of the ring stays lit and the hand sweeps."""
        import math

        self.timed = True
        self.fraction = min(max(float(fraction), 0.0), 1.0)
        self._show(math.ceil(self.fraction * self.SEGMENTS - 1e-9))
        self._point(self.fraction)
        self._paint(color)

    def state(self, rgb):
        """The task's state color (blue waiting, green open, red late): lights the ring while no countdown runs."""
        if self.timed:
            return
        name = min(COLORS, key=lambda key: sum((a - b) ** 2 for a, b in zip(COLORS[key], rgb)))
        self._show(self.SEGMENTS)
        self._point(1.0)
        self._paint(name)
