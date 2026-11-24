"""C3 (task 49) bench scale: a static pan that weighs whatever rests on it.

The reading is the measured native contact normal force of the parts on the pan divided by g,
the way a load cell reads the force on its platform: fingers pressing a part down or carrying
part of its weight change the reading. Every 0.2 s the display takes the mean of the last
0.4 s, rounded to the gram, if the load stayed within 2 g over it, and otherwise holds its last
stable value (a scale's stability gate). The value is shown on a 3-digit 7-segment remote
indicator on a post behind the input tray, tilted toward the head camera. A gripper above the pan does not hide it there (a readout on the
scale itself was covered by the wrist). A printed "MIN 125 g" sits below the digits.

The segments are non-physical quads switched by Fabric transforms (``set_visual_target``):
lit segments sit on the display face, unlit ones below the factory floor, as the QC lamp does.
"""

from pathlib import Path

import numpy as np

G = 9.81
# Pan centre (x, y) and top height. The top equals the feed-rail top, so a part rests at its
# nest height (centre z 0.94885).
PAN_XY = (0.0, 0.47)
PAN_TOP_Z = 0.92785
PAN_SIZE = (0.12, 0.10, 0.006)
BENCH_TOP_Z = 0.8875
# Remote indicator: face centre, half size (u along +x, v up the tilted face) and tilt back
# from vertical, so its normal points at the head camera (0, 0.36, 1.743).
INDICATOR_CENTER = (0.0, 0.965, 1.02)
INDICATOR_HALF = (0.11, 0.04)
INDICATOR_TILT_DEG = 50.0
UPDATE_S = 0.2
STABLE_S = 0.4
STABLE_SPAN_G = 2.0
MIN_G = 125
# Nominal part mass and the short-shot (underweight) mass, both times the task's mass_scale.
NOMINAL_KG = 0.150
UNDERWEIGHT_KG = 0.105
# A part is weighed once it rests alone and released on the pan this long.
WEIGH_DWELL_S = 0.5

# 7-segment patterns (a b c d e f g).
DIGITS = {
    " ": "0000000", "-": "0000001",
    "0": "1111110", "1": "0110000", "2": "1101101", "3": "1111001", "4": "0110011",
    "5": "1011011", "6": "1011111", "7": "1110000", "8": "1111111", "9": "1111011",
}
DIGIT_W, DIGIT_H, STROKE, PITCH = 0.030, 0.050, 0.007, 0.039


def segment_rects():
    """(centre_u, centre_v, width, height) of segments a-g in a digit's own frame (v up)."""
    w, h, s = DIGIT_W, DIGIT_H, STROKE
    vert = (s, h / 2 - s / 2)
    return [
        (0, h / 2 - s / 2, w - s, s),         # a
        (w / 2 - s / 2, h / 4, *vert),        # b
        (w / 2 - s / 2, -h / 4, *vert),       # c
        (0, -h / 2 + s / 2, w - s, s),        # d
        (-w / 2 + s / 2, -h / 4, *vert),      # e
        (-w / 2 + s / 2, h / 4, *vert),       # f
        (0, 0, w - s, s),                     # g
    ]


def text(grams):
    """The three characters shown for a reading in grams (right-aligned, 0 when empty)."""
    value = int(round(max(0.0, grams)))
    if value > 999:
        return "---"
    return f"{value:>3d}"


def build(cell):
    """Static scale bodies (colliders added by the scene's ``collide``)."""
    x, y = PAN_XY
    pan_bottom = PAN_TOP_Z - PAN_SIZE[2]
    cell.box("Scale/Base", [x, y, (BENCH_TOP_Z + pan_bottom - 0.004) / 2],
             [0.15, 0.13, pan_bottom - 0.004 - BENCH_TOP_Z], "frame")
    cell.box("Scale/Pan", [x, y, PAN_TOP_Z - PAN_SIZE[2] / 2], list(PAN_SIZE), "steel")
    cx, cy, cz = INDICATOR_CENTER
    top = cz - 0.01
    cell.box("Scale/Housing", [cx, cy + 0.012, (BENCH_TOP_Z + top) / 2], [0.03, 0.03, top - BENCH_TOP_Z], "frame")


def indicator_frame():
    """Face centre, its u (+x) and v (up the face) axes and its normal (toward the head camera)."""
    b = np.deg2rad(INDICATOR_TILT_DEG)
    up = np.array([0.0, np.sin(b), np.cos(b)])
    normal = np.array([0.0, -np.cos(b), np.sin(b)])
    return np.array(INDICATOR_CENTER), np.array([1.0, 0, 0]), up, normal


def _face_texture(directory):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1024, 372
    image = Image.new("RGB", (width, height), (14, 18, 16))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, width - 5, height - 5), outline=(120, 130, 128), width=6)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    big = ImageFont.truetype(font, 150)
    small = ImageFont.truetype(font, 64)
    # "g" right of the digits; the minimum on the lower strip.
    draw.text((832, 149), "g", fill=(235, 238, 235), font=big, anchor="mm")
    draw.text((width / 2, 325), f"MIN {MIN_G} g", fill=(235, 238, 235), font=small, anchor="mm")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ScaleFace.png"
    image.save(path)
    return path


def _quad(stage, path, half_u, half_v, *, uv=False):
    """A quad in the indicator's tilted face plane, centred on its own origin."""
    from pxr import Sdf, UsdGeom

    _, u, v, n = indicator_frame()
    corners = [(-half_u, -half_v), (half_u, -half_v), (half_u, half_v), (-half_u, half_v)]
    points = [tuple((a * u + b * v).tolist()) for a, b in corners]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([tuple(n.tolist())] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)
    pts = np.array(points)
    mesh.CreateExtentAttr([tuple(pts.min(0).tolist()), tuple(pts.max(0).tolist())])
    if uv:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    UsdGeom.Xformable(mesh).AddTranslateOp()
    return mesh


def _material(stage, path, color=None, texture=None):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    if texture is not None:
        tex = UsdShade.Shader.Define(stage, path + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        reader = UsdShade.Shader.Define(stage, path + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(3 * v for v in color)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


class WeighScale:
    """The scale's display and its reading; ``update(o)`` once per control tick."""

    def __init__(self, scene, output):
        from isaacsim.core.prims import XFormPrim
        from pxr import UsdGeom, UsdShade

        stage = scene.stage
        center, u_axis, v_axis, normal = indicator_frame()
        back = _quad(stage, "/World/Scale/Display/Back", INDICATOR_HALF[0] + 0.008, INDICATOR_HALF[1] + 0.008)
        back.GetPrim().GetAttribute("xformOp:translate").Set(tuple((center - 0.004 * normal).tolist()))
        UsdShade.MaterialBindingAPI.Apply(back.GetPrim()).Bind(
            _material(stage, "/World/Scale/Display/BackMaterial", color=(0.03, 0.035, 0.04)))
        face = _quad(stage, "/World/Scale/Display/Face", *INDICATOR_HALF, uv=True)
        face.GetPrim().GetAttribute("xformOp:translate").Set(tuple(center.tolist()))
        UsdShade.MaterialBindingAPI.Apply(face.GetPrim()).Bind(
            _material(stage, "/World/Scale/Display/FaceMaterial", texture=_face_texture(Path(output) / "labels")))
        lit = _material(stage, "/World/Scale/Display/SegmentMaterial", color=(1.0, 0.12, 0.03))
        # Digits on the upper part of the face; "g" to their right and MIN below them.
        digit_v = 0.008
        first_u = -0.05
        self.targets, self.bodies, self.lit = {}, {}, {}
        for k in range(3):
            for s, (u, v, w, h) in enumerate(segment_rects()):
                path = f"/World/Scale/Display/D{k}S{s}"
                _quad(stage, path, w / 2, h / 2)
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(lit)
                UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeVisible()
                target = center + (first_u + k * PITCH + u) * u_axis + (digit_v + v) * v_axis + 0.0008 * normal
                stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Set(tuple(target.tolist()))
                self.targets[(k, s)] = target
                self.bodies[(k, s)] = scene.world.scene.add(XFormPrim(path, name=f"scale_d{k}s{s}"))
                self.lit[(k, s)] = True
        self.hidden = {key: np.array([0.0, 1.0, -10.0 - i * 0.01]) for i, key in enumerate(self.targets)}
        self.window, self.window_start = [], None
        self.shown = None
        self.grams = 0.0

    def reset_display(self):
        """After world.reset(): show 0."""
        self.shown = None
        self._show(text(0))

    def _show(self, chars):
        from .arx_cell_task import set_visual_target

        if chars == self.shown:
            return
        self.shown = chars
        for k, ch in enumerate(chars):
            pattern = DIGITS.get(ch, DIGITS["-"])
            for s in range(7):
                on = pattern[s] == "1"
                if on != self.lit[(k, s)]:
                    set_visual_target(self.bodies[(k, s)], self.targets[(k, s)] if on else self.hidden[(k, s)])
                    self.lit[(k, s)] = on

    def update(self, t, load_n):
        """Add one control-tick load sample (N). Every UPDATE_S the display takes the mean of the
        last STABLE_S if the load stayed within STABLE_SPAN_G over it (a scale's stability
        gate), else it keeps the last stable value. ``grams`` is the displayed value."""
        self.window.append((t, load_n / G * 1000.0))
        while self.window and self.window[0][0] < t - STABLE_S - 1e-6:
            self.window.pop(0)
        if self.window_start is None:
            self.window_start = t
        if t - self.window_start >= UPDATE_S - 1e-6:
            self.window_start = t
            values = np.array([g for _, g in self.window])
            spanned = t - self.window[0][0] >= STABLE_S - 0.03
            if spanned and np.ptp(values) <= STABLE_SPAN_G:
                mean = float(values.mean())
                self.grams = 0.0 if mean < STABLE_SPAN_G else mean
                self._show(text(self.grams))
        return self.grams


def part_mass_kg(resolved, index):
    base = UNDERWEIGHT_KG if resolved["underweight"][index] else NOMINAL_KG
    return base * resolved["mass_scale"]


class WeighTracker:
    """Credits ``{i}_weighed``: part i resting alone on the pan, released (no finger force),
    at rest, for ``WEIGH_DWELL_S``. Shared by the scene (oracle sequencing) and the monitor."""

    def __init__(self):
        self.since = {}
        self.weighed = {}

    @staticmethod
    def resting(o, index):
        part = o["parts"][str(index)]
        others = any(o["parts"][k]["scale_N"] > 0.01 for k in o["parts"] if k != str(index))
        return (
            part["scale_N"] > 0.01
            and not others
            and max(f for forces in part["finger_forces_by_arm"].values() for f in forces) < 0.01
            and np.linalg.norm(part["linear_velocity"]) < 0.012
            and np.linalg.norm(part["angular_velocity"]) < 0.1
        )

    def update(self, o):
        """Returns [(index, reading_g)] newly weighed this tick."""
        t, new = o["time_s"], []
        for key in o["parts"]:
            i = int(key)
            if i in self.weighed:
                continue
            if self.resting(o, i):
                start = self.since.setdefault(i, t)
                # ...and the display has settled on the load (not the empty-pan 0).
                if t - start >= WEIGH_DWELL_S - 1e-6 and (o.get("scale_reading_g") or 0) > STABLE_SPAN_G:
                    self.weighed[i] = o.get("scale_reading_g")
                    new.append((i, self.weighed[i]))
            else:
                self.since.pop(i, None)
        return new
