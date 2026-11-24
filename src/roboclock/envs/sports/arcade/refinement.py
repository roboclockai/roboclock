"""Refined arcade fixtures and chess-club furnishing, with explicit PBR sources.

Architectural dressing is visual-only; existing task colliders remain authoritative.
Digital panel artwork is authored here, not a photograph or simulated game outcome.
"""

import hashlib

import numpy as np

from roboclock.envs.sports.arcade.scenes import ROOT


def bind(b, path, material):
    from pxr import UsdShade

    prim = b.stage.GetPrimAtPath(path)
    if prim:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )


def pbr(a, name, color, roughness=0.4, metallic=0.0, texture=None, uv=(1, 1), emission=0):
    from pxr import Gf, Sdf

    m = a.material(name, color=color, roughness=roughness, metallic=metallic)
    m.set_project_uvw(False)
    sh = m.shaders_list[0]
    if texture:
        for suffix, param in [
            ("diff", "diffuse_texture"),
            ("rough", "reflectionroughness_texture"),
            ("nor_gl", "normalmap_texture"),
        ]:
            f = next((ROOT / texture).glob(f"*_{suffix}_1k.*"))
            inp = sh.CreateInput(param, Sdf.ValueTypeNames.Asset)
            inp.Set(Sdf.AssetPath(str(f.resolve())))
            inp.GetAttr().SetMetadata("colorSpace", "sRGB" if suffix == "diff" else "raw")
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.12)
        sh.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(0.45)
        m.set_texture_scale(*uv)
    if emission:
        sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(emission)
    return m


def artwork(title, subtitle, style="title", accent=(255, 170, 75)):
    from PIL import Image, ImageDraw, ImageFont

    root = ROOT / "authored_panels"
    root.mkdir(exist_ok=True)
    key = hashlib.sha256(repr(("r2", title, subtitle, style, accent)).encode()).hexdigest()[:16]
    path = root / (key + ".png")
    if path.exists():
        return path
    if style == "banner":
        image = Image.new("RGB", (1200, 180), (13, 26, 37))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 68)
        draw.rounded_rectangle((4, 4, 1196, 176), radius=22, outline=accent, width=4)
        draw.text((600, 90), title, font=font, anchor="mm", fill=(245, 233, 207))
        image.save(path)
        return path
    image = Image.new("RGB", (1200, 600), (11, 21, 32))
    d = ImageDraw.Draw(image)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    def f(n):
        return ImageFont.truetype(font, n)

    for y in range(600):
        v = int(12 + 14 * y / 600)
        d.line((0, y, 1200, y), fill=(v, v + 11, v + 19))
    d.rounded_rectangle((18, 18, 1182, 582), radius=30, outline=accent, width=3)
    d.text((62, 48), "STUDIO / PLAY", font=f(22), fill=accent)
    d.text(
        (62, 94),
        title,
        font=ImageFont.truetype(bold, 72 if len(title) < 14 else 57),
        fill=(245, 239, 220),
    )
    d.text((64, 190), subtitle, font=f(24), fill=(154, 179, 192))
    if style == "rhythm":
        for row in range(2):
            y = 325 + row * 120
            d.rounded_rectangle(
                (70, y - 37, 1125, y + 37),
                radius=35,
                fill=(23, 42, 54),
                outline=(43, 69, 81),
                width=2,
            )
            d.ellipse((140, y - 48, 236, y + 48), outline=accent, width=7)
            for x in [400 + row * 70, 660, 860 + row * 60, 1060]:
                d.ellipse(
                    (x - 24, y - 24, x + 24, y + 24), fill=accent if row == 0 else (77, 210, 221)
                )
                d.arc((x - 14, y - 10, x + 14, y + 14), 0, 180, fill=(20, 30, 40), width=3)
    elif style == "moles":
        for row in range(2):
            for col in range(5):
                x = 140 + col * 230
                y = 335 + row * 150
                d.ellipse(
                    (x - 72, y + 22, x + 72, y + 62),
                    fill=(6, 12, 19),
                    outline=(62, 105, 97),
                    width=3,
                )
                if (row + col) % 3:
                    c = (104, 190, 126) if col != 3 else (238, 111, 98)
                    d.ellipse((x - 42, y - 42, x + 42, y + 48), fill=c)
                    for dx in (-15, 15):
                        d.ellipse((x + dx - 5, y - 8, x + dx + 5, y + 2), fill=(13, 26, 25))
                    d.ellipse((x - 7, y + 10, x + 7, y + 19), fill=(252, 210, 175))
    elif style == "chess":
        for row in range(4):
            for col in range(8):
                x = 70 + col * 66
                y = 274 + row * 66
                d.rectangle(
                    (x, y, x + 66, y + 66),
                    fill=(191, 158, 108) if (row + col) % 2 else (39, 64, 66),
                )
        d.text((680, 300), "01 / THINK", font=f(38), fill=accent)
        d.text((680, 375), "02 / MOVE", font=f(38), fill=(236, 230, 210))
        d.text((680, 450), "03 / CLOCK", font=f(38), fill=(110, 184, 186))
        d.text((170, 355), "♞", font=ImageFont.truetype(font, 118), fill=(239, 236, 217))
    elif style == "rally":
        for i in range(8):
            y = 265 + i * 42
            d.line((70, y, 1120, y), fill=(35, 64, 81), width=2)
        d.line((180, 545, 540, 260, 900, 545), fill=accent, width=7)
        d.rounded_rectangle((565, 410, 635, 500), radius=13, fill=(83, 208, 225))
        d.text((750, 320), "READY TO PLAY", font=f(28), fill=(207, 223, 221))
    else:
        for i in range(13):
            x = 65 + i * 85
            h = 35 + (i * 41) % 155
            d.rounded_rectangle(
                (x, 470 - h, x + 38, 470), radius=10, fill=accent if i % 3 else (66, 174, 186)
            )
        d.text((64, 528), "CRAFT / TIMING / PRECISION", font=f(22), fill=(139, 161, 173))
    image.save(path)
    return path


def panel(
    b, a, path, position, size, title, subtitle="", style="title", accent=(255, 170, 75), glow=0.2
):
    from isaacsim.core.api.materials import OmniPBR
    from pxr import Gf, Sdf, UsdShade

    from roboclock.simulators.isaac.visual_geometry import author_mesh

    w, h = size
    mesh = author_mesh(
        b.stage,
        path,
        (
            np.array(
                [[-w / 2, 0, -h / 2], [w / 2, 0, -h / 2], [w / 2, 0, h / 2], [-w / 2, 0, h / 2]]
            ),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.array([0, 1, 2, 3]),
        ),
        position,
    )
    mesh.CreateDoubleSidedAttr(True)
    mat = OmniPBR(path + "Material", color=np.ones(3))
    mat.set_reflection_roughness(0.35)
    mat.set_project_uvw(False)
    tex = artwork(title, subtitle, "banner" if w / h > 3 else style, accent)
    sh = mat.shaders_list[0]
    sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(tex.resolve()))
    )
    if glow:
        sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
        sh.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(tex.resolve()))
        )
        sh.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(glow)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat.material)
    return mesh


def lathe(b, a, path, profile, position, material):
    from pxr import UsdShade

    from roboclock.simulators.isaac.visual_geometry import author_mesh, revolved

    mesh = author_mesh(b.stage, path, revolved(profile, 96), position)
    mesh.CreateSubdivisionSchemeAttr("catmullClark")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(a.material(material).material)
    return mesh


def sphere(b, a, path, p, r, material, scale=None):
    from pxr import Gf, UsdGeom, UsdShade

    s = UsdGeom.Sphere.Define(b.stage, path)
    s.CreateRadiusAttr(r)
    s.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
    if scale:
        s.AddScaleOp().Set(Gf.Vec3f(*scale))
    UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(a.material(material).material)
    return s


def finish_room(b, a, theme):
    pbr(a, "ink", [0.017, 0.025, 0.037], 0.45)
    pbr(a, "cream", [0.75, 0.72, 0.64], 0.33)
    pbr(a, "red_lacquer", [0.44, 0.047, 0.025], 0.24)
    pbr(a, "teal_lacquer", [0.028, 0.25, 0.28], 0.25)
    pbr(a, "brass", [0.52, 0.32, 0.12], 0.28, 0.78)
    pbr(a, "led_warm", [1.0, 0.42, 0.12], 0.3, emission=2)
    pbr(a, "led_blue", [0.035, 0.55, 0.68], 0.3, emission=2)
    pbr(a, "clean_wood", [0.68, 0.49, 0.31], texture="wood_floor", uv=(1, 1))
    pbr(a, "skin", [0.93, 0.79, 0.55], texture="leather_white", uv=(2, 2))
    pbr(a, "felt", [0.11, 0.23, 0.19], 0.94)
    pbr(a, "foam", [0.77, 0.18, 0.06], 0.78)
    pbr(a, "black_gloss", [0.007, 0.012, 0.016], 0.19)
    pbr(a, "rose", [0.75, 0.31, 0.22], 0.54)
    from roboclock.envs.sports.arcade.daylight import dress_room

    dress_room(b, a, theme)


def cabinet(b, a, path, x, y, index):
    from pxr import Gf, UsdGeom

    accent = ("red_lacquer", "teal_lacquer", "cream")[index]
    a.rounded(path + "/Base", [x, y, 0.48], [0.63, 0.56, 0.91], "ink", 0.035)
    # Shoulder and screen box are recessed relative to the projecting controls.
    a.rounded(path + "/ScreenBox", [x, y + 0.035, 1.25], [0.61, 0.34, 0.59], "ink", 0.027)
    for side in (-1, 1):
        a.rounded(
            path + f"/Side{side + 1}",
            [x + side * 0.312, y + 0.02, 0.85],
            [0.037, 0.58, 1.64],
            accent,
            0.014,
        )
        a.rounded(
            path + f"/Strip{side + 1}",
            [x + side * 0.291, y - 0.286, 0.88],
            [0.012, 0.012, 1.20],
            "led_blue" if index != 0 else "led_warm",
            0.004,
        )
    a.rounded(path + "/Controls", [x, y - 0.25, 0.97], [0.64, 0.31, 0.085], accent, 0.022)
    a.rounded(path + "/Marquee", [x, y, 1.72], [0.65, 0.50, 0.16], accent, 0.025)
    titles = ("BEAT STUDIO", "MOLE GARDEN", "NEON RALLY")
    panel(
        b,
        a,
        path + "/Game",
        [x, y - 0.143, 1.27],
        [0.49, 0.39],
        titles[index],
        "INSERT COIN / READY",
        ("rhythm", "moles", "rally")[index],
        glow=0.28,
    )
    panel(b, a, path + "/Brand", [x, y - 0.258, 1.72], [0.53, 0.11], titles[index], "", glow=0.15)
    a.rounded(path + "/CoinDoor", [x, y - 0.287, 0.51], [0.23, 0.024, 0.28], "steel", 0.012)
    a.rounded(path + "/CoinInset", [x, y - 0.304, 0.55], [0.13, 0.009, 0.16], "ink", 0.006)
    a.rounded(path + "/CoinSlot", [x, y - 0.311, 0.59], [0.065, 0.006, 0.009], "brass", 0.002)
    for k in range(3):
        sphere(
            b,
            a,
            path + f"/Button{k}",
            [x + 0.045 + k * 0.062, y - 0.33, 1.019],
            0.020,
            ("red_lacquer", "cream", "teal_lacquer")[k],
            scale=(1, 1, 0.35),
        )
    stick = UsdGeom.Cylinder.Define(b.stage, path + "/JoystickStem")
    stick.CreateRadiusAttr(0.008)
    stick.CreateHeightAttr(0.075)
    stick.AddTranslateOp().Set(Gf.Vec3d(x - 0.14, y - 0.33, 1.047))
    bind(b, str(stick.GetPath()), a.material("steel"))
    sphere(b, a, path + "/Joystick", [x - 0.14, y - 0.33, 1.09], 0.029, "red_lacquer")
    for j in range(7):
        a.rounded(
            path + f"/Speaker{j}",
            [x - 0.115 + j * 0.038, y - 0.282, 0.78],
            [0.012, 0.012, 0.072],
            "black_gloss",
            0.004,
        )


def finish_equipment(b, a):
    from pxr import UsdGeom

    from roboclock.envs.sports.arcade.scenes import vertical_label
    from roboclock.envs.sports.arcade.tasks import station_offset
    from roboclock.envs.sports.common.equipment import annulus

    t = b.task
    bind(b, "/World/Objects/mallet/Shaft", a.material("clean_wood"))
    bind(b, "/World/Objects/mallet/GripSleeve", a.material("rubber"))
    bind(
        b,
        "/World/Objects/mallet/Head",
        a.material("foam" if t.task_id == "34" else "clean_wood"),
    )
    for i in range(7):
        bind(b, f"/World/Objects/mallet/GripBand{i}", a.material("rubber"))
    if t.task_id == "32":
        for path in ("/World/PadScreens", "/World/HoldLED", "/World/RhythmBrand"):
            prim = b.stage.GetPrimAtPath(path)
            if prim:
                UsdGeom.Imageable(prim).MakeInvisible()
        for name, sp in t.layout.controls.items():
            x, y, z = sp.position
            UsdGeom.Imageable(b.stage.GetPrimAtPath(f"/World/Drums/{name}/Shell")).MakeInvisible()
            lathe(
                b,
                a,
                f"/World/Refined/Drum_{name}",
                [
                    (0.091, -0.141),
                    (0.101, -0.131),
                    (0.108, -0.105),
                    (0.110, -0.080),
                    (0.106, -0.045),
                    (0.098, -0.022),
                    (0.094, -0.022),
                    (0.087, -0.137),
                ],
                [x, y, z],
                "red_lacquer",
            )
            bind(b, f"/World/Controls/{name}/Cap", a.material("skin"))
            bind(b, f"/World/Controls/{name}/SkinRim", a.material("cream"))
            for dz in (-0.138, -0.025):
                bind(b, f"/World/Drums/{name}/Rim{int(abs(dz) * 1000)}", a.material("rubber"))
            for j in range(16):
                theta = 2 * np.pi * j / 16
                px, py = x + 0.112 * np.cos(theta), y + 0.112 * np.sin(theta)
                sphere(
                    b,
                    a,
                    f"/World/Refined/DrumStud_{name}_{j}",
                    [px, py, z - 0.032],
                    0.0035,
                    "brass",
                )
                a.rounded(
                    f"/World/Refined/DrumLug_{name}_{j}",
                    [px, py, z - 0.084],
                    [0.005, 0.005, 0.070],
                    "brass",
                    0.002,
                )
            for yy in (y - 0.072, y + 0.072):
                a.rounded(
                    f"/World/Refined/DrumFoot_{name}_{abs(int(yy * 1000))}",
                    [x, yy, 0.766],
                    [0.23, 0.021, 0.025],
                    "ink",
                    0.006,
                )
            # Small drum identity medallion on the front of each shell.
            panel(
                b,
                a,
                f"/World/Refined/DrumBadge_{name}",
                [x, y - 0.110, z - 0.082],
                [0.056, 0.027],
                "BEAT",
                "",
                glow=0,
            )
        bind(b, "/World/Game/ScreenHousing", a.material("teal_lacquer"))
        bind(b, "/World/Game/Timeline", a.material("black_gloss"))
        for row, z in enumerate((1.205, 1.135)):
            a.rounded(
                f"/World/Refined/NoteLane{row}",
                b.screen_point([0.535, 0.326, z]),
                [0.35, 0.002, 0.002],
                "led_warm" if row == 0 else "led_blue",
                0.0005,
            )
            vertical_label(
                b,
                f"/World/Refined/LaneName{row}",
                "L" if row == 0 else "R",
                b.screen_point([0.338, 0.318, z]),
                [0.019, 0.024],
            )
        panel(
            b,
            a,
            "/World/Refined/TaikoHeader",
            b.screen_point([0.53, 0.326, 1.31]),
            [0.405, 0.060],
            "BEAT STUDIO",
            "",
            accent=(255, 177, 85),
            glow=0.1,
        )
        UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/Game/Title")).MakeInvisible()
        # Gold pinstripe, speaker grilles and a visible cable conduit.
        for side in (-1, 1):
            a.rounded(
                f"/World/Refined/ScreenTrim{side + 1}",
                b.screen_point([0.53 + side * 0.221, 0.326, 1.19]),
                [0.009, 0.005, 0.26],
                "brass",
                0.002,
            )
        for j in range(9):
            a.rounded(
                f"/World/Refined/GameSpeaker{j}",
                b.screen_point([0.36 + j * 0.042, 0.326, 1.03]),
                [0.019, 0.006, 0.012],
                "black_gloss",
                0.003,
            )
        a.rounded(
            "/World/Refined/ScreenStand",
            [0.53, 0.365, 0.955]
            if b.config["scene"].get("room_style") == "v5"
            else [0.91, 0.435, 1.09],
            [0.09, 0.05, 0.22]
            if b.config["scene"].get("room_style") == "v5"
            else [0.07, 0.05, 0.68],
            "ink",
            0.010,
        )
    else:
        UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/Game/FrontTitle")).MakeInvisible()
        bind(b, "/World/Game/Front", a.material("teal_lacquer"))
        for name, sp in t.layout.controls.items():
            bind(b, f"/World/Deck/{name}", a.material("felt"))
            bind(b, f"/World/Rims/{name}", a.material("brass"))
            x, y = sp.position[:2]
            annulus(
                b,
                f"/World/Refined/HoleCollar_{name}",
                [x, y, 0.853],
                0.034,
                0.043,
                0.004,
                [0.15, 0.24, 0.22],
            )
            bind(b, f"/World/Refined/HoleCollar_{name}", a.material("brass"))
            # Details live on the moving rigid root and stay inside its head envelope.
            root = f"/World/Controls/{name}"
            for side in (-1, 1):
                sphere(
                    b,
                    a,
                    root + f"/EarDetail{side + 1}",
                    [side * 0.018, -0.001, 0.016],
                    0.006,
                    "rose",
                )
                sphere(
                    b,
                    a,
                    root + f"/CheekDetail{side + 1}",
                    [side * 0.012, -0.022, -0.001],
                    0.005,
                    "rose",
                    scale=(1, 0.45, 0.7),
                )
                sphere(
                    b,
                    a,
                    root + f"/Muzzle{side + 1}",
                    [side * 0.0045, -0.024, -0.003],
                    0.006,
                    "cream",
                    scale=(1, 0.5, 0.8),
                )
                bind(b, root + f"/Pupil{side}".replace("-", "n"), a.material("black_gloss"))
                sphere(
                    b,
                    a,
                    root + f"/EyeGlint{side + 1}",
                    [side * 0.008 + 0.0005, -0.0246, 0.011],
                    0.0007,
                    "paper",
                )
            bind(b, root + "/Nose", a.material("black_gloss"))
            b.control_materials[name].set_roughness(0.32)
        # Cabinet trim moves with the seeded station (``tasks.station_offset``).
        dx, dy, _ = station_offset(b.config)
        panel(
            b,
            a,
            "/World/Refined/MoleBrand",
            [0.62 + dx, -0.236 + dy, 0.805],
            [0.345, 0.092],
            "MOLE GARDEN",
            "NINE HOLES / QUICK REFLEXES",
            style="moles",
            accent=(105, 220, 169),
            glow=0,
        )
        for side in (-1, 1):
            a.rounded(
                f"/World/Refined/MoleRail{side + 1}",
                [0.62 + dx + side * 0.194, dy, 0.844],
                [0.019, 0.41, 0.027],
                "cream",
                0.007,
            )
            a.rounded(
                f"/World/Refined/MoleRailLight{side + 1}",
                [0.62 + dx + side * 0.195, -0.02 + dy, 0.861],
                [0.004, 0.35, 0.003],
                "led_warm",
                0.001,
            )
        a.rounded(
            "/World/Refined/MoleBack",
            [0.62 + dx, 0.205 + dy, 0.875],
            [0.41, 0.029, 0.15],
            "teal_lacquer",
            0.018,
        )
        panel(
            b,
            a,
            "/World/Refined/MoleBackArt",
            [0.62 + dx, 0.189 + dy, 0.881],
            [0.34, 0.094],
            "MOLE GARDEN",
            "HIT GREEN / AVOID RED",
            style="moles",
            accent=(105, 220, 169),
            glow=0.1,
        )
        # Foam ball head has a recessed logo, replacing the inappropriate claw hammer.
        sphere(
            b,
            a,
            "/World/Objects/mallet/EndBadge",
            [0.027, 0, -0.125],
            0.013,
            "cream",
            scale=(0.35, 1, 1),
        )
