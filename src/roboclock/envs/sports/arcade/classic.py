"""Accepted V5 room dressing, preserved from its frozen source archive."""

from roboclock.envs.sports.arcade.refinement import bind, cabinet, panel, pbr
from roboclock.envs.sports.arcade.scenes import imported, vertical_label


def build_room_classic(b, theme):
    from pxr import Gf, UsdGeom, UsdLux

    from roboclock.envs.lab.appearance import LabAppearance

    a = LabAppearance(b.stage, b.world, b.config)
    for path in ("/World/Backdrop", "/World/TableSurface", "/World/Title"):
        prim = b.stage.GetPrimAtPath(path)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()
    a.material("navy", color=[0.025, 0.045, 0.065], roughness=0.65)
    a.material("walnut", color=[0.16, 0.07, 0.025], roughness=0.35)
    a.material("trim", color=[0.70, 0.43, 0.13], roughness=0.27, metallic=0.75)
    a.material("cyan", color=[0.05, 0.55, 0.75], emissive=True)
    a.material("coral", color=[0.75, 0.08, 0.06], emissive=True)
    a.rounded("/World/Room/Floor", [0.4, 0, -0.025], [7, 7, 0.05], "floor")
    a.rounded("/World/Room/BackWall", [0.4, 1.85, 1.6], [5.6, 0.08, 3.2], "wall")
    a.rounded("/World/Room/SideWall", [-2.35, 0, 1.6], [0.08, 3.8, 3.2], "wall")
    for z in (0.08, 1.02):
        a.rounded(
            f"/World/Room/Trim{int(z * 100)}",
            [0.4, 1.795, z],
            [5.6, 0.035, 0.06],
            "walnut" if theme == "chess" else "navy",
        )
    if theme == "chess":
        UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/Table")).MakeInvisible()
        imported(
            b.stage, "side_table_01", "/World/Room/PlayingTable", [0.40, 0, 0], [1.05, 0.95, 0.75]
        )
        for i, xy in enumerate(([-0.65, 1.0], [1.3, 0.85])):
            imported(
                b.stage, "dining_chair_02", f"/World/Room/Chair{i}", [*xy, 0], yaw=150 if i else 200
            )
        imported(
            b.stage, "side_table_01", "/World/Room/SideTable", [1.25, 1.30, 0], [0.5, 0.45, 0.64]
        )
        vertical_label(b, "/World/Room/Sign", "CHESS CLUB", [0.30, 1.795, 2.08], [1.8, 0.28])
        for i in range(3):
            x = -0.6 + i * 0.62
            a.rounded(f"/World/Room/Panel{i}", [x, 1.79, 1.48], [0.47, 0.04, 0.53], "walnut")
            vertical_label(
                b,
                f"/World/Room/PanelText{i}",
                ("THINK", "MOVE", "PLAY")[i],
                [x, 1.762, 1.48],
                [0.40, 0.12],
            )
        for j in range(6):
            a.rounded(
                f"/World/Room/WindowSlat{j}",
                [-2.29, 0.2 + j * 0.20, 1.85],
                [0.03, 0.035, 1.2],
                "walnut",
            )
    else:
        a.rounded("/World/Room/Worktop", [0.43, 0, 0.724], [1.05, 1.0, 0.05], "navy", radius=0.009)
        vertical_label(
            b, "/World/Room/Sign", "PLAY / RHYTHM / REACT", [0.3, 1.795, 2.22], [2.4, 0.24]
        )
        # Purpose-built background arcade fixtures, explicitly authored proxies.
        for i, x in enumerate((-0.9, 0.05, 1.0)):
            y = 1.25
            a.rounded(
                f"/World/Room/Cabinet{i}", [x, y, 0.64], [0.63, 0.55, 1.28], "navy", radius=0.04
            )
            a.rounded(
                f"/World/Room/Screen{i}",
                [x, y - 0.286, 1.25],
                [0.53, 0.02, 0.43],
                "polymer",
                radius=0.008,
            )
            vertical_label(
                b,
                f"/World/Room/GameTitle{i}",
                ("RHYTHM", "HIGH SCORE", "ARCADE")[i],
                [x, y - 0.300, 1.27],
                [0.47, 0.12],
            )
            a.rounded(
                f"/World/Room/Marquee{i}",
                [x, y, 1.67],
                [0.66, 0.58, 0.17],
                "coral" if i % 2 else "cyan",
                radius=0.025,
            )
            a.rounded(
                f"/World/Room/ControlShelf{i}",
                [x, y - 0.30, 0.87],
                [0.61, 0.20, 0.07],
                "steel",
                radius=0.01,
            )
            for side in (-1, 1):
                a.rounded(
                    f"/World/Room/Edge{i}_{side}".replace("-", "n"),
                    [x + side * 0.29, y - 0.282, 0.46],
                    [0.018, 0.014, 0.63],
                    "cyan",
                )
        imported(b.stage, "dining_chair_02", "/World/Room/Chair", [1.35, 0.3, 0], yaw=70)
    # Architectural ceiling fixtures provide a room-scale lighting context.
    for i, x in enumerate((-0.5, 0.8)):
        a.rounded(f"/World/Room/LightHousing{i}", [x, 0.8, 2.65], [0.65, 0.35, 0.06], "steel")
        light = UsdLux.RectLight.Define(b.stage, f"/World/Room/CeilingLight{i}")
        light.AddTranslateOp().Set(Gf.Vec3d(x, 0.8, 2.60))
        light.CreateWidthAttr(0.58)
        light.CreateHeightAttr(0.28)
        light.CreateIntensityAttr(450)
        light.CreateColorAttr(
            Gf.Vec3f(1, 0.88, 0.72) if theme == "chess" else Gf.Vec3f(0.78, 0.88, 1)
        )
    finish_room_classic(b, a, theme)
    return a


def finish_room_classic(b, a, theme):
    from pxr import Gf, UsdGeom, UsdLux

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
    bind(
        b,
        "/World/Room/Floor",
        pbr(a, "room_floor", [0.45, 0.39, 0.31], texture="wood_floor", uv=(5, 5)),
    )
    bind(
        b,
        "/World/Room/BackWall",
        pbr(
            a,
            "architectural_wall",
            [0.15, 0.19, 0.19] if theme == "chess" else [0.042, 0.065, 0.088],
            0.85,
        ),
    )
    bind(b, "/World/Room/SideWall", a.material("architectural_wall"))
    # Slatted wall with shadow gaps and a low, warm continuous light rail.
    for j in range(38):
        a.rounded(
            f"/World/Refined/Slat{j}",
            [-2.1 + j * 0.135, 1.77, 1.23],
            [0.034, 0.045, 2.32],
            "clean_wood",
            0.006,
        )
    for z in (0.10, 2.43):
        a.rounded(
            f"/World/Refined/WallLight{int(z * 100)}",
            [0.4, 1.72, z],
            [5.5, 0.025, 0.018],
            "led_warm",
            0.005,
        )
    imported(
        b.stage,
        "potted_plant_02",
        "/World/Refined/Plant",
        [1.75, 1.38, 0] if theme == "chess" else [1.4, 0.85, 0],
        [0.60, 0.57, 1.15],
        yaw=30,
    )
    if theme == "chess":
        UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/defaultGroundPlane")).MakeInvisible()
        for path in (
            "/World/Room/Sign",
            *[f"/World/Room/Panel{i}" for i in range(3)],
            *[f"/World/Room/PanelText{i}" for i in range(3)],
        ):
            prim = b.stage.GetPrimAtPath(path)
            if prim:
                UsdGeom.Imageable(prim).MakeInvisible()
        panel(
            b,
            a,
            "/World/Refined/ChessArt",
            [0.0, 1.69, 1.76],
            [1.43, 0.66],
            "THE CHESS CLUB",
            "A QUIET PLACE TO THINK",
            style="chess",
            accent=(220, 178, 104),
            glow=0,
        )
        # Clean refinished wood keeps the existing table mesh, joints and proportions.
        bind(b, "/World/Room/PlayingTable", a.material("clean_wood"))
        bind(b, "/World/Room/SideTable", a.material("clean_wood"))
        for i in range(6):
            a.rounded(
                f"/World/Refined/Book{i}",
                [1.0 + i * 0.037, 1.76, 0.88],
                [0.030, 0.19, 0.21 + (i % 3) * 0.025],
                ("ink", "cream", "red_lacquer")[i % 3],
                0.002,
            )
        a.rounded(
            "/World/Refined/BookShelf", [1.14, 1.7, 0.75], [0.64, 0.29, 0.035], "clean_wood", 0.006
        )
        for side in (-1, 1):
            a.rounded(
                f"/World/Refined/BoardInlay{side + 1}",
                [0.48, side * 0.235, 0.773],
                [0.42, 0.008, 0.004],
                "brass",
                0.001,
            )
    else:
        # Replace the generic tall boxes with shaped arcade cabinets and real controls.
        for prefix in ("Cabinet", "Screen", "GameTitle", "Marquee", "ControlShelf", "Neon"):
            for prim in list(b.stage.Traverse()):
                if str(prim.GetPath()).startswith("/World/Room/" + prefix):
                    UsdGeom.Imageable(prim).MakeInvisible()
        for path in ("/World/Room/Sign", "/World/Room/Chair"):
            prim = b.stage.GetPrimAtPath(path)
            if prim:
                UsdGeom.Imageable(prim).MakeInvisible()
        bind(b, "/World/Room/Worktop", a.material("ink"))
        for i, x in enumerate((-0.95, -0.10, 0.80)):
            cabinet(b, a, f"/World/Refined/Arcade{i}", x, 1.29, i)
        panel(
            b,
            a,
            "/World/Refined/ArcadeSign",
            [0.05, 1.67, 2.11],
            [1.86, 0.37],
            "THE PLAY ROOM",
            "RHYTHM  /  REACTION  /  SKILL",
            accent=(92, 212, 217),
            glow=0.3,
        )
        for side in (-1, 1):
            a.rounded(
                f"/World/Refined/TableEdge{side + 1}",
                [0.43, side * 0.495, 0.737],
                [1.03, 0.012, 0.016],
                "brass",
                0.004,
            )
        # The machine pedestal reads as integrated hardware instead of a lab bench.
        a.rounded(
            "/World/Refined/Pedestal", [0.54, 0.0, 0.39], [0.51, 0.58, 0.70], "teal_lacquer", 0.055
        )
        a.rounded(
            "/World/Refined/PedestalKick", [0.54, -0.294, 0.11], [0.42, 0.018, 0.17], "ink", 0.018
        )
        a.rounded(
            "/World/Refined/PedestalDoor", [0.54, -0.295, 0.42], [0.33, 0.018, 0.29], "ink", 0.025
        )
        for k in range(6):
            a.rounded(
                f"/World/Refined/Vent{k}",
                [0.54, -0.308, 0.37 + k * 0.019],
                [0.20, 0.008, 0.006],
                "steel",
                0.002,
            )
    key = UsdLux.RectLight.Define(b.stage, "/World/Refined/KeyLight")
    key.AddTranslateOp().Set(Gf.Vec3d(0.5, -1.2, 2.6))
    key.AddRotateXOp().Set(25)
    key.CreateWidthAttr(1.8)
    key.CreateHeightAttr(1.1)
    key.CreateIntensityAttr(1400)
    key.CreateColorAttr(Gf.Vec3f(1, 0.88, 0.73))
