"""Daylit chess club, percussion studio and garden arcade visual dressing.

Reuses CC0 furniture, plants and PBR maps. Architecture and fixed background
fixtures are authored visual proxies, outside the manipulation workspace.
"""

import numpy as np


def metric_wood_uv(stage, root, period=0.65):
    """Replace atlas UVs with face-varying, metre-scaled planar wood coordinates.

    Coplanar triangles share a world-space projection, avoiding the unrelated
    atlas islands that made replacement wood textures look like patchwork.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = cache.GetLocalToWorldTransform(prim)
        points = np.array(
            [transform.Transform(Gf.Vec3d(*map(float, p))) for p in mesh.GetPointsAttr().Get()]
        )
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        uv, cursor = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            polygon = points[indices[cursor : cursor + count]]
            normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            axis = int(np.argmax(np.abs(normal)))
            axes = ((1, 2), (0, 2), (0, 1))[axis]
            uv.extend(polygon[:, axes] / period)
            cursor += count
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        st.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, p)) for p in uv]))
        st.BlockIndices()


def dress_room(b, a, theme):
    from pxr import Gf, UsdGeom, UsdLux

    from roboclock.envs.sports.arcade.refinement import bind, cabinet, panel, pbr
    from roboclock.envs.sports.arcade.scenes import imported

    def hide(path):
        prim = b.stage.GetPrimAtPath(path)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()

    # Remove inherited blockout decorations and its floor; retain the physics.
    hide("/World/defaultGroundPlane")
    for prim in list(b.stage.Traverse()):
        path = str(prim.GetPath())
        if any(
            path.startswith("/World/Room/" + prefix)
            for prefix in (
                "Sign",
                "Panel",
                "WindowSlat",
                "Cabinet",
                "Screen",
                "GameTitle",
                "Marquee",
                "ControlShelf",
                "Neon",
                "Edge",
                "Trim",
                "LightHousing",
            )
        ):
            UsdGeom.Imageable(prim).MakeInvisible()
    hide("/World/Room/Chair")
    pbr(a, "day_wall", [0.78, 0.82, 0.77] if theme == "taiko" else [0.88, 0.83, 0.72], 0.88)
    pbr(a, "day_trim", [0.92, 0.91, 0.84], 0.48)
    pbr(a, "sage", [0.25, 0.48, 0.40], 0.72)
    pbr(a, "ochre", [0.85, 0.51, 0.16], 0.6)
    pbr(a, "terracotta", [0.69, 0.28, 0.17], 0.7)
    pbr(a, "linen", [0.67, 0.63, 0.48], 0.92, texture="leather_white", uv=(3, 3))
    pbr(a, "day_wood", [1.0, 0.93, 0.80], 0.52, texture="wood_floor")
    pbr(a, "day_glazing", [0.69, 0.84, 0.91], 0.30, emission=0.7)
    bind(b, "/World/Room/BackWall", a.material("day_wall"))
    bind(b, "/World/Room/SideWall", a.material("day_wall"))
    bind(
        b,
        "/World/Room/Floor",
        pbr(a, "day_floor", [0.93, 0.89, 0.77], 0.6, texture="wood_floor", uv=(4, 4)),
    )
    # Low painted wainscoting, framed panels, skirting and picture rail.
    for j in range(10):
        x = -2.0 + j * 0.52
        a.rounded(
            f"/World/Daylight/Wainscot{j}",
            [x, 1.79, 0.46],
            [0.49, 0.025, 0.80],
            "sage" if theme != "taiko" else "day_wood",
            0.008,
        )
    for z in (0.07, 0.89):
        a.rounded(
            f"/World/Daylight/Rail{int(z * 100)}",
            [0.4, 1.75, z],
            [5.5, 0.055, 0.055],
            "day_trim",
            0.008,
        )
    # Frosted clerestory glazing with real frame depth; no photographic backdrop.
    for i in range(4):
        x = -1.25 + i * 0.64
        a.rounded(
            f"/World/Daylight/WindowFrame{i}",
            [x, 1.742, 1.93],
            [0.62, 0.075, 1.09],
            "day_trim",
            0.014,
        )
        a.rounded(
            f"/World/Daylight/Glass{i}", [x, 1.694, 1.93], [0.54, 0.013, 0.99], "day_glazing", 0.002
        )
        a.rounded(
            f"/World/Daylight/Mullion{i}", [x, 1.682, 1.91], [0.54, 0.021, 0.025], "day_trim", 0.003
        )
    a.rounded(
        "/World/Daylight/WindowSill", [-0.29, 1.65, 1.37], [2.66, 0.22, 0.055], "day_trim", 0.014
    )
    imported(
        b.stage,
        "potted_plant_02",
        "/World/Daylight/Plant",
        [1.84, 1.30, 0],
        [0.60, 0.57, 1.18],
        yaw=25,
    )
    imported(
        b.stage,
        "potted_plant_02",
        "/World/Daylight/SillPlant",
        [-1.35, 1.62, 1.40],
        [0.24, 0.23, 0.40],
        yaw=80,
    )

    if theme == "chess":
        hide("/World/Room/Chair0")
        hide("/World/Room/SideTable")
        for path in ("/World/Room/PlayingTable", "/World/Room/SideTable"):
            metric_wood_uv(b.stage, path)
            bind(b, path, a.material("day_wood"))
        panel(
            b,
            a,
            "/World/Daylight/ClubArt",
            [1.48, 1.68, 1.91],
            [0.81, 0.59],
            "CHESS CLUB",
            "THINK / MOVE / CLOCK",
            "chess",
            glow=0,
        )
        # Books and a low sideboard give the furniture a coherent club setting.
        a.rounded(
            "/World/Daylight/Bookcase", [-0.25, 1.47, 0.42], [1.22, 0.34, 0.74], "day_wood", 0.016
        )
        a.rounded(
            "/World/Daylight/BookcaseInset", [-0.25, 1.292, 0.47], [1.09, 0.017, 0.48], "ink", 0.004
        )
        for j in range(18):
            a.rounded(
                f"/World/Daylight/Book{j}",
                [-0.75 + j * 0.057, 1.27, 0.39],
                [0.04, 0.16, 0.25 + (j % 4) * 0.025],
                ("cream", "sage", "terracotta", "ink")[j % 4],
                0.003,
            )
    elif theme == "taiko":
        bind(b, "/World/Room/Worktop", a.material("day_wood"))
        # Sound-absorbing panels, timber diffusers and a storage bench.
        for j in range(4):
            x = -1.32 + j * 0.58
            a.rounded(
                f"/World/Daylight/AcousticFrame{j}",
                [x, 1.64, 1.02],
                [0.51, 0.075, 0.46],
                "day_wood",
                0.02,
            )
            a.rounded(
                f"/World/Daylight/AcousticFabric{j}",
                [x, 1.594, 1.02],
                [0.46, 0.025, 0.40],
                "linen",
                0.018,
            )
        for j in range(13):
            a.rounded(
                f"/World/Daylight/Diffuser{j}",
                [1.16 + j * 0.035, 1.70, 1.93],
                [0.018, 0.04 + (j % 3) * 0.015, 0.91],
                "day_wood",
                0.003,
            )
        panel(
            b,
            a,
            "/World/Daylight/StudioSign",
            [-0.27, 1.59, 1.31],
            [1.20, 0.085],
            "PERCUSSION STUDIO",
            glow=0,
        )
        a.rounded(
            "/World/Daylight/Bench", [0.13, 1.31, 0.31], [1.42, 0.39, 0.54], "day_wood", 0.025
        )
        a.rounded(
            "/World/Daylight/BenchCushion", [0.13, 1.29, 0.62], [1.42, 0.40, 0.09], "linen", 0.035
        )
        for j in range(4):
            a.rounded(
                f"/World/Daylight/BenchDoor{j}",
                [-0.40 + j * 0.35, 1.106, 0.32],
                [0.32, 0.016, 0.43],
                "sage",
                0.009,
            )
        for x in (0.06, 0.81):
            a.rounded(
                f"/World/Daylight/Stand{int(x * 100)}",
                [x, 0, 0.36],
                [0.08, 0.72, 0.70],
                "ink",
                0.012,
            )
    else:
        bind(b, "/World/Room/Worktop", a.material("cream"))
        # One neighboring cabinet leaves negative space around the actual nine-hole game.
        cabinet(b, a, "/World/Daylight/Arcade", -0.82, 1.28, 2)
        panel(
            b,
            a,
            "/World/Daylight/GardenSign",
            [0.80, 1.68, 1.26],
            [1.25, 0.25],
            "GARDEN ARCADE",
            accent=(127, 211, 150),
            glow=0,
        )
        a.rounded(
            "/World/Daylight/SeatBase", [0.78, 1.31, 0.26], [1.18, 0.42, 0.44], "day_wood", 0.025
        )
        a.rounded("/World/Daylight/Seat", [0.78, 1.28, 0.52], [1.18, 0.46, 0.14], "ochre", 0.06)
        for j in range(3):
            a.rounded(
                f"/World/Daylight/Cushion{j}",
                [0.40 + j * 0.38, 1.43, 0.75],
                [0.35, 0.13, 0.36],
                ("sage", "terracotta", "ochre")[j],
                0.05,
            )
        a.rounded("/World/Daylight/MachineBase", [0.54, 0, 0.37], [0.58, 0.67, 0.71], "sage", 0.05)
        a.rounded(
            "/World/Daylight/MachineDoor", [0.54, -0.344, 0.38], [0.38, 0.02, 0.40], "cream", 0.025
        )
        a.rounded(
            "/World/Daylight/CoinSlot", [0.54, -0.36, 0.49], [0.09, 0.012, 0.012], "ink", 0.003
        )
        a.rounded(
            "/World/Daylight/KickPlate", [0.54, -0.354, 0.12], [0.45, 0.02, 0.12], "steel", 0.008
        )
    # Broad daylight and a front fill preserve tool-contact detail without black walls.
    for name, position, rotation, size, power, color in (
        ("Daylight", [-0.35, 1.45, 2.27], [-72, 0, 0], [2.3, 0.85], 2300, [0.90, 0.95, 1.0]),
        ("Softbox", [0.70, -1.4, 2.7], [25, 0, 0], [2.3, 1.5], 1800, [1.0, 0.95, 0.87]),
        ("WallWash", [1.8, -0.1, 2.1], [65, 0, -25], [1.4, 1.1], 1000, [1.0, 0.98, 0.91]),
    ):
        light = UsdLux.RectLight.Define(b.stage, f"/World/Daylight/{name}")
        light.AddTranslateOp().Set(Gf.Vec3d(*position))
        light.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        light.CreateWidthAttr(size[0])
        light.CreateHeightAttr(size[1])
        light.CreateIntensityAttr(power)
        light.CreateColorAttr(Gf.Vec3f(*color))
