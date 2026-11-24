"""Daylit lab context: office workstations, seating, windows and neutral tile floor."""


def add_room_details(scene, asset_root):
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

    a, stage = scene.a, scene.stage
    floor = a.material("ClinicalFloor", color=[0.57, 0.60, 0.62], roughness=0.72)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/defaultGroundPlane")):
        if prim.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(floor.material)
    grout = a.material("TileGrout", color=[0.43, 0.46, 0.48], roughness=0.85)
    for i in range(-14, 15):
        a.rounded(
            f"/World/Interior/FloorLineX{i + 14}",
            [i * 0.60, -2.0, 0.0007],
            [0.0015, 16.0, 0.001],
            grout,
            0.0002,
        )
        a.rounded(
            f"/World/Interior/FloorLineY{i + 14}",
            [0.0, i * 0.60, 0.0007],
            [16.0, 0.0015, 0.001],
            grout,
            0.0002,
        )
    # Original pack 5 furniture replaces the former primitive office proxies.
    # Normalize each mesh root independently so proportions remain unchanged.
    cx, cy = -1.25, -1.90
    for name, position, scale in [
        ("marble_slab", [cx, cy, 0.75], [0.3913, 0.48125, 0.18]),
        ("office_computer", [cx, cy - 0.015, 0.786], [1.0, 1.0, 1.0]),
        ("office_chair", [cx + 0.10, cy + 0.72, 0.0], [0.83, 0.83, 0.83]),
    ]:
        root = UsdGeom.Xform.Define(stage, "/World/Office/" + name)
        root.GetPrim().GetReferences().AddReference(str(asset_root / (name + ".usdc")))
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        root.AddScaleOp().Set(Gf.Vec3f(*scale))
        if name == "office_computer":
            root.AddRotateZOp().Set(180.0)
        elif name == "office_chair":
            root.AddRotateZOp().Set(-25.0)
    # Two slim storage pedestals leave an open knee space below the stone slab.
    for i, dx in enumerate([-0.68, 0.68]):
        a.cube(f"/World/Office/Pedestal{i}", [cx + dx, cy, 0.385], [0.34, 0.67, 0.70], "paper")
        a.cube(f"/World/Office/Plinth{i}", [cx + dx, cy, 0.055], [0.31, 0.61, 0.08], "rubber")
        for j in range(3):
            a.cube(
                f"/World/Office/Drawer{i}_{j}",
                [cx + dx, cy + 0.344, 0.22 + j * 0.18],
                [0.322, 0.018, 0.163],
                "paper",
            )
            a.cube(
                f"/World/Office/Pull{i}_{j}",
                [cx + dx, cy + 0.36, 0.26 + j * 0.18],
                [0.12, 0.018, 0.009],
                "steel",
            )
    mountain_backplate(stage, asset_root / "mountain/alps_field.jpg")
    ceiling = a.material("Ceiling", color=[0.22, 0.24, 0.26], roughness=0.85)
    a.cube("/World/Interior/Ceiling", [0.25, -0.5, 3.2], [7.5, 6.2, 0.06], ceiling)
    # Full-height window openings face a real mountain HDR panorama.
    a.cube("/World/Interior/LowerWall", [0.25, -3.6, 0.45], [7.5, 0.08, 0.90], "paper")
    a.cube("/World/Interior/UpperWall", [0.25, -3.6, 2.96], [7.5, 0.08, 0.50], "paper")
    for index, x in enumerate([-3.30, 3.80]):
        a.cube(f"/World/Interior/WallPier{index}", [x, -3.6, 1.8], [0.40, 0.08, 2.0], "paper")
    for i in range(7):
        x = -3.1 + i * 1.05
        a.cube(f"/World/Interior/Mullion{i}", [x, -3.53, 1.80], [0.026, 0.06, 1.80], "steel")
    for i, z in enumerate([0.91, 2.70]):
        a.cube(f"/World/Interior/Crossbar{i}", [0.05, -3.53, z], [6.35, 0.06, 0.028], "steel")
    a.cube("/World/Interior/WindowSill", [0.05, -3.48, 0.90], [6.45, 0.20, 0.04], "paper")
    # A quiet blank wall bounds the office side; no extra furniture is added.
    a.cube("/World/Interior/SideWall", [-2.65, -0.75, 1.50], [0.08, 5.6, 3.0], "paper")
    light = UsdLux.RectLight.Define(stage, "/World/Lighting/Daylight")
    light.CreateWidthAttr(5.8)
    light.CreateHeightAttr(1.4)
    light.CreateIntensityAttr(650.0)
    light.CreateEnableColorTemperatureAttr(True)
    light.CreateColorTemperatureAttr(6300.0)
    xf = UsdGeom.Xformable(light)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, -3.30, 1.9))
    xf.AddRotateXOp().Set(90.0)


def mountain_backplate(stage, texture):
    """Distant photographic Alps panorama, projected across all window openings."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root = "/World/Exterior/Mountain"
    material = UsdShade.Material.Define(stage, root + "Material")
    shader = UsdShade.Shader.Define(stage, root + "Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    tex = UsdShade.Shader.Define(stage, root + "Material/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
    reader = UsdShade.Shader.Define(stage, root + "Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    mesh = UsdGeom.Mesh.Define(stage, root)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(-7, -6, -0.2), Gf.Vec3f(7, -6, -0.2), Gf.Vec3f(7, -6, 4.6), Gf.Vec3f(-7, -6, 4.6)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([Gf.Vec2f(0.22, 0.50), Gf.Vec2f(0.84, 0.50), Gf.Vec2f(0.84, 0.86), Gf.Vec2f(0.22, 0.86)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
