"""Complete compact training ring; existing robot and pad physics stay at world Z."""

from roboclock.envs.sports.arcade.refinement import panel, pbr
from roboclock.envs.sports.arcade.scenes import vertical_label
from roboclock.envs.sports.arena.geometry import rod


def build_ring(b):
    from pxr import Gf, UsdGeom, UsdLux

    a = b.appearance
    # The canvas is the existing floor datum, with the surrounding room below it.
    for prim in list(b.stage.Traverse()):
        if str(prim.GetPath()).startswith(
            (
                "/World/Refined/Arcade",
                "/World/Room/Edge",
                "/World/Room/Cabinet",
                "/World/Room/Screen",
                "/World/Room/Marquee",
                "/World/Room/ControlShelf",
                "/World/Room/GameTitle",
                "/World/Room/Sign",
                "/World/Room/Chair",
                "/World/Refined/Plant",
            )
        ):
            UsdGeom.Imageable(prim).MakeInvisible()
    floor = UsdGeom.Xformable(b.stage.GetPrimAtPath("/World/Room/Floor"))
    floor.AddTranslateOp(opSuffix="ringside").Set(Gf.Vec3d(0, 0, -0.28))
    a.rounded(
        "/World/Gym/RightWall",
        [2.85, 0.05, 1.32],
        [0.08, 3.70, 3.20],
        "architectural_wall",
        0.008,
    )
    a.rounded("/World/Gym/RightSkirting", [2.80, 0.05, -0.18], [0.04, 3.70, 0.20], "ink")
    pbr(a, "ring_canvas", [0.12, 0.19, 0.27], 0.96)
    pbr(a, "ring_red", [0.58, 0.055, 0.045], 0.56, texture="leather_white", uv=(2, 2))
    pbr(a, "ring_blue", [0.025, 0.16, 0.35], 0.56, texture="leather_white", uv=(2, 2))
    a.rounded("/World/Gym/Platform", [0.28, 0, -0.145], [2.95, 2.6, 0.28], "ink", 0.04)
    a.rounded("/World/Gym/Canvas", [0.28, 0, -0.004], [2.9, 2.55, 0.008], "ring_canvas", 0.012)
    # Physical support is at Z=0; no task object is moved to fake elevation.
    b.collider(
        "/World/Gym/Support", "cube", [0.28, 0, -0.14], [2.9, 2.55, 0.28], [0.12, 0.19, 0.27]
    )
    UsdGeom.Imageable(b.stage.GetPrimAtPath("/World/Gym/Support")).MakeInvisible()
    a.rounded("/World/Equipment/RobotPlinth", [0, 0, 0.125], [0.48, 0.55, 0.25], "ink", 0.035)
    corners = [(-1.02, -1.12), (1.58, -1.12), (1.58, 1.12), (-1.02, 1.12)]
    for i, (x, y) in enumerate(corners):
        material = "ring_red" if i == 0 else "ring_blue" if i == 2 else "cream"
        rod(b, f"/World/Gym/Post{i}", [x, y, -0.2], [x, y, 1.38], 0.045, "steel", True)
        a.rounded(f"/World/Gym/CornerPad{i}", [x, y, 0.96], [0.11, 0.11, 0.72], material, 0.035)
        a.rounded(f"/World/Gym/PostFoot{i}", [x, y, 0.016], [0.19, 0.19, 0.028], "steel", 0.008)
        qx, qy = corners[(i + 1) % 4]
        for j, z in enumerate([0.43, 0.69, 0.95, 1.21]):
            rope_material = "ring_red" if j == 3 else "ring_blue" if j == 0 else "cream"
            rod(b, f"/World/Gym/Rope{i}_{j}", [x, y, z], [qx, qy, z], 0.015, rope_material, True)
        for t in [0.30, 0.70]:
            xx, yy = x + t * (qx - x), y + t * (qy - y)
            rod(
                b,
                f"/World/Gym/Spacer{i}_{int(t * 100)}",
                [xx, yy, 0.43],
                [xx, yy, 1.21],
                0.009,
                "ink",
            )
    for i in range(3):
        a.rounded(
            f"/World/Gym/Step{i}",
            [1.88 + i * 0.17, -0.78, -0.28 + (3 - i) * 0.04],
            [0.34, 0.66, (3 - i) * 0.08],
            "steel",
            0.009,
        )
    # In-world canvas branding and a legible front apron.
    b.label("/World/Gym/CanvasLogo", "ROUND / REACT", [0.3, -0.69, 0.002], [1.10, 0.17], False)
    vertical_label(b, "/World/Gym/ApronText", "THE BOXING CLUB", [0.28, -1.306, -0.13], [1.6, 0.14])
    panel(
        b,
        a,
        "/World/Gym/Sign",
        [0.25, 1.67, 2.10],
        [2.15, 0.42],
        "THE BOXING CLUB",
        "TRAINING RING / REACTION & CONTROL",
        accent=(238, 153, 92),
    )
    a.rounded("/World/Gym/Bench", [1.95, 0.70, 0.12], [0.43, 1.0, 0.08], "clean_wood", 0.015)
    for y in [0.35, 1.05]:
        a.rounded(f"/World/Gym/BenchLeg{int(y * 100)}", [1.95, y, -0.10], [0.32, 0.05, 0.38], "ink")
    # Towel and bottle on the ringside bench, clear of the manipulation volume.
    a.rounded("/World/Gym/Towel", [1.95, 0.75, 0.18], [0.30, 0.31, 0.05], "cream", 0.012)
    rod(b, "/World/Gym/Bottle", [1.95, 0.40, 0.16], [1.95, 0.40, 0.36], 0.035, "steel")
    rod(b, "/World/Gym/BottleCap", [1.95, 0.40, 0.36], [1.95, 0.40, 0.38], 0.026, "ink")
    light = UsdLux.RectLight.Define(b.stage, "/World/Lights/RingSoftbox")
    light.AddTranslateOp().Set(Gf.Vec3d(0.35, 0, 2.75))
    light.CreateWidthAttr(2.0)
    light.CreateHeightAttr(1.6)
    light.CreateIntensityAttr(750)
    for name, color in [
        ("glove_red", [0.65, 0.045, 0.025]),
        ("glove_blue", [0.02, 0.19, 0.31]),
        ("pad_leather", [0.06, 0.07, 0.08]),
    ]:
        pbr(a, name, color, texture="leather_white", uv=(3, 3))

    # A broad ringside key light makes the dark manufacturer R1 shell readable.
    light = UsdLux.RectLight.Define(b.stage, "/World/Lights/RingsideKey")
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.7, -1.9, 2.35), Gf.Vec3d(0.15, 0, 1.0), Gf.Vec3d(0, 0, 1))
    light.AddTransformOp().Set(view.GetInverse())
    light.CreateWidthAttr(1.4)
    light.CreateHeightAttr(1.0)
    light.CreateIntensityAttr(650)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.85))
