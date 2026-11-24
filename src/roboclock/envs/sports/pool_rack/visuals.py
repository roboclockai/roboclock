"""Authored numbered phenolic balls over analytic sphere collision geometry."""

from pathlib import Path

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind, pbr


def texture(number):
    from PIL import Image, ImageDraw, ImageFont

    root = Path(__file__).parent / "assets"
    root.mkdir(exist_ok=True)
    path = root / f"ball_{number:02d}.png"
    if path.exists():
        return path
    palette = [
        (237, 185, 22),
        (27, 74, 171),
        (177, 25, 25),
        (90, 36, 127),
        (222, 91, 16),
        (24, 108, 65),
        (103, 32, 27),
        (18, 20, 23),
    ]
    color = palette[(number - 1) % 8] if number else (238, 235, 220)
    im = Image.new("RGB", (1024, 512), (238, 235, 220) if number > 8 else color)
    d = ImageDraw.Draw(im)
    if number > 8:
        d.rectangle([0, 172, 1024, 340], fill=color)
    if number:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 66)
        for x in (256, 768):
            d.ellipse([x - 60, 196, x + 60, 316], fill=(245, 241, 224))
            d.text((x, 256), str(number), font=font, anchor="mm", fill=(14, 17, 20))
    im.save(path)
    return path


def ball_visual(b, root, number, radius):
    from pxr import Sdf, UsdGeom

    material = pbr(b.appearance, f"numbered_ball_{number}", [1, 1, 1], roughness=0.17)
    shader = material.shaders_list[0]
    inp = shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
    inp.Set(Sdf.AssetPath(str(texture(number).resolve())))
    inp.GetAttr().SetMetadata("colorSpace", "sRGB")
    points, normals, uv, indices = [], [], [], []
    rows, columns = 32, 64
    for i in range(rows + 1):
        theta = i * np.pi / rows
        for j in range(columns + 1):
            phi = j * 2 * np.pi / columns
            v = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
            points.append((np.array(v) * radius).tolist())
            normals.append(v)
            uv.append([j / columns, 1 - i / rows])
    for i in range(rows):
        for j in range(columns):
            a = i * (columns + 1) + j
            indices.extend([a, a + columns + 1, a + columns + 2, a, a + columns + 2, a + 1])
    mesh = UsdGeom.Mesh.Define(b.stage, root + "/Visual")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        uv
    )
    bind(b, root + "/Visual", material)


def cantilever_mount(b):
    """Move only authored fixture columns outside the table; retain real X5 CAD."""
    from pxr import Gf, Usd, UsdGeom

    from roboclock.envs.sports._shared.workflow.scenes import solid

    root = b.robot_path + "Mount"
    for prim in list(Usd.PrimRange(b.stage.GetPrimAtPath(root))):
        name = prim.GetName()
        if name.startswith(("Column", "Foot", "AnchorBolt")):
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    old = np.asarray(op.Get(), float)
                    old[0] -= 0.39
                    op.Set(Gf.Vec3d(*old))
    for i, y in enumerate((-0.30, 0.30)):
        solid(
            b,
            f"/World/RackStand/Beam{i}",
            [-0.055, y, 0.947],
            [0.48, 0.075, 0.065],
            "steel",
            0.004,
            True,
        )


def ball_handling_tips(b):
    """Visible bolt-on slender tips; real finger-link colliders, no ball attachment."""
    from pxr import PhysxSchema, UsdPhysics

    from roboclock.envs.sports._shared.workflow.scenes import solid

    for side in ("left", "right"):
        b.rig.offsets[side] = np.array([0.205, 0, 0])
        b.rig.kin[side].offset = b.rig.offsets[side].copy()
        for finger, sign in zip(b.rig.fingers[side], (1, -1)):
            root = finger + "/BallHandlingTip"
            # Local native finger origins: X=86.57 mm, Y=+/-24.90 mm at zero.
            for name, x, y, size, material in (
                ("Mount", 0.150, 0.006, [0.014, 0.012, 0.014], "steel"),
                ("Stem", 0.173, 0.007, [0.046, 0.006, 0.006], "steel"),
                ("Shoulder", 0.197, 0.004, [0.008, 0.008, 0.008], "steel"),
                ("Pad", 0.202, 0.0015, [0.008, 0.003, 0.008], "rubber"),
            ):
                position = [x - 0.08657, sign * (y - 0.02490), 0.000244]
                solid(b, root + name, position, size, material, 0.0005, True)
                api = PhysxSchema.PhysxCollisionAPI.Apply(
                    b.stage.GetPrimAtPath(root + name + "Collider")
                )
                api.CreateContactOffsetAttr(0.0003)
            mass = UsdPhysics.MassAPI.Apply(b.stage.GetPrimAtPath(finger)).GetMassAttr()
            mass.Set(float(mass.Get() or 0.04) + 0.018)
