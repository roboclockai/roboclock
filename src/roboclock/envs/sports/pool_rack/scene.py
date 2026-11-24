"""Full-size six-pocket club table, ball caddies and a thin physical rack template."""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind, panel, pbr
from roboclock.envs.sports.arena.geometry import ring
from roboclock.envs.sports._shared.workflow.geometry import rod
from roboclock.envs.sports._shared.workflow.scenes import display, rigid, solid

from .visuals import ball_visual, cantilever_mount


def supported(b, path):
    from pxr import UsdGeom, UsdPhysics

    prim = UsdGeom.Xform.Define(b.stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
    return prim


def cloth(b, material):
    """Triangulated slate top with actual openings at all six pocket mouths."""
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade
    from scipy.spatial import Delaunay

    c = b.config["rack"]
    length, half, height = c["table_length"], c["table_width"] / 2, c["bed_height"]
    pockets = np.array([[x, y] for x in (0, length / 2, length) for y in (-half, half)])
    xx, yy = np.meshgrid(np.linspace(0, length, 100), np.linspace(-half, half, 50))
    vertices = np.c_[xx.ravel(), yy.ravel()]
    vertices = vertices[
        np.min(np.linalg.norm(vertices[:, None] - pockets, axis=-1), axis=1) > 0.066
    ]
    for p in pockets:
        angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        edge = p + 0.066 * np.c_[np.cos(angles), np.sin(angles)]
        edge = edge[(edge[:, 0] >= 0) & (edge[:, 0] <= length) & (abs(edge[:, 1]) <= half)]
        vertices = np.vstack([vertices, edge])
    faces = Delaunay(vertices).simplices
    centers = vertices[faces].mean(axis=1)
    faces = faces[np.min(np.linalg.norm(centers[:, None] - pockets, axis=-1), axis=1) > 0.065]
    mesh = UsdGeom.Mesh.Define(b.stage, "/World/PoolBed/Cloth")
    mesh.CreatePointsAttr(np.c_[vertices, np.full(len(vertices), height)].tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        (vertices * 2).tolist()
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    api.CreateContactOffsetAttr(0.0005)
    api.CreateRestOffsetAttr(0.0)
    bind(b, str(mesh.GetPath()), b.appearance.material("pool_cloth"))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material.material, materialPurpose="physics"
    )
    return pockets


def template(b):
    """Thin perforated racking template; real shallow rims stabilize released balls."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    c = b.config["rack"]
    supported(b, "/World/RackTemplate")
    from scipy.spatial import Delaunay

    diameter = 2 * c["radius"]
    apex = np.asarray(c["slots"][0][:2])
    back = np.asarray(c["slots"][10][:2])
    outline = np.array(
        [
            [apex[0] + diameter, 0.0],
            [back[0] - c["radius"], -2 * diameter - np.sqrt(3) * c["radius"]],
            [back[0] - c["radius"], 2 * diameter + np.sqrt(3) * c["radius"]],
        ]
    )
    vertices = outline.tolist()
    for x, y, _ in c["slots"]:
        for j in range(48):
            angle = 2 * np.pi * j / 48
            vertices.append(
                [
                    x + c["template_hole_radius"] * np.cos(angle),
                    y + c["template_hole_radius"] * np.sin(angle),
                ]
            )
    vertices = np.asarray(vertices)
    faces = Delaunay(vertices).simplices
    centers = vertices[faces].mean(axis=1)
    holes = np.asarray(c["slots"])[:, :2]
    faces = faces[
        np.min(np.linalg.norm(centers[:, None] - holes, axis=-1), axis=1)
        > c["template_hole_radius"] * 0.98
    ]
    mesh = UsdGeom.Mesh.Define(b.stage, "/World/RackTemplate/Perforations")
    mesh.CreatePointsAttr(
        np.c_[vertices, np.full(len(vertices), c["bed_height"] + c["template_thickness"])].tolist()
    )
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    api.CreateContactOffsetAttr(0.0002)
    api.CreateRestOffsetAttr(0.0)
    bind(b, str(mesh.GetPath()), b.appearance.material("rack_template"))


def build(b):
    from isaacsim.core.api.materials import PhysicsMaterial
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdShade

    c, a = b.config["rack"], b.appearance
    from .visuals import ball_handling_tips

    ball_handling_tips(b)
    length, half = c["table_length"], c["table_width"] / 2
    pbr(a, "rack_template", [0.055, 0.07, 0.065], roughness=0.5)
    pbr(
        a,
        "pocket_leather",
        [0.055, 0.027, 0.014],
        roughness=0.58,
        texture="leather_white",
        uv=(2, 2),
    )
    contact = PhysicsMaterial(
        "/World/Materials/RackClothContact",
        static_friction=0.24,
        dynamic_friction=0.20,
        restitution=0.05,
    )
    supported(b, "/World/Pool")
    supported(b, "/World/RackStand")
    supported(b, "/World/PoolBed")
    pockets = cloth(b, contact)
    # Full slate cabinet, routed rails, leather drop pockets and brass sights.
    # Hollow cabinet: do not fill the actual pocket openings with a wood slab.
    for side, sign in enumerate((-1, 1)):
        solid(
            b,
            f"/World/Pool/ApronLong{side}",
            [length / 2, sign * (half + 0.11), 0.665],
            [length + 0.25, 0.035, 0.19],
            "clean_wood",
            0.012,
        )
        solid(
            b,
            f"/World/Pool/ApronEnd{side}",
            [length / 2 + sign * (length / 2 + 0.11), 0, 0.665],
            [0.035, 2 * half + 0.25, 0.19],
            "clean_wood",
            0.012,
        )
    for x in (0.22, length - 0.22):
        for y in (-half + 0.10, half - 0.10):
            tag = f"{int(x * 100)}_{int((y + 1) * 100)}"
            solid(b, "/World/Pool/Leg" + tag, [x, y, 0.30], [0.15, 0.15, 0.60], "clean_wood", 0.025)
            solid(b, "/World/Pool/Foot" + tag, [x, y, 0.055], [0.19, 0.19, 0.09], "brass", 0.014)
    for side, sign in enumerate((-1, 1)):
        for j, (lo, hi) in enumerate(
            ((0.075, length / 2 - 0.08), (length / 2 + 0.08, length - 0.075))
        ):
            solid(
                b,
                f"/World/Pool/Rail{side}_{j}",
                [(lo + hi) / 2, sign * (half + 0.055), 0.804],
                [hi - lo, 0.12, 0.075],
                "clean_wood",
                0.014,
                True,
            )
            rod(
                b,
                f"/World/Pool/Cushion{side}_{j}",
                [lo, sign * (half + 0.008), 0.795],
                [hi, sign * (half + 0.008), 0.795],
                0.018,
                "pool_cloth",
                True,
            )
        for j, x in enumerate(np.arange(1, 8) * length / 8):
            if j == 3:
                continue
            shape, _ = b.collider(
                f"/World/Pool/Sight{side}_{j}",
                "sphere",
                [x, sign * (half + 0.065), 0.843],
                [0.004],
                [0.85, 0.8, 0.65],
                False,
            )
            bind(b, str(shape.GetPath()), a.material("ivory"))
    for i, x in enumerate((-0.055, length + 0.055)):
        solid(
            b,
            f"/World/Pool/EndRail{i}",
            [x, 0, 0.804],
            [0.12, 2 * half - 0.15, 0.075],
            "clean_wood",
            0.014,
            True,
        )
        rod(
            b,
            f"/World/Pool/EndCushion{i}",
            [x + (0.047 if i == 0 else -0.047), -half + 0.075, 0.795],
            [x + (0.047 if i == 0 else -0.047), half - 0.075, 0.795],
            0.018,
            "pool_cloth",
            True,
        )
    for i, (x, y) in enumerate(pockets):
        root = f"/World/Pocket{i}"
        supported(b, root)
        solid(b, root + "/Floor", [x, y, 0.61], [0.14, 0.14, 0.025], "pocket_leather", 0.008, True)
        for j, (dx, dy, sx, sy) in enumerate(
            (
                (-0.073, 0, 0.014, 0.16),
                (0.073, 0, 0.014, 0.16),
                (0, -0.073, 0.16, 0.014),
                (0, 0.073, 0.16, 0.014),
            )
        ):
            solid(
                b,
                root + f"/Side{j}",
                [x + dx, y + dy, 0.695],
                [sx, sy, 0.17],
                "pocket_leather",
                0.006,
                True,
            )
        ring(b, root + "/Rim", [x, y, 0.800], 0.070, 0.009, "pocket_leather", False, 64)
    template(b)
    supported(b, "/World/BallCaddies")
    from roboclock.envs.sports.pool_rack.config import CADDY_CENTER

    cx, cy, cz = CADDY_CENTER
    for side, sign in (("Left", 1), ("Right", -1)):
        center = np.array([cx, sign * cy, cz])
        solid(
            b,
            "/World/BallCaddies/" + side + "Floor",
            center,
            [0.46, 0.23, 0.022],
            "clean_wood",
            0.009,
            True,
        )
        for j, (dx, dy, sx, sy) in enumerate(
            (
                (-0.231, 0, 0.012, 0.25),
                (0.231, 0, 0.012, 0.25),
                (0, -0.119, 0.47, 0.012),
                (0, 0.119, 0.47, 0.012),
            )
        ):
            solid(
                b,
                f"/World/BallCaddies/{side}Lip{j}",
                center + [dx, dy, 0.022],
                [sx, sy, 0.025],
                "clean_wood",
                0.005,
                True,
            )
        for j, x in enumerate((cx - 0.20, cx + 0.20)):
            rod(
                b,
                f"/World/CaddyStand/{side}{j}",
                [x, sign * 0.64, 0.03],
                [x, sign * 0.64, 0.838],
                0.025,
                "steel",
            )
        # Low cell dividers stabilize the random source positions while leaving grasp access.
        for j, x in enumerate((cx - 0.10, cx, cx + 0.10)):
            solid(
                b,
                f"/World/BallCaddies/{side}Divider{j}",
                [x, sign * cy, 0.863],
                [0.005, 0.22, 0.006],
                "rubber",
                0.001,
                True,
            )
        solid(
            b,
            f"/World/BallCaddies/{side}Cross",
            [cx, sign * cy, 0.863],
            [0.45, 0.005, 0.006],
            "rubber",
            0.001,
            True,
        )
    for name, position in {**c["sources"], "cue_ball": [1.62, 0, 0.808575]}.items():
        number = int(name[4:]) if name.startswith("ball") else 0
        root = "/World/Objects/" + name
        UsdGeom.Xform.Define(b.stage, root)
        shape, _ = b.collider(root + "/Collision", "sphere", [0, 0, 0], [c["radius"]], [1, 1, 1])
        UsdGeom.Imageable(shape).MakeInvisible()
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
            contact.material, materialPurpose="physics"
        )
        ball_visual(b, root, number, c["radius"])
        obj = rigid(b, name, position, c["mass"])
        # Number patches face the presentation hemisphere initially; physics determines subsequent spin.
        obj.set_local_pose(orientation=np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]))
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(obj.prim)
        api.CreateLinearDampingAttr(0.20)
        api.CreateAngularDampingAttr(1.5)
        api.CreateSolverPositionIterationCountAttr(32)
    cantilever_mount(b)
    # Reuse the pool-room wall materials and furniture, adding a proportionate canopy and cue rack.
    for i, x in enumerate((0.40, 1.02, 1.64)):
        solid(b, f"/World/Pool/Lamp{i}", [x, 0, 2.15], [0.54, 0.38, 0.065], "ink", 0.022)
        solid(b, f"/World/Pool/Diffuser{i}", [x, 0, 2.113], [0.49, 0.33, 0.009], "led_warm", 0.004)
        light = UsdLux.RectLight.Define(b.stage, f"/World/Pool/TaskLight{i}")
        light.AddTranslateOp().Set(Gf.Vec3d(x, 0, 2.10))
        light.CreateWidthAttr(0.55)
        light.CreateHeightAttr(0.4)
        light.CreateIntensityAttr(170)
    for j in range(5):
        x = -0.85 + j * 0.09
        rod(b, f"/World/Dressing/Cue{j}", [x, 1.38, 0.22], [x, 1.34, 1.65], 0.007, "clean_wood")
    solid(
        b, "/World/Dressing/CueRack", [-0.67, 1.36, 0.45], [0.60, 0.12, 0.05], "clean_wood", 0.008
    )
    b.stage.RemovePrim("/World/Studio/Brand")
    panel(
        b,
        a,
        "/World/Studio/Brand",
        [0.7, 1.67, 2.08],
        [2.15, 0.42],
        "THE POOL ROOM",
        "RACK / READY",
        style="title",
        accent=(220, 184, 117),
    )
    display(b, [0.62, 0.87, 1.25], 0.90)
    b.support_paths = [
        "/World/PoolBed",
        "/World/RackTemplate",
        "/World/BallCaddies",
        "/World/Pool",
        "/World/RackStand",
        *[f"/World/Pocket{i}" for i in range(6)],
        "/World/Table",
    ]
    b.view = ([1.80, -2.35, 2.65], [0.64, 0.02, 0.78])
