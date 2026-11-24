"""Scaled practice pool table with an open physical collection pocket."""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind, pbr

from .geometry import rod
from .scenes import display, rigid, solid


def build(b):
    from isaacsim.core.api.materials import PhysicsMaterial
    from pxr import UsdGeom, UsdPhysics, UsdShade

    a = b.appearance
    # A stationary, supported precision-practice shot, not a complete pool match.
    b.ball_radius = 0.023
    b.pool_y = -0.12
    b.cue_home = np.array([0.34, -0.12, 0.794])
    b.cue_start = np.array([0.56, -0.12, 0.774])
    target_x = [0.88, 0.82, 0.95][b.config["seed"] % 3]
    target_x = 0.88 + (target_x - 0.88) * b.config["scene"].get("randomness_scale", 1.0)
    if b.config["seed"] >= 3:
        target_x += float(
            np.random.default_rng(b.config["seed"]).uniform(-0.008, 0.008)
        ) * b.config["scene"].get("randomness_scale", 1.0)
    b.target_start = np.array([target_x, -0.12, 0.774])
    b.pocket = np.array([1.20, -0.12, 0.75])
    pbr(a, "ivory", [0.91, 0.88, 0.76], roughness=0.22)
    pbr(a, "ball_red", [0.70, 0.03, 0.025], roughness=0.2)
    pbr(
        a,
        "pocket_leather",
        [0.085, 0.036, 0.016],
        roughness=0.55,
        texture="leather_white",
        uv=(2, 2),
    )
    mat = PhysicsMaterial(
        "/World/Materials/ClothContact",
        static_friction=0.18,
        dynamic_friction=0.12,
        restitution=0.65,
    )
    UsdGeom.Xform.Define(b.stage, "/World/PoolBed")
    b.collider("/World/PoolBed/Bed", "cube", [0.70, 0, 0.72], [0.94, 0.76, 0.06], [0.03, 0.3, 0.2])
    bind(b, "/World/PoolBed/Bed", a.material("pool_cloth"))
    UsdShade.MaterialBindingAPI.Apply(b.stage.GetPrimAtPath("/World/PoolBed/Bed")).Bind(
        mat.material, materialPurpose="physics"
    )
    UsdPhysics.RigidBodyAPI.Apply(
        b.stage.GetPrimAtPath("/World/PoolBed")
    ).CreateKinematicEnabledAttr(True)
    solid(b, "/World/Pool/Apron", [0.70, 0, 0.64], [1.10, 0.91, 0.14], "clean_wood", 0.025)
    for i, (x, y) in enumerate([(0.3, -0.34), (0.3, 0.34), (1.08, -0.34), (1.08, 0.34)]):
        solid(b, f"/World/Pool/Leg{i}", [x, y, 0.30], [0.08, 0.08, 0.60], "clean_wood", 0.012)
        solid(b, f"/World/Pool/Foot{i}", [x, y, 0.045], [0.105, 0.105, 0.06], "brass", 0.012)
    for i, y in enumerate([-0.405, 0.405]):
        solid(
            b,
            f"/World/Pool/Rail{i}",
            [0.70, y, 0.776],
            [1.10, 0.09, 0.052],
            "clean_wood",
            0.012,
            True,
        )
        rod(
            b,
            f"/World/Pool/Cushion{i}",
            [0.22, y - np.sign(y) * 0.033, 0.766],
            [1.18, y - np.sign(y) * 0.033, 0.766],
            0.016,
            "pool_cloth",
            True,
        )
        for j, x in enumerate([0.40, 0.65, 0.90]):
            b.collider(
                f"/World/Pool/Diamond{i}{j}",
                "sphere",
                [x, y, 0.804],
                [0.004],
                [0.9, 0.8, 0.6],
                False,
            )
    solid(b, "/World/Pool/EndLeft", [0.20, 0, 0.778], [0.08, 0.76, 0.06], "clean_wood", 0.012, True)
    for i, (y, length) in enumerate([(-0.295, 0.17), (0.165, 0.43)]):
        solid(
            b,
            f"/World/Pool/EndRight{i}",
            [1.20, y, 0.778],
            [0.065, length, 0.06],
            "clean_wood",
            0.012,
            True,
        )
    # Open pocket: no collision lid across the mouth. Ball falls into a supported bin.
    UsdGeom.Xform.Define(b.stage, "/World/Pocket")
    for name, center, size in [
        ("Floor", [1.24, -0.12, 0.61], [0.18, 0.14, 0.02]),
        ("Back", [1.33, -0.12, 0.69], [0.02, 0.14, 0.16]),
        ("SideA", [1.24, -0.19, 0.69], [0.18, 0.02, 0.16]),
        ("SideB", [1.24, -0.05, 0.69], [0.18, 0.02, 0.16]),
        ("Front", [1.15, -0.12, 0.675], [0.012, 0.14, 0.13]),
    ]:
        solid(b, "/World/Pocket/" + name, center, size, "pocket_leather", 0.008, True)
    UsdPhysics.RigidBodyAPI.Apply(
        b.stage.GetPrimAtPath("/World/Pocket")
    ).CreateKinematicEnabledAttr(True)
    for name, pos, material in [
        ("cue_ball", b.cue_start, "ivory"),
        ("target_ball", b.target_start, "ball_red"),
    ]:
        root = "/World/Objects/" + name
        UsdGeom.Xform.Define(b.stage, root)
        b.collider(root + "/Surface", "sphere", [0, 0, 0], [b.ball_radius], [1, 1, 1])
        bind(b, root + "/Surface", a.material(material))
        UsdShade.MaterialBindingAPI.Apply(b.stage.GetPrimAtPath(root + "/Surface")).Bind(
            mat.material, materialPurpose="physics"
        )
        rigid(b, name, pos, 0.055)
    root = "/World/Objects/cue"
    UsdGeom.Xform.Define(b.stage, root)
    rod(b, root + "/Shaft", [-0.13, 0, 0], [0.16, 0, 0], 0.009, "clean_wood", True)
    rod(b, root + "/Grip", [-0.065, 0, 0], [0.02, 0, 0], 0.012, "rubber", True)
    rod(b, root + "/Ferrule", [0.16, 0, 0], [0.171, 0, 0], 0.009, "ivory", True)
    rod(b, root + "/Tip", [0.171, 0, 0], [0.177, 0, 0], 0.009, "teal_lacquer", True)
    rigid(b, "cue", b.cue_home, 0.045)
    for i, x in enumerate([0.23, 0.43]):
        solid(
            b, f"/World/Pool/Rest{i}", [x, -0.12, 0.769], [0.018, 0.04, 0.036], "brass", 0.003, True
        )
    # A passive bridge ahead of the grip supports the cue without a joint constraint.
    solid(b, "/World/Pool/Bridge", [0.46, -0.12, 0.765], [0.018, 0.09, 0.03], "brass", 0.004, True)
    solid(b, "/World/Pool/RobotStand", [0, 0, 0.375], [0.34, 0.34, 0.75], "ink", 0.018)
    for i in range(4):
        rod(
            b,
            f"/World/Dressing/StoredCue{i}",
            [-0.95 + i * 0.09, 1.35, 0.22],
            [-0.95 + i * 0.09, 1.30, 1.60],
            0.009,
            "clean_wood",
        )
    solid(b, "/World/Dressing/CueRack", [-0.81, 1.33, 0.40], [0.52, 0.14, 0.055], "clean_wood")
    solid(b, "/World/Pool/Lamp", [0.7, 0.45, 2.15], [1.0, 0.22, 0.085], "ink", 0.025)
    solid(b, "/World/Pool/LampDiffuser", [0.7, 0.45, 2.105], [0.92, 0.18, 0.008], "led_warm", 0.004)
    for i, x in enumerate([0.35, 1.05]):
        rod(b, f"/World/Pool/Hanger{i}", [x, 0.45, 2.19], [x, 0.45, 3.1], 0.002, "steel")
    display(b, [0.98, 0.66, 1.15], 0.87)
    b.support_paths = ["/World/PoolBed", "/World/Pocket", "/World/Table"]
    b.view = ([2.1, -2.9, 2.55], [0.62, 0, 0.95])
