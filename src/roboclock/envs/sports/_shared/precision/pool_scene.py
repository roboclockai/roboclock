"""Continuous clearance table using the established V10 materials and equipment."""

import numpy as np

from roboclock.envs.sports.arcade.refinement import bind, pbr
from roboclock.envs.sports._shared.workflow.geometry import rod
from roboclock.envs.sports._shared.workflow.scenes import display, rigid, solid


def build(b):
    from isaacsim.core.api.materials import PhysicsMaterial
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    from .pool_layout import PoolLayout

    layout = b.pool_layout = PoolLayout.from_config(b.config)
    a = b.appearance
    # A continuous clearance drill, not a complete tournament pool match.
    b.ball_radius = 0.023
    g = b.cue_grip_x = b.config.get("pool", {}).get("cue_grip_x", 0.)
    b.pool_y = -0.12
    b.cue_home = layout.cue_home
    b.cue_start = (layout.point([.56, -.12, .774]) if layout.scale == 1.
                   else np.array([layout.center_x - .02, layout.center_y - .12 * layout.scale, .774]))
    if "initial_white" in b.config.get("pool", {}):
        b.cue_start = np.array(b.config["pool"]["initial_white"], dtype=float)
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
    cushion_material = PhysicsMaterial(
        "/World/Materials/CushionContact", static_friction=.25,
        dynamic_friction=.18, restitution=.85,
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
        for path in [f"/World/Pool/Cushion{i}", f"/World/Pool/Rail{i}Collider"]:
            UsdShade.MaterialBindingAPI.Apply(b.stage.GetPrimAtPath(path)).Bind(
                cushion_material.material, materialPurpose="physics"
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
    # Mouths centered at -0.12 and +0.22, with a common visible collection depth.
    b.pockets = [np.array([x, y]) for x in [1.25, 0.15] for y in [-0.12, 0.22]]
    for i, (low, high) in enumerate(layout.end_segments):
        solid(b, f"/World/Pool/EndSegment{i}", [1.20, (low + high) / 2, 0.778],
              [0.065, high - low, 0.06], "clean_wood", 0.008, True)
    for i, (low, high) in enumerate(layout.end_segments):
        solid(b, f"/World/Pool/LeftSegment{i}", [0.20, (low + high) / 2, 0.778],
              [0.065, high - low, 0.06], "clean_wood", 0.008, True)
    h = layout.pocket_half_width
    for i, (x, y) in enumerate(b.pockets):
        sign = 1 if x > 0.7 else -1
        root = f"/World/Pocket{i}"
        UsdGeom.Xform.Define(b.stage, root)
        for name, center, size in [
            ("Floor", [x, y, 0.59], [0.21, 2 * h, 0.02]),
            ("Back", [x + sign * 0.105, y, 0.68], [0.02, 2 * h, 0.18]),
            ("SideA", [x, y - h, 0.68], [0.21, 0.02, 0.18]),
            ("SideB", [x, y + h, 0.68], [0.21, 0.02, 0.18]),
            ("Front", [x - sign * 0.10, y, 0.665], [0.012, 2 * h, 0.13]),
        ]:
            solid(b, root + "/" + name, center, size, "pocket_leather", 0.006, True)
        UsdPhysics.RigidBodyAPI.Apply(b.stage.GetPrimAtPath(root)).CreateKinematicEnabledAttr(True)
    root = "/World/Objects/cue_ball"
    UsdGeom.Xform.Define(b.stage, root)
    b.collider(root + "/Surface", "sphere", [0, 0, 0], [b.ball_radius], [1, 1, 1])
    bind(b, root + "/Surface", a.material("ivory"))
    UsdShade.MaterialBindingAPI.Apply(b.stage.GetPrimAtPath(root + "/Surface")).Bind(mat.material, materialPurpose="physics")
    rigid(b, "cue_ball", b.cue_start, 0.055)
    b.pool_initial_ball_positions = {"cue_ball": b.cue_start.copy()}
    pbr(b.appearance, "ball_gold", [0.94, 0.57, 0.035], roughness=0.19)
    pbr(b.appearance, "ball_blue", [0.04, 0.20, 0.78], roughness=0.19)
    configured_targets = b.config.get("pool", {}).get("initial_targets")
    b.pool_targets = [f"target{i + 1}" for i in range(len(configured_targets) if configured_targets is not None else 3)]
    rng = np.random.default_rng(b.config["seed"])
    variation = b.config["scene"].get("randomness_scale", 1.0)
    for name, xy, material in zip(b.pool_targets,
            [[0.78, -0.12], [1.025, -0.12], [.82 if layout.scale != 1. else 1.065, 0.10]],
            ["ball_red", "ball_gold", "ball_blue"]):
        xy = np.array(xy) + rng.uniform(-0.002, 0.002, 2) * variation
        root = "/World/Objects/" + name
        UsdGeom.Xform.Define(b.stage, root)
        b.collider(root + "/Surface", "sphere", [0, 0, 0], [b.ball_radius], [1, 1, 1])
        bind(b, root + "/Surface", b.appearance.material(material))
        UsdShade.MaterialBindingAPI.Apply(b.stage.GetPrimAtPath(root + "/Surface")).Bind(
            UsdShade.Material(b.stage.GetPrimAtPath("/World/Materials/ClothContact")), materialPurpose="physics")
        b.pool_initial_ball_positions[name] = (layout.point([*xy, .774]) if configured_targets is None
            else np.array(configured_targets[b.pool_targets.index(name)], dtype=float))
        rigid(b, name, b.pool_initial_ball_positions[name], .055)
    # Empirical cloth rolling resistance, fixed across seeds and difficulty.
    # This is passive PhysX damping, not scripted stopping or velocity edits.
    for name in ["cue_ball", *b.pool_targets]:
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(b.objects[name].prim)
        api.CreateAngularDampingAttr(1.5)
        api.CreateLinearDampingAttr(0.2)
    root = "/World/Objects/cue"
    UsdGeom.Xform.Define(b.stage, root)
    rod(b, root + "/Shaft", [-0.13, 0, 0], [0.16, 0, 0], 0.009, "clean_wood", True)
    # A visible raised saddle gives the robot a stable grip above the cue shaft,
    # keeping its fingers clear of the rail during level strokes.
    solid(b, root + "/GripFoot", [g, 0, 0], [0.06, 0.08, 0.024], "brass", 0.003, True)
    rod(b, root + "/GripStem", [g, 0, 0], [g, 0, 0.055], 0.010, "brass", True)
    solid(b, root + "/GripBlock", [g, 0, 0.055], [0.055, 0.030, 0.035], "rubber", 0.003, True)
    # A lower retaining shoulder resists slip and leaves the palm approach open.
    for i, z in enumerate([0.027]):
        solid(b, root + f"/GripShoulder{i}", [g, 0, z], [0.075, 0.060, 0.010], "brass", 0.002, True)
    grip_material = PhysicsMaterial("/World/Materials/CueGrip", static_friction=2.0, dynamic_friction=1.5, restitution=0.0)
    UsdShade.MaterialBindingAPI.Apply(b.stage.GetPrimAtPath(root + "/GripBlockCollider")).Bind(grip_material.material, materialPurpose="physics")
    rod(b, root + "/Ferrule", [0.16, 0, 0], [0.171, 0, 0], 0.009, "ivory", True)
    if b.config.get("pool", {}).get("rounded_tip", False):
        b.collider(root + "/Tip", "sphere", [.171, 0, 0], [.006], [.1, .5, .5])
        bind(b, root + "/Tip", a.material("teal_lacquer"))
    else:
        rod(b, root + "/Tip", [.171, 0, 0], [.177, 0, 0], .009, "teal_lacquer", True)
    rigid(b, "cue", b.cue_home, 0.045)
    PhysxSchema.PhysxRigidBodyAPI.Apply(b.objects["cue"].prim).CreateSolverPositionIterationCountAttr(32)
    PhysxSchema.PhysxRigidBodyAPI.Apply(b.objects["cue"].prim).CreateSolverVelocityIterationCountAttr(16)
    # The flat tool foot rests on an open platform, so vertical withdrawal is clear.
    solid(b, "/World/Pool/GripDock", layout.inverse_point([b.cue_home[0] + g, b.cue_home[1], .766]),
          [.10 / layout.scale, (.08 if layout.scale < 1. else .10) / layout.scale, .032], "ink", .003, True)
    solid(b, "/World/Pool/RobotStand", [0.65, -0.55, 0.35], [0.26, 0.22, 0.70], "ink", 0.012)
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
    display(b, [0.65, 0.66, 1.32], 0.87)
    b.support_paths = ["/World/PoolBed", *[f"/World/Pocket{i}" for i in range(4)], "/World/Table"]
    b.view = ([2.4, -0.5, 2.7], [0.68, 0.10, 0.85])

    if layout != PoolLayout():
        layout.apply_fixture_transform(b.stage)
        b.pockets = [layout.point(p) for p in b.pockets]
        b.stage.RemovePrim("/World/Pool/RobotStand")

    if b.config.get("pool", {}).get("dual_cue_exchange", False):
        b.cue_exchange_home = np.array([layout.center_x - .18 - g, layout.center_y + .04, .794])
        root = "/World/CueRest"
        UsdGeom.Xform.Define(b.stage, root)
        solid(b, root + "/Pad", [b.cue_exchange_home[0] + g, b.cue_exchange_home[1], .766], [.10, .10, .032], "rubber", .004, True)
        UsdPhysics.RigidBodyAPI.Apply(b.stage.GetPrimAtPath(root)).CreateKinematicEnabledAttr(True)
        b.support_paths.append(root)

    if layout.scale != 1.:
        rests = [(b.cue_home + [g, 0, -.028], np.array([.10, .08, .032]))]
        if hasattr(b, "cue_exchange_home"):
            rests.append((b.cue_exchange_home + [g, 0, -.028], np.array([.10, .10, .032])))
        for name, position in b.pool_initial_ball_positions.items():
            for center, size in rests:
                nearest = np.clip(position, center - size / 2, center + size / 2)
                if np.linalg.norm(position - nearest) <= b.ball_radius + .002:
                    raise ValueError(f"Initial {name} overlaps a cue rest")
