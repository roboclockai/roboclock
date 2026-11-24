"""Key objects and cues of every sports task inside the RoboDojo head camera.

The RoboDojo head camera (``roboclock.envs.robodojo_rig``: rig frame (0.04, 0, 0.543) m,
60 deg down, 96 deg horizontal FOV) sees nothing more than about 20 deg below the
horizontal, so the cue displays that the earlier presentation views read (tall panels
beside or behind the workspace) are moved low and turned toward the rig. All moved
displays are visual geometry (labels, cases, LEDs); physics is unchanged.

- ``DISPLAYS``: rigid moves of display groups (``apply`` at build, before any render).
- ``KEY_OBJECTS``: the prims each task's policy must see; ``check`` projects their
  world bounding boxes into the head image and returns the pixel margin of each
  (``MARGIN_PX`` required, corners at least 10 px inside the 640x480 image).

Projection: the renderer keeps square pixels (measured: the rendered image matches
``fy = fx``; ``get_intrinsics_matrix()`` reports ``fy`` from the 14.266 mm vertical
aperture instead, which the RTX renderer does not apply), so the effective vertical
field of view is 79.6 deg. ``effective_intrinsics`` is that matrix
(``robodojo_rig.intrinsics()``).
"""

import math

import numpy as np

from roboclock.envs import robodojo_rig as spec

MARGIN_PX = 10.0


def effective_intrinsics():
    return spec.intrinsics(spec.HEAD_CAMERA)


def head_pose(config):
    from roboclock.envs.sports.rig_cameras import head_world

    return head_world(config)


def project(points, config):
    """World points -> (u, v) pixels of the head image (u right, v down) and depth."""
    pose = head_pose(config)
    p = np.atleast_2d(np.asarray(points, float))
    local = (np.linalg.inv(pose) @ np.c_[p, np.ones(len(p))].T)[:3].T
    depth = -local[:, 2]  # USD camera looks along -Z
    k = effective_intrinsics()
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 2] + k[0, 0] * local[:, 0] / depth
        v = k[1, 2] - k[1, 1] * local[:, 1] / depth
    return np.stack([u, v], 1), depth


def box_corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def margin(points, config):
    """Smallest distance (px) of the projected points to the image border; -inf if behind."""
    uv, depth = project(points, config)
    if np.any(depth <= 1e-3):
        return -math.inf
    width, height = spec.HEAD_CAMERA["resolution"]
    return float(min(uv[:, 0].min(), width - uv[:, 0].max(), uv[:, 1].min(), height - uv[:, 1].max()))


# -- display moves ------------------------------------------------------------------
def facing_rotation(center, eye, normal=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0)):
    """Rotation turning a panel that faces ``normal`` (text up = +z) to face ``eye``.

    The text's up direction becomes ``up`` projected onto the new panel plane; with the
    head camera's image-up vector the text rows run parallel to the image rows.
    """
    n0 = np.asarray(normal, float)
    u0 = np.array([0.0, 0.0, 1.0])
    r0 = np.cross(u0, n0)
    n = np.asarray(eye, float) - np.asarray(center, float)
    n /= np.linalg.norm(n)
    u = np.asarray(up, float) - (np.asarray(up, float) @ n) * n
    u /= np.linalg.norm(u)
    r = np.cross(u, n)
    return np.column_stack([r, u, n]) @ np.column_stack([r0, u0, n0]).T


def move_matrix(move, config):
    """4x4 (column convention) rigid move + uniform scale of one display group."""
    pivot = np.asarray(move["pivot"], float)
    center = np.asarray(move["center"], float)
    if "face" in move:
        head = head_pose(config)
        # Face the head camera, text rows parallel to its image rows.
        rotation = facing_rotation(center, head[:3, 3], move.get("normal", (0.0, -1.0, 0.0)), head[:3, 1])
    else:
        rotation = np.eye(3)
    matrix = np.eye(4)
    matrix[:3, :3] = move.get("scale", 1.0) * rotation
    matrix[:3, 3] = center - matrix[:3, :3] @ pivot
    return matrix


# Each move: prims (group parents: their children follow, including labels re-created
# at run time), the panel's original centre (pivot), its new centre, the direction it
# faces ("head" = the head camera) and a uniform scale. ``hide`` lists stands and posts
# that would otherwise float (prefix patterns: rods have a ``<name>Visual`` mesh
# sibling); ``stand`` adds a visual post under the moved panel.
DISPLAYS = {
    "A1": [
        # Turn/clock panel: from above the table's left edge to a low panel under
        # BLACK's mount, between its columns, facing the rig.
        dict(
            prims=["/World/Duel"],
            pivot="bounds:/World/Duel/TurnPanel/Face",
            center=[0.755, 0.0, 0.80],
            face="head",
            scale=0.5,
            stand=dict(top=0.77, floor=0.726),
        )
    ],
    "B5": [
        # Note screen: from above the drums' left side to the table's far left corner.
        dict(
            prims=[
                "/World/Game",
                "/World/Refined/NoteLane*",
                "/World/Refined/LaneName*",
                "/World/Refined/TaikoHeader",
                "/World/Refined/ScreenTrim*",
                "/World/Refined/GameSpeaker*",
            ],
            pivot="bounds:/World/Game/ScreenHousing",
            center=[0.86, 0.36, 0.92],
            face="head",
            scale=0.75,
            hide=["/World/Refined/ScreenStand*"],
            stand=dict(top=0.80, floor=0.75),
        )
    ],
    "A2": [
        # Shot clock: from above the hoop to the free left side of the table.
        dict(
            prims=["/World/ShotClock"],
            pivot=[0.70, 0.553, 1.445],
            center=[0.65, 0.40, 0.80],
            face="head",
            scale=0.6,
            stand=dict(top=0.785, floor=0.75),
        )
    ],
    "B2": [],  # coach screen: arena/coach_display.py places the group itself
    "B3": [
        # Status board: from high on the left wall to a low stand beyond the bench end.
        dict(
            prims=["/World/Display"],
            pivot="bounds:/World/Display/Case",
            center=[1.25, 0.0, 0.93],
            face="head",
            scale=0.55,
            stand=dict(top=0.86, floor=0.05, at=[1.30, 0.30]),
        )
    ],
    "B4": [
        # Status board: from its high post to a low stand beyond the rink's far left end.
        dict(
            prims=["/World/Display"],
            pivot="bounds:/World/Display/Case",
            center=[1.05, 0.38, 0.90],
            face="head",
            scale=0.45,
            hide=["/World/Display/Post*", "/World/Display/Base*"],
            stand=dict(top=0.85, floor=0.007),
        )
    ],
    "C1": [
        # Status board: from the wall to a low stand on the bed between rack and cue ball.
        dict(
            prims=["/World/Display"],
            pivot="bounds:/World/Display/Case",
            center=[1.25, 0.0, 0.90],
            face="head",
            scale=0.55,
            stand=dict(top=0.835, floor=0.78),
        )
    ],
    "C2": [
        # Pattern screen: from above the table's left edge to a low stand beyond the
        # table's far edge, between the arms' reach and the wall.
        dict(
            prims=[
                "/World/Cups/ScreenHousing",
                "/World/Cups/Card",
                "/World/Cups/Icon*",
                "/World/Cups/Side*",
                "/World/CupScreen",
            ],
            pivot=[0.63, 0.46, 1.28],
            center=[0.98, 0.0, 0.86],
            face="head",
            scale=0.6,
            hide=["/World/Cups/ScreenStand*"],
            stand=dict(top=0.78, floor=0.0),
        )
    ],
    "C3": [
        # Status board: from its high stands to a low stand beyond the hopper, left.
        dict(
            prims=["/World/Display"],
            pivot="bounds:/World/Display/Case",
            center=[0.95, 0.30, 0.95],
            face="head",
            scale=0.55,
            hide=["/World/Display/Stand*"],
            stand=dict(top=0.885, floor=0.014),
        )
    ],
}


def _pivot(stage, move):
    pivot = move["pivot"]
    if isinstance(pivot, str) and pivot.startswith("bounds:"):
        prim = stage.GetPrimAtPath(pivot[len("bounds:"):])
        if not prim:
            raise ValueError(f"No pivot prim {pivot}")
        lo, hi = bounds(stage, [prim])
        return (lo + hi) / 2
    return np.asarray(pivot, float)


def _set_matrix(prim, matrix):
    """Apply ``matrix`` before (outside) the prim's own transform ops."""
    from pxr import Gf, UsdGeom

    if not prim.IsA(UsdGeom.Xformable):
        prim.SetTypeName("Xform")
    xf = UsdGeom.Xformable(prim)
    order = xf.GetOrderedXformOps()
    op = xf.AddTransformOp(opSuffix="headView")
    op.Set(Gf.Matrix4d(*np.asarray(matrix, float).T.reshape(-1).tolist()))
    xf.SetXformOpOrder([op, *order])


def _visual_only(prim):
    from pxr import Usd, UsdPhysics

    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"{p.GetPath()} has physics; display moves are visual only")


def apply(b, task_id):
    """Move the task's display groups into the head view once per stage; returns the moves."""
    from pxr import UsdGeom

    stage = b.stage
    if getattr(b, "head_view_stage", None) is stage:
        return b.head_view_moves
    applied = []
    for move in DISPLAYS.get(task_id, []):
        move = dict(move, pivot=_pivot(stage, move))
        matrix = move_matrix(move, b.config)
        prims = [p for pattern in move["prims"] for p in _resolve(stage, pattern)]
        if not prims:
            raise ValueError(f"{task_id}: no display prims {move['prims']}")
        for prim in prims:
            _visual_only(prim)
            _set_matrix(prim, matrix)
        for pattern in move.get("hide", []):
            for prim in _resolve(stage, pattern):
                UsdGeom.Imageable(prim).MakeInvisible()
        if "stand" in move:
            _stand(b, task_id, move)
        applied.append(
            dict(
                prims=[str(p.GetPath()) for p in prims],
                pivot=np.round(move["pivot"], 4).tolist(),
                center=list(map(float, move["center"])),
                scale=move.get("scale", 1.0),
            )
        )
    b.head_view_stage, b.head_view_moves = stage, applied
    return applied


def _stand(b, task_id, move):
    """Visual post from the floor/table to the moved panel's lower edge."""
    from pxr import Gf, UsdGeom

    stage = b.stage
    path = "/World/HeadViewStand/" + task_id
    if stage.GetPrimAtPath(path):
        return
    center = np.asarray(move["center"], float)
    top, floor = move["stand"]["top"], move["stand"]["floor"]
    x, y = move["stand"].get("at", (center[0] + 0.02, center[1]))
    rod = UsdGeom.Cylinder.Define(stage, path)
    rod.CreateRadiusAttr(0.012)
    rod.CreateHeightAttr(float(top - floor))
    rod.CreateAxisAttr("Z")
    rod.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float((top + floor) / 2)))
    rod.CreateDisplayColorAttr([(0.28, 0.30, 0.33)])
    foot = UsdGeom.Cube.Define(stage, path + "Foot")
    foot.CreateSizeAttr(1.0)
    foot.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(floor) + 0.006))
    foot.AddScaleOp().Set(Gf.Vec3f(0.14, 0.10, 0.012))
    foot.CreateDisplayColorAttr([(0.10, 0.11, 0.12)])


# -- key objects ----------------------------------------------------------------------
# Prims (or prefixes ending in "*") whose world bounds must project inside the head
# image with MARGIN_PX at reset. Dynamic objects are checked at their reset pose.
KEY_OBJECTS = {
    "A1": {
        "board": ["/World/Board"],
        "pieces": ["/World/Pieces/*"],
        "clock and buttons": ["/World/ClockHousing", "/World/Buttons/robot", "/World/Buttons/opponent"],
        "capture tray": ["/World/CaptureTray"],
        "turn panel (clock cue)": ["/World/Duel"],
    },
    "B5": {
        "drums": ["/World/Controls/pad_0", "/World/Controls/pad_1"],
        "sticks": ["/World/Objects/mallet", "/World/Objects/mallet_left"],
        "pad cues": ["/World/PadLED/*", "/World/PadScreens/*"],
        "note screen": ["/World/Game/ScreenHousing", "/World/Game/Timeline", "/World/Game/Instruction"],
    },
    "A2": {
        "ball": ["/World/Objects/ball"],
        "hoop": ["/World/Basket/Mount", "/World/Basket/MarkV0", "/World/Basket/MarkH0"],
        "return tray": ["/World/Basket/ReturnTray"],
        "shot clock": ["/World/ShotClock"],
    },
    "B1": {
        "moles": ["/World/Controls/*", "/World/Deck/*"],
        "mallets": ["/World/Objects/mallet", "/World/Objects/mallet_left"],
    },
    "B2": {
        "pads": ["/World/Controls/pad_left", "/World/Controls/pad_right"],
        "coach screen": ["/World/CoachDisplay"],
    },
    "B3": {
        "bar": ["/World/Objects/bar"],
        "rack hooks": ["/World/RackLeft/Seat", "/World/RackLeft/FrontLip",
                       "/World/RackRight/Seat", "/World/RackRight/FrontLip"],
        "lifter": ["/World/HumanTorso/Chest"],
        "display": ["/World/Display"],
    },
    "B4": {
        "rink": ["/World/Rink/Bed"],
        "puck": ["/World/Objects/puck"],
        "pushers": ["/World/Objects/left_pusher", "/World/Objects/right_pusher"],
        "goals": ["/World/Rink/Goal*"],
        "stop/go signal": ["/World/Signal/Housing", "/World/Signal/Red", "/World/Signal/Green"],
        "status board": ["/World/Display/Case", "/World/Display/Line0", "/World/Display/Line1"],
    },
    "C1": {
        "object balls": ["/World/Objects/ball*"],
        "cue ball": ["/World/Objects/cue_ball"],
        "rack template": ["/World/RackTemplate"],
        "ball caddies": ["/World/BallCaddies"],
        "display": ["/World/Display"],
    },
    "C2": {
        "cups": ["/World/Objects/left_*", "/World/Objects/right_*"],
        "bell": ["/World/Controls/bell"],
        "target marks": ["/World/Cups/TargetMark*"],
        "pattern screen": ["/World/Cups/ScreenHousing", "/World/Cups/Icon*", "/World/CupScreen"],
    },
    "C3": {
        "balls": ["/World/Objects/ball*"],
        "supply racks": ["/World/SupplyLeft/Deck", "/World/SupplyRight/Deck"],
        "hopper and gates": ["/World/Hopper", "/World/Controls/*"],
        "display": ["/World/Display"],
    },
}

# Partially visible key objects that cannot be moved without changing the task, with
# the reason (reported in the result, never silently relaxed).
EXCEPTIONS = {
    "B3": {
        "bar": (
            "The 1.44 m Olympic bar rests 0.54 m below and 0.21 m ahead of the head camera "
            "(rig z 1.08 = bar height), so the head image spans only about +/-0.60 m of it: "
            "the shaft between the hooks, both hands and both rack hooks are inside; the "
            "collars, sleeve ends (up to 53 px) and the 0.26 m plates (up to 202 px) are "
            "cut at the left and right image borders. Bringing them in needs "
            "hooks and plates moved about 0.15 m inward or the rig raised about 0.2 m; both "
            "change the catch reach the catch deadline was calibrated on."
        ),
    },
}


def _resolve(stage, pattern):
    from pxr import Usd

    if not pattern.endswith("*"):
        prim = stage.GetPrimAtPath(pattern)
        return [prim] if prim else []
    parent, prefix = pattern[:-1].rsplit("/", 1)
    root = stage.GetPrimAtPath(parent)
    if not root:
        return []
    return [p for p in root.GetChildren() if p.GetName().startswith(prefix)]


def bounds(stage, prims):
    """World AABB of the visible geometry (Gprims) under ``prims``; hidden parts excluded."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for root in prims:
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue
            lo = np.minimum(lo, box.GetMin())
            hi = np.maximum(hi, box.GetMax())
    return lo, hi


def check(b, task_id, config=None):
    """Pixel margin of ``b``'s key objects (the worst prim of each); raises nothing."""
    stage, config = b.stage, config or b.config
    report = {}
    for name, patterns in KEY_OBJECTS[task_id].items():
        prims = [p for pattern in patterns for p in _resolve(stage, pattern)]
        if not prims:
            report[name] = dict(error="no prim", patterns=patterns)
            continue
        worst = None
        for prim in prims:
            lo, hi = bounds(stage, [prim])
            if not np.all(np.isfinite(lo)):
                continue
            m = margin(box_corners(lo, hi), config)
            if worst is None or m < worst[0]:
                worst = (m, str(prim.GetPath()), lo, hi)
        if worst is None:
            report[name] = dict(error="empty bounds", patterns=patterns)
            continue
        report[name] = dict(
            margin_px=round(worst[0], 1),
            worst_prim=worst[1],
            prims=len(prims),
            lo=np.round(worst[2], 3).tolist(),
            hi=np.round(worst[3], 3).tolist(),
            inside=bool(worst[0] >= MARGIN_PX),
        )
        if name in EXCEPTIONS.get(task_id, {}):
            report[name]["exception"] = EXCEPTIONS[task_id][name]
    return report
