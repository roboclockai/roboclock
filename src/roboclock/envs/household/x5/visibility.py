"""Head-camera visibility of each task's key objects and cues (RoboDojo rig spec, requirement 6).

Projects the eight bounding-box corners of every audited object of a built initial scene
(``initial_state.json``) and every operation point into the head camera at its spec pose
(``robodojo_rig.head_camera_world``) and checks that all of them lie at least ``MARGIN_PX``
inside the 640x480 image. By default it projects with the square-pixel model the renderer
produces (``robodojo_rig.intrinsics``, ``RENDERER_SQUARE_PIXELS``: 96.0 x 79.6 deg).
``conservative=True`` also checks the ``honoured`` model (fy from the vertical aperture, what
``get_intrinsics_matrix()`` reports, 71.0 deg vertical; ``square_pixels=False``) and keeps the
smaller margin. Occlusion by the arms is not modelled; rendered frames cover it.

By decision (2026-11-04), C1's staging mat is seen by the wrist cameras only: under the rendered
model the mat and the bowl standing on it sit about 13-19 px below the head image, while the
bowl's handle point is about 20 px inside it. ``WRIST_ONLY`` lists such items; their margins are
reported but do not fail the check.

Pure numpy; ``python -m roboclock.envs.household.x5.visibility <initial_state.json>...`` prints it.
"""

import itertools
import json
import sys

import numpy as np

from roboclock.envs import robodojo_rig as RD
from roboclock.envs.household.x5 import rig

MARGIN_PX = 10.0
# Fixtures larger than the view: their task-relevant parts are separate items (B1: the lever, the
# bowl and the water cues; A2: the paddle, the outlet and the glass). A2's "Ice" is the cube pool
# hidden in the bin (cubes are seen falling into the glass). Everything else in the object audit
# is a key object or cue.
NOT_KEY = {"A2": {"IceMachine", "Ice"}, "B1": {"Sink"}}
# Key items accepted as wrist-camera-only (head margin reported, not required).
WRIST_ONLY = {"C1": {"StagingMat", "MealBowl1"}}  # R3: meal 1 starts on the mat (seed 0)
# Operation points that are motion waypoints in free space, not objects (not checked).
WAYPOINTS = {"pour_pose_spout", "door_close_push"}


def models(conservative=False):
    """Pinhole models checked: ``square`` (rendered), plus ``honoured`` if conservative."""
    out = {"square": RD.intrinsics(RD.HEAD_CAMERA)}
    if conservative:
        out["honoured"] = RD.intrinsics(RD.HEAD_CAMERA, square_pixels=False)
    return out


def project(points, bases=None, conservative=False):
    """(model -> Nx2 pixel coordinates (u right, v down)), points in world metres."""
    camera = np.linalg.inv(RD.head_camera_world(*rig.rig_origin(bases)))
    p = (camera @ np.c_[np.asarray(points, float), np.ones(len(points))].T)[:3]
    if np.any(p[2] >= 0):
        raise ValueError("point behind the head camera")
    out = {}
    for name, k in models(conservative).items():
        u = k[0, 2] + k[0, 0] * p[0] / -p[2]
        v = k[1, 2] - k[1, 1] * p[1] / -p[2]
        out[name] = np.c_[u, v]
    return out


def margin(points, bases=None, conservative=False):
    """Smallest distance (px) of any point inside the image edge (worst over the models checked)."""
    width, height = RD.HEAD_CAMERA["resolution"]
    worst = np.inf
    for uv in project(points, bases, conservative).values():
        worst = min(worst, uv[:, 0].min(), width - uv[:, 0].max(), uv[:, 1].min(), height - uv[:, 1].max())
    return float(worst)


def _row(task, name, kind, m):
    row = dict(name=name, kind=kind, margin_px=round(m, 1), ok=m >= MARGIN_PX)
    if name in WRIST_ONLY.get(task, ()):
        row.update(ok=True, wrist_only=True)
    return row


def check(state, bases=None, conservative=False):
    """Rows (name, kind, margin_px, ok) for one ``initial_state.json`` dict."""
    task = state["task"]
    rows = []
    for obj in state["objects"]:
        if obj["name"] in NOT_KEY.get(task, ()):
            continue
        lo, hi = np.array(obj["bbox_min_m"]), np.array(obj["bbox_max_m"])
        corners = [np.where(mask, hi, lo) for mask in itertools.product((0, 1), repeat=3)]
        rows.append(_row(task, obj["name"], "object", margin(corners, bases, conservative)))
    for point in state.get("operation_points", []):
        if point["name"] in WAYPOINTS:
            continue
        rows.append(_row(task, point["name"], "operation_point", margin([point["xyz"]], bases, conservative)))
    return rows


def summary(state, bases=None, conservative=False):
    rows = check(state, bases, conservative)
    return dict(margin_required_px=MARGIN_PX, pinhole_models=list(models(conservative)), ok=all(r["ok"] for r in rows),
                worst=min(rows, key=lambda r: r["margin_px"]) if rows else None, items=rows)


if __name__ == "__main__":
    bad = 0
    for path in sys.argv[1:]:
        state = json.loads(open(path).read())
        for row in check(state):
            flag = "   (wrist cameras only)" if row.get("wrist_only") else "" if row["ok"] else "   <-- OUTSIDE"
            bad += not row["ok"]
            print(f"{state['task']} {row['kind']:15s} {row['name']:24s} {row['margin_px']:8.1f} px{flag}")
    sys.exit(1 if bad else 0)
