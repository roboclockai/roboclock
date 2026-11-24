"""Kinematic reach screen of every recorded operation point (URDF limits only; no collisions/payload).

    python scripts/household/check_x5_reach.py      # needs numpy + scipy, no simulator
"""
import json
import os
from pathlib import Path

from roboclock.envs.household.x5 import rig
from roboclock.envs.household.x5.kinematics import X5, screen

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("ROBOCLOCK_X5_INITIAL", REPO / "data/household/x5_initial"))
robot = X5()
report = {}
for d in sorted(p for p in OUT.iterdir() if (p / "initial_state.json").exists()):
    rig.use(d.name)  # the task's base layout
    pts = json.loads((d / "initial_state.json").read_text())["operation_points"]
    rows = screen(pts, robot)
    for r, p in zip(rows, pts):
        if not r["reachable"]:
            other = "Right" if p["side"] == "Left" else "Left"
            alt = screen([dict(p, side=other, approaches=sorted({*p["approaches"], "down", "forward", "forward_down"}))], robot)[0]
            r["other_arm"] = dict(side=other, reachable=alt["reachable"], approach=alt["approach"])
    report[d.name] = rows
    print(d.name, f"{sum(r['reachable'] for r in rows)}/{len(rows)}", flush=True)
(OUT / "REACH_CHECK.json").write_text(json.dumps(dict(
    method="bounded numerical IK on the X5 URDF chain (x5/kinematics.py); tool point = eef_link; 5 mm / 15 deg",
    excludes="collisions, payload/torque, gripper aperture, trajectories", tasks=report), indent=2) + "\n")
