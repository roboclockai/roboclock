"""Render overview + agent-view stills for the Dual-X5 initial scenes (one Isaac process per task).

    python scripts/household/render_x5_initial.py [A1 B3 ...]
Uses ISAAC_PYTHON (default: the Isaac Sim 6.0.1 venv). Writes overview.png / agent_view.png
and the render record into data/household/x5_initial/<ID>/. Capture never advances the timeline.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("ROBOCLOCK_X5_INITIAL", REPO / "data/household/x5_initial"))
STILLS = REPO / "src/roboclock/envs/household/x5/render_stills.py"
PY = os.environ.get("ISAAC_PYTHON", "/opt/household/runtime_601/venv/bin/python")
tasks = sys.argv[1:] or ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2", "C3"]
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
for t in tasks:
    d = OUT / t
    start = time.monotonic()
    cmd = [PY, str(STILLS), "--scene", str(d / "initial_scene.usda"),
           "--shot", f"/World/Camera={d}/overview.png", "--shot", f"/World/AgentCamera={d}/agent_view.png"]
    with (d / "render.log").open("w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    for line in (d / "render.log").read_text().splitlines():
        if line.startswith("RENDER_RESULT "):
            result = json.loads(line[len("RENDER_RESULT "):])
            for shot in result["shots"]:
                shot["image"] = Path(shot["image"]).name
            result["scene"] = "initial_scene.usda"
            info = json.loads((d / "initial_state.json").read_text())
            info["render"] = result
            (d / "initial_state.json").write_text(json.dumps(info, indent=2) + "\n")
    print("DONE", t, "exit", code, f"{time.monotonic() - start:.1f}s", flush=True)
    if code:
        raise SystemExit(code)
