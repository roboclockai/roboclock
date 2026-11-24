"""Scene files for the Dual-X5 environment: initial scenes built in code, per-env work folders.

The initial scene is built with ``x5/build.py`` (kitchen, island, X5 visual pose and the task's
``<id>_x5/scene.py``), the same code that produced the recorded scenes. Built three folders below
a directory directly under the repository (as the recordings' ``data/household/x5_initial/<ID>``
were), the file is byte-identical to the recorded ``initial_scene_sha256``; elsewhere only its
relative asset paths differ. Requires ``pxr`` (Isaac Sim's, or a standalone OpenUSD).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

HOUSEHOLD = Path(__file__).resolve().parents[1]
REPO = HOUSEHOLD.parents[3]
DEMOS = REPO / "data/household/x5_demos"


def make_workdir(scene_dir=None):
    """A fresh per-environment folder; the default sits at the recordings' scene depth."""
    if scene_dir is not None:
        base = Path(scene_dir)
        prefix = "x5-scenes-"
    elif (REPO / "pyproject.toml").exists():
        base = REPO / "results/household"  # git-ignored
        prefix = ".x5-scenes-"
    else:
        base = Path(tempfile.gettempdir())
        prefix = "roboclock-household-x5-"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{prefix}{os.getpid()}-", dir=base))


def build_initial_scene(task_id, root):
    """Write ``root/<task_id>/initial_scene.usda`` and ``initial_state.json``; return the scene."""
    from roboclock.envs.household.x5 import build
    from roboclock.envs.household.x5 import scene_lib as L

    saved = L.OUTPUT, L.SCENE_DIR
    try:
        L.OUTPUT = Path(root)  # build() writes L.OUTPUT/<task>/
        build.build(task_id)
    finally:
        # build() leaves SCENE_DIR at its folder; the physics overlay (another folder) must
        # write absolute asset paths, as in the recorded episodes.
        L.OUTPUT, L.SCENE_DIR = saved
    return Path(root) / task_id / "initial_scene.usda"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_initial_sha(task_id):
    """``initial_scene_sha256`` of the accepted recording, or None if it is not available."""
    path = DEMOS / task_id / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("initial_scene_sha256")
