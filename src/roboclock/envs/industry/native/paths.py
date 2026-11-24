"""Asset path resolution for the native Industry sources.

The recorded sources name their imported assets by the absolute paths of the
machine they were generated on (``/opt/industry/...``). Those trees ship under
``assets/industry/runtime/``; this module maps one onto the other and touches no
task state. ``ROBOCLOCK_INDUSTRY_ASSET_ROOT`` overrides the bundle location.
"""

import json
import os
from pathlib import Path

PREFIX = "/opt/industry/"
REPOSITORY = Path(__file__).resolve().parents[5]


def asset_root() -> Path:
    return Path(os.environ.get("ROBOCLOCK_INDUSTRY_ASSET_ROOT", REPOSITORY / "assets/industry"))


def asset_path(original):
    if not original.startswith(PREFIX):
        raise ValueError(original)
    return str(asset_root() / "runtime" / original[len(PREFIX) :])


def local_output_root():
    """Directory that bounds standalone native-runner outputs (see runner.py)."""
    return os.environ.get("ROBOCLOCK_INDUSTRY_LOCAL_ROOT", str(REPOSITORY / "results/industry"))


def portable_json_loads(*args, **kwargs):
    def remap(value):
        if isinstance(value, str) and value.startswith(PREFIX):
            candidate = asset_path(value)
            return candidate if Path(candidate).exists() else value
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [remap(v) for v in value]
        return value

    return remap(json.loads(*args, **kwargs))


def robot_manifest() -> Path:
    return asset_root() / "runtime/arx-x5/robot-v1/prepared-v2/manifest.json"


# Environment bundles the two oldest scenes (41, 42) take as an explicit argument.
SCENE_ASSETS = {
    41: "arx-x5/environment-preview-v1/assets",
    42: "arx-x5/task42-environment-v1/assets",
}


def scene_assets(task: int) -> Path | None:
    relative = SCENE_ASSETS.get(int(task))
    return None if relative is None else asset_root() / "runtime" / relative
