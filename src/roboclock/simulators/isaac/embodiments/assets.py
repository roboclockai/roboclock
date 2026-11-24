"""Locate the converted Dual ARX X5 asset (manifest in git, USD from the sports archive)."""

import json
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name("assets.json").read_text())
# Converted manufacturer X5A pair: manifest in git, USD from the sports Drive archive.
DUAL_ARX_X5 = Path(__file__).resolve().parents[3] / "envs/sports/common/assets/dual_arx_x5"
LEGACY_DUAL_ARX_X5 = Path("results/arx-official/assets")


def asset_path(model):
    if model == "dual_arx_x5":
        for root in (DUAL_ARX_X5, LEGACY_DUAL_ARX_X5):
            manifest = root / "manifest.json"
            if not manifest.exists():
                continue
            entry = Path(json.loads(manifest.read_text())["usd"])
            entry = entry if entry.is_absolute() else root / entry
            if not entry.exists():
                raise FileNotFoundError(
                    f"{entry} is missing: extract the sports assets archive from Google Drive "
                    "or run scripts/sports/prepare_arx_x5_official.py in the Isaac environment"
                )
            return entry.resolve()
        raise FileNotFoundError("Run scripts/sports/prepare_arx_x5_official.py in the Isaac environment first")
    raise ValueError(f"Unsupported robot {model!r}: roboclock uses only dual_arx_x5")
