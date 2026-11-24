"""Import the bay shown on the LEFT of assets/scenes/02/preview.jpg (counter, sink, wall cabinets, bed).

The source OBJ is mirrored relative to the preview image: the bay kept by prepare_bedside_ward.py
(source X < -70 mm) is the preview's right bay. This script keeps source X >= -70 mm up to the
left wall, plus the shared floor, ceiling and window wall. Run with usd-core or Isaac Sim's Python.
"""

import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bedside_ward", REPO / "scripts/hospital/prepare_bedside_ward.py")
bedside_ward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bedside_ward)

LEFT_WALL_MM = 3500
ORIGIN = [0, 700, 0]  # z-up source mm: converted x = source X / 1000, bed stays at y -2.8..-0.6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=REPO.parent / "assets/scenes/02/model/02.obj"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "assets/hospital/ward_02_preview_left.usdc"
    )
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Missing supplied asset: {args.source}")
    records = bedside_ward.inventory(args.source)
    selected = {
        r["name"]
        for r in records
        if (-70 <= r["min"][0] and r["max"][0] <= LEFT_WALL_MM) or r["name"] in bedside_ward.SHARED
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "background_import", REPO / "scripts/lab/import_background_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.convert(
        args.source,
        args.output,
        0.001,
        object_names=selected,
        origin_zup_source_units=ORIGIN,
        source_encoding="gb18030",
    )
    manifest_path = args.output.with_suffix(".json")
    record = json.loads(manifest_path.read_text())
    record.update(
        scene_id="hospital_scene02_preview_left_v1",
        reference_image=str(args.source.parent.parent / "preview.jpg"),
        selection=f"-70 <= source X, max source X <= {LEFT_WALL_MM} mm (preview's left bay); shared floor, ceiling and window wall retained",
        object_count=len(selected),
        license="User-supplied pack; original rights retained, no redistribution claim",
    )
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
