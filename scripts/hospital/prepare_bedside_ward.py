"""Import the supplied scene-02 left ward with its original materials and meshes.

Run with usd-core or Isaac Sim's Python. The source OBJ is Y-up millimeters.
Furniture left of the central partition and shared architecture are retained.
The visual pack supplies no task collision or scoring geometry.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SHARED = {
    "Rectangle008",
    "Rectangle2133464527",
    "Line2065098447",
    "Rectangle2133443549",
    "Rectangle2133464531",
    "Line2065098404",
    "Line2065098403",
}


def inventory(source):
    """Read bounds by referenced faces, including exporter vertices before object headers."""
    vertices, records, indices = [], [], set()
    name = "mesh"

    def flush():
        if indices:
            points = np.asarray([vertices[i] for i in indices])
            records.append(
                {"name": name, "min": points.min(0).tolist(), "max": points.max(0).tolist()}
            )

    with source.open(encoding="gb18030", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(tuple(map(float, line.split()[1:4])))
            elif line.startswith("o "):
                flush()
                name, indices = line.strip()[2:], set()
            elif line.startswith("f "):
                indices.update(int(v.split("/")[0]) - 1 for v in line.split()[1:])
    flush()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=REPO.parent / "assets/scenes/02/model/02.obj"
    )
    parser.add_argument("--output", type=Path, default=REPO / "assets/hospital/ward_02_left.usdc")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Missing supplied asset: {args.source}")
    records = inventory(args.source)
    selected = {
        r["name"]
        for r in records
        if (-4900 <= r["min"][0] and r["max"][0] < -70) or r["name"] in SHARED
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
        origin_zup_source_units=[-3000, 700, 0],
        source_encoding="gb18030",
    )
    manifest_path = args.output.with_suffix(".json")
    record = json.loads(manifest_path.read_text())
    record.update(
        scene_id="hospital_scene02_left_ward_v2",
        reference_image=str(args.source.parent.parent / "preview.jpg"),
        selection="max source X < -70 mm; shared floor, ceiling and window wall retained",
        object_count=len(selected),
        license="User-supplied pack; original rights retained, no redistribution claim",
    )
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
