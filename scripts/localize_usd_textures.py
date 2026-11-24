"""Copy absolute texture references of USD layers next to them and make the paths relative.

Each absolute asset path in a layer is resolved to a file (as written, or by file name
under --search directories), copied to <layer dir>/textures/<layer stem>/, and replaced by
./textures/<layer stem>/<name>, the layout scripts/lab/import_background_assets.py writes.
Prints a JSON report with the SHA-256 of every copied texture and of each rewritten layer.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from pxr import Sdf


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(reference, search):
    path = Path(reference)
    if path.is_file():
        return path
    matches = [m for root in search for m in Path(root).rglob(path.name) if m.is_file()]
    return matches[0] if len({sha256(m) for m in matches}) == 1 else None


def localize(layer_path, search, dry_run):
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    folder = Path("textures") / layer_path.stem
    report = dict(layer=str(layer_path), sha256_before=sha256(layer_path), textures=[], missing=[])

    def rewrite(value):
        if not value.path.startswith("/"):
            return value
        source = locate(value.path, search)
        if source is None:
            report["missing"].append(value.path)
            return value
        target = folder / source.name
        if not dry_run:
            (layer_path.parent / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, layer_path.parent / target)
        report["textures"].append(dict(original=value.path, path=f"./{target.as_posix()}", sha256=sha256(source)))
        return Sdf.AssetPath(f"./{target.as_posix()}")

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if not isinstance(spec, Sdf.AttributeSpec) or spec.default is None:
            return
        if spec.typeName == Sdf.ValueTypeNames.Asset:
            spec.default = rewrite(spec.default)
        elif spec.typeName == Sdf.ValueTypeNames.AssetArray:
            spec.default = [rewrite(v) for v in spec.default]

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    if not dry_run and report["textures"]:
        layer.Save()
    report["sha256_after"] = sha256(layer_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layers", nargs="+", type=Path)
    parser.add_argument("--search", nargs="*", default=[], help="directories to find moved textures in")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reports = [localize(p, args.search, args.dry_run) for p in args.layers]
    print(json.dumps(reports, indent=2))
    return 1 if any(r["missing"] for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
