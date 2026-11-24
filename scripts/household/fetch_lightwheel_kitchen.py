"""Download the pinned Lightwheel kitchen and unpack the files the Dual-X5 household tasks load.

The kitchen (CC BY-NC 4.0, LightwheelAI/Lightwheel_Kitchen) stays out of git: three of its
files exceed GitHub's 100 MB limit. This script downloads the pinned archive, checks it and
every unpacked file against assets/household/lightwheel_kitchen.json, rebuilds
KitchenRoom_A1.usdc byte for byte, and links the kitchen where the household code expects it.
Run it with a Python that has pxr (Isaac Sim's, or usd-core):

    python scripts/household/fetch_lightwheel_kitchen.py          # files the X5 tasks load
    python scripts/household/fetch_lightwheel_kitchen.py --all    # whole kitchen (pre-X5 demos)
"""

import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

# The recorded KitchenRoom_A1.usdc was written as USDC 0.9.0; set before pxr is imported.
os.environ["USD_WRITE_NEW_USDC_FILES_AS_VERSION"] = "0.9.0"

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets/household/lightwheel_kitchen.json"
DEST = REPO / "assets/household/lightwheel_kitchen"
LINK = REPO / "src/roboclock/envs/household/assets/x5/lightwheel_kitchen/Collected_KitchenRoom"
PREFIX = "Collected_KitchenRoom/"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(manifest, archive):
    if archive.is_file() and sha256(archive) == manifest["archive_sha256"]:
        return
    partial = archive.with_suffix(".partial")
    print(f"Downloading {manifest['url']}")
    urllib.request.urlretrieve(manifest["url"], partial)
    if sha256(partial) != manifest["archive_sha256"]:
        partial.unlink()
        raise ValueError("Archive checksum mismatch; the upstream file changed")
    partial.replace(archive)


def derive(kitchen, spec):
    """Export KitchenRoom.usd, dropping the two missing metricsAssembler sublayers."""
    from pxr import Sdf

    out = kitchen / "KitchenRoom_A1.usdc"
    Sdf.Layer.FindOrOpen(str(kitchen / spec["source"])).Export(str(out))
    adapted = Sdf.Layer.FindOrOpen(str(out))
    adapted.subLayerPaths = [p for p in adapted.subLayerPaths if (kitchen / p).exists()]
    adapted.Save()
    if sha256(out) != spec["sha256"]:
        raise ValueError(f"{out.name} differs from the recorded layer; check the pxr version")


def link(kitchen):
    LINK.parent.mkdir(parents=True, exist_ok=True)
    if LINK.is_symlink():
        LINK.unlink()
    elif LINK.exists():
        raise FileExistsError(f"{LINK} is a real directory; move it away first")
    LINK.symlink_to(os.path.relpath(kitchen, LINK.parent), target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="unpack the whole kitchen")
    parser.add_argument("--archive", type=Path, help="an already downloaded Lightwheel_Kitchen.zip")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    [(derived_name, spec)] = manifest["derived"].items()
    kitchen = DEST / "Collected_KitchenRoom"
    kitchen.mkdir(parents=True, exist_ok=True)
    archive = args.archive or DEST / "Lightwheel_Kitchen.zip"
    download(manifest, archive)
    with zipfile.ZipFile(archive) as bundle:
        names = [n[len(PREFIX):] for n in bundle.namelist() if n.startswith(PREFIX) and not n.endswith("/")]
        wanted = names if args.all else [*manifest["files"], spec["source"]]
        for name in wanted:
            target = kitchen / name
            expected = manifest["files"].get(name)
            if not (target.is_file() and (expected is None or sha256(target) == expected)):
                bundle.extract(PREFIX + name, DEST)
            if expected and sha256(target) != expected:
                raise ValueError(f"Checksum mismatch: {name}")
    derive(kitchen, spec)
    if not args.all:
        (kitchen / spec["source"]).unlink()
    if not (args.keep_archive or args.archive):
        archive.unlink()
    link(kitchen)
    files = [p for p in kitchen.rglob("*") if p.is_file()]
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB in {kitchen}")
    print(f"{derived_name} rebuilt and verified; linked at {LINK.relative_to(REPO)}")


if __name__ == "__main__":
    main()
