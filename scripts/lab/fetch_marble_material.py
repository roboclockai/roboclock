"""Download the original CC0 ambientCG Marble012 8K maps used by the office desk."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://ambientcg.com/view?id=Marble012"
DOWNLOAD = "https://acg-download.struffelproductions.com/file/ambientCG-Web/download/Marble012_uZys50mU/Marble012_8K-JPG.zip"


def main():
    folder = Path("assets/lab/marble8k")
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / "source.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if all(
            (folder / name).exists()
            and hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
            for name, digest in record["files"].items()
        ):
            print("Marble012 8K maps already present; checksums verified")
            return
    archive = folder / "Marble012_8K-JPG.zip"
    request = urllib.request.Request(
        DOWNLOAD, headers={"User-Agent": "Mozilla/5.0", "Referer": SOURCE}
    )
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if any(token in name for token in ["Color.jpg", "Roughness.jpg", "NormalGL.jpg"]):
                (folder / Path(name).name).write_bytes(bundle.read(name))
    record_path.write_text(
        json.dumps(
            {
                "source": SOURCE,
                "download": DOWNLOAD,
                "license": "CC0",
                "resolution": 8192,
                "technique": "PBRProcedural (official asset classification)",
                "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.jpg")
                },
            },
            indent=2,
        )
    )
    archive.unlink()


if __name__ == "__main__":
    main()
