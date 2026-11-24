"""Fetch the RoboCasa objects selected for the Dual-X5 scenes by HTTP range reads (no 5.8 GB download).

    python scripts/household/collect_x5_assets.py /opt/household/V1.0_X5/assets/robocasa
Raw members land in <dest>/aigen_objs/<category>/<instance>/; build.py converts the visual meshes
into src/roboclock/envs/household/assets/x5/objects when ROBOCASA_RAW points at <dest>/aigen_objs.
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/robocasa/robocasa-assets/resolve/1b92c3d/aigen_objs.zip"
SELECTED = ["pan_2", "pan_3", "kettle_0", "baking_sheet_1", "tray_0", "tray_1", "plate_1", "plate_2",
            "mug_0", "mug_2", "bowl_1", "bowl_5", "fork_0", "spatula_0", "sponge_0"]


class RemoteZip(io.RawIOBase):
    """Seekable HTTP range reader with a single prefetched block per object."""

    def __init__(self, url):
        self.url, self.pos, self.block = url, 0, (0, b"")
        data, headers = self.fetch(-65536, None)
        self.length = int(headers["Content-Range"].split("/")[-1])
        self.tail = (self.length - len(data), data)

    def fetch(self, start, end):
        rng = f"bytes={start}" if end is None else f"bytes={start}-{end}"
        with urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": rng}), timeout=120) as r:
            if r.status != 206:
                raise RuntimeError("HTTP range support required; refusing a whole-archive download")
            return r.read(), r.headers

    def prefetch(self, start, end):
        self.block = (start, self.fetch(start, end)[0])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.length + offset
        return self.pos

    def read(self, n=-1):
        n = min(n if n >= 0 else self.length - self.pos, self.length - self.pos)
        for start, data in (self.block, self.tail):
            if start <= self.pos and self.pos + n <= start + len(data):
                out = data[self.pos - start:self.pos - start + n]
                self.pos += n
                return out
        data, _ = self.fetch(self.pos, self.pos + n - 1)
        self.pos += n
        return data


def main(dest):
    dest = Path(dest)
    rz = RemoteZip(URL)
    records = []
    with zipfile.ZipFile(rz) as z:
        names = z.namelist()
        for inst in SELECTED:
            cat = inst.rsplit("_", 1)[0]
            prefix = f"aigen_objs/{cat}/{inst}/"
            members = [z.getinfo(n) for n in names if n.startswith(prefix) and not n.endswith("/")]
            if not members or sum(m.file_size for m in members) > 40_000_000:
                raise RuntimeError(f"Unexpected archive members for {inst}")
            lo = min(m.header_offset for m in members)
            hi = max(m.header_offset + m.compress_size + 30 + len(m.filename) + 512 for m in members)
            rz.prefetch(lo, min(hi, rz.length - 1))
            for m in members:
                if ".." in Path(m.filename).parts:
                    raise RuntimeError("Unsafe archive member")
                target = dest / m.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(z.read(m.filename))
                records.append(dict(member=m.filename, bytes=target.stat().st_size,
                                    sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            print("COLLECTED", inst, len(members), flush=True)
    (dest / "x5_selected_manifest.json").write_text(json.dumps(dict(
        source=URL, revision="1b92c3d02ca4354984fec961357db0bff7b32166",
        license="CC BY 4.0 (RoboCasa dataset card)", files=records), indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/opt/household/V1.0_X5/assets/robocasa")
