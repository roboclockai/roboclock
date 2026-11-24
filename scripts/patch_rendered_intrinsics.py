"""Add the rendered (square-pixel) camera intrinsics to already-delivered result files.

RTX ignores the RoboDojo head camera's 14.266 mm vertical aperture and renders square
pixels (fx = fy = 288.13 px, 96.0 x 79.6 deg), while Isaac's ``get_intrinsics_matrix()``
reports fy = 336.46 px (71.0 deg). Results recorded before 2026-11-04 carry only the
reported matrix. This script adds ``camera_intrinsics_rendered`` (and a short
``camera_intrinsics_note``) beside every reported head/wrist intrinsics block, without
re-recording anything:

- lab ``data/lab/videos-v2``: ``camera_intrinsics`` (view -> K) in result.json and eval rows;
  ``runs-*/acceptance/visibility/visibility.json`` also gets ``margin_px_rendered`` per prim,
  recomputed from the recorded pixel ranges (u is unchanged, v is rescaled by fx / fy_reported);
- sports ``data/sports/videos``: blocks with ``cameras.<view>.intrinsics_matrix``;
- hospital ``data/hospital``: ``camera_intrinsics_reported`` (view -> K);
- household ``data/household/x5_demos``: ``camera_intrinsics`` (view -> {intrinsics_matrix}).

Industry is not handled here: it is changed only on rig/industry, which records its own
rendered intrinsics.

Every embedded ``robodojo_rig.describe()`` block gets the current spec's camera entries
(``fov_deg`` of the rendered image plus ``rendered_intrinsics``), so it equals
``robodojo_rig.describe()`` again. Folders named ``superseded*`` and ``legacy-franka`` are
skipped. None of these four domains records a sha256 of its JSON files (their manifests
and sidecars hash only videos), so no checksum needs updating. JSON formatting (indent,
trailing newline) is preserved.

Idempotent: blocks that already carry the rendered matrix are left alone.

    python scripts/patch_rendered_intrinsics.py            # patch in place
    python scripts/patch_rendered_intrinsics.py --dry-run  # report only
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roboclock.envs import robodojo_rig as RD  # noqa: E402
from roboclock.envs.household.x5.robodojo import rendered_intrinsics_report  # noqa: E402

DOMAINS = {
    "lab": "data/lab/videos-v2",
    "sports": "data/sports/videos",
    "hospital": "data/hospital",
    "household": "data/household/x5_demos",
}
SKIP_PARTS = ("superseded", "legacy-franka")
HEAD_NAMES = {"cam_head", "rgb", "head"}
WRIST_NAMES = {"cam_left_wrist", "cam_right_wrist", "left_wrist_rgb", "right_wrist_rgb", "Left", "Right"}
PATCH_NOTE = (
    "camera_intrinsics is what Isaac's get_intrinsics_matrix() reports (fy from the vertical "
    "aperture). RTX ignores the vertical aperture and renders square pixels, so the image "
    "follows camera_intrinsics_rendered (robodojo_rig.intrinsics: fy = fx, principal point at "
    "the centre). Use the rendered matrix for projection and depth unprojection."
    " camera_intrinsics_rendered was added on 2026-11-04 by scripts/patch_rendered_intrinsics.py;"
    " the recording itself was not re-run."
)
HEAD_FX = RD.intrinsics(RD.HEAD_CAMERA)[0, 0]
HEAD_FY_REPORTED = RD.intrinsics(RD.HEAD_CAMERA, square_pixels=False)[1, 1]


def template(name):
    if name in HEAD_NAMES:
        return RD.HEAD_CAMERA
    if name in WRIST_NAMES:
        return RD.WRIST_CAMERA
    return None


def check_reported(name, matrix, problems):
    """The recorded reported K must be the spec's (fx from the horizontal aperture)."""
    spec = template(name)
    expected = RD.intrinsics(spec, square_pixels=False)
    if not math.isclose(matrix[0][0], expected[0, 0], rel_tol=1e-4) or not math.isclose(
        matrix[1][1], expected[1, 1], rel_tol=1e-4
    ):
        problems.append(f"{name}: recorded K {matrix[0][0]:.3f}/{matrix[1][1]:.3f} is not the spec's reported K")


def rendered_for(reported, problems):
    """Rendered block in the shape of the domain's reported block (None stays None)."""
    if reported is None:
        return None
    if not isinstance(reported, dict):
        return None
    views = {k: v for k, v in reported.items() if template(k) is not None}
    if not views:
        return {} if not reported else None
    sample = next(iter(views.values()))
    if isinstance(sample, dict) and "intrinsics_matrix" in sample:  # household
        for name, value in views.items():
            check_reported(name, value["intrinsics_matrix"], problems)
        names = {name: view for view, name in _household_views().items()}
        return rendered_intrinsics_report([names[name] for name in views])
    out = {}  # lab / hospital: view -> K (or None without a camera)
    for name, value in views.items():
        if value is None:
            out[name] = None
            continue
        check_reported(name, value, problems)
        out[name] = RD.intrinsics(template(name)).tolist()
    return out


def _household_views():
    from roboclock.envs.household.x5.robodojo import VIEWS

    return VIEWS


def insert_after(mapping, anchor, items):
    """Rebuild ``mapping`` with ``items`` right after key ``anchor`` (in place)."""
    pairs = list(mapping.items())
    mapping.clear()
    for key, value in pairs:
        mapping[key] = value
        if key == anchor:
            mapping.update(items)


def patch_describe(block, problems):
    """Embedded robodojo_rig.describe(): camera entries -> the current spec's (idempotent)."""
    current = json.loads(json.dumps(RD.describe()))
    changed = False
    for key, spec in (("head_camera", RD.HEAD_CAMERA), ("wrist_camera", RD.WRIST_CAMERA)):
        camera = block.get(key)
        if not isinstance(camera, dict) or camera == current[key]:
            continue
        for field in ("focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm"):
            if not math.isclose(camera[field], spec[field]):
                problems.append(f"{key}.{field} {camera[field]} differs from the spec")
        block[key] = current[key]
        changed = True
    return changed


def patch(node, problems):
    """Recursively add rendered intrinsics; returns the number of blocks added/converted."""
    count = 0
    if isinstance(node, list):
        return sum(patch(item, problems) for item in node)
    if not isinstance(node, dict):
        return 0
    if "head_camera" in node and "wrist_camera" in node and "arm_separation_m" in node:
        count += patch_describe(node, problems)
    for reported_key in ("camera_intrinsics", "camera_intrinsics_reported"):
        if reported_key in node and "camera_intrinsics_rendered" not in node:
            rendered = rendered_for(node[reported_key], problems)
            items = {"camera_intrinsics_rendered": rendered}
            if rendered and "camera_intrinsics_note" not in node:
                note = PATCH_NOTE
                if reported_key != "camera_intrinsics":
                    note = note.replace("camera_intrinsics is", f"{reported_key} is", 1)
                items["camera_intrinsics_note"] = note
            insert_after(node, reported_key, items)
            count += 1
    cameras = node.get("cameras")
    if (
        isinstance(cameras, dict)
        and any(isinstance(v, dict) and "intrinsics_matrix" in v for v in cameras.values())
        and "camera_intrinsics_rendered" not in node
    ):  # sports: cameras.<view>.intrinsics_matrix
        rendered = {}
        for name, value in cameras.items():
            if template(name) is None or not isinstance(value, dict):
                continue
            check_reported(name, value["intrinsics_matrix"], problems)
            rendered[name] = RD.intrinsics(template(name)).round(4).tolist()
        insert_after(
            node,
            "cameras",
            {"camera_intrinsics_rendered": rendered, "camera_intrinsics_note": PATCH_NOTE.replace(
                "camera_intrinsics is", "cameras.<view>.intrinsics_matrix is", 1)},
        )
        count += 1
    for value in list(node.values()):
        count += patch(value, problems)
    return count


def patch_visibility(report):
    """Lab visibility.json: add rendered margins from the recorded (reported-model) pixel ranges."""
    width, height = RD.HEAD_CAMERA["resolution"]
    cy = height / 2
    count = 0
    for entry in report.values():
        if not isinstance(entry, dict) or "prims" not in entry or "minimum_margin_px_rendered" in entry:
            continue
        margins = []
        for row in entry["prims"]:
            u0, v0, u1, v1 = row["pixel_range"]
            if all(math.isfinite(x) for x in (u0, v0, u1, v1)) and math.isfinite(row["margin_px"]):
                v0, v1 = (cy - HEAD_FX * (cy - v) / HEAD_FY_REPORTED for v in (v0, v1))
                row["pixel_range_rendered"] = [round(u0, 1), round(v0, 1), round(u1, 1), round(v1, 1)]
                row["margin_px_rendered"] = min(u0, width - u1, v0, height - v1)
            else:
                row["margin_px_rendered"] = row["margin_px"]
            margins.append(row["margin_px_rendered"])
        insert_after(
            entry,
            "minimum_margin_px",
            {
                "minimum_margin_px_rendered": min(margins, default=None),
                "projection_note": "margin_px / pixel_range use the reported fy (336.46 px); "
                "*_rendered use the square pixels RTX renders (fy = fx = 288.13 px), recomputed "
                "2026-11-04 by scripts/patch_rendered_intrinsics.py",
            },
        )
        count += 1
    return count


def dump_like(original, data, lines=False):
    """Serialize ``data`` in the formatting of ``original`` (or None if no candidate matches)."""
    candidates = [
        dict(indent=indent, ensure_ascii=ascii_, separators=seps)
        for indent in (2, 1, None, 4)
        for ascii_ in (True, False)
        for seps in ((", ", ": ") if indent is None else (",", ": "), (",", ":"))
    ]
    if lines:
        rows = [json.loads(line) for line in original.splitlines() if line.strip()]
        for kwargs in candidates:
            if "\n".join(json.dumps(r, **kwargs) for r in rows) == original.rstrip("\n"):
                end = "\n" if original.endswith("\n") else ""
                return lambda new: "\n".join(json.dumps(r, **kwargs) for r in new) + end
        return None
    parsed = json.loads(original)
    for kwargs in candidates:
        text = json.dumps(parsed, **kwargs)
        for end in ("\n", ""):
            if text + end == original:
                return lambda new, kwargs=kwargs, end=end: json.dumps(new, **kwargs) + end
    return None


def skipped(path, root):
    return any(part.startswith(SKIP_PARTS) for part in path.relative_to(root).parts)


def process(domain, root, dry_run):
    changed = []
    problems = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".json", ".jsonl") or skipped(path, root) or not path.is_file():
            continue
        text = path.read_text()
        lines = path.suffix == ".jsonl"
        try:
            data = [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)
        except json.JSONDecodeError as error:
            problems.append(f"{path}: {error}")
            continue
        local = []
        count = patch(data, local)
        if domain == "lab" and path.name == "visibility.json":
            count += patch_visibility(data)
        problems += [f"{path.relative_to(ROOT)}: {p}" for p in local]
        if not count:
            continue
        writer = dump_like(text, data, lines)
        if writer is None:
            problems.append(f"{path.relative_to(ROOT)}: formatting not recognized, written with indent 2")
            writer = (lambda new: "\n".join(json.dumps(r) for r in new) + "\n") if lines else (
                lambda new: json.dumps(new, indent=2) + "\n")
        new_text = writer(data)
        if new_text != text:
            changed.append(path)
            if not dry_run:
                path.write_text(new_text)
    return changed, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    args = parser.parse_args(argv)
    failed = False
    for domain in args.domains:
        root = ROOT / DOMAINS[domain]
        if not root.is_dir():
            print(f"{domain}: {root} missing")
            continue
        changed, problems = process(domain, root, args.dry_run)
        print(f"{domain}: {len(changed)} file(s) {'would be ' if args.dry_run else ''}patched")
        for path in changed:
            print(f"  {path.relative_to(ROOT)}")
        for problem in problems:
            print(f"  WARNING {problem}")
            failed = failed or "not the spec" in problem or "differs" in problem
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
