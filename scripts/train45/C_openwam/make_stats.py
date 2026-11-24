"""Role C (OpenWAM-alpha) norm stats: the release's own stats, min/max widened to our training data.

docs/five-domain-training.md, rule 5 and section 6. OpenWAM's RoboDojo reader normalises EEF20
state and action with min-max and clips to [-1, 1]. Without a stats file in
`<dataset_dir>/meta/` it would rebuild the stats from our data alone, which moves every
coordinate away from the ones the checkpoint was trained in. Using the release file as it is
clips our targets: z (arm-base frame) of several tasks is below the release minimum.

So this script
1. computes our stats with OpenWAM's own builder (`compute_robodojo_stats`: the reader's
   `read_calibrated_eef20`, arm-base frame, EEF20 = xyz + rot6d + gripper per arm);
2. loads the release `normalization_stats.npy` and checks that it has the same payload layout
   and the same frame contract (fingerprint, contract id, frames, endpoint, gripper convention);
3. writes the release payload with `eef.min` / `eef.max` replaced by the elementwise union of
   both ranges. mean / std / q01 / q99 and all metadata stay the release's;
4. re-reads the result with the reader's own validator, and writes a JSON report of the
   widened dims (and its sha256, which run_train.sh checks before launching).

Run it with the OpenWAM venv (CPU only, ~1 min for 610 episodes):

  nice -n 10 $ROBODOJO_DIR/XPolicyLab/policy/OpenWAM/.venv/bin/python make_stats.py
"""

import os
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RELEASE = Path(os.path.expandvars("$MODELS_DIR/OpenWAM-Alpha-Sim-RoboDojo/normalization_stats.npy"))
FARM = Path(os.path.expandvars("$TRAIN45_DIR/data/RoboDojo"))
REPORT = Path(os.path.expandvars("$TRAIN45_DIR/C_openwam/normalization_widening_report.json"))
MODE = "eef"
# Metadata both payloads must agree on for the two ranges to live in the same space.
CONTRACT_KEYS = (
    "source_frame",
    "target_frame",
    "endpoint",
    "embodiment",
    "calibration_fingerprint",
    "contract_id",
    "gripper_convention",
)
DIM_NAMES = [
    f"{arm}_{name}"
    for arm in ("left", "right")
    for name in ("x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5", "gripper")
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_payload(path):
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: payload is not a dict")
    return payload


def check_same_space(release, ours, stat_keys, eef_dim):
    """Raise unless both payloads have the same layout and frame contract."""
    if set(release) != set(ours):
        raise ValueError(f"top-level keys differ: release {sorted(release)} vs ours {sorted(ours)}")
    for name, payload in (("release", release), ("ours", ours)):
        stats = payload[MODE]
        for key in stat_keys:
            value = np.asarray(stats[key])
            if value.shape != (eef_dim,) or value.dtype != np.float32:
                raise ValueError(f"{name} {MODE}.{key}: shape {value.shape} dtype {value.dtype}, expected ({eef_dim},) float32")
    for key in CONTRACT_KEYS:
        a, b = release["metadata"].get(key), ours["metadata"].get(key)
        if a is None or a != b:
            raise ValueError(f"metadata.{key} differs: release {a!r} vs ours {b!r}")
    # The release file predates the explicit variant marker; the reader defaults it to sim.
    if release["metadata"].get("variant", "sim") != ours["metadata"].get("variant", "sim"):
        raise ValueError("metadata.variant differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--farm", type=Path, default=FARM, help="dataset_dir: <farm>/<task>/arx_x5/data")
    parser.add_argument("--release", type=Path, default=RELEASE)
    parser.add_argument("--out", type=Path, help="default: <farm>/meta/robodojo_normalization_stats.npy")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--force", action="store_true", help="overwrite an existing stats file")
    args = parser.parse_args()

    from openwam.dataloader.robodojo import (
        _load_validated_stats,
        default_robodojo_stats_path,
        resolve_robodojo_calibration,
    )
    from openwam.dataloader.robodojo_contract import EEF20_DIM
    from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, STAT_KEYS
    from openwam.dataloader.utils.stats_computation.robodojo_stats_computation import (
        atomic_save_stats_npy,
        compute_robodojo_stats,
    )

    out = args.out or default_robodojo_stats_path(args.farm, variant="sim", embodiment="arx_x5")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to replace it")

    ours = compute_robodojo_stats(dataset_dir=args.farm, embodiment="arx_x5", variant="sim", action_mode=MODE)
    release = load_payload(args.release)
    check_same_space(release, ours, STAT_KEYS, EEF20_DIM)

    rel_min = np.asarray(release[MODE]["min"], dtype=np.float32)
    rel_max = np.asarray(release[MODE]["max"], dtype=np.float32)
    our_min = np.asarray(ours[MODE]["min"], dtype=np.float32)
    our_max = np.asarray(ours[MODE]["max"], dtype=np.float32)
    new_min, new_max = np.minimum(rel_min, our_min), np.maximum(rel_max, our_max)

    merged = dict(release)
    merged[MODE] = dict(release[MODE])
    merged[MODE]["min"], merged[MODE]["max"] = new_min, new_max
    atomic_save_stats_npy(out, merged)

    # The reader's own validator, as the training dataset will call it (no task check there).
    validated = _load_validated_stats(out, resolve_robodojo_calibration(None), variant="sim", embodiment="arx_x5")
    assert np.array_equal(validated["min"], new_min) and np.array_equal(validated["max"], new_max)
    for key in STAT_KEYS:
        if key not in ("min", "max"):
            assert np.array_equal(validated[key], np.asarray(release[MODE][key], dtype=np.float32)), key

    dims = []
    for i, name in enumerate(DIM_NAMES):
        low, high = float(rel_min[i] - new_min[i]), float(new_max[i] - rel_max[i])
        dims.append(
            {
                "dim": i,
                "name": name,
                "release": [float(rel_min[i]), float(rel_max[i])],
                "ours": [float(our_min[i]), float(our_max[i])],
                "merged": [float(new_min[i]), float(new_max[i])],
                "widened_low": low,
                "widened_high": high,
                "range_growth": float((new_max[i] - new_min[i]) / (rel_max[i] - rel_min[i]) - 1.0),
            }
        )
    report = {
        "stats_file": str(out),
        "sha256": sha256(out),
        "release_file": str(args.release),
        "release_sha256": sha256(args.release),
        "farm": str(args.farm),
        "rule": "release eef stats; min/max = elementwise union of release and ours; everything else from the release",
        "ours": {
            "tasks": list(ours["metadata"]["tasks"]),
            "state_rows": int(ours["metadata"]["state_rows"]),
            "action_rows": int(ours["metadata"]["action_rows"]),
        },
        "rot6d_dims_identity": [int(d) for d in ROT6D_DIMS_EEF20],
        "widened": [d["name"] for d in dims if d["widened_low"] > 0 or d["widened_high"] > 0],
        "dims": dims,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")

    print(f"wrote {out} (sha256 {report['sha256'][:12]}) from {len(report['ours']['tasks'])} tasks, "
          f"{report['ours']['state_rows']} state rows; report {args.report}")
    for d in dims:
        if d["widened_low"] > 0 or d["widened_high"] > 0:
            print(f"  {d['name']:>15}: release [{d['release'][0]:+.4f}, {d['release'][1]:+.4f}] -> "
                  f"[{d['merged'][0]:+.4f}, {d['merged'][1]:+.4f}]  (range +{100 * d['range_growth']:.1f}%)")


if __name__ == "__main__":
    main()
