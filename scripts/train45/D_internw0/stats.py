"""Normalisation stats for role D (InternW0-Delta), rule 5 of docs/five-domain-training.md.

Starts from the checkpoint's own stats (upstream assets/stats/robodojo.json, the same JSON as
the adapter's config/dataset_stats.json) and changes only `global_std`, only on the dims where
the train45 data leaves the range the stats cover. InternW0 normalises z = (x - global_mean) /
global_std and clamps z to [-5, 5] (wam/datasets/lerobot/utils/normalizer.py; the eval runtime
does the same), so a value with |z| > 5 is a clipped target. On such a dim:
    global_std_new = margin * max|x - global_mean| / 5
which puts the most extreme training value at |z| = 5 / margin. Means, min/max/quantiles and
the (unused: use_stepwise_action_norm=false) stepwise stats are left unchanged. State and
action are treated separately, with the same rule.

The data are read the way the XPolicyLab converter builds them:
[left_arm_joint_states(6), left_ee_joint_states(1), right_arm_joint_states(6), right_ee_joint_states(1)].

Run with venv-convert (numpy + h5py):
  $TRAIN45_DIR/D_internw0/venv-convert/bin/python stats.py
Writes stats_train45.json and stats_train45.report.json next to each other.
"""

import os
import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np

PARTS = ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")
Z_CLAMP = 5.0


def read(path, group):
    with h5py.File(path, "r") as f:
        return np.concatenate([np.asarray(f[f"{group}/{p}"][()], dtype=np.float32).reshape(-1, 1 if "ee" in p else 6)
                               for p in PARTS], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--farm", type=Path, default=Path(os.path.expandvars("$TRAIN45_DIR/data/RoboDojo")))
    parser.add_argument("--base", type=Path, default=Path(os.path.expandvars("$TRAIN45_DIR/D_internw0/upstream/assets/stats/robodojo.json")))
    parser.add_argument("--out", type=Path, default=Path(os.path.expandvars("$TRAIN45_DIR/D_internw0/stats_train45.json")))
    parser.add_argument("--margin", type=float, default=1.05, help="extreme training value lands at |z| = 5 / margin")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    files = sorted(glob.glob(str(args.farm / "*" / "arx_x5" / "data" / "*.hdf5")))
    tasks = sorted({Path(f).parts[-4] for f in files})
    data = {"state": [], "action": []}
    for path in files:
        for kind in data:
            data[kind].append(read(path, kind))
    data = {k: np.concatenate(v) for k, v in data.items()}

    out = json.loads(args.base.read_text())
    report = {"base": str(args.base), "base_episodes": base.get("num_episodes"), "base_frames": base.get("num_transition"),
              "tasks": len(tasks), "episodes": len(files), "frames": int(len(data["action"])),
              "margin": args.margin, "z_clamp": Z_CLAMP, "kinds": {}}
    for kind, x in data.items():
        stats = base[kind]["default"]
        mean = np.asarray(stats["global_mean"], dtype=np.float64)
        std = np.asarray(stats["global_std"], dtype=np.float64)
        dev = np.abs(x - mean).max(axis=0)
        z = np.abs(x - mean) / std
        widen = dev / std > Z_CLAMP
        new_std = std.copy()
        new_std[widen] = args.margin * dev[widen] / Z_CLAMP
        out[kind]["default"]["global_std"] = new_std.tolist()
        report["kinds"][kind] = {
            "dims_widened": np.flatnonzero(widen).tolist(),
            "per_dim": [
                {"dim": d, "data_min": float(x[:, d].min()), "data_max": float(x[:, d].max()),
                 "mean": float(mean[d]), "std_old": float(std[d]), "std_new": float(new_std[d]),
                 "max_abs_z_old": float(dev[d] / std[d]), "max_abs_z_new": float(dev[d] / new_std[d]),
                 "frac_clipped_old": float((z[:, d] > Z_CLAMP).mean())}
                for d in range(x.shape[1])
            ],
        }
    # The stats JSON keeps the base file's exact key set (both stats loaders read it as is);
    # what was widened, and why, is in the report.
    args.out.write_text(json.dumps(out, indent=2))
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=1))
    for kind, r in report["kinds"].items():
        print(f"{kind}: widened dims {r['dims_widened']}")
        for row in r["per_dim"]:
            if row["dim"] in r["dims_widened"]:
                print(f"  dim {row['dim']}: data [{row['data_min']:.3f}, {row['data_max']:.3f}] mean {row['mean']:.3f} "
                      f"std {row['std_old']:.3f} -> {row['std_new']:.3f}  max|z| {row['max_abs_z_old']:.2f} -> "
                      f"{row['max_abs_z_new']:.2f}  clipped before {100 * row['frac_clipped_old']:.2f}%")
    print(f"{report['tasks']} tasks, {report['episodes']} episodes, {report['frames']} frames -> {args.out}, {report_path}")


if __name__ == "__main__":
    main()
