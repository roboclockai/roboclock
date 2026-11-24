"""Role A (pi0.5) norm stats: RoboDojo's checkpoint stats, with state q01/q99 widened to the training data.

The rule of the 2026-11-11 lab reference run (docs/five-domain-training.md, rule 5):
- start from the init checkpoint's own `assets/arx_x5_sim/norm_stats.json`;
- on each state dim where the training data leaves [q01, q99], move that bound to the data's
  min / max (pi0.5 clamps quantile-normalised state, so out-of-range state tokens saturate);
- leave actions unchanged (recomputing them inflates near-constant wrist-roll deltas);
- mean / std stay as they are (quantile norm does not use them).

State is the 14-D vector the LeRobot converter writes and AlohaInputs(adapt_to_pi=False) passes
through unchanged: [left 6 joints, left gripper, right 6 joints, right gripper] from the HDF5
`state/` group.

  python widen_norm_stats.py --farm $TRAIN45_DIR/data/RoboDojo \
      --out $PI05_TRAIN_DIR/assets_train45/arx_x5_sim/norm_stats.json
  # reproduce the lab file: --farm <dir with only lab tasks> or --domains lab
"""

import os
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

INIT = Path(os.path.expandvars("$ROBODOJO_DIR/hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets/arx_x5_sim/norm_stats.json"))
KEYS = (
    "state/left_arm_joint_states",
    "state/left_ee_joint_states",
    "state/right_arm_joint_states",
    "state/right_ee_joint_states",
)


def state_range(files):
    lo, hi = np.full(14, np.inf), np.full(14, -np.inf)
    for path in files:
        with h5py.File(path, "r") as h5:
            state = np.concatenate([np.asarray(h5[k][()], dtype=np.float64).reshape(len(h5[k]), -1) for k in KEYS], axis=1)
        lo, hi = np.minimum(lo, state.min(0)), np.maximum(hi, state.max(0))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", type=Path, required=True, help="dir of <task>/arx_x5/data/episode_*.hdf5")
    ap.add_argument("--domains", default=None, help="only tasks of these envs (comma-separated)")
    ap.add_argument("--init", type=Path, default=INIT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.farm.glob("*/arx_x5/data/episode_*.hdf5"))
    if args.domains:
        keep = tuple(f"{d}_" for d in args.domains.split(","))
        files = [f for f in files if f.parts[-4].startswith(keep) and f.parts[-4].rsplit("_", 1)[0] in args.domains.split(",")]
    lo, hi = state_range(files)
    stats = json.loads(args.init.read_text())
    state = stats["norm_stats"]["state"]
    q01, q99 = np.array(state["q01"]), np.array(state["q99"])
    widened = {}
    for i in range(14):
        if lo[i] < q01[i]:
            widened.setdefault(i, {})["q01"] = [float(q01[i]), float(lo[i])]
            q01[i] = lo[i]
        if hi[i] > q99[i]:
            widened.setdefault(i, {})["q99"] = [float(q99[i]), float(hi[i])]
            q99[i] = hi[i]
    state["q01"], state["q99"] = q01.tolist(), q99.tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2))
    report = {"episodes": len(files), "init": str(args.init), "widened_state_dims": widened}
    Path(str(args.out) + ".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
