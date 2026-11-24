"""Write the X-VLA meta file for the five-domain training set (docs/five-domain-training.md, role B).

One datalist entry per task directory of the farm (`<farm>/<task>/arx_x5/data`); xvla's loader
expands each directory to its `.hdf5` files. Every episode is checked here, so a bad farm fails
before training starts:
- it has a prompt `"<task>|<seed>"` in the prompt file;
- its `additional_info/frequency` is 25.

  $ROBODOJO_DIR/XPolicyLab/policy/X_VLA/.venv/bin/python scripts/train45/B_xvla/make_meta.py
"""

import os
import argparse
import json
import re
import sys
from pathlib import Path

import h5py

EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--farm", default=os.path.expandvars("$TRAIN45_DIR/data/RoboDojo"))
    parser.add_argument("--prompts", default=os.path.expandvars("$TRAIN45_DIR/prompts/instructions_main_4f541f2.json"))
    parser.add_argument("--out", default=os.path.expandvars("$TRAIN45_DIR/B_xvla/roboclock45.json"))
    parser.add_argument("--expect-tasks", type=int, default=45)
    parser.add_argument("--expect-episodes", type=int, default=610)
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    farm = Path(args.farm)
    datalist, problems, n_episodes, n_frames = [], [], 0, 0
    for task_dir in sorted(p for p in farm.iterdir() if p.is_dir()):
        data_dir = task_dir / "arx_x5" / "data"
        episodes = sorted(data_dir.glob("episode_*.hdf5"))
        if not episodes:
            problems.append(f"{task_dir.name}: no episodes")
            continue
        for ep in episodes:
            key = f"{task_dir.name}|{int(EPISODE_RE.search(ep.name).group(1))}"
            if key not in prompts:
                problems.append(f"{key}: no prompt")
            with h5py.File(ep, "r") as f:
                freq = int(f["additional_info/frequency"][()])
                n_frames += f["state/left_ee_poses"].shape[0]
            if freq != 25:
                problems.append(f"{key}: frequency {freq}")
        datalist.append(str(data_dir))
        n_episodes += len(episodes)

    if len(datalist) != args.expect_tasks or n_episodes != args.expect_episodes:
        problems.append(f"{len(datalist)} tasks / {n_episodes} episodes, expected "
                        f"{args.expect_tasks} / {args.expect_episodes}")
    if problems:
        sys.exit("make_meta: " + "; ".join(problems[:10]))

    meta = {
        "dataset_name": "roboclock_ee",
        "language_instruction_key": "instruction",
        "observation_key": ["vision/cam_head/colors"],
        "prompts_path": str(Path(args.prompts).resolve()),
        "datalist": datalist,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out}: {len(datalist)} tasks, {n_episodes} episodes, {n_frames} frames")


if __name__ == "__main__":
    main()
