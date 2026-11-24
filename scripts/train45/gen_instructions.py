"""Prompts for five-domain training: the instruction this checkout's task code gives each demo episode.

The recorded HDF5 `instruction` comes from the code that recorded the demo, and the 11-10/11-12
commits rewrote instructions in every domain. A policy is evaluated with the current
`gen_instruction()`, so training uses it too (docs/five-domain-training.md, step P1). Generalises
$PI05_TRAIN_DIR/gen_lab_instructions.py (lab only) to all five domains.

For every `<demos>/<task>/arx_x5/data/episode_<seed>.hdf5` the difficulty is read from the
episode's sidecar `<demos>/_meta/<task>/seed_<seed>.json`. No Isaac is needed.

Output JSON: {"<task>|<seed>": instruction}, the key format of
convert_roboclock_to_lerobot.py --instructions. A second file `<out>.report.json` lists, per task,
whether the new instruction differs from the recorded one.

  PYTHONPATH=src python scripts/train45/gen_instructions.py \
      --demos $DEMOS_DIR/roboclock_oracle_v1 --out $TRAIN45_DIR/prompts/instructions_main_<commit>.json
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

MAKE_TASK = {
    "lab": "roboclock.envs.lab.tasks.registry",
    "hospital": "roboclock.envs.hospital.tasks.registry",
    "household": "roboclock.envs.household.tasks.registry",
    "industry": "roboclock.envs.industry.tasks",
    "sports": "roboclock.envs.sports.tasks",
}


def split_task(name):
    env, task_id = name.rsplit("_", 1)
    return env, task_id


def bind_hospital_layout(task):
    """hospital tasks that word their instruction from the scene (e.g. B1's alarm stations)
    need the layout the backend binds at reset; make_layout itself is pure Python."""
    from roboclock.envs.hospital.config import load_config
    from roboclock.envs.hospital.layout import make_layout

    config = load_config()
    task.bind(make_layout(task, config), config["sim"]["placement_dwell"])


BIND_LAYOUT = {"hospital": bind_hospital_layout}


def instruction(env, make_task, task_id, difficulty, seed):
    task = make_task(task_id, difficulty, seed)
    try:
        return task.gen_instruction()
    except AttributeError:
        if env not in BIND_LAYOUT or getattr(task, "layout", "missing") is not None:
            raise
    BIND_LAYOUT[env](task)
    return task.gen_instruction()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", default=None, help="comma-separated <env>_<task> names (default: every task dir)")
    args = ap.parse_args()

    import roboclock

    print("roboclock from", roboclock.__file__, file=sys.stderr)
    wanted = set(args.tasks.split(",")) if args.tasks else None
    out, report = {}, {}
    for task_dir in sorted(p for p in args.demos.iterdir() if (p / "arx_x5" / "data").is_dir()):
        name = task_dir.name
        if wanted and name not in wanted:
            continue
        env, task_id = split_task(name)
        make_task = importlib.import_module(MAKE_TASK[env]).make_task
        changed = same = 0
        for path in sorted((task_dir / "arx_x5" / "data").glob("episode_*.hdf5")):
            seed = int(re.search(r"episode_(\d+)", path.name).group(1))
            side = json.loads((args.demos / "_meta" / name / f"seed_{seed}.json").read_text())
            assert side["seed"] == seed and side["status"] == "success", (name, seed, side.get("status"))
            text = instruction(env, make_task, task_id, side["difficulty"], seed)
            out[f"{name}|{seed}"] = text
            if text == side["instruction"]:
                same += 1
            else:
                changed += 1
        report[name] = {"episodes": same + changed, "changed": changed}
        print(f"{name:16s} {same + changed:3d} episodes, {changed:3d} changed", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=0, ensure_ascii=False))
    Path(str(args.out) + ".report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(out)} prompts -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
