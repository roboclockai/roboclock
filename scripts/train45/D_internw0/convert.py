"""Convert the train45 task farm into one LeRobot v2.1 repo per task for InternW0-Delta (role D).

Wraps RoboDojo/XPolicyLab `scripts/transform_lerobot_v21_format.py` (imported, not edited) and
changes two things:
  * its DATA_ROOT is the task farm ($TRAIN45_DIR/data), not <RoboDojo>/data;
  * every frame gets its own task string, the shared prompt function of
    docs/five-domain-training.md (P3):
        P(task, seed, frame) = prompts[f"{task}|{seed}"] + f" Time: {frame / 25:.1f} s"
    (the XPolicyLab converter writes one HDF5 instruction per episode).
Everything else is the XPolicyLab converter's own code: 14-D state/action as
[left 6 joints, left gripper, right 6 joints, right gripper], action taken verbatim
(action[t] = state[t+1]), cameras cam_high/cam_left_wrist/cam_right_wrist at 480x640,
fps = env_cfg/arx_x5.yml collect_freq (25), AV1 (libsvtav1, crf 30, g 2) on CPU.

Output: <out>/<task>/{meta,data,videos}, plus <out>/_status/<task>.json once a task is complete.
A task with a status file is skipped unless --overwrite. Any episode error fails its task
(the XPolicyLab converter only prints and drops such episodes).

Run with venv-convert (lerobot 0.3.3 writes dataset codebase v2.1):
  nice -n 10 $TRAIN45_DIR/D_internw0/venv-convert/bin/python convert.py --jobs 8
  ... --tasks lab_A1,hospital_B1 --max-episodes 1 --out $TRAIN45_DIR/D_internw0/test/lerobot
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

ROBODOJO = Path(os.path.expandvars("$ROBODOJO_DIR"))
XPL_CONVERTER = ROBODOJO / "XPolicyLab/scripts/transform_lerobot_v21_format.py"
DEFAULT_DATA_ROOT = Path(os.path.expandvars("$TRAIN45_DIR/data"))
DEFAULT_PROMPTS = Path(os.path.expandvars("$TRAIN45_DIR/prompts/instructions_main_4f541f2.json"))
DEFAULT_OUT = Path(os.path.expandvars("$TRAIN45_DIR/D_internw0/lerobot"))
BENCH = "RoboDojo"
ENV_CFG = "arx_x5"
FPS = 25


def prompt(instructions, task, seed, frame):
    """P3 of docs/five-domain-training.md, identical for all four roles."""
    return instructions[f"{task}|{seed}"] + f" Time: {frame / 25:.1f} s"


def episode_seed(path):
    match = re.fullmatch(r"episode_(\d+)", Path(path).stem)
    if match is None:
        raise ValueError(f"cannot read the seed from {path}")
    return int(match.group(1))


def load_xpl_converter(data_root):
    """Import the XPolicyLab converter as a module and point it at the task farm."""
    if str(ROBODOJO) not in sys.path:
        sys.path.insert(0, str(ROBODOJO))  # for `from XPolicyLab.utils... import`
    spec = importlib.util.spec_from_file_location("xpl_transform_lerobot_v21", XPL_CONVERTER)
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    conv.DATA_ROOT = Path(data_root)  # read at call time by _discover_conversion_targets/_resolve_input_dir
    return conv


def convert_task(task, args):
    """Convert one task into <out>/<task>. Runs in its own (spawned) process."""
    os.environ["HF_LEROBOT_HOME"] = str(args.out)  # read by lerobot at import time
    conv = load_xpl_converter(args.data_root)
    import lerobot.datasets.lerobot_dataset as lds

    if lds.CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"lerobot writes {lds.CODEBASE_VERSION}; InternW0-Delta reads v2.1 (use lerobot 0.3.3)")
    if Path(conv.HF_LEROBOT_HOME) != Path(args.out):
        raise RuntimeError(f"HF_LEROBOT_HOME is {conv.HF_LEROBOT_HOME}, expected {args.out}")
    conv.configure_video_encoding(conv.DEFAULT_DATASET_CONFIG)

    instructions = json.loads(Path(args.prompts).read_text())
    targets = conv._discover_conversion_targets([f"{BENCH}.{task}.{ENV_CFG}"])
    if targets != [(BENCH, task, ENV_CFG)]:
        raise FileNotFoundError(f"{task}: expected one target under {args.data_root}, got {targets}")
    metadata, target_dims, fps = conv._plan_target_metadata(targets)
    if fps != FPS:
        raise ValueError(f"{task}: env_cfg fps {fps} != {FPS}")
    current_dims = metadata[targets[0]]["per_arm_dims"]
    files = conv.find_input_files(conv._resolve_input_dir(BENCH, task, ENV_CFG))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"{task}: no HDF5 files")
    missing = [f.name for f in files if f"{task}|{episode_seed(f)}" not in instructions]
    if missing:
        raise KeyError(f"{task}: no prompt for {missing[:5]} ({len(missing)} episodes)")

    dataset = conv.create_empty_dataset(
        task,  # repo_id -> <out>/<task>; removes a partial earlier attempt
        robot_type="unified_robot",
        motors=conv._build_motor_names_from_dims(target_dims),
        fps=fps,
        mode="video",
        image_height=480,
        image_width=640,
        dataset_config=conv.DEFAULT_DATASET_CONFIG,
    )
    t_task = time.time()
    episodes = []
    for path in files:
        t0 = time.time()
        seed = episode_seed(path)
        data = conv.load(str(path), data_type=BENCH, data_version="v1.0")
        state = conv._extract_qpos(data)
        action = conv._extract_action(data)
        if state.shape != action.shape:
            raise ValueError(f"{path}: state/action shape mismatch {state.shape} vs {action.shape}")
        state = conv._pad_state_to_target_dims(state, current_dims, target_dims, "state")
        action = conv._pad_state_to_target_dims(action, current_dims, target_dims, "action")
        images = {}
        for camera in conv.CAMERA_CANDIDATES:
            frames = conv._find_camera_array(data, camera, 480, 640)
            if frames is None or len(frames) != len(state):
                raise ValueError(f"{path}: camera {camera} missing or wrong length")
            images[camera] = frames
        t_read = time.time() - t0
        for i in range(len(state)):
            frame = {"observation.state": state[i], "action": action[i]}
            for camera, frames in images.items():
                frame[f"observation.images.{camera}"] = frames[i]
            dataset.add_frame(frame, task=prompt(instructions, task, seed, i))
        dataset.save_episode()
        conv.release_dataset_memory(dataset)
        episodes.append({"file": path.name, "seed": seed, "frames": int(len(state)),
                         "read_s": round(t_read, 2), "total_s": round(time.time() - t0, 2)})
        print(f"[{task}] {path.name} {len(state)} frames {time.time() - t0:.1f}s", flush=True)

    root = Path(args.out) / task
    status = {
        "task": task,
        "episodes": len(episodes),
        "frames": sum(e["frames"] for e in episodes),
        "seconds": round(time.time() - t_task, 1),
        "bytes": sum(f.stat().st_size for f in root.rglob("*") if f.is_file()),
        "prompts": str(args.prompts),
        "per_episode": episodes,
    }
    status_dir = Path(args.out) / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"{task}.json").write_text(json.dumps(status, indent=1))
    return status


def run_one(task, args):
    try:
        return task, convert_task(task, args), None
    except Exception:
        return task, None, traceback.format_exc()


def _run_one_packed(packed):
    return run_one(*packed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="all", help="comma list, or 'all' = every task in the farm")
    parser.add_argument("--jobs", type=int, default=4, help="tasks converted in parallel")
    parser.add_argument("--max-episodes", type=int, default=0, help="per task; 0 = all (testing only)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()

    # task dirs only: entries with arx_x5/data (role C keeps a meta/ dir in the farm root)
    farm = sorted(p.name for p in (args.data_root / BENCH).iterdir() if (p / ENV_CFG / "data").is_dir())
    tasks = farm if args.tasks == "all" else args.tasks.split(",")
    unknown = sorted(set(tasks) - set(farm))
    if unknown:
        parser.error(f"not in the farm: {unknown}")
    todo = [t for t in tasks if args.overwrite or not (args.out / "_status" / f"{t}.json").exists()]
    for t in todo:
        (args.out / "_status" / f"{t}.json").unlink(missing_ok=True)
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, converting {len(todo)} with {args.jobs} jobs -> {args.out}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    # Longest tasks first (by HDF5 bytes): a task is converted serially, so lab_C1/C2 (42k frames,
    # ~77 min at the measured 0.11 s/frame) set the tail.
    def size(task):
        return sum(f.stat().st_size for f in (args.data_root / BENCH / task / ENV_CFG / "data").glob("*.hdf5"))

    todo.sort(key=size, reverse=True)
    t0 = time.time()
    failed = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, args.jobs), maxtasksperchild=1) as pool:
        results = pool.imap_unordered(_run_one_packed, [(t, args) for t in todo], chunksize=1)
        for task, status, error in results:
            if error:
                failed[task] = error
                print(f"[{task}] FAILED\n{error}", flush=True)
                shutil.rmtree(args.out / task, ignore_errors=True)
            else:
                print(f"[{task}] done: {status['episodes']} episodes, {status['frames']} frames, "
                      f"{status['seconds']} s, {status['bytes'] / 1e9:.2f} GB", flush=True)
    print(f"wall {time.time() - t0:.0f} s; failed: {sorted(failed) or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
