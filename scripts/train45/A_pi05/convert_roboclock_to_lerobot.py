"""Convert roboclock oracle demos (RoboDojo v1.0 arx_x5 HDF5) into a LeRobot v3.0 dataset for openpi pi0.5.

Differences from XPolicyLab/policy/Pi_05/openpi/scripts/process_data.py (which cannot run in the openpi venv:
it imports lerobot.common, gone in lerobot 0.4.4):
  * gripper (T, 1) is reshaped, not [:, None]-ed (that made (T, 1, 1) and every episode was silently dropped);
  * the prompt comes from the `instruction` field (process_data.py reads `instructions`);
  * images are stored at 224x224 after openpi's own resize_with_pad (the exact op the policy server's
    ResizeImages transform applies to 640x480 frames), so training sees the same pixels as deployment;
  * episodes are written into N shard datasets in parallel and merged with lerobot's aggregate_datasets.
Unchanged: state = [left 6 joints, left gripper, right 6 joints, right gripper], action[t] = state[t+1]
(last frame repeats), dataset fps 50 (RoboDojo's choice for its 25 Hz data), cameras cam_high/left/right wrist.

Run with the openpi venv:
  PYTHONPATH=$ROBODOJO_DIR:$ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/src \
  HF_LEROBOT_HOME=$PI05_TRAIN_DIR/lerobot \
  $ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python convert_roboclock_to_lerobot.py \
      --src $DEMOS_DIR/roboclock_oracle_v1 --repo-id roboclock_oracle_v1_810 --shards 16
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

CAMERAS = {"cam_head": "cam_high", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
SIZE = 224
FPS = 50
CHUNK = 128


def features():
    motors = [f"left_{i}" for i in range(6)] + ["left_ee_0"] + [f"right_{i}" for i in range(6)] + ["right_ee_0"]
    feats = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": motors},
        "action": {"dtype": "float32", "shape": (14,), "names": [motors]},
    }
    for cam in CAMERAS.values():
        feats[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, SIZE, SIZE),
            "names": ["channels", "height", "width"],
        }
    return feats


def load_episode(path):
    import h5py
    import numpy as np

    from openpi.shared.image_tools import resize_with_pad
    from XPolicyLab.utils.process_data import decode_image_bit

    with h5py.File(path, "r") as ep:

        def arm(side):
            joints = ep[f"state/{side}_arm_joint_states"][:]
            grip = ep[f"state/{side}_ee_joint_states"][:].reshape(len(joints), 1)
            return np.concatenate([joints, grip], axis=1)

        state = np.concatenate([arm("left"), arm("right")], axis=1).astype(np.float32)
        instruction = ep["instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode()
        images = {}
        for src, dst in CAMERAS.items():
            raw = ep[f"vision/{src}/colors"]
            out = []
            for i in range(0, raw.shape[0], CHUNK):
                rgb = np.asarray(decode_image_bit(raw[i : i + CHUNK]))
                out.append(np.asarray(resize_with_pad(rgb, SIZE, SIZE)))
            images[dst] = np.concatenate(out)
            assert images[dst].shape == (len(state), SIZE, SIZE, 3), images[dst].shape
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    return state, action, images, str(instruction)


def episode_key(path):
    """'<env>_<task>|<seed>' for <task dir>/arx_x5/data/episode_<seed>.hdf5."""
    return f"{path.parent.parent.parent.name}|{int(path.stem.split('_')[-1])}"


def write_shard(args):
    shard, files, repo_id, root, overrides = args
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    shard_root = root / f"{repo_id}__shard{shard:02d}"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    ds = LeRobotDataset.create(
        repo_id=f"{repo_id}__shard{shard:02d}",
        fps=FPS,
        features=features(),
        root=shard_root,
        robot_type="dual_x5",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=4,
    )
    done = []
    for n, path in enumerate(files):
        t0 = time.time()
        state, action, images, recorded = load_episode(path)
        instruction = overrides.get(episode_key(path), recorded)
        for i in range(len(state)):
            frame = {"observation.state": state[i], "action": action[i], "task": instruction}
            for cam, arr in images.items():
                frame[f"observation.images.{cam}"] = arr[i]
            ds.add_frame(frame)
        ds.save_episode()
        done.append({"file": str(path), "frames": int(len(state)), "instruction": instruction,
                     "recorded_instruction": recorded})
        print(f"[shard {shard:02d}] {n + 1}/{len(files)} {path.parent.parent.parent.name}/{path.name} "
              f"{len(state)} frames {time.time() - t0:.1f}s", flush=True)
    ds.finalize()
    ds.stop_image_writer()
    return shard, str(shard_root), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--domains", default=None, help="comma-separated envs to keep, e.g. lab,hospital")
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes (smoke test)")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--instructions", type=Path, default=None,
                    help="JSON {'<env>_<task>|<seed>': prompt}: replaces the recorded instruction (e.g. the "
                         "text the current roboclock main generates for that seed)")
    args = ap.parse_args()
    overrides = json.loads(args.instructions.read_text()) if args.instructions else {}

    root = Path(os.environ["HF_LEROBOT_HOME"])
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*/arx_x5/data/episode_*.hdf5"))
    if args.domains:
        keep = set(args.domains.split(","))
        files = [f for f in files if f.parent.parent.parent.name.rsplit("_", 1)[0] in keep]
    if args.limit:
        files = files[: args.limit]
    shards = min(args.shards, len(files))
    jobs = [(k, files[k::shards], args.repo_id, root, overrides) for k in range(shards)]
    print(f"{len(files)} episodes -> {shards} shards -> {root / args.repo_id}", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(shards) as pool:
        results = sorted(pool.map(write_shard, jobs, chunksize=1))
    print(f"shards written in {(time.time() - t0) / 60:.1f} min", flush=True)

    from lerobot.datasets.aggregate import aggregate_datasets

    final = root / args.repo_id
    if final.exists():
        shutil.rmtree(final)
    aggregate_datasets(
        repo_ids=[Path(r).name for _, r, _ in results],
        aggr_repo_id=args.repo_id,
        roots=[Path(r) for _, r, _ in results],
        aggr_root=final,
    )
    episodes = [e for _, _, done in results for e in done]
    manifest = {
        "source": str(args.src),
        "episodes": len(episodes),
        "frames": sum(e["frames"] for e in episodes),
        "image_size": SIZE,
        "fps": FPS,
        "converter": str(Path(__file__).resolve()),
        "instructions_override": str(args.instructions) if args.instructions else None,
        "relabelled": sum(e["instruction"] != e["recorded_instruction"] for e in episodes),
        "files": episodes,
    }
    (final / "roboclock_manifest.json").write_text(json.dumps(manifest, indent=1))
    if not args.keep_shards:
        for _, r, _ in results:
            shutil.rmtree(r)
    print(f"done: {len(episodes)} episodes, {manifest['frames']} frames, {(time.time() - t0) / 60:.1f} min total",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
