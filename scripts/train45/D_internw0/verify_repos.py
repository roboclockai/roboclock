"""Verify converted repos with the upstream InternW0-Delta reader (CPU only).

Mode `reader` (venv-train):
  1. Composes the training config exactly as run_train.sh does (task=robodojo + the role-D
     overrides, context_len 256, ROBODOJO_NORM_STATS) and runs upstream validate_training_config.
  2. Instantiates data.train (RobotVideoDataset -> BaseLerobotDataset -> upstream v2.1 reader).
  3. For every frame of every repo: meta.tasks[task_index] == P3(task, seed, frame_index), with the
     seed of each episode from convert.py's _status file; fps 25; state/action 14-D.
  4. Builds full training samples (dataset._get, which does not silently resample like
     __getitem__) at a few indices: decoded video, 80-D action/proprio, |z| <= 5, the prompt
     equals DEFAULT_PROMPT.format(P3) of that frame, and the text context is [256, 4096].
  Step 4 needs a text cache. --fake-cache DIR writes zero embeddings for the needed prompts
  into DIR (a test-only directory whose name must contain 'fake'); never point training at it.

Mode `pixels` (venv-convert): decodes each repo's AV1 videos with PyAV and compares them with
the HDF5 frames decoded by XPolicyLab decode_image_bit (PSNR and per-channel means, which would
expose a channel swap), and checks action[t] == state[t+1] against the HDF5.
"""

import argparse
import json
import os
import sys
from pathlib import Path

D = Path(os.path.expandvars("$TRAIN45_DIR/D_internw0"))
UP = D / "upstream"
WAN = os.path.expandvars("$MODELS_DIR/Wan2.2-TI2V-5B")
PROMPTS = Path(os.path.expandvars("$TRAIN45_DIR/prompts/instructions_main_4f541f2.json"))
CONTEXT_LEN = 256


def p3(instructions, task, seed, frame):
    return instructions[f"{task}|{seed}"] + f" Time: {frame / 25:.1f} s"


def statuses(repos, tasks):
    return {t: json.loads((repos / "_status" / f"{t}.json").read_text()) for t in tasks}


def reader(args, tasks):
    os.environ["ROBODOJO_NORM_STATS"] = str(args.stats)
    os.environ["ROBODOJO_TEXT_CACHE"] = str(args.fake_cache or (D / "cache/text_embed/robodojo"))
    sys.path[:0] = [str(UP), str(UP / "src")]
    import hashlib

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from wam.training.config_validation import validate_training_config
    from wam.utils import misc
    from wam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    os.chdir(UP)  # run.sh runs from the upstream root
    work = D / "test" / "verify_work"
    work.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(work))
    dirs = ",".join(str(args.repos / t) for t in tasks)
    overrides = [  # the same as run_train.sh, minus launcher-only keys
        "task=robodojo", f"output_dir={work}", "seed=0",
        f"model.model_id={WAN}", f"model.tokenizer_model_id={WAN}", "model.redirect_common_files=false",
        f"model.tokenizer_max_len={CONTEXT_LEN}", f"data.train.context_len={CONTEXT_LEN}",
        f"data.train.dataset_dirs=[{dirs}]",
        "batch_size=8", "gradient_accumulation_steps=32", "max_steps=1550", "save_every=310", "+infra=online",
    ]
    with initialize_config_dir(config_dir=str(UP / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    validate_training_config(cfg)
    print("config ok: context_len", cfg.data.train.context_len, "val context_len", cfg.data.val.context_len,
          "stats", cfg.data.train.pretrained_norm_stats, "resume", cfg.resume,
          "global batch/GPU-count", cfg.batch_size * cfg.gradient_accumulation_steps)

    instructions = json.loads(PROMPTS.read_text())
    status = statuses(args.repos, tasks)
    if args.fake_cache:
        if "fake" not in args.fake_cache.name:
            raise SystemExit("--fake-cache directory name must contain 'fake'")
        args.fake_cache.mkdir(parents=True, exist_ok=True)
        needed = {p3(instructions, t, ep["seed"], f) for t in tasks for ep in status[t]["per_episode"] for f in range(ep["frames"])}
        for text in needed:
            prompt = DEFAULT_PROMPT.format(task=text)
            path = args.fake_cache / f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.t5_len{CONTEXT_LEN}.wan22ti2v5b.pt"
            if not path.exists():
                mask = torch.zeros(CONTEXT_LEN, dtype=torch.bool)
                mask[:8] = True
                torch.save({"context": torch.zeros(CONTEXT_LEN, 4096, dtype=torch.bfloat16), "mask": mask}, path)
        print(f"fake cache: {len(needed)} entries in {args.fake_cache}")

    ds = instantiate(cfg.data.train)
    subsets = ds.lerobot_dataset.multi_dataset._datasets
    assert len(subsets) == len(tasks), (len(subsets), len(tasks))
    assert ds.lerobot_dataset.fps == 25.0, ds.lerobot_dataset.fps
    total = 0
    expected_prompt = {}
    for task, sub in zip(tasks, subsets):
        hf = sub.hf_dataset
        tidx = np.asarray(hf["task_index"]).reshape(-1)
        fidx = np.asarray(hf["frame_index"]).reshape(-1)
        eidx = np.asarray(hf["episode_index"]).reshape(-1)
        seeds = [ep["seed"] for ep in status[task]["per_episode"]]
        bad = 0
        for k in range(len(tidx)):
            want = p3(instructions, task, seeds[int(eidx[k])], int(fidx[k]))
            bad += sub.meta.tasks[int(tidx[k])] != want
            expected_prompt[total + k] = DEFAULT_PROMPT.format(task=want)
        assert sub.meta.info["codebase_version"] == "v2.1" and sub.fps == 25
        print(f"{task}: {len(tidx)} frames, {len(sub.meta.tasks)} task strings, per-frame P3 mismatches: {bad}")
        assert bad == 0
        total += len(tidx)
    assert len(ds) == total, (len(ds), total)

    stride = max(1, total // max(1, args.samples - 1))
    for idx in sorted({0, 1, total - 1, *range(0, total, stride)})[: args.samples + 2]:
        data = ds._get(idx)
        sid = int(data["sample_id"])
        prompt_ok = data["prompt"] == expected_prompt[sid]
        z = max(float(data["action"].abs().max()), float(data["proprio"].abs().max()))
        print(f"sample {idx} (id {sid}): prompt ok {prompt_ok} ...{data['prompt'][-22:]!r} context {tuple(data['context'].shape)} "
              f"video {tuple(data['video'].shape)} action {tuple(data['action'].shape)} proprio {tuple(data['proprio'].shape)} "
              f"max|z| {z:.2f} vlm {tuple(data['vlm_current_images'].shape) if 'vlm_current_images' in data else None}")
        assert prompt_ok and tuple(data["context"].shape) == (CONTEXT_LEN, 4096) and z <= 5.0
    print("reader OK")


def pixels(args, tasks):
    sys.path.insert(0, os.path.expandvars("$ROBODOJO_DIR"))
    import av
    import h5py
    import numpy as np
    from XPolicyLab.utils.process_data import decode_image_bit

    cams = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
    import pandas as pd

    for task, st in statuses(args.repos, tasks).items():
        for ep_index, ep in enumerate(st["per_episode"]):
            h5 = args.farm / task / "arx_x5" / "data" / ep["file"]
            parquet = args.repos / task / "data" / "chunk-000" / f"episode_{ep_index:06d}.parquet"
            table = pd.read_parquet(parquet)
            state = np.stack(table["observation.state"].to_numpy())
            action = np.stack(table["action"].to_numpy())
            with h5py.File(h5, "r") as f:
                h5_state = np.concatenate([f[f"state/{p}"][()].reshape(len(state), -1) for p in
                                           ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")], 1)
                print(f"{task} ep{ep_index}: state == hdf5 {np.allclose(state, h5_state, atol=1e-6)}, "
                      f"action[t] == state[t+1] {np.allclose(action[:-1], state[1:], atol=1e-6)}")
                for cam, h5cam in cams.items():
                    video = args.repos / task / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep_index:06d}.mp4"
                    with av.open(str(video)) as container:
                        frames = [fr.to_ndarray(format="rgb24") for fr in container.decode(video=0)]
                    picks = sorted({0, len(frames) // 2, len(frames) - 1})
                    ref = np.asarray(decode_image_bit(f[f"vision/{h5cam}/colors"][picks]))
                    got = np.stack([frames[i] for i in picks]).astype(np.float64)
                    mse = ((got - ref) ** 2).mean()
                    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
                    print(f"   {cam}: {len(frames)} frames (hdf5 {len(state)}), PSNR {psnr:.1f} dB, "
                          f"RGB means video {np.round(got.mean((0, 1, 2)), 1)} hdf5 {np.round(ref.mean((0, 1, 2)), 1)}")
                    assert len(frames) == len(state) and psnr > 30


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["reader", "pixels"])
    parser.add_argument("--repos", type=Path, default=D / "lerobot")
    parser.add_argument("--tasks", required=True, help="comma list of converted tasks")
    parser.add_argument("--farm", type=Path, default=Path(os.path.expandvars("$TRAIN45_DIR/data/RoboDojo")))
    parser.add_argument("--stats", type=Path, default=D / "stats_train45.json")
    parser.add_argument("--fake-cache", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    (reader if args.mode == "reader" else pixels)(args, tasks)


if __name__ == "__main__":
    main()
