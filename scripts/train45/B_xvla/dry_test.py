"""CPU check of the role-B data path (train_roboclock.py), without training.

1. Prompts: samples of a few episodes, in frame order, equal P(task, seed, idx).
2. Static frames are kept: every candidate frame of a lab_B4 episode yields a sample. The
   released RoboDojo_ee handler is run on the same episode for comparison.
3. The longest training prompt (largest instruction at its episode's last frame) is not truncated
   by the processor at language_max_length 224.
4. Shapes and targets: a DataLoader batch (4 workers, as in training) and action == next state.
5. With --forward: one forward pass of the init checkpoint on that batch (CPU, fp32) prints the
   first losses. They should be low; random-init level means a wrong init.

  nice -n 10 $ROBODOJO_DIR/XPolicyLab/policy/X_VLA/.venv/bin/python scripts/train45/B_xvla/dry_test.py [--forward]
"""

import os
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_roboclock as tr  # noqa: E402  (puts xvla/ on sys.path)

from datasets import create_dataloader  # noqa: E402
from datasets.dataset import InfiniteDataReader, _expand_datalist  # noqa: E402
from datasets.domain_handler.simulations import RoboDojoHandler  # noqa: E402
from datasets.utils import quat_wxyz_to_rotate6d  # noqa: E402
from models.processing_xvla import XVLAProcessor  # noqa: E402

WORK = Path(os.path.expandvars("$TRAIN45_DIR/B_xvla"))


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        check.failed += 1


check.failed = 0


def episode_index(meta, task, seed):
    suffix = f"/{task}/arx_x5/data/episode_{seed:07d}.hdf5"
    return next(i for i, p in enumerate(meta["datalist"]) if p.endswith(suffix))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(WORK / "roboclock45.json"))
    parser.add_argument("--init", default=str(WORK / "init_ckpt100k"))
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()

    tr.register()
    tr.check_meta(args.meta)
    meta = _expand_datalist(json.loads(Path(args.meta).read_text()))
    prompts = tr.load_prompts(meta)
    reader = InfiniteDataReader(args.meta, num_actions=30, training=False, action_mode="ee6d")
    handler = tr.RoboclockHandler(meta=meta, num_views=3)
    common = dict(num_actions=30, training=False, image_aug=reader.image_aug, lang_aug_map=None)

    # 1 + 2: prompts per frame, every frame kept
    for task, seed in [("lab_B4", 0), ("hospital_C1", 50), ("industry_C3", 55), ("sports_B1", 0)]:
        i = episode_index(meta, task, seed)
        path = meta["datalist"][i]
        with h5py.File(path, "r") as f:
            T = f["state/left_ee_poses"].shape[0]
        samples = list(handler.iter_episode(i, **common))
        want = [f"{prompts[f'{task}|{seed}']} Time: {idx / 25:.1f} s" for idx in range(T - 5)]
        got = [s["language_instruction"] for s in samples]
        check(got == want, f"{task}|{seed}: {len(got)} samples = T-5 = {T - 5}, prompts equal P for every idx "
                           f"(idx {T - 6}: ...{got[-1][-40:]!r})")
        if task == "lab_B4":
            static = sum(float((s["abs_trajectory"][1] - s["abs_trajectory"][0]).abs().max()) < 1e-5 for s in samples)
            first_static = [k for k, s in enumerate(samples[:25])
                            if float((s["abs_trajectory"][1] - s["abs_trajectory"][0]).abs().max()) < 1e-5]
            stock = RoboDojoHandler(meta=dict(meta, language_instruction_key="instruction"), num_views=3)
            n_stock = sum(1 for _ in stock.iter_episode(i, **common))
            check(static > 0 and len(samples) - n_stock == static,
                  f"lab_B4|0: {static} static frames kept (released handler yields {n_stock} of {len(samples)}); "
                  f"static among idx 0-24: {first_static}")

    # 3: the longest prompt survives the processor
    processor = XVLAProcessor.from_pretrained(args.init)
    tok = processor.tokenizer
    longest, longest_len = None, 0
    for path in meta["datalist"]:
        key = tr.episode_key(path)
        with h5py.File(path, "r") as f:
            T = f["state/left_ee_poses"].shape[0]
        text = tr.prompt_at(prompts[key], T - 6)  # last training frame of the episode
        n = len(tok(text)["input_ids"])
        if n > longest_len:
            longest, longest_len, longest_key = text, n, key
    ids = processor.encode_language([longest])["input_ids"][0]
    n_real = int((ids != tok.pad_token_id).sum())
    decoded = tok.decode(ids, skip_special_tokens=True)
    check(processor.language_max_length == 224 and n_real == longest_len and decoded == longest,
          f"longest prompt {longest_key} ({longest_len} tokens incl. <s></s>) not truncated at "
          f"{processor.language_max_length}; ends {decoded[-30:]!r}")

    # 4: DataLoader batch (training mode, 4 forked workers) and targets
    loader = create_dataloader(batch_size=4, metas_path=args.meta, num_actions=30, training=True, action_mode="ee6d")
    batch = next(iter(loader))
    shapes = {k: tuple(v.shape) if hasattr(v, "shape") else len(v) for k, v in batch.items()}
    print("batch:", shapes, "domain_id", batch["domain_id"].tolist(), "image_mask[0]", batch["image_mask"][0].tolist())
    check(shapes["image_input"] == (4, 3, 3, 224, 224) and shapes["proprio"] == (4, 20)
          and shapes["action"] == (4, 30, 20) and batch["domain_id"].tolist() == [0] * 4
          and batch["image_mask"][0].tolist() == [True, False, False]
          and all(p.split(" Time: ")[-1].endswith(" s") for p in batch["language_instruction"]),
          "batch shapes, domain 0, one view, prompts end in 'Time: … s'")
    for p in batch["language_instruction"]:
        print("  prompt:", p[:60], "...", p[-25:])

    i = episode_index(meta, "lab_B4", 0)
    with h5py.File(meta["datalist"][i], "r") as f:
        st = {s: np.concatenate([f[f"state/{s}_ee_poses"][:, :3], quat_wxyz_to_rotate6d(f[f"state/{s}_ee_poses"][:, 3:]),
                                 f[f"state/{s}_ee_joint_states"][()]], -1) for s in ("left", "right")}
        act_l = f["action/left_ee_poses"][()]
    x = np.concatenate([st["left"], st["right"]], -1)
    idx = 100
    sample = next(s for k, s in enumerate(handler.iter_episode(i, **common)) if k == idx)
    traj = sample["abs_trajectory"].numpy()
    err = np.abs(traj - x[idx: idx + 31]).max()
    err_act = np.abs(traj[1, :3] - act_l[idx, :3]).max()
    check(err < 1e-4 and err_act < 1e-3,
          f"lab_B4|0 idx {idx}: proprio = state[idx], action[k] = state[idx+1+k] (max err {err:.1e}); "
          f"action[0] xyz vs recorded action[idx] {err_act:.1e} m")

    # 5: one forward pass
    if args.forward:
        from models.modeling_xvla import XVLA

        torch.set_num_threads(16)
        model = XVLA.from_pretrained(args.init).eval()
        lang = processor.encode_language(batch["language_instruction"])
        inputs = {k: v for k, v in batch.items() if k != "language_instruction"}
        with torch.no_grad():
            enc = model.forward_vlm(lang["input_ids"], inputs["image_input"], inputs["image_mask"])
            seq = 30 + enc["vlm_features"].shape[1] + enc["aux_visual_inputs"].shape[1]
            losses = model(**inputs, **lang)
        print("forward:", {k: round(float(v), 4) for k, v in losses.items()}, f"sequence {seq} of max_len_seq 512")
        check(seq <= 512, f"sequence length {seq} <= 512 at language_max_length 224")

    print("FAILED" if check.failed else "ALL PASS", check.failed)
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
