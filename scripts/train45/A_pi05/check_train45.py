"""End-to-end check of roboclock_train45_v1 through the exact pi05_roboclock_train45_time training transforms.

For a few frames: decode the prompt tokens back to text, save the 3 camera images the model gets
(before train-time augmentation), and invert normalization + delta to compare the action chunk with
the next 50 recorded states in the source HDF5.
"""

import dataclasses
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.expandvars("$PI05_TRAIN_DIR"))
import train_pi05_roboclock as T  # noqa: E402
import openpi.models.tokenizer as _tok  # noqa: E402
import openpi.training.data_loader as dl  # noqa: E402
import openpi.shared.normalize as _norm  # noqa: E402

OUT = Path(sys.orig_argv[-1])
OUT.mkdir(parents=True, exist_ok=True)
import os
cfg = getattr(T, os.environ.get("CFG", "train45_time"))
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
raw = dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
ds = dl.transform_dataset(raw, dc)
root = Path(os.path.expandvars("$PI05_TRAIN_DIR/lerobot/roboclock_train45_v1"))
files = json.loads((root / "roboclock_manifest.json").read_text())["files"]
ns = _norm.load(Path(os.path.expandvars("$PI05_TRAIN_DIR/assets_train45/arx_x5_sim")))
tok = _tok.PaligemmaTokenizer(cfg.model.max_token_len)
print(f"dataset frames {len(ds)}, episodes {len(files)}, model: action_horizon {cfg.model.action_horizon}, "
      f"action_dim {cfg.model.action_dim}, max_token_len {cfg.model.max_token_len}, pi05 {cfg.model.pi05}")

starts = np.cumsum([0] + [f["frames"] for f in files])
picks = {"lab_A1": 0.5, "hospital_B1": 0.6, "household_C1": 0.3, "industry_C3": 0.9, "sports_B4": 0.4}
PROMPTS = json.loads(Path(os.path.expandvars("$TRAIN45_DIR/prompts/instructions_main_4f541f2.json")).read_text())
for task, frac in picks.items():
    e = next(i for i, f in enumerate(files) if f"/{task}/" in f["file"] and "episode_00000" in f["file"])
    t = int(frac * files[e]["frames"])
    idx = int(starts[e] + t)
    s = ds[idx]
    r = raw[idx]
    assert int(r["episode_index"]) == e and int(r["frame_index"]) == t, (r["episode_index"], r["frame_index"], e, t)
    toks = np.asarray(s["tokenized_prompt"])[np.asarray(s["tokenized_prompt_mask"])]
    text = tok._tokenizer.decode(toks.tolist())
    print(f"\n=== {task} {Path(files[e]['file']).name} frame {t}/{files[e]['frames']} (dataset idx {idx})")
    print(f"prompt tokens {len(toks)} / {cfg.model.max_token_len}:\n{text}")
    row = []
    for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        img = np.asarray(s["image"][k])
        row.append(img)
        print(f"image {k}: {img.shape} {img.dtype} mask {bool(s['image_mask'][k])}")
    Image.fromarray(np.concatenate(row, axis=1)).save(OUT / f"{task}_f{t}.png")
    # invert: actions[:, :14] normalized delta -> delta -> absolute
    a = np.asarray(s["actions"])[:, :14]
    q01, q99 = ns["actions"].q01[:14], ns["actions"].q99[:14]
    delta = (a + 1) / 2 * (q99 - q01 + 1e-6) + q01
    state = np.asarray(r["observation.state"])
    absolute = delta.copy()
    m = np.r_[0:6, 7:13]
    absolute[:, m] += state[m]
    with h5py.File(files[e]["file"], "r") as h:
        def arm(side):
            j = h[f"state/{side}_arm_joint_states"][:]
            return np.concatenate([j, h[f"state/{side}_ee_joint_states"][:].reshape(len(j), 1)], 1)
        hs = np.concatenate([arm("left"), arm("right")], 1)
        hact = np.concatenate([h["action/left_arm_joint_states"][:], h["action/left_ee_joint_states"][:].reshape(-1, 1),
                               h["action/right_arm_joint_states"][:], h["action/right_ee_joint_states"][:].reshape(-1, 1)], 1)
    expect = hs[np.minimum(np.arange(t + 1, t + 51), len(hs) - 1)]
    print(f"state == hdf5 state[t]: max err {np.abs(state - hs[t]).max():.2e}")
    print(f"action chunk (50 x 14) vs hdf5 state[t+1 .. t+50]: max err {np.abs(absolute - expect).max():.2e}; "
          f"vs hdf5 action[t .. t+49]: {np.abs(absolute - hact[np.minimum(np.arange(t, t + 50), len(hact) - 1)]).max():.2e}")
    print(f"chunk motion: max |a[49]-state| over arm joints {np.abs(absolute[-1, m] - state[m]).max():.3f} rad; "
          f"grippers L {absolute[0, 6]:.2f}->{absolute[-1, 6]:.2f}  R {absolute[0, 13]:.2f}->{absolute[-1, 13]:.2f}")
    pad = np.asarray(s["actions"])[:, 14:]
    print(f"padded dims 14..31: all zero {bool(np.all(pad == 0))}; state padded dims zero {bool(np.all(np.asarray(s['state'])[14:] == 0))}")
    seed = int(Path(files[e]["file"]).stem.split("_")[-1])
    want = T.prompt_with_time(PROMPTS[f"{task}|{seed}"], t / T.DEMO_HZ)
    got = text.split("Task: ", 1)[1].split(", State:", 1)[0] if "Task: " in text else text
    print(f"prompt == relabel + time: {got.strip() == want.replace('_', ' ').strip()}; ends with Action: {text.rstrip().endswith('Action:')}")
