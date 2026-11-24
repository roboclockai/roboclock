"""X-VLA fine-tune on the roboclock five-domain set (docs/five-domain-training.md, role B).

Runs RoboDojo's vendored `xvla/train.py` unchanged, after three in-process changes; nothing under
$ROBODOJO_DIR is edited:

1. A `roboclock_ee` dataset handler, registered in the xvla handler registry. It subclasses
   `RoboDojoHandler` and keeps its reader (state ee poses and grippers, 30-step window, released
   index range), with two differences from `BaseHDF5Handler.iter_episode`:
   - every frame is kept: the static-segment `continue` (base.py:157) is gone;
   - the prompt is built per frame, P(task, seed, idx) = prompts["<task>|<seed>"] + " Time: {idx/25:.1f} s".
     The time uses the 25 Hz recording rate, not the handler's 30 Hz `ref[idx]`.
   The meta file names the prompt file (`prompts_path`). The dataset name is not in
   `DATA_DOMAIN_ID`, so it gets domain 0 like `RoboDojo_ee`.
2. `XVLAProcessor.language_max_length` = 224 (released: 50), so the longest prompt (186 tokens)
   keeps its time suffix. The policy server needs the same value.
3. `XVLA_SKIP_SAVE=1` (smoke runs) turns checkpoint writing into creating an empty `ckpt-N`
   directory; `train.py` always saves at the last step.

Every `train.py` flag passes through:

  accelerate launch ... scripts/train45/B_xvla/train_roboclock.py --models ... --train_metas_path ...

Before training, each episode in the meta's datalist must map to a prompt; otherwise this
exits. The loader would otherwise skip the episode quietly (dataset.py:144, error_log.txt).
"""

import json
import os
import random
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

XVLA_DIR = Path(
    os.environ.get("XVLA_DIR", os.path.expandvars("$ROBODOJO_DIR/XPolicyLab/policy/X_VLA/xvla"))
).resolve()
# train.py imports `datasets` and `models` from its own directory. Python puts the wrapper's
# directory there instead, so add it; the modules patched below are then the ones train.py uses.
sys.path.insert(0, str(XVLA_DIR))

from datasets.dataset import _expand_datalist  # noqa: E402
from datasets.domain_handler import registry  # noqa: E402
from datasets.domain_handler.base import _open_h5  # noqa: E402
from datasets.domain_handler.simulations import RoboDojoHandler  # noqa: E402
from models.modeling_xvla import XVLA  # noqa: E402
from models.processing_xvla import XVLAProcessor  # noqa: E402

DATASET_NAME = "roboclock_ee"
LANGUAGE_MAX_LENGTH = 224
FRAME_HZ = 25.0
EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$")


def episode_key(datapath):
    """`<task>|<seed>` from `.../<task>/arx_x5/data/episode_<seed:07d>.hdf5`."""
    path = Path(datapath)
    match = EPISODE_RE.search(path.name)
    if match is None or path.parent.name != "data" or path.parent.parent.name != "arx_x5":
        raise ValueError(f"not a roboclock demo path: {datapath}")
    return f"{path.parent.parent.parent.name}|{int(match.group(1))}"


def prompt_at(base, idx):
    """P(task, seed, frame) of docs/five-domain-training.md section 3.2 (P3)."""
    return base + f" Time: {idx / FRAME_HZ:.1f} s"


def load_prompts(meta):
    with open(meta["prompts_path"]) as f:
        return json.load(f)


class RoboclockHandler(RoboDojoHandler):
    dataset_name = DATASET_NAME

    def __init__(self, meta, num_views):
        super().__init__(meta, num_views)
        self.prompts = load_prompts(meta)

    def iter_episode(self, traj_idx, *, num_actions, training, image_aug, lang_aug_map, **kwargs):
        # BaseHDF5Handler.iter_episode (base.py:109-172) with a per-frame prompt and no static skip.
        datapath = self.meta["datalist"][traj_idx]
        if not isinstance(datapath, str):
            datapath = datapath[0]
        base = self.prompts[episode_key(datapath)]

        with _open_h5(datapath) as f:
            freq = float(np.asarray(f["additional_info/frequency"][()]))
            if freq != FRAME_HZ:
                raise ValueError(f"{datapath}: additional_info/frequency {freq}, expected {FRAME_HZ}")
            images = self.get_image_datasets(f)
            left, right, lt, rt, hfreq, qdur = self.build_left_right(f)

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: len(images)] = True
        # The released 30 Hz / 1 s window: step k is frame idx + k (end of episode: compressed).
        if lt is None:
            lt = np.arange(left.shape[0], dtype=np.float64) / float(hfreq)
        if rt is None:
            rt = np.arange(right.shape[0], dtype=np.float64) / float(hfreq)

        idxs = list(self.index_candidates(left.shape[0], training))
        if training:
            random.shuffle(idxs)

        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        ref = (lt + rt) / 2.0

        V = min(self.num_views, len(images))
        for idx in idxs:
            cur = ref[idx]
            q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
            lseq = torch.tensor(L(q))
            rseq = torch.tensor(R(q))

            imgs = [image_aug(self._pil_from_arr(images[v][idx])) for v in range(V)]
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            yield {
                "language_instruction": prompt_at(base, idx),
                "image_input": torch.stack(imgs, dim=0),
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([lseq, rseq], -1).float(),
            }


def register():
    registry._REGISTRY[DATASET_NAME] = RoboclockHandler
    XVLAProcessor.language_max_length = LANGUAGE_MAX_LENGTH
    if os.environ.get("XVLA_SKIP_SAVE", "0") not in ("", "0"):

        def skip_save(self, save_directory, *args, **kwargs):
            os.makedirs(save_directory, exist_ok=True)
            print(f"[train_roboclock] XVLA_SKIP_SAVE: no weights written to {save_directory}", flush=True)

        XVLA.save_pretrained = skip_save


def check_meta(meta_path):
    """Exit unless every datalist episode has a prompt."""
    with open(meta_path) as f:
        meta = _expand_datalist(json.load(f))
    if meta.get("dataset_name") != DATASET_NAME:
        sys.exit(f"[train_roboclock] {meta_path}: dataset_name must be {DATASET_NAME!r}")
    prompts = load_prompts(meta)
    missing = [p for p in meta["datalist"] if episode_key(p) not in prompts]
    if missing or not meta["datalist"]:
        sys.exit(f"[train_roboclock] {len(missing)} of {len(meta['datalist'])} episodes have no prompt, e.g. {missing[:3]}")
    print(f"[train_roboclock] {len(meta['datalist'])} episodes, all with prompts "
          f"({meta['prompts_path']}); language_max_length {LANGUAGE_MAX_LENGTH}", flush=True)


def metas_path_from_argv(argv):
    for i, arg in enumerate(argv):
        if arg == "--train_metas_path" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--train_metas_path="):
            return arg.split("=", 1)[1]
    return None


def main():
    register()
    meta_path = metas_path_from_argv(sys.argv[1:])
    if meta_path is None:
        sys.exit("[train_roboclock] --train_metas_path is required")
    check_meta(meta_path)
    runpy.run_path(str(XVLA_DIR / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
