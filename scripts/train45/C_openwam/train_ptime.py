"""Role C (OpenWAM-alpha) training entry point: upstream scripts/train.py with the train45 prompt.

docs/five-domain-training.md, rule 4 and section 6. The only change to upstream training is
the text prompt of each sample. Upstream's RoboDojo reader uses the episode's recorded HDF5
`instruction`; this wrapper replaces it with

    P(task, seed, frame) = prompts[f"{task}|{seed}"] + f" Time: {frame / 25:.1f} s"

where frame is the window `start` (the current observation frame), and task and seed come from
the episode path `<farm>/<task>/arx_x5/data/episode_<seed>.hdf5`. The reader's own
"A video recorded from a robot's point of view ..." prefix is kept, as at deploy time, where the
adapter wraps the client's "<instruction> Time: <sim_time:.1f> s" the same way.

It also installs fast_pp.py, a bit-identical replacement for upstream's per-frame video
preprocessing (one uint8 host-to-device copy per clip instead of one float32 copy per frame),
which takes the step from 3.35 s to 1.29 s on 8 x A100; it changes no tensor (test_fast_pp.py).

Everything else, including the Hydra arguments, passes through to upstream unchanged:

    torchrun --nnodes 1 --master_addr 127.0.0.1 --master_port 29521 --nproc_per_node 8 train_ptime.py dataloader=robodojo ...

DataLoader workers are forked, so they inherit the patch. Environment:
  OPENWAM_ROOT     vendored OpenWAM tree (default: RoboDojo's XPolicyLab/policy/OpenWAM/OpenWAM)
  TRAIN45_PROMPTS  prompt JSON (default: $TRAIN45_DIR/prompts/instructions_main_4f541f2.json)
  TRAIN45_FAST_PP  0 = keep upstream's per-frame preprocessing (default 1)
"""

import json
import os
import re
import runpy
import sys
from pathlib import Path

OPENWAM_ROOT = Path(os.environ.get("OPENWAM_ROOT", os.path.expandvars("$ROBODOJO_DIR/XPolicyLab/policy/OpenWAM/OpenWAM")))
PROMPTS = Path(os.environ.get("TRAIN45_PROMPTS", os.path.expandvars("$TRAIN45_DIR/prompts/instructions_main_4f541f2.json")))
FPS = 25.0  # HDF5 frame k is sim time k / 25 s (additional_info/frequency, RoboDojo collect_freq)
EPISODE_RE = re.compile(r"^episode_(\d+)\.hdf5$")


def episode_key(path):
    """`<task>|<seed>` from `<root>/<task>/<embodiment>/data/episode_<seed>.hdf5`."""
    path = Path(path)
    match = EPISODE_RE.match(path.name)
    if match is None or path.parent.name != "data":
        raise ValueError(f"not a formal RoboDojo episode path: {path}")
    return f"{path.parents[2].name}|{int(match.group(1))}"


def instruction_with_time(prompts, path, frame):
    """The train45 instruction for this episode at HDF5 frame `frame` (before the reader's prefix)."""
    return f"{prompts[episode_key(path)]} Time: {frame / FPS:.1f} s"


def load_prompts(path=PROMPTS):
    with open(path) as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError(f"{path}: expected a non-empty {{'<task>|<seed>': instruction}} object")
    return prompts


def install_prompt_patch(prompts=None):
    """Patch RoboDojoDataset so every sample's prompt is the reader prefix + P(task, seed, start)."""
    import openwam
    import openwam.dataloader.robodojo as robodojo
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference

    # train.py imports openwam from whatever sys.path resolves; patching a different copy
    # would silently train on the recorded instructions.
    loaded = Path(openwam.__file__).resolve()
    if OPENWAM_ROOT.resolve() not in loaded.parents:
        raise RuntimeError(f"openwam is imported from {loaded}, not from OPENWAM_ROOT={OPENWAM_ROOT}")
    dataset_cls = robodojo.RoboDojoDataset
    if getattr(dataset_cls, "_train45_prompts", None) is not None:
        return dataset_cls._train45_prompts
    prompts = load_prompts() if prompts is None else prompts
    original_init = dataset_cls.__init__
    original_build = dataset_cls._build_sample

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Fail at construction, not at a random training step, when an episode has no prompt.
        missing = [path for path in self._episode_files if episode_key(path) not in prompts]
        if missing:
            raise KeyError(f"{len(missing)} episode(s) of task {self.task_name} have no entry in {PROMPTS}: {missing[:3]}")
        for path in self._episode_files:
            if Path(path).parents[2].name != self.task_name:
                raise ValueError(f"episode path {path} does not belong to task {self.task_name}")

    def _build_sample(self, episode_index, start):
        sample = original_build(self, episode_index, start)
        path = self._episode_files[episode_index]
        sample["prompt"] = format_prompt_for_inference(instruction_with_time(prompts, path, start))
        return sample

    dataset_cls.__init__ = __init__
    dataset_cls._build_sample = _build_sample
    dataset_cls._train45_prompts = prompts
    return prompts


def main():
    train_py = OPENWAM_ROOT / "scripts" / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"upstream OpenWAM train.py not found: {train_py}")
    prompts = install_prompt_patch()
    fast = os.environ.get("TRAIN45_FAST_PP", "1") != "0"
    if fast:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fast_pp

        fast_pp.install()
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[train_ptime] {len(prompts)} prompts from {PROMPTS}; prompt = prefix + P(task, seed, start)", flush=True)
        print(f"[train_ptime] fast_pp (bit-identical video preprocessing): {'on' if fast else 'off'}", flush=True)
    sys.argv[0] = str(train_py)
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
