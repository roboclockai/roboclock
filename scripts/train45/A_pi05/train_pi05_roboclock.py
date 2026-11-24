"""openpi pi0.5 fine-tuning on the roboclock oracle demos, without editing the XPolicyLab submodule.

Registers two configs next to RoboDojo's `pi05_base_aloha_full_sim_arx-x5_seed_0` and hands control to
openpi's own scripts/train.py (tyro overrides work as usual, e.g. --exp-name, --num-train-steps):

  pi05_roboclock_arx-x5   RoboDojo's recipe (batch 256, fsdp 2, cosine LR, EMA 0.99, delta joint actions),
                        initialised from the RoboDojo arx_x5 pi0.5 checkpoint (60k steps on RoboDojo's 54
                        tasks) and normalised with that checkpoint's own norm stats, so the fine-tune starts
                        in the space the weights were trained in.
  pi05_bench_fake       same model and batch on openpi's FakeDataset, random init: step-time benchmark only.

Run (openpi venv), e.g.:
  HF_LEROBOT_HOME=$PI05_TRAIN_DIR/lerobot PYTHONPATH=$ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/src \
  .venv/bin/python $PI05_TRAIN_DIR/train_pi05_roboclock.py pi05_roboclock_arx-x5 --exp-name roboclock810 ...
"""

import os
import dataclasses
import sys
from pathlib import Path

OPENPI = Path(os.path.expandvars("$ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi"))
INIT_CKPT = Path(os.path.expandvars("$ROBODOJO_DIR/hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"))
sys.path.insert(0, str(OPENPI / "scripts"))

import openpi.training.config as _config  # noqa: E402
import openpi.training.weight_loaders as _wl  # noqa: E402

base = _config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")

roboclock = dataclasses.replace(
    base,
    name="pi05_roboclock_arx-x5",
    data=dataclasses.replace(
        base.data,
        repo_id="roboclock_oracle_v1_810",
        assets=_config.AssetsConfig(assets_dir=str(INIT_CKPT / "assets"), asset_id="arx_x5_sim"),
    ),
    weight_loader=_wl.CheckpointWeightLoader(str(INIT_CKPT / "params")),
    checkpoint_base_dir=os.path.expandvars("$PI05_TRAIN_DIR/checkpoints"),
    assets_base_dir=os.path.expandvars("$PI05_TRAIN_DIR/assets"),
    num_workers=16,
    keep_period=10000,
    wandb_enabled=False,
)

import openpi.training.optimizer as _optimizer  # noqa: E402

# Lab only (user, 2026-11-11). Prompts relabelled to origin/main gen_instruction() (C1-C3 changed 11-09);
# state quantiles widened to cover lab (assets_lab, actions unchanged). 5000 steps x 256 = 5.2 passes over
# 246k frames; cosine decays over the run; every 1000-step checkpoint is kept for evaluation.
lab = dataclasses.replace(
    roboclock,
    name="pi05_roboclock_lab",
    data=dataclasses.replace(
        roboclock.data,
        repo_id="roboclock_lab_v2",
        assets=_config.AssetsConfig(assets_dir=os.path.expandvars("$PI05_TRAIN_DIR/assets_lab"), asset_id="arx_x5_sim"),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=2.5e-5, decay_steps=5000, decay_lr=2.5e-6),
    num_train_steps=5000,
    save_interval=1000,
    keep_period=1000,
)

import numpy as np  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402

# Roboclock demos: HDF5 frame k is captured k * 2 control steps after reset, i.e. at sim time k / 25 s
# (scripts/collect/record_demo.py; the final state-only frame sits at frames * 0.04 = sidecar sim_time).
DEMO_HZ = 25.0


def prompt_with_time(instruction: str, sim_time_s: float) -> str:
    """The prompt the policy is trained and queried with: the task instruction plus the env time.

    The roboclock eval client must send exactly this (instruction + current observation sim_time).
    TokenizePrompt then builds "Task: <this>, State: <bins>;\\nAction: ".
    """
    return f"{instruction} Time: {sim_time_s:.1f} s"


@dataclasses.dataclass(frozen=True)
class AppendTimeToPrompt(_transforms.DataTransformFn):
    hz: float = DEMO_HZ

    def __call__(self, data: dict) -> dict:
        frame = int(np.asarray(data.pop("frame_index")))
        data["prompt"] = prompt_with_time(str(data["prompt"]), frame / self.hz)
        return data


_repack = lab.data.repack_transforms.inputs[0]
lab_time = dataclasses.replace(
    lab,
    name="pi05_roboclock_lab_time",
    data=dataclasses.replace(
        lab.data,
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform({**_repack.structure, "frame_index": "frame_index"}),
                AppendTimeToPrompt(),
            ]
        ),
    ),
)

# Five domains, 45 training tasks (roboclock docs/five-domain-training.md, role A; user 2026-11-14).
# Same recipe as lab_time; only what scales with the data changes:
#   data   roboclock_train45_v1 = 610 episodes / 610,098 frames of the 45 tasks (15 held out), prompts relabelled
#          to roboclock main 4f541f2 gen_instruction() (scripts/train45/gen_instructions.py) + " Time: X.X s";
#   norm   assets_train45 = RoboDojo stats, state q01/q99 widened to the 45-task min/max (lab rule;
#          scripts/train45/A_pi05/widen_norm_stats.py reproduces assets_lab exactly);
#   steps  12,400 x 256 = 3,174,400 samples = 5.2 passes, the lab run's passes; cosine decays over the run;
#   save   every 2,500 (5 checkpoints, each 640k samples apart, same spacing as roles B-D);
#   tokens max_token_len 200 -> 288: the new instructions are longer, prompt + time + state reaches 269
#          tokens (industry C3; 8 of 60 tasks exceed 200, which would truncate the state and "Action:").
#          The policy server must use this config too, or it truncates at 200.
train45_time = dataclasses.replace(
    lab_time,
    name="pi05_roboclock_train45_time",
    model=dataclasses.replace(lab_time.model, max_token_len=288),
    data=dataclasses.replace(
        lab_time.data,
        repo_id="roboclock_train45_v1",
        assets=_config.AssetsConfig(assets_dir=os.path.expandvars("$PI05_TRAIN_DIR/assets_train45"), asset_id="arx_x5_sim"),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=2.5e-5, decay_steps=12400, decay_lr=2.5e-6),
    num_train_steps=12400,
    save_interval=2500,
    keep_period=2500,
)

bench = dataclasses.replace(
    roboclock,
    name="pi05_bench_fake",
    data=_config.FakeDataConfig(),
    weight_loader=_wl.NoOpWeightLoader(),
    num_train_steps=40,
    save_interval=10**9,
    log_interval=5,
    checkpoint_base_dir=os.path.expandvars("$PI05_TRAIN_DIR/bench_ckpt"),
)

for cfg in (roboclock, lab, lab_time, train45_time, bench):
    _config._CONFIGS_DICT[cfg.name] = cfg

if __name__ == "__main__":
    import train as _train

    _train.main(_config.cli())
