"""CPU dry test of role C's training inputs, through train_ptime's patch (no GPU, no training).

Checks, with evidence written to $TRAIN45_DIR/C_openwam/cpu_dry_test.json:
1. Hydra accepts run_train.sh's overrides through the wrapper (`train_ptime.py ... --cfg job`
   prints the composed config and exits before any training code runs).
2. The dataset, built from that config, uses the widened stats file in <farm>/meta/ unchanged
   (no rebuild: same sha256 and mtime, no "computing" message, min/max equal the file's).
3. Prompts equal the reader prefix + P(task, seed, start) in the main process and in forked
   DataLoader workers.
4. At the most extreme frames of every widened dim, and at the lowest-z frame of the five
   lowest-z tasks, the unclipped normalised state and action stay within [-1, 1] and equal the
   dataset's values (so nothing was clipped); with the release stats they would leave [-1, 1].
5. The UMT5 tokenizer (the checkpoint's, seq_len 512) does not truncate the longest prompt of
   any episode (its last window, i.e. the longest time string).

  nice -n 10 $ROBODOJO_DIR/XPolicyLab/policy/OpenWAM/.venv/bin/python cpu_dry_test.py
"""

import contextlib
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_ptime  # noqa: E402

FARM = Path(os.path.expandvars("$TRAIN45_DIR/data/RoboDojo"))
STATS = FARM / "meta" / "robodojo_normalization_stats.npy"
RELEASE = Path(os.path.expandvars("$MODELS_DIR/OpenWAM-Alpha-Sim-RoboDojo/normalization_stats.npy"))
TOKENIZER = Path(os.path.expandvars("$MODELS_DIR/OpenWAM-Alpha-Sim-RoboDojo/tokenizer/google/umt5-xxl"))
OUT = Path(os.path.expandvars("$TRAIN45_DIR/C_openwam/cpu_dry_test.json"))
OVERRIDES = [
    "dataloader=robodojo",
    f"dataloader.dataset_dir={FARM}",
    os.path.expandvars("training.finetune_ckpt_path=$MODELS_DIR/OpenWAM-Alpha-Sim-RoboDojo"),
    os.path.expandvars("training.output_path=$TRAIN45_DIR/C_openwam/run1"),
    os.path.expandvars("hydra.run.dir=$TRAIN45_DIR/C_openwam/run1/hydra/${now:%Y-%m-%d_%H-%M-%S}"),
    "training.batch_size=4",
    "training.num_epochs=null",
    "training.use_gradient_checkpointing=false",
    "training.max_steps=99200",
    "training.save_steps=20000",
    "training.keep_last_k_ckpts=5",
    "project.seed=42",
]
UNIFY = np.r_[0:10, 34:44]
LOW_Z_TASKS = 5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unclipped(x, stats):
    # apply_normalization's min-max without its final clip.
    from openwam.dataloader.utils.normalization import NORM_EPS

    scale = np.maximum(stats["max"] - stats["min"], NORM_EPS)
    return ((x - stats["min"]) / scale) * 2.0 - 1.0


def hydra_check():
    cmd = [sys.executable, str(HERE / "train_ptime.py"), *OVERRIDES, "--cfg", "job"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENWAM_ROOT=str(train_ptime.OPENWAM_ROOT))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=train_ptime.OPENWAM_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"{shlex.join(cmd)} failed:\n{proc.stderr[-3000:]}")
    from omegaconf import OmegaConf

    banner = [line for line in proc.stdout.splitlines() if line.startswith("[train_ptime]")]
    cfg = OmegaConf.create("\n".join(line for line in proc.stdout.splitlines() if not line.startswith("[train_ptime]")))
    t = cfg.training
    got = {
        "max_steps": t.max_steps, "num_epochs": t.num_epochs, "batch_size": t.batch_size,
        "use_gradient_checkpointing": t.use_gradient_checkpointing, "save_steps": t.save_steps,
        "keep_last_k_ckpts": t.keep_last_k_ckpts, "learning_rate": t.learning_rate,
        "warmup_ratio": t.warmup_ratio, "lr_min_ratio": t.lr_min_ratio, "zero_stage": t.zero_stage,
        "mixed_precision": t.mixed_precision, "dataset_dir": cfg.dataloader.dataset_dir,
        "finetune_ckpt_path": t.finetune_ckpt_path,
    }
    assert got["max_steps"] == 99200 and got["num_epochs"] is None and got["batch_size"] == 4
    assert got["use_gradient_checkpointing"] is False and got["save_steps"] == 20000 and got["keep_last_k_ckpts"] == 5
    assert banner, "wrapper banner missing"
    got["wrapper_banner"] = banner[0]
    return got, cfg


def main():
    prompts = train_ptime.install_prompt_patch()
    from openwam.dataloader.registry import build_dataset
    from openwam.dataloader.robodojo import read_calibrated_eef20, resolve_robodojo_calibration
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference

    evidence = {}
    hydra_cfg, cfg = hydra_check()
    evidence["hydra_cfg_job"] = hydra_cfg

    # 2. Dataset from the composed config, stats file untouched.
    before = (sha256(STATS), STATS.stat().st_mtime_ns)
    from omegaconf import OmegaConf

    OmegaConf.update(cfg.dataloader, "seed", int(cfg.project.seed), force_add=True)  # train.py _inject_project_seed
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        ds = build_dataset(cfg.dataloader, split="train")
    after = (sha256(STATS), STATS.stat().st_mtime_ns)
    merged = np.load(STATS, allow_pickle=True).item()["eef"]
    release = np.load(RELEASE, allow_pickle=True).item()["eef"]
    report = json.loads(Path(os.path.expandvars("$TRAIN45_DIR/C_openwam/normalization_widening_report.json")).read_text())
    stats = ds.normalization_stats
    assert ds.normalization_stats_path == str(STATS), ds.normalization_stats_path
    assert before == after and before[0] == report["sha256"]
    assert "computing" not in captured.getvalue(), captured.getvalue()
    assert np.array_equal(stats["min"], merged["min"]) and np.array_equal(stats["max"], merged["max"])
    assert len(ds.tasks) == 45 and len(ds) == 610098 - 610, (len(ds.tasks), len(ds))
    evidence["dataset"] = {
        "class": type(ds).__name__, "tasks": len(ds.tasks), "windows": len(ds),
        "stats_path": ds.normalization_stats_path, "stats_sha256": before[0],
        "stats_unchanged": before == after, "rebuild_message": "computing" in captured.getvalue(),
    }

    # 4. Extreme windows: every widened dim's extreme frame, and the lowest-z frame per task.
    cal = resolve_robodojo_calibration(None)
    sub = {d.task_name: d for d in ds._sub_datasets}
    extreme = {}  # dim -> (value, task, episode_index, frame)
    task_min_z = {}
    widened = [d for d in report["dims"] if d["widened_low"] > 0 or d["widened_high"] > 0]
    for task, d in sub.items():
        for ei, path in enumerate(d._episode_files):
            x = read_calibrated_eef20(path, cal)
            for w in widened:
                i, low = w["dim"], w["widened_low"] > 0
                frame = int(x[:, i].argmin() if low else x[:, i].argmax())
                value = float(x[frame, i])
                best = extreme.get(i)
                if best is None or (value < best[0] if low else value > best[0]):
                    extreme[i] = (value, task, ei, frame)
            z = np.minimum(x[:, 2], x[:, 12])
            frame = int(z.argmin())
            if task not in task_min_z or z[frame] < task_min_z[task][0]:
                task_min_z[task] = (float(z[frame]), task, ei, frame)
    windows = []
    for i, (value, task, ei, frame) in sorted(extreme.items()):
        length = sub[task]._episode_lengths[ei]
        windows.append((f"extreme {report['dims'][i]['name']}={value:+.4f} (proprio)", task, ei, min(frame, length - 2)))
        windows.append((f"extreme {report['dims'][i]['name']}={value:+.4f} (action t+16)", task, ei, max(frame - 16, 0)))
    for value, task, ei, frame in sorted(task_min_z.values())[:LOW_Z_TASKS]:
        length = sub[task]._episode_lengths[ei]
        windows.append((f"lowest z {value:+.4f} of {task}", task, ei, max(min(frame - 8, length - 2), 0)))

    checks = []
    for label, task, ei, start in windows:
        d = sub[task]
        sample = d._build_sample(ei, start)
        path = d._episode_files[ei]
        n = min(d.num_frames, d._episode_lengths[ei] - start)
        raw = read_calibrated_eef20(path, cal, start, start + n)
        raw = np.concatenate([raw, np.repeat(raw[-1:], d.num_frames - n, axis=0)])
        ours_n, rel_n = unclipped(raw, stats), unclipped(raw, {k: np.asarray(release[k]) for k in ("min", "max")})
        got = np.concatenate([sample["proprio"].numpy(), sample["action"].numpy()])[:, UNIFY]
        key = train_ptime.episode_key(path)
        expected = format_prompt_for_inference(f"{prompts[key]} Time: {start / 25:.1f} s")
        entry = {
            "label": label, "task": task, "key": key, "start": start,
            "prompt_ok": sample["prompt"] == expected, "prompt_tail": sample["prompt"][-70:],
            "unclipped_range_merged": [float(ours_n.min()), float(ours_n.max())],
            "unclipped_range_release": [float(rel_n.min()), float(rel_n.max())],
            "dataset_equals_unclipped": bool(np.allclose(got, ours_n, atol=2e-6)),
        }
        assert entry["prompt_ok"], (sample["prompt"], expected)
        assert -1 - 1e-6 <= ours_n.min() and ours_n.max() <= 1 + 1e-6, entry
        assert entry["dataset_equals_unclipped"], entry
        checks.append(entry)
    evidence["windows"] = checks
    evidence["release_would_clip"] = sum(1 for c in checks if min(c["unclipped_range_release"]) < -1 or max(c["unclipped_range_release"]) > 1)

    # 3b. Prompts in forked DataLoader workers (as training: collate_fn=list).
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=list,
                                         generator=torch.Generator().manual_seed(0))
    batch = next(iter(loader))
    worker = []
    for s in batch:
        expected = format_prompt_for_inference(f"{prompts[train_ptime.episode_key(s['episode_path'])]} Time: {s['start_frame'] / 25:.1f} s")
        assert s["prompt"] == expected
        assert len(s["video"]) == 9 and s["video"][0].size == (320, 384) and tuple(s["action"].shape) == (32, 80)
        worker.append({"key": train_ptime.episode_key(s["episode_path"]), "start": s["start_frame"], "prompt_tail": s["prompt"][-60:]})
    evidence["dataloader_worker_samples"] = worker

    # 5. Token lengths with the checkpoint tokenizer, longest time string per episode.
    from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer

    tok = HuggingfaceTokenizer(name=str(TOKENIZER), seq_len=512, clean="whitespace")
    longest = (0, None)
    for d in ds._sub_datasets:
        for path, length in zip(d._episode_files, d._episode_lengths):
            prompt = format_prompt_for_inference(train_ptime.instruction_with_time(prompts, path, length - 2))
            full = len(tok.tokenizer([tok._clean(prompt)], add_special_tokens=True).input_ids[0])
            _, mask = tok(prompt, return_mask=True, add_special_tokens=True)
            kept = int(mask.sum())
            assert kept == full < 512, (path, kept, full)
            longest = max(longest, (full, train_ptime.episode_key(path)))
    evidence["max_prompt_tokens"] = {"tokens": longest[0], "episode": longest[1], "seq_len": 512}

    OUT.write_text(json.dumps(evidence, indent=1) + "\n")
    print(json.dumps({k: evidence[k] for k in ("dataset", "release_would_clip", "max_prompt_tokens")}, indent=1))
    for c in checks:
        print(f"{c['label']:<46} {c['key']:<20} start {c['start']:>5}  merged [{c['unclipped_range_merged'][0]:+.3f}, "
              f"{c['unclipped_range_merged'][1]:+.3f}]  release [{c['unclipped_range_release'][0]:+.3f}, "
              f"{c['unclipped_range_release'][1]:+.3f}]  prompt ok")
    print(f"all checks passed; evidence {OUT}")


if __name__ == "__main__":
    main()
