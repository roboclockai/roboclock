"""Check that the InternW0 text cache covers every prompt the converted repos need.

InternW0 training has no online text encoder (model.load_text_encoder=false). A prompt without a
cache file raises inside the dataset, and RobotVideoDataset.__getitem__ then silently returns a
random other sample instead. So this check must pass before training.

Required prompts = DEFAULT_PROMPT.format(task=t) for every task string t in every repo's
meta/tasks.jsonl. The cache file of a prompt is
    <cache>/<sha256(prompt)>.t5_len<context_len>.wan22ti2v5b.pt   (robot_video_dataset.py, text_cache.py)

Also checks, per repo, that tasks.jsonl is exactly the P3 set of its episodes
(prompts JSON + per-frame time, from convert.py's _status/<task>.json), and with --tokens that no
prompt is longer than context_len UMT5 tokens (it would be truncated). --deep N loads N random
cache files and checks their shape. Exit 1 on any failure.

  venv-train/bin/python check_cache.py --repos $TRAIN45_DIR/D_internw0/lerobot --tokens --deep 20
"""

import os
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
UPSTREAM = Path(os.path.expandvars("$TRAIN45_DIR/D_internw0/upstream"))


def p3(instructions, task, seed, frame):
    return instructions[f"{task}|{seed}"] + f" Time: {frame / 25:.1f} s"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos", type=Path, default=Path(os.path.expandvars("$TRAIN45_DIR/D_internw0/lerobot")))
    parser.add_argument("--tasks", default="farm", help="'farm' = the 45 farm tasks, 'all' = every repo with a status file, or a comma list")
    parser.add_argument("--farm", type=Path, default=Path(os.path.expandvars("$TRAIN45_DIR/data/RoboDojo")))
    parser.add_argument("--cache", type=Path, default=Path(os.path.expandvars("$TRAIN45_DIR/D_internw0/cache/text_embed/robodojo")))
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--prompts", type=Path, default=Path(os.path.expandvars("$TRAIN45_DIR/prompts/instructions_main_4f541f2.json")))
    parser.add_argument("--tokens", action="store_true", help="count UMT5 tokens (needs transformers)")
    parser.add_argument("--tokenizer", type=Path, default=Path(os.path.expandvars("$MODELS_DIR/Wan2.2-TI2V-5B/google/umt5-xxl")))
    parser.add_argument("--deep", type=int, default=0, help="load this many random cache files")
    args = parser.parse_args()

    sys.path.insert(0, str(UPSTREAM / "src"))
    try:  # the exact upstream constant, when the training env is used
        from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT as upstream_prompt
        assert upstream_prompt == DEFAULT_PROMPT, "upstream DEFAULT_PROMPT changed"
    except ImportError:
        pass

    if args.tasks == "farm":
        tasks = sorted(p.name for p in args.farm.iterdir() if (p / "arx_x5" / "data").is_dir())
    elif args.tasks == "all":
        tasks = sorted(p.stem for p in (args.repos / "_status").glob("*.json"))
    else:
        tasks = args.tasks.split(",")
    instructions = json.loads(args.prompts.read_text())
    errors = []
    required = set()
    for task in tasks:
        status_path = args.repos / "_status" / f"{task}.json"
        tasks_path = args.repos / task / "meta" / "tasks.jsonl"
        if not status_path.exists() or not tasks_path.exists():
            errors.append(f"{task}: repo not converted (no {status_path.name} or tasks.jsonl)")
            continue
        status = json.loads(status_path.read_text())
        expected = {p3(instructions, task, ep["seed"], f) for ep in status["per_episode"] for f in range(ep["frames"])}
        found = {row["task"] for row in read_jsonl(tasks_path)}
        if found != expected:
            errors.append(f"{task}: tasks.jsonl has {len(found)} strings, P3 gives {len(expected)} "
                          f"({len(found - expected)} unexpected, {len(expected - found)} absent)")
        required |= {DEFAULT_PROMPT.format(task=t) for t in found}

    suffix = f".t5_len{args.context_len}.wan22ti2v5b.pt"
    files = {p: args.cache / f"{hashlib.sha256(p.encode('utf-8')).hexdigest()}{suffix}" for p in required}
    missing = [p for p, f in files.items() if not f.exists()]
    print(f"{len(tasks)} repos, {len(required)} distinct prompts, {len(required) - len(missing)} cached, "
          f"{len(missing)} missing in {args.cache} (len {args.context_len})")
    if missing:
        errors.append(f"{len(missing)} prompts have no cache file, e.g. {missing[0][-120:]!r}")

    if args.tokens and required:
        # The cache's own tokenizer: upstream HuggingfaceTokenizer with clean="whitespace"
        # (text_cache.py), here without padding/truncation so the true length shows.
        from wam.model.backbones.wan22.wan_video_text_encoder import HuggingfaceTokenizer

        tok = HuggingfaceTokenizer(name=str(args.tokenizer), seq_len=None, clean="whitespace")
        prompts = sorted(required)
        lengths = []
        for start in range(0, len(prompts), 512):
            ids = tok(prompts[start:start + 512], return_tensors=None, add_special_tokens=True)
            lengths += [len(x) for x in ids]
        longest = max(range(len(prompts)), key=lengths.__getitem__)
        over = sum(n > args.context_len for n in lengths)
        print(f"UMT5 tokens: max {lengths[longest]} (limit {args.context_len}), {over} prompts over the limit; "
              f"longest: ...{prompts[longest][-80:]!r}")
        if over:
            errors.append(f"{over} prompts exceed {args.context_len} tokens and would be truncated")

    if args.deep and not missing and required:
        import torch

        for prompt in random.Random(0).sample(sorted(required), min(args.deep, len(required))):
            payload = torch.load(files[prompt], map_location="cpu")
            ctx, mask = payload["context"], payload["mask"]
            if tuple(ctx.shape) != (args.context_len, 4096) or ctx.dtype != torch.bfloat16 or tuple(mask.shape) != (args.context_len,):
                errors.append(f"bad cache file {files[prompt].name}: context {tuple(ctx.shape)} {ctx.dtype}, mask {tuple(mask.shape)}")
            elif bool(mask.all()):
                errors.append(f"{files[prompt].name}: mask is all ones, the prompt filled {args.context_len} tokens (truncated?)")
        print(f"deep check of {min(args.deep, len(required))} files done")

    for e in errors:
        print("FAIL:", e)
    print("OK" if not errors else f"{len(errors)} failure(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
