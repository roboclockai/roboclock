# Open-VLM LTSR runs (ideal latency), split over two nodes

LTSR is the joint success rate when every model query costs simulated time. For the open vision
LLMs we charge each query the time it takes on one 8×A100 node with the model's fastest
speculative decoder (`--inference-latency ideal`), not the time it took on the node that ran the
episode. So the result does not depend on the node or on how the servers are laid out, and one
run can be split over several nodes.

This document covers:

1. the protocol;
2. the speeds file;
3. the 2026-11-10 run and how it is split;
4. step-by-step instructions for the second node (sections 4-8).

The sweep scripts themselves are described in [vision-llm-sweep.md](vision-llm-sweep.md), and the
latency modes in [ideal-latency-eval.md](ideal-latency-eval.md).

## 1. Protocol

Every episode uses the C1 primitive agent (`roboclock.policies.vision_llm_primitives:make --depth`) as
in the 2026-11-08 open-VLM sweep, with:

- easy difficulty, the 60 tasks of the five domains, seeds 0 and 1;
- thinking on, with a 1024-token budget, and temperature 0;
- at most 300 model turns per episode for A and B tasks, and 100 for C tasks. After the cap, both
  arms hold until the episode ends;
- C tasks scored with the C-class rule of score_version 2 ([c-class-timing.md](c-class-timing.md));
- latency `ideal`, with `--model-speeds src/roboclock/evaluation/model_speeds_8xa100.yaml`;
- each model served by vLLM 0.32.0 in bf16, with the settings in `serve_vlm.sh`. The served model
  name must match the speeds file's key, for example `Qwen/Qwen3.8-27B`.

The table's TSR column is the same run at `--inference-latency zero`.

## 2. The speeds file

`src/roboclock/evaluation/model_speeds_8xa100.yaml` was measured by replaying 30 real agent requests,
one at a time, on 8 otherwise idle A100s at tensor parallel 8. Its header gives the method and its
`notes` give every variant tried. Per model it holds:

| Served model | Decoder used | Tokens/s | TTFT | Charge for a mean reply |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | DFlash2, 7 draft tokens | 274.1 | 0.32 s | 4.6 s |
| `meta-models/Muse-Glimmer-30B` | Meta's DFlash assistant | 259.8 | 0.24 s | 4.7 s |
| `google/gemma-4-26B-A4B-it` | z-lab DFlash | 289.5 | 0.44 s | 4.3 s |
| `Qwen/Qwen3.5-9B` | built-in MTP, 3 tokens | 500.4 | 0.23 s | 2.5 s |
| `Qwen/Qwen3.5-4B` | built-in MTP, 3 tokens | 562.1 | 0.22 s | 2.3 s |
| `Qwen/Qwen3.8-Flash-Next` | built-in MTP, 3 tokens | 212.0 | 0.48 s | 5.9 s |

The runs do not need these drafters. Greedy speculative decoding returns the same replies, so the
servers run without one. Only the speeds file sets the charge.

## 3. The 2026-11-10 run

| Work | Episodes | Where |
|---|---|---|
| Qwen3.8-Flash-Next, zero and ideal | 2 × 120 | main pod, done |
| 5 open VLMs, ideal | 5 × 120 | split: main pod and second node |
| 5 open VLMs, zero, C tasks only (score_version 2 reruns) | 5 × 34 | main pod |

The main pod runs everything except the 139 ideal episodes listed in
`scripts/xpolicylab/sweep/jobs/vlm-ltsr-2026-11-10-node-b.txt`:

| Model | Episodes |
|---|---|
| `qwen38` | 35 |
| `muse` | 35 |
| `gemma4` | 35 |
| `qwen35_9b` | 17 |
| `qwen35_4b` | 17 |

The counts follow the `vlm` layout's slots on the second node: 6 episodes at a time for each TP-2
model and 3 for each single-GPU model. With these counts, all five models finish at about the same
time. Each model's episodes are spread over the whole range of time budgets.

### Why the two nodes' results can be pooled

The main pod runs its share from its own scripts, at roboclock 711ac59 plus the
`VISION_LLM_MAX_QUERIES` cap. The second node runs this commit through `start.sh`. Everything that
changed in between leaves these episodes as they are:

- **Ideal-latency charge.** It comes from the same speeds file. The main pod uses a byte-identical
  copy of its numbers.
- **Server settings.** They are the same, and so is the 1024-token thinking budget.
- **Query caps.** They are the same: 300 for A and B tasks, 100 for C tasks.
- **`tidy_args` (24c5bff).** It only resolves conflicting arguments that hosted APIs produce. All
  14.8k answers of the 2026-11-08 open-VLM sweep, replayed through it, read exactly as before.
- **Request retries (24c5bff).** They change nothing unless a request to the local server fails.
- **Pre-event motion (dee7b17).** Arm motion before the event is allowed in lab B4-B7 and sports B3.
  The main pod applies the same rule to its rows offline, with `rescore_pre_event.py`.
- **`--start-pose zero` (e383ac3).** It is optional and off by default.

The time budgets of all 139 episodes, as `queue.py` plans them at this commit, equal the main pod's.

## 4. Second node: what it needs

- 8 free A100 80 GB GPUs. The `vlm` layout puts a vLLM server and 3 Isaac episodes on every GPU, so
  nothing else may run on them. Check `nvidia-smi --query-compute-apps=pid --format=csv`.
- About 200 GB of disk for the weights (section 5, step 3) and a few GB for the results.
- About 95 CPU cores, as on the main pod. Isaac start-ups wait while the 1-minute load is above
  `LOAD_MAX` (default: 1.5 × cores).
- Network access to GitHub and Hugging Face. Export your own `HF_TOKEN` before downloading.

## 5. Second node: setup

1. **roboclock.** Use a fresh checkout at this commit or later, and nothing else.
   - Do not use a working copy with local commits that are not on origin: they may change the agent
     or the task instructions, and the results would no longer pool.
   - `run_info.txt` records the commit, and adds `+local changes` if the checkout is dirty.

   ```bash
   git -C $ROBOCLOCK_DIR fetch origin
   git -C $ROBOCLOCK_DIR worktree add --detach $ROBOCLOCK_DIR origin/main
   cd $ROBOCLOCK_DIR && git log --oneline -1
   ```

2. **Isaac Sim, the kitchen, graphics libraries and Vulkan.** Follow steps 2-5 of section 4 of
   [demo-collection.md](demo-collection.md), with `$ROBOCLOCK_DIR` as the checkout:
   - the Isaac Sim 6.0.1 venv at `$ISAAC_VENV`;
   - `scripts/household/fetch_lightwheel_kitchen.py` for the household kitchen. Without it, every
     household episode that needs the kitchen fails at start with `FileNotFoundError ...
     lightwheel_kitchen/...`;
   - on the remote node node, `LD_LIBRARY_PATH=/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu` for
     `start.sh`.

3. **vLLM and the weights.** The servers use `$VLLM_VENV` and `$MODELS_DIR/<dir>`.
   - Install vLLM 0.32.0:

     ```bash
     uv venv $VLLM_VENV --python 3.12
     uv pip install --python python vllm==0.32.0
     python -c "import vllm; print(vllm.__version__)"   # 0.32.0
     ```

     The main pod has torch 2.13.0+cu130 and transformers 5.19.0 in that venv.
   - Download the weights at the main pod's revisions:

     ```bash
     cd $MODELS_DIR
     hf download Qwen/Qwen3.8-27B             --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir Qwen3.8-27B
     hf download meta-models/Muse-Glimmer-30B --revision cec94fcfd6512f2cfe708138c32533b346ea4f69 --local-dir Muse-Glimmer-30B
     hf download google/gemma-4-26B-A4B-it    --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --local-dir gemma-4-26B-A4B-it
     hf download Qwen/Qwen3.5-9B              --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir Qwen3.5-9B
     hf download Qwen/Qwen3.5-4B              --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir Qwen3.5-4B
     ```

     The total is about 185 GB. Use the `-it` (instruct) Gemma, not the base model.
   - Qwen3.8-27B's later revision 37561816 only wraps the chat template in `{% generation %}` tags.
     It renders the same prompt, so either revision gives the same results.

4. **Nothing else on the GPUs or ports.** The servers listen on 8100 (qwen38), 8110 (muse),
   8120 (gemma4), 8130 (qwen35_9b) and 8140 (qwen35_4b). `launch.sh` refuses to start if a port is
   already in use.

## 6. Second node: smoke test (optional, about 10-15 min)

This runs one short household B2 episode with Qwen3.5-4B on GPU 7, under ideal latency:

```bash
cd $ROBOCLOCK_DIR
SWEEP_ROOT=$WORK_DIR/vlm_ltsr_smoke TASKS="household:B2" SEEDS=1 LATENCY=ideal \
  MODEL_SPEEDS=$PWD/src/roboclock/evaluation/model_speeds_8xa100.yaml \
  bash scripts/xpolicylab/sweep/start.sh qwen35_4b:7
SWEEP_ROOT=$WORK_DIR/vlm_ltsr_smoke bash scripts/xpolicylab/sweep/status.sh
```

When it is done, check two things:

- `$WORK_DIR/vlm_ltsr_smoke/qwen35_4b/out/household_B2_s0/episodes.jsonl` has
  `"inference_latency_mode": "ideal"`.
- Each entry of `ideal_inference_latencies_s` equals `0.22 + tokens / 562.1`, where `tokens` is the
  matching entry of `query_output_tokens`.

Then stop it:

```bash
SWEEP_ROOT=$WORK_DIR/vlm_ltsr_smoke bash scripts/xpolicylab/sweep/stop.sh
```

## 7. Second node: run its share

```bash
cd $ROBOCLOCK_DIR
export SWEEP_ROOT=$WORK_DIR/vlm_ltsr_node_b SEEDS=2 LATENCY=ideal
export MODEL_SPEEDS=$PWD/src/roboclock/evaluation/model_speeds_8xa100.yaml
export JOBS_FILE=$PWD/scripts/xpolicylab/sweep/jobs/vlm-ltsr-2026-11-10-node-b.txt
bash scripts/xpolicylab/sweep/start.sh vlm
```

- **What it does.** `start.sh` returns at once. Its launcher queues each model's listed episodes, so
  `launch.log` should show 35/35/35/17/17 queued. It starts five servers, which take 1-6 min to
  load, and then 24 episode slots.
- **Before starting.**
  - With `LATENCY=ideal`, the launcher stops before starting any server if the speeds file has no
    entry for a model.
  - Keep `SEEDS=2`, because the job file lists seeds 0 and 1.
  - Leave `VLM_MAX_QUERIES` (300) and `VLM_MAX_QUERIES_C` (100) at their defaults.
- **Running time.** Expect about 2 hours. In ideal mode an episode makes few queries, because each
  costs 2-6 s of its simulated time budget.
- **For an automated agent session.** Run `start.sh` as an ordinary foreground command. Never run it, or
  `launch.sh`, as a background task you later stop: stopping a task kills its whole process tree,
  servers included. Stop processes with `stop.sh` or by PID, never with `pkill -f`.

Watch, stop and resume exactly as in [robodojo-sweep.md](robodojo-sweep.md), section 7. Keep the same
exported variables in the shell:

```bash
bash scripts/xpolicylab/sweep/status.sh
tail -f $SWEEP_ROOT/launch.log
```

`start.sh` again after a stop resumes the run: an episode with a result is not queued again. An
episode without one, for example after a crash or a timeout, is run again on resume.

What to expect in the logs:

- A few `vulkan_failures.txt` lines and an occasional `camera has no RGB frame` crash, which a
  resume reruns.
- Rarely, an Isaac start-up hangs with an empty log. `stop.sh` that model and resume.

## 8. Second node: hand the results back

When `status.sh` shows `queued 0` and `running 0` for all five models:

```bash
bash scripts/xpolicylab/sweep/stop.sh
for m in qwen38 muse gemma4 qwen35_9b qwen35_4b; do
  python3 scripts/xpolicylab/sweep/summarize.py $m | tee $SWEEP_ROOT/$m/final_summary.txt
done
tar czf $WORK_DIR/vlm_ltsr_node_b.tgz -C $WORK_DIR --exclude=industry_local vlm_ltsr_node_b
```

Report:

- the number of results per model (35/35/35/17/17);
- any episode without a result;
- where the archive is.

The main pod merges it: each `vlm_ltsr_node_b/<model>/out/<episode>/` goes into the main pod's
`<model>_ideal/out/`, and the table metrics are computed over the pooled 120 episodes per model.
