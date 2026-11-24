# Running the vision-LLM agents under ideal inference latency

`--inference-latency ideal` charges each model query the time it would take at the
fastest speed anyone reports for that model. The time the query took on our own GPU is
not used. Each query is charged

    L = ttft_s + output_tokens / max_tokens_per_s

- `output_tokens` is the reply's `usage.completion_tokens`, thinking tokens included.
- `max_tokens_per_s` and `ttft_s` (time to the first token) come from a speeds file.
- The time spent waiting in the server's queue, and any SAM3 calls, are not charged.

Like every latency mode, this runs inside block-and-replay (the default) or
`--time-model streaming`. See [architecture.md](architecture.md#inference-latency) for the
time model and [vision-llm-agent.md](vision-llm-agent.md) for the agents.

## 1. Code

Use `main` at or after the commit that added this file. In a checkout other than the one
your venv's editable install points to, set `PYTHONPATH=<checkout>/src`. Otherwise the
other checkout's code runs.

```bash
git pull --rebase origin main
export PYTHONPATH=$PWD/src
python -m roboclock eval --help | grep -A1 inference-latency   # lists {measured,zero,ideal}
```

## 2. Configuration

### The speeds file

The default file is `src/roboclock/evaluation/model_speeds.yaml`. It has one entry per
served model name:

```yaml
models:
  Qwen/Qwen3.8-27B:
    max_tokens_per_s: 1850   # single-request decode speed, required, > 0
    ttft_s: 0.0              # time to first token in seconds, optional, >= 0, default 0
    source: ...              # where the number comes from (not read by the code)
    surveyed: 2026-11-06     # (not read by the code)
    notes: ...               # other surveyed numbers (not read by the code)
```

- The key must equal the model name the agent sends, which is `VISION_LLM_MODEL`
  (default `Qwen/Qwen3.8-27B`). If vLLM serves the weights under another name
  (`--served-model-name`), add an entry under that name.
- The speeds are for one request, not a server's aggregate throughput.
- The shipped entry is Cerebras's listed 1850 tokens/s with `ttft_s: 0`, so only decoding
  is charged.

### Other speed profiles

To evaluate at another speed, copy the file and pass the copy with `--model-speeds`.
Leave the shipped file as the surveyed maximum.

`src/roboclock/evaluation/model_speeds_8xa100.yaml` is the profile the open-VLM LTSR runs use: each
model's single-request speed on one 8×A100 node with its fastest speculative decoder, measured
2026-11-09/10 for Qwen3.8-27B, Muse Glimmer 30B, Gemma 4 26B-A4B, Qwen3.5-9B/4B and
Qwen3.8-Flash-Next. See [vlm-ltsr.md](vlm-ltsr.md). For example, the fastest GPU clouds
measured by Artificial Analysis:

```yaml
# speeds_gpu_cloud.yaml
models:
  Qwen/Qwen3.8-27B:
    max_tokens_per_s: 190
    ttft_s: 0.66
```

Numbers surveyed on 2026-11-06 for Qwen3.8-27B, all for one request:

| Profile | tokens/s | TTFT (s) | Source |
|---|---|---|---|
| Shipped default | 1850 | 0 | Cerebras docs |
| Cerebras as measured by a user | 890 | 0.64 | Hacker News report, median |
| Fastest GPU clouds | 190 | 0.66 | Artificial Analysis (Crusoe, Multiverse) |
| One A100, bf16, our vLLM | 27.9 | 0.59 | fit over 1034 requests |

### Agent settings that change the charge

- `VISION_LLM_THINKING_BUDGET` (default 1024) caps the thinking tokens of a turn.
  - A primitive turn emits about 1115 tokens: about 0.60 s at 1850 tokens/s.
  - A waypoint-agent turn emits about 880 tokens: about 0.48 s.
- `VISION_LLM_THINKING=0` turns thinking off and makes every turn much cheaper.

## 3. What must be running

- **A Qwen3.8-27B server**, started as in
  [vision-llm-agent.md](vision-llm-agent.md#running):

  ```bash
  VLLM_GPU_UTIL=0.85 VLLM_MAX_NUM_SEQS=16 VLLM_MAX_MODEL_LEN=12288 \
    bash scripts/xpolicylab/serve_qwen.sh $MODELS_DIR/Qwen3.8-27B 8000 <gpu>
  ```

  The server must return `usage` in its replies; vLLM does. Speed does not matter in
  this mode, so a slow or shared GPU gives the same charged latency.
- **The Isaac Sim 6.0.1 venv** (`$ISAAC_VENV`), as for any roboclock eval.
- **Optionally, SAM3** for named grasp targets, through `VISION_LLM_SAM3_URL`. The
  2026-11-06 baseline `qprim3` ran without it.

## 4. Run

One episode:

```bash
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 CUDA_VISIBLE_DEVICES=<gpu>
export VISION_LLM_URL=http://127.0.0.1:8000/v1 VISION_LLM_TIMEOUT=900
VISION_LLM_LOG=qlog/household_B2_s0.jsonl \
  python -m roboclock eval --env household --task B2 --difficulty easy \
    --seeds 1 --seed-start 0 \
    --policy roboclock.policies.vision_llm_primitives:make --depth \
    --inference-latency ideal --output out/household_B2_s0
```

Add `--model-speeds speeds_gpu_cloud.yaml` for another profile. Use
`--policy roboclock.policies.vision_llm:make` for the waypoint agent.

The comparison set is the 30 episodes used for pi0.5 and the earlier Qwen runs:

| Domain | Tasks | Difficulty |
|---|---|---|
| lab | A5, B1 | easy |
| hospital | A3, B4 | easy |
| household | A2, B2 | easy |
| sports | A3, B2 | easy |
| industry | A1, B2 | reviewed_nominal |

Each task runs seeds 0, 1 and 2, one episode per process, written to
`out/<env>_<task>_s<seed>`. A sequential loop for one GPU and one server:

```bash
mkdir -p out qlog logs
for job in "lab A5" "lab B1" "hospital A3" "hospital B4" "household A2" "household B2" \
           "sports A3" "sports B2" "industry A1" "industry B2"; do
  set -- $job; E=$1; T=$2; D=easy; [ $E = industry ] && D=reviewed_nominal
  for S in 0 1 2; do
    N=${E}_${T}_s$S
    flock /tmp/.isaac-start.lock sleep 40
    VISION_LLM_LOG=qlog/$N.jsonl timeout 5400 python -m roboclock eval \
      --env $E --task $T --difficulty $D --seeds 1 --seed-start $S \
      --policy roboclock.policies.vision_llm_primitives:make --depth \
      --inference-latency ideal --output out/$N < /dev/null > logs/$N.log 2>&1
  done
done
```

To run episodes in parallel:

- Several episodes may share one server: the charged latency does not depend on the
  queue.
  - Raise `VISION_LLM_TIMEOUT`, because a queued request can wait several minutes. At
    300 s, requests timed out with three episodes on a one-sequence server.
- Keep at most about 3 Isaac processes per GPU, and at least 10 GB free on every GPU.
  - Vulkan sees all GPUs, so one full card makes Isaac fail to start everywhere.
- Stagger starts with the `flock` line above.
- `--output` will not overwrite an earlier run's `episodes.jsonl`, so delete a failed
  episode's directory before rerunning it.

## 5. Check the results

- `out/<name>/manifest.json` has `"inference_latency": "ideal"`, and `model_speeds` lists
  the speeds that were charged.
- Each row of `out/<name>/episodes.jsonl` has:
  - `task_success` (physical), `temporal_success` and `joint_success`;
  - `queries` and `mean_inference_latency`. With the shipped speeds,
    `mean_inference_latency` should be about 0.5-0.6 s for the primitive agent.
  - `measured_inference_latencies_s`: what each query really took (not charged).
- Runs under `--inference-latency zero` or `measured` record `ideal_inference_latencies_s`
  and `mean_ideal_inference_latency` too, without charging them (`docs/architecture.md`,
  Inference latency).
- `qlog/<name>.jsonl` has one row per request, with `completion_tokens` and
  `round_trip_ms` (the real time, which is not charged).

The run stops with an error, and does not fall back to measured time, if:

- the served model name has no entry: `No surveyed decode speed for '...'`;
- a reply has no token count: `... needs each query's output token count ...`;
- the policy does not name its model: `... needs a policy that names its model ...`. This
  happens for policies other than the two vision-LLM agents.

To compare: on 2026-11-06, the same 30 episodes under `--inference-latency zero` with the
primitive agent and no SAM3 (`qprim3`) gave joint 3/30 and physical 6/30. Under
`measured` on one A100 (about 40 s per turn), the earlier waypoint-agent run gave 0/30.
