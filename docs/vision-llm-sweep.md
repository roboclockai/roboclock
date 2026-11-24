# Vision-LLM models: sweeps with local vLLM servers or a hosted API

The sweep scripts in `scripts/xpolicylab/sweep/` run every roboclock task for several models at once
([robodojo-sweep.md](robodojo-sweep.md) covers the RoboDojo VLA/WAM models). They also run the
primitive agent (`roboclock.policies.vision_llm_primitives`, [vision-llm-agent.md](vision-llm-agent.md))
on vision-language models of two kinds. `VLM_AGENT=move_eef` runs the move_eef agent instead
(`roboclock.policies.vision_llm_move_eef`: RGB only, 100 model calls per episode, `VLM_EEF_MAX_QUERIES`):

| Kind | Model names | Model runs on | GPUs listed in the spec run |
|---|---|---|---|
| `vllm` | `qwen38`, `muse`, `gemma4`, `qwen35_9b`, `qwen35_4b`, `flashnext` | a local vLLM server per TP group, weights in `$MODELS_DIR` | the server and 3 Isaac episodes per GPU |
| `api` | `nv/<model id>`, e.g. `nv/gcp/google/gemini-4-argon` | NVIDIA's inference API (OpenAI-compatible) | only Isaac, 3 episodes per GPU |

Every episode uses one protocol:

- easy, `--depth`, the primitive agent with native tool calling, no SAM3;
- thinking on: a 1024-token budget on vLLM, `reasoning_effort` `medium` on a hosted API;
- past turns in view: 6 on a hosted API, 4 on vLLM (its servers have 16384 tokens of context);
- at most 300 model turns per episode for A and B tasks (`VLM_MAX_QUERIES`) and 100 for C tasks
  (`VLM_MAX_QUERIES_C`, since 2026-11-10), after which both arms hold;
- `--inference-latency zero` unless `LATENCY` says otherwise. For LTSR, `LATENCY=ideal` with
  `MODEL_SPEEDS=src/roboclock/evaluation/model_speeds_8xa100.yaml` (see [vlm-ltsr.md](vlm-ltsr.md)).

`run_info.txt` in each model's directory records all of it. Results from before 2026-11-10 used
another protocol (one JSON answer per turn, no conversation, every quantity an argument of each
call) and are not comparable.

Each model has its own queue and slots on its own GPUs. The scripts put Isaac Sim's graphics
libraries on the library path themselves when they are at `ISAAC_GRAPHICS_LIBS` (default
`/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu`).

## Hosted API models

The API serves about 250 models under ids such as `gcp/google/gemini-4-argon`, `azure/openai/gpt-5.4`,
`aws/anthropic/bedrock-claude-sonnet-5-5` and `nvidia/qwen/qwen3.8-27b`. List them with
`curl -s -H "Authorization: Bearer $KEY" https://inference-api.example.com/v1/models`.
The sweep name of a model is `nv/` plus its id. Its directory under `SWEEP_ROOT` replaces `/` with
`_` (`nv_gcp_google_gemini-4-argon`).

The key comes from `NVIDIA_API_KEY` or, if that is unset, from line `NV_API_KEY_INDEX` (default 1)
of `NV_API_KEY_FILE`. By default that file is the first of `$API_KEY_FILE` and
`$API_KEY_FILE` that exists. Lines look like `1. sk-...` or `sk-...`. Episodes get the
key in their environment only. It never appears on a command line, in `run_info.txt` or in a log.

`NV_API_KEY_INDEX` can list several lines, for example `1,2`. The model's slots then take the
keys in turn, so one sweep with one queue and one result directory spreads over several keys:

```bash
SWEEP_ROOT=$WORK_DIR/vlm_api_sweep SEEDS=2 NV_API_KEY_INDEX=1,2 SWEEP_SLOTS_PER_GPU=6 \
  bash scripts/xpolicylab/sweep/start.sh nv/azure/openai/gpt-6-astra:1,2,3
```

- Requests use the `openai` dialect: standard OpenAI fields only, with `reasoning_effort` `medium`.
  `max_tokens` is 4096, reasoning included: the provider's reasoning is not capped at the
  1024-token budget, and at 2048 Gemini 4 Argon ran out on 6% of its turns. Export
  `VISION_LLM_MAX_TOKENS`, `VISION_LLM_REASONING_EFFORT` or `VISION_LLM_HISTORY` before `start.sh`
  to change that; all are recorded in `run_info.txt`.
- The primitives go to the API as tools, so the provider's own tool calling formats the calls. A
  call with a wrong argument is answered with a refusal and costs that call, not the turn.
- A request the provider's content filter refuses costs one turn, not the episode.
- Throttling (HTTP 429), server errors and requests that hang for `VLM_API_TIMEOUT` (default 180 s)
  are retried up to `VLM_API_RETRIES` times (default 6, 5 s doubling to 60 s). Before 2026-11-10 a
  hung request waited 600 s and then ended the episode without a result. A key whose budget is spent answers HTTP 422 `budget_exceeded`; that is not
  retried, and the episode ends with an error and no result.
- The API bills every turn. With GPT 6 Astra a turn is 7 to 9k prompt tokens, about 6.1k of them
  read from the provider's cache after the first turn, and a few hundred output tokens: about
  $0.03 and 3 to 10 s.

Smoke test, one episode on GPU 3 (about 10-25 min):

```bash
SWEEP_ROOT=$WORK_DIR/vlm_api_smoke TASKS="lab:A5" SEEDS=1 \
  bash scripts/xpolicylab/sweep/start.sh nv/gcp/google/gemini-4-argon:3
bash scripts/xpolicylab/sweep/status.sh            # with the same SWEEP_ROOT
python3 scripts/xpolicylab/sweep/summarize.py nv/gcp/google/gemini-4-argon $WORK_DIR/vlm_api_smoke
```

The 60-task sweep with seeds 0-1 is 120 episodes. The pod runs about 20 Isaac episodes at once
before the CPU saturates, so 6-7 GPUs (18-21 slots) are enough:

```bash
SWEEP_ROOT=$WORK_DIR/vlm_api_sweep SEEDS=2 \
  bash scripts/xpolicylab/sweep/start.sh nv/gcp/google/gemini-4-argon:0,2,3,4,5,6,7
```

## Local open VLMs

`start.sh vlm` is the 2026-11-08 layout: `qwen38` on GPUs 0,1, `muse` on 2,3, `gemma4` on 4,5,
`qwen35_9b` on 6 and `qwen35_4b` on 7. A TP-2 model needs its GPUs in pairs; it gets one server
per pair.

```bash
SWEEP_ROOT=$WORK_DIR/vlm_sweep2 SEEDS=2 bash scripts/xpolicylab/sweep/start.sh vlm
SWEEP_ROOT=$WORK_DIR/vlm_sweep2 SEEDS=2 bash scripts/xpolicylab/sweep/start.sh qwen38:0,1 nv/gcp/google/gemini-4-argon:2,3
```

- `serve_vlm.sh MODEL PORT GPUS` starts one server (vLLM from `VLLM_PYTHON`, bf16,
  `--max-model-len 16384`, `--max-num-seqs 32`, prefix caching, the model's reasoning parser, and
  `--enable-auto-tool-choice` with the model's tool-call parser).
  It unsets the image's `OMP_NUM_THREADS=95`, which makes TP workers spin.
- The agent calls its primitives as tools with `tool_choice: "required"`, which vLLM refuses
  without a tool-call parser. `common.sh` names each model's: `qwen3_xml` for the Qwen models,
  `muse_glimmer`, `gemma4`. Only Qwen3.8-27B was checked (vLLM 0.33, 2026-11-10, FP8 weights on
  one L40S): tool calls, several per turn, a full 1024-token thinking budget, about 9.2k prompt
  tokens on the first turn and 0.7k more per past turn. vLLM did not hold the model to the
  numeric ranges in the tool schemas; the agent's own check refuses such a call.
- A request that still overflows the 16384 tokens makes the agent drop its older turns and ask
  again.
- Muse Glimmer needs `--reasoning-config` with its reasoning markers, or vLLM ignores the thinking
  budget; `common.sh` sets it.
- Ports: qwen38 8100, muse 8110, gemma4 8120, qwen35_9b 8130, qwen35_4b 8140, flashnext 8150 (one
  more per extra server).
- `flashnext` (Qwen3.8-Flash-Next, a 180B MoE in bf16) needs all eight GPUs for one TP-8 server:
  `start.sh flashnext:0,1,2,3,4,5,6,7`. It cannot share a node with the `vlm` layout.

## Settings

Set in the environment of `start.sh`, in addition to those in robodojo-sweep.md (`SWEEP_ROOT`,
`SEEDS`, `TASKS`, `FIRST_TASKS`, `ENVS`, `LATENCY`, `LOAD_MAX`, `SWEEP_SLOTS_PER_GPU`):

| Variable | Default | Meaning |
|---|---|---|
| `VLM_MAX_QUERIES` | 300 | Model turns per episode, then the arms hold (0: no cap) |
| `VLM_MAX_QUERIES_C` | 100 | The same for C-class (throughput) tasks |
| `MODEL_SPEEDS` | empty | With `LATENCY=ideal`: the speeds file passed as `--model-speeds` (empty: roboclock's surveyed `model_speeds.yaml`). The launcher refuses a model the file does not list |
| `JOBS_FILE` | empty | Queue only the listed episodes, lines `MODEL ENV TASK SEED` (section "Several nodes") |
| `NV_API_URL` | `https://inference-api.example.com/v1` | Hosted API base URL |
| `NV_API_KEY_FILE` / `NV_API_KEY_INDEX` | first of `$API_KEY_FILE`, `$API_KEY_FILE` / 1 | Where the key is read from when `NVIDIA_API_KEY` is unset; the index may list several lines (`1,2`) |
| `VISION_LLM_HISTORY` | 6 (API), 4 (vLLM) | Past turns a request carries |
| `VISION_LLM_REASONING_EFFORT` | `medium` | Hosted API: the `reasoning_effort` sent |
| `ISAAC_GRAPHICS_LIBS` | `/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu` | Directory with Isaac Sim's graphics libraries, put on the library path when it exists |
| `VLM_API_RETRIES` / `VLM_API_TIMEOUT` | 6 / 180 s | Retries of a throttled, failed or hung API request, and how long one may hang |
| `VLLM_PYTHON` / `MODELS_DIR` | `python` / `$MODELS_DIR` | Local vLLM and weights |
| `VLLM_MAX_NUM_SEQS` | 32 | Concurrent sequences per vLLM server |

`LATENCY=measured` charges each turn's round trip. For a local model that needs one episode per server
(`SWEEP_SLOTS_PER_GPU` low enough), because a shared server's queue would be charged too. For an API
model the network and the provider's queue are part of the charged time.

## Several nodes

Each node runs its own `SWEEP_ROOT`. To split one run, write the episodes a node should run into
a job file, one `MODEL ENV TASK SEED` line each (`#` starts a comment). Start that node with
`JOBS_FILE=<file>` and the run's other settings (`SEEDS`, `LATENCY`, `MODEL_SPEEDS`). Only the
listed episodes of each model are queued, and episodes that already have a result are skipped as
usual. A listed episode that the plan does not contain stops the launch. Merge the nodes' `out/`
directories afterwards. Under `LATENCY=zero` or `ideal` the charge does not depend on the node, so
results pool. Under `measured` they do not.

Example: `scripts/xpolicylab/sweep/jobs/vlm-ltsr-2026-11-10-node-b.txt` and [vlm-ltsr.md](vlm-ltsr.md).

## Watching

Watching, stopping and resuming work as in robodojo-sweep.md section 7 (`status.sh`, `stop.sh
[MODEL...]`, `start.sh` again). Each episode's requests, with reasoning and calls, are in
`<model dir>/qlog/<env>_<task>_s<seed>.jsonl`.
