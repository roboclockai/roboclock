# RoboDojo models: the full easy sweep on two nodes (roles A and B)

The same scripts also run vision-language models through the primitive agent, on local vLLM
servers or a hosted API (NVIDIA's inference API): see [vision-llm-sweep.md](vision-llm-sweep.md).

This sweep runs every roboclock task on easy, seeds 0-2, for four RoboDojo-trained models:

- Wall-X;
- InternW0-Δ;
- VPP2;
- Awomo-0.5.

That is 60 tasks × 3 seeds = 180 episodes per model, at zero inference latency. The main pod ran
the other four models (π0.5, X-VLA, G0.5, OpenWAM) the same way on 2026-11-08, so the eight
models end up in one table.

The work is split across two nodes, called roles **A** and **B**. Whoever starts a node tells it
which role it has. These letters have nothing to do with the node names A/B/C in
[demo-collection.md](demo-collection.md).

| Role | Models | GPUs (8 × A100 80 GB) | Servers / episodes per GPU | Episodes at once | Estimated wall time |
|---|---|---|---|---|---|
| A | Awomo-0.5 | 0-7 | 2 / 2 | 16 | about 10.5-11.5 h |
| B | Wall-X | 0, 1 | 2 / 2 | 4 | about 8.5-10 h |
| B | InternW0-Δ | 2, 3, 4 | 2 / 2 | 6 | about 8.5-9 h |
| B | VPP2 | 5, 6, 7 | 1 / 2 | 6 | about 10.5-11.5 h |

- **Where the estimates come from.** They use the wall time per simulated second measured in the
  2026-11-07 30-episode runs, plus about 80 s of start-up per episode.
  - The low end assumes episodes end as early as they did then. The high end assumes every episode
    runs its full budget.
  - Those runs had about two episodes per server. Awomo has one server per episode here, so it may
    finish sooner.
- **Why the split is by model.** Each node downloads only its own models. Awomo by itself is about
  half the work.
- **Why only two episodes per GPU.** These four models are limited by the server or by rendering, not
  by the simulator's CPU. A third episode per GPU does not help.

## 1. What a node needs

- 8 free 80 GB GPUs. Check `nvidia-smi`. Other jobs' GPUs must not be listed in the run (section 6).
- Disk for weights and virtual environments, plus room for the uv and Hugging Face caches:
  - role A: about 40 GB (31 GB of weights, a 3 GB venv);
  - role B: about 130 GB (103 GB of weights, 16 GB of venvs).

  The results are small: a few KB per episode, plus logs.
- Network access to GitHub, Hugging Face, PyTorch's wheel index and PyPI.
- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and the Hugging Face CLI `hf`
  (`uv tool install huggingface_hub`). Export your own Hugging Face token as `HF_TOKEN` before
  you download anything. Never paste the token into a file in the repository.

## 2. Common setup (both roles)

1. **roboclock.** Use a checkout that nobody edits while the sweep runs:

   ```bash
   git clone <repository-url> $ROBOCLOCK_DIR
   cd $ROBOCLOCK_DIR && git checkout --detach origin/main
   ```

   Use the commit that added this document, or a later one. Each model's `run_info.txt` records the
   commit.

2. **Isaac Sim and the simulator machine.** Follow steps 2-5 of section 4 in
   [demo-collection.md](demo-collection.md), using `$ROBOCLOCK_DIR` wherever it says
   `$ROBOCLOCK_DIR`:
   - the Isaac Sim 6.0.1 venv at `$ISAAC_VENV`;
   - the Lightwheel kitchen, which household needs: run
     `python scripts/household/fetch_lightwheel_kitchen.py` in the
     checkout. Both roles run household, so both need it;
   - the graphics libraries, on the remote node node only;
   - Vulkan.

3. **RoboDojo and XPolicyLab.** Use the exact adapter commit that was evaluated:

   ```bash
   git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git $ROBODOJO_DIR
   cd $ROBODOJO_DIR && git checkout f57e163
   git submodule update --init XPolicyLab          # XPolicyLab at 5d0be149
   git -C XPolicyLab log --oneline -1               # must print 5d0be149
   ```

   - Only the XPolicyLab submodule is needed. IsaacLab and curobo are not.
   - Do not move XPolicyLab to a later commit. At XPolicyLab main 6baccd57, the `Awomo05` adapter
     already differs from 5d0be149 in 90 files.
   - The roboclock clients import the XPolicyLab client from this checkout, via `PYTHONPATH`.

## 3. Role A: Awomo-0.5

Follow [awomo.md](awomo.md), section "Setup (policy machine)":

- the checkpoint and its SHA256 check;
- FLUX.2-klein-base-4B;
- the byte-identical FLUX.2 autoencoder from Comfy-Org;
- the weight symlinks;
- the uv venv.

That section assumes Qwen3-VL-4B-Instruct already exists, so download it first:

```bash
hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir $MODELS_DIR/Qwen3-VL-4B-Instruct
```

The checkpoint `Auwomo/Awomo-0.5-Robodojo` has an evaluation-only licence. Accept it on the Hub
before you download it.

## 4. Role B: Wall-X, InternW0-Δ and VPP2

Follow the "Setup (policy machine)" section of each model's document:

- [wallx.md](wallx.md);
- [internw0.md](internw0.md);
- [vpp2.md](vpp2.md).

One step differs from wallx.md. The Wall-X adapter is not in XPolicyLab 5d0be149, so take only
that directory from XPolicyLab main, and keep the rest at 5d0be149:

```bash
git -C $ROBODOJO_DIR/XPolicyLab checkout 6baccd57 -- policy/WallX
```

This is the same `policy/WallX` the main pod evaluated; it matched file for file.

wallx.md also clones `wall-x` into `$WALLX_DIR`. `serve_wallx.sh` looks for it there; set
`WALLX_REPO` if you clone it somewhere else.

## 5. Smoke test

Run one short episode per model into a scratch root before you start the sweep. Household B2 has
a 12 s budget.

```bash
cd $ROBOCLOCK_DIR
export SWEEP_ROOT=$WORK_DIR/robodojo_sweep_smoke TASKS=household:B2 SEEDS=1
bash scripts/xpolicylab/sweep/start.sh awomo:0                    # role A
bash scripts/xpolicylab/sweep/start.sh wallx:0 internw0:2 vpp2:5  # role B
bash scripts/xpolicylab/sweep/status.sh                           # repeat until results 1, queued 0
```

What to expect:

- **How long it takes.** About 5-10 minutes. Servers load in 1-6 minutes, and the first request
  compiles kernels for about a minute.
- **What success looks like.** Each model shows `results 1`:
  - `$SWEEP_ROOT/<model>/qlog/household_B2_s0.jsonl` has a row per model request;
  - `python3 scripts/xpolicylab/sweep/summarize.py <model>` prints a table.
- **The outcome itself.** Every model failed household B2 in the earlier runs. A failure is fine
  here: the smoke test checks the plumbing, not the score.

Then stop the smoke run and clear its variables:

```bash
bash scripts/xpolicylab/sweep/stop.sh
unset SWEEP_ROOT TASKS SEEDS
```

## 6. Run the sweep

```bash
cd $ROBOCLOCK_DIR
bash scripts/xpolicylab/sweep/start.sh A     # or: start.sh B
```

`start.sh` returns at once. A launcher detached from your shell then does the rest:

1. It builds each model's queue: all 180 episodes, longest budget first.
2. It starts the servers and waits until their ports accept connections.
3. It starts the episode slots. One Isaac process starts every 40 s, so 16 slots need about 11
   minutes to start.

If the GPUs differ from the table, give explicit `MODEL:GPU,GPU,...` specs instead of a role. For
example, `start.sh awomo:0,2,3,4,5,6,7` leaves out GPU 1.

The settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SWEEP_ROOT` | `$WORK_DIR/robodojo_sweep` | output root |
| `ROBODOJO` | `$ROBODOJO_DIR` | RoboDojo checkout |
| `SIM_PYTHON` | `python` | Isaac Sim python |
| `SEEDS` / `SEED_START` / `DIFFICULTY` | 3 / 0 / easy | the episode set |
| `ENVS`, `TASKS` | all five domains, all tasks | restrict the set, e.g. `TASKS="lab:A5 household:B2"` |
| `FIRST_TASKS` | none | tasks to queue first, e.g. `FIRST_TASKS="hospital:A1"` |
| `LATENCY` | zero | `roboclock eval --inference-latency`: `zero`, `measured` (see below) or `ideal` |
| `SWEEP_PER_GPU`, `SWEEP_SLOTS_PER_GPU` | from `model_cfg` | override a model's servers and episodes per GPU |
| `LOAD_MAX` | 150% of the core count | a safety net: no Isaac starts while the 1-min load is above this. At zero latency, load overstates CPU use (on the main pod, load 115 came with 51% idle CPU), so the slot count is the real limit |
| `START_STAGGER` | 20 | seconds between Isaac start-ups on the node; each start also waits under the lock until the load is below `LOAD_MAX` |
| `ROBOCLOCK_VULKAN_PROBE_TIMEOUT` | 120 | seconds for roboclock's `vulkaninfo` check before Isaac starts. roboclock's own default is 15 s, and on a busy node the check takes 5-35 s, which fails healthy start-ups as "Vulkan cannot enumerate a GPU" |
| `SWEEP_FRAMES` | 0 | 1 keeps the head JPEG of every request (large) |

Each model's per-GPU numbers, ports and episode timeout are in `model_cfg` in
`scripts/xpolicylab/sweep/common.sh`.

GPU memory per server is about:

| Model | Memory per server |
|---|---|
| Wall-X | 14 GB |
| InternW0-Δ | 29.5 GB |
| VPP2 | 48.5 GB |
| Awomo | 22 GB |

Each Isaac process takes another 5 GB or so. VPP2 needs 80 GB GPUs.

**Measured latency (LTSR).** `LATENCY=measured` lets each query's inference time pass in
simulation while the robot holds its last targets. That time is the server's `latency_ms`.
XPolicyLab starts that clock before the model lock, so an episode sharing a server would also be
charged the other episodes' inference. Run one episode per server, which `launch.sh` enforces.

Use a separate output root, so the zero-latency results stay apart. For VPP2, one 48.5 GB server per
GPU:

```bash
SWEEP_ROOT=$WORK_DIR/robodojo_sweep_measured LATENCY=measured SWEEP_PER_GPU=1 SWEEP_SLOTS_PER_GPU=1 \
  bash scripts/xpolicylab/sweep/start.sh vpp2:0,1,2,3,4,5,6,7
```

## 7. Watch, stop, resume

```bash
bash scripts/xpolicylab/sweep/status.sh       # one line per model, with a rough time left
tail -f $WORK_DIR/robodojo_sweep/launch.log  # servers and slots starting
```

Each model has its own directory, `$SWEEP_ROOT/<model>/`:

| Path | Contents |
|---|---|
| `run_info.txt` | commits, layout and episode set |
| `queue.txt` | episodes not started yet |
| `done.txt` | one line per finished episode: return code, wall time, GPU, port |
| `out/<env>_<task>_s<seed>/episodes.jsonl` | the result |
| `logs/<episode>.log` | the simulator log |
| `qlog/<episode>.jsonl` | one row per model request |
| `servers/<port>.log` | server logs |
| `servers.txt`, `slots.txt` | the PIDs this run started |

**What is normal.**
- A server log may have one handshake traceback, such as `InvalidMessage: did not receive a valid
  HTTP request`. It comes from the readiness check, which opens a plain TCP connection to the port.
- `vulkan_failures.txt` lists start-ups that Isaac retried. A few are harmless.

**What is not normal.** If an episode has no result (a crash, or `rc=124` for a timeout):

- Read its `logs/` file.
- Its `done.txt` line keeps the return code.
- The next resume runs it again.

**Stop.**
- `stop.sh [MODEL...]` kills the slots, running episodes and servers.
- To let running episodes finish first, `touch $SWEEP_ROOT/<model>/STOP`, wait until `status.sh`
  shows `running 0`, then run `stop.sh`.

**Resume.** Run `start.sh` again with the same role or specs. Episodes that already have a result
are not queued again. `start.sh` refuses to start while that model still has slots or servers.

**For automated agent sessions.**
- `start.sh` detaches everything, so run it as an ordinary foreground command.
- Do not start it, or `launch.sh`, as a background task that you later stop. Stopping a task kills
  its whole process tree, which includes `setsid` servers. This killed every server on the main pod
  once.
- Stop processes with `stop.sh` or by PID, never with `pkill -f`.

## 8. Results

When `status.sh` shows `queued 0` and `running 0` for every model on the node, summarize and stop:

```bash
cd $ROBOCLOCK_DIR
for m in awomo; do     # role B: for m in wallx internw0 vpp2
  python3 scripts/xpolicylab/sweep/summarize.py $m | tee $WORK_DIR/robodojo_sweep/$m/final_summary.txt
done
bash scripts/xpolicylab/sweep/stop.sh
tar czf $WORK_DIR/robodojo_sweep_role_A.tgz -C $WORK_DIR --exclude=frames --exclude=industry_local robodojo_sweep
```

For role B, name the archive `robodojo_sweep_role_B.tgz`.

`summarize.py` prints these, followed by roboclock's stage scores:

- a per-task table: physical, temporal and joint success, mean end time, termination reasons and
  physical failures;
- a per-domain table;
- the `TOTAL` line;
- the episodes without a result.

Report the `TOTAL` line of each model, and say where the archive is. The main pod merges the
archives into the eight-model table.

**How to read the scores.**
- Joint success requires both the physical and the temporal check.
- Household A2's `glass_picked_up` stage also fires when the glass is knocked over.
- An industry B2 physical success can be a premature press, made before the warning. Check
  `premature_press_s` against `warning_onset_s`.

## 9. The main pod's run (π0.5, X-VLA, G0.5, OpenWAM)

The main pod ran this procedure from a scratch copy of these scripts:

| Model | GPUs | Episodes at once |
|---|---|---|
| π0.5 | 0, 1 | 6 |
| X-VLA | 2, 3 | 6 |
| G0.5 | 4, 5 | 6 |
| OpenWAM | 6, 7 | 6 |

Each GPU had 3 episodes, sharing 1 or 2 servers. `start.sh fast` reproduces that layout.

π0.5 needs its checkpoint from `scripts/xpolicylab/setup_pi05_server.sh`. The other three need
the setup in their own documents:

- [xvla.md](xvla.md);
- [g05.md](g05.md);
- [openwam.md](openwam.md).

**OpenWAM's prompt cache.** Every new instruction fails once an OpenWAM server has seen 32
distinct prompts. The cause is in XPolicyLab 5d0be149: `openwam/deploy/engine.py`
`_BoundedPromptEmbedCache` raises `KeyError` when it evicts its first entry, because `popitem` calls
the overridden `__getitem__`. A 60-task sweep reaches 32 prompts.

Give the server a copy of its deploy config with a larger cache. The cache only holds text
embeddings, so this does not change the model's output:

```bash
O=$ROBODOJO_DIR/XPolicyLab/policy/OpenWAM/OpenWAM/configs/deploy.yaml
sed 's/^    maxsize: 32 .*/    maxsize: 256/' $O > $WORK_DIR/openwam_deploy_lru256.yaml
export OPENWAM_DEPLOY_CONFIG=$WORK_DIR/openwam_deploy_lru256.yaml   # read by serve_openwam.sh
```
