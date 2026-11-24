# Oracle demonstration collection on several nodes

The training set is 100 oracle demonstrations per task (50 easy + 50 medium) for all 60 roboclock
tasks: 6,000 episodes, stored as RoboDojo v1.0 `arx_x5` HDF5 episodes. Collection started on
2026-11-06 on the main pod. It is now split across three nodes by task. Each task is recorded
on exactly one node, and the finished episodes are copied back to the main pod for training.

This document covers what a node needs and which tasks it records. It also covers how to start,
watch and stop a node, and how to bring the data back.

## 1. What gets recorded

| Item | Value |
|---|---|
| Tasks | lab 16, hospital 14, household 10, sports 10, industry 10 |
| Seeds per task | train: 0-49 easy, 50-99 medium. Spares: 100-149 easy, 150-199 medium. Test: 1000-1099, never run |
| Policy | the diagnostic oracle at 50 Hz, with zero inference latency and zero control delay |
| Frames | 25 Hz; three 640x480 cameras (head, left wrist, right wrist); JPEG q95 |
| Kept | only episodes that end in joint (physical + temporal) success. A failed train seed is replaced by the next spare of the same difficulty |
| Size per episode | lab ~265 MB, household ~245 MB, sports ~170 MB, hospital ~140 MB |

The format is described in the docstrings of `scripts/collect/record_demo.py` and
`scripts/collect/splits.py`, and in the data README on the main pod
(`$DEMOS_DIR/roboclock_oracle_v1/README.md`).

## 2. Code and data paths

| Path | Role |
|---|---|
| `scripts/collect/campaign.py` | Scheduler. Runs one `record_demo.py` process per episode and can be restarted at any time |
| `scripts/collect/record_demo.py` | Records one episode |
| `scripts/collect/splits.py` | The seed split |
| `scripts/collect/start_campaigns.sh` | Starts `campaign.py` detached. Reads `ROOT`, `ENVS`, `GPUS` and `PER_GPU` from the environment; extra arguments go to `campaign.py` |
| `scripts/collect/verify_demos.py` | Checks episodes against the RoboDojo and OpenWAM readers (main pod) |
| `$ISAAC_VENV` | Isaac Sim 6.0.1 venv (Python 3.12). This path is hard-coded: it is the default `--python`, and h5py is loaded from it |
| `$DEMOS_DIR/roboclock_oracle_v1` | Data root. Use the same path on every node |
| `/tmp/.isaac-start.lock` | `flock` file that staggers Isaac starts on one machine. It is created automatically |

Layout of the data root:

```
<env>_<task>/arx_x5/data/episode_<seed:07d>.hdf5   successful demonstrations only
_meta/<env>_<task>/seed_<s>.json                    every finished episode: status success|fail|error,
                                                    metrics, timings, code commit (code.commit)
_meta/<env>_<task>/seed_<s>.running                 pid of an episode in flight (valid on this machine only)
_logs/<env>_<task>/seed_<s>.attempt<n>.log          simulator log of each attempt
_status.json  campaign.log  splits.json
_control/gpus  _control/STOP
```

A seed is finished once its `_meta/<env>_<task>/seed_<s>.json` exists, and the campaign never runs
it again. The `_meta` files are all a node needs in order to know what has already been recorded.

## 3. Node assignment

| Node | Hardware | Tasks | Estimated hours, 1 episode per GPU | New data |
|---|---|---|---|---|
| A | main pod <host-ip>: 7 x A100 (GPU 1 is reserved) | lab C1, A2, A4, A5, B2, B4, B5, B7 | ~51 | ~0.2 TB |
| B | 8 x A100 | lab A1, A3, A6, B1, B3, B6, C2, C3 | ~51 | ~0.2 TB |
| C | 4 x L40S (the remote node node `<gpu-node>`) | all of household, hospital, sports and industry (44 tasks) | ~49 | ~0.8 TB |

**Why this split.**
- L40S has RT cores; A100 has none.
- Same tasks compared: A100 recording on node A (s of wall time per s of simulation) against L40S `env.step` measured on the remote node node on 2026-11-01. The L40S was faster by these factors:

  | Task | L40S speed-up |
  |---|---|
  | household A1, C3 | 9-11x |
  | hospital B4, C1 | 6x |
  | lab C1 | 3.8x |
  | sports A1 | 2.4x |

- Rendering takes about 80% of household's wall time, so the render-bound domains go to node C. Lab is mostly physics (CPU), so it stays on the A100s.
- Lab is split between A and B so that both finish at the same time.
- lab C1, the largest task, stays on node A, which already holds its first episodes.

**How the hours were estimated.**
- About 1,420 A100 GPU-hours remain (2026-11-06 17:10). This uses per-task wall times measured on node A at one episode per GPU while the pod was loaded, oracle failures included.
- lab C1 (~270 h: ~80 min per attempt, about half fail) and lab A3 (~170 h) dominate.
- For node C, the planning speed-ups are about 2/3 of the measured ratios. That allows for what recording adds (three cameras, JPEG encoding). The values are: household 6x, hospital 4x, sports and industry 2.5x (industry not measured), lab 3x.
- **Result:** every node needs about 50 hours, which is about 2 days at one episode per GPU. Two episodes per GPU should shorten that if the CPUs keep up (section 5).

The L40S speed-ups are the weakest part of the estimate. After node C's first 2-3 hours, compare
its wall time per episode with node A's in `campaign.log`. If node C falls behind, move whole
tasks off it (section 7), starting with sports and industry.

Each node's `--tasks` list (copy it exactly):

```bash
# Node A (7 x A100)
TASKS=lab:C1,lab:A2,lab:A4,lab:A5,lab:B2,lab:B4,lab:B5,lab:B7

# Node B (8 x A100)
TASKS=lab:A1,lab:A3,lab:A6,lab:B1,lab:B3,lab:B6,lab:C2,lab:C3

# Node C (4 x L40S)
TASKS=household:A1,household:A2,household:A3,household:A4,household:B1,household:B2,household:B3,household:C1,household:C2,household:C3,hospital:A1,hospital:A2,hospital:A3,hospital:A4,hospital:B1,hospital:B2,hospital:B3,hospital:B4,hospital:B5,hospital:B6,hospital:C1,hospital:C2,hospital:C3,hospital:C4,sports:A1,sports:A2,sports:B1,sports:B2,sports:B3,sports:B4,sports:B5,sports:C1,sports:C2,sports:C3,industry:A1,industry:A2,industry:A3,industry:B1,industry:B2,industry:B3,industry:C1,industry:C2,industry:C3,industry:C4
```

## 4. Set up a node

1. **Code.** Use a dedicated checkout that nobody edits while collection runs:

   ```bash
   git clone <repository-url> $ROBOCLOCK_DIR
   cd $ROBOCLOCK_DIR && git checkout --detach origin/main
   ```

   Every node must record from the same code. Use the commit that added this document, or a later commit that does not touch `scripts/collect/` or `src/`. Each sidecar stores the commit as `code.commit`.

   Node A records from `66e69a9`. The commits after it change only docs, `data/` and default output paths, so episodes recorded from either commit are equivalent.

2. **Isaac Sim 6.0.1 venv at `$ISAAC_VENV`.** The install follows the README's Installation section:

   ```bash
   python3.12 -m venv $ISAAC_VENV
   $ISAAC_VENV/bin/pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
   $ISAAC_VENV/bin/pip install --no-deps -e $ROBOCLOCK_DIR
   $ISAAC_VENV/bin/pip install scipy pillow pyyaml gymnasium trimesh numpy-stl opencv-python-headless
   ```

   - The remote node node (node C) already has this venv, so skip the install there. Its editable roboclock install points at `$ROBOCLOCK_DIR`, which does no harm: the campaign sets `PYTHONPATH` to the collection checkout.
   - If a 6.0.1 venv already exists elsewhere, symlink it instead: `ln -s <venv> $ISAAC_VENV`.
   - Do not pip-install h5py into this venv. `record_demo.py` loads Isaac Sim's bundled h5py from under this path.
   - For reference, node A's venv has isaacsim 6.0.1.0, numpy 2.3.1, scipy 1.17.0, pillow 12.2.0 and gymnasium 1.4.0, on NVIDIA driver 580.

3. **Household kitchen (node C only).** The Lightwheel kitchen is not in git. Fetch it into the collection checkout with:

   ```bash
   python scripts/household/fetch_lightwheel_kitchen.py
   ```

4. **Graphics libraries (node C).**
   - The remote node node's system ldconfig has no Vulkan, GLU or Xt. Without them, Isaac aborts with "Missing Isaac Sim graphics libraries".
   - Export the following in the shell that runs the smoke test and `start_campaigns.sh`. The campaign passes its environment on to every episode.

     ```bash
     export LD_LIBRARY_PATH=/tmp/hospital-graphics/usr/lib/x86_64-linux-gnu
     ```

   - `/tmp` does not survive a container restart. Check that the directory exists before every start.

5. **Vulkan.** `vulkaninfo --summary` must list every GPU. On a shared node, check `nvidia-smi`
   too, and give `GPUS` only the GPUs that other jobs leave free.

6. **Smoke test.** Record one episode of one of the node's own tasks into a scratch root, not the data root:

   ```bash
   cd $ROBOCLOCK_DIR
   OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -u \
     scripts/collect/record_demo.py --root /tmp/collect-smoke --env household --task A1 --difficulty easy --seed 0
   python3 -c "import json; d=json.load(open('/tmp/collect-smoke/_meta/household_A1/seed_0.json')); print(d['status'], d['code']['commit'], 'rollout_s', round(d['rollout_s']), 'render_s', round(d['render_s']))"
   ```

   - Expected result: `success`, plus an HDF5 file in `/tmp/collect-smoke/household_A1/arx_x5/data/`.
   - Node C: use the command above as written. It also checks the kitchen and the graphics libraries. On node A this episode took about 14 minutes of wall time (`rollout_s` 810, `render_s` 709); node C should be several times faster.
   - Node B: use `--env lab --task B1` instead.
   - Both seeds succeeded on node A.

## 5. Start, watch and stop

Before the first start on nodes B and C, copy node A's sidecars (section 6.1), so that the node
skips the seeds that node A has already recorded.

**Dry run.** It prints, for each (task, difficulty), the tally and the next seed:

```bash
cd $ROBOCLOCK_DIR
PYTHONPATH=src python scripts/collect/campaign.py \
  --root $DEMOS_DIR/roboclock_oracle_v1 --envs lab,hospital,household,sports,industry \
  --tasks "$TASKS" --dry-run
```

**Start.** Set `TASKS` to the node's list from section 3, then run:

```bash
cd $ROBOCLOCK_DIR
ROOT=$DEMOS_DIR/roboclock_oracle_v1 ENVS=lab,hospital,household,sports,industry \
GPUS=0,1,2,3,4,5,6,7 PER_GPU=2 bash scripts/collect/start_campaigns.sh --tasks "$TASKS"
```

- `ENVS` must name every domain that appears in `--tasks`.
- Set `GPUS` to the node's GPU indices; on node C, for example, `GPUS=0,1,2,3`.
- **Processes per GPU.** On a node used only for collection, start with `PER_GPU=2`. Each Isaac process takes about 4-5 GB of GPU memory.
- Go to 3 per GPU if the GPUs are not saturated and the 1-minute load average stays below about 80% of the CPU cores. This is most likely to pay off on the L40S: 48 GB each, and rendering is not the bottleneck there.
- An episode starts only on a GPU with at least 8000 MiB free (`--min-free-mib`).

**Watch.**
- `_status.json` holds success/target overall and per task, plus the running episodes with their minutes so far.
- `campaign.log` has one line for each start and each finished episode.

```bash
python3 -c "import json; d=json.load(open('$DEMOS_DIR/roboclock_oracle_v1/_status.json')); print(d['updated'], d['success'], '/', d['target']); [print(r) for r in d['running']]"
```

**Change the GPU pool without a restart.** Write the GPU list and the processes per GPU, e.g.
`echo "0,1,2,3 2" > $ROOT/_control/gpus`. Once this file exists, it overrides `GPUS` and `PER_GPU`.

**Drain.** `touch $ROOT/_control/STOP` stops new episodes from starting. The campaign exits when
the running ones finish. Delete the file before the next start.

**Stop the scheduler immediately.** Use `kill -9 <campaign pid>`, and nothing else.
- SIGTERM or Ctrl-C also kills the episodes that are running.
- After `kill -9`, the running episodes finish on their own.
- A restarted campaign finds them through their `.running` files and does not start them again.
- Restarting is safe at any time, for example with a new `--tasks` list: run `kill -9`, then `start_campaigns.sh` again.

## 6. Moving sidecars and data between nodes

### 6.1 Before the first start on node B or C: copy node A's sidecars

First, on node A:
1. Restart node A's campaign with node A's own list.
2. Wait until no episode of the moved tasks is still running: their `_meta/<task>/` must contain no `.running` file whose process is alive.
3. Then run:

```bash
cd $DEMOS_DIR/roboclock_oracle_v1
tar czf $DEMOS_DIR/meta-snapshot.tgz --exclude='*.running' --exclude='*.tmp' splits.json _meta
```

Then, on node B or C:

```bash
mkdir -p $DEMOS_DIR/roboclock_oracle_v1
tar xzf meta-snapshot.tgz -C $DEMOS_DIR/roboclock_oracle_v1
```

**Never copy `.running` files to another machine.** The pid in such a file belongs to node A. If
it matches an unrelated process on the other machine, that seed is blocked forever.

Skipping 6.1 is safe, but the node then records again what node A already has. As of 2026-11-06
17:15 that was 16 finished episodes for node B's tasks and 61 for node C's. The merge keeps node
A's copies.

### 6.2 Bring data back to node A

The copy is incremental: run it as often as you like, and once more at the end. On node A:

```bash
R=$DEMOS_DIR/roboclock_oracle_v1
rsync -a --ignore-existing \
  --exclude '*.running' --exclude '*.tmp' --exclude '_status.json*' --exclude '_control/' \
  --exclude 'campaign.log' --exclude 'splits.json' --exclude 'README.md' \
  <nodeB>:$R/ $R/
rsync -a <nodeB>:$R/campaign.log $R/_logs/campaign-nodeB.log
```

- `--ignore-existing` keeps node A's file whenever both nodes have one.
- Episode and sidecar files are written to `*.tmp` and then renamed, so a copy made while collection is running never picks up a partial file.
- If the nodes cannot reach each other, tar the same paths and move the archive.

Then check the merged data on node A:

```bash
cd $ROBOCLOCK_DIR   # or any checkout at the collection commit
# Tally of every task; with no --tasks this covers all 60 tasks.
PYTHONPATH=src python scripts/collect/campaign.py \
  --root $R --envs lab,hospital,household,sports,industry --dry-run
# Format check against the RoboDojo/OpenWAM readers.
PYTHONPATH=$ROBODOJO_DIR:$ROBODOJO_DIR/XPolicyLab/policy/OpenWAM/OpenWAM \
  $ROBODOJO_DIR/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/collect/verify_demos.py $R --episodes 2
```

## 7. Move a task to another node

1. Restart the source node's campaign without the task: `kill -9`, then start it again with a shorter `--tasks` list. Wait until `_meta/<env>_<task>/` has no live `.running` file.
2. Copy `_meta/<env>_<task>/` to the target node, excluding `.running` files. Copying the task's data directory is optional, because it reaches node A from the source node through 6.2 anyway.
3. Restart the target node's campaign with the task added to `--tasks`.

## 8. Pitfalls

- **"Vulkan cannot enumerate a GPU" at start-up.**
  - On a loaded machine, `vulkaninfo` can take more than 30 s. It also fails outright when any one GPU has no free memory.
  - `record_demo.py` gives the probe 180 s and treats a failed probe as unchecked. A real rendering failure still stops the episode.
  - The campaign retries start-up failures up to 12 times and does not count them against the seed.
  - Isaac starts on one machine go through `/tmp/.isaac-start.lock`, 15 s apart. Other Isaac jobs on the node should take the same lock.
- **One episode per process.** Isaac Sim 6.0.1 hangs on a second reset in the same process; the campaign already starts a fresh process for every episode.
- **Zombies.** PID 1 in these containers does not reap child processes, so an orphaned episode that has finished stays a zombie and `kill -0` still succeeds on it. The campaign treats zombies as finished. Do not use `kill -0` to check episodes by hand.
- **`PYTHONPATH`.** The campaign sets `PYTHONPATH=<checkout>/src` for every episode, so an editable install that points at another checkout is ignored. Set it yourself for commands run by hand.
- **Killing by name.** `pkill -f record_demo` also matches the shell that runs it. Kill by pid.
- **Render speed.** A100s have no RT cores but still render over Vulkan RTX. One three-camera frame takes about 0.15 s on an idle GPU and more than 1 s under load. RTX-class GPUs render faster.
- **Training readers.** XPolicyLab's Pi_05 `process_data.py` drops every v1.0 episode until it is patched (gripper shape, and the `instructions` key). See the data README.
