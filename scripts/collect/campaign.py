"""Collect oracle demonstrations for every task on several GPUs, resumably.

  PYTHONPATH=src python scripts/collect/campaign.py \\
      --root $DEMOS_DIR/roboclock_oracle_v1 --envs lab,hospital,household,sports \\
      --gpus 2,3,4,5,6,7 --per-gpu 2

Each episode runs in its own process (`record_demo.py`; Isaac Sim 6.0.1 hangs on a second
reset in one process). Seeds come from `splits.py`: 50 easy + 50 medium successful
demonstrations per task, with a failed train seed replaced by the next spare seed of the same
difficulty. The campaign can be stopped and restarted at any time: an episode is finished once
its `_meta/<task>/seed_<s>.json` exists, and only finished episodes count.

Control files, read every loop:
- `<root>/_control/STOP`: dispatch nothing new; exit when the running episodes finish.
- `<root>/_control/gpus`: e.g. `2,3,4,5,6,7 2` (GPU list, processes per GPU) to change the
  pool without a restart.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import splits  # noqa: E402

MAX_ATTEMPTS = 3  # a crashed or timed-out episode is retried; an oracle failure is not
MAX_STARTS = 12  # start-up failures are retried up to this many attempts in total
ISAAC_START_LOCK = os.path.join(tempfile.gettempdir(), ".isaac-start.lock")


def task_list(envs, only, exclude=()):
    from roboclock.tasks.registry import task_registry

    tasks = [(env, task) for env in envs for task in task_registry(env)]
    if only:
        wanted = set(only)
        tasks = [t for t in tasks if f"{t[0]}:{t[1]}" in wanted]
    return [t for t in tasks if f"{t[0]}:{t[1]}" not in set(exclude)]


def max_time(environment, task_id, difficulty, seed):
    from roboclock.config import load_config
    from roboclock.tasks.registry import make_task

    config = load_config(environment=environment)
    return float(make_task(task_id, difficulty, seed, config, environment=environment).spec.max_time)


class Slot:
    """One (task, difficulty): needs `target` successful demonstrations."""

    def __init__(self, root, environment, task_id, difficulty, target):
        self.environment, self.task_id, self.difficulty = environment, task_id, difficulty
        self.dir = splits.task_dir(environment, task_id)
        self.meta = root / "_meta" / self.dir
        self.candidates = splits.train_seeds(difficulty) + splits.spare_seeds(difficulty)
        self.target = target
        self.running = set()
        self.attempts = {}

    def record(self, seed):
        path = self.meta / f"seed_{seed}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def errors(self, seed):
        """Attempts that ended without a result (numbers the attempt logs)."""
        return len(list(self.meta.glob(f"seed_{seed}.attempt*.error.json")))

    def crashes(self, seed):
        """Failed attempts that got past Isaac start-up and env.reset(); start-up failures
        (Vulkan enumeration, renderer not ready) are the machine's, not the seed's."""
        count = 0
        for note in self.meta.glob(f"seed_{seed}.attempt*.error.json"):
            try:
                count += bool(json.loads(note.read_text()).get("started", True))
            except json.JSONDecodeError:
                count += 1
        return count

    def tally(self):
        done = {"success": 0, "fail": 0, "error": 0}
        for seed in self.candidates:
            row = self.record(seed)
            if row:
                done[row["status"]] += 1
        return done

    def inflight(self):
        """Episodes running now: this campaign's, plus ones an earlier campaign process left
        running (they finish on their own and write their result)."""
        orphans = [
            s for s in self.candidates
            if s not in self.running and self.record(s) is None and self._alive(s)
        ]
        return len(self.running) + len(orphans)

    def next_seed(self):
        tally = self.tally()
        if tally["success"] + self.inflight() >= self.target:
            return None
        for seed in self.candidates:
            if seed in self.running or self.record(seed) is not None:
                continue
            if self._alive(seed):
                continue
            return seed
        return None

    def _alive(self, seed):
        """An episode left running by an earlier campaign process."""
        lock = self.meta / f"seed_{seed}.running"
        if not lock.exists():
            return False
        try:
            pid = int(lock.read_text().split()[0])
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError, IndexError):
            return False
        # An episode whose campaign process exited is re-parented to PID 1, which does not reap
        # children in this container: once it ends it stays a zombie, and kill(pid, 0) still
        # succeeds. Only a non-zombie process is a running episode.
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            return False
        return state != "Z"


def read_pool(root, gpus, per_gpu):
    control = root / "_control" / "gpus"
    if control.exists():
        parts = control.read_text().split()
        gpus = [int(g) for g in parts[0].split(",") if g]
        per_gpu = int(parts[1]) if len(parts) > 1 else per_gpu
    return [(g, k) for g in gpus for k in range(per_gpu)]


def free_memory_mib():
    """{gpu index: free MiB} from nvidia-smi; {} if it fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        return {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def write_status(root, slots, running, started):
    rows = {}
    for slot in slots:
        tally = slot.tally()
        rows.setdefault(slot.dir, {})[slot.difficulty] = {
            **tally,
            "running": sorted(slot.running),
            "target": slot.target,
        }
    total = sum(r[d]["success"] for r in rows.values() for d in r)
    target = sum(r[d]["target"] for r in rows.values() for d in r)
    status = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_h": round((time.time() - started) / 3600, 2),
        "success": total,
        "target": target,
        "running": [
            {"gpu": job["gpu"], "task": job["slot"].dir, "difficulty": job["slot"].difficulty,
             "seed": job["seed"], "minutes": round((time.time() - job["start"]) / 60, 1)}
            for job in running.values()
        ],
        "tasks": rows,
    }
    tmp = root / "_status.json.tmp"
    tmp.write_text(json.dumps(status, indent=1))
    os.replace(tmp, root / "_status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--envs", default="lab,hospital,household,sports")
    parser.add_argument("--exclude", default="", help="Comma-separated env:task to skip, e.g. household:C1")
    parser.add_argument("--tasks", help="Comma-separated env:task subset, e.g. lab:A1,sports:B2")
    parser.add_argument("--per-difficulty", type=int, default=splits.TRAIN_PER_DIFFICULTY)
    parser.add_argument("--difficulties", default=",".join(splits.TRAIN_DIFFICULTIES),
                        help="Comma-separated subset of easy,medium, e.g. medium for one preview demo per task")
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    parser.add_argument("--per-gpu", type=int, default=2)
    parser.add_argument("--stagger", type=float, default=30.0, help="Seconds between Isaac starts")
    parser.add_argument("--min-free-mib", type=int, default=8000,
                        help="Start an episode on a GPU only with this much free memory (Isaac ~4 GB)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-base", type=float, default=3600.0)
    parser.add_argument("--timeout-per-sim-s", type=float, default=300.0,
                        help="Hard backstop; hangs are caught by --stall-timeout")
    parser.add_argument("--stall-timeout", type=float, default=1500.0,
                        help="Kill an episode whose log has not changed for this long (s)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    for sub in ("_meta", "_logs", "_control"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    tasks = task_list(args.envs.split(","), args.tasks.split(",") if args.tasks else None,
                      [t for t in args.exclude.split(",") if t])
    table = splits.split_table(tasks)
    split_file = root / "splits.json"
    if split_file.exists():
        old = json.loads(split_file.read_text())
        for key, entry in table.items():
            if key in old and old[key] != entry:
                raise SystemExit(f"{split_file} disagrees with splits.py for {key}")
        old.update(table)
        table = old
    split_file.write_text(json.dumps(table, indent=1))

    slots = [
        Slot(root, env, task, difficulty, args.per_difficulty)
        for env, task in tasks
        for difficulty in args.difficulties.split(",")
    ]
    if args.dry_run:
        for slot in slots:
            print(slot.dir, slot.difficulty, slot.tally(), "next", slot.next_seed())
        return

    started = time.time()
    running = {}  # pid -> job
    last_start = 0.0
    stopping = False

    def terminate(signum, frame):
        for job in running.values():
            try:
                os.killpg(job["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    while True:
        # Reap finished and timed-out episodes.
        for pid, job in list(running.items()):
            code = job["proc"].poll()
            try:
                quiet = time.time() - Path(job["log"]).stat().st_mtime
            except OSError:
                quiet = 0.0
            # A live episode writes its log at least once per simulated second (record_demo.py);
            # Isaac's silent start-up hangs write nothing. The hard timeout is only a backstop.
            timed_out = code is None and (
                time.time() - job["start"] > job["timeout"] or quiet > args.stall_timeout
            )
            if timed_out:
                try:
                    os.killpg(job["proc"].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                job["proc"].wait()
                code = "timeout"
            if code is None:
                continue
            slot, seed = job["slot"], job["seed"]
            slot.running.discard(seed)
            del running[pid]
            (slot.meta / f"seed_{seed}.running").unlink(missing_ok=True)
            if slot.record(seed) is None:
                # Crash, timeout or Isaac start-up failure: keep a note and retry.
                attempt = slot.errors(seed) + 1
                try:
                    started = "[record] reset done" in Path(job["log"]).read_text(errors="replace")
                except OSError:
                    started = True
                note = {"seed": seed, "exit": code, "log": str(job["log"]), "attempt": attempt,
                        "started": started}
                (slot.meta / f"seed_{seed}.attempt{attempt}.error.json").write_text(json.dumps(note))
                if slot.crashes(seed) >= MAX_ATTEMPTS or attempt >= MAX_STARTS:
                    (slot.meta / f"seed_{seed}.json").write_text(
                        json.dumps({"status": "error", "seed": seed, "difficulty": slot.difficulty,
                                    "reason": f"{attempt} attempts without a result", "last_exit": code})
                    )
            print(f"[{time.strftime('%H:%M:%S')}] done {slot.dir} {slot.difficulty} seed={seed} exit={code} "
                  f"-> {(slot.record(seed) or {}).get('status', 'retry')}", flush=True)

        stopping = stopping or (root / "_control" / "STOP").exists()
        pool = read_pool(root, [int(g) for g in args.gpus.split(",")], args.per_gpu)
        busy = {(job["gpu"], job["k"]) for job in running.values()}
        free = [p for p in pool if p not in busy]

        if not stopping and free and time.time() - last_start >= args.stagger:
            # GPUs are shared with other jobs: only start where an Isaac process fits.
            memory = free_memory_mib()
            free = [p for p in free if memory.get(p[0], 0) >= args.min_free_mib]
        if not stopping and free and time.time() - last_start >= args.stagger:
            # Least-progressed slot first, so every task fills evenly.
            order = sorted(slots, key=lambda s: (s.tally()["success"] + s.inflight()))
            for slot in order:
                seed = slot.next_seed()
                if seed is None:
                    continue
                gpu, k = min(free, key=lambda p: (sum(j["gpu"] == p[0] for j in running.values()),
                                                  -memory.get(p[0], 0)))
                attempt = slot.errors(seed) + 1
                log = root / "_logs" / slot.dir / f"seed_{seed}.attempt{attempt}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                slot.meta.mkdir(parents=True, exist_ok=True)
                timeout = args.timeout_base + args.timeout_per_sim_s * max_time(
                    slot.environment, slot.task_id, slot.difficulty, seed
                )
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO / "src"),
                           OMNI_KIT_ACCEPT_EULA="YES")
                env.pop("TRITON_PTXAS_PATH", None)
                cmd = [
                    args.python, "-u", str(HERE / "record_demo.py"),
                    "--root", str(root), "--env", slot.environment, "--task", slot.task_id,
                    "--difficulty", slot.difficulty, "--seed", str(seed),
                ]
                # Every session on this pod staggers Isaac starts through this lock (simultaneous
                # starts fail with "Vulkan cannot enumerate a GPU").
                cmd = ["bash", "-c", f'flock {ISAAC_START_LOCK} sleep 15 && exec "$@"', "_", *cmd]
                with open(log, "w") as stream:
                    proc = subprocess.Popen(
                        cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                (slot.meta / f"seed_{seed}.running").write_text(f"{proc.pid} gpu{gpu}")
                running[proc.pid] = dict(proc=proc, slot=slot, seed=seed, gpu=gpu, k=k,
                                         start=time.time(), timeout=timeout, log=log)
                slot.running.add(seed)
                last_start = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] start {slot.dir} {slot.difficulty} seed={seed} "
                      f"gpu={gpu} timeout={timeout / 60:.0f}min", flush=True)
                break

        write_status(root, slots, running, started)
        if not running and (stopping or all(s.next_seed() is None for s in slots)):
            print("campaign finished" if not stopping else "campaign stopped", flush=True)
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
