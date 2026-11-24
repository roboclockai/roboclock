"""Write a model's episode queue: every planned episode that has no result yet, longest budget first.

usage: queue.py OUT_DIR QUEUE_FILE [MODEL]

Reads the sweep settings from the environment, as common.sh exports them (REPO, SIM_PYTHON, ENVS,
TASKS, DIFFICULTY, SEEDS, SEED_START, JOBS_FILE). One line per episode: "ENV TASK SEED MAX_TIME". An
episode counts as finished when OUT_DIR/<env>_<task>_s<seed>/episodes.jsonl is non-empty, so rerunning
after a stop queues only what is left. With JOBS_FILE (lines "MODEL ENV TASK SEED", # comments), only
MODEL's listed episodes are queued; the plan still has to contain them (ENVS, TASKS, SEEDS).
"""
import json
import os
import subprocess
import sys

out_dir, queue_file = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else None
repo, py = os.environ["REPO"], os.environ["SIM_PYTHON"]
tasks = set(os.environ.get("TASKS", "").split())
listed = None
if os.environ.get("JOBS_FILE"):
    if model is None:
        sys.exit("queue.py: JOBS_FILE needs the MODEL argument")
    listed = set()
    for line in open(os.environ["JOBS_FILE"]):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 4:
            sys.exit(f"queue.py: bad JOBS_FILE line (want MODEL ENV TASK SEED): {line.strip()}")
        if fields[0] == model:
            listed.add((fields[1], fields[2], int(fields[3])))
jobs = []
for env in os.environ["ENVS"].split():
    if tasks and not any(t.startswith(env + ":") for t in tasks):
        continue
    plan = subprocess.run(
        [py, "-m", "roboclock", "plan", "--env", env, "--difficulty", os.environ["DIFFICULTY"],
         "--seeds", os.environ["SEEDS"], "--seed-start", os.environ["SEED_START"]],
        cwd=repo, env={**os.environ, "PYTHONPATH": f"{repo}/src"},
        check=True, capture_output=True, text=True).stdout
    for ep in json.loads(plan)["plan"]:
        task, seed = ep["task_id"], ep["seed"]
        if tasks and f"{env}:{task}" not in tasks:
            continue
        if listed is not None:
            if (env, task, seed) not in listed:
                continue
            listed.discard((env, task, seed))
        result = f"{out_dir}/{env}_{task}_s{seed}/episodes.jsonl"
        if os.path.exists(result) and os.path.getsize(result) > 0:
            continue
        jobs.append((-ep["spec"]["max_time"], env, task, seed))
if listed:
    sys.exit(f"queue.py: {len(listed)} listed episodes are not in the plan (ENVS/TASKS/SEEDS), e.g. {sorted(listed)[0]}")
jobs.sort()
first = set(os.environ.get("FIRST_TASKS", "").split())
jobs.sort(key=lambda j: f"{j[1]}:{j[2]}" not in first)  # stable: listed tasks first, each still longest first
with open(queue_file, "w") as f:
    for neg_t, env, task, seed in jobs:
        f.write(f"{env} {task} {seed} {-neg_t:g}\n")
print(f"{len(jobs)} episodes queued, {sum(-j[0] for j in jobs):.0f} s of simulated budget")
