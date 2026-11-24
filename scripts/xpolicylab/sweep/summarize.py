"""Per-task and per-domain success of one model's sweep, then roboclock's stage scores.

usage: summarize.py MODEL [SWEEP_ROOT]      (SWEEP_ROOT defaults to $SWEEP_ROOT or $WORK_DIR/robodojo_sweep)

physical = task_success, temporal = temporal_success, joint = joint_success (both). Episodes without a
result (timeouts, crashes, episodes still running) are listed separately. The stage scores come from
`python -m roboclock.evaluation.score OUT --episodes`, run with $SIM_PYTHON.
"""
import collections
import json
import os
import re
import subprocess
import sys

M = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SWEEP_ROOT", os.path.expandvars("$WORK_DIR/robodojo_sweep"))
W = f"{root}/{M.replace('/', '_')}"  # an API model id: nv/gcp/google/... -> nv_gcp_google_...
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
done = {}
if os.path.exists(f"{W}/done.txt"):
    for line in open(f"{W}/done.txt"):
        g = re.match(r"(\S+) rc=(\d+) wall=(\d+)s", line)
        if g:
            done[g.group(1)] = (int(g.group(2)), int(g.group(3)))
rows, missing = [], []
for name, (rc, wall) in sorted(done.items()):
    f = f"{W}/out/{name}/episodes.jsonl"
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        missing.append(f"{name} rc={rc}")
        continue
    d = json.loads(open(f).readline())
    d["_wall"] = wall
    rows.append(d)

if os.path.exists(f"{W}/run_info.txt"):
    print(open(f"{W}/run_info.txt").read().rstrip())
print(f"{M}: {len(rows)} episodes with results, {len(missing)} without" + (f": {', '.join(missing)}" if missing else ""))
print(f"{'task':16s} {'n':>2s} {'phys':>4s} {'temp':>4s} {'joint':>5s} {'t_end':>6s}  ends | failures")
by_task = collections.defaultdict(list)
for d in rows:
    by_task[(d["environment"], d["task_id"])].append(d)
dom = collections.defaultdict(lambda: [0, 0, 0, 0])
for (env, task), g in sorted(by_task.items()):
    ph = sum(bool(d["task_success"]) for d in g)
    tp = sum(bool(d["temporal_success"]) for d in g)
    jt = sum(bool(d["joint_success"]) for d in g)
    ends = collections.Counter(str(d["termination_reason"]) for d in g)
    fails = collections.Counter(f for d in g for f in (d.get("physical_failures") or []))
    fails.update("overheat" for d in g if d.get("overheat_failure"))
    t = sum(d["elapsed_time"] for d in g) / len(g)
    desc = " ".join(f"{k}:{v}" for k, v in ends.items())
    if fails:
        desc += "  | " + " ".join(f"{k}:{v}" for k, v in fails.most_common(3))
    print(f"{env + ' ' + task:16s} {len(g):2d} {ph:4d} {tp:4d} {jt:5d} {t:6.1f}  {desc}")
    a = dom[env]
    a[0] += len(g); a[1] += ph; a[2] += tp; a[3] += jt
print("\nper domain: physical / temporal / joint")
tot = [0, 0, 0, 0]
for env, (n, ph, tp, jt) in sorted(dom.items()):
    print(f"{env:12s} n={n:3d}  physical {ph}/{n}  temporal {tp}/{n}  joint {jt}/{n}")
    tot = [x + y for x, y in zip(tot, (n, ph, tp, jt))]
print(f"TOTAL n={tot[0]}  physical {tot[1]}/{tot[0]}  temporal {tot[2]}/{tot[0]}  joint {tot[3]}/{tot[0]}")
walls = [d["_wall"] for d in rows]
if walls:
    print(f"episode wall time: {sum(walls) / 3600:.1f} slot-hours in total, longest {max(walls)} s")

print("\n== stage scores")
py = os.environ.get("SIM_PYTHON", "python")
score = subprocess.run([py, "-m", "roboclock.evaluation.score", f"{W}/out", "--episodes"], cwd=repo,
                       env={**os.environ, "PYTHONPATH": f"{repo}/src"}, capture_output=True, text=True)
print("\n".join(l for l in (score.stdout + score.stderr).splitlines() if " seed=" not in l))
