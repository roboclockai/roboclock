"""Plot measured bar motion and support transfer from an exported episode."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(run, output):
    result = json.loads((run / "result.json").read_text())
    with gzip.open(run / "trace.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    time = np.array([r["time"] for r in rows])
    states = [r["state"] for r in rows]
    bars = [s["objects"]["bar"] for s in states]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    axes[0].plot(time, [o["position"][2] for o in bars], color="#167b85", label="Measured bar height")
    for side in ("left", "right"):
        axes[0].plot(time, [s["hands"][side]["position"][2] for s in states], ls=":", lw=1, label=f"{side.title()} TCP height")
    axes[0].set_ylabel("Height (m)")
    axes[1].plot(time, np.array([s["finger_gaps"] for s in states])*1000, label=["Left jaws", "Right jaws"])
    axes[1].set_ylabel("Jaw gap (mm)")
    axes[2].plot(time, [s["human_actuator"]["force"][2] for s in states], label="Human upward force")
    axes[2].plot(time, [sum(o["forces"][:4]) for o in bars], label="Finger contact magnitudes", alpha=.7)
    axes[2].plot(time, [sum(o["support_contacts"].get(k,0) for k in ("RackLeft","RackRight")) for o in bars], label="Hook contact magnitudes")
    axes[2].set_ylabel("Force (N)")
    axes[2].set_xlabel("Simulation time (s)")
    for ax in axes:
        for key,color,label in [("exhaustion_time","#bc4651","Descent"),("catch_time","#207842","Catch"),("rack_time","#7250a0","Rack")]:
            if result[key] is not None:
                ax.axvline(result[key],color=color,ls="--",lw=1,label=label if ax is axes[0] else None)
        ax.grid(alpha=.2)
        ax.legend(loc="best",fontsize=8)
    fig.suptitle(f"B3 Bench Press Assist — {result['difficulty']}, seed {result['seed']}\nMeasured control-rate trace; predicates evaluated at 240 Hz")
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    plot(args.run,args.output)
