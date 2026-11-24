"""Plot physical, temporal and joint outcomes of the recorded diagnostic matrix."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.root / "summary.json").read_text())
    tasks = ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "C1", "C2", "C3"]
    levels = ["easy", "medium", "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 6), sharey=True, layout="constrained")
    for ax, key, title in zip(
        axes,
        ["physical_success", "temporal_success", "success"],
        ["Physical outcome", "Temporal outcome", "Joint success"],
    ):
        values = np.full((10, 3), np.nan)
        labels = {}
        for i, task in enumerate(tasks):
            for j, level in enumerate(levels):
                group = [
                    r
                    for r in rows
                    if r["task"] == task and r["difficulty"] == level and r["runtime_completed"]
                ]
                if group:
                    count = sum(bool(r[key]) for r in group)
                    values[i, j] = count / len(group)
                    labels[(i, j)] = f"{count}/{len(group)}"
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#e5e7eb")
        im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(3), levels)
        ax.set_yticks(range(10), tasks)
        ax.set_title(title)
        for (i, j), label in labels.items():
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if values[i, j] > 0.55 else "#111827",
                fontsize=12,
            )
    fig.colorbar(im, ax=axes, label="Success fraction", shrink=0.75)
    fig.suptitle("Dual ARX X5 scripted baseline — three seeds per difficulty", fontsize=14)
    fig.supxlabel(
        "Counts are successes / completed runtime episodes. Small diagnostic sample; no generalization claim.",
        fontsize=9,
    )
    fig.savefig(args.root / "outcomes.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
