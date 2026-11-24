"""Draw the README leaderboard chart (light and dark variants).

Numbers are the Overall PSR / TSR / LTSR columns (Easy level, all 60 tasks) of the
paper's end-to-end and VLM result tables.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GROUPS = [
    ("End-to-end policies, fine-tuned on RoboClock", [
        ("π0.5", "VLA", 12.8, 10.3, 8.3),
        ("X-VLA", "VLA", 10.2, 6.7, 4.8),
        ("OpenWAM-α", "WAM", 14.8, 10.0, 3.8),
        ("InternW0-Δ", "WAM", 12.3, 8.0, 2.2),
    ]),
    ("VLM agents, no robot-specific training", [
        ("Gemini 4 Argon", "VLM", 24.5, 18.2, 8.2),
        ("GPT-6 Astra", "VLM", 22.8, 19.0, 3.0),
        ("GPT-6.1 Sol", "VLM", 21.8, 17.3, 2.3),
        ("Qwen3.8-27B", "VLM, open-weight", 9.2, 5.8, 1.2),
        ("Claude Opus 5.5", "VLM", 16.5, 13.7, 1.0),
    ]),
]
METRICS = ["PSR  physical success", "TSR  + temporal constraints", "LTSR  + inference latency"]
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=["#3987e5", "#d95926", "#199e70"]),
}
BAR, GAP, ROW, HEAD = 0.20, 0.035, 1.0, 0.60


def draw(theme: str, out: Path) -> None:
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9.0, 6.4), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    y, ticks = 0.0, []
    for title, rows in GROUPS:
        y += HEAD
        ax.text(0.25, y + 0.30, title, ha="left", va="center", color=c["ink2"], fontsize=9.5, fontweight="bold",
                bbox=dict(facecolor=c["surface"], edgecolor="none", pad=2.5))
        for name, family, *vals in rows:
            y += ROW
            for k, v in enumerate(vals):
                yy = y + (k - 1) * (BAR + GAP)
                if v > 0:
                    r = min(0.09, v / 2)
                    ax.add_patch(FancyBboxPatch((0, yy - BAR / 2), v, BAR, linewidth=0, facecolor=c["series"][k],
                                                boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=1.0,
                                                clip_on=False))
                    ax.add_patch(plt.Rectangle((0, yy - BAR / 2), min(r, v), BAR, linewidth=0,
                                               facecolor=c["series"][k]))
                bold = k == 2
                ax.text(v + 0.35, yy, f"{v:.1f}", va="center", ha="left", fontsize=8.6 if bold else 7.6,
                        fontweight="bold" if bold else "normal", color=c["ink"] if bold else c["muted"])
            ticks.append((y, name, family))
    ax.set_ylim(y + 0.75, 0.15)
    ax.set_xlim(0, 30)
    ax.set_yticks([])
    for yy, name, family in ticks:
        ax.text(-0.02, yy - 0.13, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=c["ink"], fontsize=10.5)
        ax.text(-0.02, yy + 0.20, family, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=c["muted"], fontsize=7.8)
    ax.set_xticks(range(0, 31, 5))
    ax.set_xticklabels([f"{v}%" for v in range(0, 31, 5)], color=c["muted"], fontsize=8.5)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.xaxis.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(c["axis"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in c["series"]]
    leg = fig.legend(handles, METRICS, loc="upper left", bbox_to_anchor=(0.045, 0.935), ncol=3, frameon=False,
                     fontsize=9, handlelength=1.1, handleheight=0.9, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color(c["ink2"])
    fig.text(0.045, 0.965, "RoboClock leaderboard · Easy level · all 60 tasks · success rate",
             color=c["ink"], fontsize=12, fontweight="bold", va="center")
    fig.subplots_adjust(left=0.21, right=0.965, top=0.885, bottom=0.07)
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "docs" / "images" / "readme"
    for theme in THEMES:
        draw(theme, out / f"leaderboard-{theme}.png")
        print(out / f"leaderboard-{theme}.png")
