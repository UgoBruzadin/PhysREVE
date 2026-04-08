"""
plot_progress.py — PhysREVE experiment progress tracker

Reads all experiments/EXP_*/results.json files and plots how accuracy
and improvement change across experiment versions.

Usage:
    python plot_progress.py
    python plot_progress.py --save          # save figure to experiments/progress.png
    python plot_progress.py --no-show       # headless (e.g. on Colab / CI)
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

TITLE_SIZE = 16
LABEL_SIZE = 12
TICK_SIZE = 11
ANNOTATION_SIZE = 10
LEGEND_SIZE = 11
TABLE_FONT_SIZE = 10

# ── Load all results ──────────────────────────────────────────────────────────


def load_experiments(root: str = "experiments") -> list[dict]:
    records = []
    for exp_dir in sorted(Path(root).glob("EXP_*")):
        results_file = exp_dir / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            data = json.load(f)
        records.append(data)
    records.sort(key=lambda r: r["exp_id"])
    return records


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_progress(records: list[dict], save: bool = False, show: bool = True):
    if not records:
        print("No experiment results found. Run the notebook first.")
        return

    ids = [r["exp_id"] for r in records]
    baseline = [r["results"]["baseline_acc"] * 100 for r in records]
    physreve = [r["results"]["physreve_acc"] * 100 for r in records]
    improvement = [r["results"]["improvement"] * 100 for r in records]
    descs = [r["description"] for r in records]
    x = np.arange(len(ids))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "PhysREVE — Experiment Progress\nBCI IV 2a · Subject 1 · 4-class Motor Imagery",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.38)

    BLUE = "#2563eb"
    GREY = "#94a3b8"
    GREEN = "#16a34a"
    RED = "#dc2626"
    CHANCE = 25.0

    # ── A: Accuracy over experiments ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])  # full top row

    ax.plot(
        x, baseline, color=GREY, marker="o", lw=2, ms=7, label="Baseline (random init)"
    )
    ax.plot(
        x,
        physreve,
        color=BLUE,
        marker="o",
        lw=2,
        ms=7,
        label="PhysREVE (physics pretrained)",
    )
    ax.axhline(CHANCE, ls="--", color=RED, lw=1.2, alpha=0.6, label="Chance (25%)")

    # Shade the gap between the two lines
    ax.fill_between(
        x,
        baseline,
        physreve,
        where=[p >= b for p, b in zip(physreve, baseline)],
        alpha=0.12,
        color=GREEN,
        label="PhysREVE ahead",
    )
    ax.fill_between(
        x,
        baseline,
        physreve,
        where=[p < b for p, b in zip(physreve, baseline)],
        alpha=0.12,
        color=RED,
        label="Baseline ahead",
    )

    # Annotate each point with its value
    for xi, (b, p) in enumerate(zip(baseline, physreve)):
        ax.annotate(
            f"{p:.1f}%",
            (xi, p),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=ANNOTATION_SIZE,
            color=BLUE,
            fontweight="bold",
        )
        ax.annotate(
            f"{b:.1f}%",
            (xi, b),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=ANNOTATION_SIZE,
            color=GREY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=TICK_SIZE)
    ax.set_ylabel("Test Accuracy (%)", fontsize=LABEL_SIZE)
    ax.set_title("Model Accuracy per Experiment", fontsize=TITLE_SIZE)
    ax.set_ylim(0, 75)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left")
    ax.grid(alpha=0.25, axis="y")

    # ── B: Improvement (delta) bar chart ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    colors = [GREEN if v >= 0 else RED for v in improvement]
    bars = ax2.bar(x, improvement, color=colors, alpha=0.85, width=0.5)
    ax2.axhline(0, color="black", lw=0.8)

    for bar, val in zip(bars, improvement):
        ypos = val + (0.3 if val >= 0 else -0.8)
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:+.1f}%",
            ha="center",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(ids, fontsize=TICK_SIZE)
    ax2.set_ylabel("PhysREVE − Baseline (%)", fontsize=LABEL_SIZE)
    ax2.set_title("Improvement Over Baseline", fontsize=TITLE_SIZE)
    ax2.grid(alpha=0.25, axis="y")

    # ── C: Description table ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    table_data = [
        [
            r["exp_id"],
            r["description"][:55] + ("…" if len(r["description"]) > 55 else ""),
        ]
        for r in records
    ]
    table = ax3.table(
        cellText=table_data,
        colLabels=["ID", "Description"],
        cellLoc="left",
        loc="center",
        colWidths=[0.18, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONT_SIZE)
    table.scale(1, 1.6)
    # Style header
    for col in range(2):
        table[0, col].set_facecolor("#1e3a5f")
        table[0, col].set_text_props(color="white", fontweight="bold")
    # Alternate row shading
    for row in range(1, len(table_data) + 1):
        bg = "#f0f4ff" if row % 2 == 0 else "white"
        for col in range(2):
            table[row, col].set_facecolor(bg)

    ax3.set_title("Experiment Log", fontsize=11, pad=10)

    if save:
        out = Path("experiments") / "progress.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    if show:
        plt.show()

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PhysREVE experiment progress")
    parser.add_argument(
        "--save", action="store_true", help="Save figure to experiments/progress.png"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Don't call plt.show() (headless)"
    )
    args = parser.parse_args()

    records = load_experiments()
    print(f"Loaded {len(records)} experiment(s): {[r['exp_id'] for r in records]}")
    plot_progress(records, save=args.save, show=not args.no_show)
