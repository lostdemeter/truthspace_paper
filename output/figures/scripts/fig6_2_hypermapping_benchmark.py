#!/usr/bin/env python3
"""
Figure 6.2 - HyperMapping vs. neural networks: six-task benchmark.

Numbers from chapter 6 section 6.7.2:
  - Basic (position-matching only):  47.7% average
  - Full  (Self-Similar Transforms + Tachyon Navigation + Geometric RL):
    100.0% on every task.

The clearest visual is a grouped bar chart with a 100% reference line
and arrows showing the improvement on the three "hard" tasks
(sequence prediction, structure learning, function approximation).
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, MUTED, GRID, PHI)

apply_style()

tasks = [
    "XOR\n(non-linear)",
    "Image\nclassification",
    "Sentiment\nanalysis",
    "Function\napproximation",
    "Sequence\nprediction",
    "Structure\nlearning",
]
nn_kind = [
    "MLP",
    "CNN",
    "RNN / Transformer",
    "MLP regression",
    "LSTM / RNN",
    "RL + policy gradient",
]
basic = np.array([100.0, 100.0, 71.4, 15.0, 0.0, 0.0])
full  = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

x = np.arange(len(tasks))
w = 0.34

fig, ax = plt.subplots(figsize=(13, 5.4))

# Bars
b1 = ax.bar(x - w / 2, basic, width=w, color=MUTED, edgecolor=INK,
            lw=0.8, label="Basic: position-matching only",
            zorder=3)
b2 = ax.bar(x + w / 2, full, width=w, color=GOLD, edgecolor=INK,
            lw=0.8, label="Full: + self-similar / Tachyon / geometric RL",
            zorder=3)

# Value annotations
for bar in b1:
    h = bar.get_height()
    if h > 4:
        ax.text(bar.get_x() + bar.get_width() / 2, h - 3.0,
                f"{h:.1f}%", ha="center", va="top",
                color="white" if h >= 50 else INK,
                fontsize=9, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.0f}%", ha="center", va="bottom",
                color=INK_SOFT, fontsize=9, fontweight="bold")
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h - 3.0,
            f"{h:.1f}%", ha="center", va="top",
            color="white", fontsize=9, fontweight="bold")

# 100% reference line
ax.axhline(100, color=INK, ls=":", lw=0.9, alpha=0.4, zorder=1)

# NN labels under each task
for xi, name in zip(x, nn_kind):
    ax.text(xi, -7.5, name, ha="center", va="top",
            fontsize=8.0, color=INK_SOFT, style="italic")

# Delta arrows for big improvements (>50)
deltas = full - basic
for xi, b, f, d in zip(x, basic, full, deltas):
    if d >= 50:
        ax.annotate("", xy=(xi + w / 2 + 0.06, f - 4),
                    xytext=(xi + w / 2 + 0.06, b + 3),
                    arrowprops=dict(arrowstyle="-|>", color=RED,
                                    lw=1.4, mutation_scale=10))
        ax.text(xi + w / 2 + 0.10, (b + f) / 2,
                f"+{d:.0f}%", ha="left", va="center",
                color=RED, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="white", ec=RED, lw=0.6))

# Averages line as box
ax.text(0.985, 0.96,
        f"Basic mean:  47.7%\nFull  mean: 100.0%\n\u0394 = +52.3%",
        transform=ax.transAxes,
        fontsize=10.5, color=INK, ha="right", va="top",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35",
                  fc="white", ec=INK, lw=0.8))

# Axes
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=10)
ax.set_xlim(-0.6, len(tasks) - 0.4)
ax.set_ylim(-13, 112)
ax.set_ylabel("task accuracy  (%)")
ax.set_title("HyperMapping with geometric add-ons matches a NN per category",
             fontsize=12)
ax.legend(loc="lower center", fontsize=9.5,
          facecolor="white", framealpha=0.95, ncol=2,
          bbox_to_anchor=(0.5, -0.32))
ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.6)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)

fig.suptitle("Six-task benchmark: $\\mathbf{47.7\\% \\to 100\\%}$ with geometric techniques",
             fontsize=15, fontweight="bold", y=1.00)

save_fig("fig6_2_hypermapping_benchmark")
