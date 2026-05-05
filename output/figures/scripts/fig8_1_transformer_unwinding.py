#!/usr/bin/env python3
"""
Figure 8.1 - Reverse-engineering Qwen2-7B as a phi-computer.

Top: pipeline of transformer operations, each labelled with its phi-form.
Bottom: a small grid of verified results (numbers, not source-doc citations).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD, RED, TEAL, VIOLET, GOLD_DARK,
                      MUTED, GRID, PAPER)

apply_style()

fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.45)

# =========================================================================
# Top : pipeline
# =========================================================================
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 14.4); ax.set_ylim(0, 2.4)
ax.axis("off")

stages = [
    ("Token\nembed",       "lookup",           MUTED),
    ("RMS\nnorm",          r"$x \cdot \phi^{-\log_\phi r}$",  "#3F6FB3"),
    ("Q / K / V\nproject", r"$\phi$-exp add",  RED),
    ("Attention\nphi-softmax", r"$\frac{\phi^{x_i/\ln\phi}}{\sum\phi^{x_j/\ln\phi}}$", "#D67D2C"),
    ("RoPE\nphi-rotate",   r"$\phi$-phase rotation", VIOLET),
    ("MLP\nphi-SiLU",      r"$x \cdot \phi\text{-sigmoid}(x)$", "#3FA67E"),
    ("LM head\nproject",   "navigation",       MUTED),
]

w = 1.85; h = 1.5; gap = 0.16
x = 0.05
for i, (name, sub, color) in enumerate(stages):
    is_terminal = color is MUTED
    fc = color if not is_terminal else "#E1E5EC"
    txt = "white" if not is_terminal else INK
    ax.add_patch(FancyBboxPatch((x, 0.55), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.16",
                                facecolor=fc, edgecolor=INK, lw=1.1,
                                alpha=0.96))
    ax.text(x + w / 2, 0.55 + h * 0.7, name,
            ha="center", va="center", color=txt,
            fontsize=10.5, fontweight="bold")
    ax.text(x + w / 2, 0.55 + h * 0.28, sub,
            ha="center", va="center", color=txt,
            fontsize=8.5)
    if i < len(stages) - 1:
        a = FancyArrowPatch((x + w + 0.005, 1.3),
                            (x + w + gap - 0.005, 1.3),
                            arrowstyle="-|>", color=INK,
                            lw=1.4, mutation_scale=12)
        ax.add_patch(a)
    x += w + gap

ax.text(7.2, 0.2,
        "Every standard operation rewritten as an exact phi-equivalent",
        ha="center", va="center", fontsize=11, color=INK_SOFT,
        style="italic")

ax.set_title("Qwen2-7B unwinding pipeline", fontsize=13,
             fontweight="bold", color=INK, loc="center", pad=2)

# =========================================================================
# Bottom : verified results
# =========================================================================
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xlim(0, 14); ax2.set_ylim(0, 2)
ax2.axis("off")

results = [
    ("99.9991%",  "full-pass\ncorrelation",      TEAL),
    ("100%",      "token accuracy\n(3 tasks)",   GOLD_DARK),
    (r"$<10^{-14}$", "phi-sigmoid vs\nstandard sigmoid", RED),
    ("12.9x",     "lookup-table\ncompression",   VIOLET),
    ("74",        "unique tetrominoes\ncover 7B params", "#3F6FB3"),
    (r"$\phi\!\approx\!1.57$", "universal bottleneck\nat layer 27", "#D67D2C"),
]
n = len(results)
cw = 14 / n
for i, (big, sub, color) in enumerate(results):
    cx = i * cw + cw / 2
    ax2.add_patch(FancyBboxPatch((cx - cw / 2 + 0.12, 0.15),
                                 cw - 0.24, 1.7,
                                 boxstyle="round,pad=0.04,rounding_size=0.12",
                                 facecolor="white", edgecolor=color,
                                 lw=1.5))
    ax2.text(cx, 1.2, big, ha="center", va="center",
             fontsize=15, fontweight="bold", color=color)
    ax2.text(cx, 0.55, sub, ha="center", va="center",
             fontsize=9, color=INK)

ax2.set_title("Verified results", fontsize=13,
              fontweight="bold", color=INK, loc="center", pad=2)

fig.suptitle("Reverse engineering Qwen2-7B as a phi-computer",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig8_1_transformer_unwinding")
