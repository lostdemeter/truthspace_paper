#!/usr/bin/env python3
"""
Figure 12.1 - The path forward for geometric AI.

A 5-stage road of consequences building on the foundation:
phi-Lattice  |  phi-Computer  |  Irreducible Shape
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD, GOLD_SOFT, RED, TEAL,
                      VIOLET, GRID, MUTED, PAPER)

apply_style()

fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.6)
ax.axis("off")

# ----------------------------------------------------------------- foundation
foundations = ["phi-Lattice", "phi-Computer", "Irreducible Shape"]
fw = 3.4; fy = 0.45
for i, f in enumerate(foundations):
    fx = 1.2 + i * (fw + 0.6)
    ax.add_patch(FancyBboxPatch((fx, fy), fw, 0.85,
                                boxstyle="round,pad=0.04,rounding_size=0.18",
                                facecolor=GOLD_SOFT, edgecolor=GOLD,
                                lw=1.4))
    ax.text(fx + fw / 2, fy + 0.42, f,
            ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK)

ax.text(7, 0.18, "FOUNDATION  ( proven )", ha="center", va="bottom",
        fontsize=9.5, color=INK_SOFT, style="italic")

# ----------------------------------------------------------------- arrows up
for i in range(3):
    fx = 1.2 + i * (fw + 0.6) + fw / 2
    a = FancyArrowPatch((fx, 1.4), (fx + 0.3, 2.2),
                        arrowstyle="-|>", color=GOLD,
                        lw=1.0, mutation_scale=10, alpha=0.5)
    ax.add_patch(a)

# ----------------------------------------------------------------- path forward
stages = [
    ("Trivial AI",         "$O(\\log N)$\ncomplexity",   "#3F6FB3"),
    ("Platonic Ideals",    "~100 fixed-points\nin phi-space", VIOLET),
    ("Recursive\nBootstrap", "discover how\nto discover",  "#D67D2C"),
    ("Self-describing\ngeometry", "model = map\nof its own state", TEAL),
    ("Human-AI\nalignment", "shared coordinates,\nshared understanding", RED),
]
sw = 2.2; sh = 1.6; sgap = 0.45
total_w = len(stages) * sw + (len(stages) - 1) * sgap
x0 = (14 - total_w) / 2
sy = 2.5
for i, (head, sub, color) in enumerate(stages):
    sx = x0 + i * (sw + sgap)
    ax.add_patch(FancyBboxPatch((sx, sy), sw, sh,
                                boxstyle="round,pad=0.04,rounding_size=0.16",
                                facecolor=color, edgecolor=INK, lw=1.1,
                                alpha=0.95))
    ax.text(sx + sw / 2, sy + sh * 0.68, head,
            ha="center", va="center", color="white",
            fontsize=10.5, fontweight="bold")
    ax.text(sx + sw / 2, sy + sh * 0.28, sub,
            ha="center", va="center", color="white",
            fontsize=8.5, style="italic")
    if i < len(stages) - 1:
        a = FancyArrowPatch((sx + sw + 0.04, sy + sh / 2),
                            (sx + sw + sgap - 0.04, sy + sh / 2),
                            arrowstyle="-|>", color=INK,
                            lw=1.4, mutation_scale=12)
        ax.add_patch(a)

ax.text(7, sy + sh + 0.25, "PATH FORWARD  ( consequences of the proof )",
        ha="center", va="bottom", fontsize=9.5, color=INK_SOFT,
        style="italic")

ax.set_title("The path forward for geometric AI",
             fontsize=15, fontweight="bold", color=INK, pad=14, loc="center")

save_fig("fig12_1_implications")
