#!/usr/bin/env python3
"""
Figure 9.1 - Navigation replaces inference.

Panel A: full O(N^2) attention fan over N=7 tokens.
Panel B: phi-lattice navigation - the SAME flip vector connects every
         pair of semantically related words on a phi-crystal plane.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, RED, TEAL, VIOLET, GRID, MUTED)

apply_style()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6))

# =========================================================================
# Panel A : O(N^2) attention fan
# =========================================================================
panel_label(axA, "A")
axA.set_xlim(0, 1); axA.set_ylim(-0.05, 1.05)
axA.axis("off")
axA.set_title(r"Traditional inference:  $O(N^{\,2})$",
              fontsize=12, color=INK, pad=8)

N = 7
ty = 0.18
xt = np.linspace(0.07, 0.93, N)

edges = 0
for j in range(N):
    for i in range(j):
        rad = -(0.18 + 0.05 * (j - i))
        a = FancyArrowPatch((xt[i], ty + 0.04), (xt[j], ty + 0.04),
                            connectionstyle=f"arc3,rad={rad}",
                            arrowstyle="-", color=RED,
                            lw=0.9, alpha=0.4)
        axA.add_patch(a)
        edges += 1

for i, x in enumerate(xt):
    axA.add_patch(FancyBboxPatch((x - 0.038, ty - 0.05), 0.076, 0.1,
                                 boxstyle="round,pad=0.01,rounding_size=0.04",
                                 facecolor=RED, edgecolor=INK,
                                 lw=0.9, alpha=0.95, zorder=4))
    axA.text(x, ty, f"T{i+1}", ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="white", zorder=5)

axA.text(0.5, 0.94,
         f"N = {N} tokens   ->   {edges} attention edges  =  N(N-1)/2",
         ha="center", fontsize=10.5, color=INK)
axA.text(0.5, 0.04,
         "Every token attends to every previous token.\n"
         "Cost grows quadratically with sequence length.",
         ha="center", va="bottom", fontsize=9.5, color=INK_SOFT,
         style="italic")

# =========================================================================
# Panel B : phi-lattice navigation
# =========================================================================
panel_label(axB, "B")
axB.set_xlim(-0.5, 11); axB.set_ylim(-0.4, 6.6)
axB.axis("off")
axB.set_title(r"phi-lattice navigation:  $O(1)$ per step",
              fontsize=12, color=INK, pad=8)

# faint lattice grid
for gx in np.arange(0, 11.1, 0.5):
    axB.axvline(gx, color=GRID, lw=0.4, alpha=0.5, zorder=0)
for gy in np.arange(0, 6.1, 0.5):
    axB.axhline(gy, color=GRID, lw=0.4, alpha=0.5, zorder=0)

words = {
    "king":  (0.6, 4.6), "queen":  (3.6, 4.6),
    "man":   (0.6, 3.0), "woman":  (3.6, 3.0),
    "uncle": (0.6, 1.4), "aunt":   (3.6, 1.4),
    "hot":   (6.4, 1.4), "cold":   (6.4, 4.6),
    "warm":  (9.4, 1.4), "cool":   (9.4, 4.6),
}
for w, (x, y) in words.items():
    axB.scatter(x, y, s=58, color=INK, zorder=4)
    axB.text(x, y - 0.42, w, ha="center", va="top",
             fontsize=10, color=INK, zorder=4)

# gender-flip arrows (red, horizontal)
for a, b in [("king", "queen"), ("man", "woman"), ("uncle", "aunt")]:
    xa, ya = words[a]; xb, yb = words[b]
    axB.add_patch(FancyArrowPatch((xa + 0.22, ya), (xb - 0.22, yb),
                                  arrowstyle="-|>", color=RED,
                                  lw=2.0, mutation_scale=12, zorder=3))

# temperature-flip arrows (violet, vertical)
for a, b in [("hot", "cold"), ("warm", "cool")]:
    xa, ya = words[a]; xb, yb = words[b]
    axB.add_patch(FancyArrowPatch((xa, ya + 0.22), (xb, yb - 0.22),
                                  arrowstyle="-|>", color=VIOLET,
                                  lw=2.0, mutation_scale=12, zorder=3))

axB.text(2.1, 5.6, "gender flip", color=RED,
         fontsize=11, fontweight="bold", ha="center")
axB.text(2.1, 5.3, "(same sign-bits flip for every pair)",
         color=RED, fontsize=8.5, ha="center", style="italic")
axB.text(7.9, 5.6, "temperature flip", color=VIOLET,
         fontsize=11, fontweight="bold", ha="center")
axB.text(7.9, 5.3, "(orthogonal crystal plane)",
         color=VIOLET, fontsize=8.5, ha="center", style="italic")

axB.text(5.0, 0.05,
         "Each relationship is a fixed direction in the lattice.\n"
         r"Traverse it once, apply to any word:  960$\times$ compression, $O(1)$ lookup.",
         ha="center", va="bottom", fontsize=9.5, color=INK_SOFT,
         style="italic")

fig.suptitle("Navigation replaces inference",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig9_1_navigation_vs_inference")
