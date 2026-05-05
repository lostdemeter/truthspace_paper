#!/usr/bin/env python3
"""
Figure 2.2 - phi-level context decay + recursive self-similarity.

Panel A: phi^(-d/tau) context-weight decay with shaded phi-level bands.
Panel B: Recursive golden-rectangle subdivision - each subrectangle is a
         smaller copy of the whole.  No nested text, just shape repetition.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_SOFT, TEAL, RED,
                      GRID, PHI, MUTED)

apply_style()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# =========================================================================
# Panel A : phi-decay of context weights with phi-level bands
# =========================================================================
panel_label(axA, "A")
d = np.arange(1, 16)
w = PHI ** (-d / 2.5)

# phi-level boundaries (exponential bins)
boundaries = [(1, 1, "level 0", "#F4E0C2"),
              (2, 3, "level 1", "#E0EBF3"),
              (4, 7, "level 2", "#CFE0EE"),
              (8, 15, "level 3", "#BFD5E8")]
for lo, hi, name, color in boundaries:
    axA.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.55, zorder=0)
    axA.text((lo + hi) / 2, 1.02, name, ha="center", va="bottom",
             fontsize=9, color=INK_SOFT,
             bbox=dict(boxstyle="round,pad=0.18", fc="white",
                       ec="none", alpha=0.85))

axA.fill_between(d, w, color=GOLD, alpha=0.18, zorder=2)
axA.plot(d, w, color=GOLD, lw=2.2, marker="o", mfc=GOLD,
         mec=INK, mew=0.7, ms=5.5, zorder=3)

axA.set_xlim(0.5, 15.5); axA.set_ylim(0, 1.1)
axA.set_xlabel("Distance to context token (positions)")
axA.set_ylabel(r"$\phi$-weight  $=\phi^{-d/\tau}$")
axA.set_title("phi-decay: context weight by distance", fontsize=12)
axA.grid(True, axis="y", color=GRID, linewidth=0.5, alpha=0.6)
axA.set_axisbelow(True)
axA.spines["left"].set_color(INK); axA.spines["bottom"].set_color(INK)

# =========================================================================
# Panel B : recursive golden-rectangle self-similarity
# =========================================================================
axB.set_xlim(0, PHI ** 2 + 0.05); axB.set_ylim(0, PHI + 0.05)
axB.set_aspect("equal"); axB.axis("off")
panel_label(axB, "B")

# Recursive rectangles: split the long side, peel off a square, recurse.
shades = ["#F6E8C7", "#EFD9A9", "#E5C887", GOLD, GOLD_SOFT,
          "#E0D6BB", "#CFC2A0", "#B89F76"]


def draw_golden(x, y, w, h, depth, horizontal):
    """Draw nested golden rectangles by peeling off squares."""
    if depth == 0 or min(w, h) < 0.04:
        return
    color = shades[depth % len(shades)]
    axB.add_patch(Rectangle((x, y), w, h, facecolor=color,
                            edgecolor=INK, lw=0.7, alpha=0.95))
    # peel a square off the short side
    if horizontal:
        s = h
        axB.plot([x + s, x + s], [y, y + h], color=INK, lw=0.6)
        draw_golden(x + s, y, w - s, h, depth - 1, not horizontal)
    else:
        s = w
        axB.plot([x, x + w], [y + h - s, y + h - s], color=INK, lw=0.6)
        draw_golden(x, y, w, h - s, depth - 1, not horizontal)


# start: golden rectangle phi^2 wide, phi tall  (ratio = phi)
draw_golden(0, 0, PHI ** 2, PHI, depth=8, horizontal=True)

axB.set_ylim(-0.55, PHI + 0.55)
# label and equation positioned manually inside the axes
axB.text(PHI ** 2 / 2, PHI + 0.42, "Infinite self-similarity",
         ha="center", va="bottom", fontsize=12, fontweight="bold",
         color=INK)
axB.text(PHI ** 2 / 2, -0.05,
         r"$\phi = 1 + \dfrac{1}{1 + \dfrac{1}{1 + \cdots}}$",
         ha="center", va="top", fontsize=13, color=INK)

fig.suptitle("phi as the natural geometry of context",
             fontsize=15, fontweight="bold", y=1.04)

save_fig("fig2_2_self_similarity")
