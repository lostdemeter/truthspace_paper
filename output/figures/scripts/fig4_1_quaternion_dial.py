#!/usr/bin/env python3
"""
Figure 4.1 - The 4D Quaternion phi-Dial as a control panel.

Four labelled rotary dials (X: Style, Y: Perspective, Z: Depth, W: Certainty)
each parked at a meaningful value, with a printed quaternion q on the right
and a small unit-3-sphere reminder of the underlying geometry.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD, GOLD_SOFT, RED, TEAL,
                      VIOLET, GRID, PAPER, MUTED)

apply_style()

fig = plt.figure(figsize=(13, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.18)

# ===================================================================
# LEFT : four dials in a 2x2 grid, drawn on a single axes
# ===================================================================
axL = fig.add_subplot(gs[0, 0])
axL.set_xlim(0, 5.6); axL.set_ylim(0, 3.5)
axL.set_aspect("equal"); axL.axis("off")

axes_info = [
    # (col, row, label, color, value in [-1, 1], left_label, right_label)
    (0, 1, "X  -  Style",       RED,    +0.55, "formal",     "casual"),
    (1, 1, "Y  -  Perspective", TEAL,   -0.30, "subjective", "meta"),
    (0, 0, "Z  -  Depth",       VIOLET, +0.20, "terse",      "elaborate"),
    (1, 0, "W  -  Certainty",   GOLD,   -0.65, "definitive", "hedged"),
]


def draw_dial(cx, cy, label, color, value, left_lbl, right_lbl):
    r = 0.62
    # dial face
    axL.add_patch(Circle((cx, cy), r, facecolor="white",
                         edgecolor=INK, lw=1.3, zorder=2))
    # tick marks
    for t in np.linspace(np.pi, 2 * np.pi, 11):
        x0 = cx + (r - 0.05) * np.cos(t)
        y0 = cy + (r - 0.05) * np.sin(t)
        x1 = cx + r * np.cos(t)
        y1 = cy + r * np.sin(t)
        axL.plot([x0, x1], [y0, y1], color=INK_SOFT, lw=0.8, zorder=3)
    # coloured wedge from centre to value
    angle = np.pi + (1 - (value + 1) / 2) * np.pi  # value=-1 -> pi (left), +1 -> 2pi (right)
    axL.add_patch(Wedge((cx, cy), r - 0.05, np.degrees(np.pi),
                        np.degrees(angle) if value >= 0
                        else np.degrees(angle),
                        facecolor=color, alpha=0.18, zorder=2))
    # needle
    nx = cx + (r - 0.08) * np.cos(angle)
    ny = cy + (r - 0.08) * np.sin(angle)
    axL.add_patch(FancyArrowPatch((cx, cy), (nx, ny),
                                  arrowstyle="-|>", color=color,
                                  lw=2.4, mutation_scale=12, zorder=4))
    axL.add_patch(Circle((cx, cy), 0.05, facecolor=INK, zorder=5))
    # title above
    axL.text(cx, cy + r + 0.12, label, ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=color)
    # value readout INSIDE the lower half of the dial
    axL.text(cx, cy - r * 0.55, f"{value:+.2f}",
             ha="center", va="center", fontsize=12,
             color=INK, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.18", fc="white",
                       ec=color, lw=1.0, alpha=0.95))
    # endpoint labels
    axL.text(cx - r - 0.05, cy - 0.05, left_lbl, ha="right", va="center",
             fontsize=8.5, color=INK_SOFT, style="italic")
    axL.text(cx + r + 0.05, cy - 0.05, right_lbl, ha="left", va="center",
             fontsize=8.5, color=INK_SOFT, style="italic")


for col, row, lbl, color, val, ll, rl in axes_info:
    cx = 1.15 + col * 3.05
    cy = 0.95 + row * 1.65
    draw_dial(cx, cy, lbl, color, val, ll, rl)

axL.set_title("Four orthogonal semantic dials  (one per quaternion axis)",
              fontsize=12, color=INK, pad=8)

# ===================================================================
# RIGHT : quaternion expression and unit-3-sphere reminder
# ===================================================================
axR = fig.add_subplot(gs[0, 1])
axR.set_xlim(-1.4, 1.4); axR.set_ylim(-1.6, 1.6)
axR.set_aspect("equal"); axR.axis("off")

# unit sphere outline (W as radius)
theta = np.linspace(0, 2 * np.pi, 200)
axR.plot(np.cos(theta), np.sin(theta), color=INK, lw=1.2)
# meridians for 3D feel
for ang in (0.4, 0.85):
    axR.plot(np.cos(theta) * ang, np.sin(theta), color=GRID, lw=0.6)
    axR.plot(np.cos(theta), np.sin(theta) * ang, color=GRID, lw=0.6)

# the four quaternion components plotted as coloured arrows
axR.add_patch(FancyArrowPatch((0, 0), (0.55, 0), arrowstyle="-|>",
                              color=RED, lw=2.0, mutation_scale=12))
axR.text(0.6, -0.05, "x  i", color=RED, fontsize=10, fontweight="bold",
         va="top")
axR.add_patch(FancyArrowPatch((0, 0), (-0.3, 0.5), arrowstyle="-|>",
                              color=TEAL, lw=2.0, mutation_scale=12))
axR.text(-0.32, 0.55, "y  j", color=TEAL, fontsize=10, fontweight="bold",
         va="bottom", ha="right")
axR.add_patch(FancyArrowPatch((0, 0), (0.2, -0.5), arrowstyle="-|>",
                              color=VIOLET, lw=2.0, mutation_scale=12))
axR.text(0.22, -0.55, "z  k", color=VIOLET, fontsize=10, fontweight="bold",
         va="top")
# w as radius indicator (concentric)
axR.add_patch(Circle((0, 0), 0.65, facecolor="none",
                     edgecolor=GOLD, lw=2.2, ls="--"))
axR.text(0.66, 0.66, "w  (radius)", color=GOLD,
         fontsize=10, fontweight="bold")

# the equation
axR.text(0, -1.25, r"$q \;=\; w \;+\; x\,\mathbf{i} \;+\; y\,\mathbf{j} \;+\; z\,\mathbf{k}$",
         ha="center", va="center", fontsize=14, color=INK)
axR.text(0, -1.5, "unit quaternion in 4D semantic space",
         ha="center", va="center", fontsize=9, color=INK_SOFT,
         style="italic")
axR.set_title("4D quaternion", fontsize=12, color=INK)

fig.suptitle("The 4D Quaternion phi-Dial",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig4_1_quaternion_dial")
