#!/usr/bin/env python3
"""
Figure 1.1 - The Vacuum Forming Hypothesis.

Three stacked layers:
  TOP    : raw training data (scatter cloud)
  MIDDLE : the smooth 'plastic skin' that LLMs learn (surface fit)
  BOTTOM : the hidden phi-lattice interior structure (what TruthSpace seeks)

Vacuum-suction arrows pull the skin onto the unseen mold below.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD, RED, TEAL, GRID, PAPER, MUTED)

apply_style()
rng = np.random.default_rng(7)

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(-0.3, 6.2)
ax.set_aspect("equal"); ax.axis("off")

# --- shared underlying signal ---
xs = np.linspace(0.5, 9.5, 400)
mold = 0.35 * np.sin(1.6 * xs) + 0.15 * np.sin(3.1 * xs + 0.4)

# ============ TOP: training data cloud ============
y_top = 5.1
label_box = dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none", alpha=0.92)
ax.text(0.4, y_top + 0.55, "Training data",
        fontsize=11.5, fontweight="bold", color=INK, bbox=label_box,
        zorder=6)
ax.text(0.4, y_top + 0.28, "(observed token sequences)",
        fontsize=9, color=INK_SOFT, style="italic", bbox=label_box,
        zorder=6)
n = 80
sx = rng.uniform(2.4, 9.4, n)
sy = (y_top + 0.35 * np.sin(1.6 * sx) + 0.15 * np.sin(3.1 * sx + 0.4)
      + rng.normal(0, 0.18, n))
ax.scatter(sx, sy, s=22, color=GOLD, edgecolors=INK,
           linewidths=0.4, alpha=0.9, zorder=3)

# ============ MIDDLE: surface skin ============
y_mid = 3.0
ax.text(0.4, y_mid + 0.62, "Surface skin learned by training",
        fontsize=11.5, fontweight="bold", color=INK, bbox=label_box,
        zorder=6)
ax.text(0.4, y_mid + 0.35, r"smooth fit through the data $\rightarrow$ what the model 'knows'",
        fontsize=9, color=INK_SOFT, style="italic", bbox=label_box,
        zorder=6)
xs_curve = xs[xs >= 2.4]
mold_curve = 0.35 * np.sin(1.6 * xs_curve) + 0.15 * np.sin(3.1 * xs_curve + 0.4)
ax.fill_between(xs_curve, y_mid + mold_curve - 0.06, y_mid + mold_curve + 0.06,
                color=TEAL, alpha=0.22, zorder=2)
ax.plot(xs_curve, y_mid + mold_curve, color=TEAL, lw=2.4, zorder=3)

# ============ BOTTOM: phi-lattice interior ============
y_bot = 0.6
ax.text(0.4, y_bot + 0.95, r"Interior $\phi$-geometry (hidden mold)",
        fontsize=11.5, fontweight="bold", color=INK, bbox=label_box,
        zorder=6)
ax.text(0.4, y_bot + 0.68, "discrete phi-lattice that generates the surface",
        fontsize=9, color=INK_SOFT, style="italic", bbox=label_box,
        zorder=6)
# crystalline grid (starts past the labels)
gx = np.arange(2.4, 9.5, 0.42)
gy = np.array([y_bot - 0.05, y_bot + 0.18, y_bot + 0.35])
for x in gx:
    ax.plot([x, x], [gy[0], gy[-1]], color=GRID, lw=0.6, zorder=1)
for yy in gy:
    ax.plot([gx[0], gx[-1]], [yy, yy], color=GRID, lw=0.6, zorder=1)
GX, GY = np.meshgrid(gx, gy)
ax.scatter(GX.flatten(), GY.flatten(), s=14, color=RED,
           edgecolors=INK, linewidths=0.3, zorder=3)
# the mold underneath the skin (matches mid curve, exposed)
ax.plot(xs_curve, y_bot + 0.18 + mold_curve * 0.5, color=RED, lw=1.6,
        ls="--", alpha=0.8, zorder=4)

# ============ vacuum-suction arrows: data -> skin, skin -> mold ============
for x in np.linspace(3.4, 8.6, 5):
    # data being pulled down onto skin
    a1 = FancyArrowPatch(
        (x, y_top + 0.35 * np.sin(1.6 * x) + 0.15 * np.sin(3.1 * x + 0.4) - 0.45),
        (x, y_mid + 0.35 * np.sin(1.6 * x) + 0.15 * np.sin(3.1 * x + 0.4) + 0.18),
        arrowstyle="-|>", mutation_scale=10,
        color=MUTED, lw=1.0, alpha=0.75, zorder=2)
    ax.add_patch(a1)
    # skin standing on top of mold
    a2 = FancyArrowPatch(
        (x, y_mid + 0.35 * np.sin(1.6 * x) + 0.15 * np.sin(3.1 * x + 0.4) - 0.32),
        (x, y_bot + 0.18 + (0.35 * np.sin(1.6 * x) + 0.15 * np.sin(3.1 * x + 0.4)) * 0.5 + 0.32),
        arrowstyle="-", mutation_scale=8,
        color=MUTED, lw=0.9, ls=":", alpha=0.7, zorder=2)
    ax.add_patch(a2)

# vacuum-form caption (centred below)
ax.text(5.0, -0.05,
        "vacuum-form operation  =  training",
        fontsize=10, color=INK_SOFT, style="italic",
        ha="center", va="top")

ax.set_title("The Vacuum Forming Hypothesis", fontsize=14,
             fontweight="bold", color=INK, pad=12, loc="center")

save_fig("fig1_1_vacuum_forming")
