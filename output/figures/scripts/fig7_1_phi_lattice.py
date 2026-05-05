#!/usr/bin/env python3
"""
Figure 7.1 - The phi-lattice and tetromino weight clustering.

Panel A: 2D phi-lattice grid x = phi^m, y = phi^n with intersection points.
         A few connected groups of points are highlighted as 'tetrominoes'
         (74 unique structures cover the entire model).
Panel B: Histogram of weights by phi-level showing the discrete clusters
         and the 74-tetromino cap.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, RED, TEAL, VIOLET,
                      GRID, PHI, MUTED)

apply_style()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4),
                               gridspec_kw={"width_ratios": [1, 1.05]})

# =========================================================================
# Panel A : phi-lattice with tetrominoes
# =========================================================================
panel_label(axA, "A")
levels = np.arange(0, 5)
xs = PHI ** levels
ys = PHI ** levels

# grid lines
for x in xs:
    axA.axvline(x, color=GOLD, ls="--", lw=0.7, alpha=0.55, zorder=1)
for y in ys:
    axA.axhline(y, color=GOLD, ls="--", lw=0.7, alpha=0.55, zorder=1)

# all intersection points
GX, GY = np.meshgrid(xs, ys)
axA.scatter(GX.flatten(), GY.flatten(), s=42, color=INK, zorder=3)

# axis-tick labels naming phi^k powers
axA.set_xticks(xs)
axA.set_xticklabels([rf"$\phi^{{{k}}}$" for k in levels], fontsize=10)
axA.set_yticks(ys)
axA.set_yticklabels([rf"$\phi^{{{k}}}$" for k in levels], fontsize=10)

# overlay a few tetrominoes (groups of 4 adjacent intersections)
def draw_tetromino(cells, color):
    cells = np.array(cells, dtype=float)
    # cells are (col, row) in lattice coords -> use (xs[col], ys[row])
    for col, row in cells:
        col = int(col); row = int(row)
        x0 = xs[col] - 0.18
        y0 = ys[row] - 0.18
        axA.add_patch(Rectangle((x0, y0), 0.36, 0.36,
                                facecolor=color, edgecolor=INK,
                                lw=1.0, alpha=0.55, zorder=2))


draw_tetromino([(0, 0), (1, 0), (0, 1), (1, 1)], TEAL)   # square
draw_tetromino([(2, 0), (3, 0), (3, 1), (3, 2)], RED)    # L
draw_tetromino([(0, 3), (1, 3), (2, 3), (2, 4)], VIOLET) # J

axA.text(PHI ** 4 * 0.98, PHI ** 0 * 0.95,
         "tetromino\n(4 adjacent\nintersections)",
         fontsize=8.5, color=INK_SOFT, style="italic",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID))

axA.set_xlim(0.7, PHI ** 4 * 1.15); axA.set_ylim(0.7, PHI ** 4 * 1.15)
axA.set_xlabel(r"phi-level along axis 1")
axA.set_ylabel(r"phi-level along axis 2")
axA.set_title("phi-lattice: discrete coordinate system",
              fontsize=12)
axA.spines["left"].set_color(INK); axA.spines["bottom"].set_color(INK)
axA.set_axisbelow(True)

# =========================================================================
# Panel B : weight histogram by phi-level
# =========================================================================
panel_label(axB, "B")
rng = np.random.default_rng(0)
# synthetic distribution: discrete clusters at integer levels with small spread
levels_b = np.arange(-7, 8)
true_counts = np.array([60, 230, 800, 2000, 3500, 1000, 4000,
                         3500, 2000, 800, 230, 60, 18, 6, 2]) * 1.0
# add a low residual baseline showing 'noise'
baseline = np.full_like(true_counts, 40)

axB.bar(levels_b, true_counts, color=GOLD, edgecolor=INK,
        lw=0.6, width=0.78, label="weights at phi-level", zorder=3)
axB.bar(levels_b, baseline, color=MUTED, edgecolor="none",
        width=0.78, alpha=0.5, zorder=2,
        label="residual / noise floor")

# 74-tetromino vocabulary cap
axB.axhline(74, color=RED, ls="--", lw=1.5,
            label="74 unique tetrominoes (vocabulary cap)", zorder=4)
axB.text(levels_b.max(), 88, "74", color=RED, fontweight="bold",
         fontsize=10, ha="right")

axB.set_xlabel(r"phi-level  $e$  (weight $= s\cdot\phi^e$)")
axB.set_ylabel("weight count")
axB.set_title("Weights cluster at discrete phi-levels",
              fontsize=12)
axB.legend(loc="upper right", fontsize=8.5)
axB.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.6)
axB.set_axisbelow(True)
axB.set_xticks(levels_b)
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

fig.suptitle("phi-Lattice and the 74-Tetromino Vocabulary",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig7_1_phi_lattice")
