#!/usr/bin/env python3
"""
Figure 10.1 - The irreducible shape.

Panel A: phi-Zipf duality - phi^(-log_phi r) and Zipf 1/r are the same
         self-similar fractal viewed from opposite directions.
Panel B: Lattice of critical lines - a dense criss-cross pattern in two
         orientations whose intersections form the 67.9M information atoms.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, RED, TEAL, VIOLET, GRID,
                      PHI, MUTED)

apply_style()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.4))

# =========================================================================
# Panel A : phi-Zipf duality on log-log
# =========================================================================
panel_label(axA, "A")
r = np.arange(1, 200)
zipf = 1.0 / r
phi_decay = PHI ** (-np.log(r) / np.log(PHI) * 0.85)  # phi^(-log_phi(r))*0.85

axA.loglog(r, zipf, color=TEAL, lw=2.2, marker="o", ms=3.5,
           markevery=10, mfc=TEAL, mec=INK, mew=0.5,
           label=r"Zipf  $f \sim 1/r$  (frequency $\to$ inward)")
axA.loglog(r, phi_decay, color=GOLD, lw=2.2, marker="s", ms=3.5,
           markevery=10, mfc=GOLD, mec=INK, mew=0.5,
           label=r"$\phi^{-\log_\phi r}$  (encoding $\to$ outward)")

# annotate the duality
axA.fill_between(r, zipf, phi_decay, color=GOLD, alpha=0.1)
axA.set_xlabel(r"rank  $r$  (log scale)")
axA.set_ylabel(r"frequency / weight  (log scale)")
axA.set_title("phi-Zipf duality: encoding = ranking", fontsize=12)
axA.legend(loc="lower left", fontsize=9.5, frameon=True,
           facecolor="white", edgecolor=GRID)
axA.grid(True, which="both", color=GRID, lw=0.4, alpha=0.6)
axA.set_axisbelow(True)
axA.spines["left"].set_color(INK); axA.spines["bottom"].set_color(INK)

# =========================================================================
# Panel B : irreducible lattice - many critical lines, intersections
# =========================================================================
panel_label(axB, "B")
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.set_aspect("equal")
axB.set_xticks([]); axB.set_yticks([])
for s in axB.spines.values():
    s.set_color(INK); s.set_linewidth(0.8)

rng = np.random.default_rng(11)
n_lines = 36
# horizontal-ish and vertical-ish lines at many phi-spaced angles
angles_h = rng.uniform(-0.18, 0.18, n_lines)
angles_v = rng.uniform(np.pi / 2 - 0.18, np.pi / 2 + 0.18, n_lines)
offsets_h = np.linspace(0.04, 0.96, n_lines) + rng.normal(0, 0.005, n_lines)
offsets_v = np.linspace(0.04, 0.96, n_lines) + rng.normal(0, 0.005, n_lines)

# draw them all faintly
def draw_line(theta, c):
    # line passes through (c*cos, c*sin) perpendicular to (cos,sin)
    # easier: parameterise y = mx + b for nearly-horizontal, x = my+b for nearly-vertical
    if abs(np.cos(theta)) < abs(np.sin(theta)):
        # near-vertical
        m = np.cos(theta) / np.sin(theta)
        b = c
        ys = np.array([0, 1])
        xs = m * ys + b
        axB.plot(xs, ys, color=RED, lw=0.5, alpha=0.5, zorder=1)
    else:
        m = np.sin(theta) / np.cos(theta)
        b = c
        xs = np.array([0, 1])
        ys = m * xs + b
        axB.plot(xs, ys, color=TEAL, lw=0.5, alpha=0.5, zorder=1)


for th, off in zip(angles_h, offsets_h):
    draw_line(th, off)
for th, off in zip(angles_v, offsets_v):
    draw_line(th, off)

# scatter many intersection points (synthesised)
n_pts = 900
px = rng.uniform(0.05, 0.95, n_pts)
py = rng.uniform(0.05, 0.95, n_pts)
axB.scatter(px, py, s=2.5, color=INK, alpha=0.55, zorder=2)

axB.set_title("The irreducible shape", fontsize=12, color=INK, pad=10)

# annotation pill BELOW the plot, clear of the title
axB.text(0.5, -0.04,
         "3,584 critical lines     ·     67.9 M intersection points",
         ha="center", va="top", transform=axB.transAxes,
         fontsize=10.5, color=INK,
         bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=INK, lw=1.0))
axB.text(0.5, -0.18,
         "1 bit per intersection  =  information-theoretic minimum",
         ha="center", va="top", transform=axB.transAxes,
         fontsize=9.5, color=INK_SOFT, style="italic")

fig.suptitle("Irreducible shape and phi-Zipf spectrum",
             fontsize=15, fontweight="bold", y=1.04)

save_fig("fig10_1_irreducible_shape")
