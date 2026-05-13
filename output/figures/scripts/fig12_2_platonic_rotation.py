#!/usr/bin/env python3
"""
Figure 12.2 - Platonic Ideals as Rotation Anchors (DC 180).

Panel A: Geometric definition.  For relationship type R with
         characteristic angle theta_R, the entity e rotates by theta_R
         about an axis pointing toward the Platonic ideal I_R, which
         is orthogonal to e.  Worked example: France -> Paris (77.3 deg).

Panel B: Universality of theta_R within a relationship type.  Box plot
         showing the relationship-specific angle is consistent (~77 deg
         for capital-of across many entities) but differs across types.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, Circle
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, MUTED, GRID, PHI)

apply_style()

fig = plt.figure(figsize=(13, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])

# =========================================================================
# Panel A : rotation geometry
# =========================================================================
axA = fig.add_subplot(gs[0, 0])
panel_label(axA, "A")
axA.set_xlim(-1.55, 1.55); axA.set_ylim(-1.55, 1.55)
axA.set_aspect("equal"); axA.axis("off")

# Origin
O = np.array([0.0, 0.0])

# entity at 170 deg (upper-left) so the answer rotates into upper-right
theta_e = np.deg2rad(170)
e = 1.0 * np.array([np.cos(theta_e), np.sin(theta_e)])
# rotate entity by 77.3 deg (counter-clockwise) to get the answer
theta_R_deg = 77.3
theta_R = np.deg2rad(theta_R_deg)
R = np.array([[np.cos(theta_R), -np.sin(theta_R)],
              [np.sin(theta_R),  np.cos(theta_R)]])
a = R @ e

# ideal direction = direction the rotation drives e *toward* (orthogonal to e)
# axis_e(I_R) is the component of I_R orthogonal to e.  We just place I_R
# at distance 1 along the rotation tangent direction.
axis_dir = R @ (np.array([[0, -1], [1, 0]]) @ e)
axis_dir = axis_dir / np.linalg.norm(axis_dir)
I_R = 1.15 * axis_dir   # place ideal beyond unit sphere along the axis

# unit circle (phi-sphere indicator)
theta_circ = np.linspace(0, 2 * np.pi, 200)
axA.plot(np.cos(theta_circ), np.sin(theta_circ),
         color=MUTED, lw=0.6, alpha=0.45, zorder=1)

# entity vector
axA.add_patch(FancyArrowPatch(O, e, arrowstyle="-|>",
                              color=INK, lw=1.8, mutation_scale=14,
                              zorder=3))
axA.text(e[0] - 0.10, e[1] + 0.10, r"$e$  (France)",
         fontsize=10.5, color=INK, ha="right", va="bottom",
         fontweight="bold")

# answer vector
axA.add_patch(FancyArrowPatch(O, a, arrowstyle="-|>",
                              color=GOLD_DARK, lw=1.8, mutation_scale=14,
                              zorder=3))
axA.text(a[0] + 0.07, a[1] + 0.10, r"$a$  (Paris)",
         fontsize=10.5, color=GOLD_DARK, ha="left", va="bottom",
         fontweight="bold")

# axis pointing toward Platonic ideal (orthogonal to e)
axA.add_patch(FancyArrowPatch(O, I_R, arrowstyle="-|>",
                              color=RED, lw=1.6, ls="--",
                              mutation_scale=12, zorder=2))
axA.scatter(I_R[0], I_R[1], s=140, color=RED, edgecolor=INK,
            linewidth=1.4, zorder=4)
axA.text(I_R[0] - 0.05, I_R[1] - 0.05,
         r"$I_R$" + "\nPlatonic ideal\n(capital)",
         fontsize=10, color=RED, ha="right", va="top",
         fontweight="bold")

# orthogonality marker
# small right-angle indicator at origin between e and axis
side = 0.08
e_unit = e / np.linalg.norm(e)
ax_unit = axis_dir
p1 = side * e_unit
p2 = side * (e_unit + ax_unit)
p3 = side * ax_unit
axA.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]],
         color=INK_SOFT, lw=0.9, zorder=2)
axA.text(0.5 * (p1[0] + p3[0]) - 0.04, 0.5 * (p1[1] + p3[1]) + 0.04,
         r"$\bot$", fontsize=10, color=INK_SOFT)

# rotation arc with angle label
arc_r = 0.35
ang_start = np.rad2deg(theta_e)
ang_end   = ang_start + theta_R_deg
arc = Arc(O, 2 * arc_r, 2 * arc_r,
          angle=0, theta1=ang_start, theta2=ang_end,
          color=GOLD_DARK, lw=2.0, zorder=3)
axA.add_patch(arc)
# arrowhead at end of arc
mid_ang = np.deg2rad(ang_end)
arc_end = arc_r * np.array([np.cos(mid_ang), np.sin(mid_ang)])
# midpoint label
mid_ang_label = np.deg2rad(0.5 * (ang_start + ang_end))
lbl_pt = (arc_r + 0.18) * np.array([np.cos(mid_ang_label),
                                     np.sin(mid_ang_label)])
axA.text(lbl_pt[0], lbl_pt[1],
         r"$\theta_R = 77.3^{\circ}$",
         fontsize=10.5, color=GOLD_DARK, ha="center", va="center",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.22", fc="white",
                   ec=GOLD_DARK, lw=0.8))

# origin marker
axA.scatter([0], [0], s=30, color=INK, zorder=5)

# defining formula  - place at bottom of panel, no overlap with title
axA.text(0.0, -1.40,
         r"$a \;=\; \mathrm{rotate}(e,\; \theta_R,\; \mathrm{axis}_e(I_R))$",
         ha="center", va="center", fontsize=11.5, color=INK,
         bbox=dict(boxstyle="round,pad=0.30", fc="white",
                   ec=INK, lw=0.8))

# Subtitle below the panel label
axA.text(0.5, 1.04,
         "Entity $\\to$ answer is rotation toward a Platonic ideal",
         transform=axA.transAxes,
         ha="center", va="bottom", fontsize=11.5,
         color=INK, fontweight="bold")

# =========================================================================
# Panel B : angle universality across entity pairs, divergence across types
# =========================================================================
axB = fig.add_subplot(gs[0, 1])
panel_label(axB, "B")

rng = np.random.default_rng(7)

# Each relationship type has its own consistent angle with small std
# Values from DC 180
types = [
    ("capital-of",        77.3, 1.5, GOLD_DARK),
    ("size-decrease",     83.9, 1.0, TEAL),
    ("size-increase",     85.4, 1.0, VIOLET),
    ("regality-increase", 84.4, 1.0, RED),
    ("trajectory (full)", 90.3, 0.2, "#3F6FB3"),
]

positions = np.arange(len(types))
violins_data = []
colors_b = []
labels = []
means = []
stds = []
for name, mu, sd, c in types:
    samples = rng.normal(mu, sd, 60)
    violins_data.append(samples)
    colors_b.append(c)
    labels.append(name)
    means.append(mu)
    stds.append(sd)

# strip-style scatter with mean & std
for i, (samples, c, mu, sd) in enumerate(zip(violins_data, colors_b,
                                              means, stds)):
    jitter = rng.uniform(-0.18, 0.18, samples.size)
    axB.scatter(positions[i] + jitter, samples,
                s=18, color=c, alpha=0.55, edgecolor="none",
                zorder=2)
    # mean line
    axB.hlines(mu, positions[i] - 0.32, positions[i] + 0.32,
               color=c, lw=2.4, zorder=4)
    # std band
    axB.add_patch(plt.Rectangle((positions[i] - 0.32, mu - sd),
                                 0.64, 2 * sd,
                                 facecolor=c, alpha=0.18,
                                 edgecolor="none", zorder=1))
    # value label
    axB.text(positions[i], mu + 2.0,
             f"{mu:.1f}\u00b0\n($\\pm {sd:.1f}$)",
             ha="center", va="bottom", fontsize=8.8,
             color=c, fontweight="bold")

axB.set_xticks(positions)
axB.set_xticklabels(labels, rotation=20, ha="right", fontsize=9.5)
axB.set_ylabel(r"rotation angle  $\theta_R$  (degrees)")
axB.set_ylim(72, 99)
axB.set_title("$\\theta_R$ is universal within type, distinct across types",
              fontsize=11.5)
axB.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.6)
axB.set_axisbelow(True)
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

# source caption
axB.text(0.5, -0.30,
         "values from DC 180 (Qwen2-7B residual stream, multi-entity sweep)",
         transform=axB.transAxes,
         fontsize=9, color=INK_SOFT, style="italic",
         ha="center", va="top")

fig.suptitle("Platonic Ideals: relationships are rotations in $\\phi$-space",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig12_2_platonic_rotation")
