#!/usr/bin/env python3
"""
Figure 5.1 - The ENCODE = DECODE master symmetry.

Panel A: A token enters, is multiplied by phi (encode), then by 1/phi
         (decode), and emerges identical.  The two arcs are mirror images.
Panel B: The critical line sigma = 0.5 with simulated zeta-like zeros along
         it - the universal balance between under- and over-determined.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, RED, TEAL, VIOLET,
                      GRID, PHI, MUTED)

apply_style()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                               gridspec_kw={"width_ratios": [1.2, 1]})

# =========================================================================
# Panel A : encode/decode symmetry diagram
# =========================================================================
panel_label(axA, "A")
axA.set_xlim(-0.1, 5.1); axA.set_ylim(-0.4, 3.3)
axA.set_aspect("equal"); axA.axis("off")

# input and output dots on a horizontal axis
yline = 1.2
axA.plot([-0.05, 5.05], [yline, yline], color=INK, lw=1.0, alpha=0.6)

axA.add_patch(Circle((0.4, yline), 0.18, facecolor=GOLD,
                     edgecolor=INK, lw=1.2, zorder=3))
axA.text(0.4, yline - 0.45, "input  $x$", ha="center", va="top",
         fontsize=11, color=INK)

axA.add_patch(Circle((4.6, yline), 0.18, facecolor=GOLD,
                     edgecolor=INK, lw=1.2, zorder=3))
axA.text(4.6, yline - 0.45, "output  $x$", ha="center", va="top",
         fontsize=11, color=INK)

# central phi node
axA.add_patch(Circle((2.5, yline), 0.32, facecolor="white",
                     edgecolor=INK, lw=1.4, zorder=3))
axA.text(2.5, yline, r"$\phi$", ha="center", va="center",
         fontsize=20, color=INK, fontweight="bold")

# encode arc (above)
arc1 = FancyArrowPatch((0.6, yline + 0.05), (2.2, yline + 0.05),
                       connectionstyle="arc3,rad=-0.45",
                       arrowstyle="-|>", color=RED, lw=2.4,
                       mutation_scale=14, zorder=4)
axA.add_patch(arc1)
axA.text(1.4, yline + 0.95, "ENCODE", color=RED,
         fontsize=12, fontweight="bold", ha="center")
axA.text(1.4, yline + 0.7, r"$x \;\mapsto\; x \cdot \phi$",
         color=RED, fontsize=11, ha="center")

# decode arc (above, mirror)
arc2 = FancyArrowPatch((2.8, yline + 0.05), (4.4, yline + 0.05),
                       connectionstyle="arc3,rad=-0.45",
                       arrowstyle="-|>", color=TEAL, lw=2.4,
                       mutation_scale=14, zorder=4)
axA.add_patch(arc2)
axA.text(3.6, yline + 0.95, "DECODE", color=TEAL,
         fontsize=12, fontweight="bold", ha="center")
axA.text(3.6, yline + 0.7, r"$y \;\mapsto\; y \,/\, \phi$",
         color=TEAL, fontsize=11, ha="center")

# bottom equation
axA.text(2.5, -0.05, r"$\phi \,\times\, \dfrac{1}{\phi} \;=\; 1$",
         ha="center", va="center", fontsize=15, color=INK)
axA.text(2.5, 2.65, "the transformation IS the inverse",
         ha="center", fontsize=11, color=INK_SOFT, style="italic")

axA.set_title("Encoding and decoding share one operation", fontsize=12)

# =========================================================================
# Panel B : critical line sigma = 0.5
# =========================================================================
panel_label(axB, "B")
axB.set_xlim(0, 1); axB.set_ylim(-12, 12)

# heat-map background: distance to sigma=0.5
sigma = np.linspace(0, 1, 200)
t = np.linspace(-12, 12, 200)
S, T = np.meshgrid(sigma, t)
heat = np.exp(-((S - 0.5) ** 2) * 18) * (0.5 + 0.5 * np.cos(T * 0.7))
axB.imshow(heat, extent=[0, 1, -12, 12], origin="lower",
           aspect="auto", cmap="YlGnBu", alpha=0.45, zorder=0)

# critical line
axB.axvline(0.5, color=RED, lw=2.4, zorder=4,
            label=r"critical line  $\sigma = 1/2$")

# simulated zeta-like zero positions on the critical line
zero_t = np.array([14.13, 21.02, 25.01, 30.42, 32.94, 37.59, 40.92,
                   43.33, 48.00, 49.77]) - 30
# fold positive/negative
zero_t = np.concatenate([zero_t, -zero_t])
axB.scatter([0.5] * len(zero_t), zero_t, color=RED, s=42,
            edgecolors=INK, linewidths=0.6, zorder=5,
            label="zeros on the critical line")

# region labels
axB.text(0.18, 11, "under-determined\n(too few constraints)",
         color=INK, fontsize=9, ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                   ec=GRID, alpha=0.9))
axB.text(0.82, 11, "over-determined\n(too many constraints)",
         color=INK, fontsize=9, ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                   ec=GRID, alpha=0.9))

axB.set_xlabel(r"$\sigma$  (information-balance axis)")
axB.set_ylabel(r"$t$")
axB.set_title(r"Universal information limit at  $\sigma = 1/2$",
              fontsize=12)
axB.legend(loc="lower right", fontsize=9,
           facecolor="white", framealpha=0.9)
axB.set_xticks([0, 0.25, 0.5, 0.75, 1])
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

fig.suptitle("ENCODE = DECODE: the master symmetry",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig5_1_encode_decode")
