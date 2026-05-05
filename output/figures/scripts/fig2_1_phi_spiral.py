#!/usr/bin/env python3
"""
Figure 2.1 - Three views of phi self-similarity.

Panel A: Geometric proof phi = 1 + 1/phi using a golden segment.
Panel B: The Fibonacci spiral converging to phi.
Panel C: phi^n exponential ladder (clean log-scale bars).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_SOFT, GOLD_DARK,
                      TEAL, RED, GRID, PHI, MUTED)

apply_style()

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.6),
                                    gridspec_kw={"width_ratios": [1, 1.05, 1.1]})

# =========================================================================
# Panel A : geometric definition phi = 1 + 1/phi
# =========================================================================
axA.set_xlim(-0.15, PHI + 0.25); axA.set_ylim(-0.6, 1.55)
axA.set_aspect("equal"); axA.axis("off")
panel_label(axA, "A", x=-0.05)

# the long segment (length phi)
axA.add_patch(Rectangle((0, 0), 1, 0.75, facecolor=GOLD_SOFT,
                        edgecolor=INK, lw=1.1))
axA.add_patch(Rectangle((1, 0), 1 / PHI, 0.75, facecolor=GOLD,
                        edgecolor=INK, lw=1.1))

# segment labels
axA.text(0.5, 0.375, "1", fontsize=15, fontweight="bold",
         ha="center", va="center", color=INK)
axA.text(1 + 1 / (2 * PHI), 0.375, r"$1/\phi$", fontsize=13,
         fontweight="bold", ha="center", va="center", color="white")

# ratio bracket below
axA.annotate("", xy=(0, -0.05), xytext=(PHI, -0.05),
             arrowprops=dict(arrowstyle="-", color=INK, lw=1.2))
axA.plot([0, 0], [-0.05, -0.12], color=INK, lw=1.2)
axA.plot([PHI, PHI], [-0.05, -0.12], color=INK, lw=1.2)
axA.plot([1, 1], [-0.05, -0.12], color=INK, lw=1.2)
axA.text(PHI / 2, -0.25, r"$\phi \;\approx\; 1.618$",
         ha="center", fontsize=12, color=INK, fontweight="bold")

# the equation
axA.text(PHI / 2, 1.25, r"$\phi = 1 + \dfrac{1}{\phi}$",
         ha="center", va="center", fontsize=17, color=INK)
axA.text(PHI / 2, 0.95,
         "the whole : larger  =  larger : smaller",
         ha="center", fontsize=9.5, color=INK_SOFT, style="italic")

axA.set_title("Self-similar definition", fontsize=12, color=INK)

# =========================================================================
# Panel B : Fibonacci spiral - canonical CCW outward tiling
# =========================================================================
axB.set_aspect("equal"); axB.axis("off")
panel_label(axB, "B", x=-0.02)

shades = [GOLD_SOFT, "#F6E8C7", "#EFD9A9", "#E5C887", GOLD]

# (x0, y0, side, arc_corner, arc_theta1, arc_theta2)
# arc_corner = (cx, cy); arcs traverse 90 deg.
squares = [
    # F1=1: (0,0)-(1,1).  arc: center (0,0) from (1,0) to (0,1)  -> 0..90
    (0, 0, 1, (0, 0), 0, 90),
    # F2=1: (-1,0)-(0,1). arc: center (0,0) from (0,1) to (-1,0) -> 90..180
    (-1, 0, 1, (0, 0), 90, 180),
    # F3=2: (-1,-2)-(1,0). arc: center (1,0) from (-1,0) to (1,-2) -> 180..270
    (-1, -2, 2, (1, 0), 180, 270),
    # F4=3: (1,-2)-(4,1). arc: center (1,1) from (1,-2) to (4,1) -> 270..360
    (1, -2, 3, (1, 1), 270, 360),
    # F5=5: (-1,1)-(4,6). arc: center (-1,1) from (4,1) to (-1,6) -> 0..90
    (-1, 1, 5, (-1, 1), 0, 90),
]

side_of = {0: 1, 1: 1, 2: 2, 3: 3, 4: 5}
for i, ((x0, y0, s, (cx, cy), t1, t2), shade) in enumerate(zip(squares, shades)):
    axB.add_patch(Rectangle((x0, y0), s, s, facecolor=shade,
                            edgecolor=INK, lw=0.9, alpha=0.85))
    axB.text(x0 + s / 2, y0 + s / 2, str(s),
             ha="center", va="center", fontsize=11, color=INK_SOFT)
    axB.add_patch(Arc((cx, cy), 2 * s, 2 * s, angle=0,
                      theta1=t1, theta2=t2, color=INK, lw=2.0))

axB.set_xlim(-1.6, 4.6); axB.set_ylim(-2.6, 6.4)
axB.text(1.5, 6.0, r"$\dfrac{F_{n+1}}{F_n}\to\phi$",
         fontsize=14, color=INK, ha="center")
axB.set_title("Fibonacci spiral", fontsize=12, color=INK)

# =========================================================================
# Panel C : phi^n ladder
# =========================================================================
panel_label(axC, "C", x=-0.05)
ns = np.arange(-4, 5)
vals = PHI ** ns

bar_colors = [TEAL if n < 0 else GOLD for n in ns]
axC.bar(ns, vals, color=bar_colors, edgecolor=INK, lw=0.6, width=0.7)
axC.set_yscale("log")
axC.set_xticks(ns)
axC.set_xlabel(r"$n$", fontsize=11)
axC.set_ylabel(r"$\phi^{\,n}$ (log scale)", fontsize=11)
axC.set_title(r"$\phi^{\,n}$ scaling", fontsize=12, color=INK)
axC.grid(True, axis="y", which="both", color=GRID, linewidth=0.5, alpha=0.7)
axC.set_axisbelow(True)
axC.spines["left"].set_color(INK)
axC.spines["bottom"].set_color(INK)
axC.set_ylim(0.08, 12)

# value annotations - alternate above bars, comfortably spaced
for n, v in zip(ns, vals):
    axC.text(n, v * 1.15, f"{v:.3f}", ha="center", va="bottom",
             fontsize=8, color=INK)

axC.axhline(1.0, color=INK_SOFT, lw=0.6, ls="--", alpha=0.6)
axC.text(ns[-1] + 0.1, 1.05, r"$\phi^0=1$", color=INK_SOFT,
         fontsize=8.5, ha="right", va="bottom")

fig.suptitle(r"Self-similarity of the golden ratio  $\phi$",
             fontsize=15, fontweight="bold", color=INK, y=1.03)

save_fig("fig2_1_phi_spiral")
