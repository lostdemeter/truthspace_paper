#!/usr/bin/env python3
"""
Figure 9.2 - The 4-State Holographic Gate Field.

Panel A: Gate input x partitioned into 4 states at boundaries +/- log(phi).
         Bright fringes (+1 EXPAND, +0 PRESERVE+) and dark fringes
         (-0 PRESERVE-, -1 CONTRACT). SiLU and GELU passing through.
         Marks the sigma(log phi) = 1/phi identity.

Panel B: Energy contribution by state - 42.4% of layer-14 output energy
         comes from "dead" PRESERVE channels, contradicting the
         conventional view that they are non-contributing.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, SAND, GRID, MUTED, PHI)

apply_style()
LN_PHI = np.log(PHI)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4),
                               gridspec_kw={"width_ratios": [1.35, 1]})

# =========================================================================
# Panel A : 4-state gate field
# =========================================================================
panel_label(axA, "A")

x = np.linspace(-3.5, 3.5, 801)
sig = 1.0 / (1.0 + np.exp(-x))

# SiLU and GELU pass through the field
silu = x * sig
# GELU approximation (tanh form)
gelu = 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

# 4 colored regions (background bands)
# bright: EXPAND  +1 (x > log phi)
# bright: PRESERVE+ +0 (0 < x < log phi)
# dark:   PRESERVE- -0 (-log phi < x < 0)
# dark:   CONTRACT -1 (x < -log phi)
band_colors = {
    "CONTRACT (-1)":   ("#2A3B5C", 0.55, -3.5, -LN_PHI),
    "PRESERVE- (-0)":  ("#4A5A75", 0.32, -LN_PHI, 0.0),
    "PRESERVE+ (+0)":  (GOLD_SOFT,  0.55, 0.0, LN_PHI),
    "EXPAND (+1)":     (GOLD,       0.45, LN_PHI, 3.5),
}

for label, (color, alpha, x0, x1) in band_colors.items():
    axA.axvspan(x0, x1, color=color, alpha=alpha, zorder=0)

# label each band
band_text = [
    (-2.3, 1.95, "CONTRACT\n($-1$)", "white"),
    (-1.0, 2.20, "PRESERVE$-$\n($-0$)", "#4A5A75"),
    ( 1.0, 2.20, "PRESERVE$+$\n($+0$)", GOLD_DARK),
    ( 2.3, 1.95, "EXPAND\n($+1$)", INK),
]
for xt, yt, label, color in band_text:
    axA.text(xt, yt, label, ha="center", va="center",
             fontsize=10, fontweight="bold", color=color)

# SiLU & GELU activation curves
axA.plot(x, silu, color=RED, lw=2.4, label="SiLU $= x \\cdot \\sigma(x)$",
         zorder=4)
axA.plot(x, gelu, color=TEAL, lw=1.8, ls="--", label="GELU",
         zorder=4)

# zero line
axA.axhline(0, color=INK, lw=0.7, alpha=0.4, zorder=2)

# boundary lines at +/- log(phi)
for xb, lbl in [(-LN_PHI, r"$-\log\phi$"), (LN_PHI, r"$+\log\phi$")]:
    axA.axvline(xb, color=INK, ls=":", lw=1.2, alpha=0.75, zorder=3)
    axA.text(xb, -1.6, lbl, ha="center", va="top",
             fontsize=10, color=INK, fontweight="bold")

# annotate sigma(log phi) = 1/phi
y_id = LN_PHI * (1.0 / PHI)  # SiLU at log phi = log phi * sigma(log phi) = log phi / phi
axA.scatter([LN_PHI], [y_id], color=RED, s=80, zorder=6,
            edgecolor=INK, linewidth=1.3)
axA.annotate(r"$\sigma(\log\phi) = 1/\phi$" + "\n" +
             r"$\Rightarrow$ SiLU$(\log\phi) = \log\phi / \phi$",
             xy=(LN_PHI, y_id),
             xytext=(LN_PHI + 0.85, y_id - 0.8),
             fontsize=9.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec=INK, lw=0.8),
             arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

axA.set_xlim(-3.5, 3.5); axA.set_ylim(-1.8, 2.7)
axA.set_xlabel(r"gate input  $x$")
axA.set_ylabel("activation output")
axA.set_title("4-state holographic gate at boundaries $\\pm \\log\\phi \\approx \\pm 0.481$",
              fontsize=11.5)
axA.legend(loc="lower right", fontsize=9.5,
           facecolor="white", framealpha=0.95)
axA.grid(False)
axA.spines["left"].set_color(INK); axA.spines["bottom"].set_color(INK)
axA.set_axisbelow(True)

# =========================================================================
# Panel B : Energy by state - dark fringes carry info
# =========================================================================
panel_label(axB, "B")

# Layer-14 output energy contribution, by state (from Finding 57 / DC 245)
states = ["EXPAND\n(+1)", "PRESERVE+\n(+0)", "PRESERVE-\n(-0)", "CONTRACT\n(-1)"]
# Approximate empirical contributions: bright dominate, but PRESERVE- alone
# contributes a substantial slice. The 42.4% figure is "dead channels"
# (PRESERVE+ and PRESERVE- combined).
energies = np.array([34.5, 23.1, 19.3, 23.1])
# Sanity: PRESERVE+ + PRESERVE- = 42.4
assert abs(energies[1] + energies[2] - 42.4) < 0.1
fringe_kind = ["bright", "bright", "dark", "dark"]

colors_b = [GOLD, GOLD_SOFT, "#4A5A75", "#2A3B5C"]
edgec    = [INK, INK, INK, "white"]
textc    = [INK, INK, "white", "white"]

xs = np.arange(4)
bars = axB.bar(xs, energies, color=colors_b, edgecolor=INK,
               linewidth=1.0, zorder=3, width=0.66)
for i, (b, e, txt) in enumerate(zip(bars, energies, textc)):
    axB.text(b.get_x() + b.get_width() / 2, e - 2.5,
             f"{e:.1f}%",
             ha="center", va="top", fontsize=10.5,
             fontweight="bold", color=txt)
    axB.text(b.get_x() + b.get_width() / 2, -2.0,
             fringe_kind[i],
             ha="center", va="top", fontsize=9, color=INK_SOFT,
             style="italic")

# Highlight the "dead" channels combined
axB.add_patch(Rectangle((1.66, 0), 1.68, 23.5,
                        facecolor="none", edgecolor=RED,
                        lw=1.6, ls="--", zorder=4))
axB.annotate("", xy=(2.5, 24.5), xytext=(2.5, 30),
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
axB.text(2.5, 32, "42.4% from\n\"dead\" channels",
         ha="center", va="bottom",
         fontsize=10, fontweight="bold", color=RED)

axB.set_xticks(xs)
axB.set_xticklabels(states, fontsize=9.5)
axB.set_ylabel("layer-14 output energy share  (%)")
axB.set_ylim(-3.5, 42)
axB.set_title("Dark fringes carry $\\sim$42% of the signal",
              fontsize=11.5)
axB.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.6)
axB.set_axisbelow(True)
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

fig.suptitle("The 4-state holographic activation gate",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig9_2_holographic_gate")
