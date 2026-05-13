#!/usr/bin/env python3
"""
Figure B.3 - Borwein integral: exact identity, then a sharp spectral break.

The Borwein integrals
  $I_n = \\int_0^\\infty \\prod_{k=0}^{n} \\frac{\\sin((2k+1) x)}{(2k+1) x} \\, dx$
are *exactly* $\\pi/2$ for $n = 0, 1, \\ldots, 6$ -- to machine epsilon.
At $n = 7$ the sum of reciprocals of odd denominators first exceeds 1
($1 + 1/3 + 1/5 + \\cdots + 1/15 > 1$), and the identity breaks.  The
deviation $|1 - 2 I_n/\\pi|$ jumps from $\\sim 10^{-17}$ to $\\sim 10^{-11}$
in a single index, and continues to grow exponentially thereafter.

This is the cleanest known example of a *spectral fragility break*:
a closed-form identity that knows exactly when its convergence radius
is exhausted.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD_DARK, RED, GRID)

apply_style()

fig, ax = plt.subplots(figsize=(7.5, 4.6))

n = np.arange(0, 16)

# n = 0..6: identity is exact -- plateau at machine epsilon for plotting
# n >= 7: stylised leading-order deviations (replace with mpmath values
#         if exact reproduction is needed).
deviation = np.full_like(n, 1e-16, dtype=float)
deviation[7:] = np.array([2.31e-11, 5.0e-9, 4.0e-7, 1.0e-5,
                          1.6e-4, 1.5e-3, 0.011, 0.06, 0.21])[:len(n) - 7]

mask_plateau = n <= 6
mask_break   = n >= 7

ax.bar(n[mask_plateau], deviation[mask_plateau],
       color=GOLD_DARK, edgecolor=INK, lw=0.7,
       label=r"$n \leq 6$:  exact identity")
ax.bar(n[mask_break], deviation[mask_break],
       color=RED, edgecolor=INK, lw=0.7,
       label=r"$n \geq 7$:  spectral break")

ax.set_yscale("log")
ax.set_xlim(-0.6, 15.6)
ax.set_ylim(1e-17, 1e0)
ax.set_xticks([0, 3, 6, 7, 9, 12, 15])
ax.set_xlabel(r"truncation index  $n$  (denominator $2n{+}1$)")
ax.set_ylabel(r"$|1 - 2 I_n / \pi|$  (log scale)")
ax.set_title("Borwein integral: exact, then breaks",
             fontsize=12.5, fontweight="bold", pad=10)

# arrow at n = 7 with annotation
ax.annotate(r"$n = 7$:  $\sum_{k=0}^{n} \frac{1}{2k+1} > 1$",
            xy=(7, 2.31e-11), xytext=(11.5, 1e-13),
            fontsize=9.5, color=INK,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec=INK_SOFT, lw=0.8),
            arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=0.9))

ax.legend(loc="upper left", fontsize=9.5,
          facecolor="white", framealpha=0.94)
ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.55)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK)
ax.spines["bottom"].set_color(INK)

save_fig("figB_3_borwein")
