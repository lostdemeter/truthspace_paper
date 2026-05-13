#!/usr/bin/env python3
"""
Figure B.4 - Only $\\sigma = 1/2$ is conditionally convergent.

Partial sums $S_N(s) = \\sum_{n=1}^{N} n^{-s}$ for $s = \\sigma + 14.1347 i$
(the height of the first non-trivial zero) at three amplitudes:

  - $\\sigma = 0.50$:  conditional convergence -- $|S_N| \\sim \\sqrt{N}$,
                       partial sums never settle, value emerges only
                       from the Riemann--Siegel cancellation
  - $\\sigma = 0.80$:  finite limit, still oscillates during transit
  - $\\sigma = 1.20$:  absolute convergence, a few terms suffice

Only $\\sigma = 1/2$ produces the conditional regime that makes the
Riemann--Siegel formula necessary, and only this regime matches the
empirical behaviour of $|\\zeta(\\tfrac{1}{2} + it)|$ on the critical line.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD_DARK, TEAL, MUTED, GRID)

apply_style()

fig, ax = plt.subplots(figsize=(7.5, 4.6))

# Partial sums of n^{-s} for s = sigma + 14.1347 i
N_max = 200
N = np.arange(1, N_max + 1)
n_idx = N.astype(float)
ln_n = np.log(n_idx)
t_zero = 14.134725

for sigma, color, lw, lab in [
    (0.50, GOLD_DARK, 2.2, r"$\sigma = \frac{1}{2}$  (conditional)"),
    (0.80, TEAL,      1.7, r"$\sigma = 0.80$"),
    (1.20, MUTED,     1.4, r"$\sigma = 1.20$  (absolute)"),
]:
    real = np.cumsum(n_idx ** -sigma * np.cos(t_zero * ln_n))
    imag = np.cumsum(n_idx ** -sigma * np.sin(t_zero * ln_n))
    mag = np.sqrt(real * real + imag * imag)
    ax.plot(N, mag, color=color, lw=lw, label=lab)

ax.set_xlim(0, N_max)
ax.set_ylim(0, 4.5)
ax.set_xlabel(r"partial-sum length  $N$")
ax.set_ylabel(r"$\left|\sum_{n=1}^{N} n^{-s}\right|$  at  $t = 14.1347$")
ax.set_title(r"Only $\sigma = \frac{1}{2}$ is conditionally convergent",
             fontsize=12.5, fontweight="bold", pad=10)
ax.legend(loc="upper left", fontsize=9.5,
          facecolor="white", framealpha=0.94)
ax.grid(True, color=GRID, lw=0.5, alpha=0.55)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK)
ax.spines["bottom"].set_color(INK)

# annotate the conditional regime, lower-right corner
ax.text(195, 0.45,
        r"$\sigma = \frac{1}{2}$:  $|S_N| \sim \sqrt{N}$" + "\n" +
        r"$\sigma > \frac{1}{2}$:  $S_N \to \zeta(s)$",
        fontsize=9.5, color=INK,
        ha="right", va="bottom", style="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec=INK_SOFT, lw=0.8))

save_fig("figB_4_conditional_convergence")
