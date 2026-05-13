#!/usr/bin/env python3
"""
Figure B.5 - Half-step offset: $N_{\\mathrm{smooth}}(t_n) \\approx n - 1/2$.

The smooth zero count $N_{\\mathrm{smooth}}(t) = \\theta(t)/\\pi + 1$
(from Riemann--Siegel $\\theta$ via Stirling) lands almost exactly half
a step behind the integer zero index $n$.  The residual
$N_{\\mathrm{smooth}}(t_n) - (n - 1/2)$ is just the negative of
$S(t)/\\pi$ -- the bounded $S(t)$ noise -- and oscillates around zero
with RMS $\\approx 0.169$ for the first 20 zeros.

The $1/2$ offset is *exact*; the residual is just $S(t)$.  This is the
discrete signature of operating on $\\sigma = 1/2$: the smooth count is
the average, the integer count is the instantaneous, and the gap
between them is exactly half a step on average.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD_DARK, RED, GRID)

apply_style()
TWO_PI = 2.0 * np.pi


# Riemann-Siegel theta via Stirling
def rs_theta(t):
    return (0.5 * t * np.log(t / TWO_PI)
            - 0.5 * t
            - np.pi / 8.0
            + 1.0 / (48.0 * t)
            + 7.0 / (5760.0 * t ** 3))


fig, ax = plt.subplots(figsize=(7.5, 4.6))

# First 20 imaginary parts of non-trivial zeros (Odlyzko table)
zeros = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
])
n_idx = np.arange(1, len(zeros) + 1)

# N_smooth(t_n) = theta(t_n) / pi + 1
N_smooth = rs_theta(zeros) / np.pi + 1.0
residual = N_smooth - (n_idx - 0.5)

# bars: residual = -S(t_n)/pi
colors = [GOLD_DARK if r >= 0 else RED for r in residual]
ax.bar(n_idx, residual, color=colors, edgecolor=INK,
       lw=0.6, alpha=0.88)

ax.axhline(0.0, color=INK, lw=1.0, zorder=4)

# RMS lines
rms = float(np.sqrt(np.mean(residual ** 2)))
ax.axhline( rms, color=INK_SOFT, ls=":", lw=1.1, alpha=0.75)
ax.axhline(-rms, color=INK_SOFT, ls=":", lw=1.1, alpha=0.75)
ax.text(20.4, rms, f"$\\pm$ rms $= {rms:.3f}$",
        fontsize=9.5, color=INK_SOFT, ha="right", va="bottom",
        style="italic")

ax.set_xlim(0.4, 20.6)
ax.set_ylim(-0.30, 0.30)
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel(r"zero index  $n$")
ax.set_ylabel(r"$N_{\mathrm{smooth}}(t_n) - (n - \frac{1}{2}) = -S(t_n)$")
ax.set_title(r"Smooth count is half a step behind",
             fontsize=12.5, fontweight="bold", pad=10)
ax.grid(True, color=GRID, lw=0.5, alpha=0.55)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK)
ax.spines["bottom"].set_color(INK)

# annotation explaining what the bars are
ax.text(0.5, 0.97,
        r"residual is small and oscillating  $\Rightarrow$  the offset $\frac{1}{2}$ is exact",
        transform=ax.transAxes,
        fontsize=9.5, color=INK,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.24", fc="white",
                  ec=INK_SOFT, lw=0.8))

save_fig("figB_5_half_step_offset")
