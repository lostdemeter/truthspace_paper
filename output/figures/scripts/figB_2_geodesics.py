#!/usr/bin/env python3
"""
Figure B.2 - Geodesics on the conformal metric pinch toward $\\sigma = 1/2$.

The conformal factor $e^{2\\Phi}$ with potential $\\Phi = \\log|\\zeta(s)\\zeta(1-s)|$
has a parabolic well centred on $\\sigma = 1/2$ throughout the critical
strip.  Geodesics on this metric fall toward the critical line as if
into an attractor basin.

Critical-line zeros (gold dots) are completeness anchors: geodesics
terminate on them smoothly.  An injected off-line zero (red X)
immediately breaks completeness -- half the trajectories crash at the
injected zero, half escape to $\\sigma \\to 1$.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD_DARK, RED, TEAL, GRID)

apply_style()

fig, ax = plt.subplots(figsize=(7.5, 5.0))

# Critical strip: 0 <= sigma <= 1, plot a region around it
sigma_grid = np.linspace(0.02, 0.98, 200)
t_grid     = np.linspace(0.0, 6.0, 200)
SIG, TT = np.meshgrid(sigma_grid, t_grid)

# stylised conformal factor:  e^{2 phi}  with potential
#   u(sigma, t) = log|zeta(s) zeta(1-s)|, minimised at sigma=1/2
# Approximation: a parabolic well centred on sigma = 0.5
phi_field = -2.5 * (SIG - 0.5) ** 2 + 0.18 * np.cos(TT * np.pi)
metric = np.exp(2.0 * phi_field)

ax.contourf(SIG, TT, metric, levels=12, cmap="YlOrBr", alpha=0.55)
ax.contour(SIG, TT, metric, levels=8, colors=[INK_SOFT],
           linewidths=0.5, alpha=0.4)

# critical line
ax.axvline(0.5, color=GOLD_DARK, lw=2.4, alpha=0.9)
ax.text(0.50, 5.85, r"$\sigma = \frac{1}{2}$",
        ha="center", va="top", fontsize=11.5, color=GOLD_DARK,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                  ec=GOLD_DARK, lw=0.8))

# stylised geodesics curving into the line
for sigma0, color in [(0.18, TEAL), (0.82, TEAL),
                      (0.30, INK), (0.70, INK)]:
    tau = np.linspace(0, 5, 100)
    sigma_traj = 0.5 + (sigma0 - 0.5) * np.exp(-0.55 * tau)
    t_traj = 0.6 + 0.85 * tau
    ax.plot(sigma_traj, t_traj, color=color, lw=1.8,
            alpha=0.85, zorder=4)
    end = -1
    ax.annotate("", xy=(sigma_traj[end], t_traj[end]),
                xytext=(sigma_traj[end - 6], t_traj[end - 6]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
                zorder=5)

# zeros plotted on the line
zero_t = [1.5, 3.4, 5.1]
ax.scatter([0.5] * 3, zero_t, s=80, color=GOLD_DARK,
           edgecolor=INK, linewidth=1.0, zorder=6)
ax.text(0.55, 3.4, "zeros\n(geodesic\nattractors)",
        ha="left", va="center", fontsize=9.5, color=INK_SOFT,
        style="italic")

# off-line zero - incomplete geodesic warning
ax.scatter([0.74], [4.3], s=110, marker="X", color=RED,
           edgecolor=INK, linewidth=1.0, zorder=6)
ax.text(0.93, 4.3, "off-line\nbreaks\nsmoothness",
        ha="right", va="center", fontsize=9, color=RED,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                  ec=RED, lw=0.7))

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.5, 6.0)
ax.set_xlabel(r"real part  $\sigma$")
ax.set_ylabel(r"imaginary part  $t$")
ax.set_title(r"Geodesics fall toward $\sigma = \frac{1}{2}$",
             fontsize=12.5, fontweight="bold", pad=10)
ax.grid(True, color=GRID, lw=0.4, alpha=0.4)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK)
ax.spines["bottom"].set_color(INK)

save_fig("figB_2_geodesics")
