#!/usr/bin/env python3
"""
Figure B.1 - Light cone: $\\beta \\le 1/2$ is the speed limit.

The Chebyshev fluctuation $F(t)$ stays bounded after $\\sqrt{x}$-
normalisation only when the dominant zero of $\\zeta$ has real part
$\\beta \\le 1/2$.  Plotted as $|F_\\beta(t)| \\, e^{-t/2}$ for three
values of $\\beta$:

  - $\\beta = 0.40$:  sub-luminal -- curve decays
  - $\\beta = 0.50$:  light cone  -- curve bounded (the empirical case)
  - $\\beta = 0.60$:  tachyonic   -- curve blows up

Only $\\beta = 1/2$ is consistent with the observed boundedness of
prime fluctuations up to $x = 10^7$.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD_DARK, RED, TEAL, GRID)

apply_style()

fig, ax = plt.subplots(figsize=(7.5, 4.6))

t = np.linspace(0.5, 10.0, 400)
# Hypothetical Chebyshev fluctuation F_beta(t) ~ e^{beta t} with oscillation;
# the dominant zero has real part beta.  The RH bound is e^{t/2}, so we plot
# F_beta(t) / e^{t/2} = e^{(beta - 1/2) t}  (with stylised oscillation).
osc = 1.0 + 0.18 * np.cos(2.0 * t)
for beta, color, lw, ls, lab in [
    (0.40, TEAL,      1.8, ":", r"$\beta = 0.40$  (sub-RH: decays)"),
    (0.50, GOLD_DARK, 2.6, "-", r"$\beta = 0.50$  (light cone: bounded)"),
    (0.60, RED,       2.2, "-", r"$\beta = 0.60$  (tachyonic: blows up)"),
]:
    curve = np.exp((beta - 0.5) * t) * osc
    ax.plot(t, curve, color=color, lw=lw, ls=ls, label=lab)

# shade tachyonic forbidden zone
ax.axhspan(2.0, 4.5, color=RED, alpha=0.06)
ax.text(9.7, 3.6, "forbidden", fontsize=9,
        color=RED, ha="right", va="top", style="italic")

ax.set_xlim(0.5, 10.0)
ax.set_ylim(0, 4.5)
ax.set_xlabel(r"multiplicative time  $t = \log x$")
ax.set_ylabel(r"$|F_\beta(t)| \cdot e^{-t/2}$  (RH-normalised)")
ax.set_title(r"$\beta \leq \frac{1}{2}$ is the speed limit",
             fontsize=12.5, fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=9.5,
          facecolor="white", framealpha=0.94)
ax.grid(True, color=GRID, lw=0.5, alpha=0.55)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK)
ax.spines["bottom"].set_color(INK)

save_fig("figB_1_light_cone")
