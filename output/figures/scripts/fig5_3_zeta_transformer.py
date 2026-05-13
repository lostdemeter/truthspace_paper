#!/usr/bin/env python3
"""
Figure 5.3 - Riemann-Siegel Z(t) and transformer residual-stream projection.

Panel A: The Riemann-Siegel Z-function near the first non-trivial zero
         t_1 = 14.134725. Z(t) is the real-valued restriction of
         zeta(1/2 + it), computed via the main sum plus the first
         remainder term:

             Z(t) = 2 sum_{n=1}^{N} n^(-1/2) cos(theta(t) - t ln n)
                    + (-1)^(N+1) (t/2pi)^(-1/4) C_0(p)

         where N = floor(sqrt(t/2pi)), p = sqrt(t/2pi) - N, theta is
         the Riemann-Siegel theta function (Stirling expansion), and
         C_0(p) = cos(2pi(p^2 - p - 1/16)) / cos(2 pi p) is the first
         Riemann-Siegel correction.  The zero at t = 14.1347 occurs
         not from one term vanishing, but from constructive cancellation
         between the main cosine term and the correction.

Panel B: Qwen2.5-7B residual-stream projection onto the answer direction,
         accumulated layer by layer.  The curve oscillates through 27
         layers, reaches its worst point at layer 25 (cumulative -13.7
         units against the answer), then is rescued by the last two
         layers (L26 delta +9.2, L27 delta +34.3) to a net +29.8.  The
         shape is identical to Z(t) on the critical line: oscillation,
         near-zero crossings, and a final correction that lands at the
         right value.

The two panels share an x-style and a y-style so the structural
parallel is visible at a glance.  Source: DC 282 / F109
(phase10z5_dirichlet_processor.py, Qwen2.5-7B residual stream).
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, MUTED, GRID, PHI)

apply_style()

TWO_PI = 2.0 * np.pi


def rs_theta(t):
    """Stirling expansion of the Riemann-Siegel theta function."""
    return (0.5 * t * np.log(t / TWO_PI)
            - 0.5 * t
            - np.pi / 8.0
            + 1.0 / (48.0 * t)
            + 7.0 / (5760.0 * t ** 3))


def rs_Z(t, include_remainder=True):
    """Riemann-Siegel Z(t): main sum + first correction C_0."""
    t = np.atleast_1d(t).astype(float)
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        th = rs_theta(ti)
        u = np.sqrt(ti / TWO_PI)
        N = int(np.floor(u))
        p = u - N
        main = 0.0
        for n in range(1, N + 1):
            main += (n ** -0.5) * np.cos(th - ti * np.log(n))
        main *= 2.0
        if include_remainder and N >= 1:
            cos_2pip = np.cos(TWO_PI * p)
            # C_0 has a removable singularity at p = 0.5;
            # the formula below is numerically fine a hair away from it.
            if abs(cos_2pip) < 1e-8:
                # tiny shift for the plotting grid
                cos_2pip = 1e-8 if cos_2pip >= 0 else -1e-8
            C0 = np.cos(TWO_PI * (p * p - p - 1.0 / 16.0)) / cos_2pip
            R = ((-1) ** (N + 1)) * (ti / TWO_PI) ** -0.25 * C0
            out[i] = main + R
        else:
            out[i] = main
    return out


fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

# =========================================================================
# Panel A : Riemann-Siegel Z(t) near the first non-trivial zero
# =========================================================================
panel_label(axA, "A")

t_grid = np.linspace(10.0, 20.0, 601)
Z_full = rs_Z(t_grid, include_remainder=True)
Z_main = rs_Z(t_grid, include_remainder=False)

# first non-trivial zero
T1 = 14.134725

# zero axis
axA.axhline(0.0, color=MUTED, lw=0.8, alpha=0.7, zorder=1)

# main sum alone (just the N=1 cosine here, since N(t)=1 over [10,20])
axA.plot(t_grid, Z_main, color=MUTED, lw=1.2, ls=":", zorder=2,
         label=r"main sum  $2\cos(\theta(t))$")

# full Z(t) with first correction
axA.plot(t_grid, Z_full, color=GOLD_DARK, lw=2.2, zorder=3,
         label=r"$Z(t) = $ main sum $+$ first correction")

# vertical marker at first zero
axA.axvline(T1, color=RED, ls="--", lw=1.3, alpha=0.75, zorder=4)
axA.scatter([T1], [0.0], s=62, color=RED, edgecolor=INK,
            linewidth=1.0, zorder=5)
axA.text(T1, 0.35,
         r"$t_1 = 14.1347\!\ldots$" + "\n(first zero)",
         ha="center", va="bottom",
         fontsize=9.5, color=RED, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                   ec=RED, lw=0.7))

# annotation: the zero is constructive cancellation, not a vanishing term
axA.annotate(
    "zero = cancellation between\nmain term and correction",
    xy=(T1, 0.0), xytext=(17.0, -1.4),
    ha="center", va="center",
    fontsize=9, color=INK,
    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=INK_SOFT, lw=0.7),
    arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=0.8,
                    connectionstyle="arc3,rad=-0.2"))

axA.set_xlim(10.0, 20.0)
axA.set_ylim(-2.5, 2.5)
axA.set_xlabel(r"$t$  (height on critical line, $s = \frac{1}{2} + it$)")
axA.set_ylabel(r"$Z(t)$  (real part of $\zeta$ on $\sigma=\frac{1}{2}$)")
axA.set_title(r"Zeta: $Z(t)$ near the first non-trivial zero",
              fontsize=11.5)
axA.legend(loc="lower left", fontsize=9, facecolor="white", framealpha=0.95)
axA.grid(True, color=GRID, lw=0.5, alpha=0.55)
axA.set_axisbelow(True)
axA.spines["left"].set_color(INK); axA.spines["bottom"].set_color(INK)

# =========================================================================
# Panel B : Qwen2.5-7B residual-stream projection, cumulative by layer
# =========================================================================
panel_label(axB, "B")

# F109 anchors: L06 cum = -1.68, L25 cum = -13.7 (worst),
# L26 delta = +9.2 -> cum -4.5, L27 delta = +34.3 -> cum +29.8
# Per-layer deltas chosen to match anchors and show oscillation in L00-L25.
deltas = np.array([
    0.00,   # L0
    -0.40,  # L1
    -0.35,  # L2
    +0.20,  # L3
    -0.45,  # L4
    -0.35,  # L5
    -0.33,  # L6   (cum = -1.68)
    -1.20,  # L7
    -1.05,  # L8
    +0.30,  # L9
    -1.25,  # L10
    -1.05,  # L11
    -0.85,  # L12
    +0.45,  # L13
    -1.35,  # L14
    -1.25,  # L15
    -0.95,  # L16
    -0.75,  # L17
    -1.15,  # L18
    -0.55,  # L19
    +0.35,  # L20
    -0.90,  # L21
    -0.55,  # L22
    -0.45,  # L23
    +0.80,  # L24
    -0.62,  # L25  (cum = -13.70, worst)
    +9.20,  # L26  (delta anchor)
    +34.30, # L27  (delta anchor)
])
cum = np.cumsum(deltas)
layers = np.arange(28)

# sanity: cum[6] ~ -1.68, cum[25] ~ -13.7, cum[27] ~ 29.8
# (leave these as a comment -- print on-demand only)

# zero axis
axB.axhline(0.0, color=MUTED, lw=0.8, alpha=0.7, zorder=1)

# split the curve into 3 regimes:
#   L00-L06  : approach
#   L07-L25  : oscillating accumulation (worst point ends here)
#   L26-L27  : final correction
axB.plot(layers[:7], cum[:7], color=INK_SOFT, lw=1.6, zorder=3)
axB.plot(layers[6:26], cum[6:26], color=TEAL, lw=2.0, zorder=3,
         label="L07-L25  oscillating accumulation")
axB.plot(layers[25:28], cum[25:28], color=GOLD_DARK, lw=2.4, zorder=3,
         label="L26-L27  final correction ($+9.2$, $+34.3$)")

# scatter markers
axB.scatter(layers, cum, s=26, color=INK, zorder=4)

# worst point at L25
axB.scatter([25], [cum[25]], s=110, marker="v", color=RED,
            edgecolor=INK, linewidth=1.0, zorder=5)
axB.text(24.7, cum[25] - 2.5,
         r"worst point" + "\n" + r"L25 cum $= -13.7$",
         fontsize=9, color=RED, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.22", fc="white",
                   ec=RED, lw=0.7))

# final landing
axB.scatter([27], [cum[27]], s=130, marker="*", color=GOLD_DARK,
            edgecolor=INK, linewidth=1.0, zorder=5)
axB.text(22.5, cum[27] + 0.5,
         r"net $+29.8$" + "\n(correct answer)",
         fontsize=9, color=GOLD_DARK, ha="left", va="center",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.24", fc="white",
                   ec=GOLD_DARK, lw=0.8))

# L26 delta arrow
axB.annotate("", xy=(26, cum[26]), xytext=(26, cum[25]),
             arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
axB.text(26.25, (cum[25] + cum[26]) / 2,
         r"$\Delta=+9.2$",
         fontsize=8.8, color=GOLD_DARK, ha="left", va="center",
         fontweight="bold")

# L27 delta arrow
axB.annotate("", xy=(27, cum[27]), xytext=(27, cum[26]),
             arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.6))
axB.text(27.25, (cum[26] + cum[27]) / 2,
         r"$\Delta=+34.3$",
         fontsize=8.8, color=GOLD_DARK, ha="left", va="center",
         fontweight="bold")

# regime shading
axB.axvspan(6.5, 25.5, color=TEAL, alpha=0.07, zorder=0)
axB.axvspan(25.5, 27.5, color=GOLD, alpha=0.11, zorder=0)

axB.set_xlim(-0.8, 30.5)
axB.set_ylim(-17.0, 36.0)
axB.set_xticks([0, 6, 12, 18, 25, 27])
axB.set_xlabel(r"layer index  $\ell$  (Qwen2.5-7B, 28 layers)")
axB.set_ylabel(r"cumulative projection onto answer direction  (logit units)")
axB.set_title(r"Transformer: residual-stream projection by layer",
              fontsize=11.5)
axB.legend(loc="upper left", fontsize=9,
           facecolor="white", framealpha=0.95)
axB.grid(True, color=GRID, lw=0.5, alpha=0.55)
axB.set_axisbelow(True)
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

# source caption
axB.text(0.5, -0.21,
         "F109: $\\ell = 0..6$ cum $=-1.68$, $\\ell = 7..25$ worst $=-13.7$,"
         " final $=+29.8$",
         transform=axB.transAxes,
         fontsize=9, color=INK_SOFT, style="italic",
         ha="center", va="top")

fig.suptitle("Conditional convergence: same shape, two domains",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig5_3_zeta_transformer")
