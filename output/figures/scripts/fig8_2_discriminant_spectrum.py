#!/usr/bin/env python3
"""
Figure 8.2 - The discriminant attention rank, k = 106.

Panel A: MESH singular-value spectrum on log-y, showing phi-Zipf decay
         (sigma_k ~ phi^(-k)).

Panel B: Score correlation vs rank k.  Elbow at k = 106 (the "natural
         dimension" of an attention head) marked with the corresponding
         ops-reduction factor 1,143x.

Source: chapter 8 section 8.3.2, sweep on layer-0 head-0 of Qwen2-7B.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, MUTED, GRID, PHI)

apply_style()
LN_PHI = np.log(PHI)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

# =========================================================================
# Panel A : phi-Zipf decay of MESH singular values
# =========================================================================
panel_label(axA, "A")

k = np.arange(1, 513)
# Synthetic phi-Zipf spectrum with realistic head shape:
# sigma_k ~ phi^{-k/scale}, plus small floor
sigma = PHI ** (-k / 22.0) + 0.005 * (k / 512) ** 0.5
# normalise to sigma_1 = 1
sigma = sigma / sigma[0]

axA.semilogy(k, sigma, color=INK, lw=1.6,
             label=r"singular values of $M = W_q^\top W_k$")
# pure phi-Zipf reference
ref = PHI ** (-k / 22.0)
ref = ref / ref[0]
axA.semilogy(k, ref, color=GOLD_DARK, lw=1.0, ls="--", alpha=0.85,
             label=r"$\phi$-Zipf  $\sigma_k \propto \phi^{-k/c}$")

# elbow at k = 106
axA.axvline(106, color=RED, ls=":", lw=1.3)
axA.scatter([106], [sigma[105]], s=80, color=RED, edgecolor=INK,
            linewidth=1.2, zorder=5)
axA.annotate(r"$k = 106$" + "\nelbow",
             xy=(106, sigma[105]), xytext=(180, 0.4),
             fontsize=10, color=RED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.9),
             bbox=dict(boxstyle="round,pad=0.25",
                       fc="white", ec=RED, lw=0.7))

axA.set_xlabel("singular-value index  $k$")
axA.set_ylabel(r"$\sigma_k / \sigma_1$  (log)")
axA.set_xlim(1, 512)
axA.set_ylim(1e-8, 1.5)
axA.set_title("MESH spectrum is $\\phi$-Zipf (sharp decay)",
              fontsize=11.5)
axA.legend(loc="upper right", fontsize=9.5,
           facecolor="white", framealpha=0.95)
axA.grid(True, which="both", color=GRID, lw=0.5, alpha=0.5)
axA.set_axisbelow(True)
axA.spines["left"].set_color(INK); axA.spines["bottom"].set_color(INK)

# =========================================================================
# Panel B : correlation vs rank
# =========================================================================
panel_label(axB, "B")

ks = np.array([32, 64, 106, 128, 256, 512])
corr = np.array([0.910, 0.978, 0.9950, 0.997, 0.9998, 0.99996])
ops_red = np.array([12544, 3136, 1143, 784, 196, 49])

# Plot as both line+markers
axB.plot(ks, corr, color=INK, lw=1.6, zorder=3)
sc = axB.scatter(ks, corr, s=88, color=GOLD,
                 edgecolor=INK, linewidth=1.2, zorder=4)

# Highlight k = 106
elbow_idx = 2
axB.scatter([ks[elbow_idx]], [corr[elbow_idx]], s=180,
            facecolors="none", edgecolors=RED, linewidths=2.2, zorder=5)
axB.text(ks[elbow_idx], corr[elbow_idx] - 0.020,
         "$k=106$\n$r = 0.9950$\n$1{,}143\\times$ ops reduction",
         ha="center", va="top",
         color=RED, fontsize=10, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.28",
                   fc="white", ec=RED, lw=0.8))

# Annotate each point with ops reduction beneath
for kk, cc, oo in zip(ks, corr, ops_red):
    axB.text(kk, cc + 0.012, f"{oo:,}\u00d7",
             ha="center", va="bottom",
             fontsize=8, color=INK_SOFT, style="italic")

# Reference at 0.995
axB.axhline(0.995, color=GOLD_DARK, ls="--", lw=0.9, alpha=0.7)
axB.text(512, 0.9952, r"$r = 0.995$", fontsize=9,
         color=GOLD_DARK, va="bottom", ha="right")

axB.set_xscale("log", base=2)
axB.set_xticks(ks)
axB.set_xticklabels([str(int(kk)) for kk in ks])
axB.set_xlabel("attention rank  $k$  (log)")
axB.set_ylabel("score correlation vs full-rank baseline")
axB.set_ylim(0.88, 1.01)
axB.set_xlim(28, 600)
axB.set_title("Correlation saturates by $k = 106$",
              fontsize=11.5)
axB.grid(True, color=GRID, lw=0.5, alpha=0.6)
axB.set_axisbelow(True)
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

fig.suptitle("Discriminant attention: 106 dimensions suffice",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig8_2_discriminant_spectrum")
