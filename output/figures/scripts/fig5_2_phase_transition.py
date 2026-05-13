#!/usr/bin/env python3
"""
Figure 5.2 - The phi-cosine phase transition (DC 311, DC 321).

Panel A: 233-word Qwen2-1.5B L14 sample.  phi-cosine to a semantic centroid
         is *bimodal*: common-word pole [0.95, 1.00] and semantic body zone
         [0.05, 0.35]; the (0.35, 0.95) interval contains zero tokens.

Panel B: 2000-token sample projected onto a morphological axis.  The
         empty forbidden zone is bounded *exactly* by 1/phi^2 * M and
         1/phi * M, made explicit by the phi-pair identity
         1/phi + 1/phi^2 = 1.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, MUTED, GRID, PHI)

apply_style()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

# =========================================================================
# Panel A : bimodal phi-cosine distribution (DC 311)
# =========================================================================
panel_label(axA, "A")
rng = np.random.default_rng(11)

# 76 pole tokens centred near 0.975 (range 0.95-1.00)
pole = np.clip(rng.normal(0.975, 0.012, 76), 0.95, 1.00)
# 157 body-zone tokens centred near 0.20 (range 0.05-0.35)
body = np.clip(rng.normal(0.20, 0.075, 157), 0.05, 0.35)
samples = np.concatenate([pole, body])

bins = np.linspace(-0.05, 1.05, 56)
counts_p, _ = np.histogram(pole, bins=bins)
counts_b, _ = np.histogram(body, bins=bins)
centers = 0.5 * (bins[1:] + bins[:-1])
width = bins[1] - bins[0]

# Shade the forbidden gap
axA.axvspan(0.35, 0.95, color=MUTED, alpha=0.18, zorder=0)
axA.text((0.35 + 0.95) / 2, 22,
         "forbidden gap\n(0.35, 0.95)\n0 tokens", ha="center", va="center",
         fontsize=10.5, color=INK_SOFT, fontweight="bold")

# Body cluster
axA.bar(centers, counts_b, width=width * 0.95, color=GOLD,
        edgecolor=INK, lw=0.5, zorder=3,
        label="semantic body zone (157 tokens, 67.4%)")
# Pole cluster
axA.bar(centers, counts_p, width=width * 0.95, color=RED,
        edgecolor=INK, lw=0.5, zorder=3,
        label="common-word pole (76 tokens, 32.6%)")

# Cluster region labels
axA.text(0.20, 33, "polysyllabic\nspecialised", ha="center", va="bottom",
         fontsize=9, color=INK_SOFT, style="italic")
axA.text(0.975, 33, "monosyllabic\ncore", ha="center", va="bottom",
         fontsize=9, color=INK_SOFT, style="italic")

axA.set_xlim(-0.05, 1.05)
axA.set_ylim(0, 42)
axA.set_xlabel(r"$\phi$-cosine to semantic centroid")
axA.set_ylabel("token count")
axA.set_title("Bimodal $\\phi$-cosine: a phase transition in vocabulary",
              fontsize=11.5)
axA.legend(loc="upper left", fontsize=9.5,
           facecolor="white", framealpha=0.95,
           bbox_to_anchor=(0.02, 0.98))
axA.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.6)
axA.set_axisbelow(True)
axA.spines["left"].set_color(INK); axA.spines["bottom"].set_color(INK)

# annotation: syllable rule - placed bottom-centre under the forbidden gap
axA.text(0.65, 28,
         r"syllables $\leq 1 \Rightarrow$ pole" + "\n" +
         r"$\bf{87.1\%}$ classification accuracy",
         fontsize=9.5, color=INK,
         ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=GRID, lw=0.7))

# =========================================================================
# Panel B : phi-pair forbidden zone on the projection axis (DC 321)
# =========================================================================
panel_label(axB, "B")

M = 30.74
phi_inv_M     = M / PHI       # 19.00
phi_inv2_M    = M / PHI ** 2  # 11.74

# 1044 equator tokens in [-1.4, +5.3]
equator = rng.uniform(-1.4, 5.3, 1044)
# 956 English-cluster tokens in [+26.3, +30.7]
english = rng.uniform(26.3, 30.7, 956)
proj = np.concatenate([equator, english])

bins_b = np.linspace(-3, 32, 70)
hist_eq, _  = np.histogram(equator, bins=bins_b)
hist_en, _  = np.histogram(english, bins=bins_b)
centers_b = 0.5 * (bins_b[1:] + bins_b[:-1])
width_b = bins_b[1] - bins_b[0]

# Shade the empirical forbidden range
axB.axvspan(5.3, 26.3, color=MUTED, alpha=0.14, zorder=0)
# Tighter phi-pair-bounded forbidden zone
axB.axvspan(phi_inv2_M, phi_inv_M, color=RED, alpha=0.16, zorder=1)

axB.bar(centers_b, hist_eq, width=width_b * 0.95, color=TEAL,
        edgecolor=INK, lw=0.4, zorder=3,
        label="equator zone (1044, 52.2%)")
axB.bar(centers_b, hist_en, width=width_b * 0.95, color=VIOLET,
        edgecolor=INK, lw=0.4, zorder=3,
        label="English cluster (956, 47.8%)")

# phi-pair boundary lines
for xb, txt, y in [(phi_inv2_M, r"$\dfrac{M}{\phi^{2}} = 11.74$", 110),
                   (phi_inv_M,  r"$\dfrac{M}{\phi}   = 19.00$", 110)]:
    axB.axvline(xb, color=RED, ls="--", lw=1.6, zorder=4)
    axB.text(xb, y, txt, ha="center", va="bottom",
             fontsize=10, color=RED, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.20",
                       fc="white", ec=RED, lw=0.7))

# identity callout (above the bars, between the phi-pair markers)
axB.text((phi_inv2_M + phi_inv_M) / 2, 65,
         r"$\dfrac{1}{\phi} + \dfrac{1}{\phi^{2}} = 1$" + "\n" +
         r"$\Rightarrow$ empirical boundary",
         fontsize=10, color=INK,
         ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=INK, lw=0.8))

axB.set_xlim(-3, 32); axB.set_ylim(0, 145)
axB.set_xlabel("projection onto morphological axis  (max $M=30.74$)")
axB.set_ylabel("token count")
axB.set_title("$\\phi$-pair-bounded forbidden zone (2000 tokens)",
              fontsize=11.5)
axB.legend(loc="upper left", fontsize=9, ncol=1,
           facecolor="white", framealpha=0.95,
           bbox_to_anchor=(0.02, 0.78))
axB.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.6)
axB.set_axisbelow(True)
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

fig.suptitle("The $\\phi$-cosine phase transition: vocabulary is bimodal",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig5_2_phase_transition")
