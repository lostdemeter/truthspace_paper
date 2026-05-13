#!/usr/bin/env python3
"""
Figure 12.3 - Cumulative variance: about 79 Platonic Ideals.

PCA on 88 single-token concept embeddings in Qwen2-7B (DC 299):
    50% variance at  27 dimensions
    90% variance at  71 dimensions
    95% variance at  79 dimensions  <- "Platonic Ideals" working number
    99% variance at  86 dimensions

The figure presents the cumulative-variance curve, marks the four
thresholds, and contextualizes the result inside the full
3584-dimensional embedding space.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, MUTED, GRID, PHI)

apply_style()

# =========================================================================
# Construct a synthetic cumulative-variance curve that hits the four
# empirical anchor points.
# =========================================================================
anchors_k = np.array([0,  27,   71,   79,   86,   88])
anchors_y = np.array([0,  50,   90,   95,   99,  100])

k = np.arange(0, 89)
cum = np.interp(k, anchors_k, anchors_y)

# =========================================================================
fig, ax = plt.subplots(figsize=(12, 5.4))

# Shaded thresholds (50/90/95/99)
ax.axhline(50, color=MUTED, ls=":", lw=0.8, alpha=0.6)
ax.axhline(90, color=MUTED, ls=":", lw=0.8, alpha=0.6)
ax.axhline(95, color=RED,   ls="--", lw=1.4, alpha=0.85,
           label="95% variance threshold  ($\\to$ 79 Platonic Ideals)")
ax.axhline(99, color=GOLD_DARK, ls="--", lw=0.9, alpha=0.7)

# Cumulative curve
ax.plot(k, cum, color=INK, lw=2.4, zorder=3)
ax.fill_between(k, 0, cum, color=GOLD_SOFT, alpha=0.4, zorder=2)

# Threshold annotation points - staggered text positions to avoid overlap
threshold_data = [
    # (k, variance, label,                 colour,     dx,    dy)
    (27, 50, "50% at $k = 27$",            MUTED,     -2,   -4),
    (71, 90, "90% at $k = 71$",            MUTED,     -3,   -5),
    (79, 95, "95% at $k = 79$",            RED,       -2,   +4),
    (86, 99, "99% at $k = 86$",            GOLD_DARK, +3,   -5),
]

for xk, yk, lbl, col, dx, dy in threshold_data:
    ax.scatter([xk], [yk], s=110 if col == RED else 60,
               color=col, edgecolor=INK,
               linewidth=1.4 if col == RED else 0.8,
               zorder=5)
    ax.plot([xk, xk], [0, yk], color=col, lw=0.9,
            ls=":", alpha=0.5, zorder=2)
    ax.text(xk + dx, yk + dy, lbl,
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=10.5 if col == RED else 9.5,
            fontweight="bold" if col == RED else "normal",
            color=col)

# Highlight 95% threshold with a big callout in the lower middle
ax.annotate(r"$\sim 79$ Platonic Ideals" + "\n" +
            r"span $95\%$ of concept space",
            xy=(79, 95),
            xytext=(45, 38),
            fontsize=11.5, color=RED, fontweight="bold",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
            bbox=dict(boxstyle="round,pad=0.30",
                      fc="white", ec=RED, lw=1.2))

# Side annotation: 88 concepts, 3584-dim space
ax.text(0.04, 0.95,
        "Probe set:  $\\mathbf{88}$ single-token concepts in $\\mathbf{3584}$-dim embedding space.\n"
        "Manual axes ($6$) recover $9.1\\%$ of variance.\n"
        "PCA optimal $3$ axes recover the same $9.1\\%$.",
        transform=ax.transAxes,
        fontsize=9.5, color=INK,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.32",
                  fc="white", ec=GRID, lw=0.8))

# Axes
ax.set_xlim(0, 88)
ax.set_ylim(0, 105)
ax.set_xticks([0, 10, 27, 50, 71, 79, 86, 88])
ax.set_xlabel("number of PCA dimensions  $k$")
ax.set_ylabel("cumulative variance  (%)")
ax.set_title("Concept space is finite-dimensional: $\\sim 79$ Platonic Ideals at $95\\%$",
             fontsize=12)
ax.legend(loc="lower right", fontsize=10,
          facecolor="white", framealpha=0.95)
ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)

# Source caption
ax.text(0.5, -0.16,
        "source: DC 299 Phase 1 PCA on Qwen2-7B embeddings",
        transform=ax.transAxes,
        fontsize=9, color=INK_SOFT, style="italic",
        ha="center", va="top")

fig.suptitle("PCA cumulative variance: how many Platonic Ideals?",
             fontsize=15, fontweight="bold", y=1.00)

save_fig("fig12_3_pca_variance")
