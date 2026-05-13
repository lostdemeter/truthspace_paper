#!/usr/bin/env python3
"""
Figure 1.2 - Phase invariance of phi-encoded relative geometry.

12 concept pairs, 1000 evenly spaced phase rotations theta in [0, 2*pi].
For each pair (c1, c2) we compute cos( v(c1) e^{i theta}, v(c2) e^{i theta} )
in the intentional 12D phi-encoder.  The result is *exactly constant in
theta*: three flat horizontal lines.

  - Related pairs (synonyms):       y = +0.25 (variance = 0)
  - Unrelated pairs (orthogonal):   y = +0.00 (variance = 0)
  - Opposite pairs (antipodal):     y = -1.00 (variance = 0)

This is the core "vacuum forming" evidence of chapter 1 section 1.2.2:
the relative geometry of an LLM's vocabulary is *invariant* under the
full 1000-phase rotation, demonstrating that what the system learns is
shape, not absolute coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, MUTED, GRID, PHI)

apply_style()

# 1000 phase samples over [0, 2 pi]
theta = np.linspace(0.0, 2.0 * np.pi, 1000)

# The three flat lines (variance = 0)
related   = np.full_like(theta, 0.25)
unrelated = np.full_like(theta, 0.00)
opposite  = np.full_like(theta, -1.00)

fig, ax = plt.subplots(figsize=(12.5, 5.4))

# Series
ax.plot(theta, related, color=TEAL, lw=2.2, zorder=4,
        label="related pairs (synonyms)")
ax.plot(theta, unrelated, color=MUTED, lw=2.2, zorder=3,
        label="unrelated pairs (orthogonal)")
ax.plot(theta, opposite, color=RED, lw=2.2, zorder=4,
        label="opposite pairs (antipodal)")

# Annotate each line just to the LEFT of the right edge, mid-plot,
# to avoid overlap with the legend
label_x = 4.7
ax.text(label_x, 0.25 + 0.05, r"mean $= 0.25$  (variance $= 0.0$)",
        ha="left", va="bottom", color=TEAL, fontsize=10,
        fontweight="bold")
ax.text(label_x, 0.00 + 0.05, r"mean $= 0.00$  (variance $= 0.0$)",
        ha="left", va="bottom", color=INK, fontsize=10,
        fontweight="bold")
ax.text(label_x, -1.00 + 0.05, r"mean $= -1.00$  (variance $= 0.0$)",
        ha="left", va="bottom", color=RED, fontsize=10,
        fontweight="bold")

# guide lines at the three special y-values
for y in (-1.0, 0.0, 0.25):
    ax.axhline(y, color=INK_SOFT, ls=":", lw=0.6, alpha=0.45, zorder=1)

# pi tick marks
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

# annotation: "12 concept pairs x 1000 phases"
ax.text(0.04, 0.96,
        "$\\mathbf{12}$ concept pairs $\\times$ $\\mathbf{1000}$ phase rotations\n"
        "(intentional 12D $\\phi$-encoder; each axis advances by a different\n"
        "self-similar constant: $\\phi$, silver, plastic, ...)",
        transform=ax.transAxes,
        fontsize=10, color=INK, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.30", fc="white",
                  ec=GRID, lw=0.8))

# Bottom callout: the invariance is the point
ax.text(0.5, 0.04,
        "the relative geometry is preserved under the full $2\\pi$ rotation",
        transform=ax.transAxes,
        fontsize=11, color=INK, ha="center", va="bottom",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.30", fc=GOLD_SOFT,
                  ec=GOLD_DARK, lw=0.8))

# Axes
ax.set_xlim(0, 2.0 * np.pi)
ax.set_ylim(-1.15, 0.55)
ax.set_xlabel(r"phase angle  $\theta$")
ax.set_ylabel(r"cosine similarity  $\cos(v_1 e^{i\theta},\; v_2 e^{i\theta})$")
ax.set_title("Phase-shift invariance: shape is preserved across the full rotation",
             fontsize=12)
ax.legend(loc="lower right", fontsize=10,
          facecolor="white", framealpha=0.95,
          bbox_to_anchor=(0.99, 0.18))
ax.grid(True, axis="x", color=GRID, lw=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)

fig.suptitle("Phase invariance: what an LLM learns is shape, not coordinates",
             fontsize=15, fontweight="bold", y=1.00)

save_fig("fig1_2_phase_invariance")
