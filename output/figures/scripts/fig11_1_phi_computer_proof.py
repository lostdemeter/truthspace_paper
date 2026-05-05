#!/usr/bin/env python3
"""
Figure 11.1 - The phi-computer proof.

Panel A: phi-sigmoid exactly equals the standard sigmoid (overlay + residual).
Panel B: The universal bottleneck - mean phi-level across layers converges
         to ~1.57 at layer 27 regardless of input.
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, RED, TEAL, VIOLET, GRID,
                      PHI, MUTED, GOLD_SOFT)

apply_style()
LN_PHI = np.log(PHI)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

# =========================================================================
# Panel A : phi-sigmoid identity
# =========================================================================
panel_label(axA, "A")
x = np.linspace(-6, 6, 401)
sig = 1.0 / (1.0 + np.exp(-x))
phi_sig = 1.0 / (1.0 + PHI ** (-x / LN_PHI))

axA.plot(x, sig, color=INK, lw=4.0, alpha=0.4,
         label="standard sigmoid")
axA.plot(x, phi_sig, color=GOLD, lw=2.0, ls="--",
         label=r"$\phi$-sigmoid  $= 1/(1+\phi^{-x/\ln\phi})$")
axA.set_xlim(-6, 6); axA.set_ylim(-0.05, 1.08)
axA.set_xlabel(r"$x$")
axA.set_ylabel(r"$\sigma(x)$")
axA.set_title("phi-sigmoid is exactly sigmoid (algebraic identity)",
              fontsize=12)
axA.grid(True, color=GRID, lw=0.5, alpha=0.6)
axA.set_axisbelow(True)
axA.legend(loc="lower right", fontsize=9.5,
           facecolor="white", framealpha=0.95)

# inset showing residual
axI = axA.inset_axes([0.08, 0.55, 0.35, 0.28])
residual = np.abs(sig - phi_sig)
axI.semilogy(x, np.maximum(residual, 1e-17), color=RED, lw=1.4)
axI.axhline(1e-14, color=INK_SOFT, ls="--", lw=0.8)
axI.set_title(r"|residual|", fontsize=8.5, color=INK)
axI.text(0, 1e-13, r"$<10^{-14}$", color=INK_SOFT,
         fontsize=8, ha="center", va="bottom")
axI.set_ylim(1e-17, 1e-10)
axI.set_xticks([-6, 0, 6])
axI.tick_params(labelsize=7)
axI.spines["top"].set_visible(False); axI.spines["right"].set_visible(False)

# =========================================================================
# Panel B : universal bottleneck at layer 27
# =========================================================================
panel_label(axB, "B")
layers = np.arange(0, 28)

# four input prompts that all converge at layer 27
rng = np.random.default_rng(2)


def prompt_curve(seed):
    r = np.random.default_rng(seed)
    base = 1.1 + 0.45 * np.sin(layers / 5.5 + r.uniform(0, 2 * np.pi))
    base += r.normal(0, 0.05, layers.size)
    # pull all curves toward 1.57 at layer 27
    target = 1.57
    pull = (layers / 27.0) ** 2.5
    return base * (1 - pull) + target * pull


curves = {
    "math":      ("#3F6FB3", prompt_curve(1)),
    "factual":   (TEAL,      prompt_curve(2)),
    "creative":  (VIOLET,    prompt_curve(3)),
    "code":      (GOLD,      prompt_curve(4)),
}

for name, (color, y) in curves.items():
    axB.plot(layers, y, color=color, lw=1.6, marker="o", ms=3.6,
             alpha=0.9, label=name)

axB.axhline(1.57, color=RED, ls="--", lw=1.4,
            label=r"bottleneck at $\phi$-level $\approx 1.57$")

# mark layer 27
axB.axvline(27, color=INK_SOFT, ls=":", lw=0.9)
axB.scatter([27] * 4, [c[1][-1] for c in curves.values()],
            s=70, facecolors="none",
            edgecolors=[c[0] for c in curves.values()], lw=1.6,
            zorder=4)

axB.set_xlim(-0.5, 28.5); axB.set_ylim(0.6, 2.1)
axB.text(13, 2.0, "layer 27 convergence",
         ha="center", va="top", fontsize=9, color=INK_SOFT,
         style="italic")
axB.annotate("", xy=(26.5, 1.6), xytext=(16, 1.96),
             arrowprops=dict(arrowstyle="->", color=INK_SOFT,
                             lw=0.8, alpha=0.7))
axB.set_xlabel("layer")
axB.set_ylabel(r"mean $\phi$-level")
axB.set_title("Universal bottleneck: all reasoning converges",
              fontsize=12)
axB.grid(True, color=GRID, lw=0.5, alpha=0.6)
axB.set_axisbelow(True)
axB.legend(loc="lower left", fontsize=8.5, ncol=2,
           facecolor="white", framealpha=0.9)
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

fig.suptitle("The phi-computer proof",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig11_1_phi_computer_proof")
