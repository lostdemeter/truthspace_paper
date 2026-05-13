#!/usr/bin/env python3
"""
Figure 10.2 - The two complementary spectra of the irreducible shape.

Panel A: MESH magnitudes - phi-Zipf decay sigma_k ~ phi^(-k).
         Eigenvalues drop by ~1/phi per step, ~89 levels suffice (7-bit
         residual + sign + 8-bit level = the phi-2byte format of Ch 7).

Panel B: Sign matrix - near-uniform decay sigma_k ~ k^(-0.14).
         All 3584 hyperplanes are roughly equally important; the sign
         coordinate is irreducible.

The figure makes the point of section 10.2.3: magnitudes are
*compressible*, signs are *incompressible*.
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
# Panel A : phi-Zipf magnitudes
# =========================================================================
panel_label(axA, "A")
k = np.arange(1, 513)

# phi-Zipf decay
sigma_mag = PHI ** (-k / 22.0)
sigma_mag = sigma_mag / sigma_mag[0]
axA.semilogy(k, sigma_mag, color=GOLD_DARK, lw=2.0,
             label=r"magnitudes: $\sigma_k \propto \phi^{-k}$")

# k^{-ln(phi)} reference for comparison
ref_zipf = k.astype(float) ** (-LN_PHI)
ref_zipf = ref_zipf / ref_zipf[0]
axA.semilogy(k, ref_zipf, color=MUTED, lw=1.2, ls=":",
             label=r"power-law  $\sigma_k \propto k^{-\ln\phi}$")

# elbow at k = 106 (Ch 8 link)
axA.axvline(106, color=RED, ls="--", lw=1.0, alpha=0.7)
axA.text(106, 2e-2, "$k = 106$\n(discriminant\nattention rank)",
         fontsize=8.5, color=RED, ha="left", va="center", style="italic")

# 89 levels marker
axA.text(0.04, 0.05,
         r"$\sim 89$ levels capture full magnitude axis $\Rightarrow$ 8-bit storage",
         transform=axA.transAxes,
         fontsize=9.5, color=INK,
         bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=GRID, lw=0.7))

axA.set_xlim(1, 512)
axA.set_ylim(1e-8, 1.5)
axA.set_xlabel("singular-value index  $k$")
axA.set_ylabel(r"$\sigma_k / \sigma_1$  (log)")
axA.set_title("Magnitudes: $\\phi$-Zipf $\\Rightarrow$ compressible",
              fontsize=11.5)
axA.legend(loc="upper right", fontsize=9.5,
           facecolor="white", framealpha=0.95)
axA.grid(True, which="both", color=GRID, lw=0.5, alpha=0.5)
axA.set_axisbelow(True)
axA.spines["left"].set_color(INK); axA.spines["bottom"].set_color(INK)

# =========================================================================
# Panel B : near-uniform sign matrix
# =========================================================================
panel_label(axB, "B")

# sigma_k ~ k^{-0.14}  (from DC 141)
sigma_sign = k.astype(float) ** (-0.14)
sigma_sign = sigma_sign / sigma_sign[0]

axB.semilogy(k, sigma_sign, color="#3F6FB3", lw=2.0,
             label=r"signs: $\sigma_k \propto k^{-0.14}$")

# phi-Zipf reference for comparison
axB.semilogy(k, sigma_mag, color=GOLD_DARK, lw=1.2, ls=":",
             label=r"$\phi$-Zipf reference (from panel A)")

# show that decay over 3584 dims is mild
axB.axhline(sigma_sign[-1], color=INK_SOFT, ls=":", lw=0.7, alpha=0.6)
axB.annotate(r"after 512 dims:" + "\n"
             + f"$\\sigma_{{512}}/\\sigma_1 = {sigma_sign[-1]:.3f}$",
             xy=(512, sigma_sign[-1]),
             xytext=(60, 0.012),
             fontsize=9.5, color="#3F6FB3",
             arrowprops=dict(arrowstyle="->", color="#3F6FB3", lw=0.9),
             bbox=dict(boxstyle="round,pad=0.25",
                       fc="white", ec="#3F6FB3", lw=0.7))

# 3584 critical lines marker
axB.text(0.04, 0.05,
         r"all $3{,}584$ hyperplanes roughly equal $\Rightarrow$ 1 bit/sign (irreducible)",
         transform=axB.transAxes,
         fontsize=9.5, color=INK,
         bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=GRID, lw=0.7))

axB.set_xlim(1, 512)
axB.set_ylim(1e-8, 1.5)
axB.set_xlabel("singular-value index  $k$")
axB.set_ylabel(r"$\sigma_k / \sigma_1$  (log)")
axB.set_title("Signs: near-uniform $\\Rightarrow$ irreducible",
              fontsize=11.5)
axB.legend(loc="lower left", fontsize=9.5,
           facecolor="white", framealpha=0.95,
           bbox_to_anchor=(0.02, 0.18))
axB.grid(True, which="both", color=GRID, lw=0.5, alpha=0.5)
axB.set_axisbelow(True)
axB.spines["left"].set_color(INK); axB.spines["bottom"].set_color(INK)

fig.suptitle("Two complementary spectra: compressible magnitudes, irreducible signs",
             fontsize=15, fontweight="bold", y=1.02)

save_fig("fig10_2_two_spectra")
