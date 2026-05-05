#!/usr/bin/env python3
"""
Figure 6.1 - Gear-chain architecture and the emergent 5-step lifecycle.

Panel A: Input -> Gear1 -> Gear2 -> Gear3 -> Output, with the running
         quaternion product Q_total accumulating across the chain.
Panel B: The recurring Structure -> Bootstrap -> Match -> Compose -> Learn
         lifecycle that emerges across the codebase.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, RED, TEAL, VIOLET,
                      GRID, MUTED)

apply_style()

fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 6),
                               gridspec_kw={"height_ratios": [1.4, 1]})

# =========================================================================
# Panel A : the gear chain
# =========================================================================
panel_label(axA, "A", y=1.06)
axA.set_xlim(0, 14); axA.set_ylim(0, 4.0)
axA.set_aspect("auto"); axA.axis("off")

stages = [
    ("Input\nstate",  MUTED,  False),
    ("Gear 1\nAction", RED,    True),
    ("Gear 2\nRole",   TEAL,   True),
    ("Gear 3\nOutput", VIOLET, True),
    ("Final\nstate",   MUTED,  False),
]

w = 2.2; h = 1.6; gap = 0.45
x = 0.2
positions = []
for label, color, is_gear in stages:
    fc = color if is_gear else "#E1E5EC"
    txt_color = "white" if is_gear else INK
    axA.add_patch(FancyBboxPatch((x, 1.4), w, h,
                                 boxstyle="round,pad=0.04,rounding_size=0.18",
                                 facecolor=fc, edgecolor=INK, lw=1.2,
                                 alpha=0.95, zorder=3))
    axA.text(x + w / 2, 1.4 + h / 2, label,
             ha="center", va="center", color=txt_color,
             fontsize=11.5, fontweight="bold", zorder=4)
    positions.append(x + w / 2)
    x += w + gap

# arrows between stages
for i in range(len(stages) - 1):
    a = FancyArrowPatch((positions[i] + w / 2 - 0.05, 2.2),
                        (positions[i + 1] - w / 2 + 0.05, 2.2),
                        arrowstyle="-|>", color=INK, lw=1.6,
                        mutation_scale=14)
    axA.add_patch(a)

# quaternion accumulation underneath
qs = [r"$Q_1$", r"$Q_1\!\cdot\!Q_2$", r"$Q_1\!\cdot\!Q_2\!\cdot\!Q_3$"]
for i, q in enumerate(qs):
    cx = (positions[i + 1] + positions[i + 2]) / 2
    axA.text(cx, 1.05, q, ha="center", va="top",
             fontsize=10, color=INK_SOFT)

axA.annotate("", xy=(positions[-1] + 0.6, 0.45),
             xytext=(positions[1] - 0.6, 0.45),
             arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.8,
                             linestyle="--"))
axA.text((positions[1] + positions[-1]) / 2, 0.18,
         r"$Q_{\mathrm{total}} \;=\; Q_1 \cdot Q_2 \cdots Q_n$"
         "    -    quaternion path through the chain",
         ha="center", va="top", fontsize=11, color=GOLD)

axA.set_title("Gear chain  -  state flows, quaternion accumulates",
              fontsize=12, color=INK, pad=4, loc="left")

# =========================================================================
# Panel B : 5-step lifecycle
# =========================================================================
panel_label(axB, "B", y=1.08)
axB.set_xlim(0, 14); axB.set_ylim(0, 1.6)
axB.axis("off")

steps = [
    ("1. STRUCTURE", "define the space",       "#3F6FB3"),
    ("2. BOOTSTRAP", "seed with examples",     "#D67D2C"),
    ("3. MATCH",     "find nearest structure", "#D4B019"),
    ("4. COMPOSE",   "adapt to request",       "#3FA67E"),
    ("5. LEARN",     "self-improve",           VIOLET),
]
sw = 2.4; sh = 1.0; sgap = 0.2
x = 0.4
for i, (head, sub, color) in enumerate(steps):
    axB.add_patch(FancyBboxPatch((x, 0.3), sw, sh,
                                 boxstyle="round,pad=0.04,rounding_size=0.14",
                                 facecolor=color, edgecolor=INK,
                                 lw=1.0, alpha=0.95))
    axB.text(x + sw / 2, 0.3 + sh * 0.62, head,
             ha="center", va="center", color="white",
             fontsize=10.5, fontweight="bold")
    axB.text(x + sw / 2, 0.3 + sh * 0.28, sub,
             ha="center", va="center", color="white",
             fontsize=9, style="italic")
    if i < len(steps) - 1:
        a = FancyArrowPatch((x + sw + 0.02, 0.8),
                            (x + sw + sgap - 0.02, 0.8),
                            arrowstyle="-|>", color=INK,
                            lw=1.3, mutation_scale=12)
        axB.add_patch(a)
    x += sw + sgap

axB.set_title("The emergent 5-step lifecycle",
              fontsize=12, color=INK, pad=4, loc="left")

fig.suptitle("Gear architecture", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("fig6_1_gear_chain")
