#!/usr/bin/env python3
"""
Figure B.2 - The empirical zero spectrum of Qwen2.5-7B (DC 296).

The same three-stage pipeline (Compressor sweep -> Processor bisection
-> Targeter semantic analysis) that locates non-trivial zeros of the
Riemann zeta function locates 21 *non-trivial zeros* in the transformer's
logit gap when one epsilon-group of the gate projection is phase-shifted
by delta.

The 21 zeros span 3 prompts x 5 layers (with Einstein at L23 producing
no sign changes - a counterexample of unconditional commitment).

Each marker is one zero: x = layer index, y = delta*.  Colour encodes
prompt; marker shape encodes the semantic outcome at the zero.

Source data: phi_collective_zero_hunt_results.txt (Phase 10z, summary
table at lines 504-526).  Precision +/- 2.27e-13 from Stage 2 bisection.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import (apply_style, save_fig, panel_label,
                      INK, INK_SOFT, GOLD, GOLD_DARK, GOLD_SOFT,
                      RED, TEAL, VIOLET, MUTED, GRID, PHI)

apply_style()

# (prompt, layer, delta_star, baseline -> after, outcome)
# outcome categories:
#   HOLD     - baseline top1 maintained
#   REVEAL   - baseline was a placeholder; perturbation reveals the
#              correct answer (Japan ___ -> Tokyo)
#   DESTROY  - baseline destroyed, prediction is wrong / junk
#   MARGINAL - tangent zero; gap touches 0 but does not flip cleanly
ZEROS = [
    # France (baseline = "Paris")
    ("France",   5,  4.0039, "Paris",  "Paris",  "HOLD"),
    ("France",   5,  5.0000, "Paris",  "?",      "MARGINAL"),
    ("France",   5,  5.4764, "Paris",  "a",      "DESTROY"),
    ("France",  15,  6.9168, "Paris",  "Paris",  "HOLD"),
    ("France",  22,  4.6445, "Paris",  "a",      "DESTROY"),
    ("France",  23,  6.2811, "Paris",  "Paris",  "HOLD"),
    ("France",  27,  3.9902, "Paris",  "a",      "DESTROY"),
    # Japan (baseline = "______")
    ("Japan",    5, -1.5000, "______", "?",      "MARGINAL"),
    ("Japan",    5,  5.2301, "______", "Tokyo",  "REVEAL"),
    ("Japan",   15,  2.4336, "______", "Tokyo",  "REVEAL"),
    ("Japan",   15,  6.5000, "______", "Tokyo",  "REVEAL"),
    ("Japan",   15,  7.1647, "______", ".",      "DESTROY"),
    ("Japan",   22,  3.2109, "______", "Tokyo",  "REVEAL"),
    ("Japan",   22,  4.0000, "______", "?",      "MARGINAL"),
    ("Japan",   22,  4.3368, "______", "a",      "DESTROY"),
    ("Japan",   23,  4.6101, "______", "Tokyo",  "REVEAL"),
    ("Japan",   27,  3.9758, "______", "Tokyo",  "REVEAL"),
    # Einstein (baseline = "rel")
    ("Einstein", 5,  5.7396, "rel",    "the",    "DESTROY"),
    ("Einstein",15,  6.0349, "rel",    "which",  "DESTROY"),
    ("Einstein",22,  5.1454, "rel",    "the",    "DESTROY"),
    ("Einstein",27,  2.8929, "rel",    "rel",    "HOLD"),
]

# colour by prompt
PROMPT_COLOR = {
    "France":   GOLD_DARK,
    "Japan":    TEAL,
    "Einstein": VIOLET,
}

# marker by outcome
OUTCOME_MARKER = {
    "HOLD":     "o",
    "REVEAL":   "*",
    "DESTROY":  "X",
    "MARGINAL": "^",
}
OUTCOME_SIZE = {
    "HOLD":     85,
    "REVEAL":   210,
    "DESTROY":  120,
    "MARGINAL": 90,
}
OUTCOME_LABEL = {
    "HOLD":     "HOLD  (baseline maintained)",
    "REVEAL":   "REVEAL  (correct answer surfaces)",
    "DESTROY":  "DESTROY  (baseline -> junk token)",
    "MARGINAL": "MARGINAL  (tangent zero, no flip)",
}

LAYERS = [5, 15, 22, 23, 27]
LAYER_X = {l: i for i, l in enumerate(LAYERS)}      # equally-spaced columns

fig, ax = plt.subplots(1, 1, figsize=(12, 6.0))

# vertical guides at each layer column
for l, x in LAYER_X.items():
    ax.axvline(x, color=GRID, lw=0.7, alpha=0.6, zorder=0)

# Group zeros per layer, then jitter horizontally within the column
# so points do not overlap.
rng = np.random.default_rng(7)
buckets = {l: [] for l in LAYERS}
for z in ZEROS:
    buckets[z[1]].append(z)

for layer, items in buckets.items():
    n = len(items)
    if n == 1:
        offsets = np.array([0.0])
    else:
        # spread evenly across [-0.28, +0.28] within the column
        offsets = np.linspace(-0.28, 0.28, n)
    base_x = LAYER_X[layer]
    for off, (prompt, _, dstar, _, _, outcome) in zip(offsets, items):
        ax.scatter(base_x + off, dstar,
                   marker=OUTCOME_MARKER[outcome],
                   s=OUTCOME_SIZE[outcome],
                   color=PROMPT_COLOR[prompt],
                   edgecolor=INK,
                   linewidth=0.9,
                   zorder=4)

# horizontal zero gap reference (delta = 0, the baseline gate)
ax.axhline(0, color=INK, lw=0.7, alpha=0.55, zorder=1)
ax.text(-0.45, 0.0, r"$\delta = 0$",
        ha="left", va="center", fontsize=8.5,
        color=INK_SOFT, style="italic")

# annotate semantically meaningful zeros
ANNOTS = [
    # (layer, delta, dx, dy, text, color)
    (15,  2.4336, +0.55, -1.10,
     "Japan L15 $\\delta\\!=\\!2.43$\n____ $\\to$ Tokyo (correct)",
     TEAL),
    (27,  2.8929, -0.85, -1.40,
     "Einstein L27 $\\delta\\!=\\!2.89$\nrel $\\to$ rel (committed)",
     VIOLET),
    (22,  4.6445, +0.55, +1.40,
     "France L22 $\\delta\\!=\\!4.64$\nParis $\\to$ a (destroy)",
     GOLD_DARK),
]
for (layer, dstar, dx, dy, txt, color) in ANNOTS:
    base_x = LAYER_X[layer]
    ax.annotate(txt, xy=(base_x, dstar),
                xytext=(base_x + dx, dstar + dy),
                fontsize=8.5, color=color, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.20",
                          fc="white", ec=color, lw=0.7),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.7,
                                connectionstyle="arc3,rad=0.18"),
                zorder=6)

# layer L23 callout: Einstein has *no* zero here ("unconditional commit")
ax.text(LAYER_X[23], -1.95,
        "Einstein L23: $\\bf{no}$ zero in $[-5, +12]$\n(unconditional commitment)",
        ha="center", va="bottom",
        fontsize=8.5, color=INK_SOFT, style="italic",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec=INK_SOFT, lw=0.7))

# axes
ax.set_xticks(list(LAYER_X.values()))
ax.set_xticklabels([f"L{l}" for l in LAYERS])
ax.set_xlim(-0.55, len(LAYERS) - 0.40)
ax.set_ylim(-2.2, 8.5)
ax.set_xlabel(r"layer index  $\ell$  (Qwen2.5-7B, $\varepsilon$-group sweep)")
ax.set_ylabel(r"zero location  $\delta^{*}$  ($\varphi$-power phase shift)")
ax.set_title(r"Twenty-one non-trivial zeros of the transformer logit gap",
             fontsize=13)
ax.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["left"].set_color(INK); ax.spines["bottom"].set_color(INK)

# secondary y-axis showing phi^delta scaling
ax2 = ax.twinx()
ax2.set_ylim(*ax.get_ylim())
secondary_ticks = [-2, 0, 2, 4, 6, 8]
ax2.set_yticks(secondary_ticks)
ax2.set_yticklabels([f"{PHI**d:.2f}" + r"$\times$" for d in secondary_ticks])
ax2.set_ylabel(r"scaling factor  $\varphi^{\delta^{*}}$",
               color=INK_SOFT, fontsize=9.5)
ax2.tick_params(axis="y", colors=INK_SOFT, labelsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(INK_SOFT)
ax2.spines["left"].set_visible(False)

# legend (two-part: outcome by shape + prompt by colour) outside the axes
outcome_handles = [
    Line2D([0], [0], marker=OUTCOME_MARKER[k], color="w",
           markerfacecolor=MUTED, markeredgecolor=INK, markersize=11,
           label=OUTCOME_LABEL[k])
    for k in ["HOLD", "REVEAL", "DESTROY", "MARGINAL"]
]
prompt_handles = [
    Line2D([0], [0], marker="s", color="w",
           markerfacecolor=c, markeredgecolor=INK, markersize=10,
           label=p)
    for p, c in PROMPT_COLOR.items()
]
leg1 = ax.legend(handles=outcome_handles, loc="lower left",
                 fontsize=8.5, facecolor="white", framealpha=0.95,
                 title=r"outcome (shape)",
                 title_fontsize=9.0,
                 bbox_to_anchor=(0.005, 0.015), borderpad=0.7)
leg1.get_title().set_fontweight("bold")
ax.add_artist(leg1)
leg2 = ax.legend(handles=prompt_handles, loc="upper right",
                 fontsize=8.5, facecolor="white", framealpha=0.95,
                 title=r"prompt (colour)",
                 title_fontsize=9.0,
                 bbox_to_anchor=(0.985, 0.985), borderpad=0.7)
leg2.get_title().set_fontweight("bold")

# summary footer with counts
counts = {"HOLD": 0, "REVEAL": 0, "DESTROY": 0, "MARGINAL": 0}
for z in ZEROS:
    counts[z[5]] += 1
ax.text(0.5, -0.165,
        ("$N = 21$  zeros in total  "
         f"(HOLD {counts['HOLD']}, "
         f"REVEAL {counts['REVEAL']}, "
         f"DESTROY {counts['DESTROY']}, "
         f"MARGINAL {counts['MARGINAL']})"
         "  $\\cdot$  precision $\\pm 2.27 \\times 10^{-13}$  "
         "$\\cdot$  source: DC 296"),
        transform=ax.transAxes,
        fontsize=9, color=INK_SOFT, style="italic",
        ha="center", va="top")

save_fig("figB_2_empirical_zeros")
