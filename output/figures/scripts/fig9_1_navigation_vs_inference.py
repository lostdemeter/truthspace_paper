#!/usr/bin/env python3
"""
Figure 9.1: Navigation vs Inference
Panel A: Full O(N^2) attention fan - every token attends to all previous tokens.
Panel B: phi-lattice navigation - the SAME sign-flip vector moves every pair of
         semantically related words (king->queen, man->woman, ...), making
         navigation a geometric O(1) lookup rather than a statistical computation.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ------------------------------------------------------------------
# Panel A: Traditional inference - full attention fan is O(N^2)
# ------------------------------------------------------------------
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.text(0.5, 0.95, r'Traditional Inference: $O(N^2)$',
         fontsize=13, fontweight='bold', ha='center')

N = 7
token_y = 0.22
x_tokens = np.linspace(0.08, 0.92, N)

# Full attention arcs: every token attends to all previous tokens.
# Total edges = N(N-1)/2.  For N=7 that is 21 arcs.
edge_count = 0
for j in range(N):
    for i in range(j):
        dx = x_tokens[j] - x_tokens[i]
        arc = FancyArrowPatch(
            (x_tokens[i], token_y + 0.05),
            (x_tokens[j], token_y + 0.05),
            connectionstyle=f"arc3,rad=-{0.25 + 0.05 * (j - i)}",
            arrowstyle='-',
            color='#E74C3C', alpha=0.35, linewidth=1.0,
            mutation_scale=0,
        )
        ax1.add_patch(arc)
        edge_count += 1

# Token boxes on top of arcs so labels remain readable.
for i, x in enumerate(x_tokens):
    ax1.add_patch(FancyBboxPatch(
        (x - 0.045, token_y - 0.055), 0.09, 0.11,
        boxstyle="round,pad=0.015",
        facecolor='#FF6B6B', alpha=0.9, edgecolor='#333', linewidth=1.0,
        zorder=3,
    ))
    ax1.text(x, token_y, f'T{i+1}', ha='center', va='center',
             fontsize=9, fontweight='bold', color='#222', zorder=4)

# Key callout: N(N-1)/2 attention edges.
ax1.text(0.5, 0.85,
         f'N = {N} tokens  -->  {edge_count} attention edges  =  N(N-1)/2',
         ha='center', fontsize=10, color='#222')
ax1.text(0.5, 0.05,
         'Every token attends to every previous token.\nCost grows quadratically with sequence length.',
         ha='center', fontsize=9.5, color='#555')

# ------------------------------------------------------------------
# Panel B: phi-lattice navigation - same sign-flip vector for every pair
# ------------------------------------------------------------------
ax2.set_xlim(-0.5, 10.5); ax2.set_ylim(-0.2, 6.2)
ax2.axis('off')
ax2.text(5, 5.9, r'$\phi$-Lattice Navigation: $O(1)$ per step',
         fontsize=13, fontweight='bold', ha='center')

# Faint lattice grid suggesting the crystalline phi-structure
for gx in np.arange(0, 10.1, 0.5):
    ax2.axvline(gx, color='#BDC3C7', alpha=0.25, linewidth=0.5, zorder=0)
for gy in np.arange(0, 5.1, 0.5):
    ax2.axhline(gy, xmin=0.04, xmax=0.96,
                color='#BDC3C7', alpha=0.25, linewidth=0.5, zorder=0)

# Word positions.  Gender pairs on the left share a horizontal flip;
# temperature pairs on the right share a vertical flip.  The two
# relationships live on orthogonal crystal planes.
words = {
    'king':   (0.6, 4.2),  'queen':  (3.6, 4.2),
    'man':    (0.6, 2.7),  'woman':  (3.6, 2.7),
    'uncle':  (0.6, 1.2),  'aunt':   (3.6, 1.2),
    'hot':    (6.2, 1.2),  'cold':   (6.2, 4.2),
    'warm':   (9.0, 1.2),  'cool':   (9.0, 4.2),
}

for w, (x, y) in words.items():
    ax2.scatter(x, y, s=55, color='#2C3E50', zorder=4)
    ax2.text(x, y - 0.35, w, ha='center', va='top',
             fontsize=9.5, color='#222', zorder=4)

# Gender flip arrows - all identical vectors (horizontal, length 3).
gender_pairs = [('king', 'queen'), ('man', 'woman'), ('uncle', 'aunt')]
for a, b in gender_pairs:
    xa, ya = words[a]; xb, yb = words[b]
    ax2.annotate('', xy=(xb - 0.22, yb), xytext=(xa + 0.22, ya),
                 arrowprops=dict(arrowstyle='->', color='#E74C3C',
                                 lw=2.2, shrinkA=0, shrinkB=0),
                 zorder=3)

# Temperature flip arrows - all identical vectors (vertical, length 3).
temp_pairs = [('hot', 'cold'), ('warm', 'cool')]
for a, b in temp_pairs:
    xa, ya = words[a]; xb, yb = words[b]
    ax2.annotate('', xy=(xb, yb - 0.22), xytext=(xa, ya + 0.22),
                 arrowprops=dict(arrowstyle='->', color='#9B59B6',
                                 lw=2.2, shrinkA=0, shrinkB=0),
                 zorder=3)

# Relationship labels
ax2.text(2.1, 4.75, 'gender flip', color='#E74C3C',
         fontsize=10, fontweight='bold', ha='center')
ax2.text(2.1, 4.5, '(same sign-bits flip for every pair)',
         color='#E74C3C', fontsize=8.5, ha='center')

ax2.text(7.6, 4.9, 'temperature flip', color='#9B59B6',
         fontsize=10, fontweight='bold', ha='center')
ax2.text(7.6, 4.65, '(orthogonal crystal plane)',
         color='#9B59B6', fontsize=8.5, ha='center')

ax2.text(5, 0.05,
         'Each relationship is a fixed direction in the lattice.\n'
         'Traverse it once -> apply to any word: 960x compression, $O(1)$ lookup.',
         ha='center', fontsize=9.5, color='#555')

plt.suptitle('Navigation Replaces Inference', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../fig9_1_navigation_vs_inference.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig9_1 saved")
