#!/usr/bin/env python3
"""
Figure 2.2: phi-Level Binning and Geometric Context Decay
Panel A: phi-decay of context weights with distance
Panel B: phi continued fraction self-similarity tree
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PHI = (1 + np.sqrt(5)) / 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: phi-decay attention weights
d = np.arange(1, 16)
w = PHI**(-d/5)
ax1.plot(d, w, 'o-', color='#2196F3', linewidth=2.5, markersize=8)
ax1.fill_between(d, 0, w, alpha=0.15, color='#2196F3')
ax1.set_xlabel('Distance (tokens)', fontsize=11); ax1.set_ylabel(r'$\phi$-weight', fontsize=11)
ax1.set_title(r'$\phi$-Decay: $\phi^{-d/\tau}$ Context Weights', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
for level, (lo, hi) in enumerate([(1,1),(2,3),(4,7),(8,12)]):
    ax1.axvspan(lo-0.4, hi+0.4, alpha=0.1, color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'][level])
    ax1.annotate(f'level {level}', xy=((lo+hi)/2, ax1.get_ylim()[1]*0.85), fontsize=8, ha='center', color='#555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

# Panel B: phi continued fraction self-similarity tree
ax2.axis('off')
def draw_tree(ax, x, y, w, h, d, max_depth=4):
    if d >= max_depth: return
    ax.add_patch(Rectangle((x,y), w, h, fill=True, alpha=0.4,
                           facecolor=plt.cm.viridis(d/max_depth), edgecolor='#333', lw=1.5))
    label = r'$\phi = 1 + \frac{1}{\phi}$' if d==0 else (r'$1 + \frac{1}{1+\frac{1}{\phi}}$' if d==1 else '')
    ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=8, fontweight='bold' if d<2 else 'normal')
    sw = w/PHI
    draw_tree(ax, x+w-sw, y, sw, h, d+1)
    draw_tree(ax, x, y+h*0.1, w-sw, h*0.8, d+1)
draw_tree(ax2, -2, 0, 6, 3, 0)
ax2.set_xlim(-3, 5); ax2.set_ylim(-1, 4)
ax2.set_title(r'$\phi = 1 + \frac{1}{1+\frac{1}{1+\cdots}}$ - Infinite Self-Similarity', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('../fig2_2_self_similarity.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig2_2 saved")
