#!/usr/bin/env python3
"""
Figure 2.1: phi Self-Similarity — 3 panels
Panel A: phi = 1 + 1/phi decomposition
Panel B: Fibonacci squares with golden spiral
Panel C: phi^n scaling
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle

PHI = (1 + np.sqrt(5)) / 2

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                          gridspec_kw={'width_ratios': [1.2, 1.8, 1.2]})

# Panel A: phi decomposition
ax = axes[0]
ax.barh([0], [1], height=0.6, color='#2196F3', alpha=0.8, label='1')
ax.barh([0], [1/PHI], left=1, height=0.6, color='#FF9800', alpha=0.8, label=r'$1/\phi$')
ax.set_yticks([]); ax.set_xlim(0, 2.2)
ax.set_title(r'$\phi = 1 + 1/\phi$', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.spines['bottom'].set_position(('data', -0.35))

# Panel B: Fibonacci squares + golden spiral
ax = axes[1]
fib = [1, 1, 2, 3, 5, 8, 13, 21]
colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#98D8C8','#F7DC6F']
# Correct CCW Fibonacci tiling (bottom-left corner of each square):
# sq0 (0,0) 1x1; sq1 right of sq0; sq2 above; sq3 left; sq4 below;
# sq5 right; sq6 above; sq7 left. Overall bbox: (-24,-5) to (10,16).
fib_positions = [(0, 0), (1, 0), (0, 1), (-3, 0), (-3, -5), (2, -5), (-3, 3), (-24, -5)]
for i, (x, y) in enumerate(fib_positions):
    side = fib[i]
    rect = Rectangle((x, y), side, side, fill=True, alpha=0.55,
                     facecolor=colors[i], edgecolor='#333', linewidth=1.2)
    ax.add_patch(rect)
# Fibonacci spiral: one quarter-circle arc per square.
# (center_x, center_y, radius, theta1, theta2)
arc_params = [
    (1, 1, 1, 180, 270),    # sq0
    (1, 1, 1, 270, 360),    # sq1
    (0, 1, 2, 0, 90),       # sq2
    (0, 0, 3, 90, 180),     # sq3
    (2, 0, 5, 180, 270),    # sq4
    (2, 3, 8, 270, 360),    # sq5
    (-3, 3, 13, 0, 90),     # sq6
    (-3, -5, 21, 90, 180),  # sq7
]
for cx, cy, r, t1, t2 in arc_params:
    arc = Arc((cx, cy), 2*r, 2*r, angle=0, theta1=t1, theta2=t2,
              edgecolor='#2C3E50', linewidth=2.2)
    ax.add_patch(arc)
ax.set_xlim(-25, 11); ax.set_ylim(-6.5, 17)
ax.set_aspect('equal')
ax.set_title(r'Fibonacci Spiral $\to \phi$', fontsize=13, fontweight='bold')
ax.axis('off')

# Panel C: phi^n scaling
ax = axes[2]
n_values = np.arange(-4, 5)
phi_powers = PHI**n_values
bar_colors = ['#FF9800' if n < 0 else '#2196F3' for n in n_values]
ax.bar(n_values, phi_powers, color=bar_colors, alpha=0.8, edgecolor='#333', linewidth=1.2)
ax.set_yscale('log')
ax.set_xlabel('n', fontsize=11); ax.set_ylabel(r'$\phi^n$', fontsize=12)
ax.set_title(r'$\phi^n$ Scaling', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
for n, v in zip(n_values, phi_powers):
    ax.text(n, v*1.4, f'{v:.3f}', ha='center', fontsize=7, rotation=45)
ax.set_ylim(0.1, 12)

plt.tight_layout()
plt.savefig('../fig2_1_phi_spiral.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig2_1 saved")
