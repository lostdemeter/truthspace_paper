#!/usr/bin/env python3
"""
Figure 7.1: phi-Lattice Coordinate System
Panel A: 2D projection of phi-power grid with intersection points
Panel B: Weight count by phi-level showing clustering at discrete levels
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PHI = (1 + np.sqrt(5)) / 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: phi-lattice grid
ax1.set_xlim(-1, 6); ax1.set_ylim(-1, 6)
for level in range(5):
    val = PHI ** level
    ax1.axhline(y=val, alpha=0.2, color='#2196F3', linestyle='--')
    ax1.axvline(x=val, alpha=0.2, color='#FF6B6B', linestyle='--')
    ax1.plot(val, 0, 'o', color='#FF6B6B', markersize=8)
    ax1.plot(0, val, 'o', color='#2196F3', markersize=8)
    ax1.text(val, -0.3, f'phi^{level}={val:.2f}', fontsize=7.5, ha='center', color='#FF6B6B')
for i in range(5):
    for j in range(5):
        x = PHI ** i; y = PHI ** j
        ax1.plot(x, y, 'o', color='#333', markersize=4, alpha=0.6)
        if i == j:
            ax1.text(x+0.08, y+0.08, f'({i},{j})', fontsize=6, alpha=0.5)
ax1.set_xlabel('phi X-axis', fontsize=11); ax1.set_ylabel('phi Y-axis', fontsize=11)
ax1.set_title('phi-Lattice: Discrete phi-Power Grid', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.1); ax1.set_aspect('equal')

# Panel B: Tetromino weight distribution
levels = range(-6, 7)
freq = [80, 250, 800, 2000, 3500, 1000, 4000, 1000, 3500, 2000, 800, 250, 80]
bars = ax2.bar(levels, freq, color='#4ECDC4', alpha=0.8, edgecolor='#333', linewidth=1.2)
ax2.axhline(y=74, color='red', linestyle='--', alpha=0.7, label='74 unique tetrominoes')
ax2.set_xlabel('phi-Level', fontsize=11); ax2.set_ylabel('Weight Count', fontsize=11)
ax2.set_title('Weights Cluster at phi-Levels (74 Unique)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../fig7_1_phi_lattice.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig7_1 saved")
