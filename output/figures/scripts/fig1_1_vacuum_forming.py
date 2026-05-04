#!/usr/bin/env python3
"""
Figure 1.1: Vacuum Forming Hypothesis
Surface vs. interior geometric structure contour plot.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2

fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z_interior = np.sin(X * np.log(PHI)) * np.cos(Y * np.log(PHI)) * 0.5 + 0.5
Z_surface = gaussian_filter(Z_interior, sigma=1.5)
contour_interior = ax.contour(X, Y, Z_interior, levels=8, colors='#FF6B6B', linewidths=1.5, alpha=0.8)
contour_surface = ax.contour(X, Y, Z_surface, levels=6, colors='#2196F3', linewidths=2.5, alpha=0.9)
sample_x = np.random.uniform(-2.5, 2.5, 30)
sample_y = np.random.uniform(-2.5, 2.5, 30)
ax.scatter(sample_x, sample_y, c='#FF9800', s=30, alpha=0.7, label='Training Data (surface samples)', zorder=5)
ax.legend(fontsize=10)
ax.set_xlabel('Semantic Axis 1', fontsize=11)
ax.set_ylabel('Semantic Axis 2', fontsize=11)
ax.set_title('Vacuum Forming Hypothesis: Surface vs. Interior Geometric Structure', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('../fig1_1_vacuum_forming.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig1_1 saved")
