#!/usr/bin/env python3
"""
Figure 5.1: ENCODE = DECODE Symmetry
Panel A: The symmetry diagram — encode and decode as same phi-operation
Panel B: Critical strip information limit (sigma = 0.5)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Symmetry diagram
ax1.axis('off')
ax1.arrow(-0.5, 0.3, 3.0, 0, head_width=0.08, head_length=0.08, fc='#333', ec='#333', linewidth=1.5)
circle = plt.Circle((0.5, 0.3), 0.3, fill=True, alpha=0.3, facecolor='#2196F3', edgecolor='#1565C0', linewidth=2)
ax1.add_patch(circle)
ax1.text(0.5, 0.3, r'$\phi$', fontsize=20, fontweight='bold', ha='center', va='center', color='#1565C0')

ax1.add_patch(FancyArrowPatch((0.1, 0.55), (0.35, 0.35),
                              arrowstyle='->', color='#FF6B6B', linewidth=3,
                              connectionstyle='arc3,rad=-0.3'))
ax1.text(0.15, 0.65, 'ENCODE', fontsize=14, fontweight='bold', color='#FF6B6B', ha='center')
ax1.text(0.15, 0.58, r'$x \times \phi$', fontsize=11, color='#FF6B6B', ha='center')

ax1.add_patch(FancyArrowPatch((0.65, 0.35), (0.9, 0.55),
                              arrowstyle='->', color='#4CAF50', linewidth=3,
                              connectionstyle='arc3,rad=-0.3'))
ax1.text(0.85, 0.65, 'DECODE', fontsize=14, fontweight='bold', color='#4CAF50', ha='center')
ax1.text(0.85, 0.58, r'$x \div \phi$', fontsize=11, color='#4CAF50', ha='center')

ax1.text(-0.2, 0.3, 'Input', fontsize=12, ha='center', va='center', color='#333')
ax1.text(1.1, 0.3, 'Output', fontsize=12, ha='center', va='center', color='#333')
ax1.text(0.5, -0.1, r'$\phi \times 1/\phi = 1$', fontsize=16, fontweight='bold', ha='center')
ax1.text(0.5, -0.2, 'The transformation IS the understanding', fontsize=11, color='#555', ha='center')
ax1.set_xlim(-0.5, 1.5); ax1.set_ylim(-0.3, 0.8)

# Panel B: Critical strip
ax2.axis('off')
x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.sinc(5 * (X - 0.5)) * np.exp(-Y**2 / 4)
ax2.imshow(Z, extent=[0, 1, -2, 2], origin='lower', aspect='auto', cmap='viridis', alpha=0.7)
ax2.axvline(x=0.5, color='red', linewidth=3, linestyle='-', alpha=0.8, label=r'Critical line $\sigma=0.5$')
ax2.axvline(x=0, color='#555', linewidth=1.5, linestyle='--', alpha=0.5)
ax2.axvline(x=1, color='#555', linewidth=1.5, linestyle='--', alpha=0.5)
ax2.text(0.25, 1.8, 'Under-determined', fontsize=9, ha='center', color='#555')
ax2.text(0.75, 1.8, 'Over-constrained', fontsize=9, ha='center', color='#555')
ax2.text(0.5, -1.8, r'$\sigma=0.5$: Universal Information Limit', fontsize=11, fontweight='bold', ha='center', color='red')
ax2.set_xlabel(r'$\sigma$ (real part)', fontsize=11)
ax2.set_ylabel(r'$t$ (imaginary part)', fontsize=11)
ax2.set_title(r'Critical Strip: $\sigma = 0.5$', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')

plt.suptitle('ENCODE = DECODE: The Master Symmetry', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../fig5_1_encode_decode.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig5_1 saved")
