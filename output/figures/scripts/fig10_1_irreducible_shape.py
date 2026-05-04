#!/usr/bin/env python3
"""
Figure 10.1: Irreducible Shape and phi-Zipf Duality
Panel A: phi-Zipf dual fractal — encoding = ranking
Panel B: Critical line lattice showing irreducible intersection points
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: phi-Zipf dual fractal
ax1.set_xscale('log', base=2); ax1.set_yscale('log', base=2)
ranks = np.arange(1, 100)
zipf = 1.0 / ranks ** 0.8
phi_pow = PHI ** (-np.log2(ranks))
ax1.plot(ranks, zipf, 'o-', color='#2196F3', linewidth=2, markersize=3, alpha=0.7, label='Zipf: f ~ 1/r')
ax1.plot(ranks, phi_pow, 's-', color='#FF6B6B', linewidth=2, markersize=3, alpha=0.7, label=r'$\phi^{-\log_2(r)}$')
ax1.set_xlabel('Rank (log)', fontsize=11); ax1.set_ylabel('Frequency (log)', fontsize=11)
ax1.set_title(r'$\phi$-Zipf Duality: Encoding = Ranking, Inward = Outward', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Panel B: Critical line lattice
ax2.set_xlim(-0.5, 5.5); ax2.set_ylim(-0.5, 5.5)
for i in range(6):
    ax2.axhline(i, alpha=0.4, color='#FF6B6B', linestyle='--', linewidth=0.8)
    ax2.axvline(i, alpha=0.4, color='#4ECDC4', linestyle='--', linewidth=0.8)
for i in range(6):
    for j in range(6):
        ax2.plot(i, j, 'o', color='#333', markersize=4, alpha=0.8)
ax2.text(2.5, 5.3, '3584 critical lines', ha='center', fontsize=10, fontweight='bold', color='#FF6B6B')
ax2.text(5.3, 2.5, '67.9M intersection points', ha='center', fontsize=10, fontweight='bold', color='#4ECDC4', rotation=270)
ax2.set_title('The Irreducible Shape: Lattice of Critical Lines', fontsize=11, fontweight='bold')
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.1)

plt.tight_layout()
plt.savefig('../fig10_1_irreducible_shape.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig10_1 saved")
