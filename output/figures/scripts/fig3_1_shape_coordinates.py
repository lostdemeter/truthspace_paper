#!/usr/bin/env python3
"""
Figure 3.1: Geometric Model Hypothesis
Panel A: 3D phi-structured weight manifold with signal (69%) and noise (31%)
Panel B: Training fidelity curve — shape discovery over time
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')

np.random.seed(42)
n = 300
t = np.random.uniform(0, 2*np.pi, n)
s = np.random.uniform(0, np.pi, n)
X = np.cos(t)*np.sin(s)*(0.5+0.5*np.sin(s*np.log(PHI)))
Y = np.sin(t)*np.sin(s)*(0.5+0.5*np.sin(s*np.log(PHI)))
Z = np.cos(s)*(0.5+0.3*np.sin(t*np.log(PHI)))
noise_mask = np.random.choice([True, False], n, p=[0.31, 0.69])
Xn, Yn, Zn = X.copy(), Y.copy(), Z.copy()
Xn[noise_mask] += np.random.normal(0, 0.15, noise_mask.sum())
Yn[noise_mask] += np.random.normal(0, 0.15, noise_mask.sum())
Zn[noise_mask] += np.random.normal(0, 0.15, noise_mask.sum())
sig = ~noise_mask
ax1.scatter(X[sig], Y[sig], Z[sig], c=t[sig], cmap='viridis', alpha=0.6, s=15, label='Signal (69%)')
ax1.scatter(Xn[noise_mask], Yn[noise_mask], Zn[noise_mask], c='red', alpha=0.2, s=10, label='Noise (31%)')
ax1.set_xlabel('dim 1', fontsize=9); ax1.set_ylabel('dim 2', fontsize=9); ax1.set_zlabel('dim 3', fontsize=9)
ax1.set_title('Weights as phi-Coordinates of a Shape', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')

ax2 = fig.add_subplot(122)
epochs = np.array([0, 1, 5, 20, 50, 100, 200])
fidelity = 1 - np.exp(-epochs/30)
noise = 0.3 * np.exp(-epochs/20) + 0.02
ax2.plot(epochs, fidelity, 'o-', color='#2196F3', linewidth=2.5, markersize=8)
ax2.fill_between(epochs, 0, fidelity, alpha=0.15, color='#2196F3')
ax2.fill_between(epochs, fidelity-noise, fidelity+noise, alpha=0.2, color='#FF6B6B')
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='phi-shape (discovered)')
ax2.set_xlabel('Training Steps', fontsize=11); ax2.set_ylabel('Shape Fidelity', fontsize=11)
ax2.set_title('Training Discovers the phi-Shape, It Does Not Create It', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.05, 1.15)

plt.tight_layout()
plt.savefig('../fig3_1_shape_coordinates.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig3_1 saved")
