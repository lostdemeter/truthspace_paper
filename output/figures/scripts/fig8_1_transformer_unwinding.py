#!/usr/bin/env python3
"""
Figure 8.1: Transformer Unwinding Pipeline
Pipeline schematic showing the reverse engineering of Qwen2-7B
with key discoveries annotated below.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis('off')

pipeline = [
    ('Token\nEmbed', '#95A5A6'), ('RMS\nNorm', '#3498DB'), ('Q/K/V\nProject', '#E74C3C'),
    ('Attention\nphi-Softmax', '#F39C12'), ('RoPE\nphi-Rotate', '#9B59B6'),
    ('MLP\nphi-SiLU', '#2ECC71'), ('Output\nProject', '#95A5A6'),
]
x_pos = np.linspace(0.04, 0.96, len(pipeline))
for i, (label, color) in enumerate(pipeline):
    x = x_pos[i]
    box = FancyBboxPatch((x-0.05, 0.15), 0.1, 0.55, boxstyle="round,pad=0.03",
                          facecolor=color, alpha=0.8, edgecolor='#333', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, 0.425, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

for i in range(len(pipeline)-1):
    x1 = x_pos[i] + 0.05; x2 = x_pos[i+1] - 0.05
    ax.annotate('', xy=(x2, 0.425), xytext=(x1, 0.425),
                arrowprops=dict(arrowstyle='->', color='#333', linewidth=2))

results_text = [
    "Qwen2-7B: 99.9991% correlation (Doc 129)",
    "phi-sigmoid = sigmoid EXACT (Doc 191)",
    "Rank-1 in layers 3-27 (Doc 186)",
    "Transformer = Lookup Table: 12.9x compression (Doc 187)",
    "Universal Bottleneck: phi~1.57 at layer 27 (Doc 200)",
    "74 tetromino structures cover all weights (Doc 162)",
]
y = -0.1
for t in results_text:
    ax.text(0.5, y, t, ha='center', va='center', fontsize=9, color='#555')
    y -= 0.1

ax.set_title('Qwen2-7B Reverse Engineering: Unwinding the Transformer', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.8)

plt.tight_layout()
plt.savefig('../fig8_1_transformer_unwinding.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig8_1 saved")
