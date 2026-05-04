#!/usr/bin/env python3
"""
Figure 6.1: Gear Chain Architecture
Top: Gear pipeline with quaternion accumulation
Bottom: 5-step emergent pattern lifecycle
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')

stages = [('Input\nState', '#95A5A6'), ('Gear 1\nAction', '#FF6B6B'), ('Gear 2\nRole', '#4ECDC4'),
          ('Gear 3\nOutput', '#45B7D1'), ('Final\nState', '#95A5A6')]
y_pos = 0.5
x_positions = [0.08, 0.25, 0.42, 0.59, 0.78]
for i, (label, color) in enumerate(stages):
    x = x_positions[i]
    box = FancyBboxPatch((x-0.06, y_pos-0.3), 0.12, 0.6,
                          boxstyle="round,pad=0.04", facecolor=color, alpha=0.7,
                          edgecolor='#333', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y_pos, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

for i in range(len(stages)-1):
    x1 = x_positions[i] + 0.06; x2 = x_positions[i+1] - 0.06
    ax.annotate('', xy=(x2, y_pos), xytext=(x1, y_pos),
                arrowprops=dict(arrowstyle='->', color='#333', linewidth=2.5))

ax.annotate('', xy=(0.78, 0.1), xytext=(0.08, 0.1),
            arrowprops=dict(arrowstyle='->', color='#9B59B6', linewidth=2, linestyle='--'))
ax.text(0.43, 0.06, r'$Q_{total} = Q_1 \times Q_2 \times \cdots \times Q_n$',
        ha='center', va='center', fontsize=11, color='#9B59B6')

emergent_steps = ['1.STRUCTURE', '2.BOOTSTRAP', '3.MATCH', '4.COMPOSE', '5.LEARN']
emergent_colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB']
for i, (step, c) in enumerate(zip(emergent_steps, emergent_colors)):
    x = 0.08 + i * 0.185
    rect = Rectangle((x, -0.35), 0.16, 0.22, fill=True, alpha=0.85,
                     facecolor=c, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+0.08, -0.24, step, ha='center', va='center', fontsize=7.5, fontweight='bold', color='white')
    if i < 4:
        ax.annotate('', xy=(x+0.185, -0.24), xytext=(x+0.16, -0.24),
                    arrowprops=dict(arrowstyle='->', color='#333', linewidth=1.5))

ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.9)
ax.set_title('Gear Chain Architecture + Emergent Pattern', fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('../fig6_1_gear_chain.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig6_1 saved")
