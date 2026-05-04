#!/usr/bin/env python3
"""
Figure 4.1: 4D Quaternion phi-Dial
Panel A: 3D quaternion axes (X:Style, Y:Perspective, Z:Depth, W: Certainty as radius)
Panel B: Control sliders showing each axis modulation
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PHI = (1 + np.sqrt(5)) / 2

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')

for (dx, dy, dz, label, c) in [(1.3,0,0,'X: Style','#FF6B6B'), (0,1.3,0,'Y: Perspective','#4ECDC4'),
                                (0,0,1.3,'Z: Depth','#45B7D1')]:
    ax.quiver(0,0,0, dx,dy,dz, color=c, arrow_length_ratio=0.15, linewidth=2.5)
    ax.text(dx*1.1, dy*1.1, dz*1.1, label, color=c, fontsize=11, fontweight='bold')

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
r = 0.7
ax.plot_wireframe(r*np.cos(u)*np.sin(v), r*np.sin(u)*np.sin(v), r*np.cos(v),
                  alpha=0.08, color='#9B59B6')
ax.text(0, 0, 1.5, 'W: Certainty (radius)', color='#9B59B6', fontsize=11, fontweight='bold', ha='center')
ax.set_title('4D Quaternion phi-Dial', fontsize=14, fontweight='bold')
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=25, azim=-45)

ax2 = fig.add_subplot(122)
ax2.axis('off')
controls = [
    ('X: Style', 'Formal <-> Casual', '#FF6B6B', 0.8),
    ('Y: Perspective', 'Subjective <-> Meta', '#4ECDC4', 0.6),
    ('Z: Depth', 'Terse <-> Elaborate', '#45B7D1', 0.4),
    ('W: Certainty', 'Definitive <-> Hedged', '#9B59B6', 0.2),
]
for i, (name, desc, color, val) in enumerate(controls):
    y = 1 - i * 0.22
    ax2.text(0.05, y, name, fontsize=12, fontweight='bold', color=color, va='center')
    ax2.text(0.05, y-0.08, desc, fontsize=9, color='#555', va='center')
    ax2.add_patch(Rectangle((0.55, y-0.04), 0.35, 0.06, fill=True, alpha=0.2,
                             facecolor=color, edgecolor='#333', linewidth=1))
    ax2.add_patch(Rectangle((0.55, y-0.04), 0.35*val, 0.06, fill=True, alpha=0.8,
                             facecolor=color, edgecolor='#333', linewidth=1))

ax2.text(0.05, -0.1, r'$q = w + xi + yj + zk$', fontsize=14, fontstyle='italic', va='center')
ax2.set_xlim(0, 1); ax2.set_ylim(-0.3, 1.1)

plt.tight_layout()
plt.savefig('../fig4_1_quaternion_dial.png', dpi=200, bbox_inches='tight')
plt.close()
print("fig4_1 saved")
