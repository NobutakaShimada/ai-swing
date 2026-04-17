"""トルク可視化グラフ: 腰・膝トルクの時系列 + スイング角 (重ね表示)"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

csv = sys.argv[1] if len(sys.argv) > 1 else "ai_swing_rotation.csv"
df = pd.read_csv(csv)
t = df['t'].values
x_deg = np.degrees(df['x'].values)
torq_body = df['torq_ai'].values
torq_knee = df['torq_ai_knee'].values

fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1.8], hspace=0.10)

# ---- 上段: スイング角 ----
ax0 = fig.add_subplot(gs[0])
ax0.plot(t, x_deg, color='#333', lw=1.2, label='swing angle x [deg]')
ax0.axhline(0, color='gray', lw=0.5, ls='--')
ax0.set_ylabel('x [deg]', fontsize=11)
ax0.set_xticklabels([])
ax0.legend(loc='upper left', fontsize=9)
ax0.grid(True, alpha=0.3)
ax0.set_xlim(t[0], t[-1])

# ---- 下段: 腰・膝トルク 重ね表示 ----
ax1 = fig.add_subplot(gs[1])

# 腰トルク: 実線 + 半透明塗りつぶし
ax1.plot(t, torq_body, color='#2255cc', lw=1.5, label='waist torque [Nm]', zorder=3)
ax1.fill_between(t, torq_body, alpha=0.25, color='#2255cc')

# 膝トルク: 実線 + 半透明塗りつぶし
ax1.plot(t, torq_knee, color='#22aa44', lw=1.5, label='knee torque [Nm]', zorder=3)
ax1.fill_between(t, torq_knee, alpha=0.25, color='#22aa44')

ax1.axhline(0, color='gray', lw=0.7)
ax1.set_ylabel('Torque [Nm]', fontsize=11)
ax1.set_xlabel('time [s]', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(t[0], t[-1])

fig.suptitle('Joint Torques — AI Swing (waist + knee, overlaid)', fontsize=13, y=0.99)
plt.savefig('torque_graph.png', dpi=150, bbox_inches='tight')
print("torque_graph.png saved")
plt.show()
