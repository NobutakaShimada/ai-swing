"""検証用: 外力なしの自由振動アニメーション
初期角度を与えてブランコ+胴体+頭部の時間発展を見る
"""
import math
import sys
sys.path.insert(0, '.')
from swing_physics import Swing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 初期条件
x_init = 30 * math.pi/180    # ブランコ 30度
phi_init = 15 * math.pi/180  # 胴体 15度
alpha_init = -20 * math.pi/180  # 頭部 -20度(前傾)

# Swingオブジェクト作成 (VR=0: Real)
sw = Swing(eps=1, VR=0, x=0, coef_Hooke=0)
sw.reset()

# 初期値を上書き
sw.x = x_init
sw.z = 0
sw.phi = phi_init
sw.d_phi = 0
sw.alpha = alpha_init
sw.d_alpha = 0

# シミュレーション (10秒間)
dt_sim = 0.001  # 1ms
t_end = 10.0
n_steps = int(t_end / dt_sim)
save_every = 10  # 10msごとに記録

ts, xs, phis, alphas = [], [], [], []

for i in range(n_steps):
    if i % save_every == 0:
        ts.append(sw.t)
        xs.append(sw.x)
        phis.append(sw.phi)
        alphas.append(sw.alpha)
    sw.rk4_step(0.0, 0.0)  # 外力なし
    sw.t += dt_sim

print(f"シミュレーション完了: {len(ts)} フレーム")
print(f"x: {min(xs)*180/math.pi:.1f} ~ {max(xs)*180/math.pi:.1f} deg")
print(f"phi: {min(phis)*180/math.pi:.1f} ~ {max(phis)*180/math.pi:.1f} deg")
print(f"alpha: {min(alphas)*180/math.pi:.1f} ~ {max(alphas)*180/math.pi:.1f} deg")

# アニメーション作成
L = sw.L
L1t = sw.L1t
L1h = sw.L1h

fig, (ax_anim, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))

# 右側: 時系列グラフ
DEG = 180/math.pi
ax_plot.plot(ts, [v*DEG for v in xs], 'r-', label='x (swing)')
ax_plot.plot(ts, [v*DEG for v in phis], 'b-', label='phi (torso)')
ax_plot.plot(ts, [v*DEG for v in alphas], 'g-', label='alpha (head)')
ax_plot.set_xlabel('sec')
ax_plot.set_ylabel('deg')
ax_plot.legend()
ax_plot.grid(True)
ax_plot.set_title('Angle time series')
time_line = ax_plot.axvline(0, color='k', linestyle='--', alpha=0.5)

# 左側: アニメーション
ax_anim.set_xlim(-3, 3)
ax_anim.set_ylim(-3, 1)
ax_anim.set_aspect('equal')
ax_anim.grid(True)
ax_anim.set_title('Free swing (no external torque)')

rope_line, = ax_anim.plot([], [], 'k-', linewidth=2)
torso_line, = ax_anim.plot([], [], 'b-', linewidth=3)
head_line, = ax_anim.plot([], [], 'g-', linewidth=3)
head_dot, = ax_anim.plot([], [], 'go', markersize=10)
pivot_dot, = ax_anim.plot([0], [0], 'ko', markersize=8)
time_text = ax_anim.text(-2.8, 0.8, '', fontsize=12)

def animate(frame):
    x_val = xs[frame]
    phi_val = phis[frame]
    alpha_val = alphas[frame]

    # 支点 (0,0)
    # 座面
    seat_x = L * math.sin(x_val)
    seat_y = -L * math.cos(x_val)

    # 胴体先端 (首)
    neck_x = seat_x - L1t * math.sin(phi_val)
    neck_y = seat_y + L1t * math.cos(phi_val)

    # 頭頂
    head_x = neck_x - L1h * math.sin(alpha_val)
    head_y = neck_y + L1h * math.cos(alpha_val)

    rope_line.set_data([0, seat_x], [0, seat_y])
    torso_line.set_data([seat_x, neck_x], [seat_y, neck_y])
    head_line.set_data([neck_x, head_x], [neck_y, head_y])
    head_dot.set_data([head_x], [head_y])
    time_text.set_text(f't = {ts[frame]:.2f} s')
    time_line.set_xdata([ts[frame]])

    return rope_line, torso_line, head_line, head_dot, time_text, time_line

ani = animation.FuncAnimation(fig, animate, frames=len(ts), interval=10, blit=True)

# MP4保存
print("動画を保存中...")
ani.save('test_freeswing.mp4', writer='ffmpeg', fps=100, dpi=100)
print("test_freeswing.mp4 に保存しました")

# 静止画も保存
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(ts, [v*DEG for v in xs], 'r-', label='x (swing)')
ax2.plot(ts, [v*DEG for v in phis], 'b-', label='phi (torso)')
ax2.plot(ts, [v*DEG for v in alphas], 'g-', label='alpha (head)')
ax2.set_xlabel('sec')
ax2.set_ylabel('deg')
ax2.legend()
ax2.grid(True)
ax2.set_title('Free swing - angle time series')
fig2.savefig('test_freeswing.png', dpi=150)
print("test_freeswing.png に保存しました")
