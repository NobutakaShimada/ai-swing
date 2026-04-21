"""README用の3種類の画像を生成:
1. max_amplitude.gif — 最大振幅付近の3周期ループGIF
2. hero_cold_start.png — 4時点のスナップショット (ゴースト重畳)
3. one_period_strip.png — 最大振幅時の1周期9コマストリップ
"""
import pygame
import pandas as pd
import numpy as np
import os
import math
import subprocess
from PIL import Image
import io

# ---- CSV読み込み ----
obs = pd.read_csv("ai_swing_rotation.csv")
t_data = obs['t'].values
x_data = obs['x'].values
phi_data = obs['phi'].values
d_phi_data = obs['d_phi'].values
psi_data = obs['psi'].values

# atMLB: d_phiが+→-に変わる点
atMLB = []
for i in range(len(d_phi_data) - 1):
    atMLB.append(1 if d_phi_data[i] > 0 and d_phi_data[i+1] <= 0 else 0)
atMLB.append(0)

# ---- 人体パラメータ ----
L = 1.61; Lh = 1.58; M = 50
m1 = 0.626 * M; m2 = 0.246 * M; m3 = 0.128 * M
L1 = 0.501 * Lh; L2 = 0.249 * Lh; L3 = 0.25 * Lh
a = L2 * (m3 + 0.5 * m2) / M; b = L2 - a

ANGLE_OFFSET = -0.5 * np.pi

os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
font = pygame.font.SysFont("Arial", 22)
font_small = pygame.font.SysFont("Arial", 18)


def draw_figure(surf, cx, cy, idx, colors, widths, head_color, scale=100):
    """1フレーム分の人体+ロープ+座面を描画 (中心座標 cx,cy 指定)"""
    d = ANGLE_OFFSET
    rx0 = L * scale * np.cos(x_data[idx] + d)
    ry0 = L * scale * np.sin(x_data[idx] + d)
    pivot = (cx, cy)
    seat = (int(cx + rx0), int(cy - ry0))
    pygame.draw.line(surf, colors['rope'], pivot, seat, widths['rope'])
    pygame.draw.circle(surf, (40, 40, 40), pivot, 4)
    # 座面
    sdx = -np.sin(x_data[idx] + d)
    sdy = np.cos(x_data[idx] + d)
    sL = int(24 * scale / 100)
    seat1 = (int(seat[0] - sdx * sL), int(seat[1] + sdy * sL))
    seat2 = (int(seat[0] + sdx * sL), int(seat[1] - sdy * sL))
    pygame.draw.line(surf, colors['seat'], seat1, seat2, max(3, int(6 * scale / 100)))
    # 腰
    rx2a = -a * scale * np.sin(x_data[idx] - d)
    ry2a = a * scale * np.cos(x_data[idx] - d)
    hip = (int(seat[0] + rx2a), int(seat[1] - ry2a))
    # 上体: phi は絶対角(鉛直基準)なので x を加算しない
    rx1 = -L1 * scale * np.cos(phi_data[idx] + d)
    ry1 = -L1 * scale * np.sin(phi_data[idx] + d)
    torso_frac = 0.80
    neck = (int(hip[0] + rx1 * torso_frac), int(hip[1] - ry1 * torso_frac))
    pygame.draw.line(surf, colors['torso'], hip, neck, widths['torso'])
    # 頭部
    head_r = max(4, int(L1 * scale * 0.13))
    head_c = (int(hip[0] + rx1 * (torso_frac + 0.10)),
              int(hip[1] - ry1 * (torso_frac + 0.10)))
    pygame.draw.circle(surf, head_color, head_c, head_r)
    pygame.draw.circle(surf, (40, 40, 40), head_c, head_r, 2)
    # 上腿
    rx2 = -b * scale * np.sin(x_data[idx] + d)
    ry2 = b * scale * np.cos(x_data[idx] + d)
    knee = (int(seat[0] + rx2), int(seat[1] - ry2))
    pygame.draw.line(surf, colors['leg'], hip, knee, widths['leg'])
    # 下腿
    rx3 = L3 * scale * np.cos(psi_data[idx] + d)
    ry3 = L3 * scale * np.sin(psi_data[idx] + d)
    foot = (int(knee[0] + rx3), int(knee[1] - ry3))
    pygame.draw.line(surf, colors['leg'], knee, foot, widths['leg'])
    # 関節
    for jp in (hip, knee, foot):
        pygame.draw.circle(surf, (40, 40, 40), jp, max(2, int(3 * scale / 100)))


ghost_colors = {'rope': (200, 200, 210), 'seat': (210, 190, 170),
                'torso': (255, 200, 170), 'leg': (255, 200, 170)}
ghost_widths = {'rope': 2, 'torso': 5, 'leg': 4}
cur_colors = {'rope': (120, 60, 60), 'seat': (90, 50, 20),
              'torso': (40, 90, 200), 'leg': (40, 90, 200)}
cur_widths = {'rope': 4, 'torso': 14, 'leg': 11}


def surf_to_pil(surf):
    data = pygame.image.tostring(surf, 'RGB')
    return Image.frombytes('RGB', surf.get_size(), data)


# =============================================
# x_dataのゼロクロス(+→-)を探して周期を把握
# =============================================
zero_cross_pm = []  # x が + → - に変わるフレーム
for i in range(1, len(x_data)):
    if x_data[i-1] > 0 and x_data[i] <= 0:
        zero_cross_pm.append(i)

print(f"ゼロクロス(+→-) 数: {len(zero_cross_pm)}")

# 最大振幅付近を見つける (360°回転に入る前)
# x_data の絶対値が最大になる直前の安定した大振幅区間を使う
# 回転前 (|x| < 90°) での最大振幅区間を探す
max_amp_before_rotation = 0
max_amp_frame = 0
for i in range(len(x_data)):
    ax = abs(x_data[i])
    if ax > math.radians(90):
        break  # 回転領域に入ったら止める
    if ax > max_amp_before_rotation:
        max_amp_before_rotation = ax
        max_amp_frame = i

print(f"回転前最大振幅: {math.degrees(max_amp_before_rotation):.1f}° at t={t_data[max_amp_frame]:.1f}s")

# 最大振幅付近のゼロクロスを探す
target_t = t_data[max_amp_frame]
nearby_zc = [zc for zc in zero_cross_pm if abs(t_data[zc] - target_t) < 15]
print(f"最大振幅付近のゼロクロス: {len(nearby_zc)} 個")

# =============================================
# 1. max_amplitude.gif — 3周期ループ GIF
# =============================================
print("\n=== max_amplitude.gif ===")

# 最大振幅付近で3周期分を取る
if len(nearby_zc) >= 4:
    gif_start_zc = nearby_zc[-4]
    gif_end_zc = nearby_zc[-1]
else:
    gif_start_zc = nearby_zc[0]
    gif_end_zc = nearby_zc[-1]

print(f"GIF区間: t={t_data[gif_start_zc]:.1f}s ~ {t_data[gif_end_zc]:.1f}s")

GIF_W, GIF_H = 500, 500
gif_cx, gif_cy = GIF_W // 2, GIF_H // 2
gif_frames = []
SKIP_GIF = 4  # 4フレームに1枚 = 80ms間隔

for frame in range(gif_start_zc, gif_end_zc, SKIP_GIF):
    surf = pygame.Surface((GIF_W, GIF_H))
    surf.fill((245, 245, 250))
    # ゴースト (エピソード先頭からの全履歴)
    for i in range(frame - 1, -1, -1):
        if atMLB[i] > 0:
            draw_figure(surf, gif_cx, gif_cy, i, ghost_colors, ghost_widths, (255, 220, 190))
    # 現在
    draw_figure(surf, gif_cx, gif_cy, frame, cur_colors, cur_widths, (255, 210, 170))
    # テキスト
    tline = font.render(f"t = {t_data[frame]:5.1f} s", True, (40, 40, 40))
    surf.blit(tline, (20, 20))
    xdeg = abs(np.degrees(x_data[frame]))
    xline = font.render(f"|x| = {xdeg:4.1f}°", True, (40, 40, 40))
    surf.blit(xline, (20, 48))
    gif_frames.append(surf_to_pil(surf))

# GIF保存 (等速、80ms/frame)
gif_frames[0].save("max_amplitude.gif", save_all=True,
                    append_images=gif_frames[1:],
                    duration=80, loop=0)
print(f"max_amplitude.gif: {len(gif_frames)} frames")


# =============================================
# 2. hero_cold_start.png — 4時点スナップショット
# =============================================
print("\n=== hero_cold_start.png ===")

# 4時点: 序盤 / 中盤 / 大振幅 / 回転直前
# 各時点付近で |x| が最大(ピーク)のフレームを探す
hero_times = [20, 45, 70, 95]
hero_frames = []
for ht in hero_times:
    # ht ± 5秒の範囲で |x| が最大のフレームを探す
    window = 5.0
    mask = (t_data >= ht - window) & (t_data <= ht + window)
    indices = np.where(mask)[0]
    best = indices[np.argmax(np.abs(x_data[indices]))]
    hero_frames.append(best)
    print(f"  target t={ht}s → frame {best}, t={t_data[best]:.1f}s, |x|={math.degrees(abs(x_data[best])):.1f}°")

PANEL_W, PANEL_H = 300, 400
HERO_W = PANEL_W * 4 + 10  # 少し余白
HERO_H = PANEL_H + 60
hero_surf = pygame.Surface((HERO_W, HERO_H))
hero_surf.fill((245, 245, 250))

for pi, fidx in enumerate(hero_frames):
    px = pi * PANEL_W + PANEL_W // 2
    py = PANEL_H // 2 + 20
    # ゴースト: エピソード先頭からの全履歴
    for i in range(fidx - 1, -1, -1):
        if atMLB[i] > 0:
            draw_figure(hero_surf, px, py, i, ghost_colors, ghost_widths, (255, 220, 190), scale=80)
    # 現在フレーム
    draw_figure(hero_surf, px, py, fidx, cur_colors, cur_widths, (255, 210, 170), scale=80)
    # ラベル
    label = font.render(f"t = {t_data[fidx]:.0f} s", True, (40, 40, 40))
    hero_surf.blit(label, (pi * PANEL_W + 10, 5))
    xdeg = abs(np.degrees(x_data[fidx]))
    amp_label = font_small.render(f"|x| = {xdeg:.1f}°", True, (100, 40, 40))
    hero_surf.blit(amp_label, (pi * PANEL_W + 10, 30))

pil_hero = surf_to_pil(hero_surf)
pil_hero.save("hero_cold_start.png")
print(f"hero_cold_start.png: {HERO_W}x{HERO_H}")


# =============================================
# 3. one_period_strip.png — 1周期9コマ
# =============================================
print("\n=== one_period_strip.png ===")

# 最大振幅付近の1周期を取る
if len(nearby_zc) >= 2:
    strip_start = nearby_zc[-2]
    strip_end = nearby_zc[-1]
else:
    strip_start = nearby_zc[0]
    # 1周期分を推定
    period_frames = int(2.5 / 0.02)  # ~2.5s
    strip_end = min(strip_start + period_frames, len(t_data) - 1)

period_len = strip_end - strip_start
print(f"1周期: frame {strip_start}~{strip_end}, t={t_data[strip_start]:.2f}~{t_data[strip_end]:.2f}s ({t_data[strip_end]-t_data[strip_start]:.2f}s)")

NUM_COLS = 9
STRIP_PANEL_W = 180
STRIP_PANEL_H = 340
STRIP_W = STRIP_PANEL_W * NUM_COLS
STRIP_H = STRIP_PANEL_H + 40

strip_surf = pygame.Surface((STRIP_W, STRIP_H))
strip_surf.fill((245, 245, 250))

for ci in range(NUM_COLS):
    frac = ci / (NUM_COLS - 1)
    fidx = strip_start + int(frac * period_len)
    fidx = min(fidx, len(t_data) - 1)

    px = ci * STRIP_PANEL_W + STRIP_PANEL_W // 2
    py = STRIP_PANEL_H // 2 + 25
    scale = 70

    # 現在フレームのみ (ストリップはゴーストなし)
    draw_figure(strip_surf, px, py, fidx, cur_colors, cur_widths, (255, 210, 170), scale=scale)

    # 時刻ラベル
    dt = t_data[fidx] - t_data[strip_start]
    label = font_small.render(f"t+{dt:.2f}s", True, (40, 40, 40))
    strip_surf.blit(label, (ci * STRIP_PANEL_W + 5, 5))
    xdeg = np.degrees(x_data[fidx])
    xlab = font_small.render(f"x={xdeg:+.0f}°", True, (100, 40, 40))
    strip_surf.blit(xlab, (ci * STRIP_PANEL_W + 5, 25))

pil_strip = surf_to_pil(strip_surf)
pil_strip.save("one_period_strip.png")
print(f"one_period_strip.png: {STRIP_W}x{STRIP_H}")

print("\n完了!")
