"""360°回転エピソードの動画をオフスクリーン描画 → ffmpegでmp4化"""
import pygame
import pandas as pd
import numpy as np
import subprocess
import os
import math

# CSV読み込み
obs = pd.read_csv("ai_swing_rotation.csv")
t = obs['t'].values
x = obs['x'].values
phi = obs['phi'].values
d_phi = obs['d_phi'].values
psi = obs['psi'].values

# atMLB: d_phiが+→-に変わる点
atMLB = []
for i in range(len(d_phi) - 1):
    atMLB.append(1 if d_phi[i] > 0 and d_phi[i+1] <= 0 else 0)
atMLB.append(0)

# 人体パラメータ
L = 1.61
Lh = 1.58
M = 50
m1 = 0.626 * M
m2 = 0.246 * M
m3 = 0.128 * M
L1 = 0.501 * Lh
L2 = 0.249 * Lh
L3 = 0.25 * Lh
a = L2 * (m3 + 0.5 * m2) / M
b = L2 - a

# 動画設定
SPEEDUP = 2
W, H = 500, 500
FPS = 50  # 出力FPS
SKIP = max(1, int(SPEEDUP * 1000 / (FPS * 20)))  # 20ms/frame → skip frames

os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
screen = pygame.Surface((W, H))
font = pygame.font.SysFont("Arial", 24)

x0, y0 = W // 2, H // 2
d = -0.5 * np.pi

def draw_figure(surf, frame_idx, colors, widths, head_color):
    rx0 = L * 100 * np.cos(x[frame_idx] + d)
    ry0 = L * 100 * np.sin(x[frame_idx] + d)
    pivot = (x0, y0)
    seat = (int(x0 + rx0), int(y0 - ry0))
    # ロープ
    pygame.draw.line(surf, colors['rope'], pivot, seat, widths['rope'])
    # 支点
    pygame.draw.circle(surf, (40, 40, 40), pivot, 5)
    # 座面
    sdx = -np.sin(x[frame_idx] + d)
    sdy = np.cos(x[frame_idx] + d)
    sL = 24
    seat1 = (int(seat[0] - sdx * sL), int(seat[1] + sdy * sL))
    seat2 = (int(seat[0] + sdx * sL), int(seat[1] - sdy * sL))
    pygame.draw.line(surf, colors['seat'], seat1, seat2, 6)

    # 腰
    rx2a = -a * 100 * np.sin(x[frame_idx] - d)
    ry2a = a * 100 * np.cos(x[frame_idx] - d)
    hip = (int(seat[0] + rx2a), int(seat[1] - ry2a))

    # 上体(腰→首)
    rx1 = -L1 * 100 * np.cos(x[frame_idx] + phi[frame_idx] + d)
    ry1 = -L1 * 100 * np.sin(x[frame_idx] + phi[frame_idx] + d)
    torso_frac = 0.80
    neck = (int(hip[0] + rx1 * torso_frac), int(hip[1] - ry1 * torso_frac))
    pygame.draw.line(surf, colors['torso'], hip, neck, widths['torso'])

    # 頭部
    head_r = int(L1 * 100 * 0.13)
    head_c = (int(hip[0] + rx1 * (torso_frac + 0.10)),
              int(hip[1] - ry1 * (torso_frac + 0.10)))
    pygame.draw.circle(surf, head_color, head_c, head_r)
    pygame.draw.circle(surf, (40, 40, 40), head_c, head_r, 2)

    # 上腿(腰→膝)
    rx2 = -b * 100 * np.sin(x[frame_idx] + d)
    ry2 = b * 100 * np.cos(x[frame_idx] + d)
    knee = (int(seat[0] + rx2), int(seat[1] - ry2))
    pygame.draw.line(surf, colors['leg'], hip, knee, widths['leg'])

    # 下腿(膝→足先) — psiは絶対角
    rx3 = L3 * 100 * np.cos(psi[frame_idx] + d)
    ry3 = L3 * 100 * np.sin(psi[frame_idx] + d)
    foot = (int(knee[0] + rx3), int(knee[1] - ry3))
    pygame.draw.line(surf, colors['leg'], knee, foot, widths['leg'])

    # 関節マーカー
    for jp in (hip, knee, foot):
        pygame.draw.circle(surf, (40, 40, 40), jp, 3)

# ffmpegプロセス起動
cmd = [
    'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{W}x{H}', '-pix_fmt', 'rgb24', '-r', str(FPS),
    '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-crf', '20', 'rotation_swing.mp4'
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

ghost_colors = {'rope': (200, 200, 210), 'seat': (210, 190, 170),
                'torso': (255, 200, 170), 'leg': (255, 200, 170)}
ghost_widths = {'rope': 2, 'torso': 5, 'leg': 4}
cur_colors = {'rope': (120, 60, 60), 'seat': (90, 50, 20),
              'torso': (40, 90, 200), 'leg': (40, 90, 200)}
cur_widths = {'rope': 4, 'torso': 14, 'leg': 11}

frames_written = 0
total_frames = len(t)

for frame in range(0, total_frames, SKIP):
    screen.fill((245, 245, 250))

    # 残像描画 (エピソード先頭まで全履歴)
    for i in range(frame - 1, -1, -1):
        if atMLB[i] > 0:
            draw_figure(screen, i, ghost_colors, ghost_widths, (255, 220, 190))

    # 現在フレーム
    draw_figure(screen, frame, cur_colors, cur_widths, (255, 210, 170))

    # テキスト
    tline = font.render(f"t = {t[frame]:6.1f} s", True, (40, 40, 40))
    screen.blit(tline, (20, 20))
    xdeg = np.degrees(x[frame])
    xline = font.render(f"x = {xdeg:+.1f} deg", True, (40, 40, 40))
    screen.blit(xline, (20, 50))

    # 360°達成マーカー
    if abs(x[frame]) > 2 * math.pi:
        bang = font.render("360 deg!", True, (200, 0, 0))
        screen.blit(bang, (W // 2 - 60, 20))

    # フレーム書き出し
    raw = pygame.image.tostring(screen, 'RGB')
    proc.stdin.write(raw)
    frames_written += 1

    if frames_written % 100 == 0:
        print(f"  {frames_written} frames ({t[frame]:.1f}s / {t[-1]:.1f}s)")

proc.stdin.close()
proc.wait()

duration = frames_written / FPS
print(f"\n完了: rotation_swing.mp4 ({frames_written} frames, {duration:.1f}s, {SPEEDUP}x speed)")
