"""汎用スイング動画生成スクリプト (トルク矢印 + 両側ゴースト対応)
Usage:
  python render_video.py <csv> <output.mp4> [title]
"""
import sys
import os
import math
import numpy as np
import pandas as pd
import pygame
import subprocess

csv_path   = sys.argv[1] if len(sys.argv) > 1 else "ai_swing_rotation.csv"
output_mp4 = sys.argv[2] if len(sys.argv) > 2 else "swing_output.mp4"
title_str  = sys.argv[3] if len(sys.argv) > 3 else ""

obs = pd.read_csv(csv_path)
obs = obs[obs['episode'] == 0].reset_index(drop=True)

t_data     = obs['t'].values
x_data     = obs['x'].values
z_data     = obs['z'].values        # dx (ロープ角速度)
phi_data   = obs['phi'].values
psi_data   = obs['psi'].values
torq_body  = obs['torq_ai'].values
torq_knee  = obs['torq_ai_knee'].values

# ゴーストマーカー: x の速度 z がゼロクロスする点 (両方向の峰を捕捉)
atPeak = []
for i in range(len(z_data)-1):
    # + → − (前方峰) または − → + (後方峰)
    cross = (z_data[i] > 0 and z_data[i+1] <= 0) or \
            (z_data[i] < 0 and z_data[i+1] >= 0)
    atPeak.append(1 if cross else 0)
atPeak.append(0)

L = 1.61; Lh = 1.58; M = 50
m1 = 0.626*M; m2 = 0.246*M; m3 = 0.128*M
L1 = 0.501*Lh; L2 = 0.249*Lh; L3 = 0.25*Lh
a = L2*(m3+0.5*m2)/M; b = L2-a
D = -0.5*np.pi

MAX_BODY = 40.0
MAX_KNEE = 30.0

SPEEDUP = 2
W, H = 520, 500
FPS = 50
SKIP = max(1, int(SPEEDUP * 1000 / (FPS * 20)))

os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
screen = pygame.Surface((W, H))
font   = pygame.font.SysFont("Arial", 22)
font_s = pygame.font.SysFont("Arial", 17)

x0, y0 = W//2 - 10, H//2

def joint_pos(idx):
    xi = x_data[idx]; ph = phi_data[idx]; ps = psi_data[idx]
    sx = x0 + 100*L*math.cos(xi+D);  sy = y0 - 100*L*math.sin(xi+D)
    hx = sx - 100*a*math.sin(xi-D);  hy = sy - 100*a*math.cos(xi-D)
    kx = sx - 100*b*math.sin(xi+D);  ky = sy - 100*b*math.cos(xi+D)
    fx = kx + 100*L3*math.cos(ps+D); fy = ky - 100*L3*math.sin(ps+D)
    nx = hx - 100*L1*0.80*math.cos(ph+D)      # ph は絶対角 → xi を足さない
    ny = hy + 100*L1*0.80*math.sin(ph+D)
    return (int(sx),int(sy)),(int(hx),int(hy)),(int(kx),int(ky)),(int(fx),int(fy)),(int(nx),int(ny))

def draw_figure(idx, colors, widths, head_color):
    xi = x_data[idx]; ph = phi_data[idx]
    seat,hip,knee,foot,neck = joint_pos(idx)
    pivot = (x0, y0)
    pygame.draw.line(screen, colors['rope'], pivot, seat, widths['rope'])
    pygame.draw.circle(screen, (40,40,40), pivot, 5)
    sdx = -math.sin(xi+D); sdy = math.cos(xi+D); sL = 22
    pygame.draw.line(screen, colors['seat'],
                     (int(seat[0]-sdx*sL),int(seat[1]+sdy*sL)),
                     (int(seat[0]+sdx*sL),int(seat[1]-sdy*sL)), 5)
    pygame.draw.line(screen, colors['torso'], hip, neck, widths['torso'])
    hr = int(L1*100*0.13)
    hcx = int(hip[0] - 100*L1*0.90*math.cos(ph+D))   # ph は絶対角
    hcy = int(hip[1] + 100*L1*0.90*math.sin(ph+D))
    pygame.draw.circle(screen, head_color, (hcx,hcy), hr)
    pygame.draw.circle(screen, (40,40,40), (hcx,hcy), hr, 2)
    pygame.draw.line(screen, colors['leg'], hip, knee, widths['leg'])
    pygame.draw.line(screen, colors['leg'], knee, foot, widths['leg'])
    for jp in (hip, knee, foot):
        pygame.draw.circle(screen, (40,40,40), jp, 3)

def draw_torque_arrow(cx, cy, torque, max_val, col_pos, col_neg):
    if abs(torque) < 0.3:
        return
    norm = min(abs(torque) / max_val, 1.0)
    radius = int(12 + norm * 20)
    width  = max(2, int(norm * 7))
    col = col_pos if torque > 0 else col_neg
    arc_span = math.pi * 1.5
    start = math.pi * 0.25 if torque > 0 else math.pi * 1.75
    sign  = 1 if torque > 0 else -1
    n = 30
    pts = [(cx + radius*math.cos(start + sign*arc_span*i/n),
            cy - radius*math.sin(start + sign*arc_span*i/n))
           for i in range(n+1)]
    for i in range(len(pts)-1):
        pygame.draw.line(screen, col,
                         (int(pts[i][0]),int(pts[i][1])),
                         (int(pts[i+1][0]),int(pts[i+1][1])), width)
    tip = pts[-1]; prev = pts[-3]
    dx = tip[0]-prev[0]; dy = tip[1]-prev[1]
    ln = math.hypot(dx,dy)
    if ln > 0:
        ux=dx/ln; uy=dy/ln; aw=max(4,width+4)
        p1=(int(tip[0]-ux*aw*1.2-uy*aw*0.7), int(tip[1]-uy*aw*1.2+ux*aw*0.7))
        p2=(int(tip[0]-ux*aw*1.2+uy*aw*0.7), int(tip[1]-uy*aw*1.2-ux*aw*0.7))
        pygame.draw.polygon(screen, col, [tip, p1, p2])

ghost_colors = {'rope':(200,200,210),'seat':(210,190,170),
                'torso':(255,200,170),'leg':(255,200,170)}
ghost_widths = {'rope':2,'torso':5,'leg':4}
cur_colors   = {'rope':(120,60,60),'seat':(90,50,20),
                'torso':(40,90,200),'leg':(40,90,200)}
cur_widths   = {'rope':4,'torso':14,'leg':11}

COL_W_POS=(220,60,60); COL_W_NEG=(60,100,220)
COL_K_POS=(40,180,80); COL_K_NEG=(180,160,20)

cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
       '-s',f'{W}x{H}','-pix_fmt','rgb24','-r',str(FPS),
       '-i','-','-c:v','libx264','-pix_fmt','yuv420p',
       '-crf','20', output_mp4]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

total = len(t_data)
frames_written = 0
for frame in range(0, total, SKIP):
    screen.fill((245,245,250))

    # ゴースト (両側峰)
    for i in range(frame-1, -1, -1):
        if atPeak[i]:
            draw_figure(i, ghost_colors, ghost_widths, (255,220,190))

    # 現在フレーム
    draw_figure(frame, cur_colors, cur_widths, (255,210,170))

    # トルク矢印
    _,hip,knee,_,_ = joint_pos(frame)
    draw_torque_arrow(hip[0],  hip[1],  torq_body[frame], MAX_BODY, COL_W_POS, COL_W_NEG)
    draw_torque_arrow(knee[0], knee[1], torq_knee[frame], MAX_KNEE, COL_K_POS, COL_K_NEG)

    # HUD
    screen.blit(font.render(f"t = {t_data[frame]:6.1f} s", True, (40,40,40)), (20,18))
    xdeg = np.degrees(x_data[frame])
    screen.blit(font.render(f"x = {xdeg:+.1f}°", True, (40,40,40)), (20,44))
    tb = torq_body[frame]; tk = torq_knee[frame]
    screen.blit(font_s.render(f"W:{tb:+.0f}Nm", True, COL_W_POS if tb>=0 else COL_W_NEG), (20,76))
    screen.blit(font_s.render(f"K:{tk:+.0f}Nm", True, COL_K_POS if tk>=0 else COL_K_NEG), (20,96))
    if title_str:
        ts = font_s.render(title_str, True, (80,80,160))
        screen.blit(ts, (W//2 - ts.get_width()//2, H-24))

    proc.stdin.write(pygame.image.tostring(screen, 'RGB'))
    frames_written += 1
    if frames_written % 200 == 0:
        print(f"  {t_data[frame]:.1f}s / {t_data[-1]:.1f}s", flush=True)

proc.stdin.close(); proc.wait()
print(f"完了: {output_mp4} ({frames_written} frames, {frames_written/FPS:.1f}s, {SPEEDUP}x)", flush=True)
