"""エネルギー変化率可視化スイング動画
各パーツの色 = dE/dt (エネルギーの増減)
  増加中 (+) : 赤
  ゼロ付近   : 薄いグレー
  減少中 (−) : 青
関節の光彩  : 腰・膝に筋力仕事率 P = u·dq に比例した後光（黄）
右上に dE/dt バー、右下に P バー表示

Usage: python render_energy_video.py <csv> <output.mp4> [title]
"""
import sys, os, math
import numpy as np
import pandas as pd
import pygame
import subprocess
sys.path.insert(0, '.')
from swing_physics import Swing

csv_path   = sys.argv[1] if len(sys.argv) > 1 else "ai_swing_rotation.csv"
output_mp4 = sys.argv[2] if len(sys.argv) > 2 else "energy_swing.mp4"
title_str  = sys.argv[3] if len(sys.argv) > 3 else ""

obs = pd.read_csv(csv_path)
obs = obs[obs['episode'] == 0].reset_index(drop=True)

t_data   = obs['t'].values
x_data   = obs['x'].values
z_data   = obs['z'].values
phi_data = obs['phi'].values
psi_data = obs['psi'].values
dphi_data = obs['d_phi'].values
dpsi_data = obs['d_psi'].values
torq_body = obs['torq_ai'].values
torq_knee = obs['torq_ai_knee'].values

# ---- 物理パラメータ ----
sw = Swing(eps=1, VR=0, x=15, coef_Hooke=0)
L=sw.L; Lk=sw.Lk; L1=sw.L1; L3=sw.L3
m0=sw.m0; m1=sw.m1; m3=sw.m3; G=sw.G
k_waist=144.0; k_knee=50.0
B = m1*L1**2/3;  C = m3*L3**2/3
A_arr = (m0/3 + m1)*L**2 + m3*Lk**2   # scalar (approximately constant)

# ---- エネルギー計算（厳密: 各剛体の重心速度から直接計算）----
#
# 座標系: ピボット原点, x右, y上
# 各剛体の KE = ½·m·|v_cm|² + ½·I_cm·ω²  (並進 + 重心まわり回転)
# PE = m·g·(y_cm − y_cm_equilibrium)

x  = x_data;  dx = z_data
ph = phi_data; dph = dphi_data
ps = psi_data; dps = dpsi_data

# ---- ロープ (均一棒, 長さ L, 端がピボット) ----
# KE = ½·(m0/3)·L²·dx²  (棒の端まわり回転 = 並進+回転の厳密な合計)
# PE equilibrium: x=0 → y_cm = -L/2
KE_rope = 0.5*(m0/3)*L**2 * dx**2
PE_rope = m0*G*(L/2)*(1 - np.cos(x))          # Δ from x=0

# ---- 胴体 (m1, 均一棒, 長さ L1) ----
# 腰の位置 (ピボット基準): hip ≈ (L·sin x, −L·cos x)
# 胴体方向 (腰→首): (−sin φ, cos φ)
#   ← φ は鉛直上向きを基準とした胴体の絶対角度
#   ← 質量行列 D = -m1·(L1/2)·L·cos(φ-x) と整合する唯一の解釈
# 重心位置:
cm_tx = L*np.sin(x) - (L1/2)*np.sin(ph)
cm_ty = -L*np.cos(x) + (L1/2)*np.cos(ph)
# 重心速度:
vcm_tx = L*np.cos(x)*dx - (L1/2)*np.cos(ph)*dph
vcm_ty = L*np.sin(x)*dx - (L1/2)*np.sin(ph)*dph
omega_t = dph                                   # 胴体の絶対角速度
Icm_t   = m1*L1**2/12                          # 重心まわり慣性モーメント
KE_torso = 0.5*m1*(vcm_tx**2 + vcm_ty**2) + 0.5*Icm_t*omega_t**2
PE_torso = m1*G*(cm_ty - (-L + L1/2))          # Δ from x=φ=0
E_torso_arr = KE_torso + PE_torso + 0.5*k_waist*(ph - x)**2

# ---- 膝下 (m3, 均一棒, 長さ L3) ----
# 膝の位置: knee ≈ (Lk·sin x, −Lk·cos x)
# 膝下方向 (膝→足先): (sin ψ, −cos ψ)
# 重心位置:
cm_lx = Lk*np.sin(x) + (L3/2)*np.sin(ps)
cm_ly = -Lk*np.cos(x) - (L3/2)*np.cos(ps)
# 重心速度:
vcm_lx = Lk*np.cos(x)*dx + (L3/2)*np.cos(ps)*dps
vcm_ly = Lk*np.sin(x)*dx + (L3/2)*np.sin(ps)*dps
omega_l = dps                                   # 膝下の絶対角速度
Icm_l   = m3*L3**2/12
KE_leg  = 0.5*m3*(vcm_lx**2 + vcm_ly**2) + 0.5*Icm_l*omega_l**2
PE_leg  = m3*G*(cm_ly - (-Lk - L3/2))          # Δ from x=ψ=0
E_leg_arr = KE_leg + PE_leg + 0.5*k_knee*(ps - x)**2

# ロープエネルギー (胴体・膝下の PE の x 依存分は既に torso/leg に含む)
E_swing_arr = KE_rope + PE_rope

# ---- 数値確認: 合計が質量行列の全エネルギーと一致するか ----
E_total_rigorous = E_swing_arr + E_torso_arr + E_leg_arr
# (デバッグ用: 最初の数点を出力)
# print("E_total sample:", E_total_rigorous[:5])

P_knee_arr  = torq_knee * dpsi_data   # 膝筋力仕事率
P_body_arr  = torq_body * dphi_data   # 腰筋力仕事率

# ---- dE/dt 計算 (軽い平滑化でノイズ除去) ----
def smooth(a, n=30):
    return np.convolve(a, np.ones(n)/n, mode='same')

dEs_arr = smooth(np.gradient(E_swing_arr, t_data))  # ロープ dE/dt
dEt_arr = smooth(np.gradient(E_torso_arr, t_data))  # 胴体 dE/dt
dEl_arr = smooth(np.gradient(E_leg_arr,   t_data))  # 膝下 dE/dt

# 正規化スケール: 各 dE/dt の 97 パーセンタイル絶対値
DES_max = np.percentile(np.abs(dEs_arr), 97)
DET_max = np.percentile(np.abs(dEt_arr), 97)
DEL_max = np.percentile(np.abs(dEl_arr), 97)

# 参照用: 絶対エネルギーのスケール
ES_max  = np.percentile(E_swing_arr, 97)
ET_max  = np.percentile(E_torso_arr, 97)
EL_max  = np.percentile(E_leg_arr,   97)
PK_max  = np.percentile(np.abs(P_knee_arr), 97)
PB_max  = np.percentile(np.abs(P_body_arr), 97)

def norm(v, vmax): return float(np.clip(v / max(vmax, 1e-6), 0, 1))
def norm_signed(v, vmax): return float(np.clip(v / max(vmax, 1e-6), -1, 1))

# ---- ゴーストマーカー (dx ゼロクロス) ----
atPeak = []
for i in range(len(z_data)-1):
    cross = (z_data[i]>0 and z_data[i+1]<=0) or (z_data[i]<0 and z_data[i+1]>=0)
    atPeak.append(1 if cross else 0)
atPeak.append(0)

# ---- diverging カラースケール: 赤(+増加) ↔ グレー(0) ↔ 青(−減少) ----
def lerp_color(c0, c1, t):
    t = max(0.0, min(1.0, t))
    return (int(c0[0]+(c1[0]-c0[0])*t),
            int(c0[1]+(c1[1]-c0[1])*t),
            int(c0[2]+(c1[2]-c0[2])*t))

C_GRAY = (130, 130, 135)   # ゼロ付近
C_RED  = (220,  35,  20)   # 増加(+)
C_BLUE = ( 20,  70, 220)   # 減少(−)

def dE_color(signed_norm):
    """signed_norm ∈ [-1, 1]: -1→青, 0→グレー, +1→赤"""
    s = max(-1.0, min(1.0, signed_norm))
    if s >= 0:
        return lerp_color(C_GRAY, C_RED,  s)
    else:
        return lerp_color(C_GRAY, C_BLUE, -s)

def glow_alpha(p_norm):
    """仕事率に応じた後光の透明度 0〜200"""
    return int(p_norm * 200)

# ---- 描画定数 ----
SPEEDUP = 2
W, H = 560, 520
FPS  = 50
SKIP = max(1, int(SPEEDUP * 1000 / (FPS * 20)))
D    = -0.5 * math.pi

x0, y0 = W//2 - 10, H//2

os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
screen = pygame.Surface((W, H))
font   = pygame.font.SysFont("Arial", 20)
font_s = pygame.font.SysFont("Arial", 15)
font_b = pygame.font.SysFont("Arial", 13)

def joint_pos(idx):
    xi = x_data[idx]; ph = phi_data[idx]; ps = psi_data[idx]
    sx = x0 + 100*L*math.cos(xi+D);  sy = y0 - 100*L*math.sin(xi+D)
    hx = sx - 100*(sw.a)*math.sin(xi-D); hy = sy - 100*(sw.a)*math.cos(xi-D)
    kx = sx - 100*(sw.b)*math.sin(xi+D); ky = sy - 100*(sw.b)*math.cos(xi+D)
    fx = kx + 100*L3*math.cos(ps+D); fy = ky - 100*L3*math.sin(ps+D)
    nx = hx - 100*L1*0.80*math.cos(ph+D)      # ph は絶対角 → xi を足さない
    ny = hy + 100*L1*0.80*math.sin(ph+D)
    return (int(sx),int(sy)),(int(hx),int(hy)),(int(kx),int(ky)),(int(fx),int(fy)),(int(nx),int(ny))

def draw_glow(cx, cy, radius, color, alpha):
    """半透明の後光円を描画"""
    if alpha < 5:
        return
    glow_surf = pygame.Surface((radius*4, radius*4), pygame.SRCALPHA)
    for r in range(radius, 0, -2):
        a = int(alpha * (r / radius) * 0.6)
        pygame.draw.circle(glow_surf, (*color, a), (radius*2, radius*2), r)
    screen.blit(glow_surf, (cx - radius*2, cy - radius*2))

def draw_figure_energy(idx):
    """エネルギーに応じた色で人物を描画"""
    xi  = x_data[idx]; ph = phi_data[idx]
    seat, hip, knee, foot, neck = joint_pos(idx)
    pivot = (x0, y0)

    # --- dE/dt 正規化 (符号付き, -1〜+1) ---
    des = norm_signed(dEs_arr[idx], DES_max)
    det = norm_signed(dEt_arr[idx], DET_max)
    del_ = norm_signed(dEl_arr[idx], DEL_max)
    pk  = norm(max(P_knee_arr[idx], 0), PK_max)
    pb  = norm(max(P_body_arr[idx], 0), PB_max)

    # --- 色決定: dE/dt の符号・大きさで赤↔グレー↔青 ---
    c_rope  = dE_color(des)
    c_torso = dE_color(det)
    c_leg   = dE_color(del_)
    c_head  = dE_color(det)
    c_seat  = dE_color(des)

    # --- ロープ ---
    pygame.draw.line(screen, c_rope, pivot, seat, 4)
    pygame.draw.circle(screen, (40, 40, 40), pivot, 5)

    # --- 座板 ---
    sdx = -math.sin(xi+D); sdy = math.cos(xi+D); sL = 22
    pygame.draw.line(screen, c_seat,
                     (int(seat[0]-sdx*sL), int(seat[1]+sdy*sL)),
                     (int(seat[0]+sdx*sL), int(seat[1]-sdy*sL)), 5)

    # --- 後光: 腰 (P_body) ---
    if pb > 0.05:
        draw_glow(hip[0], hip[1], 28, (255, 220, 40), glow_alpha(pb))

    # --- 後光: 膝 (P_knee) ---
    if pk > 0.05:
        draw_glow(knee[0], knee[1], 22, (255, 220, 40), glow_alpha(pk))

    # --- 胴体 ---
    pygame.draw.line(screen, c_torso, hip, neck, 13)

    # --- 頭部 ---
    hr  = int(L1*100*0.13)
    hcx = int(hip[0] - 100*L1*0.90*math.cos(ph+D))   # ph は絶対角
    hcy = int(hip[1] + 100*L1*0.90*math.sin(ph+D))
    pygame.draw.circle(screen, c_head,   (hcx, hcy), hr)
    pygame.draw.circle(screen, (40,40,40),(hcx, hcy), hr, 2)

    # --- 脚: 上腿はロープと剛結合 → c_rope、下腿のみ c_leg ---
    pygame.draw.line(screen, c_rope, hip,  knee, 10)   # 上腿 = ロープと同系
    pygame.draw.line(screen, c_leg,  knee, foot, 10)   # 下腿 = 膝下 DOF
    for jp in (hip, knee, foot):
        pygame.draw.circle(screen, (30,30,30), jp, 3)

def draw_figure_ghost(idx):
    """ゴースト (薄く)"""
    seat, hip, knee, foot, neck = joint_pos(idx)
    pivot = (x0, y0)
    gc = (200, 200, 210)
    pygame.draw.line(screen, gc, pivot, seat, 2)
    pygame.draw.line(screen, (220,190,160), hip, neck, 5)
    pygame.draw.line(screen, (220,190,160), hip, knee, 4)
    pygame.draw.line(screen, (220,190,160), knee, foot, 4)
    ph = phi_data[idx]
    hr = int(sw.L1*100*0.13)
    hcx = int(hip[0] - 100*sw.L1*0.90*math.cos(ph+D))
    hcy = int(hip[1] + 100*sw.L1*0.90*math.sin(ph+D))
    pygame.draw.circle(screen, (240,210,190), (hcx,hcy), hr)

# ---- エネルギーバー描画 ----
BAR_X, BAR_Y = W - 130, 20
BAR_W, BAR_H = 100, 12

def draw_energy_bars(idx):
    des = dEs_arr[idx]
    det = dEt_arr[idx]
    del_ = dEl_arr[idx]
    pk_raw = P_knee_arr[idx]
    pb_raw = P_body_arr[idx]

    # diverging バー: 中央ゼロ、右=増加(赤)、左=減少(青)
    parts = [
        ("dE swing", des,  DES_max),
        ("dE torso", det,  DET_max),
        ("dE shin",  del_, DEL_max),
    ]
    pygame.draw.rect(screen, (30,30,40), (BAR_X-8, BAR_Y-6, 158, 105), border_radius=6)
    MID = BAR_X + BAR_W//2
    for i, (lbl, val, vmax) in enumerate(parts):
        by = BAR_Y + i * 26
        s  = float(np.clip(val / max(vmax, 1e-6), -1, 1))
        col = dE_color(s)
        pygame.draw.rect(screen, (55,55,65), (BAR_X, by, BAR_W, BAR_H))
        bar_len = int((BAR_W//2) * abs(s))
        if s >= 0:
            pygame.draw.rect(screen, col, (MID, by, bar_len, BAR_H))
        else:
            pygame.draw.rect(screen, col, (MID - bar_len, by, bar_len, BAR_H))
        pygame.draw.line(screen, (180,180,180), (MID, by), (MID, by+BAR_H))
        pygame.draw.rect(screen, (150,150,150), (BAR_X, by, BAR_W, BAR_H), 1)
        screen.blit(font_b.render(lbl, True, (220,220,220)), (BAR_X - 58, by - 1))
        screen.blit(font_b.render(f"{val:+.0f}", True, col), (BAR_X + BAR_W + 4, by - 1))

    # 仕事率テキスト
    pk_col = (230,80,80) if pk_raw >= 0 else (80,120,220)
    pb_col = (230,80,80) if pb_raw >= 0 else (80,120,220)
    py = BAR_Y + 82
    screen.blit(font_b.render(f"P knee: {pk_raw:+.0f} W", True, pk_col), (BAR_X-58, py))
    screen.blit(font_b.render(f"P waist:{pb_raw:+.0f} W", True, pb_col), (BAR_X-58, py+16))

# ---- カラースケール凡例 ----
def draw_legend():
    lx, ly = 10, H - 96
    bw = 130
    pygame.draw.rect(screen, (30,30,40), (lx-4, ly-4, bw+12, 94), border_radius=5)

    # diverging グラデーションバー
    screen.blit(font_b.render("-", True, C_BLUE), (lx, ly))
    screen.blit(font_b.render("0", True, C_GRAY), (lx + bw//2 - 3, ly))
    screen.blit(font_b.render("+", True, C_RED),  (lx + bw - 7,    ly))
    for px in range(bw):
        s = (px / bw) * 2 - 1   # -1 〜 +1
        c = dE_color(s)
        pygame.draw.rect(screen, c, (lx, ly+14, px+1, 10))
    pygame.draw.rect(screen, (160,160,160), (lx, ly+14, bw, 10), 1)

    # スケール注記
    rows = [
        f"Rope  +-{DES_max:.0f} W",
        f"Torso +-{DET_max:.0f} W",
        f"Shin  +-{DEL_max:.0f} W",
    ]
    for i, lbl in enumerate(rows):
        screen.blit(font_b.render(lbl, True, (175,175,175)), (lx, ly+30+i*18))

    # 後光の凡例
    gy2 = ly + 30 + 3*18
    pygame.draw.circle(screen, (255,215,0), (lx+6, gy2+6), 5)
    screen.blit(font_b.render("glow = muscle power > 0", True, (175,175,175)), (lx+16, gy2))

# ---- ffmpeg ----
cmd = ['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo',
       '-s',f'{W}x{H}','-pix_fmt','rgb24','-r',str(FPS),
       '-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','20', output_mp4]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

total = len(t_data); frames = 0
for frame in range(0, total, SKIP):
    screen.fill((18, 18, 28))   # 暗い背景

    # ゴースト
    for i in range(frame-1, -1, -1):
        if atPeak[i]:
            draw_figure_ghost(i)
            break   # 直近1つのゴーストのみ表示 (重くならないよう)

    # 現在フレーム (エネルギー色)
    draw_figure_energy(frame)

    # UI
    draw_energy_bars(frame)
    draw_legend()

    # HUD
    screen.blit(font.render(f"t = {t_data[frame]:5.1f} s", True, (200,200,210)), (20, 18))
    xdeg = math.degrees(x_data[frame])
    screen.blit(font.render(f"x = {xdeg:+.1f}°", True, (200,200,210)), (20, 42))
    if title_str:
        ts = font_s.render(title_str, True, (160,160,220))
        screen.blit(ts, (W//2 - ts.get_width()//2, H-22))

    proc.stdin.write(pygame.image.tostring(screen, 'RGB'))
    frames += 1
    if frames % 300 == 0:
        print(f"  {t_data[frame]:.1f}s / {t_data[-1]:.1f}s", flush=True)

proc.stdin.close(); proc.wait()
print(f"完了: {output_mp4} ({frames} frames, {frames/FPS:.1f}s, {SPEEDUP}x)", flush=True)
