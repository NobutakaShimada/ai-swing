import pygame
import time
import pandas as pd
import sys
import numpy as np

def p2mA(x):
    z=[] # + から - へ移るとき 1
    for i in range(0, len(x)-1):
        if (x[i]>0) and (x[i+1]<=0 ):
            z.append(1)
        else :
             z.append(0)
    return z


if len(sys.argv)<2 :
    print("コマンドライン引数で ファイル名 を入力")
    sys.exit()

# CSVファイルの読み込み, データの取得
obs = pd.read_csv( sys.argv[1])
epi = obs['episode'].values + 1
t = obs['t'].values
x = obs['x'].values
phi = obs['phi'].values
d_phi = obs['d_phi'].values
psi = 0
atMLB = p2mA(d_phi)

# パラメータ設定

L = 1.61
Lh = 1.58
M = 50
m1=0.626*M
m2=0.246*M
m3=0.128*M
L1=0.501*Lh
L2=0.249*Lh
L3=0.25*Lh 
a=L2*(m3+0.5*m2)/M
b=(L2-a)
L=L


pygame.init()
font = pygame.font.SysFont("Arial", 24)

x0=250
y0=250
screen = pygame.display.set_mode((x0*2, y0*2))
clock = pygame.time.Clock()

running = True
frame = 0

while running :
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill((245, 245, 250))
    d = -0.5 * np.pi  # 角度のオフセット

    def draw_figure(frame_idx, colors, widths, head_color):
        """1フレーム分の人体+ロープ+座面を太線で描画"""
        rx0 = L*100*np.cos(x[frame_idx]+d)
        ry0 = L*100*np.sin(x[frame_idx]+d)
        pivot = (x0, y0)
        seat  = (int(x0+rx0), int(y0-ry0))
        # ロープ
        pygame.draw.line(screen, colors['rope'], pivot, seat, widths['rope'])
        # 支点
        pygame.draw.circle(screen, (40,40,40), pivot, 5)
        # 座面 (ロープに垂直な短い線)
        sdx = -np.sin(x[frame_idx]+d)
        sdy =  np.cos(x[frame_idx]+d)
        sL = 24
        seat1 = (int(seat[0] - sdx*sL), int(seat[1] + sdy*sL))
        seat2 = (int(seat[0] + sdx*sL), int(seat[1] - sdy*sL))
        pygame.draw.line(screen, colors['seat'], seat1, seat2, 6)

        # 腰の位置 (座面からのオフセット)
        rx2a = -a*100*np.sin(x[frame_idx]-d)
        ry2a =  a*100*np.cos(x[frame_idx]-d)
        hip = (int(seat[0]+rx2a), int(seat[1]-ry2a))

        # 上体 (腰 → 首)
        rx1 = -L1*100*np.cos(x[frame_idx]+phi[frame_idx]+d)
        ry1 = -L1*100*np.sin(x[frame_idx]+phi[frame_idx]+d)
        torso_frac = 0.80
        neck = (int(hip[0] + rx1*torso_frac), int(hip[1] - ry1*torso_frac))
        pygame.draw.line(screen, colors['torso'], hip, neck, widths['torso'])

        # 頭部 (首の先に円)
        head_r = int(L1*100*0.13)
        head_c = (int(hip[0] + rx1*(torso_frac + 0.10)),
                  int(hip[1] - ry1*(torso_frac + 0.10)))
        pygame.draw.circle(screen, head_color, head_c, head_r)
        pygame.draw.circle(screen, (40,40,40), head_c, head_r, 2)

        # 上腿 (腰 → 膝)
        rx2 = -b*100*np.sin(x[frame_idx]+d)
        ry2 =  b*100*np.cos(x[frame_idx]+d)
        knee = (int(seat[0]+rx2), int(seat[1]-ry2))
        pygame.draw.line(screen, colors['leg'], hip, knee, widths['leg'])

        # 下腿 (膝 → 足先)
        rx3 = L3*100*np.cos(x[frame_idx]+psi+d)
        ry3 = L3*100*np.sin(x[frame_idx]+psi+d)
        foot = (int(knee[0]+rx3), int(knee[1]-ry3))
        pygame.draw.line(screen, colors['leg'], knee, foot, widths['leg'])

        # 関節マーカー
        for jp in (hip, knee, foot):
            pygame.draw.circle(screen, (40,40,40), jp, 3)

    # atMLBの残像 (薄色細線)
    ghost_colors = {'rope':(200,200,210),'seat':(210,190,170),
                    'torso':(255,200,170),'leg':(255,200,170)}
    ghost_widths = {'rope':2,'torso':5,'leg':4}
    for i in range(frame-1, -1, -1):
        if epi[i+1] != epi[i]:
            break
        if atMLB[i] > 0:
            draw_figure(i, ghost_colors, ghost_widths, (255, 220, 190))

    # 現在フレーム (濃い太線)
    cur_colors = {'rope':(120,60,60),'seat':(90,50,20),
                  'torso':(40,90,200),'leg':(40,90,200)}
    cur_widths = {'rope':4,'torso':14,'leg':11}
    draw_figure(frame, cur_colors, cur_widths, (255, 210, 170))

    text_surface = font.render(f"Episode: {epi[frame]}", True, (200, 0, 0))
    screen.blit(text_surface, (2*x0-150, 2*y0-40))
    tline = font.render(f"t = {t[frame]:6.1f} s", True, (40, 40, 40))
    screen.blit(tline, (20, 20))
    xline = font.render(f"|x| = {abs(np.degrees(x[frame])):5.1f} deg", True, (40, 40, 40))
    screen.blit(xline, (20, 50))

    pygame.display.flip()
    clock.tick(100)  # 5 FPS
    if frame < len(t) - 1:
        frame += 1    

pygame.quit()


