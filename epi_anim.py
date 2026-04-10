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
    
    screen.fill((255, 255, 255))
    d = -0.5 * np.pi  # 角度のオフセット
    
    # atMLBを描画
    for i in range( frame-1, -1,-1):
        if epi[i+1] != epi[i]:
            break
        if atMLB[i]>0:
            rx0=L*100*np.cos(x[i]+d)
            ry0=L*100*np.sin(x[i]+d)
            pygame.draw.aalines(screen, (200, 200, 200),False, [(x0,y0),(x0+rx0,y0-ry0)])
            rx1=-L1*100*np.cos(x[i]+phi[i]+d)
            ry1=-L1*100*np.sin(x[i]+phi[i]+d)
            rx2a =-a*100*np.sin(x[i]-d)
            ry2a = a*100*np.cos(x[i]-d)
            pygame.draw.aalines(screen, (255, 165, 100),False
                        ,[(x0+rx0+rx2a,y0-ry0-ry2a),(x0+rx0+rx2a+rx1,y0-ry0-ry2a-ry1)])  #上体描画
            pygame.draw.aalines(screen, (255, 165, 100),False
                        ,[(x0+rx0,y0-ry0),(x0+rx0+rx2a,y0-ry0-ry2a)])  #

    
    rx0=L*100*np.cos(x[frame]+d)
    ry0=L*100*np.sin(x[frame]+d)
    pygame.draw.aalines(screen, (255, 0, 0),False, [(x0,y0),(x0+rx0,y0-ry0)])

    rx1=-L1*100*np.cos(x[frame]+phi[frame]+d)
    ry1=-L1*100*np.sin(x[frame]+phi[frame]+d)
    rx2a =-a*100*np.sin(x[frame]-d)
    ry2a = a*100*np.cos(x[frame]-d)
    pygame.draw.aalines(screen, (0, 0, 255),False
                        ,[(x0+rx0+rx2a,y0-ry0-ry2a),(x0+rx0+rx2a+rx1,y0-ry0-ry2a-ry1)])  #上体描画

    rx2=-b*100*np.sin(x[frame]+d)
    ry2=b*100*np.cos(x[frame]+d)
    pygame.draw.aalines(screen, (0, 0, 255),False
                        ,[(x0+rx0+rx2a,y0-ry0-ry2a),(x0+rx0+rx2,y0-ry0-ry2)]) #上腿描画
    
    rx3 = L3*100*np.cos(x[frame]+psi+d)
    ry3 = L3*100*np.sin(x[frame]+psi+d)
    pygame.draw.aalines(screen, (0, 0, 255),False
                        ,[(x0+rx0+rx2,y0-ry0-ry2),(x0+rx0+rx2+rx3,y0-ry0-ry2-ry3)]) #下腿描画

    text_surface = font.render(f"Episode: {epi[frame]}", True, (255, 0, 0))
    screen.blit(text_surface, (2*x0-120, 2*y0-40)) # 

    pygame.display.flip()
    clock.tick(100)  # 5 FPS
    if frame < len(t) - 1:
        frame += 1    

pygame.quit()


