import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

def p2m(x):
    F_idx = []
    i = 0  # Pythonは0始まりのインデックス
    while i < len(x) - 1:  # ループ内でlen(d1)-1の要素までアクセスする
        if x[i] >= 0 and x[i+1] < 0:
            F_idx.append(i)
            i += 5  # インデックスで5は離れているべき
        i += 1
    return F_idx

def m2p(x):
    B_idx = []
    i = 0  # Pythonは0始まりのインデックス
    while i < len(x) - 1:  # ループ内でlen(d1)-1の要素までアクセスする
        if x[i] <= 0 and x[i+1] > 0:
            B_idx.append(i)
            i += 5  # インデックスで5は離れているべき
        i += 1
    return B_idx


if len(sys.argv)<2 :
    print("コマンドライン引数で ファイル名 を入力")
    sys.exit()


DEG = 180/np.pi

filename =  sys.argv[1]
obs = pd.read_csv(filename)
episodes = max( obs['episode'].values)
print(episodes)


for i in range(0 , episodes+1):
    indices = np.where(obs['episode'].values == i)[0]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7)) 

    t = obs['t'].values[indices]
    x = obs['x'].values[indices]
    z = obs['z'].values[indices]
    phi = obs['phi'].values[indices]
    d_phi = obs['d_phi'].values[indices]
    torq_ai = obs['torq_ai'].values[indices]
    torq_Hooke = obs['torq_Hooke'].values[indices]
    torq_LB = obs['torq_LB'].values[indices]
    torq_iner = obs['torq_iner'].values[indices]
    torq_cent = obs['torq_cent'].values[indices]
    torq_grav = obs['torq_grav'].values[indices]
    F_idx = p2m(z)
    B_idx = m2p(z)
    #print(F_idx)
    #print(B_idx)


    atMLB=[]
    for i in range(0,len(B_idx)-2):
        atMLB.append( B_idx[i]-1 +np.argmax( phi[B_idx[i]:F_idx[i+1]] )) 

    atMLB_pi=[]
    A=[]
    for k in range(0,len(B_idx)-2):
        idx1=B_idx[k]
        idx2=B_idx[k+1]
        idx12=F_idx[k+1]

        atMLB_pi.append( (atMLB[k]-idx1)/(idx12-idx1) )
        A.append(max(abs( x[idx1:idx2] ))*DEG)



    ax=axes.flatten()[0]

    ax.plot(t, x*DEG, linestyle='-',color='red',label='Swing')  
    ax.plot(t, phi*DEG, linestyle='-',color='blue',label='LB')  
    ax.legend()
    ax.set_xlabel('sec')           
    ax.set_ylabel('deg')           
    ax.grid(True)      
    #ax.set_xlim(0, 120)

    ax=axes.flatten()[1]
    ax.plot(t, torq_LB, linestyle='-',color='green',label='total',linewidth=0.5)  
    ax.plot(t, torq_ai, linestyle='-',color='red',label='AI',linewidth=0.5)  
    ax.plot(t, torq_iner, linestyle='-',color='orange',label='iner',linewidth=0.5)  
    ax.plot(t, torq_cent, linestyle='-',color='cyan',label='cent',linewidth=0.5)  
    ax.plot(t, torq_grav, linestyle='-',color='blue',label='grav',linewidth=0.5)  
    ax.set_xlabel('sec')           
    ax.set_ylabel('Torque [Nm]')           
    #ax.set_xlim(0, 120)
    ax.legend()
    ax.grid(True)      



    ax=axes.flatten()[2]
    ax.plot(A, atMLB_pi, marker='o', linestyle='None',color='blue')  
    ax.set_xlabel('A[deg]')           
    ax.set_ylabel('@MLB')        
    ax.set_ylim(0, 1)
    ax.grid(True)      



    plt.tight_layout() # グラフが重ならないように調整 
    plt.show()                

#plt.title('') 
#plt.xlim(0, 90)
#plt.ylim(0, 1)
#plt.xlabel('A')           
#plt.ylabel('@MLB')        
