import math
import sys
import random

class Swing:

    cN=["2XS","XS", "Sv", "Sr", "M", "L"] 
    cNM = [1.21, 1.41, 1.61, 1.61, 1.81, 2.01] 
    cNG = [7.47, 7.77, 8.12, 8.12, 8.56, 9.32] 

    h=0.001   #ミリ秒で系を更新する

    G = 9.80665
    RAD= math.pi/180  
    DEG=180/ math.pi   

    M = 50   
    Lh = 1.58  
    m1t=0.545*M   # 胴体(頭部除く上半身)
    m1h=0.081*M   # 頭部
    m1=m1t+m1h    # 旧m1互換(上半身全体)
    m2=0.246*M
    m3=0.128*M
    L1t=0.411*Lh  # 胴体長(座面～首)
    L1h=0.089*Lh  # 頭部長(首～頭頂)
    L1=L1t+L1h    # 旧L1互換
    L2=0.249*Lh
    L3=0.25*Lh
    a=L2*(m3+0.5*m2)/M
    b=L2-a


    def __init__(self,eps = -0.006, chn = 4,x=0, phi_init=-20, delta_phi_0=15,
                     VR=0, coef_Hooke=1):
                  #  外力の反映 1:2XS 2:XS 3:Sv 4:Sr 5:M 6:L,初期座面角(deg)
                  #                  ,初期上体角(振動中心からの),上体振動の中心(deg)
                  # 何往復 , 座面の前進局面での限界値 , VR=0:Realブランコ

        self.eps = eps 
        self.chn = chn -1 # matlabの配列は1から,pythonは0から。その補正!
        self.x_init = -x*self.RAD
        r=random.uniform(0.9, 1.1) #少しのランダム性
        self.x_init = -r*x*self.RAD

        self.delta_phi_0 = delta_phi_0 *self.RAD
        self.phi_init = phi_init *self.RAD 
        #self.nmax = nmax  
        #self.xDegmax = xDegmax 
        self.VR = VR
        self.coef_Hooke = coef_Hooke
        self.torq_ai = 0
        self.torq_ai_head = 0  # 頭部AIトルク
        self.u_body_filt = 0.0  # ローパスフィルター後の腰トルク
        self.u_head_filt = 0.0  # ローパスフィルター後の頭トルク
        self.tau_u = 0.1        # トルクフィルターの時定数 [s]

        self.L =self.cNM[self.chn]  
        self.m0=self.cNG[self.chn]  


        self.coef_iner=self.eps*int(not bool(VR)) #慣性力トルクの係数 VR=0でnonzero
        self.coef_cent=self.eps*int(not bool(VR)) #遠心力トルク VR=0でnonzero
        self.coef_grav=self.eps     #重力トルク

        self.Res_sw2 =0.  #0.4*0.3   #座面の速度2乗に比例する抵抗
        self.Res_sw1 =0.  #座面の速度に比例する抵抗
        self.Res_sw0 =0.  #0.4*0.7   #速度に依存しない抵抗

        self.omega_phi= math.sqrt(self.G/(self.L-0.5*self.L1))  




    def reset(self):
        r=random.uniform(0.9, 1.1) #少しのランダム性
        # バイアス付きランダム初期条件: 静止スタートを優先
        # 40%の確率で完全静止スタート、60%の確率でランダム振幅(0〜最大)
        if random.random() < 0.4:
            scale = 0.0  # 完全静止スタート
        else:
            scale = random.uniform(0.0, 1.0)  # 0〜最大初期振幅
        phi_init_s = scale * self.phi_init
        x_init_s   = scale * self.x_init
        self.phi=r*phi_init_s* math.cos(0.1* math.pi) + self.delta_phi_0 #体初期角
        self.d_phi=-r*self.omega_phi*phi_init_s* math.sin( 0.1* math.pi) #初期角速度
        self.z=1.e-60
        self.z00=1.
        self.t= self.t1= self.t2 =0
        self.n = 0
        self.x = x_init_s

        self.u_body_filt = 0.0
        self.u_head_filt = 0.0
        self.oldz= self.dz= self.oldd_phi= self.d2_phi=0
        self.alpha= 0       # 頭部の頸部関節角(胴体からの相対角)
        self.d_alpha= 0     # 頭部角速度
        self.d2_alpha= 0    # 頭部角加速度
        self.psi= self.d_psi= self.d2_psi=0

        self.phimax= self.xmax= -99 
        self.phimin= self.xmin =99 

        self.MLB= self.atMLB= self.arg= self.torq_LB= self.torq_Hooke=0 
        self.E=self.Esw=self.torq_iner= self.torq_grav=torq_cent=0  

        #self.Tp=2* math.pi/self.omega_phi 

        self.data_csv = "t,x,z,dz,phi,d_phi,d2_phi,T,torq_LB,torq_Hooke,"\
                        "torq_iner,torq_grav,torq_cent,E,Esw\n"
        print((f"iner:{self.coef_iner},cent:{self.coef_cent},grav:{self.coef_grav}"
               f",Hooke:{self.coef_Hooke},omega:{self.omega_phi:.2f}"))

    def calc_accel(self, x, dx, phi, dphi, alpha, dalpha, u_body, u_head):
        """3x3連立を解いて d2x, d2phi, d2alpha を返す"""
        L = self.L
        L1t, L1h = self.L1t, self.L1h
        m0, m1t, m1h = self.m0, self.m1t, self.m1h
        G = self.G

        s_px = math.sin(phi - x)
        c_px = math.cos(phi - x)
        s_ax = math.sin(alpha - x)
        c_ax = math.cos(alpha - x)
        s_ap = math.sin(alpha - phi)
        c_ap = math.cos(alpha - phi)

        # 質量行列
        A = (m0/3 + m1t + m1h) * L**2
        B = m1t * L1t**2 / 3 + m1h * L1t**2
        C = m1h * L1h**2 / 3
        D = -(m1t*L1t/2 + m1h*L1t) * L * c_px
        E = -m1h * (L1h/2) * L * c_ax
        F = m1h * L1t * (L1h/2) * c_ap

        # 右辺
        Rx = (-(m1t*L1t/2 + m1h*L1t) * L * s_px * dphi**2
              - m1h*(L1h/2) * L * s_ax * dalpha**2
              - (m0/2 + m1t + m1h) * G * L * math.sin(x))

        # 腰部のバネ復元力 + ダンパ
        k_waist = 144.0  # [Nm/rad] 重力不安定化(~113)+ 余裕(~31)で固有周期2.65s
        c_waist = 10.0   # [Nm·s/rad] 漕ぎを許しつつ第2モードを抑制
        torq_waist = -k_waist * (phi - x) - c_waist * (dphi - dx)

        Rphi = (-(m1t*L1t/2 + m1h*L1t) * L * s_px * dx**2
                + m1h*L1t*(L1h/2) * s_ap * dalpha**2
                - (m1t/2 + m1h) * G * L1t * math.sin(phi)
                + torq_waist
                + u_body)

        # 首のバネ復元力 + ダンパ (頭部を胴体に剛結合相当)
        k_neck = 2000.0  # [Nm/rad]
        c_neck = 50.0    # [Nm·s/rad]
        torq_neck = -k_neck * (alpha - phi) - c_neck * (dalpha - dphi)

        Ralpha = (-m1h*(L1h/2) * L * s_ax * dx**2
                  - m1h*L1t*(L1h/2) * s_ap * dphi**2
                  - (m1h/2) * G * L1h * math.sin(alpha)
                  + torq_neck
                  + u_head)

        # 3x3連立 (クラメルの公式)
        det = A*(B*C - F**2) - D*(D*C - F*E) + E*(D*F - B*E)
        d2x     = ((B*C-F**2)*Rx + (E*F-D*C)*Rphi + (D*F-B*E)*Ralpha) / det
        d2phi   = ((E*F-D*C)*Rx + (A*C-E**2)*Rphi + (D*E-A*F)*Ralpha) / det
        d2alpha = ((D*F-B*E)*Rx + (D*E-A*F)*Rphi + (A*B-D**2)*Ralpha) / det

        # トルク記録
        self.torq_ai = u_body
        self.torq_ai_head = u_head
        self.torq_grav = (m0/2 + m1t + m1h) * G * L * math.sin(x)

        return d2x, d2phi, d2alpha


    def rk4_step(self, u_body, u_head):
        """6変数 (x,dx,phi,dphi,alpha,dalpha) を1ステップ進める"""
        h = self.h
        # 1次ローパスフィルター (時定数 tau_u)
        alpha_f = h / self.tau_u
        self.u_body_filt += (u_body - self.u_body_filt) * alpha_f
        self.u_head_filt += (u_head - self.u_head_filt) * alpha_f
        u_body = self.u_body_filt
        u_head = self.u_head_filt

        x0, dx0 = self.x, self.z
        p0, dp0 = self.phi, self.d_phi
        a0, da0 = self.alpha, self.d_alpha

        # k1
        ax1, ap1, aa1 = self.calc_accel(x0, dx0, p0, dp0, a0, da0, u_body, u_head)
        kx1, kdx1 = h*dx0, h*ax1
        kp1, kdp1 = h*dp0, h*ap1
        ka1, kda1 = h*da0, h*aa1

        # k2
        ax2, ap2, aa2 = self.calc_accel(
            x0+kx1/2, dx0+kdx1/2, p0+kp1/2, dp0+kdp1/2, a0+ka1/2, da0+kda1/2, u_body, u_head)
        kx2, kdx2 = h*(dx0+kdx1/2), h*ax2
        kp2, kdp2 = h*(dp0+kdp1/2), h*ap2
        ka2, kda2 = h*(da0+kda1/2), h*aa2

        # k3
        ax3, ap3, aa3 = self.calc_accel(
            x0+kx2/2, dx0+kdx2/2, p0+kp2/2, dp0+kdp2/2, a0+ka2/2, da0+kda2/2, u_body, u_head)
        kx3, kdx3 = h*(dx0+kdx2/2), h*ax3
        kp3, kdp3 = h*(dp0+kdp2/2), h*ap3
        ka3, kda3 = h*(da0+kda2/2), h*aa3

        # k4
        ax4, ap4, aa4 = self.calc_accel(
            x0+kx3, dx0+kdx3, p0+kp3, dp0+kdp3, a0+ka3, da0+kda3, u_body, u_head)
        kx4, kdx4 = h*(dx0+kdx3), h*ax4
        kp4, kdp4 = h*(dp0+kdp3), h*ap4
        ka4, kda4 = h*(da0+kda3), h*aa4

        # 更新
        self.x      += (kx1  + 2*kx2  + 2*kx3  + kx4 ) / 6
        self.z      += (kdx1 + 2*kdx2 + 2*kdx3 + kdx4) / 6
        self.phi    += (kp1  + 2*kp2  + 2*kp3  + kp4 ) / 6
        self.d_phi  += (kdp1 + 2*kdp2 + 2*kdp3 + kdp4) / 6
        self.alpha  += (ka1  + 2*ka2  + 2*ka3  + ka4 ) / 6
        self.d_alpha+= (kda1 + 2*kda2 + 2*kda3 + kda4) / 6

        self.dz     = (kdx1 + 2*kdx2 + 2*kdx3 + kdx4) / (h*6)
        self.d2_phi = (kdp1 + 2*kdp2 + 2*kdp3 + kdp4) / (h*6)
        self.d2_alpha=(kda1 + 2*kda2 + 2*kda3 + kda4) / (h*6)


    def observe(self):
        """ブランコのエネルギーを返す (報酬計算用)
        ロープ+人体全体が座面位置で振れる運動エネルギーと位置エネルギー。
        頭部・胴体の相対振動エネルギーは含めない。"""
        L = self.L
        m0, M, G = self.m0, self.M, self.G
        x, dx = self.x, self.z

        T_sw = 0.5 * (m0/3 + M) * L**2 * dx**2
        V_sw = -(m0/2 + M) * G * L * math.cos(x)

        return T_sw + V_sw

