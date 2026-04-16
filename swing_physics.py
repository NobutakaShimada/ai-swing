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
    m1=m1t+m1h    # 上半身全体(頭部を胴体に統合)
    m2=0.246*M    # 上腿(ロープと剛結合)
    m3=0.128*M    # 下腿(膝関節で接続, 新DOF)
    L1t=0.411*Lh  # 胴体長(座面～首)
    L1h=0.089*Lh  # 頭部長(首～頭頂)
    L1=L1t+L1h    # 上半身全長(統合)
    L2=0.249*Lh   # 上腿長(座面～膝)
    L3=0.25*Lh    # 下腿長(膝～足先)
    a=L2*(m3+0.5*m2)/M
    b=L2-a


    def __init__(self,eps = -0.006, chn = 4,x=0, phi_init=-20, delta_phi_0=15,
                     VR=0, coef_Hooke=1):

        self.eps = eps
        self.chn = chn -1
        self.x_init = -x*self.RAD
        r=random.uniform(0.9, 1.1)
        self.x_init = -r*x*self.RAD

        self.delta_phi_0 = delta_phi_0 *self.RAD
        self.phi_init = phi_init *self.RAD
        self.VR = VR
        self.coef_Hooke = coef_Hooke
        self.torq_ai = 0
        self.torq_ai_knee = 0    # 膝AIトルク
        self.u_body_filt = 0.0   # ローパスフィルター後の腰トルク
        self.u_knee_filt = 0.0   # ローパスフィルター後の膝トルク
        self.tau_u = 0.1         # トルクフィルターの時定数 [s]

        self.L =self.cNM[self.chn]
        self.m0=self.cNG[self.chn]

        # 膝の位置 (支点からの距離)
        self.Lk = self.L + self.L2  # 支点→膝

        self.coef_iner=self.eps*int(not bool(VR))
        self.coef_cent=self.eps*int(not bool(VR))
        self.coef_grav=self.eps

        self.Res_sw2 =0.
        self.Res_sw1 =0.
        self.Res_sw0 =0.

        self.omega_phi= math.sqrt(self.G/(self.L-0.5*self.L1))



    def reset(self):
        r=random.uniform(0.9, 1.1)
        # バイアス付きランダム初期条件: 静止スタートを優先
        if random.random() < 0.4:
            scale = 0.0
        else:
            scale = random.uniform(0.0, 1.0)
        phi_init_s = scale * self.phi_init
        x_init_s   = scale * self.x_init
        self.phi=r*phi_init_s* math.cos(0.1* math.pi) + self.delta_phi_0
        self.d_phi=-r*self.omega_phi*phi_init_s* math.sin( 0.1* math.pi)
        self.z=1.e-60
        self.z00=1.
        self.t= self.t1= self.t2 =0
        self.n = 0
        self.x = x_init_s

        self.u_body_filt = 0.0
        self.u_knee_filt = 0.0
        self.oldz= self.dz= self.oldd_phi= self.d2_phi=0
        # 下腿: 初期状態はロープと同方向 (psi = x)
        self.psi = x_init_s
        self.d_psi = 0.0
        self.d2_psi = 0.0

        self.phimax= self.xmax= -99
        self.phimin= self.xmin =99

        self.MLB= self.atMLB= self.arg= self.torq_LB= self.torq_Hooke=0
        self.E=self.Esw=self.torq_iner= self.torq_grav=torq_cent=0

        self.data_csv = "t,x,z,dz,phi,d_phi,d2_phi,psi,d_psi,d2_psi\n"
        print((f"iner:{self.coef_iner},cent:{self.coef_cent},grav:{self.coef_grav}"
               f",Hooke:{self.coef_Hooke},omega:{self.omega_phi:.2f}"))

    def calc_accel(self, x, dx, phi, dphi, psi, dpsi, u_body, u_knee):
        """3x3連立 (x, phi, psi) を解いて d2x, d2phi, d2psi を返す

        自由度:
          x   : ブランコ(ロープ)角 (鉛直下向きから)
          phi : 胴体角 (頭部質量統合, 鉛直下向きから, 座面から上方に伸びる)
          psi : 下腿角 (鉛直下向きから, 膝から下方に伸びる)

        上腿(m2)はロープと剛結合。胴体-下腿の直接結合なし (F=0)。
        """
        L = self.L
        Lk = self.Lk       # 支点→膝 = L + L2
        L1 = self.L1       # 上半身全長 (統合)
        L3 = self.L3
        m0 = self.m0
        m1 = self.m1       # 上半身全体 (m1t + m1h)
        m3 = self.m3       # 下腿
        G = self.G

        s_px = math.sin(phi - x)
        c_px = math.cos(phi - x)
        s_kx = math.sin(psi - x)
        c_kx = math.cos(psi - x)

        # ---- 質量行列 ----
        # A: ブランコ(ロープ) + 上半身(座面に集中) + 下腿(膝に集中)
        A = (m0/3 + m1) * L**2 + m3 * Lk**2
        # B: 上半身の腰まわり慣性 (統合した一様棒)
        B = m1 * L1**2 / 3
        # C: 下腿の膝まわり慣性
        C = m3 * L3**2 / 3
        # D: ロープ-胴体結合 (胴体は上向き → 負符号)
        D = -m1 * (L1/2) * L * c_px
        # E: ロープ-下腿結合 (下腿は下向き=ロープと同方向 → 正符号)
        E = m3 * Lk * (L3/2) * c_kx
        # F: 胴体-下腿結合 → 0 (直接接続なし)

        # ---- 右辺 (速度依存 + 重力 + 外力) ----
        # Rx: ブランコ方程式
        Rx = (-(m1*(L1/2)) * L * s_px * dphi**2        # 胴体からの遠心力
              + m3 * Lk * (L3/2) * s_kx * dpsi**2      # 下腿からの遠心力
              - (m0/2 + m1) * G * L * math.sin(x)       # 重力(ロープ+上半身)
              - m3 * G * Lk * math.sin(x))               # 重力(下腿, 膝位置)

        # 腰部のバネ復元力 + ダンパ
        k_waist = 144.0   # [Nm/rad]
        c_waist = 10.0    # [Nm·s/rad]
        torq_waist = -k_waist * (phi - x) - c_waist * (dphi - dx)

        # Rphi: 胴体方程式
        Rphi = (-(m1*(L1/2)) * L * s_px * dx**2         # ロープからの遠心力
                - (m1/2) * G * L1 * math.sin(phi)        # 重力
                + torq_waist
                + u_body)

        # 膝のバネ復元力 + ダンパ (自然姿勢 = ロープと同方向)
        k_knee = 50.0    # [Nm/rad]
        c_knee = 5.0     # [Nm·s/rad]
        torq_knee = -k_knee * (psi - x) - c_knee * (dpsi - dx)

        # Rpsi: 下腿方程式
        Rpsi = (-m3 * Lk * (L3/2) * s_kx * dx**2       # ロープからの遠心力
                - (m3/2) * G * L3 * math.sin(psi)        # 重力(吊り下げ→復元力)
                + torq_knee
                + u_knee)

        # ---- 3x3連立 (クラメルの公式, F=0で簡略化) ----
        det = A*B*C - D**2*C - B*E**2
        d2x   = ( B*C*Rx     - D*C*Rphi    - B*E*Rpsi) / det
        d2phi = (-D*C*Rx     + (A*C-E**2)*Rphi + D*E*Rpsi) / det
        d2psi = (-B*E*Rx     + D*E*Rphi    + (A*B-D**2)*Rpsi) / det

        # トルク記録
        self.torq_ai = u_body
        self.torq_ai_knee = u_knee
        self.torq_grav = (m0/2 + m1 + m3*Lk/L) * G * L * math.sin(x)

        return d2x, d2phi, d2psi


    def rk4_step(self, u_body, u_knee):
        """6変数 (x,dx,phi,dphi,psi,dpsi) を1ステップ進める"""
        h = self.h
        # 1次ローパスフィルター
        alpha_f = h / self.tau_u
        self.u_body_filt += (u_body - self.u_body_filt) * alpha_f
        self.u_knee_filt += (u_knee - self.u_knee_filt) * alpha_f
        u_body = self.u_body_filt
        u_knee = self.u_knee_filt

        x0, dx0 = self.x, self.z
        p0, dp0 = self.phi, self.d_phi
        k0, dk0 = self.psi, self.d_psi

        # k1
        ax1, ap1, ak1 = self.calc_accel(x0, dx0, p0, dp0, k0, dk0, u_body, u_knee)
        kx1, kdx1 = h*dx0, h*ax1
        kp1, kdp1 = h*dp0, h*ap1
        kk1, kdk1 = h*dk0, h*ak1

        # k2
        ax2, ap2, ak2 = self.calc_accel(
            x0+kx1/2, dx0+kdx1/2, p0+kp1/2, dp0+kdp1/2, k0+kk1/2, dk0+kdk1/2, u_body, u_knee)
        kx2, kdx2 = h*(dx0+kdx1/2), h*ax2
        kp2, kdp2 = h*(dp0+kdp1/2), h*ap2
        kk2, kdk2 = h*(dk0+kdk1/2), h*ak2

        # k3
        ax3, ap3, ak3 = self.calc_accel(
            x0+kx2/2, dx0+kdx2/2, p0+kp2/2, dp0+kdp2/2, k0+kk2/2, dk0+kdk2/2, u_body, u_knee)
        kx3, kdx3 = h*(dx0+kdx2/2), h*ax3
        kp3, kdp3 = h*(dp0+kdp2/2), h*ap3
        kk3, kdk3 = h*(dk0+kdk2/2), h*ak3

        # k4
        ax4, ap4, ak4 = self.calc_accel(
            x0+kx3, dx0+kdx3, p0+kp3, dp0+kdp3, k0+kk3, dk0+kdk3, u_body, u_knee)
        kx4, kdx4 = h*(dx0+kdx3), h*ax4
        kp4, kdp4 = h*(dp0+kdp3), h*ap4
        kk4, kdk4 = h*(dk0+kdk3), h*ak4

        # 更新
        self.x      += (kx1  + 2*kx2  + 2*kx3  + kx4 ) / 6
        self.z      += (kdx1 + 2*kdx2 + 2*kdx3 + kdx4) / 6
        self.phi    += (kp1  + 2*kp2  + 2*kp3  + kp4 ) / 6
        self.d_phi  += (kdp1 + 2*kdp2 + 2*kdp3 + kdp4) / 6
        self.psi    += (kk1  + 2*kk2  + 2*kk3  + kk4 ) / 6
        self.d_psi  += (kdk1 + 2*kdk2 + 2*kdk3 + kdk4) / 6

        self.dz     = (kdx1 + 2*kdx2 + 2*kdx3 + kdx4) / (h*6)
        self.d2_phi = (kdp1 + 2*kdp2 + 2*kdp3 + kdp4) / (h*6)
        self.d2_psi = (kdk1 + 2*kdk2 + 2*kdk3 + kdk4) / (h*6)


    def observe(self):
        """ブランコのエネルギーを返す (報酬計算用)"""
        L = self.L
        m0, M, G = self.m0, self.M, self.G
        x, dx = self.x, self.z

        T_sw = 0.5 * (m0/3 + M) * L**2 * dx**2
        V_sw = -(m0/2 + M) * G * L * math.cos(x)

        return T_sw + V_sw
